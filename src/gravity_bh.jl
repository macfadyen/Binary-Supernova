# Softened black-hole gravitational potentials and gas source terms.
# Two moving BH point masses with Plummer softening; G = 1 in code units.
#
# Potential:      Φ_i(r) = −M_i / sqrt(|r − r_i|² + ε_i²)
# Acceleration:   −∇Φ_i(r) = −M_i (r − r_i) / (|r − r_i|² + ε_i²)^(3/2)
#                             (points toward BH i; G = 1)
#
# Gas momentum / energy source terms (method-of-lines RHS):
#   d(ρv)/dt += ρ * Σ_i [−∇Φ_i(r_cell)]
#   d(E)/dt  += ρ * v · Σ_i [−∇Φ_i(r_cell)]
#
# Force on BH i from gas (Newton's 3rd law, attractive toward gas):
#   F_gas→i = +∫ ρ ∇Φ_i dV
#   where ∇Φ_i(r) = M_i (r − r_i) / (|r − r_i|² + ε_i²)^(3/2)
#
# GPU: add_bh_gravity_source! packs BH data into NTuples (value types,
# safe to pass to GPU kernels) and dispatches to _bh_gravity_kernel!.

# ---------------------------------------------------------------------------
# Point evaluations

"""
    bh_potential(bh, x, y, z) -> Float64

Gravitational potential at (x, y, z) due to `bh` (G = 1, Plummer softening).
Φ = −M / sqrt(r² + ε²),  r² = |pos − bh.pos|².
"""
@inline function bh_potential(bh::BlackHole, x::Float64, y::Float64, z::Float64)
    ddx = x - bh.pos[1]; ddy = y - bh.pos[2]; ddz = z - bh.pos[3]
    return -bh.mass / sqrt(ddx^2 + ddy^2 + ddz^2 + bh.eps^2)
end

"""
    bh_accel(bh, x, y, z) -> (ax, ay, az)

Gravitational acceleration at (x, y, z) due to `bh` (G = 1).
Returns −∇Φ_bh, which points toward the BH (attractive).
"""
@inline function bh_accel(bh::BlackHole, x::Float64, y::Float64, z::Float64)
    ddx = x - bh.pos[1]; ddy = y - bh.pos[2]; ddz = z - bh.pos[3]
    r2  = ddx^2 + ddy^2 + ddz^2 + bh.eps^2
    r3  = r2 * sqrt(r2)                      # (r² + ε²)^(3/2)
    fac = -bh.mass / r3                       # − M (r − r_i) / r³ → points toward BH
    return fac * ddx, fac * ddy, fac * ddz
end

# ---------------------------------------------------------------------------
# Grid-level source terms

"""
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz, bhs, x0, y0, z0)

Add BH gravitational source terms to the method-of-lines RHS `dU`.

BH data is packed into NTuples before kernel dispatch so the kernel arguments
are pure value types, compatible with both CPU and CUDA backends.
"""
function add_bh_gravity_source!(dU, U,
                                 nx::Int, ny::Int, nz::Int,
                                 dx::Real, dy::Real, dz::Real,
                                 bhs, x0::Real, y0::Real, z0::Real)
    nbh = length(bhs)
    bh_px   = ntuple(n -> Float64(bhs[n].pos[1]),  nbh)
    bh_py   = ntuple(n -> Float64(bhs[n].pos[2]),  nbh)
    bh_pz   = ntuple(n -> Float64(bhs[n].pos[3]),  nbh)
    bh_mass = ntuple(n -> Float64(bhs[n].mass),    nbh)
    bh_eps  = ntuple(n -> Float64(bhs[n].eps),     nbh)

    backend = KA.get_backend(U)
    kern = _bh_gravity_kernel!(backend, _WGSIZE_3D)
    kern(dU, U, nx, ny, nz,
         Float64(dx), Float64(dy), Float64(dz),
         Float64(x0), Float64(y0), Float64(z0),
         bh_px, bh_py, bh_pz, bh_mass, bh_eps;
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
    return nothing
end

"""
    gas_force_on_bh(U, nx, ny, nz, dx, dy, dz, bh, x0, y0, z0) -> Vector{Float64}

Gravitational force on `bh` from the gas: F = +∫ ρ ∇Φ_bh dV (G = 1).
∇Φ_bh(r) = M_bh (r − r_bh) / (|r − r_bh|² + ε²)^(3/2).
By Newton's third law this is the force the gas exerts on the BH (attractive,
pointing toward the centroid of the gas distribution).

Runs on CPU (called once per BH per timestep; result used in N-body update).
For GPU simulations, the active-cell density slice is copied to host first.
"""
function gas_force_on_bh(U,
                          nx::Int, ny::Int, nz::Int,
                          dx::Real, dy::Real, dz::Real,
                          bh::BlackHole, x0::Real, y0::Real, z0::Real)
    ng = NG
    dV = dx * dy * dz
    # Copy active density and momenta to CPU if on GPU
    ρ_host = Array(view(U, 1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        xc = x0 + (i - 0.5) * dx
        yc = y0 + (j - 0.5) * dy
        zc = z0 + (k - 0.5) * dz
        ddx = xc - bh.pos[1]; ddy = yc - bh.pos[2]; ddz = zc - bh.pos[3]
        r2  = ddx^2 + ddy^2 + ddz^2 + bh.eps^2
        r3  = r2 * sqrt(r2)
        fac = bh.mass / r3 * ρ_host[i, j, k] * dV
        Fx += fac * ddx
        Fy += fac * ddy
        Fz += fac * ddz
    end
    return [Fx, Fy, Fz]
end
