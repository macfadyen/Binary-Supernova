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

For each active cell, the source is:
  d(ρv)/dt += ρ * a_BH
  d(E)/dt  += ρ * v · a_BH
where a_BH = Σ_i [−∇Φ_i(r_cell)] is the total BH gravitational acceleration.

Arguments:
- `x0, y0, z0`: physical left edge of the active domain.
  Active cell (ng+1,ng+1,ng+1) has centre at (x0+dx/2, y0+dy/2, z0+dz/2).
"""
function add_bh_gravity_source!(dU, U,
                                 nx::Int, ny::Int, nz::Int,
                                 dx::Real, dy::Real, dz::Real,
                                 bhs, x0::Real, y0::Real, z0::Real)
    ng = NG
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz

        ρ  = U[1, i, j, k]
        vx = U[2, i, j, k] / ρ
        vy = U[3, i, j, k] / ρ
        vz = U[4, i, j, k] / ρ

        ax = 0.0; ay = 0.0; az = 0.0
        for bh in bhs
            dax, day, daz = bh_accel(bh, xc, yc, zc)
            ax += dax; ay += day; az += daz
        end

        dU[2, i, j, k] += ρ * ax
        dU[3, i, j, k] += ρ * ay
        dU[4, i, j, k] += ρ * az
        dU[5, i, j, k] += ρ * (vx * ax + vy * ay + vz * az)
    end
    return nothing
end

"""
    gas_force_on_bh(U, nx, ny, nz, dx, dy, dz, bh, x0, y0, z0) -> Vector{Float64}

Gravitational force on `bh` from the gas: F = +∫ ρ ∇Φ_bh dV (G = 1).
∇Φ_bh(r) = M_bh (r − r_bh) / (|r − r_bh|² + ε²)^(3/2).
By Newton's third law this is the force the gas exerts on the BH (attractive,
pointing toward the centroid of the gas distribution).
"""
function gas_force_on_bh(U,
                          nx::Int, ny::Int, nz::Int,
                          dx::Real, dy::Real, dz::Real,
                          bh::BlackHole, x0::Real, y0::Real, z0::Real)
    ng = NG
    dV = dx * dy * dz
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        ddx = xc - bh.pos[1]; ddy = yc - bh.pos[2]; ddz = zc - bh.pos[3]
        r2  = ddx^2 + ddy^2 + ddz^2 + bh.eps^2
        r3  = r2 * sqrt(r2)
        fac = bh.mass / r3 * U[1, i, j, k] * dV   # M ρ dV / r³
        Fx += fac * ddx
        Fy += fac * ddy
        Fz += fac * ddz
    end
    return [Fx, Fy, Fz]
end
