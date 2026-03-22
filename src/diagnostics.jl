# Scalar diagnostics for a BinarySupernova run.
#
# All functions use view-based broadcasting so they work with both CPU Arrays
# and CuArrays (GPU).  Coordinate arrays are constructed on CPU and moved to
# the device via Adapt.adapt when U is a CuArray.
#
# Quantities:
#   gas_energy_total   — E_gas  = Σ U[5] dV
#   gas_kinetic_total  — KE_gas = Σ (1/2 ρ |v|²) dV
#   gas_momentum_total — P_gas  = Σ (ρv) dV  (length-3 vector)
#   gas_angular_momentum_total — L_gas = Σ r × (ρv) dV about origin (length-3)
#   bh_energy_total    — KE_BH  = Σ_i (1/2 M_i |v_i|²)
#   bh_angular_momentum_total — L_BH = Σ_i M_i (r_i × v_i) about origin (length-3)
#   bound_gas_mass     — mass of gas cells with total energy ≤ 0 (BH gravity included)

"""
    gas_energy_total(U, nx, ny, nz, dx, dy, dz) -> Float64

Total gas energy: E = Σ U[5,i,j,k] × dV  (internal + kinetic).
"""
function gas_energy_total(U, nx::Int, ny::Int, nz::Int,
                           dx::Real, dy::Real, dz::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    return Float64(sum(view(U, 5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))) * dV
end

"""
    gas_kinetic_total(U, nx, ny, nz, dx, dy, dz) -> Float64

Total kinetic energy of the gas: KE = Σ (|ρv|² / 2ρ) dV.
"""
function gas_kinetic_total(U, nx::Int, ny::Int, nz::Int,
                            dx::Real, dy::Real, dz::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    ρ  = view(U, 1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mx = view(U, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    my = view(U, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mz = view(U, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    ρ_s = max.(ρ, 1e-30)
    return Float64(sum(@. 0.5 * (mx^2 + my^2 + mz^2) / ρ_s)) * dV
end

"""
    gas_momentum_total(U, nx, ny, nz, dx, dy, dz) -> Vector{Float64}

Total linear momentum of the gas: P = Σ (ρv) dV (length-3 vector).
"""
function gas_momentum_total(U, nx::Int, ny::Int, nz::Int,
                             dx::Real, dy::Real, dz::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    Px = Float64(sum(view(U, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))) * dV
    Py = Float64(sum(view(U, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))) * dV
    Pz = Float64(sum(view(U, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))) * dV
    return [Px, Py, Pz]
end

"""
    gas_angular_momentum_total(U, nx, ny, nz, dx, dy, dz, x0, y0, z0) -> Vector{Float64}

Total angular momentum of the gas about the origin: L = Σ r × (ρv) dV.
Cell-centre coordinate arrays are adapted to the device of U via Adapt.adapt.
"""
function gas_angular_momentum_total(U, nx::Int, ny::Int, nz::Int,
                                     dx::Real, dy::Real, dz::Real,
                                     x0::Real, y0::Real, z0::Real)
    ng = NG
    dV  = Float64(dx) * Float64(dy) * Float64(dz)
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)

    # Cell-centre coordinate arrays (broadcast shape nx × 1 × 1, etc.)
    xc_cpu = reshape([x0 + (i - ng - 0.5) * fdx for i in ng+1:ng+nx], nx, 1, 1)
    yc_cpu = reshape([y0 + (j - ng - 0.5) * fdy for j in ng+1:ng+ny], 1, ny, 1)
    zc_cpu = reshape([z0 + (k - ng - 0.5) * fdz for k in ng+1:ng+nz], 1, 1, nz)

    backend = KA.get_backend(U)
    xc = adapt(backend, xc_cpu)
    yc = adapt(backend, yc_cpu)
    zc = adapt(backend, zc_cpu)

    mx = view(U, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    my = view(U, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mz = view(U, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)

    Lx = Float64(sum(@. yc * mz - zc * my)) * dV
    Ly = Float64(sum(@. zc * mx - xc * mz)) * dV
    Lz = Float64(sum(@. xc * my - yc * mx)) * dV
    return [Lx, Ly, Lz]
end

"""
    bh_kinetic_total(bhs) -> Float64

Total kinetic energy of all BHs: KE_BH = Σ_i 1/2 M_i |v_i|².
"""
function bh_kinetic_total(bhs)
    KE = 0.0
    for bh in bhs
        KE += 0.5 * bh.mass * (bh.vel[1]^2 + bh.vel[2]^2 + bh.vel[3]^2)
    end
    return KE
end

"""
    bh_angular_momentum_total(bhs) -> Vector{Float64}

Total angular momentum of all BHs about the origin: L_BH = Σ_i M_i (r_i × v_i).
"""
function bh_angular_momentum_total(bhs)
    Lx = 0.0;  Ly = 0.0;  Lz = 0.0
    for bh in bhs
        M = bh.mass
        x, y, z    = bh.pos[1], bh.pos[2], bh.pos[3]
        vx, vy, vz = bh.vel[1], bh.vel[2], bh.vel[3]
        Lx += M * (y * vz - z * vy)
        Ly += M * (z * vx - x * vz)
        Lz += M * (x * vy - y * vx)
    end
    return [Lx, Ly, Lz]
end

"""
    bound_gas_mass(U, nx, ny, nz, dx, dy, dz, x0, y0, z0, bhs, γ) -> Float64

Mass of gas cells whose total specific energy (kinetic + thermal + gravitational)
is ≤ 0.  A cell is "bound" when:

    e_kin + e_int + Φ_BH ≤ 0

where e_kin = |v|²/2, e_int = P/((γ−1)ρ), Φ_BH = Σ_i −M_i / r_i (G=1).

Uses broadcasting for GPU compatibility; BH potential is computed via a loop
over (small) BH list on the device.
"""
function bound_gas_mass(U, nx::Int, ny::Int, nz::Int,
                         dx::Real, dy::Real, dz::Real,
                         x0::Real, y0::Real, z0::Real,
                         bhs, γ::Real)
    ng = NG
    dV  = Float64(dx) * Float64(dy) * Float64(dz)
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)

    ρ_v  = view(U, 1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mx_v = view(U, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    my_v = view(U, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mz_v = view(U, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    E_v  = view(U, 5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)

    backend = KA.get_backend(U)
    xc_cpu = reshape([x0 + (i - ng - 0.5) * fdx for i in ng+1:ng+nx], nx, 1, 1)
    yc_cpu = reshape([y0 + (j - ng - 0.5) * fdy for j in ng+1:ng+ny], 1, ny, 1)
    zc_cpu = reshape([z0 + (k - ng - 0.5) * fdz for k in ng+1:ng+nz], 1, 1, nz)
    xc = adapt(backend, xc_cpu)
    yc = adapt(backend, yc_cpu)
    zc = adapt(backend, zc_cpu)

    ρ_s     = max.(ρ_v, 1e-30)
    KE_spec = @. 0.5 * (mx_v^2 + my_v^2 + mz_v^2) / ρ_s
    P_v     = max.((γ - 1) .* (E_v .- ρ_s .* KE_spec), 0.0)
    e_int   = @. P_v / ((γ - 1) * ρ_s)

    # Accumulate BH gravitational potential (sum over typically 2 BHs)
    T = eltype(U)
    Φ = KA.zeros(backend, T, nx, ny, nz)
    for bh in bhs
        px = T(bh.pos[1]);  py = T(bh.pos[2]);  pz = T(bh.pos[3])
        M  = T(bh.mass);    ε  = T(bh.eps)
        @. Φ -= M / sqrt((xc - px)^2 + (yc - py)^2 + (zc - pz)^2 + ε^2)
    end

    e_tot = @. KE_spec + e_int + Φ
    return Float64(sum(ifelse.(e_tot .<= 0, ρ_s, zero(T)))) * dV
end
