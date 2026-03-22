# Scalar diagnostics for a BinarySupernova run.
#
# All functions operate on the conserved-variable array U (active cells only)
# plus the list of BlackHole structs.  No HDF5 I/O here — that is in io.jl.
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
    E  = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        E += U[5, i, j, k]
    end
    return E * dV
end

"""
    gas_kinetic_total(U, nx, ny, nz, dx, dy, dz) -> Float64

Total kinetic energy of the gas: KE = Σ (|ρv|² / 2ρ) dV.
"""
function gas_kinetic_total(U, nx::Int, ny::Int, nz::Int,
                            dx::Real, dy::Real, dz::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    KE = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ  = U[1, i, j, k]
        ρ  = max(ρ, 1e-30)
        KE += 0.5 * (U[2,i,j,k]^2 + U[3,i,j,k]^2 + U[4,i,j,k]^2) / ρ
    end
    return KE * dV
end

"""
    gas_momentum_total(U, nx, ny, nz, dx, dy, dz) -> Vector{Float64}

Total linear momentum of the gas: P = Σ (ρv) dV (length-3 vector).
"""
function gas_momentum_total(U, nx::Int, ny::Int, nz::Int,
                             dx::Real, dy::Real, dz::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    Px = 0.0;  Py = 0.0;  Pz = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        Px += U[2, i, j, k]
        Py += U[3, i, j, k]
        Pz += U[4, i, j, k]
    end
    return [Px * dV, Py * dV, Pz * dV]
end

"""
    gas_angular_momentum_total(U, nx, ny, nz, dx, dy, dz, x0, y0, z0) -> Vector{Float64}

Total angular momentum of the gas about the origin: L = Σ r × (ρv) dV.
"""
function gas_angular_momentum_total(U, nx::Int, ny::Int, nz::Int,
                                     dx::Real, dy::Real, dz::Real,
                                     x0::Real, y0::Real, z0::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    Lx = 0.0;  Ly = 0.0;  Lz = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        mx = U[2, i, j, k];  my = U[3, i, j, k];  mz = U[4, i, j, k]
        # r × (ρv): [y*mz - z*my, z*mx - x*mz, x*my - y*mx]
        Lx += yc * mz - zc * my
        Ly += zc * mx - xc * mz
        Lz += xc * my - yc * mx
    end
    return [Lx * dV, Ly * dV, Lz * dV]
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
"""
function bound_gas_mass(U, nx::Int, ny::Int, nz::Int,
                         dx::Real, dy::Real, dz::Real,
                         x0::Real, y0::Real, z0::Real,
                         bhs, γ::Real)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    M_bound = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        ρ  = U[1, i, j, k]
        ρ  = max(ρ, 1e-30)
        vx = U[2, i, j, k] / ρ;  vy = U[3, i, j, k] / ρ;  vz = U[4, i, j, k] / ρ
        KE_spec = 0.5 * (vx^2 + vy^2 + vz^2)
        P       = max((γ - 1.0) * (U[5, i, j, k] - ρ * KE_spec), 0.0)
        e_int   = P / ((γ - 1.0) * ρ)
        # Sum BH gravitational potential at this cell (no softening for binding check)
        Φ = 0.0
        for bh in bhs
            ddx = xc - bh.pos[1];  ddy = yc - bh.pos[2];  ddz = zc - bh.pos[3]
            r   = sqrt(ddx^2 + ddy^2 + ddz^2 + bh.eps^2)
            Φ  -= bh.mass / r
        end
        (KE_spec + e_int + Φ) <= 0.0 && (M_bound += ρ * dV)
    end
    return M_bound
end
