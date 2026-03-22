# 3D adiabatic Euler equations on a uniform Cartesian grid.
# Conserved variables: U[q,i,j,k], q ∈ 1:5
#   1 = ρ,  2 = ρvx,  3 = ρvy,  4 = ρvz,  5 = E
# EOS: P = (γ−1)(E − ½ρ|v|²)
#
# Ghost cells: NG = 3 on each face.
# Active cells: i ∈ NG+1:NG+nx, j ∈ NG+1:NG+ny, k ∈ NG+1:NG+nz.
# Total array size: (5, nx+2*NG, ny+2*NG, nz+2*NG).
#
# Spatial operator (method of lines, unsplit):
#   L(U) = −∂F/∂x − ∂G/∂y − ∂H/∂z
# F, G, H computed via WENO5-Z reconstruction + HLLC (hllc.jl).
#
# Time integration: SSP-RK3 via rk3_step! (rk3.jl).
# Positivity: density floor + pressure floor applied at each RK stage.

# ---------------------------------------------------------------------------
# Boundary conditions

"""
    fill_ghost_3d_outflow!(U, nx, ny, nz)

Zero-gradient (copy-boundary-cell) ghost fill in all six face directions.
Order: x faces first, then y, then z — corners are filled correctly.
"""
function fill_ghost_3d_outflow!(U, nx::Int, ny::Int, nz::Int)
    ng    = NG
    nxtot = nx + 2*ng
    nytot = ny + 2*ng
    nztot = nz + 2*ng

    # x faces
    for k in 1:nztot, j in 1:nytot
        for q in 1:5, m in 1:ng
            U[q, m,       j, k] = U[q, ng+1,  j, k]
            U[q, ng+nx+m, j, k] = U[q, ng+nx, j, k]
        end
    end
    # y faces (includes x-ghost columns)
    for k in 1:nztot, i in 1:nxtot
        for q in 1:5, m in 1:ng
            U[q, i, m,       k] = U[q, i, ng+1,  k]
            U[q, i, ng+ny+m, k] = U[q, i, ng+ny, k]
        end
    end
    # z faces (includes x- and y-ghost cells)
    for j in 1:nytot, i in 1:nxtot
        for q in 1:5, m in 1:ng
            U[q, i, j, m      ] = U[q, i, j, ng+1 ]
            U[q, i, j, ng+nz+m] = U[q, i, j, ng+nz]
        end
    end
end

"""
    fill_ghost_3d_periodic!(U, nx, ny, nz)

Periodic ghost fill in all six face directions.
"""
function fill_ghost_3d_periodic!(U, nx::Int, ny::Int, nz::Int)
    ng    = NG
    nxtot = nx + 2*ng
    nytot = ny + 2*ng
    nztot = nz + 2*ng

    for k in 1:nztot, j in 1:nytot
        for q in 1:5, m in 1:ng
            U[q, m,       j, k] = U[q, ng+nx-(ng-m), j, k]
            U[q, ng+nx+m, j, k] = U[q, ng+m,         j, k]
        end
    end
    for k in 1:nztot, i in 1:nxtot
        for q in 1:5, m in 1:ng
            U[q, i, m,       k] = U[q, i, ng+ny-(ng-m), k]
            U[q, i, ng+ny+m, k] = U[q, i, ng+m,         k]
        end
    end
    for j in 1:nytot, i in 1:nxtot
        for q in 1:5, m in 1:ng
            U[q, i, j, m      ] = U[q, i, j, ng+nz-(ng-m)]
            U[q, i, j, ng+nz+m] = U[q, i, j, ng+m        ]
        end
    end
end

# ---------------------------------------------------------------------------
# Floor application

"""
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

Clamp density ≥ ρ_floor and pressure ≥ P_floor in all active cells.
If P < P_floor, internal energy is raised while keeping momentum fixed.
Applied at the beginning of each SSP-RK3 stage.
"""
function apply_floors_3d!(U, nx::Int, ny::Int, nz::Int,
                           ρ_floor::Real, P_floor::Real, γ::Real)
    ng = NG
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ = U[1, i, j, k]
        if ρ < ρ_floor
            # Unphysical density: reset to full floor state (zero velocity).
            # Keeping the original momentum with ρ = ρ_floor gives v = m/ρ_floor
            # which can be enormous, causing WENO5 to reconstruct energy spikes.
            U[1, i, j, k] = ρ_floor
            U[2, i, j, k] = 0.0
            U[3, i, j, k] = 0.0
            U[4, i, j, k] = 0.0
            U[5, i, j, k] = P_floor / (γ - 1)
        else
            KE = 0.5 * (U[2,i,j,k]^2 + U[3,i,j,k]^2 + U[4,i,j,k]^2) / ρ
            P  = (γ - 1) * (U[5, i, j, k] - KE)
            if P < P_floor
                U[5, i, j, k] = P_floor / (γ - 1) + KE
            end
        end
    end
end

# ---------------------------------------------------------------------------
# WENO5+HLLC flux computation

# Helper: cell-average pressure, guaranteed ≥ 0.
@inline function _cell_pressure(ρ, mx, my, mz, E, γ)
    ρ  = max(ρ, 1e-30)
    KE = 0.5 * (mx^2 + my^2 + mz^2) / ρ
    return max((γ - 1) * (E - KE), 0.0)
end

# Reconstruct total energy from pressure + reconstructed primitives.
# Ensures E ≥ KE ≥ 0 by construction (prevents WENO5 overshoot of E at
# strong blast-background discontinuities where E contrast can exceed 3×10⁶).
@inline function _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
    ρ_pos = max(ρL, 1e-30)
    KE    = ρL > 0.0 ? 0.5 * (mxL^2 + myL^2 + mzL^2) / ρ_pos : 0.0
    return PL / (γ - 1) + KE
end

# x-direction fluxes.
# Fx[q, i, j, k] = flux at right face of cell (i,j,k), i.e., between i and i+1.
# Computed for i ∈ NG:NG+nx (nx+1 interfaces).
function _weno_fluxes_x!(Fx, U, nx::Int, ny::Int, nz::Int, γ::Real)
    ng = NG
    Threads.@threads :static for k in ng+1:ng+nz
        for j in ng+1:ng+ny, i in ng:ng+nx
            ρL  = weno5_left( U[1,i-2,j,k],U[1,i-1,j,k],U[1,i,j,k],U[1,i+1,j,k],U[1,i+2,j,k])
            ρR  = weno5_right(U[1,i-1,j,k],U[1,i,j,k],U[1,i+1,j,k],U[1,i+2,j,k],U[1,i+3,j,k])
            mxL = weno5_left( U[2,i-2,j,k],U[2,i-1,j,k],U[2,i,j,k],U[2,i+1,j,k],U[2,i+2,j,k])
            mxR = weno5_right(U[2,i-1,j,k],U[2,i,j,k],U[2,i+1,j,k],U[2,i+2,j,k],U[2,i+3,j,k])
            myL = weno5_left( U[3,i-2,j,k],U[3,i-1,j,k],U[3,i,j,k],U[3,i+1,j,k],U[3,i+2,j,k])
            myR = weno5_right(U[3,i-1,j,k],U[3,i,j,k],U[3,i+1,j,k],U[3,i+2,j,k],U[3,i+3,j,k])
            mzL = weno5_left( U[4,i-2,j,k],U[4,i-1,j,k],U[4,i,j,k],U[4,i+1,j,k],U[4,i+2,j,k])
            mzR = weno5_right(U[4,i-1,j,k],U[4,i,j,k],U[4,i+1,j,k],U[4,i+2,j,k],U[4,i+3,j,k])
            # Reconstruct pressure via WENO5 on cell-average P; clamp to stencil
            # range to prevent overshoot, then reconstruct E = P/(γ-1) + KE.
            pi2 = _cell_pressure(U[1,i-2,j,k],U[2,i-2,j,k],U[3,i-2,j,k],U[4,i-2,j,k],U[5,i-2,j,k],γ)
            pi1 = _cell_pressure(U[1,i-1,j,k],U[2,i-1,j,k],U[3,i-1,j,k],U[4,i-1,j,k],U[5,i-1,j,k],γ)
            p0  = _cell_pressure(U[1,i  ,j,k],U[2,i  ,j,k],U[3,i  ,j,k],U[4,i  ,j,k],U[5,i  ,j,k],γ)
            pp1 = _cell_pressure(U[1,i+1,j,k],U[2,i+1,j,k],U[3,i+1,j,k],U[4,i+1,j,k],U[5,i+1,j,k],γ)
            pp2 = _cell_pressure(U[1,i+2,j,k],U[2,i+2,j,k],U[3,i+2,j,k],U[4,i+2,j,k],U[5,i+2,j,k],γ)
            pp3 = _cell_pressure(U[1,i+3,j,k],U[2,i+3,j,k],U[3,i+3,j,k],U[4,i+3,j,k],U[5,i+3,j,k],γ)
            P_maxL = max(pi2, pi1, p0, pp1, pp2)
            P_maxR = max(pi1, p0,  pp1, pp2, pp3)
            PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
            PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
            EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
            ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
            Fx[1,i,j,k],Fx[2,i,j,k],Fx[3,i,j,k],Fx[4,i,j,k],Fx[5,i,j,k] =
                hllc_flux_x(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
        end
    end
end

# y-direction fluxes.
function _weno_fluxes_y!(Fy, U, nx::Int, ny::Int, nz::Int, γ::Real)
    ng = NG
    Threads.@threads :static for k in ng+1:ng+nz
        for j in ng:ng+ny, i in ng+1:ng+nx
            ρL  = weno5_left( U[1,i,j-2,k],U[1,i,j-1,k],U[1,i,j,k],U[1,i,j+1,k],U[1,i,j+2,k])
            ρR  = weno5_right(U[1,i,j-1,k],U[1,i,j,k],U[1,i,j+1,k],U[1,i,j+2,k],U[1,i,j+3,k])
            mxL = weno5_left( U[2,i,j-2,k],U[2,i,j-1,k],U[2,i,j,k],U[2,i,j+1,k],U[2,i,j+2,k])
            mxR = weno5_right(U[2,i,j-1,k],U[2,i,j,k],U[2,i,j+1,k],U[2,i,j+2,k],U[2,i,j+3,k])
            myL = weno5_left( U[3,i,j-2,k],U[3,i,j-1,k],U[3,i,j,k],U[3,i,j+1,k],U[3,i,j+2,k])
            myR = weno5_right(U[3,i,j-1,k],U[3,i,j,k],U[3,i,j+1,k],U[3,i,j+2,k],U[3,i,j+3,k])
            mzL = weno5_left( U[4,i,j-2,k],U[4,i,j-1,k],U[4,i,j,k],U[4,i,j+1,k],U[4,i,j+2,k])
            mzR = weno5_right(U[4,i,j-1,k],U[4,i,j,k],U[4,i,j+1,k],U[4,i,j+2,k],U[4,i,j+3,k])
            pi2 = _cell_pressure(U[1,i,j-2,k],U[2,i,j-2,k],U[3,i,j-2,k],U[4,i,j-2,k],U[5,i,j-2,k],γ)
            pi1 = _cell_pressure(U[1,i,j-1,k],U[2,i,j-1,k],U[3,i,j-1,k],U[4,i,j-1,k],U[5,i,j-1,k],γ)
            p0  = _cell_pressure(U[1,i,j  ,k],U[2,i,j  ,k],U[3,i,j  ,k],U[4,i,j  ,k],U[5,i,j  ,k],γ)
            pp1 = _cell_pressure(U[1,i,j+1,k],U[2,i,j+1,k],U[3,i,j+1,k],U[4,i,j+1,k],U[5,i,j+1,k],γ)
            pp2 = _cell_pressure(U[1,i,j+2,k],U[2,i,j+2,k],U[3,i,j+2,k],U[4,i,j+2,k],U[5,i,j+2,k],γ)
            pp3 = _cell_pressure(U[1,i,j+3,k],U[2,i,j+3,k],U[3,i,j+3,k],U[4,i,j+3,k],U[5,i,j+3,k],γ)
            P_maxL = max(pi2, pi1, p0, pp1, pp2)
            P_maxR = max(pi1, p0,  pp1, pp2, pp3)
            PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
            PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
            EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
            ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
            Fy[1,i,j,k],Fy[2,i,j,k],Fy[3,i,j,k],Fy[4,i,j,k],Fy[5,i,j,k] =
                hllc_flux_y(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
        end
    end
end

# z-direction fluxes.
function _weno_fluxes_z!(Fz, U, nx::Int, ny::Int, nz::Int, γ::Real)
    ng = NG
    Threads.@threads :static for k in ng:ng+nz
        for j in ng+1:ng+ny, i in ng+1:ng+nx
            ρL  = weno5_left( U[1,i,j,k-2],U[1,i,j,k-1],U[1,i,j,k],U[1,i,j,k+1],U[1,i,j,k+2])
            ρR  = weno5_right(U[1,i,j,k-1],U[1,i,j,k],U[1,i,j,k+1],U[1,i,j,k+2],U[1,i,j,k+3])
            mxL = weno5_left( U[2,i,j,k-2],U[2,i,j,k-1],U[2,i,j,k],U[2,i,j,k+1],U[2,i,j,k+2])
            mxR = weno5_right(U[2,i,j,k-1],U[2,i,j,k],U[2,i,j,k+1],U[2,i,j,k+2],U[2,i,j,k+3])
            myL = weno5_left( U[3,i,j,k-2],U[3,i,j,k-1],U[3,i,j,k],U[3,i,j,k+1],U[3,i,j,k+2])
            myR = weno5_right(U[3,i,j,k-1],U[3,i,j,k],U[3,i,j,k+1],U[3,i,j,k+2],U[3,i,j,k+3])
            mzL = weno5_left( U[4,i,j,k-2],U[4,i,j,k-1],U[4,i,j,k],U[4,i,j,k+1],U[4,i,j,k+2])
            mzR = weno5_right(U[4,i,j,k-1],U[4,i,j,k],U[4,i,j,k+1],U[4,i,j,k+2],U[4,i,j,k+3])
            pi2 = _cell_pressure(U[1,i,j,k-2],U[2,i,j,k-2],U[3,i,j,k-2],U[4,i,j,k-2],U[5,i,j,k-2],γ)
            pi1 = _cell_pressure(U[1,i,j,k-1],U[2,i,j,k-1],U[3,i,j,k-1],U[4,i,j,k-1],U[5,i,j,k-1],γ)
            p0  = _cell_pressure(U[1,i,j,k  ],U[2,i,j,k  ],U[3,i,j,k  ],U[4,i,j,k  ],U[5,i,j,k  ],γ)
            pp1 = _cell_pressure(U[1,i,j,k+1],U[2,i,j,k+1],U[3,i,j,k+1],U[4,i,j,k+1],U[5,i,j,k+1],γ)
            pp2 = _cell_pressure(U[1,i,j,k+2],U[2,i,j,k+2],U[3,i,j,k+2],U[4,i,j,k+2],U[5,i,j,k+2],γ)
            pp3 = _cell_pressure(U[1,i,j,k+3],U[2,i,j,k+3],U[3,i,j,k+3],U[4,i,j,k+3],U[5,i,j,k+3],γ)
            P_maxL = max(pi2, pi1, p0, pp1, pp2)
            P_maxR = max(pi1, p0,  pp1, pp2, pp3)
            PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
            PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
            EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
            ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
            Fz[1,i,j,k],Fz[2,i,j,k],Fz[3,i,j,k],Fz[4,i,j,k],Fz[5,i,j,k] =
                hllc_flux_z(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
        end
    end
end

# ---------------------------------------------------------------------------
# Flux divergence

function _flux_divergence_3d!(dU, Fx, Fy, Fz,
                               nx::Int, ny::Int, nz::Int,
                               dx::Real, dy::Real, dz::Real)
    ng = NG
    fill!(dU, zero(eltype(dU)))
    inv_dx = 1.0 / dx;  inv_dy = 1.0 / dy;  inv_dz = 1.0 / dz
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        for q in 1:5
            dU[q,i,j,k] = -(Fx[q,i,j,k] - Fx[q,i-1,j,k]) * inv_dx -
                            (Fy[q,i,j,k] - Fy[q,i,j-1,k]) * inv_dy -
                            (Fz[q,i,j,k] - Fz[q,i,j,k-1]) * inv_dz
        end
    end
end

# ---------------------------------------------------------------------------
# CFL timestep

"""
    cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, cfl) -> dt

CFL-limited timestep for 3D adiabatic Euler.
dt = cfl × min over active cells of min(dx,dy,dz) / (|v| + cs).
"""
function cfl_dt_3d(U, nx::Int, ny::Int, nz::Int,
                   dx::Real, dy::Real, dz::Real,
                   γ::Real, cfl::Real)
    ng = NG
    smax = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ  = U[1, i, j, k]
        ρ  = max(ρ, 1e-30)
        vx = U[2, i, j, k] / ρ
        vy = U[3, i, j, k] / ρ
        vz = U[4, i, j, k] / ρ
        KE = 0.5 * (vx^2 + vy^2 + vz^2)
        P  = max((γ - 1) * (U[5, i, j, k] / ρ - KE) * ρ, 0.0)
        cs = sqrt(γ * P / ρ)
        sx = (abs(vx) + cs) / dx
        sy = (abs(vy) + cs) / dy
        sz = (abs(vz) + cs) / dz
        s  = max(sx, sy, sz)
        smax = max(smax, s)
    end
    return smax > 0.0 ? cfl / smax : Inf
end

# ---------------------------------------------------------------------------
# Main step function

"""
    euler3d_step!(U, nx, ny, nz, dx, dy, dz, dt, γ;
                  bc=:outflow, ρ_floor=0.0, P_floor=0.0)

Advance 3D adiabatic Euler by one SSP-RK3 step.
`U` has size `(5, nx+2*NG, ny+2*NG, nz+2*NG)`.
`bc` ∈ {:outflow, :periodic}.
Floors applied at the start of each RK stage.
"""
function euler3d_step!(U, nx::Int, ny::Int, nz::Int,
                       dx::Real, dy::Real, dz::Real, dt::Real, γ::Real;
                       bc::Symbol    = :outflow,
                       ρ_floor::Real = 0.0,
                       P_floor::Real = 0.0)
    ng    = NG
    nxtot = nx + 2*ng
    nytot = ny + 2*ng
    nztot = nz + 2*ng

    # Pre-allocate scratch (avoid allocations inside L!).
    Fx = zeros(eltype(U), 5, nxtot+1, nytot,   nztot  )
    Fy = zeros(eltype(U), 5, nxtot,   nytot+1, nztot  )
    Fz = zeros(eltype(U), 5, nxtot,   nytot,   nztot+1)
    dU = zeros(eltype(U), 5, nxtot,   nytot,   nztot  )

    fill_bc! = bc === :periodic ?
        (V -> fill_ghost_3d_periodic!(V, nx, ny, nz)) :
        (V -> fill_ghost_3d_outflow!( V, nx, ny, nz))

    function L!(dV, V)
        fill_bc!(V)
        apply_floors_3d!(V, nx, ny, nz, ρ_floor, P_floor, γ)
        fill!(Fx, zero(eltype(U)))
        fill!(Fy, zero(eltype(U)))
        fill!(Fz, zero(eltype(U)))
        _weno_fluxes_x!(Fx, V, nx, ny, nz, γ)
        _weno_fluxes_y!(Fy, V, nx, ny, nz, γ)
        _weno_fluxes_z!(Fz, V, nx, ny, nz, γ)
        _flux_divergence_3d!(dV, Fx, Fy, Fz, nx, ny, nz, dx, dy, dz)
    end

    tmp1 = similar(U)
    Un   = similar(U)
    Un  .= U

    # SSP-RK3 (Shu-Osher) with explicit positivity floors after each stage
    # combination.  apply_floors_3d! is also called at the *start* of L!, but
    # the stage combinations U = ... can introduce ρ < 0 between L! calls.
    # Flooring after each combination prevents WENO5 from seeing unphysical
    # states (near-zero ρ with large momentum → enormous velocity spikes).

    # Stage 1: u⁽¹⁾ = uⁿ + dt L(uⁿ)
    L!(tmp1, U)
    @. U = Un + dt * tmp1
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 2: u⁽²⁾ = ¾ uⁿ + ¼ u⁽¹⁾ + ¼ dt L(u⁽¹⁾)
    L!(tmp1, U)
    @. U = 0.75 * Un + 0.25 * U + 0.25 * dt * tmp1
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 3: uⁿ⁺¹ = ⅓ uⁿ + ⅔ u⁽²⁾ + ⅔ dt L(u⁽²⁾)
    L!(tmp1, U)
    @. U = (1/3) * Un + (2/3) * U + (2/3) * dt * tmp1
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    return nothing
end

# ---------------------------------------------------------------------------
# Sedov-Taylor initial condition (point explosion at domain centre).
# Used in tests and in the thermal-bomb phase (Phase 5).

"""
    sedov_ic_3d!(U, nx, ny, nz, dx, dy, dz, γ;
                 E_blast=1.0, r_inject=nothing,
                 ρ_bg=1.0, P_floor=1e-5)

Initialise a Sedov-Taylor point explosion centred at the origin.
Energy `E_blast` is deposited uniformly in cells within radius `r_inject`
(default: 2.5 × min(dx,dy,dz)).  Background: ρ = ρ_bg, P = P_floor.

The domain origin corresponds to array index (NG + nx÷2 + 1, ...) when the
grid is centred, or (NG+1, ...) when the corner is at the origin.
Pass `x_offset, y_offset, z_offset` for the centre in physical coordinates.
"""
function sedov_ic_3d!(U, nx::Int, ny::Int, nz::Int,
                      dx::Real, dy::Real, dz::Real, γ::Real;
                      E_blast::Real   = 1.0,
                      r_inject        = nothing,
                      ρ_bg::Real      = 1.0,
                      P_floor::Real   = 1e-5,
                      x_offset::Real  = 0.0,
                      y_offset::Real  = 0.0,
                      z_offset::Real  = 0.0)
    ng = NG
    r_inj = isnothing(r_inject) ? 2.5 * min(dx, dy, dz) : Float64(r_inject)

    # x_offset, y_offset, z_offset are the physical coordinates of the LEFT edge
    # of the active domain.  Cell centre of active cell i (1-indexed within active
    # region) is at x_offset + (i - 0.5)*dx.  Set offsets to -L for a [-L,L]³ domain.

    # Background state.
    fill!(U, 0.0)
    E_bg = P_floor / (γ - 1)    # internal energy density at background P
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = ρ_bg
        U[5, i, j, k] = E_bg
    end

    # Find injection cells and total injection volume.
    V_inj = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = (i - ng - 0.5) * dx + x_offset
        yc = (j - ng - 0.5) * dy + y_offset
        zc = (k - ng - 0.5) * dz + z_offset
        r  = sqrt(xc^2 + yc^2 + zc^2)
        r <= r_inj && (V_inj += dx * dy * dz)
    end

    # Deposit E_blast uniformly in injection sphere.
    dE_cell = (V_inj > 0.0) ? E_blast * (γ - 1) / V_inj : 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = (i - ng - 0.5) * dx + x_offset
        yc = (j - ng - 0.5) * dy + y_offset
        zc = (k - ng - 0.5) * dz + z_offset
        r  = sqrt(xc^2 + yc^2 + zc^2)
        if r <= r_inj
            U[5, i, j, k] += dE_cell   # add pressure / (γ-1) to E
        end
    end
end
