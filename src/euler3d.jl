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
#
# GPU: all hot loops dispatch to KA kernels (gpu_kernels.jl).
# Backend detected via KA.get_backend(U):
#   CPU Array  → KA.CPU()      (multithreaded, replaces Threads.@threads)
#   CuArray    → CUDABackend() (A100 PTX)

# ---------------------------------------------------------------------------
# Boundary conditions
#
# Ghost fills use view-based broadcasting — compatible with CPU Array and CuArray.

"""
    fill_ghost_3d_outflow!(U, nx, ny, nz)

Zero-gradient (copy-boundary-cell) ghost fill in all six face directions.
Order: x faces first, then y, then z — corners are filled correctly.
"""
function fill_ghost_3d_outflow!(U, nx::Int, ny::Int, nz::Int)
    ng = NG
    # x faces
    @views U[:, 1:ng,          :, :] .= U[:, ng+1:ng+1,  :, :]
    @views U[:, ng+nx+1:ng+nx+ng, :, :] .= U[:, ng+nx:ng+nx, :, :]
    # y faces (includes x-ghost columns already filled)
    @views U[:, :, 1:ng,          :] .= U[:, :, ng+1:ng+1,  :]
    @views U[:, :, ng+ny+1:ng+ny+ng, :] .= U[:, :, ng+ny:ng+ny, :]
    # z faces (includes x- and y-ghost cells)
    @views U[:, :, :, 1:ng         ] .= U[:, :, :, ng+1:ng+1  ]
    @views U[:, :, :, ng+nz+1:ng+nz+ng] .= U[:, :, :, ng+nz:ng+nz]
end

"""
    fill_ghost_3d_periodic!(U, nx, ny, nz)

Periodic ghost fill in all six face directions.
"""
function fill_ghost_3d_periodic!(U, nx::Int, ny::Int, nz::Int)
    ng = NG
    # x: left ghost ← last ng active cells; right ghost ← first ng active cells
    @views U[:, 1:ng,              :, :] .= U[:, nx+1:nx+ng,     :, :]
    @views U[:, ng+nx+1:ng+nx+ng,  :, :] .= U[:, ng+1:ng+ng,     :, :]
    # y
    @views U[:, :, 1:ng,              :] .= U[:, :, ny+1:ny+ng,    :]
    @views U[:, :, ng+ny+1:ng+ny+ng,  :] .= U[:, :, ng+1:ng+ng,    :]
    # z
    @views U[:, :, :, 1:ng             ] .= U[:, :, :, nz+1:nz+ng    ]
    @views U[:, :, :, ng+nz+1:ng+nz+ng] .= U[:, :, :, ng+1:ng+ng     ]
end

# ---------------------------------------------------------------------------
# Floor application

"""
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

Clamp density ≥ ρ_floor and pressure ≥ P_floor in all active cells.
Dispatches to KA kernel for CPU/GPU portability.
"""
function apply_floors_3d!(U, nx::Int, ny::Int, nz::Int,
                           ρ_floor::Real, P_floor::Real, γ::Real)
    backend = KA.get_backend(U)
    kern = _apply_floors_kernel!(backend, _WGSIZE_3D)
    kern(U, nx, ny, nz, Float64(ρ_floor), Float64(P_floor), Float64(γ);
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
end

# ---------------------------------------------------------------------------
# WENO5+HLLC flux computation — helpers for pressure reconstruction

# Helper: cell-average pressure, guaranteed ≥ 0.
@inline function _cell_pressure(ρ, mx, my, mz, E, γ)
    ρ  = max(ρ, 1e-30)
    KE = 0.5 * (mx^2 + my^2 + mz^2) / ρ
    return max((γ - 1) * (E - KE), 0.0)
end

# Reconstruct total energy from pressure + reconstructed primitives.
@inline function _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
    ρ_pos = max(ρL, 1e-30)
    KE    = ρL > 0.0 ? 0.5 * (mxL^2 + myL^2 + mzL^2) / ρ_pos : 0.0
    return PL / (γ - 1) + KE
end

# x-direction fluxes — dispatches to KA kernel.
function _weno_fluxes_x!(Fx, U, nx::Int, ny::Int, nz::Int, γ::Real)
    backend = KA.get_backend(U)
    kern = _weno_fluxes_x_kernel!(backend, _WGSIZE_3D)
    kern(Fx, U, nx, ny, nz, Float64(γ); ndrange = (nx+1, ny, nz))
    KA.synchronize(backend)
end

# y-direction fluxes — dispatches to KA kernel.
function _weno_fluxes_y!(Fy, U, nx::Int, ny::Int, nz::Int, γ::Real)
    backend = KA.get_backend(U)
    kern = _weno_fluxes_y_kernel!(backend, _WGSIZE_3D)
    kern(Fy, U, nx, ny, nz, Float64(γ); ndrange = (nx, ny+1, nz))
    KA.synchronize(backend)
end

# z-direction fluxes — dispatches to KA kernel.
function _weno_fluxes_z!(Fz, U, nx::Int, ny::Int, nz::Int, γ::Real)
    backend = KA.get_backend(U)
    kern = _weno_fluxes_z_kernel!(backend, _WGSIZE_3D)
    kern(Fz, U, nx, ny, nz, Float64(γ); ndrange = (nx, ny, nz+1))
    KA.synchronize(backend)
end

# ---------------------------------------------------------------------------
# Flux divergence

function _flux_divergence_3d!(dU, Fx, Fy, Fz,
                               nx::Int, ny::Int, nz::Int,
                               dx::Real, dy::Real, dz::Real)
    fill!(dU, zero(eltype(dU)))
    backend = KA.get_backend(dU)
    kern = _flux_divergence_kernel!(backend, _WGSIZE_3D)
    kern(dU, Fx, Fy, Fz, nx, ny, nz,
         1.0/Float64(dx), 1.0/Float64(dy), 1.0/Float64(dz);
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
end

# ---------------------------------------------------------------------------
# CFL timestep

"""
    cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, cfl) -> dt

CFL-limited timestep for 3D adiabatic Euler.
dt = cfl × min over active cells of min(dx,dy,dz) / (|v| + cs).
Dispatches to KA kernel; final maximum reduction uses Base.maximum.
"""
function cfl_dt_3d(U, nx::Int, ny::Int, nz::Int,
                   dx::Real, dy::Real, dz::Real,
                   γ::Real, cfl::Real)
    backend = KA.get_backend(U)
    speeds  = KA.zeros(backend, Float64, nx, ny, nz)
    kern = _cfl_speeds_kernel!(backend, _WGSIZE_3D)
    kern(speeds, U, nx, ny, nz, Float64(γ),
         1.0/Float64(dx), 1.0/Float64(dy), 1.0/Float64(dz);
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
    smax = maximum(speeds)
    return smax > 0.0 ? cfl / smax : Inf
end

# ---------------------------------------------------------------------------
# RHS function (method of lines operator) — used by euler3d_step! and fmr3d.

"""
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                 bc=:outflow, ρ_floor=0.0, P_floor=0.0, fill_ghosts=true)

Compute the method-of-lines RHS: dU = L(U) = −∂F/∂x − ∂G/∂y − ∂H/∂z.
Fills `Fx`, `Fy`, `Fz` (WENO5+HLLC fluxes) and `dU` (flux divergence).
When `fill_ghosts=false`, skips the BC ghost fill (caller is responsible).
"""
function euler3d_rhs!(dU, Fx, Fy, Fz, U,
                      nx::Int, ny::Int, nz::Int,
                      dx::Real, dy::Real, dz::Real, γ::Real;
                      bc::Symbol     = :outflow,
                      ρ_floor::Real  = 0.0,
                      P_floor::Real  = 0.0,
                      fill_ghosts::Bool = true)
    if fill_ghosts
        if bc === :periodic
            fill_ghost_3d_periodic!(U, nx, ny, nz)
        else
            fill_ghost_3d_outflow!(U, nx, ny, nz)
        end
    end
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)
    fill!(Fx, zero(eltype(U)))
    fill!(Fy, zero(eltype(U)))
    fill!(Fz, zero(eltype(U)))
    _weno_fluxes_x!(Fx, U, nx, ny, nz, γ)
    _weno_fluxes_y!(Fy, U, nx, ny, nz, γ)
    _weno_fluxes_z!(Fz, U, nx, ny, nz, γ)
    _flux_divergence_3d!(dU, Fx, Fy, Fz, nx, ny, nz, dx, dy, dz)
    return nothing
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
Works with CPU Array or CuArray — flux buffers allocated via `similar`.
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

    # similar allocates on the same device as U (CPU Array → Array, CuArray → CuArray)
    Fx = similar(U, 5, nxtot+1, nytot,   nztot  )
    Fy = similar(U, 5, nxtot,   nytot+1, nztot  )
    Fz = similar(U, 5, nxtot,   nytot,   nztot+1)
    dU = similar(U, 5, nxtot,   nytot,   nztot  )

    Un = copy(U)

    # SSP-RK3 (Shu-Osher).  Floors after each stage combination prevent
    # WENO5 from seeing ρ < 0 between calls.

    # Stage 1: u⁽¹⁾ = uⁿ + dt L(uⁿ)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                 bc, ρ_floor, P_floor)
    @. U = Un + dt * dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 2: u⁽²⁾ = ¾ uⁿ + ¼ u⁽¹⁾ + ¼ dt L(u⁽¹⁾)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                 bc, ρ_floor, P_floor)
    @. U = 0.75 * Un + 0.25 * U + 0.25 * dt * dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 3: uⁿ⁺¹ = ⅓ uⁿ + ⅔ u⁽²⁾ + ⅔ dt L(u⁽²⁾)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                 bc, ρ_floor, P_floor)
    @. U = (1/3) * Un + (2/3) * U + (2/3) * dt * dU
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

    # Background state.
    fill!(U, 0.0)
    E_bg = P_floor / (γ - 1)
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
            U[5, i, j, k] += dE_cell
        end
    end
end
