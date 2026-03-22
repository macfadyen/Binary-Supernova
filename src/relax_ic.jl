# Roche potential relaxation initial conditions for BinarySupernova.
#
# Starting from a spherical polytrope (polytrope_ic_3d!) in the full binary
# potential, velocity damping drives the gas to the tidal equilibrium (Roche)
# shape.  The relaxed state is then used as the pre-explosion IC.
#
# Algorithm (CLAUDE.md §6.3):
#   1. Caller places polytrope on grid via polytrope_ic_3d!.
#   2. relax_ic! evolves with SSP-RK3 using:
#        hydro RHS  +  BH gravity  +  [self-gravity]  +  velocity damping
#      d(ρv)/dt += −ρ (v − v_rot) / t_damp        [damps residual oscillations]
#      d(E)/dt  += −ρ v · (v − v_rot) / t_damp    [removes KE, conserves E_th]
#      where v_rot = Ω × r = (−Ω y, Ω x, 0) is the co-rotation velocity.
#      For Ω = 0 (default), this reduces to simple velocity damping.
#   3. Stops when KE_gas / E_thermal < KE_tol or t ≥ t_max.
#   4. Returns (t, n_steps, KE_ratio) — caller verifies stability.
#
# Energy accounting: the damping force removes KE at rate 2 KE / t_damp and
# subtracts the same amount from the total energy → E_thermal is conserved
# exactly.  KE decays as exp(−2 t / t_damp) → use t_damp ≈ 0.1 P₀ to reach
# KE < 1% E_th in ≈ 0.23 P₀.
#
# Co-rotating frame (Ω > 0): damping drives gas toward solid-body rotation at
# Ω, which is the expected equilibrium in the binary co-rotating frame.  BH1
# should be held at a fixed position (or moved by nbody_step! each step) while
# the gas settles.  Initialize the star with v_y = Ω × r_com for the stellar
# CoM to avoid large secular CoM drift.
#
# Self-gravity (Phase 7, optional): stabilises the star in hydrostatic
# equilibrium between relaxation steps and gives the correct Roche shape.
# Recommended for production runs (self_gravity = true).
#
# Reference: CLAUDE.md §6.3; Dempsey+ 2020 (damping analogue for BDs).

# ---------------------------------------------------------------------------
# Velocity damping source term

"""
    relax_damping_source!(dU, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, t_damp;
                          Ω = 0.0)

Add velocity-damping source to the RHS array `dU`:

  d(ρv)/dt += −ρ (v − v_rot) / t_damp
  d(E)/dt  += −ρ v · (v − v_rot) / t_damp

where v_rot = (−Ω y, Ω x, 0) is the co-rotation velocity at angular rate `Ω`.
For Ω = 0 this reduces to simple velocity damping d(ρv)/dt += −ρv / t_damp.

Thermal energy is conserved exactly: the work done by the damping force equals
the kinetic energy removed.  Uses view broadcasting for CPU/GPU portability.

Arguments:
- `x0, y0, z0` : physical left edges of the active domain (needed for cell
                  centres when Ω ≠ 0).
- `t_damp`      : damping timescale (code units).
- `Ω`           : co-rotation angular velocity (default 0).
"""
function relax_damping_source!(dU, U,
                                nx::Int, ny::Int, nz::Int,
                                dx::Real, dy::Real, dz::Real,
                                x0::Real, y0::Real, z0::Real,
                                t_damp::Float64;
                                Ω::Float64 = 0.0)
    ng = NG
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)

    ρ  = view(U, 1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mx = view(U, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    my = view(U, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    mz = view(U, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)

    dU_mx = view(dU, 2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    dU_my = view(dU, 3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    dU_mz = view(dU, 4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    dU_E  = view(dU, 5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)

    ρ_s = max.(ρ, 1e-30)
    inv_td = 1.0 / t_damp

    if Ω == 0.0
        # Simple velocity damping: damp all momentum toward zero.
        @. dU_mx -= mx * inv_td
        @. dU_my -= my * inv_td
        @. dU_mz -= mz * inv_td
        # Energy: work by force = v · f = -(mx² + my² + mz²) / (ρ t_damp)
        @. dU_E  -= (mx^2 + my^2 + mz^2) / (ρ_s * t_damp)
    else
        # Co-rotating damping: damp relative to solid-body rotation at Ω.
        # v_rot = (−Ω y, Ω x, 0);  cell centres are x0+(i−ng−0.5)dx, etc.
        backend = KA.get_backend(U)
        xc_cpu = reshape([x0 + (i - ng - 0.5)*fdx for i in ng+1:ng+nx], nx, 1, 1)
        yc_cpu = reshape([y0 + (j - ng - 0.5)*fdy for j in ng+1:ng+ny], 1, ny, 1)
        xc = adapt(backend, xc_cpu)
        yc = adapt(backend, yc_cpu)

        # Primitive velocities
        vx = @. mx / ρ_s
        vy = @. my / ρ_s
        vz = @. mz / ρ_s
        # Co-rotation velocity
        vrotx = @. -Ω * yc
        vroty = @.  Ω * xc
        # Relative velocity (residual to be damped)
        vrx = @. vx - vrotx
        vry = @. vy - vroty
        # No z-rotation component (2D rotation about z)
        @. dU_mx -= ρ_s * vrx * inv_td
        @. dU_my -= ρ_s * vry * inv_td
        @. dU_mz -= mz * inv_td        # vz relative to co-rotating frame = vz
        # Energy: v · f = −ρ (vx vrx + vy vry + vz vz) / t_damp
        @. dU_E  -= ρ_s * (vx*vrx + vy*vry + vz^2) * inv_td
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Main relaxation driver

"""
    relax_ic!(U, nx, ny, nz, dx, dy, dz, γ;
              bhs=[], x0=0.0, y0=0.0, z0=0.0,
              t_damp=0.1, t_max=2.0, cfl=0.4,
              ρ_floor=1e-10, P_floor=1e-10,
              KE_tol=0.01, Ω=0.0,
              self_gravity=false, verbose=false)
    -> NamedTuple (t, n_steps, KE_ratio)

Drive the gas in `U` toward tidal (Roche) equilibrium by velocity damping.

SSP-RK3 time integration uses the combined RHS:
  L(U) = euler_hydro + BH_gravity + [self_gravity] + relax_damping

Stops when KE_gas / E_thermal < `KE_tol` or t ≥ `t_max`.

Arguments:
- `bhs`          : Vector of BlackHole structs; held at fixed positions unless
                   the caller wraps `relax_ic!` with their own nbody_step! loop.
- `x0, y0, z0`   : physical left edges of the active domain.
- `t_damp`        : velocity damping timescale (code units; ≈ 0.1 P₀ recommended).
- `t_max`         : maximum relaxation time; exit early if KE_tol is met.
- `KE_tol`        : fractional KE threshold: KE / E_thermal < KE_tol → converged.
- `Ω`             : co-rotation angular velocity; 0.0 = simple damping.
- `self_gravity`  : include gas self-gravity via `add_self_gravity_source!`.
- `verbose`       : print convergence info every 20 steps.
"""
function relax_ic!(U, nx::Int, ny::Int, nz::Int,
                   dx::Real, dy::Real, dz::Real, γ::Real;
                   bhs          = BlackHole[],
                   x0::Real     = 0.0,
                   y0::Real     = 0.0,
                   z0::Real     = 0.0,
                   t_damp::Float64   = 0.1,
                   t_max::Float64    = 2.0,
                   cfl::Float64      = 0.4,
                   ρ_floor::Float64  = 1e-10,
                   P_floor::Float64  = 1e-10,
                   KE_tol::Float64   = 0.01,
                   Ω::Float64        = 0.0,
                   self_gravity::Bool = false,
                   verbose::Bool      = false)
    ng    = NG
    nxtot = nx + 2ng;  nytot = ny + 2ng;  nztot = nz + 2ng

    # Pre-allocate flux and RHS buffers (same device as U).
    Fx = similar(U, 5, nxtot+1, nytot,   nztot  )
    Fy = similar(U, 5, nxtot,   nytot+1, nztot  )
    Fz = similar(U, 5, nxtot,   nytot,   nztot+1)
    dU = similar(U, 5, nxtot,   nytot,   nztot  )
    Un = similar(U)

    t        = 0.0
    n_steps  = 0
    KE_ratio = Inf

    while t < t_max && KE_ratio > KE_tol
        dt = min(cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, cfl), t_max - t)
        Un .= U

        # ---- SSP-RK3 (Shu-Osher) with full source stack ----

        # Stage 1: U⁽¹⁾ = Uⁿ + dt L(Uⁿ)
        euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                     bc=:outflow, ρ_floor, P_floor)
        isempty(bhs) || add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz,
                                                bhs, x0, y0, z0)
        self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz)
        relax_damping_source!(dU, U, nx, ny, nz, dx, dy, dz, x0, y0, z0,
                               t_damp; Ω)
        @. U = Un + dt * dU
        apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

        # Stage 2: U⁽²⁾ = ¾ Uⁿ + ¼ (U⁽¹⁾ + dt L(U⁽¹⁾))
        euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                     bc=:outflow, ρ_floor, P_floor)
        isempty(bhs) || add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz,
                                                bhs, x0, y0, z0)
        self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz)
        relax_damping_source!(dU, U, nx, ny, nz, dx, dy, dz, x0, y0, z0,
                               t_damp; Ω)
        @. U = 0.75 * Un + 0.25 * U + 0.25 * dt * dU
        apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

        # Stage 3: Uⁿ⁺¹ = ⅓ Uⁿ + ⅔ (U⁽²⁾ + dt L(U⁽²⁾))
        euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                     bc=:outflow, ρ_floor, P_floor)
        isempty(bhs) || add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz,
                                                bhs, x0, y0, z0)
        self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz)
        relax_damping_source!(dU, U, nx, ny, nz, dx, dy, dz, x0, y0, z0,
                               t_damp; Ω)
        @. U = (1/3) * Un + (2/3) * U + (2/3) * dt * dU
        apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

        t       += dt
        n_steps += 1

        # Convergence check: KE / E_thermal.
        # E_thermal = E_total − KE is conserved by damping so this ratio
        # decreases monotonically if damping dominates forcing.
        KE    = gas_kinetic_total(U, nx, ny, nz, dx, dy, dz)
        E_tot = gas_energy_total(U, nx, ny, nz, dx, dy, dz)
        E_th  = max(E_tot - KE, 0.0)
        KE_ratio = E_th > 0.0 ? KE / E_th : 0.0

        if verbose && n_steps % 20 == 0
            @info "relax_ic!" t=round(t,digits=3) KE_ratio=round(KE_ratio,sigdigits=3) n_steps
        end
    end

    return (t=t, n_steps=n_steps, KE_ratio=KE_ratio)
end
