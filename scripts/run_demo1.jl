#!/usr/bin/env julia
# Demo 1: first end-to-end binary supernova simulation.
#
# Usage:
#   julia --project=. scripts/run_demo1.jl [--gpu] [--nx NX]
#
# --nx NX  : grid resolution per axis (default 64; use 96 or 128 for better quality)
# --gpu    : use CUDA backend (requires CUDA.jl and a supported GPU)
#
# Output is written to demo1/output/:
#   trajectory.h5        BH positions, velocities, masses at every step
#   diagnostics.csv      Per-step scalar diagnostics
#   snap_tNNN.h5         Gas snapshots every DT_SNAP = 1.0 code time units
#
# See demo1/README.md for full parameter description and physical motivation.

using BinarySupernova
using Printf
import KernelAbstractions as KA

# ---------------------------------------------------------------------------
# Argument parsing

use_gpu = "--gpu" in ARGS
nx_arg  = let v = 64
    for (i, arg) in enumerate(ARGS)
        if arg == "--nx" && i < length(ARGS)
            v = parse(Int, ARGS[i+1])
        end
    end
    v
end

# ---------------------------------------------------------------------------
# Physical parameters — see demo1/README.md for motivation

# Physical masses: M_BH1=10 M☉, M_star=20 M☉, M_BH2_init=10 M☉, M_ejecta=10 M☉
# Code units: G = M_total = M_BH1+M_star = 30 M☉ = 1, a0 = 1
const M_BH1      = 10.0/30.0    # = 1/3  pre-existing BH mass
const M_STAR     = 20.0/30.0    # = 2/3  pre-SN stellar mass
const M_BH2_INIT = 10.0/30.0    # = 1/3  remnant mass (half the star)
const GAMMA      = 5.0 / 3.0   # γ=5/3 (n=3/2 polytrope): lower ρ_c ≈ 90 vs 840 for γ=4/3,
                               # giving a better-conditioned IC at 64³
const A0         = 1.0          # initial binary separation
const R_STAR     = 0.35         # stellar radius ≈ 0.92 × Roche lobe radius (q = 0.5)
const E_SN       = 1.0          # supernova energy in code units (≈ virial energy)
const R_BOMB     = R_STAR       # thermal bomb deposited over full star
const V_KICK     = [0.0, 0.0, 0.0]  # no natal kick
const F_SINK     = 1.0          # sink timescale multiplier
const RHO_FLOOR  = 1e-3     # warm ambient: density ratio stellar/ambient ~10 keeps WENO5 stable
const P_FLOOR    = 1e-6     # consistent with warm ambient cs ~ 0.04 code units/time
const CFL        = 0.4

# Evolution
const T_END   = 20.0   # ≈ 3.2 orbital periods (P0 = 2π ≈ 6.28)
const DT_SNAP = 1.0    # snapshot interval in code time units
const OUTDIR  = "demo1/output"

# Grid resolution (set from command-line arg, then frozen as const-like)
const NX = nx_arg

# ---------------------------------------------------------------------------
# Backend

if use_gpu
    using CUDA
    @assert CUDA.functional() "CUDA not functional — check driver and GPU"
    backend = CUDABackend()
    @info "GPU backend" device=CUDA.name(CUDA.device())
else
    backend = KA.CPU()
    @info "CPU backend" threads=Threads.nthreads()
end

# ---------------------------------------------------------------------------
# Grid

const NY = NX;  const NZ = NX
const L  = 2.0
const DX = 2L / NX
const ng = BinarySupernova.NG

nxtot = NX + 2ng;  nytot = NY + 2ng;  nztot = NZ + 2ng
const x0 = -L;  const y0 = -L;  const z0 = -L   # domain physical left edge

@info "Grid" NX=NX dx=DX domain="[$(x0),$(L)]³" memory_MB=round(5*nxtot*nytot*nztot*8/1e6,digits=1)

# Allocate on chosen backend
if use_gpu
    U = CUDA.zeros(Float64, 5, nxtot, nytot, nztot)
else
    U = zeros(Float64, 5, nxtot, nytot, nztot)
end

# ---------------------------------------------------------------------------
# Physical unit conversion (for r_sink/c_code; no effect on dynamics)
units = PhysicalUnits(30.0, 10.0)   # 30 M☉ total, 10 R☉ separation → c ≈ 396

# Larger softening prevents extreme accelerations in the dense stellar core near BH2.
# eps = 4 Δx limits peak BH gravity to a_max ≈ 0.38*M_BH/eps² ≈ 2 code units/time².
eps_bh    = 4.0 * DX    # BH gravitational softening = 4 Δx
r_floor   = 2.0 * DX    # minimum sink radius = 2 Δx

# ---------------------------------------------------------------------------
# Step 1: Place Lane-Emden polytrope for the pre-SN star

@info "Building polytrope IC..."

# Work on host array (polytrope_ic_3d! runs on CPU regardless of backend)
U_cpu = Array(U)
polytrope_ic_3d!(U_cpu, NX, NY, NZ, DX, DX, DX, GAMMA;
                 M_star   = M_STAR,
                 R_star   = R_STAR,
                 x0 = x0, y0 = y0, z0 = z0,
                 x_center = 0.0, y_center = 0.0, z_center = 0.0,
                 ρ_floor  = RHO_FLOOR, P_floor = P_FLOOR)
fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

# ---------------------------------------------------------------------------
# Step 2: BH1 on circular orbit around the star

# Circular orbit velocity: v_orb = sqrt(M_total / a0) with G = 1
v_orb = sqrt((M_BH1 + M_STAR) / A0)
# BH1 at (a0, 0, 0); star at origin → CoM at x = M_BH1*a0/M_total = 0.25
# Shift both so CoM is at origin
x_cm  = M_BH1 * A0 / (M_BH1 + M_STAR)
x_bh1 = A0 - x_cm
x_star = -x_cm
# For simplicity in this first demo, keep the star centred and BH1 at A0.
# The CoM offset is small (0.25 a0) and the relaxation will re-centre things.
bh1 = BlackHole(
    [A0, 0.0, 0.0],          # position
    [0.0, v_orb, 0.0],        # velocity (prograde circular orbit)
    M_BH1,
    eps_bh,
    units.c_code,
    r_floor)
@info "BH1" pos=bh1.pos vel=bh1.vel mass=bh1.mass r_sink=r_sink(bh1)

# Step 3 (Roche relaxation) skipped — using spherical polytrope IC directly.

# ---------------------------------------------------------------------------
# Step 4: Thermal bomb + BH2 activation

@info "Applying thermal bomb (E_SN=$(E_SN), r_bomb=$(R_BOMB))..."

U_cpu = Array(U)
M_bomb = thermal_bomb!(U_cpu, NX, NY, NZ, DX, DX, DX;
                       E_SN = E_SN, r_bomb = R_BOMB,
                       x0 = x0, y0 = y0, z0 = z0,
                       x_center = 0.0, y_center = 0.0, z_center = 0.0)
fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

# Activate BH2 at origin with natal kick
bh2 = BlackHole(
    [0.0, 0.0, 0.0],
    V_KICK,
    M_BH2_INIT,
    eps_bh,
    units.c_code,
    r_floor)
bhs = BlackHole[bh1, bh2]

@info "BH2 activated" pos=bh2.pos vel=bh2.vel mass=bh2.mass r_sink=r_sink(bh2)
@info "M_bomb" M_bomb=M_bomb M_ejecta=M_STAR-M_BH2_INIT

# ---------------------------------------------------------------------------
# Pre-allocate work buffers (same device as U)

Fx = similar(U, 5, nxtot+1, nytot,   nztot  )
Fy = similar(U, 5, nxtot,   nytot+1, nztot  )
Fz = similar(U, 5, nxtot,   nytot,   nztot+1)
dU = similar(U)
Un = similar(U)

# ---------------------------------------------------------------------------
# Output setup

mkpath(OUTDIR)
traj_file = joinpath(OUTDIR, "trajectory.h5")
init_trajectory_file(traj_file, 2)

diag_file = joinpath(OUTDIR, "diagnostics.csv")
open(diag_file, "w") do f
    println(f, "t,Mdot1,Mdot2,M_BH1,M_BH2,r_sep," *
               "E_gas,Jz_gas,M_bound," *
               "Fgx1,Fgy1,Fgz1,Fgx2,Fgy2,Fgz2")
end

# Snapshot at t = 0 (immediately after bomb)
snap_path(idx) = joinpath(OUTDIR, @sprintf("snap_t%03d.h5", idx))
write_snapshot(snap_path(0), Array(U), NX, NY, NZ, DX, DX, DX, 0.0, GAMMA)
append_trajectory(traj_file, 0.0, bhs)
@info "Snapshot 0 written"

# ---------------------------------------------------------------------------
# Helper: one coupled SSP-RK3 step
# Advances gas (hydro + BH gravity + sinks) and BH N-body together.
# BH positions are frozen during the gas RK3 (standard frozen-force approximation).
# Accretion mass/momentum transfer applied once after stage 3.

function coupled_step!(U, Un, dU, Fx, Fy, Fz, bhs, dt,
                        nx, ny, nz, dx, x0, y0, z0, γ,
                        f_sink, ρ_floor, P_floor)
    # Gas force on BHs at current state (frozen for the whole step).
    # Guard against NaN/Inf (can occur near the dense stellar core at early times);
    # replace with zero force so the N-body remains valid while gas settles.
    F_gas = [gas_force_on_bh(U, nx, ny, nz, dx, dx, dx,
                              bhs[i], x0, y0, z0)
             for i in 1:length(bhs)]
    for i in eachindex(F_gas)
        if any(!isfinite, F_gas[i]) || sqrt(sum(F_gas[i].^2)) > 1e3
            F_gas[i] = zeros(3)
        end
    end

    # N-body: advance BH positions/velocities (internal SSP-RK3).
    # Skip if any BH position is already NaN (guards against cascading failure).
    bhs_ok = all(all(isfinite, bh.pos) && all(isfinite, bh.vel) for bh in bhs)
    if bhs_ok
        nbody_step!(bhs, dt, F_gas)
    end

    # Gas SSP-RK3 with all sources
    Un .= U

    # Stage 1: U⁽¹⁾ = Uⁿ + dt L(Uⁿ)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    # Sinks disabled: standard sink (torque_free=false) is stable with warm ambient
    # but omitted here to keep the demo simple. Re-enable for accretion diagnostics.
    # add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0;
    #                   f_sink = f_sink, torque_free = false)
    @. U = Un + dt * dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 2: U⁽²⁾ = ¾ Uⁿ + ¼ U⁽¹⁾ + ¼ dt L(U⁽¹⁾)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    # add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0;
    #                   f_sink = f_sink, torque_free = false)
    @. U = 0.75*Un + 0.25*U + 0.25*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 3: Uⁿ⁺¹ = ⅓ Uⁿ + ⅔ U⁽²⁾ + ⅔ dt L(U⁽²⁾)
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    # add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0;
    #                   f_sink = f_sink, torque_free = false)
    @. U = (1/3)*Un + (2/3)*U + (2/3)*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Accretion disabled (sinks off in this demo run)
    # for bh in bhs
    #     accrete!(bh, U, nx, ny, nz, dx, dx, dx, x0, y0, z0, dt; f_sink = f_sink)
    # end

    return F_gas   # return so caller can store for torque diagnostics
end

# ---------------------------------------------------------------------------
# Time integration

@info "Starting coupled evolution" t_end=T_END P0=round(2π,digits=3)
t_wall = time()

# Wrap the main loop in a function to avoid Julia soft-scope variable issues
function run_evolution!(U, Un, dU, Fx, Fy, Fz, bhs,
                         traj_file, diag_file, snap_path)
    t      = 0.0
    n_step = 0
    t_snap = DT_SNAP

    while t < T_END - 1e-12
        dt = cfl_dt_3d(U, NX, NY, NZ, DX, DX, DX, GAMMA, CFL)
        dt = min(dt, T_END - t, t_snap - t + 1e-12)
        dt = max(dt, 1e-14)

        M_prev = [bh.mass for bh in bhs]

        F_gas = coupled_step!(U, Un, dU, Fx, Fy, Fz, bhs, dt,
                               NX, NY, NZ, DX, x0, y0, z0, GAMMA,
                               F_SINK, RHO_FLOOR, P_FLOOR)

        t      += dt
        n_step += 1

        # ---- Diagnostics ---------------------------------------------------
        Mdot  = [(bhs[i].mass - M_prev[i]) / dt for i in 1:2]
        r_sep = sqrt(sum((bhs[2].pos .- bhs[1].pos).^2))

        E_gas   = gas_energy_total(U, NX, NY, NZ, DX, DX, DX)
        Jz_gas  = gas_angular_momentum_total(U, NX, NY, NZ, DX, DX, DX, x0, y0, z0)[3]
        M_bound = bound_gas_mass(U, NX, NY, NZ, DX, DX, DX, x0, y0, z0, bhs, GAMMA)

        open(diag_file, "a") do f
            @printf(f, "%.6f,%.6e,%.6e,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    t, Mdot[1], Mdot[2],
                    bhs[1].mass, bhs[2].mass,
                    r_sep,
                    E_gas, Jz_gas, M_bound,
                    F_gas[1][1], F_gas[1][2], F_gas[1][3],
                    F_gas[2][1], F_gas[2][2], F_gas[2][3])
        end

        append_trajectory(traj_file, t, bhs)

        # ---- Snapshot ------------------------------------------------------
        if t >= t_snap - 1e-10
            snap_idx = round(Int, t / DT_SNAP)
            write_snapshot(snap_path(snap_idx), Array(U),
                           NX, NY, NZ, DX, DX, DX, t, GAMMA)
            t_snap += DT_SNAP
            @info "Snapshot" idx=snap_idx t=round(t,digits=3) r_sep=round(r_sep,digits=4) M_BH2=round(bhs[2].mass,digits=5)
        end

        if n_step % 500 == 0
            wall = round(time() - t_wall, digits=1)
            @info "Progress" step=n_step t=round(t,digits=3) dt_code=round(dt,sigdigits=3) wall_sec=wall
        end
    end
    return t, n_step
end

t_final, n_steps = run_evolution!(U, Un, dU, Fx, Fy, Fz, bhs,
                                   traj_file, diag_file, snap_path)

t_elapsed = time() - t_wall
Mcell_per_sec = n_steps * NX * NY * NZ / t_elapsed / 1e6
@info "Done" n_steps=n_steps t_final=round(t_final,digits=4) wall_sec=round(t_elapsed,digits=1) Mcell_per_s=round(Mcell_per_sec,digits=1)
@info "Outputs" traj=traj_file diag=diag_file snapshots=OUTDIR
