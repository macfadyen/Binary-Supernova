#!/usr/bin/env julia
# Binary supernova: 30 M☉ pre-existing BH + 60 M☉ low-Z WR progenitor
# → 30 M☉ remnant BH + 30 M☉ ejecta, uniform grid, sinks OFF (Run 1).
#
# Usage:
#   julia --project=. scripts/run_sn30_nosink.jl [--gpu] [--nx NX]
#
# --nx NX  : grid resolution per axis (default 128; use 64 for quick smoke test,
#            192 for higher-quality production)
# --gpu    : use CUDA backend (requires CUDA.jl and a supported GPU)
#
# Output written to demo1/output_sn30_nosink/:
#   trajectory.h5        BH positions, velocities, masses at every step
#   diagnostics.csv      Per-step scalar diagnostics (M_bound is the headline)
#   snap_tNNN.h5         Gas snapshots every DT_SNAP = 0.5 code time units
#
# Physical scenario and parameter rationale: see scripts/README_sn30.md.

using BinarySupernova
using Printf
import KernelAbstractions as KA

# ---------------------------------------------------------------------------
# Argument parsing

use_gpu = "--gpu" in ARGS
nx_arg  = let v = 128
    for (i, arg) in enumerate(ARGS)
        if arg == "--nx" && i < length(ARGS)
            v = parse(Int, ARGS[i+1])
        end
    end
    v
end

# ---------------------------------------------------------------------------
# Physical parameters
#
# Pre-SN system: M_BH1 = 30 M☉, M_star = 60 M☉ → M_tot = 90 M☉
# Code units (G = M_tot = a0 = 1):  M_BH1 = 1/3, M_star = 2/3
# Post-SN remnant: M_BH2_init = 30 M☉ (direct collapse, low Z) → 1/3 in code
# Ejecta:          M_ejecta = M_star − M_BH2_init = 30 M☉ → 1/3 in code
#
# PhysicalUnits(90.0, 30.0): v_unit ≈ 756 km/s, c_code ≈ 397, t_unit ≈ 2.8e4 s,
# T_orb = 2π t_unit ≈ 2 days, energy unit ≈ 1 Bethe (10^51 erg).

const M_BH1      = 30.0 / 90.0        # = 1/3  pre-existing BH
const M_STAR     = 60.0 / 90.0        # = 2/3  WR/He-star progenitor
const M_BH2_INIT = 30.0 / 90.0        # = 1/3  direct-collapse remnant
const GAMMA      = 5.0 / 3.0          # shakedown γ; switch to 4/3 at NX≥128 later
const A0         = 1.0                # initial binary separation (= 30 R☉)
const R_STAR     = 0.1                # stripped WR He-star ≈ 3 R☉; R_L/a ≈ 0.44
const E_SN       = 0.3                # ≈ 3e50 erg; e_SN/|Φ| ≈ 1 → most falls back
const R_BOMB     = R_STAR             # deposit over whole progenitor
const V_KICK     = [0.0, 0.0, 0.0]    # direct collapse → near-zero kick
const F_SINK     = 1.0                # demo default (sinks off in this run)
const RHO_FLOOR  = 1e-3               # warm ambient for WENO5 stability
const P_FLOOR    = 1e-6               # consistent ambient sound speed
const CFL        = 0.4

# Evolution
const T_END   = 30.0                  # ≈ 5 P₀
const DT_SNAP = 0.5                   # ≈ 10 h physical
const OUTDIR  = "demo1/output_sn30_nosink"

# Grid
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
# Grid — domain ±4 a₀ = ±120 R☉ so ejecta stays on-grid for t ≳ 5.

const NY = NX;  const NZ = NX
const L  = 4.0
const DX = 2L / NX
const ng = BinarySupernova.NG

nxtot = NX + 2ng;  nytot = NY + 2ng;  nztot = NZ + 2ng
const x0 = -L;  const y0 = -L;  const z0 = -L

@info "Grid" NX=NX dx=DX domain="[$(x0),$(L)]³" R_star_over_dx=round(R_STAR/DX, digits=2) memory_MB=round(5*nxtot*nytot*nztot*8/1e6,digits=1)

if use_gpu
    U = CUDA.zeros(Float64, 5, nxtot, nytot, nztot)
else
    U = zeros(Float64, 5, nxtot, nytot, nztot)
end

# ---------------------------------------------------------------------------
# Physical unit conversion (for r_sink/c_code; no effect on dynamics)
units = PhysicalUnits(90.0, 30.0)     # 90 M☉ total, 30 R☉ separation → c ≈ 397

# BH softening and sink floor — tighter than demo1 because R_STAR=0.1 is small.
eps_bh  = 2.0 * DX                    # ≈ 0.13 at NX=64, 0.063 at NX=128, 0.042 at NX=192
r_floor = 1.5 * DX

# ---------------------------------------------------------------------------
# CoM recentring
#
# q_pre = M_star / M_BH1 = 2, so CoM offset from the star is a₀/3 toward BH1.
# Place CoM at origin:
#   x_star_center = −M_BH1 * a₀ / M_tot = −1/3
#   x_bh1         = +M_star * a₀ / M_tot = +2/3
# Keplerian velocities (reduced-mass orbit with relative speed v_rel = √(M_tot/a₀) = 1):
#   v_bh1 = +(M_star / M_tot) × v_rel = +2/3
#   v_bh2 = −(M_BH1  / M_tot) × v_rel = −1/3   (BH2 inherits pre-SN star velocity)

const v_rel   = sqrt((M_BH1 + M_STAR) / A0)   # = 1.0 in code units
const x_star_center = -M_BH1  * A0 / (M_BH1 + M_STAR)   # = −1/3
const x_bh1         = +M_STAR * A0 / (M_BH1 + M_STAR)   # = +2/3
const v_bh1_y       = +M_STAR * v_rel / (M_BH1 + M_STAR)
const v_bh2_y       = -M_BH1  * v_rel / (M_BH1 + M_STAR)

# ---------------------------------------------------------------------------
# Step 1: polytrope for the pre-SN star, centred at x_star_center

@info "Building polytrope IC..." M_star=M_STAR R_star=R_STAR center_x=x_star_center

U_cpu = Array(U)
polytrope_ic_3d!(U_cpu, NX, NY, NZ, DX, DX, DX, GAMMA;
                 M_star   = M_STAR,
                 R_star   = R_STAR,
                 x0 = x0, y0 = y0, z0 = z0,
                 x_center = x_star_center, y_center = 0.0, z_center = 0.0,
                 ρ_floor  = RHO_FLOOR, P_floor = P_FLOOR)
fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

# ---------------------------------------------------------------------------
# Step 2: BH1 on CoM-centred circular orbit

bh1 = BlackHole(
    [x_bh1, 0.0, 0.0],
    [0.0,   v_bh1_y, 0.0],
    M_BH1,
    eps_bh,
    units.c_code,
    r_floor)
@info "BH1" pos=bh1.pos vel=bh1.vel mass=bh1.mass r_sink=r_sink(bh1)

# Step 3 (Roche relaxation) skipped — using spherical polytrope IC directly.

# ---------------------------------------------------------------------------
# Step 4: thermal bomb + BH2 activation (at the star centre)

@info "Applying thermal bomb..." E_SN=E_SN r_bomb=R_BOMB center_x=x_star_center

U_cpu = Array(U)
M_bomb = thermal_bomb!(U_cpu, NX, NY, NZ, DX, DX, DX;
                       E_SN = E_SN, r_bomb = R_BOMB,
                       x0 = x0, y0 = y0, z0 = z0,
                       x_center = x_star_center, y_center = 0.0, z_center = 0.0)
fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

# BH2 activated at the former stellar centre, inheriting the pre-SN star's
# orbital velocity plus any natal kick.
bh2 = BlackHole(
    [x_star_center, 0.0, 0.0],
    [0.0, v_bh2_y, 0.0] .+ V_KICK,
    M_BH2_INIT,
    eps_bh,
    units.c_code,
    r_floor)
bhs = BlackHole[bh1, bh2]

@info "BH2 activated" pos=bh2.pos vel=bh2.vel mass=bh2.mass r_sink=r_sink(bh2)
@info "M_bomb" M_bomb=M_bomb M_ejecta=M_STAR-M_BH2_INIT

# ---------------------------------------------------------------------------
# Scratch buffers

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

snap_path(idx) = joinpath(OUTDIR, @sprintf("snap_t%03d.h5", idx))
write_snapshot(snap_path(0), Array(U), NX, NY, NZ, DX, DX, DX, 0.0, GAMMA)
append_trajectory(traj_file, 0.0, bhs)
@info "Snapshot 0 written"

# ---------------------------------------------------------------------------
# One coupled SSP-RK3 step — sinks OFF in this script (Run 1).

function coupled_step!(U, Un, dU, Fx, Fy, Fz, bhs, dt,
                        nx, ny, nz, dx, x0, y0, z0, γ,
                        f_sink, ρ_floor, P_floor)
    F_gas = [gas_force_on_bh(U, nx, ny, nz, dx, dx, dx,
                              bhs[i], x0, y0, z0)
             for i in 1:length(bhs)]
    for i in eachindex(F_gas)
        if any(!isfinite, F_gas[i]) || sqrt(sum(F_gas[i].^2)) > 1e3
            F_gas[i] = zeros(3)
        end
    end

    bhs_ok = all(all(isfinite, bh.pos) && all(isfinite, bh.vel) for bh in bhs)
    if bhs_ok
        nbody_step!(bhs, dt, F_gas)
    end

    Un .= U

    # Stage 1
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    @. U = Un + dt * dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 2
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    @. U = 0.75*Un + 0.25*U + 0.25*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 3
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    @. U = (1/3)*Un + (2/3)*U + (2/3)*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    return F_gas
end

# ---------------------------------------------------------------------------
# Time integration

@info "Starting coupled evolution" t_end=T_END P0=round(2π,digits=3)
t_wall = time()

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

        if t >= t_snap - 1e-10
            snap_idx = round(Int, t / DT_SNAP)
            write_snapshot(snap_path(snap_idx), Array(U),
                           NX, NY, NZ, DX, DX, DX, t, GAMMA)
            t_snap += DT_SNAP
            @info "Snapshot" idx=snap_idx t=round(t,digits=3) r_sep=round(r_sep,digits=4) M_bound=round(M_bound,digits=4)
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
