#!/usr/bin/env julia
# Fiducial production driver: 50 M☉ BH + 55 M☉ stripped He/C-O progenitor
# → 45 M☉ remnant BH + 10 M☉ ejecta, tidally synchronized, a₀ = 20 R☉ (default).
# Live orbit + sinks + torque-free accretion.
#
# Physical motivation: low-Z close-binary channel producing a ~50+50 M☉ BH
# binary, sitting on the upper edge of the pair-instability mass gap.  Maps to
# GW150914-/GW190521-like systems.  See memory project_fiducial_progenitor.md.
#
# Physical parameters (inputs):
#   M_BH1        = 50 M☉
#   M_star       = 55 M☉   (stripped He/C-O, no H envelope)
#   M_BH2_init   = 45 M☉   → M_ejecta = 10 M☉
#   R_star       = 1 R☉    (compact; resolution tight at NX<256)
#   a₀           = 20 R☉  (override with --a0-rsun)
#   E_SN         = 10⁵¹ erg
#   tidal sync   = true    (star spin Ω_spin = Ω_orb, aligned with +ẑ)
#   self_gravity = false   (M_gas/M_BH ≈ 0.1 << 1; leave off by default)
#
# Usage:
#   julia --project=. scripts/run_sn50_fiducial.jl [--gpu] [--nx NX]
#                                                  [--a0-rsun A0]
#                                                  [--torque-free]
#                                                  [--scf-ic | --relax-ic]
#                                                  [--no-spin]
#                                                  [--outdir PATH]

using BinarySupernova
using Printf
import KernelAbstractions as KA

# ---------------------------------------------------------------------------
# Argument parsing

use_gpu     = "--gpu" in ARGS
torque_free = "--torque-free" in ARGS
no_spin     = "--no-spin" in ARGS
scf_ic_arg  = "--scf-ic" in ARGS
relax_ic_arg = "--relax-ic" in ARGS
self_gravity_arg = "--self-gravity" in ARGS
nx_arg = let v = 128
    for (i, arg) in enumerate(ARGS)
        if arg == "--nx" && i < length(ARGS)
            v = parse(Int, ARGS[i+1])
        end
    end
    v
end
outdir_arg = let v = "demo1/output_sn50_fiducial"
    for (i, arg) in enumerate(ARGS)
        if arg == "--outdir" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end
# Checkpoint/restart (for PBS walltime-capped runs; see scripts/run_relax_optimal.pbs).
# --restart PATH        resume from a uniform checkpoint, skipping the IC build.
# --checkpoint-every-min M  write OUTDIR/chkpt.h5 every M wall-minutes (0 = off).
# --wall-budget-min M       checkpoint and exit(2) once M wall-minutes elapse, so
#                           the batch script can auto-resubmit (0 = run to t_end).
restart_arg = let v = ""
    for (i, arg) in enumerate(ARGS)
        if arg == "--restart" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end
checkpoint_every_min_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--checkpoint-every-min" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
wall_budget_min_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--wall-budget-min" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
L_arg = let v = 4.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--L" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
a0_rsun_arg = let v = 20.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--a0-rsun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
rho_sink_min_arg = let v = -1.0   # -1 ⇒ default = 2 × RHO_FLOOR
    for (i, arg) in enumerate(ARGS)
        if arg == "--rho-sink-min" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
rho_floor_arg = let v = 3e-5
    for (i, arg) in enumerate(ARGS)
        if arg == "--rho-floor" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
p_floor_arg = let v = 1e-6
    for (i, arg) in enumerate(ARGS)
        if arg == "--p-floor" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
v_kick_x_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--v-kick-x" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
v_kick_y_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--v-kick-y" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
v_kick_z_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--v-kick-z" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
bipolar_theta_deg_arg = let v = 180.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--bipolar-theta-deg" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
bh2_sink_delay_arg = let v = -1.0   # -1 ⇒ default = 2·r_sink_BH2 / c_s_post_bomb
    for (i, arg) in enumerate(ARGS)
        if arg == "--bh2-sink-delay" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
t_end_arg = let v = 30.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--t-end" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
dt_snap_arg = let v = 0.5
    for (i, arg) in enumerate(ARGS)
        if arg == "--dt-snap" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Scale on E_SN: actual E_SN = E_sn_frac · 10⁵¹ erg.  For weak-SN / fallback
# studies, values in 0.05–0.5 probe the partial-to-failed explosion regime.
e_sn_frac_arg = let v = 1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--e-sn-frac" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Shell-bomb inner cutoff as a fraction of R_STAR.  Negative ⇒ fall back to
# R_CORE (existing default: exclude the hollowed core only).  Use e.g. 0.5 to
# restrict bomb to the outer half of the envelope, leaving inner layers
# unshocked — source of high-AM fallback material.
r_bomb_inner_frac_arg = let v = -1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--r-bomb-inner-frac" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Outer cutoff of bomb as fraction of R_STAR.  Default 1.0 = R_STAR (whole star).
# Use e.g. 0.5 to bomb only the inner half of the envelope, leaving outer
# layers unshocked with their pre-SN tidal-sync velocity (high-AM reservoir).
r_bomb_outer_frac_arg = let v = 1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--r-bomb-outer-frac" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Mass-based bomb: deposit E_SN in the innermost FRAC of the envelope mass.
# When > 0 this overrides --r-bomb-outer-frac — the bomb radius is found from
# the *actual* (post-relaxation) stellar mass profile, so it is robust to a
# relaxed star whose radius differs from the input R_STAR.  Recommended for
# --relax-ic; assumes a spherical bomb (a bipolar cone cuts M_bomb further).
bomb_mass_frac_arg = let v = -1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--bomb-mass-frac" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Rigid spin of the stellar envelope as a multiple of Ω_orb.  Default 1.0 =
# tidal synchronization.  Values > 1 represent super-synchronous rotation
# (e.g. CHE progenitors: k ≈ 3–7 with surface velocity a significant fraction
# of breakup).  Ignored if --no-spin is set.
spin_omega_frac_arg = let v = 1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--spin-omega-frac" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Progenitor mass inputs in M☉.  Defaults reproduce the 50/45/10 fiducial.
# M_tot = M_BH1 + M_star; M_ejecta = M_star − M_BH2_init is derived.
m_bh1_msun_arg = let v = 50.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--m-bh1-msun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
m_star_msun_arg = let v = 55.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--m-star-msun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
m_bh2_msun_arg = let v = 45.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--m-bh2-msun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# SCF-IC axis ratio α = r_p / r_eq.  Only consulted with --scf-ic.
# α = 1 ⇒ non-rotating (Lane-Emden limit); α ↘ α_peak ≈ 0.66 (n=3) increases
# rotation up to the mass-shedding limit.  Because the SCF surface condition
# snaps r_p to a grid cell, α is *discretised* on Δα ≈ 1 / i_eq; for a given
# (M_star, R_star), Ω_spin is whatever the equilibrium sequence dictates — not
# a free target.  Use --scf-axis-ratio to pick the figure; --spin-omega-frac
# is ignored by the SCF path.
scf_axis_ratio_arg = let v = 1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--scf-axis-ratio" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
# Roche-relaxation IC controls (--relax-ic).  t_damp ≈ 0.1 P₀ ≈ 0.6; the
# relaxation runs the full t_max — KE_tol = 0 disables the KE/E_th early exit.
relax_t_damp_arg = let v = 0.6
    for (i, arg) in enumerate(ARGS)
        if arg == "--relax-t-damp" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
relax_t_max_arg = let v = 3.0       # safety cap; relax_ic! stops at the KE minimum
    for (i, arg) in enumerate(ARGS)
        if arg == "--relax-t-max" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
relax_ke_tol_arg = let v = 0.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--relax-ke-tol" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end

# ---------------------------------------------------------------------------
# Physical inputs (solar units) + code-unit conversion

const M_BH1_MSUN  = m_bh1_msun_arg
const M_STAR_MSUN = m_star_msun_arg
const M_BH2_MSUN  = m_bh2_msun_arg
const M_TOT_MSUN  = M_BH1_MSUN + M_STAR_MSUN       # total system mass
const A0_RSUN     = a0_rsun_arg
const R_STAR_RSUN = 1.0
const E_SN_ERG    = 1.0e51

const units = PhysicalUnits(M_TOT_MSUN, A0_RSUN)

# M_tot = a₀ = 1 in code units
const M_BH1      = M_BH1_MSUN  / M_TOT_MSUN
const M_STAR     = M_STAR_MSUN / M_TOT_MSUN
const M_BH2_INIT = M_BH2_MSUN  / M_TOT_MSUN
const A0         = 1.0
const R_STAR     = R_STAR_RSUN / A0_RSUN                   # = 0.05
# Total-energy unit is M_unit · v_unit² (not E_unit which is pressure).
const E_SN       = e_sn_frac_arg * E_SN_ERG / (units.M_unit * units.v_unit^2)

# γ = 4/3: radiation-dominated stripped He/C-O progenitor interior; matches
# the n = 3 polytrope index used by the Lane-Emden IC.  (CLAUDE.md §3.1, §6.1.)
const GAMMA        = 4.0 / 3.0
const R_BOMB       = r_bomb_outer_frac_arg * R_STAR
const V_KICK       = [v_kick_x_arg, v_kick_y_arg, v_kick_z_arg]
const F_SINK       = 1.0
const RHO_FLOOR    = rho_floor_arg
const P_FLOOR      = p_floor_arg
const RHO_SINK_MIN = rho_sink_min_arg > 0 ? rho_sink_min_arg : 2.0 * RHO_FLOOR
const CFL          = 0.4

# Physics toggles
const SELF_GRAVITY = self_gravity_arg || relax_ic_arg   # --relax-ic forces self-gravity on
const TIDAL_SYNC   = !no_spin         # star rotates at Ω_orb; flag --no-spin disables

const T_END   = t_end_arg
const DT_SNAP = dt_snap_arg
const OUTDIR  = outdir_arg
const NX      = nx_arg

# Checkpoint/restart configuration.
const RESTART_PATH         = restart_arg
const RESTARTING           = !isempty(RESTART_PATH) && isfile(RESTART_PATH)
const CHECKPOINT_EVERY_MIN = checkpoint_every_min_arg
const WALL_BUDGET_MIN      = wall_budget_min_arg
const CHKPT_PATH           = joinpath(OUTDIR, "chkpt.h5")
if !isempty(RESTART_PATH) && !RESTARTING
    @warn "--restart path not found; starting fresh" RESTART_PATH
end

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
const L  = L_arg
const DX = 2L / NX
const ng = BinarySupernova.NG

nxtot = NX + 2ng;  nytot = NY + 2ng;  nztot = NZ + 2ng
const x0 = -L;  const y0 = -L;  const z0 = -L

@info "Grid" NX=NX dx=round(DX,digits=4) domain="[$(x0),$(L)]³" R_star_over_dx=round(R_STAR/DX, digits=2) memory_MB=round(5*nxtot*nytot*nztot*8/1e6,digits=1)
@info "Physical scales" M_tot_Msun=M_TOT_MSUN a0_Rsun=A0_RSUN c_code=round(units.c_code, digits=1) t_unit_s=round(units.t_unit, sigdigits=3) v_unit_kms=round(units.v_unit/1e5, digits=1)
@info "Physics toggles" torque_free=torque_free tidal_sync=TIDAL_SYNC self_gravity=SELF_GRAVITY ρ_floor=RHO_FLOOR ρ_sink_min=RHO_SINK_MIN
@info "Code-unit params" M_BH1=round(M_BH1,digits=4) M_STAR=round(M_STAR,digits=4) M_BH2_INIT=round(M_BH2_INIT,digits=4) R_STAR=round(R_STAR,digits=4) E_SN=round(E_SN,sigdigits=4)

R_STAR / DX < 2.0 && @warn "Star is under-resolved (R_star / dx < 2). Consider NX ≥ 256."

if use_gpu
    U = CUDA.zeros(Float64, 5, nxtot, nytot, nztot)
else
    U = zeros(Float64, 5, nxtot, nytot, nztot)
end

# ---------------------------------------------------------------------------
# Softening + sink floor

eps_bh  = 2.0 * DX
r_floor = 1.5 * DX

# ---------------------------------------------------------------------------
# CoM recentring — identical convention to run_sn30_sinks.jl

const v_rel         = sqrt((M_BH1 + M_STAR) / A0)          # = 1 in code units
const x_star_center = -M_BH1  * A0 / (M_BH1 + M_STAR)
const x_bh1         = +M_STAR * A0 / (M_BH1 + M_STAR)
const v_bh1_y       = +M_STAR * v_rel / (M_BH1 + M_STAR)
const v_bh2_y       = -M_BH1  * v_rel / (M_BH1 + M_STAR)

# Orbital angular frequency (inertial frame, aligned with +ẑ, prograde).
const Ω_ORB = v_bh1_y / x_bh1                               # = 1 in code units

# ---------------------------------------------------------------------------
# Breakup frequency (diagnostic, shared by both IC paths)
const Ω_BRK = sqrt(M_STAR / R_STAR^3)

U_cpu = Array(U)

# v_gas(r) = v_star + Ω × (r - r_star),   Ω = Ω_spin ẑ,   v_star = (0, v_bh2_y, 0).
# Hoisted above the IC block so the --relax-ic path can reuse it.
function apply_stellar_rotation!(U_cpu, γ, Ω, v_star_y,
                                  x_star, y_star, z_star,
                                  ρ_thresh)
    ng = BinarySupernova.NG
    nxtot, nytot, nztot = size(U_cpu, 2), size(U_cpu, 3), size(U_cpu, 4)
    nx = nxtot - 2ng;  ny = nytot - 2ng;  nz = nztot - 2ng
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ = U_cpu[1, i, j, k]
        ρ <= ρ_thresh && continue
        xc = x0 + (i - ng - 0.5) * DX - x_star
        yc = y0 + (j - ng - 0.5) * DX - y_star
        vx = -Ω * yc
        vy = v_star_y + Ω * xc
        vz = 0.0
        # U[5] currently holds only E_internal (polytrope sets no KE); P = (γ-1)·E_int.
        P_internal = U_cpu[5, i, j, k] * (γ - 1.0)
        U_cpu[2, i, j, k] = ρ * vx
        U_cpu[3, i, j, k] = ρ * vy
        U_cpu[4, i, j, k] = ρ * vz
        U_cpu[5, i, j, k] = P_internal / (γ - 1.0) + 0.5 * ρ * (vx^2 + vy^2 + vz^2)
    end
    return nothing
end

# BH1 on the CoM-centred circular orbit.  Built before the IC block so the
# --relax-ic path can use it as a fixed gravity source during relaxation.
bh1 = BlackHole(
    [x_bh1, 0.0, 0.0],
    [0.0,   v_bh1_y, 0.0],
    M_BH1,
    eps_bh,
    units.c_code,
    r_floor)
@info "BH1" pos=bh1.pos vel=bh1.vel mass=round(bh1.mass,digits=4) r_sink=round(r_sink(bh1),digits=4)

# Radius enclosing the innermost `frac` of the gas mass beyond `r_inner`
# (gas above ρ_thresh), measured from the star centre.  Keys the core-carve
# and the thermal bomb to mass fractions of the *relaxed* star — robust to a
# relaxed star whose extent differs from the input R_STAR.
function radius_for_mass_fraction(U_cpu, frac, x_star, r_inner,
                                        nx, ny, nz, dx, x0, y0, z0, ρ_thresh)
    ng = BinarySupernova.NG
    dV = dx^3
    rmax = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U_cpu[1, i, j, k] > ρ_thresh || continue
        xc = x0 + (i-ng-0.5)*dx - x_star
        yc = y0 + (j-ng-0.5)*dx
        zc = z0 + (k-ng-0.5)*dx
        r  = sqrt(xc*xc + yc*yc + zc*zc)
        r > rmax && (rmax = r)
    end
    rmax <= r_inner && return r_inner
    nbin = 256
    Δb   = rmax / nbin
    mbin = zeros(nbin)
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ = U_cpu[1, i, j, k]
        ρ > ρ_thresh || continue
        xc = x0 + (i-ng-0.5)*dx - x_star
        yc = y0 + (j-ng-0.5)*dx
        zc = z0 + (k-ng-0.5)*dx
        r  = sqrt(xc*xc + yc*yc + zc*zc)
        r <= r_inner && continue
        b = clamp(floor(Int, r/Δb) + 1, 1, nbin)
        mbin[b] += ρ * dV
    end
    M_env = sum(mbin)
    M_env <= 0.0 && return r_inner
    target = frac * M_env
    acc = 0.0
    for b in 1:nbin
        acc += mbin[b]
        acc >= target && return b * Δb
    end
    return rmax
end

# --- Initial conditions.  On restart the entire IC build (polytrope,
#     relaxation, thermal bomb, BH2 activation) is SKIPPED and the evolved
#     state is loaded from the checkpoint instead — re-running the relaxation
#     on every PBS resubmit would defeat the checkpoint's purpose.
if !RESTARTING
if relax_ic_arg
    # ---- Roche-potential relaxation IC (CLAUDE.md §6.3).
    #      A spherical Lane-Emden polytrope is relaxed by velocity damping in
    #      the binary's CO-ROTATING frame (relax_ic! with Ω = Ω_orb adds the
    #      centrifugal + Coriolis forces).  Centrifugal support holds the star
    #      at its orbital position with BH1 fixed, so it settles into the tidal
    #      (Roche) shape — the L1/L2 bulge the SCF figure lacks — with no
    #      orbital drift.  The relaxed gas is at rest in the rotating frame;
    #      the inertial co-rotation velocity is added by the overlay below
    #      (which assumes synchronization, Ω_spin = Ω_orb).  self_gravity is
    #      forced on (see SELF_GRAVITY).
    @info "Building Roche-relaxation IC (Lane-Emden polytrope + velocity damping)..." M_star=round(M_STAR,digits=4) M_core=round(M_BH2_INIT,digits=4) R_star=round(R_STAR,digits=4)

    # Relax the COMPLETE star — the pre-SN star is whole; the core becomes BH2
    # only at the explosion.  Paint un-hollowed (M_core = 0); the core is carved
    # out after relaxation, mass-based, from the actual relaxed profile.
    polytrope_ic_3d!(U_cpu, NX, NY, NZ, DX, DX, DX, GAMMA;
                     M_star = M_STAR, R_star = R_STAR, M_core = 0.0,
                     x0 = x0, y0 = y0, z0 = z0,
                     x_center = x_star_center, y_center = 0.0, z_center = 0.0,
                     ρ_floor = RHO_FLOOR, P_floor = P_FLOOR)
    @info "Polytrope painted (full star; core carved mass-based after relaxation)"

    fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
    copyto!(U, U_cpu)

    @info "Relaxing toward Roche equilibrium..." t_damp=relax_t_damp_arg t_max=relax_t_max_arg KE_tol=relax_ke_tol_arg self_gravity=SELF_GRAVITY
    relax_info = relax_ic!(U, NX, NY, NZ, DX, DX, DX, GAMMA;
                           bhs = [bh1],
                           x0 = x0, y0 = y0, z0 = z0,
                           t_damp  = relax_t_damp_arg,
                           t_max   = relax_t_max_arg,
                           cfl     = CFL,
                           ρ_floor = RHO_FLOOR, P_floor = P_FLOOR,
                           KE_tol  = relax_ke_tol_arg,
                           Ω       = Ω_ORB,
                           self_gravity = SELF_GRAVITY,
                           verbose = true)
    @info "Relaxation done" t=round(relax_info.t,digits=3) n_steps=relax_info.n_steps KE_ratio=round(relax_info.KE_ratio,sigdigits=3)
    relax_info.KE_ratio > 0.05 &&
        @warn "Relaxation ended with KE/E_th > 5% — star may not be fully settled" KE_ratio=round(relax_info.KE_ratio,sigdigits=3)

    # Overlay rigid rotation on the relaxed (tidally distorted) star.
    U_cpu = Array(U)
    Ω_SPIN = spin_omega_frac_arg * Ω_ORB
    if TIDAL_SYNC
        apply_stellar_rotation!(U_cpu, GAMMA, Ω_SPIN, v_bh2_y,
                                x_star_center, 0.0, 0.0,
                                2.0 * RHO_FLOOR)
        @info "Stellar rotation applied (overlay on relaxed star)" Ω_spin=round(Ω_SPIN,digits=3) Ω_orb=round(Ω_ORB,digits=3) frac_of_breakup=round(Ω_SPIN/Ω_BRK,digits=3)
        Ω_SPIN / Ω_BRK > 0.9 && @warn "Requested Ω_spin > 0.9 Ω_brk — overlay is unphysical above breakup"
    else
        @info "Stellar rotation DISABLED (--no-spin)"
    end

    # Carve the core, mass-based: the relaxed star has expanded/distorted, so
    # the radius enclosing M_BH2_INIT differs from the input polytrope R_CORE.
    # Find it from the relaxed profile, then hollow r < R_CORE for BH2.
    R_CORE = radius_for_mass_fraction(U_cpu, M_BH2_INIT / M_STAR, x_star_center,
                                      0.0, NX, NY, NZ, DX, x0, y0, z0,
                                      2.0 * RHO_FLOOR)
    let ng = BinarySupernova.NG
        @inbounds for k in ng+1:ng+NZ, j in ng+1:ng+NY, i in ng+1:ng+NX
            xc = x0 + (i - ng - 0.5) * DX - x_star_center
            yc = y0 + (j - ng - 0.5) * DX
            zc = z0 + (k - ng - 0.5) * DX
            if xc*xc + yc*yc + zc*zc < R_CORE^2
                U_cpu[1, i, j, k] = RHO_FLOOR
                U_cpu[2, i, j, k] = 0.0
                U_cpu[3, i, j, k] = 0.0
                U_cpu[4, i, j, k] = 0.0
                U_cpu[5, i, j, k] = P_FLOOR / (GAMMA - 1.0)
            end
        end
    end
    @info "Core carved (mass-based)" r_core=round(R_CORE, digits=4) r_core_over_dx=round(R_CORE/DX, digits=2)

elseif scf_ic_arg
    # ---- Self-consistent rotating polytrope (Hachisu 1986a SCF).
    #      The equilibrium sequence is parameterised by the axis ratio α;
    #      Ω_spin is an *output*, not an input.  Because the SCF surface
    #      condition snaps r_p to a cell, Ω_spin is quantised on Δα ≈ 1/i_eq —
    #      practical control only for moderate rotation (a₀ ≲ 5 R☉).
    α_ic = clamp(Float64(scf_axis_ratio_arg), 0.55, 1.0)
    @info "Building self-consistent rotating polytrope IC (Hachisu SCF)..." M_star=round(M_STAR,digits=4) M_core=round(M_BH2_INIT,digits=4) R_star=round(R_STAR,digits=4) axis_ratio=round(α_ic,digits=4) Ω_brk=round(Ω_BRK,digits=3)

    info_ic = rotating_polytrope_ic_3d!(U_cpu, NX, NY, NZ, DX, DX, DX, GAMMA;
                    M_star     = M_STAR,
                    R_star     = R_STAR,
                    axis_ratio = α_ic,
                    M_core     = M_BH2_INIT,
                    x0 = x0, y0 = y0, z0 = z0,
                    x_center = x_star_center, y_center = 0.0, z_center = 0.0,
                    v_star   = (0.0, v_bh2_y, 0.0),
                    ρ_floor  = RHO_FLOOR, P_floor = P_FLOOR,
                    Nr_scf = 256, Nμ_scf = 33, lmax_scf = 12,
                    scf_tol = 1e-7, scf_maxiter = 3000, scf_mix = 0.35)

    R_CORE = info_ic.r_core
    Ω_SPIN = info_ic.Ω_spin
    @info "SCF IC built" axis_ratio=round(α_ic,digits=4) Ω_spin=round(Ω_SPIN,digits=4) Ω_over_Ω_orb=round(Ω_SPIN/Ω_ORB,digits=3) frac_of_breakup=round(Ω_SPIN/Ω_BRK,digits=3) ρ_c=round(info_ic.ρ_c,sigdigits=4) M_mapped=round(info_ic.M_mapped,digits=4) r_core=round(R_CORE,digits=4) r_core_over_dx=round(R_CORE/DX,digits=2)
    Ω_SPIN / Ω_BRK > 0.9 && @warn "Ω_spin > 0.9 Ω_brk — SCF solution near mass-shedding; envelope may shed on first dynamic step"
else
    # ---- Legacy: static Lane-Emden polytrope + rigid-rotation velocity overlay.
    #      Unphysical when Ω_spin > Ω_brk (surface exceeds breakup); use --scf-ic
    #      for a self-consistent figure instead.
    @info "Building polytrope IC (static Lane-Emden + rigid-rotation overlay)..." M_star=round(M_STAR,digits=4) M_core=round(M_BH2_INIT,digits=4) R_star=round(R_STAR,digits=4) center_x=round(x_star_center,digits=4)

    _, _, _, R_CORE = polytrope_ic_3d!(U_cpu, NX, NY, NZ, DX, DX, DX, GAMMA;
                     M_star   = M_STAR,
                     R_star   = R_STAR,
                     M_core   = M_BH2_INIT,
                     x0 = x0, y0 = y0, z0 = z0,
                     x_center = x_star_center, y_center = 0.0, z_center = 0.0,
                     ρ_floor  = RHO_FLOOR, P_floor = P_FLOOR)
    @info "Polytrope core hollowed" r_core=round(R_CORE, digits=4) r_core_over_dx=round(R_CORE/DX, digits=2)

    Ω_SPIN = spin_omega_frac_arg * Ω_ORB
    if TIDAL_SYNC
        apply_stellar_rotation!(U_cpu, GAMMA, Ω_SPIN, v_bh2_y,
                                x_star_center, 0.0, 0.0,
                                2.0 * RHO_FLOOR)
        @info "Stellar rotation applied (overlay)" Ω_spin=round(Ω_SPIN,digits=3) Ω_orb=round(Ω_ORB,digits=3) Ω_brk_surface=round(Ω_BRK,digits=3) frac_of_breakup=round(Ω_SPIN/Ω_BRK,digits=3) v_star_y=round(v_bh2_y,digits=4)
        Ω_SPIN / Ω_BRK > 0.9 && @warn "Requested Ω_spin > 0.9 Ω_brk — overlay is unphysical above breakup; use --scf-ic"
    else
        @info "Stellar rotation DISABLED (--no-spin)"
    end
end

fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

# ---------------------------------------------------------------------------
# Step 4: thermal bomb + BH2 activation

# Inner cutout for the thermal bomb.  Default: exclude the hollowed core only
# (r < R_CORE), matching CLAUDE.md §6.1 ("No energy deposited in the BH2 sink
# region").  With `--r-bomb-inner-frac FRAC` > 0, the cutoff is pushed out to
# FRAC · R_STAR so only the outer envelope is shocked — inner layers retain
# their pre-SN tidal-sync velocity as high-AM fallback.
R_BOMB_INNER = r_bomb_inner_frac_arg >= 0.0 ?
               r_bomb_inner_frac_arg * R_STAR : R_CORE

U_cpu = Array(U)

# Bomb outer radius: geometric (r_bomb_outer_frac · R_STAR) by default, or — if
# --bomb-mass-frac > 0 — the radius enclosing that fraction of the built star's
# envelope mass (robust to a relaxed star whose extent differs from R_STAR).
R_BOMB_EFF = bomb_mass_frac_arg > 0.0 ?
    radius_for_mass_fraction(U_cpu, bomb_mass_frac_arg, x_star_center,
                                  R_BOMB_INNER, NX, NY, NZ, DX, x0, y0, z0,
                                  2.0 * RHO_FLOOR) :
    R_BOMB

@info "Applying thermal bomb..." E_SN=round(E_SN,sigdigits=4) r_bomb=round(R_BOMB_EFF,digits=4) r_bomb_inner=round(R_BOMB_INNER,digits=4) bomb_mass_frac=bomb_mass_frac_arg center_x=round(x_star_center,digits=4) bipolar_theta_deg=bipolar_theta_deg_arg

M_bomb = thermal_bomb!(U_cpu, NX, NY, NZ, DX, DX, DX;
                       E_SN = E_SN, r_bomb = R_BOMB_EFF, r_bomb_inner = R_BOMB_INNER,
                       x0 = x0, y0 = y0, z0 = z0,
                       x_center = x_star_center, y_center = 0.0, z_center = 0.0,
                       bipolar_theta_deg = bipolar_theta_deg_arg)
fill_ghost_3d_outflow!(U_cpu, NX, NY, NZ)
copyto!(U, U_cpu)

bh2 = BlackHole(
    [x_star_center, 0.0, 0.0],
    [0.0, v_bh2_y, 0.0] .+ V_KICK,
    M_BH2_INIT,
    eps_bh,
    units.c_code,
    r_floor)
bhs = BlackHole[bh1, bh2]

@info "BH2 activated" pos=bh2.pos vel=bh2.vel mass=round(bh2.mass,digits=4) r_sink=round(r_sink(bh2),digits=4)
@info "M_bomb" M_bomb=round(M_bomb,digits=4) M_ejecta=round(M_STAR-M_BH2_INIT,digits=4)

# BH2 sink activation delay: lets the bomb-driven blast clear r_sink(BH2)
# before sinks turn on, so we don't accrete the cold envelope material that
# starts inside r_sink at IC.  Default = 2 · r_sink / c_s_post_bomb where
# c_s² = γ(γ-1) · E_SN/M_bomb (specific thermal energy in the bombed shell).
    c_s_post_bomb = sqrt(GAMMA * (GAMMA - 1.0) * E_SN / M_bomb)
    t_bh2_sink_on_val = bh2_sink_delay_arg >= 0.0 ? bh2_sink_delay_arg :
                        2.0 * r_sink(bh2) / c_s_post_bomb
    @info "BH2 sink delay" t_activate=round(t_bh2_sink_on_val, sigdigits=3) c_s_post_bomb=round(c_s_post_bomb, sigdigits=3) frac_of_P0=round(t_bh2_sink_on_val/(2π), sigdigits=3)

    t0      = 0.0
    n_step0 = 0
    t_snap0 = DT_SNAP
else
    # ---- Restart: load the evolved uniform state; the IC build above is skipped.
    @info "Restarting from checkpoint (IC build skipped)" path=RESTART_PATH
    U_load, bhs_load, grid_load, meta_load = load_checkpoint_uniform(RESTART_PATH)
    (grid_load.nx == NX && grid_load.ny == NY && grid_load.nz == NZ) ||
        error("checkpoint grid $(grid_load.nx)×$(grid_load.ny)×$(grid_load.nz) ≠ run grid $(NX)×$(NY)×$(NZ)")
    isapprox(grid_load.dx, DX; rtol = 1e-9) ||
        error("checkpoint dx=$(grid_load.dx) ≠ run dx=$(DX) — different --L?")
    size(U_load) == size(U) ||
        error("checkpoint U shape $(size(U_load)) ≠ allocated $(size(U))")
    copyto!(U, U_load)
    bhs = bhs_load
    t_bh2_sink_on_val = meta_load.t_bh2_sink_on
    t0      = meta_load.t
    n_step0 = Int(meta_load.step)
    t_snap0 = meta_load.t_snap
    @info "Restart state loaded" t=round(t0, digits=4) step=n_step0 n_bhs=length(bhs) M_BH1=round(bhs[1].mass, digits=5) M_BH2=round(bhs[2].mass, digits=5)
end

# Sink-activation time — derived from M_bomb (fresh) or carried in the
# checkpoint (restart).  Const so coupled_step!'s `t < T_BH2_SINK_ON` gate
# stays type-stable in the hot loop.
const T_BH2_SINK_ON = t_bh2_sink_on_val

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
diag_file = joinpath(OUTDIR, "diagnostics.csv")
snap_path(idx) = joinpath(OUTDIR, @sprintf("snap_t%03d.h5", idx))

if !RESTARTING
    init_trajectory_file(traj_file, 2)
    open(diag_file, "w") do f
        println(f, "t,Mdot1,Mdot2,M_BH1,M_BH2,r_sep," *
                   "E_gas,Jz_gas,M_bound," *
                   "Fgx1,Fgy1,Fgz1,Fgx2,Fgy2,Fgz2")
    end
    write_snapshot(snap_path(0), Array(U), NX, NY, NZ, DX, DX, DX, 0.0, GAMMA)
    append_trajectory(traj_file, 0.0, bhs)
    @info "Snapshot 0 written"
else
    # Restart appends to the existing trajectory.h5 / diagnostics.csv; the
    # snapshots already on disk up to t0 are kept.
    @info "Restart — appending to existing trajectory.h5 / diagnostics.csv" from_t=round(t0, digits=4)
end

# ---------------------------------------------------------------------------
# Coupled SSP-RK3 step: live orbit + torque-free sinks + density-threshold
# accretion.  Optional gas self-gravity (FFT Poisson solver, isolated BCs)
# added at each RK stage when --self-gravity is set.

function coupled_step!(U, Un, dU, Fx, Fy, Fz, bhs, dt, t,
                        nx, ny, nz, dx, x0, y0, z0, γ,
                        f_sink, ρ_floor, P_floor, torque_free, self_gravity)
    # Gas always feels gravity from all BHs; sinks are gated by t ≥ t_activate.
    bhs_sink = t < T_BH2_SINK_ON ? bhs[1:1] : bhs

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
    self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx)
    add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs_sink, x0, y0, z0;
                       f_sink = f_sink, torque_free = torque_free,
                       ρ_sink_min = RHO_SINK_MIN)
    @. U = Un + dt * dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 2
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx)
    add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs_sink, x0, y0, z0;
                       f_sink = f_sink, torque_free = torque_free,
                       ρ_sink_min = RHO_SINK_MIN)
    @. U = 0.75*Un + 0.25*U + 0.25*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    # Stage 3
    euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                 bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    add_bh_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, y0, z0)
    self_gravity && add_self_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx)
    add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs_sink, x0, y0, z0;
                       f_sink = f_sink, torque_free = torque_free,
                       ρ_sink_min = RHO_SINK_MIN)
    @. U = (1/3)*Un + (2/3)*U + (2/3)*dt*dU
    apply_floors_3d!(U, nx, ny, nz, ρ_floor, P_floor, γ)

    for bh in bhs_sink
        accrete!(bh, U, nx, ny, nz, dx, dx, dx, x0, y0, z0, dt;
                 f_sink = f_sink, ρ_sink_min = RHO_SINK_MIN)
    end

    return F_gas
end

# ---------------------------------------------------------------------------
# Time integration

@info "Starting coupled evolution" t_end=T_END P0=round(2π,digits=3) torque_free=torque_free
t_wall = time()

function run_evolution!(U, Un, dU, Fx, Fy, Fz, bhs,
                         traj_file, diag_file, snap_path, torque_free,
                         t0, n_step0, t_snap0, t_wall0;
                         checkpoint_every_min = 0.0, wall_budget_min = 0.0)
    t      = t0
    n_step = n_step0
    t_snap = t_snap0
    dt     = 0.0
    last_ckpt_wall = time()

    # Atomic checkpoint write: serialise to a temp file then rename, so a
    # SIGKILL mid-write can never corrupt the live chkpt.h5 a resubmit reads.
    function do_checkpoint(t, n_step, dt_last, t_snap)
        tmp = CHKPT_PATH * ".tmp"
        save_checkpoint_uniform(tmp, Array(U), bhs;
            t = t, step = n_step, dt_last = dt_last, t_snap = t_snap,
            t_bh2_sink_on = T_BH2_SINK_ON,
            nx = NX, ny = NY, nz = NZ, dx = DX, dy = DX, dz = DX, γ = GAMMA,
            x0 = x0, y0 = y0, z0 = z0, ρ_floor = RHO_FLOOR, P_floor = P_FLOOR)
        mv(tmp, CHKPT_PATH; force = true)
        @info "Checkpoint written" path=CHKPT_PATH t=round(t, digits=4) step=n_step wall_sec=round(time()-t_wall0, digits=1)
    end

    while t < T_END - 1e-12
        dt = cfl_dt_3d(U, NX, NY, NZ, DX, DX, DX, GAMMA, CFL)
        dt = min(dt, T_END - t, t_snap - t + 1e-12)
        dt = max(dt, 1e-14)

        M_prev = [bh.mass for bh in bhs]

        F_gas = coupled_step!(U, Un, dU, Fx, Fy, Fz, bhs, dt, t,
                               NX, NY, NZ, DX, x0, y0, z0, GAMMA,
                               F_SINK, RHO_FLOOR, P_FLOOR, torque_free,
                               SELF_GRAVITY)

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
            @info "Snapshot" idx=snap_idx t=round(t,digits=3) r_sep=round(r_sep,digits=4) M_BH2=round(bhs[2].mass,digits=5) M_bound=round(M_bound,digits=4)
        end

        if n_step % 500 == 0
            wall = round(time() - t_wall, digits=1)
            @info "Progress" step=n_step t=round(t,digits=3) dt_code=round(dt,sigdigits=3) wall_sec=wall
        end

        # Periodic checkpoint on a wall-clock cadence.
        if checkpoint_every_min > 0.0 && (time() - last_ckpt_wall) >= checkpoint_every_min * 60
            do_checkpoint(t, n_step, dt, t_snap)
            last_ckpt_wall = time()
        end

        # Graceful wall-budget exit: checkpoint and stop before PBS SIGKILLs
        # the job, so the batch script can auto-resubmit from chkpt.h5.
        if wall_budget_min > 0.0 && (time() - t_wall0) >= wall_budget_min * 60
            @info "Wall budget reached — checkpointing and exiting for resubmit" budget_min=wall_budget_min t=round(t, digits=4)
            do_checkpoint(t, n_step, dt, t_snap)
            return t, n_step, false   # incomplete → driver exits 2
        end
    end

    # Reached t_end.  Only when checkpointing is enabled, leave a final
    # checkpoint so a stray resubmit is a clean no-op (loads t≈t_end, exits at
    # once); runs without --checkpoint-every-min/--wall-budget-min are unchanged.
    if checkpoint_every_min > 0.0 || wall_budget_min > 0.0
        do_checkpoint(t, n_step, dt, t_snap)
    end
    return t, n_step, true
end

t_final, n_steps, completed = run_evolution!(U, Un, dU, Fx, Fy, Fz, bhs,
                                   traj_file, diag_file, snap_path, torque_free,
                                   t0, n_step0, t_snap0, t_wall;
                                   checkpoint_every_min = CHECKPOINT_EVERY_MIN,
                                   wall_budget_min = WALL_BUDGET_MIN)

t_elapsed = time() - t_wall
Mcell_per_sec = n_steps * NX * NY * NZ / t_elapsed / 1e6
@info "Done" n_steps=n_steps t_final=round(t_final,digits=4) wall_sec=round(t_elapsed,digits=1) Mcell_per_s=round(Mcell_per_sec,digits=1)
@info "BH accretion" M_BH1_final=round(bhs[1].mass,digits=4) M_BH2_final=round(bhs[2].mass,digits=4) ΔM_BH1=round(bhs[1].mass-M_BH1,digits=4) ΔM_BH2=round(bhs[2].mass-M_BH2_INIT,digits=4)
@info "Outputs" traj=traj_file diag=diag_file snapshots=OUTDIR

if completed
    # Completion sentinel for the PBS auto-resubmit chain (rc=0 + done.marker).
    open(joinpath(OUTDIR, "done.marker"), "w") do f
        println(f, "t_final=", t_final, " n_steps=", n_steps)
    end
    @info "Run complete — reached t_end" t_final=round(t_final, digits=4)
else
    @info "Run incomplete (wall-budget exit) — resubmit to continue from checkpoint" chkpt=CHKPT_PATH
    exit(2)
end
