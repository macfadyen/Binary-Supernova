#!/usr/bin/env julia
# Production runner for BinarySupernova on NVIDIA A100.
#
# Usage:
#   julia --project=. scripts/run_production.jl [--gpu] [--nx NX] [--outdir DIR]
#
# --gpu     : use CUDA backend (A100); default is CPU (testing/debug)
# --nx NX   : grid resolution per dimension (default 128; A100 target ≥ 256)
# --outdir  : output directory for HDF5 snapshots (default "output/")
#
# Environment:
#   Loads CUDA.jl only when --gpu is passed, so the script can run on CPU
#   workstations for testing without a CUDA installation.
#
# Example (A100, 256³):
#   julia --project=. -t 1 scripts/run_production.jl --gpu --nx 256 --outdir output/run01/

using BinarySupernova

# ---------------------------------------------------------------------------
# Argument parsing

use_gpu = "--gpu" in ARGS
nx = 128
outdir = "output/"
for (i, arg) in enumerate(ARGS)
    if arg == "--nx" && i < length(ARGS)
        nx = parse(Int, ARGS[i+1])
    end
    if arg == "--outdir" && i < length(ARGS)
        outdir = ARGS[i+1]
    end
end

# ---------------------------------------------------------------------------
# Backend selection

import KernelAbstractions as KA

if use_gpu
    using CUDA
    @assert CUDA.functional() "CUDA not functional — check driver and GPU"
    backend = CUDABackend()
    @info "GPU backend: $(CUDA.name(CUDA.device()))"
    # Warm up JIT on a tiny problem before the main run
    U_warm = CUDA.zeros(Float32, 5, nx÷4 + 6, nx÷4 + 6, nx÷4 + 6)
    @info "CUDA JIT warm-up complete"
else
    backend = KA.CPU()
    @info "CPU backend: $(Threads.nthreads()) threads"
end

# ---------------------------------------------------------------------------
# Physical parameters (compact BH-star binary, code units G = M_tot = a0 = 1)

units = PhysicalUnits(20.0, 10.0)   # 20 M☉ total, 10 R☉ separation → c ≈ 485

params = SimParams(
    M_BH1      = 0.5,
    M_star     = 0.5,
    M_BH2_init = 0.1,
    a0         = 1.0,
    E_SN       = 1.0,
    r_bomb     = 0.1,
    v_kick     = [0.0, 0.0, 0.0],
    gamma      = 4/3,
    f_sink     = 1.0,
    rho_floor  = 1e-10,
    cfl        = 0.4,
)

@info "SimParams" M_BH1=params.M_BH1 M_star=params.M_star M_BH2=params.M_BH2_init

# ---------------------------------------------------------------------------
# Grid setup

ny = nx;  nz = nx
L  = 2.0         # domain half-size in code units: [-L, L]³
dx = 2L / nx
ng = BinarySupernova.NG

nxtot = nx + 2ng;  nytot = ny + 2ng;  nztot = nz + 2ng
γ  = params.gamma
x0 = -L;  y0 = -L;  z0 = -L   # physical left edge of active domain

@info "Grid" nx=nx ng=ng dx=dx nxtot=nxtot
@info "Memory" array_GB = round(5*nxtot*nytot*nztot*8/1e9, digits=2)

# ---------------------------------------------------------------------------
# Allocate state array on the selected backend

if use_gpu
    U = CUDA.zeros(Float64, 5, nxtot, nytot, nztot)
else
    U = zeros(Float64, 5, nxtot, nytot, nztot)
end

# ---------------------------------------------------------------------------
# Initial condition: Sedov-Taylor at domain centre (Phase 1 test IC)
# Replace with polytrope_ic_3d! + thermal_bomb! for production SN runs.

sedov_ic_3d!(Array(U), nx, ny, nz, dx, dx, dx, γ;
             E_blast = 1.0, ρ_bg = 1.0, P_floor = 1e-5)
if use_gpu
    U_cpu = Array(U)
    sedov_ic_3d!(U_cpu, nx, ny, nz, dx, dx, dx, γ;
                 E_blast = 1.0, ρ_bg = 1.0, P_floor = 1e-5,
                 x_offset = x0, y_offset = y0, z_offset = z0)
    copyto!(U, U_cpu)
else
    sedov_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                 E_blast = 1.0, ρ_bg = 1.0, P_floor = 1e-5,
                 x_offset = x0, y_offset = y0, z_offset = z0)
end

# ---------------------------------------------------------------------------
# Time integration

t_end   = 0.1
t       = 0.0
n_step  = 0
t_snap  = 0.0
dt_snap = t_end / 10

mkpath(outdir)

@info "Starting integration" t_end=t_end

t_start = time()

while t < t_end
    dt = cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, params.cfl)
    dt = min(dt, t_end - t)

    euler3d_step!(U, nx, ny, nz, dx, dx, dx, dt, γ;
                  bc = :outflow,
                  ρ_floor = params.rho_floor,
                  P_floor = 1e-10)

    t      += dt
    n_step += 1

    if t >= t_snap
        E_tot = gas_energy_total(U, nx, ny, nz, dx, dx, dx)
        @info "Step" n=n_step t=round(t,digits=4) dt=round(dt,sigdigits=3) E=round(E_tot,sigdigits=6)
        t_snap += dt_snap
    end
end

t_elapsed = time() - t_start
@info "Integration complete" n_steps=n_step t=t wall_sec=round(t_elapsed,digits=1) Mcell_per_sec=round(n_step*nx*ny*nz/t_elapsed/1e6, digits=1)

# ---------------------------------------------------------------------------
# Write final snapshot

U_snap = Array(U)   # no-op on CPU; device→host copy on GPU
snap_file = joinpath(outdir, "final.h5")
write_snapshot(snap_file, U_snap, nx, ny, nz, dx, dx, dx, t, BlackHole[])
@info "Snapshot written" file=snap_file
