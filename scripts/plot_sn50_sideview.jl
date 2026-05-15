#!/usr/bin/env julia
# Side-view (xz-plane, y=0 slice) density figure for the SN50 sink-delay run.
# Shows ejecta vertical extent and orbital-plane confinement at t = 0, 5, 15, 30.
#
# Usage:  julia --project=. scripts/plot_sn50_sideview.jl [--outdir PATH]
# Output: <outdir>/figures/sn50_sideview.png

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

outdir_arg = let v = "demo1/output_sn50_a0_5rsun_nx128_sinkdelay"
    for (i, arg) in enumerate(ARGS)
        if arg == "--outdir" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end

const OUTDIR  = outdir_arg
const FIGDIR  = joinpath(OUTDIR, "figures")
const DT_SNAP = 0.5
const SNAP_TIMES = [0, 5, 15, 30]
mkpath(FIGDIR)

fig = Figure(size = (1500, 450))

Label(fig[0, 1:5],
      "SN50 a₀=5 R☉ sink-delay — side view (xz plane, y=0 slice)",
      fontsize = 15, font = :bold)

t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
bh1 = bh_data[1]; bh2 = bh_data[2]

last_hm = nothing
for (k, t) in enumerate(SNAP_TIMES)
    idx = round(Int, t / DT_SNAP)
    f   = joinpath(OUTDIR, @sprintf("snap_t%03d.h5", idx))
    isfile(f) || continue

    U_s, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(f)
    jy  = ny ÷ 2 + 1                    # y = 0 slice
    ρ   = U_s[1, :, jy, :]              # [nx, nz]
    L   = nx * dx / 2
    xs  = LinRange(-L + dx/2, L - dx/2, nx)
    zs  = LinRange(-L + dx/2, L - dx/2, nz)

    ax = Axis(fig[1, k],
              title  = @sprintf("t = %d  (%.2f P₀)", t, t/(2π)),
              xlabel = "x / a₀",
              ylabel = k == 1 ? "z / a₀" : "",
              aspect = DataAspect())
    k > 1 && hideydecorations!(ax, ticks=false)

    hm = heatmap!(ax, xs, zs, log10.(max.(ρ, 1e-12)),
                   colormap = :inferno, colorrange = (-5, 1))
    global last_hm = hm

    ti = argmin(abs.(t_traj .- Float64(t)))
    scatter!(ax, [bh1.pos[ti, 1]], [bh1.pos[ti, 3]],
             color = :cyan, markersize = 9, marker = :circle,
             strokecolor = :black, strokewidth = 0.5)
    scatter!(ax, [bh2.pos[ti, 1]], [bh2.pos[ti, 3]],
             color = :yellow, markersize = 9, marker = :diamond,
             strokecolor = :black, strokewidth = 0.5)
    hlines!(ax, [0.0], color = :white, linestyle = :dash,
            linewidth = 0.5, alpha = 0.4)
end
last_hm !== nothing && Colorbar(fig[1, 5], last_hm, label = "log₁₀ ρ")

outpath = joinpath(FIGDIR, "sn50_sideview.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved" path=outpath
