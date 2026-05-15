#!/usr/bin/env julia
# Quick-look figure of a single snapshot: top-down (xy) + side (xz) density slices.
#
# Usage:
#   julia --project=. scripts/plot_sn50_snap.jl [--outdir PATH] [--snap N]
#                                                [--name NAME]
# Defaults to the highest-numbered snap in --outdir.

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

outdir_arg = let v = "demo1/output_sn50_a0_1rsun_nx256_1orbit"
    for (i, arg) in enumerate(ARGS)
        if arg == "--outdir" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end
snap_arg = let v = -1
    for (i, arg) in enumerate(ARGS)
        if arg == "--snap" && i < length(ARGS)
            v = parse(Int, ARGS[i+1])
        end
    end
    v
end
name_arg = let v = ""
    for (i, arg) in enumerate(ARGS)
        if arg == "--name" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end

# Find latest snap if not specified
snap_idx = snap_arg
if snap_idx < 0
    snaps = filter(f -> startswith(f, "snap_t") && endswith(f, ".h5"),
                   readdir(outdir_arg))
    idxs = [parse(Int, match(r"snap_t(\d+)\.h5", s).captures[1]) for s in snaps]
    isempty(idxs) && error("No snapshots in $outdir_arg")
    snap_idx = maximum(idxs)
end
snap_f = joinpath(outdir_arg, @sprintf("snap_t%03d.h5", snap_idx))
@info "Reading" snap=snap_f

U, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(snap_f)

# BH positions at this time
traj_f = joinpath(outdir_arg, "trajectory.h5")
t_traj, bh_data = read_trajectory(traj_f)
bh1 = bh_data[1]; bh2 = bh_data[2]
ti = argmin(abs.(t_traj .- t_s))

L  = nx * dx / 2
xs = LinRange(-L + dx/2, L - dx/2, nx)

ρ_xy = U[1, :, :, nz ÷ 2 + 1]
ρ_xz = U[1, :, ny ÷ 2 + 1, :]

figname = isempty(name_arg) ? @sprintf("snap_t%03d", snap_idx) : name_arg
fig = Figure(size = (1100, 480))

Label(fig[0, 1:3],
      @sprintf("%s  —  snap %d, t = %.3f (%.3f P₀)", basename(outdir_arg),
               snap_idx, t_s, t_s / (2π)),
      fontsize = 14, font = :bold)

ax1 = Axis(fig[1, 1], title = "xy (z=0)",
           xlabel = "x / a₀", ylabel = "y / a₀", aspect = DataAspect())
hm1 = heatmap!(ax1, xs, xs, log10.(max.(ρ_xy, 1e-12)),
               colormap = :inferno, colorrange = (-5, 1))
scatter!(ax1, [bh1.pos[ti, 1]], [bh1.pos[ti, 2]],
         color = :cyan, markersize = 10, marker = :circle,
         strokecolor = :black, strokewidth = 0.5, label = "BH1")
scatter!(ax1, [bh2.pos[ti, 1]], [bh2.pos[ti, 2]],
         color = :yellow, markersize = 10, marker = :diamond,
         strokecolor = :black, strokewidth = 0.5, label = "BH2")
axislegend(ax1, position = :rt)

ax2 = Axis(fig[1, 2], title = "xz (y=0)",
           xlabel = "x / a₀", ylabel = "z / a₀", aspect = DataAspect())
hm2 = heatmap!(ax2, xs, xs, log10.(max.(ρ_xz, 1e-12)),
               colormap = :inferno, colorrange = (-5, 1))
scatter!(ax2, [bh1.pos[ti, 1]], [bh1.pos[ti, 3]],
         color = :cyan, markersize = 10, marker = :circle,
         strokecolor = :black, strokewidth = 0.5)
scatter!(ax2, [bh2.pos[ti, 1]], [bh2.pos[ti, 3]],
         color = :yellow, markersize = 10, marker = :diamond,
         strokecolor = :black, strokewidth = 0.5)

Colorbar(fig[1, 3], hm1, label = "log₁₀ ρ")

figdir = joinpath(outdir_arg, "figures")
mkpath(figdir)
outpath = joinpath(figdir, figname * ".png")
save(outpath, fig, px_per_unit = 2)
@info "Saved" path=outpath

@printf "  t = %.4f   BH1 at (%.3f, %.3f, %.3f)   M = %.4f\n" t_s bh1.pos[ti, 1] bh1.pos[ti, 2] bh1.pos[ti, 3] bh1.mass[ti]
@printf "              BH2 at (%.3f, %.3f, %.3f)   M = %.4f\n" bh2.pos[ti, 1] bh2.pos[ti, 2] bh2.pos[ti, 3] bh2.mass[ti]
