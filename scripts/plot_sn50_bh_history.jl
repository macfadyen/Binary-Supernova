#!/usr/bin/env julia
# BH position and mass time histories in physical units (R☉ and M☉).
#
# Usage:  julia --project=. scripts/plot_sn50_bh_history.jl [--outdir PATH]
#                                                            [--a0-rsun A0]
#                                                            [--mtot-msun M]
# Output: <outdir>/figures/bh_history.png

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
a0_rsun_arg = let v = 1.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--a0-rsun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end
mtot_msun_arg = let v = 105.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--mtot-msun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end

const OUTDIR    = outdir_arg
const A0_RSUN   = a0_rsun_arg
const MTOT_MSUN = mtot_msun_arg

@info "Reading trajectory" path=joinpath(OUTDIR, "trajectory.h5")
t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
bh1 = bh_data[1]; bh2 = bh_data[2]

# Code→physical unit conversions
x1 = bh1.pos[:, 1] .* A0_RSUN   # R☉
y1 = bh1.pos[:, 2] .* A0_RSUN
z1 = bh1.pos[:, 3] .* A0_RSUN
x2 = bh2.pos[:, 1] .* A0_RSUN
y2 = bh2.pos[:, 2] .* A0_RSUN
z2 = bh2.pos[:, 3] .* A0_RSUN

M1 = bh1.mass .* MTOT_MSUN      # M☉
M2 = bh2.mass .* MTOT_MSUN

t_orb = t_traj ./ (2π)          # orbital periods P₀

r_sep = sqrt.((x2 .- x1).^2 .+ (y2 .- y1).^2 .+ (z2 .- z1).^2)

fig = Figure(size = (1400, 800))

Label(fig[0, 1:3],
      @sprintf("BH time history — %s  (a₀ = %.1f R☉, M_tot = %.0f M☉)",
               basename(OUTDIR), A0_RSUN, MTOT_MSUN),
      fontsize = 15, font = :bold)

# ---- Row 1: orbit traces in xy and xz

ax_xy = Axis(fig[1, 1],
             title  = "Orbit (xy)",
             xlabel = "x (R☉)", ylabel = "y (R☉)",
             aspect = DataAspect())
lines!(ax_xy, x1, y1, color = :crimson,   linewidth = 1.5, label = "BH1")
lines!(ax_xy, x2, y2, color = :darkgreen, linewidth = 1.5, label = "BH2")
scatter!(ax_xy, [x1[1]],   [y1[1]],   color = :crimson,   markersize = 9)
scatter!(ax_xy, [x2[1]],   [y2[1]],   color = :darkgreen, markersize = 9)
scatter!(ax_xy, [x1[end]], [y1[end]], color = :crimson,   marker = :star5, markersize = 14)
scatter!(ax_xy, [x2[end]], [y2[end]], color = :darkgreen, marker = :star5, markersize = 14)
axislegend(ax_xy, position = :rt)

ax_xz = Axis(fig[1, 2],
             title  = "Orbit (xz)",
             xlabel = "x (R☉)", ylabel = "z (R☉)",
             aspect = DataAspect())
lines!(ax_xz, x1, z1, color = :crimson,   linewidth = 1.5, label = "BH1")
lines!(ax_xz, x2, z2, color = :darkgreen, linewidth = 1.5, label = "BH2")
scatter!(ax_xz, [x1[1]],   [z1[1]],   color = :crimson,   markersize = 9)
scatter!(ax_xz, [x2[1]],   [z2[1]],   color = :darkgreen, markersize = 9)
scatter!(ax_xz, [x1[end]], [z1[end]], color = :crimson,   marker = :star5, markersize = 14)
scatter!(ax_xz, [x2[end]], [z2[end]], color = :darkgreen, marker = :star5, markersize = 14)
axislegend(ax_xz, position = :rt)

# ---- (right) separation vs time

ax_sep = Axis(fig[1, 3],
              title  = "Binary separation",
              xlabel = "t (P₀)", ylabel = "|r₂ − r₁| (R☉)")
lines!(ax_sep, t_orb, r_sep, color = :royalblue, linewidth = 2)
hlines!(ax_sep, [A0_RSUN], color = :gray, linestyle = :dash,
        linewidth = 1, label = "a₀")
axislegend(ax_sep, position = :rb)

# ---- Row 2: position components vs time

ax_posx = Axis(fig[2, 1],
               title  = "Position x(t)",
               xlabel = "t (P₀)", ylabel = "x (R☉)")
lines!(ax_posx, t_orb, x1, color = :crimson,   linewidth = 1.5, label = "BH1")
lines!(ax_posx, t_orb, x2, color = :darkgreen, linewidth = 1.5, label = "BH2")
axislegend(ax_posx, position = :rt)

ax_posy = Axis(fig[2, 2],
               title  = "Position y(t)",
               xlabel = "t (P₀)", ylabel = "y (R☉)")
lines!(ax_posy, t_orb, y1, color = :crimson,   linewidth = 1.5, label = "BH1")
lines!(ax_posy, t_orb, y2, color = :darkgreen, linewidth = 1.5, label = "BH2")
axislegend(ax_posy, position = :rt)

# ---- (right) masses in M☉

ax_m = Axis(fig[2, 3],
            title  = "BH masses",
            xlabel = "t (P₀)", ylabel = "M (M☉)")
lines!(ax_m, t_orb, M1, color = :crimson,   linewidth = 2, label = "BH1")
lines!(ax_m, t_orb, M2, color = :darkgreen, linewidth = 2, label = "BH2")
hlines!(ax_m, [M1[1]], color = :crimson,   linestyle = :dash, linewidth = 1)
hlines!(ax_m, [M2[1]], color = :darkgreen, linestyle = :dash, linewidth = 1)
axislegend(ax_m, position = :rc)

figdir = joinpath(OUTDIR, "figures")
mkpath(figdir)
outpath = joinpath(figdir, "bh_history.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved" path=outpath

# ---- Summary printout ----
println()
@printf "Final t       : %.3f code (%.3f P₀)\n"           t_traj[end]  t_orb[end]
@printf "BH1: M %.3f → %.3f M☉  (ΔM = %+.4f M☉)\n"       M1[1] M1[end] (M1[end]-M1[1])
@printf "BH2: M %.3f → %.3f M☉  (ΔM = %+.4f M☉)\n"       M2[1] M2[end] (M2[end]-M2[1])
@printf "r_sep: init %.3f R☉  min %.3f  max %.3f  final %.3f\n" r_sep[1] minimum(r_sep) maximum(r_sep) r_sep[end]
