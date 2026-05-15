#!/usr/bin/env julia
# Resolution-convergence comparison: NX=64 vs NX=128 SN50 a₀=5 R☉ sink-delay runs.
#
# Shows:
#   Row 1: density slices at t=0, 5, 15, 30 (NX=64)
#   Row 2: density slices at t=0, 5, 15, 30 (NX=128)
#   Row 3: M_BH1(t), M_BH2(t), r_sep(t), M_bound(t) overlaid (both resolutions)
#
# Usage:  julia --project=. scripts/plot_sn50_resolution.jl
# Output: demo1/sn50_resolution_comparison.png

using BinarySupernova
using CairoMakie
using DelimitedFiles: readdlm
using Printf

CairoMakie.activate!()

const DIR_64  = "demo1/output_sn50_a0_5rsun_nx64_sinkdelay"
const DIR_128 = "demo1/output_sn50_a0_5rsun_nx128_sinkdelay"
const FIGOUT  = "demo1/sn50_resolution_comparison.png"
const DT_SNAP = 0.5
const M_EJECTA = 10.0 / 105.0

const SNAP_TIMES = [0, 5, 15, 30]

function panel_density!(fig, row, col, dir, t, dt_snap, label_y)
    idx = round(Int, t / dt_snap)
    f   = joinpath(dir, @sprintf("snap_t%03d.h5", idx))
    isfile(f) || return nothing

    U_s, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(f)
    kz  = nz ÷ 2 + 1
    ρ   = U_s[1, :, :, kz]
    L   = nx * dx / 2
    xs  = LinRange(-L + dx/2, L - dx/2, nx)

    ax = Axis(fig[row, col],
              title  = col == 1 ? @sprintf("t = %d", t) : @sprintf("t = %d", t),
              xlabel = row == 2 && col == 1 ? "x / a₀" : "",
              ylabel = col == 1 ? label_y * "\ny / a₀" : "",
              aspect = DataAspect())
    col > 1 && hideydecorations!(ax, ticks=false)
    row == 1 && hidexdecorations!(ax, ticks=false)

    hm = heatmap!(ax, xs, xs, log10.(max.(ρ, 1e-12)),
                   colormap = :inferno, colorrange = (-5, 1))

    # trajectory markers at this time
    t_traj, bh_data = read_trajectory(joinpath(dir, "trajectory.h5"))
    bh1 = bh_data[1]; bh2 = bh_data[2]
    ti = argmin(abs.(t_traj .- Float64(t)))
    scatter!(ax, [bh1.pos[ti, 1]], [bh1.pos[ti, 2]],
             color = :cyan, markersize = 8, marker = :circle,
             strokecolor = :black, strokewidth = 0.5)
    scatter!(ax, [bh2.pos[ti, 1]], [bh2.pos[ti, 2]],
             color = :yellow, markersize = 8, marker = :diamond,
             strokecolor = :black, strokewidth = 0.5)
    return hm
end

fig = Figure(size = (1500, 1000))

Label(fig[0, 1:5],
      "SN50 a₀=5 R☉ sink-delay: resolution comparison (NX=64 vs NX=128)",
      fontsize = 16, font = :bold)

# --- Rows 1 & 2: density slices
last_hm = nothing
for (k, t) in enumerate(SNAP_TIMES)
    panel_density!(fig, 1, k, DIR_64,  t, DT_SNAP, "NX=64")
    hm = panel_density!(fig, 2, k, DIR_128, t, DT_SNAP, "NX=128")
    hm !== nothing && (last_hm = hm)
end
if last_hm !== nothing
    Colorbar(fig[1:2, 5], last_hm, label = "log₁₀ ρ")
end

# --- Row 3: time-series overlays (four panels)

function load_diag(dir)
    D = readdlm(joinpath(dir, "diagnostics.csv"), ',', Float64; skipstart=1)
    t       = D[:, 1]
    M_BH1   = D[:, 4]
    M_BH2   = D[:, 5]
    r_sep   = D[:, 6]
    M_bound = D[:, 9]
    (t=t, M_BH1=M_BH1, M_BH2=M_BH2, r_sep=r_sep, M_bound=M_bound)
end

d64  = load_diag(DIR_64)
d128 = load_diag(DIR_128)

# Panel (1) M_BH1
ax1 = Axis(fig[3, 1], xlabel = "t", ylabel = "M_BH1", title = "BH1 mass")
lines!(ax1, d64.t,  d64.M_BH1,  color = :royalblue, linewidth = 1.5, label = "NX=64")
lines!(ax1, d128.t, d128.M_BH1, color = :crimson,   linewidth = 1.5, label = "NX=128")
axislegend(ax1, position = :rb)

# Panel (2) M_BH2
ax2 = Axis(fig[3, 2], xlabel = "t", ylabel = "M_BH2", title = "BH2 mass (+fallback)")
lines!(ax2, d64.t,  d64.M_BH2,  color = :royalblue, linewidth = 1.5, label = "NX=64")
lines!(ax2, d128.t, d128.M_BH2, color = :crimson,   linewidth = 1.5, label = "NX=128")
hlines!(ax2, [d128.M_BH2[1] + M_EJECTA], color = :black, linestyle = :dot,
        linewidth = 1, label = "M_BH2_0 + M_ejecta")
axislegend(ax2, position = :rb)

# Panel (3) r_sep
ax3 = Axis(fig[3, 3], xlabel = "t", ylabel = "r_sep / a₀", title = "Binary separation")
lines!(ax3, d64.t,  d64.r_sep,  color = :royalblue, linewidth = 1.5, label = "NX=64")
lines!(ax3, d128.t, d128.r_sep, color = :crimson,   linewidth = 1.5, label = "NX=128")
hlines!(ax3, [1.0], color = :gray, linestyle = :dash, linewidth = 1, label = "a₀")
axislegend(ax3, position = :rb)

# Panel (4) M_bound (log scale)
ax4 = Axis(fig[3, 4], xlabel = "t", ylabel = "M_bound", title = "Bound gas mass",
           yscale = log10)
M_bound_64  = max.(d64.M_bound,  1e-6)
M_bound_128 = max.(d128.M_bound, 1e-6)
lines!(ax4, d64.t,  M_bound_64,  color = :royalblue, linewidth = 1.5, label = "NX=64")
lines!(ax4, d128.t, M_bound_128, color = :crimson,   linewidth = 1.5, label = "NX=128")
hlines!(ax4, [M_EJECTA], color = :gray, linestyle = :dash, linewidth = 1, label = "M_ejecta")
axislegend(ax4, position = :rt)

save(FIGOUT, fig, px_per_unit = 2)
@info "Saved" path=FIGOUT

# --- Text summary
println("\n=== SN50 a₀=5 R☉ sink-delay: resolution comparison ===")
@printf "  NX=64   final:  ΔM_BH1 = %+.4f   ΔM_BH2 = %+.4f   r_sep = %.4f   M_bound = %.4e\n" (d64.M_BH1[end]-d64.M_BH1[1])   (d64.M_BH2[end]-d64.M_BH2[1])   d64.r_sep[end]  d64.M_bound[end]
@printf "  NX=128  final:  ΔM_BH1 = %+.4f   ΔM_BH2 = %+.4f   r_sep = %.4f   M_bound = %.4e\n" (d128.M_BH1[end]-d128.M_BH1[1]) (d128.M_BH2[end]-d128.M_BH2[1]) d128.r_sep[end] d128.M_bound[end]
@printf "  M_ejecta reference = %.4f\n" M_EJECTA
