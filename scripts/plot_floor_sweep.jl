#!/usr/bin/env julia
# Plot the floor-value sweep (scripts/run_floor_sweep.jl).
#
# Four log-log panels vs ρ_floor: ΔM_BH1, ΔM_BH2, M_bound, max|F_gas→BH|.
# Horizontal reference lines for M_ejecta and the converged plateau.
#
# Usage:  julia --project=. scripts/plot_floor_sweep.jl
# Output: demo1/floor_sweep/sweep_summary.png

using CairoMakie
using DelimitedFiles: readdlm
using Printf

CairoMakie.activate!()

const SUMMARY = "demo1/floor_sweep/sweep_summary.csv"
const FIGOUT  = "demo1/floor_sweep/sweep_summary.png"

D = readdlm(SUMMARY, ',', skipstart=1)
ρ_floor = Float64.(D[:, 1])
dM_BH1  = Float64.(D[:, 8])
dM_BH2  = Float64.(D[:, 9])
M_bound = Float64.(D[:, 10])
max_Fg1 = Float64.(D[:, 13])
max_Fg2 = Float64.(D[:, 14])

# Reference values for the SN30 fixed-orbit setup used by the sweep
const M_EJECTA = 30.0 / 90.0      # M_star − M_BH2_init in code units

fig = Figure(size = (1200, 800))

Label(fig[0, 1:2],
      "Floor-value sweep — SN30 fixed-orbit, NX=64, ρ_sink_min = 2·ρ_floor",
      fontsize = 15, font = :bold)

# Panel 1: ΔM_BH1
ax1 = Axis(fig[1, 1],
           xscale = log10, yscale = log10,
           xlabel = "ρ_floor", ylabel = "ΔM_BH1",
           title  = "BH1 mass gain (artifact; should → 0)")
scatterlines!(ax1, ρ_floor, dM_BH1, color = :crimson, markersize = 12, linewidth = 2)
hlines!(ax1, [0.01 * 30.0/90.0], color = :gray, linestyle = :dash,
        label = "1% of M_BH1")
axislegend(ax1, position = :rb)

# Panel 2: ΔM_BH2
ax2 = Axis(fig[1, 2],
           xscale = log10, yscale = log10,
           xlabel = "ρ_floor", ylabel = "ΔM_BH2",
           title  = "BH2 mass gain (physical fallback + artifact)")
scatterlines!(ax2, ρ_floor, dM_BH2, color = :darkgreen, markersize = 12, linewidth = 2)
hlines!(ax2, [M_EJECTA], color = :gray, linestyle = :dash, label = "M_ejecta")
axislegend(ax2, position = :rt)

# Panel 3: M_bound
ax3 = Axis(fig[2, 1],
           xscale = log10, yscale = log10,
           xlabel = "ρ_floor", ylabel = "M_bound",
           title  = "Bound gas mass at t_final")
scatterlines!(ax3, ρ_floor, M_bound, color = :teal, markersize = 12, linewidth = 2)
hlines!(ax3, [M_EJECTA], color = :gray, linestyle = :dash, label = "M_ejecta")
axislegend(ax3, position = :rb)

# Panel 4: max|F_gas→BH|
ax4 = Axis(fig[2, 2],
           xscale = log10, yscale = log10,
           xlabel = "ρ_floor", ylabel = "max |F_gas→BH|",
           title  = "Peak gas force (diagnostic, not fed back)")
scatterlines!(ax4, ρ_floor, max_Fg1, color = :royalblue,
              markersize = 12, linewidth = 2, label = "on BH1")
scatterlines!(ax4, ρ_floor, max_Fg2, color = :orange,
              markersize = 12, linewidth = 2, label = "on BH2")
axislegend(ax4, position = :rt)

save(FIGOUT, fig, px_per_unit = 2)
@info "Saved" path=FIGOUT

# ---- Short text summary ----

println("\n=== Floor-value sweep summary ===")
for k in 1:length(ρ_floor)
    @printf("  ρ_floor = %.1e:  ΔM_BH1 = %.4f   ΔM_BH2 = %.4f   M_bound = %.4f\n",
            ρ_floor[k], dM_BH1[k], dM_BH2[k], M_bound[k])
end
println("  M_ejecta (reference) = $(round(M_EJECTA, digits=4))")
