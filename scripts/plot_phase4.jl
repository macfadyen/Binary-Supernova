# Phase 4 plotting script: Lane-Emden solution and polytrope IC.
# Reproduces computations from test_stellar_ic.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie

CairoMakie.activate!()

println("Computing Lane-Emden solutions for n=1, 1.5, 3...")

# --- Lane-Emden curves ---
ns_plot = [1.0, 1.5, 3.0]
colors_le = [:royalblue, :crimson, :forestgreen]
labels_le = ["n = 1  (γ = 2)", "n = 1.5  (γ = 5/3)", "n = 3  (γ = 4/3)"]

fig1 = Figure(size=(650, 420))
ax1 = Axis(fig1[1,1], xlabel="ξ", ylabel="θ(ξ)",
           title="Lane-Emden solutions",
           limits=(0, 8, 0, 1.05))

for (n, col, lbl) in zip(ns_plot, colors_le, labels_le)
    ξs, θs, _ = BinarySupernova.lane_emden(n)
    lines!(ax1, ξs, max.(θs, 0.0), color=col, linewidth=2, label=lbl)
    # Mark ξ₁
    scatter!(ax1, [ξs[end]], [0.0], color=col, markersize=8, marker=:circle)
end
hlines!(ax1, [0.0], color=:black, linewidth=0.8, linestyle=:dot)
axislegend(ax1, position=:rt)

save("docs/figures/phase4_lane_emden.png", fig1, px_per_unit=2)
println("Saved: docs/figures/phase4_lane_emden.png")

# --- Polytrope IC: radial density profile ---
println("Building 32³ polytrope IC (γ=5/3)...")

ng = BinarySupernova.NG
γ  = 5.0 / 3.0
nx = ny = nz = 32
L  = 0.4
dx = 2L / nx
x0 = -L

M_star = 0.7
R_star = 0.3

U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
ρ_c, r_scale, K = polytrope_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                                    M_star = M_star, R_star = R_star,
                                    x0 = x0, y0 = x0, z0 = x0)

# Compute analytic Lane-Emden profile for same γ
n_le = 1.0 / (γ - 1)   # = 1.5
ξs, θs, _ = BinarySupernova.lane_emden(n_le)
ξ1 = ξs[end]
r_scale_le = R_star / ξ1

# Scatter all cell radii vs ρ/ρ_c
rs_all = Float64[]
ρs_norm = Float64[]
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    xc = x0 + (i - ng - 0.5)*dx
    yc = x0 + (j - ng - 0.5)*dx
    zc = x0 + (k - ng - 0.5)*dx
    r  = sqrt(xc^2 + yc^2 + zc^2)
    ρv = U[1, i, j, k]
    push!(rs_all, r / R_star)
    push!(ρs_norm, ρv / ρ_c)
end

# Analytic curve
r_ana = range(0, 1.0, length=300)
θ_ana = [let ξ = r * ξ1
    if ξ <= 0
        1.0
    elseif ξ >= ξ1
        0.0
    else
        # interpolate θ
        idx = searchsortedfirst(ξs, ξ)
        idx = clamp(idx, 2, length(ξs))
        frac = (ξ - ξs[idx-1]) / (ξs[idx] - ξs[idx-1])
        θv = θs[idx-1] + frac * (θs[idx] - θs[idx-1])
        max(θv, 0.0)
    end
end for r in r_ana]
ρ_ana_norm = [θ^n_le for θ in θ_ana]

fig2 = Figure(size=(650, 420))
ax2 = Axis(fig2[1,1], xlabel="r / R_star", ylabel="ρ / ρ_c",
           title="Polytrope IC: density profile (32³, γ=5/3)")
scatter!(ax2, rs_all, ρs_norm, color=(:royalblue, 0.5), markersize=3,
         label="Grid cells")
lines!(ax2, collect(r_ana), ρ_ana_norm, color=:black, linewidth=2.5,
       label="Lane-Emden θⁿ")
vlines!(ax2, [1.0], color=:gray, linestyle=:dash, linewidth=1.5, label="R_star")
xlims!(ax2, 0, 1.4)
ylims!(ax2, -0.05, 1.1)
axislegend(ax2, position=:rt)

save("docs/figures/phase4_polytrope.png", fig2, px_per_unit=2)
println("Saved: docs/figures/phase4_polytrope.png")
