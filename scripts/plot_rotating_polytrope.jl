#!/usr/bin/env julia
# Diagnostics for the Hachisu (1986a) SCF rotating polytrope solver.
#
# Produces:
#   docs/figures/rotating_polytrope_n3.png — 4-panel summary:
#     (a) Ω²/(πGρ_c) and T/|W| along the n=3 axis-ratio sequence, with mass-
#         shedding peak marked; literature reference (0.0085) overlaid.
#     (b) Total mass M̂ along the sequence (ρ_c = r_eq = 1 units).
#     (c) Density contours in the (ϖ, z) meridional plane at three axis
#         ratios: α = 1.0 (spherical), 0.8 (oblate), α_peak (near mass
#         shedding).
#     (d) Equatorial density profile at α = 1 vs. the reference Lane-Emden
#         solution θ³(ξ).
#
# Usage:  julia --project=. scripts/plot_rotating_polytrope.jl

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

const FIGDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# (a) + (b): scan the n=3 sequence

αs_seq = collect(0.50:0.02:0.98)
@info "Scanning n=3 sequence" length=length(αs_seq)
Ω²_seq = Float64[];  T_over_W_seq = Float64[];  M_seq = Float64[]
for α in αs_seq
    sol = scf_rotating_polytrope(3.0, α;
                                  Nr = 256, Nμ = 17, lmax = 12,
                                  tol = 1e-7, maxiter = 2000, mix = 0.4)
    push!(Ω²_seq,        sol.Ω²)
    push!(T_over_W_seq,  sol.T_over_W)
    push!(M_seq,         sol.M)
end
k_peak  = argmax(Ω²_seq)
α_peak  = αs_seq[k_peak]
Ω²_peak = Ω²_seq[k_peak]
@info "Peak located" α_peak=α_peak Ω²_peak_over_πGρc=round(Ω²_peak/π, digits=5)

# ---------------------------------------------------------------------------
# (c): meridional-plane density for three representative axis ratios

sol_sph    = scf_rotating_polytrope(3.0, 1.00; Nr = 256, Nμ = 33, lmax = 12,
                                     tol = 1e-8, maxiter = 3000, mix = 0.3)
sol_oblate = scf_rotating_polytrope(3.0, 0.80; Nr = 256, Nμ = 33, lmax = 12,
                                     tol = 1e-8, maxiter = 3000, mix = 0.3)
sol_mshed  = scf_rotating_polytrope(3.0, α_peak; Nr = 256, Nμ = 33, lmax = 12,
                                     tol = 1e-8, maxiter = 3000, mix = 0.3)

"Convert (r, μ) grid + density to a (ϖ, z) meridional array on a half-plane."
function meridional(sol)
    r  = sol.r;  μ  = sol.μ;  ρ  = sol.ρ
    Nr = length(r);  Nμ = length(μ)
    ϖ = zeros(Nr, Nμ);  z = zeros(Nr, Nμ)
    for i in 1:Nr, j in 1:Nμ
        ϖ[i, j] = r[i] * sqrt(1 - μ[j]^2)
        z[i, j] = r[i] * μ[j]
    end
    return ϖ, z, ρ
end

# ---------------------------------------------------------------------------
# (d): equatorial profile at α = 1 vs Lane-Emden

ξs, θs, _ = lane_emden(3.0; dξ = 1e-3)
θ_of_r(r) = begin
    ξ = r * 6.8969
    ξ ≥ ξs[end] && return 0.0
    k = searchsortedlast(ξs, ξ)
    k == length(ξs) && return 0.0
    frac = (ξ - ξs[k]) / (ξs[k+1] - ξs[k])
    max(θs[k] + frac * (θs[k+1] - θs[k]), 0.0)
end

# ---------------------------------------------------------------------------
# Figure

fig = Figure(size = (1400, 900))

Label(fig[0, 1:3],
      "Hachisu SCF rotating polytrope — n = 3",
      fontsize = 16, font = :bold)

# (a) Ω² / πGρ_c vs α
ax_a = Axis(fig[1, 1],
            title  = "Ω² / (π G ρ_c)  vs  axis ratio α = r_p / r_eq",
            xlabel = "α",
            ylabel = "Ω̂² = Ω² / (π G ρ_c)")
lines!(ax_a, αs_seq, Ω²_seq ./ π,
       color = :royalblue, linewidth = 2.0, label = "SCF")
hlines!(ax_a, [0.0085], color = :gray, linestyle = :dash,
        label = "literature Ω̂²_ms ≈ 0.0085")
scatter!(ax_a, [α_peak], [Ω²_peak/π], color = :crimson, markersize = 12,
         marker = :star5, label = @sprintf("peak  α = %.2f", α_peak))
axislegend(ax_a, position = :lt)

# (b) Mass M̂ vs α
ax_b = Axis(fig[1, 2],
            title  = "Total mass  M̂ = M / (ρ_c r_eq³)",
            xlabel = "α",
            ylabel = "M̂")
lines!(ax_b, αs_seq, M_seq, color = :darkgreen, linewidth = 2.0)
hlines!(ax_b, [4π * 2.01824 / 6.8969^3], color = :gray,
        linestyle = :dash, label = "Lane-Emden M̂ = 0.0773")
axislegend(ax_b, position = :lt)

# (c) Meridional density contours for three axis ratios
ax_c = Axis(fig[1, 3],
            title  = "Meridional density ρ(ϖ, z) / ρ_c",
            xlabel = "ϖ / r_eq",
            ylabel = "z / r_eq",
            aspect = DataAspect())
levels = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9]
for (sol, col, lbl) in ((sol_sph,    :gray,   @sprintf("α = 1.00")),
                         (sol_oblate, :royalblue, @sprintf("α = 0.80")),
                         (sol_mshed,  :crimson,
                          @sprintf("α = %.2f (peak)", α_peak)))
    ϖ, z, ρ = meridional(sol)
    contour!(ax_c, ϖ, z, ρ; levels = levels, color = col,
             linewidth = 1.2, label = lbl)
    contour!(ax_c, ϖ, -z, ρ; levels = levels, color = col,
             linewidth = 1.2)
end
axislegend(ax_c, position = :rt)
xlims!(ax_c, 0, 1.1);  ylims!(ax_c, -1.1, 1.1)

# (d) Equatorial profile at α = 1 vs Lane-Emden
ax_d = Axis(fig[2, 1:3],
            title  = "Equatorial density profile, α = 1  (SCF non-rotating limit vs. Lane-Emden)",
            xlabel = "r / r_eq",
            ylabel = "ρ / ρ_c")
rr  = sol_sph.r
ρeq = [sol_sph.ρ[i, 1] for i in 1:length(rr)]
ρ_LE = [θ_of_r(r)^3 for r in rr]
lines!(ax_d, rr, ρeq,  color = :royalblue, linewidth = 2.5, label = "SCF (α = 1, μ = 0)")
lines!(ax_d, rr, ρ_LE, color = :crimson, linewidth = 2.5, linestyle = :dash,
       label = "Lane-Emden θ³(r·ξ₁)")
axislegend(ax_d, position = :rt)
xlims!(ax_d, 0, 1.05);  ylims!(ax_d, -0.02, 1.05)

outpath = joinpath(FIGDIR, "rotating_polytrope_n3.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved" path=outpath

println()
@printf "Summary for n = 3:\n"
@printf "  Mass-shedding peak:  α = %.3f   Ω²/(πGρ_c) = %.5f   (literature ≈ 0.0085)\n" α_peak Ω²_peak/π
@printf "  Non-rotating mass :  M = %.5f  (theory %.5f)   error %.2f%%\n" sol_sph.M (4π*2.01824/6.8969^3) 100*(sol_sph.M/(4π*2.01824/6.8969^3)-1)
