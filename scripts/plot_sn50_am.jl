#!/usr/bin/env julia
# Angular-momentum diagnostics from diagnostics.csv:
#   (1) Jz_gas(t), M_bound(t)
#   (2) specific ℓ = Jz_gas / M_bound  vs  Kepler threshold √(M_bin × r_sep)
#   (3) ℓ / ℓ_kep (CBD-formation criterion)
#   (4) ΔM_BH2(t), for context
#
# Usage: julia --project=. scripts/plot_sn50_am.jl --outdir PATH [--a0-rsun 2.0] [--mtot-msun 50.0]

using CairoMakie
using DelimitedFiles
using Printf

CairoMakie.activate!()

outdir = let v = "."
    for (i, a) in enumerate(ARGS); a == "--outdir" && i < length(ARGS) && (v = ARGS[i+1]); end; v
end
a0_rsun = let v = 2.0
    for (i, a) in enumerate(ARGS); a == "--a0-rsun" && i < length(ARGS) && (v = parse(Float64, ARGS[i+1])); end; v
end
mtot_msun = let v = 50.0
    for (i, a) in enumerate(ARGS); a == "--mtot-msun" && i < length(ARGS) && (v = parse(Float64, ARGS[i+1])); end; v
end

csv_path = joinpath(outdir, "diagnostics.csv")
@info "Reading" csv_path
raw, hdr = readdlm(csv_path, ',', header=true)
cols = Dict(strip(String(n)) => i for (i, n) in enumerate(vec(hdr)))

t      = Float64.(raw[:, cols["t"]])
M_BH1  = Float64.(raw[:, cols["M_BH1"]])
M_BH2  = Float64.(raw[:, cols["M_BH2"]])
r_sep  = Float64.(raw[:, cols["r_sep"]])
Jz     = Float64.(raw[:, cols["Jz_gas"]])
M_bnd  = Float64.(raw[:, cols["M_bound"]])

# Kepler specific AM at current binary separation for current binary mass
M_bin  = M_BH1 .+ M_BH2
ℓ_kep  = sqrt.(max.(M_bin .* r_sep, 1e-30))
ℓ      = Jz ./ max.(M_bnd, 1e-12)
ratio  = ℓ ./ ℓ_kep
ΔM_BH2 = (M_BH2 .- M_BH2[1]) .* mtot_msun  # M☉
M_bnd_Msun = M_bnd .* mtot_msun

P_orb = 2π
t_P   = t ./ P_orb

fig = Figure(size = (1200, 900))

Label(fig[0, 1:2],
      @sprintf("M20 bipolar45 — angular-momentum diagnostics  (a₀=%g R☉, M_tot=%g M☉)",
               a0_rsun, mtot_msun);
      fontsize = 15, font = :bold)

ax1 = Axis(fig[1, 1], xlabel = "t / P_orb", ylabel = "J_z,gas  (code)",
           title = "Total gas angular momentum")
lines!(ax1, t_P, Jz, color = :royalblue, linewidth = 2)

ax2 = Axis(fig[1, 2], xlabel = "t / P_orb", ylabel = "M_bound  (M☉)",
           title = "Bound gas mass", yscale = log10)
lines!(ax2, t_P, max.(M_bnd_Msun, 1e-3), color = :seagreen, linewidth = 2)

ax3 = Axis(fig[2, 1], xlabel = "t / P_orb",
           ylabel = "specific ℓ  (code)",
           title = "Specific AM of bound gas vs Kepler at r_sep")
lines!(ax3, t_P, ℓ,     color = :royalblue,  linewidth = 2, label = "ℓ = J_z / M_bound")
lines!(ax3, t_P, ℓ_kep, color = :crimson,    linewidth = 2, linestyle = :dash,
       label = "ℓ_kep = √(M_bin·r_sep)")
axislegend(ax3, position = :lt)

ax4 = Axis(fig[2, 2], xlabel = "t / P_orb",
           ylabel = "ℓ / ℓ_kep",
           title = "CBD criterion (> 1 → centrifugal support)")
lines!(ax4, t_P, ratio, color = :purple, linewidth = 2)
hlines!(ax4, [1.0]; color = :black, linestyle = :dash, linewidth = 1)

ax5 = Axis(fig[3, 1], xlabel = "t / P_orb",
           ylabel = "r_sep  (R☉)",
           title = "Binary separation")
lines!(ax5, t_P, r_sep .* a0_rsun, color = :darkorange, linewidth = 2)
hlines!(ax5, [a0_rsun]; color = :gray, linestyle = :dash, linewidth = 1, label = "a₀")
axislegend(ax5, position = :rt)

ax6 = Axis(fig[3, 2], xlabel = "t / P_orb",
           ylabel = "ΔM_BH2  (M☉)",
           title = "Fallback onto BH2")
lines!(ax6, t_P, ΔM_BH2, color = :firebrick, linewidth = 2)

figdir = joinpath(outdir, "figures")
mkpath(figdir)
outpath = joinpath(figdir, "am_diagnostics.png")
save(outpath, fig; px_per_unit = 2)
@info "Saved" outpath

println()
@printf "Final t             : %.3f code (%.3f P_orb)\n" t[end] t_P[end]
@printf "J_z,gas (final)     : %.4f code\n" Jz[end]
@printf "M_bound (final)     : %.4f code  = %.3f M☉\n" M_bnd[end] M_bnd_Msun[end]
@printf "specific ℓ (final)  : %.4f code\n" ℓ[end]
@printf "ℓ_kep at r_sep      : %.4f code   (r_sep = %.3f R☉)\n" ℓ_kep[end] (r_sep[end]*a0_rsun)
@printf "ℓ / ℓ_kep (final)   : %.4f\n" ratio[end]
@printf "ΔM_BH2 (final)      : %.3f M☉\n" ΔM_BH2[end]

peak_ratio_idx = argmax(ratio)
@printf "Max ℓ / ℓ_kep       : %.4f at t=%.2f (%.2f P_orb)\n" ratio[peak_ratio_idx] t[peak_ratio_idx] t_P[peak_ratio_idx]
