#!/usr/bin/env julia
# Hachisu SCF (n = 3) calibration table: α → Ω_spin at fiducial (M_star, R_star).
#
# Motivation: the SCF path in run_sn50_fiducial.jl is parameterised by the
# axis ratio α = r_p / r_eq, not by Ω_spin.  This script maps the relevant
# portion of the stable branch (α ≥ α_peak ≈ 0.66) to code-unit spin for
# several fiducial orbital separations, so runs can be specified as
# "pick α to get k ≈ Ω_spin / Ω_orb" without trial-and-error.
#
# Outputs:
#   docs/figures/scf_calibration_n3.png   — k vs α and Ω/Ω_brk vs α across a₀
#   stdout table per a₀: α, Ω_code, Ω/Ω_orb (=k), Ω/Ω_brk, T/|W|, M̂
#
# Usage:  julia --project=. scripts/calibrate_scf_ic.jl

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

const FIGDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(FIGDIR)

# Fiducial progenitor (matches run_sn50_fiducial.jl)
const M_TOT_MSUN  = 105.0
const M_STAR_MSUN = 55.0
const R_STAR_RSUN = 1.0
const M_STAR      = M_STAR_MSUN / M_TOT_MSUN           # = 0.5238 code units

# a₀ cases: the a₀=2 R☉ pair is the only one where rotation near Ω_orb is a
# substantial fraction of breakup; wider separations make SCF ≈ Lane-Emden.
const A0_CASES = [2.0, 3.0, 5.0, 10.0, 20.0]

# α scan: finely spaced near 1 (weak rotation, most relevant for wide orbits)
# and stepped down to the peak (≈ 0.66 for n = 3).
const ALPHAS = [0.999, 0.995, 0.99, 0.98, 0.97, 0.95, 0.92, 0.90,
                0.87, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70, 0.68, 0.66]

# Shared SCF parameters (higher Nr than the driver for a smoother curve).
const NR_SCF    = 512
const NMU_SCF   = 33
const LMAX_SCF  = 12
const TOL_SCF   = 1e-8
const MAX_SCF   = 3000
const MIX_SCF   = 0.30

# Precompute sequence (α-dependent only; a₀ rescales).
@info "Running SCF sequence" n=3 points=length(ALPHAS) Nr=NR_SCF
seq = map(ALPHAS) do α
    sol = scf_rotating_polytrope(3.0, α;
                                  Nr = NR_SCF, Nμ = NMU_SCF, lmax = LMAX_SCF,
                                  tol = TOL_SCF, maxiter = MAX_SCF, mix = MIX_SCF)
    (α_req = α, α_act = sol.axis_ratio_actual, Ω² = max(sol.Ω², 0.0),
     M̂ = sol.M, T_over_W = sol.T_over_W, converged = sol.converged)
end

# Report per-case tables
println()
@printf "%-8s %8s %8s %8s %10s %10s %10s %10s %8s\n" "a₀/R☉" "α_req" "α_act" "M̂" "ρ_c" "Ω_code" "Ω/Ω_orb" "Ω/Ω_brk" "T/|W|"
println(repeat('-', 100))

cals = Dict{Float64, Vector{NamedTuple}}()
for a0_rsun in A0_CASES
    R_star = R_STAR_RSUN / a0_rsun                      # code units
    Ω_brk  = sqrt(M_STAR / R_star^3)
    Ω_orb  = 1.0
    rows = NamedTuple[]
    for s in seq
        ρ_c    = M_STAR / (R_star^3 * s.M̂)
        Ω_code = sqrt(s.Ω² * ρ_c)
        k      = Ω_code / Ω_orb
        fbrk   = Ω_code / Ω_brk
        @printf "%-8.2f %8.4f %8.4f %8.5f %10.3e %10.4f %10.4f %10.4f %8.4f\n" a0_rsun s.α_req s.α_act s.M̂ ρ_c Ω_code k fbrk s.T_over_W
        push!(rows, (α = s.α_act, Ω_code = Ω_code, k = k, fbrk = fbrk,
                     T_over_W = s.T_over_W))
    end
    cals[a0_rsun] = rows
    println()
end

# ---------------------------------------------------------------------------
# Figure

fig = Figure(size = (1250, 520))

Label(fig[0, 1:2],
      "Hachisu SCF (n=3) calibration — M_star = 0.5238, R_star = 1 R☉ / (a₀/R☉)";
      fontsize = 15, font = :bold)

ax_k = Axis(fig[1, 1],
            title  = "k = Ω_spin / Ω_orb  vs  axis ratio α",
            xlabel = "axis ratio α = r_p / r_eq",
            ylabel = "k = Ω_spin / Ω_orb",
            yscale = log10)

ax_f = Axis(fig[1, 2],
            title  = "Ω_spin / Ω_brk  vs  axis ratio α",
            xlabel = "axis ratio α",
            ylabel = "Ω_spin / Ω_brk")

colors = (:royalblue, :darkorange, :seagreen, :purple, :gray30)
for (col, a0) in zip(colors, A0_CASES)
    rows = cals[a0]
    αs = [r.α for r in rows]
    ks = [max(r.k, 1e-6) for r in rows]   # log scale safety
    fs = [r.fbrk for r in rows]
    lines!(ax_k, αs, ks; color = col, linewidth = 2.2,
           label = @sprintf("a₀ = %g R☉", a0))
    scatter!(ax_k, αs, ks; color = col, markersize = 6)
    lines!(ax_f, αs, fs; color = col, linewidth = 2.2,
           label = @sprintf("a₀ = %g R☉", a0))
    scatter!(ax_f, αs, fs; color = col, markersize = 6)
end
hlines!(ax_k, [1.0];  color = :black, linestyle = :dash, linewidth = 1)
hlines!(ax_f, [1.0];  color = :crimson, linestyle = :dash, linewidth = 1,
        label = "breakup")
axislegend(ax_k, position = :rt)
axislegend(ax_f, position = :rt)

outpath = joinpath(FIGDIR, "scf_calibration_n3.png")
save(outpath, fig; px_per_unit = 2)
@info "Figure saved" path=outpath

# ---------------------------------------------------------------------------
# Suggest alpha for common k targets at each a₀

println()
println("Suggested α for target k (linear-interpolation on the tabulated sequence):")
@printf "%-8s " "a₀/R☉"
k_targets = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
for k in k_targets
    @printf "k=%-5.2f " k
end
println()

function interp_alpha_for_k(rows, k_target)
    # rows sorted α decreasing, k increasing
    length(rows) < 2 && return NaN
    k_vec = [r.k for r in rows]
    α_vec = [r.α for r in rows]
    k_target < k_vec[1]        && return α_vec[1]        # weaker than table → α≈1
    k_target > k_vec[end]      && return NaN             # exceeds mass-shedding table
    for i in 1:length(rows)-1
        if k_vec[i] <= k_target <= k_vec[i+1]
            t = (k_target - k_vec[i]) / (k_vec[i+1] - k_vec[i])
            return α_vec[i] + t * (α_vec[i+1] - α_vec[i])
        end
    end
    return NaN
end

for a0 in A0_CASES
    @printf "%-8.2f " a0
    rows = cals[a0]
    for k in k_targets
        α = interp_alpha_for_k(rows, k)
        if isnan(α)
            @printf "%-8s " "—"
        else
            @printf "%-8.4f " α
        end
    end
    println()
end
println("(— means target k exceeds the mass-shedding limit at this a₀)")
