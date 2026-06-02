# CBD-from-fallback campaign — the kick lever.
#
# Tests whether tuning the BH2 natal kick can shrink the post-SN orbit enough
# that spin-fed fallback (whose circularisation radius r_circ is set by the
# pre-SN star and is FIXED) becomes circumbinary.  Sweeps the anti-orbital kick
# k_y for the CBD-optimal config (15/30/20 M☉, a₀=2.5 R☉, SCF α=0.72) and plots
# the post-SN orbit (a_post, periastron, cavity edge 2·a_post) against the fixed
# r_circ.  Mirrors `scripts/predict_disk.jl --scan-kick`.
#
# Saves docs/figures/cbd_campaign_kick.png.

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

# --- CBD-optimal configuration ---------------------------------------------
m_bh1, m_star, m_bh2 = 15.0, 30.0, 20.0       # M☉
a0_rsun     = 2.5
R_star_rsun = 1.0
α           = 0.72                            # SCF axis ratio
cavity_fac  = 2.0                             # CBD inner edge / a (Artymowicz-Lubow)

M_TOT = m_bh1 + m_star
M_BH1, M_STAR, M_BH2 = m_bh1 / M_TOT, m_star / M_TOT, m_bh2 / M_TOT
M_bin  = M_BH1 + M_BH2                         # post-SN binary
M_pre  = M_BH1 + M_STAR                        # = 1 (code units)
R_STAR = R_star_rsun / a0_rsun
coll   = R_STAR                                # collision proxy: periastron within R★

eggleton(q) = 0.49 * q^(2/3) / (0.6 * q^(2/3) + log1p(q^(1/3)))
RL_bh2 = eggleton(M_BH2 / M_BH1)               # post-SN BH2 Roche-lobe / a

# v_orb in km/s, to label the kick axis against real natal kicks
const G_CGS, MSUN_G, RSUN_CM = 6.674e-8, 1.989e33, 6.957e10
v_orb_kms = sqrt(G_CGS * M_TOT * MSUN_G / (a0_rsun * RSUN_CM)) / 1e5

# --- r_circ from the SCF rotating-polytrope figure (= predict_disk budget) --
println("SCF solve (n=3, α=$α) ...")
sol = BinarySupernova.scf_rotating_polytrope(3.0, α; Nr = 256, Nμ = 33, lmax = 12,
                                             tol = 1e-7, mix = 0.35, maxiter = 3000)
ω_over_brk = sqrt(max(sol.Ω², 0.0) / sol.M)    # Ω_spin/Ω_brk, pure (n,α)
Ω_BRK  = sqrt(M_STAR / R_STAR^3)
Ω_SPIN = ω_over_brk * Ω_BRK
j_spin = Ω_SPIN * R_STAR^2                      # specific AM about BH2
r_circ = j_spin^2 / M_BH2                       # spin-fed mini-disc radius (FIXED)
a_cbd  = r_circ / cavity_fac                    # a_post a settled CBD would require
@printf("r_circ = %.4f a0  (settled CBD needs a_post <= %.4f a0)\n", r_circ, a_cbd)

# --- Post-SN orbit (run frame: sep along −x, v_rel along −y; k_y>0 anti-orbital)
function orbit(ky)
    v_orb = sqrt(M_pre)
    vy = -v_orb + ky                            # tangential; k_y>0 reduces |v|
    ε  = 0.5 * vy^2 - M_bin                      # specific energy at separation a₀=1
    h2 = (1.0 - ky)^2                            # |r×v|² with r=(−1,0,0), v=(0,vy,0)
    bound  = ε < 0.0
    a_post = bound ? -M_bin / (2ε) : Inf
    e      = sqrt(max(1.0 + 2.0 * ε * h2 / M_bin^2, 0.0))
    r_peri = bound ? a_post * (1.0 - e) : NaN
    return a_post, e, r_peri
end

kys      = collect(range(0.0, 0.6, length = 361))
a_post   = similar(kys);  r_peri = similar(kys)
for (i, ky) in enumerate(kys)
    a_post[i], _, r_peri[i] = orbit(ky)
end
cav_edge = cavity_fac .* a_post                 # CBD cavity edge = 2·a_post
clear    = r_circ ./ cav_edge                   # ≥ 1 would clear the bar

# Regime onsets along the (monotone) k_y sweep
i_over  = findfirst(i -> r_circ > RL_bh2 * a_post[i], eachindex(kys))  # BH2 lobe overflow
i_merge = findfirst(i -> r_peri[i] < coll,            eachindex(kys))  # periastron plunge
ky_over  = i_over  === nothing ? NaN : kys[i_over]
ky_merge = i_merge === nothing ? NaN : kys[i_merge]

# tightest bound, non-merging orbit
nm = findall(i -> r_peri[i] >= coll, eachindex(kys))
i_tight = nm[argmin(a_post[nm])]
@printf("tightest non-merging a_post = %.4f a0 (k_y=%.3f); r_circ/cavity = %.3f (<1 => no CBD)\n",
        a_post[i_tight], kys[i_tight], clear[i_tight])

# --- Figure ----------------------------------------------------------------
fig = Figure(size = (900, 760))

Label(fig[0, 1], "Can a natal kick shrink the orbit into a CBD?\n" *
      "CBD-optimal: M = 15/30/20 M⊙,  a₀ = 2.5 R⊙,  SCF α = 0.72",
      fontsize = 16, font = :bold, padding = (0, 0, 4, 0))

ax1 = Axis(fig[1, 1], ylabel = "separation  [a₀]",
           xticklabelsvisible = false)

# regime shading
isnan(ky_merge) || vspan!(ax1, ky_merge, kys[end]; color = (:firebrick, 0.10))
(isnan(ky_over) || isnan(ky_merge)) ||
    vspan!(ax1, ky_over, ky_merge; color = (:seagreen, 0.12))

lines!(ax1, kys, a_post,   color = :royalblue, linewidth = 2.5, label = "a_post (semi-major axis)")
lines!(ax1, kys, cav_edge, color = :royalblue, linewidth = 2, linestyle = :dash,
       label = "cavity edge  2·a_post")
lines!(ax1, kys, r_peri,   color = :crimson,   linewidth = 2.5, label = "periastron")
hlines!(ax1, [r_circ], color = :seagreen, linewidth = 2.5, linestyle = :dot,
        label = "r_circ (spin-fed disc — FIXED)")
hlines!(ax1, [coll],   color = :gray40,   linewidth = 1.5, linestyle = :dashdot,
        label = "collision (r_peri < R★)")
axislegend(ax1, position = :rt, framevisible = true, padding = (6, 6, 4, 4), rowgap = 0)
ylims!(ax1, 0, 2.0)
text!(ax1, isnan(ky_over) ? 0.05 : 0.5*(ky_over+ky_merge), 0.16; text = "L2 spill\n(CBD seed)",
      align = (:center, :bottom), color = :seagreen, fontsize = 11)
isnan(ky_merge) || text!(ax1, 0.5*(ky_merge+kys[end]), 0.16; text = "BH–BH\nmerger",
      align = (:center, :bottom), color = :firebrick, fontsize = 11)

ax2 = Axis(fig[2, 1],
           xlabel = "anti-orbital kick  k_y   [code units;  v_orb = $(round(Int, v_orb_kms)) km/s]",
           ylabel = "r_circ / cavity edge")
isnan(ky_merge) || vspan!(ax2, ky_merge, kys[end]; color = (:firebrick, 0.10))
(isnan(ky_over) || isnan(ky_merge)) ||
    vspan!(ax2, ky_over, ky_merge; color = (:seagreen, 0.12))
lines!(ax2, kys, clear, color = :purple, linewidth = 2.5)
hlines!(ax2, [1.0], color = :black, linewidth = 1.5, linestyle = :dash, label = "CBD threshold = 1")
scatter!(ax2, [kys[i_tight]], [clear[i_tight]], color = :purple, markersize = 10)
text!(ax2, kys[i_tight], clear[i_tight]; text = @sprintf("  max (non-merging) = %.2f ≪ 1", clear[i_tight]),
      align = (:left, :center), fontsize = 11, color = :purple)
axislegend(ax2, position = :rb, framevisible = true)
ylims!(ax2, 0, 1.1)

rowgap!(fig.layout, 6)
save("docs/figures/cbd_campaign_kick.png", fig, px_per_unit = 2)
println("Saved: docs/figures/cbd_campaign_kick.png")
