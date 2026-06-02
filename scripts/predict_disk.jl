#!/usr/bin/env julia
# predict_disk.jl — pre-run angular-momentum budget for run_sn50_fiducial.jl.
#
# Predicts the fate of supernova fallback — radial accretion onto BH2, a
# mini-disc bound to BH2, or a circumbinary disc — from the IC parameters
# alone, before committing GPU hours.  Accepts the physically-relevant flags
# of run_sn50_fiducial.jl; paste a run command and swap the script name.
# Unrecognised flags (--gpu, --torque-free, --outdir, ...) are ignored.
#
# Physics (CLAUDE.md §4, §6; binary-SN discussion notes):
#   A fluid parcel at stellar cylindrical radius ϖ, in a star rotating rigidly
#   at Ω_spin, carries specific angular momentum  j = Ω_spin ϖ²  about the
#   stellar centre.  The thermal bomb is radial and adds no AM about the spin
#   axis, so j is frozen through the explosion and the fallback circularises
#   around BH2 at
#       r_circ = j² / (G M_BH2) = Ω_spin² ϖ⁴ / M_BH2            [G = 1].
#   Outer-envelope material (ϖ → R_star) sets the disc's outer edge.  Compared
#   against BH2's Roche lobe R_L,BH2: r_circ < ~0.4 R_L is a clean mini-disc;
#   r_circ > R_L,BH2 means the disc overflows BH2's lobe and spills through L1
#   (→ BH1) and L2 (→ circumbinary).  A *settled* circumbinary disc needs
#   r_circ ≥ r_cav ≈ 2 a, i.e. specific AM j_CBD,min = √(G M_bin r_cav).
#   Stellar rotation is capped at breakup; even there
#       r_circ|_brk = (M_star/M_BH2) · R_star < a  — always a mini-disc.
#
# For the SCF path, Ω_spin/Ω_brk is a fixed (n, α) property of the figure, so
#   r_circ(a₀) = (Ω_spin/Ω_brk)² (M_star/M_BH2) (R_star/a₀)   ∝ 1/a₀,
# and one SCF solve covers a whole a₀ scan.
#
# A natal kick on BH2 (Blaauw mass loss + --v-kick-*) reshapes the post-SN
# orbit; the kick modes solve that orbit and re-test the disc fate at the new
# separation a_post (r_circ stays fixed — set by the pre-SN star — while the
# cavity bar 2·a_post moves).  A kick can shrink a_post to at most a0/2, and
# only by driving r_peri -> 0 (a BH-BH plunge); mass loss alone widens.
#
# Usage:
#   julia --project=. scripts/predict_disk.jl [run_sn50_fiducial.jl flags]
#   julia --project=. scripts/predict_disk.jl [flags] --scan-a0  LO,HI,N
#   julia --project=. scripts/predict_disk.jl [flags] --scan-alpha LO,HI,N
#   julia --project=. scripts/predict_disk.jl [flags] --v-kick-y K [--v-kick-x/-z K]
#   julia --project=. scripts/predict_disk.jl [flags] --scan-kick LO,HI,N  # sweep ky
#
# Examples:
#   julia --project=. scripts/predict_disk.jl                       # fiducial
#   julia --project=. scripts/predict_disk.jl \                     # CBD-optimal
#       --m-bh1-msun 15 --m-star-msun 30 --m-bh2-msun 20 \
#       --a0-rsun 3 --L 2 --nx 384 --scf-ic --scf-axis-ratio 0.72 \
#       --r-bomb-outer-frac 0.5
#   julia --project=. scripts/predict_disk.jl \                     # a₀ scan
#       --m-bh1-msun 15 --m-star-msun 30 --m-bh2-msun 20 --L 2 --nx 384 \
#       --scf-ic --scf-axis-ratio 0.72 --scan-a0 1.5,4.0,11

using BinarySupernova
using Printf

# ---------------------------------------------------------------------------
# Argument parsing — same flag names as run_sn50_fiducial.jl.

function arg_float(flag::AbstractString, default::Float64)
    for (i, a) in enumerate(ARGS)
        a == flag && i < length(ARGS) && return parse(Float64, ARGS[i+1])
    end
    return default
end
function arg_int(flag::AbstractString, default::Int)
    for (i, a) in enumerate(ARGS)
        a == flag && i < length(ARGS) && return parse(Int, ARGS[i+1])
    end
    return default
end
function arg_str(flag::AbstractString, default::String)
    for (i, a) in enumerate(ARGS)
        a == flag && i < length(ARGS) && return String(ARGS[i+1])
    end
    return default
end
arg_flag(flag::AbstractString) = flag in ARGS

# Parse a "LO,HI,N" (or "LO,HI") scan spec into (lo, hi, n).
function parse_scan(s::AbstractString)
    p = split(s, ",")
    length(p) >= 2 || error("scan spec must be LO,HI[,N] — got \"$s\"")
    lo = parse(Float64, p[1]);  hi = parse(Float64, p[2])
    n  = length(p) >= 3 ? parse(Int, p[3]) : 11
    return lo, hi, n
end

# ---------------------------------------------------------------------------
# Linear interpolation on an ascending grid (clamped at the ends).

function interp(xq::Float64, xs::Vector{Float64}, ys::Vector{Float64})
    xq <= xs[1]   && return ys[1]
    xq >= xs[end] && return ys[end]
    i = clamp(searchsortedlast(xs, xq), 1, length(xs) - 1)
    t = (xq - xs[i]) / (xs[i+1] - xs[i])
    return (1.0 - t) * ys[i] + t * ys[i+1]
end

# Eggleton (1983) Roche-lobe radius / separation for mass ratio q = M_lobe/M_other.
eggleton(q) = 0.49 * q^(2/3) / (0.6 * q^(2/3) + log1p(q^(1/3)))

# Bisect a monotone-decreasing f for its root in [lo, hi]; NaN if not bracketed.
function bisect(f, lo::Float64, hi::Float64; tol = 1e-4)
    flo = f(lo);  fhi = f(hi)
    sign(flo) == sign(fhi) && return NaN
    for _ in 1:100
        mid = 0.5 * (lo + hi)
        sign(f(mid)) == sign(flo) ? (lo = mid) : (hi = mid)
        hi - lo < tol && break
    end
    return 0.5 * (lo + hi)
end

# ---------------------------------------------------------------------------
# Post-SN orbit:  Blaauw mass loss + natal kick on BH2.
#
# Pre-SN the binary is BH1 + star on a circular orbit of separation a₀ and
# relative speed v_orb = √(G M_pre / a₀) with M_pre = M_BH1 + M_star = 1 (code
# units).  At the explosion (CLAUDE.md §7.2) the star → BH2 (mass M_BH2) +
# ejecta, the ejecta is removed instantaneously (Blaauw), and a natal kick is
# added to BH2.  The post-SN *relative* orbit is governed by M_bin = M_BH1 +
# M_BH2.  Work in run_sn50_fiducial.jl's frame (l.407-410): the BH2−BH1
# separation is along −x, the pre-SN relative velocity along −y, the orbital AM
# along +z.  The kick V_KICK is added to BH2, so it adds straight to the
# relative velocity, and the components keep the run's exact meaning:
#   kx ≡ --v-kick-x   along x  (kx>0 points toward BH1 — radially inward)
#   ky ≡ --v-kick-y   along y  (ky>0 OPPOSES the −y orbital motion — anti-
#                               orbital / retrograde → shrinks the orbit)
#   kz ≡ --v-kick-z   along z  (out of plane)
# All in code units (velocity unit = v_orb).  Because the tool reproduces the
# run frame, run --v-kick-* values can be pasted verbatim and mean the same.
#
# Two-body invariants of the relative orbit (G = 1):
#   ε = ½|v_rel|² − M_bin/|r_rel|         specific energy   → a = −M_bin/(2ε)
#   h = |r_rel × v_rel|                   specific AM       → e = √(1 + 2ε h²/M_bin²)
# Mass loss alone (k = 0) keeps |r_rel| = a₀ as the periastron and pushes
# apastron out; it unbinds the binary when M_ejecta/M_pre > ½ (Blaauw).
function post_sn_orbit(M_BH1, M_BH2, M_pre, kx, ky, kz; a0 = 1.0)
    M_bin = M_BH1 + M_BH2
    v_orb = sqrt(M_pre / a0)
    rx, ry, rz = -a0, 0.0, 0.0                          # BH2 − BH1, along −x
    vx, vy, vz = kx, -v_orb + ky, kz                    # v_rel_pre along −y, + kick
    v2 = vx^2 + vy^2 + vz^2
    ε  = 0.5 * v2 - M_bin / a0                          # specific orbital energy
    hx = ry*vz - rz*vy;  hy = rz*vx - rx*vz;  hz = rx*vy - ry*vx
    h2 = hx^2 + hy^2 + hz^2
    bound  = ε < 0.0
    a_post = bound ? -M_bin / (2ε) : Inf
    e      = sqrt(max(1.0 + 2.0 * ε * h2 / M_bin^2, 0.0))
    r_peri = bound ? a_post * (1.0 - e) : NaN
    r_apo  = bound ? a_post * (1.0 + e) : NaN
    return (; M_bin, v_orb, ε, h2, bound, a_post, e, r_peri, r_apo)
end

# Disc fate at the POST-SN separation a_post, with r_circ held fixed (the kick
# moves the BHs but not the spin-fed mini-disc, whose size is set by the pre-SN
# star).  Roche lobe and cavity edge both scale with the separation a_post.
function disc_fate_post(r_circ, a_post, b)
    RL  = b.RL_bh2     * a_post                         # eggleton fraction × a_post
    tr  = b.trunc_fac  * RL
    cav = b.cavity_fac * a_post                         # tidal cavity edge ~ 2 a_post
    r_circ <  tr  ? "mini-disc around BH2"  :
    r_circ <  RL  ? "mini-disc (stripped)"  :
    r_circ <  cav ? "overflows BH2 lobe (L2 spill)" :
                    "CIRCUMBINARY DISC"
end

# v_orb in km/s (code velocity unit = orbital speed) for sanity vs real kicks.
const _G_CGS   = 6.674e-8
const _MSUN_G  = 1.989e33
const _RSUN_CM = 6.957e10
vorb_kms(M_TOT_MSUN, a0_rsun) =
    sqrt(_G_CGS * M_TOT_MSUN * _MSUN_G / (a0_rsun * _RSUN_CM)) / 1e5

# ---------------------------------------------------------------------------
# Stellar rotation figure — dimensionless, solved ONCE (independent of a₀).
#   --scf-ic : Hachisu SCF; ω_over_brk ≡ Ω_spin/Ω_brk = √(Ω̂²/M̂) is a pure
#              (n, α) property — see rotating_polytrope.jl l.338-339.
#   --no-spin / else : Lane-Emden polytrope; legacy rigid overlay Ω_spin=f·Ω_orb.
# Both return the dimensionless mass profile r/R_star vs M(<r)/M_star.

function rotation_figure(; scf_ic::Bool, α_request::Float64, no_spin::Bool,
                           spin_frac::Float64, n_poly::Float64, quiet::Bool = false)
    if scf_ic
        α_ic = clamp(α_request, 0.55, 1.0)              # run clamps identically
        if !quiet
            print("  SCF solve (n=3, α=", round(α_ic, digits=3), ") ... ")
            flush(stdout)
        end
        sol = BinarySupernova.scf_rotating_polytrope(n_poly, α_ic;
                  Nr = 256, Nμ = 33, lmax = 12,
                  tol = 1e-7, mix = 0.35, maxiter = 3000)
        quiet || println(sol.converged ? "converged" : "NOT converged")
        ω_over_brk = sqrt(max(sol.Ω², 0.0) / sol.M)     # = Ω_spin/Ω_brk, pure (n,α)
        note = ""
        sol.converged || (note = "SCF did not converge — Ω_spin approximate.")
        sol.Ω² > 0 ||
            (note = "SCF Ω² ≤ 0 (α below mass-shedding) — Ω_spin floored to 0.")
        # angle-averaged cumulative mass (mirrors rotating_polytrope.jl l.356-363)
        Δr_s = sol.r[2] - sol.r[1];  Δμ_s = sol.μ[2] - sol.μ[1]
        Menc = zeros(length(sol.r));  acc = 0.0
        for i in eachindex(sol.r)
            ρ̄ = sum(@view sol.ρ[i, :]) * Δμ_s
            acc += 4π * sol.r[i]^2 * ρ̄ * Δr_s
            Menc[i] = acc
        end
        return (; mode = :scf, ω_over_brk,
                  α_actual = sol.axis_ratio_actual, converged = sol.converged,
                  r_grid_frac = collect(sol.r), Menc_frac = Menc ./ Menc[end],
                  label = @sprintf("SCF rotating polytrope   α_req=%.3f  α_actual=%.3f",
                                    α_ic, sol.axis_ratio_actual),
                  note)
    else
        ξs, θs, dθs = BinarySupernova.lane_emden(n_poly)
        ωn = -ξs[end]^2 * dθs[end]                      # M(<ξ)/M = -ξ²θ'/ωn
        r_grid_frac = ξs ./ ξs[end]
        Menc_frac   = clamp.([-ξs[i]^2 * dθs[i] / ωn for i in eachindex(ξs)], 0.0, 1.0)
        if no_spin
            return (; mode = :nospin, r_grid_frac, Menc_frac,
                      label = "Lane-Emden polytrope, no rotation (--no-spin)", note = "")
        else
            return (; mode = :legacy, spin_frac, r_grid_frac, Menc_frac,
                      label = @sprintf("Lane-Emden polytrope + rigid overlay   Ω_spin/Ω_orb=%.2f",
                                        spin_frac),
                      note = "")
        end
    end
end

# ---------------------------------------------------------------------------
# Budget at one orbital separation a₀ (R☉).  Pure arithmetic given `fig`.

function budget(a0_rsun::Float64, fig;
                m_bh1_msun, m_star_msun, m_bh2_msun, R_star_rsun,
                NX, L, rbo_frac, rbi_frac, cavity_fac, trunc_fac)
    M_TOT_MSUN = m_bh1_msun + m_star_msun
    M_BH1  = m_bh1_msun  / M_TOT_MSUN                   # code units (M_tot=a₀=G=1)
    M_STAR = m_star_msun / M_TOT_MSUN
    M_BH2  = m_bh2_msun  / M_TOT_MSUN
    M_EJ   = M_STAR - M_BH2
    M_BIN  = M_BH1 + M_BH2                              # post-SN binary
    R_STAR = R_star_rsun / a0_rsun
    DX     = 2L / NX
    Ω_ORB  = 1.0
    Ω_BRK  = sqrt(M_STAR / R_STAR^3)                    # surface breakup (run §l.337)
    Ω_SPIN = fig.mode === :scf    ? fig.ω_over_brk * Ω_BRK :
             fig.mode === :legacy ? fig.spin_frac  * Ω_ORB : 0.0

    j_spin     = Ω_SPIN * R_STAR^2                      # specific AM about BH2
    r_circ     = j_spin^2 / M_BH2                       # circularisation radius (·a₀)
    j_cbd      = sqrt(M_BIN * cavity_fac)               # CBD inner-edge AM (a = 1)
    j_ratio    = j_cbd > 0 ? j_spin / j_cbd : 0.0
    r_circ_brk = (M_STAR / M_BH2) * R_STAR              # breakup ceiling (any spin)

    RL_star   = eggleton(M_STAR / M_BH1)                # pre-SN: star in (BH1+star)
    RL_bh2    = eggleton(M_BH2  / M_BH1)                # post-SN: BH2 in (BH1+BH2)
    trunc_bh2 = trunc_fac * RL_bh2
    fill_star = R_STAR / RL_star

    r_core     = interp(m_bh2_msun / m_star_msun, fig.Menc_frac, fig.r_grid_frac) * R_STAR
    r_bomb_out = rbo_frac * R_STAR
    r_bomb_in  = rbi_frac >= 0 ? rbi_frac * R_STAR : r_core
    Mlt(r)     = interp(r / R_STAR, fig.r_grid_frac, fig.Menc_frac) * M_STAR
    M_bomb     = max(Mlt(r_bomb_out) - Mlt(r_bomb_in), 0.0)
    M_cold     = max(M_STAR - Mlt(r_bomb_out), 0.0)

    return (; m_bh1_msun, m_star_msun, m_bh2_msun, M_TOT_MSUN,
              a0_rsun, R_star_rsun, M_BH1, M_STAR, M_BH2, M_BIN, M_EJ,
              R_STAR, NX, L, DX, Ω_ORB, Ω_SPIN, Ω_BRK,
              RL_star, fill_star, RL_bh2, trunc_bh2,
              j_spin, j_cbd, j_ratio, r_circ, r_circ_brk,
              r_core, r_bomb_in, r_bomb_out, M_bomb, M_cold,
              cavity_fac, trunc_fac)
end

# Physics-only disc classification (no grid resolution).
classify(rc, b) =
    rc < b.trunc_bh2     ? "mini-disc around BH2" :
    rc < b.RL_bh2        ? "mini-disc (tidally stripped)" :
    rc < b.cavity_fac    ? "overflows BH2 lobe (L2 spill)" :
                           "CIRCUMBINARY DISC"

# Scan-row fate, including the pre-SN star-overflow check.
function scan_fate(b)
    b.fill_star >= 1.0      && return "STAR overflows lobe -- use relax IC"
    b.r_circ < b.trunc_bh2  && return "mini-disc (clean)"
    b.r_circ < b.RL_bh2     && return "mini-disc (stripped)"
    b.r_circ < b.cavity_fac && return "OVERFLOWS BH2 lobe -> L1/L2 spill"
    return "CIRCUMBINARY DISC"
end

# ---------------------------------------------------------------------------
# Single-config report.

function print_report(b, fig)
    println()
    println("="^72)
    println("  SUPERNOVA-FALLBACK DISC BUDGET            predict_disk.jl")
    println("="^72)
    println("  IC: ", fig.label)
    isempty(fig.note) || println("  WARN: ", fig.note)
    println()

    @printf("PROGENITOR / ORBIT  [M_sun, R_sun]\n")
    @printf("  M_BH1 %5.1f   M_star %5.1f   M_BH2,init %5.1f   M_ejecta %5.1f\n",
            b.m_bh1_msun, b.m_star_msun, b.m_bh2_msun, b.m_star_msun - b.m_bh2_msun)
    @printf("  a0 = %.2f R_sun      R_star = %.2f R_sun\n", b.a0_rsun, b.R_star_rsun)
    println()

    @printf("CODE UNITS  [M_tot = a0 = G = 1]\n")
    @printf("  M_BH1 %.4f   M_STAR %.4f   M_BH2 %.4f   M_bin %.4f\n",
            b.M_BH1, b.M_STAR, b.M_BH2, b.M_BIN)
    @printf("  R_STAR %.5f   grid: NX %d  L %.2f  dx %.5f\n", b.R_STAR, b.NX, b.L, b.DX)
    @printf("  R_star/dx = %.2f   %s\n", b.R_STAR / b.DX,
            b.R_STAR / b.DX < 2 ? "** STAR UNDER-RESOLVED (run warns < 2; want >= 4)" : "ok")
    println()

    @printf("SPIN\n")
    @printf("  Omega_orb %.3f   Omega_spin %.4f   ratio Omega_s/Omega_o %.3f   Omega_s/Omega_brk %.3f\n",
            b.Ω_ORB, b.Ω_SPIN, b.Ω_SPIN / b.Ω_ORB, b.Ω_SPIN / b.Ω_BRK)
    println()

    @printf("ROCHE GEOMETRY  [units of a0]\n")
    @printf("  star pre-SN : R_L %.3f   R_star/R_L %.3f   -> %s\n",
            b.RL_star, b.fill_star,
            b.fill_star < 0.90 ? "detached" :
            b.fill_star < 1.05 ? "Roche-filling" : "OVERFLOWING")
    @printf("  BH2 post-SN : R_L %.3f   mini-disc truncation ~ %.3f\n",
            b.RL_bh2, b.trunc_bh2)
    println()

    @printf("ANGULAR-MOMENTUM BUDGET  [outer-envelope material, varpi -> R_star]\n")
    @printf("  j_spin    (about BH2)     = %.4e\n", b.j_spin)
    @printf("  j_CBD,min (about COM)     = %.4e   [circular orbit at r_cav = %.1f a0]\n",
            b.j_cbd, b.cavity_fac)
    @printf("  j_spin / j_CBD,min        = %.2f %%\n", 100 * b.j_ratio)
    @printf("  r_circ around BH2         = %.4e a0  = %.4e R_sun  = %.3g dx\n",
            b.r_circ, b.r_circ * b.a0_rsun, b.r_circ / b.DX)
    @printf("  r_circ at breakup spin    = %.4f a0    (hard ceiling, ANY rotation)\n",
            b.r_circ_brk)
    println()

    @printf("ENVELOPE PARTITION  [M_sun; %s mass profile]\n",
            fig.mode === :scf ? "SCF rotating" : "n=3 Lane-Emden")
    @printf("  r_core = %.4f a0  (%.3f R_sun = %.2f R_star)\n",
            b.r_core, b.r_core * b.a0_rsun, b.r_core / b.R_STAR)
    @printf("  envelope (r_core -> R_star)        %6.2f M_sun\n", b.M_EJ * b.M_TOT_MSUN)
    @printf("  bombed shell [%.2f -> %.2f R_star]  %6.2f M_sun\n",
            b.r_bomb_in / b.R_STAR, b.r_bomb_out / b.R_STAR, b.M_bomb * b.M_TOT_MSUN)
    @printf("  unshocked outer shell              %6.2f M_sun   <- cold disc seed\n",
            b.M_cold * b.M_TOT_MSUN)
    @printf("  (bound fallback fraction needs the energy budget — not computed here)\n")
    println()

    # spin sweep: r_circ vs rotation rate, capped at breakup
    op = b.Ω_SPIN / b.Ω_ORB
    ratios = Float64[1.0, 1.5, 2.0, 3.0, b.Ω_BRK / b.Ω_ORB]
    op > 0 && !any(r -> isapprox(r, op; rtol=1e-6), ratios) && push!(ratios, op)
    filter!(x -> 0 < x <= b.Ω_BRK / b.Ω_ORB * (1 + 1e-9), ratios)
    sort!(unique!(ratios))
    @printf("SPIN SWEEP  r_circ(BH2) vs rotation rate   [Omega_brk/Omega_orb = %.2f]\n",
            b.Ω_BRK / b.Ω_ORB)
    @printf("  %14s   %13s   %s\n", "Om_spin/Om_orb", "r_circ/a0", "fate")
    for f in ratios
        rc   = f^2 * b.R_STAR^4 / b.M_BH2
        tags = String[]
        isapprox(f, b.Ω_BRK / b.Ω_ORB; rtol=1e-6) && push!(tags, "breakup")
        op > 0 && isapprox(f, op; rtol=1e-3)       && push!(tags, "<- this run")
        @printf("  %14.2f   %13.4e   %-30s %s\n", f, rc, classify(rc, b), join(tags, ", "))
    end
    println()

    # verdict
    println("-"^72)
    phys = classify(b.r_circ, b)
    if b.Ω_SPIN == 0.0
        println("  VERDICT:  RADIAL ACCRETION ONTO BH2  (no disc)")
        println()
        println("  Omega_spin = 0: fallback carries no angular momentum and accretes")
        println("  straight onto BH2 along the radial infall. No disc of any kind.")
    elseif phys == "CIRCUMBINARY DISC"
        println("  VERDICT:  CIRCUMBINARY DISC")
        println()
        @printf("  r_circ = %.3f a0 >= %.1f a0 (cavity edge): outer-envelope fallback\n",
                b.r_circ, b.cavity_fac)
        println("  has enough AM to settle into a rotationally-supported circumbinary")
        println("  disc. Check r_circ/dx >= 4 for the run to resolve it.")
    else
        res = b.r_circ >= 4*b.DX ? "RESOLVED" :
              b.r_circ >= 2*b.DX ? "marginally resolved" : "SUB-GRID"
        println("  VERDICT:  ", uppercase(phys), "   [", res, " at NX=", b.NX, "]")
        println()
        @printf("  Outer-envelope fallback circularises at r_circ = %.3e a0, set by\n",
                b.r_circ)
        @printf("  stellar spin alone — %.1f%% of the circumbinary AM threshold.\n",
                100 * b.j_ratio)
        if phys == "overflows BH2 lobe (L2 spill)"
            @printf("  r_circ = %.2f a0 exceeds R_L,BH2 = %.2f a0: the fallback overflows\n",
                    b.r_circ, b.RL_bh2)
            println("  BH2's Roche lobe, fed through L1 (-> BH1) and L2 (-> circumbinary).")
            println("  This L2-spill is the CBD seed; a settled CBD (r_circ >= 2 a0) still")
            println("  needs binary-torque pumping over many orbits.")
        else
            println("  It forms a disc bound to BH2, not around the binary.")
        end
        if res == "SUB-GRID"
            @printf("  At NX=%d the disc spans only %.3g dx: the run CANNOT resolve it;\n",
                    b.NX, b.r_circ / b.DX)
            println("  accretion will be set by the sink prescription and will not")
            println("  converge in NX (cf. the 88%->46% Delta-M_BH2 non-convergence).")
            println("  Fix: raise NX until r_circ >= 4 dx, or tighten a0.")
        end
        a0_ceiling = (b.m_star_msun / b.m_bh2_msun) * b.R_star_rsun / b.cavity_fac
        println()
        if a0_ceiling < b.R_star_rsun
            @printf("  Even at breakup rotation r_circ reaches the cavity edge only for\n")
            @printf("  a0 <= %.2f R_sun < R_star = %.2f R_sun — the star larger than its\n",
                    a0_ceiling, b.R_star_rsun)
            println("  orbit (a contact/merger configuration, not a binary). Stellar")
            println("  spin provably cannot make a CBD.")
        else
            @printf("  Reaching the cavity edge at breakup needs a0 <= %.2f R_sun.\n",
                    a0_ceiling)
        end
        println()
        println("  To form a genuine CBD the fallback needs ORBITAL-scale AM:")
        println("   (1) Roche-OVERFLOW progenitor — outer envelope already co-orbits;")
        println("   (2) pre-existing circumbinary gas (post-common-envelope IC);")
        println("   (3) integrate >> tens of P0 so binary torques pump the mini-disc")
        println("       outward — and bookkeep the binary hardening that pays for it.")
    end
    println("-"^72)
    @printf("  [assumptions: r_cav = %.1f a0 (Artymowicz-Lubow tidal cavity); mini-disc\n",
            b.cavity_fac)
    @printf("   truncation = %.2f R_L,BH2; n=3 polytrope. Override: --cavity-factor,\n",
            b.trunc_fac)
    println("   --trunc-factor.]")
    println()
    return nothing
end

# ---------------------------------------------------------------------------
# Scan over a₀ (fixed figure) — find where r_circ crosses R_L,BH2.

function scan_a0(fig, lo, hi, n; bp...)
    println()
    println("="^72)
    println("  DISC-FATE SCAN over a0            predict_disk.jl --scan-a0")
    println("="^72)
    println("  IC: ", fig.label)
    isempty(fig.note) || println("  WARN: ", fig.note)
    bp_nt = (; bp...)
    @printf("  fixed: M_BH1=%.0f M_star=%.0f M_BH2=%.0f M_sun   NX=%d  L=%.2f\n",
            bp_nt.m_bh1_msun, bp_nt.m_star_msun, bp_nt.m_bh2_msun, bp_nt.NX, bp_nt.L)
    fig.mode === :scf &&
        @printf("  fixed: Omega_spin/Omega_brk = %.3f  (set by alpha; a0-independent)\n",
                fig.ω_over_brk)
    println()
    @printf("  %8s  %10s  %10s  %11s  %14s   %s\n",
            "a0[Rsun]", "R*/R_L,*", "Om_s/Om_o", "r_circ/a0", "r_circ/R_L,BH2", "fate")
    for k in 0:n-1
        a0 = lo + (hi - lo) * k / max(n - 1, 1)
        b  = budget(a0, fig; bp...)
        @printf("  %8.2f  %10.3f  %10.3f  %11.4e  %14.3f   %s\n",
                a0, b.fill_star, b.Ω_SPIN / b.Ω_ORB, b.r_circ,
                b.r_circ / b.RL_bh2, scan_fate(b))
    end
    println()

    # crossings (a₀ is monotone: r_circ ∝ 1/a₀ for SCF, ∝ 1/a₀⁴ for legacy)
    b_ref   = budget(0.5 * (lo + hi), fig; bp...)
    a0_RL   = bisect(a0 -> (bb = budget(a0, fig; bp...); bb.r_circ - bb.RL_bh2), 0.3, 200.0)
    a0_cbd  = bisect(a0 -> (bb = budget(a0, fig; bp...); bb.r_circ - bb.cavity_fac), 0.05, 200.0)
    a0_fill = bp_nt.R_star_rsun / b_ref.RL_star      # R_star = R_L,star

    println("-"^72)
    println("  THRESHOLDS  [a0 in R_sun]")
    if isnan(a0_RL)
        println("  r_circ = R_L,BH2 : not crossed within a0 in [0.3, 200]")
    else
        @printf("  r_circ = R_L,BH2  : a0 = %.2f  — below this the mini-disc overflows\n", a0_RL)
        println("                      BH2's lobe and spills through L1 (-> BH1) and L2.")
    end
    @printf("  star Roche-fills  : a0 = %.2f  — below this the SCF single-star figure\n", a0_fill)
    println("                      is invalid; use the Roche-relaxation IC (relax_ic.jl).")
    if !isnan(a0_RL) && a0_fill < a0_RL
        @printf("  => SCF-valid L2-spill window:  %.2f < a0 < %.2f R_sun\n", a0_fill, a0_RL)
    end
    if isnan(a0_cbd)
        println("  settled CBD (r_circ = r_cav) : not reached for a0 in [0.05, 200]")
    elseif a0_cbd < bp_nt.R_star_rsun
        @printf("  settled CBD (r_circ = r_cav) : would need a0 = %.2f R_sun < R_star = %.2f\n",
                a0_cbd, bp_nt.R_star_rsun)
        println("                      — impossible (star larger than orbit). Spin cannot")
        println("                      make a settled CBD; the L2 spill must seed it.")
    else
        @printf("  settled CBD (r_circ = r_cav) : a0 = %.2f R_sun\n", a0_cbd)
    end
    println("-"^72)
    println()
    return nothing
end

# ---------------------------------------------------------------------------
# Scan over SCF axis ratio α (fixed a₀) — one SCF solve per α.

function scan_alpha(α_lo, α_hi, n, n_poly; a0_rsun, bp...)
    println()
    println("="^72)
    println("  DISC-FATE SCAN over SCF alpha     predict_disk.jl --scan-alpha")
    println("="^72)
    bp_nt = (; bp...)
    @printf("  fixed: M_BH1=%.0f M_star=%.0f M_BH2=%.0f M_sun   a0=%.2f R_sun  NX=%d\n",
            bp_nt.m_bh1_msun, bp_nt.m_star_msun, bp_nt.m_bh2_msun, a0_rsun, bp_nt.NX)
    println("  (alpha decreasing -> more rotation; n=3 mass-shedding near alpha~0.66)")
    println()
    @printf("  %8s  %10s  %12s  %10s  %14s   %s\n",
            "alpha", "alpha_act", "Om_s/Om_brk", "Om_s/Om_o", "r_circ/R_L,BH2", "fate")
    for k in 0:n-1
        α  = α_hi - (α_hi - α_lo) * k / max(n - 1, 1)   # high -> low (more rotation)
        fig = rotation_figure(; scf_ic = true, α_request = α, no_spin = false,
                                spin_frac = 1.0, n_poly = n_poly, quiet = true)
        b  = budget(a0_rsun, fig; bp...)
        @printf("  %8.3f  %10.3f  %12.3f  %10.3f  %14.3f   %s\n",
                α, fig.α_actual, fig.ω_over_brk, b.Ω_SPIN / b.Ω_ORB,
                b.r_circ / b.RL_bh2, scan_fate(b))
    end
    println()
    println("-"^72)
    println("  Note: Ω_spin/Ω_brk is the figure's rotation parameter; for n=3 it tops")
    println("  out near the mass-shedding peak (alpha~0.66). run_sn50_fiducial.jl warns")
    println("  above Ω_spin/Ω_brk = 0.9. a0 is the stronger lever — see --scan-a0.")
    println("-"^72)
    println()
    return nothing
end

# ---------------------------------------------------------------------------
# Kick fate: classify a post-SN orbit (disruption / merger / disc class).

function kick_classify(o, r_circ, b, coll)
    o.bound          || return "DISRUPTED (unbound)"
    o.r_peri < coll  && return "BH-BH MERGER (plunge)"
    r_circ <= 0.0    && return "RADIAL ACCRETION (no disc)"
    return uppercase(disc_fate_post(r_circ, o.a_post, b))
end

# ---------------------------------------------------------------------------
# Single-kick report: post-SN orbit + whether it clears the CBD bar.

function kick_report(b, fig, kx, ky, kz, coll)
    M_pre = b.M_BH1 + b.M_STAR
    o     = post_sn_orbit(b.M_BH1, b.M_BH2, M_pre, kx, ky, kz)
    vk    = vorb_kms(b.M_TOT_MSUN, b.a0_rsun)
    kmag  = sqrt(kx^2 + ky^2 + kz^2)
    f_ej  = b.M_EJ / M_pre

    println()
    println("="^72)
    println("  POST-SN KICK ORBIT + DISC FATE     predict_disk.jl --v-kick-*")
    println("="^72)
    println("  IC: ", fig.label)
    isempty(fig.note) || println("  WARN: ", fig.note)
    println()
    @printf("KICK  [run frame; code velocity unit = v_orb = %.0f km/s]\n", vk)
    @printf("  kx --v-kick-x = %+.3f  (%+.0f km/s)   radial%s\n", kx, kx*vk,
            kx > 0 ? " (toward BH1)" : kx < 0 ? " (away from BH1)" : "")
    @printf("  ky --v-kick-y = %+.3f  (%+.0f km/s)   %s\n", ky, ky*vk,
            ky > 0 ? "anti-orbital -> SHRINKS" : ky < 0 ? "prograde -> widens" : "tangential")
    @printf("  kz --v-kick-z = %+.3f  (%+.0f km/s)   out-of-plane\n", kz, kz*vk)
    @printf("  |v_kick| = %.3f  (%.0f km/s)\n", kmag, kmag*vk)
    println()

    @printf("BLAAUW MASS LOSS\n")
    @printf("  M_ejecta/M_pre = %.3f   %s\n", f_ej,
            f_ej > 0.5 ? "** > 0.5: UNBINDS on mass loss alone" : "<= 0.5 (mass loss alone keeps it bound)")
    println()

    @printf("POST-SN ORBIT  [a0 = pre-SN separation = 1]\n")
    if !o.bound
        @printf("  UNBOUND: specific energy = %+.4f > 0  -> binary disrupted\n", o.ε)
    else
        @printf("  a_post = %.4f a0 = %.4f R_sun   (%s vs a0)\n",
                o.a_post, o.a_post * b.a0_rsun, o.a_post < 1 ? "SHRUNK" : "widened")
        @printf("  e_post = %.4f\n", o.e)
        @printf("  r_peri = %.4f a0 = %.4f R_sun = %.2f R_star   %s\n",
                o.r_peri, o.r_peri * b.a0_rsun, o.r_peri / b.R_STAR,
                o.r_peri < coll ? "** < collision radius" : "")
        @printf("  r_apo  = %.4f a0 = %.4f R_sun\n", o.r_apo, o.r_apo * b.a0_rsun)
    end
    println()

    @printf("CBD BAR  [r_circ fixed by the pre-SN star; cavity edge = %.1f a_post]\n", b.cavity_fac)
    @printf("  r_circ              = %.4f a0 = %.4f R_sun\n", b.r_circ, b.r_circ * b.a0_rsun)
    if o.bound
        cav = b.cavity_fac * o.a_post
        @printf("  cavity edge (2a)    = %.4f a0 = %.4f R_sun\n", cav, cav * b.a0_rsun)
        @printf("  r_circ / cavity     = %.3f   %s\n", b.r_circ / cav,
                b.r_circ >= cav ? ">= 1: CLEARS the bar (CBD)" : "< 1: below the bar (no settled CBD)")
        @printf("  CBD needs a_post <= r_circ/%.1f = %.4f a0\n", b.cavity_fac, b.r_circ / b.cavity_fac)
    end
    println()

    println("-"^72)
    println("  VERDICT:  ", kick_classify(o, b.r_circ, b, coll))
    println("-"^72)
    println()
    return o
end

# ---------------------------------------------------------------------------
# Scan the anti-orbital (tangential) kick — the orbit-shrinking lever.

function scan_kick(b, fig, lo, hi, n, kx, kz, coll)
    M_pre = b.M_BH1 + b.M_STAR
    vk    = vorb_kms(b.M_TOT_MSUN, b.a0_rsun)
    a_cbd = b.r_circ / b.cavity_fac                     # a_post below which a CBD clears
    println()
    println("="^72)
    println("  POST-SN KICK SCAN (tangential)    predict_disk.jl --scan-kick")
    println("="^72)
    println("  IC: ", fig.label)
    isempty(fig.note) || println("  WARN: ", fig.note)
    @printf("  fixed: kx=%.2f  kz=%.2f   v_orb=%.0f km/s   M_ej/M_pre=%.3f\n",
            kx, kz, vk, b.M_EJ / M_pre)
    @printf("  r_circ=%.4f a0 (fixed);  settled CBD needs a_post <= %.4f a0;  coll r_peri<%.3f a0\n",
            b.r_circ, a_cbd, coll)
    println("  (ky>0 = anti-orbital/retrograde = orbit-shrinking; ky<0 = prograde)")
    println()
    @printf("  %8s %9s  %6s  %9s  %7s  %9s   %s\n",
            "ky", "ky[km/s]", "bound", "a_post", "e", "r_peri", "fate")
    a_min = Inf;  ky_min = NaN
    for i in 0:n-1
        ky = lo + (hi - lo) * i / max(n - 1, 1)
        o  = post_sn_orbit(b.M_BH1, b.M_BH2, M_pre, kx, ky, kz)
        if o.bound && o.r_peri >= coll && o.a_post < a_min
            a_min = o.a_post;  ky_min = ky
        end
        @printf("  %8.3f %9.0f  %6s  %9.4f  %7.4f  %9.4f   %s\n",
                ky, ky * vk, o.bound ? "yes" : "NO",
                o.a_post, o.e, o.r_peri, kick_classify(o, b.r_circ, b, coll))
    end
    println()
    println("-"^72)
    if isfinite(a_min)
        @printf("  tightest bound, non-merging orbit: a_post = %.4f a0  (ky = %.3f)\n", a_min, ky_min)
        cav = b.cavity_fac * a_min
        if b.r_circ >= cav
            @printf("  -> cavity edge %.4f a0 <= r_circ %.4f a0: a CBD CLEARS here.\n", cav, b.r_circ)
        else
            @printf("  -> cavity edge %.4f a0 still > r_circ %.4f a0: NO settled CBD.\n", cav, b.r_circ)
            @printf("     would need a_post <= %.4f a0; tighter only via r_peri -> 0\n", a_cbd)
            println("     (a near-radial plunge = BH-BH merger, not a tightened binary).")
        end
    else
        println("  -> no bound, non-merging orbit in the scanned ky range.")
    end
    @printf("  [floor: an instantaneous kick at a0 can shrink a_post to at most a0/2,\n")
    println("   and only at r_peri -> 0 (collision).  Mass loss pushes the other way.]")
    println("-"^72)
    println()
    return nothing
end

# ---------------------------------------------------------------------------

function main()
    m_bh1_msun  = arg_float("--m-bh1-msun",  50.0)
    m_star_msun = arg_float("--m-star-msun", 55.0)
    m_bh2_msun  = arg_float("--m-bh2-msun",  45.0)
    a0_rsun     = arg_float("--a0-rsun",     20.0)
    NX          = arg_int(  "--nx",          128)
    L           = arg_float("--L",           4.0)
    scf_ic      = arg_flag( "--scf-ic")
    α_request   = arg_float("--scf-axis-ratio", 1.0)
    spin_frac   = arg_float("--spin-omega-frac", 1.0)
    no_spin     = arg_flag( "--no-spin")
    rbo_frac    = arg_float("--r-bomb-outer-frac", 1.0)
    rbi_frac    = arg_float("--r-bomb-inner-frac", -1.0)
    cavity_fac  = arg_float("--cavity-factor", 2.0)     # CBD inner edge / a₀
    trunc_fac   = arg_float("--trunc-factor",  0.4)     # mini-disc edge / R_L,BH2
    scan_a0_s   = arg_str(  "--scan-a0",    "")
    scan_α_s    = arg_str(  "--scan-alpha", "")
    kx          = arg_float("--v-kick-x", 0.0)          # radial (run frame)
    ky          = arg_float("--v-kick-y", 0.0)          # tangential; >0 = anti-orbital
    kz          = arg_float("--v-kick-z", 0.0)          # out-of-plane
    scan_kick_s = arg_str(  "--scan-kick", "")          # sweep ky: LO,HI[,N]
    coll_arg    = arg_float("--coll-radius", -1.0)      # code units; <0 -> default R_STAR

    R_star_rsun = 1.0                                   # hardcoded in run_sn50_fiducial.jl
    n_poly      = 1.0 / (4.0/3.0 - 1.0)                 # γ = 4/3 -> n = 3

    m_bh2_msun < m_star_msun ||
        error("M_BH2_init ($m_bh2_msun) must be < M_star ($m_star_msun)")

    bp = (; m_bh1_msun, m_star_msun, m_bh2_msun, R_star_rsun,
            NX, L, rbo_frac, rbi_frac, cavity_fac, trunc_fac)

    if !isempty(scan_α_s)
        lo, hi, n = parse_scan(scan_α_s)
        scan_alpha(lo, hi, n, n_poly; a0_rsun = a0_rsun, bp...)
    elseif !isempty(scan_a0_s)
        lo, hi, n = parse_scan(scan_a0_s)
        fig = rotation_figure(; scf_ic, α_request, no_spin, spin_frac, n_poly)
        scan_a0(fig, lo, hi, n; bp...)
    elseif !isempty(scan_kick_s)
        lo, hi, n = parse_scan(scan_kick_s)
        fig  = rotation_figure(; scf_ic, α_request, no_spin, spin_frac, n_poly)
        b    = budget(a0_rsun, fig; bp...)
        coll = coll_arg >= 0 ? coll_arg : b.R_STAR
        scan_kick(b, fig, lo, hi, n, kx, kz, coll)
    else
        fig = rotation_figure(; scf_ic, α_request, no_spin, spin_frac, n_poly)
        b   = budget(a0_rsun, fig; bp...)
        print_report(b, fig)
        if kx != 0.0 || ky != 0.0 || kz != 0.0           # kick given: add orbit report
            coll = coll_arg >= 0 ? coll_arg : b.R_STAR
            kick_report(b, fig, kx, ky, kz, coll)
        end
    end
    return nothing
end

main()
