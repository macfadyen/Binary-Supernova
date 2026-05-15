# Hachisu (1986a, ApJS 61, 479) self-consistent field method for rigidly
# rotating axisymmetric polytropes.
#
# Solves hydrostatic + centrifugal equilibrium
#     H + Φ_grav − ½ Ω² ϖ² = C        (constant inside the star)
#     P = K ρ^(1+1/n),   H = (n+1) K ρ^(1/n)
# for a polytrope of index `n` with prescribed axis ratio α = r_p / r_eq.
#
# The axisymmetric Poisson equation is solved on a 2D spherical (r, μ=cosθ)
# grid via the Green's-function multipole expansion.  Density is assumed
# symmetric about the equator (μ → −μ), so only even l contribute.
#
# Normalisation: ρ_c = 1 and r_eq = 1 in the returned solution; G = 1 in the
# potential (the code uses the convention Φ = −2π Σ_l P_l(μ) u_l(r), which
# corresponds to G = 1, so Ω² has units of Gρ_c and M is in units of ρ_c r_eq³).
#
# Axis-ratio sequence:
#     α = 1         → non-rotating Lane-Emden (Ω² = 0)
#     α ↘ α_ms      → rotating, ever-more-oblate; Ω² rises
#     α ≤ α_ms      → mass-shedding limit; below it no equilibrium exists
# For n = 3 (radiation-dominated), α_ms ≈ 0.58.

# ---------------------------------------------------------------------------
# Legendre polynomials via Bonnet's recurrence.

function _Plegendre(l::Int, μ::Float64)
    l == 0 && return 1.0
    l == 1 && return μ
    p0 = 1.0; p1 = μ
    for k in 2:l
        pk = ((2k - 1) * μ * p1 - (k - 1) * p0) / k
        p0 = p1; p1 = pk
    end
    return p1
end

# ---------------------------------------------------------------------------
# SCF solver

"""
    scf_rotating_polytrope(n, axis_ratio; Nr=128, Nμ=33, lmax=12,
                           mix=0.5, tol=1e-7, maxiter=2000, verbose=false)

Solve Hachisu's SCF for a rigidly rotating polytrope with index `n` and
prescribed pole-to-equator axis ratio `axis_ratio ∈ (0, 1]`.

Returns a NamedTuple:
- `r`, `μ`           : spherical grid (μ ∈ [0, 1], exploiting equatorial symmetry)
- `ρ`, `Φ`           : density and gravitational potential on the (r, μ) grid
- `Ω²`               : Ω² in units of G ρ_c
- `M`, `J`, `T`, `W` : mass, angular momentum, rotational KE, grav binding energy
                       (with ρ_c = r_eq = G = 1)
- `T_over_W`         : T/|W| rotation parameter
- `iters`, `converged`
- `n`, `axis_ratio_requested`, `axis_ratio_actual`
- `r_eq`, `r_p`      : equatorial / polar radii (grid-aligned)

The SCF is considered converged when max |Δρ| per iteration drops below `tol`.
A negative Ω² signals that the requested axis ratio is below the mass-shedding
limit for this `n`; in that case `converged = false` and Ω² is returned as-is.
"""
function scf_rotating_polytrope(n::Real, axis_ratio::Real;
                                 Nr::Int = 128,
                                 Nμ::Int = 33,
                                 lmax::Int = 12,
                                 mix::Float64 = 0.5,
                                 tol::Float64 = 1e-7,
                                 maxiter::Int = 2000,
                                 verbose::Bool = false)

    @assert 0 < axis_ratio <= 1       "axis_ratio must be in (0, 1]"
    @assert iseven(lmax) && lmax >= 0 "lmax must be a non-negative even integer"
    @assert Nr > 10 && Nμ > 4
    nf = Float64(n)

    # --- radial grid: r_eq lands on index i_eq with r[i_eq] = 1
    i_eq = round(Int, 0.8 * Nr)
    Δr   = 1.0 / (i_eq - 0.5)
    r    = [(i - 0.5) * Δr for i in 1:Nr]
    r_eq = r[i_eq]

    i_p = clamp(round(Int, axis_ratio * (i_eq - 0.5) + 0.5), 1, i_eq)
    r_p = r[i_p]
    α_actual = r_p / r_eq

    # --- μ grid: cell centres on [0, 1]; density is symmetric about μ = 0
    Δμ   = 1.0 / Nμ
    μ    = [(j - 0.5) * Δμ for j in 1:Nμ]

    # --- Legendre tables for even l = 0, 2, …, lmax
    ls   = collect(0:2:lmax)
    nl   = length(ls)
    Pμ   = zeros(nl, Nμ);  Pl0 = zeros(nl);  Pl1 = zeros(nl)
    for (k, l) in enumerate(ls)
        Pl0[k] = _Plegendre(l, 0.0)
        Pl1[k] = _Plegendre(l, 1.0)
        for j in 1:Nμ
            Pμ[k, j] = _Plegendre(l, μ[j])
        end
    end

    # --- initial guess: spherical (1 − r²)^n inside r ≤ 1, zero outside;
    #     ρ_c = 1 by construction
    ρ = zeros(Nr, Nμ)
    for i in 1:Nr, j in 1:Nμ
        ρ[i, j] = r[i] < 1.0 ? (1.0 - r[i]^2)^nf : 0.0
    end

    # Scratch arrays
    ρ̃_l  = zeros(nl, Nr)   # multipole moment ρ̃_l(r) = ∫_{-1}^{1} ρ P_l dμ
    u_l  = zeros(nl, Nr)   # radial Green's function coefficient
    Φ    = zeros(Nr, Nμ)
    ρ_new = similar(ρ)

    converged = false
    iters_used = maxiter
    local Ω² = 0.0
    local H_c = 0.0

    for iter in 1:maxiter
        # Multipole moments: ρ̃_l(r) = 2 ∫_0^1 ρ(r, μ) P_l(μ) dμ (factor 2 from
        # folding over μ ∈ [−1, 0] via equatorial symmetry; odd l vanish).
        @inbounds for k in 1:nl, i in 1:Nr
            s = 0.0
            for j in 1:Nμ
                s += ρ[i, j] * Pμ[k, j]
            end
            ρ̃_l[k, i] = 2.0 * s * Δμ
        end

        # Radial Green's-function integrals via midpoint rule:
        #   u_l(r_i) = r_i^{-l-1} ∫_0^{r_i} ρ̃_l(r') r'^{l+2} dr'
        #            + r_i^{l}    ∫_{r_i}^{r_max} ρ̃_l(r') r'^{1-l} dr'
        @inbounds for k in 1:nl
            l = ls[k]
            I1 = 0.0
            for i in 1:Nr
                I1 += ρ̃_l[k, i] * r[i]^(l + 2) * Δr
                u_l[k, i] = r[i]^(-l - 1) * I1
            end
            I2 = 0.0
            for i in Nr:-1:1
                I2 += ρ̃_l[k, i] * r[i]^(1 - l) * Δr
                u_l[k, i] += r[i]^l * I2
            end
        end

        # Φ(r, μ) = −2π Σ_l u_l(r) P_l(μ)       (G = 1 here)
        @inbounds for i in 1:Nr, j in 1:Nμ
            s = 0.0
            for k in 1:nl
                s += u_l[k, i] * Pμ[k, j]
            end
            Φ[i, j] = -2π * s
        end

        # Surface reference potentials via analytic P_l(0), P_l(1).
        Φ_A = 0.0; Φ_B = 0.0
        @inbounds for k in 1:nl
            Φ_A += u_l[k, i_eq] * Pl0[k]
            Φ_B += u_l[k, i_p ] * Pl1[k]
        end
        Φ_A *= -2π; Φ_B *= -2π

        # Centre (r → 0): only the monopole survives; u_l(0) = 0 for l ≥ 2 and
        # u_0(0) = ∫_0^∞ r' ρ̃_0(r') dr'.
        I2_mono = 0.0
        @inbounds for i in 1:Nr
            I2_mono += ρ̃_l[1, i] * r[i] * Δr
        end
        Φ_c = -2π * I2_mono

        # Solve for Ω² from the two surface constraints:
        #   Φ_A − ½ Ω² r_eq² = Φ_B  ⇒  Ω² = 2(Φ_A − Φ_B) / r_eq²
        Ω² = 2.0 * (Φ_A - Φ_B) / r_eq^2
        C  = Φ_B
        H_c = C - Φ_c   # central enthalpy; polytrope gives ρ = (H/H_c)^n

        # Update density: ρ = (H / H_c)^n where H = C + ½Ω²ϖ² − Φ; clip at 0.
        @inbounds for i in 1:Nr, j in 1:Nμ
            ϖ² = r[i]^2 * (1.0 - μ[j]^2)
            H  = C + 0.5 * Ω² * ϖ² - Φ[i, j]
            ρ_new[i, j] = (H > 0.0 && H_c > 0.0) ? (H / H_c)^nf : 0.0
        end

        # Under-relaxed update + residual (max absolute density change).
        res = 0.0
        @inbounds for i in 1:Nr, j in 1:Nμ
            Δρ = ρ_new[i, j] - ρ[i, j]
            if abs(Δρ) > res
                res = abs(Δρ)
            end
            ρ[i, j] = mix * ρ_new[i, j] + (1.0 - mix) * ρ[i, j]
        end

        if verbose && (iter % 50 == 0 || iter == 1)
            @info "SCF iter" iter Ω²=round(Ω², digits=6) res=round(res, sigdigits=4) H_c=round(H_c, digits=4)
        end

        if res < tol
            converged  = true
            iters_used = iter
            break
        end
    end

    # ------------------------------------------------------------------------
    # Integrated diagnostics.  With μ ∈ [0, 1] and equatorial symmetry,
    #   dV = 4π r² dr dμ  (= 2π dφ · 2 dμ · r² dr, axisymmetric, μ-symmetric).
    M = 0.0;  Jint = 0.0;  Tkin = 0.0;  Wpot = 0.0
    Ω = sqrt(max(Ω², 0.0))
    @inbounds for i in 1:Nr, j in 1:Nμ
        ρij = ρ[i, j]
        ρij == 0.0 && continue
        dV   = 4π * r[i]^2 * Δr * Δμ
        ϖ²   = r[i]^2 * (1.0 - μ[j]^2)
        M    += ρij * dV
        Jint += ρij * Ω * ϖ² * dV
        Tkin += 0.5 * ρij * Ω² * ϖ² * dV
        Wpot += 0.5 * ρij * Φ[i, j] * dV   # (Φ < 0), W is negative
    end
    T_over_W = Wpot < 0.0 ? Tkin / abs(Wpot) : NaN

    return (; r, μ, ρ = copy(ρ), Φ = copy(Φ),
              Ω² = Ω², H_c = H_c,
              M = M, J = Jint, T = Tkin, W = Wpot, T_over_W,
              iters = iters_used, converged,
              n = nf, axis_ratio_requested = Float64(axis_ratio),
              axis_ratio_actual = α_actual,
              r_eq = r_eq, r_p = r_p)
end

# ---------------------------------------------------------------------------
# Helpers

"""
    mass_shedding_limit(n; α_lo=0.45, α_hi=0.95, steps=40, kw...) -> (α_peak, Ω²_peak)

Scan the Hachisu sequence between `α_lo` and `α_hi` and return the axis ratio
at which Ω² attains its maximum — the mass-shedding limit.  Below α_peak the
sequence continues to exist mathematically (Hachisu's surface condition can
always be satisfied) but Ω² decreases because increasing oblateness costs
more mass than it gains in centrifugal support.

Returns the pair `(α_peak, Ω²_peak)`.  The returned Ω²_peak is in units of
G ρ_c; for n = 3 the literature value is Ω²_peak/(π G ρ_c) ≈ 0.0085.
"""
function mass_shedding_limit(n::Real; α_lo::Real = 0.45, α_hi::Real = 0.95,
                              steps::Int = 40, kw...)
    αs = range(Float64(α_hi), Float64(α_lo); length = steps)
    α_peak = first(αs);  Ω²_peak = -Inf
    for α in αs
        sol = scf_rotating_polytrope(n, α; kw...)
        if sol.converged && sol.Ω² > Ω²_peak
            Ω²_peak = sol.Ω²
            α_peak  = sol.axis_ratio_actual
        end
    end
    return (α_peak, Ω²_peak)
end

# ---------------------------------------------------------------------------
# 3D Cartesian mapping
#
# Rescale the dimensionless SCF solution (ρ_c = r_eq = G = 1) to code units
# with prescribed physical M_star and R_star (= r_eq_code), paint onto the
# Cartesian grid, and impose the rigid rotation + bulk translation velocity
# field.  P is derived from the polytropic EoS
#     P = K ρ^(1 + 1/n),   K = H_c_dimless / (n+1) · ρ_c^{1-1/n} · r_eq²
# and thermal energy from e_int = P / (ρ (γ-1)).

"""
    rotating_polytrope_ic_3d!(U, nx, ny, nz, dx, dy, dz, γ;
        M_star, R_star, axis_ratio, M_core=0.0,
        x0=0, y0=0, z0=0,
        x_center=0, y_center=0, z_center=0,
        v_star=(0.0, 0.0, 0.0),
        ρ_floor=1e-10, P_floor=1e-8,
        Nr_scf=256, Nμ_scf=33, lmax_scf=12,
        scf_tol=1e-7, scf_mix=0.4, scf_maxiter=3000,
        verbose=false)

Build a rigidly-rotating, self-consistent axisymmetric polytrope (Hachisu SCF)
of index n = 1/(γ-1), total mass `M_star`, equatorial radius `R_star`, and
prescribed axis ratio `axis_ratio ∈ (0, 1]`, and paint it onto the active
cells of `U`.

The Ω_spin that comes out of the SCF is **not** a free parameter: it is
determined by the axis ratio together with (M_star, R_star).  If you have a
target Ω_spin (e.g. a multiple of Ω_orb), use [`axis_ratio_for_spin`](@ref)
to bisect on `axis_ratio` first.

If `M_core > 0`, the inner spherical region of radius `r_core` (determined so
the enclosed angle-averaged mass equals `M_core`) is hollowed to floor values
— the future sink location for BH2.

Returns a NamedTuple:
  - `Ω_spin`   : stellar spin (code units)
  - `ρ_c`      : central density (code units)
  - `K`        : polytropic constant in code units
  - `r_core`   : radius of hollowed core (0 if M_core = 0)
  - `α_actual` : grid-aligned axis ratio
  - `Ω²_dimless`, `H_c_dimless`, `M_mapped` (integrated on the 3D grid)
"""
function rotating_polytrope_ic_3d!(U::AbstractArray{T, 4}, nx, ny, nz,
                                    dx::Real, dy::Real, dz::Real, γ::Real;
                                    M_star::Real,
                                    R_star::Real,
                                    axis_ratio::Real,
                                    M_core::Real = 0.0,
                                    x0::Real = 0.0, y0::Real = 0.0, z0::Real = 0.0,
                                    x_center::Real = 0.0,
                                    y_center::Real = 0.0,
                                    z_center::Real = 0.0,
                                    v_star::NTuple{3, <:Real} = (0.0, 0.0, 0.0),
                                    ρ_floor::Real = 1e-10,
                                    P_floor::Real = 1e-8,
                                    Nr_scf::Int = 256,
                                    Nμ_scf::Int = 33,
                                    lmax_scf::Int = 12,
                                    scf_tol::Real = 1e-7,
                                    scf_mix::Real = 0.4,
                                    scf_maxiter::Int = 3000,
                                    verbose::Bool = false) where {T}

    n = 1.0 / (γ - 1.0)

    # --- SCF in dimensionless units
    sol = scf_rotating_polytrope(n, axis_ratio;
                                  Nr = Nr_scf, Nμ = Nμ_scf, lmax = lmax_scf,
                                  tol = scf_tol, mix = scf_mix,
                                  maxiter = scf_maxiter, verbose = verbose)
    sol.converged || @warn "SCF did not fully converge" iters=sol.iters axis_ratio=axis_ratio
    sol.Ω² > 0.0 || @warn "Non-positive Ω² from SCF — axis ratio may be too small (super-shedding)" Ω²=sol.Ω²

    # --- scale factors
    r_eq_code = Float64(R_star)
    ρ_c       = Float64(M_star) / (r_eq_code^3 * sol.M)
    Ω_spin    = sqrt(max(sol.Ω², 0.0) * ρ_c)     # G = 1, Ω_code² = Ω̂² · G ρ_c
    # Polytropic constant in code units.
    # Dimensionless: P̂ = ρ̂ · Ĥ / (n+1) = (Ĥ_c / (n+1)) · ρ̂^{1+1/n}  →  K̂ = Ĥ_c/(n+1)
    # Physical scaling: [K] = [P]/[ρ]^{1+1/n} = (ρ_c² r_eq²) / ρ_c^{1+1/n}
    #                      = ρ_c^{1-1/n} · r_eq²
    K_code = (sol.H_c / (n + 1.0)) * ρ_c^(1.0 - 1.0/n) * r_eq_code^2

    # --- Core hollow radius: find spherical r_core such that
    #     M_core = ∫_0^{r_core} 4π r² ρ̄(r) dr,
    #     ρ̄(r) = ∫_0^1 ρ(r, μ) dμ (angle-averaged, μ ∈ [0,1] half-domain).
    r_core = 0.0
    if M_core > 0.0
        Nr_s = length(sol.r)
        Δr_s = sol.r[2] - sol.r[1]
        Nμ_s = length(sol.μ)
        Δμ_s = sol.μ[2] - sol.μ[1]
        M_enc_prev = 0.0
        for i in 1:Nr_s
            ρ̄ = 0.0
            for j in 1:Nμ_s
                ρ̄ += sol.ρ[i, j]
            end
            ρ̄ *= Δμ_s
            # Dimensionless enclosed mass increment (spherical shell 4π r² dr)
            dM_dim = 4π * sol.r[i]^2 * ρ̄ * Δr_s
            M_enc_dim_next = M_enc_prev + dM_dim
            M_enc_next_code = M_enc_dim_next * ρ_c * r_eq_code^3
            M_enc_prev_code = M_enc_prev * ρ_c * r_eq_code^3
            if M_enc_next_code >= M_core
                frac = (M_core - M_enc_prev_code) /
                       max(M_enc_next_code - M_enc_prev_code, 1e-30)
                r_core = r_eq_code * (sol.r[i] - 0.5 * Δr_s + frac * Δr_s)
                break
            end
            M_enc_prev = M_enc_dim_next
        end
    end

    # --- Bilinear (r, μ) → ρ̂ lookup
    Nr_s = length(sol.r)
    Nμ_s = length(sol.μ)
    Δr_s = sol.r[2] - sol.r[1]
    Δμ_s = sol.μ[2] - sol.μ[1]
    r_scf_max = sol.r[end] + 0.5 * Δr_s
    ρ_scf = sol.ρ

    @inline function interp_ρ̂(r_dim::Float64, μ_abs::Float64)
        (r_dim <= 0.0 || r_dim >= r_scf_max) && return 0.0
        μ_cl = clamp(μ_abs, 0.0, 1.0)
        i_real = r_dim / Δr_s + 0.5
        i_lo = clamp(floor(Int, i_real), 1, Nr_s - 1)
        ri = i_real - i_lo
        j_real = μ_cl / Δμ_s + 0.5
        j_lo = clamp(floor(Int, j_real), 1, Nμ_s - 1)
        rj = j_real - j_lo
        ρ00 = ρ_scf[i_lo,     j_lo    ]
        ρ10 = ρ_scf[i_lo + 1, j_lo    ]
        ρ01 = ρ_scf[i_lo,     j_lo + 1]
        ρ11 = ρ_scf[i_lo + 1, j_lo + 1]
        return (1-ri)*(1-rj)*ρ00 + ri*(1-rj)*ρ10 + (1-ri)*rj*ρ01 + ri*rj*ρ11
    end

    # --- paint onto grid
    vsx, vsy, vsz = Float64(v_star[1]), Float64(v_star[2]), Float64(v_star[3])
    γf = Float64(γ)
    exponent = 1.0 + 1.0/n
    M_mapped = 0.0
    ng = NG
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx - x_center
        yc = y0 + (j - ng - 0.5) * dy - y_center
        zc = z0 + (k - ng - 0.5) * dz - z_center

        r_sph = sqrt(xc*xc + yc*yc + zc*zc)
        r_dim = r_sph / r_eq_code
        μ_abs = r_sph > 0.0 ? abs(zc) / r_sph : 0.0

        # Core hollow → floor values (no velocity)
        if M_core > 0.0 && r_sph < r_core
            U[1, i, j, k] = ρ_floor
            U[2, i, j, k] = 0.0
            U[3, i, j, k] = 0.0
            U[4, i, j, k] = 0.0
            U[5, i, j, k] = P_floor / (γf - 1.0)
            continue
        end

        ρ̂  = interp_ρ̂(r_dim, μ_abs)
        ρ = ρ_c * ρ̂

        if ρ <= ρ_floor
            U[1, i, j, k] = ρ_floor
            U[2, i, j, k] = 0.0
            U[3, i, j, k] = 0.0
            U[4, i, j, k] = 0.0
            U[5, i, j, k] = P_floor / (γf - 1.0)
            continue
        end

        P = K_code * ρ^exponent
        P = max(P, P_floor)

        # v_gas = v_star + Ω_spin ẑ × (r − r_star)
        vx = vsx - Ω_spin * yc
        vy = vsy + Ω_spin * xc
        vz = vsz

        U[1, i, j, k] = ρ
        U[2, i, j, k] = ρ * vx
        U[3, i, j, k] = ρ * vy
        U[4, i, j, k] = ρ * vz
        U[5, i, j, k] = P / (γf - 1.0) + 0.5 * ρ * (vx*vx + vy*vy + vz*vz)

        M_mapped += ρ * dx * dy * dz
    end

    return (; Ω_spin, ρ_c, K = K_code, r_core,
              α_actual = sol.axis_ratio_actual,
              Ω²_dimless = sol.Ω², H_c_dimless = sol.H_c,
              M_mapped, M_target = Float64(M_star),
              converged = sol.converged)
end

"""
    axis_ratio_for_spin(n, Ω_target_dimless;
                        α_lo=0.50, α_hi=0.999, bisect_tol=1e-4,
                        Nr=128, Nμ=17, lmax=10,
                        scf_tol=1e-7, maxiter=2000, mix=0.4)

Bisect on axis ratio to find the Hachisu solution whose dimensionless spin
`Ω / sqrt(G ρ_c)` matches `Ω_target_dimless`.  Only the stable branch
(axis ratio ≥ α_peak) is probed.  Throws if the target exceeds the
mass-shedding maximum for this `n`.
"""
function axis_ratio_for_spin(n::Real, Ω_target_dimless::Real;
                              α_lo::Real = 0.50, α_hi::Real = 0.999,
                              bisect_tol::Real = 1e-4,
                              Nr::Int = 128, Nμ::Int = 17, lmax::Int = 10,
                              scf_tol::Real = 1e-7,
                              maxiter::Int = 2000, mix::Real = 0.4)
    Ω²_target = Float64(Ω_target_dimless)^2
    α_peak, Ω²_peak = mass_shedding_limit(n;
                                            α_lo = α_lo, α_hi = α_hi, steps = 25,
                                            Nr = Nr, Nμ = Nμ, lmax = lmax,
                                            tol = scf_tol, mix = mix,
                                            maxiter = maxiter)
    Ω²_target > Ω²_peak && error(
        "Requested Ω² = $Ω²_target exceeds mass-shedding maximum " *
        "Ω²_peak = $Ω²_peak at α_peak = $α_peak.  Star cannot rotate " *
        "this fast in hydrostatic equilibrium for n = $n.")
    lo = max(α_peak, Float64(α_lo));  hi = Float64(α_hi)
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        sol = scf_rotating_polytrope(n, mid;
                                      Nr = Nr, Nμ = Nμ, lmax = lmax,
                                      tol = scf_tol, mix = mix,
                                      maxiter = maxiter)
        if sol.Ω² < Ω²_target
            hi = mid
        else
            lo = mid
        end
        (hi - lo) < bisect_tol && return 0.5 * (lo + hi)
    end
    return 0.5 * (lo + hi)
end

