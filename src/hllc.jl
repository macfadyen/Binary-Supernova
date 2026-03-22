# HLLC Riemann solver for 3D adiabatic Euler.
# Conserved variables: (ρ, ρvx, ρvy, ρvz, E)
# EOS: P = (γ−1)(E − ½ρ|v|²)
#
# Strategy: one generic normal-direction solver `_hllc_normal` that takes
# (normal momentum, transverse-1 momentum, transverse-2 momentum) and returns
# fluxes in the same ordering.  The public x/y/z wrappers rotate inputs in and
# rotate outputs back so the caller always works with (ρ, mx, my, mz, E).
#
# Reference: Toro (2009) §10.6 — multi-dimensional HLLC.
# Wave-speed estimates: PVRS (Toro §10.5.1), same as HighMachCBD 1D solver.

# ---------------------------------------------------------------------------
# Core solver: normal direction n, two transverse directions t1, t2.
# Returns (Fρ, Fmn, Fmt1, Fmt2, FE) — numerical flux along n.

@inline function _hllc_normal(ρL, mnL, mt1L, mt2L, EL,
                               ρR, mnR, mt1R, mt2R, ER, γ)
    # Reset full state when ρ ≤ 0 — WENO5 reconstructs ρ and E independently,
    # so a negative reconstructed ρ (floored to 1e-30) can pair with a large
    # positive E from the neighboring normal cells, giving E/ρ → ∞ in the
    # HLLC star-state formula.  Zeroing momentum and E prevents this.
    if ρL <= 0.0
        ρL = 1e-30; mnL = 0.0; mt1L = 0.0; mt2L = 0.0; EL = 0.0
    else
        ρL = max(ρL, 1e-30)
    end
    if ρR <= 0.0
        ρR = 1e-30; mnR = 0.0; mt1R = 0.0; mt2R = 0.0; ER = 0.0
    else
        ρR = max(ρR, 1e-30)
    end

    # Clamp total energy to ≥ kinetic energy (ensures internal energy ≥ 0).
    # WENO5 can reconstruct E < 0 at strong discontinuities; a negative E
    # in the physical flux (E + p)*vn would drain total energy unphysically.
    KE_L = 0.5 * (mnL^2 + mt1L^2 + mt2L^2) / ρL
    KE_R = 0.5 * (mnR^2 + mt1R^2 + mt2R^2) / ρR
    EL   = max(EL, KE_L)
    ER   = max(ER, KE_R)

    vnL  = mnL / ρL;  vt1L = mt1L / ρL;  vt2L = mt2L / ρL
    vnR  = mnR / ρR;  vt1R = mt1R / ρR;  vt2R = mt2R / ρR

    # Pressure is now guaranteed ≥ 0 by the E ≥ KE clamp above.
    pL = (γ - 1) * (EL - KE_L)
    pR = (γ - 1) * (ER - KE_R)

    csL = sqrt(γ * pL / ρL)
    csR = sqrt(γ * pR / ρR)

    # Wave-speed estimate: PVRS (Toro §10.5.1) when both pressures are positive;
    # Davis (1988) min/max estimate when either side is vacuum (p = 0).
    # Davis guarantees SL ≤ 0 ≤ SR across a contact/rarefaction even for vacuum,
    # preventing the degenerate SR = 0 that traps energy at blast-zone boundaries.
    if pL > 0.0 && pR > 0.0
        ρ_avg  = 0.5 * (ρL + ρR)
        cs_avg = 0.5 * (csL + csR)
        p_star = max(0.0, 0.5 * (pL + pR) - 0.5 * (vnR - vnL) * ρ_avg * cs_avg)
        qL = p_star > pL ? sqrt(1.0 + (γ + 1) / (2γ) * (p_star / pL - 1.0)) : 1.0
        qR = p_star > pR ? sqrt(1.0 + (γ + 1) / (2γ) * (p_star / pR - 1.0)) : 1.0
        SL = vnL - csL * qL
        SR = vnR + csR * qR
    else
        SL = min(vnL - csL, vnR - csR)
        SR = max(vnL + csL, vnR + csR)
    end

    # Contact wave speed (Toro Eq. 10.37).
    denom  = ρL * (SL - vnL) - ρR * (SR - vnR)
    # Guard: double-vacuum degenerate state (both p=0, gas diverging) → zero flux.
    abs(denom) < 1e-100 && return 0.0, 0.0, 0.0, 0.0, 0.0
    S_star = (pR - pL + ρL * vnL * (SL - vnL) - ρR * vnR * (SR - vnR)) / denom

    # Physical fluxes.
    FρL   = mnL
    FmnL  = ρL * vnL^2 + pL
    Fmt1L = mt1L * vnL           # transverse: ρ vt1 vn
    Fmt2L = mt2L * vnL
    FEL   = (EL + pL) * vnL

    if SL >= 0.0
        return FρL, FmnL, Fmt1L, Fmt2L, FEL
    end

    FρR   = mnR
    FmnR  = ρR * vnR^2 + pR
    Fmt1R = mt1R * vnR
    Fmt2R = mt2R * vnR
    FER   = (ER + pR) * vnR

    if SR <= 0.0
        return FρR, FmnR, Fmt1R, Fmt2R, FER
    end

    # HLL fallback for large density contrasts (carbuncle prevention).
    # When ρL/ρR > 20 (or vice-versa) — e.g., blast interior meeting background —
    # HLLC concentrates energy into low-density cells causing runaway cs and CFL collapse.
    # HLL is more dissipative but avoids this instability at strong rarefaction fronts.
    if max(ρL, ρR) > 3.0 * min(ρL, ρR)
        inv_dS = 1.0 / (SR - SL)
        return (SR * FρL   - SL * FρR   + SL * SR * (ρR  - ρL )) * inv_dS,
               (SR * FmnL  - SL * FmnR  + SL * SR * (mnR - mnL)) * inv_dS,
               (SR * Fmt1L - SL * Fmt1R + SL * SR * (mt1R - mt1L)) * inv_dS,
               (SR * Fmt2L - SL * Fmt2R + SL * SR * (mt2R - mt2L)) * inv_dS,
               (SR * FEL   - SL * FER   + SL * SR * (ER  - EL )) * inv_dS
    end

    # HLLC star states (Toro §10.6).
    if S_star >= 0.0
        denom_L = ρL * (SL - vnL)           # = ρL*(SL-vnL); same sign as denom above
        χ      = denom_L / (SL - S_star)
        ρsL    = χ
        mnsL   = χ * S_star
        mt1sL  = χ * vt1L
        mt2sL  = χ * vt2L
        # Guard: when csL = 0 → SL = vnL → denom_L = 0 → χ = 0 → EsL = 0.
        # Without guard, pL/denom_L = 0/0 = NaN and χ*NaN = NaN (IEEE 754).
        EsL    = abs(denom_L) > 1e-100 ?
                     χ * (EL / ρL + (S_star - vnL) * (S_star + pL / denom_L)) :
                     0.0
        return FρL  + SL * (ρsL  - ρL),
               FmnL + SL * (mnsL - mnL),
               Fmt1L + SL * (mt1sL - mt1L),
               Fmt2L + SL * (mt2sL - mt2L),
               FEL  + SL * (EsL  - EL)
    else
        denom_R = ρR * (SR - vnR)
        χ      = denom_R / (SR - S_star)
        ρsR    = χ
        mnsR   = χ * S_star
        mt1sR  = χ * vt1R
        mt2sR  = χ * vt2R
        # Guard: when csR = 0 → SR = vnR → denom_R = 0 → χ = 0 → EsR = 0.
        EsR    = abs(denom_R) > 1e-100 ?
                     χ * (ER / ρR + (S_star - vnR) * (S_star + pR / denom_R)) :
                     0.0
        return FρR  + SR * (ρsR  - ρR),
               FmnR + SR * (mnsR - mnR),
               Fmt1R + SR * (mt1sR - mt1R),
               Fmt2R + SR * (mt2sR - mt2R),
               FER  + SR * (EsR  - ER)
    end
end

# ---------------------------------------------------------------------------
# Public x/y/z interfaces.
# All take (ρ, mx, my, mz, E)_L/R and return (Fρ, Fmx, Fmy, Fmz, FE).

"""
    hllc_flux_x(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ) → (Fρ,Fmx,Fmy,Fmz,FE)

HLLC flux in the x-direction.  Normal = x, transverse = (y, z).
"""
@inline function hllc_flux_x(ρL, mxL, myL, mzL, EL,
                              ρR, mxR, myR, mzR, ER, γ)
    return _hllc_normal(ρL, mxL, myL, mzL, EL,
                        ρR, mxR, myR, mzR, ER, γ)
end

"""
    hllc_flux_y(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ) → (Fρ,Fmx,Fmy,Fmz,FE)

HLLC flux in the y-direction.  Normal = y, transverse = (x, z).
Rotates (my→mn, mx→mt1, mz→mt2) into the normal solver, then unrotates output.
"""
@inline function hllc_flux_y(ρL, mxL, myL, mzL, EL,
                              ρR, mxR, myR, mzR, ER, γ)
    # Rotate in: normal=y, t1=x, t2=z
    Fρ, Fmy, Fmx, Fmz, FE = _hllc_normal(ρL, myL, mxL, mzL, EL,
                                           ρR, myR, mxR, mzR, ER, γ)
    return Fρ, Fmx, Fmy, Fmz, FE
end

"""
    hllc_flux_z(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ) → (Fρ,Fmx,Fmy,Fmz,FE)

HLLC flux in the z-direction.  Normal = z, transverse = (x, y).
Rotates (mz→mn, mx→mt1, my→mt2) into the normal solver, then unrotates output.
"""
@inline function hllc_flux_z(ρL, mxL, myL, mzL, EL,
                              ρR, mxR, myR, mzR, ER, γ)
    # Rotate in: normal=z, t1=x, t2=y
    Fρ, Fmz, Fmx, Fmy, FE = _hllc_normal(ρL, mzL, mxL, myL, EL,
                                           ρR, mzR, mxR, myR, ER, γ)
    return Fρ, Fmx, Fmy, Fmz, FE
end
