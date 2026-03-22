# Phase 1: 3D Adiabatic Euler Solver

## Overview

Phase 1 delivers a fully tested 3D adiabatic Euler solver on a uniform Cartesian grid with WENO5-Z reconstruction and HLLC Riemann fluxes, integrated with SSP-RK3.

## Components

### `src/weno5.jl`
WENO5-Z reconstruction (Borges et al. 2008) with left/right interface reconstruction functions. Characteristic stencil uses 5-cell support. Ghost cells (NG=3) required on each face.

### `src/hllc.jl`
3D HLLC Riemann solver (Toro 2009, §10.6). Single generic solver `_hllc_normal` rotated for x/y/z faces. Key robustness features:
- Full state reset when reconstructed ρ ≤ 0 (zeros momentum and energy)
- KE clamp: E = max(E, KE) ensures non-negative internal energy
- PVRS wave-speed estimate (Toro §10.5.1) with Davis (1988) vacuum fallback
- HLL fallback when ρ_max/ρ_min > 3 (carbuncle prevention at strong shocks)
- Guard against degenerate 0/0 in star-state energy when c_s = 0

### `src/euler3d.jl`
3D method-of-lines operator with SSP-RK3 (Shu-Osher). Key design decisions:
- **Pressure reconstruction**: WENO5-Z is applied to cell-average pressure P (not total energy E) to prevent the 3-million:1 blast-background energy contrast from causing WENO5 overshoots. Total energy is reconstructed as E = P/(γ-1) + KE from the primitive-based interface pressure. Pressure is clamped to [0, max_stencil_P] before the Riemann solve.
- Positivity floors applied after each SSP-RK3 stage (density floor + pressure floor, full state reset when ρ < ρ_floor)
- Outflow and periodic boundary conditions

## Tests

All Phase 1 tests pass (`julia --project=. -e 'using Pkg; Pkg.test()'`):

### Sod Shock Tube (3D, x/y/z sweeps)
- IC: ρ_L=1, P_L=1, ρ_R=0.125, P_R=0.1, γ=5/3
- Grid: 256 × 4 × 4 (active), t_end=0.2
- L1 errors: ρ ≈ 0.0038, v ≈ 0.0073, P ≈ 0.0031 (threshold: 0.05)
- Sweep symmetry: x/y/z errors identical to 5 significant figures

### Sedov-Taylor Blast Wave (3D, γ=5/3)
- IC: E_blast=1 deposited in sphere of radius 2.5 dx, uniform ρ=1 background
- Grid: 32³ (active), t_end=0.1
- Shock radius: R_num=0.445, R_exact=0.458, **error=2.9%** (threshold: 10%)
- Spherical symmetry: x-y error 0.012%, x-z error 0.068% (threshold: 10%)
- Energy conservation: < 5% loss from outflow BC

## Key Numerical Challenges Solved

1. **WENO5 energy overshoot at blast discontinuity**: The initial blast creates a ~3×10⁶ contrast in total energy E between blast cells and background. Direct WENO5 on E produces reconstructed values 270× above the stencil maximum by step 14, driving a runaway instability (E → 10¹⁸ by step 60). **Fix**: reconstruct pressure P instead and cap to stencil maximum.

2. **Star-state NaN when c_s = 0**: WENO5 can reconstruct E < KE, giving P = 0 → c_s = 0 → SR = v_n. The contact speed formula then divides by ρ(SR − v_n) = 0. **Fix**: guard with `abs(denom) > 1e-100` check.

3. **Runaway from negative reconstructed ρ**: WENO5 reconstructs ρ < 0 at strong density gradients. The HLLC floor raises ρ to 1e-30 but keeps original E, giving E/ρ → ∞. **Fix**: zero out momentum and E when ρ ≤ 0.

## Performance

On a MacBook Pro (M-series), the 32³ Sedov test completes ~29 timesteps to t=0.1 in about 2 seconds. The Sod test (256×4×4 × 3 sweeps) takes ~10 seconds total.
