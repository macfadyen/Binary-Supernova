# SN30 science run — 30 M☉ BH + 60 M☉ WR → 30 M☉ BH + 30 M☉ fallback

This directory contains two scripts for the baseline Binary-Supernova science
scenario: a 30 M☉ pre-existing black hole in a close circular orbit with a
60 M☉ low-metallicity stripped progenitor that core-collapses to a 30 M☉
remnant and ejects ~30 M☉ of gas. The science question is whether the ejecta
falls back onto the binary and how the accretion reshapes the post-SN orbit.

## Scripts

- `run_sn30_nosink.jl` — Run 1, shakedown with sinks OFF. Validates gas
  dynamics and reports `M_bound(t)` as the "did it fall back?" proxy.
- `run_sn30_sinks.jl` — Run 2, sinks + `accrete!` ON. Same IC as Run 1 with
  Ṁ₁(t), Ṁ₂(t) and BH-mass growth now meaningful.

## Physical setup (code units, G = M_tot = a₀ = 1)

| Parameter    | Value          | Physical              | Notes                                              |
|--------------|----------------|-----------------------|----------------------------------------------------|
| M_BH1        | 1/3            | 30 M☉                 | Pre-existing BH                                    |
| M_star       | 2/3            | 60 M☉                 | Low-Z progenitor (stripped WR / He-star)           |
| M_BH2_init   | 1/3            | 30 M☉                 | Direct-collapse remnant                            |
| M_ejecta     | 1/3            | 30 M☉                 | Implied                                            |
| a₀           | 1              | 30 R☉                 | Detached pre-SN binary                             |
| R_star       | 0.1            | 3 R☉                  | Stripped He-star; R_L/a ≈ 0.44 so detached         |
| γ            | 5/3            | —                     | Shakedown; revisit 4/3 once baseline stable        |
| E_SN         | 0.3            | ≈ 3 × 10⁵⁰ erg        | Specific e_SN ≈ binary well depth → most bound    |
| V_kick       | [0,0,0]        | 0 km/s                | Direct-collapse 30 M☉ BH → near-zero kick          |
| L (domain)   | 4              | ±120 R☉               | Ejecta stays on-grid for t ≳ 5                    |
| T_END        | 30             | ≈ 60 d                | ≈ 5 orbital periods (P₀ = 2π)                      |
| DT_SNAP      | 0.5            | ≈ 10 h                | Dense during active fallback                       |

Physical-unit conversion is `PhysicalUnits(90.0, 30.0)` (M_tot = 90 M☉ pre-SN,
a₀ = 30 R☉) → v_unit ≈ 756 km/s, c_code ≈ 397, t_unit ≈ 2.8 × 10⁴ s, one
code energy unit ≈ 1 Bethe.

## Usage

```
# Quick smoke test (a few minutes on M3 Max)
julia --project=. scripts/run_sn30_nosink.jl --nx 64

# Baseline resolution
julia --project=. scripts/run_sn30_nosink.jl --nx 128

# Once Run 1 is stable, accretion run at same resolution
julia --project=. scripts/run_sn30_sinks.jl --nx 128
# Flip the sink prescription to torque-free for the science result:
julia --project=. scripts/run_sn30_sinks.jl --nx 128 --torque-free

# GPU (A100):
julia --project=. scripts/run_sn30_nosink.jl --gpu --nx 192
```

Outputs land in `demo1/output_sn30_nosink/` or `demo1/output_sn30_sinks/`:
`trajectory.h5`, `diagnostics.csv`, `snap_tNNN.h5`.

## What to look for

**Run 1 headlines (sinks off):**
- `M_bound(t)` in `diagnostics.csv`: should rise sharply as the bomb drives
  gas outward, then settle to a fraction of M_ejecta (≈ 1/3) as the unbound
  tail leaves and the bound component fallback-accretes toward the binary.
  "Most falls onto the binary" → end-run `M_bound / M_ejecta` ≳ 0.5 expected
  for E_SN = 0.3.
- Post-process BH trajectory for orbit evolution:
  ```julia
  using BinarySupernova
  el = orbit_elements_from_trajectory("demo1/output_sn30_nosink/trajectory.h5";
                                       G_grav = 1.0)
  # el.t, el.a, el.e, el.bound, el.T_orb, el.ε
  ```
  Circular IC should give `e ≈ 0` at t = 0 and `bound = true` throughout.
  Drift in `a(t)` during fallback quantifies the orbital response.
- Watch E_gas drift (≲ 1% over one orbit after outflow losses settle) and
  that the BH trajectories remain finite.

**Run 2 additions (sinks on):**
- `Mdot1`, `Mdot2` columns become non-zero and track the accretion timing.
- `M_BH1` and `M_BH2` grow. End-run `ΔM_BH1 + ΔM_BH2` should be a meaningful
  fraction of M_ejecta (≥ 0.1 at NX = 128; higher with the torque-free flag
  and/or better resolution).
- Compare `M_bound` trajectory with Run 1: accretion drains bound mass onto
  the BHs, so `M_bound` in Run 2 falls below the Run 1 curve over time.

## Caveats

- **Resolution**: R_star / dx ≈ 1.6 at NX = 64, 2.4 at NX = 128. The 3 R☉
  star is small on the ±120 R☉ domain. The bomb + fallback is dominated by
  the global binary potential, not sub-R_star structure, so this is adequate
  for a first science run. Production-quality would use FMR (not yet wired
  to BHs/sinks — follow-up dev task).
- **γ = 5/3, not 4/3**: ρ_c ≈ 90 (n=3/2) is better-conditioned at modest
  resolution than n=3 (ρ_c ≈ 840). The physically correct γ for a radiation-
  dominated massive star interior is 4/3; rerun at γ = 4/3 once a γ = 5/3
  baseline is stable and you trust the results.
- **Self-gravity OFF**: for M_bound ~ M_BH ~ 0.3 code, gas self-gravity could
  matter during peak fallback. Toggle on in a follow-up via
  `add_self_gravity_source!` in the coupled step (uniform grid) or via the
  `fmr_gravity.jl` path once an FMR+BH driver exists.
- **Natal kick = 0**: matches the direct-collapse heavy-BH expectation
  (Fryer+ 2012 rapid; Mandel-Müller 2020). For a follow-up population study
  sweep |V_kick| ∈ {0, 50, 100, 200} km/s (code: |V| ≈ {0, 0.066, 0.13, 0.26}).
