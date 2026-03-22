# Phase 3: N-body BHs + Gas Sinks

## Objective

Phase 3 adds the two black-hole point masses and their coupling to the gas.
Three new modules are introduced:

- `gravity_bh.jl` — Plummer-softened BH gravitational potentials and the
  gas-momentum/energy source terms they generate; also computes the reaction
  force on each BH from the gas (Newton's third law).
- `nbody.jl` — SSP-RK3 N-body integrator for an arbitrary number of BHs,
  integrated in lockstep with the fluid using the same Butcher table as the
  gas solver.
- `sinks.jl` — Torque-free gas-sink prescription (Dempsey, Munoz & Lithwick
  2020) and a standard uniform-drain fallback; plus `accrete!` for
  fully-conservative BH mass and momentum bookkeeping.

---

## Implementation Notes

### `gravity_bh.jl`

The gravitational potential (G = 1, Plummer softening ε) is:

```
Φ_i(r) = −M_i / sqrt(|r − r_i|² + ε_i²)
```

The gas source terms per cell are:

```
d(ρv)/dt += ρ * Σ_i [−∇Φ_i(r_cell)]
d(E)/dt  += ρ * v · Σ_i [−∇Φ_i(r_cell)]
```

where `−∇Φ_i` points toward BH i.

The force on BH i from the gas (`gas_force_on_bh`) is:

```
F_gas→i = +∫ ρ ∇Φ_i dV
```

This is the Newton's-third-law reaction to the force the BH exerts on the gas.
Note: CLAUDE.md §3.2 has a sign inconsistency in the formula for `F_gas→i`; the
correct expression is `+∫ ρ ∇Φ_i dV` (attractive, pointing toward the gas centroid),
which is what is implemented here.

### `nbody.jl`

The equations of motion (G = 1) are:

```
dr_i/dt = v_i
dv_i/dt = Σ_{j≠i} M_j (r_j − r_i) / (|r_j − r_i|² + ε_j²)^(3/2) + F_gas→i / M_i
```

The integrator uses the same SSP-RK3 Shu-Osher scheme as the gas solver.
The gas forces `F_gas` are computed before the step and held constant
throughout the three sub-stages.  For Phase 3 (pure N-body test) these are
set to zero.

### `sinks.jl` — Torque-free prescription

For each cell within `r_sink(bh)` the source terms are (Dempsey+ 2020 eq. 6–7):

```
d(ρ)/dt    = −ρ / t_sink
d(ρv)/dt   = −(ρ v_r) r̂ / t_sink        [radial component only]
d(E)/dt    = −(½ρ v_r² + ρ e_int) / t_sink
```

where `v_r = (v_gas − v_BH) · r̂` is the radial speed in the BH rest frame
and `r̂ = (r_cell − r_BH) / |r_cell − r_BH|`.

The tangential kinetic energy `½ρ|v_tan|²` is not removed; gas surrounding the
sink conserves angular momentum.  The angular momentum drain is algebraically
zero: `r × d(ρv)/dt ∝ r × r̂ = 0`.

The `accrete!` function updates `bh.mass` and `bh.vel` using the **full** gas
momentum (both radial and tangential), ensuring strict N-body momentum
conservation (CLAUDE.md §4.3).

### Design decisions

- **F_gas held constant within nbody_step!**: For the Phase 3 tests (no gas),
  this is irrelevant.  In the full coupled simulation, `gas_force_on_bh` is
  called once per timestep at the start; updating it per sub-stage would
  require passing a callback and is deferred to Phase 6 if needed.
- **`torque_free = false` fallback**: The standard drain (all conserved variables
  at `1/t_sink`) is available via `add_sink_sources!(...; torque_free = false)`.
  Used for early cross-checks.
- **No accretion sub-step correction**: `accrete!` uses the pre-step gas state.
  At CFL = 0.4 the mass accreted per step is small and first-order accuracy is
  acceptable; a predictor-corrector can be added in Phase 6 diagnostics.

---

## Test Results

### Kepler orbit — energy and angular momentum conservation

**Setup**: two equal-mass BHs (M₁ = M₂ = 0.5) in a circular orbit,
separation a₀ = 1, G = 1, ε = 10⁻⁴, no gas.  Orbital period T = 2π.
Integration: SSP-RK3, dt = T/500 = 2π/500, 10 orbits (t_end = 20π).

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| Relative energy error \|ΔE/E₀\| | **0.0073%** | < 0.1% | ✅ |
| Relative angular-momentum error \|ΔL/L₀\| | **0.0036%** | < 0.1% | ✅ |

### Sink — mass accounting

Uniform gas (ρ = 1) on an 8³ grid, BH at domain centre, `r_sink = 3 Δx`.
Gas mass removal rate from `dU[1]` agrees with `accrete!` computation to
floating-point precision (difference < 10⁻¹⁴).

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| \|dm_gas_rate − dm_BH_rate\| / dm_BH_rate | **0.0%** | < 10⁻¹² | ✅ |

### Sink — torque-free angular-momentum drain

Gas in solid-body rotation (ω = 1) around a stationary BH.  Maximum torque
`|r × d(ρv)/dt|` over all sink cells.

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| max \|r × d(ρv)/dt\| | **2.8 × 10⁻¹⁷** | < 10⁻¹⁴ | ✅ |

The residual is at floating-point rounding level (≈ machine epsilon × |ρ v_r / t_sink|),
confirming that the torque is algebraically zero.

### `accrete!` mass conservation

Gas with ρ₀ = 1.5 on an 8³ grid, BH at domain centre.  ΔM_BH from `accrete!`
matches the expected value `M_gas_inside_sink / t_sink * dt` to relative error
< 10⁻¹³.

| Metric | Result | Threshold | Pass? |
|--------|--------|-----------|-------|
| \|ΔM_BH − expected\| / expected | **< 10⁻¹³** | < 10⁻¹⁰ | ✅ |

---

## Known Limitations

- **F_gas constant within step**: The gas-on-BH force is not updated at each
  SSP-RK3 sub-stage, introducing a splitting error O(dt²).  For Phase 3 this
  is not exercised (no gas in the N-body test); addressed in Phase 6.
- **No coupled gas+N-body test**: Phase 3 tests the N-body and sink modules
  independently.  A coupled run (BH orbit in gas potential) is deferred to
  Phase 5 (full supernova run) where all modules operate together.
- **Softening ε is static**: `bh.eps` is set at construction and does not
  adapt to the local fine-grid resolution.  Phase 5 initial conditions will
  set ε ~ 0.5 r_floor.
- **accrete! uses pre-step gas**: First-order accurate in Δm/M_BH; acceptable
  when accretion is a small perturbation per step.

---

## Next Steps

Phase 4 adds `stellar_ic.jl`: Lane-Emden polytrope solver, mapping the 3D
density and pressure profile to the grid, and placing BH1 on a Keplerian
orbit.  The stability test (star + BH1 stable for 2 P₀) validates that the
gravity source terms and N-body integrator work correctly together with the
hydro solver.

---

*All 52 tests pass (`julia --project=. -e 'using Pkg; Pkg.test()'`).*
