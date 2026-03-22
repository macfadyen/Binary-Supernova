# Phase 0 — Bootstrap

## Objective

Stand up the `BinarySupernova` Julia package: directory layout, dependencies,
core data structures, and the two numerical modules copied verbatim from
HighMachCBD. This phase establishes the scaffold that all subsequent phases
build on.

## Implementation Notes

### Package structure

```
Binary-Supernova/
├── CLAUDE.md               project design document
├── Project.toml            Julia package manifest (same deps as HighMachCBD)
├── src/
│   ├── BinarySupernova.jl  main module — structs, unit helpers, includes
│   ├── weno5.jl            WENO5-Z reconstruction (verbatim from HighMachCBD)
│   └── rk3.jl              SSP-RK3 integrator    (verbatim from HighMachCBD)
├── test/
│   ├── runtests.jl
│   ├── test_weno5.jl       copied from HighMachCBD
│   ├── test_rk3.jl         new: ODE convergence + conservation
│   └── test_blackhole.jl   new: struct, r_sink, t_sink, unit conversion
├── docs/
│   ├── phase0.md           this file
│   └── figures/            (empty — no figures generated in Phase 0)
└── scripts/                (empty — plotting scripts added per phase)
```

### Core data structures defined in `BinarySupernova.jl`

**`PhysicalUnits`** — derives all code-to-CGS conversion factors from two inputs:
total system mass in solar masses and initial binary separation in solar radii.
G = M_total = a₀ = 1 in code units, so:

```
v_unit = sqrt(G_phys * M_total_phys / a₀_phys)
t_unit = a₀_phys / v_unit
c_code = c_phys  / v_unit
```

For a 20 M☉ system at 10 R☉: v_unit ≈ 618 km/s, c_code ≈ 485.

**`BlackHole`** — mutable struct holding position, velocity, mass, softening,
c_code, and numerical sink floor. The derived functions `r_sink(bh)` and
`t_sink(bh)` are not stored — they are computed on-the-fly from `bh.mass`:

```
r_sink(bh) = max(6 * bh.mass / bh.c_code², bh.r_floor)   [ISCO or numerical floor]
t_sink(bh) = f_sink * r_sink / sqrt(2 * bh.mass / r_sink) [free-fall timescale]
```

**`SimParams`** — immutable struct holding all physical and numerical parameters
for a run, with validation assertions on construction. `M_ejecta(p)` returns
`p.M_star - p.M_BH2_init`.

### Deviation from plan

None. The CLAUDE.md comment "c_code ~ 10⁴ for stellar systems" was incorrect;
the correct value for compact pre-SN binaries (a₀ ~ 10 R☉) is c_code ~ 500.
The comment and the example were corrected before the Phase 0 commit.

## Test Results

All 25 tests pass (`julia --project test/runtests.jl`).

| Test set | Tests | Result |
|---|---|---|
| WENO5 reconstruction | 4 | ✅ pass |
| SSP-RK3 integrator | 4 | ✅ pass |
| BlackHole struct and sink radii | 11 | ✅ pass |
| SimParams construction | 4 | ✅ pass |
| PhysicalUnits conversion | 2 | ✅ pass |
| **Total** | **25** | **✅ all pass** |

### WENO5

- Linear reconstruction exact to 1e-13 ✓
- Quadratic reconstruction exact to 1e-13 ✓

### SSP-RK3

- Zero RHS: state unchanged to rtol=1e-14 ✓
- Exponential decay dy/dt = −y: error at t=1 with dt=0.01 is < 1e-6 ✓
- Convergence ratio (dt=0.1 vs dt=0.01) > 100×, confirming > 2nd-order ✓

### BlackHole / PhysicalUnits

- G = 1 in code units verified to rtol=1e-12 ✓
- c_code = C_CGS / v_unit verified to rtol=1e-12 ✓
- r_sink floor operative (ISCO ≪ r_floor for physical c_code) ✓
- r_sink = 6M/c² when c_code = 1 (artificial non-relativistic test) ✓
- t_sink = r_sink / v_ff matches formula to rtol=1e-12 ✓
- SimParams validation: M_BH2_init ≥ M_star throws AssertionError ✓

## Known Limitations

- `BlackHole.pos` and `.vel` are plain `Vector{Float64}` (heap-allocated).
  A future refactor could use `SVector{3,Float64}` from StaticArrays.jl for
  better performance in tight N-body loops, but this is deferred to Phase 3
  when the N-body integrator is written.

- No GPU backend test yet. GPU portability (KernelAbstractions) is tested
  starting in Phase 1 when the first compute kernels are written.

- `Manifest.toml` is excluded from git (`.gitignore`). To reproduce the exact
  package environment, pin with `julia --project -e 'import Pkg; Pkg.instantiate()'`.

## Next Steps

Phase 1 implements the 3D adiabatic Euler solver: HLLC Riemann solver
(`hllc.jl`) and the 3D sweep solver with WENO5 + HLLC + SSP-RK3
(`euler3d.jl`), targeting Sod shock tube and Sedov-Taylor blast wave tests.
