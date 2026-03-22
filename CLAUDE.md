# BinarySupernova — Project Design Document

## 1. Overview

Simulate a supernova explosion in a close binary system with a pre-existing black hole
companion. The exploding star collapses to a second black hole (possibly with a natal
kick), and both black holes are then embedded in expanding supernova ejecta and fallback
material.

Built in Julia for GPU execution via KernelAbstractions.jl. Uses HighMachCBD
(`~/sandbox/HighMachCBD`) as the numerical foundation; several source modules are
adopted verbatim and the rest follow the same design conventions.

### Scientific goals
- Determine whether the binary survives the explosion (bound BH-BH pair → future GW
  source) or is disrupted (natal kick + mass loss exceeds binding energy)
- Track fallback accretion history onto each BH: M_BH(t), Ṁ(t)
- Measure how ejecta angular momentum and BH natal kicks shape the post-SN orbital
  parameters (separation, eccentricity)
- Advanced: assess role of gas self-gravity in fallback; use realistic MESA stellar
  profiles; improve ICs via Roche-potential relaxation

---

## 2. Code Units

```
G      = 1
M_tot  = M_BH1 + M_star = 1        (total system mass)
a₀     = 1                          (initial binary separation)
→  Ω₀  = sqrt(G M_tot / a₀³) = 1
   P₀  = 2π
   c   = c_phys / sqrt(G_phys * M_tot_phys / a₀_phys)   [>> 1; ~500 for compact binaries]
```

Physical conversion example: M_tot = 20 M☉, a₀ = 10 R☉ → c ≈ 485 (code units).
For wider pre-SN orbits (a₀ ~ 100 R☉): c ≈ 1500; for very wide systems: c ≈ 10⁴.

---

## 3. Governing Equations

### 3.1 Gas — 3D Adiabatic Euler

Conserved variables: **U** = (ρ, ρvₓ, ρvᵧ, ρvᵤ, E)

```
∂ρ/∂t     + ∇·(ρv)                 = S_ρ
∂(ρv)/∂t  + ∇·(ρvv + P I)          = −ρ ∇(Φ₁ + Φ₂) + S_ρv
∂E/∂t     + ∇·((E + P)v)            = −ρ v·∇(Φ₁ + Φ₂)  + S_E

P = (γ − 1)(E − ½ρ|v|²)            [γ = 4/3 radiation-dominated star / 5/3 for ejecta]
```

S terms are sink source terms (Section 4). Gas self-gravity (Phase 7) adds
−ρ ∇Φ_gas to the momentum and energy equations.

### 3.2 Black Hole Point Masses — N-body

```
dM_i/dt = Ṁ_i(t)                              [grows via accretion]
dv_i/dt = Σ_{j≠i} −∇Φ_j|_{r_i} + F_gas→i / M_i
dr_i/dt = v_i
```

where:
- Φ_j(r) = −G M_j / sqrt(|r − r_j|² + ε_j²)   [softened BH potential]
- F_gas→i = −∫ ρ ∇Φ_i dV                        [gas gravitational force on BH i]

BH equations of motion are integrated with SSP-RK3 in lockstep with the fluid.

---

## 4. Sink Prescription

### 4.1 BlackHole struct

```julia
mutable struct BlackHole
    pos     :: SVector{3, Float64}   # current position (N-body updated)
    vel     :: SVector{3, Float64}   # current velocity (N-body + accretion updated)
    mass    :: Float64               # current mass (grows via accretion)
    eps     :: Float64               # gravitational softening radius
    c       :: Float64               # speed of light in code units (constant)
    r_floor :: Float64               # minimum sink radius ≈ 2 Δx_fine (set at init)
end

r_sink(bh::BlackHole) = max(6.0 * bh.mass / bh.c^2,  bh.r_floor)
t_sink(bh::BlackHole) = f_sink * r_sink(bh) / sqrt(2.0 * bh.mass / r_sink(bh))
```

`r_sink = 6 G M_BH / c²` is the ISCO radius of a Schwarzschild BH. As fallback
accretion grows M_BH, r_sink grows automatically. In practice r_sink ≪ Δx_fine so
r_floor is the operative value throughout; the formula is the correct physical limit and
ensures r_sink scales correctly in the sub-grid model.

`t_sink` is set to a multiple `f_sink ~ 1–10` of the free-fall time at the sink
boundary, so the removal rate scales naturally with the BH mass as accretion proceeds.

### 4.2 Torque-free gas source terms

For each cell within r_sink(BH i) of BH i, the following source is applied at each
RK sub-stage:

```
d     = r_cell − bh.pos           # displacement from BH (3-vector)
r̂    = d / |d|                    # unit radial vector in BH frame
v     = ρv_cell / ρ_cell          # primitive velocity
v_rel = v − bh.vel                # relative to BH
v_r   = v_rel · r̂                # radial speed in BH frame (scalar)
e_int = E/ρ − ½|v|²              # specific internal energy

d(ρ)/dt    = −ρ / t_sink
d(ρv)/dt   = −(ρ v_r) r̂ / t_sink           # radial component only
d(E)/dt    = −(½ρ v_r² + ρ e_int) / t_sink  # radial KE + thermal only
```

Torque on gas about BH centre:  **d** × d(ρ**v**)/dt  ∝  **d** × **r̂** = **0** ✓

The tangential kinetic energy (½ρ|v_tan|²) remains in the gas. Gas surrounding the
sink region conserves angular momentum — speeds up slightly as mass is removed.

This is the 3D adiabatic generalisation of the Dempsey, Munoz & Lithwick (2020)
torque-free prescription used in HighMachCBD.

A `torque_free = false` flag reverts to the standard prescription (all conserved
variables drained at rate 1/t_sink), useful for comparison and early testing.

### 4.3 BH N-body update from accretion

The BH receives the **full** momentum of the accreted gas (strict momentum
conservation for the N-body trajectory), even though only the radial component was
removed from the gas source terms. The difference is the torque-free correction that
keeps the surrounding gas angular momentum intact.

```julia
function accrete!(bh::BlackHole, sink_cells, dV, dt)
    Δm = sum(ρ[c] * dV  for c in sink_cells) / t_sink(bh) * dt
    ΔP = sum(ρv[c] * dV for c in sink_cells) / t_sink(bh) * dt   # full 3-vector
    bh.vel  = (bh.mass * bh.vel + ΔP) / (bh.mass + Δm)
    bh.mass += Δm
    # r_sink(bh) and t_sink(bh) update automatically
end
```

### 4.4 Diagnostics per BH

```
Ṁ(t)        mass accretion rate
M_BH(t)     cumulative mass including fallback
r_sink(t)   = max(6 G M_BH / c², r_floor) — growing ISCO radius
Ṗ(t)        momentum accretion rate vector
J̇_acc(t)   angular momentum accretion rate (= 0 for torque-free, by construction)
```

---

## 5. Fixed-Mesh Refinement

### 5.1 Layout

A **single fine zone** is used, centred on the binary, covering both BHs and the inner
ejecta simultaneously. Three levels, refinement ratio 4:1 at each interface:

```
Level 0 (coarse):  [−L,   L  ]³    Δx₀          full domain, far ejecta, outer BC
Level 1 (medium):  [−L/4, L/4]³    Δx₀/4        mid ejecta
Level 2 (fine):    [−L/16,L/16]³   Δx₀/16       both BHs + inner ejecta
```

L is chosen so Level 0 contains several stellar radii of ejecta. Level 2 must remain
large enough to keep both BHs within it throughout the simulation.

If the binary is disrupted and the BHs separate beyond the fine zone, Level 2 can be
disabled between restarts and Level 1 takes over.

### 5.2 FMR numerics

- **Prolongation**: 5th-order Lagrange interpolation (generalised from HighMachCBD's
  2:1 stencil to the 4:1 case; WENO5 requires 3 ghost cells, 4:1 provides 12 fine
  cells of coverage — well within the 5-point stencil)
- **Restriction**: conservative averaging over 4³ = 64 fine cells per coarse cell
- **Flux correction**: Berger-Colella at coarse-fine interfaces (same algorithm as
  HighMachCBD, generalised to 3D and ratio 4)
- **Time subcycling**: fine level takes 4 steps per coarse step (ratio 4); each fine
  step uses the same CFL condition as the uniform-grid solver
- **`refinement_ratio` parameter**: accepts 2 or 4; 2:1 available for testing and
  for levels where 4:1 is not needed

---

## 6. Initial Conditions

### 6.1 Baseline: Polytrope (Phases 4–6)

Solve the Lane-Emden ODE for a polytrope of index n = 3 (γ = 4/3,
radiation-dominated massive star):

```
(1/ξ²) d/dξ [ξ² dθ/dξ] = −θⁿ,   θ(0) = 1,  θ'(0) = 0
```

The numerical solution gives θ(ξ) on [0, ξ₁] where θ(ξ₁) = 0 (stellar surface).
Map onto the 3D grid:

```
ρ(r)  = ρ_c  θ(r / r_scale)^n
P(r)  = K ρ(r)^(1 + 1/n)
```

with ρ_c and r_scale set so that the total stellar mass = M_ejecta = M_star − M_BH2_init
and the stellar radius = R_star (input parameter).

The inner core (mass = M_BH2_init, radius = r_core satisfying
∫₀^r_core 4πr²ρ dr = M_BH2_init) is **not placed on the grid**; it becomes BH2 at t = 0.

BH1 is placed at position (a₀, 0, 0) with velocity (0, v_orb, 0), the star at the
origin. Both on circular Keplerian orbit.

### 6.2 Advanced: MESA Stellar Model (Phase 8)

Read a 1D MESA stellar profile (radius, density, temperature, pressure columns).
Map onto the 3D grid assuming spherical symmetry. Derive a local effective γ(r) from
the MESA EOS at each radius to be consistent with the adiabatic solver.

### 6.3 Advanced: Roche Potential Relaxation (Phase 9)

Improved baseline IC that accounts for tidal distortion and potential Roche lobe
overflow:

1. Place the polytrope (or MESA profile) on the grid
2. Add BH1 gravity field (full binary potential)
3. Apply velocity damping source: `d(ρv)/dt += −ρv / t_damp` with t_damp ~ 0.1 P₀
4. Evolve for 1–2 P₀ until kinetic energy < 1% of thermal energy
5. Remove damping; confirm stability for 0.5 P₀ without it
6. Trigger thermal bomb

This correctly shapes the star into its Roche geometry before the explosion, which
affects the early shock propagation and the fallback trajectory.

---

## 7. Supernova Explosion Trigger (t = 0)

### 7.1 Thermal bomb

Deposit explosion energy E_SN as thermal energy, mass-weighted over the ejecta gas:

```
ΔE[cell] = E_SN × (ρ[cell] dV) / M_ejecta    for r_cell < r_bomb
```

Sum over cells = E_SN exactly. Typical r_bomb = R_star (whole star) or R_star/2
(inner half). No energy deposited in the BH2 sink region (r < r_core).

### 7.2 BH2 activation

Simultaneously with the thermal bomb:
1. Activate BH2 sink at the stellar centre with mass M_BH2_init
2. Set r_sink = max(6 G M_BH2_init / c², r_floor)
3. Apply natal kick: `bh2.vel += v_kick`  (v_kick is a free 3-vector parameter)

### 7.3 Parameters

```
M_star        initial stellar mass
M_BH2_init    remnant BH mass at collapse (M_ejecta = M_star − M_BH2_init)
E_SN          explosion energy in code units
r_bomb        thermal bomb deposition radius (default = R_star)
v_kick        natal kick velocity vector for BH2
f_sink        sink timescale multiplier (default 1.0)
```

---

## 8. Module Map

```
src/
├── BinarySupernova.jl      module, exports, parameter struct
├── weno5.jl                WENO5-Z reconstruction            [from HighMachCBD verbatim]
├── rk3.jl                  SSP-RK3 integrator                [from HighMachCBD verbatim]
├── hllc.jl                 HLLC Riemann solver, adiabatic    [Phase 1 — new]
├── euler3d.jl              3D adiabatic solver (KA kernels)  [Phase 1 — new]
├── fmr3d.jl                3D FMR, ratio 2 or 4              [Phase 2 — new]
├── gravity_bh.jl           two moving softened BH potentials [Phase 3 — adapted]
├── nbody.jl                BH-BH + gas→BH N-body             [Phase 3 — new]
├── sinks.jl                torque-free dynamic sinks         [Phase 3 — adapted]
├── stellar_ic.jl           Lane-Emden polytrope + bomb       [Phase 4–5 — new]
├── io.jl                   HDF5 snapshots + BH trajectory    [Phase 6 — adapted]
├── diagnostics.jl          energy, AM, bound mass, BH state  [Phase 6 — new]
├── self_gravity.jl         Poisson solver (FFT / multigrid)  [Phase 7 — advanced]
├── mesa_ic.jl              MESA 1D profile reader            [Phase 8 — advanced]
└── relax_ic.jl             Roche potential relaxation IC     [Phase 9 — advanced]
```

---

## 9. Phased Development Plan

Each phase ends with: (a) all tests passing, (b) a phase document written to `docs/`,
and (c) a tagged commit pushed to GitHub. No phase is considered complete until all
three are done.

---

### Phase 0 — Bootstrap
- New Julia package `BinarySupernova`; `Project.toml` with same deps as HighMachCBD
- Copy `weno5.jl`, `rk3.jl` verbatim; confirm all copied tests pass
- Define `BlackHole` struct, `SimParams` struct, unit conversion helpers
- KernelAbstractions GPU scaffolding (identical pattern to HighMachCBD)
- Initialise GitHub repository; add `.gitignore` (HDF5 data files, figures, Julia
  artefacts); push initial commit
- **Exit**: package compiles; copied unit tests pass; GPU backend selectable at
  runtime; `docs/phase0.md` written; commit tagged `v0.0`

### Phase 1 — Adiabatic Hydrodynamics
- `hllc.jl`: HLLC flux for adiabatic Euler; Roe-averaged wave speeds
- `euler3d.jl`: 3D WENO5 + HLLC + SSP-RK3; θ-blend positivity on ρ and E
- Tests: Sod shock tube in x, y, z; 3D Sedov-Taylor vs. exact solution
- **Exit**: Sedov-Taylor shock position error < 2% at t = 0.1; all three sweep
  directions give identical results on symmetric IC; positivity maintained;
  `docs/phase1.md` with shock tube and Sedov-Taylor plots; commit tagged `v0.1`

### Phase 2 — 3D FMR (4:1 ratio)
- `fmr3d.jl`: multi-level 3D hierarchy; prolongation, restriction, flux correction,
  time subcycling; `refinement_ratio` ∈ {2, 4}
- Tests: blast wave crosses level boundary without spurious reflection; convergence
  rate maintained across interface vs. uniform fine-grid reference
- **Exit**: L1 density error at refinement boundary < 5% vs. uniform fine reference;
  `docs/phase2.md` with convergence plots and cross-boundary density profiles;
  commit tagged `v0.2`

### Phase 3 — N-body BHs + Sinks
- `gravity_bh.jl`, `nbody.jl`, `sinks.jl`
- Tests: isolated Kepler orbit (two BHs, no gas) conserves E and L < 0.1% over
  10 orbits; single BH sink mass accounting exact (Ṁ_BH = −Ṁ_gas)
- **Exit**: both tests pass; torque-free flag verified (J̇_acc = 0 to machine
  precision); `docs/phase3.md` with Kepler orbit energy drift plot and sink
  accretion rate time series; commit tagged `v0.3`

### Phase 4 — Pre-Explosion IC
- `stellar_ic.jl`: Lane-Emden solver; polytrope mapped to 3D grid; BH1 on orbit
- Tests: star + BH1 system stable for 2 P₀; stellar mass conserved < 0.1%
- **Exit**: no spurious explosion; stellar radius stable; `docs/phase4.md` with
  initial density profile plot and stability time series; commit tagged `v0.4`

### Phase 5 — Supernova Explosion
- Thermal bomb; BH2 activation with r_sink = 6 G M_BH2 / c²; natal kick
- Tests: isolated SN (no BH1) reproduces Sedov-Taylor; BH2 fallback growth measured
- **Exit**: shock position matches Sedov-Taylor < 5%; total energy conserved < 1%;
  `docs/phase5.md` with shock profile plots, M_BH2(t) fallback curve, energy budget
  time series; commit tagged `v0.5`

### Phase 6 — Production Runs + Diagnostics
- Full coupled evolution; HDF5 snapshots; BH trajectory file
- Diagnostics: a(t), M_BH(t), r_sink(t), M_bound(t), E_kin/E_th/E_grav, J_total
- Parameter study: M_BH2_init/M_star, a₀, E_SN, |v_kick|, kick direction
- **Exit**: parameter sweep at low resolution completed; binary fate classified;
  `docs/phase6.md` with BH separation, accretion history, bound mass fraction,
  and energy budget plots for each run; commit tagged `v0.6`

### Phase 7 — Gas Self-Gravity *(Advanced)*
- `self_gravity.jl`: FFT Poisson solver (uniform grids); multigrid extension for FMR
- Enable via `self_gravity = true`; off by default
- Relevant when bound fallback mass ~ M_BH
- **Exit**: Poisson solver convergence test; self-gravity vs. no-self-gravity
  comparison run; `docs/phase7.md`; commit tagged `v0.7`

### Phase 8 — MESA Stellar Model *(Advanced)*
- `mesa_ic.jl`: read MESA 1D profile; map to 3D; derive local effective γ(r)
- Match code units to MESA physical units at import
- **Exit**: MESA star stable for 1 orbit; density profile comparison (MESA vs.
  polytrope); `docs/phase8.md`; commit tagged `v0.8`

### Phase 9 — Roche Potential Relaxation *(Advanced)*
- `relax_ic.jl`: velocity damping for 1–2 P₀ in full binary potential before explosion
- Accounts for tidal distortion and Roche lobe shape
- **Exit**: kinetic energy < 1% thermal at end of relaxation; stable for 0.5 P₀ after
  damping removed; `docs/phase9.md` with kinetic/thermal energy relaxation curves and
  density slice before/after; commit tagged `v0.9`

---

## 10. Documentation Standards

### Phase documents (`docs/phaseN.md`)

Each phase document is written at phase completion and covers:

1. **Objective** — one paragraph stating what the phase implements and why
2. **Implementation notes** — key design choices made during coding; anything that
   deviated from the plan and why
3. **Test results** — for every test in the phase exit criteria:
   - Quantitative result (error value, convergence rate, conservation violation)
   - Pass/fail against the stated criterion
   - Plot (saved to `docs/figures/phaseN_*.png`) embedded as a Markdown image link
4. **Known limitations** — anything deferred to a later phase or noted as approximate
5. **Next steps** — brief pointer to what Phase N+1 will build on

### Figures

All figures generated by Julia scripts in `scripts/` (not by test code directly):

```
scripts/
├── plot_phase1.jl     Sod shock, Sedov-Taylor
├── plot_phase2.jl     FMR convergence, cross-boundary profiles
├── plot_phase3.jl     Kepler orbit drift, sink accretion rate
├── ...
docs/figures/
├── phase1_sod_shock.png
├── phase1_sedov_taylor.png
├── phase2_convergence.png
├── ...
```

Figures use CairoMakie (same as HighMachCBD). Each script reads HDF5 output produced
by the corresponding test and regenerates all figures from scratch. Figures are
committed to the repository so the docs render correctly on GitHub.

### Inline comments

- Comment every non-obvious numerical formula with the equation number or reference
  it implements (e.g. `# Dempsey+ 2020 eq. 6`)
- Comment every physical constant or parameter with its units in code units
- Do not comment self-evident Julia syntax

---

## 11. Version Control and GitHub Workflow

### Repository layout

```
Binary-Supernova/
├── .gitignore          HDF5 data (*.h5), Julia depot artefacts, OS files
├── CLAUDE.md           this document
├── Project.toml
├── Manifest.toml
├── src/                source modules
├── test/               test suite (Julia Test.jl)
├── scripts/            plotting and analysis scripts
└── docs/
    ├── phaseN.md       one file per phase, written at completion
    └── figures/        PNG figures committed alongside docs
```

### `.gitignore`

```
*.h5
*.hdf5
data/
output/
.DS_Store
*.jl.cov
*.jl.mem
/Manifest.toml        # omit if pinning deps; include if reproducing exact env
```

### Commit conventions

- Commit message format: `<phase>: <short description>`
  - Examples: `phase1: add HLLC solver and Sedov-Taylor test`
  - Examples: `phase3: torque-free sink passes J_dot=0 check`
- Every commit that touches `src/` must have passing tests (`julia --project test/`)
- Figures and docs are committed together with the code that produced them, in the
  same commit or in an immediately following one on the same branch
- No force-pushing to `main`

### Branching

- `main`: stable, tagged releases only; always corresponds to a completed phase
- `dev`: active development; merged to `main` at each phase completion
- Feature branches (e.g. `feature/hllc`, `feature/fmr4x`) for work in progress;
  merged to `dev` when tests pass

### Tags

Each phase completion is tagged on `main`:

```
v0.0   Phase 0 complete
v0.1   Phase 1 complete (adiabatic hydro)
v0.2   Phase 2 complete (FMR)
...
```

Tags allow reproducible reproduction of results in each phase document.

---

## 12. Key Design Decisions

1. **WENO5-Z + SSP-RK3 + HLLC**: proven combination for shocked flows; retains
   HighMachCBD's high-order accuracy and shock-capturing properties.

2. **3D from Phase 1**: 2D can be used for early unit tests but the production solver
   is 3D. The SN ejecta geometry and natal kick direction require 3D for correct
   dynamics.

3. **Single fine FMR zone**: simpler than per-BH zones; avoids interpolation artefacts
   when BHs are close; resizing between restarts handles large separations.

4. **4:1 refinement ratio**: reduces number of levels (3 vs. 6 for same total
   refinement factor) and total cell count; verified feasible with WENO5's 3-ghost-cell
   stencil (12 fine cells available vs. 3 needed).

5. **r_sink = 6 G M_BH / c²** (ISCO): physically motivated, mass-dependent, grows
   with fallback accretion. Floor at ~2 Δx_fine is the operative numerical limit;
   the formula ensures correct sub-grid scaling.

6. **Torque-free sinks**: gas angular momentum budget correctly tracked; BH receives
   full accreted momentum for N-body accuracy. Generalises Dempsey+ 2020 to 3D
   adiabatic with moving BHs.

7. **N-body in lockstep with SSP-RK3**: BH positions and velocities updated at each
   RK sub-stage using the same Butcher weights; avoids operator-splitting errors in
   the BH-BH and BH-gas gravitational coupling.

8. **GPU portability**: all compute kernels written with KernelAbstractions.jl;
   CPU (dev/test) and CUDA (production, A100) selected at runtime via backend
   argument, identical to HighMachCBD.

9. **Docs + plots at every phase**: phase documents and figures are committed with
   the code. This ensures that results are reproducible and progress is legible
   from the git history alone.

---

## 13. Pitfalls and Notes

- **c in code units is large**: r_sink = 6 G M_BH / c² ≪ Δx for any realistic
  stellar binary. The floor r_floor = 2 Δx_fine is always operative. The ISCO formula
  is retained for physical correctness and correct scaling; do not confuse the floor
  with a physical radius.

- **γ choice**: γ = 4/3 is appropriate for the radiation-dominated star interior.
  The ejecta at late times may be better described by γ = 5/3. A spatially varying
  or time-switched γ can be considered in advanced phases.

- **Softening ε**: BH gravitational softening must be small compared to r_sink_floor
  but large enough that the force gradient does not require a timestep smaller than
  the CFL limit. Typical: ε ~ 0.5 r_sink_floor.

- **Energy conservation check**: total energy E_gas + E_kin_BH + E_grav_BH-BH should
  be conserved to < 1% over the simulation. Violations indicate flux correction errors
  at FMR boundaries or sink accounting errors.

- **Torque-free residual**: total angular momentum J_gas + J_BH1 + J_BH2 should be
  conserved. The torque-free sink sets J̇_acc = 0; any drift comes from boundary
  fluxes or BH-gas gravity integration errors.

- **Data management**: HDF5 snapshot files are excluded from git (they are large).
  The scripts in `scripts/` must be self-contained and reproducible given the HDF5
  output; document the run command needed to regenerate data in each `docs/phaseN.md`.

---

## 14. References

- Dempsey, Munoz & Lithwick 2020, arXiv:2002.05164 — torque-free sink prescription
- Duffell et al. 2024, arXiv:2402.13039 — Santa Barbara binary-disk code comparison
- Borges et al. 2008 — WENO-Z reconstruction
- Shu & Osher 1988 — SSP-RK3 integrator
- Toro 1999 — HLLC Riemann solver
- Berger & Colella 1989 — AMR flux correction
- Hu, Adams & Shu 2013 — positivity-preserving limiter
- Lane-Emden (polytrope): Chandrasekhar 1939, §4
