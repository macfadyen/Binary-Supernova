# CBD negative-result confirming grid — cabeus runbook

**Goal.** The single A100 run `G0` (`docs/cbd_campaign.md`) found that SN fallback
from a Roche-distorted progenitor makes **no circumbinary disc** — only a
sub-Keplerian, binary-hardening cloud. This grid confirms that the null result is
robust across the *CBD-favorable* corner of parameter space, so the negative-result
paper (`docs/cbd_paper_outline.md`) can claim it with a stated scope rather than
from one point.

The spin-cap theorem already covers everything *outside* this corner analytically:
for the 15/30/20 masses, spin-fed fallback is provably sub-CBD for any
a₀ > 1.03 R☉ (= a contact configuration). The only door the theorem leaves open is
the **Roche-overflow regime** (star over its own lobe below a₀ = 2.27 R☉), where the
relax-IC injects *orbital*-scale AM the spin model does not see. The grid pushes the
three levers that are simulation-only **and** CBD-favorable: deeper overflow (↓a₀),
more bound mass (↓E_SN), longer integration (↑t_end), plus a resolution check.

> **This is a cabeus/A100 runbook — it cannot be validated on the dev Mac**
> (NX≥256 at L=6 is multi-day on CPU; prior long CPU runs were externally killed).
> Run it on cabeus exactly as below; record the gate outcomes in §5.

---

## 0. Preconditions (cabeus)

Same environment as the G0 run (see `scripts/run_relax_optimal.pbs` header and memory
`project_predict_disk.md`, "Cabeus run path" paragraph):

```bash
# on cabeus, from /nobackup/$USER/Binary-Supernova
git pull origin main
module purge && module load gcc/13.2        # gcc ONLY — NOT nvhpc (CUDA.jl artifact bug)
export JULIA_DEPOT_PATH=/nobackup/$USER/.julia:
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=. -e 'using BinarySupernova; import CUDA; @assert CUDA.functional()'
```

Queue: `gpu_normal` (24 h cap), group `s1994`, `model=mil_a100` (A100-80GB). The
checkpoint / auto-resubmit chain in the PBS script carries any run that exceeds the
24 h cap across as many jobs as needed.

---

## 1. The grid

All runs: masses **M_BH1=15, M★=30, M_BH2,init=20 M☉** (ΔM/M_preSN = 0.22, Blaauw-safe),
relax-IC (co-rotating Roche relaxation), L=6, self-gravity on, torque-free sinks,
`--bomb-mass-frac 0.5` spherical, `--v-kick-y -0.015`. Only the four grid axes change.

| Run | a₀ [R☉] | E_SN frac | NX | t_end | Lever / objection it closes | est. wall* |
|-----|---------|-----------|-----|-------|-----------------------------|-----------|
| **G0** ✓ | 2.5 | 0.05 | 256 | 20 | baseline (done — the decisive run) | 5.9 h |
| **G1** | 2.5 | **0.02** | 256 | 20 | "a weaker SN leaves more bound mass → disc" | ~6 h |
| **G2** | **2.0** | 0.05 | 256 | 20 | "deeper Roche overflow → orbital AM clears the bar" | ~6 h |
| **G3** | 2.5 | 0.05 | 256 | **50** | "integrate longer → binary torques pump a CBD" | ~7–8 h** |
| **G4** | **2.0** | **0.02** | 256 | 30 | combined worst case (overflow + weak SN + time) | ~8–9 h |
| **G5** | 2.5 | 0.05 | **384** | 20 | "the null is a resolution artifact" + pins hardening rate | ~30 h (chains 2 jobs) |

\* Extrapolated from G0 (5.9 h). **Confirm with a devel probe before the full run**
(§2), especially G5 (NX=384, unprobed) and G2/G4 (new a₀ — the deeper-overflow
relaxation may disperse differently; the KE-minimum auto-stop should still catch it).
\*\* G3 as a *continuation* of G0's checkpoint (§3); a fresh t=50 run is ~13 h.

Total ≈ 55–60 A100-h, the bulk in the NX=384 check. If A100 time is tight, NX=320
(~15 h) is an acceptable cheaper resolution check than NX=384.

> **One-command driver.** `scripts/launch_cbd_grid.sh` wraps the probe→launch
> sequence below, with the G3 checkpoint-continuation check baked in:
> `scripts/launch_cbd_grid.sh probe`, then `scripts/launch_cbd_grid.sh launch`
> (or a subset, e.g. `… launch G2 G5`). Add `--dry-run` to preview the exact `qsub`
> lines without submitting, or `… status` for `qstat` + per-run output-dir state.
> The explicit `qsub` lines it issues are documented in §2–§3 for reference.

---

## 2. Probe wall first (devel queue, ~2 min of sim)

```bash
# size the wall + confirm the GPU path for each NEW config before committing hours
qsub -q devel -l walltime=00:45:00 \
     -v T_END=2,DT_SNAP=2,WALL_BUDGET_MIN=40,NX=256,A0_RSUN=2.0,E_SN_FRAC=0.05 \
     scripts/run_relax_optimal.pbs        # G2-shape probe
qsub -q devel -l walltime=00:45:00 \
     -v T_END=2,DT_SNAP=2,WALL_BUDGET_MIN=40,NX=384,A0_RSUN=2.5,E_SN_FRAC=0.05 \
     scripts/run_relax_optimal.pbs        # G5-shape probe (NX=384)
```

Read `Mcell/s` + `wall_sec` from the log tail; extrapolate to the target `t_end`.
If the extrapolation exceeds 24 h, the run will auto-chain — no action needed.

---

## 3. Launch the grid

```bash
cd /nobackup/$USER/Binary-Supernova

# G1 — weaker SN, more bound mass
qsub -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.02,T_END=20 scripts/run_relax_optimal.pbs

# G2 — deeper Roche overflow
qsub -v NX=256,A0_RSUN=2.0,E_SN_FRAC=0.05,T_END=20 scripts/run_relax_optimal.pbs

# G4 — combined worst case
qsub -v NX=256,A0_RSUN=2.0,E_SN_FRAC=0.02,T_END=30 scripts/run_relax_optimal.pbs

# G5 — resolution check (auto-chains across the 24 h cap)
qsub -v NX=384,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=20 scripts/run_relax_optimal.pbs

# G3 — long integration.  PREFERRED: continue G0's checkpoint to t=50 by reusing
# its OUTDIR (the PBS script auto-detects chkpt.h5 and resumes; ~7-8 h).  Verify
# the checkpoint exists first:
G0=demo1/output_relax_optimal_a25rsun_L6_bh15_20_esn005_bombmf05_kickm0.015_nx256
ls -la "$G0/chkpt.h5" && \
qsub -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=50,OUTDIR="$G0" scripts/run_relax_optimal.pbs
#   If $G0/chkpt.h5 is absent (cleaned after G0's rc=0), run G3 fresh into its own dir:
#   qsub -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=50,OUTDIR=demo1/output_G3_t50_nx256 \
#        scripts/run_relax_optimal.pbs
```

Output dirs (auto-named by the PBS `TAG`; `--gpu` run writes `diagnostics.csv`,
`bh_history.csv`, snapshots, periodic `chkpt.h5`):

| Run | OUTDIR |
|-----|--------|
| G0 | `demo1/output_relax_optimal_a25rsun_L6_bh15_20_esn005_bombmf05_kickm0.015_nx256` |
| G1 | `demo1/output_relax_optimal_a25rsun_L6_bh15_20_esn002_bombmf05_kickm0.015_nx256` |
| G2 | `demo1/output_relax_optimal_a20rsun_L6_bh15_20_esn005_bombmf05_kickm0.015_nx256` |
| G3 | `= G0` (continuation) or `demo1/output_G3_t50_nx256` (fresh) |
| G4 | `demo1/output_relax_optimal_a20rsun_L6_bh15_20_esn002_bombmf05_kickm0.015_nx256` |
| G5 | `demo1/output_relax_optimal_a25rsun_L6_bh15_20_esn005_bombmf05_kickm0.015_nx384` |

The PBS script runs `plot_sn50_cbd.jl` automatically on clean (rc=0) completion.

---

## 4. Analysis (per completed run)

```bash
D=<OUTDIR from the table above>
A0=<2.5 or 2.0>
julia --project=. scripts/plot_sn50_cbd.jl        --outdir "$D" --a0-rsun "$A0"
julia --project=. scripts/plot_sn50_bh_history.jl --outdir "$D" --a0-rsun "$A0" --mtot-msun 45
julia --project=. scripts/plot_sn50_am.jl         --outdir "$D" --a0-rsun "$A0" --mtot-msun 45
```

(`--mtot-msun 45` = M_BH1 + M★ = the code-unit total mass, as in `docs/cbd_campaign.md`.)

### Pass/fail gate — the CBD criterion

A run forms a **circumbinary disc** iff, in an annulus *beyond* the tidal cavity
(R ≳ 2·r_sep), it shows **all** of:

1. **Rotational support:** azimuthally-averaged ⟨v_φ⟩(R) / v_K(R) ≳ 0.8, where
   v_K = √(M_bin / R)  (the blue-vs-red-dashed panel of `plot_sn50_cbd.jl`);
2. **Centrifugal AM:** ℓ_bound / ℓ_kep ≥ 1  (`plot_sn50_am.jl`);
3. **Sustained:** conditions 1–2 hold for ≳ 1 orbit and the annulus is **not**
   draining to the density floor.

Anything else is the **null** (sub-Keplerian, pressure/turbulence-supported,
draining cloud — e.g. G0: ⟨v_φ⟩/v_K ≈ 0.2–0.4, ℓ/ℓ_kep < 1 at all R).

### Decision gate

- **All five null →** the negative result holds across the favorable corner.
  Proceed to the paper (`docs/cbd_paper_outline.md`); G0+grid become the
  results-section figures, with the grid as the "money plot" (every run far below
  the CBD line).
- **Any run passes (sustained supported annulus) →** a genuine CBD detection in
  that corner. Do **not** publish the null. Investigate: re-run at higher NX to
  test convergence; check it is not a relax-IC artifact (dispersed star) per the
  G0-development notes in `project_predict_disk.md`.

---

## 5. Results ledger (fill in on cabeus)

| Run | status | M_bound peak [M☉] | max ⟨v_φ⟩/v_K beyond 2·r_sep | max ℓ_bound/ℓ_kep | gas mass R>2·r_sep trend | gate |
|-----|--------|-------------------|------------------------------|-------------------|--------------------------|------|
| G0 | done | ~19 | ~0.2–0.4 | < 1 | drains 19→5 M☉ | **NULL** |
| G1 |  |  |  |  |  |  |
| G2 |  |  |  |  |  |  |
| G3 |  |  |  |  |  |  |
| G4 |  |  |  |  |  |  |
| G5 |  |  |  |  |  |  |

Also record per run: final r_sep range / eccentricity / apastron-decay slope
(the gas-hardening positive result), ΔM_BH1, ΔM_BH2, and walltime / # chained jobs.
