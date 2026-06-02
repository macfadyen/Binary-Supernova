# Paper outline + figure plan — "No circumbinary disc from supernova fallback"

**Status:** scaffold, written before the confirming grid (`docs/cbd_grid_runbook.md`)
completes. The results sections and the headline scope are **contingent on the grid
landing null**; if any run forms a disc, do not write this paper (see the runbook
decision gate). Fill in numbers/figures as runs land.

**Working title.** *Supernova fallback cannot assemble a circumbinary disc around a
black-hole binary: an angular-momentum ceiling and its implication for
electromagnetic counterparts to gravitational-wave sources.*

**Target.** ApJ (full paper: theorem + methods + grid). A tightly-scoped ApJL
(theorem + kick + one run, headlined on the AGN implication) is the fallback if we
want it out fast — but we chose the survey, so ApJ.

---

## Contingent abstract (≈150 words, draft)

Whether a stellar-mass black-hole binary can be accompanied by gas — and so produce
an electromagnetic counterpart to its gravitational-wave signal — depends on its
formation channel. We ask whether the binary's *own* formation supernova can endow it
with a circumbinary disc (CBD), closing the last endogenous-gas loophole for the
isolated-binary channel. We show analytically that the disc-forming angular momentum
available to supernova fallback is bounded: spin-fed fallback circularises at a radius
of order the stellar radius — interior to the orbit for any detached binary — so
stellar rotation provably cannot make a CBD, independent of explosion geometry. The
only reservoir that can exceed this ceiling is the binary's orbital angular momentum,
accessible to a tidally-distorted (Roche) progenitor. Using 3D adiabatic
hydrodynamics with live black-hole N-body and torque-free sinks, we simulate a grid
spanning the CBD-favorable corner (deep Roche overflow, weak explosions, long
integrations, a resolution check) and find **no** circumbinary disc in any case: the
bound fallback is a sub-Keplerian, turbulent cloud that **hardens** the surviving
eccentric binary rather than forming a disc. A natal kick cannot rescue the disc
either — it lands the system in L2-spill or a black-hole merger. We conclude the
isolated channel is electromagnetically dark from the supernova onward, and that an
EM counterpart requiring circumbinary gas favors an externally gas-supplied (e.g.
AGN-disc) environment.

---

## The certain / caveat ledger (the spine of the claim)

State this explicitly in the paper (Methods or a dedicated "Scope" subsection) so the
negative result is honest about what is proven vs. simulated vs. open.

| Leg | Strength | What it establishes |
|-----|----------|---------------------|
| **Spin channel** | **Certain (analytic).** `r_circ ≤ (M★/M_BH2)·R★`; for our masses < a for all a₀ > 1.03 R☉ (= contact). Explosion-geometry-independent. | Stellar rotation cannot make a CBD in any detached binary. |
| **Kick lever** | **Certain (analytic).** Blaauw + kick: a_post ≥ a₀/2, and only at r_peri→0 (merger). Cavity edge never reaches r_circ. | A natal kick cannot tighten the orbit into a CBD; it yields L2-spill or a BH–BH merger. |
| **Orbital channel** | **Strong (numerical, this grid).** Roche-distorted progenitor injects orbital-scale AM, yet G0–G5 all stay sub-Keplerian (ℓ/ℓ_kep < 1), no disc. | Even the only reservoir that *can* beat the spin ceiling fails to assemble a CBD across the favorable corner. |
| **Contact / CE regime** (a₀ ≲ 2 R☉, star overfilling) | **Open — out of scope.** Not a detached binary; a mass-transfer/CE problem. | Explicitly excluded; flagged as a distinct (and itself gas-producing) channel. |
| **Pumping over ≫10 P₀** | **Open — argued, not run.** Our gas is sub-Keplerian and *draining*, so there is no reservoir to viscously spread; but we integrate only ~3–8 orbits. | Addressed by the G3 long-integration run; note the AM-flux argument. |
| **Aspherical / jet explosions** | **Open — argued, not simulated.** The spin ceiling is geometry-independent; aspherical injection changes *how much* spin AM is bound, not the ceiling. | Stated as a consistency argument; not directly tested. |
| **Microphysics** | **Caveat.** Newtonian, adiabatic, no neutrino/radiation transport; single resolution for the hardening *rate* (null is resolution-robust per G5). | Standard simulation caveats; the dynamical AM argument is insensitive to them. |

---

## Section outline

**1. Introduction.**
- BBH mergers as GW sources; the EM-counterpart question and why the formation
  channel is the crux. Isolated/field vs. AGN/dynamical (gas-rich) channels.
- Circumbinary gas: CBD-driven inspiral and EM counterparts (GW190521 / ZTF19abanrhr).
- The endogenous-gas loophole: *can the binary's own (second) SN leave a CBD?* — the
  one place the isolated channel might still produce gas. Nobody has simulated it.
- This paper: an AM ceiling (analytic) + a confirming simulation grid → no; the
  implication for counterparts.

**2. The angular-momentum argument.**
- 2.1 Setup: SN in a close BH-binary; the two — and only two — AM reservoirs for
  bound fallback (stellar spin; binary orbital AM).
- 2.2 **Spin-cap theorem.** r_circ = Ω_spin²R★⁴/M_BH2; breakup ceiling
  r_circ|_brk = (M★/M_BH2)R★ < a for any detached binary; explosion-geometry-independent.
  (From `predict_disk.jl`; cite the equatorial-belt upper bound.)
- 2.3 The orbital channel: a Roche-distorted progenitor co-orbits its outer envelope,
  the only way to exceed the spin ceiling — hence a *simulation* question.
- 2.4 **The kick lever, excluded.** Post-SN orbit from Blaauw mass-loss + natal kick;
  a_post ≥ a₀/2 only at r_peri→0; cavity edge never reaches r_circ → L2-spill or merger.

**3. Methods.**
- 3.1 Code: 3D adiabatic Euler, WENO5-Z + HLLC + SSP-RK3, 3D FMR (4:1), KernelAbstractions
  GPU (BinarySupernova; HighMachCBD lineage). Code units (G=1, M_tot=a₀=1).
- 3.2 Live BH N-body + torque-free dynamic sinks (3D adiabatic generalization of
  Dempsey, Muñoz & Lithwick 2020); BH receives full accreted momentum.
- 3.3 The Roche progenitor: co-rotating-frame relaxation IC (centrifugal+Coriolis;
  KE-minimum auto-stop); self-gravity on. Contrast with the symmetric SCF figure.
- 3.4 Explosion: mass-based thermal bomb (inner half by mass), BH2 activation with
  sink-delay, natal kick.
- 3.5 The CBD diagnostic: azimuthal ⟨v_φ⟩(R)/v_K(R) and ℓ_bound/ℓ_kep; the gate.
- 3.6 The confirming grid (G0–G5) and the `predict_disk.jl` AM pre-flight that
  selected the CBD-favorable corner.

**4. Results.**
- 4.1 Baseline (G0): the sub-Keplerian cloud — density + ⟨v_φ⟩/v_K at three phases.
- 4.2 The grid: null across deep overflow (G2), weak SN / bound mass (G1), long
  integration (G3), combined worst case (G4), and resolution (G5). **The money plot.**
- 4.3 AM budget: ℓ_bound/ℓ_kep < 1 at all R and t across the grid; M_bound peaks then
  drains (accreted/ejected, not retained).
- 4.4 The positive result: **gas-assisted hardening** — surviving eccentric binary,
  apastron decay via drag from the bound fallback.

**5. Discussion / implications.**
- 5.1 The isolated channel is **EM-dark from the SN onward**: no endogenous CBD, and
  the transient fallback is accreted/ejected within a few orbits — Myr–Gyr before
  merger. (CE/wind gas from earlier phases also disperses long before merger.)
- 5.2 Therefore a robust EM counterpart requiring circumbinary gas **favors an
  externally gas-supplied environment** — most naturally an AGN accretion disc.
  GW190521/ZTF context; McKernan & Ford, Bartos+2017, Stone+2017, Tagawa+2020.
- 5.3 Honest alternatives & caveats: gaseous tertiary, still-embedded cluster gas,
  the Graham+2020 association debate; "favors," not "proves."
- 5.4 Gas-hardening as a distinct, separately-interesting channel result (cleaner GW
  progenitor story than a CBD).

**6. Caveats & scope.** The ledger above, in prose.

**7. Conclusions.**

---

## Figure plan

| Fig | Content | Source | Status |
|-----|---------|--------|--------|
| **1** | The AM ceiling: r_circ(a₀) (spin + breakup) vs. cavity bar 2a₀ and the contact floor; shaded "no-CBD" region. The theorem in one plot. | `predict_disk.jl` numbers → **new** `scripts/plot_cbd_theory.jl` | to write |
| **2** | The Roche progenitor: density slice of the relaxed co-rotating star (L1/L2 bulge) vs. the symmetric SCF figure. Methods. | G0 IC snapshot via `plot_sn50_sideview.jl`/`plot_sn50_snap.jl` | to render |
| **3** | Baseline G0: density + ⟨v_φ⟩/v_K at t = 7 / 14 / 20. | `cbd_campaign_t07/t14/t20_*.png` (exist) | done (G0) |
| **4** | **Money plot:** max ⟨v_φ⟩/v_K beyond 2·r_sep for G0–G5 vs. the lever pushed, all far below the CBD threshold. | **new** `scripts/plot_cbd_grid.jl` | to write (needs grid) |
| **5** | AM budget: ℓ_bound/ℓ_kep(t) and M_bound(t) overlaid for the grid. | extend `cbd_campaign_am.png` / `plot_sn50_am.jl` | to extend |
| **6** | Gas-assisted hardening: r_sep / apastron decay, eccentricity (the positive result). | `cbd_campaign_bh_history.png` (exists) | done (G0) |
| **7** | The kick lever excluded: a_post, r_peri, r_circ/cavity vs. anti-orbital kick. | `cbd_campaign_kick.png` (exists) | done |

Two new plotting scripts (Fig 1 theory curve, Fig 4 grid money-plot); Fig 5 extends
the existing AM plot to overlay the grid. All committed to the repo per the project
figure convention.

---

## Key references (science-critical; numerics refs per CLAUDE.md §14)

- Graham et al. 2020 — GW190521 candidate EM counterpart (ZTF19abanrhr).
- McKernan & Ford et al.; Bartos et al. 2017; Stone, Metzger & Haiman 2017;
  Tagawa, Haiman & Kocsis 2020 — the AGN-disc BBH channel.
- Artymowicz & Lubow 1994 — tidal cavity (r_cav ≈ 2a).
- Muñoz, Miranda & Lai 2019; Muñoz et al. 2020; Duffell et al. 2024;
  Moody, Shi & Stone 2019; Dempsey, Muñoz & Lithwick 2020 — CBD torques & sinks.
- Blaauw 1961 — SN mass-loss orbital change.
- Hachisu 1986 — SCF rotating-polytrope method; Chandrasekhar 1939 — Lane-Emden.

---

## Open decisions / TODO before submission

- Confirm the grid lands null (the runbook decision gate) — **blocking**.
- Decide whether the G3 long run needs extending to ≫10 P₀, or whether the
  draining-reservoir + AM-flux argument suffices for the pumping caveat.
- Write `plot_cbd_theory.jl` (Fig 1) and `plot_cbd_grid.jl` (Fig 4).
- Lock the physical scaling: state the fiducial in M☉/R☉ and the GW150914/GW190521
  context (this grid uses 15/30/20; note generality via code units).
- Author list, journal (ApJ vs. ApJL), and how hard to push the AGN framing in the title.
