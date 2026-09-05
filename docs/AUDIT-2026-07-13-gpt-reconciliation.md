# Reconciliation: prior fan-out audit vs. external GPT audit — qpsim @ 7e72b52

**Method.** An external GPT audit filed ~20 findings (5 rated High). Each was
independently re-verified by a dedicated Fable-5 agent that had to **reproduce**
GPT's specific numbers against the actual code at `7e72b52` (running the venv
Python), classify confirmed / partially-confirmed / refuted / already-documented,
and assign a realistic severity. A final agent reconciled the two audits. 21
agents, 0 errors, ~834k tokens.

## Bottom line

Of GPT's 20 findings, **19 are real issues on current code** (all reproduced);
**1 (G5) is refuted** as a factual misread. After severity adjustment: **4
medium + 15 low, zero high/blocker, zero corrupted published/validated results,
zero default-path crashes.** By category (real issues): numerics-correctness 3
(G1, G2, G8); robustness/DoS 3 (G3, G16, G17); validation-integrity 9 (G4, G6,
G7, G9, G12, G13, G14, G15, G18); api-contract 2 (G10, G11); security 2 (G19,
G20).

**The prior "essentially clean" verdict does not hold as stated.** It was
directionally correct on the thing it measured — no blocker-class physics error,
no corrupted figure, core detailed-balance and kinetics are sound — but it
materially undercounted: it confirmed 2 low/latent findings where 19 real ones
exist, including **four genuine medium defects**. Honest framing: *no smoking-gun
physics bug, but the repo is not "clean" — it has a consistent pattern of real
robustness / validation-honesty gaps the prior methodology was structurally
blind to.*

GPT also **overstated several severities** and got one diagnosis backwards
(details in §4): High→low for G6/G7/G19, and in G6 it named the wrong
`coupled_newton` fallback as the offender.

## Findings table

| ID | Short title | GPT sev | Verdict | Real? | Adj sev |
|----|-------------|---------|---------|-------|---------|
| G1 | Collision midpoint dE vs observable exact BCS cell weights | High | confirmed | yes | **medium** |
| G2 | Nonequilibrium gap solver assumes unique monotone root | High | confirmed | yes | **medium** |
| G3 | Snapshot cadence: unbounded work before cancel (DoS) | High | confirmed | yes | **medium** |
| G4 | Fischer Fig.3 numerical pin vacuous for every curve | High | partially-confirmed | yes | **medium** |
| G6 | Newton fallback bypasses relative/balance certificate | High | partially-confirmed | yes | low |
| G7 | Nbar loop certifies convergence at pre-advance state | High | partially-confirmed | yes | low |
| G8 | Near-Tc gap discontinuity (coupling derivation) | Medium | partially-confirmed | yes | low |
| G9 | Diffusion apply_collisions: NaN/nonpositive dt + gap mismatch accepted | Medium | partially-confirmed | yes | low |
| G10 | Zero-active-bin shortcut: NaN passthrough + KeyError | Medium | partially-confirmed | yes | low |
| G11 | Spectral caches alias caller arrays / writable | Medium | partially-confirmed | yes | low |
| G12 | High-energy tail silently zero-filled, no warning | Medium | partially-confirmed | yes | low |
| G13 | Effective-temperature fit bound reported as measurement | Medium | confirmed | yes | low |
| G14 | Overnight resume: run-id omits grid/dt; nonconverged=completed | Medium | confirmed | yes | low |
| G15 | Fig.6 marked complete without current numerical evidence | Medium | confirmed | yes | low |
| G16 | Cancel during 0-D steady / final M25 point ignored, persisted done | Medium | confirmed | yes | low |
| G17 | Concurrent same-slug saves race on one fixed .tmp path | Medium | partially-confirmed | yes | low |
| G18 | Below-threshold M25 drive validates OK, returns all-NaN | Medium | partially-confirmed | yes | low |
| G19 | Non-loopback bind exposes unauthenticated read/write/delete/compute | Cond-High | partially-confirmed | yes | low |
| G20 | load_material path escapes supplied database dir | Low | partially-confirmed | yes | low |
| G5 | "8 Fischer reproductions complete" overstates F24 fig5/fig8 | High | **refuted** | no | none |

## Confirmed real issues, by adjusted severity

### Medium (4)

**G1 — Collision operators use midpoint ρ·dE; observables use exact singular BCS
cell weights** (`collisions/phonon.py:618-620,637-638` vs
`observables/density.py:46-47` / `physics/bcs_quadrature.py:143-145`). Two
inconsistent discrete measures for the same singular DOS in adjacent paths.
Observed at default NE=400: low-T x_qp differs **14.77% at 0.15 K** (12.68% at
0.2 K, up to 25% at 0.05 K), converging only ~1/√NE
(20.5/14.8/10.4/7.3/5.1% at NE=200/400/800/1620/3200). Decisive: scattering
conserves Σρf·dE to machine zero (rel 5e-17) but the *reported* exact-weight
number drifts **rel 1.1e-2** under the same number-conserving relaxation.
`diffusion` even mixes the two measures within one backend. *Fix:* route
collision dE-contractions and observables through a single cell-weight vector on
`SpectralContext` (exact singular weights for pure-BCS, `ρ·dE` for Dynes).

**G2 — Nonequilibrium gap solver assumes a unique monotone residual**
(`physics/gap_equation.py:230-231` comment). Monotonicity holds only for
equilibrium f. Observed: an accepted gap-edge-bump 0≤f≤1 produces **three
residual sign changes** (roots ~[173.66, 176.55, 182.41] µeV); `solve_gap`
returns 173.66 at `bracket_factor=0.05` but 182.41 at other factors — a physical
branch selected by a numerical knob. (The separate "returns 0 on same-sign
endpoints" sub-claim did not reproduce.) *Fix:* global sign-change scan over the
physical interval + pick the root continuous with the previous iterate (or warn
on multiplicity); correct the false comment.

**G3 — Snapshot cadence permits unbounded work before cancellation is polled**
(`webui/schemas.py:173,226`; `services/transient.py:207-217`). `snapshot_interval`
has only `gt=0`; the inner cadence loop drains fully before the cancel hook runs.
Observed: interval=1e-5 → **100,002 snapshots**; a cancel after step 1 still
emitted 10,002 snapshots from that single step; interval=1e-300 is accepted and
is an uninterruptible, memory-growing hang. Single-worker runner → blocks the
queue. *Fix:* schema lower bound / max-snapshot cap, and poll cancellation
*inside* the inner while loop.

**G4 — Fischer Fig.3 numerical pin is vacuous for every curve, not just ratio-10**
(`validation/fischer_2023/test_fig3_paper.py:93-100`). `atol=1e-6, rtol=0` on
curves of magnitude ~1e-8. Observed: all four Fig.3 curves (max|f| 3.6e-10 …
1.5e-8) pass even if replaced by zero; Fig.5 15/81 and Fig.6 34/66 x_qp points
also vacuously covered. Broader than the single all-zero column the brief
documented. (Fig.5/6 are only *partially* vacuous — their large-value pins are
tight, so "Fig.5/6 similarly weak" overstates Fig.6.) *Fix:* signal-scaled
tolerance, e.g. `atol = max(1e-14, 1e-6·max|curve|)` or an rtol.

### Low (15)

- **G6 — Newton false convergence.** `newton_steady_state.py:214` not-accepted
  fallback uses an absolute-only gate; observed a saturated **f=1 returned as
  converged** for an infeasible root — but only in a sub-1e-14-rate regime
  unreachable through real phonon physics (normal rates correctly raise). GPT's
  `coupled_newton` half is **inverted**: line 427 *does* enforce the balance
  certificate. *Fix:* apply the relative certificate in the
  `newton_steady_state` fallback.
- **G7 — Nbar loop certifies the pre-advance state** (`services/nbar_loop.py:244-266`).
  `converged` certifies the point before the final re-solve. Observed:
  near-discontinuous injected map returns converged with **residual 0.999990**;
  every smooth physical map returns O(tol) (1.4e-7). *Fix:* re-check residual
  after the final re-solve.
- **G8 — Near-Tc gap discontinuity** (`physics/gap_equation.py:115`). Coupling
  anchored to δ₀=1.764·kTc at T=0 makes effective Tc ≠ declared Tc. Observed: Δ
  plateaus at **3.847 µeV** as T→Tc⁻ then hard-clamps to 0; only within ~0.08% of
  Tc, irrelevant at operating T≪Tc. Distinct from the documented "solve_gap
  biased near Tc" warning (this is in `calibrate_gap`, no grid, no warning).
  *Fix:* derive 1/λ from the linearized gap equation at the declared Tc.
- **G9 — Diffusion apply_collisions missing guards** (`backends/diffusion.py:866`).
  Observed: dt=NaN → all-NaN output, no raise; `state.gap=999` vs
  `spectral.gap=180` silently ignored. Only reachable by directly invoking the
  backend method (the supported driver validates dt). *Fix:* mirror the sibling
  backend's dt/gap guards.
- **G10 — Zero-active-bin shortcut** (`services/steady_state.py:210-212`).
  Observed: returns NaN `initial_guess` verbatim; Picard path raises **KeyError
  'n_ph'** downstream. Requires a pathological grid where every above-gap bin is
  within the active margin. *Fix:* honor the `phonon_out` contract or raise a
  clear error.
- **G11 — Mutable/aliased spectral caches** (`physics/spectral.py:149`). Observed:
  `np.shares_memory(E, ctx.E)==True` for contiguous float64 input; returned `rho`
  is writable and mutates the cache. No consumer currently mutates them. *Fix:*
  defensive-copy + read-only flags.
- **G12 — High-energy tail silently zero-filled** (`physics/gap_equation.py:87`,
  `observables/ac_conductivity.py:77`). Observed: constant f=0.05 gives gap
  **149.8→109.4 µeV** as Emax grows, no warning; physically decaying f is
  Emax-independent to 6 sig figs. *Fix:* high-edge occupation warning mirroring
  the existing low-edge one.
- **G13 — Effective-temperature fit bound reported as a measurement**
  (`observables/effective_temperature.py:107-118`). Observed: T_true=20 K returns
  **T_eff=9.999995 K** (upper bound), silently, reachable via the webui
  driven-phonon summary. *Fix:* flag/warn on boundary solutions.
- **G14 — Overnight resume corruption**
  (`scripts/run_prelim_readout_heating_overnight.py:200,357`). run-id omits
  NX/NE/dt/tmax; observed both smoke run-ids are an exact subset of overnight ids
  → a 3-ns 5-cell smoke result silently substitutes for overnight cases;
  nonconverged runs marked `completed`. Research script, off default path. *Fix:*
  fold resolution params into the run-id; don't overload `completed`.
- **G15 — Fig.6 marked complete without current numerical evidence**
  (`validation/fischer_2023/test_fig6_paper.py:92`; `docs/FISCHER-BASELINE-REGEN-2026-07-12.md:74`).
  The only full-sweep value test is slow+manual_slow (~14 h), CI-excluded, and
  its baseline predates the corrected-quadrature regen. Openly documented
  elsewhere; STATUS.md ✅ is an un-cross-referenced honesty gap. *Fix:* regen and
  re-pin, or footnote STATUS.md.
- **G16 — Cancellation ignored in two of four executors** (`webui/execute.py:154`,
  M25 loop). Observed: 0-D steady and final M25 point return `done` with a full
  valid payload despite a cancel in the solve window; transient/spatial have the
  guard. Result is scientifically correct — only the terminal label is dishonest.
  *Fix:* symmetric post-solve `_check_cancel`.
- **G17 — Concurrent same-slug saves race on a fixed .tmp** (`webui/store.py:112`).
  Observed: 16 threads × 100 saves of one slug → **757/1600 failures**
  (PermissionError/FileNotFoundError); committed file never corrupted; manifests
  unaffected. *Fix:* unique tmp name (uuid/mkstemp).
- **G18 — Below-threshold M25 drive validates OK, returns all-NaN**
  (`webui/builders.py:275`). Observed: 50 GHz drive (threshold 103 GHz) validates
  clean, finishes `done`, x_L all NaN, points_converged=0/5 — but per-point notes
  *do* name the below-threshold cause (GPT's "no error surfaced" is overstated).
  *Fix:* add the threshold check to `validate_setup`, mirroring the existing
  omega_10 guard.
- **G19 — Unauthenticated webui API** (`webui/cli.py:27`, `webui/server.py:81`).
  Confirmed no auth on any route (read/write/delete/compute); default bind is
  loopback-only, exposure requires a deliberate `--host`. *Fix:* warn or require
  `--allow-remote` on non-loopback bind; optional token.
- **G20 — load_material path traversal** (`materials/database.py:198`). Observed:
  `../secretzone/secret` escapes `database_dir`; but no untrusted caller supplies
  material names (webui derives them from a glob; slugs already guarded). *Fix:*
  add a containment/segment guard (defense-in-depth).

## Why the prior fan-out audit missed the real ones — structural causes

Self-critical by design. Each miss maps to a specific methodological blind spot,
not bad luck.

- **File-partitioning hid the cross-operator inconsistency (G1).** A 12-agent
  fan-out gives each auditor a slice of files. G1 lives *between* files —
  midpoint ρ·dE in `collisions/phonon.py` vs. exact cell weights in
  `observables/` and `bcs_quadrature.py`. No single auditor owned "the discrete
  DOS measure" as a global invariant, so the inconsistency was invisible to
  everyone. The highest-impact real finding is precisely the one only a
  cross-cutting view could catch.
- **Detailed-balance checks are measure-agnostic (G1, G2).** The prior audit
  leaned on thermal-residual / detailed-balance checks. Detailed balance holds in
  *any single consistent measure*, so it structurally cannot detect a mismatch
  *between* the operator's measure and the observable's, nor a non-monotone
  residual for a nonequilibrium f. The verification was blind to the whole class
  by construction.
- **In-code comments were trusted as specs (G2).** `gap_equation.py` asserts
  monotonicity; auditors accepted uniqueness rather than adversarially building a
  gap-edge-bump f. A comment claiming an invariant is exactly where an audit
  should push hardest.
- **The "concrete reachable wrong-output" discipline under-weighted a whole
  severity axis (G3, G9–G20).** Good for physics correctness, but it structurally
  discards robustness/DoS, validation-honesty, api-contract, and security/posture
  — none of which produce a wrong number on a default path. ~12 of the 19 real
  findings were invisible to this bar. The bar wasn't wrong; it was too narrow to
  be the *only* bar.
- **Adversarial/pathological inputs were never exercised (G2, G6, G7, G9, G10,
  G12).** Each needs a hostile input (non-monotone f, sub-tol rates,
  near-discontinuous map, NaN dt, degenerate grid, non-decaying tail). The prior
  audit tested the physical/contractive regime where all behave correctly, and
  conflated "unreachable through default callers" with "cannot be triggered."
- **The slow validation gate was never run (G4, G15).** The fast 1060-pass suite
  excludes the only value-checking Fig.3/Fig.6 tests. "Green baseline" was read as
  "figures validated," but the tests that would expose the vacuous Fig.3 pin and
  the stale Fig.6 baseline are `-m "not slow"`-excluded.
- **The audit stopped at the first smoking gun (G4).** The brief documented *one*
  all-zero Fig.3 column; the prior audit treated "found the degenerate column" as
  done rather than comparing every column's atol against its signal magnitude.
- **Research scripts and the webui were out of scope by habit (G3, G13, G14, G16,
  G17, G18, G19, G20).** The audit concentrated on `qpsim/` physics, but
  `scripts/` and `webui/` are reachable end-to-end (the webui passes user input
  straight into the drivers), and that's where the robustness/honesty cluster
  lives.

## Refuted / overstated

- **G5 (refuted).** GPT claims "8 Fischer reproductions complete" overstates the
  incomplete F24 `fig5_paper.py`/`fig8_paper.py`. False premise: the
  authoritative enumerations (STATUS table, Validation_Chain, ARCHITECTURE) count
  `figs_5_7_fe_pb` + `fig8_xqp_pb` for F24 — the `*_paper.py` scripts are
  separate, honestly self-labeled ("qpsim-native characterization," native-CSV
  filenames, "overlays not implemented"). No claim references them; no reader is
  misled. Prior audit was right here.
- **Severity overstatements (real but over-rated):** G6/G7 (High→low: false
  convergence only in sub-tol / near-discontinuous regimes unreachable through
  physics; G6 additionally inverts the `coupled_newton` diagnosis). G19
  (Cond-High→low: loopback default, opt-in only). G18 (Medium→low: GPT's "no
  error surfaced" is wrong — per-point diagnostics name the cause). G8/G12
  (Medium→low: only manifest with unphysical inputs / within 0.08% of Tc).
- **Documented false positives:** none of the 19 are the repo's known false
  positives. G8 and G12 are explicitly distinct from the documented "solve_gap
  biased near Tc" warning; G1 is not the documented Δ-vs-Δ₀ normalization note.

## Net recommendation

**Fix first (medium, real, reachable at default settings):**
1. **G1** — unify the discrete DOS measure across collision kernels and
   observables. Highest physics impact; drives a 10–15% x_qp inconsistency in the
   low-T regime the tool targets. **⚠️ ATTEMPTED 2026-07-14 and REJECTED** — the
   naive exact-cell-weight swap passes detailed balance / conservation / the fast
   suite but **catastrophically degrades the driven Fischer reproductions**
   (fig3/5/6 collapse 5–8 orders of magnitude; A/B-isolated to the measure
   change). The exact singular gap-edge weight over-weights recombination in the
   driven self-consistent solve. See `G1-MEASURE-ATTEMPT-2026-07-14.md`; needs a
   physicist. Do NOT ship the swap; the inconsistency stays a documented ~1/√NE
   convergence budget for now.
2. **G3** — clamp `snapshot_interval` in schema + bound/poll the inner cadence
   loop. Trivial; closes an uninterruptible DoS.
3. **G4** — make Fig.3 (and Fig.5/6 tail) numerical pins signal-scaled; restores
   real regression coverage.
4. **G2** — replace the assumed-monotone bracket with a global root scan +
   continuity selection; correct the false monotonicity comment.

**Fix next (cheap low-severity hardening, batchable):** G6/G7/G9
(certificate/guard symmetry), G13/G16/G18 (honesty flags + upfront M25 threshold
validation), G17 (unique tmp name), G8 (linearized-Tc coupling), G11/G20
(defensive copies / path guard).

**Documentation-only:** G14 (run-id namespacing), G15 (footnote STATUS.md), G19
(non-loopback warning).

**Run next to close remaining unknowns:**
- Run the slow + manual_slow validation gate (`-m "slow or manual_slow"`),
  especially regenerate/re-pin **Fig.6** under corrected `bcs_dos_cell_weights` —
  the single largest current blind spot (G15).
- After the **G1** fix, re-run the pinned Fischer baselines + the phonon
  detailed-balance suite to confirm the unified measure doesn't move validated
  figures beyond tolerance.
- Add a standing **cross-operator invariant test** ("conserved dynamical number
  == reported observable number to O(1/√NE)") so this class is caught by CI, not
  by an external auditor.

## Addendum (2026-07-14): the slow validation gate was already red

After applying the 16 non-baseline-moving fixes, the slow gate
(`pytest -m "slow and not manual_slow"`, ~8.5 h) was run — apparently for the
first time in a while; neither GPT's audit nor the fan-out ran it (both used the
fast gate, and the repo's "green" claim was fast-gate-only). It has **4
pre-existing failures at `7e72b52`, unrelated to the fixes**:

- `validation/transient/test_photon_kick_response.py` — **PROVEN pre-existing**:
  fails identically at clean `HEAD` (snapshot times `96.0…114.0` vs a stale
  baseline `96.1…114.1`; the baseline predates a snapshot-cadence/scheduler fix).
- `validation/fischer_2024/test_fig5_paper.py`, `test_fig8_paper.py`,
  `test_figs_5_7_fe_pb.py` — strongly-inferred pre-existing: the fixes cannot
  alter their computed values (guards/warnings/read-only views/behavior-preserving;
  `grep` confirms no in-place ctx-array mutation in `validation/`), and these are
  the same F24 modules G5/G15 flag as placeholder/unit-mismatched. (Clean-HEAD
  confirmation not yet run — ~1–2 h.)

This is a **new validation-integrity finding** beyond the 19: the pinned "green"
baselines were fast-gate-only, and the slow gate has been red. It compounds G4/G15
(the vacuous fig3 pin, the stale fig6 baseline) — a baseline regeneration +
re-pinning pass is needed regardless of G1/G8, and should be bundled with them.

## Verdict

**Verdict on "essentially clean":** overturned as an overall characterization,
with an honest caveat — the prior audit was correct that there is no
blocker-class physics error and no corrupted published result. What it missed is
a broad, consistent layer of medium/low robustness and validation-honesty
defects its physics-correctness-only methodology could not see. The repo is sound
at its physics core and not submission-blocked by these, but it is not "clean,"
and the four medium findings (G1 above all) warrant fixes before the numbers are
relied on at the default grid.
