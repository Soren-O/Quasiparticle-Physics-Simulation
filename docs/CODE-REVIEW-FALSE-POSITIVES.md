# Code Review False Positives

A running ledger of findings filed during code audits/reviews that were
adjudicated as **not defects** — so the next audit doesn't burn effort
re-deriving them, and so nobody "fixes" intentional behavior.

**Process:** every audit round, add newly adjudicated false positives here
with the claim as filed, the verdict, and the evidence. Never delete an
entry; if a verdict is later overturned, move it to §3 with the reversal
recorded (that has happened — see §3). Check this file **before** filing a
finding; if you believe an entry is wrong, you may re-file, but you must
state explicitly why the recorded adjudication fails.

Entry format: **Claim** → *Verdict* — evidence (date/round).

---

## 1. Refuted findings (checked against code/execution — not bugs)

- **`solve_gap` is biased high near `T_c`** → *Documented domain limit* —
  the energy grid cannot sample below-gap occupation; it warns. Not a
  quadrature bug. (pre-2026-07-13 audits; re-verified since)
- **Phonon pair-breaking kernel normalizes by current Δ, not Δ₀** →
  *Documented approximation* — Δ ≈ Δ₀ at the Fischer temperatures; stated
  in the kernel docs. (pre-2026-07-13)
- **`_tau_R_inverse` should use mean-gap Δ̄ instead of Δ_R in the D11/S48
  formula itself** → *Version/label false alarm* — arXiv:2408.17218 **v2**
  Eq. D11/S48 carries Δ_R. ⚠️ Do not confuse with the *r^{R<} conversion
  prefactor*, which genuinely needed (Δ_R/Δ̄)³ — see §3.1. (2026-07-13)
- **TiN `rho_F ≈ 3.8e28 eV⁻¹m⁻³` is a unit typo** → *Film/disorder-
  dependent value, defensible*. (2026-07-13)
- **M25 steady-state acceptance should be a fixed absolute tolerance** →
  *The row-wise source-scaled + backward-error gate is intentional* — a
  bare `1e-14`/`1.0 Hz` reading is the OLD, already-fixed miscalibration.
  (2026-07-13)
- **Spatial Crank–Nicolson crashes on large `D₀·dt/dx²`** → *Fail-loud by
  design* — keep the diffusion number ≲ 5. (2026-07-13)
- **Direct gap integrals raise when the grid omits the superconducting
  edge** → *By design*; only roundoff-sized positive face offsets are
  aligned. (2026-07-13/15)
- **Fig. 6 negative direct suppression values are a sign bug** → *Retained
  and plotted signed by design*; only explicit SC collapse maps to NaN.
  (2026-07-15)
- **`sweep_cache` Fig. 7 solve-source digest omits downstream
  observables** → *Intentional* — observables are covered by separate
  artifact/dependency tests. (2026-07-15)
- **Canonical Fig. 6 vs `_direct` output paths should be unified** →
  *Deliberately distinct.* (2026-07-15)
- **Default `pytest` skips the paper baselines** → *By design* — CI runs
  `-m "slow and not manual_slow"` as its own step (and a guard test now
  pins that step's existence). (2026-07-13; guard added 2026-07-19)
- **`build_variable_diffusion_laplacian` lacks the sibling's
  `missing_edges` completeness check** → *Refuted by execution* — the
  operators come out byte-identical. (2026-07-13)
- **webui `occupation_heatmap` `vmin>vmax` LogNorm guard can 500** →
  *Dead branch* — the T3 backend's QP floor prevents all-underflow of
  `f_final`. (2026-07-13)
- **F24 strict-v2 read-time certification is forgeable (stamp-based, not
  verifying)** → *No accidental path* — the "attack" requires deliberately
  recomputing and rewriting the in-file SHA-256. (2026-07-19 audit)
- **`density.py` strict `f ∈ [0,1]` gate breaks on roundoff-negative
  occupations** → *All shipped callers clip first.* (2026-07-19)
- **F24 slow pins' `atol=1e-14` is vacuous on f(E) tails down to 1e-90** →
  *Head is rtol-controlled*; the tail below 1e-14 is physically irrelevant
  there. (2026-07-19)
- **`_K_incomplete` has unbounded relative error at large z** → *In-domain
  on every shipped path*; consuming products keep it there. (2026-07-19)
- **"No solver-vs-paper analytic quantitative test exists"** → *They
  exist* (the eq53 comparison layer); their *correctness* was a separate,
  real finding. (2026-07-19)

## 2. Documented approximations & accepted limitations
(Real physics/engineering gaps, adjudicated as documented contracts —
don't re-file as new bugs; re-open only with new evidence or a design.)

- **Moving-gap stranded-tail policy**: up to 1e-3 of QP *number* may sit
  hidden above `E_max` (warn above 1e-9); wholly-hidden rows are
  collisionless, straddling rows co-evolve. Bounded experimental
  approximation, not a validated recovery method — an `E_max`-convergence
  error-budget study is the open TODO. (user-adjudicated 2026-07-20;
  see `Moving_Gap_Time_Integration.md`)
- **Phonon gap-cut cell ω-labeling**: a supported cut cell's pair events
  can be binned below 2Δ (≤ one `dE` mislabel; discrete detailed balance
  exact; vanishes on covered grids). Masking is NOT the fix (deletes
  physical rate); the designed fix is a rate-preserving gap-aware ω-remap
  across rates + Jacobians. The *photon* channel, by contrast, is hard-
  gated at 2Δ — that part was a real bug, fixed. (2026-07-20 rounds 3–4)
- **Marchegiani strict pins are win32-stamped; ubuntu CI runs the 1e-3
  fallback** → accepted trade-off, documented in `baselines/README.md`;
  a Linux-stamped twin was considered and declined. (user-adjudicated
  2026-07-20)
- **Fig. 5 low-drive `atol=1e-6` gates are vacuous** → known; deferred BY
  DESIGN to the Fig. 5 regeneration campaign, where signal-scaled
  tolerances are a required part of the re-pin. (2026-07-20)
- **Fig. 7 solve uses fixed Δ₀ and one grid for all temperatures** →
  documented convention at `_build_grid` (~0.17% gap mismatch at 0.34 K);
  the pin is a self-consistent regression under that convention, and
  re-gridding is a baseline-moving change. (2026-07-20 round 3)
- **Device conserved-mode certificate is scoped** to symmetric ratio-1
  matched-weight junctions (bin-wise cancellation provable); other
  junction sets get a loud once-per-solve warning. Known open gap, not a
  silent one. (2026-07-20 round 4)
- **Sub-threshold photon `loss_rate` coefficient stays finite below 2Δ** —
  that is the scattering channel (physical at any ω); only pair
  *generation* is gated. A `loss == 0` expectation below 2Δ is wrong.
  (2026-07-20 round 3, caught in our own test)
- **Mismatched-T device fixture needs ~1200 outer iterations at
  non-default budgets** — genuine slow outer mode under honest
  scale-aware certification; defaults fail *safely*. (2026-07-20 round 4)

## 3. ⚠️ Traps: things that LOOKED like false positives but were real
(The anti-overconfidence section. Read before refuting anything.)

1. **The (Δ_R/Δ̄)³ τ_R normalization factor** — refuted in one review
   round "against the paper text", then **confirmed** in the next: the
   refutation had relied on a plain-text extraction that truncated the
   equation exactly before the `Δ̄³` tail, corroborated by this repo's own
   doc which had transcribed `Δ_R³`. The MathML alt text in the saved
   ar5iv HTML settled it: `r^{R>} ≃ r^{<>} ≃ r^{R<} ≃ 8π b_R Δ̄³`.
   *Lesson: verify formula claims against the equation SOURCE (MathML/
   LaTeX alt text), never a plain-text conversion, and never treat the
   repo's own transcription as independent evidence.*
2. **The "21% τ₀ᴾᴮ shift" used to justify reverting the phonon pair
   mask** — a measurement artifact: the experimental mask broke the
   canonical-kernel detection behind the Kaplan endpoint correction, and
   the same shift reproduced on grids with zero sub-threshold pairs.
   The revert itself was still right, but for the principled reason only.
   *Lesson: a side-effect measurement needs a control run (apply the same
   change where the claimed cause is absent).*
3. **A regression test that passes is not evidence the bug existed** —
   a round-3 sub-2Δ test used a face-aligned gap with no supported cut
   cell, so the PRE-fix code also returned zero and the test was vacuous
   (its quoted pre-fix repro number was mis-transferred from a different
   parameterization). *Lesson: run the claimed failing scenario against
   the pre-fix code (or verify the construction actually exhibits the
   defect) before pinning it as a regression test.*
4. **"Green CI + green fan-out audit" ≠ correct** — four consecutive
   review rounds each found real defects in a tree with fully green
   hosted CI and a 123-agent audit behind it. Per-file audits are
   structurally blind to cross-file/cross-layer issues (`scripts/`,
   paper-anchor conventions, solver certification null spaces).
