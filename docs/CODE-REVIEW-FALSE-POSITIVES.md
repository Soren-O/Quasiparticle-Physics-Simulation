# Code Review False Positives and Adjudications

A running ledger of findings filed during code audits/reviews.  Only §1 is
a list of findings currently adjudicated as **not defects**.  Section 2
contains real, accepted limitations and §3 contains overturned
adjudications.  Those sections are retained so future reviewers have the
history and evidence, not so they suppress a valid finding.

**Process:** every audit round, add newly adjudicated claims with the claim
as filed, the verdict, and the evidence. Never erase the history: if a
verdict is later overturned, move it to §3 and record the reversal. Check
this file before filing a finding, but treat an entry as evidence rather
than a veto. A change to any code or assumption in an entry's stated scope
automatically reopens it for verification. If an entry is wrong, re-file it
and state why the recorded evidence or scope no longer holds.
Audit reports should say “no blocker found in the reviewed scope,” never the
durable-sounding “no remaining code blocker.”

New entry format:

> **ID / current status / claim** → *verdict* — evidence and reproducer;
> exact scope; commit verified; regression test (if any); explicit reopen
> condition; date/round.

Legacy entries below predate that schema. Treat them as provisional when
their implementation or assumptions have changed since the recorded date.

> **Live Round-7 status (2026-07-25):** The replacement `NE=1620` Fig. 3
> solve completed all 14 continuation steps in `10671.777 s`, passed
> independent certificate/readback and visual checks, and was promoted.
> After final-equation recertification, its current
> CSV/PDF/validation-record SHA-256 values are
> `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
> `7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
> and
> `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`.
> The four Fischer-2024 families were genuinely regenerated through fresh
> solve processes after the final digest change and promoted as strict-v3
> pairs. Fig. 7 also completed all 48 exact current-source targets under
> hardened identity `82ef6da8…f320`; matched CSV/PDF/attestation promotion,
> independent readback, and visual inspection passed. The final non-slow
> aggregate is `2188 passed, 18 deselected, 12 warnings, 0 failed`.
> Hosted CI remains separate post-push evidence. The working tree is
> `codex/audit-round7-fixes`, based on
> `cd4fb81`; historical counts below remain historical rather than proof for
> this tree.

---

## 1. Refuted findings (checked against code/execution — not bugs)

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
- **`density.py` strict `f ∈ [0,1]` gate breaks on roundoff-negative
  occupations** → *All shipped callers clip first.* (2026-07-19)
- **"No solver-vs-paper analytic quantitative test exists"** → *They
  exist* (the eq53 comparison layer); their *correctness* was a separate,
  real finding. (2026-07-19)
- **A below-2Δ photon must have `loss_rate == 0`** → *Refuted physical
  expectation* — the number-conserving scattering channel remains physical
  at sub-threshold photon energies. Only pair generation/recombination is
  gated by the strict `ω > 2Δ` predicate. (2026-07-20 round 3)
- **R7-CI-TIMEOUT / refuted / the existing 180-minute hosted slow-step
  timeout was already too short** → *Cross-host timing extrapolation, not an
  observed CI failure.* At exact hosted head `cd4fb81`, GitHub Actions run
  [29845574296](https://github.com/Soren-O/Quasiparticle-Physics-Simulation/actions/runs/29845574296)
  completed the slow step in 39m39s on Python 3.13 and 40m09s on Python 3.14,
  well inside the configured 180-minute step timeout. The much slower local
  Windows Fig. 3 execution is real performance evidence for that host, but it
  does not show that the existing hosted run timed out. Scope is deliberately
  narrow: this evidence predates the uncommitted Round-6/7 tree and does not
  prove that a future enlarged slow suite will remain below 180 minutes.
  The later 49.29-hour local Windows run does not overturn this narrow
  hosted-CI adjudication: it exposed real local algorithm/runtime defects, but
  it is still not an exact-head hosted timeout observation.
  Re-open from an exact-head hosted timing near or above the limit, or after
  materially changing the selected slow tests. No code regression test;
  evidence is the hosted step timestamps and `.github/workflows/ci.yml`.
  (2026-07-22 Round 7)
- **R7-FIG3-ROUTING / refuted / the explicit Fig. 3
  `number_polish_shape_tol=1e-8` weakens solver acceptance** → *False within
  its exact scope.* The parameter changes only when the scalar number-mode
  polish is attempted. Every candidate still re-enters the unchanged
  residual, aggregate-backward-error, and pair-number certificates at
  `1e-10`; the generic default remains conservative. On the paper grid it
  reduced the ratio-zero path from about `32.8 s` and 16 dense Jacobians to
  `5.628 s` and 3 Jacobians with the returned occupation unchanged in that
  routing-only A/B to the comparison precision. Re-open if
  the routed candidate ever bypasses a return certificate or the
  independently measured Fig. 3 shape floor changes. (2026-07-25 Round 7
  follow-up)
- **R7-FIG3-AA-BRANCH / refuted on the 81-bin diagnostic / safeguarded
  Anderson necessarily changes the continuation branch** → *Not observed in
  the measured scope.* Historical plain Picard took `314.2773 s`; depth-3
  safeguarded Anderson took `17.9573 s` (`17.5×`, 94.29% less wall time).
  Target full-array maximum differences divided by the corresponding legacy
  peak were `[0, 1.727e-7, 3.745e-6, 7.176e-9]` for ratios
  `[0, 0.1, 1, 10]`, and both paths passed the independent certificates. This
  did **not** itself substitute for the later completed 1620-bin result. Re-open on
  grid/ladder/mixing changes or any full-grid branch/certificate disagreement.
  (2026-07-25 Round 7 follow-up)
- **R7-FIG7-LOCK-FALLBACK / refuted / the Fig. 7 campaign lock returns a
  Linux process identity on non-Linux hosts** → *Test-fixture false
  positive.* `_process_identity` deliberately selects the robust
  boot-ID/start-time identity when the Linux `/proc` creation-time
  interfaces are present and otherwise uses the conservative stable-live
  fallback. The hosted test changed `sys.platform` to `darwin` but left the
  Linux runner's real `/proc` visible, so it never exercised the fallback it
  claimed to test. The fixture now hides that capability as well; the
  production driver and its attested SHA-256 remain unchanged. Re-open if a
  real supported non-Linux host exposes those paths with incompatible
  semantics, or if capability-based dispatch is replaced by platform-label
  dispatch. Regression:
  `tests/scripts/test_regenerate_fischer_fig7_parallel.py`.
  (2026-07-25 Round 7 hosted follow-up)
- **R7-FIG7-PLATEAU-THRESHOLD / refuted / hosted Fig. 7 returned a
  non-extrinsic-limited low-temperature plateau** → *Stale test premise.*
  The live Linux result at 0.06 K and −64 dBm was
  `Q_qp=4.191009106e9`, matching the authenticated Windows pin
  `4.191009023e9` to about `2e-8` relative. The failed test instead demanded
  an undocumented `Q_qp > 1e12`, directly contradicting that canonical
  artifact. The physical claim is `Q_qp >> Q_ext`: here
  `Q_qp/Q_ext ≈ 5987`, so the total quality factor is extrinsic dominated.
  The live gate now asserts that loss hierarchy and compares both QP loss and
  total Q against the authenticated pin with the measured cross-platform
  envelopes. Re-open on failure of those comparisons, not the retired
  absolute threshold. Regression:
  `validation/fischer_2023/test_fig7_paper.py`.
  (2026-07-25 Round 7 hosted follow-up)
- **R7-F24-CERTIFICATE-PIN / refuted / Fischer-2024 Fig. 8 physics drifted
  on hosted Linux** → *Diagnostic-roundoff false positive.* All physical
  `x_qp` arrays matched before the failure; only three backward-error
  certificates changed at approximately `1e-12`, still six orders below the
  advertised `1e-6` acceptance limit. A normwise residual at the last accepted
  floating-point iterate is a validity certificate, not a reproducible
  observable. The slow regression continues to compare the physical curves
  and now requires each freshly reassembled backward error and raw residual
  to satisfy its actual contract. Re-open if a certificate exceeds its
  threshold or a physical curve drifts. Regression:
  `validation/fischer_2024/test_fig8_xqp_pb.py`.
  (2026-07-25 Round 7 hosted follow-up)

## 2. Documented approximations & accepted limitations
(These are **real physics/engineering gaps**, not false positives. Do not
present the already-recorded limitation as a newly discovered fact, but do
re-open it when new evidence, a concrete design, or a changed caller makes
the impact actionable.)

- **Moving-gap stranded-tail policy**: up to 1e-3 of QP *number* may sit
  hidden above `E_max` (warn above 1e-9); wholly-hidden rows are
  collisionless, straddling rows co-evolve. Bounded experimental
  approximation, not a validated recovery method — an `E_max`-convergence
  error-budget study is the open TODO. (user-adjudicated 2026-07-20;
  see `Moving_Gap_Time_Integration.md`)
- **`solve_gap` near-`T_c` grid bias:** the represented energy grid cannot
  sample below-gap occupation; the solver warns. This is a real documented
  domain/discretization limit, not a quadrature implementation bug.
  (pre-2026-07-13 audits; re-verified since)
- **Direct gap-integral grid coverage:** direct integrals require the
  represented grid to cover the superconducting edge; only roundoff-sized
  positive face offsets are aligned. This is an explicit domain contract,
  not proof that omitted-edge inputs are physically resolved. Re-open if a
  caller needs extrapolation or a changed grid violates the contract.
  (2026-07-13/15)
- **Current-gap phonon pair-breaking normalization:** the kernel uses current
  Δ rather than Δ₀. This is a documented approximation whose impact is small
  only in the Fischer regimes where Δ ≈ Δ₀. (pre-2026-07-13)
- **Phonon gap-cut cell ω-labeling**: a supported cut cell's pair events
  can be binned below 2Δ (≤ one `dE` mislabel; discrete detailed balance
  exact; vanishes on covered grids). Masking is NOT the fix (deletes
  physical rate); the designed fix is a rate-preserving gap-aware ω-remap
  across rates + Jacobians. The *photon pair-generation/recombination block*,
  by contrast, is gated by the strict `ω > 2Δ` predicate; physical sub-gap
  K+ scattering remains active. The pair-block omission was a real bug and is
  fixed. (2026-07-20 rounds 3–4)
- **Marchegiani strict pins are win32-stamped; ubuntu CI runs the 1e-3
  fallback** → accepted trade-off, documented in
  `validation/baselines/README.md`;
  a Linux-stamped twin was considered and declined. (user-adjudicated
  2026-07-20)
- **Fig. 5 low-drive `atol=1e-6` gates are vacuous** → known; deferred BY
  DESIGN to the Fig. 5 regeneration campaign, where signal-scaled
  tolerances are a required part of the re-pin. (2026-07-20)
- **Fig. 7 solve uses fixed Δ₀ and one grid for all temperatures** →
  documented convention at `_build_grid` (~0.17% gap mismatch at 0.34 K);
  the pin is a self-consistent regression under that convention, and
  re-gridding is a baseline-moving change. (2026-07-20 round 3)
- **F24 strict-v2/v3 in-file hashes are currency/integrity checks, not authenticated
  tamper defense.** They catch accidental stale or corrupted payloads, but a
  writer who deliberately changes data and recomputes the in-file SHA-256 can
  forge a self-consistent stamp. Do not cite these hashes as provenance
  against a malicious or mistaken re-pin. (2026-07-19 audit, scope corrected
  2026-07-21)
- **F24 tail validation:** the resolved head is relative-tolerance controlled,
  while the `1e-14` absolute floor deliberately does not constrain f(E) tails
  down to ~1e-90. That is a scoped validation limit, not a general statement
  that arbitrary tail errors are harmless. Re-open when a tail-sensitive
  observable is added. (2026-07-19; classification corrected 2026-07-21)
- **R6-DEVICE-SCOPE / accepted limitation / Device conserved-mode support
  contract.**
  Active conservative transfer edges declare a public scalar `C_a/C_b`
  contract and their evaluated flux is checked; the shipped symmetric edge
  additionally requires its matched finite-volume measure. Consistent unequal
  ratios work, while inconsistent/extreme graphs refuse. Zero-rate edges are
  inert and do not join components. The finite-phonon-specific restriction is
  limited to active conservative cross-region components; independent or
  disabled-edge Devices remain valid. In either phonon mode, unknown nonzero
  state-dependent Junction flux needs a prescribed-source or conservative-
  capacity contract and otherwise refuses. Regression:
  `tests/devices/test_device.py`; working tree based on `cd4fb81`, durable
  commit pending. Re-open for
  any new Junction family, finite-phonon component certificate, or change to
  capacity accounting. (2026-07-21 follow-up)
- **Mismatched-T frozen-flux convergence is slow and budget-dependent.** A
  `5e-5` request reached about `3.8e-5`; a two-iteration continuation at
  `1e-5` failed safely. That test does **not** establish an intrinsic accuracy
  floor or prove that a full iteration budget cannot converge more tightly.
  Re-open after any full-budget convergence study or outer-solver change.
  (2026-07-21 follow-up)
- **R7-FIG3-RESTART / accepted durability boundary.** Restart state is
  content-bound, schema-checked, atomically replaced, and records every
  completed continuation step. With restart enabled, callbacks are delivered
  at least once and completed targets are replayed before further
  computation; callback side effects must therefore be idempotent. Pure
  uncached `run()` remains fresh unless restart is explicitly requested. A
  completed checkpoint is recovery state, not a cross-version result cache,
  and is retained until an outer owner makes its final artifact durable. This
  code does not automatically delete it; explicit owner cleanup is required.
  This is not multi-process locking or a transactional atomic commit across
  checkpoint, cache, CSV, and PDF files. Re-open if ownership/deletion
  semantics or concurrent-writer support changes. (2026-07-25 Round 7
  follow-up)
- **R7-FISCHER-NUMBER-SCHEMA / accepted staged rollout.** The independent
  Fischer certificate now detects the cold amplitude/null mode. Fig. 3
  persists and requires that extended field. Its
  `producer_solve_contract_digest` identifies the contract that actually
  generated stored `f`, while `validated_solve_contract_digest` identifies a
  later contract that re-certified it. Finite-escape re-certification
  reconstructs the affine Ph0 root implied by stored `f`; passing proves
  current-equation root membership for that reconstructed pair, not the
  producer's original `n_ph` or execution of the current solver algorithm.
  Figs. 5--7 gate the number certificate during live solves. Fig. 7's
  regenerated schema now persists `qp_number_backward_error` and carries a
  promotion attestation; Figs. 5--6 retain legacy summary schemas that do not
  contain the returned `f`/`n_ph` state needed for honest reconstruction.
  Those two old artifacts are not artifact-level number certificates. A
  persisted upgrade requires genuine regeneration; fabricating fields or
  treating a metadata-only rebind as numerical evidence would be a defect.
  All four F24 families and Fig. 7 have now been genuinely regenerated under
  their extended/current schemas. Full-state re-certification under any future
  contract change must still retain producer and validator identities
  separately. Re-open for Figs. 5--6 regeneration, or if any future schema
  change loses that provenance distinction.
  (2026-07-25 Round 7 follow-up)

Audit-history note: Round 5 filed six findings and all six were confirmed
real; this is history, not an accepted limitation and not evidence that future
rounds must have the same outcome.

Round 6 added no new item to §1. Instead it overturned or materially narrowed
several older classifications, including the certificate scope, spatial
subcycling/completeness checks, WebUI all-underflow branch, large-`z`
incomplete-Bessel limitation, late Anderson-collapse guard, and the coupled
thermal-shortcut contract (see §3.5–11).
That imbalance is intentional: this ledger records evidence, not a quota of
findings that must be refuted.

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
4. **"Green CI + green fan-out audit" ≠ correct** — the original 123-agent
   audit, four external-review rounds, and the subsequent Round-6 pass all
   preceded or found real defects in a tree with fully green
   hosted CI and a 123-agent audit behind it. Per-file audits are
   structurally blind to cross-file/cross-layer issues (`scripts/`,
   paper-anchor conventions, solver certification null spaces).
5. **The Round-5 standalone/device number-certificate scope was a
   deliberate, complete contract** — **overturned on 2026-07-21.** A
   zero-valued `ExternalFlux` changed no equation but bypassed the
   standalone certificate; number-conserving sub-gap photon scattering
   diluted the purported pair-only denominator; a PB-photon-only solve
   skipped the check when `K_r0` was absent; and unsupported Device modes
   warned before returning known wrong fixed points. The mixed-channel
   numerator also rejected exact cold thermal roots because scattering
   roundoff was divided by exponentially small pair turnover. *Lesson:
   certify a conserved quantity using an explicit decomposition of only
   the channels that change it, and test physically equivalent API forms
   (for example `None` versus an all-zero source).* The follow-up repair
   now uses explicit phonon-pair, PB K⁻ pair, and local-source components and
   ties acceptance to the advertised tolerance. During an earlier Fig. 7
   campaign this adjudication's own reopen condition fired: the amplitude
   polish and floating-point policy changed again. The re-audit required
   direct assembly of the number-conserving shape channels, per-row physical-
   turnover scaling without a global floor, normalized-balance line-search
   merit once dimensional residuals are already below tolerance, a raw-Newton
   fallback, and delayed log-amplitude globalization only after both Newton
   directions fail. The reopen condition fired again during the 2026-07-22
   final slow gate: a refined Fig. 3 point had passing absolute/aggregate
   errors but number error `1.09e-10` against `1e-10`, and the line search
   spent 500 iterations accepting progress on the wrong merit. When number is
   the only failed certificate, the solver now preserves the passing aggregate
   gate and returns immediately when a trial clears the number gate. If none
   clears it, the solver scans all 20 backtracks and chooses the feasible trial
   with the best number error. The final source-bound original/refined ladder
   comparison passed through ratio 10 in 4906.90 s without a tolerance or
   baseline change, down from 6100.62 s before the safe early exit. Its refined
   ratio-10 certificate was `η_qp=6.393e-17` and scaled
   `η_ph=3.803e-20`. A one-ULP
   saturation regression also prevents the amplitude candidate from escaping
   `f ∈ [0,1]`. Representable wrong-amplitude
   seeds are corrected;
   genuinely unrepresentable configured turnover refuses, while an exact
   absorbing vacuum with no actual or declared pair-generating source is
   accepted. The total-number certificate and amplitude polish apply only
   when the solve mask equals `ctx.active_mask`; any nonempty proper subset
   is now rejected because transition flux across its artificial boundary is
   unaccounted. An all-false mask remains an explicit no-op, and a mask that
   enables any row outside `ctx.active_mask` is invalid. Prescribed `ExternalFlux`
   turnover stays in the normalizer; only explicitly tagged conservative
   Device exchange is excluded, so a large balanced transfer cannot hide
   internal imbalance. Device backends are pure-map boundaries: input
   mutation, incoherent/non-finite ancillary T3 state, and overflow/NaN in a
   declared transfer check refuse rather than self-certify. Regression:
   `tests/solvers/test_newton_steady_state.py` and `tests/devices/test_device.py`;
   historical Round-6 production evidence included the certified
   `−68 dBm / 0.06 K`
   Fig. 7 point and the complete 48-target regeneration (maximum QP backward
   error `3.70072e-10`, maximum phonon backward error `9.82942e-9`, both below
   `2e-8`, solve digest
   `71d2730e43b41fba106e3066de156eb6d4f69a23ea4aac91c64de6bd552d0503`);
   the then-working tree was based on `cd4fb81`. The reduced Fig. 6
   source-bound pin was also legitimately moved by this
   repair: its old state had independent pair-number error `7.4462e-6 > 1e-6`,
   while the new deterministic state has `4.4959e-7`; the regression now
   reassembles that balance independently and retains its original pin
   tolerances. Re-open on any channel decomposition, amplitude-polish, or
   floating-point policy change. Its reopen condition fired again on
   2026-07-25: the old full-grid Fig. 3 ratio-zero pin carried `3.659e-3`
   pair-number error and was shown by a certified diagnostic amplitude root to
   require replacement. A source-frozen replacement was completed, independently
   revalidated, and promoted on 2026-07-25; see item 12.
   (2026-07-21 Round 6 re-adjudication; reopened and resolved 2026-07-25)
6. **Large spatial Crank–Nicolson diffusion numbers were merely a permanent
   fail-loud design limit** — **superseded/fixed.** The July-15 monotonicity
   repair added automatic subcycling based on the maximum exit rate; the live
   backend chooses the required substep count and refuses only an excessive
   (>1,000,000) request. The old “keep `D₀·dt/dx² ≲ 5`” entry described a
   prior implementation, not the current contract. Regression/history:
   `docs/AUDIT-2026-07-15-numerical-software.md` N16. Re-open on subcycling or
   monotonicity-policy changes. (classification corrected 2026-07-21)
7. **The variable-diffusion missing-edge check was a false positive because
   the two assembled operators were byte-identical** — **overturned in
   Round 6.** Byte equality on a fully assigned mesh did not test the API
   completeness contract. The variable builder accepted an unassigned empty
   declared segment (and could miss a redundant already-covered declaration)
   that its sibling rejected, so global edge-ID assignment completeness was
   not enforced. Both builders now share the same validation
   for invalid directions/coordinates, duplicate or interior faces, and
   unassigned declared segments, including empty segments; an assigned empty
   segment remains valid.
   Regression: `tests/grid/test_spatial_grid.py`. *Lesson: matching outputs on
   valid input do not refute a missing invalid-input/domain guard.*
   (2026-07-21 Round 6)
8. **The WebUI all-underflow heatmap branch was unreachable because the T3
   backend floors occupations** — **overturned in Round 6.** The plotting API
   accepts schema-valid arrays independently of that one backend path, and an
   ultracold/direct payload can have no positive representable samples. The
   old `LogNorm` construction then had an invalid range; it now falls back to
   a valid linear normalization when every positive value lies at or below
   the plotting floor. Regression:
   `tests/webui/test_review_fixes.py`. *Lesson: prove reachability against the
   public function's input contract, not one upstream producer.*
   (2026-07-21 Round 6)
9. **`_K_incomplete` was merely an accepted out-of-domain large-`z`
   limitation** — **superseded/fixed in Round 6.** The failure was reachable
   through finite positive M25 temperatures: at `z≈1199.81` the branching
   fraction became `0/0` even though its true value
   `2.11583916322907e-106` is representable. The implementation now uses a
   Gaussian-scaled endpoint quadrature, log-domain full/incomplete/difference
   forms, and `kve`/`erfcx` for cancellation-prone consumers. At still colder
   points a mathematically subnormal ratio becomes exact zero without making
   the coefficient bundle non-finite. Regression:
   `tests/services/test_rate_equation_coefficients.py`. *Lesson: individual
   factors underflowing does not make their ratio unrepresentable.*
   (2026-07-21 Round 6)
10. **The existing late Anderson branch-collapse guard already protected
    cold Fig. 7** — **overturned in Round 6.** The guard waited until the outer
    convergence check, but the observed collapsed branch entered an exact
    200-iteration cycle before that point, so the recovery path was never
    reached. The guard now runs immediately after the collapsed inner solve:
    it restores the last matched `(f, n_ph)` pair, disables Anderson once, and
    continues with plain Picard. In the exact `−68 dBm / 0.06 K` production
    run, call 6 collapsed to `xqp=3.23e-106`; call 7 restored the physical
    `xqp=1.22e-10` state, and calls 8–251 converged monotonically to the
    independently certified endpoint. The Round-6 48-target artifact
    regeneration repeated the point from a fresh seed. That earlier campaign's
    2835.1 s timing is historical; the historical Round-6 final-source campaign
    completed the same point in 3415.6 s under solve digest
    `71d2730e43b41fba106e3066de156eb6d4f69a23ea4aac91c64de6bd552d0503`.
    *Lesson: a recovery branch is not
    coverage unless the failure reaches it before the algorithm cycles or
    exhausts its budget.* Regression: `tests/services/test_steady_state.py`.
    (2026-07-21 Round 6)
11. **Exact Fermi/Bose-shaped arrays were sufficient to justify the coupled
    solver's ultracold thermal shortcut** — **overturned in Round 6.** The
    arrays are an independently known fixed point only for the exact phonon
    frequency/index map derived from the QP energy grid. A custom map with the
    correct shapes, integer types, and in-range indices could route every
    recombination pair to the zero-frequency bin; the shortcut then returned
    the input unchanged even though the QP-number backward error was `1.0` and
    the slow-phonon error was `1.088`. Shortcut eligibility now requires exact
    value equality with `build_phonon_frequency_map(ctx.E)` for the frequency
    axis, difference map, sum map, and sign map. Noncanonical maps remain valid
    solver inputs, but must pass the ordinary and slow-channel certificates.
    *Lesson: an analytic fixed-point shortcut certifies the operator and its
    discretization together, not state-vector shape alone.* Regression:
    `tests/solvers/test_coupled_newton.py`.
    (2026-07-22 Round 6)
12. **The stamped, nonzero, previously green Fig. 3 ratio-zero baseline was a
    valid numerical reference** — **overturned on 2026-07-25.** Its shape was
    nearly correct, but its amplitude was an early-accepted root with
    pair-number error about `3.659e-3`. The corrected result is approximately
    a uniform `1.0036657949566×` rescaling and passes the number certificate
    near roundoff. The replacement 1620-bin solve completed in `10671.78 s`,
    was revalidated under its exact producer runtime
    (Python 3.14.3 / NumPy 2.5.1 / SciPy 1.18.0), and was promoted with
    After the final invalid-input guards conservatively advanced the source
    digest, the authenticated raw state was reassembled under the final
    equations while retaining the original producer identity. Current
    CSV/PDF/validation-record SHA-256 values are
    `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
    `7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
    and `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`.
    The raw payload did not persist finite-ratio `n_ph`; validation therefore
    reconstructs the unique Ph0 affine fixed point implied by each saved
    `f(E)` and does not claim to authenticate the producer's original stored
    phonon residual independently. *Lesson: a baseline is evidence only for the quantities
    independently certified at the stored snapshot; nonzero shape, hashes,
    and prior green runs do not certify a null or slow mode.*
13. **The 49-hour Fig. 3 wall time was intrinsic to the physics/full grid** —
    **overturned.** Two algorithm-policy costs dominated: an unnecessarily
    late scalar number-mode route repeatedly formed 1620×1620 dense Newton
    systems, and high-ratio continuation used 5% plain Picard with Anderson
    disabled. Routing-only scalar polish and safeguarded depth-3 Anderson
    preserved the independent return gates and reduced the measured full-grid
    wall time from `177440.15 s` (49.29 h) to `10671.78 s` (2.96 h), about
    `16.6×`. *Lesson: before accepting an extreme simulation time as physical
    complexity, profile solver routing and fixed-point iteration counts, then
    require same-seed/certificate A/B evidence for an acceleration.*
14. **An end-of-run regression comparison is adequate for a multi-hour
    continuation sweep** — **overturned.** The old test spent 49.29 hours
    before its first curve comparison exposed a ratio-zero mismatch. Target
    callbacks now validate each completed paper ratio immediately, and opt-in
    content-bound checkpoints preserve each continuation state. *Lesson:
    validate the cheapest decisive invariant at the earliest durable
    boundary.*
