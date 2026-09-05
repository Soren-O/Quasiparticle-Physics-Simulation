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

> **This ledger is not the authoritative live-status page.** See
> [`CURRENT-STATUS.md`](CURRENT-STATUS.md) and
> [`STATUS.md`](STATUS.md) for the active producer/gate state. This ledger
> may retain completed hashes, timings, and test counts when they are evidence
> for an adjudication, but volatile process state must not be read as current.

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
  artifact/dependency tests. This exclusion applies only to downstream
  observables: every dependency used directly by a cache or publisher must
  still be enumerated in that layer's source manifest. (2026-07-15; boundary
  clarified 2026-07-27)
- **Canonical Fig. 6 vs `_direct` output paths should be unified** →
  *Deliberately distinct.* Direct/diagnostic mode must never overwrite a
  canonical bundle member. (2026-07-15; boundary clarified 2026-07-27)
- **Default `pytest` excludes slow live numerical comparisons** → *By
  design* — fast artifact/currentness gates and M25 regressions remain in the
  default suite; CI runs `-m "slow and not manual_slow"` as its own step (and
  a guard test pins that step's existence). (2026-07-13; scope corrected and
  guard noted 2026-07-27)
- **`density.py` strict `f ∈ [0,1]` gate breaks on roundoff-negative
  occupations** → *Not reproduced in shipped paths.* The direct callers do
  not all clip locally, as an earlier version of this entry claimed; they
  consume solver states whose acceptance/projection contracts already enforce
  the occupation domain. Re-open if a caller bypasses those contracts or a
  returned state contains an out-of-domain value. (2026-07-19; evidence
  wording corrected 2026-07-28)
- **A below-2Δ photon must have `loss_rate == 0`** → *Refuted physical
  expectation* — the number-conserving scattering channel remains physical
  at sub-threshold photon energies. Only pair generation/recombination is
  gated by the shared tolerance-aware, strictly-above
  `pair_channel_open` predicate. (2026-07-20 round 3; collar wording
  corrected 2026-07-27)
- **R7-CI-TIMEOUT / refuted / the existing 180-minute hosted slow-step
  timeout was already too short** → *Cross-host timing extrapolation, not an
  observed CI failure; historical exact-head evidence only.* At exact hosted
  head `cd4fb81`, GitHub Actions run
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
  routing-only A/B to the comparison precision. Those timings are historical
  source-frozen Round-7 evidence; Round-8 changes to `fig3_solve.py` and
  `newton_steady_state.py` automatically reopen current-tree performance and
  branch equivalence, while the current return-certificate tests retain the
  structural no-bypass claim. Re-open if
  the routed candidate ever bypasses a return certificate or the
  independently measured Fig. 3 shape floor changes. (2026-07-25 Round 7
  follow-up; current-tree A/B pending 2026-07-27)
- **R7-FIG3-AA-BRANCH / refuted on the 81-bin diagnostic / safeguarded
  Anderson necessarily changes the continuation branch** → *Not observed in
  the measured scope.* Historical plain Picard took `314.2773 s`; depth-3
  safeguarded Anderson took `17.9573 s` (`17.5×`, 94.29% less wall time).
  Target full-array maximum differences divided by the corresponding legacy
  peak were `[0, 1.727e-7, 3.745e-6, 7.176e-9]` for ratios
  `[0, 0.1, 1, 10]`, and both paths passed the independent certificates. This
  did **not** itself substitute for the later completed 1620-bin result.
  Round-8 solver/continuation changes reopen current-tree A/B verification;
  retain these values only as historical evidence from the recorded Round-7
  tree. Re-open on grid/ladder/mixing changes or any full-grid
  branch/certificate disagreement. (2026-07-25 Round 7 follow-up;
  current-tree A/B pending 2026-07-27)
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
  **Round-8 follow-up:** the physics-drift verdict remains refuted, but this
  entry did not mean every F24 reader had adopted the stated contract. The
  shared full-state `bind_certificate()` path still required producer stamps
  and fresh reassemblies to agree at `rtol=1e-12`, so canonical Fig. 5 and
  Figs. 5–7 readback could fail solely when the BLAS thread count changed.
  That separate reader bug is now fixed: stored and reassembled certificates
  must each satisfy the advertised bounds independently, and the reader
  returns the fresh measurement without treating diagnostic last bits as a
  portable observable. Regression:
  `validation/fischer_2024/test_artifact.py`.
  (2026-07-25 Round 7 hosted follow-up; reader-wide correction 2026-07-27)
- **R8-M25-DENSITY-MARKERS / narrowly refuted /
  `chemical_potentials_kelvin` should reject NaN failed-point markers or the
  exact-zero logarithmic boundary** → *Intentional plotting/sweep-alignment
  contract, not a solver-input loophole.* Shipped sweep callers use NaN to
  retain a failed point's temperature coordinate; exact zero maps to the
  mathematical `−inf` log boundary. This exception does **not** cover finite
  negative densities: no shipped caller uses them as markers, and silently
  mapping one to NaN can hide an upstream bug. The low-level solver rejects
  negative states, while branch/multi-seed acceptance requires positive
  densities. Scope is only the inversion helper's NaN/zero behavior; do not
  generalize it to solver states or other observables. Finite-negative input
  handling was reopened for correction on 2026-07-27.
- **R9-FIG6-FROZEN-PHONON-UNITS / refuted / the giant qpsim phonon residual
  on the exact Fischer Fig. 6 author state proves a unit, prefactor, or
  formula defect** → *Foreign-root cancellation false inference.* With the
  author grid, constants, coherence, and pair-frequency labels held fixed, a
  qpsim-free transcription reproduces the captured author QP channels to
  about `3e-14` of turnover, and at the captured A1 frozen state the
  author-equivalent phonon accumulator's residual is about `7.0e-11` of
  turnover. The author state is a stiff
  cancellation root with approximately `8.19e5 s^-1` of turnover.
  Re-evaluating it after changing several discretization conventions at
  once—and replacing the attachment's legacy
  `k_B/e = 86.1733034152 µeV/K` with qpsim's
  `86.17333262145 µeV/K`—does not isolate a prefactor or formula error. The
  prefactor/unit conversion agrees when evaluated under common conventions.
  This adjudication does **not** dismiss the real left-edge versus
  finite-volume coherence difference or the pair-frequency
  `2 Delta + (i+j)h` versus `2 Delta + (i+j+1)h` difference. Re-open the
  unit/prefactor claim only if a common-convention, channel-level comparison
  fails; do not infer it from a native qpsim residual at the foreign author
  root. A subsequent same-seed staged-resolve pilot also reproduced the
  captured author state to `3.951e-16` full-state relative L2 and converged
  all primary variants with QP residual/turnover approximately
  `1.3e-15`–`2.1e-15` and phonon residual/turnover approximately
  `6.2e-12`–`1.60e-11`; the independently re-solved author control is the
  `1.596e-11` endpoint, distinct from the frozen captured-A1 diagnostic. This
  further rules out a unit/prefactor inference from a poorly converged or
  different control root at this point.
  (2026-07-29 author-first audit and staged pilot)
- **R9-FIG6-AUTHOR-JACOBIAN / refuted / a finite-difference mismatch proves
  the staged author-control Jacobian is wrong and should be corrected** →
  *Authenticated historical-numerics trap.* The supplied source has three
  real residual/Jacobian inconsistencies: the photon residual omits
  terminal-bin transitions whose derivatives remain in the matrix, photon
  off-diagonals use the partner occupation rather than the row occupation,
  and the phonon-pair `D_n N` diagonal uses two shifted occupation indices.
  A mathematically consistent derivative silently changes the authors'
  Newton path and is not an exact author-control replay. The staged producer
  now pins these matrix entries to an executable oracle bound to the
  authenticated source hashes. Re-open only if the staged matrix differs
  from those source-bound entries; do not “repair” the author-control branch.
  A corrected Jacobian can be studied only as a separately labelled solver
  substitution. (2026-07-29 staged-pilot adversarial review)

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
  by contrast, is gated by the shared tolerance-aware, strictly-above
  `pair_channel_open` predicate; physical sub-gap K+ scattering remains
  active. The pair-block omission was a real bug and is fixed. (2026-07-20
  rounds 3–4; collar wording corrected 2026-07-27)
- **Fischer-2023 Fig. 6 author-to-qpsim discretization attribution remains
  staged work:** one exact full-resolution author point matches the digitized
  paper trace. With the attachment's exact binary64 `Delta_0`, the direct-gap
  observable reproduces the driven gap, independently reconstructed thermal
  gap, and ordinate bit-for-bit. The earlier one-ULP statement was a
  validation-script parameter leak (`180.0e-6` instead of the author's
  `180*10**-6`), not observable disagreement. The first substantive operator
  differences are localized to author left-edge versus qpsim finite-volume
  coherence and to the `+h` pair-frequency center shift; a third grid-level
  substep separately isolates the much smaller native-micro-eV binary64 DOS
  arithmetic difference. The formal qpsim-free C0 port verifies every
  Newton transition and reproduces the A1 full state to `3.951e-16` relative
  L2; formal C1 reproduces both gaps and the ordinate bit-for-bit. Formal C2
  independently recomputes all 124 frozen parameter/operator arrays and
  measures the fixed-`n_bar` Eq. 35 coordinate shift from
  `0.33990789737294363` to `0.3399503360830364`, but deliberately does not
  solve a changed root or ordinate. A same-seed single-point pilot has
  re-solved the grid changes in four Newton iterations:
  author control
  `0.12090908988993258`; coherence only `0.12070758916263027`
  (`-0.00020150072730`); pair label only `0.14590106106941977`
  (`+0.02499197117949`); and both `0.14570562776829468`
  (`+0.02479653787836`); and C3c native-DOS arithmetic
  `0.14570561703489288` (`-1.07334e-8` from C3b). Coherence and native-DOS
  arithmetic are negligible at this point. The pair-label
  change is material but moves upward, away from promoted qpsim
  `0.08967258`, so none of the three localized changes explains qpsim's lower
  ordinate.
  Do not promote the pilot into a formal C3 ordinate: it used author
  parameters and a 1620-cell coefficient carrier, not the accepted C2
  endpoint and live 1640-cell grid. Formal C3 was subsequently completed as a
  separate frozen differential: parent cell `i` maps to child `i+20` with no
  interpolation, all active C2b5 channels reproduce bit-for-bit at the
  projection control, and independent verification reassembles the
  C3a/C3b/C3c channels on the true grid. The evidence distinguishes
  roundoff-sized mapped-left-face differences from the real `+0.5 micro-eV`
  author-left-edge-to-qpsim-center carrier shift, whose observable effect is
  reported separately. Like C2, formal C3 claims no changed root or
  ordinate. Formal frozen C4 and C5 were subsequently completed; C6–C7 and
  the full 300-point author replay remain incomplete. (2026-07-30
  author-first audit, staged pilot, and formal frozen C3/C4/C5)
- **Marchegiani strict pins are win32-stamped; ubuntu CI runs the 1e-3
  fallback** → accepted trade-off, documented in
  `validation/baselines/README.md`;
  a Linux-stamped twin was considered and declined. (user-adjudicated
  2026-07-20)
- **Fig. 7 solve uses fixed Δ₀ and one grid for all temperatures** →
  documented convention at `_build_grid` (~0.17% gap mismatch at 0.34 K);
  the pin is a self-consistent regression under that convention, and
  re-gridding is a baseline-moving change. (2026-07-20 round 3)
- **In-file artifact hashes are currency/integrity checks, not authenticated
  tamper defense.** Regardless of schema version, they catch accidental stale
  or corrupted payloads, but a writer who deliberately changes data and
  recomputes the in-file SHA-256 can forge a self-consistent stamp. Do not cite
  those hashes alone as provenance against a malicious or mistaken re-pin.
  (2026-07-19 audit, scope made schema-independent 2026-07-27)
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
- **R7-FISCHER-NUMBER-SCHEMA / rollout completed in the Round-8 working
  tree.** The
  independent Fischer certificate detects the cold amplitude/null mode.
  Fig. 3 persists `f` and distinguishes the producer solve contract from the
  later validator contract; finite-escape validation reconstructs the affine
  phonon fixed point implied by stored `f`, not the producer's omitted original
  `n_ph`. The locally promoted F23 Fig. 5 v3 and Fig. 6 v2 working-tree
  canonicals persist complete `f/n_ph` state and independently reassemble the
  number certificate. This is not yet a claim about a durable pushed commit.
  Fig. 7 is the intentional exception: summary-v2 persists authenticated
  producer certificate assertions but omits solved state, requires explicit
  opt-in, and cannot claim reader-side reconstruction. Fabricating fields or
  treating a metadata-only rebind as numerical evidence remains a defect.
  Re-open if any future schema loses its stated state/provenance distinction.
  (2026-07-25 Round 7 follow-up; rollout completed locally 2026-07-28 Round 8)
- **R8-FISCHER-ROW-DTYPE / open campaign-archive validation gap.** The frozen
  Fig. 5/6 resumable-row readers coerce raw NPZ arrays to `float` before all
  type checks, so complex, boolean, or integer payloads can lose their original
  dtype. This is not evidence that solver-produced rows are corrupt; current
  promotion requires a separate pre-coercion, per-field dtype-schema check of
  every row. Add fail-closed dtype validation and adversarial regression cases
  in the next provenance-breaking regeneration. (2026-07-28 Round 8;
  cross-reference §3.28)
- **R8-FIG6-READER-RECERT-DUPLICATION / open source-frozen efficiency
  gap.** The current public `read_baseline()` and `read_baseline_metadata()`
  each replay the complete 66-state artifact and then call promotion-record
  validation, which replays it again. The fast configuration preflight and
  signed-diagnostic publisher no longer compose those APIs repeatedly: the
  preflight authenticates bytes/config/axes/stored certificate columns and a
  separate `slow` test performs the full replay; the diagnostic locks both
  output resources plus the canonical publication tuple and passes one
  authenticated result through all internal commit-marker checks. Refactor the
  public reader to return one
  artifact-plus-record snapshot at the next provenance-breaking revision.
  This is a validation-runtime/lock-contention defect, not evidence that the
  promoted numerical states are wrong. (2026-07-28 Round 8;
  cross-reference §3.30)
- **R8-FIG5-PUBLISH-RECERT-DUPLICATION / open source-frozen efficiency
  gap.** The Fig. 5 campaign publisher validates the same assembled 81 states
  five times across row assembly, campaign validation, CSV writing, staged
  readback, and final readback. After the sixth row became durable, those
  passes added `504.201 s` before final status publication. The default test's
  separate two-replay composition has been fixed by a scalar fast preflight
  plus one explicit slow recertification, but deduplicating the publisher
  requires a provenance-breaking revision that preserves rollback and
  currentness guarantees. This is unnecessary computation, not evidence that
  the promoted states are wrong. (2026-07-28 Round 8; cross-reference §3.31)

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
   mutation, incoherent/non-finite ancillary backend state, and overflow/NaN in a
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
8. **The WebUI all-underflow heatmap branch was unreachable because the spatial
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
    (Python 3.14.3 / NumPy 2.5.1 / SciPy 1.18.0), and was promoted. After
    the final invalid-input guards conservatively advanced the source
    digest, the authenticated raw state was reassembled under the final
    equations while retaining the original producer identity. At the
    2026-07-25 Round-7 snapshot, CSV/PDF/validation-record SHA-256 values were
    `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
    `7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
    and `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`.
    The raw payload did not persist finite-ratio `n_ph`; validation therefore
    reconstructs the unique phonon affine fixed point implied by each saved
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

    **Round-8 evidence scope for items 15–29:** verified on the publication
    candidate on branch `codex/audit-round7-fixes`, based on `6d3c512`; its
    durable revision is the commit containing this ledger. Hosted CI remains
    separate post-push evidence. Unless an item states a narrower trigger,
    reopen it when its cited validator, publisher, solver, or evidence-scope
    contract changes. The principal
    regression families are `validation/fischer_2023/test_fig{3,5,6,7}_paper.py`,
    `tests/validation/test_fig7_promotion.py`,
    `validation/fischer_2024/test_artifact.py`,
    `validation/marchegiani_2025/test_artifact_contract.py`, and
    `tests/validation/test_transient_artifact_io.py`.

15. **"No solver-vs-paper analytical quantitative test exists" was a false
    positive because the Eq. 53 layer provides one** — **overturned on
    2026-07-27.** The Eq. 53 tests validate the analytical helper against a
    separately transcribed formula and float64 pins. The numerical Fig. 6
    curve and analytical curve are each compared with their own
    qpsim-generated baseline columns; no test compares those curves with each
    other at shared coordinates. The same separation holds across the
    Fischer figure suite, and no checked-in test compares a numerical curve
    with digitized paper data. *Lesson: formula-transcription coverage,
    numerical self-regression, and solver-vs-reference agreement are three
    distinct validation layers; evidence for one must not be relabeled as
    another.*
16. **The vacuous Fig. 5 low-drive `atol=1e-6` was an acceptable permanent
    limitation** — **overturned; validator fixed and state-bound v3 canonical
    locally promoted in Round 8.** The old floor exceeded the entire
    low-drive signal and could accept
    amplitude collapse. The current comparison uses `rtol=5e-3`,
    `atol=1e-30`, and the v3 contract persists `f/n_ph` for independent
    certificate reassembly. The six-row 81-point campaign, canonical
    promotion, exact-dtype row audit, one-pass full recertification, and visual
    inspection all passed. This closes a working-tree qpsim regression, not
    commensurate-grid refinement or paper-data parity. *Lesson: an absolute
    tolerance must be calibrated against the smallest claimed signal, not the
    largest curve in the panel.*
17. **A `%PDF` prefix, a large byte count, and terminal `%%EOF` prove a plot is
    valid** — **overturned in Round 8.** Token-shaped junk passed those checks.
    The audited F23/F24/M25/transient canonical figure-bundle validators now
    require structural xref/catalog/page parsing, exactly one page, and a
    semantic paint/text mark beyond Matplotlib's initial white canvas fill.
    This rejects the reproduced blank-page failure but remains an
    accidental-blank heuristic, not proof that a curve is correct or agrees
    with a paper; `/Producer` metadata is not treated as evidence. *Lesson:
    magic bytes and nonempty page streams are file-structure hints, not plot
    validation.*
18. **Atomic replacement of each output file makes a multi-file evidence
    bundle transactional** — **overturned in Round 8.** Readers could observe
    mixed CSV/PDF/record generations, and a failing concurrent publisher could
    roll back over a successful one. The audited F23/F24/M25/transient
    canonical publishers use one OS lock, staged members, semantic
    recertification, and a manifest/record promoted last. A later audit also
    found direct F23 Fig. 5/6 writer entrypoints that could still target
    canonical members outside this path; those entrypoints now refuse
    canonical resolved paths. This statement does not cover unrelated
    campaign/development publishers. *Lesson: per-file atomicity does not
    define a bundle commit, and every public writer must share the same
    canonical-path policy.*
19. **Capturing source identity at publication time is sufficient for a
    multi-hour solve** — **overturned in Round 8.** A mid-solve edit could
    stamp an old in-memory result with the later source. The audited canonical
    F23/F24/M25/transient producers now freeze source/config/runtime before
    solving and recheck it before and throughout publication; authenticated
    cache/restart evidence remains explicitly distinguished from a fresh
    invocation. This proves stability only over the dependency closure
    actually enumerated by the manifest: F23 Fig. 6 directly used
    `sweep_cache.py` while omitting it from its artifact fingerprint, a later
    hole now closed by a regression test. It is not a claim about every
    script-level campaign driver.
20. **A reader need not participate in the publisher lock because the commit
    marker is promoted last** — **overturned in Round 8.** An unlocked reader
    can still span replacements and assemble a mixed snapshot. The audited
    F23/F24/M25/transient canonical readers now hold the same OS lock across
    data plus record/manifest authentication, and canonical identity is
    resolved-path based so an explicit spelling of the canonical path cannot
    bypass that contract. This does not generalize to unreviewed
    development-output readers.
21. **A paper raster beside a qpsim raster is a digitization/parity pipeline**
    — **overturned in Round 8.** The optional helpers perform no calibration,
    registration, curve extraction, uncertainty analysis, or score, and the
    repository ships no paper raster corpus. They are now labeled as manual
    visual aids and fail nonzero if they produce nothing. This does not make
    every paper-style renderer harmless: the promoted canonical Fig. 6 panel
    keeps the paper's finite axis window and therefore leaves 27 finite
    numerical and 26 finite Eq. 53 out-of-window samples invisible.
    The separate `scripts.render_fischer_fig6_signed_diagnostic` renderer
    shows every finite stored point, marks the paper window, and records
    clipping plus the count of nonfinite points that cannot be plotted,
    without claiming digitized-data parity.
22. **Individually hashed M25 CSV/PDF files were a complete artifact contract**
    — **overturned in Round 8.** Current canonical M25 readers and publishers
    are OS-locked,
    manifest-authenticated bundles, including when callers pass an explicit
    resolved canonical path. The numerical Fig. 3 branch-state bundles persist
    state for reader-side residual reassembly; the Eq. 8 crossover is
    reassembled from its closed form. Fig. 4 remains honestly summary-only:
    reading its producer assertions requires explicit opt-in and returns that
    evidence scope instead of fabricating reader certification.
23. **A projected exact vacuum returned by threaded LAPACK is automatically an
    absorbing physical root** — **overturned in Round 8.** Historically, a
    threaded-LAPACK dense Newton step could overshoot a tiny positive
    occupation to exact zero, and the line search accepted that nonabsorbing
    vacuum. The regression deterministically forces the equivalent
    overshooting Newton direction; it does not attempt to reproduce LAPACK
    scheduling. The solver now backtracks such a projection unless the
    assembled state is genuinely absorbing.
24. **A structurally valid sparse transient curve independently certifies the
    integrator trajectory** — **overturned in Round 8.** Stored snapshots can
    validate their domain, thermal seed, ordering, monotone response,
    reconstructed stored `x_qp` values, endpoint proximity, and the separately
    stored steady state. They cannot prove which ETD2 substeps produced the path
    between snapshots. The slow live test executes the current `run()`
    implementation and compares its stored snapshots, so it exercises the
    omitted interval; it still does not persist or independently authenticate
    every internal ETD2 substep. *Lesson: snapshot semantics, live-path
    execution, and substep-level dynamics provenance are distinct evidence
    layers.*
25. **The Fischer Fig. 3/5/6 `tau_0^PB` diagnostic independently extracts
    the paper's approximately 255 ps lifetime** — **overturned in Round 8.**
    The diagnostic first constructs the phonon-side kernel with the paper
    input, then inverts a sink assembled from that same kernel. Full-grid
    probes at input lifetimes `0.123`, `0.255`, and `0.510 ns` returned
    `0.122829344535`, `0.254646202086`, and `0.509292404171 ns`:
    the constant ratio `0.998612557199` is a discretization/threshold
    quadrature effect. This is a useful normalization round-trip, not an
    independent parameter recovery or paper-data comparison. The frozen
    Fig. 3/5/6 solve modules still print or describe this as a phonon-side
    extraction that reproduces the paper's ≈255 ps value; correcting that
    source wording is deliberately deferred because it participates in the
    newly promoted source fingerprints. Correct it in a coordinated
    provenance-breaking regeneration rather than immediately making the
    accepted bundles stale. *Lesson: trace every alleged extracted quantity
    back to its inputs; inversion of an operator normalized by the target
    value is a self-consistency check.*
26. **Finite persisted Fig. 3 occupation columns are automatically physical**
    — **overturned in Round 8.** The CSV writer, reader, and staged plot path
    accepted any finite value, including one binary64 ULP below zero or above
    one. The post-solve validation/publication paths now enforce the exact inclusive Pauli domain
    `[0,1]`; exact boundary values remain valid, and failed writes preserve an
    existing artifact. A follow-up audit then found that premature
    `dtype=float` coercion discarded an imaginary component before the domain
    check; the validator, writer, plot path, and reader now reject complex
    occupations explicitly, including a smallest-subnormal imaginary part;
    the CSV text reader instead rejects complex spellings as nonnumeric.
    Both repairs changed only the post-solve publication layer, so the final bundle was
    republished from the unchanged authenticated raw solve instead of
    relabeling or rerunning it. *Lesson: finiteness, shape, and provenance do
    not imply a physical value domain; reject complex data before coercion and
    test the nearest representable real and imaginary violations.*
27. **A sidecar that hashes a PDF automatically authenticates every render
    claim in that sidecar** — **overturned in Round 8.** The Fig. 6 signed
    diagnostic originally authenticated the PDF bytes and recomputed marker
    counts, but a reviewer could change the recorded PCHIP request/count
    without changing the PDF and the reader still accepted the pair. The
    renderer now embeds a commitment to the complete render-evidence object
    and plotting/numerical runtime in the PDF metadata; the reader recomputes
    the expected PCHIP count from the canonical samples and requires the PDF
    commitment to match. This authenticates pairing and the renderer's stated
    intent, not the semantic artist count inside the PDF: the reader does not
    decode the page back into markers/curves, so visual inspection remains a
    separate closeout step. *Lesson: a data-file hash binds bytes to a record,
    not arbitrary descriptive fields in that record; claims not derivable from
    the data need an explicit commitment, and a commitment is not itself
    semantic render inspection.*
28. **Physical-domain checks after `np.asarray(..., dtype=float)` preserve the
    type integrity of a resumable row archive** — **overturned in Round 8.**
    The frozen Fig. 5/6 campaign readers convert raw NPZ members before their
    value checks, so a complex member can lose its imaginary component and
    boolean/integer members can be normalized into apparently ordinary
    reals. This is a real generic validator gap, but not evidence that the
    live campaigns contain such payloads: Round-8 closeout separately required
    every raw archive member to match its declared dtype before accepting the
    campaigns. State/axis/certificate arrays are real `float64`,
    while explicitly integral metadata such as Fig. 5 `num_bins` remains
    non-boolean `int64`. All three Fig. 6 rows and all six Fig. 5 rows passed
    the exact pre-coercion schema, hash, size, producer, and certificate
    reauthentication checks.
    Pre-coercion rejection, including
    smallest-subnormal-imaginary and boolean/integer regression cases, is
    intentionally deferred to the next provenance-breaking regeneration
    rather than invalidating an otherwise authenticated in-flight solve.
    *Lesson: validate dtype and complexness before coercion; a validator's
    synthetic acceptance gap does not by itself prove that a specific,
    independently type-checked artifact is corrupt.*
29. **A tiny ambient-environment recertification drift proves a campaign row
    is corrupt** — **refuted for the current Round-8 rows, while exposing a
    real reader-portability defect.** Fig. 6 row authentication intentionally
    uses a near-bitwise certificate equality under its recorded eight-variable
    single-thread environment. Rechecking the first completed row under a
    different BLAS thread setup changed one normalized certificate by
    `2.916e-11` relative (`7.69e-19` absolute), only about `7.7e-14` of the
    `1e-5` scientific gate; exact-environment revalidation passed. The false
    inference is that this negligible reduction-order drift invalidates the
    row or its physics. The genuine issue is that the frozen public Fig. 5/6
    canonical readers reuse producer-exact equality under the caller's ambient
    environment and emit only a generic mismatch. Closeout and CI now use all
    eight controls; the next provenance-breaking revision must split strict
    resume-row identity from portable canonical semantic recertification.
    *Lesson: do not weaken producer identity because another environment
    differs, and do not mistake producer identity tolerance for a portable
    scientific acceptance tolerance.*
30. **Calling several authenticated Fig. 6 readers is cheap defense in
    depth** — **overturned in Round 8.** Each frozen public reader performs
    full state-derived recertification, and `read_baseline()` plus
    `read_baseline_metadata()` each trigger another replay through promotion
    validation. The advertised fast preflight therefore replayed all 66
    states four times: the first closeout run took `336.06 s` before reporting
    two stale-test-schema failures. A promotion-locked scalar preflight now
    checks exact artifact identities, current fingerprint/generation evidence,
    axes, and stored certificate columns; the two corrected checks complete
    in `5.2 s`, while
    `test_canonical_bundle_authenticates_and_recertifies` remains the separate
    full `slow` gate. The signed-diagnostic publisher likewise reuses one
    authenticated snapshot for staging and promotion, but now rebinds that
    snapshot under the canonical lock and locks both output resources so
    overlapping PDF/sidecar publications cannot race; external readers still
    re-certify once. This removes redundant computation, not independent
    evidence. The duplicate replay inside the source-frozen public reader
    remains open as documented in §2. *Lesson: count expensive validators
    transitively; repeated calls can serialize identical evidence rather than
    strengthen it.*
31. **Repeated Fig. 5 validation calls are cheap defense in depth** —
    **overturned in Round 8.** The advertised fast preflight composed
    `read_baseline_metadata()` and `read_baseline()`, replaying all 81 states
    twice (`160.88 s` test body). A promotion-locked scalar preflight now checks
    exact identities, current fingerprint, metadata, axes, table hash, and
    stored certificate gates in `1.71 s`; one explicit slow full replay takes
    `82.58 s`. Separately, source tracing showed five full validation passes in
    the publisher itself, measured as `504.201 s` of post-solve overhead. The
    fast-test composition is fixed; publisher deduplication remains open in §2
    because changing the frozen source now would stale the accepted campaign.
    *Lesson: map validator calls transitively and preserve one authenticated
    snapshot across stages instead of recomputing identical evidence.*
32. **The Fischer/Catelani canonical CSVs are scraped or digitized paper
    curves** — **refuted in Round 8, while exposing a real missing validation
    capability.** Repository-wide source and manifest inspection found no
    tracked paper raster/data oracle, figure scraper, OCR/digitizer,
    pixel-to-axis calibration, or quantitative curve-alignment score.
    `rasterize_baselines.py` and `validation/rasterize_pdf.py` rasterize qpsim
    PDFs; `make_comparison.py` and `make_isolated_comparison.py` merely place a
    caller-supplied paper image beside a qpsim image. The canonical CSVs are
    qpsim-generated states/summaries whose manifests bind qpsim source,
    configuration, runtime, and output—not DOI/version/page/panel/crop or
    extracted paper coordinates. Their passing gates establish regression,
    provenance, and discretized-equation consistency, plus a few broad manual
    anchors; they do not establish paper-curve parity. The false positive is
    treating those CSVs as hidden scraped evidence. The genuine open work is a
    versioned paper-data pipeline with source/crop hashes, calibrated extracted
    points and uncertainties, explicit units/normalizations, preregistered
    interpolation/error metrics, and manifests binding the independent oracle
    to the qpsim result. *Lesson: “paper topology” is not paper data, and
    side-by-side pixels are not a quantitative validation oracle.*
    **Round-9 update:** the premise remains false for every canonical
    qpsim-generated CSV. The missing capability has now been implemented
    separately for Fischer-2023 Fig. 6 under `validation/paper_data/`; it does
    not retroactively turn the canonical baseline into scraped data.
33. **A passing digitized analytic overlay validates the corresponding
    sampled numerical branch** — **refuted in Round 9.** In the first independent
    Fig. 6 paper-data score, all three dashed Eq. 53 controls agree within
    `0.388` normalized raster uncertainty or better. That is strong evidence
    for the axis calibration, color identity, curve mapping, and formula
    transcription. Yet all three solid numerical curves fail by `7.59–9.13`
    normalized units, with maximum relative discrepancies around `33–39%`
    over seven sampled points on the visible rising branch
    (`T*/Delta ≈ 0.250–0.410`); unsampled curve regions remain uncharacterized.
    Treating the dashed control as a proxy for the solid result would have
    hidden the central finding. *Lesson: use analytic traces as controls for
    the digitization and transcription path, then score numerical traces
    separately; common axes and good control agreement do not establish
    numerical parity.*
34. **qpsim's public sub-gap photon operator materially disagrees with the
    frozen C3c author-form photon loss** — **refuted by formal C4.** The
    apparent large mismatch comes from comparing different return semantics:
    qpsim returns a loss-rate coefficient, while C3c stores physical loss.
    The valid comparison is
    `loss_s_inv = loss_rate_ns_inv * frozen_f / 1e-9`. After that conversion,
    the full photon net differs by only about `2.02e-15` symmetric relative
    L1. A real but numerically tiny endpoint-policy difference remains:
    qpsim includes the representable child-cell pair `1619 <-> 1639`, while
    the authenticated author residual omits transitions touching its final
    QP cell. The two net contributions are about `-2.88825e-35` and
    `+2.88858e-35 s^-1`, far too small at this frozen state to explain the
    Figure 6 discrepancy. *Lesson: compare physical gain/loss terms, not a
    rate coefficient to an already occupation-weighted loss; isolate genuine
    endpoint semantics from floating-point operation ordering.*
35. **Formal C5 shows that qpsim's QP-phonon scattering gain/loss disagree
    materially with the author-form operator, and the nonzero pair number
    moment violates conservation** — **refuted by like-for-like bookkeeping
    and process-specific conservation.** The author source-order scattering
    gain and loss buckets both include the same Pauli cross-term
    `n f_i f_j`; public qpsim removes it from both. The raw public-minus-parent
    gain/loss L1 differences (about `1.006e-4 s^-1` each) are therefore
    bookkeeping differences that cancel from the physical net, not changed
    scattering physics. After subtracting the shared term from the author
    buckets, the like-for-like gain/loss L1 differences are only
    `9.50321362825228e-14` and `1.9342844642219452e-13 s^-1`; the physical
    scattering net agrees to `5.682685376326191e-16` symmetric relative L1.
    The pair number moment is intentionally nonzero because pair breaking and
    recombination create and destroy two quasiparticles. It is retained as a
    diagnostic (`-0.1738186684181618 s^-1 micro-eV`, relative about
    `0.0321`), not subjected to the zero-drift gate used for
    number-conserving scattering. *Lesson: compare like-for-like physical
    buckets, and apply conservation gates only to channels that conserve the
    measured quantity.*

36. **`qpsim/services/transient.py:327` — the early-stop finite-difference
    veto loosened by `a47bad3`** — flagged post-hoc as a change that
    "should have been held back", and left in place only because reverting
    it would advance the source digest. **Adjudicated 2026-08-11: keep the
    change; do not revert.** The old veto tested *smallness* (any bin
    `<= 32*eps` distrusts the finite difference), which has no physical
    meaning for occupations — every cold thermal state has most of its tail
    below that, so the fallback rate was pinned at `inf` on every step and
    `stop_tol` could never fire. Measured on a 100 mK thermal state, NE=40:
    35 of 40 bins sit at or below `32*eps` (min `f = 2.1e-54`), so the old
    veto was permanently armed and the documented early-stop feature was
    structurally dead for all mK states. The new veto tests *saturation*,
    which is the correct complementarity condition: `f >= 1-32eps` always
    disqualifies (hidden blocked gain `G(1-f)` is unbounded) and an active
    low clip disqualifies, but an untouched `f=0` bin hides nothing, because
    under the public contract the RHS at `f=0` is `gain >= 0` (`ExternalFlux`
    hard-rejects negative gain). On a real FD-routed run (1e-6 above-gap
    kick, `dt=100 ns`, `stop_tol=1e-10`) the new veto converges at
    `t=6100 ns`/61 steps where the old logic never converged through the
    full 20000 ns horizon, and the exact raw `max|df/dt|` at that stop,
    evaluated independently through `apply_collisions_with_diagnostics`, is
    `9.788e-11 /ns <= stop_tol`. The veto gates only the *stop decision*:
    the old-veto trajectory truncated at the new stop time is bitwise
    identical to the new stop state, and the default `DiffusionBackend`
    (the exact-residual path every shipped caller uses) stops at the
    identical step. Pinned by
    `tests/review_2026_08_03/test_P15.py::TestFiniteDifferenceCertificate`.
    *The legitimate residue is a process failure, not a code failure:
    `a47bad3`'s message claimed behaviour-neutrality that this hunk does not
    have — it is behaviour-changing on the custom-backend FD-fallback path.
    Lesson: "behaviour-neutral" must be asserted per hunk against the
    reachable call paths, not per commit.*

37. **"The recombination phonon source is 2× too large" — the sum-lattice
    unordered-pair double count** — measured as exactly
    `2.000000000000` (fifteen digits, three grid resolutions, two drive
    amplitudes, so structural rather than discretisation) when the
    quasiparticle energy-loss moment
    `cell_weights @ (E * (gain - loss*f))` is compared against
    `dE[0] * (omega @ a_ph)`, the phonon energy measure that balances
    *exactly* for scattering. The mechanism is real and visible at
    `qpsim/collisions/phonon.py:461-470`: the phonon source bincounts the
    full `(i, j)` matrix. On the **difference** lattice used by scattering,
    `(i, j)` and `(j, i)` are distinct events — an emission and an
    absorption — so the plain sum is correct. On the **sum** lattice used by
    recombination they are the *same* event, so the plain sum counts every
    unordered pair twice. The engine is right; the naive measure is not.
    With the factor of one half the ledger closes to `1.5e-15` on every
    grid tested. Corroborating evidence that this is a measure convention
    rather than a defect: the C6 evidence bundle compares this channel
    against the author's own implementation and finds `9.2e-3` symmetric
    relative L1, not a factor of two, and
    `docs/REVIEW-2026-08-03-HELD-BACK.md` records the Kaplan S₊ ledger
    closing to `3.2e-14` in exactly the uncorrected configuration that
    produces the 2× under the naive measure. Both halves are now pinned in
    `validation/analytic/test_minimal_models.py::TestClosedLoopEnergyLedger`
    — the halving is asserted load-bearing for recombination *and* asserted
    absent for scattering, so it cannot be dropped or copied onto the wrong
    lattice. *Lesson: an exact small integer ratio in a heavily audited
    kernel is far more likely to be a counting convention than a bug. Before
    filing, check whether an existing author-comparison or ledger already
    measures the same quantity and gets a different answer — if it does, the
    disagreement is in the measure, not the code.*
