# qpsim current status — AI-agent handoff

> **2026-07-25 live Round-7 update:** work is on branch
> `codex/audit-round7-fixes`, based on `cd4fb81`. The replacement `NE=1620`
> Fig. 3 producer completed all 14 continuation steps in `10671.777 s`
> (2 h 57 m 51.8 s), versus 49.29 h on the superseded pathological route.
> Its authenticated raw state was subsequently reassembled and recertified
> under the final current equations, preserving producer digest `522539fc…`
> and recording validated digest `34fd48de…`. Current
> CSV/PDF/validation-record SHA-256 values are
> `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
> `7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
> and
> `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`.
>
> All four Fischer-2024 families were freshly regenerated again after the
> final source-digest change and promoted as strict-v3 CSV/PDF pairs. Their
> focused collection passes `69` tests with `4` slow deselections. Exact
> current-source Fig. 7 completed all 48 fresh targets under hardened identity
> `82ef6da816fedbe89d6920b51cdcbd3d1dabe40d8b265f38bbaf997d0639f320`
> and solve digest
> `5d66e4de331acaa73c1d190e71b40cb05c503789efbd94f84ee4d9ec37d86502`.
> The six-worker campaign took `3642.094 s` wall time (`13292.818 s`
> aggregate worker time) and transactionally promoted CSV, PDF, and
> attestation hashes `3298d00b…8628`, `3ad21536…b9b`, and
> `a5d31ac4…357`. Maximum QP, QP-number, and representability-aware phonon
> backward errors were `3.701e-10`, `8.006e-9`, and `9.687e-9`, all below
> the `2e-8` gate. Its focused suite passed `74` tests with `2` slow
> deselections; independent attestation and visual inspection passed.
> The raw direct-form phonon diagnostic reached `0.429269` where the exact
> bath correction is below one binary64 ULP; it is retained as a
> representability diagnostic, not used as the acceptance metric.
>
> The final consolidated non-slow aggregate passes **2188 tests with
> 0 failures**, `18` intentional deselections, and `12` warnings in
> `716.22 s`. Hosted CI remains separate post-push evidence.
>
> Fig. 3 now persists an amplitude-sensitive pair-number certificate and
> distinct producer/validated solve-contract records. Finite-escape validation
> reconstructs the unique affine Ph0 root implied by stored `f`; this proves
> current-equation root membership for that reconstructed pair, not the
> producer's original omitted finite-ratio `n_ph`. The promoted validation
> record states that qualification explicitly.

> **Historical 2026-07-22 working-tree update (superseded by the banner
> above):** this file intentionally remains the
> historical handoff for pre-fix tree `71c5f02`; it is not the status of the
> current `codex/audit-fixes-2026-07-19` working tree. The original
> 123-agent audit, four external-review rounds, and the current Round-6
> repair are recorded in
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md), and overturned
> adjudications are tracked in
> [`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md). Round 6
> is still uncommitted, but all five affected numerical artifact families
> (CSV/PDF pairs) have been regenerated through their real solve paths and
> pass their strict currentness readers. The authoritative post-regeneration
> default collection passed: **1866 passed, 17 deselected, 13 warnings in
> 657.40 s**. The current Fig. 7 regeneration covered all 48 independent
> targets (six workers, 3421.3 s wall time; 13582.0 aggregate worker-seconds)
> under solve-contract digest
> `71d2730e43b41fba106e3066de156eb6d4f69a23ea4aac91c64de6bd552d0503`;
> maximum QP/phonon backward errors were `3.70072e-10` and `9.82942e-9`
> against the `2e-8` limit. The historical tables below still describe
> `71c5f02`; do not quote their old green counts as evidence for this working
> tree. The exact refinement slow gate passed in 4906.90 s, and the transient
> slow battery passed 4/4 in 850.77 s. The Fig. 5 high-drive and reduced Fig. 6
> slow checks pass; the two legacy baselines remain documented xfails. A
> durable commit and completion of the full 1620-bin Fig. 3 pinned-baseline
> slow node were still pending. That node was restarted on 2026-07-22 in the
> original checkout and remained active at that historical update. Its prior
> interrupted run's durable log preserves a certified full-grid `0.1` target
> (QP backward error
> `1.218e-16`, scaled phonon backward error `1.321e-6`) and successful
> continuation returns at `0.3` and `0.5`; those partial checkpoints are not
> represented here as a completed baseline test.

> **Historical 2026-07-22 Round-7 isolated-working-tree update (superseded by
> the banner above):** a further audit found
> and repaired issues in gap/threshold physics, WebUI/backend contract parity,
> public complex-input validation, overflow-safe spatial diffusion, Picard
> controls, and campaign artifact promotion/resume binding. Those edits live
> on uncommitted branch `codex/audit-round7-fixes`, based on `cd4fb81` plus the
> copied Round-6 tree; see the Round-7 section in
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md). The active Fig. 3
> process was deliberately left in the original checkout, so it is not
> Round-7 verification. Focused regressions are green, and the final
> un-rebound default collection produced **1922 passed, 5 failed, 17
> deselected, and 13 warnings in 545.55 s**. The five failures are exactly the
> strict Fig. 7/F24 source-provenance checks; bypass-only validation retained
> all payload/certificate checks and passed, and an exact reachability audit
> found the Kaplan endpoint numerically inert on all five artifact paths.
> Final static gates are clean. Exact regeneration of Fig. 7 and the
> summary-only F24 artifacts, source-honest handling of full-state F24
> re-certification, hosted CI, and a durable commit remain pending.

> **2026-07-19 follow-up (historical and subsequently superseded):** an
> independent 123-agent audit initially reported that it had confirmed the
> core engine and this document's claims, but found four
> distinct high-severity defect classes in the `scripts/` campaign
> drivers and the paper-anchor/M25 layer (an earlier tally said five;
> see the errata in the fixes doc). All are fixed on branch
> `codex/audit-fixes-2026-07-19`. Four later external-review rounds and Round
> 6 overturned parts of that initial conclusion — see
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md).
> The tables below describe the pre-fix tree `71c5f02`.

Snapshot date: **2026-07-19**
Audited code head: **`71c5f02310db0d65e7a9aa0bc5e09a4034d97bf3`**

Read the working-tree banner above first, then use
[`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md) for the live branch.
The remainder of this file is the compact historical `71c5f02` handoff, not a
replacement for the evidence in
[`AUDIT-2026-07-15-numerical-software.md`](AUDIT-2026-07-15-numerical-software.md).

## Scope of the completed work

The completed review was a **code-only numerical-software audit**. It was not a
paper audit. Papers were consulted only where necessary to check that an
implemented code path used the physics model, normalization, or operating mode
that it claimed to use.

Do not infer paper agreement from a passing figure-baseline test. Most figure
CSVs are self-pinned qpsim regression artifacts, not independently digitized
paper data.

## Repository state

| Item | Historical snapshot value |
|---|---|
| Remote | `Soren-O/Quasiparticle-Physics-Simulation` |
| Base | `origin/main` at `b92571a635aee9e8efd3fc228ac4ac7a7e69c150` |
| Audit branch | `codex/qpsim-deep-audit-fixes` |
| Audited code head | `71c5f02310db0d65e7a9aa0bc5e09a4034d97bf3` |
| Pull request | [Draft PR #5 — Fix qpsim numerical correctness and validation](https://github.com/Soren-O/Quasiparticle-Physics-Simulation/pull/5) |
| PR state at snapshot | Open, draft, merge state `CLEAN` |
| Exact-head CI | [Run 29667989929](https://github.com/Soren-O/Quasiparticle-Physics-Simulation/actions/runs/29667989929), green on Python 3.13 and 3.14 |
| Branch size | 8 commits ahead and 0 behind; 150 files changed; 26,623 insertions and 6,425 deletions relative to `origin/main` |
| Python | 3.13+; CI covers 3.13 and 3.14 |

The branch has **not been merged to `main`**. Before this handoff document was
created, the branch and remote were synchronized and the worktree was clean.
Treat `71c5f02` as the exact code/test/validation tree for the results below; a
later commit containing only this document does not alter that numerical tree.

## What the audit branch changes

This is a broad numerical repair branch, not a one-bug patch. The main repaired
contracts are recorded as findings N1–N43 and include:

- one matched finite-volume measure for BCS singular capacity, support,
  coherence factors, collision terms, photon partners, spatial transport,
  remapping, and observables;
- deterministic, branch-anchored nonequilibrium gap solving and fail-closed
  support coverage for self-consistent and moving-gap states;
- a stage-constrained moving-gap ETD2 update whose public occupation and gap
  satisfy the same algebraic constraint;
- raw residual and gain/loss backward-error certificates for Newton, coupled
  Newton, and Picard-facing steady-state paths;
- structured handling of coupled-Newton line-search failure and
  superconducting gap collapse, without converting unrelated failures into
  fallback states or silent `NaN` output;
- balance-preserving stiff collision stepping and fail-loud retry behavior;
- stricter grid/domain validation for collision operators, external sources,
  direct gap observables, and spatial transport;
- cache, provenance, schema, and atomic-write protections for validation
  artifacts;
- cross-platform deterministic validation policy: CI pins OpenBLAS, OMP, and
  MKL to one thread and uses measured OS-family envelopes only where exact
  Windows/Linux agreement is not justified;
- adversarial tests for unsupported grids, nonlinear pseudo-roots, non-finite
  derived measurements, singular-edge omissions, malformed inputs, and stale
  artifact metadata.

The final follow-up commit additionally fixes:

- the Fischer 2023 Fig. 5 high-drive thermal pseudo-root by tightening the
  inner Newton and final Picard/certificate contracts;
- Fischer 2023 Fig. 6 direct-mode grid coverage, crossing reconstruction,
  structured failure propagation, signed plotting, and canonical-file
  no-clobber behavior;
- the self-consistent diffusion benchmark's discrete fixed-point calibration,
  edge reconstruction, iteration cap, invalid-depth checks, and zero-capacity
  drift handling;
- roundoff-scale alignment of direct BCS gap-edge support;
- exact-source Fig. 7 regeneration and source-honest Fischer 2024
  regeneration/re-certification (producer and validator identities kept
  distinct);
- Windows CP1252-safe executable messages in the four Fischer 2024 generators,
  guarded by normalized-AST tests so numerical expressions remain unchanged.

## Historical `71c5f02` verified state

| Gate | Exact evidence |
|---|---|
| Local default aggregate | **1549 passed, 17 deselected, 4 warnings in 525.03 s** |
| Collection inventory | 1566 tests: 16 non-manual `slow` tests and one `manual_slow` test |
| Hosted Python 3.13 default | **1549 passed, 17 deselected, 4 warnings in 302.96 s** |
| Hosted Python 3.13 slow | **14 passed, 2 expected xfailed, 1550 deselected, 1 warning in 2235.62 s**; job wall 47m19s |
| Hosted Python 3.14 default | **1549 passed, 17 deselected, 4 warnings in 246.92 s** |
| Hosted Python 3.14 slow | **14 passed, 2 expected xfailed, 1550 deselected, 1 warning in 2140.25 s**; job wall 44m08s |
| Static/tooling gates | `ruff check .`, `mypy qpsim` over 75 source files, `compileall`, all seven bundled symbolic-verification scripts, and diff checks passed |
| Fischer 2023 Fig. 7 | Exact 48-target recertification completed in 1123.7 s; all axes, observables, and certificate arrays were bitwise identical to the active rows |
| Fischer 2024 artifacts | Exact current-solver recertification covered 84 states in 95.07 s; all certificates passed; maximum pinned/live row drift `2.58e-14` |
| Final focused slice | 82 passed, 2 deselected; Fig. 6 fast file 45 passed, 2 deselected; diffusion feedback 10 passed; gap suppression 18 passed |

The Fischer 2023 Fig. 7 solve-contract digest active at historical head
`71c5f02` is:

```text
ebe1382d509f6c52f11bca95b8d0161a211c4002a59f38de942cb2aefd193165
```

The only annotation on the exact-head hosted run is GitHub's non-failing
Node.js-20 deprecation warning for `actions/checkout@v4` and
`actions/setup-python@v5`; it is not a qpsim test failure.

The four default-suite warnings are expected: one Starlette dependency warning
and three explicit high-energy-tail diagnostics. The two slow xfails are the
intentional Fischer 2023 Fig. 5 pre-v2 artifact quarantine and the Figs. 9–13
uncertified legacy-artifact quarantine.

## Figure and numerical-validation status

| Surface | Historical snapshot status plus superseding notes |
|---|---|
| Fischer 2023 Fig. 3 | **Corrected strict-v3 replacement promoted and current-equation recertified.** The source-frozen `NE=1620` producer completed all 14 continuation steps in `10671.777 s`; the preserved raw state was later reassembled under the final equations with distinct producer/validator identities. Current CSV/PDF/validation-record hashes are `1f9250…60c8`, `7e38be…5812`, and `680454…371`. The finite-ratio `n_ph` omission qualification is explicit in the validation record. This is a fixed-grid regression, not paper parity. |
| Fischer 2023 Fig. 5 | **Quarantined.** The tested `T_star/Delta=0.60` forward/reverse split was a loose-tolerance pseudo-root; direct/forward/reverse now agree within 0.046%. The full sweep still needs tight-contract regeneration and refinement. |
| Fischer 2023 Fig. 6 | **Qualified and not closed.** The historical 66-point sweep certified its declared loose solver contract, but the default suppression pin is tighter than that state contract justifies. The repaired fixed-gap/direct path produces the strictly certified signed full-grid value `-0.0168056838447`, with QP backward error `1.619e-11` and certified phonon error zero; there is no full direct canonical or refinement result yet. |
| Fischer 2023 Fig. 7 | **Exact current-source regeneration complete and promoted.** All 48 targets completed under hardened identity `82ef6da8…f320` and solve digest `5d66e4de…6502` in `3642.094 s` wall time. CSV/PDF/attestation hashes are `3298d0…8628`, `3ad215…b9b`, and `a5d31a…357`; strict readback, independent attestation, and visual inspection passed. Maximum gated QP/QP-number/representability-aware phonon errors were `3.701e-10`/`8.006e-9`/`9.687e-9`. The raw phonon diagnostic `0.429269` records sub-ULP direct-form representability loss and is not the gate. This remains a scoped fixed-grid regression, not bitwise portability or paper parity. |
| Fischer 2023 Figs. 9–13 | **Quarantined.** Low-power `Q_i` is still nonconverged: the aligned `NE=3240 -> 6480` rung changes by 4.44368%. Existing evidence does not justify rewriting the photon operator. |
| Fischer 2024, four families | Freshly regenerated through their real solve paths and promoted as strict-v3 CSV/PDF pairs; the focused collection passes 69 tests with 4 slow deselections. These are independently certified **qpsim-native** regressions at paper topology. Analytic paper-target overlays remain incomplete. |
| Diffusion feedback benchmark | The default `NE=24`, `NX=201`, 10% well now has a discrete fixed point and certifies its raw map. The bounded guard plateau and fixed edge-node reconstruction are intentional parts of this benchmark contract. |
| Moving-gap integration | Verified second-order only within its documented ideal-BCS, uniform-work-grid, spatially homogeneous DAE and support domain. |

## Open work and residual risk

The historical `71c5f02` snapshot had no known failing tests or merge
conflicts. The final Round-7 non-slow aggregate passes 2188 tests with
0 failures, 18 intentional deselections, and 12 warnings. Fig. 3, Fig. 7,
and all four F24 pairs are current and promoted; hosted CI is separate
post-push evidence. The important remaining items are
numerical-qualification work, not hidden green checkmarks:

1. **Fig. 5:** run a full tight-contract regeneration and commensurate-grid
   refinement campaign before promoting any replacement baseline.
2. **Fig. 6:** run the complete repaired fixed-gap/direct-observable production
   sweep, investigate/refine the signed negative low-drive result, and define a
   replacement pin only after observable convergence is demonstrated. Running
   the old loose-contract `manual_slow` wrapper does not close this item.
3. **Figs. 9–13:** continue commensurate refinement. The audit proposes—without
   claiming a derived error budget—`<=1%` maximum `Q_i` change on two
   consecutive rungs plus `<=0.25%` exact-cell/FV observable discrepancy before
   promotion.
4. **Fig. 3:** keep quantitative claims in scalar moments and weak measures
   unless the ideal-BCS threshold model is explicitly regularized; strong-norm
   threshold layers remain slowly convergent.
5. **Fischer 2024:** paper-target analytic overlays are still incomplete even
   though the qpsim-native fixed-grid regressions are certified.
6. **Roadmap:** T2/T1 electronic backends and Ph1/Ph2 phonon transport are not
   implemented. These are roadmap gaps, not regressions introduced by this
   branch.
7. **Orthogonal manuscript question:** `papers/qp-diffusion` still has contested
   physics items documented elsewhere. Passing its symbolic scripts proves the
   code/manuscript identities agree; it does not prove the manuscript physics.
   Do not turn the next code audit into a paper audit unless explicitly asked.

## Behaviors that are intentional

Do not “fix” these without new evidence:

- Direct gap integrals raise when the energy grid does not cover the
  superconducting edge. Only roundoff-sized positive face offsets are aligned.
- Finite signed Fig. 6 direct suppression values are retained and plotted on a
  signed scale. Only explicit superconducting collapse maps to `NaN`; other
  numerical failures propagate.
- `solve_gap` warns near `T_c` when the grid cannot represent below-gap
  occupation. This is a declared domain limitation.
- The phonon pair-breaking kernel uses the current gap, not the zero-temperature
  gap, under its documented approximation.
- Spatial Crank–Nicolson transport now subcycles under-resolved diffusion
  steps automatically and fails loudly only when the required substep count
  exceeds its explicit safety cap.
- The default pytest configuration excludes `slow`; CI runs
  `slow and not manual_slow` separately.
- M25 steady-state acceptance uses a row-wise source-scaled residual plus a
  backward-error gate. Replacing it with a bare absolute threshold would
  reinstate an already-fixed conditioning error.
- `validation/sweep_cache.py` intentionally excludes downstream observables
  from the Fig. 7 **solve-source** digest. Artifact/dependency tests cover the
  downstream contracts separately.
- Canonical Fig. 6 output and direct-mode output are distinct. Programmatic or
  CLI direct generation must use the `_direct` CSV/PDF paths.
- A Windows/Linux regression envelope is an OS-family calibration, not a claim
  of hardware-independent bitwise identity.

## Documentation hierarchy and known doc debt

Use the documents in this order:

1. [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md) — live branch repair and verification record.
2. [`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md) — adjudication ledger, including overturned verdicts.
3. [`CURRENT-STATUS.md`](CURRENT-STATUS.md) — this historical `71c5f02` branch/CI handoff; use its opening banner for current evidence.
4. [`AUDIT-2026-07-15-numerical-software.md`](AUDIT-2026-07-15-numerical-software.md) — full original findings, derivations, timings, hashes, and unresolved-risk evidence.
5. [`NEXT-AUDIT-BRIEF.md`](NEXT-AUDIT-BRIEF.md) — concise rabbit-hole and false-positive guide.
6. [`STATUS.md`](STATUS.md) — broad project/gate history, including work outside this audit.
7. [`../ENGINE-REVIEW-NOTES.md`](../ENGINE-REVIEW-NOTES.md) — integration history and older timing context.

Known documentation debt:

- `README.md` still marks Fischer paper-reproduction parity and the Layer-4
  chain with unqualified checkmarks. The qualified status above and in the
  numerical audit is authoritative.
- `.github/workflows/ci.yml` still describes the manual Fig. 6 sweep as roughly
  14 hours. Measured rows total 6.04 hours serial or 2.11 hours concurrent, and
  the old wrapper still would not close the repaired direct-observable question.
- Older hosted counts of 1513 tests belong to earlier exact trees. The current
  `71c5f02` hosted count is 1549.
- `ENGINE-REVIEW-NOTES.md` retains an explicitly historical 802-pass command;
  its opening disclaimer is important.
- Focused test counts overlap. Do not add them together to manufacture an
  aggregate larger than the collected suite.

## Recommended first pass for the next agent

1. Confirm identity before drawing conclusions:

   ```bash
   git status -sb
   git rev-parse HEAD
   git diff --stat origin/main...HEAD
   gh pr checks 6
   ```

2. Read the audit outcome, fixed-findings table, verification table, and
   explicit unresolved-risk section. Do not start with the historical audits.
3. Independently review the highest-risk invariants in:

   - `qpsim/physics/spectral.py`
   - `qpsim/physics/gap_equation.py`
   - `qpsim/observables/gap_suppression.py`
   - `qpsim/solvers/coupled_newton.py`
   - `qpsim/solvers/etd.py`
   - `qpsim/backends/t3_diffusion.py`
   - `qpsim/backends/t3_spatial_1d.py`
   - `validation/fischer_2023/`
   - `validation/fischer_2024/`

4. Prefer adversarial invariant checks over reproducing plots: represented
   support, matched measures, raw residuals, gain/loss backward error,
   conservation, branch continuity, cache keys, and fail-loud boundaries.
5. If a change touches solver-source provenance, do not casually rewrite a
   canonical CSV. Recompute into a temporary path, independently certify it,
   read it back through the strict artifact reader, compare rows/certificates,
   and promote only source-honest evidence. Fig. 7 and summary-only F24
   artifacts require exact regeneration. A full-state re-certification may
   record a later validator identity, but it must preserve the producer
   identity and must not be described as current-algorithm execution.

## Reproduction commands

Install with Python 3.13 or 3.14:

```bash
pip install -e ".[dev,ui]"
```

Use one BLAS thread for comparable numerical evidence:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

PowerShell equivalents are `$env:OPENBLAS_NUM_THREADS="1"`,
`$env:OMP_NUM_THREADS="1"`, and `$env:MKL_NUM_THREADS="1"`.

Core gates:

```bash
ruff check .
mypy qpsim
python -m compileall -q qpsim scripts tests validation
make -C papers/qp-diffusion verify PY=python
pytest -q
pytest -q -m "slow and not manual_slow"
```

Useful focused gates:

```bash
pytest -q tests/observables/test_gap_suppression.py
pytest -q validation/diffusion_operators/test_self_consistent_feedback.py
pytest -q validation/fischer_2023/test_fig6_paper.py
pytest -q validation/fischer_2024
```

The full Fig. 6 `manual_slow` wrapper is intentionally excluded from CI and is
not the right first next step: its historical contract is already known to be
too loose for the pinned observable.
