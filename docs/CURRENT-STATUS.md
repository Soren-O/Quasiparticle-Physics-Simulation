# qpsim current status — AI-agent handoff

> **2026-07-19 follow-up:** an independent 123-agent audit of this tree
> confirmed the core engine and this document's claims, but found 5
> high-severity defects in the `scripts/` campaign drivers and the
> paper-anchor/M25 layer. All are fixed on branch
> `codex/audit-fixes-2026-07-19` — see
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md). The tables
> below describe the pre-fix tree `71c5f02`.

Snapshot date: **2026-07-19**
Audited code head: **`71c5f02310db0d65e7a9aa0bc5e09a4034d97bf3`**

Read this file first when taking over the repository. It is a compact handoff,
not a replacement for the evidence in
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

| Item | Current value |
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
- exact provenance rebinding after successful Fig. 7 and Fischer 2024
  recertification;
- Windows CP1252-safe executable messages in the four Fischer 2024 generators,
  guarded by normalized-AST tests so numerical expressions remain unchanged.

## Verified state

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

The active Fischer 2023 Fig. 7 solve-contract digest is:

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

| Surface | Current status |
|---|---|
| Fischer 2023 Fig. 3 | Promoted as a nonzero, independently certified `NE=1620` qpsim regression. Scalar mass and weak Wasserstein shape converge much faster than strong pointwise/capacity-L1 metrics near ideal-BCS photon thresholds. Do not claim pointwise continuum convergence or paper parity. |
| Fischer 2023 Fig. 5 | **Quarantined.** The tested `T_star/Delta=0.60` forward/reverse split was a loose-tolerance pseudo-root; direct/forward/reverse now agree within 0.046%. The full sweep still needs tight-contract regeneration and refinement. |
| Fischer 2023 Fig. 6 | **Qualified and not closed.** The historical 66-point sweep certified its declared loose solver contract, but the default suppression pin is tighter than that state contract justifies. The repaired fixed-gap/direct path produces the strictly certified signed full-grid value `-0.0168056838447`, with QP backward error `1.619e-11` and certified phonon error zero; there is no full direct canonical or refinement result yet. |
| Fischer 2023 Fig. 7 | Exact tight-contract 48/48 production evidence exists on Windows and Linux. The pin is a certified cross-platform regression under scoped OS-family envelopes, not bitwise portability or paper parity. BLAS thread variables are enforced in CI runtime policy but are not serialized into the CSV header. |
| Fischer 2023 Figs. 9–13 | **Quarantined.** Low-power `Q_i` is still nonconverged: the aligned `NE=3240 -> 6480` rung changes by 4.44368%. Existing evidence does not justify rewriting the photon operator. |
| Fischer 2024, four families | Promoted as strict-v2, independently certified **qpsim-native** regressions at paper topology. Pre-v2 files remain archived and rejected. Analytic paper-target overlays remain incomplete. |
| Diffusion feedback benchmark | The default `NE=24`, `NX=201`, 10% well now has a discrete fixed point and certifies its raw map. The bounded guard plateau and fixed edge-node reconstruction are intentional parts of this benchmark contract. |
| Moving-gap integration | Verified second-order only within its documented ideal-BCS, uniform-work-grid, spatially homogeneous DAE and support domain. |

## Open work and residual risk

There are no known failing tests or merge conflicts on the audited code head.
The important remaining items are numerical-qualification work, not hidden green
checkmarks:

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
- Spatial Crank–Nicolson transport raises on under-resolved diffusion steps; it
  is meant to fail loudly.
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

1. [`CURRENT-STATUS.md`](CURRENT-STATUS.md) — exact branch/CI handoff.
2. [`AUDIT-2026-07-15-numerical-software.md`](AUDIT-2026-07-15-numerical-software.md) — full findings, derivations, timings, hashes, and unresolved-risk evidence.
3. [`NEXT-AUDIT-BRIEF.md`](NEXT-AUDIT-BRIEF.md) — concise rabbit-hole and false-positive guide.
4. [`STATUS.md`](STATUS.md) — broad project/gate history, including work outside this audit.
5. [`../ENGINE-REVIEW-NOTES.md`](../ENGINE-REVIEW-NOTES.md) — integration history and older timing context.

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
   gh pr checks 5
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
   and only then decide whether a metadata rebind or numerical regeneration is
   justified.

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
python -m compileall -q qpsim tests validation
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
