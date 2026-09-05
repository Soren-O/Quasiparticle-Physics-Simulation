# Engine review — integration notes

This file began as historical integration context for branch
`fix/gpt-review-engine`. Current code-audit work is on
`codex/qpsim-deep-audit-fixes`; the dated counts below remain historical and
must not be read as results for later follow-up edits.

## 2026-07-18 numerical-audit correction

- Fig. 5's tested forward/reverse split at `T_star/Delta=0.60` is not evidence
  of a physical branch ambiguity. Tightening inner Newton backward error to
  `1e-10` brings direct/forward/reverse `NE=1620` values within `0.046%`. The legacy pin remains
  quarantined pending full tight-contract regeneration/refinement.
- Fig. 6's measured production-row times sum to `21726.6965 s` (`6.04 h`)
  serial, versus `7599.292171 s` (`2.11 h`) concurrent; the old 14-hour estimate
  below is superseded. More importantly, the default self-consistent observable
  is tighter than its accepted gap-map state justifies. The repaired fixed-gap/
  direct path has a strictly certified full-grid point, but its corrected
  low-drive observable is negative and no full canonical exists, so the old
  wrapper is not closure.
- Figs. 9--13 remains quarantined through an aligned `NE=6480`, `-100 dBm`
  point (`4.44368%` rung). Observable variants and an overlap-aware photon
  prototype do not justify rewriting the photon operator.
- The self-consistent diffusion-feedback benchmark now represents the direct
  gap closure down to its `0.5*Delta_0` floor. Its seed is calibrated at the
  requested fixed point, and the raw map raises if it does not converge within
  64 iterations instead of returning a truncated or unconverged well. Its
  intentional guard plateau uses fixed edge-node reconstruction, so the
  advertised 10% default well no longer loses its root at a cell-face stencil
  jump; analytic drift ignores zero-capacity cells just as transport does.
- Fischer 2023 Fig. 5/6 and all four Fischer 2024 generators keep executable
  console, exception, and CLI-help text ASCII-safe, so a long Windows CP1252 run
  cannot finish a solve and then fail while reporting it. Source-level AST
  regressions guard those paths without constraining plot typography.

## Slow-validation integration gate

This branch adds a slow-validation CI step and now also contains the fixes needed
to make that step safe for pull requests:

1. **Fig. 5 exposed a real Picard false-convergence regression.** Commit `f041a85`
   floored every Picard denominator at `1e-3 * max(n_ph)`. Large low-energy phonon
   occupations could therefore hide tiny but dynamically decisive above-gap bins,
   and the solver incorrectly stopped on the thermal branch at high drive. The
   convergence test now uses the standard per-bin condition
   `abs(delta) <= atol + rtol * max(abs(old), abs(new))`. The occupation-space
   `picard_atol` is explicit and independent of the inner Newton collision-
   residual tolerance. A two-endpoint, full-grid Fig. 5 regression test protects
   the high-drive branch. Reverting `1c5af1a` alone does **not** restore it;
   reverting `f041a85` does.

2. **Fig. 5 and Fig. 7 baselines have been regenerated** from the corrected
   solver. Their sweep axes and metadata are unchanged. Rendered plots were also
   compared with the published Fischer–Catelani figures: the curve families,
   knees, ordering, and scales agree qualitatively.

3. **The full Fig. 6 sweep is now `manual_slow`.** Pull-request CI runs
   `pytest -m "slow and not manual_slow"`, so the measured roughly 6.04-hour serial full-resolution
   regression cannot stall the gate. The automated slow step has a 180-minute
   timeout. Fig. 6 remains runnable explicitly with
   `pytest -m "slow and manual_slow" validation/fischer_2023/test_fig6_paper.py`.

The authors' source in `PhysApplPaper_Figure_6/examples/Figure_6.py` uses a fixed
kinetic gap, ten coupled Newton iterations, and evaluates the driven gap directly
from the converged quasiparticle distribution. qpsim's `--direct-gap` path encodes
those semantics. Guard-cell-invariant edge reconstruction, a narrowly scoped
strict Picard fallback, paired mode flags, and signed-observable retention now
make that path runnable and certified at the diagnosed full-grid point. Its
corrected negative low-drive value remains a paper-parity/refinement warning; a
complete direct-gap campaign is required before replacing the tolerance-limited
self-consistent-gap baseline.
Direct-mode generation now derives `_direct` CSV/PDF paths from its public
arguments, so programmatic use is as no-clobber-safe as the CLI.

## Original engine change set (13 files, all verified)

- **[P1] `solve_gap`** (`qpsim/physics/gap_equation.py`): widen the bracket search
  to the Debye cutoff instead of giving up after 5 steps; a colder-than-thermal
  population near T_c now returns the true gap (~182 µeV) instead of an
  Δ_eq underestimate. Behavior-preserving except in the previously-failing fallback.
- **[P1] lint**: fix the demo scripts (drop unused noqa, `itertools.pairwise`) and
  pin `ruff==0.15.17` so a new ruff release can't redden CI on unchanged code.
- **[P2] webui path traversal**: `_safe_segment` guards every slug/run_id path join
  in `qpsim/webui/store.py`; `qpsim/webui/server.py` maps the `ValueError` → 404.
  Closes the Windows `..%5C` (encoded-backslash) arbitrary read. + regression tests.
- **[P2] `spatial` backend**: reject degenerate/descending meshes and non-positive
  `interface_conductance` in `_validate_state`. + tests.
- **[P2]** `sympy` added to the `[dev]` extra; the `pytest -m slow` CI step.
- **[P3] `picard`**: reject `mixing ∉ (0, 1]` (`mixing=0` falsely "converged").

## Integration-gate additions

- Replace the global peak-scaled Picard floor with a per-bin absolute-plus-relative
  convergence test, expose the occupation-space `picard_atol` separately from
  `newton_tol`, and add focused unit and full-grid Fig. 5 regression coverage.
- Regenerate the corrected Fig. 5 and Fig. 7 CSV/PDF baselines.
- Add the `manual_slow` marker, exclude the full Fig. 6 sweep from PR CI, and cap
  that automated step at 180 minutes while preserving an explicit manual command.

## Reproduce green

```
pip install -e ".[dev,ui]"        # Python 3.13 or 3.14
ruff check .                       # All checks passed
mypy qpsim                         # Success
pytest tests/solvers/test_picard.py tests/backends/test_spatial.py \
       tests/webui/test_server.py qpsim/physics/gap_equation.py -q   # 50 passed
pytest -q                                                         # 802 passed
pytest -m "slow and not manual_slow"                               # 14 passed; fig6 excluded
```

The slow surface was executed in visible shards on macOS/Python 3.14: the Fig. 5
high-drive regression took 80.23s, its full uncached baseline check took 37m25s,
both Fig. 7 guards took 3m25s, and Fig. 3 took 35m52s (mostly its bounded
ratio-10 coupled-Newton endpoint). The remaining nine tests took 4m03s. These
timings are useful when setting the CI job timeout; they do not include the
manual Fig. 6 sweep.

_(This is branch-scoped integration context — safe to delete at merge.)_
