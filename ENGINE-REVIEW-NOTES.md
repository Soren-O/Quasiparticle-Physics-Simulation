# Engine review — integration notes

Branch `fix/gpt-review-engine` (base: `main`). This is the **code half** of a
10-finding review; the **paper half** is on `fix/gpt-review-2026-07-05`. The two
change sets are independent. Everything here is verified: `ruff check .`,
`mypy qpsim`, and the touched-code tests pass (see "Reproduce green" below).

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
   `pytest -m "slow and not manual_slow"`, so the roughly 14-hour full-resolution
   regression cannot stall the gate. The automated slow step has a 180-minute
   timeout. Fig. 6 remains runnable explicitly with
   `pytest -m "slow and manual_slow" validation/fischer_2023/test_fig6_paper.py`.

The authors' source in `PhysApplPaper_Figure_6/examples/Figure_6.py` uses a fixed
kinetic gap, ten coupled Newton iterations, and evaluates the driven gap directly
from the converged quasiparticle distribution. qpsim's `--direct-gap` path encodes
those semantics and is fast on a reduced grid, but it does not yet reproduce the
published high-drive turnover reliably. Replacing the current self-consistent-gap
baseline therefore remains follow-up physics/numerics work, separate from merging
the engine fixes.

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
- **[P2] `t3_spatial_1d`**: reject degenerate/descending meshes and non-positive
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
pytest tests/solvers/test_picard.py tests/backends/test_t3_spatial_1d.py \
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
