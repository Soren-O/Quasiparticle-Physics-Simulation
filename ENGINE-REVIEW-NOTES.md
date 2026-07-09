# Engine review — integration notes

Branch `fix/gpt-review-engine` (base: `main`). This is the **code half** of a
10-finding review; the **paper half** is on `fix/gpt-review-2026-07-05`. The two
change sets are independent. Everything here is verified: `ruff check .`,
`mypy qpsim`, and the touched-code tests pass (see "Reproduce green" below).

## ⚠️ Read before merging — the new slow-CI step exposes pre-existing issues

This branch adds a `pytest -m slow` CI step, which exercises the paper/slow
validation surface for the first time. That surface has **two pre-existing
problems that live in `main`, not introduced here**:

1. **Stale Fischer baselines — fig5 & fig7 fail** (fig3 and figs_9_13 pass). This
   is NOT a code regression and NOT platform noise (byte-identical results across
   machines). The numerical curves legitimately moved because of physics/solver
   corrections committed *after* the baselines were pinned (May 2026): confirmed
   contributor `1c5af1a` (removed a spurious 2× in `phonon_collision_rates`
   recombination/pair-breaking — reverting it recovers 6 of 14 deviating fig5
   points); leading suspect for the dominant deviation `f041a85` (steady-state
   Picard near-zero-bin fix). **The current curves are correct.**
   → **Fix = regenerate the fig5/fig7 pinned baseline CSVs. Do NOT change code to
   match the old baselines** (that re-introduces the bugs the corrections removed).
   Physics/decision call — needs a human.

2. **fig6 regression test hangs.** It did not complete on the origin machine
   (~7×-slow Windows / Python 3.14) across multiple multi-hour attempts; behavior
   on other platforms is unknown. A prior `fig6-reproduction-fixes` branch was
   already merged into `main`, yet the hang persists — so this is **unresolved**,
   needing a fresh convergence/iteration guard or a platform investigation, not
   coordination on that (already-merged) branch. Don't run `pytest -m slow` blind
   on an unknown platform.

**Recommended order:** regenerate the fig5/fig7 baselines and resolve or quarantine
(e.g. mark fig6 `xfail`/`skip` with a tracking note) the fig6 hang **first**, then
merge this branch — otherwise CI goes red or hangs on the new slow step.

## The change set (13 files, all verified)

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

## Reproduce green

```
pip install -e ".[dev,ui]"        # Python 3.13 or 3.14
ruff check .                       # All checks passed
mypy qpsim                         # Success
pytest tests/solvers/test_picard.py tests/backends/test_t3_spatial_1d.py \
       tests/webui/test_server.py qpsim/physics/gap_equation.py -q   # 50 passed
```

_(This is branch-scoped integration context — safe to delete at merge.)_
