# Next-audit brief — qpsim repo state and rabbit-hole flags

Deliberately-lean starting point for a fresh code audit of `qpsim`. **Read this
first, then audit with independent eyes.** The dated `docs/AUDIT-2026-07-12-
integrated.md` and `docs/AUDIT-2026-07-13-reaudit.md` are historical findings
records: useful for context, but treat their conclusions as *done* and verify
independently rather than assuming the areas they touched are now flawless. This
brief exists so a fresh pass need not re-derive settled false alarms — not to
tell you what is or isn't a bug.

## Current state (2026-07-13)

- Active branch `codex/reaudit-fixes-2026-07-13` (open as PR #2); **not merged to main**.
- Green baseline: `pytest` (fast gate) **1060 passed / 15 slow deselected**;
  `ruff check` clean; `mypy qpsim` strict clean (73 files); the seven
  `papers/qp-diffusion` symbolic verifiers pass.
- Highest-churn areas (where a new regression would most likely hide — scrutinize):
  the M25 rate-equation acceptance layer (`services/rate_equation.py`), the
  moving-gap remap (`backends/t3_diffusion.py`), the BCS gap-edge quadrature
  (`physics/bcs_quadrature.py` + `observables/`), and spatial/webui robustness.

## Gate gotchas (a green run can still be hiding a break)

- The default `pytest` **deselects `slow`**. A numerical change can leave the
  fast gate green while breaking a self-pinned *slow* baseline. CI runs
  `pytest -m "slow and not manual_slow"` as a separate step — run that (or read
  CI) before trusting green.
- Fischer **Fig. 6** is `manual_slow` (~14 h). Excluded from CI; don't run it by accident.
- Fischer/Marchegiani figure CSVs are **self-pinned regression baselines**, not
  digitized paper data. Passing them proves code stability, *not* paper fidelity.

## False positives — look like bugs, are intentional (don't re-file)

- `solve_gap` is biased high near `T_c` because the energy grid can't sample
  below-gap occupation. This is a documented domain-contract limitation and it
  now **warns** — not a quadrature bug.
- The phonon-side pair-breaking kernel normalizes by the *current* Δ (not Δ₀) —
  a documented approximation (Δ ≈ Δ₀ at Fischer temperatures).
- `_tau_R_inverse` uses `Δ_R` — correct per arXiv:2408.17218 **v2** Eq. D11. (A
  "should be mean-gap Δ̄" claim came from a different version/label; false alarm.)
- TiN material `rho_F ≈ 3.8e28 eV⁻¹ m⁻³` — film/disorder-dependent, defensible;
  not a unit typo.
- M25 steady-state acceptance uses a **row-wise source-scaled + backward-error**
  residual gate, not a fixed tolerance. A bare `1e-14` or `1.0 Hz` reading is the
  *old* miscalibration (already fixed); the current gate is intentional.
- Spatial Crank–Nicolson transport **raises** on an under-resolved step
  (diffusion number `D₀·dt/dx²` too large) by design — fail-loud, not a crash to
  "fix". Keep that number ≲ 5 (validate_setup warns above ~8).

## Real gaps — tests won't catch these (worth examining)

- Transient self-consistent-gap **trajectory** accuracy is capped at ~0.1% by
  `solve_gap`'s edge interpolation of `f` (`_gap_integral_f`'s `np.interp`), NOT
  by the moving-gap remap (which is exact per update). Logical next fix: a
  cell-exact gap residual.
- `validation/baselines/ph0_constant/fischer_fig3_paper.csv` `f_ratio_10` column
  is **all zeros** (unphysical); `test_fig3_paper` compares it at `atol=1e-6`
  while the signal is ≤ 1.5e-8, so that assertion is **vacuous**. Green ≠ correct there.
- Known-minor, flagged not fixed: duplicate terminal snapshot (1 ulp apart) in
  transient/spatial output; webui `kind="step"` with equal gaps + `interface_G_N`
  fails at runtime instead of returning a 400; a run whose manifest write keeps
  failing stays undeletable until restart.
- **fig7 (fischer_2023) Picard chain is non-deterministic at low-occupancy
  points** (found landing PR #3, 2026-07-14/15): across repeated ubuntu CI runs
  of identical code, `run()` results at losses 1/Q_qp ≲ 1e-9 vary by up to
  ~2.5e-11 absolute loss (up to ~2.4% relative), and the 3.13/3.14 legs disagree
  with each other; points at loss ≥ 3e-8 reproduce to ≤ 2e-4. The pinned test
  now gates losses at rtol=1e-3 / atol=1e-10, which absorbs the noise, but the
  underlying run-to-run irreproducibility (thread-order/BLAS-reduction dependent
  accumulation in the Picard chain?) is an unexplained engine finding worth a
  root-cause pass.

## Orthogonal: paper physics

The bundled `papers/qp-diffusion` manuscript still has contested blocker physics
(B1–B5, M1). `verify_*.py` passing means the code **matches the manuscript**, not
that the manuscript is correct — that needs a human physicist. The engine work is
independent and could land without it.
