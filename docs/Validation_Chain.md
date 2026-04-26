# Validation Chain

Inventory of test and validation tiers under `tests/` and
`validation/`. The canonical reference is `New Framework Plan.md` §6;
this document maps the tiers onto the live test directories.

## Tier 1 — Analytic fixed points (`validation/analytic/`)

Identities that must hold at thermal equilibrium or at known
parameter limits. Fast, opt-in (not slow-marked).

- `test_detailed_balance.py` — e-ph, sub-gap photon, and pair-breaking
  photon channels each vanish at `(f = f_FD(T), n_ph = n_BE(T))` to
  roundoff on the active window.
- `test_mattis_bardeen_thermal.py` — `σ_1 → 0` as `T → 0`;
  `σ_2 → π Δ/ω` (kinetic-inductance limit).
- `test_gap_equation_equilibrium.py` — `solve_gap(f_FD(T_B))` recovers
  `Δ_eq` from `calibrate_gap(T_c, T_B)`.

## Tier 2 — Tier reductions (`validation/tier_reductions/`)

Structural placeholder for T1 → T2 → T3 reductions when those
backends ship. Empty in v1.

## Tier 3 — Paper reproductions

Pinned against self-checked CSV baselines and PDF plots under
`validation/baselines/{ph0_constant, ph0_kaplan, transient,
marchegiani_2025}/`. Each module has a paired `test_*.py` that
re-runs the baseline at registration time and asserts at the
documented tolerance tier.

### Fischer 2023 (`validation/fischer_2023/`)

| Figure | Module | Tolerance |
|---|---|---|
| Fig 3, τ_l = 0 | `fig3_tau_l_zero.py` | 1e-12 (bit-identical) |
| Fig 3, finite τ_l (ratios 0.5 / 1 / 2 / 5 / 10) | `fig3_finite_tau_l.py` | 1e-6 (iterative); ratio 10 via coupled Newton |
| Fig 5, x_qp vs T* | `fig5_xqp.py` | 1e-6 |
| Fig 6, gap suppression | `fig6_gap_suppression.py` | self-pinned, sc-gap + acoustic-escape τ_l |
| Fig 7, Q_i(T_B) thermal | `fig7_qi_vs_t.py` | 1e-6 |
| Fig 7, Q_i(T_B) with drive | `fig7_with_drive.py` | 1e-6 (via `nbar_loop`) |
| Figs 9–13, Q_i(P_read) | `figs_9_13_qi_vs_pread.py` | 1e-6 (via `nbar_loop`) |

### Fischer 2024 (`validation/fischer_2024/`)

| Figure | Module | Tolerance |
|---|---|---|
| Figs 5–7, f(E) | `figs_5_7_fe_pb.py` | 1e-6 |
| Fig 8, x_qp(T_B) | `fig8_xqp_pb.py` | 1e-6 |

### Marchegiani 2025 (`validation/marchegiani_2025/`)

| Figure | Module | Status |
|---|---|---|
| Eq. 8 Lambert-W T̄ | `fig3_crossover_temperature.py` | closed-form, machine precision |
| Fig 3, μ_α vs T (small + large gap asymmetry) | `fig3_chemical_potentials.py` | both panels match qualitatively; quantitative within ~5% of paper at the cited reference T |
| Fig 4, Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T | `fig4_parity_rates.py` | qualitative trends pinned at rtol=5e-2; panel a has multi-stability noise from competing M25 fixed points (max-x_L branch picker). Paper-grade smoothness needs a proper bifurcation tracker — outstanding gap. |

### Transient (`validation/transient/`)

`photon_kick_response.py` — drives the ETD2 transient stepper from a
thermal initial state under a step photon kick; pins the resulting
`f(E, t)` snapshot CSV. **No paired regression test in v1** — flagged
as outstanding gap.

## Tier 4 — Unit tests (`tests/`)

Per-module tests mirroring the library layout. Run with
`pytest -q`. As of the seventh session: 542 pass, 9 deselected (slow
opt-in via `-m slow`).

## Slow tier (`pytest -m slow`)

Fischer reproductions at Fischer-scale grids. The Marchegiani sweeps
and the transient demo are fast and not slow-marked.

## See also

- `STATUS.md` — running gate tracker, current test count.
- `Part_II_Physics.md`, `Part_III_Numerics.md` — what's being
  validated.
