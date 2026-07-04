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
| Fig 3, paper legend ratios 0 / 0.1 / 1 / 10 | `fig3_paper.py` | slow; 1620-bin paper grid + phonon-side Eq. 12 kernels |
| Fig 5, paper-topology x_qp two-panel | `fig5_paper.py` | slow; Eq. 47 + Appendix-E analytic overlay |
| Fig 6, paper-topology gap suppression | `fig6_paper.py` | slow; Eq. 53 overlay; (δΔ_T − δΔ)/δΔ_T ordinate |
| Fig 7, paper-facing Q_i,tot(T_B) | `fig7_paper.py` | slow; Tables II/III parameters + Eq. 65 |
| Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | 1e-4 (via `nbar_loop`); not a literal paper figure |

### Fischer 2024 (`validation/fischer_2024/`)

| Figure | Module | Tolerance |
|---|---|---|
| Figs 5–7, f(E) | `figs_5_7_fe_pb.py` | 1e-6 |
| Fig 8, x_qp(T_B) | `fig8_xqp_pb.py` | 1e-6 |

### Marchegiani 2025 (`validation/marchegiani_2025/`)

All sweeps run through the branch-continuation driver
(`qpsim.services.rate_equation.solve_rate_equation_branch`) on the
Γ̄-normalized density equations (2026-07-04); the historical
multi-stability noise was a conditioning artifact of the missing
`Γ̄ = Γ̃/N_CP(R)` normalization and is gone. All fast — the whole
directory runs in the default gate (~15 s).

| Figure | Module | Status |
|---|---|---|
| Eq. 8 Lambert-W T̄ | `fig3_crossover_temperature.py` | closed-form, machine precision |
| Fig 3, μ_α vs T (small + large gap asymmetry) | `fig3_chemical_potentials.py` | matches the published curves (paper-exact μ inversions, SI Eqs. S2–S5): panel a merged μ_α linear from 0.94 to 0 at T̄ ≈ 146 mK; panel b ordering μ_L ≳ μ_R> > μ_R< with the R-band merge at ~50 mK; smooth through the crossover |
| Fig 3, paper-styled panels + insets | `fig3_paper.py` | paper-faithful artifacts `m25_fig3{a,b}_paper.csv` / `m25_fig3_paper.pdf`; strict same-platform pin at rtol=1e-6 |
| Fig 4, Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T | `fig4_parity_rates.py` | smooth curves incl. the paper's low-T nonmonotonic dip (panel a); smoothness enforced (max adjacent |Δlog10 Γ_P| < 0.2); strict same-platform pin via robust comparison |
| Fig 4, paper-styled two-stack with comparison models | `fig4_paper.py` | paper-faithful `m25_fig4_paper.{csv,pdf}`: full model + global-quasiequilibrium reduction (density ratios per SI Note 1 last paragraph; total-density closure summing Eqs. 4–6; generation per Eq. 7) + renormalized global-QE (Fig. 4 caption parameters Γ^ph_00 = 600 Hz, ω_LR/2π = 6 GHz; large-asymmetry family only) + exp(−ω_10/T) dotted reference; strict pin rtol=1e-6 |

### Transient (`validation/transient/`)

`photon_kick_response.py` — drives the ETD2 transient stepper from a
thermal initial state under a step photon kick; pins the resulting
`f(E, t)` snapshot CSV. Paired regression test added 2026-07-03
(`test_photon_kick_response.py`, slow-marked): baseline pin at rtol=1e-6,
monotone x_qp rise, late-time agreement with the independent Newton
steady state, and observable-plumbing consistency. Baseline regenerated
same day — the v1 baseline predated both the ×2 recombination fix
(x_qp_ss moved by exactly the predicted ×1.41) and the
`run_time_dependent` total-time fix (snapshot grid was 96.1/102.1/108.1
ns from float accumulation).

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
