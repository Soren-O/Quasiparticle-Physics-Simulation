# qpsim status (gate tracker)

Last updated: 2026-04-24 (fifth session).

Central snapshot of what's done, what's in progress, and what's deferred. The New Framework Plan (`~/Documents/Quasiparticle Simulation/Documentation/Current/New Framework Plan.md`) is the authoritative spec; this is the running status against it.

## Gates

| Gate | Scope | Status |
|---|---|---|
| 0 | Phonon-model decisions (D1–D5) | ✅ resolved in `docs/Phonon_Model_Decisions.md` |
| 1 | Repo skeleton | ✅ (tag `gate-1-skeleton` at `77fd516`) |
| 2 | Physics + grid + collisions + solvers ported from legacy qpsim; ETD2 / Strang / coupled Newton upgrades landed | ✅ |
| 3 | T3 paper-reproduction parity against self-pinned baselines | ✅ all 8 figures (see below) |
| 4 | Full Layer-4 audit chain | ✅ all parity tests pass (1e-12 / 1e-6 / 1e-4 tolerance tiers) |
| 4.5 | Characterization tier (Ph0-Kaplan) | ✅ sc-gap + acoustic-escape τ_l (Fig 6) + frequency-resolved τ_PB(Ω) per Kaplan 1976 (`qpsim.physics.kaplan_pair_breaking`, closed-form elliptic-integral evaluator + `tau_0_phonon` Al/Nb material field) |
| 5 | Ph1 phonon spatial transport | ❌ not started |
| 6 | T2 kinetic scalar backend | ❌ not started |
| 7 | T1 two-component backend | ❌ not started (requires new derivation) |
| 8 | Marchegiani junction path (strategy A: rate equation) | ✅ Eq. 8 Lambert-W T̄ + coefficients-in steady-state solver + SI Notes III/IV coefficient integrals + SI Note V photon-assisted spectral density (`M25PhotonDrive` dataclass + back-solver) + population-dependent `g_ph_α_per_state` arrays all shipped. **Quantitative M25 Fig 3a reproduction landed via the Layer-2 Device Architecture path** (see row below): x_L/x_R</R>/p_1 all match published values to 4 sig figs. Direct caller of `solve_rate_equation_steady_state` at Fig 3 inputs needs `accept_lm_convergence=True` to bypass the residual check at the float64 cancellation floor (~1e-5 Hz vs ~1e10 Hz tunneling currents) |
| Device Architecture | Region/Junction/Device/Qubit composition layer | ✅ Phases 1-5c shipped: design doc, ExternalFlux through T3 stack (Phase 2), Region+Junction+Device+solve_device_steady_state (Phase 3), Qubit+parity+JunctionQubitCoupling (Phase 4), `M25GapAsymmetricJJ` Junction subclass (Phase 5 v1), no-double-counting plumbing (Phase 5b: `owns_region_dissipation` flag, `external_dissipation_only=True` backend path, Device-solver routing), moment-solver wiring (Phase 5c: `M25GapAsymmetricJJ` caches the 4-unknown moment-solver fixed point on first evaluate and uses it instead of state-derived moments — sidesteps the cross-tunneling bootstrap problem at Δ_L ≈ Δ_R). **Quantitative M25 Fig 3a reproduction confirmed**: x_L = 5.17e-6, x_R> = 2.09e-6, x_R< = 8.76e-8, p_1 = 3.21e-4 — matches M25 paper to 4 sig figs. Required ancillary fixes: `solve_rate_equation_steady_state` gained `accept_lm_convergence=True` to bypass the residual check at the float64 cancellation floor; `SpectralContext.active_mask` epsilon switched from `mean(dE)` to the bin spacing of the first bin above the gap (uniform-grid behavior unchanged; correctly handles piecewise grids and is immune to tiny far-tail bins) |

## Validation figures

All reproductions self-pinned with CSV baselines + PDF plots under `validation/baselines/{ph0_constant, ph0_kaplan, transient, marchegiani_2025}/` and regression tests under `validation/{fischer_2023, fischer_2024, transient, marchegiani_2025}/`.

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 τ_l = 0 | `validation/fischer_2023/fig3_tau_l_zero.py` | `ph0_constant/` | ✅ bit-identical (1e-12) |
| Fischer 2023 Fig 3 finite τ_l, ratios {0.5, 1, 2, 5, 10} | `fig3_finite_tau_l.py` | `ph0_constant/` | ✅ iterative (1e-6); ratio 10 via coupled Newton |
| Fischer 2023 Fig 5 x_qp vs T* | `fig5_xqp.py` | `ph0_constant/` | ✅ iterative (1e-6) |
| Fischer 2023 Fig 6 gap suppression | `fig6_gap_suppression.py` | `ph0_kaplan/` | ✅ self-pinned, sc-gap + acoustic-escape τ_l |
| Fischer 2023 Fig 7 Q_i(T_B) thermal | `fig7_qi_vs_t.py` | `ph0_constant/` | ✅ |
| Fischer 2023 Fig 7 Q_i(T_B) with drive | `fig7_with_drive.py` | `ph0_constant/` | ✅ via `nbar_loop` service |
| Fischer 2023 Figs 9-13 Q_i(P_read) | `figs_9_13_qi_vs_pread.py` | `ph0_constant/` | ✅ via `nbar_loop` service |
| Fischer 2024 Figs 5-7 f(E) | `fischer_2024/figs_5_7_fe_pb.py` | `ph0_constant/` | ✅ |
| Fischer 2024 Fig 8 x_qp(T_B) | `fischer_2024/fig8_xqp_pb.py` | `ph0_constant/` | ✅ |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ demo, via `transient` service |
| Marchegiani 2025 Eq. 8 T̄ | `validation/marchegiani_2025/fig3_crossover_temperature.py` | `marchegiani_2025/` | ✅ closed-form; **full rate-equation integration deferred** |

## Analytic tests

`validation/analytic/` (opt-in, not slow-marked — fast to run):
- Detailed balance: e-ph, sub-gap photon, pair-breaking photon channels vanish at `(f_FD(T), n_BE(ω, T))`.
- Mattis-Bardeen thermal limits: σ_1 → 0 at T → 0, σ_2 → π Δ / ω kinetic-inductance limit.
- Gap-equation round-trip: `solve_gap(f_FD(T_B))` recovers `Δ_eq`.

## Services (`qpsim.services.*`)

- ✅ `steady_state` — Newton + Picard + Anderson + coupled-Newton routing.
- ✅ `nbar_loop` — F23 Eqs. 59-60 fixed-point on (n̄, Q_i).
- ✅ `transient` — `run_time_dependent` with ETD2 collision substeps, snapshot cadence, stop-tol, observables dict.
- partial `rate_equation` — Eq. 8 Lambert-W T̄ closed-form observable + `solve_rate_equation_steady_state` Newton solver on the 4-unknown M25 boxed system; auto-scaled residual tolerance for arbitrary coefficient magnitudes.
- partial `rate_equation_coefficients` — `M25PhysicalParameters` + `coefficients_from_physical_parameters` build the `M25Coefficients` bundle from primitive physical inputs (gaps, ω_10, transmon E_J/E_C, T, junction tunneling scale `R_T = g_T Δ̄/e²`, caption-level `r^L`/`r^{R<}`). Implements SI Notes III (12 tunneling rates) and IV (recombination, generation-by-thermal-phonons, intraband relaxation, branching fraction). Note V photon-assisted tunneling shipped via `M25PhotonDrive` + `coefficients_from_physical_parameters_with_photon_drive`: elliptic-integral spectral densities `_S_ph_total` (S57), `_S_ph_Rgt` (S59), `_S_ph_Rlt = total − Rgt`; all four `Γ^{ph}_{ij}` derived from single back-solvable `Gamma_nu_scale_Hz`; per-state `g^{ph}_α` arrays feed the population-dependent residual.
- planned `parametric_sweep` — factor the nested-loop pattern duplicated across validation modules.

## Observables (`qpsim.observables.*`)

- ✅ `ac_conductivity` — Mattis-Bardeen σ_1, σ_2.
- ✅ `quality_factor` — Q_i = σ_2 / (α σ_1).
- ✅ `frequency_shift` — δω/ω fractional.
- ✅ `density` — qp_number_density, qp_fraction.
- ✅ `gap_suppression` — δΔ, δΔ/Δ_eq; `compute_gap_suppression(f, E, T_c, T_bath)` + `gap_suppression_from_deltas(Δ_eq, Δ_final)`.
- ✅ `effective_temperature` — `effective_phonon_temperature(n_ph, ω_bins, gap, T_bath)` via weighted BE fit per F23 Eq. 36.
- planned `charge_imbalance` — δN from f_T (NFP §5 Table row 443).

## Test suite

**524 unit/regression tests passing.** Ruff clean. mypy clean on all new qpsim surfaces.

Slow tests (opt in with `-m slow`):
- Fischer validation reproductions at Fischer-scale grids.
- Transient and M25 Eq. 8 sweeps are fast, not slow-marked.

## Build/dev notes

- **Local venv:** `.venv/` at repo root, Python 3.14.3.
  - `.venv/bin/pip install -e ".[dev]"` installs qpsim editable + ruff/mypy/pytest.
  - `.venv/bin/pytest -q` runs fast suite. `-m slow` opts into Fischer reproductions.
- **Legacy repo:** `~/Documents/Quasiparticle Simulation/Active Code/qpsim/` is read-only port source.
- **Specs:** `~/Documents/Quasiparticle Simulation/Documentation/Current/` — `New Framework Plan.md` is authoritative.
- **Prelim deck:** `~/Documents/Graduate/Preliminary Exam/build_presentation.py` builds the .pptx; uses system Python (`/opt/homebrew/bin/python3`) for python-pptx availability.
