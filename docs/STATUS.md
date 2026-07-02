# qpsim status (gate tracker)

Last updated: 2026-07-02 (a1-diffusion-operators merged to main; CI green; code-health review fixes; prior: 2026-06-09 spatial diffusion-operator family A1 + §7.5 benchmarks).

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
| 8 | Marchegiani junction path (strategy A: rate equation) | ✅ Eq. 8 Lambert-W T̄ + coefficients-in steady-state solver + SI Notes III/IV coefficient integrals + SI Note V photon-assisted spectral density (`M25PhotonDrive` dataclass + back-solver) + population-dependent `g_ph_α_per_state` arrays all shipped. **M25 Fig 3a reproduced via the Layer-2 Device Architecture path** (see row below): the plotted quantity μ_α/Δ_L matches paper to within ~2% across T ≤ 75 mK; underlying moment quantities (x_L etc.) sit on a flat-valley multi-stable manifold where many seeds give the same plotted μ_α. Direct path: `solve_rate_equation_steady_state_multi_seed` (post-Phase-6 deterministic helper); the lower-level `solve_rate_equation_steady_state` accepts a single seed and may need `accept_lm_convergence=True` for the cancellation-floor "no progress" stall (the bypass is gated to that specific status + a 1.0 Hz residual ceiling, not arbitrary failures) |
| Device Architecture | Region/Junction/Device/Qubit composition layer | ✅ Phases 1-6 shipped: design doc, ExternalFlux through T3 stack (Phase 2), Region+Junction+Device+solve_device_steady_state (Phase 3), Qubit+parity+JunctionQubitCoupling (Phase 4), `M25GapAsymmetricJJ` Junction subclass (Phase 5 v1), no-double-counting plumbing (Phase 5b: `owns_region_dissipation` flag, `external_dissipation_only=True` backend path, Device-solver routing), moment-solver wiring (Phase 5c: caches the 4-unknown moment-solver fixed point on first evaluate, sidesteps the cross-tunneling bootstrap problem), Fig 3 + Fig 4 paper-figure baselines under `validation/marchegiani_2025/` (Phase 6), and a switch from non-deterministic `scipy.optimize.root(method='lm')` to deterministic `method='hybr')` plus a `solve_rate_equation_steady_state_multi_seed` helper for paper-matching branch selection (Phase 6 hardening). **M25 Fig 3a comparison** (deterministic hybr+lm multi-seed branch picker, eighth-session improvement): the plotted quantity in M25 Fig 3a is μ_α/Δ_L vs T. New picker matches paper to within ~2% across the photon-driven regime: T = 10 mK → μ_L/Δ_L = 0.9534 (paper ≈0.95), T = 20 mK → 0.9076 (paper ≈0.91, **0.2% agreement**), T = 30 mK → 0.8563 (paper ≈0.87), T = 40 mK → 0.8010 (paper ≈0.81), T = 60 mK → 0.6403 (paper ≈0.65), T = 75 mK → 0.5318 (paper ≈0.55). Above ~80 mK the picker stays on the photon-driven branch instead of transitioning to the thermal branch around T̄ ≈ 150 mK — visible as a high-T residual that proper bifurcation tracking would close. Underlying moment quantities (x_L = 1.5e-5, x_R> = 6.3e-6, x_R< = 5.8e-7, p_1 = 1.4e-4 at T = 20 mK) sit on a flat valley of fixed points where many seeds give different (x_L, x_R>) combinations that produce the same plotted μ_L within visual resolution. **Picker history:** the previous picker (hybr-only with `p_1 = 1e-3` seeds) under-sampled the high-x_L manifold and landed at x_L = 3.58e-6, μ_L/Δ_L = 0.8944 (-0.3% vs paper); the eighth-session pass added (a) varied p_1 ∈ {1e-4, 3e-4, 1e-3} seeds with tunneling-correct x_R/x_L = 0.4 ratio, (b) `scipy.optimize.root(method='lm')` as an additional candidate source filtered by 1.0 Hz residual ceiling. The "FORTRAN COMMON-block state" non-determinism originally cited for the lm→hybr switch does not reproduce on supported scipy versions (regression-tested at `tests/services/test_rate_equation.py::TestLmDeterminism`). The 119 Hz residual at paper-exact moment values that prompted the seventh-session investigation is dominated by paper's 3-sig-fig rounding of x_R> (corrected x_R> = 2.094e-6 drops the residual to 0.7 Hz) — not a coefficient bug. Ancillary fixes: `SpectralContext.active_mask` epsilon switched from `mean(dE)` to the bin spacing of the first bin above the gap (uniform-grid behavior unchanged; correctly handles piecewise grids) |

## Validation figures

All reproductions self-pinned with CSV baselines + PDF plots under `validation/baselines/{ph0_constant, ph0_kaplan, transient, marchegiani_2025}/` and regression tests under `validation/{fischer_2023, fischer_2024, marchegiani_2025}/` (the `transient/` photon-kick output is a demo — baseline committed, no regression test). M25 baselines carry a `# pinned_on:` platform stamp; their strict pin tests run only on the generating platform (fixed-point selection is platform-dependent).

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 paper legend ratios {0, 0.1, 1, 10} | `validation/fischer_2023/fig3_paper.py` | `ph0_constant/` | ✅ paper-faithful (1620-bin grid, phonon-side Eq. 12 + pair-breaking kernels) |
| Fischer 2023 Fig 5 paper-topology x_qp two-panel | `fig5_paper.py` | `ph0_constant/` | ✅ paper-faithful; Eq. 47 + Appendix-E analytic overlay |
| Fischer 2023 Fig 6 paper-topology gap suppression | `fig6_paper.py` | `ph0_kaplan/` | ✅ paper-faithful ordinate (δΔ_T − δΔ)/δΔ_T; Eq. 53 overlay |
| Fischer 2023 Fig 7 paper-facing Q_i,tot(T_B) | `fig7_paper.py` | `ph0_constant/` | ✅ Tables II/III parameters + Eq. 65 extrinsic-loss cap |
| Fischer 2023 Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | `ph0_constant/` | ✅ via `nbar_loop` service; not a literal paper figure |
| Fischer 2024 Figs 5-7 f(E) | `fischer_2024/figs_5_7_fe_pb.py` | `ph0_constant/` | ✅ |
| Fischer 2024 Fig 8 x_qp(T_B) | `fischer_2024/fig8_xqp_pb.py` | `ph0_constant/` | ✅ |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ demo, via `transient` service |
| Marchegiani 2025 Eq. 8 T̄ | `validation/marchegiani_2025/fig3_crossover_temperature.py` | `marchegiani_2025/` | ✅ closed-form Lambert-W |
| Marchegiani 2025 Fig 3 (μ_α vs T) | `validation/marchegiani_2025/fig3_chemical_potentials.py` | `marchegiani_2025/` | ✅ both panels (small + large gap asymmetry) match qualitatively; μ_L → 0 at T̄ ≈ 150 mK |
| Marchegiani 2025 Fig 4 (Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T) | `validation/marchegiani_2025/fig4_parity_rates.py` | `marchegiani_2025/` | partial — qualitative trends pinned; panel a has multi-stability noise from competing M25 fixed points (max-x_L branch picker); paper-grade smoothness needs proper bifurcation tracking |

## Analytic tests

`validation/analytic/` (runs in the default suite — fast, no markers):
- Detailed balance: e-ph, sub-gap photon, pair-breaking photon channels vanish at `(f_FD(T), n_BE(ω, T))`.
- Mattis-Bardeen thermal limits: σ_1 → 0 at T → 0, σ_2 → π Δ / ω kinetic-inductance limit.
- Gap-equation round-trip: `solve_gap(f_FD(T_B))` recovers `Δ_eq`.

## Spatial diffusion operators (A1 / §7.5)

`qpsim.transport.diffusion.base` — `DiffusionModel` operator family
parametrized by `(p, q)`, `L_{p,q}[f] = N_1^{-p} d_x(D_N N_1^q d_x f)`:
**A1 = (1, 0)** dirty-limit Keldysh–Usadel (default; conserves `N_1 f`,
undressed flux `D_L` — 1 above the local gap edge, 0 below — rate
`D_N / N_1`, same uniform-gap rate as C), A1P = (1, 2) transverse-dressing
diagnostic (the pre-errata energy-channel assignment), A2 = (2, 2)
diagnostic, C = (0, -1) legacy `D_E` closure, B = (0, -2) constant-τ.
June 2026 advanced-propagator errata (g^A = -τ³(g^R)†τ³): the `N_1²` flux
dressing belongs to the transverse (charge) channel, so the energy-mode
operator is (1, 0), not (1, 2); see `docs/Diffusion_Operators.md` and the
qp-diffusion paper's `verify_gA_convention.py`.

`T3Spatial1DBackend.apply_transport` is an exactly-conservative
finite-volume Crank–Nicolson step on the conserved density `u = N_1^p f`
(harmonic-mean faces, reflective ends; `Σ_x N_1^p f` conserved to ~1e-15);
reproduces the legacy modal C-step to round-off (A1 coincides with it at a
uniform gap). Optional `gap_profile`
(per-cell DOS) + `interface_conductance` add a Kupriyanov–Lukichev two-gap
interface `F = G_N (N_1^L N_1^R - N_2^L N_2^R)(f_L - f_R)` — the
coherence-factor energy-channel weight, regular at matched gaps
(current-continuous, f-discontinuous). Prelim scripts unchanged
(their committed outputs are historical C runs).

§7.5 benchmarks in `validation/diffusion_operators/` (fast co-located tests +
`python -m …` scripts → CSV/figure under `outputs/`): `uniform_gap_packet`
(measured `D_eff(E)/D_N` traces `N_1^{q-p}`; A1 ≡ C at uniform gap),
`gap_gradient_drift` (COM drift
`D_N q N_1^{q-p-1} d_x N_1`; A1 no drift, A1P/A2 drift up the gap, C/B down),
`interface_trap`
(current-continuity + f-discontinuity with the coherence weight; A1 vs A2
distinct closed equilibria).

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

**~700 tests collected; fast suite green on Windows and ubuntu CI** (slow Fischer/M25 reproductions opt-in via `-m slow`; two M25 strict pins skip off-platform by design). Ruff clean. mypy clean on all new qpsim surfaces. GitHub Actions CI (3.13 + 3.14) green since fe4ec54.

Recent hardening pass (Claude+GPT cross-review, seventh session):
- Photon collision kernels (`sub_gap_photon`, `pair_breaking_photon`) and the analytic Newton Jacobian now hard-reject nonuniform energy grids via `qpsim.collisions._uniform_grid.uniform_grid_spacing`; previously they silently used `dE[0]` as a uniform stride.
- `PhononState.__post_init__` now validates finite/nonneg `n_ph`, finite/nonneg/strict-monotone `omega_bins`, and finite/nonneg `tau_l` (was shape-only).
- `phonon_steady_state` raises on singular/runaway phonon balances instead of clipping negative occupations to zero — turns "no Ph0 fixed point" from a fake solution into a loud failure.
- `accept_lm_convergence` bypass narrowed to its documented scope: only the `is_no_progress_stall` case is exempted from the residual check; success-with-high-residual now raises.
- `transient.run_time_dependent` truncates the final substep so `total_time` is honored exactly.
- Four M25 Fig 3/4 multi-stable branch points shifted under the stricter residual acceptance; CSV+PDF baselines regenerated; visual paper agreement preserved.

Slow tests (opt in with `-m slow`):
- Fischer validation reproductions at Fischer-scale grids.
- Transient and M25 Eq. 8 sweeps are fast, not slow-marked.

## Build/dev notes

- **Local venv:** `.venv/` at repo root, Python ≥ 3.13.
  - macOS/Linux: `.venv/bin/pip install -e ".[dev]"`, `.venv/bin/pytest -q`.
  - Windows: `.venv/Scripts/python.exe -m pip install -e ".[dev]"`,
    `.venv/Scripts/python.exe -m pytest -q`. (`PYTHONUTF8=1` is no longer
    required since the material-YAML loader reads UTF-8 explicitly, but
    remains harmless belt-and-braces.)
  - `-m slow` opts into the Fischer/M25 full reproductions.
- **Mac-only paths** (do not exist on the Windows box):
  - Legacy repo: `~/Documents/Quasiparticle Simulation/Active Code/qpsim/` (read-only port source).
  - Specs: `~/Documents/Quasiparticle Simulation/Documentation/Current/` — `New Framework Plan.md` is authoritative.
  - Prelim deck: `~/Documents/Graduate/Preliminary Exam/build_presentation.py` (uses `/opt/homebrew/bin/python3` for python-pptx).
