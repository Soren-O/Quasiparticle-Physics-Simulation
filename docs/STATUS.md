# qpsim status (gate tracker)

Last updated: 2026-07-04 (M25 closeout: single-quasiparticle Γ̄ =
Γ̃/N_CP(R) normalization fix in the density equations
(`M25Coefficients.cooper_pair_number_R`), branch-continuation driver
`solve_rate_equation_branch`, paper-exact μ inversions, real Fig-4
comparison models (global quasiequilibrium + renormalized), all M25
baselines regenerated and pinned tight — see Gate 8 / Device
Architecture rows and the Marchegiani validation-figure rows below.
Prior: 2026-07-03 web frontend `qpsim.webui` shipped — see
`docs/Frontend.md`; optional extra `qpsim[ui]`, CI installs it; the
only engine changes are the physics-neutral `progress_hook` on
`services.transient.run_time_dependent` and
`T3Spatial1DBackend.run_until_steady_state`, plus — from the same-day
frontend code review — float coercion in the materials loader
(YAML 1.1 loads unsigned-exponent notation like `1.74e28` as a
*string*; Al/Nb/TiN `v_F`/`rho_F` were affected), a public
`COMMENSURATE_TOL` alias in `collisions.sub_gap_photon`, and a derived
`H_OVER_KB_K_PER_HZ` in `qpsim.constants`. Prior: 2026-07-02
a1-diffusion-operators merged to main; CI green; code-health review
fixes).

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
| 8 | Marchegiani junction path (strategy A: rate equation) | ✅ Eq. 8 Lambert-W T̄ + coefficients-in steady-state solver + SI Notes III/IV coefficient integrals + SI Note V photon-assisted spectral density (`M25PhotonDrive` dataclass + back-solver) + population-dependent `g_ph_α_per_state` arrays + **(2026-07-04) the single-quasiparticle normalization fix and the branch-continuation driver**. The normalization fix: M25 Eqs. 4–6 run on Γ̄ = Γ̃/N_CP(R) (M25 text below Eq. 6), not the ensemble Γ̃ rates the qubit equation uses; the residual previously used Γ̃ (~1e10 Hz) in the density equations, which (a) mixed 1e10 Hz tunneling currents with 1e-8 Hz generation in one system — the entire "flat-valley multi-stability / cancellation-floor" pathology was this conditioning artifact — and (b) drained x_R< ~8 orders below the paper's own small-asymmetry approximation. New `M25Coefficients.cooper_pair_number_R` field (default 1.0 = legacy for opaque bundles; the Note-V builder sets 2ν₀Δ_R·V ≈ 1.61e10). With the fix the Fig 3/4 system has a **unique physical root** at every T (verified by 1-D reduced-residual scans: exactly one sign change), hybr converges from the default seed with residuals 1e-12–1e-24 Hz, and figures match the published curves including panel-a's merged μ_α and Fig 4a's low-T nonmonotonic dip. Sweep entry point: `solve_rate_equation_branch` (bidirectional natural continuation with warm starts + adaptive step bisection + reduced 1-D bracketing fold handler + documented exchange rule); `solve_rate_equation_steady_state_multi_seed` retained for single-point solves (default picker now `min_residual`; the old `max_x_L` default admitted sub-1-Hz pseudo-roots on the recombination slope). **Corrected paper anchors** (paper-exact μ inversion, SI Eqs. S2–S5): μ_L/Δ_L = 0.938 @ 10 mK, 0.872 @ 20 mK, 0.804 @ 30 mK, 0.669 @ 50 mK, linear to ≈0 at T̄ ≈ 146 mK — matching the published Fig 3a curve (linear from ~0.94 to zero at ~0.146 K). The previously quoted anchors (0.9534/0.9076/0.8563/…) were pixel misreads that happened to coincide with a max-x_L pseudo-root (residual 5.7e-4 Hz vs the 5e-8 Hz source scale — not a fixed point) and are inconsistent with the paper's own linearity statement and its x_L ≈ √(g^ph_R/r^L) approximation |
| Device Architecture | Region/Junction/Device/Qubit composition layer | ✅ Phases 1-6 shipped: design doc, ExternalFlux through T3 stack (Phase 2), Region+Junction+Device+solve_device_steady_state (Phase 3), Qubit+parity+JunctionQubitCoupling (Phase 4), `M25GapAsymmetricJJ` Junction subclass (Phase 5 v1), no-double-counting plumbing (Phase 5b: `owns_region_dissipation` flag, `external_dissipation_only=True` backend path, Device-solver routing), moment-solver wiring (Phase 5c: caches the 4-unknown moment-solver fixed point on first evaluate, sidesteps the cross-tunneling bootstrap problem), Fig 3 + Fig 4 paper-figure baselines under `validation/marchegiani_2025/` (Phase 6), deterministic `method='hybr'` + `solve_rate_equation_steady_state_multi_seed` (Phase 6 hardening), and **(2026-07-04) the Γ̄ = Γ̃/N_CP(R) normalization fix flowing through the junction's density-equation flux assembly** (`evaluate` now divides the tunneling gammas by `cooper_pair_number_R`, matching the corrected residual; qubit channels keep Γ̃). Junction default `branch_picker_mode` switched to `min_residual` — with the normalization fix the moment system's root is unique and `max_x_L` (which could pick sub-1-Hz pseudo-roots on the recombination slope) is retained for legacy comparison only. **M25 Fig 3a comparison, corrected:** device-path moments at T = 20 mK are x_L = 5.58e-8, x_R> = 2.02e-8, x_R< = 4.70e-8, p_1 = 8.3e-4 (the unique root; consistent with the paper's small-asymmetry x_L ≈ x_R> + x_R< ≈ √(g^ph_R/r^L)), giving μ_L/Δ_L = 0.864 (leading-order inversion) / 0.872 (full SI-Eq.-S3 inversion) vs the published figure's ≈0.87 at 20 mK. The eighth-session claim of 0.9076 at 20 mK rested on a pseudo-root (x_L = 1.5e-5, residual 5.7e-4 Hz ≈ 10⁴× the source scale) matched against a misread of the figure; both sides of that comparison were wrong and canceled. Picker/lm-determinism history retained in git; `TestLmDeterminism` still pins lm reproducibility. Ancillary fixes: `SpectralContext.active_mask` epsilon switched from `mean(dE)` to the bin spacing of the first bin above the gap (uniform-grid behavior unchanged; correctly handles piecewise grids); default solver seed's p₁ now includes the SI-Eq.-S73 photon term (reduces to the old ee-balance seed when Γ̃^ph_01 = 0) |

## Validation figures

All reproductions self-pinned with CSV baselines + PDF plots under `validation/baselines/{ph0_constant, ph0_kaplan, transient, marchegiani_2025}/` and regression tests under `validation/{fischer_2023, fischer_2024, marchegiani_2025, transient}/` (the `transient/` photon-kick demo gained its paired regression test 2026-07-03; baseline regenerated same day — the v1 pin predated the ×2 recombination fix and the total-time scheduler fix). M25 baselines carry a `# pinned_on:` platform stamp; their strict pin tests run only on the generating platform. (Historical reason: pre-normalization-fix fixed-point selection was platform-sensitive. With the Γ̄ fix the tracked root is unique and residuals sit at ~1e-12 Hz, so cross-platform scatter should now be rounding-level — the stamp is kept as cheap insurance until that's verified on a second machine.)

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 paper legend ratios {0, 0.1, 1, 10} | `validation/fischer_2023/fig3_paper.py` | `ph0_constant/` | ✅ paper-faithful (1620-bin grid, phonon-side Eq. 12 + pair-breaking kernels) |
| Fischer 2023 Fig 5 paper-topology x_qp two-panel | `fig5_paper.py` | `ph0_constant/` | ✅ paper-faithful; Eq. 47 + Appendix-E analytic overlay |
| Fischer 2023 Fig 6 paper-topology gap suppression | `fig6_paper.py` | `ph0_kaplan/` | ✅ paper-faithful ordinate (δΔ_T − δΔ)/δΔ_T; Eq. 53 overlay — ⚠️ **numerical ordinate not currently re-validated**: the only full-sweep value test is `manual_slow` (~14 h, CI-excluded) and its baseline predates the corrected `bcs_dos_cell_weights` regeneration deferred in `FISCHER-BASELINE-REGEN-2026-07-12.md` §4. The Eq. 53 analytic overlay + thermal x_qp are fast-tested; the numerical sweep is not. Re-pin under corrected quadrature to clear. |
| Fischer 2023 Fig 7 paper-facing Q_i,tot(T_B) | `fig7_paper.py` | `ph0_constant/` | ✅ Tables II/III parameters + Eq. 65 extrinsic-loss cap |
| Fischer 2023 Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | `ph0_constant/` | ✅ via `nbar_loop` service; not a literal paper figure |
| Fischer 2024 Figs 5-7 f(E) | `fischer_2024/figs_5_7_fe_pb.py` | `ph0_constant/` | ✅ |
| Fischer 2024 Fig 8 x_qp(T_B) | `fischer_2024/fig8_xqp_pb.py` | `ph0_constant/` | ✅ |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ demo, via `transient` service |
| Marchegiani 2025 Eq. 8 T̄ | `validation/marchegiani_2025/fig3_crossover_temperature.py` | `marchegiani_2025/` | ✅ closed-form Lambert-W |
| Marchegiani 2025 Fig 3 (μ_α vs T) | `validation/marchegiani_2025/fig3_chemical_potentials.py` | `marchegiani_2025/` | ✅ branch-continuation driver + Γ̄-normalized density eqs + paper-exact μ inversion (SI Eqs. S2–S5); both panels match the published curves (panel a: three μ_α merged, linear 0.94 → 0 at T̄ ≈ 146 mK; panel b: μ_L ≳ μ_R> > μ_R< at low T with the R-band merge at ~50 mK); smooth through the crossover, μ → 0 at T̄ |
| Marchegiani 2025 Fig 3 paper-styled panels | `validation/marchegiani_2025/fig3_paper.py` | `marchegiani_2025/` | ✅ paper-faithful (`m25_fig3{a,b}_paper.csv`, insets + regime shading); same solver path as the row above; strict pin rtol=1e-6, fast (runs in the default gate) |
| Marchegiani 2025 Fig 4 (Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T) | `validation/marchegiani_2025/fig4_parity_rates.py` | `marchegiani_2025/` | ✅ multi-stability scatter gone (was a conditioning artifact of the missing Γ̄ normalization); Γ_P smooth with the paper's low-T nonmonotonic dip on panel a; smoothness enforced in the test (max adjacent |Δlog10 Γ_P| < 0.2) |
| Marchegiani 2025 Fig 4 paper-styled two-stack | `validation/marchegiani_2025/fig4_paper.py` | `marchegiani_2025/` | ✅ paper-faithful (`m25_fig4_paper.csv`): full model (both ω_LR) + global-quasiequilibrium reduction (SI Note 1 density ratios + total-density closure, M25 Eq. 7) + renormalized global-QE curve (Fig. 4 caption: Γ^ph_00 = 600 Hz, ω_LR/2π = 6 GHz; 5 GHz family only) + exp(−ω_10/T) detailed-balance dotted reference; reproduces the paper's diagnostic (renorm mimics full Γ_P but deviates on the excitation/relaxation ratio at low T) |

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
- ✅ `rate_equation` — Eq. 8 Lambert-W T̄ closed-form observable + `solve_rate_equation_steady_state` Newton solver on the 4-unknown M25 boxed system (density equations on Γ̄ = Γ̃/`cooper_pair_number_R`; auto-scaled residual tolerance) + `solve_rate_equation_branch` bidirectional continuation driver for temperature sweeps (warm-started natural continuation, adaptive step bisection, reduced 1-D bracketing fold handler, documented photon/thermal exchange rule) + `thermal_equilibrium_seed` / `analytic_low_T_seed` seeds.
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

**779 passed, 14 deselected (slow) in ~4.5 min on Windows** as of the 2026-07-04 M25 closeout (slow Fischer reproductions opt-in via `-m slow`; the M25 strict pins skip off-platform by design; the whole `validation/marchegiani_2025/` directory is now fast — the fig3_paper/fig4_paper pins moved into the default gate at rtol=1e-6 since branch-tracked sweeps take seconds, and the default gate itself dropped from ~12 min to ~4.5 min because the M25 sweeps no longer grind the multi-seed grid). Ruff clean. mypy clean on all qpsim surfaces. GitHub Actions CI (3.13 + 3.14) green since fe4ec54 (needs re-verification on ubuntu after the M25 closeout — the strict pins will skip there, the qualitative tests run everywhere).

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
