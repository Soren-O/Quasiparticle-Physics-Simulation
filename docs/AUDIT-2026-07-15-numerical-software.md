# qpsim numerical-software audit — 2026-07-15

## Outcome

This was a **code-only numerical audit** of `qpsim` at base commit
`b92571a635aee9e8efd3fc228ac4ac7a7e69c150`, followed by repairs in the
`codex/qpsim-deep-audit-fixes` working tree. It was not an audit of any paper or
manuscript. Paper equations were consulted only as narrow checks that an
implemented code path used the model and normalization it claimed to use.

The audit found and repaired several defects that could silently select a wrong
nonlinear branch, accept a false steady state, lose a discrete invariant in a
stiff step, or apply an operator outside its implemented domain. The repaired
paths now fail loudly where the code lacks a justified model. In particular:

- nonequilibrium gap solves can continue deterministically through multiple
  roots instead of letting `bracket_factor` select the branch;
- self-consistent steady-gap solves reject roots below the first represented
  energy-cell face and return only the kinetic state whose own branch-anchored
  gap map was checked;
- a moving-gap update now returns an occupation and gap that satisfy the same
  algebraic constraint, rather than a one-pass solve/remap mismatch;
- ETD2 collision stepping preserves configured semidiscrete balances per
  spatial column and rejects/retries steps when either stage reveals a stiff
  nonlinear loss;
- convergence checks use raw operator residuals and gain/loss backward-error
  certificates rather than clipped updates or small dimensional residuals;
- clean-BCS collision kernels, photon partner maps, external sources, and local
  spatial gaps now enforce their actual support and grid contracts;
- singular BCS capacity, cell support, coherence factors, photon partners,
  phonon line sources, Jacobians, spatial A1 transport, remap, and observables
  now share one finite-volume measure;
- gap calibration closes continuously at the declared critical temperature,
  and the runtime gap integral no longer interpolates through the singular edge.
- thermal Bose factors and public collision/observable boundaries remain
  finite and fail closed at cancellation-prone or malformed inputs;
- steady-state continuation now carries both quasiparticle and phonon state,
  and every T3 solve checks that its public gap and phonon model match the
  spectral operators it will actually use.

This does **not** justify a blanket paper-parity claim. Fischer 2023 Fig. 3 has
now produced and promoted a nonzero, independently balance-certified NE=1620
artifact, and Fig. 6 completed all 66 production targets with QP, phonon, and
gap-map certificates. Fig. 3's scalar-density refinement estimate is small.
Its former 30--42% distribution-shape warning is now diagnosed as a strong-norm
response to clean-BCS photon-threshold layers: weak shape transport and mass
converge much faster, but pointwise/capacity-total-variation convergence is not
certified. An uncached full-pin repeat reproduced the promoted artifact. The repaired moving-gap
method is verified second-order for its documented DAE and support contract. Fig. 7 is
certified at all 48 targets under exact tight-contract Windows and Linux production
runs. Its historical loose-inner-Newton stopping defect is repaired. Later hosted
single-thread Linux runs exposed a smaller, deterministic, independently certified
Windows/Linux envelope in the tight-contract solution; platform-scoped regression bounds
cover it without claiming bitwise portability. The historical
Fig. 5 and Figs. 9--13 pins remain explicitly quarantined. The four Fischer
2024 families were subsequently promoted as certified qpsim-native regressions
at paper topology after commensurate-grid refinement; their pre-strict-v2
artifacts remain archived as rejected audit evidence. The post-repair
unit/API suite passes 1272/1272 tests, and the recorded complete non-slow
validation selection passed all 127 then-selected tests before the additional
N29 artifact regressions landed. Figure CSVs are self-pinned regression
data—not independent measurements of agreement with a publication. The Fig. 7
cross-platform result is a software-regression and balance certificate, not a
bitwise-portability or paper-parity claim.

Throughout this audit, a “Windows/Linux envelope” denotes the OS-family case,
not an exact hardware/runtime identity; its bounds are calibrated from the
recorded hosted runs.

## Scope and method

The review covered the numerical engine and its reachable orchestration layers:

- spectral support, BCS gap calibration, gap quadrature, and moving-gap remap;
- electron–phonon, sub-gap-photon, and pair-breaking-photon collision operators;
- ETD, Newton, coupled-Newton, Picard-facing, and finite-volume TVD paths;
- homogeneous, transient, and one-dimensional spatial T3 backends;
- external fluxes, symmetric junction exchange, observables, validation caches,
  and the web execution boundary.

The main audit questions were code invariants rather than line-by-line style:

1. Does every stored degree of freedom have represented spectral capacity?
2. Do a discrete operator and its stopping/conservation test use the same
   measure, support, frozen coefficients, and time sample?
3. Can a tolerance, bracket, clipping operation, or underflowing tail certify a
   state that is not a numerical fixed point?
4. Are nonuniform-grid geometry, photon partner lattices, and finite-volume
   edge cells handled consistently?
5. Are caches keyed and bounded by every input that changes the operator?
6. Does an unsupported model fail before it can produce plausible-looking
   output?

The review used adversarial manufactured states in addition to thermal fixed
points: a three-root gap-edge occupation bump, stiff conservative exchange,
columns with vastly different balance scales, gain on zero-DOS rows,
incommensurate photon energies, nonuniform reconstruction geometry, and unequal
region capacities. Thermal detailed balance was retained as a useful check, but
was not treated as sufficient: it cannot detect every discrete-measure or branch
selection error.

## Fixed findings

Severity describes possible numerical impact before the repair, not code size.
“High” means a supported-looking call could silently return the wrong state or
violate a governing invariant; “Medium” means a narrower configuration could do
so or an important diagnostic was unreliable; “Low” is robustness, reporting,
or validation-honesty hardening.

| ID | Severity | Finding and impact | Repair | Regression evidence |
|---|---|---|---|---|
| N1 | High | `solve_gap` assumed an effectively monotone nonequilibrium residual. A manufactured gap-edge bump has three roots, and the returned root changed with `bracket_factor`; a numerical tuning knob therefore selected the physical branch. | Added bounded adaptive sign-change scanning and optional `reference_gap`; reference-anchored calls select the nearest root deterministically and warn on multiplicity. Moving-gap and outer self-consistency loops pass their current gap. Legacy unanchored calls retain behavior but warn when ambiguity is detected. | `tests/physics/test_gap_equation.py::TestSolveGap::test_three_root_gap_edge_bump_uses_nearest_continuation_branch` checks all three roots, branch continuity, and bracket-factor independence. |
| N2 | High | `apply_gap_update` solved the gap once and then remapped `f` along frozen-ξ characteristics. The remap changes the occupation entering the gap equation, so the returned `(f, Δ)` did not in general satisfy its own constraint. | Iterates solve → conservative remap until the final gap is unchanged at `1e-10 µeV + 1e-12` relative tolerance; uses a tighter Brent tolerance and fails after 50 iterations instead of returning an uncertified state. | `tests/backends/test_t3_transient.py` checks projection idempotence, the final gap-equation residual, and loud iteration-cap failure. |
| N3 | High | A large stiff ETD2 collision step could drift a conserved weighted quasiparticle balance. In spatial arrays, a global correction could also compensate one column with another, and a very large-mass column could hide an error in a small column. The original rate limiter inspected only the initial-stage loss, so a source-driven predictor could reveal a loss six orders of magnitude larger and still take the unrestricted step. | Added optional weighted Heun-balance projection, an explicit `balance_axis`, slice-local scale-invariant tolerances, and positivity-preserving occupation/hole scaling. Stage-aware subcycling (`max_loss_step=0.25` on collision paths) now rejects and retries from the unchanged state until both initial and predictor losses satisfy the bound. Accepted substeps are reported independently of rejected RHS trials. | `tests/solvers/test_etd.py` covers stiff exchange, independent columns, extreme mass scales, and the manufactured `x' = 1 - 10^6 x^2` predictor-loss adversary. Backend/transient tests exercise the integrated path and exact internal-step reporting. |
| N4 | Medium | BCS coupling was inferred from a zero-temperature gap ratio, producing a small nonzero limiting gap immediately below the declared `T_c`. The runtime integral interpolated `f` off-grid through the singular edge, imposing a trajectory-accuracy floor. | Derives `1/λ` from the finite-cutoff linearized equation at `T_c`; a supplied material `Delta_0` is diagnostic metadata rather than an inconsistent second anchor. Replaced edge interpolation with analytic vacuum `acosh` plus exact per-cell occupation weights, using the state’s actual `dE`. | `tests/physics/test_gap_equation.py` covers continuous closure at `T_c`, shipped-material references, exact piecewise-constant integrals, manufactured cell averages, and thermal round trips; `validation/analytic/test_gap_equation_equilibrium.py` was updated accordingly. |
| N5 | High | Collision code could be called with a Dynes-broadened spectrum even though the implemented coherence factors and lattice partners are clean-BCS formulas. Target rows with zero represented capacity could also acquire nonzero rates. | All public phonon, sub-gap-photon, and pair-breaking-photon collision boundaries reject Dynes contexts. Gain/loss rates and Jacobian rows are zeroed outside finite-volume support, and backend/service boundaries enforce the same contract. | Focused collision suites include public-boundary Dynes rejection, zero-support rows, detailed balance, and Jacobian coverage in `tests/collisions/` and `tests/backends/`. |
| N6 | High | Photon lattice mismatches were sometimes snapped silently. A nominally commensurate pair-breaking energy can still have a misaligned reflected partner origin, breaking the pair map; the sub-gap API also accepted energies at or above pair-breaking threshold. | Mismatches greater than `0.01` of an energy-grid bin now raise; accepted values snap only within that lattice tolerance. Pair-breaking reflection-origin alignment is validated whenever the pair channel is open. Sub-gap photon APIs enforce `omega_0 < 2Δ`. | `tests/collisions/test_sub_gap_photon.py`, `tests/collisions/test_pair_breaking_photon.py`, and `tests/webui/test_schemas_builders.py` cover domain, commensurability, and pre-run validation. |
| N7 | Medium | A transient `stop_tol` check rebuilt collision kernels and could resample a time-dependent external-flux callback at the endpoint. Even with one frozen midpoint sample, an instantaneous zero of a future time-dependent drive cannot certify steady state. Driver steps also hid internal rate subcycling. | Autonomous runs use a raw endpoint collision residual from the same frozen kernels and constant flux as the accepted ETD step. Callable time-dependent flux remains midpoint-sampled when integrating without early stop, but `stop_tol` plus a callable now raises: convergence of a non-autonomous future problem is undefined from one instantaneous residual. Results report both driver `n_steps` and accepted `n_etd_substeps`. | `tests/services/test_transient.py` covers one kernel build, midpoint-only callback sampling without stopping, isolated-zero-drive rejection, constant-flux stopping, subclass fallback, and stiff accepted-step reporting; backend and web-execute tests cover diagnostics propagation. |
| N8 | High | Coupled Newton could walk into a near-vacuum/zero branch with a tiny dimensional residual and then use an absolute-only line-search fallback, bypassing the requested relative step/balance certificate. Pointwise tail ratios also made a useful certificate numerically unattainable. | The absolute-only escape is available only when callers explicitly set `step_rtol=0`. The default is `1e-8`; convergence uses active-support variables plus normwise block gain/loss backward error, with finite/positive option validation. | `tests/solvers/test_coupled_newton.py` checks that a tiny absolute residual cannot bypass the balance certificate, nonfinite residuals fail closed, and the analytic/finite-difference paths agree on acceptance. |
| N9 | Medium | `SpectralContext.active_mask` excluded legitimate near-gap bins through an arbitrary margin. Conversely, source gain on a genuinely zero-capacity row represented injection into no state at all. | Support is now exactly `cell_weights > 0`. A BCS cell cut by the gap remains active even if its center has point `rho == 0`; homogeneous, coupled, transient, and spatial source paths reject positive gain only outside represented capacity while treating loss-only terms there as inert. | `tests/physics/test_spectral.py`, `tests/devices/test_external_flux.py`, `tests/services/test_steady_state.py`, and backend tests cover near-edge and cut-cell support, Dynes support, rejected gain, and inert loss. |
| N10 | Medium | Spatial collisions needed local-gap spectral operators, but rebuilding or retaining dense kernels for arbitrary per-cell gaps is both expensive and potentially unbounded at Fischer-sized energy grids. Reusing a context without a complete physics key risks stale operators. | Added a complete, gap-keyed LRU collision cache, with lookup before context construction, bounded to two entries. The collision path currently accepts at most two distinct local gaps and fails explicitly beyond that supported memory contract. External-flux support is validated per local gap. | `tests/backends/test_t3_spatial_1d.py` covers cache hits before context construction, LRU bounds, physics-input invalidation, local zero-capacity gain, and two-gap execution. |
| N11 | Medium | The nonuniform TVD reconstruction used `dE[cell]/2` as the center-to-face distance. That equality fails next to cells of different widths and can undo the limiter through an overlong extrapolation. | Reconstructs to the actual midpoint face from neighboring energy centers. | `tests/solvers/test_spectral_flow_tvd.py::TestAdvectSpectralFlow::test_nonuniform_reconstruction_uses_actual_face_distance` uses the counterexample grid `[1, 11, 12, 13]`; conservation/subcycling coverage remains in the same suite. |
| N12 | Medium | `SymmetricGapTunnelingJunction` applied equal rates to both regions, which conserves population only when their quasiparticle capacities are equal. | Added validated `capacity_ratio_a_to_b = C_a/C_b` and sets `α_b = α_a C_a/C_b`, so `C_a f_a + C_b f_b` is conserved. Diagnostics expose both rates while retaining the legacy key. | `tests/devices/test_device.py` covers unequal-capacity conservation and rejects nonfinite/nonpositive capacity ratios and invalid rates. |
| N13 | Medium | The historical `qp_fraction` is `n_qp/(4 rho_F Δ_0)`, half the convention used by Fischer/Catelani. A paper comparison could therefore carry a silent factor-of-two error. | Kept the historical API for compatibility and added/exported `qp_fraction_paper = n_qp/(2 rho_F Δ_0)`. Web summaries, arrays, labels, and CSVs name both conventions. | `tests/observables/test_density.py::TestQpFraction::test_paper_convention_is_explicit_factor_two` and web execution tests pin the conversion and labels. |
| N14 | Medium | Dynamic Ph0 runs defaulted away from the implemented F&C Eq. 12 phonon-side kernel, and a missing/invalid phonon pair-breaking time could silently steer users to legacy behavior. | Dynamic Ph0 backend/UI defaults now select the phonon-side kernel; explicit `False` is a legacy opt-out. Dynamic use requires a finite positive `tau_0_pb_ns`; the material/API documentation states that contract, shipped Al metadata supplies it, and materials without it fail loudly unless the caller selects thermal phonons or the legacy opt-out. | Focused backend, schema/builder, material, and phonon tests cover defaults, opt-out, required parameter, and prefactor normalization. |
| N15 | Low | Several orchestration paths could make correct numerics hard to reproduce safely: UI Newton controls did not reach coupled Newton, an equal-gap “step” failed only at runtime, terminal manifest failures could strand undeletable runs, validation cache identifiers could escape/alias paths, and the unauthenticated UI could be bound remotely. | Mapped controls to coupled-Newton keywords; reject equal-gap interfaces before execution; made terminal deletion race-safe; strictly validate and contain cache identifiers; restrict CLI binds and Host headers to loopback (with `testserver` only for in-process tests), including correct IPv6 browser URLs. | `tests/webui/test_cli.py`, `tests/webui/test_schemas_builders.py`, `tests/webui/test_server.py`, `tests/webui/test_store_execute.py`, and `tests/validation/test_sweep_cache.py` cover these boundaries. |
| N16 | High | Crank–Nicolson is A-stable but not L-stable. At an allowed large diffusion step its stiff amplification became negative without leaving `[0,1]`: a two-cell pulse was nearly swapped rather than damped, so the clipping diagnostic remained silent. | The spatial backend computes each energy channel's maximum finite-volume exit rate and subcycles until `h max(exit_rate) <= 1`. Since the generator obeys `|lambda_max| <= 2 max(exit_rate)`, retained CN modal amplifications are non-negative. A one-million-step guard fails loudly on unreasonable work. | `tests/backends/test_t3_spatial_1d.py::TestT3Spatial1DTransport::test_huge_dt_subcycles_before_cn_can_swap_two_cell_pulse` checks damping toward the mean, sign preservation, and mass conservation at `dt=100`. |
| N17 | Medium | Spatial `run_until_steady_state` used only the accepted, clipped step difference. A saturated `f=1` state under a positive source therefore appeared stationary and was reported converged despite a large outward raw RHS. | Collision stepping and diagnostics share one cached gain/loss system. The endpoint certificate reconstructs the transport generator from the same cached CN matrix, adds the raw collision/source rate, and only then takes the norm; this preserves cancellation at a true driven steady state while exposing infeasible saturated sources. | Spatial backend tests cover the saturated-source counterexample, raw transport/collision cancellation, and thermal equilibrium. |
| N18 | Low | Several public no-op/low-level paths validated too late: `etd1_step` accepted nonfinite time/rates or broadcast shapes, spectral-flow advection returned before validating an invalid grid/gap, and homogeneous collisions could reach kernel work with a malformed occupation. | Added exact-shape, finite-value, physical-gap/domain, and occupation guards before numerical work, including no-op calls and a gap trajectory that would cross zero. | Focused ETD, spectral-flow, and homogeneous transient tests cover invalid shapes, nonfinite values, gap crossing, and bounds. |
| N19 | High | The observable/remap BCS measure used exact singular cell integrals while collisions conserved midpoint `rho*dE`. A first repair changed only the QP phonon measure; photon partners and phonon-side sources stayed point-sampled, breaking `w_i T_ij = w_j T_ji` and collapsing driven Fischer states by 5--8 orders. | Defines `w_i = integral_cell rho dE`, `rho_bar_i = w_i/dE`, capacity support `w_i > 0`, and product-measure-averaged BCS coherence factors. QP contractions/ETD use `w`; photon partners and phonon line terms/Jacobians use `rho_bar`; spatial A1 capacity, dirty-limit support fraction, KL interface weight, remap, sources, and observables use the same represented cells. | An adversarial six-cell drive reproduces the historical hybrid's `O(10^-2)` number drift while the matched operator is conservative below `2e-15`; QP/phonon event energy and analytical Jacobians agree to roundoff/finite differences. `validation/fischer_2023/test_fig3_finite_volume_reduced.py` retains resolved driven peaks instead of the historical `1.52e-17` collapse. |
| N20 | High | The outer self-consistent-gap loop could accept a small raw gap-map residual, under-relax to a different gap, then run an unchecked final fixed-gap kinetic polish. The returned occupation could therefore invalidate the gap map that had just passed. A candidate below the first represented cell face was also allowed to rely on constant-left extrapolation through unsampled gap-edge support. | Each outer iterate completes the fixed-gap kinetic solve, computes a branch-anchored raw gap map on that exact occupation, and returns that same state only when the unrelaxed relative residual passes. There is no post-convergence polish. A positive candidate below the reconstructed first cell face now raises and tells the caller to extend the grid. | `tests/backends/test_t3_self_consistent_gap.py` independently recomputes the returned branch-anchored map, manufactures a post-polish occupation-branch jump, rejects an under-relaxation false positive, and checks the below-face failure. The focused file passed 8 tests without the former extrapolation warnings; a broader T3 backend group passed 37 tests. |
| N21 | Medium | The former project–collide–project moving-gap step composed a `dt`-independent algebraic remap with ETD2, so the coupled method was only first-order even though the collision substep was second-order. | `T3DiffusionBackend.step` now advances persistent BCS material-coordinate cell averages `g(xi)` with stage-constrained ETD2. Every stage jointly materializes the fixed-energy occupation, solves the algebraic gap constraint, evaluates collisions, and conservatively lifts events back to the persistent cells. | `tests/backends/test_t3_transient.py::TestStep::test_moving_gap_coupling_is_second_order_and_constrained` measures orders 1.8–2.2 against a refined reference for both `f` and `Delta`. Companion regressions cover zero-collision material shells, scattering mass conservation, the fixed-gap limit, photon channels, bounds, and rejected-stage nonmutation. |
| N22 | High | Homogeneous Newton could accept a tiny dimensional residual without a scale-independent physical balance, including through an absolute-only line-search fallback. Picard could likewise stop on a small mixed-iterate change even when its mapped fixed point or affine Ph0 balance was not certified. Conversely, direct evaluation of `a*n+b` can report a large relative residual at the correctly rounded binary64 affine root because the exact real root is not representable. | Newton now requires both its dimensional residual and a normwise L1 gain/loss backward error and has no absolute-only fallback. Picard requires local and normwise mapped-fixed-point tests plus a shared physical Ph0 certificate. That certificate records the raw residual separately, accounts for operation rounding and the nearest-binary64 half-ULP interval, and certifies only a finite nonnegative representable root; negative, signed-underflow, singular, overflowed, and nonfinite balances fail closed. If the mixed phonon iterate is several ULPs from the certified map, the service projects to that map and performs one final Newton solve so the returned `(f, n_ph)` is a matched pair. | Scale adversaries, false tiny roots, exact-nearest-root acceptance, two-ULP rejection, underflow/overflow, negative-root, and invalid-control cases pass across `tests/solvers/test_newton_steady_state.py`, `tests/services/test_steady_state.py`, and `tests/phonon_models/test_ph0_local.py`. The shared independent certificate is perturbation-tested in `validation/fischer_2023/test_steady_state_certificate.py`; the combined focused group passed 132 tests. |
| N23 | Medium | Thermal Bose factors used `1 / (exp(x) - 1)`. A positive transfer below machine epsilon could therefore cancel to a zero denominator: the public occupation was replaced by zero, while a collision helper could form infinity even though the limiting kernel product is finite. Direct collision, density, and spectral-flow APIs also accepted malformed occupations or coerced numeric support masks, including on nominal no-op paths. | A shared evaluator uses `expm1`, distinguishes the exact decoupled zero-transfer bin from an underflowed positive transfer, and saturates only occupations outside binary64 range. Public collision/observable calls now require exact shapes, finite physical occupations and temperatures, paired override matrices, and Boolean masks before doing work; kernel grids and coherence matrices no longer flatten or broadcast silently. | Kernel/phonon focused checks passed 69 tests; the three public collision suites passed 79; density/spectral-flow checks passed 44. Regressions include a `1e-18` microelectronvolt frequency, nextafter-spaced energies, maximum finite temperature, and malformed no-op calls. |
| N24 | High | `T3DiffusionBackend.steady_state` could accept `state.gap != state.spectral.gap`, so returned metadata named one gap while kernels used another. It also accepted non-Ph0 model tags even though it always solved Ph0. Separately, backend Picard passed only `f` to the service, which reset `n_ph` to thermal at every call; the Fig. 3/6 claim of full-state branch continuation was false. Coupled Newton treated a negative bath temperature as the zero-temperature limit. | Steady-state entry now enforces the state/spectral gap invariant and `PhononModel.PH0_LOCAL`; the state dataclass rejects non-enum tags. The finite-escape Picard service accepts and validates an explicit phonon seed, and the backend forwards it only when it lies on the exact QP-derived frequency grid. Both Picard and coupled Newton reject invalid bath temperatures. | A 130-test backend/service/Newton group passed after the repair; the service file then passed 56 tests including first-iteration seed capture and no-active-bin guards, and state/coupled-path checks passed 19. The subsequent Fig. 3 and Fig. 6 production sweeps described below exercised full-state continuation. |
| N25 | High | `SymmetricGapTunnelingJunction` checked energy centers and cross-region gaps but not cell widths, Dynes broadening, per-bin finite-volume capacities, or either state's own public/spectral gap coherence. A scalar material/volume capacity ratio cannot conserve a population when the two discrete capacity vectors differ with energy. | Runtime evaluation requires both input states to be individually coherent and physical, and the two regions to have matching centers, widths, spectral gaps, Dynes model, and cell-weight vector before applying the scalar capacity ratio. | The device/junction group passed 51 tests, including unequal global capacities, mismatched widths, mismatched Dynes measures, malformed state gaps, and coherent cross-gap rejection. |
| N26 | High | Validation CSV readers could silently overwrite duplicate coordinates, tolerate missing Cartesian points as NaN, or accept certificate columns/metadata that were absent, stale, nonfinite, or above the declared gate. Interrupted direct writes could also leave a partial artifact. Thus a self-pinned figure test could pass without certifying the artifact it appeared to validate. | Fig. 3/6/7 readers now require their current exact schemas, metadata and certificate versions; reject duplicate/missing coordinates, wrong dimensions, nonfinite fields, and over-target accepted certificates; and quarantine stale pre-schema pins. Writers use same-directory temporary files plus atomic replacement with explicit UTF-8/LF CSV output. Solver point failures are distinguished from certificate/observable failures so only genuine fold points become NaN. | Fig. 3 production completed at NE=1620 and its independently certified artifact was promoted after focused checks; Fig. 6 completed and certified all 66 exact production targets; Fig. 7 artifact checks and its clean local slow recomputation passed. Their detailed numerical evidence and remaining qualifications appear below. |
| N27 | Medium | If the gap residual stayed positive from the numerical zero-gap floor to the Debye cutoff, `solve_gap` warned and returned `delta_eq` even though it knew that value was not a root. An outer gap loop could then treat this fallback as its exact gap map. | The impossible positive-endpoint case now raises with both residual values; only the legitimate negative-low-end normal-state case returns zero. No non-root fallback is exposed to downstream convergence tests. | The gap-equation file passed 30 tests, including an inconsistent-calibration adversary that must fail without returning a gap. |
| N28 | Medium | Three diffusion benchmark oracles still described the pre-repair, point-sampled discretization. The uniform-packet and self-consistent-feedback checks used center-point DOS where production transport conserves against the exact cell-average DOS; the packet check also inverted a single Crank–Nicolson step even when the backend subcycled. The interface check formed a product of separately averaged `N_1`/`N_2` factors, which is not the finite-volume average of the complete Kupriyanov–Lukichev coherence product. These stale oracles produced five false validation failures and could have encouraged a mathematically incorrect rollback. | Rewrote the references against the actual represented measure: `cell_weights / dE` for capacity and conserved-density COM, the exact above-gap support fraction for `q = 0`, and the analytic inversion of all `m` equal CN substeps using `expm1`. The KL reference now independently integrates the complete coherence-factor product over the energy cell and applies a raw-flux continuity certificate. | The five initially failing nodeids were the two interface current/jump checks, two self-consistent feedback drift checks, and the uniform-packet effective-diffusivity check. After the oracle repairs, all **14/14** tests in `validation/diffusion_operators/` passed in **209.10 s**. This verifies qpsim's finite-volume discretization; it is not paper-parity evidence. |
| N29 | High | The Fischer 2023 Fig. 5 and Figs. 9--13 artifacts and four Fischer 2024 artifact families could treat pre-schema, stale, incomplete, or caller-forged CSV data as a usable pin. Several readers lacked exact source/configuration/axis contracts; supplied certificate scalars were not bound to the stored payload; broad `ArtifactValidationError` catches converted future corrupt current-schema artifacts into expected xfails; and direct writes could leave a partial pin. The numerical evidence also showed that the historical pins are stale: the repaired Fig. 5 branch guard passed while its upper-panel pin failed, Figs. 9--13 changed under the matched finite-volume measure, and all four F24 families disagreed with their legacy snapshots. | Added exact versioned schemas, dimensions, columns, axes, resolved configurations, dependency-complete source hashes, finite/unique/physical-domain checks, per-point independently reassembled balance certificates, and same-directory fsync-plus-atomic replacement. Stored tables and ordered certificates are bound by SHA-256; Fig. 5 persists exact compressed `f`/`n_ph` states and recertifies them on both write and read; the F23 outer `nbar` residual is recomputed from persisted `P_read`, `Q_tot`, and `n_bar`; F24 generation retains the live `f(E)` long enough to recompute `x_qp` and QP balance instead of trusting stamps. Distinct `LegacyArtifactError` paths quarantine only genuinely pre-schema pins, while a malformed/stale current schema fails. F24 Fig. 8 resets at every temperature and continues the full state strong-to-weak. F23 legacy replacements remain blocked on refinement. After a separate commensurate-grid study, the four F24 pre-v2 pairs were moved intact into `validation/baselines/legacy/fischer_2024_pre_strict_v2/` and strict-v2 replacements were promoted. | Cache-off F23 Fig. 5 ran **8088.34 s**: the high-drive nonthermal-branch check passed, but the pin failed **20/42** upper-panel values (max absolute difference `1.3979e-4`, max relative `0.98119093`; at `T_B=0.10 K`, `nbar=2.89e7`, current `4.5783954713003457e-8` versus pin `2.4341423568665584e-6`). Its new schema suite passed **9 fast tests**, and only the exact legacy canonical xfails. Figs. 9--13 had `Q_i` drift at all 21 powers (max relative `14.5144%`) while fresh certificates were small. The F24 hardening suite passed **51 fast tests** plus four reduced live solve/write/read paths. Its four promoted pins passed live in **76.86 s**; all 84 points passed strict readback with family maximum QP backward errors no larger than `9.104e-7`. These are code-regression and balance results, not paper-parity evidence. |
| N30 | Medium | Spatial transport accepted smooth gap profiles, but the local collision path rejected more than two distinct gaps to avoid retaining `O(NX * NE**2)` dense kernels. This left a physically inconsistent transport-only escape hatch for general profiles and made an implementation memory bound look like a model limitation. | One- and two-gap profiles retain the existing batched ETD2 path. Profiles with three or more exact gaps now stream independent gap groups through ETD2 and stream the raw endpoint-rate diagnostic through the same group evaluator. The two-entry LRU remains the only cross-group dense-kernel owner, so resident collision-kernel memory stays `O(NE**2)` while support, finite-volume weights, sources, and rates remain local-gap matched. Resident gaps are visited first in both stepping and endpoint diagnostics, avoiding zero-hit cyclic thrashing. | The **55-test** spatial backend file passes. New regressions compare a three-gap streamed step with independent single-gap collision solves to `2e-13` relative tolerance, compare a four-gap raw rate column-by-column, verify the two-entry cache bound and `N-2` rebuild behavior on repeated diagnostics, and reject unsupported external gain in a late streamed group. Ruff and strict mypy pass on the changed source/test files. |
| N31 | Medium | Direct `solve_gap` warned but still returned a gap below the reconstructed first energy-cell face by silently extending the first stored occupation across unsampled gap-edge states. The T3 backend rejected this geometry, but low-level callers could ignore the warning and consume a quantitatively unsupported root or normal-state decision. | The public solve now fails closed by default whenever its continuation anchor, selected root, or collapse lies below grid coverage. Historical constant-left extrapolation requires the explicit `allow_gap_edge_extrapolation=True` opt-in and still emits a runtime warning. A fixed `1e-9` relative geometric edge-coincidence tolerance admits only sub-part-per-billion numerical equality and is independent of caller `xtol`, so a large root tolerance cannot opt into missing support. Covered normal-state grids continue to return zero without an extrapolation flag. | The gap-equation file passes **35 tests**, including default root/anchor rejection, explicit legacy opt-in, type validation, a fully covered non-equilibrium collapse, a large-`xtol` bypass adversary, and the Fischer-scale `8.28e-8 micro-eV` numerical edge coincidence. Existing branch-continuation and equilibrium recovery tests remain green. |
| N32 | High | Fischer 2023 Fig. 7 used an inner Newton tolerance/backward-error gate of `1e-11/1e-6`. Exact loose-contract Windows/Linux solves could therefore stop at different tolerance-limited approximations of the same root while still passing the old `1e-5` artifact gate. The former loss comparison also used `atol=1e-10`, masking every loss below `Q=1e10`. | Tightened the inner Newton contract to `1e-13/1e-8`, the Picard map/change/balance gates to `1e-9/1e-13/1e-8`, and the persisted certificate gate to `2e-8`. Fig. 7 now validates requested axes and every raw/result field before deriving or writing observables; binds `Q_qp` to `sigma1` and `Q_tot` to `Q_qp/Q_ext`; stamps source/runtime provenance; and uses same-OS loss/`Q_tot` bounds of `4e-3`/`1e-4` with loss `atol=2e-19`. N35 adds narrowly measured `8e-3`/`2e-4` Windows/Linux bounds while retaining the `2e-8` certificate gate. The loose canonical is retained under `ph0_constant/archive/`. | Repeated same-platform/thread probes were bitwise stable; tightening Picard alone did not move the selected points, while tightening inner Newton collapsed the two visible cross-platform probes to `1.30e-6` and `3.78e-7` relative loss differences. Exact 48-point tight runs passed on Windows 11 / Python 3.14.3 / NumPy 2.4.2 / SciPy 1.17.0 in **982.54 s** and Linux / Python 3.13.14 / NumPy 2.5.1 / SciPy 1.18.0 in **946.16 s**. Across them, meaningful (`T>=0.10 K`) loss drift was at most `0.24396%`, `Q_tot` drift at most `6.3863e-5`, and the larger low-temperature relative tail represented only `1.0941e-19` absolute loss. Both full runs stayed below the `2e-8` certificate gate. |
| N33 | Medium | After N31, two integration boundaries still assumed an occupation sampled only above the original gap could support an arbitrarily suppressed root. The web executor also coupled independent equilibrium calibration metadata to that fallible occupation-backed diagnostic, so a support failure removed `delta_eq_ueV` from an otherwise valid result. | `compute_gap_suppression` now documents and enforces its low-edge coverage contract; covered observable fixtures extend their first cell face to zero, while an explicit regression proves the former `1.001*Delta` grid fails closed. The web executor calibrates and persists `delta_eq_ueV` independently, then attempts derived suppression; unsupported suppression fields remain absent with a diagnostic note instead of erasing equilibrium metadata. | The combined affected slice passed **50 tests**; the complete web suite passed **112 tests**. The final repository aggregate passes with the intentionally constant hot-tail warning still visible. Ruff, mypy, and compileall are clean. |
| N34 | Low | The transient photon-kick CSV writer inherited the Windows locale encoding. Its title's em dash was therefore persisted as CP1252 byte `0x97`, even though `.gitattributes` correctly pinned LF; GitHub Actions Ubuntu/Python 3.14 completed the expensive solve and then failed while decoding the baseline as UTF-8. The artifact has prose provenance but no byte checksum contract, so neither caught the encoding drift. | Both reader and writer now select UTF-8 explicitly while retaining `newline=""` and the CSV writer's `lineterminator="\n"`. The canonical was migrated by replacing only the CP1252 encoding of that em dash with its UTF-8 bytes. A fast artifact-I/O regression requires strict BOM-free UTF-8, LF-only bytes, and a locale-independent write/read round trip. | The pre/post CSV hashes are `1fb7ef59ec969c8e4a705fa0283006476292ffae04f79a7284215ee02494dbc9` / `18e2a2424c037e2b6dd64189848765d0a0c75a6b6cc4bed63364c3f2d05c51d1`. File size changed by exactly two bytes (`447856 -> 447858`); all `447332` bytes from `E_uev` through the final numerical row are identical with SHA-256 `7df15a69abc0fed86494d101d40b300605692dc77c9ebe037743d4d59a6e067a`. The two fast portability tests pass; Ruff and strict mypy are clean. |
| N35 | Medium | The first hosted slow gate exposed both an uncontrolled BLAS-thread contract and under-sampled hardware/runtime portability envelopes. Four-thread reduction order made Fig. 3's roundoff-limited coupled-Newton line search discard a polished state and retain its Picard predictor. Serializing BLAS removed that fallback, but exact hosted Linux Python 3.13/3.14 runs still produced stable, fully polished, independently certified values outside Windows-calibrated strict gates: Fig. 3 ratio 10 differed by `1.27723%`, while Fig. 7 loss and `Q_tot` drift reached `4.63267e-3` and `1.634166e-4`. | CI pins `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1`, with a fast workflow-contract regression. Fig. 3 keeps its caller-local `coupled_newton_step_rtol=1e-6`, stamps the generating platform, retains `rtol=1e-4` for ratios through one and same-platform ratio 10, and uses `1.5e-2` only for ratio 10 in the Windows/Linux OS-family case calibrated by hosted runs. Fig. 7 retains same-OS loss/`Q_tot` bounds of `4e-3`/`1e-4` and uses `8e-3`/`2e-4` for that same OS-family case. Independent certificate gates are unchanged. Fig. 3 numerical rows/PDF and the entire Fig. 7 artifact remain unchanged. | The two hosted Python versions agreed closely; Fig. 3 completed coupled Newton without fallback and all Fig. 7 states stayed below the `2e-8` certificate gate. Tightening Fig. 7 beyond its current inner-Newton contract was rejected: `1e-10` selected a different low-occupation branch at `T=0.14 K`, moving loss by `0.8768%--16.1554%`, while `1e-9` moved low-temperature `Q_tot` beyond the strict gate. Runtime provenance reports BLAS variables, but the Fig. 7 CSV header does not serialize them; the workflow/test contract and this audit record that remaining metadata limitation. |

## Physics-consistency spot checks

These checks were deliberately narrow. They establish that code implements its
declared formula; they do not establish that the publication’s model is correct
or that every figure has been reproduced.

- The default dynamic-phonon path uses the phonon-side kernel labeled F&C
  Eq. 12. Unit tests pin the implemented prefactors
  `2 K_minus/(pi Δ tau_0_pb)` for scattering and
  `K_plus/(pi Δ tau_0_pb)` for pair processes.
- The quasiparticle recombination loss retains the Kaplan per-QP
  normalization with no extra leading factor of two: the partner quasiparticle
  is removed in its own energy row. Dedicated absolute-rate tests compare both
  the same-grid quadrature and the gap-edge rate against independent formulas.
- BCS coupling is normalized by the linearized finite-cutoff equation at the
  declared `T_c`; a measured `Delta_0` can reveal material/model mismatch but no
  longer moves that same weak-coupling model to a different critical
  temperature.
- The public sub-gap path enforces `omega_0 < 2Δ`; the web-facing pair-breaking
  configuration enforces `omega_PB > 2Δ`, while the underlying kernel enables
  reflected pair terms only when that channel is open. Both require the
  discrete partner maps they actually use to exist.
- Junction exchange now conserves the capacity-weighted quasiparticle
  population, the correct discrete statement when two equal-gap regions have
  different `rho_F * volume`.
- Both quasiparticle-fraction conventions are explicit, preventing a silent
  factor-of-two in code-to-paper comparisons.

No manuscript derivation, parameter inference, plotted curve, or claimed
experimental agreement was audited here.

## Verification performed

The final default repository aggregate is current through all audit follow-ups
described above. Slow/manual figure selections are reported separately because
they have different costs and status. Targeted counts overlap one another and
must not be added.

| Area | Observed result |
|---|---|
| Final default repository aggregate (`pytest -q`) | **1513 passed, 17 deselected, 4 warnings in 457.57 s** on the exact final code/test/validation tree before documentation-only edits. The deselections are the opt-in slow/manual selections. |
| Hosted CI, run `29629653466` | Python 3.13 default: **1513 passed, 17 deselected, 4 warnings in 289.82 s**; slow: **14 passed, 2 expected xfailed, 1514 deselected, 1 warning in 2224.94 s**. Python 3.14 default: same counts in **272.11 s**; slow: same counts in **2035.36 s**. Ruff, mypy, symbolic identities, and default tests passed; the non-manual slow matrix completed with its expected outcomes. Run wall: **46m59s**. |
| Historical unit/API checkpoint (`pytest tests -q`) | **1272/1272 passed, 13 warnings in 258.88 s** on the earlier post-repair combined tree |
| Default non-slow validation discovery (initial audit run) | **5 failed, 122 passed, 17 deselected, 3 expected xfailed, 1 warning in 556.56 s.** All five failures were stale finite-volume oracles identified in N28; this initial run is diagnostic history. |
| Historical post-repair non-slow validation (`pytest validation -q`) | **127 passed, 17 deselected, 3 expected xfailed, 1 warning in 217.75 s.** The expected xfails are the two quarantined Fig. 3 baseline schema/nonvacuity checks and the stale Fig. 6 configuration-metadata check. The sole warning is the explicit `8.28066788472e-08 µeV` support-edge diagnostic in the analytic gap-equilibrium check. |
| Diffusion-operator validation, including cell-measure references, subcycled-CN analytic inversion, and independent KL product quadrature | **14/14 passed in 209.10 s** after repairing the five stale oracles |
| Public phonon/sub-gap/PB collision APIs | 59 passed |
| ETD solver | 12 passed |
| Coupled-Newton solver | 23 passed |
| Coupled Newton + backend + web-builder integration | 70 passed |
| Homogeneous Newton/Picard/Ph0 certificate group | **132 passed in 2.50 s**; includes the independent certificate and backend threading checks |
| Transient/backend/web-execute integration | 79 passed; two expected low-support warnings |
| Spectral support | Historical focused snapshot: 50 passed, 1 expected xfail; the U1 cut-cell xfail is repaired and the final focused count is pending |
| Fischer 2023 Fig. 3 production and repeat | **NE=1620 completed in 5768.40 s** with nonzero peak occupations `[4.7378e-10, 1.4641e-9, 1.79765e-8, 1.88885e-7]`; maximum QP/Ph0 backward errors `9.596746994e-11`/`2.370677984e-6`. The canonical CSV/PDF SHA-256 are now `b99ef820ae9f3430e6fbcbb86fcf8f236f23ed6296504c6d2ff29a1ce8a4eb0e` / `3f2de753cb697c68a307ae1b1ad1f1aa01a9b3975d16a734448b34baf7df7106`; the former CSV hash `bb5482fe831b2073b4c7ad8dfa6159f4f4d473f80e3ef23edac9fa19e2e61a6f` predates the one-line `# pinned_on: win32` stamp, and all numerical rows plus the PDF are unchanged. An uncached full-pin repeat passed in **5224.19 s** on Windows 11 / Python 3.14.3: grid, thermal seed, and escape-time inputs were bitwise equal; certificate maxima were exact; curve max absolute/relative differences were `2.2958874e-41`/`4.52827e-15`; and the PDF was pixel-identical. Source digest: `d176a783fef29e5eff14b0ec8dc27224fc5751af46e30431f7daf44a95d7cd18`. Exact-capacity scalar-density refinement estimates `0.77--1.41%` remaining error. The prior `30--42%` Richardson strong-L1 remainder is retained as a pointwise/total-variation limitation, but a dedicated threshold study establishes weak convergence; see U3. The combined Fig. 3/Fig. 7 non-slow policy suite passed **44 tests with 4 slow deselected in 4.70 s**; the reduced ladder-halving repeat passed in 48.07 s. |
| Fischer 2023 Fig. 5 exact cache-off audit and artifact hardening | **1 passed, 1 stale-pin failure in 8088.34 s.** The original 349.17 s branch guard passed (`low < 1e-8`, `high > 1e-3`, ratio `> 1e6`); its current post-schema rerun also passed in **287.42 s**. The full pin failed after 7737.30 s: 20/42 upper-panel values exceeded `atol=1e-6`, with max absolute/relative differences `1.3979e-4`/`0.98119093`. A subsequent `NE=162/324/648/1296/1620` representative ladder found residual scalar and strong-shape error plus certified upper-panel continuation hysteresis; see U8. The pre-schema pin is quarantined and was not replaced. Its state-bound schema suite passed 9 fast tests; the legacy slow pin xfailed before solving. |
| Fischer 2023 Fig. 6 production | **66/66 targets** in the exact 3x22, 15-column schema; concurrent wall `7599.292171 s` (temperature rows `6693.5442286/7599.1238508/7434.0284306 s`). Max QP backward error `9.266889496677232e-7`, certified/raw phonon backward errors `3.310602868482188e-6`/`3.342306208118692e-6`, and gap-map error `9.92486093309708e-11 µeV`, below `1e-5/1e-5/1e-10` gates. CSV/PDF SHA-256 `9559accfab2d64a834299d12ab0ce5885b52f9278b1b479f991007675a3a5adc` / `e7a7ccae510d50738f5da335998e8a233bfc80912e5f968c104af112405b0fd4`; the production source digest remained `44d6482885a4d2dd197748a92d6406e5034f532e52ce527a577b43b19b80a54c`. The reduced full-state continuation node passed in **3.71 s**. The exact serial `manual_slow` pin node was not invoked; the certified production sweep is equivalent numerical evidence, not an exact pytest-node result. |
| F23/F24 artifact hardening | Fig. 5: 9 fast passed. Figs. 9--13: 12 fast passed; its fresh 21-point run had max QP backward error `1.3659922777e-11`, max QP residual `6.6019352639e-22`, and max outer-map residual `1.55066e-11`, while every historical `Q_i` point drifted (max `14.5144%`). Its `NE=405/810/1620/3240` follow-up shows weak measure convergence but a nonconverged pinned `Q_i` observable; see U9. F24: **58 fast tests passed, 4 slow deselected in 16.32 s** after adding mandatory canonical read/re-certification gates and a cross-libm thermal-seed check; four reduced live solve/write/read paths also passed. Linux NumPy 2.5.1 differed from the Windows-generated Fermi-Dirac seed at only 3/810 entries, with maximum error 2 ULP / `2.71e-16` relative; the reader admits `8*eps` relative with no absolute floor and rejects `1e-9` drift. The four full live pins passed in **76.86 s**. Pre-v2 files remain under `validation/baselines/legacy/fischer_2024_pre_strict_v2/` and are still rejected by current readers. |
| Fischer 2024 refinement and strict-v2 promotion | All four production grids are `NE=810`; the nested `405/810/1620` ladder keeps `omega_PB/dE=126/252/504` exactly integral. At the native five-drive points, `NE=810 -> 1620` changed `x_qp` by `0.0062--0.1903%` and BCS-capacity total-variation shape by `0.495--1.156%` across the fixed-temperature and endpoint-temperature probes. The old/current `x_qp` offsets were `0.858--4.283%`, far larger than the refinement change and primarily an N19 measure/amplitude shift. At paper-topology Fig. 5 the production-grid `x_qp` change was `0.0794--0.0862%` versus `13.37--14.84%` old/current offsets; shape TV was `2.258--2.834%`. At paper-topology Fig. 8, the low-temperature production-grid `x_qp` change was `0.0830--0.1089%`, and the thermal-dominated 0.30 K change was `0.3102%`; each decreased by about `2.5--2.7x` from the preceding refinement. Direct 0.05 K thermal seeds fail the backward-error gate at the two weak drives, while strong-to-weak continuation converges on every grid; at 0.30 K direct and continued results agree within `7.3e-7` relative. These are fixed-grid qpsim-native regression certificates, not pointwise continuum-shape or paper-parity claims. |
| Fischer 2024 promoted artifact hashes | Canonical CSV/PDF SHA-256: Fig. 5 paper topology `de560f1ca01052350122fa8cb9ddce5b363e34c97f6d9c58bb2515af8a9bf944` / `8a11a07739a770dd784cbaececbc8f8911cff8ca249428ce507cab8e64e0fd46`; Fig. 8 paper topology `ca77082ff8449869a30356145d3ebe0cb508baaf56ea914fe00ed90b2b13b0bc` / `3e89910af8777541f88bdd71c5308c170dc3e17764052b2eda013690ca3dfd12`; Fig. 8 native `9c196b9803dc0e0df94d9b9492a2eeebb974b045f12fe77b413514c221c0e925` / `8636d8d9dd0d4900e1481de02187d177c6bb42d68af62e29769d309a7b0354c1`; Figs. 5--7 native `e11d718b6da7a659e6baa51850b9f55aa4ebb2beeed039f1342cd4e73552c4b8` / `069aa4f61e9d36ba6ac0ea0eb35d26c115ad79869948a6d7261d501e370d59f9`. The 84 persisted solve points all passed strict schema, dependency-hash, ordered-certificate, and certified-payload readback. Maximum `(QP backward error, residual)` by family was `(1.13e-7, 1.10e-16)`, `(9.10e-7, 5.92e-13)`, `(2.72e-11, 3.29e-15)`, and `(3.39e-13, 7.45e-17)`, respectively. Provenance-only LF/CRLF and thermal-seed portability rebinds changed the CSV hashes but preserved every table row, ordered certificate, certified-payload hash, and PDF byte-for-byte. |
| Moving-gap/gap-focused groups | Refined-reference moving-gap order and invariant regressions passed; self-consistent-gap file: 8 passed without extrapolation warnings; broader affected T3 backend group: 37 passed |
| Fischer 2023 Fig. 7 certified validation | **48/48 exact tight-contract targets passed on both Windows and Linux.** The portability runs completed on Windows in **982.54 s** and Linux in **946.16 s**. Conservative whole-`qpsim` digest changes were followed by exact uncached Windows recertifications in **901.13 s** and, after the post-publish NumPy-2.5 typing repair, **975.48 s**. The active run's maximum QP/Ph0 backward errors are `9.818804622232902e-9`/`8.270891010038062e-9`, below the `2e-8` gate; newline-canonical solve-contract SHA-256 is `c9c7cc5530ed35e195d6a7da8ffed8c0236d28d49683b966a580201f0ea30ab9`; canonical CSV/PDF SHA-256 remain `e0f957c0d4a43a8ae77d8217eb0e232b8f823634e7d69a9e7a163f0c0f63983a` / `93fc6db803dd8fc0226a3fd137a9052d5a96409c166684b2563f5d2bae524d05`. The portability rebind changed provenance metadata only; all 48 numerical rows and the PDF are unchanged. Hosted single-thread CI later measured loss drift `0.463267%` and `Q_tot` drift `1.634166e-4` at `P=-68 dBm`, `T=0.18 K`, plus `1.2107e-4` `Q_tot` drift at `P=-72 dBm`, `T=0.18 K`; every state remained below the `2e-8` certificate gate. Same-OS comparisons retain `0.4%`/`1e-4`, while the measured Windows/Linux pair uses `0.8%`/`2e-4`. The `2e-19` loss floor is unchanged. The loose pair and exact-source Linux predecessor are archived. The low-temperature plateau node passed in **4.20 s**. |
| Transient photon-kick slow validation | **4 passed in 752.05 s** after regenerating the matched-finite-volume CSV/PDF pin; the exact module passed again in **726.54 s** after the encoding repair. Canonical CSV/PDF SHA-256: `18e2a2424c037e2b6dd64189848765d0a0c75a6b6cc4bed63364c3f2d05c51d1` / `f8d7be510eee34c3a294dc3743035c22a8c2209987695575670794b7bc66ea81`. The CSV hash change is encoding-only: its numerical payload is byte-identical. |
| Web host contract | 15 focused tests passed |
| Static and source-integrity checks | Final repository-wide `ruff check .`, `mypy qpsim` (75 source files), and `python -m compileall -q qpsim tests validation` all passed. `git diff --check` was clean; Git emitted only the repository's Windows LF-to-CRLF conversion notices. |

The Fig. 3 `d176a783...` solve-source digest quoted in its row belongs to the
recorded 5224.19 s repeat. Later conservative whole-tree source changes advanced
the current digest; the historical value is not presented as the current-tree
fingerprint.

### Default-gate deselection inventory — 2026-07-17

The exact current-tree collection commands were:

- `python -m pytest --collect-only -q -m "slow and not manual_slow"`: 16/1530
  selected, 1514 deselected in 3.62 s;
- `python -m pytest --collect-only -q -m manual_slow`: one of 1530 selected,
  1529 deselected in 3.57 s.

Every deselected node is accounted for below. “Direct” means that exact test
node ran; “quarantined” means the current node deliberately xfailed before its
solve because the canonical is pre-schema and must not be overwritten. Four
F24 nodes that originally took this path were later promoted and rerun.

| Deselected node | Marker | Execution/certification status |
|---|---|---|
| `fischer_2023/test_fig3_paper.py::test_reduced_ladder_refinement_preserves_nonzero_branch` | `slow` | Direct pass, 48.07 s |
| `fischer_2023/test_fig3_paper.py::test_matches_pinned_baseline` | `slow` | Direct uncached pass, 5224.19 s |
| `fischer_2023/test_fig5_paper.py::test_high_drive_does_not_false_converge_to_thermal_branch` | `slow` | Direct current-tree pass, 287.42 s; earlier pass was 349.17 s |
| `fischer_2023/test_fig5_paper.py::test_matches_pinned_baseline` | `slow` | Current expected quarantine xfail; the old cache-off numerical comparison failed its stale pin after 7737.30 s |
| `fischer_2023/test_fig6_paper.py::test_reduced_full_state_continuation_is_certified_and_repeatable` | `slow` | Direct current-tree pass, 3.71 s |
| `fischer_2023/test_fig6_paper.py::test_matches_pinned_baseline` | `slow`, `manual_slow` | **Exact node not run.** Equivalent exact-grid production completed and certified 66/66 targets in 7599.292171 s concurrent wall time; the serial pytest wrapper documents an approximately 14-hour cost. |
| `fischer_2023/test_fig7_paper.py::test_matches_pinned_baseline` | `slow` | Direct current-tree pass on hosted Linux under Python 3.13 and 3.14 in CI run `29629653466`. The complete slow matrices each finished with 14 passes and two intended F23 quarantine xfails. Earlier exact 48-target production passed under Linux in 946.16 s and Windows in 982.54 s; later Windows recertifications passed in 901.13 s and 975.48 s. |
| `fischer_2023/test_fig7_paper.py::test_low_temperature_plateau_is_extrinsic_limited` | `slow` | Direct tight-contract pass, 4.20 s |
| `fischer_2023/test_figs_9_13_qi_vs_pread.py::test_matches_pinned_baseline` | `slow` | Current expected quarantine xfail |
| `fischer_2024/test_fig5_paper.py::test_matches_pinned_baseline` | `slow` | Direct promoted-pin pass in the four-node run |
| `fischer_2024/test_fig8_paper.py::test_matches_pinned_baseline` | `slow` | Direct promoted-pin pass in the four-node run |
| `fischer_2024/test_fig8_xqp_pb.py::test_matches_pinned_baseline` | `slow` | Direct promoted-pin pass in the four-node run |
| `fischer_2024/test_figs_5_7_fe_pb.py::test_matches_pinned_baseline` | `slow` | Direct promoted-pin pass in the four-node run; all four passed in 76.86 s |
| `transient/test_photon_kick_response.py::test_matches_pinned_baseline` | module `slow` | Direct numerical pass in the four-node shared-fixture run. A later Ubuntu CI run exposed only the CP1252 title-byte defect after solving; N34 migrates that header to UTF-8 without changing the numerical payload. |
| `transient/test_photon_kick_response.py::test_x_qp_rises_monotonically` | module `slow` | Direct pass in the four-node shared-fixture run |
| `transient/test_photon_kick_response.py::test_late_time_approaches_newton_steady_state` | module `slow` | Direct pass in the four-node shared-fixture run |
| `transient/test_photon_kick_response.py::test_snapshot_x_qp_consistent_with_f` | module `slow` | Direct pass in the four-node shared-fixture run; all four passed in 752.05 s |

The lowest-cost uncertain selections were rerun explicitly on the final tree:

- the six then-pre-schema pin nodes together initially: 6 expected xfails in
  1.97 s; four F24 nodes were later promoted and passed live;
- the two remaining F23 pre-schema pins on the final tree: 2 expected
  xfails in 1.84 s;
- Fig. 7's tight-contract one-point plateau: 1 passed in 4.20 s;
- Fig. 6's reduced full-state continuation: 1 passed in 3.71 s;
- Fig. 5's post-schema high-drive endpoints: 1 passed in 287.42 s.

Thus all 16 non-manual slow nodeids have direct current-tree CI evidence: 14
numerical tests pass and two F23 pre-schema pins remain honest quarantine
xfails. The Fig. 7 full-pin wrapper passed on hosted Linux under both supported
Python versions. Only the serial Fig. 6 `manual_slow` wrapper was not separately
invoked on the final contract; its 66-target production configuration completed
and certified, but that equivalence is not reported as an exact node pass.
No canonical baseline was regenerated during the initial inventory. The later
F24 refinement campaign promoted four strict-v2 replacements and moved the
untouched pre-v2 CSV/PDF pairs into the explicit legacy archive above.

The final aggregate's 4 warnings are three explicit high-energy-tail
diagnostics plus one upstream FastAPI/Starlette deprecation notice; there were no
test failures. It excludes 17 slow/manual selections and therefore does not
clear the figure qualifications described below. In particular, the default aggregate
does not by itself establish a production result or paper parity. The later
Fig. 3 and Fig. 6 production evidence is recorded separately below and remains
subject to its stated refinement and paper-parity limits.

The regenerated transient pin was also qualified by full 810-bin trajectories
to 120 ns at driver steps `dt = 0.2`, `0.1`, and `0.05 ns`. The largest canonical
`0.1`-versus-`0.05` difference was `2.56e-11` in `f` and `5.89e-12` in `x_qp`;
all trajectories remained bounded and monotone at the stored snapshots. This is
evidence of driver-partition step-insensitivity. It is **not** a formal-order
measurement: adaptive ETD subcycling dominated the accepted partitions, and the
remaining differences were at partition/roundoff scale.

The promoted tight-contract Fig. 7 artifact stores five independent certificate fields at
each of 48 `(P_read, T_bath)` targets. Their maxima were: dimensional QP
residual `3.545257078359232e-14` at `(-72 dBm, 0.30 K)`; normwise QP backward
error `9.818804631182414e-9` at `(-100 dBm, 0.14 K)`; dimensional Ph0 residual
`8.640035930462195e-11` at `(-68 dBm, 0.34 K)`; raw direct-form Ph0 backward
error `0.454787067197511` at `(-100 dBm, 0.06 K)`; and
nearest-binary64-certified Ph0 backward error `9.6543334410508e-9` at
`(-80 dBm, 0.26 K)`. The large raw value is retained as a diagnostic rather
than used as a false rejection of a correctly rounded affine root. The two
gated normwise backward errors were both below `2e-8` at every target.

The completed Fig. 3 production artifact is nonvacuous: the four maximum
occupations are `[4.7378e-10, 1.4641e-9, 1.79765e-8, 1.88885e-7]`, and its
maximum QP/Ph0 backward errors are `9.596746994e-11` and `2.370677984e-6`.
Every exact-capacity scalar `x_qp` difference shrank from NE=648 to NE=1620,
with an estimated `0.77--1.41%` remaining scalar error. The distribution-level
Richardson strong-L1 remainder is much larger (`30--42%`). Follow-up refinement
locates that discrepancy at the ideal-BCS threshold ladder and shows much
smaller weak/moment errors, but does not supply pointwise or total-variation
convergence. The promoted artifact is therefore a certified production
regression, not a claim of continuum-shape convergence or paper agreement. Its
uncached full-pin repeat passed in 5224.19 s: input arrays and
certificate maxima reproduced exactly, curve differences were at roundoff
(`2.2958874e-41` absolute, `4.52827e-15` relative), and the PDF was
pixel-identical. Fig. 6 completed its
exact 3x22 production sweep at all 66 targets: the maximum QP, certified Ph0,
raw Ph0, and gap-map diagnostics were respectively
`9.266889496677232e-7`, `3.310602868482188e-6`,
`3.342306208118692e-6`, and `9.92486093309708e-11 µeV`, all inside the
declared `1e-5/1e-5/1e-10` gates.

The cache-disabled Fig. 5 audit deliberately separated branch validity from
pin parity. The two-endpoint high-drive check passed in 349.17 s, proving that
the high point did not falsely collapse to the thermal branch (`low < 1e-8`,
`high > 1e-3`, and `high/low > 1e6`). The complete run then failed the legacy
pin after 7737.30 s: 20 of 42 upper-panel values differed beyond `1e-6`
absolute. At the sharp branch-transition point (`T_B=0.10 K`,
`nbar=2.89e7`) the current value was `4.5783954713003457e-8` versus
`2.4341423568665584e-6` in the pin. The maximum absolute and relative
differences were `1.3979e-4` and `0.98119093`. Total wall time for both nodes
was 8088.34 s. The old pin remains unchanged and is now a legacy xfail; no
replacement is justified without a refinement study.

The Fig. 5 replacement path now retains the exact returned `f` and `n_ph`
states and independently reassembles `x_qp` plus all five QP/phonon certificate
fields on both write and read. Per-point state hashes and a whole logical-row
hash bind the axes, observables, certificates, and compressed float64 states.
Only the exact logical SHA-256 of the untouched pre-schema canonical raises
`LegacyArtifactError`; a merely legacy-looking or malformed current artifact
fails. Its focused schema suite passed 9 fast tests, and the legacy slow pin
xfails before entering the solve.

A representative commensurate-grid follow-up covered the upper-panel
`T_B=0.10 K` transition and high-drive endpoint plus four high-drive lower-panel
temperatures at `NE=162/324/648/1296/1620`. The 12-point rung wall times were
`8.71/30.14/134.14/652.83/999.63 s`. Maximum successive `x_qp` changes were
`23.89%`, `10.75%`, `3.80%`, and `0.443%`; a dyadic Richardson estimate still
leaves about `2.21%` at `T_star/Delta=0.55`. From `NE=1296` to 1620 the strong
BCS-capacity L1 shape change was `4.25--9.24%`, while unit-mass W1 was only
`0.037--0.065%`; roughly `60--68%` of the strong difference remained near the
monochromatic threshold ladder. The current paper-grid transition value
`4.5783536e-8` reproduces the earlier full-run forward branch to `9.2e-6`
relative.

Continuation direction is nevertheless decisive. At `NE=648`, reverse
high-to-low continuation differed from the forward branch by up to `99.987%`
through the upper-panel transition, while the lower-panel direction control
differed by at most `0.0572%`. A minimal paper-grid reverse path
`T_star/Delta=0.95 -> 0.75 -> 0.65 -> 0.60` confirmed persistence: at `0.60`,
forward continuation gives `x_qp=4.5783536e-8` and reverse continuation gives
`4.8295221e-6`, a `99.052%` separation. Both states are numerical roots, not
false convergence: across the forward and reverse paper-grid probes the maximum
independently rebuilt QP/phonon backward errors were `4.56e-7`/`5.70e-7`, well
inside the `1e-5` gate. Without an explicit physical branch-selection contract,
neither direction can be silently promoted as canonical truth. The legacy
CSV/PDF therefore remain untouched and quarantined.

For Figs. 9--13, all 21 historical `Q_i` values moved under the matched
finite-volume measure. The worst relative difference was `14.5143739505%` at
`-66 dBm` (`1.9193302321e11` current versus `2.2452081371e11` pinned), with a
maximum absolute difference `4.5913918014e10`. `Q_tot` and `nbar` remained
numerically close because `Q_tot` is already coupling-limited; that closeness
does not rescue the stale internal-loss pin. The fresh run's maximum QP
backward error, maximum dimensional QP residual, and maximum outer fixed-point
residual were `1.3659922777e-11`, `6.6019352639e-22`, and `1.55066e-11`.
Restoring only the old midpoint density measure in memory reduced the worst
`Q_i` shift from 14.51% to 4.77%, identifying N19 as the majority contributor
but not proving grid convergence. The pre-schema canonical therefore remains
quarantined.

The required commensurate refinement then ran all 21 powers at
`NE=405/810/1620/3240` in `4.90/19.69/80.81/340.88 s`. Successive maximum
`Q_i` changes were `2.83%`, `3.99%`, and `4.38%`; at `-100 dBm` they grew from
`2.20%` to `3.99%` to `4.38%`, so the pinned internal-loss observable is not
grid-converged. This contrasts with improving represented-mass changes
`4.12% -> 1.83% -> 0.938%` and normalized W1 maxima
`0.242% -> 0.209% -> 0.188%`. Strong BCS-capacity L1 remained
`24.43% -> 19.84% -> 17.30%`, with `84--89%` localized near the exact photon
threshold ladder. At `-60 dBm`, `Q_i` changes did decrease to `2.83%`, `0.526%`,
and `0.336%`, but convergence of only the high-power end cannot certify the
21-point family.

An `NE=1620` reverse-power control agrees with the forward sweep to
`6.67e-6` relative in `Q_i` and `9.35e-12` in `nbar`, consistent with the
declared `1e-4` outer tolerance and ruling out a separate sweep branch. Across
all four grids, maxima remained `1.37e-11` for QP backward error,
`7.91e-22` for the dimensional QP residual, and `1.56e-11` for the outer fixed
point. Thus the failure is observable discretization, not branch selection or
residual acceptance. The `NE=405` canonical is not promotable; its pre-schema
CSV/PDF remain untouched and quarantined.

The four Fischer 2024 legacy families also failed current numerics. The
paper-topology Fig. 5 numerical distributions mismatched 83, 72, and 64 of 810
bins across its three drives, with approximately 32--33% maximum relative
differences and `x_qp` shifts of 11.79--12.92%; the fresh maximum QP backward
error/residual were `1.12968e-7`/`1.096e-16`. The five Figs. 5--7 native curves
mismatched 324, 403, 576, 720, and 810 bins; their `x_qp` shifts ranged from
4.107% to 0.851%, while fresh certificate maxima were `3.39e-13` and
`7.45e-17`. Every point in the eight-temperature Fig. 8 native summary moved.
For paper-topology Fig. 8, the old weak-drive low-temperature values were the
thermal seed (`1.589e-20`), an absolute-tolerance false convergence. Resetting
per temperature and continuing the full state strong-to-weak yields
`4.045e-7` and `4.048e-8` at those weak drives; all 36 current solves certified
with maximum backward error `9.10396187215466e-7` and residual
`5.923401480897071e-13`.

A subsequent refinement campaign used the exactly commensurate nested
`NE=405/810/1620` ladder (`omega_PB/dE=126/252/504`). Native-drive scalar
changes from the production grid to `NE=1620` were `0.0062--0.1903%` over
representative weak/strong drives and low/fixed/high temperatures; normalized
BCS-capacity shape TV decreased more slowly, to `0.495--1.156%`. Those scalar
changes are much smaller than the `0.858--4.283%` old/current offsets. For the
paper-topology Fig. 5 points, the production-grid scalar changes were
`0.0794--0.0862%` with `2.258--2.834%` shape TV, versus `13.37--14.84%`
old/current scalar offsets. The legacy differences are therefore primarily the
N19 operator-measure/amplitude change, not current-grid false convergence.

Paper-topology Fig. 8 required a separate branch diagnosis. At 0.05 K, direct
thermal seeds at the two weak drives now fail immediately with gain/loss
backward error one; strong-to-weak full-state continuation instead converges
on all three grids to nonzero certified states. Its `NE=810 -> 1620` scalar
changes were `0.0830--0.1089%`; at 0.30 K they were `0.3102%`, and direct and
continued states agreed within `7.3e-7` relative. Each refinement was about
`2.5--2.7x` smaller than the preceding one.

On that evidence, all four canonical CSV/PDF pairs were regenerated as
strict-v2, dependency-fingerprinted, payload-hashed artifacts with 84 ordered
QP certificates. Strict readback passed, and the four full live pin nodes
passed in 76.86 s. The old pairs are preserved under
`validation/baselines/legacy/fischer_2024_pre_strict_v2/` and remain explicitly
rejected. Promotion establishes qpsim-native fixed-grid regression stability
at paper topology. It does not establish paper parity, and the placeholder
analytic overlays remain outside every acceptance claim.

## Explicit unresolved risks and limits

### U1 — Resolved follow-up: finite-volume cut-cell/support/measure contract

A BCS gap can cut through a finite-volume cell whose center lies below the gap.
For that cell, point-sampled `rho` is zero while the analytic integrated BCS DOS
weight is positive. This item is no longer unresolved on the audit branch:
`SpectralContext.cell_weights` defines capacity and support, and the former
strict xfail is now a passing regression.

Direct `solve_gap` now shares the fail-closed grid-coverage contract: a selected
root or collapse below the first reconstructed cell edge raises unless the
caller explicitly requests the historical constant-left extrapolation, which
still warns. The self-consistent T3 backend never opts in, and the moving-gap
DAE likewise requires lower-face coverage at every stage. Quantitative
moving/self-consistent-gap runs therefore need a grid extending below the
minimum candidate gap.

The repair is deliberately cross-operator rather than a local mask change.
Support, singular capacity, product-measure coherence factors, drive terms,
phonon-side line terms, photon partners, Jacobians, spatial transport, remap,
and observables use the matched `w`/`rho_bar` contract described in N19. The
historical rejected patch and the root-cause derivation are preserved in
`docs/G1-MEASURE-ATTEMPT-2026-07-14.md`.

This is a mass-lumped finite-volume method: singular capacity and BCS coherence
ratios are integrated analytically, while remaining smooth energy/frequency
factors are evaluated on the existing center lattice. A moving edge crossing a
center therefore retains ordinary cut-cell quadrature error; the repair claims
discrete support/conservation consistency, not exact integration of every
kernel over a partially filled cell. Grid refinement remains required for
quantitative moving-edge trajectories.

### U2 — Resolved follow-up: second-order moving-gap coupling

`apply_gap_update` remains a complete algebraic projection whose result does not
scale with positive `dt`; calling it with `dt/2` is not a half flow. It is no
longer composed as the time integrator in `T3DiffusionBackend.step`. The backend
now advances the reduced index-one DAE in persistent `xi` cell averages with a
stage-constrained ETD2 predictor/corrector and rechecks the final public gap
constraint.

Refined-reference trajectories measure approximately second-order convergence
for both public `f` and the self-consistent gap, with an accepted order band of
1.8--2.2. This claim is limited to the documented ideal-BCS, uniform-work-grid,
spatially homogeneous DAE and its enforced support boundaries; see
`docs/Moving_Gap_Time_Integration.md`.

### U3 — Qualified follow-up: Fischer 2023 Fig. 3 threshold-layer refinement

The historical all-zero ratio-10 pin is no longer current. Full-grid
branch-preserving continuation completed at NE=1620 in 5768.40 s and produced
nonzero ordered peaks `[4.7378e-10, 1.4641e-9, 1.79765e-8, 1.88885e-7]` with
independent QP/Ph0 backward errors below the `1e-5` gate. The certified CSV/PDF
were promoted, and 21 focused tests passed.

The former shape warning is now quantitatively classified. On the exact union
partition with BCS capacity `dC = E dE/sqrt(E^2-Delta^2)`, the direct
NE=648-to-1620 capacity-L1 changes are `18.93--19.04%` across all four paper
ratios. `87.5--88.0%` of that strong error lies within one NE=648 cell width
(`2.5 ueV`) of the exact ladder `Delta + n * 20 ueV`. Fits immediately above
the first six thresholds give `f ~ delta^-alpha` with
`alpha = 0.466--0.493`; multiplying by the BCS capacity singularity makes the
represented layer nearly marginal in the strong norm. The four mass-normalized
NE=1620 shapes agree with one another to `1.05e-5` capacity-L1, so the
ratio-zero thermal-phonon path is a representative cheap refinement probe.

That probe was extended without touching the certified artifact to NE=5184
(`dE = 0.3125 ueV`). From NE=2592 to NE=5184, total represented mass changes by
only `0.613%` and the unit-mass Wasserstein-1 shape distance is `0.178%` of the
full BCS-capacity span, while strong capacity-L1 remains `12.34%`; `82.25%` of
that strong error is still within one coarse cell of a threshold. The first-cell
mass decreases under refinement even though the sampled peak rises, consistent
with a narrowing integrable layer rather than a stable numerical atom.

Controls rule out the suspected code defects at the measured scale. Gap-face
capacity, photon shifts, and event-number conservation close to
`1.3e-17` relative; an `E_max = 5 Delta` run at unchanged spacing agrees below
that cutoff to `2.98e-10` of the peak; halving the nonlinear continuation steps
changes only the ratio-10 curve, by `1.00e-6`; and replacing the separable
photon-cell coefficient with an exact correlated cell integral leaves the
successive strong-L1 values unchanged to five significant figures. A fast
manufactured regression in `tests/validation/test_refinement_metrics.py` now
pins the distinction between strong capacity-L1, mass-separated Wasserstein-1,
and threshold-localized error.

The first full slow-CI execution later exposed a separate runtime-contract
issue, not a threshold-refinement defect. Hosted four-thread BLAS made the
roundoff-limited ratio-10 Newton line search reject its polished iterate and
fall back to the certified Picard predictor. Pinning the documented
single-thread contract and using a Fig. 3-only
`coupled_newton_step_rtol=1e-6` removed that reduction-order fallback. Hosted
Python 3.13/3.14 nevertheless converged to nearly identical, fully polished
ratio-10 states that differ from the Windows pin by `1.27723%` over 290/1620
cells, with excellent independent certificates. Ratios 0/0.1/1 remain inside
`1e-4`. The canonical now stamps `win32`; only ratio 10 in the Windows/Linux
OS-family case calibrated by hosted runs receives a `1.5e-2` envelope. Numerical rows and the PDF
were not regenerated.

Thus no collision, quadrature, or physical-regularization source patch is
justified for the threshold-layer refinement behavior. The ideal clean-BCS,
monochromatic-drive model is weakly converging in mass and smooth shape
observables, but its threshold values and total-variation-style curve metric
remain grid-dependent. A grid-independent pointwise curve would require an
explicitly justified physical regularization (for example photon linewidth or
spectral broadening), not a silent interpolation change. The uncached full-pin
repeat passed in 5224.19 s with roundoff-level curve differences and a
pixel-identical PDF, closing local repeatability. Fig. 3 remains a certified
paper-grid regression, not a blanket paper-faithful or continuum-shape claim.

### U4 — Medium: self-pinned curves are not independent paper truth

CSV baselines under `validation/baselines/` primarily prove repeatability
against prior qpsim output. They are not digitized publication data and often
exercise the same implementation on both sides of the comparison. Passing
them cannot establish physics fidelity by itself.

The separate Fischer 2024 paper-target scripts are also explicit about their
incomplete status: `validation/fischer_2024/fig5_paper.py` contains three
`TODO(paper-parity)` analytic overlays, and `fig8_paper.py` contains a placeholder
analytic density formula. All four promoted artifacts must be described as
certified qpsim-native regressions at paper topology, not completed
reproductions or independent paper truth.

### U5 — Medium: one exact validation wrapper has equivalent production evidence

All 16 non-manual slow nodeids have direct current-tree CI evidence: 14
numerical tests passed and two F23 pre-schema pins took their intended narrow
quarantine xfails. The tightened Fig. 7 full-pin wrapper passed on hosted Linux
under Python 3.13 and 3.14 in run `29629653466`. Only the serial Fig. 6
`manual_slow` full-pin wrapper was not invoked, while its equivalent 66-target
production configuration completed with independent certificates. Equivalent
production evidence does not turn that uninvoked pytest node into a pass.

The remaining figure qualifications are substantive rather than missing-test
bookkeeping. Fig. 3's repeat passes while its ideal threshold layers remain
strong-norm/pointwise grid-dependent;
F23 Fig. 5 proved the high-drive branch and then failed its stale pin, and
Figs. 9--13 remains quarantined. The four F24 replacements pass their certified
local pins, but their incomplete analytic overlays and self-pinned nature still
preclude a paper-parity claim.

### U6 — Resolved/qualified: Fig. 7 stopping defect and portability envelope

An exact 48-point Linux rerun of the historical loose contract reproduced the
reported class of discrepancy while still passing every old certificate. Against
the loose Windows canonical, its largest relative loss difference was `23.7528%`
in a vanishing `0.06 K` tail (`1.0944e-19` absolute); visible points included
`0.1793%` and `0.1402%` loss differences. Repeated same-platform and thread-count
probes were stable. Tightening Picard alone did not move the selected points,
whereas tightening the inner Newton tolerance/backward-error gate from
`1e-11/1e-6` to `1e-13/1e-8` collapsed the two visible probes to `1.30e-6` and
`3.78e-7` relative differences. This identifies tolerance-limited termination
at the same root, not competing physical branches or stochastic nondeterminism.

The strict contract was exercised at all 48 points on Windows 11 /
Python 3.14.3 / NumPy 2.4.2 / SciPy 1.17.0 in `982.54 s` and on a Linux ABI /
Python 3.13.14 / NumPy 2.5.1 / SciPy 1.18.0 in `946.16 s`, with BLAS thread
variables fixed to one. The Windows run used the exact promoted numerical knobs
and equations; only source comments/provenance machinery changed afterward. The
Linux run used the exact promoted source, recorded by solve-contract digest
`f9014d9921e690ea00975b249ca9a9807d1941b36f95ce6b2cf25710de6d6372`.
Across those full runs, loss drift for `T>=0.10 K` was at most `0.243957%`,
`Q_tot` drift at most `6.3863e-5`, and the larger relative low-temperature tail
was bounded by `1.0941e-19` absolute loss. Both runs passed the `2e-8` balance
gate. The Linux environment was WSL on the same physical host, so this is an
OS/runtime/library portability test rather than independent-hardware evidence.

The later N31/N33 integration repairs changed files outside the Fig. 7 call
path, but the deliberately conservative solve-contract digest covers all
`qpsim` sources. A frozen-source Windows run therefore repeated all 48 targets
in `901.13 s`. The post-publish CI run then exposed NumPy 2.5's stricter
`np.sum` overload typing in the unrelated ETD balance projection. Narrowing
the scalar/axis-preserving branches changed no Fig. 7 call-path code but again
advanced the whole-tree digest, so a second exact uncached Windows
recertification repeated all 48 targets in `975.48 s`. It passed the established
envelope and certificate gates: versus the preceding canonical, meaningful
loss moved by at most `0.136833%`, `Q_tot` by `4.8194e-5`, and maximum QP/Ph0
backward errors were `9.818804622232902e-9`/`8.270891010038062e-9`. The active
CSV/PDF carry exact final-source digest
`c9c7cc5530ed35e195d6a7da8ffed8c0236d28d49683b966a580201f0ea30ab9`
and SHA-256 `e0f957c0d4a43a8ae77d8217eb0e232b8f823634e7d69a9e7a163f0c0f63983a` /
`93fc6db803dd8fc0226a3fd137a9052d5a96409c166684b2563f5d2bae524d05`.
The final provenance-only LF/CRLF normalization rebind changed only the CSV
header: all 48 numerical rows and the PDF remained byte-for-byte unchanged.
The loose pair and the exact-source Linux predecessor are archived intact.
Those initial regression tolerances reflected the then-measured scale-aware
envelope and rejected the old-to-tight shifts (`3.4523%` in loss, `0.5732%` in
`Q_tot`). The later hosted measurements below refine that portability contract;
none of this asserts bitwise identity, continuum convergence, or paper parity.

The first full GitHub slow gate exposed that the workflow had not actually
enforced the one-thread BLAS condition used above. At `P=-72 dBm`, `T=0.18 K`,
one/two/four-thread WSL probes moved `Q_tot` from the active Windows pin by
`6.0039e-5`/`9.2573e-5`/`1.2099e-4`, with the four-thread value reproducing
CI's `1.2109e-4` miss. CI now pins all three BLAS/OMP thread variables to one.
Exact hosted single-thread Python 3.13/3.14 runs then exposed the residual
tight-contract Windows/Linux envelope: loss drift reached `0.463267%` and
`Q_tot` drift `1.634166e-4`, while every independently assembled balance
certificate remained below `2e-8`. Same-OS comparisons retain loss/`Q_tot`
bounds of `0.4%`/`1e-4`; the Windows/Linux OS-family case calibrated by hosted
runs uses `0.8%`/`2e-4`. The loss `atol=2e-19` is unchanged, and those bounds still
reject the old-to-tight shifts (`3.4523%` loss and `0.5732%` `Q_tot`).

A global tighter inner-Newton gate is not a sound substitute. At `1e-10`, four
`T=0.14 K` loss points moved by `0.8768%--16.1554%` because the solve selected
a different low-occupation branch; `1e-9` also moved low-temperature `Q_tot`
beyond the strict gate. The current solver contract and all Fig. 7 artifact
bytes therefore remain unchanged. Runtime provenance reports BLAS variables,
although the CSV header does not serialize them, so the workflow guard and N35
document the remaining metadata limitation.

### U7 — Resolved follow-up: bounded-memory smooth-gap collisions

The spatial collision cache still retains no more than two exact local-gap
operators, each of which owns three dense `NE x NE` matrices. Profiles with
more than two distinct gaps no longer fail, however: the collision step and raw
endpoint-rate diagnostic stream one exact gap group at a time. This makes
resident kernel memory independent of `NX` while preserving matched local
spectral support. Smooth profiles remain compute-intensive because each
distinct gap still requires an operator build; the repair removes the memory
failure, not that `O(N_gap * NE**2)` construction cost.

### U8 — Medium: F23 Fig. 5 has a certified continuation-direction branch split

The representative grid campaign confirms that the upper-panel sharp
transition is not merely an old-pin or residual problem. Forward low-to-high
and reverse high-to-low continuations reach different independently certified
roots. The split persists on the `NE=1620` paper grid and is two orders of
magnitude in `x_qp` at `T_star/Delta=0.60`; scalar grid error and strong
threshold-layer error also remain measurable. No code-only criterion in the
current validation path establishes which branch a canonical should name. The
pre-schema artifact must remain quarantined until the model or validation
contract supplies a physical branch-selection rule and a new refinement study
certifies that chosen branch.

### U9 — Medium: F23 Figs. 9--13 `Q_i` is not grid-converged

The `NE=405/810/1620/3240` study separates weak convergence of the represented
quasiparticle measure from the plotted internal-loss observable. Mass and W1
improve, and forward/reverse power sweeps select the same branch, but the
largest `Q_i` rung change grows to `4.38%` at the finest comparison. The
production `NE=405` family therefore cannot be promoted merely because
`Q_tot` and `nbar` are coupling-limited and visually stable. A future
replacement needs an observable-level refinement target, especially at the
low-power end; strong pointwise/L1 shape agreement is not an appropriate sole
gate for its singular threshold layers.

## Merge and validation recommendation

The repaired code is materially safer and more numerically honest than the
base, but release claims should remain scoped:

1. retain the green 1513-pass default aggregate and final Ruff, mypy, compile,
   and diff-clean results; if exact marker-gate closure is required, run only
   the remaining Fig. 6 `manual_slow` wrapper and report its serial cost;
2. keep Fig. 3 qualified as a certified paper-grid regression rather than a
   pointwise or capacity-total-variation continuum result; use scalar moments
   and weak measure metrics for refinement unless the physical model is
   explicitly regularized;
3. retain fail-loud quarantines for F23 Fig. 5 and Figs. 9--13: Fig. 5 needs an
   explicit physical continuation-branch contract, while Figs. 9--13 needs
   low-power `Q_i` refinement; retain the four promoted F24 strict-v2 artifacts
   only as qpsim-native fixed-grid regressions,
   preserve their pre-v2 archive, and do not label them paper parity;
4. preserve the matched finite-volume identities and gate any future measure
   change on the adversarial conservation and driven reduced-grid tests, not
   only thermal detailed balance;
5. describe fixed-gap and documented moving-gap transients as verified ETD2,
   keep the moving-gap claim within its enforced DAE/support domain, and describe
   self-pinned figure tests as regression parity rather than paper truth.
