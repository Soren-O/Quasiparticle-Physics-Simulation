# Part III — Numerics

Numerics half of the software-paper skeleton (§3 of
`papers/qpsim/ARCHITECTURE.md`); §1 is the paper's introduction and has no
in-tree index, so there is no Part I.

Index of the discretizations and solvers in `qpsim.solvers` and the
companion driver services. This document points at the live code.

## Discretizations

- **Energy grid** (`qpsim.grid.energy_grid`): cell-centered uniform
  bins in `[E_min·Δ, E_max·Δ]`. Photon collision channels assume
  uniform spacing and hard-reject nonuniform grids via
  `qpsim.collisions._uniform_grid.uniform_grid_spacing`.
  `SpectralContext.cell_weights` is the represented DOS capacity,
  `cell_density = cell_weights/dE` is the matched transition density, and
  `active_mask` is exactly `cell_weights > 0`. Thus a cell cut by the BCS gap
  remains represented even when its center has point `rho == 0`. Pure-BCS
  coherence ratios are analytically cell-averaged under the same product
  measure; remaining smooth kernel factors use mass-lumped center values.
- **Phonon frequency grid** (`qpsim.collisions.phonon.build_phonon_frequency_map`):
  derived from the QP grid as the union of `{|E_i − E_j|, E_i + E_j}`
  per D3. Guarantees no off-grid interpolation in the e-ph collision
  integral and exact discrete detailed balance at equilibrium. Thermal Bose
  factors use the stable identity `exp(-x) / -expm1(-x)`; the exact
  zero-transfer bin has zero occupation by convention because it is a
  decoupled bookkeeping mode. This preserves positive occupations through
  the binary64 subnormal range, returns zero only on physical underflow, and
  saturates at the largest finite value only when the mathematical occupation
  exceeds binary64 range.
- **Spatial grid** (`qpsim.grid.spatial_grid`): a uniform mesh over a cell
  mask, with the boundary-condition dataclasses and Laplacian assembly that
  go with it. `SpatialBackend` uses a conservative finite-volume
  nearest-neighbour flux operator over that mask. Every exposed face
  carries its own condition — `reflective`, `absorbing`, `dirichlet`,
  `neumann` or `robin` — declared in `SpatialState.conditions` and resolved
  face by face by `qpsim.transport.spatial_operator.face_condition_lookup`;
  a state that declares none is reflective on every face, which is the
  closed-device default. Absorbing, Dirichlet, Neumann and Robin faces each
  carry their own analytic benchmark (`qpsim/webui/bench/bc_*.py`); Robin is
  first order because β is evaluated on the cell centre. Two-gap interfaces
  are optional and add their face conductances to the same assembly.

## Steady-state solvers (`qpsim.solvers.*`)

- `newton_steady_state.py` — Newton with analytic Jacobian on `f(E)`
  alone (frozen `n_ph`). Backtracking line search; raises on singular
  Jacobian. Every return requires both the dimensional maximum-residual
  tolerance and an L1 normwise gain/loss backward-error certificate; a tiny
  absolute collision rate cannot trigger an absolute-only escape. The
  Jacobian's photon paths share the `_uniform_grid` guard with the runtime
  kernels.
- `coupled_newton.py` — Joint Newton on `(f, n_ph)` with a block
  Jacobian. Cross blocks can use scale-aware finite differences or their
  implemented analytic forms. Convergence requires both a relative Newton
  step and a normwise gain/loss backward-error certificate. This remains a
  local solve: strong-bottleneck branches can require parameter continuation,
  and the solver does not choose among multiple physical branches.
- `picard.py` — Mixed-Picard fixed-point primitive with optional Anderson
  acceleration. The coupled phonon convergence policy lives in the steady-state
  service below.
- `anderson.py` — Type-II Anderson (least-squares on residual
  differences; "bad Broyden"), regularized with `rcond=1e-10`. Returns
  `None` when `depth == 0` (acceleration disabled, including the
  branch-collapse reset) or when the history is empty
  (`m = min(depth, len(X_hist)) < 1`), so Picard falls back to a plain mixed
  step. A single history pair already gives a one-column secant update:
  exactly one plain mixed step precedes the first extrapolation.

## Time integrators

- `etd.py` — ETD2 collision substep (Heun-style predictor–corrector
  on the affine `(gain, loss_rate)` form). `expm1` evaluates the small-rate
  exponential weight without a rate floor. Collision paths use a weighted
  Heun-balance projection and stage-aware rate subcycling: a predictor that
  reveals a larger nonlinear loss is rejected and retried at a shorter step.
  Accepted internal steps are counted independently of rejected trials.
  The fixed-gap method is verified second-order in `dt`.
- `DiffusionBackend.step` — for a self-consistent moving ideal-BCS gap,
  stage-constrained ETD2 advances cell-average occupation on persistent
  material coordinates `xi = sqrt(E**2 - Delta**2)`. Predictor and corrector
  materialize onto the fixed energy work grid and solve the branch-anchored
  algebraic gap constraint before evaluating collisions; the final accepted
  state is constrained and materialized again. A refined-reference regression
  verifies second order for both public `f` and `Delta`; zero-collision shell
  invariance, scattering mass conservation, bounds, fixed-gap equivalence, and
  rejected-stage nonmutation are separate gates. The claim applies only to the
  enforced domain in
  `Moving_Gap_Time_Integration.md`.
- `crank_nicolson.py` — Diffusion stepper with `(I − αL)` /
  `(I + αL)` operators, sparse LU factorization. The spatial backend
  subcycles before a stiff CN amplification can become negative, preserving
  damping as well as conservation on reflective boundaries.
- `spectral_flow_tvd.py` — TVD spectral-flow advection for the
  gap-update term `(Δ/E) · Δ̇ ∂_E f`. Monotonized-centered (MC) limiter
  (3-arg minmod reconstruction), upwind flux, zero-flux boundary at `E < Δ`.
- `ssprk.py` — SSPRK(2,2) (Heun) for combined transport+collision
  steps when ETD2 is not the right tool.

## Driver services (`qpsim.services.*`)

- `steady_state.py` — routes between Newton (frozen `n_ph`) and Picard
  (finite `τ_l`) on the phonon configuration; Anderson is a Picard
  sub-mode (`anderson_depth > 0`), and coupled-Newton is invoked directly
  by callers that need it (e.g. fig6), not via this router. Shape and grid
  validation happens at this layer. Finite-`τ_l` convergence combines a local
  `atol + rtol` occupation-change test, an amplitude-independent normwise
  fixed-point guard, and a normwise physical phonon balance certificate
  assembled from the already computed affine source/sink coefficients. The
  generic `picard_balance_tol` is `max(10*picard_tol, 1e-6)` when omitted;
  candidate acceptance requires this physical certificate as well as both
  iterate tests. The phonon certificate retains the raw direct-form error but
  gates the positive excess beyond an operation-level and half-ULP bound for
  the nearest representable affine root. This matters when a bath-pinned
  correction is smaller than one binary64 spacing; resolvable multi-ULP errors
  and negative or non-finite roots still fail. Validation sweeps pass explicit
  limits and independently reassemble both raw and representability-aware
  diagnostics.
  A finite-escape Picard caller may supply `initial_phonon_guess` on the exact
  QP-derived frequency grid. `DiffusionBackend.steady_state` forwards both
  `f` and same-grid `n_ph`, so parameter sweeps continue the full nonlinear
  state rather than silently restarting the phonons at the bath on every row.
  The backend also requires `state.gap == state.spectral.gap` before
  building any phonon operator.
- `DiffusionBackend.steady_state(self_consistent_gap=True)` — the outer
  loop completes a fixed-gap kinetic solve, evaluates a branch-anchored raw gap
  map on that exact occupation, and returns that same state only if the
  unrelaxed relative map residual passes `gap_tol`. Under-relaxation changes the
  next iterate, not the acceptance test; there is no unchecked final kinetic
  polish. A candidate below the first represented energy-cell face raises
  instead of relying on gap-edge occupation extrapolation.
- `nbar_loop.py` — Fischer 2023 Eqs. 59–60 fixed point on `(n̄, Q_i)`
  for KID readout-power sweeps.
- `transient.py` — `run_time_dependent` driver: ETD2 substeps,
  configurable snapshot cadence, optional autonomous early-stop on the raw
  endpoint collision rate, and a midpoint-sampled callable
  `external_flux(t)` for time-varying junction couplings. Callable fluxes
  cannot be combined with early stopping: an instantaneous zero cannot
  certify future convergence of a non-autonomous drive.
  The final substep is truncated so `total_time` is honored exactly.
- `rate_equation.py` — M25 4-unknown closed-form solver. Uses
  `scipy.optimize.root(method='hybr')` for determinism. Every returned
  state passes a row-wise residual gate,
  `|R_i| ≤ max(1e-14 Hz, residual_tol_relative·source_i + 64 eps Σ_j|term_ij|)`:
  a floor with no ceiling, so each row is held to ~`residual_tol_relative`
  of its own physical drive plus that row's float64 cancellation
  granularity. `accept_lm_convergence=True` relaxes only the MINPACK
  "iteration is not making good progress" solver status (`hybrd` info 4/5);
  it never bypasses the residual gate, and every other failure (`maxfev`,
  "no further improvement possible") raises.
  `solve_rate_equation_steady_state_multi_seed`
  brackets the multi-stable branch space and defaults to the minimum-residual
  fixed point. The max-`x_L` picker is deprecated and emits a
  `DeprecationWarning` when selected.
- `rate_equation_coefficients.py` — SI Notes III/IV/V coefficient
  integrals. See `M25_coefficient_integrals.md` for the formulas.

## Stability and safety floors

- `phonon_steady_state` raises on singular or runaway phonon balances
  rather than clipping negative occupations to zero. A "no phonon fixed
  point exists" condition is a loud `RuntimeError`.
- `PhononState.__post_init__` validates finite/nonneg `n_ph`,
  finite/nonneg/strict-monotone `omega_bins`, and finite/nonneg
  `tau_l`.
- Photon collision kernels reject nonuniform energy grids at entry.
- Moving- and self-consistent-gap backend paths require represented lower-edge
  support and fail before extrapolated gap-edge occupation can certify a state.
- M25 has no residual bypass. `accept_lm_convergence` relaxes the MINPACK
  stall status only; the row-wise residual gate (floored at 1e-14 Hz,
  otherwise scaled by each row's own source term) applies to every returned
  state, on the single-seed and multi-seed paths alike.

## See also

- `Part_II_Physics.md` — physics-side index.
- `Validation_Chain.md` — what each tier of test enforces.
- `Device_Architecture.md` — outer Picard / inner Newton split for
  multi-region Devices.
