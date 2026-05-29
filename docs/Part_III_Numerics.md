# Part III — Numerics

Index of the discretizations and solvers in `qpsim.solvers` and the
companion driver services. The canonical reference remains
`New Framework Plan.md` §4.3; this document points at the live code.

## Discretizations

- **Energy grid** (`qpsim.grid.energy_grid`): cell-centered uniform
  bins in `[E_min·Δ, E_max·Δ]`. Photon collision channels assume
  uniform spacing and now hard-reject nonuniform grids via
  `qpsim.collisions._uniform_grid.uniform_grid_spacing`. The
  active-energy margin in `SpectralContext` is keyed off the local
  bin spacing at the gap edge to behave correctly on the piecewise
  grids used by M25's two-band R electrode.
- **Phonon frequency grid** (`qpsim.collisions.phonon.build_phonon_frequency_map`):
  derived from the QP grid as the union of `{|E_i − E_j|, E_i + E_j}`
  per D3. Guarantees no off-grid interpolation in the e-ph collision
  integral and exact discrete detailed balance at equilibrium.
- **Spatial grid** (`qpsim.grid.spatial_grid`): uniform 1D mesh with a
  5-point Laplacian. Gate 2 ships the homogeneous `N_spatial = 1`
  case only; spatial T3 lands at Gate 5.

## Steady-state solvers (`qpsim.solvers.*`)

- `newton_steady_state.py` — Newton with analytic Jacobian on `f(E)`
  alone (frozen `n_ph`). Backtracking line search; raises on singular
  Jacobian. The Jacobian's photon paths share the `_uniform_grid`
  guard with the runtime kernels.
- `coupled_newton.py` — Joint Newton on `(f, n_ph)` with a block
  Jacobian. The cross blocks are evaluated by finite difference for
  v1 (acceptable up to Fischer-scale `(NE, N_ω)`; an analytic
  cross-block remains a post-Gate-4 optimization). Required for the
  `τ_l/τ_PB ≥ 10` regime where Picard stalls.
- `picard.py` — Mixed-Picard fixed point with optional Anderson
  acceleration; relative-L∞ convergence test.
- `anderson.py` — Type-II Anderson (least-squares on residual
  differences; "bad Broyden"), regularized with `rcond=1e-10`. Returns
  `None` on `m < 2` so Picard falls back to a plain mixed step.

## Time integrators

- `etd.py` — ETD2 collision substep (Heun-style predictor–corrector
  on the affine `(gain, loss_rate)` form). The exponential factor is
  fallen back to its first-order Taylor series for `μ < 1e-14` (a
  `dt`-independent decay-rate floor) to avoid catastrophic cancellation.
  Verified second-order in `dt`.
- `crank_nicolson.py` — Diffusion stepper with `(I − αL)` /
  `(I + αL)` operators, sparse LU factorization. Conservation
  preserved on reflective boundaries.
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
  validation happens at this layer.
- `nbar_loop.py` — Fischer 2023 Eqs. 59–60 fixed point on `(n̄, Q_i)`
  for KID readout-power sweeps.
- `transient.py` — `run_time_dependent` driver: ETD2 substeps,
  configurable snapshot cadence, optional early-stop on `dt`-rate, and
  a callable `external_flux(t)` for time-varying junction couplings.
  The final substep is truncated so `total_time` is honored exactly.
- `rate_equation.py` — M25 4-unknown closed-form solver. Uses
  `scipy.optimize.root(method='hybr')` for determinism;
  `accept_lm_convergence=True` exempts only the `is_no_progress_stall`
  case (cancellation-floor regime ~1e-5 Hz, hard-capped at 1.0 Hz)
  from the residual check. `solve_rate_equation_steady_state_multi_seed`
  brackets the multi-stable branch space and picks the max-`x_L`
  candidate (paper-matching nonequilibrium branch).
- `rate_equation_coefficients.py` — SI Notes III/IV/V coefficient
  integrals. See `M25_coefficient_integrals.md` for the formulas.

## Stability and safety floors

- `phonon_steady_state` raises on singular or runaway phonon balances
  rather than clipping negative occupations to zero. A "no Ph0 fixed
  point exists" condition is now a loud `RuntimeError`.
- `PhononState.__post_init__` validates finite/nonneg `n_ph`,
  finite/nonneg/strict-monotone `omega_bins`, and finite/nonneg
  `tau_l`.
- Photon collision kernels reject nonuniform energy grids at entry.
- M25 residual bypass requires both the cancellation-floor status
  marker and the user-supplied flag, and is hard-capped at 1.0 Hz.

## See also

- `Part_II_Physics.md` — physics-side index.
- `Validation_Chain.md` — what each tier of test enforces.
- `Device_Architecture.md` — outer Picard / inner Newton split for
  multi-region Devices.
