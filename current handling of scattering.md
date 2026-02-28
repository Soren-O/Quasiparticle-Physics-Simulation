# Current handling of scattering and recombination

## Scope
This note documents how the repo currently implements quasiparticle scattering and recombination kernels/dynamics, mainly in:
- `qpsim/solver.py`
- `qpsim/precompute.py`
- `qpsim/models.py`
- validation/parity tests under `tests/`

## High-level flow
1. `SimulationParameters` exposes `enable_scattering`, `enable_recombination`, `tau_s`, `tau_r`, `T_c`, and `bath_temperature`.
2. In energy-resolved mode (`energy_gap > 0`), `run_2d_crank_nicolson(...)` builds an energy grid and prepares collision arrays.
3. Kernels are either:
- loaded from precompute (`precompute_arrays`), or
- built on-the-fly in the solver.
4. Time stepping applies collisions with a BoltzPhlow-style time-relaxation update (`apply_collision_step_time_relaxation*`), optionally split with diffusion as:
- `C(dt/2) -> D(dt) -> C(dt/2)` when both are enabled.

## Kernel definitions

### Recombination kernel
Defined in `recombination_kernel(...)`:
- Inputs: `E_bins`, `gap`, `tau_0` (effectively `tau_r` at call sites), `T_c`, `bath_temperature`
- Formula implemented (code comment labels this Eq. 17 form):

`K_r(i,j) = (1/tau_r) * ((E_i + E_j)/(kB*T_c))^2 * (1/(kB*T_c)) * (1 + gap^2/(E_i*E_j)) * N_p(E_i+E_j)`

- `N_p` is emission factor:
  - finite bath temperature: `1 + n_BE`
  - zero bath temperature: `1` (spontaneous emission only)

### Scattering kernel
Defined in `scattering_kernel(...)`:
- Inputs: `E_bins`, `gap`, `tau_0` (effectively `tau_s`), `T_c`, `bath_temperature`
- Formula implemented (code comment labels this Eq. 16 form):

`K_s(i,j) = (1/tau_s) * (E_i - E_j)^2 / (kB*T_c)^3 * (1 - gap^2/(E_i*E_j)) * N_p(E_i-E_j)`

- Coherence factor is clamped nonnegative.
- `N_p` uses sign of `E_i - E_j`:
  - `> 0`: emission (`1 + n_BE`)
  - `< 0`: absorption (`n_BE`)
  - `= 0`: diagonal explicitly zero (`K_s(i,i)=0`)
- At `T_bath=0`, only emission is allowed.

### Units and conventions
- Energies are in `ueV`, time in `ns`.
- Parity tests indicate kernels are treated as `1/(ns*ueV)` (legacy kernels in `1/(ns*eV)` are converted by `1e-6` before comparison).

## Thermal quantities used by collisions
- `thermal_qp_weights(...)` computes `n_eq(E) = rho_Dynes(E) * f_FD(E,T)`.
- `rho_Dynes` falls back to BCS DOS when `dynes_gamma=0`.
- Thermal recombination source is precomputed as:

`G_therm = 2 * n_eq * dE * (K_r @ n_eq)`

This makes equilibrium stationary when recombination is active.

## Collision RHS used in time stepping
Current runtime path uses `apply_collision_step_time_relaxation(...)` (uniform kernels) or `_nonuniform(...)` (per-pixel kernels).

For each energy bin and pixel:
- Recombination contributes:
  - loss rate: `2*dE*(K_r @ n)`
  - gain: `G_therm`
- Scattering contributes with Pauli blocking:
  - `f = n/rho`
  - blocking factor `max(1-f, 0)`
  - scattering-in: `dE * rho * (1-f) * (K_s.T @ n)`
  - scattering-out rate: `dE * ((K_s * rho_j) @ (1-f_j))`

Then state is advanced via `_apply_time_relaxation_update(...)`:
- solves `dn/dt = gain - loss_rate*n` with frozen coefficients over the step
- uses exponential form
- clamps to nonnegative output

## Uniform vs nonuniform gap handling

### Uniform gap
Precompute stores:
- `K_r`, `K_s` (NE x NE)
- `rho_bins`, `G_therm` (NE)

### Nonuniform gap (`gap_expression`)
Precompute evaluates gap per interior pixel and stores:
- `K_r_all`, `K_s_all` (N_spatial x NE x NE)
- `rho_all`, `G_therm_all` (N_spatial x NE)
- `D_array` (NE x N_spatial)

Implementation detail:
- kernels are cached by unique gap value, then broadcast to pixels sharing that gap.

## Where precompute is used
In `run_2d_crank_nicolson(...)`:
- if `gap_expression` is non-empty and `precomputed` is not passed, solver auto-calls `precompute_arrays(...)`.
- nonuniform precompute triggers per-pixel collision stepping (`einsum`-based path).
- otherwise uniform arrays are used.

## Diagnostics and safeguards related to collisions
- Pauli diagnostics check occupancy `f=n/rho` each step (`_pauli_occupancy_stats`).
- Can warn or raise if occupancy exceeds configured thresholds.
- Also detects forbidden population where `rho ~ 0` but `n > floor`.
- Collision solver name is normalized, but currently only one supported implementation exists: `boltzphlow_relaxation`.

## Legacy/alternate routines currently not on the runtime path
`solver.py` still contains:
- `apply_scattering_step(...)`
- `apply_recombination_step(...)`
- `_collision_rhs(...)`

These are explicit RHS/Euler-style helpers but `run_2d_crank_nicolson(...)` currently applies collisions through the time-relaxation routines above.

## What tests currently assert
- Kernel parity vs legacy MKID simulator at tight tolerance:
  - `tests/test_old_mkid_simulation_parity.py`
- Collision-step closeness to legacy for small `dt`.
- Nonuniform-gap per-pixel precompute arrays match legacy kernels/rho and thermal generation.
- Detailed balance check for scattering kernel:
  - `tests/test_physics_safety.py` and `qpsim/validation.py`
- Thermal stationarity and pure-process checks:
  - `qpsim/validation.py`
- Analytic-style regression cases for scattering/recombination behavior:
  - `qpsim/test_cases.py`

## Current implementation summary
- Kernel physics is implemented explicitly and consistently in one place (`solver.py`).
- Runtime stepping uses a positivity-preserving time-relaxation collision integrator.
- Nonuniform gap is handled via precomputed per-pixel kernels and vectorized `einsum` collision updates.
- Validation coverage is strong for parity, detailed balance, and equilibrium behavior.
