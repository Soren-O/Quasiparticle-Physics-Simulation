# Integrated physics and code audit — 2026-07-12

> **Historical record.** For the current repo state and audit rabbit-hole flags,
> start at [`docs/NEXT-AUDIT-BRIEF.md`](NEXT-AUDIT-BRIEF.md). Treat the findings
> below as done and verify independently.

## Scope and provenance

This audit reviewed the qpsim kinetic, diffusion, phonon, junction, observable,
Web UI, validation, and bundled `papers/qp-diffusion` paths.  It started from
the earlier 18-agent audit at commit `c358a9c`, independently rechecked its
highest-impact claims, then reviewed the newer engine-integration and corrected
paper branches.  The integrated working branch is
`codex/integrated-audit-fixes-2026-07-12`.

The review used independent numerical probes, symbolic checks, mutation/cache
tests, targeted unit and integration suites, Ruff, and strict mypy.  It also
separated three different claims that are easy to conflate:

1. the continuous equations are derived correctly;
2. the discrete implementation converges to those equations;
3. a plotted regression matches independent paper data rather than qpsim's own
   prior output.

## Bottom line

The paper's central longitudinal result remains sound: the A1 diffusion
operator, conservative moving-gap equation, M25 moment equations, collision
integrals, spectral functions, and corrected nonadiabatic derivation agree with
their governing equations.  The published fixed-gap steady-state pipelines are
the most mature part of the project.

The audit did find real defects outside that core: stale mutable caches,
fail-open NaN handling, a moving-gap conservation error, under-resolved BCS
gap-edge observables, an invalid closed-bath Newton path, unit labels/conversion
errors, and validation scripts that could print `FAIL` while exiting zero.
Those issues are fixed on the integrated branch and covered by regressions.

The main remaining physics limitation is now explicit rather than silently
papered over: a self-consistent gap cannot be accurately solved far below the
energy grid's lower support without specifying the occupation of newly opened
states.  `solve_gap` therefore remains biased near `T_c` on a grid anchored at
the zero-temperature gap; inventing thermal values would be wrong for a
nonequilibrium distribution.

## Correctness fixes applied

### Moving-gap and spatial transport

- Moving-gap recovery now conserves the finite-volume invariant
  `sum(rho*f*dE)`, including on nonuniform grids.
- Population crossing a rising gap is spread over bounded above-gap capacity;
  it is no longer discarded or forced into one saturating bin.
- Spectral flow subcycles on the displacement CFL `|gap_dot|*dt`.  The old
  advice to reduce `dt` was ineffective when `gap_dot` was recomputed from the
  same total gap jump.
- Collapsed/nonfinite gap roots, Dynes gap motion, inconsistent scalar/spectral
  gaps, insufficient lower-grid support, and insufficient occupation capacity
  now fail loudly.
- Spatial interface conductance `G_N=0` is accepted as the physical opaque
  limit; `None` retains its distinct meaning of no explicit interface.
- Spatial clipping/conservation warnings are surfaced in Web UI run notes.

### Gap-edge observables and material units

- Pure-BCS density and `sigma_1` integrals use the exact cell measure
  `sqrt(E_hi^2-Delta^2)-sqrt(E_lo^2-Delta^2)` for the integrable DOS
  singularity.  Dynes contexts keep ordinary numerical quadrature.
- At 1620 bins, independent adaptive-quadrature errors at 0.1–0.2 K fell from
  roughly 8–13% to below 1.2% for both `x_qp` and `sigma_1`.
- The material database again stores the conventional `rho_F` in
  `eV^-1 m^-3`; `qp_number_density` explicitly converts the micro-eV energy
  integral to eV.  Persisted Web UI setups now carry a schema version; legacy
  v1 `micro-eV^-1 m^-3` values are migrated by `1e6` when loaded, while new
  eV-unit values round-trip unchanged. Custom YAML and direct observable inputs
  in the old material-scale range are rejected with the same migration hint.
- The preliminary spatial calibration is pinned at `3.132e12 QP/s` for its
  documented source setting.
- Negative `sigma_1` is now reported as signed gain/negative damping in both
  lumped and spatial quality factors, rather than being mislabeled as infinite
  passive Q.

### Solver and cache failure handling

- M25 residual gates, coefficient bundles, physical input bundles, and helper
  candidate paths reject NaN and infinity fail-closed.
- The M25 junction cache uses value fingerprints for all physical, drive, and
  branch-selection inputs.  Doubling the photon scale now updates
  `Gamma_ph_00` from 300 to 600 Hz instead of reusing 300 Hz.
- The final `nbar` re-solve validates its observable exactly like loop
  iterations; physical `+inf` remains the zero-loss limit.
- Newton solvers reject malformed/nonfinite/out-of-range initial states and
  return the exact state whose residual was certified, not a different
  post-hoc clipped vector.
- Unsuccessful Levenberg–Marquardt results cannot enter the M25 branch picker.
- `anderson_depth=1` now performs a real one-history secant update.
- Coupled Newton rejects the closed-phonon `tau_l=0` limit.  That residual has
  an unconstrained conserved-energy mode and no unique joint steady-state
  root; the previous test exited at its exact initial root without exercising
  the singular Jacobian.  Closed-bath Picard/time-domain routes remain valid.
- The ETD source weight uses `expm1`, preserving finite gain when `mu*dt` is
  below subtraction precision.
- Annotation-only device imports no longer create a collection-order-dependent
  solver import cycle.

### Phonon and configuration robustness

- Phonon source/sink and its analytic Jacobian reject unsupported nonuniform
  grids consistently.
- The analytic Kaplan `S_+` endpoint correction is applied only to a
  proportional pure-BCS `K_plus` kernel.  It no longer overwrites a `K_minus`,
  custom, or Dynes kernel merely because the array shape matches.
- Web UI models reject NaN/infinity globally and require ordered energy-grid
  bounds.  Invalid bounds now produce validation errors rather than a server
  error during construction.
- The deprecated, badly biased M25 `max_x_L` picker is rejected by the Web UI.

## Paper derivation and verification

The corrected nonadiabatic calculation includes the previously omitted
`d h + h d` contribution.  The combined space/time verifier now also includes
the outer star products in the spatial Usadel current, rather than correcting
only the inner Keldysh ansatz.  With the full first-order current:

- the longitudinal `hbar * dot(Delta) * grad(Delta)` source cancels;
- the `f_L` sector does not source the transverse channel;
- a transverse first-order correction may remain, as the manuscript now
  states.

All verifier scripts exit nonzero when a checked identity fails, and the full
symbolic verification target runs in CI.  The standalone verification checkout
is synchronized on its own local branch as a separate deliverable.

## Independently confirmed faithful components

- A1 `(p,q)=(1,0)` diffusion, including mutation coverage against the obsolete
  `(1,2)` form.
- M25 residual structure and the S25–S59 coefficient family.
- QP collision detailed balance and the analytic Newton Jacobian.
- Pure-BCS spectral identities and the corrected advanced-Green-function
  convention.
- Kaplan pair-breaking normalization and SciPy's elliptic-parameter convention.
- Kupriyanov–Lukichev energy-channel interface weight and conservative spatial
  flux signs.
- Physical constants and frequency/energy unit conversions used by the core
  solvers.

## Remaining limitations and decisions

### Elevated-temperature/self-consistent gap support

With a 1620-bin grid whose first cell is anchored at `Delta_0`, feeding an
equilibrium occupation into `solve_gap` overestimates the gap by approximately
11.6%, 37.0%, and 79.6% at `T/T_c = 0.8, 0.9, 0.95`.  If the grid instead
starts at the actual equilibrium gap, the corresponding errors are only about
0.015%, 0.027%, and 0.22%.  The singular integral is already removed by the
`E=Delta*cosh(u)` substitution; the error is missing occupation support in
`[Delta_candidate, E_min]`.  A correct general repair requires an adaptive
energy domain or an explicit physical extrapolation/state-remap contract.

### Validation baselines

Several Fischer CSVs are regression baselines generated from older qpsim
output, not digitized paper data.  The corrected BCS quadrature changes
1620-bin low-temperature values by about +8–18% in `x_qp`/`sigma_1` and
-9–15% in passive Q.  Fast analytic pins were updated from independent
quadrature; expensive self-generated figure baselines should be regenerated
and then compared against digitized paper curves.  Passing a regenerated CSV
alone establishes stability, not paper fidelity.

The changed 810-bin transient benchmark was rerun in full: its four slow tests
pass, its stored occupation trajectory is unchanged, and only the derived
steady-state `x_qp` header moved from `0.0329947` to `0.0340626` under the
corrected quadrature.

### Other bounded approximations

- The M25 low-temperature approximation in `_tau_R_inverse` is about 10% off
  near its crossover and remains an explicitly documented approximation.
- Full nonadiabatic spatial/transverse dynamics beyond first order remain
  outside the manuscript and implementation scope.
- Dynes-broadened moving-gap transport is rejected until its complex spectral
  flux is derived; silently combining Dynes DOS with clean-BCS flow would be
  less faithful than rejecting it.

## Verification results

- Default gate: **888 passed, 15 slow/manual tests deselected**.
- Changed transient slow gate: **4 passed**.
- Repository lint: Ruff clean.
- Type checking: strict mypy clean across **73 source files**.
- Symbolic paper verification: all seven scripts passed; the corrected combined
  space/time check reaches numerical residuals around `1e-125`.
- Standalone verifier mirror: all seven scripts passed in 380.8 seconds on its
  synchronized local branch.
- `git diff --check`: clean.

## Reproduction commands

```text
pytest -q
pytest tests/backends tests/solvers tests/collisions tests/observables -q
pytest tests/devices/test_m25_junction.py tests/services/test_rate_equation.py -q
make -C papers/qp-diffusion verify PY=python
ruff check .
mypy qpsim
```

The expensive `slow and not manual_slow` Fischer gate is intentionally
separate; the full Fig. 6 sweep remains a many-hour manual target.
