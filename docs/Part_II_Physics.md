# Part II — Physics

Index of the in-tree physics surfaces. This document points at the
authoritative implementations and committed decisions in this repo.

## State variables

- **Quasiparticle distribution `f(E, r)`** on a uniform energy grid
  above the gap and a cell mask over the film: one scalar occupation per
  energy cell per spatial cell, the isotropic dirty-limit unknown of the
  Keldysh kinetic equation. A one-cell mask is the homogeneous case; the
  operator that couples cells is indexed in `Diffusion_Operators.md`.
- **Phonon occupation `n_ph(ω, r)`** on a frequency grid derived from the
  QP grid (see D3 in `Phonon_Model_Decisions.md`). The bath is local — one
  occupation per frequency per spatial cell — with acoustic escape to the
  substrate.
- **Gap `Δ(r)`**: BCS weak-coupling self-consistent value per spatial
  cell, optionally re-solved each step from the local `f` via
  `qpsim.physics.gap_equation`.

## Spectral functions

Implemented in `qpsim.physics.spectral`:

- BCS DOS `ρ(E) = E/√(E²−Δ²)` and the Dynes-broadened variant.
- Coherence factors `K⁺ = 1 + Δ²/(EE')` and
  `K⁻ = max(0, 1 − Δ²/(EE'))`.
- `SpectralContext` caches `ρ`, `K±`, `D(E)`, and an
  active-energy mask; it rebuilds only when `Δ` moves beyond a
  fractional tolerance.

## Collision channels

- **Electron–phonon** (`qpsim.collisions.phonon`): scattering uses
  `K⁻`, recombination/pair-breaking uses `K⁺`. Optional dynamic
  `n_ph` coupling via `compute_phonon_source_sink`. Detailed balance
  vanishes at `(f_FD, n_BE)` to roundoff.
- **Sub-gap photon** (`qpsim.collisions.sub_gap_photon`): single-mode
  scattering at `ω₀ < 2Δ`, photon convention swaps to `K⁺`.
- **Pair-breaking photon** (`qpsim.collisions.pair_breaking_photon`):
  scattering (`K⁺`) plus pair generation/recombination via the
  reflection partner `E_j = ω_PB − E_i` (`K⁻`).

All three channels are validated at thermal equilibrium under
`validation/analytic/test_detailed_balance.py`.

## Spatial transport

`qpsim.transport.spatial_transport` diffuses `f` across the cell mask and
is composed with the collision integral by `qpsim.backends.spatial`. The
operator is one member of the `(p, q)` dressing family in
`qpsim.transport.diffusion.base` — `A1`, `A1P`, `A2`, `C`, `B` — whose
conserved density is `N_1^p f` and whose flux coefficient vanishes
wherever `N_1 = 0`: below the local gap edge there are no states, so
there is no flux. `A1` = `(1, 0)` is the dirty-limit
Keldysh–Usadel projection and the default; `Diffusion_Operators.md`
derives the family and lists the benchmarks that separate its members.

Every exposed face of the mask carries its own boundary condition
(`reflective`, `absorbing`, `dirichlet`, `neumann`, `robin`) from
`qpsim.grid.spatial_grid`. A gap profile that steps between regions adds a
Kupriyanov–Lukichev interface conductance per energy bin across the
faces that separate them. Transport requires a pure-BCS spectral context;
a Dynes-broadened one is refused rather than approximated.

## Gap self-consistency

`qpsim.physics.gap_equation` exposes a two-step API:

- `calibrate_gap(T_c, T_bath)` — equilibrium `Δ_eq` plus cached `1/λ`
  and Debye cutoff.
- `solve_gap(calibration, f, E_bins)` — runtime gap from an arbitrary
  `f(E)` via Brent's method on the reference-subtracted residual.

The reconstructed first energy-cell face must lie at or below the selected
gap. Otherwise the sampled occupation does not cover the newly available
gap-edge states, and `solve_gap` fails closed. A constant-left occupation
extrapolation is available only through the explicit
`allow_gap_edge_extrapolation=True` opt-in and always warns; state-backed
quantitative paths do not enable it.

For a non-equilibrium occupation the residual need not be monotone and can have
several physical roots. Continuation callers pass
`reference_gap=<previous/current Δ>`; `solve_gap` performs a bounded adaptive
sign-change scan around that value and selects the nearest detected
sign-changing root, independent of `bracket_factor`. Tangent
(even-multiplicity) roots do not change sign and are outside this continuation
contract. The solver still emits a runtime warning listing detected branches
because deterministic continuation does not remove the underlying physical
ambiguity. The self-consistent-gap and moving-gap loops provide this reference
automatically.

The cosh substitution `E = Δ cosh u` removes the `1/√(E²−Δ²)`
singularity at the gap edge. The coupling is fixed by the finite-cutoff gap
equation linearized at the declared `T_c`, so `Δ_eq` closes continuously there.
The default cutoff implies `Δ₀^BCS/(k_B T_c) = 1.76374`. An optional measured
`Delta_0` is retained in the calibration as diagnostic metadata only; it cannot
serve as a second independent anchor. Materials that depart from weak coupling
(notably Nb at ~1.88) need an explicit strong-coupling or phenomenological gap
model to reproduce both measured `Delta_0` and `T_c` quantitatively.

## Phonon sector

Two timescales, distinct physics, distinct equations — see
`Phonon_Model_Decisions.md` and `Phonon_Escape_Time.md` for the full
glossary. Briefly:

- **`τ_PB(ω)`** (Kaplan 1976, `qpsim.physics.kaplan_pair_breaking`):
  bulk BCS pair-breaking lifetime. Lives in the collision integral.
  Closed form `S₊(x) = x · E(1 − 4/x²)` via `scipy.special.ellipe`.
- **`τ_l(ω)`** (Kaplan 1979, `qpsim.physics.phonon_escape`): thin-film
  acoustic escape time `≈ 4d/(η s)`. Lives in the phonon transport
  (relaxation) term. Two builders: `constant_tau_l` (Fischer
  convention) and `acoustic_escape_tau_l` (from material geometry).

The Rothwarf–Taylor trapping factor `ζ = 1 + τ_l/τ_PB` is **not**
applied in PDE backends that evolve `n_ph` dynamically — reabsorption
is already in the collision integral. The rate-equation service
(`qpsim.services.rate_equation`, M25-style) is the only place `ζ`-style
closures legitimately appear.

## See also

- `Diffusion_Operators.md` — the `(p, q)` spatial diffusion-operator
  family, its dirty-limit Keldysh–Usadel default, and the benchmarks that
  separate the members.
- `Phonon_Model_Decisions.md` — committed decisions D1–D5.
- `Phonon_Escape_Time.md` — `τ_l` derivation and reference chain.
- `M25_coefficient_integrals.md` — SI Notes III/IV/V tunneling,
  recombination, intraband, and photon-assisted integrals.
- `Device_Architecture.md` — Region/Junction/Qubit/Device composition
  layer; how M25 lands as a specific Junction implementation.
- `Part_III_Numerics.md` — solver and discretization choices.
- `Validation_Chain.md` — inventory of the validation layers and what
  each tier of test enforces.
