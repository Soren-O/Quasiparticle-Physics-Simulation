# Part II — Physics

Index of the in-tree physics surfaces. The canonical scope reference
remains `New Framework Plan.md` §2–3; this document points at the
authoritative implementations and committed decisions in this repo.

## State variables

- **Quasiparticle distribution `f(E)`** on a uniform energy grid above
  the gap. T3 (isotropic dirty limit) is the only tier shipped in v1.
  T2 (scalar kinetic with angle) and T1 (two-component) are reserved
  in `qpsim.backends.base.Tier`.
- **Phonon occupation `n_ph(ω)`** on a frequency grid derived from the
  QP grid (see D3 in `Phonon_Model_Decisions.md`). Ph0 (local with
  escape) is the v1 target; Ph1/Ph2 are reserved.
- **Gap `Δ`**: BCS weak-coupling self-consistent value, optionally
  re-solved each step from the current `f(E)` via
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

## Gap self-consistency

`qpsim.physics.gap_equation` exposes a two-step API:

- `calibrate_gap(T_c, T_bath)` — equilibrium `Δ_eq` plus cached `1/λ`
  and Debye cutoff.
- `solve_gap(calibration, f, E_bins)` — runtime gap from an arbitrary
  `f(E)` via Brent's method on the reference-subtracted residual.

The cosh substitution `E = Δ cosh u` removes the `1/√(E²−Δ²)`
singularity at the gap edge. Hard-coded BCS weak-coupling ratio
`Δ₀/(k_B T_c) = 1.764`. Materials that depart from weak coupling
(notably Nb at ~1.88) need explicit `Δ_0` rather than relying on
`calibrate_gap` to derive it from `T_c`.

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

## Tier-reduction map

T1 → T2 → T3 reductions are not implemented; T3 is the only backend.
The structural slots for T1 and T2 (`qpsim.backends.base.Tier`,
`qpsim.phonon_models.PhononModel`) are reserved so future additions
are forward-compatible.

## See also

- `Phonon_Model_Decisions.md` — Gate 0 committed decisions D1–D5.
- `Phonon_Escape_Time.md` — `τ_l` derivation and reference chain.
- `M25_coefficient_integrals.md` — SI Notes III/IV/V tunneling,
  recombination, intraband, and photon-assisted integrals.
- `Device_Architecture.md` — Region/Junction/Qubit/Device composition
  layer; how M25 lands as a specific Junction implementation.
- `Part_III_Numerics.md` — solver and discretization choices.
- `Validation_Chain.md` — test/validation tier inventory.
