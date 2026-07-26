# Moving-gap time integration

## Implemented contract

`T3DiffusionBackend.step` advances the homogeneous, ideal-BCS collision
problem with a self-consistent scalar gap using a stage-constrained ETD2
method. The authoritative transient unknown is not the public fixed-energy
array. It is a cell-average occupation `g(xi)` on persistent material
coordinates

```text
xi = sqrt(E**2 - Delta**2),       rho_BCS(E, Delta) dE = dxi.
```

The persistent array is stored privately in each accepted
`T3DiffusionState`. A later accepted step reuses it only while the public
energy grid, widths, gap, and occupation still match their synchronization
snapshot. If a caller edits the public state, the backend deliberately
reinitializes the material state rather than using stale hidden data.

At every predictor and corrector evaluation:

1. the fixed energy-cell faces are mapped to material-coordinate faces at a
   trial gap;
2. exact `dxi` overlaps materialize `g` as fixed-energy cell averages `f`;
3. the discrete BCS gap equation is solved together with that materialization,
   so the stage satisfies `G(P_Delta g, Delta) = 0`;
4. phonon and optional photon collision rates are evaluated on the existing
   uniform energy work grid; and
5. individual gain and loss events are conservatively lifted back to the
   persistent material cells.

ETD2 then advances the reduced index-one DAE

```text
g_dot = C(g, Delta),              G(P_Delta g, Delta) = 0.
```

The final accepted occupation is constrained and materialized once more. The
returned public `f`, `gap`, and `SpectralContext` therefore describe the same
discrete constrained state.

`apply_gap_update` remains available as a standalone algebraic projection for
callers that need to constrain and remap one public fixed-energy state. Its
positive `dt` argument is only an API label: changing its magnitude does not
create a fractional gap flow, and it is not used to construct `step`.
`apply_collisions` remains the fixed-gap ETD2 path.

## Conservative work-grid bridge

Let `a[j, i]` be the exact `dxi` overlap of material cell `j` with energy
work cell `i`, `dxi[j]` the material-cell width, and
`w[i] = sum_j a[j, i]` the exact BCS capacity of the energy cell. The public
occupation is

```text
f[i] = sum_j a[j, i] g[j] / w[i].
```

The internal collision routines return `gain[i] - loss[i] f[i]`, where the
internal gain already includes the target Pauli factor. Its available hole
capacity is exactly

```text
h[i] = sum_j a[j, i] (1 - g[j]) = w[i] (1 - f[i]).
```

For `h[i] > 0`, the lift recovers the bare gain-event density
`b[i] = w[i] gain[i] / h[i]` and deposits it over the actual material holes.
Loss remains a local per-particle coefficient:

```text
gain_xi[j] = (1 - g[j]) / dxi[j] * sum_i a[j, i] b[i]
loss_xi[j] =                 1 / dxi[j] * sum_i a[j, i] loss[i].
```

Consequently, summing the material events reproduces the work-grid events to
roundoff:

```text
sum_j dxi[j] gain_xi[j]          = sum_i w[i] gain[i]
sum_j dxi[j] loss_xi[j] g[j]     = sum_i w[i] loss[i] f[i].
```

`ExternalFlux.gain` keeps its documented additive fixed-energy-cell semantics
rather than being reinterpreted as Pauli limited. It is overlap-deposited
directly; its loss array remains a local coefficient. The same aggregate
identities are audited at runtime.

This bridge is also why the collision lattice does not move. Frozen phonon
occupations remain indexed by the original pair-frequency map, and sub-gap and
pair-breaking photon partner rules retain their fixed-energy commensurability
contract. Only the conservative materialization and event lift connect that
work lattice to the moving spectral state.

## Accuracy and invariant evidence

`tests/backends/test_t3_transient.py` independently integrates a driven,
self-consistent trajectory to the same final time with successively halved
steps and compares each result with a much finer reference. The observed
orders in the audit fixture are approximately

| quantity | observed orders |
|---|---|
| public occupation `f` | 1.978, 1.993, 2.011 |
| self-consistent `Delta` | 1.991, 1.985, 2.029 |

The regression gate accepts the asymptotic interval 1.8--2.2 and separately
checks the final public gap equation to `5e-10` micro-eV and the occupation
bounds. Additional tests establish:

- exact frozen-material-shell invariance when collision rates vanish;
- roundoff-level conservation of `sum(dxi * g)` for scattering-only dynamics;
- equivalence with `apply_collisions` when the gap is held fixed;
- nonmutation of accepted persistent state across rejected stiff predictor
  trials; and
- operation of both fixed-energy photon channels during a moving-gap step.

## Domain and fail-loud boundaries

The implemented moving-gap DAE intentionally has a narrower domain than a
general spectral solver:

- `dynes_gamma` must be zero. A real material coordinate satisfying the BCS
  measure identity has not been derived for a Dynes-broadened spectrum.
- The fixed energy grid must be uniform for the collision and photon kernels,
  and its lower face must cover every gap reached by the trajectory.
- The fixed upper boundary must be high enough that stranded occupied
  persistent characteristics above `E_max` stay a small correction. The
  backend measures this tail every materialization: up to `1e-3` of the
  quasiparticle NUMBER may sit above the window (warning above `1e-9`),
  kept at its true `xi` in the persistent representation and re-entering
  if the gap falls; beyond `1e-3` it raises (2026-07-20 adjudication —
  previously any tail above ~`5e-12` raised, which barred
  rising-gap/recovery trajectories). Caveats: (i) the bound is on QP
  number, not energy or collision-rate error — the hidden tail is
  excluded from public observables, collisions, and gap feedback;
  (ii) only persistent rows lying WHOLLY above the window are truly
  frozen — a straddling row shares one occupation value between its
  visible and hidden portions, so the hidden portion co-evolves with
  the visible dynamics (and the visible portion is correspondingly
  mis-weighted). This is a bounded experimental approximation, not a
  validated recovery method: for quantitative recovery studies,
  demonstrate `E_max`-independence of the observables (an error-budget
  study for this regime has not been performed).
- Gap collapse to the normal state is not implemented; a non-positive stage
  solution raises.
- The current path is spatially homogeneous and freezes `n_ph` over the QP
  step, matching the pre-existing transient collision contract.

These checks prevent the second-order claim from being extrapolated to a
different DAE, spectral measure, moving photon lattice, or unresolved finite
energy domain.
