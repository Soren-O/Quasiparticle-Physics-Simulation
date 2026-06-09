# Spatial diffusion operators (A1 / A2 / B / C)

The 1D spatial T3 backend (`qpsim.backends.t3_spatial_1d`) diffuses the
quasiparticle occupation `f(E, x)` under a selectable energy-dependent
operator. The four operators are one family in the dressing exponents
`(p, q)`, defined in `qpsim.transport.diffusion.base`:

```
L_{p,q}[f] = N_1^{-p} d/dx ( D_N N_1^q  d f/dx )      (equivalently)
d/dt ( N_1^p f ) = d/dx ( D_N N_1^q  d f/dx )
```

where `N_1(E) = Re[E / sqrt(E^2 - Δ^2)]` is the BCS density of states (the
`rho` carried by `SpectralContext`) and `D_N` is the scalar normal-state
diffusivity. Each member therefore fixes three linked quantities:

| `DiffusionModel` | `(p, q)` | conserved density | uniform-gap `D_eff = D_N N_1^{q-p}` | role |
|---|---|---|---|---|
| `A1` | `(1, 2)` | `N_1 f` | `D_N N_1`  (rises at the gap edge) | dirty-limit Keldysh–Usadel — **default** |
| `A2` | `(2, 2)` | `N_1^2 f` | `D_N`  (flat) | diagnostic foil (**rejected**) |
| `C`  | `(0, -1)` | `f` | `D_N / N_1`  (falls at the edge) | clean / BRT / const-ℓ — legacy `D_E` |
| `B`  | `(0, -2)` | `f` | `D_N / N_1^2` | constant-τ scalar Boltzmann |

`A1` is the physically correct operator: the dirty-limit Usadel kinetic
equation conserves the energy-resolved quasiparticle density `N_1 f`, dresses
the flux by `N_1^2`, and gives a rate that **rises** toward the gap edge.

## The A2-mislabel correction

Earlier `qpsim.transport.diffusion` docstrings (and the April
"Energy-Dependent Diffusion Analysis" note) called the operator
`(D/N_1^2) d_x[N_1^2 d_x f]` "usadel". That operator is `A2 = (2, 2)`: it
conserves `N_1^2 f`, **not** `N_1 f`. It is the *rejected diagnostic*, not
the dirty-limit Usadel reduction — which is `A1 = (1, 2)`. The June 2026
theory closed this (conserved quantity is `N_1 f`). The legacy enum names map
onto the family as:

| legacy name | resolves to | `from_name(...)` |
|---|---|---|
| `LEGACY` | `C`  | clean / `D_E` closure |
| `BOLTZMANN` | `B` | constant-τ |
| `USADEL` | `A2` | the rejected diagnostic (**not** A1) |

`SpectralContext.D_E = D_N sqrt(1 - (Δ/E)^2) = D_N / N_1` is exactly closure
`C`; the spatial backend reproduces the legacy modal step to round-off when
`diffusion_model = C` at a uniform gap.

## Using the operators

```python
from qpsim.backends.t3_spatial_1d import T3Spatial1DState
from qpsim.transport.diffusion import DiffusionModel

state = T3Spatial1DState(..., diffusion_model=DiffusionModel.A1)   # default
```

The scheme is an exactly-conservative finite-volume Crank–Nicolson step on
the conserved density `u = N_1^p f` with harmonic-mean face weights
`W = D_N N_1^q` and reflective (zero-flux) ends, so `Σ_x N_1^p f` is conserved
per energy to round-off. `f = u / N_1^p` is recovered and clipped to `[0, 1]`.

A spatially-varying gap is supplied via `gap_profile` (shape `(NX,)`); the DOS
`N_1(E, x)` is then evaluated per cell. A finite `interface_conductance`
`G_N` turns every face where the gap steps into a **Kaplan–Larkin interface**
carrying the current `F = G_N N_1^L N_1^R (f_L - f_R)` (dx-independent,
current-continuous, `f`-discontinuous), matched to the bulk flux
`-D_N N_1^2 d_x f`. Both only affect transport; the collision term still uses
the scalar-gap `SpectralContext`.

## §7.5 benchmarks

`validation/diffusion_operators/` separates the four operators (run any with
`python -m validation.diffusion_operators.<name>`; CSV + figure land in
`outputs/diffusion_operators/`):

1. **`uniform_gap_packet`** — the measured `D_eff(E)/D_N` traces `N_1^{q-p}`:
   rising (A1), flat (A2), falling (C), steeply falling (B); `n_qp` conserved
   to ~1e-15.
2. **`gap_gradient_drift`** — the COM drift velocity matches
   `v = D_N q N_1^{q-p-1} d_x N_1`, so A1/A2 (`q = 2`) drift *up* the gap
   gradient and C/B (`q < 0`) drift *down* it (opposite signs); A1 vs A2
   differ by one power of `N_1`.
3. **`interface_trap`** — a two-gap Kaplan–Larkin interface: current is
   continuous across it while `f` is discontinuous (jump = bulk current /
   `G_N N_1^L N_1^R`); a closed relaxation shows A1 and A2 reaching *distinct*
   equilibria (the `p` dressing the driven steady state cannot see).

## Scope

This targets the working `t3_spatial_1d` backend. Prelim experiment scripts
were written before the family existed; their committed outputs are historical
`C`-closure runs. The default is now `A1`, so re-running them uses `A1` unless
they are pinned to `C` explicitly. The homogeneous `t3_diffusion` Gate-5
spatial path is separate and out of scope here.
