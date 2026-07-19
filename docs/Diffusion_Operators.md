# Spatial diffusion operators (A1 / A1P / A2 / B / C)

The 1D spatial T3 backend (`qpsim.backends.t3_spatial_1d`) diffuses the
quasiparticle occupation `f(E, x)` under a selectable energy-dependent
operator. The operators are one family in the dressing exponents
`(p, q)`, defined in `qpsim.transport.diffusion.base`:

```
L_{p,q}[f] = N_1^{-p} d/dx ( D_N N_1^q  d f/dx )      (equivalently)
d/dt ( N_1^p f ) = d/dx ( D_N N_1^q  d f/dx )
```

where `N_1(E) = Re[E / sqrt(E^2 - Δ^2)]` is the BCS density of states (the
`rho` carried by `SpectralContext`) and `D_N` is the scalar normal-state
diffusivity. The flux coefficient vanishes wherever `N_1 = 0` (no states
below the local gap edge means no flux); for `q = 0` this implements the
dirty-limit longitudinal spectral coefficient `D_L(E, x)` — 1 above the
local gap edge, 0 below it. Each member fixes three linked quantities:

| `DiffusionModel` | `(p, q)` | conserved density | uniform-gap `D_eff = D_N N_1^{q-p}` | role |
|---|---|---|---|---|
| `A1` | `(1, 0)` | `N_1 f` | `D_N / N_1`  (falls at the gap edge) | dirty-limit Keldysh–Usadel — **default** |
| `A1P` | `(1, 2)` | `N_1 f` | `D_N N_1`  (rises) | transverse-dressing diagnostic |
| `A2` | `(2, 2)` | `N_1^2 f` | `D_N`  (flat) | diagnostic foil |
| `C`  | `(0, -1)` | `f` | `D_N / N_1`  (falls at the edge) | clean / BRT / const-ℓ — legacy `D_E` |
| `B`  | `(0, -2)` | `f` | `D_N / N_1^2` | constant-τ scalar Boltzmann |

`A1` is the operator selected by the dirty-limit Keldysh–Usadel projection:
the conserved energy-resolved quasiparticle density is `N_1 f` and the
longitudinal (energy-mode) flux is **undressed** — `D_L = 1` above the local
gap edge, 0 below — giving a uniform-gap rate `D_N / N_1` that **falls**
toward the gap edge. At a uniform gap this is the same rate (and the same
`f`-dynamics) as the clean/BRT closure `C`: the dirty and clean reductions
agree on the uniform-gap rate and differ only in conserved density and
inhomogeneous-gap structure (`A1` has no DOS-gradient drift; `C` dresses the
flux inside the divergence).

`A1P` attaches the **transverse** (charge-imbalance) flux dressing
`D_T = N_1^2` to the occupation mode. It is not selected by the Keldysh
projection for the energy mode — the `N_1^2` dressing belongs to the
charge-imbalance channel — and is kept as a labeled diagnostic.

## Channel-assignment correction (June 2026)

The advanced-propagator convention controls which scalar channel carries the
`N_1^2` dressing. With the physical conjugation `g^A = -τ³ (g^R)† τ³`
(equal to `-g^R` above the gap), the dirty-limit spectral coefficients are

```
D_L = ¼ Tr[1 - g^R g^A]        = 1    (above the gap; 0 below)   — energy mode
D_T = ¼ Tr[1 - g^R τ³ g^A τ³]  = N_1² (above the gap)            — charge mode
```

so the energy-mode operator is `A1 = (1, 0)`, not `(1, 2)`. The `(1, 2)`
assignment (kept as `A1P`) followed from the opposite conjugation
`g^A = -(g^R)†`, which fails the advanced spectral equation and the
equilibrium gap-equation check. See the qp-diffusion paper
(`~/Documents/qp-diffusion-paper/paper.tex`) and its
`verify_gA_convention.py` for the full audit; thesis Chapter 4 carries the
same correction.

## The A2-mislabel correction

Earlier `qpsim.transport.diffusion` docstrings (and the April
"Energy-Dependent Diffusion Analysis" note) called the operator
`(D/N_1^2) d_x[N_1^2 d_x f]` "usadel". That operator is `A2 = (2, 2)`: it
conserves `N_1^2 f`, **not** `N_1 f`. It is a *diagnostic*, not
the dirty-limit Usadel reduction. The legacy enum names map
onto the family as:

| legacy name | resolves to | `from_name(...)` |
|---|---|---|
| `LEGACY` | `C`  | clean / `D_E` closure |
| `BOLTZMANN` | `B` | constant-τ |
| `USADEL` | `A2` | the diagnostic (**not** A1) |

`SpectralContext.D_E = D_N sqrt(1 - (Δ/E)^2) = D_N / N_1` is exactly closure
`C`; the spatial backend reproduces the legacy modal step to round-off when
`diffusion_model = C` at a uniform gap — and, since the correction, also when
`diffusion_model = A1` (the two coincide there).

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
In these discrete expressions `N_1` is the exact represented cell-average
DOS, `cell_weights / dE`, rather than the point value at the energy-cell
center. For `q = 0`, a cell cut by the gap uses the exact above-gap support
fraction for `D_L`; a cell with only partial support does not receive a
full-strength longitudinal flux.
Because Crank–Nicolson is not L-stable, a single very stiff step can otherwise
flip the sign of a decaying spatial mode and nearly swap a two-cell pulse. The
backend conservatively subcycles each energy channel until
`h max(exit_rate) <= 1`. The finite-volume generator satisfies
`|lambda_max| <= 2 max(exit_rate)`, so every retained CN modal amplification
is non-negative. This preserves positivity and damping without changing the
second-order CN method; the final occupation clip is normally a round-off
no-op and remains only as a fail-loud backstop.

A spatially-varying gap is supplied via `gap_profile` (shape `(NX,)`); the DOS
`N_1(E, x)` is then evaluated per cell. A finite `interface_conductance`
`G_N` turns every face where the gap steps into a
**Kupriyanov–Lukichev interface** carrying the energy-channel current
`F = G_N overline[(N_1^L N_1^R - N_2^L N_2^R)] (f_L - f_R)`
(dx-independent, current-continuous, `f`-discontinuous). The overbar is the
single energy-cell average of the complete coherence-factor product, not a
product of separately averaged `N_1`/`N_2` values. This Maki–Griffin weight is
regular at matched gaps; the bare DOS product `N_1 N_1'` belongs to the charge
channel. Local collisions use the matching gap and spectral support in each
region. Because an exact local-gap collision operator retains three dense
`NE x NE` matrices, a two-entry LRU bounds retained kernels. One- and two-gap
profiles advance in one batched ETD2 call; profiles with more distinct values
stream one exact gap group at a time. Smooth `gap_profile` collisions therefore
retain `O(NE**2)` resident kernel memory rather than `O(NX * NE**2)`, at the
cost of building one operator per distinct gap.

Collision capacity and support use the same exact cell-integrated BCS measure
as transport. A finite-volume energy cell whose center lies below the gap but
which straddles the edge therefore remains represented with its correct
non-zero partial capacity.

## Validation-oracle correction (July 2026)

The numerical-software audit exposed five failures in the diffusion benchmark
tests after the production finite-volume repairs. They were stale validation
oracles, not evidence that transport should return to point sampling:

1. `uniform_gap_packet` compared the measured rate with center-point DOS and
   inverted one caller-visible CN step, although production uses represented
   cell-average DOS and may take `m` internal CN substeps. Its reference now
   uses `cell_weights / dE`, the exact above-gap support fraction for `q = 0`,
   and the analytic `m`-substep inversion.
2. `self_consistent_feedback` built its conserved-density COM and drift
   prediction from center-point DOS. Both now use exact BCS cell weights, so
   the oracle and transport act on the same finite-volume state.
3. `interface_trap` multiplied separately cell-averaged `N_1` and `N_2` terms.
   The correct KL coefficient is the single cell average of the complete
   product `N_1^L N_1^R - N_2^L N_2^R`; an independent direct-energy
   quadrature now supplies that reference, with a raw current-continuity gate.

The two interface failures, two feedback-drift failures, and one uniform-packet
failure were thereby resolved; all **14/14** tests in
`validation/diffusion_operators/` passed in **209.10 s**. This establishes
internal agreement with qpsim's declared finite-volume discretization. It does
not establish agreement with a paper or experiment.

## §7.5 benchmarks

`validation/diffusion_operators/` separates the operators (run any with
`python -m validation.diffusion_operators.<name>`; CSV + figure land in
`outputs/diffusion_operators/`):

1. **`uniform_gap_packet`** — the measured `D_eff(E)/D_N` traces the
   finite-volume `N_1^{q-p}`:
   falling (A1 and C, identical curves), rising (A1P), flat (A2), steeply
   falling (B); `n_qp` conserved to ~1e-15.
2. **`gap_gradient_drift`** — the COM drift velocity matches
   `v = D_N q N_1^{q-p-1} d_x N_1`: A1 (`q = 0`) shows *no* drift, A1P/A2
   (`q = 2`) drift *up* the gap gradient (differing by one power of `N_1`),
   and C/B (`q < 0`) drift *down* it.
3. **`interface_trap`** — a two-gap Kupriyanov–Lukichev interface: current is
   continuous across it while `f` is discontinuous (jump = bulk current /
   the cell-averaged KL conductance); a closed relaxation shows A1 and A2
   reaching *distinct* equilibria (the `p` dressing the driven steady state
   cannot see).
4. **`self_consistent_feedback`** — a heavy population digs a gap well through
   the direct gap closure, and a weak probe's finite-volume conserved-density
   COM follows the operator-family drift prediction on that realized well.

## Scope

This targets the working `t3_spatial_1d` backend. Prelim experiment scripts
were written before the family existed; their committed outputs are historical
`C`-closure runs. The default is now `A1`, so re-running them uses `A1` unless
they are pinned to `C` explicitly — at a uniform gap the two coincide. The
homogeneous `t3_diffusion` Gate-5 spatial path is separate and out of scope
here.
