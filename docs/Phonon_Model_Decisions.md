---
title: Phonon Model Decisions (Gate 0, Committed)
description: Committed decisions D1–D5 fixing the v1 phonon-sector model for the qpsim greenfield rewrite.
sourced_from: Phonon Analysis.md (Gate 0 working draft, 2026-04-12)
---

# Phonon Model Decisions

## Motivation

The phonon sector is the single most consequential modeling choice in a
nonequilibrium BCS kinetic framework. Two distinct phonon timescales set
different observables:

- $\tau_{\mathrm{PB}}(\omega)$ (Kaplan 1976) is the **pair-breaking lifetime**.
  It lives inside the collision integral $I_{qp\to ph}$ and $\Gamma_{e\text{-}ph}(E)$
  and controls the quasiparticle lifetime via phonon-mediated recombination.
- $\tau_l(\omega)$ is the **acoustic escape time** of the local phonon bath
  into the substrate. It lives in the transport (relaxation) term of the
  phonon kinetic equation and controls how quickly a nonequilibrium phonon
  population re-equilibrates to the substrate bath.

Confusing these two timescales, or double-counting them via a
Rothwarf–Taylor trapping factor applied on top of a dynamic $n_{ph}(\omega,t)$
solve, produces qualitatively wrong steady states and incorrect transient
kinetics. The decisions below fix each choice unambiguously for v1.

---

## Phonon hierarchy (tiering)

The framework reserves three phonon tiers, matching the QP tier hierarchy
(T1/T2/T3):

- **Ph0 — local phonon field with escape.** Single scalar equation per
  spatial point per frequency:
  $$
  \partial_t n_{ph}(\omega,\mathbf{r},t)
  = I_{qp\to ph}[f,n_{ph}](\omega,\mathbf{r})
  - \frac{n_{ph}(\omega,\mathbf{r},t) - n_{BE}(\omega,T_B)}{\tau_l(\omega)}.
  $$
  No lateral transport; through-thickness uniformity is assumed
  ($d/s \ll \tau_0$). **This is the v1 target.**
- **Ph1 — lateral transport.** Adds an in-plane transport operator
  (diffusion $D_{ph}\nabla^2 n_{ph}$ with $D_{ph} = sd/3$ as the tentative
  default, or ballistic streaming $s\hat{\mathbf q}\cdot\nabla n_{ph}$).
  Deferred to v2. The Ph1 transport operator is specified to degrade to
  a no-op for v1 backends, so switching Ph0 → Ph1 is forward-compatible
  once real transport lands.
- **Ph2 — explicit substrate kinetics.** Couples film and substrate
  phonons at the interface; required for phonon-detector simulations.
  Deferred to v3.

v1 scope targets Ph0. Ph1 and Ph2 are specified here so the state
representation can carry them without restructuring later, but neither
is implemented in v1.

---

## Committed decisions (D1–D5)

### D1 — How $\tau_l$ is specified

Two equally first-class modes for supplying $\tau_l(\omega)$. The
caller picks at `PhononState` construction time; the framework picks
no default.

- **Constant mode.** Caller supplies a scalar $\tau_l$, tiled across
  the phonon frequency grid. Closed-form, no material parameters
  required. Matches the Fischer–Catelani 2023 convention and the
  current reference implementation.
- **Acoustic-escape mode.** $\tau_l(\omega) \approx 4d/(\eta(\omega)\,s)$,
  derived in `Phonon_Escape_Time.md`. Computes $\tau_l$ from `Material`
  fields (film thickness $d$, substrate transmission $\eta$, sound
  velocity $s$). Appropriate when the caller wants $\tau_l$ to track
  geometry changes (multi-material sweeps, thickness sweeps).
  Frequency dependence of $\eta$ is permitted; a frequency-independent
  $\eta$ from the acoustic-mismatch model (AMM) is the minimal concrete
  form.

Same underlying equation in either case — only the numerical value of
$\tau_l$ differs.

### D2 — Phonon source/sink evaluation

**Committed:** the framework retains the `build_phonon_frequency_map`
helper that enumerates the pair-sum and pair-difference frequency mesh
$\{E_i+E_j,\ |E_i-E_j|\}$ from the QP energy grid and precomputes the
phonon-occupation lookup for $I_{qp\to ph}$.

Justification: this is a solved numerical problem in the current `qpsim`
and the implementation is pure-physics; no upgrade is planned. The
spectral-density convention vs occupation convention is handled at the
kernel level ($K_0^s$, $K_0^r$).

### D3 — Frequency grid

**Committed:** the phonon frequency grid is **derived from** the QP energy
grid rather than specified independently. Nodes are the set of distinct
$|E_i - E_j|$ (scattering transfers) and $E_i + E_j$ (recombination
emissions), $i,j$ indexing QP grid points.

Justification: this is the unique grid that makes the $I_{qp\to ph}$
collision integral evaluable without any off-grid interpolation of
$n_{ph}$ when assembling the QP kernels. It guarantees exact detailed
balance at equilibrium on the discrete grid and is a prerequisite for
Validation Check 1 (equilibrium detailed balance) and Check 2 (energy
conservation of e-ph exchange).

### D4 — Ph1 transport operator (tentative, deferred)

**Specified for v1:** the Ph1 transport operator shall degrade to a
no-op so that backends can advertise "Ph1-compatible" without depending
on a real implementation. When Ph1 is implemented (v2), the planned
default closure is isotropic diffusion with
$$
D_{ph} = \frac{s\, d}{3},
$$
appropriate for polycrystalline films with diffuse boundary scattering.
Ballistic streaming remains under consideration for epitaxial/specular
films.

Justification: v1 deliverables (Fischer-style MKID reproductions) do not
require lateral phonon transport, so the operator is specified as
trivial in v1. The diffusion closure is the pragmatic engineering
default; the ballistic option is revisited when Ph1 lands.

### D5 — Sound velocity in the `Material` dataclass

**Committed:** the `Material` dataclass carries all three sound
velocities ($s_L$, $s_T$, $s_D$). The Debye average
$$
s_D^{-3} = \tfrac{1}{3}\big(s_L^{-3} + 2 s_T^{-3}\big)
$$
is the v1 default for scalar-$s$ evaluations (single-branch Debye,
$N_\text{branch}=1$). The $s_L$ and $s_T$ fields are reserved for
multi-branch v3 work.

Justification: single-branch Debye with $s_D$ is consistent with the
Ph0 / v1 scope and matches the Kaplan 1976 evaluation of $\tau_0^{PB}$.
Storing $s_L$ and $s_T$ at the material level costs nothing and makes
the v3 multi-branch extension a pure-additive change.

---

## Additional committed conventions

### No Rothwarf–Taylor $\zeta$ in the PDE

The trapping factor
$$
\zeta(\omega) = 1 + \tau_l / \tau_{PB}(\omega)
$$
is a rate-equation closure quantity. It **is not applied** in the PDE
backends that evolve $n_{ph}(\omega,\mathbf{r},t)$ dynamically. Phonon
reabsorption is already contained in the $(1-f(E))(1-f(\omega-E))\,n(\omega)$
absorption term of $I_{qp\to ph}$; multiplying the recombination rate by
$\zeta$ on top of a dynamic $n_{ph}$ solve double-counts this physics.

Specified behavior: the `PhononState` constructor (Gate 2) shall reject
mixed configurations (dynamic $n_{ph}$ + $\zeta$-renormalized $\tau_0$).
The $\zeta$ route remains available in the separate rate-equation
service (M25 module) where $n_{ph}$ is not tracked dynamically.

### Kaplan 1976 fixes $\tau_{PB}$, not $\tau_l$

The Kaplan 1976 $S_+(\omega/\Delta)$ kernel produces the pair-breaking
lifetime:
$$
\frac{1}{\tau_{PB}(\omega)} = \frac{\Delta}{\pi\,\Delta_0\,\tau_0^{PB}}\,
S_+(\omega/\Delta),
\qquad
S_+(x) = (x+2)\,E(k) - \frac{4x}{x+2}\,K(k), \quad k = \frac{x-2}{x+2}.
$$
This is the lifetime for a phonon of energy $\omega\ge 2\Delta$ against
pair-breaking **inside the superconductor**. It is a bulk BCS quantity
and lives in $I_{qp\to ph}$.

$\tau_l$ is a separate quantity from a **thin-film acoustics** problem
and is derived in `Phonon_Escape_Time.md`.

---

### The phonon spectrum is Debye, and that is why the kernels look like this

**Recorded 2026-08-26.** This assumption was load-bearing, implicit and
undocumented, and its absence caused a real misreading: the phonon side
appears to have *no* density of states, which invites "fixing" by adding one.
Doing so would double-count.

The collision integrals assume a **Debye phonon spectrum**,
$$
\alpha^2 F(\omega) = b\,\omega^2 ,
$$
and $\tau_0$ is *defined* by Kaplan (1976) under exactly that assumption — it
absorbs $b$ together with the $(k_B T_c)^3$ scale. The $\omega^2$ in the
quasiparticle kernel

$$
K_0^s(E_i,E_j) = \frac{1}{\tau_0}\,\frac{(E_i-E_j)^2}{(k_B T_c)^3}\,K^-
$$

is therefore **not kinematic**. It is $\alpha^2 F(\omega)/b$. Assume a
different phonon spectrum and both this frequency dependence and the meaning
of $\tau_0$ change together; neither may be varied alone.

**Where the density of states lives, and where it must not.** The two kernels
are deliberately asymmetric:

| | frequency dependence | why |
|---|---|---|
| quasiparticle side, `build_scattering_kernel_base` | $\propto\omega^2$ | the QP equation **integrates over phonon modes**, so it carries $D(\omega)\propto\omega^2$ |
| phonon side, `build_scattering_kernel_phonon_side` | none — $2K^-/(\pi\Delta\tau_0^{PB})$ | the phonon equation is written **per mode**, and a single mode has no density of states |

Their ratio is exactly $\omega^2$ times a constant (verified numerically to
$7\times10^{-16}$), which is the signature of this structure and a cheap check
that it has not been disturbed.

Consequences worth stating, because each has been misread at least once:

- `n_ph` is an **occupation** (Bose), the direct analogue of the
  quasiparticle $f$ — intensive, dimensionless, unbounded above. It is *not*
  a density.
- There is deliberately **no per-bin phonon density-of-states array**, and
  none should be added. The absence mirrors `SpectralContext.cell_density` on
  the quasiparticle side only in appearance; the two sides are asymmetric on
  purpose because the mode integral happens on one of them.
- `compute_phonon_source_sink` sums $dE\times(\dots)$ along an anti-diagonal.
  That is discretising $\int dE$ **at fixed $\omega$** — a per-mode source —
  not a sum over modes. It therefore needs no $1/D(\omega)$ normalisation.
- Counting total excitations *does* need the measure: the conserved
  combination weights $n_{ph}$ by $\propto\omega^2$. Conservation checks must
  use it; the kinetic equations must not.
- A frequency-dependent $\alpha^2F$ (a real measured spectrum, an Einstein
  mode, a soft-mode material) is **not** expressible by editing $\tau_0$. It
  requires replacing the $\omega^2$ in the QP kernel and re-deriving the
  normalisation, and would break the ratio check above.

---

### Quasiparticles are stored as cell averages, phonons as point samples

The two populations are discretised with **different and deliberately
different** conventions. Neither is wrong; writing code that forgets which one
it is holding is what goes wrong.

**Quasiparticles: a cell average.** The energy axis is a finite-volume mesh.
What is stored for cell $i$ is not "the occupation at $E_i$" but the
occupation averaged over the whole cell, and the number of quasiparticles in
that cell is
$$ n_i \;=\; \rho_i\,f_i, \qquad \rho_i \;=\; \int_{E_i^-}^{E_i^+}\! N(E)\,dE $$
with $\rho_i$ the **integral** of the BCS density of states across the cell,
not $N(E_i)$ times a width. This matters entirely because $N(E)$ diverges at
the gap edge: the first cell has finite weight $\sqrt{E^+{}^2-\Delta^2}$ while
$N(E_i)\,\Delta E$ is unbounded as the mesh refines. `SpectralContext.
cell_density` is that integral, and it is what the collision integrals use in
all fifteen places they need a quasiparticle count.

**Phonons: a point sample.** The phonon equation is solved **per mode**. Every
frequency bin is an exact event frequency — a value $|E_i-E_j|$ or $E_i+E_j$
that some quasiparticle pair actually emits (verified: all 900 bins of the
shipped grid, no exceptions). $n_{ph}$ at that bin is the occupation number of
that one mode, a dimensionless number, and it carries no measure — for the
same reason given in the Debye section above: the per-mode equation
discretises $\int dE$ at fixed $\omega$, so a density of states would
double-count.

**So the convention is mixed, and that is correct** — the quasiparticle
equation integrates *over* states and needs the measure, the phonon equation
is stated *per* state and must not have it.

**The interface between them is where this can go wrong, and did.** A
quasiparticle pair is a two-dimensional *cell* in $(E,E')$; the phonon it emits
is a *point* in $\omega$. Converting one to the other is a real operation, and
depositing the whole cell into the single bin nearest its centre is not that
operation — it silently reinterprets a cell average as a point sample. That is
the origin of the pair-marginal defect recorded in
`docs/HELD-BACK-ADJUDICATION-2026-08-11.md` (item 101): whole-cell deposit
along the anti-diagonal gives a threshold of $4\Delta$ where Kaplan gives
$\pi\Delta$, and refining the mesh does not remove it, because the error is in
the *representation*, not the resolution. The correct conversion splits each
cell by the area it actually shares with each frequency strip
(`qpsim/collisions/pair_split.py`), and the read-back must be its transpose or
detailed balance breaks. **Rule: whenever a quantity crosses between the two
populations, say in the code which representation it is in and convert
explicitly.**

**How wrong is it to confuse them? Exactly $1/\sqrt{2}$.** Worth having the
constant rather than a warning. On a grid whose lowest face sits on the gap,
the first cell is $[\Delta,\Delta+h]$, and
$$ \underbrace{\sqrt{(\Delta+h)^2-\Delta^2}}_{\text{cell integral}} \to \sqrt{2\Delta h},
\qquad
\underbrace{N(\Delta+h/2)\,h}_{\text{point sample}} \to \sqrt{\Delta h}. $$
Both vanish like $\sqrt{h}$, so nothing looks unstable — but their ratio tends
to $1/\sqrt2$ and *stays* there. A point sample undercounts the gap-edge cell
by **29.3% at every resolution**. That is the signature to recognise: an error
that survives refinement is in the representation, not the mesh, which is
exactly why no convergence study catches it and why $4\Delta$ vs $\pi\Delta$
went unnoticed for so long.

**One known exception, currently unreachable — and it is a *different* fault.**
The junction band weights use the exact cell integral for a pure BCS spectrum,
but fall back to centre-value × overlap when Dynes broadening is switched on.
The two failures look alike and are not, and the distinction is the useful
part: broadening replaces the singularity with a Lorentzian of finite width
$\Gamma$, and a point sample of a *smooth* function is merely under-resolved.
It converges normally once the cells are narrower than the peak. Measured
gap-edge error against the exact cell integral, versus $\Gamma/\Delta E$:

| $\Gamma/\Delta E$ | 0.05 | 0.27 | 1.09 | 5.4 | 21.7 |
|---|---|---|---|---|---|
| gap-edge cell error | 82% | 14% | 2.9% | 0.10% | 0.01% |

So this one is a **resolution condition** — cells no wider than the broadening
— and refining the mesh at fixed $\Gamma$ moves *right* along that table and
repairs it, unlike the pure-BCS case above, which refinement cannot repair at
all. The hazard was only that nothing stated the condition, and a physically
small $\Gamma$ sits at the left end on a default grid. It is not currently a
live defect: the engine refuses to run with Dynes broadening ("spatial
transport requires a pure-BCS spectral context"), so only the pure-BCS branch
ships. The guard exists so that implementing broadened kernels has to confront
the condition rather than inherit it silently.

`tests/physics/test_collocation_convention.py` pins all of it: the $1/\sqrt2$
constant, the phonon bins being exact event frequencies, and the guard.

---

## Glossary: three distinct timescales

| Symbol | Name | Physics | Where it enters |
|---|---|---|---|
| $\tau_{\mathrm{esc}}$ | Phonon escape time | Mean time before a phonon in the film crosses into the substrate. Pure thin-film acoustics (AMM/Kapitza). Independent of superconductivity. | Transport (relaxation) term of phonon kinetic equation. |
| $\tau_l$ | "Local" phonon lifetime (Fischer convention) | Identified with $\tau_{\mathrm{esc}}$ in v1: $\tau_l = \tau_{\mathrm{esc}}$. The "lifetime" of the local phonon population against loss to the substrate bath. | Bath-relaxation term: $-(n_{ph} - n_{BE}(T_B))/\tau_l$. |
| $\tau_{\mathrm{PB}}$ | Pair-breaking lifetime | Mean time before a phonon of energy $\omega \ge 2\Delta$ breaks a Cooper pair. Bulk BCS quantity, Kaplan 1976 via $S_+(\omega/\Delta)$. | Collision integral $I_{qp\to ph}$ (specifically the $K^+$ recombination/pair-breaking term), and the QP-side $\Gamma_{e\text{-}ph}(E)$. |

**Common pitfalls:**

1. **Writing $\tau_l$ when you mean $\tau_{\mathrm{PB}}$.** In Fischer 2023,
   $\tau_l$ is unambiguously the escape time; it has no $S_+(\omega/\Delta)$
   structure. Do not cite Kaplan 1976 as the source of $\tau_l$; the
   correct citations are Kaplan 1979, Eisenmenger 1976, Little 1959.
2. **Applying $\zeta = 1 + \tau_l/\tau_{\mathrm{PB}}$ on top of a dynamic
   phonon solve.** The reabsorption physics $\zeta$ encodes is already
   present in $I_{qp\to ph}$. Dynamic $n_{ph}$ + $\zeta$ = double-counting.
   See `Phonon_Escape_Time.md` for the detailed argument.
3. **Using $\tau_{\mathrm{transport}} \approx d/s$ as $\tau_l$.** The
   transport time (one bounce, direction randomization) is
   $\tau_{\mathrm{transport}} \approx d/s$. The escape time has an
   extra factor $4/\eta$ because (a) the free surface reflects totally
   and (b) each substrate-side bounce transmits with probability only
   $\eta$. Conflating the two gives a factor-$4/\eta \sim 20$ error in
   the phonon diffusion length.

---

## Cross-references

- `Phonon_Escape_Time.md` — derivation of $\tau_l \approx 4d/(\eta s)$,
  reference chain (Kaplan 1979, Eisenmenger 1976, Little 1959), numerical
  example for Al on sapphire, and the double-counting argument.
- Kaplan et al. 1976 — source of $\tau_{\mathrm{PB}}(\omega)$ via
  $S_+(\omega/\Delta)$.
- Fischer & Catelani 2023 — MKID synthesis; Appendix A ties $\tau_{PB}$
  (Eq. A2) and $\tau_l$ (Eq. A5) together; Table I parameter set is the
  v1 acceptance-test baseline.
