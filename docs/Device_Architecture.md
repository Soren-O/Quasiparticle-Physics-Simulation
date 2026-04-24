---
title: Device Architecture (Region / Junction / Qubit / Device)
description: Three-layer bottom-up architecture for multi-region superconducting devices with tunnel coupling, optional qubit coupling, and moment-closure reductions. Obsoletes the hand-coded M25 rate-equation service.
status: design proposal, 2026-04-24
---

# Device Architecture

## 0. Summary

The current backend hierarchy (T1/T2/T3) describes the QP kinetic
equation in **one** superconducting region. Everything device-level
— two electrodes coupled by a tunnel junction, a Josephson-qubit
two-level system coupled to those tunneling events, parity selection
rules — currently lives in a hand-coded 4-variable ODE system
(`qpsim.services.rate_equation`). That worked for Gate 8 Strategy A
but has architectural problems: physics baked in at the moment-closure
level, M25-specific choices fused with general structure, and numerical
pathologies from fast tunneling terms dominating the residual of a
4-variable system that's really the moment image of a much
better-conditioned kinetic equation.

This document proposes a three-layer replacement:

* **Layer 1 — Region.** One superconducting region. Owns material,
  geometry, backend, state. T1/T2/T3 are the three possible backends
  (v1 ships T3 only). Gains one new surface: an external flux
  ``G_ext(E, r, t)`` on the RHS of the kinetic equation.
* **Layer 2 — Device.** Composition of 1+ Regions + 0+ Junctions +
  0-or-1 Qubit. A Junction couples two named regions via an
  E-resolved tunneling-rate evaluator. A Qubit is an optional TLS
  whose transitions are driven by junction tunneling events. The
  top-level Device solver iterates: evaluate junctions → push fluxes
  into regions → step each region's backend → evolve qubit → repeat
  to steady state or over time.
* **Layer 3 — Moment-closure reductions.** Specific ansätze on the
  Region state (e.g. Fermi-Dirac per sub-band) reduce Layer 2 to
  algebraic rate equations. M25's 4-variable system is one specific
  Layer-3 reduction; the mapping makes that explicit rather than
  hard-coding.

The M25 validation then runs as: compose a ``Device`` with two
regions (L and R), one ``Junction``, one ``Qubit`` with parity, and
a photon drive. Solve Layer 2 at physical rate scales (region-local
~10³ Hz terms, no 10¹⁸ cancellation pathology) and post-process for
the densities/chemical potentials the paper plots. The existing
Stage A coefficient-integral machinery
(``M25PhysicalParameters``, the S_ph / Γ̃ / r / τ_R/E evaluators)
all **reuse** — they become the guts of a specific M25
``Junction`` implementation and the derivation of Layer 3 for
comparison/speed.

This doc is the design proposal only. The implementation plan is
§7, phased across 5–6 sessions. Every phase lands green with
intermediate checkpoints.

---

## 1. Why a new layer (not a patch on top)

### 1.1 What Gate 8 Strategy A actually does

The service at `qpsim.services.rate_equation.solve_rate_equation_steady_state`
solves a 4-variable algebraic system `(p_1, x_L, x_{R>}, x_{R<})`
for the M25 boxed equations (main-text Eqs. 3-6). The inputs are
a fully-packed `M25Coefficients` bundle: 12 tunneling rates
Γ̃^α_{ij}, recombination r^α/r^{<>}, thermal-phonon and photon-
assisted generation, intraband τ_R/τ_E, branching ξ. The
Stage A+B machinery builds that bundle from primitive physical
parameters via the SI Note III/IV/V coefficient integrals.

This **works** — mathematically correct, well-tested — but:

1. **The Fermi-Dirac ansatz is baked in.** The x_α's are moments
   under the assumption `f_α(E) = f_FD(E − µ_α)` per sub-band. You
   cannot run this machinery on a **non-Fermi** distribution
   (athermal photon drive, injection pumping, transient relaxation)
   without writing an entirely new solver.
2. **The gap-asymmetric two-electrode topology is baked in.** The
   three sub-band split (L, R<, R>) lives in the signatures of
   Γ̃^α, r^α, g^α. Single-electrode, N-electrode, NIS, or
   symmetric-gap JJ problems each need a new hand-coded variant.
3. **The qubit level structure is baked in.** Two levels with
   parity → specific matrix elements in S10, fused into the
   coefficient evaluators. Transmon qutrit or multi-qubit coupling
   would need a different service.
4. **Numerical problems.** The 4-variable residual has tunneling
   terms ~10¹¹ Hz cancelling to source rates ~10⁻⁸ Hz. A
   19-order-of-magnitude cancellation that Newton + FD-Jacobian
   can't resolve cleanly at float64 (Stage B investigation,
   2026-04-24). The kinetic-equation image of the same system has
   individual term magnitudes ~10³ Hz — well-conditioned.

These are all consequences of writing the moment closure **as the
primary object**. The kinetic equation is the primary object; the
moment closure is a derived convenience.

### 1.2 What T1/T2/T3 don't provide

All three tiers (per NFP §2.2) describe one superconducting region:
one `f` field (or `(f_L, f_T)`), one gap `Δ`, one material, one
spatial mesh. They have no notion of:

* Multiple discrete regions coupled by tunneling — L and R as
  **separate** kinetic-equation instances.
* A **Junction** — a coupling that transfers QPs between two
  regions based on their f(E) distributions.
* A **Qubit** — a discrete two-level system that drives and is
  driven by tunneling events.
* **Parity** selection rules.

This is the physics the M25 rate equation adds on top. The
design below makes those additions first-class, so they apply
to any tier (T1/T2/T3) and any region count (1, 2, N), and the
M25-specific choices become specializations at the Junction level
rather than framework-level assumptions.

---

## 2. Layer 1 — Region

A `Region` is one superconducting region with one QP distribution
and one gap. Existing `T3DiffusionState` is the v1 implementation;
T2 and T1 slot in later.

### 2.1 Dataclass sketch

```python
@dataclass
class Region:
    """One superconducting region with its own backend state.

    Regions are the primitives the Device composes. A 0D region is a
    `geometry` with a single spatial cell; a spatial region carries a
    mesh. A Device may contain regions with heterogeneous backends
    (T3 on one, T2 on another) as long as every Junction between them
    has a compatible cross-region flux evaluator.
    """
    name: str                               # unique within the Device
    material: Material
    geometry: Geometry                      # 0D or spatial mesh
    energy_grid: np.ndarray                 # shape (NE,) — E values ≥ Δ
    backend_kind: Tier                      # T3_DIFFUSION | T2_KINETIC | T1_TWO_COMPONENT
    state: BackendState                     # tier-specific: f(E) / (f,p̂) / (f_L,f_T)
    phonon_state: PhononState               # local thermal bath or n_ph field
    gap_self_consistent: bool = False       # solve Δ together with f
    # Populated by the solver at each step — NOT user input.
    current_Delta: float | None = None
```

### 2.2 The external-flux surface

The existing kinetic equation for T3 is

    ∂_t f + (Δ/E) Δ̇ ∂_E f
      = ∇_r · [D(E, Δ) ∇_r f] + I_coll[f, n_ph](E, r) + G_ext(E, r, t)

where `G_ext` is the existing `external_generation_runtime`
plumbing on `T3DiffusionBackend`. Phase 2 of the implementation
plan renames/generalizes this to **`external_flux`** so it can
accept either (positive) injection or (negative) extraction, and
it's called once per Region per solver step with the aggregated
fluxes from **all** Junctions attached to that Region.

`external_flux` has the same dimensional shape as the collision
integral: `(NE, NR)` for T3 (energy × spatial), `(NE, NR, Nθ, Nφ)`
for T2/T1. 0D regions collapse NR → 1.

**No other changes to the Region interface.** All of Gate 3's
Fischer validations run with `external_flux = 0` and are bit-for-
bit identical to today.

### 2.3 Backend choice is per-Region

A Device may mix tiers: L region on T3, R region on T2, qubit-
readout electrode on T3. The backend choice affects only the
per-Region step; the Junction interface is tier-agnostic because
the flux `G_ext(E, …)` is the common surface.

---

## 3. Layer 2 — Device, Junction, Qubit

### 3.1 Device

```python
@dataclass
class Device:
    regions: dict[str, Region]              # keyed by Region.name
    junctions: list[Junction]
    qubit: Qubit | None = None
```

A Device is just data. The solver is the `solve_device_steady_state`
free function in §3.5. No "Device" class methods for evolution —
keeps the data/behavior split clean.

**Single-region Devices exist.** For Gate 3's existing Fischer
reproductions, a `Device(regions={"main": region}, junctions=[])`
is a trivial wrapper. All existing services
(`solve_steady_state`, `solve_nbar_loop`, `run_time_dependent`)
gain an internal `device = _wrap_single_region(state)` call and
behave identically — backward compatible.

### 3.2 Junction

```python
@dataclass
class Junction:
    """A tunnel coupling between two named Regions.

    At each solver step, ``evaluate`` is called with the current
    state of the two regions (and the qubit, if any), and returns:
      * E-resolved fluxes into each Region (negative = extraction),
      * Qubit transition rate matrix (if ``qubit_coupling``).

    The M25 gap-asymmetric JJ with photon drive is **one specific
    implementation** of the ``evaluate`` protocol — the framework
    carries no M25-specific assumptions.
    """
    name: str
    region_a: str                           # name in Device.regions
    region_b: str
    matrix_elements: JunctionMatrixElements
    photon_drive: PhotonDrive | None = None # pair-breaking / photon-assisted
    qubit_coupling: JunctionQubitCoupling | None = None

    def evaluate(
        self,
        region_a_state: RegionState,
        region_b_state: RegionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionFluxes:
        """Compute fluxes + qubit rates from current region/qubit state.

        Returns
        -------
        JunctionFluxes
            * ``flux_a(E, r)``, ``flux_b(E, r)`` — injections into
              regions A and B (Hz · density-units, matching the
              region's f-equation).
            * ``qubit_rates(i, j)`` — rate matrix if qubit_coupling
              is set, otherwise None.
        """
        ...
```

**Concrete Junction implementations ship as subclasses or free-
function registrations**. M25GapAsymmetricJJ is the first one:
it knows about the three-sub-band kinematic split (R</R>),
holds the Γ̃^α_{ij} integrand from Stage A (SI Note III), wires
in photon-assisted tunneling from Stage A's Note V module, and
returns E-resolved fluxes instead of the integrated-over-E Γ̃×x
rates. The symmetric-gap (z=1) single-qubit transmon NIS junction
is another; `Junction` subclass `NISNormalInsulatorSuperconductor`
yet another.

The point: **the framework owns the `Junction` protocol; specific
physics owns specific `Junction` implementations**. No more "M25
hand-coded into the service".

### 3.3 Qubit

```python
@dataclass
class Qubit:
    """Optional coupled two-level-system (or more) driven by junction tunneling.

    Transmon convention: ``levels = 2`` for logical {|0>, |1>},
    ``levels = 3`` for qutrit including |2>. Parity is tracked as
    a separate ``parity_state`` axis when ``track_parity = True``.
    """
    n_levels: int = 2
    E_J_kelvin: float = 0.0                 # transmon Josephson energy
    E_C_kelvin: float = 0.0                 # transmon charging energy
    omega_kelvin: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
                                            # level energies
    track_parity: bool = True
    state: QubitState | None = None         # populated at solver init

@dataclass
class QubitState:
    p: np.ndarray                           # shape (n_levels,) or (n_levels, 2) if parity
    t: float = 0.0                          # lab-frame time if evolving

@dataclass
class JunctionQubitCoupling:
    """How a Junction drives the Qubit.

    The tunneling-rate evaluator returns a per-(i,j) rate matrix;
    this struct holds the matrix elements (s_ii', c_ii' in the M25
    convention) used to weight ``sin(φ̂/2)`` vs ``cos(φ̂/2)``
    contributions. Parity-selection rules live here: each tunneling
    event flips parity, so every channel advances the parity axis.
    """
    sin_matrix_elements: np.ndarray         # shape (n_levels, n_levels) — s²_{ii'}
    cos_matrix_elements: np.ndarray         # shape (n_levels, n_levels) — c²_{ii'}
    photon_conserving: bool = False         # for ee (parity-preserving) channels
```

### 3.4 Matrix elements: where M25 Stage A lives

The M25 Stage A module `qpsim.services.rate_equation_coefficients`
currently evaluates:

* `_s_10_squared`, `_c_ii_squared` — transmon matrix elements
  (SI Eq. S25–S28).
* `_gamma_L_10`, `_gamma_Rgt_10`, `_gamma_Rlt_10`, `_gamma_L_01`,
  `_gamma_Rgt_01`, `_gamma_L_ii`, `_gamma_Rgt_ii` — the 12
  Γ̃^α_{ij} tunneling-rate integrands (SI Eqs. S30–S36).
* `_tau_R_inverse`, `_tau_E_inverse` — intraband relaxation (SI S50).
* `_branching_fraction` — ξ (SI S37).
* `_S_ph_total`, `_S_ph_Rgt`, `_S_ph_Rlt` — photon spectral density
  (Note V, Eqs. S57–S59).

Most of these stay **exactly as-is**. Their new home is as private
helpers of a specific `Junction` subclass:

```python
class M25GapAsymmetricJJ(Junction):
    """Gap-asymmetric JJ coupled to a transmon qubit, with pair-
    breaking photon drive. Implements the M25 physics exactly.
    """
    def evaluate(
        self,
        region_a_state: RegionState,        # L electrode
        region_b_state: RegionState,        # R electrode
        qubit_state: QubitState,
    ) -> JunctionFluxes:
        # 1. Evaluate Γ̃^α_{ij} from the region-local f_α(E) —
        #    for now, assume Fermi-Dirac ansatz → use Stage A
        #    closed-form evaluators that take (E_J, E_C, gaps, T,
        #    μ_α) and return Hz rates per qubit state. A future
        #    generalization drops the Fermi-Dirac ansatz and
        #    integrates directly over f_α(E).
        # 2. Weight by qubit state p_i.
        # 3. Assemble region fluxes:
        #    flux_L(E) = Σ_{i,j} p_i × Γ̃^L_{ij} × (f-shape function)
        #    (signs: outflow from L when the tunneling lands the QP in R)
        # 4. Return JunctionFluxes with L/R fluxes and qubit rate matrix.
        ...
```

The moment-closure specialization (“Fermi-Dirac on each region”)
is encoded in *this* subclass via the choice to call Stage A's
closed-form evaluators. A different subclass could integrate over
f(E) directly, producing a Junction that works on non-thermal
distributions — that's Layer 2 in its general form, and is the
architectural escape hatch the user asked for.

### 3.5 Top-level solver

```python
def solve_device_steady_state(
    device: Device,
    *,
    residual_tol: float | None = None,
    max_outer_iterations: int = 50,
    ...
) -> DeviceSolution:
    """Fixed-point iteration on (region states, qubit state).

    Each outer iteration:
      1. For each junction, evaluate fluxes + qubit rates given
         current region/qubit states.
      2. Aggregate fluxes per region (sum over junctions connecting
         to that region).
      3. For each region, step the backend to its local steady state
         given the junction fluxes as external sources. ``external_flux``
         is region-local; the per-region Newton is well-conditioned.
      4. If a qubit exists, evolve its master equation using the
         aggregated rate matrix.
      5. Check convergence: max fractional change in region states,
         qubit populations, junction fluxes < tol.

    Returns converged Device state + per-junction flux diagnostics.
    """
    ...
```

The **outer iteration** is a Picard fixed-point on
(region states, qubit state). The **inner** per-region Newton
operates on a single-region kinetic equation with known external
fluxes — well-conditioned, fast, converges at the source-rate
accuracy the Stage A guard already enforces.

---

## 4. Layer 3 — Moment-closure reductions

The existing `solve_rate_equation_steady_state` is **exactly** the
moment-closure reduction of Layer 2 under the assumption that
f_α(E) is Fermi-Dirac per sub-band. Making that explicit is a
documentation win even if we keep the fast algebraic path.

### 4.1 Reduction operator

```python
def fermi_dirac_moments(
    region_state: RegionState,
    sub_bands: list[tuple[float, float]],   # (E_min, E_max) per sub-band
) -> list[tuple[float, float]]:             # (x_α, μ_α) per sub-band
    """Extract (density, chemical potential) moments under Fermi ansatz.

    Integrates region_state.f(E) over each sub-band and fits
    x_α = √(2πT/Δ_α) exp(−(Δ_α − µ_α)/T) for µ_α.
    """
    ...
```

### 4.2 M25 rate equation as derived Layer 3

With the Fermi-Dirac ansatz on each Region, the Junction.evaluate()
call reduces to the Γ̃^α_{ij} × x_α representation and the
kinetic equation becomes algebraic in (p_1, x_α). This is exactly
Eqs. 4–6 of M25 — the current `_rate_equation_residual`. So
Layer 3 can be thought of as:

    Layer 3 = Layer 2 solved on a `Device` with every region
              backend replaced by a moment-closure "dummy backend"
              that stores (x_α, µ_α) instead of f(E).

This view is what makes the current 4-variable system the specific
case it is, not the architectural norm.

### 4.3 When to use which layer

| Use case | Layer |
|---|---|
| Non-thermal f(E), athermal drive, transient relaxation | Layer 2 |
| Paper reproduction of M25 Fig 3/4/5 (steady-state, thermal) | Either; Layer 3 faster if the ansatz holds |
| Multi-region Fischer-Catelani-style MKID networks | Layer 2 |
| Quick engineering estimate of junction current at known T | Layer 3 |
| Single-region Fischer validation (existing Gate 3) | Layer 1 (unchanged) |

The framework offers both. Layer 3 is a *derived*, faster, less
general reduction of Layer 2 — **not the primary object**.

---

## 5. Backward compatibility

* **Gate 3 Fischer validations** (single region, T3, thermal phonon
  bath): unchanged. A Device wrapping one Region is a no-op over
  the existing solver.
* **`qpsim.services.steady_state.solve_steady_state`,
  `nbar_loop.solve_nbar_loop`,
  `transient.run_time_dependent`**: add internal
  `_wrap_single_region_device()` calls. External API stable. All
  360 Gate-3 tests pass.
* **Stage A M25 machinery** (`M25PhysicalParameters`,
  `coefficients_from_physical_parameters`, the `_S_ph_*` and `_gamma_*`
  evaluators): retained as the internals of `M25GapAsymmetricJJ`
  and the Layer-3 reduction path. Zero math re-derivation.
* **`qpsim.services.rate_equation.solve_rate_equation_steady_state`**:
  retained as the Layer-3 reduction entry point. Gains a
  docstring note: "this is the Fermi-Dirac moment closure of
  `solve_device_steady_state`; prefer the Device path unless you
  have a reason to pre-compute the coefficient bundle". Existing
  22 tests (Strategy A limiting cases + roundtrips) still pass.
* **Gate 3 baselines** (`validation/baselines/ph0_*/`) untouched.

No deprecations in Phase 1–4. Phase 5 (M25 Fig 3 reproduction via
Layer 2) produces a **new** baseline distinct from any current
one; the rate-equation path remains as documentation-only
reference.

---

## 6. Open questions (design tradeoffs to lock before Phase 2)

### 6.1 Junction evaluator API: moment closure vs. E-resolved

**The clean answer** is that `Junction.evaluate` receives full
`RegionState` objects (which contain f(E)) and returns E-resolved
fluxes. Concrete implementations can internally choose to
marginalize over E via a moment closure if that's physically
justified.

**But** the Stage A closed-form Γ̃ evaluators assume Fermi-Dirac
and work on (x_α, µ_α) tuples. A pragmatic v1 ships a specific
`M25GapAsymmetricJJ(MomentClosureJunction)` that internally
converts its input RegionStates to (x_α, µ_α) before calling the
Stage A Γ̃ math. The fully-general E-resolved version is a v2
refinement.

**Proposal:** ship both, with clear labels:
* `MomentClosureJunction` — subclass that first reduces each
  region's state to (x_α, µ_α), then calls the physics. Stage A
  machinery is a natural implementation.
* `KineticJunction` — subclass that operates on f(E) directly via
  the E-resolved tunneling current formula. Future refinement;
  not required for M25 Fig 3 reproduction.

### 6.2 How does the solver handle mixed-tier Devices?

Example: region L on T3 (diffusion, angular-averaged), region R
on T2 (kinetic-scalar, with momentum direction). The Junction
fluxes need to be consistent: T3 wants an `(NE, NR)` flux array;
T2 wants `(NE, NR, Nθ, Nφ)`.

**Proposal:** Junctions are tier-aware. Each Junction declares
what tiers it can connect. A future `KineticJunction` handling
T2↔T2 coupling would emit angular-resolved fluxes; the current
`MomentClosureJunction` only emits E-resolved (angular-averaged)
fluxes, so it's T3↔T3 only. M25 Fig 3 is a T3↔T3 problem → fine
for v1.

### 6.3 Qubit time evolution vs. steady state

For steady-state solves, the qubit master equation is another
algebraic equation: solve `ṗ = 0` jointly with the region states.
For transient solves, qubit state evolves via RK or ETD. Same
underlying physics, different solver loops. Both fit the Device
abstraction.

**Proposal:** two entry points in v1 —
`solve_device_steady_state(device)` and
`evolve_device(device, t_span, ...)` — matching the existing
`solve_steady_state` / `run_time_dependent` split.

### 6.4 Where does photon drive live?

Currently `PhotonDrive` and `PhotonState` live in different places
for different drive mechanisms (pair-breaking vs sub-gap vs
photon-assisted tunneling). For Devices, pair-breaking drive is a
per-Region property (absorbed in the film), while photon-assisted
tunneling (M25 Γ_ν) is a per-Junction property.

**Proposal:** `Region` owns its thermal phonon state AND pair-
breaking photon bath. `Junction` owns the photon-assisted-
tunneling drive (Γ_ν × n̄). No field has two homes.

### 6.5 Naming: Region or Electrode?

"Region" is more general; "Electrode" implies a two-electrode
device topology. MKIDs are "films" or "resonators", JJs are
"electrodes". The framework needs to work for all.

**Proposal:** `Region` is the primitive. `Junction.region_a` /
`region_b` for the names. When someone builds a JJ they'll use
names `"L"` / `"R"` by convention; the framework doesn't care.

---

## 7. Implementation plan

### Phase 1 — Design (this doc)
Scope: write this document, review with GPT, fix structural
issues before any code changes.
**Status: in progress.**

### Phase 2 — External flux on T3
Rename/generalize `external_generation_runtime` →
`external_flux`. Add acceptance of arbitrary E-resolved flux array
on `T3DiffusionBackend.step()`. No new physics. Every existing
Gate-3 test passes unchanged because `external_flux = 0` is the
default. Add one new test: stepping a T3 region with a
non-trivial `external_flux` produces the expected shift in the
steady-state f(E).
**Est: 1 session. Ships green.**

### Phase 3 — Region / Junction / Device, no qubit
* Create `qpsim.devices.region.Region` wrapping
  `T3DiffusionState`.
* Create `qpsim.devices.junction.Junction` protocol + one test
  implementation: a **symmetric-gap two-region tunneling coupling**
  (no qubit, no photon drive) that reaches thermal equilibrium
  when L and R are at matched temperature.
* Create `qpsim.devices.device.Device` with
  `solve_device_steady_state(device, ...)` Picard loop.
* Test: at `T_L = T_R = 100 mK`, no drive, the two regions relax
  to identical thermal f(E). Residual in junction flux < physical
  tolerance.
**Est: 1–2 sessions. Ships green.**

### Phase 4 — Qubit + parity + JunctionQubitCoupling
* Create `qpsim.devices.qubit.Qubit` + `QubitState` + master-
  equation evolution (both steady-state algebraic + transient
  ODE paths).
* Extend `Junction` to carry a `qubit_coupling` and return
  `qubit_rates` alongside region fluxes.
* Test: thermal QP bath + matched qubit freq → qubit relaxation
  at the Boltzmann rate matching detailed balance.
**Est: 1 session. Ships green.**

### Phase 5 — M25 Fig 3 via Layer 2
* Implement `M25GapAsymmetricJJ(MomentClosureJunction)`
  internally calling the Stage A Γ̃ / r / τ evaluators.
* Compose `Device(L, R, M25GapAsymmetricJJ, Qubit)` at M25 Fig 3a
  and Fig 3b parameters.
* Sweep T, solve Layer 2, extract densities and chemical potentials,
  pin CSV + PDF baselines under
  `validation/baselines/marchegiani_2025/m25_fig3_device.{csv,pdf}`.
* Regression test.
**Est: 1 session. Ships green.**

### Phase 6 — M25 Fig 4/5 + closure
* Fig 4 (density ratios or transition-rate ratios) and Fig 5
  (parity-switching rate Γ_P vs T) reproductions via same
  Device setup.
* STATUS + memory updates: Gate 8 Strategy B closes.
**Est: 1 session. Ships green.**

**Total:** 5–6 sessions from Phase 2 onward, with a green
checkpoint after each. No big-bang refactor; every phase adds one
layer and keeps the existing behavior intact.

---

## 8. Not in scope for this doc

* Multi-qubit coupling (e.g., two transmons sharing a bus
  resonator). Architecture admits it — just a list of Qubits
  instead of one — but deferred.
* Electromagnetic mode solving (Josephson junction → microwave
  resonator coupling). Left as a later extension.
* T2 and T1 backends. Orthogonal — Device composition doesn't
  require T2/T1, and when they land they plug into the existing
  Region abstraction.
* Rewriting the NFP. This doc is the **concrete device-level
  design**; NFP §2 continues to hold for the within-region
  kinetic equations.

---

## 9. Next action

After review of this doc, start Phase 2: add `external_flux` to
`T3DiffusionBackend`. That unlocks everything downstream and
is the smallest possible first step that touches real code.
