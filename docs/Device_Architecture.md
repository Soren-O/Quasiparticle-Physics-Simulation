---
title: Device Architecture (Region / Junction / Qubit / Device)
description: Implemented T3 steady-state Device core, with a limited M25 adapter, historical design rationale, and current certification limits.
status: core implemented; planned mixed-tier, transient, kinetic-junction, and full M25 Device validation remain open; updated 2026-07-21
---

# Device Architecture

> **Current-status banner (2026-07-21):** The repository contains the T3
> steady-state Region/Junction/Device core, multiple Junction composition, and
> one optional steady-state Qubit. `M25GapAsymmetricJJ` is a limited adapter
> around the existing cached four-variable moment solve; it is not the planned
> generic `MomentClosureJunction` or an E-resolved `KineticJunction`. Mixed-tier
> Devices, multi-qubit Devices, Device time evolution, and the Phase-5/6 full
> Device validation artifacts described below are not implemented. Sections
> 1–7 preserve the April proposal as history; §3.5 and §7's status notes state
> the live contract.

## 0. Summary

The within-region backend hierarchy (T1/T2/T3) describes the QP kinetic
equation in **one** superconducting region. The implemented `qpsim.devices`
layer composes those regions with Junctions and an optional Qubit. The older
hand-coded 4-variable M25 service (`qpsim.services.rate_equation`) remains as
a tested Layer-3 moment reduction, rather than the only device abstraction.

The design is organized into three layers; implementation coverage differs:

* **Layer 1 — Region.** One superconducting region. Owns material,
  geometry, backend, state. T1/T2/T3 are the three possible backends
  (v1 ships T3 only). Gains one new surface: an external flux
  ``G_ext(E, r, t)`` on the RHS of the kinetic equation.
* **Layer 2 — Device.** The live implementation composes one or more T3
  Regions, zero or more Junctions, and zero or one Qubit. The top-level solver
  is steady-state only: evaluate junctions → push fluxes into regions → solve
  each region → solve the optional qubit master equation → repeat. Mixed T1/
  T2/T3 backends, multiple Qubits, and Device transient evolution remain
  design extensions.
* **Layer 3 — Moment-closure reductions.** The standalone M25 four-variable
  service is implemented and tested. The Device-facing M25 adapter reuses and
  caches that isolated closure; a general closure consuming each evolving
  Region state has not landed.

The M25 adapter composes two Regions, one Junction, and a parity-tracking
Qubit in focused tests, while reusing the Stage-A coefficient machinery. The
published Fig. 3/4 validation sweeps remain service-driven; the proposed
`m25_fig3_device.{csv,pdf}` full Device artifact does not exist.

Two honest notes on what this *does* and *does not* automatically
fix:

1. **Architectural separation: implemented within scope.** Multi-region,
   multi-junction T3 steady-state setups and one optional Qubit are first-class
   compositions. That does not imply mixed tiers, multiple Qubits, or Device
   transients.
2. **Stage B numerical pathology: not removed by the M25 adapter.** The
   adapter calls the existing isolated moment solver, so it retains the
   coefficient-to-density conditioning and does not respond to evolving f(E).
   Moving to a true `KineticJunction` that
   operates on E-resolved f(E) on each region would cleanly side-
   step the cancellation (tunneling becomes a boundary
   ``ExternalFlux`` on well-conditioned region-local kinetic
   equations), but that's strictly more physics work than v1
   ships. §6.1 lays out the still-open decision tree.

This document began as the design proposal. Section 7 preserves its phased
implementation plan for provenance; it is not the current task list.

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

### 2.2 The external-flux surface: ExternalFlux contract

The existing T3 kinetic equation is

    ∂_t f + (Δ/E) Δ̇ ∂_E f
      = ∇_r · [D(E, Δ) ∇_r f] + I_coll[f, n_ph](E, r)

and **every collision / drive term is structured as a
``(gain, loss_rate)`` pair** so the positivity-preserving ETD /
Newton machinery works (``df/dt = gain − loss_rate · f``; ``f ≥ 0``
is preserved when both are non-negative). Examples: photon-assisted
pair breaking, sub-gap photon scattering, e-phonon — all return a
``(gain, loss_rate)`` pair from their evaluator, both in units of
``1/ns``.

There is **no pre-existing external-generation path** to rename.
Phase 2 is a real solver-surface change that adds a new input
contract — ``ExternalFlux`` — that threads through
``T3DiffusionBackend``, ``solve_steady_state``, ``newton_solve_f``,
``coupled_newton_solve``, and the transient ETD stepper. No change
to collision physics; just a new additive term consumed on the RHS.

#### 2.2.1 ExternalFlux dataclass

```python
@dataclass
class ExternalFlux:
    """A boundary / junction-injected source and sink on a Region's f-equation.

    Decomposed into a (gain, loss_rate) pair to match the existing
    collision-term contract — signed fluxes are explicitly rejected.
    Extraction ("flow out of this region via a junction") is
    encoded as ``loss_rate * f``, NOT as a negative gain. This
    preserves the positivity and Jacobian structure that T3's
    ETD / Newton solvers rely on.

    Units are ``1/ns`` to match the rest of the T3 stack (not Hz).
    Converters from Stage-A-style Hz rates live in the Junction
    implementation, not in the ExternalFlux contract.

    Shape matches the current T3 state layout: v1 is ``(NE,)``
    because Gate-2 T3 is lumped-0D (single spatial cell). When
    Gate 5 adds spatial T3, both fields broaden to ``(NE, NR)``
    and ``target_cells`` becomes a meaningful mask. v1 ships with
    a shape check that accepts ``(NE,)`` and optionally
    ``(NE, 1)`` (transparent squeeze) so the contract is forward-
    compatible without premature spatial plumbing.
    """
    gain: np.ndarray          # ≥ 0 everywhere. Shape (NE,) in v1.
                              # Additive source rate, units 1/ns.
    loss_rate: np.ndarray     # ≥ 0 everywhere. Shape (NE,) in v1.
                              # Multiplier on f in the damping term,
                              # so the RHS contribution is -loss_rate*f,
                              # units 1/ns.
    target_cells: np.ndarray | None = None
                              # Optional boolean mask (NR,) for spatial
                              # T3; IGNORED in v1 (0D has one cell).
                              # Reserved for Gate 5.
    diagnostics: dict[str, float] = field(default_factory=dict)
                              # Junction name, total injected current,
                              # etc. — passed to the solver logs.
```

#### 2.2.2 How it threads through the solvers

| Surface | Change |
|---|---|
| ``T3DiffusionBackend.step(state, dt, external_flux=...)`` | New kwarg. Adds ``gain`` to the explicit piece and ``loss_rate`` to the damping piece in the ETD2 substep. |
| ``solve_steady_state(..., external_flux=...)`` | New kwarg, propagates to Newton and coupled-Newton paths. |
| ``newton_solve_f(..., external_flux=...)`` | Adds ``+ gain − loss_rate * f`` to the residual and ``-loss_rate`` to the Jacobian diagonal. |
| ``coupled_newton_solve(..., external_flux=...)`` | Same; the ``(f, n_ph)`` coupled Newton sees the extra terms on f only. |
| ``run_time_dependent(..., external_flux=callable)`` | Callable returning ``ExternalFlux`` at each substep midpoint. This time-varying path requires ``stop_tol=None`` because an instantaneous residual cannot certify future convergence of a non-autonomous drive. |

When ``external_flux=None`` (the default), the code path is bit-
for-bit identical to today. All 360+ Gate-3 tests remain green.

#### 2.2.3 Regression tests for Phase 2

The Phase 2 surface is *just* the new RHS term. To test it cleanly
without tangling with the (nonlinear in f, via cross-bin partners)
collision kernels, the contract tests run with
``enable_recombination=False, enable_scattering=False,
enable_photon_scattering=False`` so only the ExternalFlux term
plus the spectral-flow term is live:

* **Zero flux identity**: ``ExternalFlux=None`` ⇒ bit-for-bit match
  with today's Gate-3 Fischer validations across the full T3 test
  suite.
* **Linear ODE closed form**: collision kernels disabled,
  ``ExternalFlux(gain=g(E), loss_rate=r(E))`` with constant ``g, r``.
  Steady state is ``f(E) = g(E) / r(E)`` by direct construction of
  the ODE ``df/dt = g − r f``. Compare to float64 precision.
* **Detailed-balance ansatz**: with kernels disabled, set
  ``gain, loss_rate`` such that ``gain/loss_rate = f_FD(E, T_bath)``
  pointwise in E. Steady state is Fermi-Dirac by construction.
  This is the same linear-ODE test as above — the point is that
  the contract *supports* a detailed-balance setup.
* **Conservation under injection**: with kernels disabled,
  ``∂_t n_qp`` from ExternalFlux equals
  ``4 ρ_F ∫ ρ(E) (gain − loss_rate · f) dE`` — a linear identity
  checked directly; confirms that the observable-level
  conservation law is wired consistently with the Fischer-
  convention normalization.

Nonlinear behavior with collisions on is exercised in Phase 3
(two-region device reaching thermal equilibrium), where the
invariant is architecturally richer.

### 2.3 Backend choice is per-Region

A Device may mix tiers: L region on T3, R region on T2, qubit-
readout electrode on T3. The backend choice affects only the
per-Region step; the Junction interface is tier-agnostic because
the ``ExternalFlux`` contract is the common surface.

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

**Single-region Devices exist.** A caller can explicitly construct
`Device(regions={"main": region}, junctions=[])`. Existing Fischer services
continue to call their original single-region solver paths directly; there is
no `_wrap_single_region_device()` compatibility layer.

### 3.2 Junction

```python
@dataclass
class Junction:
    """A tunnel coupling between two named Regions.

    At each solver step, ``evaluate`` is called with the current
    state of the two regions (and the qubit, if any), and returns:
      * ``ExternalFlux`` contributions for each Region (see §2.2),
      * Channel-tagged qubit transition rates (if ``qubit_coupling``).

    The M25 gap-asymmetric JJ with photon drive is **one specific
    implementation** of the ``evaluate`` protocol — the framework
    carries no M25-specific assumptions.
    """
    name: str
    region_a: str                           # name in Device.regions
    region_b: str
    matrix_elements: JunctionMatrixElements
    photon_drive: PhotonDrive | None = None # photon-assisted tunneling
    qubit_coupling: JunctionQubitCoupling | None = None

    def evaluate(
        self,
        region_a_state: RegionState,
        region_b_state: RegionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        """Compute per-region flux contracts + qubit rates.

        Returns
        -------
        JunctionResult
            * ``external_flux_a``, ``external_flux_b``
              — ``ExternalFlux`` (gain, loss_rate) contributions for
              each region. The solver sums contributions from all
              junctions that touch a region.
            * ``qubit_channels`` — list of ``QubitTransitionChannel``
              records (see §3.3.2) if qubit_coupling is set, each
              carrying a rate and a parity-flip flag. Otherwise empty.
        """
        ...
```

#### 3.2.1 Boundary-current normalization (critical for conservation)

A junction transports QPs between two regions. Converting a
**per-energy-bin tunneling rate** ``I_J(E)`` (events/(time · E-bin))
into an ``ExternalFlux.gain(E)`` (units ``1/ns``, matching the
per-E-bin collision-term convention) requires the E-resolved DOS
normalization, **not** the integrated moment one. The framework
uses the Fischer convention from ``qpsim.observables.density``:

  ``n_qp = 4 ρ_F ∫ ρ(E) f(E) dE``

where ``ρ_F`` is the single-spin normal-state DOS at the Fermi
level in eV⁻¹ m⁻³, the integration measure in this formula is in eV,
and ``ρ(E)`` is the BCS (or Dynes-broadened) spectral enhancement.
Code operating on qpsim's µeV grids must divide the energy measure by
``1e6``. Write the continuous junction spectrum as ``j_J(E)`` in
events/(ns·eV), so a qpsim cell receives
``I_{J,i} = j_J(E_i) dE_i[eV]`` events/ns. Matching
``I_{J,i}/V_region`` to
``4 ρ_F ρ(E_i) ∂_t f_i dE_i[eV]`` gives the equivalent forms:

* **0D regions (v1):**
  ``gain_i = j_J^{in}(E_i) / (4 ρ_F ρ_i V_region)`` or
  ``I_{J,i}^{in} / (4 ρ_F ρ_i V_region dE_i[eV])``;
  ``loss_rate_i = j_J^{out}(E_i) / (4 ρ_F ρ_i V_region f_i)`` or
  ``I_{J,i}^{out} / (4 ρ_F ρ_i V_region f_i dE_i[eV])``.
  The in/out spectra are split by sign; the loss-rate form recovers
  positivity-preserving solver structure — see §2.2.
* **Spatial regions (Gate 5+):** same formula per spatial cell with
  ``V_cell`` and with ``target_cells`` masking the junction
  interface. Not implemented in v1.

**Relationship to Stage A's moment normalization.** Stage A's
``g^{ph}_R = Γ^{ph} / (2 ν_0 Δ_R V)`` is the E-*integrated* rate
per Cooper pair, the appropriate normalization for the M25
moment-level ``dx_α/dt`` equations. It relates to the E-resolved
gain via

  ``g_moment = (2 ρ_F/(ν_0 Δ)) ∫ gain(E) ρ(E) dE``

Every quantity in that expression must use one energy unit. Stage A stores
``ν_0`` in J⁻¹m⁻³, so convert it to eV⁻¹m⁻³ via
``ν_0[eV⁻¹m⁻³] = ν_0[J⁻¹m⁻³] × 1.602176634e-19 J/eV`` before comparing it
with ``ρ_F``; likewise use ``Δ[eV]`` and ``dE[eV]``. When the converted
``ρ_F = ν_0`` (same DOS convention on both sides), this
simplifies to ``g_moment = (2/Δ) × ∫ gain(E) ρ(E) dE``. The
``M25GapAsymmetricJJ`` implementation is responsible for this
mapping: it evaluates Stage A's moment-rate ``Γ̃^α`` per qubit
channel, then distributes that rate across the E-grid in a way
that preserves the moment sum (e.g., a δ-like distribution at
the kinematically-selected partner energies). This distribution
choice is an *approximation* introduced by the MomentClosureJunction
wrapper — a KineticJunction would compute ``gain(E)`` directly from
f_L(E), f_R(E') at paired energies.

**Conservation invariants (pinned by Phase 3 tests).**

1. **Per-region total injection matches the junction current**:
   ``4 ρ_F V_region Σ_i ρ_i gain_i dE_i[eV] = I_J^{total}`` — the
   device-level junction diagnostic — to float64.
2. **Cross-region balance at detailed balance**: summing over
   every region of ``∂_t N_qp = 4 ρ_F V_region Σ_i ρ_i (gain_i −
   loss_rate_i f_i) dE_i[eV]`` equals zero when the Device is at thermal
   equilibrium with matched temperature and no drive.
3. **Steady-state matched-T limit**: two-region Device at
   ``T_L = T_R``, no drive: both regions land on ``f = f_FD(T)``,
   junction flux ``I_J → 0`` as convergence proceeds.

For the shipped `SymmetricGapTunnelingJunction`, the scalar
`capacity_ratio_a_to_b` represents only a global material/volume factor. It is
valid only when both regions share the same per-bin finite-volume spectral
measure. Runtime evaluation therefore requires each state's public gap to
match its spectral gap and requires identical energy centers, cell widths,
spectral gaps, Dynes broadening, and `cell_weights` across the junction. A
different energy-dependent capacity vector needs an energy-resolved asymmetric
junction model; it is not approximated by the scalar ratio.

#### 3.2.2 Unit convention summary

| Quantity | Unit | Why |
|---|---|---|
| ``ExternalFlux.gain``, ``loss_rate`` | 1/ns | Matches T3 stack |
| Junction spectrum ``j_J(E)`` | events/(ns·eV) | Continuous normalization |
| Per-bin current ``I_{J,i}`` | events/ns | Includes ``dE_i[eV]`` |
| Stage A ``Γ̃^α_{ij}`` | Hz | Paper convention; converted at the Junction boundary |
| Gaps, energies | μeV or K (per backend) | Existing T3 convention |
| ``tau_0``, ``tau_0_phonon`` | ns | Material YAML |

``M25GapAsymmetricJJ`` is responsible for converting from Stage A's
Hz rates to the Junction's internal ns units and for the
``N_CP = 2 ν_0 Δ V`` normalization. A dedicated conversion helper
lives on the Junction base class.

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
    """Optional coupled TLS driven by junction tunneling.

    The qubit state space is ``(level, parity)``: e.g. a two-level
    transmon with parity tracking has four discrete states
    |0,e>, |0,o>, |1,e>, |1,o>. Channels driving transitions between
    these are tagged by WHICH axis they advance (see
    ``QubitTransitionChannel`` below).
    """
    n_levels: int = 2
    track_parity: bool = True
    E_J_kelvin: float = 0.0                 # transmon Josephson energy
    E_C_kelvin: float = 0.0                 # transmon charging energy
    omega_kelvin: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
                                            # level energies
    state: QubitState | None = None         # populated at solver init


@dataclass
class QubitState:
    """Probabilities over the (level, parity) state space.

    Shape is ``(n_levels, 2)`` when ``track_parity`` is set, else
    ``(n_levels,)``. The parity axis has two entries: [0] = even,
    [1] = odd. ``np.sum(p) = 1`` always.
    """
    p: np.ndarray
    t_ns: float = 0.0                       # lab-frame time if evolving
```

#### 3.3.1 QubitTransitionChannel — parity-resolved rates

A single ``qubit_rates[i, j]`` matrix can't distinguish a
parity-preserving (ee) transition |i, e> → |j, e> from a
parity-changing (eo) transition |i, e> → |j, o>, yet both appear
in the M25 master equation (Γ̃^{ee} vs Γ̃^{eo}). The Junction
returns a list of channel records:

```python
@dataclass
class QubitTransitionChannel:
    """One addressable qubit transition produced by a Junction.

    A tunneling event that flips parity (the default for QP
    tunneling) has ``flips_parity = True``. A photon-mediated
    parity-preserving process (ee in M25) has
    ``flips_parity = False``.
    """
    level_from: int
    level_to: int
    rate_per_ns: float                      # transition rate
    flips_parity: bool                      # True for eo channels
    label: str = ""                         # "ph_00", "ee_10", etc.
                                            # — diagnostic only
```

The qubit-master-equation evolver consumes a list of these channels
and assembles a transition matrix on the full ``(level, parity)``
state space. For the M25 setup with 2 levels × 2 parities, this
produces a 4×4 rate matrix, which the solver evolves either to
steady state (algebraic) or in time (ODE).

#### 3.3.2 JunctionQubitCoupling

```python
@dataclass
class JunctionQubitCoupling:
    """How a Junction drives the Qubit.

    Holds the transmon matrix elements (M25 SI Eqs. S25–S28) used
    to split the tunneling rate between sin(φ̂/2) (parity-flipping,
    logical-changing) and cos(φ̂/2) (parity-flipping, logical-
    conserving) channels. Parity-preserving (ee) channels — if any —
    are declared separately via ``parity_preserving_rates``.
    """
    sin_matrix_elements: np.ndarray         # shape (n_levels, n_levels) — s²_{ii'}
    cos_matrix_elements: np.ndarray         # shape (n_levels, n_levels) — c²_{ii'}
    # Optional parity-preserving channel rates (ee in M25). Zero by
    # default because the default tunneling event flips parity; the
    # M25 setup includes these as independent drivers.
    parity_preserving_rates: np.ndarray | None = None
                                            # shape (n_levels, n_levels) — 1/ns
```

The Junction.evaluate() implementation decides which channels to
emit — for an M25GapAsymmetricJJ, each Γ̃^α_{ij} tunneling rate
produces a ``QubitTransitionChannel(level_from=i, level_to=j,
rate_per_ns=..., flips_parity=True)``; the Γ̃^{ee} terms produce
``flips_parity=False`` channels. The qubit evolver sees both
kinds and handles them correctly on the 4-state grid.

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
class M25GapAsymmetricJJ(MomentClosureJunction):
    """Gap-asymmetric JJ coupled to a transmon qubit, with pair-
    breaking photon drive. Implements the M25 physics exactly.

    ``MomentClosureJunction`` is the abstract base that handles the
    RegionState-to-moments reduction; this subclass supplies the
    M25-specific physics in the reduced coordinates.
    """
    def evaluate(
        self,
        region_a_state: RegionState,        # L electrode
        region_b_state: RegionState,        # R electrode
        qubit_state: QubitState,            # shape (n_levels, 2) parity-resolved
    ) -> JunctionResult:
        # 1. Reduce each RegionState to (x_α, µ_α) under the
        #    Fermi-Dirac ansatz — the Layer-3 moment closure
        #    implemented on MomentClosureJunction.
        # 2. Call Stage A closed-form evaluators at (E_J, E_C,
        #    gaps, T, µ_α) → per-channel moment-level rates in Hz:
        #       * Γ̃^α_{ij} × x_α (tunneling, one rate per α, i, j)
        #       * r^α × x_α² (recombination), g^{pn}_α (thermal gen),
        #         τ_R⁻¹/τ_E⁻¹ (intraband), ξ (branching),
        #         g^{ph}_α/Γ̃^{ph}_{ij} (Note V photon-assisted)
        # 3. For each (α, i, j) channel, convert the Hz rate to a
        #    per-E-bin gain on that region via §3.2.1 normalization,
        #    concentrating the injection at the kinematically-
        #    selected partner energy (e.g. E_partner = E + ω_LR for
        #    L→R> tunneling). Collect per-region ExternalFlux.
        # 4. Emit one QubitTransitionChannel per (i → j) transition:
        #       * eo channels (Γ̃^α, Γ̃^{ph}): flips_parity=True
        #       * ee channels (Γ̃^{ee}): flips_parity=False
        #    with rate_per_ns converted from the Stage-A Hz output.
        return JunctionResult(
            external_flux_a=flux_L,
            external_flux_b=flux_R,
            qubit_channels=channels,
        )
```

The moment-closure specialization (“Fermi-Dirac on each region”)
is encoded in *this* subclass via the choice to call Stage A's
closed-form evaluators. A different subclass could integrate over
f(E) directly, producing a Junction that works on non-thermal
distributions — that's Layer 2 in its general form, and is the
architectural escape hatch the user asked for.

### 3.5 Top-level solver

The following block is the original design sketch. The current callable uses
the controls shown here (names abbreviated only by the ellipsis):

```python
def solve_device_steady_state(
    device: Device,
    *,
    backend: T3DiffusionBackend | None = None,
    use_thermal_phonons: bool = True,
    inner_anderson_depth: int = 3,
    outer_tol: float = 1e-6,
    outer_max_iter: int = 100,
    outer_damping: float = 0.5,
    inner_newton_tol: float = 1e-12,
    inner_newton_max_iter: int = 200,
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
      5. Check scale-aware fixed-point defects in region occupations and
         qubit populations; certify supported conservative components with a
         capacity-weighted QP-number backward error.

    Returns the certified region/qubit snapshot and convergence diagnostics.
    """
    ...
```

The **outer iteration** is a Picard fixed-point on (region states, qubit
state). The **inner** per-region Newton operates on a single-region kinetic
equation with frozen external flux. It is usually much better scaled than the
four-variable closure, but that is not an unconditional convergence promise:
cold number modes can be unrepresentable in float64 and fail loudly.

**Current certification contract (2026-07-21).** The implemented solver
certifies the fixed-point defect and, for every supported conservative
component, a capacity-weighted QP-number backward error against the same
public `outer_tol`. An active conservative Junction declares its scalar
`C_a/C_b` ratio through the public safety contract, and the solver checks the
evaluated weighted transfer. `SymmetricGapTunnelingJunction` additionally
requires matched finite-volume spectral measures; a zero-rate instance is a
true inert edge and does not join components. Unknown state-dependent flux
must declare a prescribed-source or conservative-capacity contract;
otherwise the solve refuses. Exclusive dissipation-owning M25 closures are
certified locally by Newton. `use_thermal_phonons=False` is allowed for
independent/disabled-edge Devices, but refuses when an active conservative
cross-region component would require the still-missing nonequilibrium-
phonon component certificate. Exact absorbing vacuum states are accepted;
unresolved finite-temperature number modes still fail loudly. Prescribed
sources retain their own turnover in the Newton normalizer; only explicitly
identified conservative exchange is excluded from that scale. A returned
state is the same snapshot at which its defect was measured. An injected
backend must act as a pure map (in-place mutation refuses), and its complete
returned T3 state -- occupation, positive matching gap/spectral context,
fixed finite grid, bath temperature, and PH0 phonons -- is validated before
convergence arithmetic. Conservative-transfer verification scales raw rates
and finite-volume weights separately before multiplication, so large finite
inputs cannot overflow into a NaN that bypasses the contract. Invalid qubit
outputs are rejected at the same boundary.

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
  `nbar_loop.solve_nbar_loop`, and `transient.run_time_dependent`** remain
  direct single-region APIs. They do not internally construct a Device.
* **Stage A M25 machinery** (`M25PhysicalParameters`,
  `coefficients_from_physical_parameters`, the `_S_ph_*` and `_gamma_*`
  evaluators): retained as the internals of `M25GapAsymmetricJJ`
  and the Layer-3 reduction path. Zero math re-derivation.
* **`qpsim.services.rate_equation.solve_rate_equation_steady_state`** remains
  the primary implemented M25 moment-reduction entry point. The repository
  does not currently add the proposed Device-preference docstring or wrapper.
* **Gate 3 baselines** (`validation/baselines/ph0_*/`) untouched.

No deprecations in Phase 1–4. Phase 5 (M25 Fig 3 reproduction via
Layer 2) produces a **new** baseline distinct from any current
one; the rate-equation path remains as documentation-only
reference.

---

## 6. Open questions (design tradeoffs to lock before Phase 2)

### 6.1 Junction evaluator API: moment closure vs. E-resolved — and what this means for Stage B

**The clean answer** is that `Junction.evaluate` receives full
`RegionState` objects (which contain f(E)) and returns an
``ExternalFlux`` (gain, loss_rate) pair per region. Concrete
implementations can internally choose to marginalize over E via a
moment closure if that's physically justified.

**But** the Stage A closed-form Γ̃ evaluators assume Fermi-Dirac
and work on (x_α, µ_α) tuples — they **don't** operate on f(E)
directly. A pragmatic v1 ships a specific
`M25GapAsymmetricJJ(MomentClosureJunction)` that internally reduces
its input RegionStates to (x_α, µ_α) before calling the Stage A
Γ̃ math, then spreads the resulting integrated junction current
back across the E-grid into an ``ExternalFlux`` with the right
total current.

**Proposal:** ship both classes, with clear labels:
* `MomentClosureJunction` — base class that first reduces each
  region's state to moments (x_α, µ_α) and calls physics on those.
  `M25GapAsymmetricJJ` subclasses this. Cheap, exact for
  thermalized regions, re-uses Stage A as-is.
* `KineticJunction` — base class that operates on f(E) directly
  via the E-resolved tunneling current formula ``I_J(E) = |t|²
  N_L(E) N_R(E ∓ ω) [f_L(E) − f_R(E ∓ ω)] × (coherence)``. True
  Layer 2. Not required for M25 Fig 3 reproduction, but is the
  architectural escape hatch for athermal distributions.

**What this means for the Stage B conditioning claim.** The
original rationale for the layered rewrite was partly that Stage B
numerical pathology (19-order coefficient-to-density ratio) would
resolve automatically by moving to Layer 2. That claim was
**optimistic** for the `MomentClosureJunction` path: it still
reduces to the same algebraic structure inside the Junction, and
can reintroduce the ill-conditioning if the outer Picard loop and
inner per-region Newton don't decouple the scales cleanly.

The clean resolution of Stage B requires one of:

1. **KineticJunction + full f(E) on each region.** The tunneling
   current becomes a boundary ``ExternalFlux`` with gain/loss
   magnitudes set by *local f(E)* — naturally at the 10⁻⁸ scale,
   not at the 10¹¹ Γ̃-vs-Γ̃ cancellation scale. Per-region Newton
   on a kinetic equation is well-conditioned. This is the
   *architecturally true* fix.
2. **MomentClosureJunction + variable-rescaling inside its
   internal moment solve.** Acceptable shortcut that reuses Stage
   A wholesale but does the rescaling that I failed to make work
   in the standalone rate-equation solver. Whether this actually
   converges at float64 precision at the Fig 3 parameter set is
   still an open numerical question.

**Decision for Phase 5:** M25 Fig 3 reproduction uses
`MomentClosureJunction` with variable rescaling done properly on
the internal moment residual. If that still doesn't give
physical-accuracy convergence, the `KineticJunction` implementation
is the required next step (effectively Phase 5b). Either way,
Phase 5 is **explicit** about which class it uses and which
numerical issue it depends on resolving.

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

## 7. Historical implementation plan (partially implemented)

The scopes below are the original proposal, not a completion ledger. Current
status: Phases 1–2 are implemented; Phase 3's T3 steady-state core is
implemented; Phase 4 has steady-state qubit support but not Device/qubit
transients; Phase 5 has a limited cached-closure M25 adapter but neither the
generic `MomentClosureJunction`, an E-resolved `KineticJunction`, nor the named
full Device sweep artifact; Phase 6's validation “via the same Device setup”
is not implemented (separate service-driven M25 figure validations do exist).

### Phase 1 — Design (this doc)
Scope: write this document, review with GPT, fix structural
issues before any code changes.
**Status: completed.**

### Phase 2 — ExternalFlux contract through T3 solver stack
**Status: implemented.** Historical target follows.
Introduce the new ``ExternalFlux(gain, loss_rate, target_cells,
diagnostics)`` dataclass per §2.2.1. Thread it through:
* ``T3DiffusionBackend.step`` (adds gain to the explicit piece,
  loss_rate to the damping piece in the ETD2 substep);
* ``solve_steady_state`` (kwarg, forwards down);
* ``newton_solve_f`` (residual + Jacobian-diagonal update);
* ``coupled_newton_solve`` (same, f-side only);
* ``run_time_dependent`` (takes a callable returning
  ``ExternalFlux`` at each step).

All with positivity-preserving (gain, loss_rate) semantics — no
signed fluxes. Default ``external_flux=None`` is bit-for-bit
identical to today; every existing Gate-3 Fischer test remains
green. New tests per §2.2.3 pin the contract: zero-flux identity,
constant-gain recombination-only steady state, detailed-balance
thermal source, and conservation under non-zero flux.

**Not in scope for Phase 2**: multi-region coupling, Junctions.
The Phase 2 scope is the surface change only, plus its tests.
**Est: 1–2 sessions. Ships green.**

### Phase 3 — Region / Junction / Device, no qubit
**Status: implemented for T3 steady-state composition.** Historical target
follows.
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
**Status: partially implemented (steady-state master equation only; no Device
transient evolution).** Historical target follows.
* Create `qpsim.devices.qubit.Qubit` + `QubitState` + master-
  equation evolution (both steady-state algebraic + transient
  ODE paths).
* Extend `Junction` to carry a `qubit_coupling` and return
  `qubit_rates` alongside region fluxes.
* Test: thermal QP bath + matched qubit freq → qubit relaxation
  at the Boltzmann rate matching detailed balance.
**Est: 1 session. Ships green.**

### Phase 5 — M25 Fig 3 via MomentClosureJunction (Layer-3-in-Layer-2-harness)
**Status: partially implemented by `M25GapAsymmetricJJ`, which adapts a cached
isolated moment solve. The generic/kinetic Junctions and named full Device
baseline below remain open.** Historical target follows.
* Implement `M25GapAsymmetricJJ(MomentClosureJunction)` internally
  calling the Stage A Γ̃ / r / τ evaluators, returning per-region
  ``ExternalFlux`` with the boundary-current normalization of
  §3.2.1.
* Qubit is the 2-level × 2-parity setup from Phase 4 with the
  M25 matrix elements.
* Compose `Device(L, R, M25GapAsymmetricJJ, Qubit)` at M25 Fig 3a
  and Fig 3b parameter sets.
* Sweep T, solve Layer 2 (Picard outer + per-region Newton inner),
  extract densities and chemical potentials, pin CSV + PDF
  baselines under
  `validation/baselines/marchegiani_2025/m25_fig3_device.{csv,pdf}`.
* Regression test with GPT's recommended absolute-residual
  assertion (each sweep point converges to within physical tol,
  not just auto-tol).

**Caveat from §6.1:** this path still uses the Stage A moment
closure inside the Junction. If the per-region Picard + inner
Newton does NOT decouple the 10¹¹-vs-10⁻⁸ scale pathology (an
open numerical question I flagged in §6.1), Phase 5 needs a
follow-up Phase 5b that implements a `KineticJunction` doing
E-resolved tunneling. Phase 5 ships with an **explicit
convergence-behavior report** in its commit message so the
decision tree is transparent.
**Est: 1–2 sessions depending on whether 5b is needed. Ships green.**

### Phase 6 — M25 Fig 4/5 + closure
**Status: not implemented through the Device setup described here.** Separate
service-based Fig. 3/4 validations must not be counted as this deliverable.
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

## 9. Current next actions

Remaining work includes both numerical qualification and planned architecture:
extend the conservative component certificate to stored non-equilibrium
phonons, study tighter coupled solvers for slow mismatched-temperature modes,
implement a Junction that consumes evolving f(E), and decide whether to build
the proposed full M25 Device validation artifacts. New Junction families must
remain inside the explicit capacity/source safety contracts in §3.5.
