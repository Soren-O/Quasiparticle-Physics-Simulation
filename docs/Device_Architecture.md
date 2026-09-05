---
title: Device Architecture (Region / Junction / Qubit / Device)
description: The Region/Junction/Qubit/Device composition layer — the ExternalFlux contract, the Junction protocol and its boundary-current normalization, the qubit master equation, and the Device solver's certification contract.
---

# Device Architecture

## 0. Summary

The diffusion backend describes the QP kinetic equation in **one** superconducting
region. The `qpsim.devices` layer composes those regions with Junctions and an
optional Qubit.

The composition is organized into three layers:

* **Layer 1 — Region.** One superconducting region: material, energy grid,
  spectral context, phonon state, and the current `f(E)`. It carries one
  surface beyond the single-region kinetic equation — an external flux
  ``G_ext(E, t)`` on the RHS.
* **Layer 2 — Device.** One or more Regions, zero or more Junctions, and zero
  or one Qubit. The top-level solver is steady-state: evaluate junctions →
  push fluxes into regions → solve each region → solve the optional qubit
  master equation → repeat.
* **Layer 3 — Moment-closure reductions.** `qpsim.services.rate_equation`
  solves the M25 four-variable algebraic system directly.
  `M25GapAsymmetricJJ` carries that closure inside the Junction protocol, so
  the same physics composes as a Device.

The M25 adapter composes two Regions, one Junction, and a parity-tracking
Qubit while reusing the Stage-A coefficient machinery. The published Fig. 3/4
validation sweeps are service-driven.

Two notes on what the composition layer does and does not do:

1. **Architectural separation.** Multi-region, multi-junction steady-state
   setups with one optional Qubit are first-class compositions.
2. **Stage B numerical conditioning.** The M25 adapter calls the isolated
   moment solver, so it carries the coefficient-to-density conditioning of
   that solve (§1) and does not respond to evolving f(E). A Junction that
   operates on E-resolved f(E) side-steps that cancellation entirely, because
   tunneling becomes a boundary ``ExternalFlux`` on well-conditioned
   region-local kinetic equations.

---

## 1. Why the composition layer is separate from the moment closure

The service at `qpsim.services.rate_equation.solve_rate_equation_steady_state`
solves a 4-variable algebraic system `(p_1, x_L, x_{R>}, x_{R<})`
for the M25 boxed equations (main-text Eqs. 3-6). The inputs are
a fully-packed `M25Coefficients` bundle: 12 tunneling rates
Γ̃^α_{ij}, recombination r^α/r^{<>}, thermal-phonon and photon-
assisted generation, intraband τ_R/τ_E, branching ξ. The
Stage A+B machinery builds that bundle from primitive physical
parameters via the SI Note III/IV/V coefficient integrals.

That system is mathematically correct and well tested, and it is specific in
four ways:

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
   coefficient evaluators. A transmon qutrit or multi-qubit coupling
   would need a different service.
4. **Numerical conditioning.** The 4-variable residual has tunneling
   terms ~10¹¹ Hz cancelling to source rates ~10⁻⁸ Hz. A
   19-order-of-magnitude cancellation that Newton + FD-Jacobian
   can't resolve cleanly at float64 (Stage B investigation,
   2026-04-24). The kinetic-equation image of the same system has
   individual term magnitudes ~10³ Hz — well-conditioned.

These are all consequences of writing the moment closure **as the
primary object**. The kinetic equation is the primary object; the
moment closure is a derived convenience.

---

## 2. Layer 1 — Region

A `Region` is one superconducting region with one QP distribution
and one gap.

### 2.1 Region

```python
@dataclass
class Region:
    """One named superconducting region with its own backend state."""
    name: str                # unique within the Device
    state: DiffusionState    # material, energy grid, phonon state, f(E)
```

`DiffusionState` carries `f` on the energy grid (shape ``(NE,)``), the scalar
gap `Δ` in μeV, the `SpectralContext` (DOS, K±, `D(E)`) that must match that
gap, the `PhononState` (`n_ph`, `τ_l`, branches), the `Material`, and the
substrate bath temperature `T_bath` in K.

The mapping key in `Device.regions` is the only identity the composition layer
resolves by: region state, junction endpoints, capacity weights and the
conserved-number certificate are all keyed. A key that contradicts its own
`Region.name` is a caller-side transposition, and the `Device` constructor
rejects it rather than bind silently to the key.

### 2.2 The external-flux surface: ExternalFlux contract

The kinetic equation on a Region is

    ∂_t f + (Δ/E) Δ̇ ∂_E f
      = I_coll[f, n_ph](E) + gain(E) − loss_rate(E) · f(E)

and **every collision / drive term is structured as a
``(gain, loss_rate)`` pair** so the positivity-preserving ETD /
Newton machinery works (``df/dt = gain − loss_rate · f``; ``f ≥ 0``
is preserved when both are non-negative). Examples: photon-assisted
pair breaking, sub-gap photon scattering, e-phonon — all return a
``(gain, loss_rate)`` pair from their evaluator, both in units of
``1/ns``.

``ExternalFlux`` is the input contract for a boundary- or junction-injected
source and sink. It threads through ``DiffusionBackend``,
``solve_steady_state``, ``newton_solve_f``, ``coupled_newton_solve``, and the
transient ETD stepper. It changes no collision physics; it is an additive term
consumed on the RHS.

#### 2.2.1 ExternalFlux dataclass

```python
@dataclass(frozen=True)
class ExternalFlux:
    """A boundary / junction-injected source and sink on a Region's f-equation.

    Decomposed into a (gain, loss_rate) pair to match the collision-term
    contract — signed fluxes are explicitly rejected.  Extraction ("flow
    out of this region via a junction") is encoded as ``loss_rate * f``,
    NOT as a negative gain.  This preserves the positivity and Jacobian
    structure the ETD / Newton solvers rely on.

    Units are ``1/ns`` to match the rest of the diffusion stack (not Hz).
    Converters from Stage-A-style Hz rates live in the Junction
    implementation, not in the ExternalFlux contract.
    """
    gain: np.ndarray          # ≥ 0 everywhere. Shape (NE,).
                              # Additive source rate, units 1/ns.
    loss_rate: np.ndarray     # ≥ 0 everywhere. Shape (NE,).
                              # Multiplier on f in the damping term,
                              # so the RHS contribution is -loss_rate*f,
                              # units 1/ns.
    diagnostics: dict[str, str | float] = field(default_factory=dict)
                              # Junction name, total injected current,
                              # etc. — passed to the solver logs.
```

Construction validates the contract and cannot be bypassed afterwards:
complex input is rejected before the float cast (NumPy would otherwise drop an
imaginary NaN with only a `ComplexWarning`), ``(NE, 1)`` is squeezed to
``(NE,)`` transparently, shapes must match, entries must be finite and
non-negative, and the stored arrays are copies marked read-only so
``ef.gain[0] = -1`` raises instead of corrupting solver state.
``ExternalFlux.zero(NE)`` builds the zero-flux instance for a grid.

Two further checks run at the solver boundary rather than in the constructor:

* ``_validate_for_NE(NE)`` rejects a flux whose length disagrees with the
  solver grid. The pathological case is length 1, which would broadcast
  silently across every bin and turn a single-bin contract into an all-bin one.
* ``_validate_gain_support(mask)`` rejects a positive gain on a bin whose
  represented spectral capacity is exactly zero. A loss rate on such a row is
  harmless; injection into an unrepresented state is undefined.

#### 2.2.2 How it threads through the solvers

| Surface | Behaviour |
|---|---|
| ``DiffusionBackend.step(state, dt, external_flux=...)`` | Adds ``gain`` to the explicit piece and ``loss_rate`` to the damping piece in the ETD2 substep. |
| ``DiffusionBackend.apply_collisions(..., external_flux=...)`` | Same pair added to the collision RHS; ``apply_collisions_with_diagnostics`` reports it. |
| ``DiffusionBackend.steady_state(..., external_flux=...)`` | Validates length and gain support, then routes to the fixed-gap or self-consistent-gap path. |
| ``solve_steady_state(..., external_flux=...)`` | Propagates to the Newton and coupled-Newton paths. |
| ``newton_solve_f(..., external_flux=...)`` | Adds ``+ gain − loss_rate * f`` to the residual and ``-loss_rate`` to the Jacobian diagonal. |
| ``coupled_newton_solve(..., external_flux=...)`` | Same; the ``(f, n_ph)`` coupled Newton sees the extra terms on f only. |
| ``run_time_dependent(..., external_flux=callable)`` | Callable returning ``ExternalFlux`` at each substep midpoint. This time-varying path requires ``stop_tol=None`` because an instantaneous residual cannot certify future convergence of a non-autonomous drive. |

``external_flux=None`` is the default and leaves the residual untouched.

Unaccelerated Picard (``method="picard"``, ``anderson_depth=0``,
``use_thermal_phonons=False``) is brittle when any perturbation feeds through
the phonon-emission cycle, and a non-zero ``external_flux`` reliably pushes it
into oscillating non-convergence. ``steady_state`` catches that combination at
the API boundary with a routing hint — raise ``anderson_depth``, switch to
``method="coupled_newton"``, or use ``use_thermal_phonons=True`` when the
τ_l → 0 limit applies — rather than failing after 200 iterations.
``external_flux_is_conservative_transfer=True`` requires an ``ExternalFlux``
and is not available on the coupled-Newton path.

#### 2.2.3 Contract tests

The contract tests (`tests/devices/test_external_flux.py`) run with
``enable_recombination=False, enable_scattering=False,
enable_photon_scattering=False`` so only the ExternalFlux term plus the
spectral-flow term is live, keeping the assertions clear of the (nonlinear in
f, via cross-bin partners) collision kernels:

* **Zero flux identity**: ``ExternalFlux=None`` reproduces the no-flux path
  bit for bit across the Fischer validations.
* **Linear ODE closed form**: collision kernels disabled,
  ``ExternalFlux(gain=g(E), loss_rate=r(E))`` with constant ``g, r``.
  Steady state is ``f(E) = g(E) / r(E)`` by direct construction of
  the ODE ``df/dt = g − r f``. Compared to float64 precision.
* **Detailed-balance ansatz**: with kernels disabled, set
  ``gain, loss_rate`` such that ``gain/loss_rate = f_FD(E, T_bath)``
  pointwise in E. Steady state is Fermi-Dirac by construction —
  the same linear ODE as above, pinning that the contract
  *supports* a detailed-balance setup.
* **Conservation under injection**: with kernels disabled,
  ``∂_t n_qp`` from ExternalFlux equals
  ``4 ρ_F ∫ ρ(E) (gain − loss_rate · f) dE`` — a linear identity
  checked directly; confirms that the observable-level
  conservation law is wired consistently with the Fischer-
  convention normalization.

Nonlinear behavior with collisions on is exercised by the two-region
thermal-equilibrium tests in `tests/devices/test_device.py`, where the
invariant is architecturally richer.

---

## 3. Layer 2 — Device, Junction, Qubit

### 3.1 Device

```python
@dataclass
class Device:
    regions: dict[str, Region]              # keyed by Region.name
    junctions: list[Junction] = field(default_factory=list)
    qubit: Qubit | None = None
```

A Device is just data. The solver is the `solve_device_steady_state`
free function in §3.5. No "Device" class methods for evolution —
keeps the data/behavior split clean.

The constructor validates the topology: at least one region; every
`regions` key equal to its own `Region.name`; every Junction's `region_a` /
`region_b` a known key; and no region shared by two Junctions when one of them
declares `requires_exclusive_regions`. A reduced closure that solves an
isolated subsystem internally cannot respond to another Junction changing
either region's f(E), so that topology would produce a superficially converged
but non-self-consistent solution.

**Single-region Devices exist.** A caller can explicitly construct
`Device(regions={"main": region}, junctions=[])`. The Fischer services call
their single-region solver paths directly.

### 3.2 Junction

`Junction` is an abstract base. The framework owns the protocol; the physics
lives in the subclasses.

```python
class Junction(ABC):
    """A tunnel coupling between two named Regions."""

    name: str
    region_a: str                           # key in Device.regions
    region_b: str

    # Class-level safety contracts read by the Device solver.
    owns_region_dissipation: bool = False
    requires_exclusive_regions: bool = False
    prescribed_region_flux: ClassVar[bool] = False

    def qp_number_capacity_ratio_a_to_b(self) -> float | None:
        """``C_a / C_b`` for an active conservative QP-transfer edge."""
        return None

    @abstractmethod
    def evaluate(
        self,
        state_a: DiffusionState,
        state_b: DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        """Compute per-region flux contracts + qubit rates."""
```

`JunctionResult` carries ``external_flux_a`` and ``external_flux_b`` — the
``ExternalFlux`` contributions for each region, which the solver sums over
every junction touching that region — and ``qubit_channels``, a list of
``QubitTransitionChannel`` records (§3.3.1), empty when the Junction has no
qubit coupling. ``qubit_state`` is supplied for the small set of Junctions
whose rates depend on qubit populations; most Junctions ignore it.

The three class-level flags are safety contracts, not hints:

* **`owns_region_dissipation`.** `True` means the emitted flux already
  includes the moment-integrated dissipation physics (M25's ``r_α x_α²``
  recombination and ``g_α`` thermal-phonon generation), so the Device solver
  runs the touched regions with the backend's ``external_dissipation_only=True``
  path to avoid double-counting the e-ph collision kernel. At most one such
  Junction may touch any region; the solver enforces this.
* **`requires_exclusive_regions`.** `True` means the closure is valid only
  when each touched region belongs to this Junction alone.
* **`prescribed_region_flux`.** `True` means each emitted region flux is a
  prescribed local source/sink rather than a state-dependent exchange between
  the two regions, and is certified by each region's Newton number-mode check.
  The conservative component certificate cannot infer this from arbitrary
  `evaluate` code, so the default is `False`.

`qp_number_capacity_ratio_a_to_b` returns ``C_a / C_b`` when the Junction
transfers quasiparticles conservatively between its two regions, where the
conserved discrete population is
``C_a * sum(w_a * f_a) + C_b * sum(w_b * f_b)``. It must stay present when the
instantaneous *net* current happens to vanish, and every `evaluate` result must
conserve the stated weighted population; the Device solver verifies that
identity before using the ratio in its component number-mode certificate.
`None` covers Junctions without such a contract and couplings identically
disabled by configuration.

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

* ``gain_i = j_J^{in}(E_i) / (4 ρ_F ρ_i V_region)`` or
  ``I_{J,i}^{in} / (4 ρ_F ρ_i V_region dE_i[eV])``;
* ``loss_rate_i = j_J^{out}(E_i) / (4 ρ_F ρ_i V_region f_i)`` or
  ``I_{J,i}^{out} / (4 ρ_F ρ_i V_region f_i dE_i[eV])``.

The in/out spectra are split by sign; the loss-rate form recovers the
positivity-preserving solver structure — see §2.2.

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
that preserves the moment sum — uniformly in spectral measure over each
electrode's active sub-band(s), normalized so that
``(2/Δ_α) ∫ ρ × gain dE = gain_moment`` holds exactly. That
distribution choice is an *approximation* introduced by the moment-closure
wrapper; a Junction consuming f_L(E), f_R(E′) at paired energies computes
``gain(E)`` directly instead.

**Conservation invariants** (pinned by `tests/devices/test_device.py`):

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

**`SymmetricGapTunnelingJunction`** is the energy-conserving coupling between
two regions with matched gaps: at each energy `E`, quasiparticles tunnel in
proportion to ``[f_a(E) − f_b(E)]``, giving ``gain_a(E) = α_a f_b(E)``,
``loss_rate_a(E) = α_a``, and the analogous ``α_b`` decomposition for region B.
The rates obey ``C_a α_a = C_b α_b``, so the weighted population
``C_a f_a + C_b f_b`` is conserved even when two matched-gap electrodes have
unequal volumes or normal-state densities of states. The (1 − f)
Pauli-blocking factors of the full tunneling matrix element are absorbed into
the cross-region detailed balance: the pair above is valid in the
non-degenerate limit ``f_α ≪ 1`` that holds throughout the superconducting
regime where this framework lives.

Its scalar `capacity_ratio_a_to_b` represents a global material/volume factor
only. It is valid when both regions share the same per-bin finite-volume
spectral measure, so runtime evaluation requires each state's public gap to
match its spectral gap and requires identical energy centers, cell widths,
spectral gaps, Dynes broadening, and `cell_weights` across the junction. An
energy-dependent capacity vector needs an energy-resolved asymmetric junction
model; it is not approximated by the scalar ratio. ``alpha_per_ns == 0`` is a
genuinely disabled coupling rather than a graph edge: `evaluate` returns zero
fluxes before the cross-region compatibility checks, and
`qp_number_capacity_ratio_a_to_b` returns `None`, so an inert Junction cannot
merge independent conservation components or put its unused capacity ratio into
certification.

#### 3.2.2 Unit convention summary

| Quantity | Unit | Why |
|---|---|---|
| ``ExternalFlux.gain``, ``loss_rate`` | 1/ns | Matches every collision evaluator |
| Junction spectrum ``j_J(E)`` | events/(ns·eV) | Continuous normalization |
| Per-bin current ``I_{J,i}`` | events/ns | Includes ``dE_i[eV]`` |
| Stage A ``Γ̃^α_{ij}`` | Hz | Paper convention; converted at the Junction boundary |
| Gaps, energies | μeV or K (per backend) | Diffusion-stack convention |
| ``tau_0``, ``tau_0_phonon`` | ns | Material YAML |

``M25GapAsymmetricJJ`` is responsible for converting from Stage A's
Hz rates to the Junction's internal ns units and for the
``N_CP = 2 ν_0 Δ V`` normalization.

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
    omega_kelvin: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
                                            # level energies, Kelvin
    E_J_kelvin: float = 0.0                 # transmon Josephson energy
    E_C_kelvin: float = 0.0                 # transmon charging energy


@dataclass
class QubitState:
    """Probabilities over the (level, parity) state space.

    Shape is ``(n_levels, 2)`` when ``track_parity`` is set, else
    ``(n_levels,)``. The parity axis has two entries: [0] = even,
    [1] = odd. ``np.sum(p) = 1`` always.
    """
    p: np.ndarray
    t_ns: float = 0.0                       # lab-frame time; zero at steady state
```

`Qubit` validates `n_levels ≥ 1`, the length of `omega_kelvin`, and that the
level energies are non-decreasing (level 0 is the ground reference).
`QubitState` validates shape, finiteness, non-negativity, and
``|sum(p) − 1| ≤ 1e-12``.

`omega_kelvin`, `E_J_kelvin` and `E_C_kelvin` are annotations: they are
validated on the `Qubit` but read by nothing in the solver. The splitting that
sets the rates, and the only detailed-balance relation in the code, is
`M25PhysicalParameters.omega_10_kelvin` carried by the junction, and the M25
matrix elements come from `M25PhysicalParameters.E_J_kelvin` /
`E_C_kelvin`. Nothing cross-checks the two, so a value set on the `Qubit` does
not move any population.

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
    rate_per_ns: float                      # transition rate, ≥ 0, finite
    flips_parity: bool                      # True for eo channels
    label: str = ""                         # "ph_00", "ee_10", etc.
                                            # — diagnostic only
```

``level_from`` may equal ``level_to`` when the transition is purely
parity-flipping. The rate is state-independent at the master-equation level:
any dependence on the source region's f(E) is computed inside the Junction's
`evaluate`.

`build_rate_matrix` consumes a list of these channels and assembles a
transition matrix on the full ``(level, parity)`` state space. For the M25
setup with 2 levels × 2 parities this produces a 4×4 rate matrix;
`solve_qubit_master_equation_steady_state` returns the populations satisfying
``M @ p = 0`` with ``sum(p) = 1``.

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
    parity_preserving_rates: np.ndarray | None = None
                                            # shape (n_levels, n_levels) — 1/ns
```

In the single-junction transmon approximation (E_J ≫ E_C) only the off-diagonal
``s²_{10}``, ``s²_{01}`` are nonzero — the diagonals are exponentially
suppressed by charge dispersion — while only the diagonals ``c²_{ii}`` are
nonzero and ≈ 1 at leading order in ``E_C / E_J``.
``parity_preserving_rates`` defaults to `None`, meaning no ``ee`` channels are
emitted, because the default tunneling event flips parity; the M25 setup
supplies these as independent drivers.

The `Junction.evaluate` implementation decides which channels to
emit — for an `M25GapAsymmetricJJ`, each Γ̃^α_{ij} tunneling rate
produces a ``QubitTransitionChannel(level_from=i, level_to=j,
rate_per_ns=..., flips_parity=True)``; the Γ̃^{ee} terms produce
``flips_parity=False`` channels. The qubit evolver sees both
kinds and handles them correctly on the 4-state grid.

### 3.4 Matrix elements: where M25 Stage A lives

The M25 Stage A module `qpsim.services.rate_equation_coefficients`
evaluates:

* `_s_10_squared`, `_c_ii_squared` — transmon matrix elements
  (SI Eq. S25–S28).
* `_gamma_L_10`, `_gamma_Rgt_10`, `_gamma_Rlt_10`, `_gamma_L_01`,
  `_gamma_Rgt_01`, `_gamma_L_ii`, `_gamma_Rgt_ii` — the 12
  Γ̃^α_{ij} tunneling-rate integrands (SI Eqs. S30–S36).
* `_tau_R_inverse`, `_tau_E_inverse` — intraband relaxation (SI S50).
* `_branching_fraction` — ξ (SI S37).
* `_S_ph_total`, `_S_ph_Rgt`, `_S_ph_Rlt` — photon spectral density
  (Note V, Eqs. S57–S59).

`M25GapAsymmetricJJ` is the `Junction` subclass that carries them:

```python
@dataclass
class M25GapAsymmetricJJ(Junction):
    """Gap-asymmetric JJ coupled to a transmon qubit, with pair-
    breaking photon drive. Implements the M25 physics exactly.
    """
    name: str
    region_a: str                           # L electrode (high gap)
    region_b: str                           # R electrode (low gap)
    m25_params: M25PhysicalParameters       # gaps, ω_10, E_J/E_C, T
    m25_drive: M25PhotonDrive               # pair-breaking photon channel
    branch_picker_mode: str = "min_residual"
    expected_ordering: tuple[str, ...] | None = None
```

Region B's energy grid must span ``[Δ_R, ∞]`` so both R< (E ∈ [Δ_R, Δ_L]) and
R> (E ≥ Δ_L) sub-band moments can be extracted. `evaluate`:

1. Caches `M25Coefficients` (state-independent for fixed parameters) and the
   moment-solver fixed point ``(p_1, x_L, x_{R>}, x_{R<})`` from
   `solve_rate_equation_steady_state_multi_seed` on the first call. Both caches
   are keyed by a value fingerprint of every primitive input field, so
   replacing either bundle rebuilds the affected cache.
2. Validates that the region-state gaps agree with the `m25_params` bundle the
   cached coefficients were built from, on both `state.gap` and
   `state.spectral.gap`. The quantity at risk is not Δ but the asymmetry
   ω_LR = Δ_L − Δ_R, which the whole sub-band structure is built on and which
   is ~1/99 of Δ at the Fig. 3a point: a 0.1% per-gap slack there admits ~20%
   of ω_LR — ≈9% in the emitted f_{R<} (the R< band measure is
   √(Δ_L²−Δ_R²)) and ≈17% in the moments the cached coefficients carry.
3. Builds per-region ``ExternalFlux`` from the cached moment values, spreading
   each moment rate uniformly in spectral measure over the electrode's active
   sub-band(s) under the §3.2.1 normalization. Pure-BCS cells use analytic DOS
   weights, and the R</R> boundary must be a cell face so one cell never
   carries two incompatible moment rates.
4. Emits one `QubitTransitionChannel` per transition: ``eo`` from QP tunneling
   (``Γ̃^α_{ij} × x_α`` for α ∈ {L, R>, R<}) and from photon-assisted
   tunneling (``Γ̃^{ph}_{ij}``), both `flips_parity=True`; ``ee``
   (``Γ̃^{ee}_{ij}``) with `flips_parity=False`. The Device's qubit master
   equation reaches the same ``p_1`` as the moment solver because the channels
   carry the same effective rates.

The moment-level ``g_α`` and ``r_α x_α²`` already include the
moment-integrated e-ph generation and recombination, so the class sets
``owns_region_dissipation = True`` and the Device solver routes the inner solve
through ``external_dissipation_only=True``: the inner Newton sees only the
M25-supplied (gain, loss_rate), not also the e-ph collision kernel that would
crush f(E) to thermal. Because the cached fixed point is an isolated
two-electrode closure — `evaluate` reads `state_a.f` and `state_b.f` for grid
metadata only, never for their occupation values — the class also sets
``requires_exclusive_regions = True``. Building the flux from the cached
moments sidesteps the cross-electrode bootstrap at Δ_L ≈ Δ_R, where the inner
Newton's residual floor masks the x_L ↔ x_R> exchange and a state-driven
Picard locks orders of magnitude below the M25 fixed point; the Device Picard
then converges in 2 iterations because the emitted flux is constant across
iterates. `qubit_state` is accepted for API parity and ignored: the M25 master
equation owns ``p_1`` self-consistently with the QP densities, so an external
qubit-state perturbation would over-determine the coupled system.

The moment-closure specialization ("Fermi-Dirac on each region")
is encoded in *this* subclass via the choice to call Stage A's
closed-form evaluators. A subclass integrating over f(E) directly produces a
Junction that works on non-thermal distributions — Layer 2 in its general
form.

The `branch_picker_mode` default `"min_residual"` selects the true fixed
point: with the single-quasiparticle normalization of the density equations
(`M25Coefficients.cooper_pair_number_R`, set by the Note-V builder used here)
the M25 system has a unique physical root and this mode finds it.
`expected_ordering` is an optional moment-ordering hint forwarded to the
solver.

### 3.5 Top-level solver

```python
def solve_device_steady_state(
    device: Device,
    *,
    backend: DiffusionBackend | None = None,
    use_thermal_phonons: bool = True,
    inner_anderson_depth: int = 3,
    outer_tol: float = 1e-6,
    outer_max_iter: int = 100,
    outer_damping: float = 0.5,
    inner_newton_tol: float = 1e-12,
    inner_newton_max_iter: int = 200,
) -> DeviceSolution:
    """Damped outer Picard loop on (junction fluxes ↔ region states).

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
      5. Measure the UNDAMPED fixed-point defects in f, stored n_ph and
         qubit populations, then take the damped step
         ``x_next = x + outer_damping*(F(x) - x)``; certify supported
         conservative components with a capacity-weighted QP-number
         backward error.

    Returns the certified region/qubit snapshot and convergence diagnostics.
    """
```

The **outer iteration** is a Picard fixed-point on (region states, qubit
state). The **inner** per-region Newton operates on a single-region kinetic
equation with frozen external flux. It is usually much better scaled than the
four-variable closure, but that is not an unconditional convergence promise:
cold number modes can be unrepresentable in float64 and fail loudly.

Convergence is certified against a **scale-aware** tolerance: every region's
defects must satisfy `outer_tol` relative to the corresponding `f` / `n_ph`
scale (with a tiny absolute floor for all-zero states), and the qubit defect
likewise against ``||p||_inf``. At 100 mK the entire occupation signal is
~8e-10, so an absolute threshold would certify any state pair — including a
50%-wrong region swap on a period-2 orbit of the Jacobi map. Damping breaks
two-cycles; the `outer_damping = 0.5` default suppresses the period-2 Jacobi
oscillations of symmetric exchange-coupled regions, and the full step is taken
when the map's output has stopped moving, which is not an orbit.

`DeviceSolution` returns the converged `DiffusionState` per region, the
`QubitState` when one is configured, `n_outer_iterations`, and the last
iteration's `final_max_delta_f`, `final_max_delta_n_ph`, `final_max_delta_p`
and `final_max_number_backward_error`.

**Certification contract (2026-07-21).** The solver
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
cross-region component would require a nonequilibrium-phonon component
certificate. Exact absorbing vacuum states are accepted;
unresolved finite-temperature number modes still fail loudly. Prescribed
sources retain their own turnover in the Newton normalizer; only explicitly
identified conservative exchange is excluded from that scale. A returned
state is the same snapshot at which its defect was measured. An injected
backend must act as a pure map (in-place mutation refuses), and its complete
returned state — occupation, positive matching gap/spectral context,
fixed finite grid, bath temperature, and phonon state — is validated before
convergence arithmetic. Conservative-transfer verification scales raw rates
and finite-volume weights separately before multiplication, so large finite
inputs cannot overflow into a NaN that bypasses the contract. Invalid qubit
outputs are rejected at the same boundary.

---

## 4. Layer 3 — Moment-closure reductions

`solve_rate_equation_steady_state` is **exactly** the moment-closure reduction
of Layer 2 under the assumption that f_α(E) is Fermi-Dirac per sub-band.

### 4.1 Reduction operator

The M25-convention dimensionless density of sub-band α is

    x_α = (2/Δ_α) ∫_{Δ_α}^∞ ρ_α(E) f_α(E) dE

implemented as `_moment_x_M25` in `qpsim/devices/m25_junction.py`, which
integrates a region's occupation against the per-cell spectral measure for one
physical energy band. Cells crossed by a band bound are split by their
spectral measure rather than assigned by centre location.

Pure-BCS cells use the analytic DOS primitive — the DOS *integrated* across
the cell, which stays finite at the gap edge where the DOS itself does not; a
point sample there undercounts the gap-edge cell by exactly 1/√2, 29.3%, at
every resolution. Broadened cells take the DOS at the cell centre times the
geometric overlap, which is accurate while the broadening is at least as wide
as the cells near the gap; `_spectral_band_weights` warns below that ratio,
because the weights stay positive and plausible and only the gap-edge cell is
wrong. See “Quasiparticles are stored as cell averages, phonons as point
samples” in `docs/Phonon_Model_Decisions.md`.

### 4.2 M25 rate equation as derived Layer 3

With the Fermi-Dirac ansatz on each Region, the `Junction.evaluate`
call reduces to the Γ̃^α_{ij} × x_α representation and the
kinetic equation becomes algebraic in (p_1, x_α). This is exactly
Eqs. 4–6 of M25 — `_rate_equation_residual`. So
Layer 3 can be thought of as:

    Layer 3 = Layer 2 solved on a `Device` with every region
              backend replaced by a moment-closure "dummy backend"
              that stores (x_α, µ_α) instead of f(E).

This view is what makes the 4-variable system the specific
case it is, not the architectural norm.

### 4.3 When to use which layer

| Use case | Layer |
|---|---|
| Non-thermal f(E), athermal drive, transient relaxation | Layer 2 |
| Paper reproduction of M25 Fig 3/4/5 (steady-state, thermal) | Either; Layer 3 faster if the ansatz holds |
| Multi-region Fischer-Catelani-style MKID networks | Layer 2 |
| Quick engineering estimate of junction current at known T | Layer 3 |
| Single-region Fischer validation | Layer 1 |

The framework offers all three. Layer 3 is a *derived*, faster, less
general reduction of Layer 2 — **not the primary object**.

---

## 5. Single-region entry points

* **`qpsim.services.steady_state.solve_steady_state`,
  `nbar_loop.solve_nbar_loop`, and `transient.run_time_dependent`** are
  direct single-region APIs. They do not internally construct a Device.
* **Stage A M25 machinery** (`M25PhysicalParameters`,
  `coefficients_from_physical_parameters`, the `_S_ph_*` and `_gamma_*`
  evaluators) is the internals of `M25GapAsymmetricJJ`
  and of the Layer-3 reduction path — one derivation, two consumers.
* **`qpsim.services.rate_equation.solve_rate_equation_steady_state`** is
  the M25 moment-reduction entry point.
* **Single-region baselines** (`validation/baselines/constant/`,
  `validation/baselines/kaplan/`) are produced through the single-region
  paths.

---

## 6. Conventions

### 6.1 Where photon drive lives

`PhotonDrive` and `PhotonState` sit in different places for different drive
mechanisms (pair-breaking vs sub-gap vs photon-assisted tunneling). For
Devices, pair-breaking drive is a per-Region property (absorbed in the film),
while photon-assisted tunneling (M25 Γ_ν) is a per-Junction property.

A `Region` owns its phonon state AND its pair-breaking photon bath. A
`Junction` owns the photon-assisted-tunneling drive (Γ_ν × n̄). No field has
two homes.

### 6.2 Region, not electrode

"Region" is the general term; "electrode" implies a two-electrode device
topology. MKIDs are films or resonators, JJs are electrodes, and the framework
works for all of them. `Region` is therefore the primitive, and
`Junction.region_a` / `region_b` hold the names. A JJ built on this layer will
use names `"L"` / `"R"` by convention; the framework does not care.

### 6.3 Qubit steady state and time evolution

For steady-state solves, the qubit master equation is another algebraic
equation: `solve_qubit_master_equation_steady_state` solves `ṗ = 0` jointly
with the region states inside the outer Picard loop. The same rate matrix,
assembled by `build_rate_matrix` from the pooled channel list, is what a
time-marched qubit integrates. Same underlying physics, different solver loop.
