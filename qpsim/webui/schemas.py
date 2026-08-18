"""Pydantic setup models — the frontend's serializable simulation configs.

One model per run mode, discriminated on ``mode``:

* ``steady_state_0d`` — 0-D T3 kinetic steady state (Newton / Picard /
  coupled-Newton) with optional photon drives.
* ``transient_0d`` — ETD2 collisional transient ``f(E, t)``.
* ``spatial_1d`` — 1D strip driven to steady state (diffusion-operator
  family, optional two-gap step + Kupriyanov–Lukichev interface).
* ``m25_junction`` — M25 gap-asymmetric junction moment layer over a
  temperature sweep.

These models validate *shape and static bounds* only. Cross-field
physics checks (drive frequencies vs 2Δ, Dynes × spatial transport,
grid commensurability) live in :mod:`qpsim.webui.builders` so their
messages can reference derived quantities like the grid spacing.

Units follow the engine convention: energies in μeV, times in ns,
temperatures in K, lengths in μm — except the M25 layer, whose inputs
are in the paper's natural GHz (÷h) and Hz units and are converted to
Kelvin at the builder boundary.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from qpsim.materials import load_material
from qpsim.materials.database import validate_rho_F_eV

# Default material values come straight from the YAML database so the
# frontend never carries a second, drifting copy of Al's parameters.
_AL = load_material("Al")

# Upper bound on emitted snapshots. A ``snapshot_interval`` far below the
# integration step would otherwise drain an unbounded inner cadence loop
# before cancellation is ever polled (single-worker runner → memory blow-up
# and an uninterruptible job). Reject such setups at validation time.
#
# Taken from the ENGINE rather than restated. This was written twice with
# values 25x apart -- 100_000 here against the spatial backend's 4_000 -- so
# the 2-D mode's own defaults sat in the gap: a setup asking for a frame per
# step validated green in the editor and then died at run time with a bare
# engine ValueError, in the one place the message could not be acted on. Two
# numbers for one rule also means tuning either silently moves where setups
# are rejected.
from qpsim.backends.t3_spatial import _SNAPSHOT_HARD_CAP as _MAX_SNAPSHOTS


def _reject_dense_snapshots(snapshot_interval: float | None, run_time: float) -> None:
    """Raise if ``snapshot_interval`` would emit more than ``_MAX_SNAPSHOTS``."""
    if snapshot_interval is None:
        return
    n_snapshots = run_time / snapshot_interval
    if n_snapshots > _MAX_SNAPSHOTS:
        raise ValueError(
            f"snapshot_interval={snapshot_interval:g} would emit ~{n_snapshots:.3g} "
            f"snapshots over a run time of {run_time:g} (cap {_MAX_SNAPSHOTS}). "
            f"Increase snapshot_interval to at least "
            f"{run_time / _MAX_SNAPSHOTS:g}."
        )


class StrictModel(BaseModel):
    """Base: reject unknown keys so stale setup files fail loudly."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MaterialParams(StrictModel):
    """Superconductor parameters (editable copy of a database material)."""

    name: str = _AL.name
    Delta_0: Annotated[float, Field(gt=0.0)] = _AL.Delta_0  # gap Δ₀ (μeV)
    T_c: Annotated[float, Field(gt=0.0)] = _AL.T_c  # critical temperature (K)
    tau_0: Annotated[float, Field(gt=0.0)] = _AL.tau_0  # e-ph characteristic time (ns)
    tau_0_pb_ns: Annotated[float, Field(gt=0.0)] | None = _AL.tau_0_pb_ns  # τ₀^PB (ns)
    D_0: Annotated[float, Field(ge=0.0)] = _AL.D_0  # normal-state diffusion (μm²/ns)
    rho_F: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = _AL.rho_F  # eV⁻¹ m⁻³
    dynes_gamma: Annotated[float, Field(ge=0.0)] = 0.0  # Dynes broadening Γ (μeV)

    @model_validator(mode="after")
    def reject_legacy_rho_f_units(self) -> MaterialParams:
        validate_rho_F_eV(self.rho_F, allow_zero=True)
        return self


class EnergyGrid(StrictModel):
    """Uniform cell-centered energy grid in units of the gap."""

    # Sub-gap cells are required when a self-consistent gap can fall below
    # Delta_0. They are inert (rho=0) for the initial pure-BCS spectrum.
    min_factor: Annotated[float, Field(ge=0.0)] = 1.0
    max_factor: Annotated[float, Field(gt=1.0)] = 10.0
    # 405, not 400. The phonon frequency grid is the union of a difference
    # lattice (scattering) and a sum lattice (pair breaking), and they share
    # bins only when 2*E_face/dE is an integer. On the default [Delta, 10*Delta]
    # window 400 gives 88.889 and the two channels land on disjoint
    # sublattices, so a scattering-emitted phonon above 2*Delta can never break
    # a pair. 405 gives 90 exactly. Only dynamic-phonon runs are affected --
    # a thermal bath never builds the map -- but the default must be one a
    # dynamic run can legally use.
    num_bins: Annotated[int, Field(ge=8, le=5000)] = 405

    @model_validator(mode="after")
    def max_must_exceed_min(self) -> EnergyGrid:
        if self.max_factor <= self.min_factor:
            raise ValueError("max_factor must be greater than min_factor")
        return self


class PhononInitialCondition(StrictModel):
    """Where the phonon population starts, when that is not the bath.

    The default seed and the target the escape term relaxes toward are the
    same Bose-Einstein value, so a run left at ``bath`` starts exactly at that
    term's own fixed point and the phonons do not move. Escape is then
    unmeasurable: a check on it would agree just as well with the term
    switched off. Every other kind here is a deliberate departure that gives
    the relaxation something to relax from.
    """

    kind: Literal["bath", "thermal_at", "scaled", "expression"] = "bath"
    T_eff: Annotated[float, Field(gt=0.0)] | None = None   # thermal_at (K)
    factor: Annotated[float, Field(ge=0.0)] = 1.0          # scaled x n_bath
    # Variables in scope: omega (ueV), n_bath (the Bose value there), params.
    expression: str | None = None
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def the_departure_must_be_real(self) -> PhononInitialCondition:
        if self.kind == "thermal_at" and self.T_eff is None:
            raise ValueError("kind='thermal_at' needs T_eff (K).")
        if self.kind == "expression" and not self.expression:
            raise ValueError("kind='expression' needs an expression.")
        if self.kind == "scaled" and self.factor == 1.0:
            raise ValueError(
                "kind='scaled' with factor=1 is the bath value exactly, so "
                "the phonons would start at the escape term's own fixed point "
                "and nothing would relax. Use a factor != 1, or kind='bath'."
            )
        return self


class PhononSector(StrictModel):
    """Ph0 phonon-sector choice.

    * ``thermal_bath`` — n_ph pinned at the Bose–Einstein bath
      (Fischer τ_l → 0 limit; Newton steady-state path).
    * ``dynamic_escape`` — dynamic n_ph with finite acoustic escape
      time ``tau_l_ns`` (Picard/Anderson or coupled-Newton).
    * ``dynamic_closed`` — dynamic n_ph with no substrate escape
      (τ_l → ∞; the engine's ``tau_l = 0.0`` sentinel).

    Dynamic modes use the F&C Eq. 12 phonon-side kernel by default.
    ``use_phonon_side_kernel=False`` is retained only for reproducing legacy
    runs that reused the quasiparticle-side kernel in the phonon equation.
    """

    mode: Literal["thermal_bath", "dynamic_escape", "dynamic_closed"] = "thermal_bath"
    tau_l_ns: Annotated[float, Field(gt=0.0)] = 0.170
    use_phonon_side_kernel: bool = True
    # Only meaningful in the dynamic modes; thermal_bath pins n_ph outright.
    initial: PhononInitialCondition = PhononInitialCondition()


class CollisionTerms(StrictModel):
    """Which electron-phonon channels are present in the kinetic equation.

    Both default to on, which is the physical model. Switching one off is a
    deliberate *reduction*: it removes that term from the right-hand side so a
    single mechanism can be studied on its own.

    A reduction is not a physical state. With ``scattering=False`` the
    quasiparticle energies cannot relax, and with ``recombination=False``
    nothing sets the quasiparticle number, so a driven run has no steady state
    at all. Neither reduced model has a thermal fixed point, so detailed
    balance and the number-conservation certificate do not apply to it.
    """

    scattering: bool = True
    recombination: bool = True
    # The phonon-side counterparts. Each pair is ONE physical process booked
    # on both sides of the ledger, so switching a side off on its own is an
    # energy-conservation violation, not a smaller model: one population
    # trades energy with another that does not record it. Allowed on purpose,
    # with a warning, because seeing the imbalance is the point.
    phonon_scattering_source: bool = True
    phonon_recombination_source: bool = True


class SubGapDrive(StrictModel):
    """Single-mode sub-gap photon drive (requires ω₀ < 2Δ)."""

    enabled: bool = False
    omega_0: Annotated[float, Field(gt=0.0)] = 22.0  # photon energy (μeV)
    n_bar: Annotated[float, Field(ge=0.0)] = 0.0  # mean photon number
    c_phot: Annotated[float, Field(ge=0.0)] = 0.06e-9  # coupling (1/ns)


class PairBreakingDrive(StrictModel):
    """Pair-breaking photon drive (requires ω_PB > 2Δ)."""

    enabled: bool = False
    omega_PB: Annotated[float, Field(gt=0.0)] = 529.2  # photon energy (μeV)
    n_bar_PB: Annotated[float, Field(ge=0.0)] = 0.0
    c_phot_PB: Annotated[float, Field(ge=0.0)] = 1e-9  # coupling (1/ns)


class SolverOptions(StrictModel):
    """0-D steady-state solver knobs.

    ``method="auto"`` picks the canonical route for the phonon sector:
    the Newton thermal path for ``thermal_bath``, Anderson-accelerated
    Picard for the dynamic sectors.
    """

    method: Literal["auto", "picard", "coupled_newton"] = "auto"
    self_consistent_gap: bool = False
    picard_tol: Annotated[float, Field(gt=0.0)] = 1e-8
    picard_max_iter: Annotated[int, Field(ge=1, le=100000)] = 500
    picard_mixing: Annotated[float, Field(gt=0.0, le=1.0)] = 0.2
    anderson_depth: Annotated[int, Field(ge=0, le=50)] = 3
    newton_tol: Annotated[float, Field(gt=0.0)] = 1e-12
    newton_max_iter: Annotated[int, Field(ge=1, le=100000)] = 300
    # The tolerance that actually gates the COUPLED Newton route. Its `tol`
    # (which newton_tol feeds) is read only under `step_rtol <= 0`, so with
    # the shipped default it gates nothing and the displayed "Newton
    # tolerance" had no effect on that route at all -- an inert control. `tol`
    # is disabled there deliberately: an absolute residual is unreliable when
    # every amplitude is ~1e-10, where a warm seed can sit below it and exit
    # at iteration 0 with a stale state. So the fix is to expose the knob that
    # acts, not to re-enable the one that was switched off for a reason.
    # Default matches the engine exactly, so numerics are unchanged.
    coupled_newton_step_rtol: Annotated[float, Field(gt=0.0)] = 1e-8
    # Selects how the coupled-Newton cross blocks are BUILT, not what is
    # solved: True is the exact closed-form derivative of the discrete
    # residual (two assemblies per iteration, what every certified driver
    # runs); False is the legacy finite-difference secant, which needs
    # NE + N_omega residual assemblies per iteration for the same root.
    # Defaults True because the alternative is a large, silent cost for no
    # difference in the answer.
    coupled_newton_analytic_cross: bool = True


class ProbeConfig(StrictModel):
    """Mattis–Bardeen probe for σ₁/σ₂, Q_i, and frequency shift.

    Requires ω₀ < Δ (sub-gap probe) and a pure-BCS spectral context
    (the observables raise for Dynes Γ > 0; the frontend skips them
    with a note instead).
    """

    enabled: bool = True
    omega_0: Annotated[float, Field(gt=0.0)] = 22.0  # probe photon energy (μeV)
    alpha: Annotated[float, Field(gt=0.0, le=1.0)] = 0.08  # kinetic-inductance fraction
    Q_ext: Annotated[float, Field(gt=0.0)] | None = None  # extrinsic-loss cap


class EnergyProfileSpec(StrictModel):
    """A shape in energy, peak-normalised to 1.

    Peak-normalised so that the amplitude beside it means the same thing
    whichever shape is chosen: with unit-integral normalisation, narrowing a
    spectral line would silently raise its height and "the same amplitude"
    would be a different experiment on every grid.
    """

    kind: Literal[
        "flat", "thermal", "monoenergetic", "gap_edge", "expression"
    ] = "flat"
    T_eff: Annotated[float, Field(gt=0.0)] | None = None   # thermal (K)
    E_0: Annotated[float, Field(gt=0.0)] | None = None     # line centre (μeV)
    width: Annotated[float, Field(gt=0.0)] | None = None   # line σ (μeV)
    # Variables in scope: E (μeV), gap (μeV), params.
    expression: str | None = None


class SpatialProfileSpec(StrictModel):
    """A shape over the cells, peak-normalised to 1.

    Coordinates are normalised cell centres over the mask's bounding box, so
    ``0.5`` is the middle of the device whatever its size, and a single-cell
    device sits at 0.5 rather than at an end.
    """

    kind: Literal["uniform", "gaussian", "point", "expression"] = "uniform"
    x_0: float = 0.5
    y_0: float = 0.5
    sigma: Annotated[float, Field(gt=0.0)] = 0.12
    # Variables in scope: x, y (normalised), x_um, y_um (μm), params.
    expression: str | None = None


class TimeProfileSpec(StrictModel):
    """How a drive varies in time, as a dimensionless factor.

    ``pulse`` is the one that matters: switch a drive on, switch it off, and
    fit the decay. That measurement was inexpressible while a drive was a
    fixed array held for the whole run.
    """

    kind: Literal[
        "constant", "pulse", "ramp", "exponential", "expression"
    ] = "constant"
    t_on: Annotated[float, Field(ge=0.0)] = 0.0            # (ns)
    t_off: Annotated[float, Field(gt=0.0)] | None = None   # (ns)
    tau: Annotated[float, Field(gt=0.0)] | None = None     # decay/rise (ns)
    # Variables in scope: t (ns), params.
    expression: str | None = None

    @model_validator(mode="after")
    def window_is_ordered(self) -> TimeProfileSpec:
        if self.t_off is not None and self.t_off <= self.t_on:
            raise ValueError(
                f"t_off={self.t_off:g} ns is not after t_on={self.t_on:g} ns, "
                "so the pulse would never be on."
            )
        if self.kind in ("ramp", "exponential") and self.tau is None:
            raise ValueError(f"A {self.kind!r} time profile needs tau (ns).")
        if self.kind == "expression" and not self.expression:
            raise ValueError("An expression time profile needs an expression.")
        return self


class InitialCondition(StrictModel):
    """Where the run starts.

    ``thermal`` is equilibrium at ``T_bath`` and is the default, so every
    existing setup is unaffected. The other kinds are deliberate departures
    from equilibrium, which is what makes a *relaxation* observable: a term
    acting on a state that is already its own fixed point has nothing to
    relax, and its rate cannot be measured.

    ``amplitude`` is a peak excess **occupation**, not a particle count. The
    state here is an occupation ``f in [0, 1]``; the archived implementation
    carried a density and its numbers do not transfer.
    """

    kind: Literal["thermal", "excess", "absolute"] = "thermal"
    amplitude: Annotated[float, Field(ge=0.0)] = 0.0
    energy: EnergyProfileSpec = EnergyProfileSpec()
    space: SpatialProfileSpec = SpatialProfileSpec()
    # Non-separable f(E, x, y). Variables: E, gap, x, y, x_um, y_um, params.
    expression: str | None = None
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def amplitude_present_when_it_matters(self) -> InitialCondition:
        if self.kind == "excess" and self.amplitude == 0.0 and not self.expression:
            raise ValueError(
                "kind='excess' with amplitude=0 prepares exactly the thermal "
                "state, so the run would measure nothing. Set an amplitude, or "
                "use kind='thermal' if that is what you meant."
            )
        return self


class DriveSpec(StrictModel):
    """A prescribed external source over energy, space and time.

    ``A · g_E(E) · g_S(x,y) · g_T(t)`` by default, with ``expression`` as the
    escape hatch for a drive that is not separable — a spot that moves, a
    spectrum that hardens as it decays.

    ``channel`` says which side of the kinetic equation it enters. ``gain`` is
    a source added to df/dt; ``loss`` is a rate coefficient multiplying f, so
    it drains in proportion to what is there — a trap, or an out-tunnelling
    channel.
    """

    enabled: bool = False
    channel: Literal["gain", "loss"] = "gain"
    amplitude: Annotated[float, Field(ge=0.0)] = 0.0      # (1/ns)
    energy: EnergyProfileSpec = EnergyProfileSpec()
    space: SpatialProfileSpec = SpatialProfileSpec()
    time: TimeProfileSpec = TimeProfileSpec()
    # Non-separable g(E, x, y, t). Variables: E, gap, x, y, x_um, y_um, t, params.
    expression: str | None = None
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def an_enabled_drive_must_do_something(self) -> DriveSpec:
        # The recurring failure in this codebase is a default that makes the
        # measurement inert while the code looks right. An enabled drive at
        # zero amplitude is exactly that, so it is refused rather than run.
        if self.enabled and self.amplitude == 0.0 and not self.expression:
            raise ValueError(
                "An enabled drive with amplitude=0 applies nothing, so the run "
                "would look driven and be undriven. Set an amplitude, or set "
                "enabled=false."
            )
        return self


class SteadyState0DSetup(StrictModel):
    """0-D T3 kinetic steady state."""

    mode: Literal["steady_state_0d"] = "steady_state_0d"
    material: MaterialParams = MaterialParams()
    T_bath: Annotated[float, Field(gt=0.0)] = 0.1  # bath temperature (K)
    grid: EnergyGrid = EnergyGrid()
    phonons: PhononSector = PhononSector()
    collisions: CollisionTerms = CollisionTerms()
    subgap_drive: SubGapDrive = SubGapDrive()
    pb_drive: PairBreakingDrive = PairBreakingDrive()
    solver: SolverOptions = SolverOptions()
    probe: ProbeConfig = ProbeConfig()


class Transient0DSetup(StrictModel):
    """0-D ETD2 collisional transient at frozen Δ.

    ``phonons.mode`` decides whether the phonon population is frozen or
    solved in time. ``thermal_bath`` pins n_ph at the Bose-Einstein seed,
    which is the historical behaviour. The dynamic modes co-evolve it with
    ``f`` by operator splitting: ``f`` advances at frozen n_ph, then n_ph
    advances at the new ``f`` under the exact solution of its affine ODE.

    Drives are constant across the transient. Δ is held fixed in every mode.
    """

    mode: Literal["transient_0d"] = "transient_0d"
    material: MaterialParams = MaterialParams()
    T_bath: Annotated[float, Field(gt=0.0)] = 0.1
    grid: EnergyGrid = EnergyGrid()
    phonons: PhononSector = PhononSector()
    collisions: CollisionTerms = CollisionTerms()
    subgap_drive: SubGapDrive = SubGapDrive()
    pb_drive: PairBreakingDrive = PairBreakingDrive()
    dt: Annotated[float, Field(gt=0.0)] = 0.1  # ETD2 substep (ns)
    total_time: Annotated[float, Field(gt=0.0)] = 120.0  # (ns)
    snapshot_interval: Annotated[float, Field(gt=0.0)] | None = None  # default total/50
    stop_tol: Annotated[float, Field(ge=0.0)] | None = None  # early stop on max|df|/dt
    probe: ProbeConfig = ProbeConfig()

    @model_validator(mode="after")
    def snapshot_interval_not_pathological(self) -> Transient0DSetup:
        _reject_dense_snapshots(self.snapshot_interval, self.total_time)
        return self


class GapStepProfile(StrictModel):
    """Optional two-gap step along the strip.

    ``uniform`` uses the material gap everywhere. ``step`` sets the
    left fraction of the strip to ``gap_left`` and the rest to
    ``gap_right``; a finite ``interface_G_N`` turns the step face into
    a Kupriyanov–Lukichev interface.
    """

    kind: Literal["uniform", "step"] = "uniform"
    gap_left: Annotated[float, Field(gt=0.0)] = 180.0  # (μeV)
    gap_right: Annotated[float, Field(gt=0.0)] = 200.0  # (μeV)
    step_position_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    interface_G_N: Annotated[float, Field(ge=0.0)] | None = None


class InjectionConfig(StrictModel):
    """Continuous QP injection for the 1D strip (Gaussian in energy)."""

    enabled: bool = True
    center_over_delta: Annotated[float, Field(gt=1.0)] = 2.0  # line center (×Δ)
    sigma_over_delta: Annotated[float, Field(gt=0.0)] = 0.1  # line width (×Δ)
    rate_per_ns: Annotated[float, Field(gt=0.0)] = 2e-5  # peak gain (1/ns)
    where: Literal["left_end", "uniform"] = "left_end"


class Spatial1DSetup(StrictModel):
    """1D strip driven to steady state (T3 spatial backend).

    Spatial transport requires a pure-BCS spectral context — the
    builder rejects ``dynes_gamma > 0`` here (the engine does too).
    """

    mode: Literal["spatial_1d"] = "spatial_1d"
    material: MaterialParams = MaterialParams()
    T_bath: Annotated[float, Field(gt=0.0)] = 0.1
    # Gap-edge x_qp and Mattis-Bardeen observables require the first physical
    # cell edge at Delta; a grid starting above Delta silently drops the BCS
    # singular spectral weight.
    # 66 rather than 64: on [Delta, 4*Delta] the two phonon lattices share
    # bins only for a multiple of 3, and 64 gives 42.667. (The 2-D preset's 48
    # is already valid -- 32 exactly.) See EnergyGrid.num_bins.
    grid: EnergyGrid = EnergyGrid(min_factor=1.0, max_factor=4.0, num_bins=66)
    length_um: Annotated[float, Field(gt=0.0)] = 100.0
    num_cells: Annotated[int, Field(ge=2, le=2000)] = 31
    diffusion_model: Literal["A1", "A1P", "A2", "C", "B"] = "A1"
    # D_0 = 0 is the transport off-switch: the flux coefficient is D_0*N_1**q,
    # so zero gives an identically zero operator for every member. There is no
    # OFF member in diffusion_model because the enum value IS the (p, q) pair.
    collisions: CollisionTerms = CollisionTerms()
    gap_profile: GapStepProfile = GapStepProfile()
    injection: InjectionConfig = InjectionConfig()
    dt: Annotated[float, Field(gt=0.0)] = 1.0  # split step (ns); D0*dt/dx^2~5 at defaults
    max_time: Annotated[float, Field(gt=0.0)] = 20000.0  # (ns)
    stop_tol: Annotated[float, Field(ge=0.0)] = 2e-10
    snapshot_interval: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def snapshot_interval_not_pathological(self) -> Spatial1DSetup:
        _reject_dense_snapshots(self.snapshot_interval, self.max_time)
        return self
    # No probe here: strip-resonator response needs a current-weighted
    # treatment (observables.spatial_ac_response) this mode doesn't
    # drive yet — carrying a probe config would validate and render a
    # field with no effect.


class M25DriveConfig(StrictModel):
    """M25 photon drive, calibrated to a target Γ^ph_00 (paper Fig. 3)."""

    omega_nu_GHz: Annotated[float, Field(gt=0.0)] = 119.0  # drive frequency ω_ν/2π (GHz)
    Gamma_ph_00_Hz: Annotated[float, Field(gt=0.0)] = 300.0  # calibration target (Hz)
    nu_0_per_J_per_m3: Annotated[float, Field(gt=0.0)] = 0.73e47  # DOS at E_F
    volume_m3: Annotated[float, Field(gt=0.0)] = 506e-6 * 240e-6 * 0.028e-6


class M25JunctionSetup(StrictModel):
    """M25 gap-asymmetric junction moment layer over a T sweep.

    Inputs in the paper's natural units (GHz ÷ h for energies, Hz for
    rates); the builder converts to the Kelvin/Hz convention of
    :mod:`qpsim.services.rate_equation_coefficients`. Defaults follow
    the Fig. 3 reproduction in ``validation/marchegiani_2025``.
    """

    mode: Literal["m25_junction"] = "m25_junction"
    Delta_R_over_h_GHz: Annotated[float, Field(gt=0.0)] = 49.0
    omega_LR_over_h_GHz: Annotated[float, Field(gt=0.0)] = 5.0  # gap asymmetry Δ_L − Δ_R
    omega_10_over_h_GHz: Annotated[float, Field(gt=0.0)] = 5.5
    E_J_over_h_GHz: Annotated[float, Field(gt=0.0)] = 14.5
    E_C_over_h_GHz: Annotated[float, Field(gt=0.0)] = 0.290
    r_L_Hz: Annotated[float, Field(gt=0.0)] = 6.25e6
    r_Rlt_Hz: Annotated[float, Field(gt=0.0)] = 6.25e6
    Gamma_ee_10_Hz: Annotated[float, Field(ge=0.0)] = 100e3
    drive: M25DriveConfig = M25DriveConfig()
    T_start_mK: Annotated[float, Field(gt=0.0)] = 10.0
    T_stop_mK: Annotated[float, Field(gt=0.0)] = 150.0
    T_points: Annotated[int, Field(ge=1, le=500)] = 29
    # "max_x_L" is DEPRECATED (selects pseudo-roots; see
    # docs/REVIEW-2026-07-04-deep-review.md finding 10). It stays in the
    # Literal so previously saved setups still validate, but it is no
    # longer offered in the UI dropdown and the engine emits a
    # DeprecationWarning when it runs.
    branch_picker_mode: Literal["max_x_L", "min_residual", "lock_to_preferred"] = (
        "lock_to_preferred"
    )


class GeometrySource(StrictModel):
    """Where the 2-D mask comes from.

    ``rectangle`` needs nothing but its extent. ``gds`` rasterises one layer
    of a layout file; that needs the optional ``gdstk`` package, and the
    builder says so rather than failing on an import.

    The mask decides the dimensionality: ``rows = 1`` is a 1-D strip and
    ``rows = cols = 1`` is a single 0-D cell, both solved by the same core.
    """

    kind: Literal["rectangle", "gds"] = "rectangle"
    rows: Annotated[int, Field(ge=1, le=512)] = 24
    cols: Annotated[int, Field(ge=1, le=512)] = 24
    mesh_size_um: Annotated[float, Field(gt=0.0)] = 4.0
    gds_path: str | None = None
    gds_layer: Annotated[int, Field(ge=0)] = 0
    # A rasterised layout that lands in more than one piece is nearly always
    # a stray polygon or too coarse a mesh, not a real device.
    require_connected: bool = True


BoundaryKind = Literal[
    "reflective", "absorbing", "dirichlet", "neumann", "robin"
]


class EdgeCondition(StrictModel):
    """One boundary condition, for one edge.

    ``robin`` is the physically interesting one and the reason this exists:
    ``∂ₙφ + βφ = γ`` is the finite-transparency interface, the lossy contact,
    the trap with a finite escape rate. ``absorbing`` is its β → ∞ limit and
    ``reflective`` its β = 0 limit, so without it the whole continuum between
    "nothing leaves" and "everything leaves" — which is where every real
    contact sits — cannot be stated.
    """

    kind: BoundaryKind = "reflective"
    value: float = 0.0          # dirichlet/neumann value, or robin β
    aux_value: float | None = None   # robin γ


class EdgeConditions(StrictModel):
    """Boundary conditions on the device rim.

    ``kind``/``value``/``aux_value`` set the default for every edge, and
    ``per_edge`` overrides individual ones by id. For a rectangular mask the
    ids are ``up``, ``down``, ``left`` and ``right``.

    Per-edge assignment is what makes a *device* rather than a slab: an
    absorbing normal-metal trap on one end with a reflective rim elsewhere is
    the standard quasiparticle-trapping experiment. It also removes a trap in
    the uniform form — on a one-cell-wide strip, a rim-wide "absorbing" also
    absorbs through the long sides, so the 1-D reduction silently leaked.
    """

    kind: BoundaryKind = "reflective"
    value: float = 0.0
    aux_value: float | None = None
    per_edge: dict[str, EdgeCondition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def robin_needs_a_coefficient(self) -> EdgeConditions:
        for label, spec in [("boundary", self), *self.per_edge.items()]:
            if spec.kind == "robin" and spec.aux_value is None:
                raise ValueError(
                    f"A robin condition on {label!r} needs aux_value (γ in "
                    "∂ₙφ + βφ = γ); `value` is β."
                )
        return self


class GapRegions(StrictModel):
    """Local gap across the mask.

    ``uniform`` uses the material gap everywhere. ``column_step`` puts
    ``gap_left`` on the columns before ``step_fraction`` and ``gap_right``
    after, which is the 2-D reading of the 1-D strip's step: in 2-D the
    boundary between the two is a curve of faces rather than one face, and a
    finite ``interface_G_N`` turns every face along it into a
    Kupriyanov-Lukichev barrier.
    """

    kind: Literal["uniform", "column_step", "expression"] = "uniform"
    gap_left: Annotated[float, Field(gt=0.0)] = 180.0
    gap_right: Annotated[float, Field(gt=0.0)] = 180.0
    step_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.5
    interface_G_N: Annotated[float, Field(ge=0.0)] | None = None
    # A prescribed map in micro-eV over the cells. Variables in scope:
    # x, y (normalised cell centres), x_um, y_um, gap (the material value),
    # params. A step is one gap profile; a proximitised finger, a radial
    # well, a gradient used as a trap, or a map computed elsewhere are not
    # reachable from any finite set of presets.
    expression: str | None = None
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def expression_present_when_selected(self) -> GapRegions:
        if self.kind == "expression" and not self.expression:
            raise ValueError("kind='expression' needs an expression.")
        return self


class Injection2D(StrictModel):
    """Continuous Gaussian-in-energy quasiparticle source."""

    enabled: bool = False
    center_over_delta: Annotated[float, Field(gt=1.0)] = 2.0
    sigma_over_delta: Annotated[float, Field(gt=0.0)] = 0.1
    rate_per_ns: Annotated[float, Field(gt=0.0)] = 2e-5
    where: Literal["left_edge", "uniform", "centre_cell"] = "left_edge"


class KineticsSetup(StrictModel):
    """Quasiparticle kinetics on a geometry, driven to steady state.

    Deliberately NOT named for a dimensionality. The geometry is a mask, so a
    single cell is 0-D, a one-cell-wide mask is a 1-D strip, and anything else
    is 2-D — configurations of one model rather than separate ones. The former
    name ``spatial_2d`` described the mask's maximum rank rather than the
    physics, and stopped being true the moment the 0-D and 1-D modes reduced
    onto this one.

    Spatial transport requires a pure-BCS spectral context; the builder
    rejects ``dynes_gamma > 0`` here, as the engine does.
    """

    mode: Literal["kinetics"] = "kinetics"
    material: MaterialParams = MaterialParams()
    T_bath: Annotated[float, Field(gt=0.0)] = 0.1
    grid: EnergyGrid = EnergyGrid(min_factor=1.0, max_factor=4.0, num_bins=48)
    geometry: GeometrySource = GeometrySource()
    boundary: EdgeConditions = EdgeConditions()
    diffusion_model: Literal["A1", "A1P", "A2", "C", "B"] = "A1"
    collisions: CollisionTerms = CollisionTerms()
    gap_regions: GapRegions = GapRegions()
    injection: Injection2D = Injection2D()
    initial: InitialCondition = InitialCondition()
    # Any number of prescribed drives, summed. A device can be under a steady
    # bias and a pulse at once, and that is two drives rather than one harder
    # expression. `injection` above is the older, narrower knob and still
    # works; these are the general form.
    drives: list[DriveSpec] = Field(default_factory=list)
    subgap_drive: SubGapDrive = SubGapDrive()
    pb_drive: PairBreakingDrive = PairBreakingDrive()
    phonons: PhononSector = PhononSector()
    self_consistent_gap: bool = False
    # Gap quantum as a fraction of the energy-grid spacing. Collisions group
    # by exact gap, so a continuous profile gives one group per cell; snapping
    # bounds that. Measured cost: the kernel moves about 2e-2 per FULL grid
    # spacing of gap shift and smoothly, so a tenth costs ~2e-3. Do not push
    # past ~0.5: crossing a cell edge moves it 3.3e-1, because a bin enters or
    # leaves the above-gap support.
    gap_quantum_over_dE: Annotated[float, Field(gt=0.0, le=1.0)] = 0.1
    dt: Annotated[float, Field(gt=0.0)] = 1.0
    max_time: Annotated[float, Field(gt=0.0)] = 5000.0
    stop_tol: Annotated[float, Field(ge=0.0)] = 2e-10
    # Cadence for recording the evolving field (ns). ``None`` keeps only the
    # final state, which is the right default for a steady-state search and
    # the wrong one for every dynamical question: without frames, "has it
    # settled" cannot be distinguished from "is still drifting". Each frame is
    # a full (NE, Ncells) field plus the phonon map, so the cost is real and
    # the choice is the user's.
    snapshot_interval: Annotated[float, Field(gt=0.0)] | None = None
    # WHICH engine entry point runs -- deliberately not inside SolverOptions.
    # `solver.method` there selects the root-find ALGORITHM; this selects
    # whether a root-find happens at all. It also decides which of the two
    # knob groups is live (dt/max_time/stop_tol, or solver.*), so it sits
    # above both rather than inside one of them.
    #
    # "steady_state" routes to T3DiffusionBackend.steady_state, whose state is
    # f:(NE,) with a SCALAR gap -- it has no cell axis, so it requires a
    # one-cell mask. That is a property of the solver, not of the geometry:
    # multi-cell devices reach a steady state perfectly well by time-marching
    # to stop_tol, which is what "time_march" does.
    strategy: Literal["time_march", "steady_state"] = "time_march"
    solver: SolverOptions = SolverOptions()
    probe: ProbeConfig = ProbeConfig()

    @model_validator(mode="after")
    def snapshot_interval_not_pathological(self) -> KineticsSetup:
        _reject_dense_snapshots(self.snapshot_interval, self.max_time)
        return self

    @model_validator(mode="after")
    def one_authority_for_the_gap_switch(self) -> KineticsSetup:
        """``self_consistent_gap`` is the top-level field, and only that one.

        SolverOptions carries a field of the same name for the 0-D mode this
        one is replacing. Two settable fields meaning the same thing is the
        two-authorities defect that keeps producing an interface which claims
        one thing while the engine does another, so the duplicate is refused
        out loud rather than silently losing to whichever is read last.
        Transitional: it disappears with the 0-D mode.
        """
        if self.solver.self_consistent_gap:
            raise ValueError(
                "Set self_consistent_gap at the top level of the setup, not on "
                "solver. The top-level field is what the term panel's 'gapeq' "
                "switch and the time-march path both read; solver."
                "self_consistent_gap belongs to the 0-D mode this one replaces "
                "and is ignored here."
            )
        return self


AnySetup = (
    SteadyState0DSetup
    | Transient0DSetup
    | Spatial1DSetup
    | KineticsSetup
    | M25JunctionSetup
)


# Retired mode names, and what each is now. A stored setup and a run manifest
# both name their mode as a plain string, so every one already on disk still
# says the old name -- without this map, renaming a mode does not date saved
# work, it makes it UNLOADABLE. Entries are permanent: the cost of keeping one
# is a dict line, and the cost of dropping one is somebody's saved device.
LEGACY_MODE_ALIASES: dict[str, str] = {
    "spatial_2d": "kinetics",
}


def canonical_mode(mode: str) -> str:
    """The current name for a possibly-retired mode string.

    Used everywhere a BARE mode string arrives rather than a whole setup --
    ``/api/defaults/{mode}``, manifest listings — because those never pass
    through :class:`SetupEnvelope` and would otherwise 404 on a name the
    envelope accepts happily.
    """
    return LEGACY_MODE_ALIASES.get(mode, mode)


class SetupEnvelope(StrictModel):
    """A named, persistable setup."""

    name: str = "Untitled setup"
    setup: AnySetup = Field(discriminator="mode")

    @field_validator("setup", mode="before")
    @classmethod
    def _upgrade_retired_mode(cls, value: object) -> object:
        """Rewrite a retired mode name before the discriminator sees it.

        Has to be ``mode="before"``: the union is discriminated ON ``mode``, so
        by the time a validator could run on the parsed model the parse has
        already failed. Copies rather than mutating -- the caller's dict is
        often the raw request body or a just-read JSON file, and rewriting it
        in place would make the on-disk name depend on whether anything
        happened to load it.
        """
        if isinstance(value, dict):
            mode = value.get("mode")
            if isinstance(mode, str) and mode in LEGACY_MODE_ALIASES:
                return {**value, "mode": LEGACY_MODE_ALIASES[mode]}
        return value
    # Name of an analytic benchmark to check this run against
    # (:mod:`qpsim.webui.benchmarks`). It lives on the envelope rather than
    # inside the setup because it is an assertion ABOUT the run, not part of
    # the physics being solved: the same setup is the same physics whether or
    # not anybody checks it against a closed form. Keeping it out of the setup
    # models also keeps the solved result independent of whether a check was
    # requested. Unknown names are reported as a note, not an error, so a run
    # is never lost to a mistyped check.
    benchmark: str | None = None


MODE_LABELS: dict[str, str] = {
    "steady_state_0d": "0-D steady state",
    "transient_0d": "0-D transient",
    "spatial_1d": "1D strip",
    "kinetics": "Kinetics (any geometry)",
    "m25_junction": "M25 junction",
}

MODE_CLASSES: dict[
    str,
    type[
        SteadyState0DSetup
        | Transient0DSetup
        | Spatial1DSetup
        | KineticsSetup
        | M25JunctionSetup
    ],
] = {
    "steady_state_0d": SteadyState0DSetup,
    "transient_0d": Transient0DSetup,
    "spatial_1d": Spatial1DSetup,
    "kinetics": KineticsSetup,
    "m25_junction": M25JunctionSetup,
}
