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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qpsim.materials import load_material
from qpsim.materials.database import validate_rho_F_eV

# Default material values come straight from the YAML database so the
# frontend never carries a second, drifting copy of Al's parameters.
_AL = load_material("Al")

# Upper bound on emitted snapshots. A ``snapshot_interval`` far below the
# integration step would otherwise drain an unbounded inner cadence loop
# before cancellation is ever polled (single-worker runner → memory blow-up
# and an uninterruptible job). Reject such setups at validation time.
_MAX_SNAPSHOTS = 100_000


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
    num_bins: Annotated[int, Field(ge=8, le=5000)] = 400

    @model_validator(mode="after")
    def max_must_exceed_min(self) -> EnergyGrid:
        if self.max_factor <= self.min_factor:
            raise ValueError("max_factor must be greater than min_factor")
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
    grid: EnergyGrid = EnergyGrid(min_factor=1.0, max_factor=4.0, num_bins=64)
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


AnySetup = SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | M25JunctionSetup


class SetupEnvelope(StrictModel):
    """A named, persistable setup."""

    name: str = "Untitled setup"
    setup: AnySetup = Field(discriminator="mode")


MODE_LABELS: dict[str, str] = {
    "steady_state_0d": "0-D steady state",
    "transient_0d": "0-D transient",
    "spatial_1d": "1D strip",
    "m25_junction": "M25 junction",
}

MODE_CLASSES: dict[
    str, type[SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | M25JunctionSetup]
] = {
    "steady_state_0d": SteadyState0DSetup,
    "transient_0d": Transient0DSetup,
    "spatial_1d": Spatial1DSetup,
    "m25_junction": M25JunctionSetup,
}
