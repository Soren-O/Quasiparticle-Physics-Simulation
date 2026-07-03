"""Setup → engine-object construction and cross-field validation.

Pure functions from the pydantic setup models to the engine's state
dataclasses, plus :func:`validate_setup`, which reports every
cross-field physics problem the static schema can't see (drive
frequencies vs 2Δ, Dynes × spatial transport, grid commensurability,
solver-route conflicts) with messages that reference derived
quantities like the actual grid spacing.

No heavy UI imports here — this module depends only on the core
engine and :mod:`qpsim.webui.schemas`, so the builder logic is
testable without the ``ui`` extra's server stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.backends.t3_spatial_1d import T3Spatial1DState, T3SpatialFlux1D
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.rate_equation_coefficients import M25PhotonDrive, M25PhysicalParameters
from qpsim.transport.diffusion.base import from_name as diffusion_model_from_name
from qpsim.webui.schemas import (
    AnySetup,
    M25JunctionSetup,
    MaterialParams,
    Spatial1DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)

# Kelvin per Hz (h/k_B) — the M25 layer's GHz→Kelvin conversion.
H_OVER_KB_K_PER_HZ = 4.799243e-11

# Photon partners land on bins i ± round(ω/dE); beyond this fractional
# mismatch the engine warns and snaps (see qpsim.collisions.sub_gap_photon).
COMMENSURATE_TOL = 0.01


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_setup`."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def material_from_params(mp: MaterialParams) -> Material:
    """Engine Material from the editable frontend copy."""
    return Material(
        name=mp.name,
        Delta_0=mp.Delta_0,
        T_c=mp.T_c,
        tau_0=mp.tau_0,
        tau_0_pb_ns=mp.tau_0_pb_ns,
        D_0=mp.D_0,
        rho_F=mp.rho_F,
    )


def fermi_dirac(E: np.ndarray, T_bath: float) -> np.ndarray:
    """Thermal seed occupation (clipped exponent, house convention)."""
    kT = KB_UEV_PER_K * T_bath
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _grid_spacing(setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup) -> float:
    span = (setup.grid.max_factor - setup.grid.min_factor) * setup.material.Delta_0
    return span / float(setup.grid.num_bins)


def _check_photon_commensurate(
    report: ValidationReport, label: str, omega: float, dE: float
) -> None:
    m = round(omega / dE)
    if m <= 0:
        report.errors.append(
            f"{label}: photon energy {omega:g} μeV is below one grid spacing "
            f"(dE = {dE:.4g} μeV) — increase the photon energy or the number of bins."
        )
        return
    frac_err = abs(omega - m * dE) / dE
    if frac_err > COMMENSURATE_TOL:
        report.warnings.append(
            f"{label}: {omega:g} μeV is not grid-commensurate (dE = {dE:.4g} μeV, "
            f"fractional error {frac_err:.3f}); the engine will snap it to "
            f"{m * dE:.4g} μeV. Nearest commensurate choices: ω = {m * dE:.4g} μeV, "
            f"or adjust the bin count."
        )


def _validate_drives_and_probe(
    report: ValidationReport,
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup,
) -> None:
    gap = setup.material.Delta_0
    dE = _grid_spacing(setup)

    subgap = getattr(setup, "subgap_drive", None)
    if subgap is not None and subgap.enabled:
        if subgap.omega_0 >= 2.0 * gap:
            report.errors.append(
                f"Sub-gap drive: ω₀ = {subgap.omega_0:g} μeV must be < 2Δ = {2 * gap:g} μeV "
                f"(use the pair-breaking drive above 2Δ)."
            )
        _check_photon_commensurate(report, "Sub-gap drive", subgap.omega_0, dE)

    pb = getattr(setup, "pb_drive", None)
    if pb is not None and pb.enabled:
        if pb.omega_PB <= 2.0 * gap:
            report.errors.append(
                f"Pair-breaking drive: ω_PB = {pb.omega_PB:g} μeV must be > 2Δ = {2 * gap:g} μeV."
            )
        _check_photon_commensurate(report, "Pair-breaking drive", pb.omega_PB, dE)

    if setup.probe.enabled:
        if setup.probe.omega_0 >= gap:
            report.errors.append(
                f"Probe: Mattis–Bardeen observables need a sub-gap probe, "
                f"ω₀ = {setup.probe.omega_0:g} μeV ≥ Δ = {gap:g} μeV."
            )
        if setup.material.dynes_gamma > 0.0:
            report.warnings.append(
                "Probe: Mattis–Bardeen σ₁/σ₂ (and Q_i, δω/ω) require a pure-BCS "
                "spectral context — they will be skipped because Dynes Γ > 0."
            )

    if setup.T_bath >= setup.material.T_c:
        report.errors.append(
            f"T_bath = {setup.T_bath:g} K must be below T_c = {setup.material.T_c:g} K."
        )
    elif setup.T_bath > 0.5 * setup.material.T_c:
        report.warnings.append(
            f"T_bath = {setup.T_bath:g} K is above T_c/2 — strong thermal gap "
            f"suppression; the fixed-gap kinetics are increasingly approximate there."
        )

    if setup.grid.num_bins > 2500:
        report.warnings.append(
            f"{setup.grid.num_bins} energy bins builds ~{setup.grid.num_bins}² collision "
            f"kernels — expect slow solves and high memory."
        )


def validate_setup(setup: AnySetup) -> ValidationReport:
    """Cross-field physics validation (schema-level checks already passed)."""
    report = ValidationReport()

    if isinstance(setup, SteadyState0DSetup):
        _validate_drives_and_probe(report, setup)
        if setup.solver.method == "coupled_newton" and setup.phonons.mode == "thermal_bath":
            report.errors.append(
                "Solver: coupled-Newton solves (f, n_ph) jointly and cannot be combined "
                "with the pinned thermal bath — pick a dynamic phonon sector or the "
                "auto/picard route."
            )
        if (
            setup.phonons.use_phonon_side_kernel
            and setup.material.tau_0_pb_ns is None
        ):
            report.errors.append(
                "Phonons: the phonon-side kernel needs the material's τ₀^PB (tau_0_pb_ns)."
            )

    elif isinstance(setup, Transient0DSetup):
        _validate_drives_and_probe(report, setup)
        if setup.dt > setup.material.tau_0 / 10.0:
            report.warnings.append(
                f"dt = {setup.dt:g} ns exceeds τ₀/10 = {setup.material.tau_0 / 10:g} ns — "
                f"ETD2 stability limits may distort the transient."
            )
        n_steps = setup.total_time / setup.dt
        if n_steps > 2e5:
            report.warnings.append(
                f"total_time/dt ≈ {n_steps:.3g} substeps — this run will take a while."
            )

    elif isinstance(setup, Spatial1DSetup):
        _validate_drives_and_probe(report, setup)
        if setup.material.dynes_gamma > 0.0:
            report.errors.append(
                "Spatial transport requires a pure-BCS spectral context: Dynes Γ must be 0 "
                "for the 1D strip (Dynes with 0-D collisions is fine)."
            )
        if setup.material.D_0 <= 0.0:
            report.errors.append("1D strip: the material needs a positive D₀ (μm²/ns).")
        if setup.injection.enabled:
            e_center = setup.injection.center_over_delta
            if not (setup.grid.min_factor < e_center < setup.grid.max_factor):
                report.errors.append(
                    f"Injection: line center {e_center:g}×Δ lies outside the energy grid "
                    f"[{setup.grid.min_factor:g}, {setup.grid.max_factor:g}]×Δ."
                )
        if setup.gap_profile.kind == "step":
            e_max = setup.grid.max_factor * setup.material.Delta_0
            for side, val in (
                ("gap_left", setup.gap_profile.gap_left),
                ("gap_right", setup.gap_profile.gap_right),
            ):
                if val >= e_max:
                    report.errors.append(
                        f"Gap profile: {side} = {val:g} μeV is at or above the grid top "
                        f"{e_max:g} μeV — no quasiparticle states would exist there."
                    )

    elif isinstance(setup, M25JunctionSetup):
        if setup.E_J_over_h_GHz <= setup.E_C_over_h_GHz:
            report.errors.append("M25: requires E_J > E_C (transmon regime).")
        if setup.T_stop_mK < setup.T_start_mK:
            report.errors.append("M25: T_stop must be ≥ T_start.")
        if setup.omega_10_over_h_GHz >= 2.0 * setup.Delta_R_over_h_GHz:
            report.errors.append("M25: needs ω₁₀ < 2Δ_R (no direct pair-breaking by the qubit).")

    return report


def build_spectral(
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup,
) -> SpectralContext:
    """Spectral context on the setup's uniform energy grid."""
    E, _ = build_energy_grid(
        gap=setup.material.Delta_0,
        energy_min_factor=setup.grid.min_factor,
        energy_max_factor=setup.grid.max_factor,
        num_energy_bins=setup.grid.num_bins,
    )
    return SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=setup.material.Delta_0,
        dynes_gamma=setup.material.dynes_gamma,
        diffusion_coefficient=setup.material.D_0,
    )


def build_state_0d(
    setup: SteadyState0DSetup | Transient0DSetup,
) -> T3DiffusionState:
    """Thermal-seed 0-D T3 state on the physics ω-grid."""
    spectral = build_spectral(setup)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)

    if isinstance(setup, SteadyState0DSetup) and setup.phonons.mode == "dynamic_escape":
        tau_l_value = setup.phonons.tau_l_ns
    else:
        # thermal_bath (value unused on the Newton path), dynamic_closed
        # (0.0 is the engine's no-substrate τ_l → ∞ sentinel), and all
        # transients (n_ph frozen at the thermal seed).
        tau_l_value = 0.0

    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, setup.T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_value),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=fermi_dirac(spectral.E, setup.T_bath),
        gap=setup.material.Delta_0,
        spectral=spectral,
        phonon=phonon,
        material=material_from_params(setup.material),
        T_bath=setup.T_bath,
    )


def drive_dicts(
    setup: SteadyState0DSetup | Transient0DSetup,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """(photon_params, pb_photon_params) in the backend's dict format."""
    photon_params = None
    if setup.subgap_drive.enabled:
        photon_params = {
            "omega_0": setup.subgap_drive.omega_0,
            "n_bar": setup.subgap_drive.n_bar,
            "c_phot": setup.subgap_drive.c_phot,
        }
    pb_params = None
    if setup.pb_drive.enabled:
        pb_params = {
            "omega_PB": setup.pb_drive.omega_PB,
            "n_bar_PB": setup.pb_drive.n_bar_PB,
            "c_phot_PB": setup.pb_drive.c_phot_PB,
        }
    return photon_params, pb_params


def steady_state_solver_kwargs(setup: SteadyState0DSetup) -> dict[str, object]:
    """Backend ``steady_state`` kwargs for the chosen phonon sector + method."""
    s = setup.solver
    kwargs: dict[str, object] = {
        "self_consistent_gap": s.self_consistent_gap,
        "use_phonon_side_kernel": setup.phonons.use_phonon_side_kernel,
        "newton_tol": s.newton_tol,
        "newton_max_iter": s.newton_max_iter,
    }
    if setup.phonons.mode == "thermal_bath":
        # Newton path with n_ph pinned at the bath (validate_setup has
        # already rejected an explicit coupled_newton request here).
        kwargs["use_thermal_phonons"] = True
        return kwargs

    method = "picard" if s.method == "auto" else s.method
    kwargs["method"] = method
    kwargs["picard_tol"] = s.picard_tol
    kwargs["picard_max_iter"] = s.picard_max_iter
    kwargs["picard_mixing"] = s.picard_mixing
    kwargs["anderson_depth"] = s.anderson_depth
    return kwargs


def build_state_1d(setup: Spatial1DSetup) -> T3Spatial1DState:
    """Thermal-seed 1D strip state with optional two-gap step profile."""
    spectral = build_spectral(setup)
    x = np.linspace(0.0, setup.length_um, setup.num_cells)
    f_col = fermi_dirac(spectral.E, setup.T_bath)
    f = np.tile(f_col[:, None], (1, setup.num_cells))

    gap_profile: np.ndarray | None = None
    interface_conductance: float | None = None
    if setup.gap_profile.kind == "step":
        split = setup.gap_profile.step_position_fraction * setup.length_um
        gap_profile = np.where(x < split, setup.gap_profile.gap_left, setup.gap_profile.gap_right)
        interface_conductance = setup.gap_profile.interface_G_N

    return T3Spatial1DState(
        f=f,
        x=x,
        gap=setup.material.Delta_0,
        spectral=spectral,
        material=material_from_params(setup.material),
        T_bath=setup.T_bath,
        diffusion_model=diffusion_model_from_name(setup.diffusion_model),
        gap_profile=gap_profile,
        interface_conductance=interface_conductance,
    )


def build_injection_flux(
    setup: Spatial1DSetup, state: T3Spatial1DState
) -> T3SpatialFlux1D | None:
    """Gaussian-in-energy continuous QP source for the strip."""
    if not setup.injection.enabled:
        return None
    E = state.spectral.E
    gap = setup.material.Delta_0
    center = setup.injection.center_over_delta * gap
    sigma = setup.injection.sigma_over_delta * gap
    line = np.exp(-0.5 * ((E - center) / sigma) ** 2)

    NE, NX = state.f.shape
    gain = np.zeros((NE, NX))
    if setup.injection.where == "left_end":
        gain[:, 0] = setup.injection.rate_per_ns * line
    else:
        gain[:, :] = setup.injection.rate_per_ns * line[:, None]
    return T3SpatialFlux1D(
        gain=gain,
        loss_rate=np.zeros((NE, NX)),
        diagnostics={
            "source": "webui_gaussian_injection",
            "center_uev": center,
            "sigma_uev": sigma,
        },
    )


def build_m25_inputs(
    setup: M25JunctionSetup, T_kelvin: float
) -> tuple[M25PhysicalParameters, M25PhotonDrive]:
    """M25 physical parameters + photon drive at one temperature.

    Follows the Fig. 3 reproduction's conventions: R_T from E_J via
    ``8 E_J (Δ̄/Δ_L)``, drive scale left at 1.0 for the caller to
    calibrate against the Γ^ph_00 target.
    """
    ghz_to_K = 1e9 * H_OVER_KB_K_PER_HZ
    Delta_L_GHz = setup.Delta_R_over_h_GHz + setup.omega_LR_over_h_GHz
    params = M25PhysicalParameters(
        Delta_L_kelvin=Delta_L_GHz * ghz_to_K,
        Delta_R_kelvin=setup.Delta_R_over_h_GHz * ghz_to_K,
        omega_10_kelvin=setup.omega_10_over_h_GHz * ghz_to_K,
        T_kelvin=T_kelvin,
        E_J_kelvin=setup.E_J_over_h_GHz * ghz_to_K,
        E_C_kelvin=setup.E_C_over_h_GHz * ghz_to_K,
        R_T_Hz=8.0
        * setup.E_J_over_h_GHz
        * 1e9
        * ((Delta_L_GHz + setup.Delta_R_over_h_GHz) / 2.0 / Delta_L_GHz),
        r_L_Hz=setup.r_L_Hz,
        r_Rlt_Hz=setup.r_Rlt_Hz,
        Gamma_ee_10_Hz=setup.Gamma_ee_10_Hz,
    )
    drive = M25PhotonDrive(
        omega_nu_kelvin=setup.drive.omega_nu_GHz * ghz_to_K,
        Gamma_nu_scale_Hz=1.0,
        nu_0_per_J_per_m3=setup.drive.nu_0_per_J_per_m3,
        volume_m3=setup.drive.volume_m3,
    )
    return params, drive
