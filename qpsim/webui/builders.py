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
from qpsim.backends.t3_spatial import T3SpatialState
from qpsim.backends.t3_spatial_1d import T3Spatial1DState, T3SpatialFlux1D
from qpsim.collisions.pair_breaking_photon import (
    validate_pair_breaking_photon_grid,
)
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.collisions.sub_gap_photon import COMMENSURATE_TOL
from qpsim.constants import H_OVER_KB_K_PER_HZ
from qpsim.geometries import Geometry, from_gds, gds_support_available, rectangle
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.materials.database import Material
from qpsim.observables import fermi_dirac_distribution
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.rate_equation_coefficients import M25PhotonDrive, M25PhysicalParameters
from qpsim.transport.diffusion.base import DiffusionModel
from qpsim.transport.diffusion.base import from_name as diffusion_model_from_name
from qpsim.webui.schemas import (
    AnySetup,
    M25JunctionSetup,
    MaterialParams,
    ProbeConfig,
    Spatial1DSetup,
    Spatial2DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)


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


def mb_probe_invalid_reason(
    probe: ProbeConfig, dynes_gamma: float, gap: float
) -> str | None:
    """Why Mattis–Bardeen probe observables can't run, or ``None`` if valid.

    One author for the guard that the engine enforces in
    ``observables.ac_conductivity`` — used by setup validation and by
    both executors that skip the probe with a note.
    """
    if not probe.enabled:
        return "probe disabled"
    if probe.omega_0 >= gap:
        return (
            f"Mattis–Bardeen observables need a sub-gap probe: "
            f"ω₀ = {probe.omega_0:g} μeV is not below Δ = {gap:g} μeV."
        )
    if dynes_gamma > 0.0:
        return (
            "Mattis–Bardeen σ₁/σ₂ (and Q_i, δω/ω) require a pure-BCS "
            "spectral context — skipped because Dynes Γ > 0."
        )
    return None


def _grid_spacing(
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | Spatial2DSetup,
) -> float:
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
        report.errors.append(
            f"{label}: {omega:g} μeV is not grid-commensurate (dE = {dE:.4g} μeV, "
            f"fractional error {frac_err:.3f}); nearest commensurate choice: "
            f"ω = {m * dE:.4g} μeV, "
            f"or adjust the bin count."
        )


def _check_pb_runtime_contract(
    report: ValidationReport,
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | Spatial2DSetup,
) -> None:
    """Run the production PB kernel's static grid-contract preflight.

    The shared helper is ``O(NE)`` and avoids constructing a spectral context
    with dense ``NE × NE`` coherence matrices merely to validate a request.
    Do not duplicate its roundoff-sensitive finite-volume predicates here.
    """
    pb = getattr(setup, "pb_drive", None)
    if pb is None or not pb.enabled or setup.material.dynes_gamma > 0.0:
        return
    E, dE = build_energy_grid(
        gap=setup.material.Delta_0,
        energy_min_factor=setup.grid.min_factor,
        energy_max_factor=setup.grid.max_factor,
        num_energy_bins=setup.grid.num_bins,
    )
    try:
        contract = validate_pair_breaking_photon_grid(
            E,
            np.full_like(E, dE),
            setup.material.Delta_0,
            pb.omega_PB,
        )
    except ValueError as exc:
        report.errors.append(f"Pair-breaking drive: {exc}")
        return
    if contract.fractional_error > 1e-6:
        report.warnings.append(
            "Pair-breaking drive: nominal ω_PB snaps to "
            f"{contract.snapped_omega:.6g} μeV "
            f"({contract.fractional_error:.2e} bins); evaluate any "
            "energy-dependent photon occupancy at the snapped energy."
        )


def _validate_drives_and_probe(
    report: ValidationReport,
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | Spatial2DSetup,
) -> None:
    gap = setup.material.Delta_0
    dE = _grid_spacing(setup)

    if setup.material.dynes_gamma > 0.0:
        report.errors.append(
            "Kinetic collision solves require a pure-BCS spectral context: "
            "Dynes Γ must be 0 for both 0-D and spatial modes because the "
            "collision kernels do not implement Dynes broadening."
        )

    if setup.material.dynes_gamma == 0.0 and setup.grid.min_factor > 1.0:
        report.errors.append(
            "Grid: pure-BCS x_qp and Mattis-Bardeen observables require "
            "grid.min_factor <= 1 so the first finite-volume cell covers "
            "the gap edge. Starting above Delta drops singular spectral "
            "support that cannot be reconstructed from the sampled f(E)."
        )

    subgap = getattr(setup, "subgap_drive", None)
    if subgap is not None and subgap.enabled and subgap.c_phot > 0.0:
        if subgap.omega_0 >= 2.0 * gap:
            report.errors.append(
                f"Sub-gap drive: ω₀ = {subgap.omega_0:g} μeV must be < 2Δ = {2 * gap:g} μeV "
                f"(use the pair-breaking drive above 2Δ)."
            )
        _check_photon_commensurate(report, "Sub-gap drive", subgap.omega_0, dE)

    pb = getattr(setup, "pb_drive", None)
    if pb is not None and pb.enabled and pb.c_phot_PB > 0.0:
        if pb.omega_PB <= 2.0 * gap:
            report.errors.append(
                f"Pair-breaking drive: ω_PB = {pb.omega_PB:g} μeV must be > 2Δ = {2 * gap:g} μeV."
            )
        _check_pb_runtime_contract(report, setup)

    probe = getattr(setup, "probe", None)
    if probe is not None and probe.enabled:
        reason = mb_probe_invalid_reason(probe, setup.material.dynes_gamma, gap)
        if reason is not None:
            if probe.omega_0 >= gap:
                report.errors.append(f"Probe: {reason}")
            else:
                report.warnings.append(f"Probe: {reason}")

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

        c = setup.collisions
        if setup.phonons.mode == "thermal_bath":
            # No phonon equation exists in this sector: n is pinned at the bath.
            # The phonon-source flags have no term to act on, so they are inert
            # rather than wrong, and a channel switched off on the quasiparticle
            # side alone is not a split.
            if not (c.phonon_scattering_source and c.phonon_recombination_source):
                report.warnings.append(
                    "Collision terms: the phonon source switches have no effect with a "
                    "thermal-bath phonon sector, because n_ph is pinned and there is no "
                    "phonon equation to remove a term from."
                )
        else:
            sc_split = c.scattering != c.phonon_scattering_source
            rc_split = c.recombination != c.phonon_recombination_source
            if sc_split or rc_split:
                channels = " and ".join(
                    name for name, split in
                    (("scattering", sc_split), ("recombination", rc_split)) if split
                )
                if not setup.phonons.use_phonon_side_kernel:
                    report.errors.append(
                        f"Collision terms: the {channels} channel cannot be split while "
                        "use_phonon_side_kernel is off, because the phonon equation then "
                        "reuses the quasiparticle-side kernel and the two sides are the "
                        "same matrix. Enable the phonon-side kernel, or set both sides "
                        "of the channel the same."
                    )
                elif setup.solver.method == "coupled_newton":
                    report.errors.append(
                        f"Collision terms: the {channels} channel cannot be split on the "
                        "coupled-Newton route, which assembles the phonon source and its "
                        "Jacobian through a path that does not carry these flags. Use the "
                        "Picard solver, or set both sides of the channel the same."
                    )
                else:
                    report.warnings.append(
                        f"Energy conservation is not being tracked: the {channels} "
                        "channel is switched on for one population and off for the "
                        "other, so one trades energy with the other without it being "
                        "recorded. Detailed balance no longer holds and there is no "
                        "thermal fixed point. Proceed at your own risk."
                    )

        if setup.solver.self_consistent_gap and setup.grid.min_factor >= 1.0:
            report.warnings.append(
                "Self-consistent gap: the energy grid does not extend below "
                "Delta_0, so any suppressed-gap solution lacks occupation "
                "samples near its new edge. Set grid.min_factor below the "
                "smallest expected Delta/Delta_0 for quantitative results."
            )
        if setup.solver.method == "coupled_newton" and setup.phonons.mode == "thermal_bath":
            report.errors.append(
                "Solver: coupled-Newton solves (f, n_ph) jointly and cannot be combined "
                "with the pinned thermal bath — pick a dynamic phonon sector or the "
                "auto/picard route."
            )
        if setup.solver.method == "coupled_newton" and setup.phonons.mode == "dynamic_closed":
            report.errors.append(
                "Solver: coupled-Newton requires finite phonon escape; the dynamic_closed "
                "sector has an unconstrained conserved-energy mode. Use Picard/auto or "
                "select dynamic_escape."
            )
        if (
            setup.phonons.mode != "thermal_bath"
            and setup.phonons.use_phonon_side_kernel
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

    elif isinstance(setup, Spatial2DSetup):
        _validate_drives_and_probe(report, setup)
        # Dynes broadening is reported by _validate_drives_and_probe above,
        # which covers every kinetic mode; do not repeat it here.
        if setup.material.D_0 < 0.0:
            report.errors.append("2D geometry: D₀ (μm²/ns) cannot be negative.")
        elif setup.material.D_0 == 0.0:
            report.warnings.append(
                "2D geometry: D₀ = 0 switches spatial transport off entirely; "
                "every cell then evolves independently."
            )
        source = setup.geometry
        if source.kind == "gds":
            if not source.gds_path:
                report.errors.append("2D geometry: a GDS source needs gds_path.")
            elif not gds_support_available():
                report.errors.append(
                    "2D geometry: GDS import needs the optional 'gdstk' "
                    'package. Install it with: pip install -e ".[gds]" — or '
                    "choose the rectangle geometry."
                )
        else:
            cells = source.rows * source.cols
            if cells > 40_000:
                report.warnings.append(
                    f"2D geometry: {source.rows}×{source.cols} is {cells:,} "
                    "cells. Each energy bin factorises its own sparse operator, "
                    "so both memory and time grow with this; start smaller and "
                    "refine once the physics looks right."
                )
            dimensionality = (
                0 if cells <= 1 else (1 if 1 in (source.rows, source.cols) else 2)
            )
            if dimensionality < 2:
                report.warnings.append(
                    f"2D geometry: a {source.rows}×{source.cols} mask is the "
                    f"{dimensionality}-D reduction, which is intended and "
                    "supported — the same core solves it — but say so "
                    "deliberately rather than by accident."
                )
        if setup.boundary.kind in ("dirichlet", "neumann") and (
            not np.isfinite(setup.boundary.value)
        ):
            report.errors.append(
                f"2D geometry: a {setup.boundary.kind} boundary needs a finite "
                "value."
            )

    elif isinstance(setup, Spatial1DSetup):
        _validate_drives_and_probe(report, setup)
        if setup.material.D_0 < 0.0:
            report.errors.append("1D strip: D₀ (μm²/ns) cannot be negative.")
        elif setup.material.D_0 == 0.0:
            # Not an error: the flux coefficient is D_0*N_1**q, so zero gives an
            # exactly zero transport operator and the strip becomes a set of
            # independent 0-D cells. That is the way to isolate the other terms.
            report.warnings.append(
                "1D strip: D₀ = 0 switches spatial transport off entirely; each "
                "cell evolves independently."
            )
        # A large diffusion number is safe because the backend subcycles CN
        # below its non-negative-amplification stiffness bound, but it can make
        # one visible driver step substantially more expensive. Warn about the
        # hidden work rather than claiming the old full-step ringing/clip
        # failure.
        if setup.material.D_0 > 0.0 and setup.num_cells > 1:
            dx_um = setup.length_um / setup.num_cells
            diffusion_number = setup.material.D_0 * setup.dt / (dx_um * dx_um)
            if diffusion_number > 8.0:
                report.warnings.append(
                    f"1D strip: diffusion number D₀·dt/dx² ≈ {diffusion_number:.1f} "
                    f"(dt = {setup.dt:g} ns, dx = {dx_um:.2g} µm). The "
                    "Crank–Nicolson transport step will be internally subcycled "
                    "to prevent stiff ringing; reduce dt or increase dx if the "
                    "run is too slow."
                )
        if setup.injection.enabled:
            e_center = setup.injection.center_over_delta
            if not (setup.grid.min_factor < e_center < setup.grid.max_factor):
                report.errors.append(
                    f"Injection: line center {e_center:g}×Δ lies outside the energy grid "
                    f"[{setup.grid.min_factor:g}, {setup.grid.max_factor:g}]×Δ."
                )
        if setup.gap_profile.kind == "step":
            e_max = setup.grid.max_factor * setup.material.Delta_0
            e_min = setup.grid.min_factor * setup.material.Delta_0
            if (
                setup.gap_profile.interface_G_N is not None
                and setup.gap_profile.gap_left == setup.gap_profile.gap_right
            ):
                report.errors.append(
                    "Gap profile: interface_G_N requires distinct gap_left and "
                    "gap_right values; equal gaps do not define a step face."
                )
            for side, val in (
                ("gap_left", setup.gap_profile.gap_left),
                ("gap_right", setup.gap_profile.gap_right),
            ):
                if val < e_min:
                    report.errors.append(
                        f"Gap profile: {side} = {val:g} micro-eV lies below the "
                        f"grid bottom {e_min:g} micro-eV. Lower grid.min_factor "
                        "so the local BCS edge is represented."
                    )
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
        # The photon drive must clear the junction pair-breaking threshold
        # Δ_L + Δ_R (with Δ_L = Δ_R + ω_LR); below it S⁻ = 0 makes the Γ_ph
        # calibration singular and every sweep point returns NaN. This is a
        # deterministic, T-independent setup error — reject it up front here
        # rather than running the whole sweep to an all-NaN "done".
        drive_threshold_GHz = 2.0 * setup.Delta_R_over_h_GHz + setup.omega_LR_over_h_GHz
        if setup.drive.omega_nu_GHz <= drive_threshold_GHz:
            report.errors.append(
                f"M25: drive ω_ν = {setup.drive.omega_nu_GHz:g} GHz is at or below the "
                f"pair-breaking threshold Δ_L + Δ_R = {drive_threshold_GHz:g} GHz; the "
                "Γ_ph calibration is singular (S⁻ = 0) and the sweep returns all-NaN."
            )
        if setup.branch_picker_mode == "max_x_L":
            # The engine only emits a (default-suppressed) DeprecationWarning
            # and the M25 executor captures no warnings, so a saved setup using
            # this deprecated picker ran silently to "done" on pseudo-roots
            # (~60x wrong x_L, ~600x parity rates on the default parameters).
            # The schema keeps the value so old setups still validate-load;
            # reject it here at run time.
            report.errors.append(
                "M25: branch_picker_mode 'max_x_L' is deprecated — it selects "
                "sub-1-Hz slope pseudo-roots (≈60× wrong x_L). Use "
                "'min_residual'."
            )

    return report


def build_spectral(
    setup: SteadyState0DSetup | Transient0DSetup | Spatial1DSetup | Spatial2DSetup,
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

    has_sector = isinstance(setup, (SteadyState0DSetup, Transient0DSetup))
    if has_sector and setup.phonons.mode == "dynamic_escape":
        tau_l_value = setup.phonons.tau_l_ns
    else:
        # thermal_bath (value unused: n_ph is pinned on both the Newton path
        # and the frozen transient) and dynamic_closed, where 0.0 is the
        # engine's no-substrate τ_l → ∞ sentinel.
        tau_l_value = 0.0

    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, setup.T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_value),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=fermi_dirac_distribution(spectral.E, setup.T_bath),
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
        "enable_scattering": setup.collisions.scattering,
        "enable_recombination": setup.collisions.recombination,
        "enable_phonon_scattering_source": setup.collisions.phonon_scattering_source,
        "enable_phonon_recombination_source": (
            setup.collisions.phonon_recombination_source
        ),
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
    if method == "coupled_newton":
        # The UI exposes one pair of Newton controls. The backend's monolithic
        # path deliberately has distinct keyword names, so map the displayed
        # values rather than silently falling back to its 1e-10 / 50 defaults.
        # NOT mapped: ``coupled_newton_analytic_cross``. SolverOptions has no
        # field for it, so this route takes the backend default (False) and
        # builds the cross blocks by finite differences — (NE + N_ω) residual
        # evaluations per Newton iteration instead of two assemblies (the
        # 2026-08-03 review measured 0.38 s vs 34 s per iteration at the
        # shipped 400-bin default). Every in-tree driver passes
        # analytic_cross=True. Exposing it is a 2026-08-03 review item held
        # for recertification (it selects a different Jacobian, not a
        # different root).
        kwargs["coupled_newton_tol"] = s.newton_tol
        kwargs["coupled_newton_max_iter"] = s.newton_max_iter
    kwargs["picard_tol"] = s.picard_tol
    kwargs["picard_max_iter"] = s.picard_max_iter
    kwargs["picard_mixing"] = s.picard_mixing
    kwargs["anderson_depth"] = s.anderson_depth
    return kwargs


def build_state_1d(setup: Spatial1DSetup) -> T3Spatial1DState:
    """Thermal-seed 1D strip state with optional two-gap step profile."""
    spectral = build_spectral(setup)
    # Equal finite-volume cell centers on [0, length_um]. Reflective
    # boundaries sit half a cell outside the first/last centers, so
    # ``state.cell_widths.sum()`` is exactly the requested physical length.
    dx_um = setup.length_um / setup.num_cells
    x = (np.arange(setup.num_cells, dtype=float) + 0.5) * dx_um
    f_col = fermi_dirac_distribution(spectral.E, setup.T_bath)
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


def build_geometry_2d(setup: Spatial2DSetup) -> Geometry:
    """Geometry for the 2-D mode, from extent or from a layout file."""
    source = setup.geometry
    if source.kind == "rectangle":
        return rectangle(
            source.rows, source.cols, mesh_size=source.mesh_size_um,
        )
    if source.gds_path is None:
        raise ValueError("A GDS geometry needs gds_path.")
    if not gds_support_available():
        raise ValueError(
            "GDS import needs the optional 'gdstk' package, which is not "
            'installed. Install it with: pip install -e ".[gds]" -- or choose '
            "the rectangle geometry."
        )
    return from_gds(
        source.gds_path, source.gds_layer, source.mesh_size_um,
        require_connected=source.require_connected,
    )


def build_state_2d(setup: Spatial2DSetup) -> T3SpatialState:
    """Thermal-seed state on the setup's geometry."""
    geometry = build_geometry_2d(setup)
    spectral = build_spectral(setup)
    condition = BoundaryCondition(
        setup.boundary.kind,
        None if setup.boundary.kind in ("reflective",) else setup.boundary.value,
    )
    thermal = fermi_dirac_distribution(spectral.E, setup.T_bath)
    return T3SpatialState(
        f=np.repeat(thermal[:, None], geometry.cell_count, axis=1),
        geometry=geometry,
        spectral=spectral,
        material=material_from_params(setup.material),
        T_bath=setup.T_bath,
        conditions=geometry.conditions(condition),
        diffusion_model=DiffusionModel[setup.diffusion_model],
    )
