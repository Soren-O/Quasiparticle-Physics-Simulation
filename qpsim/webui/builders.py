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

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.backends.t3_spatial import T3SpatialState
from qpsim.collisions.pair_breaking_photon import (
    validate_pair_breaking_photon_grid,
)
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    validate_phonon_lattice_coupling,
)
from qpsim.collisions.sub_gap_photon import COMMENSURATE_TOL
from qpsim.constants import H_OVER_KB_K_PER_HZ
from qpsim.fields.drive import (
    ExpressionDrive,
    ExternalDrive,
    SeparableDrive,
    SumDrive,
    cell_coordinates,
)
from qpsim.fields.initial import (
    energy_profile,
    seed_occupation,
    separable_excess,
    spatial_profile,
)
from qpsim.fields.safe_eval import compile_expression
from qpsim.geometries import (
    Geometry,
    discover_gds_layers,
    from_gds,
    from_polygons,
    gds_support_available,
    rectangle,
)
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
    EdgeCondition,
    EdgeConditions,
    EnergyProfileSpec,
    M25JunctionSetup,
    MaterialParams,
    ProbeConfig,
    KineticsSetup,
    SpatialProfileSpec,
    TimeProfileSpec,
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
    setup: KineticsSetup,
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
    setup: KineticsSetup,
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
    setup: KineticsSetup,
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


def _validate_phonon_lattice(report: ValidationReport, setup: Any) -> None:
    """Reject an energy grid whose two phonon channels would decouple.

    Only for a LIVE phonon population: a thermal bath never builds the omega
    map, so the constraint does not apply to it and enforcing it there would
    reject grids that are perfectly fine for the run being asked for.

    Checked here as well as in the engine so the answer arrives at submission,
    with the corrected bin counts, rather than as a ValueError from inside a
    collision layer after the job has been queued.
    """
    phonons = getattr(setup, "phonons", None)
    if phonons is None or getattr(phonons, "mode", "thermal_bath") == "thermal_bath":
        return
    try:
        E = build_energy_grid(
            gap=setup.material.Delta_0,
            energy_min_factor=setup.grid.min_factor,
            energy_max_factor=setup.grid.max_factor,
            num_energy_bins=setup.grid.num_bins,
        )[0]
        omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(E)
        validate_phonon_lattice_coupling(
            omega, idx_diff, idx_sum, sign, E_bins=E,
        )
    except ValueError as exc:
        report.errors.append(f"Phonon frequency grid: {exc}")


def validate_setup(setup: AnySetup) -> ValidationReport:
    """Cross-field physics validation (schema-level checks already passed)."""
    report = ValidationReport()
    _validate_phonon_lattice(report, setup)

    if isinstance(setup, KineticsSetup):
        _validate_drives_and_probe(report, setup)
        # Strategy-specific guards, carried over from the modes this one
        # replaced. They were never about those modes -- they are about the
        # SOLVER each strategy selects, and deleting them with the modes
        # would have dropped six real checks on the route that still runs
        # them (the equal-gap interface guard below was found exactly that
        # way, by a test that posted a retired mode name and got a 200).
        if setup.strategy == "steady_state":
            # This strategy reads a STRICT SUBSET of the setup: material,
            # phonons.mode and probe. Every physical input listed below is
            # discarded -- measured bit-identical to 11 digits with injection
            # at 2e-4 and at 2e-1, while the terms panel still reported the
            # source as on. Before the mode collapse these fields could not be
            # set on this route at all, because it was a separate mode with a
            # narrower schema; making `strategy` a setting is what created a
            # path where they can be set, look accepted and do nothing.
            #
            # Refused rather than warned: a setup carrying a drive that is
            # dropped is not the setup the user described, and a run that
            # returns the thermal answer to a driven question is wrong in the
            # way that is hardest to notice.
            # "Enabled" is not the same as "acting", and only an ACTING term
            # is one this strategy discards. A photon drive is switched by its
            # COUPLING -- every kernel term is multiplied by it, so c_phot = 0
            # applies nothing however enabled it is -- and injection by its
            # rate. Keying the refusal on `enabled` alone would reject setups
            # where nothing would have happened anyway, which is the mirror of
            # the defect this check exists to prevent. Same rule as
            # `terms._photon` / `terms._injection`, deliberately.
            def _acting(node_name: str, magnitude: str) -> bool:
                node = getattr(setup, node_name, None)
                if node is None or not bool(getattr(node, "enabled", False)):
                    return False
                return float(getattr(node, magnitude, 0.0) or 0.0) > 0.0

            dropped: list[str] = []
            if _acting("injection", "rate_per_ns"):
                dropped.append("injection")
            if getattr(setup, "drives", None):
                dropped.append("drives")
            initial = getattr(setup, "initial", None)
            initial_kind = getattr(initial, "kind", "thermal") if initial else "thermal"
            if initial_kind != "thermal":
                dropped.append(f"initial.kind={initial_kind!r}")
            if _acting("subgap_drive", "c_phot"):
                dropped.append("subgap_drive")
            if _acting("pb_drive", "c_phot_PB"):
                dropped.append("pb_drive")
            if dropped:
                report.errors.append(
                    "strategy='steady_state' solves a 0-D steady state from the "
                    "material, the phonon sector and the probe alone, so it would "
                    f"silently discard: {', '.join(dropped)}. Choose "
                    "strategy='time_march' to drive the system, or clear these "
                    "fields to state that the run is undriven."
                )

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
            if setup.self_consistent_gap and setup.grid.min_factor >= 1.0:
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
        else:
            if setup.dt > setup.material.tau_0 / 10.0:
                report.warnings.append(
                    f"dt = {setup.dt:g} ns exceeds τ₀/10 = "
                    f"{setup.material.tau_0 / 10:g} ns — ETD2 stability "
                    f"limits may distort the transient."
                )
            n_steps = setup.max_time / setup.dt
            if n_steps > 2e5:
                report.warnings.append(
                    f"max_time/dt ≈ {n_steps:.3g} substeps — this run will "
                    f"take a while."
                )
        # Dynes broadening is reported by _validate_drives_and_probe above,
        # which covers every kinetic mode; do not repeat it here.
        if setup.material.D_0 < 0.0:
            report.errors.append("Kinetics: D₀ (μm²/ns) cannot be negative.")
        elif setup.material.D_0 == 0.0:
            report.warnings.append(
                "Kinetics: D₀ = 0 switches spatial transport off entirely; "
                "every cell then evolves independently."
            )
        source = setup.geometry
        if source.kind == "gds":
            if not source.gds_path:
                report.errors.append("Kinetics: a GDS source needs gds_path.")
            elif not Path(source.gds_path).is_file():
                report.errors.append(
                    f"Kinetics: GDS file not found: {source.gds_path}"
                )
            elif not gds_support_available():
                report.errors.append(
                    "Kinetics: GDS import needs the optional 'gdstk' "
                    'package. Install it with: pip install -e ".[gds]" — or '
                    "choose the rectangle geometry."
                )
        else:
            cells = source.rows * source.cols
            if cells > 40_000:
                report.warnings.append(
                    f"Kinetics: {source.rows}×{source.cols} is {cells:,} "
                    "cells. Each energy bin factorises its own sparse operator, "
                    "so both memory and time grow with this; start smaller and "
                    "refine once the physics looks right."
                )
            # There used to be a warning here that a mask of reduced rank "is
            # the N-D reduction ... but say so deliberately rather than by
            # accident". It is gone, because it stopped being true: with the
            # 0-D and 1-D modes retired, a 1x1 mask is not an accident to flag,
            # it is HOW you ask for 0-D. The warning would now fire on the
            # ordinary path, and a warning that fires when nothing is wrong
            # teaches people to ignore warnings.
            regions = setup.gap_regions
            if (
                regions.kind == "column_step"
                and regions.interface_G_N is not None
                and regions.gap_left == regions.gap_right
            ):
                # Ported from the retired 1-D mode, which was the only place
                # this was checked. A conductance describes how hard it is to
                # cross a step FACE, and equal gaps do not define one -- the
                # setting would be read, stored, and mean nothing. Retiring a
                # mode must not quietly drop the guards it carried.
                report.errors.append(
                    "Gap regions: interface_G_N requires distinct gap_left and "
                    "gap_right values; equal gaps do not define a step face."
                )
            if setup.strategy == "steady_state" and cells != 1:
                # Caught here as well as at execution so the UI can say it
                # before a run starts. Blames the solver, not the device: a
                # multi-cell mask is a perfectly good device, and the thing
                # that cannot be done is asking THIS solver for its fixed point.
                report.errors.append(
                    f"Kinetics: strategy 'steady_state' uses the 0-D "
                    f"steady-state solver, whose state has no cell axis, and "
                    f"this mask has {cells} cells. Use a single cell, or keep "
                    "the geometry and switch to 'time_march', which reaches a "
                    "steady state by advancing to stop_tol."
                )
        if setup.boundary.kind in ("dirichlet", "neumann") and (
            not np.isfinite(setup.boundary.value)
        ):
            report.errors.append(
                f"Kinetics: a {setup.boundary.kind} boundary needs a finite "
                "value."
            )
        # The 1-D branch has always checked this and the 2-D branch never did,
        # so the identical setup was rejected on one mode and silently ran on
        # the other. A line outside the grid is not a small source: at
        # centre = 6Delta on a grid stopping at 4Delta the peak gain is 5e-95
        # against a nominal 2e-5, and terms.py reads only `enabled`, so the
        # panel reports "External injection: on" for a device that reaches the
        # bath value and nothing else.
        if setup.injection.enabled:
            e_center = setup.injection.center_over_delta
            if not (setup.grid.min_factor < e_center < setup.grid.max_factor):
                report.errors.append(
                    f"Injection: line center {e_center:g}×Δ lies outside the "
                    f"energy grid [{setup.grid.min_factor:g}, "
                    f"{setup.grid.max_factor:g}]×Δ, so the source would be "
                    "numerically absent."
                )
        _validate_gap_map_against_grid(report, setup)

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
    setup: KineticsSetup,
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
    setup: KineticsSetup,
) -> T3DiffusionState:
    """Thermal-seed 0-D T3 state on the physics ω-grid.

    Also serves ``KineticsSetup`` under ``strategy="steady_state"``, which is
    what makes the merged mode reproduce the 0-D mode BIT-IDENTICALLY: the two
    do not build equivalent states by parallel code, they build the same state
    by the same code. A second implementation that merely agreed today is the
    standing defect of this repo.
    """
    spectral = build_spectral(setup)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)

    has_sector = isinstance(
        setup, KineticsSetup,
    )
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
    setup: KineticsSetup,
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


def steady_state_solver_kwargs(
    setup: KineticsSetup,
) -> dict[str, object]:
    """Backend ``steady_state`` kwargs for the chosen phonon sector + method."""
    s = setup.solver
    # ONE authority for the gap switch. The 0-D mode carries it on `solver`;
    # the merged mode carries it at the top level, which is also what the term
    # panel's `gapeq` switch and the time-march path read. Reading `solver` for
    # both would make the merged mode's displayed switch inert.
    self_consistent_gap = (
        setup.self_consistent_gap if isinstance(setup, KineticsSetup)
        else s.self_consistent_gap
    )
    kwargs: dict[str, object] = {
        "self_consistent_gap": self_consistent_gap,
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
        kwargs["coupled_newton_tol"] = s.newton_tol
        kwargs["coupled_newton_max_iter"] = s.newton_max_iter
        # How the cross blocks are BUILT, not what is solved: the same root,
        # reached by an exact closed-form derivative of the discrete residual
        # instead of a finite-difference secant. This route used to take the
        # backend's legacy default and rebuild the cross blocks by finite
        # differences -- NE + N_omega residual assemblies per Newton iteration
        # rather than two -- so every web-UI coupled-Newton run paid tens of
        # seconds per iteration for a Jacobian every in-tree driver already
        # builds analytically.
        kwargs["coupled_newton_analytic_cross"] = s.coupled_newton_analytic_cross
        # The one that actually gates this route. Without it the displayed
        # "Newton tolerance" was routed into `coupled_newton_tol`, which is
        # read only when step_rtol <= 0 -- so tightening it from 1e-12 to
        # 1e-20, or loosening it to 1e-2, returned bit-identical f and n_ph.
        kwargs["coupled_newton_step_rtol"] = s.coupled_newton_step_rtol
    kwargs["picard_tol"] = s.picard_tol
    kwargs["picard_max_iter"] = s.picard_max_iter
    kwargs["picard_mixing"] = s.picard_mixing
    kwargs["anderson_depth"] = s.anderson_depth
    return kwargs






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


def build_geometry_2d(setup: KineticsSetup) -> Geometry:
    """Geometry for the 2-D mode, from extent or from a layout file."""
    source = setup.geometry
    if source.kind == "rectangle":
        return rectangle(
            source.rows, source.cols, mesh_size=source.mesh_size_um,
        )
    if source.kind == "polygons":
        if not source.polygons:
            raise ValueError("A polygons geometry needs at least one polygon.")
        return from_polygons(
            source.polygons, source.mesh_size_um,
            require_connected=source.require_connected,
        )
    if source.gds_path is None:
        raise ValueError("A GDS geometry needs gds_path.")
    if not gds_support_available():
        raise ValueError(
            "GDS import needs the optional 'gdstk' package, which is not "
            'installed. Install it with: pip install -e ".[gds]" -- or choose '
            "the rectangle geometry."
        )
    # Checked here rather than left to the reader: gdstk answers a missing
    # file with an OSError and a wrong layer with "No polygons found on
    # layer N", neither of which tells the person what to type instead.
    path = Path(source.gds_path)
    if not path.is_file():
        raise ValueError(f"GDS file not found: {source.gds_path}")
    layers = discover_gds_layers(path)
    if source.gds_layer not in layers:
        raise ValueError(
            f"Layer {source.gds_layer} carries no polygons in {path.name}; the "
            f"layers with polygons are: {', '.join(str(n) for n in layers)}."
        )
    return from_gds(
        source.gds_path, source.gds_layer, source.mesh_size_um,
        require_connected=source.require_connected,
    )


def build_state_2d(setup: KineticsSetup) -> T3SpatialState:
    """Thermal-seed state on the setup's geometry."""
    geometry = build_geometry_2d(setup)
    spectral = build_spectral(setup)
    conditions = build_boundary_conditions_2d(setup, geometry)
    thermal = fermi_dirac_distribution(spectral.E, setup.T_bath)
    return T3SpatialState(
        f=np.repeat(thermal[:, None], geometry.cell_count, axis=1),
        geometry=geometry,
        spectral=spectral,
        material=material_from_params(setup.material),
        T_bath=setup.T_bath,
        conditions=conditions,
        diffusion_model=DiffusionModel[setup.diffusion_model],
        gap_per_cell=build_gap_per_cell_2d(setup, geometry),
        interface_conductance=(
            None if setup.gap_regions.kind == "uniform"
            else setup.gap_regions.interface_G_N
        ),
    )


def _boundary_condition(spec: EdgeCondition | EdgeConditions) -> BoundaryCondition:
    """Schema condition -> engine condition.

    `reflective` and `absorbing` carry no value at all: passing 0.0 for them
    would read as a Dirichlet wall pinned at zero, which is a different
    boundary and a silently different device.
    """
    if spec.kind in ("reflective", "absorbing"):
        return BoundaryCondition(spec.kind)
    return BoundaryCondition(spec.kind, spec.value, spec.aux_value)


_DIRECTIONS = ("up", "down", "left", "right")


def _edges_facing(geometry: Geometry, direction: str) -> list[str]:
    """Edge ids whose every face points ``direction``.

    Segments are run-merged and named ``edge_0001``..., which is the right
    identity for a GDS-imported outline but useless to type: on a rectangle
    nobody knows which number is the left end. Directions are how a person
    describes a rim, so they are accepted as aliases. A merged segment that
    turns a corner belongs to no single direction and is deliberately not
    matched -- silently including it would apply a condition to a face the
    author did not mean.
    """
    return [
        edge.edge_id
        for edge in geometry.edges
        if edge.faces and all(f.direction == direction for f in edge.faces)
    ]


def build_boundary_conditions_2d(
    setup: KineticsSetup, geometry: Geometry,
) -> dict[str, BoundaryCondition]:
    """Per-edge conditions: the rim default, then any named overrides.

    An override key is either a real segment id or one of the four direction
    aliases.
    """
    conditions = geometry.conditions(_boundary_condition(setup.boundary))
    for key, spec in setup.boundary.per_edge.items():
        condition = _boundary_condition(spec)
        if key in conditions:
            conditions[key] = condition
            continue
        if key in _DIRECTIONS:
            matched = _edges_facing(geometry, key)
            if not matched:
                raise ValueError(
                    f"This geometry has no edge facing {key!r}. Its segments "
                    f"are: {', '.join(sorted(conditions))}."
                )
            for edge_id in matched:
                conditions[edge_id] = condition
            continue
        # Assembly requires every outward face to be named, so an
        # unrecognised id is a condition that would silently never apply.
        raise ValueError(
            f"This geometry has no edge {key!r}. Use one of "
            f"{', '.join(_DIRECTIONS)}, or a segment id: "
            f"{', '.join(sorted(conditions))}."
        )
    return conditions


def build_gap_per_cell_2d(setup: KineticsSetup, geometry: Geometry) -> np.ndarray | None:
    """Local gap for every solved cell, or ``None`` for a uniform gap."""
    regions = setup.gap_regions
    if regions.kind == "uniform":
        return None
    if regions.kind == "expression":
        x_um, y_um, x_norm, y_norm = cell_coordinates(geometry)
        fn = compile_expression(
            regions.expression, variables=(*_SPACE_VARS, "gap"),
        )
        gaps = np.broadcast_to(
            np.asarray(
                fn(
                    x=x_norm, y=y_norm, x_um=x_um, y_um=y_um,
                    gap=float(setup.material.Delta_0),
                    params=dict(regions.params),
                ),
                dtype=float,
            ),
            x_norm.shape,
        ).astype(float, copy=True)
        # A non-positive gap is not a smaller gap, it is a normal metal, and
        # every kernel here assumes a superconducting spectrum. Refuse rather
        # than produce a spectral context nothing downstream can interpret.
        if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0.0):
            raise ValueError(
                "The prescribed gap map must be finite and strictly positive "
                f"everywhere; it ranges over [{np.nanmin(gaps):g}, "
                f"{np.nanmax(gaps):g}] micro-eV."
            )
        return gaps
    _rows, cols = np.nonzero(geometry.mask)
    boundary_col = regions.step_fraction * geometry.shape[1]
    return np.where(cols < boundary_col, regions.gap_left, regions.gap_right)


def _validate_gap_map_against_grid(
    report: ValidationReport, setup: KineticsSetup,
) -> None:
    """The energy grid must reach below the SMALLEST local gap.

    A varying gap moves the band edge cell by cell, and the BCS weights
    cannot reconstruct singular support that was never sampled. Raised deep
    in the quadrature this reads as an opaque bound violation; here it can
    name the offending gap and the grid factor that would cover it.
    """
    regions = setup.gap_regions
    if regions.kind == "uniform":
        return
    delta_0 = setup.material.Delta_0
    largest: float | None = None
    if regions.kind == "column_step":
        smallest = min(regions.gap_left, regions.gap_right)
        largest = max(regions.gap_left, regions.gap_right)
    else:
        try:
            geometry = build_geometry_2d(setup)
            gaps = build_gap_per_cell_2d(setup, geometry)
        except Exception as exc:
            report.errors.append(f"Gap map: {exc}")
            return
        if gaps is None:
            return
        smallest = float(np.min(gaps))
        largest = float(np.max(gaps))
    # Only the LOW end was checked. A region above the grid top validates
    # clean and then dies inside SpectralContext with a message naming
    # neither the gap nor the field that set it -- an Al/Nb bilayer entered
    # in micro-eV is the obvious way in. The 1-D branch checks both ends.
    ceiling = setup.grid.max_factor * delta_0
    if largest is not None and largest >= ceiling:
        needed = largest / delta_0
        report.errors.append(
            f"Gap map: a local gap of {largest:.4g} micro-eV is at or above "
            f"the grid top {ceiling:.4g} "
            f"(max_factor={setup.grid.max_factor:g} x Delta_0={delta_0:g}), so "
            "no quasiparticle states would exist there. Set grid.max_factor > "
            f"{needed:.4g}, or lower the gap."
        )
    floor = setup.grid.min_factor * delta_0
    if smallest < floor:
        needed = smallest / delta_0
        report.errors.append(
            f"Gap map: the smallest local gap is {smallest:.4g} micro-eV "
            f"but the energy grid starts at {floor:.4g} "
            f"(min_factor={setup.grid.min_factor:g} x Delta_0={delta_0:g}). "
            f"Set grid.min_factor <= {needed:.4g} so the grid covers the "
            "band edge everywhere on the device."
        )


def build_injection_2d(
    setup: KineticsSetup, state: T3SpatialState,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian-in-energy source as ``(gain, loss_rate)`` over the cells."""
    injection = setup.injection
    if not injection.enabled:
        return None
    energies = state.spectral.E
    gap = setup.material.Delta_0
    centre = injection.center_over_delta * gap
    sigma = injection.sigma_over_delta * gap
    line = np.exp(-0.5 * ((energies - centre) / sigma) ** 2)

    rows, cols = np.nonzero(state.geometry.mask)
    gain = np.zeros((energies.size, state.f.shape[1]))
    if injection.where == "uniform":
        gain[:, :] = injection.rate_per_ns * line[:, None]
    elif injection.where == "left_edge":
        # Every cell on the lowest occupied column, so a ragged mask still
        # gets a whole edge injected rather than a single corner.
        target = cols == cols.min()
        gain[:, target] = injection.rate_per_ns * line[:, None]
    else:  # centre_cell
        centre_row = 0.5 * (rows.min() + rows.max())
        centre_col = 0.5 * (cols.min() + cols.max())
        nearest = int(np.argmin((rows - centre_row) ** 2 + (cols - centre_col) ** 2))
        gain[:, nearest] = injection.rate_per_ns * line
    return gain, np.zeros_like(gain)


# -- prescribed fields: initial conditions and drives ------------------
#
# The schema states a field; these turn it into the engine object. Kept here
# rather than in qpsim.fields so the engine layer stays free of pydantic and
# can be driven from a plain script.


def _compiled(source: str | None, variables: tuple[str, ...]) -> Any:
    """Compile a prescribed expression, or None when none was given."""
    if not source:
        return None
    return compile_expression(source, variables=variables)


_ENERGY_VARS = ("E", "gap")
_SPACE_VARS = ("x", "y", "x_um", "y_um")
_TIME_VARS = ("t",)


def _energy_shape(
    spec: EnergyProfileSpec,
    spectral: SpectralContext,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    # `params` was not forwarded, so every constant an author defined for a
    # separable expression was silently discarded -- and the whitelisted
    # `params.get('E0', 400.0)` form the codebase steers them toward then
    # returns its DEFAULT, so the run completes at the wrong line centre with
    # no error, no warning and no note. The schema documents params as in
    # scope for exactly these expressions.
    return energy_profile(
        spec.kind, spectral,
        T_eff=spec.T_eff, E_0=spec.E_0, width=spec.width,
        expression=_compiled(spec.expression, _ENERGY_VARS),
        params=params,
    )


def _space_shape(
    spec: SpatialProfileSpec,
    geometry: Geometry,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    x_um, y_um, x_norm, y_norm = cell_coordinates(geometry)
    return spatial_profile(
        spec.kind, x_norm, y_norm,
        x_0=spec.x_0, y_0=spec.y_0, sigma=spec.sigma,
        expression=_compiled(spec.expression, _SPACE_VARS),
        x_um=x_um, y_um=y_um,
        params=params,
    )


def _time_factor(
    spec: TimeProfileSpec, params: dict[str, float],
) -> tuple[Callable[[float], float], bool]:
    """``(factor(t), is_static)`` for a time profile.

    The static flag is what lets a steady drive cost nothing: the run loop
    samples once and hoists it out of the step loop entirely.
    """
    if spec.kind == "constant":
        return (lambda t: 1.0), True

    t_on = float(spec.t_on)
    if spec.kind == "pulse":
        t_off = float("inf") if spec.t_off is None else float(spec.t_off)

        def pulse(t: float) -> float:
            return 1.0 if t_on <= t < t_off else 0.0

        return pulse, False

    if spec.kind == "ramp":
        tau = float(spec.tau)  # validated present on the model

        def ramp(t: float) -> float:
            return float(np.clip((t - t_on) / tau, 0.0, 1.0))

        return ramp, False

    if spec.kind == "exponential":
        tau = float(spec.tau)

        def decay(t: float) -> float:
            return float(np.exp(-(t - t_on) / tau)) if t >= t_on else 0.0

        return decay, False

    compiled = _compiled(spec.expression, _TIME_VARS)

    def prescribed(t: float) -> float:
        return float(compiled(t=float(t), params=params))

    return prescribed, False


def build_initial_state_2d(
    setup: KineticsSetup, state: T3SpatialState,
) -> tuple[T3SpatialState, list[str]]:
    """Apply the setup's initial condition, returning the state and any notes.

    ``kind='thermal'`` returns the state untouched, so every setup written
    before initial conditions existed produces exactly the run it always did.
    """
    spec = setup.initial
    if spec.kind == "thermal":
        return state, []

    spectral = state.spectral
    n_cells = state.f.shape[1]
    if spec.expression is not None:
        x_um, y_um, x_norm, y_norm = cell_coordinates(state.geometry)
        fn = compile_expression(
            spec.expression, variables=_ENERGY_VARS + _SPACE_VARS,
        )
        ones = np.ones((spectral.E.size, 1))
        field = np.asarray(
            fn(
                E=np.broadcast_to(
                    spectral.E[:, None], (spectral.E.size, n_cells)
                ),
                gap=float(spectral.gap),
                x=ones * x_norm[None, :], y=ones * y_norm[None, :],
                x_um=ones * x_um[None, :], y_um=ones * y_um[None, :],
                params=dict(spec.params),
            ),
            dtype=float,
        )
        field = np.broadcast_to(field, (spectral.E.size, n_cells))
        seeded = seed_occupation(
            spectral, n_cells, setup.T_bath,
            **({"absolute": field} if spec.kind == "absolute" else {"excess": field}),
        )
    else:
        excess = separable_excess(
            _energy_shape(spec.energy, spectral, spec.params),
            _space_shape(spec.space, state.geometry, spec.params),
            spec.amplitude,
        )
        seeded = seed_occupation(
            spectral, n_cells, setup.T_bath,
            **({"absolute": excess} if spec.kind == "absolute" else {"excess": excess}),
        )
    return replace(state, f=seeded.f), list(seeded.notes)


def build_drives_2d(
    setup: KineticsSetup, state: T3SpatialState,
) -> ExternalDrive | None:
    """Every enabled drive on the setup, summed, or None if there are none."""
    parts: list[ExternalDrive] = []
    for spec in setup.drives:
        if not spec.enabled:
            continue
        params = dict(spec.params)
        if spec.expression is not None:
            x_um, y_um, x_norm, y_norm = cell_coordinates(state.geometry)
            parts.append(ExpressionDrive(
                fn=compile_expression(
                    spec.expression,
                    variables=_ENERGY_VARS + _SPACE_VARS + _TIME_VARS,
                ),
                energies=state.spectral.E,
                gap=float(state.spectral.gap),
                x_um=x_um, y_um=y_um, x_norm=x_norm, y_norm=y_norm,
                params=params,
                channel=spec.channel,
            ))
            continue
        pattern = spec.amplitude * np.outer(
            _energy_shape(spec.energy, state.spectral, params),
            _space_shape(spec.space, state.geometry, params),
        )
        factor, static = _time_factor(spec.time, params)
        parts.append(SeparableDrive(
            pattern=pattern, time_factor=factor,
            channel=spec.channel, static=static,
        ))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else SumDrive(tuple(parts))


def build_phonon_seed_2d(
    setup: KineticsSetup, geometry: Geometry,
) -> np.ndarray | None:
    """The phonon population a run starts from, or ``None`` for the bath.

    Returns a ``(n_omega,)`` profile; ``SpatialCollisions`` broadcasts it
    across cells. The frequency grid is derived from the energy grid, not from
    the geometry, so the seed does not depend on the mask.
    """
    spec = setup.phonons.initial
    if spec.kind == "bath" or setup.phonons.mode == "thermal_bath":
        return None
    spectral = build_spectral(setup)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    bath = thermal_phonon_occupation(omega, setup.T_bath)

    if spec.kind == "thermal_at":
        return thermal_phonon_occupation(omega, float(spec.T_eff))
    if spec.kind == "scaled":
        return float(spec.factor) * bath
    fn = compile_expression(spec.expression, variables=("omega", "n_bath"))
    seed = np.broadcast_to(
        np.asarray(
            fn(omega=omega, n_bath=bath, params=dict(spec.params)), dtype=float,
        ),
        omega.shape,
    ).astype(float, copy=True)
    if not np.all(np.isfinite(seed)) or np.any(seed < 0.0):
        raise ValueError(
            "The prescribed phonon seed must be finite and non-negative; a "
            "negative occupation is not a colder phonon field."
        )
    return seed
