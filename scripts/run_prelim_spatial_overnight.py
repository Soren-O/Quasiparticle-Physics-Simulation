"""Overnight 1D spatial sweep for the prelim validation experiment.

This is the slower companion to ``run_prelim_spatial_100um.py``.  It
uses a finer time step and mesh, sweeps diffusion/source assumptions, and
reports absolute resonant-frequency shifts for the six target modes from
the TeX presentation:

    f_n = 5 + n / 7 GHz, n = 1, ..., 6.

The source is still normalized as a local ``x_qp`` generation rate.  Once
the SIS I(V) / junction-resistance calibration is ready, replace that
normalization with a spectral current-to-gain conversion.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import os
import platform
import secrets
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise, product
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_spatial import (
    SpatialSnapshot,
    T3SpatialBackend,
    T3SpatialState,
)
from qpsim.geometries import strip
from qpsim.constants import KB_UEV_PER_K
from qpsim.experiments.prelim_resonators import (
    AL_STRIP_LENGTH_UM,
    AL_STRIP_THICKNESS_UM,
    AL_STRIP_WIDTH_UM,
    COUPLING_LENGTH_UM,
    FIXED_RESONATOR_LENGTH_UM,
    KINETIC_INDUCTANCE_FRACTION,
    PRELIM_RESONATORS,
    TARGET_RESONATOR_FREQUENCIES_GHZ,
    TARGET_RESONATOR_LENGTHS_UM,
    VARIABLE_PIECE_LENGTHS_UM,
    current_squared_profile,
    full_resonator_current_integral_um,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.observables.spatial_ac_response import compute_current_weighted_ac_response
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation


LENGTH_UM = AL_STRIP_LENGTH_UM
T_BATH_K = 0.1
ENERGY_MAX_FACTOR = 5.0
# Max transport diffusion number D0*dt/dx^2 per run; each config's dt_ns is a
# cap that is reduced per D0 below. The spatial Crank-Nicolson step clips [0,1]
# over/undershoot and the backend raises once a step alters >0.1% of the
# conserved density (~diffusion number 11); 4 keeps a clip-free margin.
CFL_TARGET = 4.0
N_SUBGAP_QUAD = 120
SPATIAL_GRID_CONVENTION = "uniform_finite_volume_cell_centers"


class _HasNX(Protocol):
    """Structural type shared by the two campaign configuration records."""

    @property
    def NX(self) -> int: ...


_CAMPAIGN_LOCK_SCHEMA = "qpsim.prelim-campaign-lock.v2"
_CAMPAIGN_LOCK_FILENAME = ".prelim-campaign.lock"
_CAMPAIGN_LOCK_OWNER_FILENAME = ".prelim-campaign.lock.owner.json"


@dataclass(frozen=True)
class CampaignLockHandle:
    """Open OS-lock handle retained for the full campaign lifetime."""

    lock_path: Path
    owner_path: Path
    stream: BinaryIO
    token: str
    owner_record: Mapping[str, object]


@lru_cache(maxsize=1)
def _source_material_code_digest() -> str:
    """Hash a conservative, newline-independent campaign source manifest.

    The overnight artifacts are expensive enough that stale reuse is worse
    than an occasional unnecessary rerun.  Hash every Python/YAML input under
    ``qpsim`` plus the campaign scripts themselves, including the Al material
    record.  Cache the result because every generated run id asks for it.
    """
    source_paths = {
        path.resolve()
        for path in (ROOT / "qpsim").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    source_paths.update(
        path.resolve()
        for path in (
            Path(__file__),
            ROOT / "scripts" / "run_prelim_spatial_finite_phonon_one.py",
            ROOT / "scripts" / "run_prelim_readout_heating_overnight.py",
        )
    )

    digest = hashlib.sha256()
    for path in sorted(source_paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        contents = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


@dataclass(frozen=True)
class SweepConfig:
    name: str
    NX: int
    NE: int
    dt_ns: float
    max_time_ns: float
    stop_tol: float
    snapshot_interval_ns: float
    D0_values: tuple[float, ...]
    source_rates_per_ns: tuple[float, ...]
    source_centers_delta: tuple[float, ...]
    source_sigmas_delta: tuple[float, ...]


SMOKE_CONFIG = SweepConfig(
    name="smoke",
    NX=21,
    NE=24,
    dt_ns=2.0,
    max_time_ns=20_000.0,
    stop_tol=2e-10,
    snapshot_interval_ns=2_000.0,
    D0_values=(6.0, 60.0),
    source_rates_per_ns=(2e-5,),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)

OVERNIGHT_CONFIG = SweepConfig(
    name="overnight",
    NX=41,
    NE=40,
    dt_ns=1.0,
    max_time_ns=60_000.0,
    stop_tol=5e-11,
    snapshot_interval_ns=5_000.0,
    D0_values=(3.0, 6.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0),
    source_rates_per_ns=(1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7),
    source_centers_delta=(1.95, 2.0, 2.1),
    source_sigmas_delta=(0.05, 0.08),
)

CALIBRATED_CONFIG = SweepConfig(
    name="calibrated",
    NX=41,
    NE=40,
    dt_ns=1.0,
    max_time_ns=60_000.0,
    stop_tol=5e-11,
    snapshot_interval_ns=5_000.0,
    D0_values=(3.0, 6.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0),
    # For the current 100/41 um finite-volume source cell, these span roughly
    # 3e10--3e12 QP/s, covering the SIS threshold estimate.
    source_rates_per_ns=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
    source_centers_delta=(1.95, 2.0, 2.1),
    source_sigmas_delta=(0.05, 0.08),
)


SUMMARY_FIELDS = [
    "run_id",
    "status",
    "NX",
    "D0_um2_per_ns",
    "source_rate_per_ns",
    "estimated_source_qp_per_s",
    "source_cell_volume_um3",
    "qps_per_xqp_source_cell",
    "source_center_delta",
    "source_sigma_delta",
    "dt_ns",
    "max_time_ns",
    "total_time_ns",
    "n_steps",
    "converged",
    "final_max_dfdt_per_ns",
    "xqp_mean",
    "xqp_source",
    "xqp_open_end",
    "delta_fr_hz_min",
    "delta_fr_hz_max",
    "error",
    "trace_csv",
    "profile_csv",
]

TRACE_FIELDS = [
    "t_ns",
    "max_dfdt_per_ns",
    "xqp_mean",
    "xqp_source",
    "xqp_open_end",
]

PROFILE_FIELDS = ["x_um", "xqp"]

RESONATOR_SHIFT_FIELDS = [
    "resonator_index",
    "resonator_label",
    "f0_ghz",
    "resonator_length_um",
    "variable_piece_length_um",
    "current_participation",
    "current_integral_strip_um",
    "current_integral_full_um",
    "xqp_current_weighted",
    "sigma1_current_weighted_norm",
    "sigma2_current_weighted_norm",
    "sigma1_ref_current_weighted_norm",
    "sigma2_ref_current_weighted_norm",
    "relative_sigma2_change_current_weighted",
    "inverse_Qi_current_weighted",
    "Qi_current_weighted",
    "Qi_weighted_f_legacy",
    "frac_freq_shift_current_weighted",
    "delta_fr_hz_current_weighted",
    "shifted_fr_ghz_current_weighted",
    "frac_freq_shift_weighted_f_legacy",
    "delta_fr_hz_weighted_f_legacy",
    "Qi_uniform_strip_average",
    "frac_freq_shift_uniform_strip_average",
    "delta_fr_hz_uniform_strip_average",
]

SHIFT_FIELDS = [
    "run_id",
    "D0_um2_per_ns",
    "source_rate_per_ns",
    "estimated_source_qp_per_s",
    "source_cell_volume_um3",
    "qps_per_xqp_source_cell",
    "source_center_delta",
    "source_sigma_delta",
    *RESONATOR_SHIFT_FIELDS,
]

EXPECTED_RESONATOR_INDICES = frozenset(
    int(resonator.index) for resonator in PRELIM_RESONATORS
)
EXPECTED_RESONATOR_LABELS = {
    int(resonator.index): resonator.label for resonator in PRELIM_RESONATORS
}

# Values written with Python's round-trip float representation should compare
# exactly after CSV parsing. Keep a small ulp-scale allowance so a harmless
# formatter change cannot make an otherwise complete campaign rerun.
_CSV_TIME_RTOL = 64.0 * float(np.finfo(float).eps)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    return fermi_dirac_occupation(E, T)


@dataclass
class StripFlux:
    """External source/sink for ``f(E, cell)`` on a campaign strip.

    Lives here rather than in the engine because the unified backend's API is
    two plain arrays -- ``external_gain`` and ``external_loss`` -- and this is
    a caller-side convenience that bundles them with the diagnostics these
    campaigns report. Carried over from the retired backend's StripFlux,
    validation included: extraction must be encoded as ``loss_rate * f`` rather
    than negative gain, matching qpsim.devices.external_flux.ExternalFlux.
    """

    gain: np.ndarray
    loss_rate: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def zero(cls, n_energy: int, n_cells: int) -> "StripFlux":
        """A source that does nothing, for the no-drive control run."""
        shape = (int(n_energy), int(n_cells))
        return cls(gain=np.zeros(shape), loss_rate=np.zeros(shape))

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.gain) or np.iscomplexobj(self.loss_rate):
            raise ValueError("gain and loss_rate must be real-valued.")
        gain = np.asarray(self.gain, dtype=float)
        loss = np.asarray(self.loss_rate, dtype=float)
        if gain.ndim != 2 or loss.shape != gain.shape:
            raise ValueError(
                f"gain and loss_rate must both have shape (NE, Ncells); got "
                f"{gain.shape} and {loss.shape}."
            )
        if np.any(gain < 0.0):
            raise ValueError("gain must be non-negative; encode removal as loss_rate.")
        self.gain, self.loss_rate = gain, loss


def strip_coordinates(state: T3SpatialState) -> np.ndarray:
    """Cell centres of a one-row mask, in um.

    The retired 1-D state carried an explicit ``x`` array; the unified state
    carries a mask and one mesh size, which is the SAME measure -- equal-volume
    cells at (i + 1/2) h -- stated once instead of sampled. Derived here rather
    than stored so it cannot drift from the geometry the solver actually used.
    """
    n = int(state.f.shape[1])
    return (np.arange(n, dtype=float) + 0.5) * float(state.geometry.mesh_size)


def _cell_centered_strip_grid(
    num_cells: int,
    *,
    length_um: float = LENGTH_UM,
) -> tuple[np.ndarray, float]:
    """Return equal-volume cell centers spanning exactly ``length_um``.

    The spatial backend conserves an equal-cell finite-volume sum.  Using
    endpoint samples here would give those endpoint samples a full backend
    cell but only half weight in physical trapezoidal observables.  Midpoint
    cells make the numerical and physical measures identical.
    """
    if not isinstance(num_cells, (int, np.integer)) or num_cells < 2:
        raise ValueError(f"num_cells must be an integer >= 2; got {num_cells!r}.")
    if (
        isinstance(length_um, (bool, np.bool_))
        or not isinstance(length_um, (int, float, np.integer, np.floating))
        or not np.isfinite(float(length_um))
        or float(length_um) <= 0.0
    ):
        raise ValueError(
            f"length_um must be finite and positive; got {length_um!r}."
        )
    dx_um = float(length_um) / int(num_cells)
    centers = (np.arange(int(num_cells), dtype=float) + 0.5) * dx_um
    return centers, dx_um


def _campaign_cell_widths(state: T3SpatialState) -> np.ndarray:
    """Return the campaign cell measure, rejecting a geometry mismatch."""
    expected_x, expected_dx = _cell_centered_strip_grid(state.f.shape[1])
    position_tol = 64.0 * np.finfo(float).eps * max(1.0, LENGTH_UM)
    if not np.allclose(strip_coordinates(state), expected_x, rtol=0.0, atol=position_tol):
        raise ValueError(
            "Prelim campaign cell centres must match the configured strip's "
            "finite-volume cell centers."
        )
    # The unified state carries ONE mesh size for a uniform mask instead of a
    # per-cell width array. Rebuilding the array here keeps the campaign's
    # trapezoidal observables reading a per-cell measure, and the check below
    # is now on the single authoritative number rather than on N copies of it.
    mesh = float(state.geometry.mesh_size)
    if not np.isclose(mesh, expected_dx, rtol=0.0, atol=position_tol):
        raise ValueError("Prelim campaign state cell widths do not match LENGTH_UM/NX.")
    return np.full(int(state.f.shape[1]), mesh, dtype=float)


def _build_state(config: SweepConfig, D0: float) -> T3SpatialState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=config.NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    x, dx_um = _cell_centered_strip_grid(config.NX)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], config.NX, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=strip(int(x.size), mesh_size=float(dx_um)),
        spectral=spectral,
        material=material,
        T_bath=T_BATH_K,
    )


def _source_flux(
    state: T3SpatialState,
    *,
    local_xqp_generation_rate_per_ns: float,
    center_delta: float,
    sigma_delta: float,
) -> StripFlux:
    center = center_delta * state.gap
    sigma = sigma_delta * state.gap
    profile = np.exp(-0.5 * ((state.spectral.E - center) / sigma) ** 2)
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    xqp_norm = float(np.sum(spectral_weights * profile) / state.gap)
    if xqp_norm <= 0.0:
        raise RuntimeError("Could not normalize source spectrum.")
    gain_spectrum = local_xqp_generation_rate_per_ns * profile / xqp_norm

    gain = np.zeros_like(state.f)
    gain[:, 0] = gain_spectrum
    return StripFlux(
        gain=gain,
        loss_rate=np.zeros_like(gain),
        diagnostics={
            "local_xqp_generation_rate_per_ns": local_xqp_generation_rate_per_ns,
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
            "source_center_uev": center,
            "source_sigma_uev": sigma,
        },
    )


def _source_calibration(
    config: _HasNX,
    source_rate_per_ns: float,
) -> dict[str, float]:
    """Estimate QP/s represented by a local source-cell x_qp rate.

    The Al database value ``rho_F = 1.74e28`` is the customary Al DOS in
    eV^-1 m^-3, so pair it with ``Delta`` in eV here.
    """
    material = load_material("Al")
    dx_um = LENGTH_UM / config.NX
    source_cell_volume_um3 = dx_um * AL_STRIP_WIDTH_UM * AL_STRIP_THICKNESS_UM
    source_cell_volume_m3 = source_cell_volume_um3 * 1e-18
    delta_eV = material.Delta_0 * 1e-6
    qps_per_xqp_source_cell = (
        4.0 * float(material.rho_F) * delta_eV * source_cell_volume_m3
    )
    return {
        "source_cell_volume_um3": source_cell_volume_um3,
        "qps_per_xqp_source_cell": qps_per_xqp_source_cell,
        "estimated_source_qp_per_s": (
            source_rate_per_ns * 1e9 * qps_per_xqp_source_cell
        ),
    }


def _xqp_profile(state: T3SpatialState) -> np.ndarray:
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    return np.sum(spectral_weights[:, None] * state.f, axis=0) / state.gap


def _mean_f(state: T3SpatialState) -> np.ndarray:
    return np.mean(state.f, axis=1)


def _current_weights(state: T3SpatialState, resonator_length_um: float) -> np.ndarray:
    """Current-squared profile over the strip at the shorted end.

    Gao writes the modal weighting as ``sin^2(pi x / 2l)`` with ``x`` measured
    from the open end, so a strip coordinate ``s`` measured away from the short
    maps to ``cos^2(pi s / 2l)``.
    """
    return current_squared_profile(strip_coordinates(state), resonator_length_um)


def _current_weighted_f(
    state: T3SpatialState,
    resonator_length_um: float,
) -> tuple[np.ndarray, float]:
    weights = _current_weights(state, resonator_length_um)
    cell_widths = _campaign_cell_widths(state)
    norm_um = float(np.sum(weights * cell_widths))
    if norm_um <= 0.0:
        raise RuntimeError("Current weighting normalization vanished.")
    weighted_f = (
        np.sum(state.f * weights[None, :] * cell_widths[None, :], axis=1)
        / norm_um
    )

    # Integral of cos^2(pi s / 2l) over the full quarter-wave resonator is l/2.
    current_participation = norm_um / (0.5 * resonator_length_um)
    return weighted_f, current_participation


def _current_weighted_xqp(
    state: T3SpatialState,
    resonator_length_um: float,
) -> float:
    weights = _current_weights(state, resonator_length_um)
    cell_widths = _campaign_cell_widths(state)
    norm_um = float(np.sum(weights * cell_widths))
    profile = _xqp_profile(state)
    return float(np.sum(profile * weights * cell_widths) / norm_um)


def _trace_observables() -> dict[str, Callable[[T3SpatialState], float]]:
    def xqp_mean(state: T3SpatialState) -> float:
        return float(np.mean(_xqp_profile(state)))

    def xqp_source(state: T3SpatialState) -> float:
        return float(_xqp_profile(state)[0])

    def xqp_open_end(state: T3SpatialState) -> float:
        return float(_xqp_profile(state)[-1])

    return {
        "xqp_mean": xqp_mean,
        "xqp_source": xqp_source,
        "xqp_open_end": xqp_open_end,
    }


def _resonator_shifts(
    state: T3SpatialState,
    f_ref: np.ndarray,
) -> list[dict[str, float | str]]:
    f_uniform = _mean_f(state)
    rows: list[dict[str, float | str]] = []
    for resonator in PRELIM_RESONATORS:
        f0_ghz = resonator.frequency_ghz
        length_um = resonator.total_length_um
        probe_energy = resonator.probe_energy_uev
        response = compute_current_weighted_ac_response(
            state.f,
            f_ref,
            strip_coordinates(state),
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            current_weights=_current_weights(state, length_um),
            full_current_integral_um=full_resonator_current_integral_um(length_um),
            spatial_cell_widths_um=_campaign_cell_widths(state),
            n_subgap=N_SUBGAP_QUAD,
        )
        delta_hz_current_weighted = response.frac_freq_shift * f0_ghz * 1e9

        f_current_weighted, _ = _current_weighted_f(state, resonator_length_um=length_um)
        f_ref_current_weighted, _ = _current_weighted_f(
            T3SpatialState(
                f=np.repeat(f_ref[:, None], state.f.shape[1], axis=1),
                geometry=state.geometry,
                spectral=state.spectral,
                material=state.material,
                T_bath=state.T_bath,
            ),
            resonator_length_um=length_um,
        )
        qi_current_weighted = compute_quality_factor(
            f_current_weighted,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_strip_only = compute_frequency_shift(
            f_current_weighted,
            f_ref_current_weighted,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_weighted_f_legacy = (
            frac_shift_strip_only * response.current_participation
        )

        qi_uniform = compute_quality_factor(
            f_uniform,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_uniform = compute_frequency_shift(
            f_uniform,
            f_ref,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        delta_hz_uniform = frac_shift_uniform * f0_ghz * 1e9
        rows.append(
            {
                "resonator_index": float(resonator.index),
                "resonator_label": resonator.label,
                "f0_ghz": f0_ghz,
                "resonator_length_um": length_um,
                "variable_piece_length_um": resonator.variable_piece_length_um,
                "current_participation": response.current_participation,
                "current_integral_strip_um": response.strip_current_integral_um,
                "current_integral_full_um": response.full_current_integral_um,
                "xqp_current_weighted": _current_weighted_xqp(state, length_um),
                "sigma1_current_weighted_norm": response.sigma1_current_weighted_norm,
                "sigma2_current_weighted_norm": response.sigma2_current_weighted_norm,
                "sigma1_ref_current_weighted_norm": (
                    response.sigma1_ref_current_weighted_norm
                ),
                "sigma2_ref_current_weighted_norm": (
                    response.sigma2_ref_current_weighted_norm
                ),
                "relative_sigma2_change_current_weighted": (
                    response.relative_sigma2_change_current_weighted
                ),
                "inverse_Qi_current_weighted": response.inverse_qi_qp,
                "Qi_current_weighted": response.qi_qp,
                "Qi_weighted_f_legacy": qi_current_weighted,
                "frac_freq_shift_current_weighted": response.frac_freq_shift,
                "delta_fr_hz_current_weighted": delta_hz_current_weighted,
                "shifted_fr_ghz_current_weighted": (
                    f0_ghz + delta_hz_current_weighted / 1e9
                ),
                "frac_freq_shift_weighted_f_legacy": frac_shift_weighted_f_legacy,
                "delta_fr_hz_weighted_f_legacy": (
                    frac_shift_weighted_f_legacy * f0_ghz * 1e9
                ),
                "Qi_uniform_strip_average": qi_uniform,
                "frac_freq_shift_uniform_strip_average": frac_shift_uniform,
                "delta_fr_hz_uniform_strip_average": delta_hz_uniform,
            }
        )
    return rows


def require_matching_header(path: Path, fieldnames: list[str]) -> None:
    """Refuse to append rows beneath a stale aggregate-CSV header.

    Appending a wider/reordered row set under an old header silently
    mislabels every later column (2026-07-20 review: 40-column rev2 rows
    under a 37-column legacy header shifted values across fields, with
    overflow landing in csv.DictReader's None key). A schema change
    requires a fresh output directory or an explicit schema migration;
    selective ``--no-resume`` deliberately preserves unrelated campaigns.
    Shared by the spatial and readout overnight runners.
    """
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as fp:
        first = fp.readline().strip()
    if not first:
        return
    existing = first.split(",")
    if existing != list(fieldnames):
        raise SystemExit(
            f"{path} has a {len(existing)}-column header that does not match "
            f"the current {len(fieldnames)}-column schema. Appending would "
            "silently mislabel columns. Use a fresh --out-dir or migrate "
            "the aggregate schema explicitly; --no-resume deliberately "
            "preserves rows owned by other campaigns."
        )


def _validated_campaign_run_ids(run_ids: Iterable[str]) -> tuple[str, ...]:
    """Materialize and validate every id before any destructive operation."""
    owned = tuple(run_ids)
    for run_id in owned:
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in {".", ".."}
            or Path(run_id).name != run_id
            or any(character in run_id for character in "*?[]")
        ):
            raise ValueError(f"Invalid campaign run id {run_id!r}.")
    return owned


def purge_run_ids_rows(path: Path, run_ids: Iterable[str]) -> None:
    """Atomically remove every row owned by the supplied exact run ids.

    A retried case (failed or max_time_reached) previously APPENDED a
    second row set for the same run id, leaving duplicates and orphan
    shift rows that no reader disambiguates (2026-07-20 round-4 review).
    A collision-safe temporary plus atomic replacement keeps re-runs
    idempotent without exposing a truncated aggregate. Shared by the spatial
    and readout runners.
    """
    owned = set(_validated_campaign_run_ids(run_ids))
    if not owned:
        return
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    try:
        idx = header.index("run_id")
    except ValueError:
        return
    kept = [r for r in data if len(r) <= idx or r[idx] not in owned]
    if len(kept) == len(data):
        return
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.purge.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            writer.writerows(kept)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def purge_run_id_rows(path: Path, run_id: str) -> None:
    """Atomically remove every row for ``run_id`` before a re-run."""
    purge_run_ids_rows(path, {run_id})


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write finite JSON through a collision-safe, fsynced replacement."""
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True, allow_nan=False)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _lock_campaign_stream(stream: BinaryIO) -> None:
    """Take one non-blocking exclusive OS lock on byte zero."""
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_campaign_stream(stream: BinaryIO) -> None:
    """Release the platform advisory lock; closing is the crash fallback."""
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _lock_contention(exc: OSError) -> bool:
    """Whether an OS lock failure means another descriptor owns the lock."""
    return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(
        exc, "winerror", None
    ) in {32, 33, 36}


def _campaign_owner_diagnostic(owner_path: Path) -> str:
    """Best-effort diagnostic text that never participates in lock authority."""
    try:
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "owner metadata unavailable or stale"
    if not isinstance(payload, dict):
        return "owner metadata malformed or stale"
    return (
        f"last owner pid={payload.get('pid')!r}, "
        f"owner={payload.get('owner_label')!r}, "
        f"acquired={payload.get('acquired_unix_s')!r}"
    )


def _close_campaign_stream(stream: BinaryIO, *, locked: bool) -> None:
    """Close every post-open path, releasing a held OS lock even on failure."""
    try:
        if locked and not stream.closed:
            _unlock_campaign_stream(stream)
    finally:
        stream.close()


def acquire_campaign_lock(
    out_dir: Path,
    *,
    owner_label: str,
    config_digest: str,
) -> CampaignLockHandle:
    """Acquire the sole shared writer lock for an output directory.

    The open descriptor's OS advisory lock is the *only* ownership authority.
    It is released automatically by the OS after a process crash. The
    persistent JSON sidecar is diagnostic only: it may be absent, corrupt, or
    stale and is never consulted to permit acquisition.
    """
    lock_path = out_dir / _CAMPAIGN_LOCK_FILENAME
    owner_path = out_dir / _CAMPAIGN_LOCK_OWNER_FILENAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        stream = os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    locked = False
    try:
        try:
            _lock_campaign_stream(stream)
            locked = True
        except OSError as exc:
            if _lock_contention(exc):
                diagnostic = _campaign_owner_diagnostic(owner_path)
                raise RuntimeError(
                    "Campaign output directory already has a live writer "
                    f"({diagnostic}); lock={lock_path}."
                ) from None
            raise
        # Initialize the persistent lock byte only after ownership is held.
        # Windows byte-range locks can cover a range beyond EOF; doing this
        # before the lock allowed two first-open contenders to race, with the
        # loser raising a raw PermissionError while writing beneath the
        # winner's mandatory lock.
        if os.fstat(stream.fileno()).st_size == 0:
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())

        token = secrets.token_hex(32)
        owner: dict[str, object] = {
            "schema": _CAMPAIGN_LOCK_SCHEMA,
            "token": token,
            "pid": os.getpid(),
            "owner_label": owner_label,
            "config_digest": config_digest,
            "platform": platform.platform(),
            "acquired_unix_s": time.time(),
            "released_unix_s": None,
        }
        # Failure to publish diagnostics must fail the invocation, but the
        # cleanup below always unlocks/closes so metadata can never wedge the
        # output directory.
        _atomic_write_json(owner_path, owner)
        return CampaignLockHandle(
            lock_path,
            owner_path,
            stream,
            token,
            owner,
        )
    except BaseException:
        _close_campaign_stream(stream, locked=locked)
        raise


def release_campaign_lock(handle: CampaignLockHandle) -> None:
    """Publish a clean-release diagnostic, then release/close unconditionally."""
    if handle.stream.closed:
        raise RuntimeError(f"Campaign lock is already released: {handle.lock_path}.")
    try:
        try:
            released = {
                **handle.owner_record,
                "released_unix_s": time.time(),
            }
            _atomic_write_json(handle.owner_path, released)
        except Exception as exc:
            warnings.warn(
                f"Could not update campaign lock diagnostics at release: {exc!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
    finally:
        _close_campaign_stream(handle.stream, locked=True)


def validate_campaign_limits(
    max_runs: object,
    wall_hours: object,
) -> tuple[int | None, float | None]:
    """Validate destructive campaign controls before touching the output path."""
    if max_runs is not None:
        if (
            isinstance(max_runs, (bool, np.bool_))
            or not isinstance(max_runs, (int, np.integer))
            or int(max_runs) <= 0
        ):
            raise ValueError(
                f"max_runs must be a positive integer when supplied; got {max_runs!r}."
            )
        max_runs = int(max_runs)
    if wall_hours is not None:
        if (
            isinstance(wall_hours, (bool, np.bool_))
            or not isinstance(wall_hours, (int, float, np.integer, np.floating))
            or not math.isfinite(float(wall_hours))
            or float(wall_hours) <= 0.0
        ):
            raise ValueError(
                "wall_hours must be finite and positive when supplied; "
                f"got {wall_hours!r}."
            )
        wall_hours = float(wall_hours)
    return max_runs, wall_hours


def purge_run_artifacts(out_dir: Path, run_ids: Iterable[str]) -> None:
    """Remove only artifacts owned by the supplied exact campaign run ids."""
    owned = _validated_campaign_run_ids(run_ids)
    for run_id in owned:
        for prefix in ("trace", "profile"):
            final_path = out_dir / f"{prefix}_{run_id}.csv"
            final_path.unlink(missing_ok=True)
            final_path.with_suffix(final_path.suffix + ".tmp").unlink(
                missing_ok=True
            )
            for orphan in out_dir.glob(f".{final_path.name}.*.tmp"):
                orphan.unlink(missing_ok=True)


def _append_csv(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    # Header when the file is missing OR zero-byte: a truncated/empty file
    # must not silently accumulate headerless rows (2026-07-20 review).
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        fp.flush()


def _write_trace(path: Path, snapshots: Sequence[SpatialSnapshot]) -> None:
    rows: list[dict[str, float]] = []
    for snapshot_index, snap in enumerate(snapshots):
        t_ns = float(snap.t)
        max_rate = float(snap.max_rate)
        # T3SpatialBackend uses +inf only for the unevaluated t=0 residual.
        # Persist a finite sentinel so the artifact passes the same finite-data
        # checks used by resume.  No other non-finite value is legitimate.
        if t_ns == 0.0 and max_rate == math.inf:
            max_rate = 0.0
        raw_row = {
            "t_ns": t_ns,
            "max_dfdt_per_ns": max_rate,
            **snap.observables,
        }
        row: dict[str, float] = {}
        for field in TRACE_FIELDS:
            try:
                value = float(raw_row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Trace snapshot {snapshot_index} has invalid {field}."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"Trace snapshot {snapshot_index} has non-finite {field}: "
                    f"{value!r}."
                )
            row[field] = value
        rows.append(row)

    _atomic_write_csv(path, TRACE_FIELDS, rows)


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write one complete CSV beside its destination, then promote it.

    A killed campaign must leave either the previous complete artifact or no
    committed artifact, never a truncated final-name CSV that discards an
    hours-long solve. Resume keys only on final paths, so orphan ``*.tmp``
    files from a hard process/host failure remain inert (and may be removed
    during ordinary output-directory maintenance).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_profile(path: Path, state: T3SpatialState) -> None:
    profile = _xqp_profile(state)
    rows = [
        {"x_um": float(x_um), "xqp": float(xqp)}
        for x_um, xqp in zip(strip_coordinates(state), profile, strict=True)
    ]
    _atomic_write_csv(path, PROFILE_FIELDS, rows)


# Physics/schema revision folded into every run id — BUMP whenever the
# runner's physics or summary schema changes so resume cannot silently
# accept invalidated rows (2026-07-20 review).
#   rev2 (2026-07-20): non-converged runs record status=max_time_reached
#     (previously mislabeled "completed" and permanently resume-excluded).
#   rev3 (2026-07-21): summary records NX; resume verifies the profile
#     cardinality and the trace endpoint against the committed summary.
#   rev4 (2026-07-21): resolution, time-step cap, run horizon, and stop
#     tolerance are part of the run id, preventing a smoke/calibrated result
#     from satisfying resume for a different numerical campaign.
#   rev5 (2026-07-21): exact point/config/source fingerprints make resume
#     sensitive to all campaign dimensions, physics constants, material data,
#     and repository physics code.
#   rev6 (2026-07-21): the strip uses equal finite-volume cell centers
#     (dx=L/NX); source calibration and every spatial observable use that same
#     midpoint measure. Endpoint-grid artifacts are intentionally stale.
_PHYSICS_REV = "rev6"


def _physics_constants() -> dict[str, object]:
    """Return imported/module constants that affect campaign physics."""
    return {
        "AL_STRIP_LENGTH_UM": AL_STRIP_LENGTH_UM,
        "AL_STRIP_THICKNESS_UM": AL_STRIP_THICKNESS_UM,
        "AL_STRIP_WIDTH_UM": AL_STRIP_WIDTH_UM,
        "CFL_TARGET": CFL_TARGET,
        "COUPLING_LENGTH_UM": COUPLING_LENGTH_UM,
        "ENERGY_MAX_FACTOR": ENERGY_MAX_FACTOR,
        "FIXED_RESONATOR_LENGTH_UM": FIXED_RESONATOR_LENGTH_UM,
        "KB_UEV_PER_K": KB_UEV_PER_K,
        "KINETIC_INDUCTANCE_FRACTION": KINETIC_INDUCTANCE_FRACTION,
        "LENGTH_UM": LENGTH_UM,
        "N_SUBGAP_QUAD": N_SUBGAP_QUAD,
        "SPATIAL_GRID_CONVENTION": SPATIAL_GRID_CONVENTION,
        "T_BATH_K": T_BATH_K,
        "TARGET_RESONATOR_FREQUENCIES_GHZ": list(TARGET_RESONATOR_FREQUENCIES_GHZ),
        "TARGET_RESONATOR_LENGTHS_UM": list(TARGET_RESONATOR_LENGTHS_UM),
        "VARIABLE_PIECE_LENGTHS_UM": list(VARIABLE_PIECE_LENGTHS_UM),
    }


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_environment() -> dict[str, object]:
    """Numerical runtime identity that can change a resumed solve."""
    build_dependencies = np.__config__.CONFIG.get("Build Dependencies", {})
    blas = build_dependencies.get("blas", {})
    lapack = build_dependencies.get("lapack", {})
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "BLIS_NUM_THREADS",
            "MKL_CBWR",
            "MKL_DYNAMIC",
            "OPENBLAS_NUM_THREADS",
            "OPENBLAS_CORETYPE",
            "OMP_DYNAMIC",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        },
        "lapack": {
            "name": lapack.get("name"),
            "version": lapack.get("version"),
        },
        "thread_environment": thread_variables,
    }


def _config_digest(config: SweepConfig) -> str:
    return _json_digest(
        {
            "config": config.__dict__,
            "physics_constants": _physics_constants(),
            "physics_revision": _PHYSICS_REV,
            "runtime_environment": _runtime_environment(),
            "source_material_code_sha256": _source_material_code_digest(),
        }
    )


def _point_digest(
    D0: float,
    rate: float,
    center_delta: float,
    sigma_delta: float,
) -> str:
    """Hash exact, round-trip float values instead of lossy id labels."""
    return _json_digest(
        {
            "D0_um2_per_ns": float(D0),
            "source_center_delta": float(center_delta),
            "source_rate_per_ns": float(rate),
            "source_sigma_delta": float(sigma_delta),
        }
    )


def _run_id(
    D0: float,
    rate: float,
    center_delta: float,
    sigma_delta: float,
    config: SweepConfig,
) -> str:
    readable = (
        f"{_PHYSICS_REV}_D0_{D0:g}_rate_{rate:.0e}_center_{center_delta:g}"
        f"_sigma_{sigma_delta:g}_nx{config.NX}_ne{config.NE}_"
        f"dt{config.dt_ns:g}_tmax{config.max_time_ns:g}_"
        f"tol{config.stop_tol:.0e}_snap{config.snapshot_interval_ns:g}_"
        f"point{_point_digest(D0, rate, center_delta, sigma_delta)[:20]}_"
        f"cfg{_config_digest(config)[:12]}"
    )
    return readable.replace("+", "").replace(".", "p")


def _read_exact_csv(
    path: Path,
    expected_fields: list[str],
) -> list[dict[str, str]] | None:
    """Read a CSV only when every row has the current exact schema."""
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header != expected_fields:
                return None
            rows: list[dict[str, str]] = []
            for values in reader:
                if len(values) != len(expected_fields):
                    return None
                rows.append(dict(zip(expected_fields, values, strict=True)))
    except (OSError, UnicodeError, csv.Error):
        return None
    return rows


def _read_attempt_csv(
    path: Path,
    expected_fields: list[str],
) -> list[tuple[str, dict[str, str] | None]] | None:
    """Read aggregate rows while retaining attributable malformed attempts.

    A truncated row that still names its run id invalidates only that run and
    can be removed by ``purge_run_id_rows`` on retry. A row with no run id
    cannot be repaired safely, so fail loud instead of entering a rerun loop
    that can never clean the aggregate.
    """
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header != expected_fields:
                return None
            run_id_index = expected_fields.index("run_id")
            rows: list[tuple[str, dict[str, str] | None]] = []
            for values in reader:
                raw_run_id = values[run_id_index] if len(values) > run_id_index else ""
                run_id = raw_run_id.strip()
                if not run_id:
                    raise SystemExit(
                        f"{path} contains a malformed row with no run_id. "
                        "It cannot be attributed or removed by an idempotent "
                        "retry. Repair/remove that row manually or use a "
                        "fresh output directory."
                    )
                if raw_run_id != run_id:
                    raise SystemExit(
                        f"{path} contains non-canonical run_id {raw_run_id!r}. "
                        "Whitespace-normalizing it would make per-id purge "
                        "non-idempotent. Repair/remove that row manually or "
                        "use a fresh output directory."
                    )
                if len(values) != len(expected_fields):
                    rows.append((run_id, None))
                    continue
                rows.append(
                    (
                        run_id,
                        dict(zip(expected_fields, values, strict=True)),
                    )
                )
    except (OSError, UnicodeError, csv.Error):
        return None
    return rows


def validate_campaign_aggregate_rows(
    path: Path,
    expected_fields: list[str],
) -> None:
    """Fail loud when a fresh-start purge cannot safely attribute every row."""
    if not path.exists():
        return
    attempts = _read_attempt_csv(path, expected_fields)
    if attempts is None:
        raise SystemExit(
            f"{path} cannot be safely parsed with the current UTF-8 aggregate "
            "schema. Repair it manually or use a fresh output directory."
        )


def _numeric_artifact_rows(
    *,
    out_dir: Path,
    run_id: str,
    reference: str,
    prefix: str,
    expected_fields: list[str],
) -> list[dict[str, float]] | None:
    """Read a contained run-local CSV only when every value is finite."""
    reference = reference.strip()
    if not reference or reference != f"{prefix}_{run_id}.csv":
        return None
    try:
        root = out_dir.resolve(strict=True)
        artifact = (out_dir / reference).resolve(strict=True)
    except OSError:
        return None
    if not artifact.is_relative_to(root) or not artifact.is_file():
        return None
    rows = _read_exact_csv(artifact, expected_fields)
    if not rows:
        return None
    numeric_rows: list[dict[str, float]] = []
    for row in rows:
        try:
            numeric_row = {
                field: float(row[field]) for field in expected_fields
            }
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in numeric_row.values()):
            return None
        numeric_rows.append(numeric_row)
    return numeric_rows


def _verify_completed_run_ids(
    summary_path: Path,
    shifts_path: Path | None,
    out_dir: Path | None,
    *,
    summary_fields: list[str],
    shift_fields: list[str],
    trace_fields: list[str],
    profile_fields: list[str],
    expected_run_ids: set[str] | None = None,
    expected_rows: Mapping[str, Mapping[str, float | int]] | None = None,
    stop_tol: float | None = None,
) -> set[str]:
    """Return only run ids with a complete, internally consistent output set.

    The last summary row is the authoritative attempt for a run id. Any
    malformed aggregate row makes the file unverifiable, so resume safely
    recomputes instead of falling back to an older ``completed`` row.
    """
    if shifts_path is None or out_dir is None:
        return set()
    summary_rows = _read_attempt_csv(summary_path, summary_fields)
    shift_rows = _read_attempt_csv(shifts_path, shift_fields)
    if summary_rows is None or shift_rows is None:
        return set()

    aggregate_rows = [*summary_rows, *shift_rows]
    for run_id, _row in aggregate_rows:
        if run_id in {".", ".."} or Path(run_id).name != run_id:
            raise SystemExit(
                f"Aggregate CSV contains invalid run_id {run_id!r}. It "
                "cannot be repaired by an idempotent retry. Rerun with "
                "--no-resume or repair/remove that row."
            )
    known_run_ids = set(expected_run_ids or ())
    known_run_ids.update(run_id for run_id, row in aggregate_rows if row is not None)
    unknown_malformed = {
        run_id
        for run_id, row in aggregate_rows
        if row is None and run_id not in known_run_ids
    }
    if unknown_malformed:
        sample = sorted(unknown_malformed)[0]
        raise SystemExit(
            f"Aggregate CSV contains a malformed row for unknown run_id "
            f"{sample!r}. It cannot be attributed to a retry safely. Rerun "
            "with --no-resume or repair/remove that row."
        )

    latest_summary: dict[str, dict[str, str] | None] = {}
    for run_id, row in summary_rows:
        latest_summary[run_id] = row

    shifts_by_run: dict[str, list[dict[str, str] | None]] = {}
    for run_id, row in shift_rows:
        shifts_by_run.setdefault(run_id, []).append(row)

    completed: set[str] = set()
    for run_id, row in latest_summary.items():
        if row is None:
            continue
        if row["status"].strip() != "completed":
            continue
        if row["converged"].strip().casefold() not in {"true", "1"}:
            continue
        if row["error"].strip():
            continue
        summary_numeric_fields = set(summary_fields) - {
            "run_id",
            "status",
            "converged",
            "error",
            "trace_csv",
            "profile_csv",
        }
        try:
            summary_values = {
                field: float(row[field]) for field in summary_numeric_fields
            }
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in summary_values.values()):
            continue
        try:
            nx_float = float(row["NX"])
            dt_ns = float(row["dt_ns"])
            max_time_ns = float(row["max_time_ns"])
            total_time_ns = float(row["total_time_ns"])
            n_steps_float = float(row["n_steps"])
        except ValueError:
            continue
        if (
            not nx_float.is_integer()
            or nx_float < 2.0
            or dt_ns <= 0.0
            or max_time_ns <= 0.0
            or total_time_ns < 0.0
            or not n_steps_float.is_integer()
            or n_steps_float < 1.0
        ):
            continue
        # The backend advances by ``dt_ns`` except for at most one shortened
        # final step that lands exactly on ``max_time_ns``.  A status row
        # whose step count and elapsed time violate that invariant is not a
        # resumable completion, even when its artifacts are otherwise
        # self-consistent.  Scale the comparison for worst-case accumulated
        # binary64 addition error over many steps.
        scheduled_time_ns = n_steps_float * dt_ns
        previous_step_time_ns = (n_steps_float - 1.0) * dt_ns
        accumulation_rtol = max(
            _CSV_TIME_RTOL,
            8.0 * n_steps_float * float(np.finfo(float).eps),
        )
        time_consistency_tol = accumulation_rtol * max(
            1.0,
            abs(total_time_ns),
            abs(scheduled_time_ns),
            abs(max_time_ns),
        )
        if (
            not math.isfinite(scheduled_time_ns)
            or total_time_ns > max_time_ns + time_consistency_tol
            or total_time_ns <= previous_step_time_ns - time_consistency_tol
            or total_time_ns > scheduled_time_ns + time_consistency_tol
            or (
                not math.isclose(
                    total_time_ns,
                    scheduled_time_ns,
                    rel_tol=accumulation_rtol,
                    abs_tol=time_consistency_tol,
                )
                and not math.isclose(
                    total_time_ns,
                    max_time_ns,
                    rel_tol=accumulation_rtol,
                    abs_tol=time_consistency_tol,
                )
            )
        ):
            continue
        expected_nx = int(nx_float)
        if expected_rows is not None:
            expected = expected_rows.get(run_id)
            if expected is None:
                continue
            expected_matches = True
            for field, expected_value in expected.items():
                try:
                    actual_value = float(row[field])
                except (KeyError, ValueError):
                    expected_matches = False
                    break
                scale = max(1.0, abs(float(expected_value)))
                if not math.isclose(
                    actual_value,
                    float(expected_value),
                    rel_tol=_CSV_TIME_RTOL,
                    abs_tol=_CSV_TIME_RTOL * scale,
                ):
                    expected_matches = False
                    break
            if not expected_matches:
                continue
        if stop_tol is not None:
            residual_fields = ["final_max_dfdt_per_ns"]
            if "final_max_dnphdt_per_ns" in summary_values:
                residual_fields.append("final_max_dnphdt_per_ns")
            if any(
                summary_values[field] < 0.0
                or summary_values[field]
                > float(stop_tol) * (1.0 + _CSV_TIME_RTOL)
                for field in residual_fields
            ):
                continue
        if any(
            summary_values[field] < 0.0
            for field in ("xqp_mean", "xqp_source", "xqp_open_end")
        ):
            continue
        if "wall_seconds" in summary_values and summary_values["wall_seconds"] < 0.0:
            continue

        run_shifts = shifts_by_run.get(run_id, [])
        if len(run_shifts) != len(EXPECTED_RESONATOR_INDICES):
            continue
        if any(shift is None for shift in run_shifts):
            continue
        valid_run_shifts = [shift for shift in run_shifts if shift is not None]
        if any(
            not value.strip()
            for shift in valid_run_shifts
            for value in shift.values()
        ):
            continue
        try:
            indices_float = [
                float(shift["resonator_index"]) for shift in valid_run_shifts
            ]
        except ValueError:
            continue
        if any(
            not math.isfinite(index) or not index.is_integer()
            for index in indices_float
        ):
            continue
        indices = {int(index) for index in indices_float}
        if indices != EXPECTED_RESONATOR_INDICES:
            continue
        labels_match = all(
            shift["resonator_label"].strip()
            == EXPECTED_RESONATOR_LABELS[int(index)]
            for shift, index in zip(valid_run_shifts, indices_float, strict=True)
        )
        if not labels_match:
            continue
        shift_numeric_fields = set(shift_fields) - {"run_id", "resonator_label"}
        try:
            shift_values = [
                float(shift[field])
                for shift in valid_run_shifts
                for field in shift_numeric_fields
            ]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in shift_values):
            continue
        common_numeric_fields = (
            set(summary_numeric_fields)
            & set(shift_numeric_fields)
        ) - {"resonator_index"}
        if any(
            not math.isclose(
                float(shift[field]),
                summary_values[field],
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL
                * max(1.0, abs(summary_values[field])),
            )
            for shift in valid_run_shifts
            for field in common_numeric_fields
        ):
            continue
        delta_values = [
            float(shift["delta_fr_hz_current_weighted"])
            for shift in valid_run_shifts
        ]
        if not math.isclose(
            summary_values["delta_fr_hz_min"],
            min(delta_values),
            rel_tol=_CSV_TIME_RTOL,
            abs_tol=_CSV_TIME_RTOL * max(1.0, abs(min(delta_values))),
        ) or not math.isclose(
            summary_values["delta_fr_hz_max"],
            max(delta_values),
            rel_tol=_CSV_TIME_RTOL,
            abs_tol=_CSV_TIME_RTOL * max(1.0, abs(max(delta_values))),
        ):
            continue
        if "abs_delta_fr_hz_median" in summary_values:
            expected_abs_median = float(np.median(np.abs(delta_values)))
            if not math.isclose(
                summary_values["abs_delta_fr_hz_median"],
                expected_abs_median,
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL * max(1.0, expected_abs_median),
            ):
                continue
        if "Qi_min" in summary_values and "Qi_current_weighted" in shift_fields:
            qi_values = [
                float(shift["Qi_current_weighted"])
                for shift in valid_run_shifts
            ]
            if not math.isclose(
                summary_values["Qi_min"],
                min(qi_values),
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL * max(1.0, abs(min(qi_values))),
            ) or not math.isclose(
                summary_values["Qi_max"],
                max(qi_values),
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL * max(1.0, abs(max(qi_values))),
            ):
                continue

        trace_rows = _numeric_artifact_rows(
            out_dir=out_dir,
            run_id=run_id,
            reference=row["trace_csv"],
            prefix="trace",
            expected_fields=trace_fields,
        )
        if trace_rows is None:
            continue
        trace_times = [trace_row["t_ns"] for trace_row in trace_rows]
        time_tol = _CSV_TIME_RTOL * max(1.0, abs(total_time_ns))
        if not math.isclose(
            trace_times[-1],
            total_time_ns,
            rel_tol=_CSV_TIME_RTOL,
            abs_tol=_CSV_TIME_RTOL,
        ):
            continue
        final_trace = trace_rows[-1]
        trace_summary_fields = {
            "max_dfdt_per_ns": "final_max_dfdt_per_ns",
            "max_dnphdt_per_ns": "final_max_dnphdt_per_ns",
            "xqp_mean": "xqp_mean",
            "xqp_source": "xqp_source",
            "xqp_open_end": "xqp_open_end",
            "nph_mean": "nph_mean",
            "nph_max": "nph_max",
        }
        if any(
            not math.isclose(
                final_trace[trace_field],
                summary_values[summary_field],
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL
                * max(1.0, abs(summary_values[summary_field])),
            )
            for trace_field, summary_field in trace_summary_fields.items()
            if trace_field in final_trace and summary_field in summary_values
        ):
            continue
        if not math.isclose(
            trace_times[0],
            0.0,
            rel_tol=0.0,
            abs_tol=time_tol,
        ) or any(
            later < earlier - time_tol
            for earlier, later in pairwise(trace_times)
        ):
            continue

        profile_rows = _numeric_artifact_rows(
            out_dir=out_dir,
            run_id=run_id,
            reference=row["profile_csv"],
            prefix="profile",
            expected_fields=profile_fields,
        )
        if profile_rows is None or len(profile_rows) != expected_nx:
            continue
        position_tol = _CSV_TIME_RTOL * max(1.0, abs(LENGTH_UM))
        profile_valid = True
        for index, profile_row in enumerate(profile_rows):
            expected_x = (index + 0.5) * LENGTH_UM / expected_nx
            if not math.isclose(
                profile_row["x_um"],
                expected_x,
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=position_tol,
            ) or profile_row["xqp"] < 0.0:
                profile_valid = False
                break
        if not profile_valid:
            continue
        profile_xqp = [profile_row["xqp"] for profile_row in profile_rows]
        profile_summary = {
            "xqp_mean": float(np.mean(profile_xqp)),
            "xqp_source": profile_xqp[0],
            "xqp_open_end": profile_xqp[-1],
        }
        if any(
            not math.isclose(
                summary_values[field],
                value,
                rel_tol=_CSV_TIME_RTOL,
                abs_tol=_CSV_TIME_RTOL * max(1.0, abs(value)),
            )
            for field, value in profile_summary.items()
        ):
            continue
        completed.add(run_id)
    return completed


def _completed_run_ids(
    summary_path: Path,
    shifts_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    expected_run_ids: set[str] | None = None,
    expected_rows: Mapping[str, Mapping[str, float | int]] | None = None,
    stop_tol: float | None = None,
) -> set[str]:
    """Run ids whose latest attempt has all six verified output rows/files."""
    return _verify_completed_run_ids(
        summary_path,
        shifts_path,
        out_dir,
        summary_fields=SUMMARY_FIELDS,
        shift_fields=SHIFT_FIELDS,
        trace_fields=TRACE_FIELDS,
        profile_fields=PROFILE_FIELDS,
        expected_run_ids=expected_run_ids,
        expected_rows=expected_rows,
        stop_tol=stop_tol,
    )


def _write_metadata(out_dir: Path, config: SweepConfig) -> Path:
    config_sha256 = _config_digest(config)
    source_material_code_sha256 = _source_material_code_digest()
    metadata = {
        "physics_revision": _PHYSICS_REV,
        "config_sha256": config_sha256,
        "physics_constants": _physics_constants(),
        "runtime_environment": _runtime_environment(),
        "source_material_code_sha256": source_material_code_sha256,
        "preset": config.name,
        "target_resonator_frequencies_ghz": TARGET_RESONATOR_FREQUENCIES_GHZ,
        "target_resonator_lengths_um": TARGET_RESONATOR_LENGTHS_UM,
        "fixed_resonator_length_um": FIXED_RESONATOR_LENGTH_UM,
        "coupling_length_um": COUPLING_LENGTH_UM,
        "variable_piece_lengths_um": VARIABLE_PIECE_LENGTHS_UM,
        "kinetic_inductance_fraction": KINETIC_INDUCTANCE_FRACTION,
        "length_um": LENGTH_UM,
        "T_bath_K": T_BATH_K,
        "energy_max_factor": ENERGY_MAX_FACTOR,
        "n_subgap_quad": N_SUBGAP_QUAD,
        "spatial_grid_convention": SPATIAL_GRID_CONVENTION,
        "spatial_grid_note": (
            "Uniform finite-volume cell centers x_i=(i+1/2)L/NX with "
            "dx=L/NX. Diffusion, source volume, profile validation, and "
            "spatial observables all use the same equal-cell measure."
        ),
        "config": config.__dict__,
        "source_note": (
            "Source is normalized as a local source-cell x_qp generation rate. "
            "CSV fields estimate QP/s using the source-cell volume and the "
            "Al rho_F database value as eV^-1 m^-3."
        ),
    }
    path = out_dir / f"metadata_{_PHYSICS_REV}_{config_sha256[:12]}.json"
    fd, temporary_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True, allow_nan=False)
            fp.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "overnight", "calibrated"),
        default="overnight",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "prelim_spatial_overnight",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _run_campaign(
    args: argparse.Namespace,
    config: SweepConfig,
    out_dir: Path,
) -> None:
    summary_path = out_dir / "summary.csv"
    shifts_path = out_dir / "resonator_shifts.csv"
    all_combinations = list(
        product(
            config.D0_values,
            config.source_rates_per_ns,
            config.source_centers_delta,
            config.source_sigmas_delta,
        )
    )
    combinations = (
        all_combinations
        if args.max_runs is None
        else all_combinations[: args.max_runs]
    )
    expected_run_ids = {
        _run_id(D0, rate, center_delta, sigma_delta, config)
        for D0, rate, center_delta, sigma_delta in all_combinations
    }
    selected_run_ids = {
        _run_id(D0, rate, center_delta, sigma_delta, config)
        for D0, rate, center_delta, sigma_delta in combinations
    }
    expected_rows: dict[str, dict[str, float | int]] = {}
    for D0, rate, center_delta, sigma_delta in all_combinations:
        run_id = _run_id(D0, rate, center_delta, sigma_delta, config)
        dx_um = LENGTH_UM / config.NX
        dt_run = min(config.dt_ns, CFL_TARGET * dx_um * dx_um / D0)
        expected_rows[run_id] = {
            "NX": config.NX,
            "D0_um2_per_ns": D0,
            "source_rate_per_ns": rate,
            **_source_calibration(config, rate),
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
            "dt_ns": dt_run,
            "max_time_ns": config.max_time_ns,
        }
    if len(expected_run_ids) != len(all_combinations):
        raise RuntimeError(
            "Spatial campaign generated duplicate run ids; every exact sweep "
            "point must have a unique persistent identity."
        )
    require_matching_header(summary_path, SUMMARY_FIELDS)
    require_matching_header(shifts_path, SHIFT_FIELDS)
    if args.no_resume:
        # Validate both shared aggregates before deleting any selected rows
        # or artifacts. Rows without one exact canonical id cannot be
        # attributed to a campaign and selected purge cannot repair them.
        validate_campaign_aggregate_rows(summary_path, SUMMARY_FIELDS)
        validate_campaign_aggregate_rows(shifts_path, SHIFT_FIELDS)
        # Restart only the selected exact run ids. The aggregate files are
        # intentionally shared by config-addressed presets, so wholesale
        # truncation would destroy other campaigns' resume state.
        purge_run_ids_rows(summary_path, selected_run_ids)
        purge_run_ids_rows(shifts_path, selected_run_ids)
        purge_run_artifacts(out_dir, selected_run_ids)
        completed: set[str] = set()
    else:
        completed = _completed_run_ids(
            summary_path,
            shifts_path,
            out_dir,
            expected_run_ids=expected_run_ids,
            expected_rows=expected_rows,
            stop_tol=config.stop_tol,
        )
    # Header/resume validation comes first: a rejected invocation must not
    # rewrite provenance. Config-addressed metadata lets multiple presets
    # coexist safely in the shared output directory.
    _write_metadata(out_dir, config)

    start_wall = time.monotonic()
    wall_limit_s = None if args.wall_hours is None else args.wall_hours * 3600.0
    print(
        f"Preset {config.name}: {len(combinations)} queued runs, "
        f"dt={config.dt_ns:g} ns, NX={config.NX}, NE={config.NE}.",
        flush=True,
    )

    for run_number, (D0, rate, center_delta, sigma_delta) in enumerate(combinations, start=1):
        if wall_limit_s is not None and time.monotonic() - start_wall >= wall_limit_s:
            print("Wall-time limit reached; stopping cleanly.", flush=True)
            break

        run_id = _run_id(D0, rate, center_delta, sigma_delta, config)
        if run_id in completed:
            print(f"[{run_number}/{len(combinations)}] skip completed {run_id}", flush=True)
            continue

        print(f"[{run_number}/{len(combinations)}] start {run_id}", flush=True)
        # Idempotent re-run: drop any stale rows from a prior failed or
        # non-converged attempt of this run id (2026-07-20 round-4 review).
        purge_run_id_rows(summary_path, run_id)
        purge_run_id_rows(shifts_path, run_id)
        trace_path = out_dir / f"trace_{run_id}.csv"
        profile_path = out_dir / f"profile_{run_id}.csv"
        source_calibration = _source_calibration(config, rate)

        # Bound the transport diffusion number per D0 (see CFL_TARGET): the
        # spatial Crank-Nicolson step clip-raises on an under-resolved dt.
        dx_um = LENGTH_UM / config.NX
        dt_run = min(config.dt_ns, CFL_TARGET * dx_um * dx_um / D0)

        base_row: dict[str, object] = {
            "run_id": run_id,
            "NX": config.NX,
            "D0_um2_per_ns": D0,
            "source_rate_per_ns": rate,
            **source_calibration,
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
            "dt_ns": dt_run,
            "max_time_ns": config.max_time_ns,
            "trace_csv": trace_path.name,
            "profile_csv": profile_path.name,
        }
        shift_base_row: dict[str, object] = {
            "run_id": run_id,
            "D0_um2_per_ns": D0,
            "source_rate_per_ns": rate,
            **source_calibration,
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
        }

        try:
            state = _build_state(config, D0)
            f_ref = _mean_f(state)
            flux = _source_flux(
                state,
                local_xqp_generation_rate_per_ns=rate,
                center_delta=center_delta,
                sigma_delta=sigma_delta,
            )
            backend = T3SpatialBackend()
            result = backend.run_until_steady_state(
                state,
                dt=dt_run,
                max_time=config.max_time_ns,
                external_flux=flux,
                stop_tol=config.stop_tol,
                snapshot_interval=config.snapshot_interval_ns,
                observables=_trace_observables(),
            )
            _write_trace(trace_path, result.snapshots)
            _write_profile(profile_path, result.state)
            shift_rows = _resonator_shifts(result.state, f_ref)
            for shift_row in shift_rows:
                _append_csv(
                    shifts_path,
                    {
                        **shift_base_row,
                        **shift_row,
                    },
                    SHIFT_FIELDS,
                )

            final_obs = result.snapshots[-1].observables
            delta_values = [row["delta_fr_hz_current_weighted"] for row in shift_rows]
            # Don't overload "completed": a run that hit max_time without
            # meeting stop_tol is not converged, and resume gates on
            # status == "completed" — marking it completed would both
            # misrepresent the row and permanently exclude the case from
            # re-running on resume (2026-07-19 audit).
            summary_row = {
                **base_row,
                "status": "completed" if result.converged else "max_time_reached",
                "total_time_ns": result.total_time,
                "n_steps": result.n_steps,
                "converged": result.converged,
                "final_max_dfdt_per_ns": result.snapshots[-1].max_rate,
                **final_obs,
                "delta_fr_hz_min": min(delta_values),
                "delta_fr_hz_max": max(delta_values),
                "error": "",
            }
            _append_csv(summary_path, summary_row, SUMMARY_FIELDS)
            print(
                f"[{run_number}/{len(combinations)}] done {run_id}: "
                f"xqp_mean={final_obs['xqp_mean']:.3e}, "
                f"delta_fr=[{min(delta_values):.3e}, {max(delta_values):.3e}] Hz, "
                f"converged={result.converged}",
                flush=True,
            )
        except Exception as exc:
            _append_csv(
                summary_path,
                {
                    **base_row,
                    "status": "failed",
                    "error": repr(exc),
                },
                SUMMARY_FIELDS,
            )
            print(f"[{run_number}/{len(combinations)}] FAILED {run_id}: {exc!r}", flush=True)


def main() -> None:
    args = _parse_args()
    args.max_runs, args.wall_hours = validate_campaign_limits(
        args.max_runs,
        args.wall_hours,
    )
    config_by_name = {
        "smoke": SMOKE_CONFIG,
        "overnight": OVERNIGHT_CONFIG,
        "calibrated": CALIBRATED_CONFIG,
    }
    config = config_by_name[args.preset]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_campaign_lock(
        out_dir,
        owner_label=f"spatial:{config.name}",
        config_digest=_config_digest(config),
    )
    try:
        _run_campaign(args, config, out_dir)
    finally:
        release_campaign_lock(lock)


if __name__ == "__main__":
    main()
