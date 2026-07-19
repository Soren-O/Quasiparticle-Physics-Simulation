"""Post-processing observables.

Exports are loaded lazily so importing one lightweight observable does not pull
in every spectral, fitting, or spatial-response dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GapSuppressionResult": (
        "qpsim.observables.gap_suppression",
        "GapSuppressionResult",
    ),
    "SpatialAcResponse": (
        "qpsim.observables.spatial_ac_response",
        "SpatialAcResponse",
    ),
    "compute_ac_conductivity": (
        "qpsim.observables.ac_conductivity",
        "compute_ac_conductivity",
    ),
    "compute_current_weighted_ac_response": (
        "qpsim.observables.spatial_ac_response",
        "compute_current_weighted_ac_response",
    ),
    "compute_frequency_shift": (
        "qpsim.observables.frequency_shift",
        "compute_frequency_shift",
    ),
    "compute_gap_suppression": (
        "qpsim.observables.gap_suppression",
        "compute_gap_suppression",
    ),
    "compute_gap_suppression_direct": (
        "qpsim.observables.gap_suppression",
        "compute_gap_suppression_direct",
    ),
    "compute_quality_factor": (
        "qpsim.observables.quality_factor",
        "compute_quality_factor",
    ),
    "delta_suppression_from_distribution_direct": (
        "qpsim.observables.gap_suppression",
        "delta_suppression_from_distribution_direct",
    ),
    "edge_samples_from_centers": (
        "qpsim.observables.gap_suppression",
        "edge_samples_from_centers",
    ),
    "effective_phonon_temperature": (
        "qpsim.observables.effective_temperature",
        "effective_phonon_temperature",
    ),
    "fermi_dirac_distribution": (
        "qpsim.observables.gap_suppression",
        "fermi_dirac_distribution",
    ),
    "gap_from_distribution_direct": (
        "qpsim.observables.gap_suppression",
        "gap_from_distribution_direct",
    ),
    "gap_integral_from_distribution_direct": (
        "qpsim.observables.gap_suppression",
        "gap_integral_from_distribution_direct",
    ),
    "gap_suppression_from_deltas": (
        "qpsim.observables.gap_suppression",
        "gap_suppression_from_deltas",
    ),
    "gap_suppression_from_integrals_direct": (
        "qpsim.observables.gap_suppression",
        "gap_suppression_from_integrals_direct",
    ),
    "gap_suppression_ratio_from_integrals": (
        "qpsim.observables.gap_suppression",
        "gap_suppression_ratio_from_integrals",
    ),
    "left_edges_from_centers": (
        "qpsim.observables.gap_suppression",
        "left_edges_from_centers",
    ),
    "qp_fraction": (
        "qpsim.observables.density",
        "qp_fraction",
    ),
    "qp_fraction_paper": (
        "qpsim.observables.density",
        "qp_fraction_paper",
    ),
    "qp_number_density": (
        "qpsim.observables.density",
        "qp_number_density",
    ),
    "thermal_gap_integral_direct": (
        "qpsim.observables.gap_suppression",
        "thermal_gap_integral_direct",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
