"""Pure physics: spectral, kernels, gap equation, phonon escape, pair breaking.

Exports are loaded lazily so importing one physics helper does not eagerly import
all SciPy-backed helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GapCalibration": ("qpsim.physics.gap_equation", "GapCalibration"),
    "SpectralContext": ("qpsim.physics.spectral", "SpectralContext"),
    "acoustic_escape_tau_l": ("qpsim.physics.phonon_escape", "acoustic_escape_tau_l"),
    "bcs_density_of_states": ("qpsim.physics.spectral", "bcs_density_of_states"),
    "build_tau_l": ("qpsim.physics.phonon_escape", "build_tau_l"),
    "calibrate_gap": ("qpsim.physics.gap_equation", "calibrate_gap"),
    "coherence_factor_minus": ("qpsim.physics.spectral", "coherence_factor_minus"),
    "coherence_factor_plus": ("qpsim.physics.spectral", "coherence_factor_plus"),
    "constant_tau_l": ("qpsim.physics.phonon_escape", "constant_tau_l"),
    "dynes_density_of_states": ("qpsim.physics.spectral", "dynes_density_of_states"),
    "kaplan_S_plus": ("qpsim.physics.kaplan_pair_breaking", "kaplan_S_plus"),
    "kaplan_S_plus_numerical": (
        "qpsim.physics.kaplan_pair_breaking",
        "kaplan_S_plus_numerical",
    ),
    "recombination_kernel": ("qpsim.physics.kernels", "recombination_kernel"),
    "recombination_kernel_base": (
        "qpsim.physics.kernels",
        "recombination_kernel_base",
    ),
    "scattering_kernel": ("qpsim.physics.kernels", "scattering_kernel"),
    "scattering_kernel_base": ("qpsim.physics.kernels", "scattering_kernel_base"),
    "solve_gap": ("qpsim.physics.gap_equation", "solve_gap"),
    "tau_PB_inverse_Hz": (
        "qpsim.physics.kaplan_pair_breaking",
        "tau_PB_inverse_Hz",
    ),
    "thermal_phonon_occupation": (
        "qpsim.physics.kernels",
        "thermal_phonon_occupation",
    ),
    "thermal_qp_weights": ("qpsim.physics.spectral", "thermal_qp_weights"),
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
