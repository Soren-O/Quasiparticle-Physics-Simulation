"""Pure physics: spectral, kernels, gap equation.

Ported (Gate 2):
- spectral.py — BCS/Dynes DOS, K±, thermal_qp_weights, SpectralContext
- kernels.py — K₀ˢ / K₀ʳ scattering & recombination kernels, thermal_phonon_occupation
- gap_equation.py — BCS calibration + reference-subtracted runtime solve

Planned (New Framework Plan §5):
- phonon_escape.py — τ_l(ω) acoustic-escape form + constant variant (Gate 2+)
- mattis_bardeen.py — σ_1, σ_2 (Gate 2+)
"""

from qpsim.physics.gap_equation import GapCalibration, calibrate_gap, solve_gap
from qpsim.physics.kernels import (
    recombination_kernel,
    recombination_kernel_base,
    scattering_kernel,
    scattering_kernel_base,
    thermal_phonon_occupation,
)
from qpsim.physics.spectral import (
    SpectralContext,
    bcs_density_of_states,
    coherence_factor_minus,
    coherence_factor_plus,
    dynes_density_of_states,
    thermal_qp_weights,
)

__all__ = [
    "GapCalibration",
    "SpectralContext",
    "bcs_density_of_states",
    "calibrate_gap",
    "coherence_factor_minus",
    "coherence_factor_plus",
    "dynes_density_of_states",
    "recombination_kernel",
    "recombination_kernel_base",
    "scattering_kernel",
    "scattering_kernel_base",
    "solve_gap",
    "thermal_phonon_occupation",
    "thermal_qp_weights",
]
