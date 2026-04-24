"""Pure physics: spectral, kernels, gap equation, phonon escape, pair breaking.

Ported/written (Gate 2):
- spectral.py — BCS/Dynes DOS, K±, thermal_qp_weights, SpectralContext
- kernels.py — K₀ˢ / K₀ʳ scattering & recombination kernels, thermal_phonon_occupation
- gap_equation.py — BCS calibration + reference-subtracted runtime solve
- phonon_escape.py — τ_l(ω) builders (constant and acoustic-escape modes)

Added (Gate 4.5 closure):
- kaplan_pair_breaking.py — τ_PB(Ω) per Kaplan 1976 (pair-breaking
  phonon lifetime, distinct from the thin-film escape time τ_l of
  Kaplan 1979).

Planned (New Framework Plan §5):
- mattis_bardeen.py — σ_1, σ_2 (currently under qpsim.observables.ac_conductivity)
"""

from qpsim.physics.gap_equation import GapCalibration, calibrate_gap, solve_gap
from qpsim.physics.kaplan_pair_breaking import (
    kaplan_S_plus,
    kaplan_S_plus_numerical,
    tau_PB_inverse_Hz,
)
from qpsim.physics.kernels import (
    recombination_kernel,
    recombination_kernel_base,
    scattering_kernel,
    scattering_kernel_base,
    thermal_phonon_occupation,
)
from qpsim.physics.phonon_escape import acoustic_escape_tau_l, constant_tau_l
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
    "acoustic_escape_tau_l",
    "bcs_density_of_states",
    "calibrate_gap",
    "coherence_factor_minus",
    "coherence_factor_plus",
    "constant_tau_l",
    "dynes_density_of_states",
    "kaplan_S_plus",
    "kaplan_S_plus_numerical",
    "recombination_kernel",
    "recombination_kernel_base",
    "scattering_kernel",
    "scattering_kernel_base",
    "solve_gap",
    "tau_PB_inverse_Hz",
    "thermal_phonon_occupation",
    "thermal_qp_weights",
]
