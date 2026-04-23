"""Post-processing observables.

Ported / written (Gate 2 + Gate 3 + Gate 4.5):
- ac_conductivity.py — Mattis–Bardeen σ₁, σ₂ (normalized)
- quality_factor.py — Q_i = σ₂ / (α σ₁)
- frequency_shift.py — fractional δω/ω vs a reference occupation
- density.py — qp_number_density, qp_fraction (Fischer x_qp convention)
- gap_suppression.py — δΔ / Δ from solve_gap compared against Δ_eq
- effective_temperature.py — T_* from weighted BE fit to n_ph (F23 Eq. 36)
"""

from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.observables.density import qp_fraction, qp_number_density
from qpsim.observables.effective_temperature import effective_phonon_temperature
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.gap_suppression import (
    GapSuppressionResult,
    compute_gap_suppression,
    gap_suppression_from_deltas,
)
from qpsim.observables.quality_factor import compute_quality_factor

__all__ = [
    "GapSuppressionResult",
    "compute_ac_conductivity",
    "compute_frequency_shift",
    "compute_gap_suppression",
    "compute_quality_factor",
    "effective_phonon_temperature",
    "gap_suppression_from_deltas",
    "qp_fraction",
    "qp_number_density",
]
