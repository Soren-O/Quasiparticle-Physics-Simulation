"""Post-processing observables.

Ported / written (Gate 2 + Gate 3):
- ac_conductivity.py — Mattis–Bardeen σ₁, σ₂ (normalized)
- quality_factor.py — Q_i = σ₂ / (α σ₁)
- frequency_shift.py — fractional δω/ω vs a reference occupation
- density.py — qp_number_density, qp_fraction (Fischer x_qp convention)

Planned (New Framework Plan §5):
- effective_temperature.py — T_* from fitting n_ph to a Bose-Einstein
- gap_suppression.py — δΔ / Δ from solve_gap compared against Δ_eq
"""

from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.observables.density import qp_fraction, qp_number_density
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor

__all__ = [
    "compute_ac_conductivity",
    "compute_frequency_shift",
    "compute_quality_factor",
    "qp_fraction",
    "qp_number_density",
]
