"""Collision integrals (tier-agnostic where possible).

Ported (Gate 2):
- phonon.py — electron–phonon J_1^eph; thermal-bath and dynamic-n_ph paths
- sub_gap_photon.py — ω₀ < 2Δ single-mode photon scattering
- pair_breaking_photon.py — ω_PB > 2Δ pair-breaking photon (scatter + gen + rec)

Planned:
- external_generation.py — G_ext(E, r, t) external QP source (no old-code source)
"""

from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    CoherenceAssignment,
    apply_phonon_collision,
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates

__all__ = [
    "CoherenceAssignment",
    "apply_phonon_collision",
    "build_phonon_frequency_map",
    "build_recombination_kernel_base",
    "build_recombination_kernel_phonon_side",
    "build_scattering_kernel_base",
    "build_scattering_kernel_phonon_side",
    "compute_phonon_source_sink",
    "pair_breaking_photon_collision_rates",
    "phonon_collision_rates",
    "phonon_occupation_matrices_from_state",
    "sub_gap_photon_collision_rates",
]
