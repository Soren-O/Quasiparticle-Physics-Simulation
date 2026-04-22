"""Phonon hierarchy. Orthogonal to the electronic tier.

Modules planned (New Framework Plan §5):
- state.py — PhononState dataclass (shape (N_branch, N_omega, N_spatial))
- ph0_local.py — local bath with escape time (v1 target)
- ph1_ballistic.py — ballistic with escape (v2)
- ph2_diffusive.py — substrate diffusion (v3; skeleton only)

See docs/Phonon_Model_Decisions.md for the Gate 0 decisions.
"""
