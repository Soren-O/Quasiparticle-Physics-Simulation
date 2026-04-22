"""Numerical solvers (tier-agnostic).

Ported (Gate 2):
- anderson.py — Type-I Anderson extrapolation helper
- etd.py — ETD1 exponential-Euler stepper (ETD2 coming per Build Handoff)
- newton_steady_state.py — Newton with analytical Jacobian for f
- picard.py — generic Picard + Anderson fixed-point iteration

Planned (New Framework Plan §5, later in Gate 2):
- coupled_newton.py — joint (f, n_ph) Newton (unlocks F23 ratio=10)
- crank_nicolson.py — Crank–Nicolson diffusion substep
- spectral_flow_tvd.py — TVD finite volume
- ssprk.py — SSPRK(2,2) time stepper
"""

# Note: newton_solve_f is *not* re-exported here because it imports
# from qpsim.collisions.phonon, and qpsim.collisions.phonon imports
# etd1_step from qpsim.solvers.etd. Pulling it into this __init__
# would trigger a circular import. Use:
#     from qpsim.solvers.newton_steady_state import newton_solve_f

from qpsim.solvers.anderson import anderson_extrapolate
from qpsim.solvers.etd import etd1_step
from qpsim.solvers.picard import PicardInfo, picard_iterate

__all__ = [
    "PicardInfo",
    "anderson_extrapolate",
    "etd1_step",
    "picard_iterate",
]
