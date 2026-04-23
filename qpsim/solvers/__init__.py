"""Numerical solvers (tier-agnostic).

Ported / written (Gate 2):
- anderson.py — Type-I Anderson extrapolation helper
- coupled_newton.py — joint (f, n_ph) Newton (unlocks F23 ratio=10)
- crank_nicolson.py — Crank–Nicolson operator builder for parabolic solves
- etd.py — ETD1 (exponential Euler) and ETD2 (Heun-type second-order)
- newton_steady_state.py — Newton with analytical Jacobian for f
- picard.py — generic Picard + Anderson fixed-point iteration
- spectral_flow_tvd.py — TVD finite-volume advection for spectral-flow
- ssprk.py — SSPRK(2,2) second-order TVD time stepper
"""

# Note: newton_solve_f and coupled_newton_solve are *not* re-exported
# here because both import from qpsim.collisions.phonon, and
# qpsim.collisions.phonon imports etd2_step from qpsim.solvers.etd.
# Pulling either into this __init__ would trigger a circular import.
# Use the submodule directly:
#     from qpsim.solvers.newton_steady_state import newton_solve_f
#     from qpsim.solvers.coupled_newton import coupled_newton_solve

from qpsim.solvers.anderson import anderson_extrapolate
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.solvers.etd import etd1_step, etd2_step
from qpsim.solvers.picard import PicardInfo, picard_iterate
from qpsim.solvers.spectral_flow_tvd import advect_spectral_flow
from qpsim.solvers.ssprk import ssprk22_step

__all__ = [
    "PicardInfo",
    "advect_spectral_flow",
    "anderson_extrapolate",
    "build_cn_operators",
    "etd1_step",
    "etd2_step",
    "picard_iterate",
    "ssprk22_step",
]
