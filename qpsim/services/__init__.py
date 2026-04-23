"""Orchestration — user-facing entry points.

Ported (Gate 2):
- steady_state.py — solve_steady_state (thermal-phonon + finite-τ_l paths)

Ported (Gate 3):
- nbar_loop.py — self-consistent n̄(P_read) fixed-point iteration

Planned (New Framework Plan §5):
- transient.py — run_time_dependent
- parametric_sweep.py — sweep over (T_B, n̄, P_read, …)
- rate_equation.py — Marchegiani rate-equation module
"""

from qpsim.services.nbar_loop import (
    NbarLoopIteration,
    NbarLoopResult,
    dbm_to_uev_per_ns,
    solve_nbar_loop,
)
from qpsim.services.steady_state import solve_steady_state

__all__ = [
    "NbarLoopIteration",
    "NbarLoopResult",
    "dbm_to_uev_per_ns",
    "solve_nbar_loop",
    "solve_steady_state",
]
