"""Orchestration — user-facing entry points.

- steady_state.py — solve_steady_state (thermal-phonon + finite-τ_l paths)
- nbar_loop.py — self-consistent n̄(P_read) fixed-point iteration
- transient.py — run_time_dependent driver for f(E, t) under frozen n_ph
- rate_equation.py — Marchegiani rate-equation module
"""

# Note: run_time_dependent / TransientResult / TransientSnapshot are *not*
# re-exported here because qpsim.services.transient imports from
# qpsim.backends.diffusion, which in turn imports from
# qpsim.services.steady_state. Pulling transient into this __init__ would
# create a circular import at package load time. Import directly:
#     from qpsim.services.transient import run_time_dependent

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
