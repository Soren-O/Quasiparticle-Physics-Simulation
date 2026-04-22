"""Orchestration — user-facing entry points.

Ported (Gate 2):
- steady_state.py — solve_steady_state (thermal-phonon + finite-τ_l paths)

Planned (New Framework Plan §5):
- transient.py — run_time_dependent
- parametric_sweep.py — sweep over (T_B, n̄, P_read, …)
- rate_equation.py — Marchegiani rate-equation module
"""

from qpsim.services.steady_state import solve_steady_state

__all__ = ["solve_steady_state"]
