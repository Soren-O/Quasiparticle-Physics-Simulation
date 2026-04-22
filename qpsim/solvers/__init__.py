"""Numerical solvers (tier-agnostic).

Modules planned (New Framework Plan §5):
- newton_steady_state.py — Newton with analytic Jacobian for f
- picard.py — Picard outer loop for (f, n_ph)
- anderson.py — Anderson acceleration helper
- coupled_newton.py — NEW: joint (f, n_ph) Newton (resolves F23 τ_l/τ_PB=10 case)
- crank_nicolson.py — C–N diffusion substep
- spectral_flow_tvd.py — TVD finite volume (van Leer 1979, Ch2)
- ssprk.py — SSPRK(2,2) time stepper (Gottlieb 2001)
"""
