"""Transport operators.

Implemented:
- diffusion/ — DiffusionModel operator family (A1/A2/B/C) + dressing
  helpers; see qpsim.transport.diffusion.base. Legacy aliases:
  LEGACY→C, BOLTZMANN→B, USADEL→A2 (USADEL is the rejected diagnostic
  A2, not the dirty-limit Usadel operator A1).

Modules planned (New Framework Plan §5):
- ballistic.py — v_g p̂·∇ streaming (T1/T2)
- spectral_flow.py — (Δ/E) Δ̇ ∂_E conservation law
- gap_gradient_force.py — (Δ/E) v_F p̂·∇Δ ∂_E f (T1/T2)
"""
