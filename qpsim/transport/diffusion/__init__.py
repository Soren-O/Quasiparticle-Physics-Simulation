"""Diffusion-model dispatch.

Modules planned (New Framework Plan §5):
- base.py — DiffusionModel enum
- legacy.py — D(E) = D_0 √(1 − (Δ/E)²) (Fischer-matching)
- boltzmann.py — D(E) = D_0 [1 − (Δ/E)²]
- usadel.py — (D/N_1²) ∇ [N_1² ∇ f]
"""
