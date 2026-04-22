"""Tier-specific solvers. Each implements the Backend protocol.

Modules planned (New Framework Plan §5):
- base.py — Backend protocol, BackendState union
- t3_diffusion.py — T3 diffusion-scalar (v1 target; port from current ScalarBackend)
- t2_kinetic.py — T2 scalar-kinetic with angular axis (v2)
- t1_two_component.py — T1 (f_L, f_T) two-component (v3)
- reductions.py — tier-to-tier state conversions (T1→T2, T2→T3, …)
"""
