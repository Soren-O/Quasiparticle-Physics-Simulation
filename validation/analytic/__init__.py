"""Analytic fixed-point tests.

- test_detailed_balance.py — thermal (f, n_ph) is a fixed point of the
  collision integral
- test_mattis_bardeen_thermal.py — σ at thermal → textbook BCS
  coherence-factor forms
- test_gap_equation_equilibrium.py — gap self-consistency recovers Δ_eq(T_B)
  to rel < 1e-8 at T_bath = 0.1 K and rel < 9e-5 at T_c/2 (NE = 1620); the
  achieved errors are 4.6e-10 and 3.0e-5 (Brent xtol floors the cold case,
  cell-constant representation error the hot one)

The uniform-Δ diffusion reduction is *not* in this package: it lives in
``validation/diffusion_operators/{uniform_gap_packet,test_uniform_gap_packet}.py``
(D_eff vs analytic per model, rel < 1e-9). Note that "reduces to D∇² for
uniform Δ" identifies A2 ``(p, q) = (2, 2)`` — the rejected diagnostic —
not the dirty-limit Usadel operator A1 ``(1, 0)``, whose uniform-gap rate is
``D_0 / N_1(E)`` (see ``qpsim.transport.diffusion.base``).
"""
