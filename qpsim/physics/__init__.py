"""Pure physics: spectral functions (BCS DOS, coherence factors), kernels (K₀ˢ/K₀ʳ), gap equation, Mattis–Bardeen, τ_l acoustic-escape form.

Modules planned (New Framework Plan §5):
- spectral.py — N_1(E), K±, coherence factors, SpectralContext
- kernels.py — K_0^s, K_0^r, photon kernels
- coherence.py — coherence factors (may fold into spectral.py)
- gap_equation.py — calibrate_gap, solve_gap (Brent on reference-subtracted BCS)
- phonon_escape.py — τ_l(ω) acoustic-escape form + constant variant
- mattis_bardeen.py — σ_1, σ_2
"""
