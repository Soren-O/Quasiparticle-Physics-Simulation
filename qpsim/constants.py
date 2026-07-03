"""Shared physical constants for the qpsim package."""

import math

# Boltzmann constant in μeV/K.
# k_B = 8.617333262145e-5 eV/K = 86.17333262145 μeV/K.
KB_UEV_PER_K = 86.17333262145

# Reduced Planck constant in μeV·ns.
# ℏ = 1.054571817e-34 J·s = 6.582119569e-10 eV·s = 0.6582119569 μeV·ns.
HBAR_UEV_NS = 0.6582119569

# h/k_B in K/Hz — the GHz↔Kelvin conversion the M25 layer runs on.
# Derived from the two constants above (h = 2πℏ; the 1e-9 turns
# μeV·ns into μeV/Hz): ≈ 4.799243e-11, the literal several M25
# validation modules and tests carry locally.
H_OVER_KB_K_PER_HZ = 2.0 * math.pi * HBAR_UEV_NS / KB_UEV_PER_K * 1e-9
