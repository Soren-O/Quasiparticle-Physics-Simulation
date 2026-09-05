"""Transport operators.

- diffusion/ — DiffusionModel operator family (A1/A1P/A2/B/C) + dressing
  helpers; see qpsim.transport.diffusion.base. A1 = (1, 0) is the
  dirty-limit default (undressed energy-channel flux); A1P = (1, 2)
  carries the transverse N_1^2 dressing as a diagnostic. Legacy aliases:
  LEGACY→C, BOLTZMANN→B, USADEL→A2 (USADEL is the diagnostic A2, not the
  dirty-limit Usadel operator A1).
- spatial_operator.py — the per-energy finite-volume operator on a cell
  mask of any dimensionality: the stencil an arbitrary mask produces and
  the per-face boundary conditions the active region inherits from the
  device rim.
- spatial_transport.py — the conservative Crank-Nicolson transport step on
  the mask, one operator per energy bin, subcycled under a monotonicity
  bound, with a bounded factor cache keyed on the active mask.
- interface.py — Kupriyanov-Lukichev face conductances where the gap steps
  between neighbouring cells, cached per gap pair.
"""
