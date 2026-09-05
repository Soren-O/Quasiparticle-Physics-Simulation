"""Transport operators.

- diffusion/ — DiffusionModel operator family (A1/A1P/A2/B/C) + dressing
  helpers; see qpsim.transport.diffusion.base. A1 = (1, 0) is the
  dirty-limit default (undressed energy-channel flux); A1P = (1, 2)
  carries the transverse N_1^2 dressing as a diagnostic. Legacy aliases:
  LEGACY→C, BOLTZMANN→B, USADEL→A2 (USADEL is the diagnostic A2, not the
  dirty-limit Usadel operator A1).
"""
