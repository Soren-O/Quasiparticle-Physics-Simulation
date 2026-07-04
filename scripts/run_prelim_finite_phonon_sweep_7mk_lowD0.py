"""Low-D0 extension of the 7 mK finite-phonon sweep.

Adds the diffusion-transition points D0 = {0.6, 1, 3} um^2/ns that the
main sweep (D0 = {6, 20, 60}) lacks — the prelim deck's diffusion slide
showed transport mattering strongly below ~6 um^2/ns and saturating
above, so the paper's D0-sweep figure needs both sides of the knee.
Together the two output directories give the full 6-point D0 axis at
identical (rate, tau_l) grids.

Longer max_time than the main sweep: the strip diffusion time L^2/D0
reaches ~17 us at D0 = 0.6 um^2/ns, so the 12 us cap would truncate the
slowest runs. Same grid, dt, and stop tolerance as the main sweep.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_prelim_finite_phonon_sweep_7mk as base
from scripts.run_prelim_spatial_overnight import SweepConfig

base.CONFIG = SweepConfig(
    name="finite_phonon_sweep_7mk_lowD0",
    NX=21,
    NE=28,
    dt_ns=1.0,
    max_time_ns=48_000.0,
    stop_tol=2e-9,
    snapshot_interval_ns=1_000.0,
    D0_values=(0.6, 1.0, 3.0),
    source_rates_per_ns=(1e-4, 5e-4, 1e-3),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)
base.OUT_DIR = ROOT / "outputs" / "prelim_finite_phonon_sweep_7mk_lowD0"


if __name__ == "__main__":
    base.main()
