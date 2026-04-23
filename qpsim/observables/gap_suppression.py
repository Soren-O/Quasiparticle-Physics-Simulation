"""Gap suppression ``δΔ`` relative to the equilibrium BCS gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpsim.physics.gap_equation import calibrate_gap, solve_gap


@dataclass(frozen=True)
class GapSuppressionResult:
    """Equilibrium and driven-gap comparison at fixed ``T_bath``."""

    delta_eq: float
    delta_final: float
    delta_suppression: float
    rel_suppression: float


def gap_suppression_from_deltas(
    delta_eq: float,
    delta_final: float,
) -> GapSuppressionResult:
    """Package ``Δ_eq`` and ``Δ_final`` as suppression observables."""
    if delta_eq < 0:
        raise ValueError("delta_eq must be non-negative.")
    if delta_final < 0:
        raise ValueError("delta_final must be non-negative.")

    delta_suppression = float(delta_eq - delta_final)
    rel_suppression = delta_suppression / delta_eq if delta_eq > 0 else 0.0
    return GapSuppressionResult(
        delta_eq=float(delta_eq),
        delta_final=float(delta_final),
        delta_suppression=delta_suppression,
        rel_suppression=float(rel_suppression),
    )


def compute_gap_suppression(
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    T_c: float,
    T_bath: float,
) -> GapSuppressionResult:
    """Solve the gap equation on ``f(E)`` and compare against ``Δ_eq(T_bath)``."""
    calibration = calibrate_gap(T_c=T_c, T_bath=T_bath)
    delta_final = solve_gap(
        calibration,
        np.asarray(f, dtype=float),
        np.asarray(E_bins, dtype=float),
    )
    return gap_suppression_from_deltas(calibration.delta_eq, delta_final)
