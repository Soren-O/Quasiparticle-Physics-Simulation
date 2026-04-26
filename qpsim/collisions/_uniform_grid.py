"""Shared grid checks for collision channels with index-shift partners."""

from __future__ import annotations

import numpy as np


def uniform_grid_spacing(E: np.ndarray, dE: np.ndarray, channel: str) -> float:
    """Return scalar dE after verifying that the energy grid is uniform."""
    E_arr = np.asarray(E, dtype=float).reshape(-1)
    dE_arr = np.asarray(dE, dtype=float).reshape(-1)
    if E_arr.size == 0 or dE_arr.size == 0:
        raise ValueError(f"{channel} requires a non-empty energy grid.")
    if E_arr.size != dE_arr.size:
        raise ValueError(
            f"{channel} requires E and dE with matching lengths; "
            f"got {E_arr.size} and {dE_arr.size}."
        )

    dE_scalar = float(dE_arr[0])
    if not np.isfinite(dE_scalar) or dE_scalar <= 0.0:
        raise ValueError(
            f"{channel} requires a positive finite uniform dE; got {dE_scalar}."
        )

    scale = max(abs(dE_scalar), 1.0)
    atol = 1e-12 * scale
    rtol = 1e-10
    if (
        not np.all(np.isfinite(E_arr))
        or not np.all(np.isfinite(dE_arr))
        or np.any(dE_arr <= 0.0)
    ):
        raise ValueError(f"{channel} requires finite positive grid values.")

    if E_arr.size > 1:
        spacings = np.diff(E_arr)
        if not np.allclose(spacings, dE_scalar, rtol=rtol, atol=atol):
            raise ValueError(
                f"{channel} requires a uniform energy grid because it maps "
                "photon partners by fixed index offsets. Use a uniform grid "
                "or add an energy-search/interpolation implementation for "
                "nonuniform grids."
            )

    if not np.allclose(dE_arr, dE_scalar, rtol=rtol, atol=atol):
        raise ValueError(
            f"{channel} requires uniform integration widths because it maps "
            "photon partners by fixed index offsets."
        )

    return dE_scalar
