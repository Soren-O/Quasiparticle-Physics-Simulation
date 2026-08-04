"""Shared grid checks for collision channels with index-shift partners."""

from __future__ import annotations

import numpy as np

# Successful validations, keyed by exact grid content. Grid arrays are built
# once per solve and never mutated, but hot loops (residual/Jacobian
# evaluations) re-validate them thousands of times per solve; remembering
# recent successes replaces the diff/allclose scan with a hash lookup.
# Identity (``id``) is not safe as a key — the same discipline as
# T3Spatial1DBackend._spectral_cache_key: SpectralContext.E/.dE mint a fresh
# read-only view per access, so those ids are freed and recycled, and neither
# an id nor the endpoints constrain the interior spacings this guard checks.
# Keying on content makes a stale hit impossible: a changed interior, a
# changed endpoint, or an in-place mutation is a different key and falls
# through to the full check. Failures are never cached.
_VALIDATED_GRIDS: dict[tuple[bytes, bytes], float] = {}
_VALIDATED_GRIDS_MAX = 16


def _clear_validated_grids() -> None:
    """Forget every remembered validation (test/diagnostic hook)."""
    _VALIDATED_GRIDS.clear()


def uniform_grid_spacing(E: np.ndarray, dE: np.ndarray, channel: str) -> float:
    """Return scalar dE after verifying that the energy grid is uniform."""
    E_arr = np.asarray(E, dtype=float).reshape(-1)
    dE_arr = np.asarray(dE, dtype=float).reshape(-1)
    if E_arr.size and dE_arr.size:
        cache_key = (E_arr.tobytes(), dE_arr.tobytes())
        cached = _VALIDATED_GRIDS.get(cache_key)
        if cached is not None:
            return cached
    else:
        cache_key = None
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

    if cache_key is not None:
        while len(_VALIDATED_GRIDS) >= _VALIDATED_GRIDS_MAX:
            _VALIDATED_GRIDS.pop(next(iter(_VALIDATED_GRIDS)))
        _VALIDATED_GRIDS[cache_key] = dE_scalar
    return dE_scalar
