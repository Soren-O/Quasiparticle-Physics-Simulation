"""Shared fail-closed validation for public collision-array inputs."""

from __future__ import annotations

import numpy as np


def validated_occupation(
    f: np.ndarray,
    expected_shape: tuple[int, ...],
    operation: str,
) -> np.ndarray:
    """Return a finite occupation vector in ``[0, 1]`` with exact shape."""
    # Reject complex inputs before conversion.  ``np.asarray(...,
    # dtype=float)`` otherwise discards the imaginary component with only a
    # warning; in particular, an imaginary NaN can become an apparently
    # finite real occupation.
    if np.iscomplexobj(f):
        raise ValueError(f"{operation} requires real-valued occupations f.")
    occupation = np.asarray(f, dtype=float)
    if occupation.ndim != 1 or occupation.shape != expected_shape:
        raise ValueError(
            f"{operation} requires f with shape {expected_shape}; got "
            f"{occupation.shape}."
        )
    if np.any(~np.isfinite(occupation)) or np.any(
        (occupation < 0.0) | (occupation > 1.0)
    ):
        raise ValueError(
            f"{operation} requires finite occupations f in [0, 1]."
        )
    return occupation


def validated_rate_matrix(
    matrix: np.ndarray | None,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    """Return a finite non-negative collision matrix with exact shape."""
    if matrix is None:
        return None
    # As above, fail before a float cast can erase an invalid imaginary part.
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be real-valued.")
    result = np.asarray(matrix, dtype=float)
    if result.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {result.shape}."
        )
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative rates.")
    return result
