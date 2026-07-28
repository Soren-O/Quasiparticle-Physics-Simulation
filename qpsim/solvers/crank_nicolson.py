"""Crank–Nicolson time stepper for linear parabolic operators.

Given a precomputed Laplacian (from :mod:`qpsim.grid.spatial_grid`),
builds the implicit ``A`` and explicit ``B`` matrices for one CN step
and factors ``A`` with SuperLU so repeated per-step solves are fast.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def _finite_real_scalar(name: str, raw: float) -> float:
    """Return one finite real scalar without silent complex coercion."""
    if isinstance(raw, (bool, np.bool_)) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real scalar; got {raw!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {raw!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return value


def build_cn_operators(
    laplacian: sparse.spmatrix,
    dt: float,
    diffusion_coefficient: float,
) -> tuple[sparse.csr_matrix, spla.SuperLU]:
    """Build CN operators ``(B, LU[A])`` for ``∂_t u = D · L · u``.

    The Crank–Nicolson update is
    ``(I − α L) u^{n+1} = (I + α L) u^n``
    with ``α = ½ dt · D``. Returns:

    * ``B = (I + α L)`` as CSR.
    * ``LU``: SuperLU factor of ``A = (I − α L)`` (CSC), so the per-step
      solve is ``u^{n+1} = LU.solve(B · u^n)``.

    The factorization is the expensive part of a time loop; reuse this
    pair across steps while ``laplacian``, ``dt``, and
    ``diffusion_coefficient`` are unchanged.
    """
    if not sparse.issparse(laplacian):
        raise ValueError("laplacian must be a scipy sparse matrix.")
    if (
        len(laplacian.shape) != 2
        or laplacian.shape[0] == 0
        or laplacian.shape[0] != laplacian.shape[1]
    ):
        raise ValueError(
            "laplacian must be a nonempty square sparse matrix; "
            f"got shape {laplacian.shape}."
        )
    # Normalize the sparse storage before inspecting values.  Formats such as
    # LIL expose ``.data`` as ragged Python lists and DOK uses a mapping, even
    # though both are valid scipy sparse operators.
    try:
        laplacian_csr = laplacian.tocsr()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("laplacian must contain real numeric values.") from exc
    laplacian_data = np.asarray(laplacian_csr.data)
    if np.iscomplexobj(laplacian_data):
        raise ValueError("laplacian must be real-valued.")
    if np.any(~np.isfinite(laplacian_data)):
        raise ValueError("laplacian must contain only finite values.")

    dt = _finite_real_scalar("dt", dt)
    diffusion_coefficient = _finite_real_scalar(
        "diffusion_coefficient", diffusion_coefficient
    )
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative.")

    n = laplacian.shape[0]
    identity = sparse.eye(n, format="csr")
    alpha = 0.5 * dt * diffusion_coefficient
    a_mat = (identity - alpha * laplacian_csr).tocsc()
    b_mat = (identity + alpha * laplacian_csr).tocsr()
    lu = spla.splu(a_mat)
    return b_mat, lu
