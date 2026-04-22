"""Crank–Nicolson time stepper for linear parabolic operators.

Given a precomputed Laplacian (from :mod:`qpsim.grid.spatial_grid`),
builds the implicit ``A`` and explicit ``B`` matrices for one CN step
and factors ``A`` with SuperLU so repeated per-step solves are fast.
"""

from __future__ import annotations

from scipy import sparse
from scipy.sparse import linalg as spla


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
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if diffusion_coefficient < 0:
        raise ValueError("diffusion_coefficient must be non-negative.")

    n = laplacian.shape[0]
    identity = sparse.eye(n, format="csr")
    alpha = 0.5 * dt * diffusion_coefficient
    a_mat = (identity - alpha * laplacian).tocsc()
    b_mat = (identity + alpha * laplacian).tocsr()
    lu = spla.splu(a_mat)
    return b_mat, lu
