"""Current-weighted ac response from a spatial quasiparticle distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.physics.spectral import SpectralContext


@dataclass(frozen=True)
class SpatialAcResponse:
    """Spatially integrated Mattis-Bardeen response for a resonator section."""

    sigma1_current_weighted_norm: float
    sigma2_current_weighted_norm: float
    sigma1_ref_current_weighted_norm: float
    sigma2_ref_current_weighted_norm: float
    relative_sigma2_change_current_weighted: float
    frac_freq_shift: float
    inverse_qi_qp: float
    qi_qp: float
    current_participation: float
    strip_current_integral_um: float
    full_current_integral_um: float


def _as_spatial_distribution(f: np.ndarray, n_x: int, name: str) -> np.ndarray:
    arr = np.asarray(f, dtype=float)
    if arr.ndim == 1:
        return np.repeat(arr[:, None], n_x, axis=1)
    if arr.ndim == 2 and arr.shape[1] == n_x:
        return arr
    raise ValueError(f"{name} must have shape (NE,) or (NE, NX).")


def _local_conductivities(
    f_spatial: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    *,
    n_subgap: int,
) -> tuple[np.ndarray, np.ndarray]:
    sigma1 = np.empty(f_spatial.shape[1], dtype=float)
    sigma2 = np.empty(f_spatial.shape[1], dtype=float)
    for ix in range(f_spatial.shape[1]):
        sigma1[ix], sigma2[ix] = compute_ac_conductivity(
            f_spatial[:, ix],
            ctx,
            omega_0,
            n_subgap=n_subgap,
        )
    return sigma1, sigma2


def compute_current_weighted_ac_response(
    f: np.ndarray,
    f_ref: np.ndarray,
    x_um: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    *,
    alpha: float,
    current_weights: np.ndarray,
    full_current_integral_um: float,
    n_subgap: int = 500,
) -> SpatialAcResponse:
    """Integrate local Mattis-Bardeen response over a resonator current profile.

    This differs from applying Mattis-Bardeen to a pre-averaged occupation:
    ``sigma_1`` and ``sigma_2`` are computed at each position first, and only
    then integrated with the local ``I^2`` weights.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if full_current_integral_um <= 0.0:
        raise ValueError("full_current_integral_um must be positive.")

    x = np.asarray(x_um, dtype=float)
    weights = np.asarray(current_weights, dtype=float)
    if x.ndim != 1:
        raise ValueError("x_um must be one-dimensional.")
    if weights.shape != x.shape:
        raise ValueError("current_weights must have the same shape as x_um.")

    f_spatial = _as_spatial_distribution(f, x.size, "f")
    f_ref_spatial = _as_spatial_distribution(f_ref, x.size, "f_ref")
    if f_spatial.shape != f_ref_spatial.shape:
        raise ValueError("f and f_ref must have compatible energy dimensions.")

    strip_integral = float(np.trapezoid(weights, x))
    if strip_integral <= 0.0:
        raise ValueError("current weighting normalization vanished.")

    sigma1, sigma2 = _local_conductivities(
        f_spatial,
        ctx,
        omega_0,
        n_subgap=n_subgap,
    )
    sigma1_ref, sigma2_ref = _local_conductivities(
        f_ref_spatial,
        ctx,
        omega_0,
        n_subgap=n_subgap,
    )

    sigma1_eff = float(np.trapezoid(weights * sigma1, x) / strip_integral)
    sigma2_eff = float(np.trapezoid(weights * sigma2, x) / strip_integral)
    sigma1_ref_eff = float(np.trapezoid(weights * sigma1_ref, x) / strip_integral)
    sigma2_ref_eff = float(np.trapezoid(weights * sigma2_ref, x) / strip_integral)

    rel_sigma2 = np.divide(
        sigma2 - sigma2_ref,
        sigma2_ref,
        out=np.zeros_like(sigma2),
        where=sigma2_ref > 0.0,
    )
    weighted_rel_sigma2 = float(np.trapezoid(weights * rel_sigma2, x) / strip_integral)
    frac_shift = (
        0.5
        * alpha
        * float(np.trapezoid(weights * rel_sigma2, x))
        / full_current_integral_um
    )

    local_loss_ratio = np.divide(
        sigma1,
        sigma2,
        out=np.zeros_like(sigma1),
        where=sigma2 > 0.0,
    )
    inverse_qi = (
        alpha
        * float(np.trapezoid(weights * local_loss_ratio, x))
        / full_current_integral_um
    )
    qi_qp = 1.0 / inverse_qi if inverse_qi > 0.0 else float(np.inf)

    return SpatialAcResponse(
        sigma1_current_weighted_norm=sigma1_eff,
        sigma2_current_weighted_norm=sigma2_eff,
        sigma1_ref_current_weighted_norm=sigma1_ref_eff,
        sigma2_ref_current_weighted_norm=sigma2_ref_eff,
        relative_sigma2_change_current_weighted=weighted_rel_sigma2,
        frac_freq_shift=float(frac_shift),
        inverse_qi_qp=float(inverse_qi),
        qi_qp=float(qi_qp),
        current_participation=strip_integral / full_current_integral_um,
        strip_current_integral_um=strip_integral,
        full_current_integral_um=float(full_current_integral_um),
    )
