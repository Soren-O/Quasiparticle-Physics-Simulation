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


def _as_real_array(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc


def _as_spatial_distribution(f: np.ndarray, n_x: int, name: str) -> np.ndarray:
    arr = _as_real_array(f, name)
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
    spatial_cell_widths_um: np.ndarray | None = None,
    n_subgap: int = 500,
) -> SpatialAcResponse:
    """Integrate local Mattis-Bardeen response over a resonator current profile.

    This differs from applying Mattis-Bardeen to a pre-averaged occupation:
    ``sigma_1`` and ``sigma_2`` are computed at each position first, and only
    then integrated with the local ``I^2`` weights.

    When ``spatial_cell_widths_um`` is supplied, ``x_um`` is interpreted as
    finite-volume cell centers and every spatial integral uses the explicit
    cell measure.  Omitting it preserves the endpoint-sample/trapezoidal API
    used by older callers.
    """
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if (
        not np.isfinite(full_current_integral_um)
        or full_current_integral_um <= 0.0
    ):
        raise ValueError("full_current_integral_um must be finite and positive.")

    x = _as_real_array(x_um, "x_um")
    weights = _as_real_array(current_weights, "current_weights")
    if x.ndim != 1:
        raise ValueError("x_um must be one-dimensional.")
    if x.size == 0 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x_um must be non-empty, finite, and strictly increasing.")
    if weights.shape != x.shape:
        raise ValueError("current_weights must have the same shape as x_um.")
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            "current_weights must be finite, non-negative, and contain "
            "at least one positive value."
        )

    cell_widths: np.ndarray | None = None
    if spatial_cell_widths_um is not None:
        cell_widths = _as_real_array(
            spatial_cell_widths_um,
            "spatial_cell_widths_um",
        )
        if cell_widths.shape != x.shape:
            raise ValueError(
                "spatial_cell_widths_um must have the same shape as x_um."
            )
        if np.any(~np.isfinite(cell_widths)) or np.any(cell_widths <= 0.0):
            raise ValueError(
                "spatial_cell_widths_um must be finite and positive."
            )

    def integrate(values: np.ndarray) -> np.ndarray:
        if cell_widths is None:
            return np.asarray(np.trapezoid(values, x, axis=-1))
        return np.asarray(np.sum(values * cell_widths, axis=-1))

    f_spatial = _as_spatial_distribution(f, x.size, "f")
    f_ref_spatial = _as_spatial_distribution(f_ref, x.size, "f_ref")
    if f_spatial.shape != f_ref_spatial.shape:
        raise ValueError("f and f_ref must have compatible energy dimensions.")

    strip_integral = float(integrate(weights))
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

    sigma1_eff = float(integrate(weights * sigma1) / strip_integral)
    sigma2_eff = float(integrate(weights * sigma2) / strip_integral)
    sigma1_ref_eff = float(integrate(weights * sigma1_ref) / strip_integral)
    sigma2_ref_eff = float(integrate(weights * sigma2_ref) / strip_integral)

    supported_weight = weights > 0.0
    delta_sigma2 = sigma2 - sigma2_ref
    zero_reference_with_change = (
        supported_weight & (sigma2_ref == 0.0) & (delta_sigma2 != 0.0)
    )
    if np.any(zero_reference_with_change):
        raise ValueError(
            "Cannot form the local reactive-conductivity change: reference "
            "sigma2 is zero while the current sigma2 differs at one or more "
            "spatial points."
        )
    rel_sigma2 = np.divide(
        delta_sigma2,
        sigma2_ref,
        out=np.zeros_like(sigma2),
        where=supported_weight & (sigma2_ref != 0.0),
    )
    weighted_rel_sigma2 = float(integrate(weights * rel_sigma2) / strip_integral)
    frac_shift = (
        0.5
        * alpha
        * float(integrate(weights * rel_sigma2))
        / full_current_integral_um
    )

    zero_reactive_with_loss = (
        supported_weight & (sigma2 == 0.0) & (sigma1 != 0.0)
    )
    if np.any(zero_reactive_with_loss):
        raise ValueError(
            "Cannot form the local quasiparticle loss ratio: sigma2 is zero "
            "while sigma1 is non-zero at one or more spatial points."
        )
    # Preserve the sign of the reactive response. In particular, sigma2 < 0
    # is an active response and must produce a signed inverse Q instead of
    # being silently replaced by the zero-loss limit.
    local_loss_ratio = np.divide(
        sigma1,
        sigma2,
        out=np.zeros_like(sigma1),
        where=supported_weight & (sigma2 != 0.0),
    )
    inverse_qi = (
        alpha
        * float(integrate(weights * local_loss_ratio))
        / full_current_integral_um
    )
    # Negative inverse Q is active gain (negative damping), not the zero-loss
    # limit. Preserve its sign just like the lumped quality-factor observable.
    qi_qp = 1.0 / inverse_qi if inverse_qi != 0.0 else float(np.inf)

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
