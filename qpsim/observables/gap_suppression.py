"""Gap suppression ``δΔ`` relative to the equilibrium BCS gap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpsim.physics import fermi_dirac_occupation


def _finite_real_scalar(name: str, raw: float) -> float:
    """Return one finite real scalar without silently dropping complexity."""
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


def _finite_real_array(name: str, values: np.ndarray) -> np.ndarray:
    """Return a finite float array, rejecting complex input before casting."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


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
    delta_eq = _finite_real_scalar("delta_eq", delta_eq)
    delta_final = _finite_real_scalar("delta_final", delta_final)
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
    """Solve the gap equation on ``f(E)`` and compare against ``Δ_eq(T_bath)``.

    The finite-volume cells reconstructed from ``E_bins`` must extend below
    every physically possible driven gap.  A distribution that can collapse
    to the normal state therefore needs support down to zero; missing low-edge
    occupation data is rejected rather than extrapolated.
    """
    from qpsim.physics.gap_equation import calibrate_gap, solve_gap

    f_arr = _finite_real_array("f", f)
    E_arr = _finite_real_array("E_bins", E_bins)
    T_c = _finite_real_scalar("T_c", T_c)
    T_bath = _finite_real_scalar("T_bath", T_bath)
    if T_c <= 0.0:
        raise ValueError("T_c must be positive.")
    if T_bath < 0.0:
        raise ValueError("T_bath must be non-negative.")
    calibration = calibrate_gap(T_c=T_c, T_bath=T_bath)
    delta_final = solve_gap(
        calibration,
        f_arr,
        E_arr,
    )
    return gap_suppression_from_deltas(calibration.delta_eq, delta_final)


def left_edges_from_centers(E_bins: np.ndarray) -> np.ndarray:
    """Return left bin edges for an approximately uniform center grid."""
    E = _finite_real_array("E_bins", E_bins).reshape(-1)
    if E.size == 0:
        raise ValueError("E_bins must be non-empty.")
    if E.size == 1:
        raise ValueError("At least two energy bins are required.")
    dE = np.diff(E)
    if np.any(dE <= 0.0):
        raise ValueError("E_bins must be strictly increasing.")
    h = float(np.mean(dE))
    if not np.allclose(dE, h, rtol=1e-9, atol=1e-12 * max(1.0, abs(h))):
        raise ValueError("Direct gap integral currently requires a uniform grid.")
    return E - 0.5 * h


def edge_samples_from_centers(f: np.ndarray, E_bins: np.ndarray) -> np.ndarray:
    """Map center-grid occupations to the left-edge nodes used by Fischer Fig. 6."""
    E = _finite_real_array("E_bins", E_bins).reshape(-1)
    f_arr = _finite_real_array("f", f).reshape(-1)
    if f_arr.shape != E.shape:
        raise ValueError(f"f and E_bins must have the same shape, got {f_arr.shape} and {E.shape}.")
    edges = left_edges_from_centers(E)
    out = np.interp(edges, E, f_arr)
    slope_left = (f_arr[1] - f_arr[0]) / (E[1] - E[0])
    out[0] = f_arr[0] + slope_left * (edges[0] - E[0])
    return np.maximum(out, 0.0)


def fermi_dirac_distribution(E: np.ndarray, T_bath: float) -> np.ndarray:
    """Fermi-Dirac occupation at bath temperature ``T_bath`` in kelvin."""
    E_arr = _finite_real_array("E", E)
    temperature = _finite_real_scalar("T_bath", T_bath)
    if temperature < 0.0:
        raise ValueError("T_bath must be non-negative.")
    return fermi_dirac_occupation(E_arr, temperature)


def gap_integral_from_distribution_direct(
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    gap: float,
    samples: str = "centers",
) -> float:
    """Return ``I[f] = integral 2 rho(E) f(E) / E dE`` on a uniform grid.

    This mirrors the Fischer Fig. 6 author code path: occupations are linearly
    interpolated between left-edge nodes and the BCS square-root singularity is
    integrated analytically. ``samples="centers"`` maps qpsim's cell-centered
    occupations onto the left-edge nodes first; ``samples="edges"`` treats
    ``f`` as already sampled at those left edges.
    """
    gap = _finite_real_scalar("gap", gap)
    if gap <= 0.0:
        raise ValueError("gap must be finite and positive.")
    E = _finite_real_array("E_bins", E_bins).reshape(-1)
    f_arr = _finite_real_array("f", f).reshape(-1)
    if f_arr.shape != E.shape:
        raise ValueError(f"f and E_bins must have the same shape, got {f_arr.shape} and {E.shape}.")
    if np.any((f_arr < 0.0) | (f_arr > 1.0)):
        raise ValueError("f must contain physical occupations in [0, 1].")
    if E.size < 2:
        raise ValueError("At least two energy bins are required.")

    edges = left_edges_from_centers(E)
    h = float(E[1] - E[0])
    # Missing even a tiny interval directly above Delta is amplified by the
    # ideal-BCS square-root singularity.  Admit only coordinate roundoff, not a
    # user-scale geometric tolerance, and align that roundoff-sized offset to
    # Delta before integrating so the singular support is not silently dropped.
    tol = 64.0 * np.finfo(float).eps * max(gap, h, 1.0)
    grid_lo = float(edges[0])
    if grid_lo > gap + tol:
        raise ValueError(
            "Energy grid does not cover the superconducting gap: "
            f"first cell edge {grid_lo:g} > gap {gap:g}."
        )
    if grid_lo > gap:
        edges = edges - (grid_lo - gap)
    grid_hi = float(edges[-1] + h)
    if grid_hi <= gap + tol:
        raise ValueError("Energy grid lies entirely below the superconducting gap.")

    mode = samples.lower()
    if mode in {"center", "centers"}:
        # Ideal-BCS cells wholly below Delta have zero spectral capacity, so
        # their stored occupations are placeholders rather than physical
        # sub-gap samples. (A gap-cut cell remains represented even if its
        # center is below Delta.) Reconstruct the gap-edge value only from
        # positive-capacity centers; otherwise an arbitrary guard value leaks
        # through np.interp into the singular first active interval.
        active = gap < edges + h
        if np.count_nonzero(active) < 2:
            raise ValueError(
                "Center-sampled direct gap integration requires at least two "
                "positive-capacity energy-bin centers at the superconducting gap."
            )
        E_active = E[active]
        f_active = f_arr[active]
        vals = np.interp(edges, E_active, f_active)
        slope_left = (f_active[1] - f_active[0]) / (
            E_active[1] - E_active[0]
        )
        below_first_active = edges < E_active[0]
        vals[below_first_active] = f_active[0] + slope_left * (
            edges[below_first_active] - E_active[0]
        )
        vals = np.maximum(vals, 0.0)
    elif mode in {"edge", "edges", "authors"}:
        # The final interval is held constant, matching the bundled author code.
        vals = np.maximum(f_arr, 0.0)
    else:
        raise ValueError("samples must be 'centers' or 'edges'.")

    # Cells wholly below the ideal-BCS gap have exactly zero spectral measure.
    # Keeping them in the kinetic grid is useful for a later moving-gap solve
    # and must not invalidate this fixed-gap integral.  Clamp their transformed
    # bounds to zero; the first gap-crossing cell then starts exactly at Delta.
    cell_x_lo = edges - gap
    x_lo = np.maximum(cell_x_lo, 0.0)
    x_hi = np.maximum(cell_x_lo + h, 0.0)

    a_hi = np.arcsinh(np.sqrt(x_hi / (2.0 * gap)))
    a_lo = np.arcsinh(np.sqrt(x_lo / (2.0 * gap)))
    const_term = float(np.sum(4.0 * vals * (a_hi - a_lo)))
    if E.size <= 1:
        return const_term

    x0 = x_lo[:-1]
    x1 = x_hi[:-1]
    da = a_hi[:-1] - a_lo[:-1]
    lin_weight = (
        np.sqrt(x1 * (x1 + 2.0 * gap))
        - np.sqrt(x0 * (x0 + 2.0 * gap))
        # Keep the affine occupation anchored at the cell's true left edge.
        # For a gap-crossing cell that edge is below Delta even though the
        # spectral integration bound is clamped to Delta.
        - 2.0 * (cell_x_lo[:-1] + gap) * da
    )
    lin_term = float(np.sum(2.0 * (vals[1:] - vals[:-1]) / h * lin_weight))
    return const_term + lin_term


def gap_from_distribution_direct(
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    gap: float,
    delta0: float | None = None,
    samples: str = "centers",
) -> float:
    """Return the non-self-consistent gap ``Delta[f] = Delta0 * exp(-I[f])``."""
    gap0 = _finite_real_scalar(
        "delta0", gap if delta0 is None else delta0
    )
    if gap0 <= 0.0:
        raise ValueError("delta0 must be positive.")
    integral = gap_integral_from_distribution_direct(
        f,
        E_bins,
        gap=gap,
        samples=samples,
    )
    return gap0 * float(np.exp(-integral))


def delta_suppression_from_distribution_direct(
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    gap: float,
    samples: str = "centers",
) -> float:
    """Return ``(Delta0 - Delta[f]) / Delta0`` without subtracting gaps."""
    integral = gap_integral_from_distribution_direct(
        f,
        E_bins,
        gap=gap,
        samples=samples,
    )
    return float(-np.expm1(-integral))


def gap_suppression_ratio_from_integrals(
    driven_integral: float,
    thermal_integral: float,
) -> float:
    """Return ``(delta_T - delta_driven) / delta_T`` from direct integrals."""
    driven_integral = _finite_real_scalar("driven_integral", driven_integral)
    thermal_integral = _finite_real_scalar("thermal_integral", thermal_integral)
    if thermal_integral <= 0.0:
        return float("nan")
    denominator = -np.expm1(-thermal_integral)
    if denominator <= 0.0:
        return float("nan")
    numerator = np.exp(-thermal_integral) * np.expm1(thermal_integral - driven_integral)
    return float(numerator / denominator)


def gap_suppression_from_integrals_direct(
    driven_integral: float,
    thermal_integral: float,
    *,
    delta0: float,
) -> GapSuppressionResult:
    """Package a thermal-vs-driven direct gap comparison without cancellation."""
    driven_integral = _finite_real_scalar("driven_integral", driven_integral)
    thermal_integral = _finite_real_scalar("thermal_integral", thermal_integral)
    delta0 = _finite_real_scalar("delta0", delta0)
    if delta0 <= 0.0:
        raise ValueError("delta0 must be positive.")
    delta_eq = delta0 * float(np.exp(-thermal_integral))
    delta_final = delta0 * float(np.exp(-driven_integral))
    rel_suppression = float(-np.expm1(thermal_integral - driven_integral))
    delta_suppression = delta_eq * rel_suppression
    return GapSuppressionResult(
        delta_eq=delta_eq,
        delta_final=delta_final,
        delta_suppression=float(delta_suppression),
        rel_suppression=rel_suppression,
    )


def thermal_gap_integral_direct(
    E_bins: np.ndarray,
    *,
    gap: float,
    T_bath: float,
    samples: str = "centers",
) -> float:
    """Direct gap integral for a thermal Fermi-Dirac distribution."""
    E = _finite_real_array("E_bins", E_bins).reshape(-1)
    mode = samples.lower()
    if mode in {"edge", "edges", "authors"}:
        sample_E = left_edges_from_centers(E)
    elif mode in {"center", "centers"}:
        sample_E = E
    else:
        raise ValueError("samples must be 'centers' or 'edges'.")
    f = fermi_dirac_distribution(sample_E, T_bath)
    return gap_integral_from_distribution_direct(
        f,
        E,
        gap=gap,
        samples=samples,
    )


def compute_gap_suppression_direct(
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    gap: float,
    T_bath: float,
    samples: str = "centers",
    thermal_samples: str | None = None,
) -> GapSuppressionResult:
    """Compare a driven distribution to thermal equilibrium via direct integrals."""
    thermal_mode = samples if thermal_samples is None else thermal_samples
    driven_integral = gap_integral_from_distribution_direct(
        f,
        E_bins,
        gap=gap,
        samples=samples,
    )
    thermal_integral = thermal_gap_integral_direct(
        E_bins,
        gap=gap,
        T_bath=T_bath,
        samples=thermal_mode,
    )
    return gap_suppression_from_integrals_direct(
        driven_integral,
        thermal_integral,
        delta0=gap,
    )
