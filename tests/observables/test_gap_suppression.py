"""Tests for qpsim.observables.gap_suppression."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.gap_suppression import (
    compute_gap_suppression,
    compute_gap_suppression_direct,
    delta_suppression_from_distribution_direct,
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
    gap_suppression_from_deltas,
    gap_suppression_ratio_from_integrals,
    thermal_gap_integral_direct,
)
from qpsim.physics.gap_equation import calibrate_gap


def _thermal_f(
    T_bath: float = 0.3,
    T_c: float = 1.2,
    num: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    delta_0 = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=delta_0,
        # ``compute_gap_suppression`` may find any gap down to the normal
        # state, so the input occupation must cover that entire interval.
        energy_min_factor=0.0,
        energy_max_factor=10.0,
        num_energy_bins=num,
    )
    dE = integration_widths_from_centers(E)
    del dE  # grid spacing not needed; helper mirrors observables tests style.
    kT = KB_UEV_PER_K * T_bath
    f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return E, f


class TestGapSuppressionFromDeltas:
    def test_positive_suppression(self) -> None:
        result = gap_suppression_from_deltas(1.0, 0.9)
        assert result.delta_suppression == pytest.approx(0.1)
        assert result.rel_suppression == pytest.approx(0.1)

    def test_negative_suppression_is_allowed(self) -> None:
        result = gap_suppression_from_deltas(1.0, 1.1)
        assert result.delta_suppression == pytest.approx(-0.1)
        assert result.rel_suppression == pytest.approx(-0.1)

    def test_rejects_negative_inputs(self) -> None:
        with pytest.raises(ValueError, match="delta_eq"):
            gap_suppression_from_deltas(-1.0, 0.5)
        with pytest.raises(ValueError, match="delta_final"):
            gap_suppression_from_deltas(1.0, -0.5)


class TestComputeGapSuppression:
    def test_thermal_roundtrip_gives_near_zero_suppression(self) -> None:
        T_c, T_bath = 1.2, 0.3
        E, f = _thermal_f(T_bath=T_bath, T_c=T_c, num=300)
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)

        result = compute_gap_suppression(f, E, T_c=T_c, T_bath=T_bath)

        assert result.delta_eq == pytest.approx(cal.delta_eq, rel=1e-12)
        assert result.delta_final == pytest.approx(cal.delta_eq, rel=3e-3)
        assert result.rel_suppression == pytest.approx(0.0, abs=3e-3)

    def test_hot_nonequilibrium_suppresses_gap(self) -> None:
        E, _ = _thermal_f(T_bath=0.3, T_c=1.2, num=300)
        # Over-occupy the spectrum relative to thermal equilibrium.
        f_hot = np.full_like(E, 0.35)

        result = compute_gap_suppression(f_hot, E, T_c=1.2, T_bath=0.3)

        assert result.delta_final <= result.delta_eq
        assert result.delta_suppression >= 0.0

    def test_rejects_missing_low_energy_support(self) -> None:
        T_c, T_bath = 1.2, 0.3
        delta_0 = 1.764 * KB_UEV_PER_K * T_c
        E, _ = build_energy_grid(
            gap=delta_0,
            energy_min_factor=1.001,
            energy_max_factor=10.0,
            num_energy_bins=300,
        )
        f = 1.0 / (
            np.exp(np.minimum(E / (KB_UEV_PER_K * T_bath), 500.0)) + 1.0
        )

        with pytest.raises(ValueError, match="below the reconstructed"):
            compute_gap_suppression(f, E, T_c=T_c, T_bath=T_bath)


class TestDirectGapSuppression:
    def test_subgap_guard_cells_have_zero_measure(self) -> None:
        gap = 180.0
        E_reference, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=1620,
        )
        E_guarded, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=(gap - 20.0) / gap,
            energy_max_factor=10.0,
            num_energy_bins=1640,
        )
        f_reference = 1.0e-3 * np.exp(-(E_reference - gap) / 400.0)
        f_guarded = np.full_like(E_guarded, 0.75)
        active = E_guarded >= gap
        np.testing.assert_array_equal(E_guarded[active], E_reference)
        f_guarded[active] = f_reference

        reference = gap_integral_from_distribution_direct(
            f_reference,
            E_reference,
            gap=gap,
        )
        guarded = gap_integral_from_distribution_direct(
            f_guarded,
            E_guarded,
            gap=gap,
        )

        assert guarded == pytest.approx(reference, rel=1e-13, abs=1e-15)

    def test_rejects_grid_entirely_below_gap(self) -> None:
        gap = 180.0
        E = np.arange(170.5, 180.0, 1.0)

        with pytest.raises(ValueError, match="entirely below"):
            gap_integral_from_distribution_direct(
                np.full_like(E, 1.0e-3),
                E,
                gap=gap,
            )

    def test_rejects_grid_that_starts_above_gap(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.01,
            energy_max_factor=10.0,
            num_energy_bins=180,
        )

        with pytest.raises(ValueError, match="does not cover"):
            gap_integral_from_distribution_direct(
                np.full_like(E, 1.0e-3),
                E,
                gap=gap,
            )

    def test_rejects_subnanometric_gap_edge_omission(self) -> None:
        gap = 180.0
        # The former 1e-10 relative geometry tolerance admitted this grid even
        # though the singular interval [Delta, first_face) was missing.
        omitted = 1.0e-8
        E = gap + omitted + 0.5 + np.arange(4, dtype=float)

        with pytest.raises(ValueError, match="does not cover"):
            gap_integral_from_distribution_direct(
                np.full_like(E, 0.5),
                E,
                gap=gap,
            )

    def test_roundoff_sized_gap_edge_offset_is_aligned_to_gap(self) -> None:
        gap = 180.0
        E_exact = gap + 0.5 + np.arange(4, dtype=float)
        offset = 8.0 * np.finfo(float).eps * gap
        E_roundoff = E_exact + offset
        f = np.full_like(E_exact, 1.0e-3)

        exact = gap_integral_from_distribution_direct(f, E_exact, gap=gap)
        aligned = gap_integral_from_distribution_direct(f, E_roundoff, gap=gap)

        assert aligned == pytest.approx(exact, rel=2e-15, abs=0.0)

    def test_gap_crossing_cell_retains_true_linear_anchor(self) -> None:
        gap = 180.0
        h = 1.0
        left_edges = np.array([gap - 0.25, gap + 0.75, gap + 1.75])
        E = left_edges + 0.5 * h
        f_left = np.array([1.0e-3, 3.0e-3, 3.0e-3])

        actual = gap_integral_from_distribution_direct(
            f_left,
            E,
            gap=gap,
            samples="edges",
        )

        crossing_hi = left_edges[1]
        grid_hi = left_edges[-1] + h
        acosh_crossing = np.arccosh(crossing_hi / gap)
        affine_slope = (f_left[1] - f_left[0]) / h
        crossing = (
            2.0 * f_left[0] * acosh_crossing
            + 2.0
            * affine_slope
            * (
                np.sqrt(crossing_hi**2 - gap**2)
                - left_edges[0] * acosh_crossing
            )
        )
        constant_tail = 2.0 * f_left[1] * (
            np.arccosh(grid_hi / gap) - acosh_crossing
        )

        assert actual == pytest.approx(
            crossing + constant_tail,
            rel=2e-13,
            abs=1e-15,
        )

    def test_constant_distribution_matches_analytic_integral(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=180,
        )
        f0 = 1.0e-3
        f = np.full_like(E, f0)

        integral = gap_integral_from_distribution_direct(f, E, gap=gap)

        expected = 4.0 * f0 * np.arcsinh(np.sqrt((10.0 * gap - gap) / (2.0 * gap)))
        assert integral == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_direct_gap_uses_expm1_for_tiny_suppression(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=180,
        )
        f = np.full_like(E, 1.0e-10)

        integral = gap_integral_from_distribution_direct(f, E, gap=gap)
        delta_rel = delta_suppression_from_distribution_direct(f, E, gap=gap)
        direct_gap = gap_from_distribution_direct(f, E, gap=gap)

        assert delta_rel == pytest.approx(-np.expm1(-integral), rel=1e-15)
        assert (gap - direct_gap) / gap == pytest.approx(delta_rel, rel=1e-8)

    def test_ratio_from_integrals_resolves_fig6_scale(self) -> None:
        thermal_integral = 4.6e-10
        driven_integral = 0.82 * thermal_integral

        ratio = gap_suppression_ratio_from_integrals(
            driven_integral,
            thermal_integral,
        )

        assert ratio == pytest.approx(0.18, rel=1e-8)

    def test_ratio_from_integrals_matches_decimal_reference(self) -> None:
        thermal_integral = 4.6e-10
        driven_integral = 0.82 * thermal_integral
        with localcontext() as ctx:
            ctx.prec = 60
            drv = Decimal(str(driven_integral))
            th = Decimal(str(thermal_integral))
            reference = float(((-drv).exp() - (-th).exp()) / (Decimal(1) - (-th).exp()))

        ratio = gap_suppression_ratio_from_integrals(
            driven_integral,
            thermal_integral,
        )
        naive = (
            np.exp(-driven_integral) - np.exp(-thermal_integral)
        ) / (1.0 - np.exp(-thermal_integral))

        assert ratio == pytest.approx(reference, rel=1e-14)
        assert abs(naive - reference) / reference > 1e-8

    def test_thermal_integral_defaults_to_center_samples(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=180,
        )

        integral = thermal_gap_integral_direct(E, gap=gap, T_bath=0.2)

        f_centers = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * 0.2), 700.0)) + 1.0)
        expected = gap_integral_from_distribution_direct(f_centers, E, gap=gap)
        assert integral == pytest.approx(expected, rel=1e-15)

    def test_compute_direct_returns_stable_small_difference(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=180,
        )
        f_thermal = np.zeros_like(E)
        f_driven = np.full_like(E, 1.0e-10)

        thermal_integral = gap_integral_from_distribution_direct(
            f_thermal,
            E,
            gap=gap,
        )
        driven_integral = gap_integral_from_distribution_direct(
            f_driven,
            E,
            gap=gap,
        )
        result = compute_gap_suppression_direct(
            f_driven,
            E,
            gap=gap,
            T_bath=0.0,
        )

        assert result.delta_eq == pytest.approx(gap)
        assert result.delta_suppression == pytest.approx(
            gap * -np.expm1(thermal_integral - driven_integral),
            rel=1e-15,
        )
        assert result.rel_suppression == pytest.approx(
            -np.expm1(thermal_integral - driven_integral),
            rel=1e-15,
        )
