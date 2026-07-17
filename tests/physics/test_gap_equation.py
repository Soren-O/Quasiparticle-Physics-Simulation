"""Tests for qpsim.physics.gap_equation — BCS calibration and runtime solve."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)
from qpsim.materials.database import load_material
from qpsim.physics.gap_equation import _gap_integral_f, calibrate_gap, solve_gap


def _legacy_interpolated_gap_integral(
    delta: float,
    f: np.ndarray,
    E: np.ndarray,
    omega_D: float,
) -> float:
    """Pre-finite-volume residual, retained only for accuracy regressions."""
    u_max = np.arccosh(omega_D / delta)
    u = np.linspace(0.0, u_max, max(E.size * 2, 256) + 1)
    f_interp = np.interp(
        delta * np.cosh(u),
        E,
        f,
        left=float(f[0]),
        right=0.0,
    )
    return float(np.trapezoid(1.0 - 2.0 * f_interp, u))


class TestCalibrateGap:
    def test_bcs_zero_T_ratio(self) -> None:
        # The default finite cutoff ω_D/(k_B T_c)=100 is already very close
        # to the universal infinite-cutoff BCS ratio 1.763876..., with the
        # small finite-cutoff correction retained by the calibration.
        cal = calibrate_gap(T_c=1.2, T_bath=0.0)
        assert cal.delta_eq == pytest.approx(cal.delta_0_bcs, rel=0.0, abs=0.0)
        assert cal.delta_0_bcs / (KB_UEV_PER_K * 1.2) == pytest.approx(
            1.763739802445,
            rel=2e-11,
        )
        assert cal.delta_0_reference is None

    @pytest.mark.parametrize("material_name", ["Al", "TiN", "Nb"])
    def test_supplied_material_gap_is_diagnostic_not_a_second_anchor(
        self,
        material_name: str,
    ) -> None:
        material = load_material(material_name)
        baseline = calibrate_gap(T_c=material.T_c, T_bath=0.0)
        with_reference = calibrate_gap(
            T_c=material.T_c,
            T_bath=0.0,
            Delta_0=material.Delta_0,
        )

        assert with_reference.delta_0_reference == material.Delta_0
        assert with_reference.delta_0_bcs == pytest.approx(
            baseline.delta_0_bcs,
            rel=0.0,
            abs=0.0,
        )
        assert with_reference.delta_eq == pytest.approx(
            baseline.delta_eq,
            rel=0.0,
            abs=0.0,
        )
        assert with_reference._inv_lambda == pytest.approx(
            baseline._inv_lambda,
            rel=0.0,
            abs=0.0,
        )

    @pytest.mark.parametrize("reference_ratio", [1.2, 2.2])
    def test_reference_below_or_above_bcs_ratio_cannot_move_Tc(
        self,
        reference_ratio: float,
    ) -> None:
        T_c = 3.0
        T_bath = 0.999 * T_c
        baseline = calibrate_gap(T_c=T_c, T_bath=T_bath, xtol=1e-12)
        supplied = reference_ratio * KB_UEV_PER_K * T_c
        diagnostic = calibrate_gap(
            T_c=T_c,
            T_bath=T_bath,
            Delta_0=supplied,
            xtol=1e-12,
        )

        assert diagnostic.delta_0_reference == supplied
        assert diagnostic.delta_eq == pytest.approx(
            baseline.delta_eq,
            rel=0.0,
            abs=0.0,
        )
        assert (
            calibrate_gap(
                T_c=T_c,
                T_bath=T_c,
                Delta_0=supplied,
            ).delta_eq
            == 0.0
        )

    @pytest.mark.parametrize("material_name", ["Al", "TiN", "Nb"])
    def test_shipped_material_gap_closes_continuously_at_declared_Tc(
        self,
        material_name: str,
    ) -> None:
        material = load_material(material_name)
        epsilons = np.array([1e-2, 1e-3, 1e-4])
        calibrations = [
            calibrate_gap(
                T_c=material.T_c,
                T_bath=(1.0 - epsilon) * material.T_c,
                Delta_0=material.Delta_0,
                xtol=1e-10,
            )
            for epsilon in epsilons
        ]
        gaps = np.array([cal.delta_eq for cal in calibrations])

        assert np.all(gaps > 0.0)
        assert np.all(np.diff(gaps) < 0.0)
        # Mean-field BCS closes as sqrt(1 - T/Tc).  The normalized values
        # should therefore approach a common finite coefficient, rather than
        # the O(0.1-1) residual gap produced by a Δ0-anchored coupling.
        scaled = gaps / np.sqrt(epsilons)
        assert np.max(scaled) / np.min(scaled) < 1.01
        assert gaps[-1] / calibrations[-1].delta_0_bcs < 0.02
        assert (
            calibrate_gap(
                T_c=material.T_c,
                T_bath=material.T_c,
                Delta_0=material.Delta_0,
            ).delta_eq
            == 0.0
        )

    def test_normal_state_above_Tc(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=2.0)
        assert cal.delta_eq == 0.0

    def test_gap_monotonically_decreases_with_T(self) -> None:
        cals = [calibrate_gap(T_c=1.2, T_bath=T) for T in (0.0, 0.3, 0.6, 0.9, 1.1)]
        deltas = [c.delta_eq for c in cals]
        for a, b in pairwise(deltas):
            assert a >= b - 1e-12

    def test_rejects_non_positive_Tc(self) -> None:
        with pytest.raises(ValueError, match="T_c must be positive"):
            calibrate_gap(T_c=0.0, T_bath=0.5)

    def test_rejects_negative_Tbath(self) -> None:
        with pytest.raises(ValueError, match="T_bath must be non-negative"):
            calibrate_gap(T_c=1.2, T_bath=-0.1)

    @pytest.mark.parametrize("bad_gap", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_supplied_zero_temperature_gap(self, bad_gap: float) -> None:
        with pytest.raises(ValueError, match="Delta_0 must be finite and positive"):
            calibrate_gap(T_c=1.2, T_bath=0.1, Delta_0=bad_gap)


class TestSolveGap:
    def test_cell_exact_integral_matches_piecewise_constant_reference(self) -> None:
        from scipy.integrate import quad

        delta = 2.0
        omega_D = 5.2
        edges = np.array([2.0, 2.3, 3.0, 4.2, 6.0])
        E = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        f = np.array([0.2, 0.05, 0.4, 0.1])

        actual = _gap_integral_f(delta, f, E, omega_D, widths)
        reference = 0.0
        for i, value in enumerate(f):
            lo = max(float(edges[i]), delta)
            hi = min(float(edges[i + 1]), omega_D)
            if hi <= lo:
                continue
            contribution, _ = quad(
                lambda energy, occupation=value: (
                    (1.0 - 2.0 * occupation) / np.sqrt(energy * energy - delta * delta)
                ),
                lo,
                hi,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=200,
            )
            reference += contribution

        assert actual == pytest.approx(reference, rel=0.0, abs=2e-12)
        legacy = _legacy_interpolated_gap_integral(delta, f, E, omega_D)
        assert abs(legacy - reference) > 4e-3

    def test_manufactured_cell_average_gap_has_no_interpolation_floor(self) -> None:
        from scipy.optimize import brentq

        delta_0 = 180.0
        T_c = delta_0 / (1.764 * KB_UEV_PER_K)
        cal = calibrate_gap(
            T_c=T_c,
            T_bath=0.3,
            Delta_0=delta_0,
            n_quadrature=4096,
            xtol=1e-12,
        )
        target = 0.95 * cal.delta_eq
        legacy_errors: list[float] = []
        finite_volume_errors: list[float] = []

        for num_bins in (40, 80, 160, 320):
            E, dE = build_energy_grid(cal.delta_eq, 0.75, 6.0, num_bins)
            widths = np.full(num_bins, dE)
            shape = np.exp(-np.maximum(E - target, 0.0) / (0.35 * cal.delta_eq))
            vacuum = _gap_integral_f(
                target,
                np.zeros_like(shape),
                E,
                cal._omega_D,
                widths,
            )
            unit_profile = _gap_integral_f(
                target,
                shape,
                E,
                cal._omega_D,
                widths,
            )
            amplitude = (vacuum - cal._ref_integral) / (vacuum - unit_profile)
            assert 0.0 < amplitude < 1.0
            f = amplitude * shape

            recovered = solve_gap(cal, f, E, widths, xtol=1e-12)
            finite_volume_errors.append(abs(recovered - target) / target)

            def legacy_residual(
                candidate: float,
                f_state: np.ndarray = f,
                energies: np.ndarray = E,
            ) -> float:
                return (
                    _legacy_interpolated_gap_integral(
                        candidate,
                        f_state,
                        energies,
                        cal._omega_D,
                    )
                    - cal._ref_integral
                )

            legacy = brentq(
                legacy_residual,
                0.8 * target,
                1.1 * cal.delta_eq,
                xtol=1e-12,
            )
            legacy_errors.append(abs(legacy - target) / target)

        assert max(finite_volume_errors) < 2e-12
        assert legacy_errors[0] > 5e-4
        assert all(fine < coarse for coarse, fine in pairwise(legacy_errors))
        assert legacy_errors[-1] < legacy_errors[0] / 50.0

    def test_explicit_standard_widths_match_derived_widths(self) -> None:
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        E, _ = build_energy_grid(cal.delta_eq, 0.75, 6.0, 200)
        widths = integration_widths_from_centers(E)
        f = np.exp(-E / (0.2 * cal.delta_eq)) * 0.01

        implicit = solve_gap(cal, f, E, xtol=1e-12)
        explicit = solve_gap(cal, f, E, widths, xtol=1e-12)

        assert explicit == pytest.approx(implicit, rel=0.0, abs=0.0)

    def test_smooth_thermal_roundtrip_improves_continuous_reference(self) -> None:
        from scipy.optimize import brentq

        delta_0 = 180.0
        T_c = delta_0 / (1.764 * KB_UEV_PER_K)
        T_bath = 0.5 * T_c
        # delta_eq is obtained independently from the continuous thermal cosh
        # quadrature in calibrate_gap, not from the cell-exact runtime residual.
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        E, _ = build_energy_grid(cal.delta_eq, 1.0, 10.0, 400)
        widths = integration_widths_from_centers(E)
        f = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * T_bath), 500.0)) + 1.0)

        finite_volume = solve_gap(cal, f, E, widths, xtol=1e-12)

        def legacy_residual(candidate: float) -> float:
            return (
                _legacy_interpolated_gap_integral(
                    candidate,
                    f,
                    E,
                    cal._omega_D,
                )
                - cal._ref_integral
            )

        legacy = brentq(
            legacy_residual,
            0.5 * cal.delta_eq,
            1.5 * cal.delta_eq,
            xtol=1e-12,
        )
        finite_volume_error = abs(finite_volume - cal.delta_eq)
        legacy_error = abs(legacy - cal.delta_eq)

        assert finite_volume_error < 0.92 * legacy_error

    def test_three_root_gap_edge_bump_uses_nearest_continuation_branch(self) -> None:
        delta_0 = 180.0
        T_c = delta_0 / (1.764 * KB_UEV_PER_K)
        calibration = calibrate_gap(
            T_c=T_c,
            T_bath=0.01,
            Delta_0=delta_0,
            xtol=1e-12,
        )
        E, dE_scalar = build_energy_grid(delta_0, 0.75, 6.0, 600)
        widths = np.full_like(E, dE_scalar)
        # A narrow, valid 0 <= f <= 1 occupation bump immediately below the
        # equilibrium edge makes the gap residual cross zero three times.
        f = 0.22 * np.exp(-0.5 * ((E - 175.75) / 0.75) ** 2)

        selected: list[float] = []
        for bracket_factor in (0.001, 0.01, 0.05, 0.5, 1.0):
            with pytest.warns(
                RuntimeWarning,
                match=r"multiple roots \(3 detected.*reference_gap",
            ):
                selected.append(
                    solve_gap(
                        calibration,
                        f,
                        E,
                        widths,
                        bracket_factor=bracket_factor,
                        reference_gap=176.5,
                        xtol=1e-12,
                    )
                )

        np.testing.assert_allclose(selected, selected[0], rtol=0.0, atol=2e-12)
        assert selected[0] == pytest.approx(176.6170286607, abs=2e-9)

        # Moving the continuation reference selects the adjacent physical
        # branches deterministically, while preserving the ambiguity warning.
        branch_roots: list[float] = []
        for reference in (174.0, 176.5, 179.5):
            with pytest.warns(RuntimeWarning, match="multiple roots"):
                branch_roots.append(
                    solve_gap(
                        calibration,
                        f,
                        E,
                        widths,
                        reference_gap=reference,
                        xtol=1e-12,
                    )
                )
        np.testing.assert_allclose(
            branch_roots,
            [173.8326437527, 176.6170286607, 179.9734475079],
            rtol=0.0,
            atol=2e-9,
        )

    def test_equilibrium_roundtrip(self) -> None:
        # Feeding in the thermal Fermi-Dirac occupation f_FD(E, T_bath)
        # must reproduce Δ_eq(T_bath) to high accuracy.
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        # Grid spans past ω_D = 100·kB·T_c so the integrand is captured.
        omega_D = 100.0 * KB_UEV_PER_K * T_c
        E = np.linspace(cal.delta_eq * 1.001, omega_D * 1.01, 3000)
        kT = KB_UEV_PER_K * T_bath
        f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        delta = solve_gap(cal, f, E)
        assert delta == pytest.approx(cal.delta_eq, rel=1e-3)

    def test_normal_state_returns_zero(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=2.0)  # Δ_eq = 0
        E = np.linspace(0.1, 5.0, 100)
        f = np.zeros_like(E)
        assert solve_gap(cal, f, E) == 0.0

    def test_sub_milli_microev_gap_is_not_hidden_by_absolute_floor(self) -> None:
        # A deliberately low-Tc BCS model has a perfectly physical
        # zero-temperature gap below 1e-3 μeV.  The former fixed lo_floor =
        # 1e-3 μeV put the entire root below the search domain.
        calibration = calibrate_gap(T_c=1e-6, T_bath=0.0, xtol=1e-16)
        assert 0.0 < calibration.delta_eq < 1e-3
        E, dE_scalar = build_energy_grid(calibration.delta_eq, 0.75, 6.0, 200)
        widths = np.full_like(E, dE_scalar)
        f = np.zeros_like(E)

        legacy_bracket = solve_gap(
            calibration,
            f,
            E,
            widths,
            xtol=1e-16,
        )
        continuation = solve_gap(
            calibration,
            f,
            E,
            widths,
            reference_gap=calibration.delta_eq,
            xtol=1e-16,
        )

        assert legacy_bracket == pytest.approx(calibration.delta_eq, abs=2e-15)
        assert continuation == pytest.approx(calibration.delta_eq, abs=2e-15)

    @pytest.mark.parametrize("bad_occupation", [-1e-12, 1.0 + 1e-12])
    def test_rejects_out_of_range_occupation(self, bad_occupation: float) -> None:
        calibration = calibrate_gap(T_c=1.2, T_bath=0.3)
        E = np.linspace(0.75 * calibration.delta_eq, 6.0 * calibration.delta_eq, 100)
        f = np.zeros_like(E)
        f[50] = bad_occupation

        with pytest.raises(ValueError, match=r"occupations in \[0, 1\]"):
            solve_gap(calibration, f, E)

    def test_extreme_nonequilibrium_gives_normal_state(self) -> None:
        # f ≈ 1 everywhere above the gap pushes (1 − 2f) < 0, so the
        # gap integral cannot equal the positive reference integral.
        # Expect the solver to detect this and return 0.
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        # Cover the entire possible gap interval down to the normal state;
        # collapse must not rely on inventing occupations below the grid.
        E = np.linspace(0.0, cal.delta_eq * 60.0, 2001)
        f = np.ones_like(E)
        assert solve_gap(cal, f, E) == 0.0

    def test_positive_residual_without_root_fails_closed(self) -> None:
        calibration = calibrate_gap(T_c=1.2, T_bath=0.3)
        inconsistent = replace(
            calibration,
            _ref_integral=-1.0,
            _inv_lambda=-1.0,
        )
        E = np.linspace(
            0.75 * calibration.delta_eq,
            6.0 * calibration.delta_eq,
            200,
        )

        with pytest.raises(RuntimeError, match="no fallback gap is returned"):
            solve_gap(inconsistent, np.zeros_like(E), E)

    def test_rejects_candidate_gap_below_grid_support_by_default(self) -> None:
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        omega_D = 100.0 * KB_UEV_PER_K * T_c
        # Deliberately omit all occupation support below 2 Δ_eq. The cold
        # distribution solves near Δ(0), far below the first reconstructed
        # cell edge, so the constant-left extrapolation is quantitatively
        # unsupported and must be reported.
        E = np.linspace(2.0 * cal.delta_eq, omega_D, 1000)
        f = np.zeros_like(E)

        with pytest.raises(
            ValueError,
            match=r"below the reconstructed.*allow_gap_edge_extrapolation",
        ):
            solve_gap(cal, f, E)

    def test_explicit_gap_edge_extrapolation_retains_legacy_warning(self) -> None:
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        omega_D = 100.0 * KB_UEV_PER_K * T_c
        E = np.linspace(2.0 * cal.delta_eq, omega_D, 1000)
        f = np.zeros_like(E)

        with pytest.warns(RuntimeWarning, match="explicitly enabled"):
            candidate = solve_gap(
                cal,
                f,
                E,
                allow_gap_edge_extrapolation=True,
            )

        lower_edge = E[0] - 0.5 * (E[1] - E[0])
        assert candidate < lower_edge

    def test_large_root_tolerance_cannot_bypass_gap_edge_contract(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=0.3)
        omega_D = 100.0 * KB_UEV_PER_K * 1.2
        E = np.linspace(2.0 * cal.delta_eq, omega_D, 1000)

        # Root accuracy and grid coverage are independent contracts. Before
        # this regression guard, xtol entered the support tolerance directly,
        # so this call returned a deeply off-grid gap without the explicit
        # extrapolation opt-in.
        with pytest.raises(
            ValueError,
            match=r"below the reconstructed.*allow_gap_edge_extrapolation",
        ):
            solve_gap(
                cal,
                np.zeros_like(E),
                E,
                xtol=cal.delta_eq,
            )

    def test_sub_part_per_billion_gap_edge_mismatch_is_numerical_coincidence(
        self,
    ) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=0.0)
        E, _ = build_energy_grid(
            cal.delta_eq,
            1.0 + 1e-10,
            6.0,
            200,
        )

        candidate = solve_gap(cal, np.zeros_like(E), E)

        lower_edge = E[0] - 0.5 * (E[1] - E[0])
        assert 0.0 < lower_edge - candidate < 1e-9 * lower_edge

    def test_rejects_continuation_anchor_below_grid_support(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=0.3)
        omega_D = 100.0 * KB_UEV_PER_K * 1.2
        E = np.linspace(2.0 * cal.delta_eq, omega_D, 1000)
        with pytest.raises(ValueError, match=r"candidate gap.*below the reconstructed"):
            solve_gap(
                cal,
                np.zeros_like(E),
                E,
                reference_gap=cal.delta_eq,
            )

    def test_rejects_non_boolean_gap_edge_extrapolation_flag(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=0.3)
        E = np.linspace(0.5 * cal.delta_eq, 6.0 * cal.delta_eq, 100)
        with pytest.raises(TypeError, match="must be a bool"):
            solve_gap(
                cal,
                np.zeros_like(E),
                E,
                allow_gap_edge_extrapolation=1,  # type: ignore[arg-type]
            )
