"""Unit tests for the self-consistent n̄(P_read) fixed-point service."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from qpsim.constants import HBAR_UEV_NS
from qpsim.services.nbar_loop import (
    dbm_to_uev_per_ns,
    solve_nbar_loop,
)


class TestDbmToUevPerNs:
    def test_0_dbm_is_one_mW(self) -> None:
        # 0 dBm = 1 mW = 6.2415e12 μeV/ns.
        assert dbm_to_uev_per_ns(0.0) == pytest.approx(6.241509074e12, rel=1e-6)

    def test_minus_100_dbm_is_one_hundred_microattowatts(self) -> None:
        # −100 dBm = 1e-10 mW = 624 μeV/ns.
        assert dbm_to_uev_per_ns(-100.0) == pytest.approx(624.1509074, rel=1e-6)

    def test_decade_doubling(self) -> None:
        # 10 dB increase = 10× power.
        lo = dbm_to_uev_per_ns(-80.0)
        hi = dbm_to_uev_per_ns(-70.0)
        assert hi / lo == pytest.approx(10.0, rel=1e-10)

    @pytest.mark.parametrize("dbm", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_dbm_rejected(self, dbm: float) -> None:
        with pytest.raises(ValueError, match="dbm must be finite"):
            dbm_to_uev_per_ns(dbm)


class TestFixedQi:
    """With a constant Q_i the loop reduces to a 1-step analytic formula."""

    def _fixture(
        self, Q_i: float,
    ) -> tuple[Callable[[float], np.ndarray], Callable[[np.ndarray], float]]:
        def solve_f(n_bar: float) -> np.ndarray:
            return np.array([n_bar])  # placeholder f; compute_Q_i ignores it
        def compute_Q_i(f: np.ndarray) -> float:
            return Q_i
        return solve_f, compute_Q_i

    def test_converges_in_two_iterations(self) -> None:
        # Initial guess uses Q_i → ∞ (i.e. Q_tot = Q_c); iteration 0
        # steps to the correct n̄, iteration 1 confirms convergence.
        Q_i, Q_c, omega_0, P_read = 1e6, 5e5, 20.0, 1.0e5
        solve_f, compute_Q_i = self._fixture(Q_i)
        result = solve_nbar_loop(
            P_read_uev_per_ns=P_read, Q_c=Q_c, omega_0=omega_0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-8, max_iter=5,
        )
        Q_tot = 1.0 / (1.0 / Q_i + 1.0 / Q_c)
        expected = 2.0 * HBAR_UEV_NS * Q_tot**2 * P_read / (omega_0**2 * Q_c)
        assert result.converged
        assert result.iterations == 2
        assert result.n_bar == pytest.approx(expected, rel=1e-12)
        assert result.Q_i == Q_i
        assert result.Q_tot == pytest.approx(Q_tot, rel=1e-12)

    def test_large_Q_c_returns_to_prefactor_limit(self) -> None:
        # Q_c → ∞ ⇒ Q_tot → Q_i ⇒ n̄ → 2ℏ·Q_i²·P / (ω²·Q_c), which also
        # diverges as Q_c → ∞ — but the ratio n̄·Q_c / (Q_i²·2ℏP/ω²)
        # → 1. Spot-check with a very large finite Q_c.
        Q_i = 1e4
        Q_c = 1e12
        omega_0, P_read = 20.0, 1.0
        solve_f, compute_Q_i = self._fixture(Q_i)
        result = solve_nbar_loop(
            P_read_uev_per_ns=P_read, Q_c=Q_c, omega_0=omega_0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-8, max_iter=5,
        )
        expected = 2.0 * HBAR_UEV_NS * Q_i**2 * P_read / (omega_0**2 * Q_c)
        assert result.n_bar == pytest.approx(expected, rel=1e-4)


class TestNbarDependentQi:
    """When Q_i depends on n̄, the loop iterates to a self-consistent point.

    Model: ``Q_i(n̄) = Q_0 / (1 + n̄/n_sat)``. A saturation-driven Q
    drop of this form admits a closed-form fixed point for validation.
    """

    def test_converges_with_Qi_rolloff(self) -> None:
        Q_0, n_sat, Q_c, omega_0, P_read = 1e6, 1e5, 5e5, 20.0, 1.0e7

        def Q_i_of_nbar(n_bar: float) -> float:
            return Q_0 / (1.0 + n_bar / n_sat)

        # Injected callables — solve_f caches n_bar into the return
        # array so compute_Q_i can recover it.
        def solve_f(n_bar: float) -> np.ndarray:
            return np.array([n_bar])
        def compute_Q_i(f: np.ndarray) -> float:
            return Q_i_of_nbar(float(f[0]))

        result = solve_nbar_loop(
            P_read_uev_per_ns=P_read, Q_c=Q_c, omega_0=omega_0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-10, max_iter=100, under_relaxation=0.5,
        )
        assert result.converged

        # Validate: residual of the fixed-point map should vanish at
        # the converged state.
        Q_tot_check = 1.0 / (1.0 / Q_i_of_nbar(result.n_bar) + 1.0 / Q_c)
        n_bar_check = 2.0 * HBAR_UEV_NS * Q_tot_check**2 * P_read / (omega_0**2 * Q_c)
        assert result.n_bar == pytest.approx(n_bar_check, rel=1e-8)

    def test_history_length_equals_iterations(self) -> None:
        def solve_f(n_bar: float) -> np.ndarray:
            return np.array([0.0])
        def compute_Q_i(f: np.ndarray) -> float:
            return 1e6

        result = solve_nbar_loop(
            P_read_uev_per_ns=1.0, Q_c=5e5, omega_0=20.0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-8, max_iter=10,
        )
        assert len(result.history) == result.iterations
        assert result.history[0].iteration == 0


class TestInputValidation:
    @staticmethod
    def _trivial_solvers() -> tuple[
        Callable[[float], np.ndarray], Callable[[np.ndarray], float],
    ]:
        return (lambda n_bar: np.zeros(1)), (lambda f: 1e6)

    def test_negative_P_read_rejected(self) -> None:
        sf, cq = self._trivial_solvers()
        with pytest.raises(ValueError, match="P_read"):
            solve_nbar_loop(
                P_read_uev_per_ns=-1.0, Q_c=1e5, omega_0=20.0,
                solve_f=sf, compute_Q_i=cq,
            )

    def test_zero_omega_rejected(self) -> None:
        sf, cq = self._trivial_solvers()
        with pytest.raises(ValueError, match="omega_0"):
            solve_nbar_loop(
                P_read_uev_per_ns=1.0, Q_c=1e5, omega_0=0.0,
                solve_f=sf, compute_Q_i=cq,
            )

    def test_out_of_range_under_relaxation_rejected(self) -> None:
        sf, cq = self._trivial_solvers()
        with pytest.raises(ValueError, match="under_relaxation"):
            solve_nbar_loop(
                P_read_uev_per_ns=1.0, Q_c=1e5, omega_0=20.0,
                solve_f=sf, compute_Q_i=cq, under_relaxation=1.5,
            )

    @pytest.mark.parametrize(
        ("name", "overrides"),
        [
            ("P_read_uev_per_ns", {"P_read_uev_per_ns": float("nan")}),
            ("Q_c", {"Q_c": float("inf")}),
            ("omega_0", {"omega_0": float("nan")}),
            ("tol", {"tol": float("inf")}),
            ("under_relaxation", {"under_relaxation": float("nan")}),
            ("n_bar_initial", {"n_bar_initial": float("inf")}),
        ],
    )
    def test_non_finite_scalar_inputs_rejected(
        self, name: str, overrides: dict[str, float],
    ) -> None:
        sf, cq = self._trivial_solvers()
        kwargs: dict[str, object] = {
            "P_read_uev_per_ns": 1.0,
            "Q_c": 1e5,
            "omega_0": 20.0,
            "solve_f": sf,
            "compute_Q_i": cq,
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError, match=rf"{name} must be finite"):
            solve_nbar_loop(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_q_i", [float("nan"), 0.0, -1.0, float("-inf")])
    def test_final_resolve_rejects_invalid_q_i(self, bad_q_i: float) -> None:
        calls = 0

        def compute_Q_i(f: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            return 1e6 if calls == 1 else bad_q_i

        with pytest.raises(RuntimeError, match="final re-solve"):
            solve_nbar_loop(
                P_read_uev_per_ns=0.0,
                Q_c=1e5,
                omega_0=20.0,
                solve_f=lambda n_bar: np.array([n_bar]),
                compute_Q_i=compute_Q_i,
            )

    def test_positive_infinite_q_i_is_preserved_after_final_resolve(self) -> None:
        result = solve_nbar_loop(
            P_read_uev_per_ns=1.0,
            Q_c=1e5,
            omega_0=20.0,
            solve_f=lambda n_bar: np.array([n_bar]),
            compute_Q_i=lambda f: float("inf"),
        )

        assert np.isposinf(result.Q_i)
        assert result.Q_tot == 1e5


class TestNonConvergence:
    def test_returns_not_converged_at_max_iter(self) -> None:
        # Construct a map that oscillates: Q_i depends non-smoothly on n_bar.
        call_count = [0]

        def solve_f(n_bar: float) -> np.ndarray:
            call_count[0] += 1
            return np.array([n_bar])

        def compute_Q_i(f: np.ndarray) -> float:
            # Alternate between two very different Q_i values → oscillation.
            return 1e6 if call_count[0] % 2 == 0 else 1e3

        result = solve_nbar_loop(
            P_read_uev_per_ns=1e10, Q_c=1e5, omega_0=20.0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-12, max_iter=5, under_relaxation=1.0,
        )
        assert not result.converged
        assert result.iterations == 5


class TestFalloffToUnderRelaxation:
    def test_relaxation_does_not_scale_convergence_residual(self) -> None:
        # At zero drive the raw map is exactly zero. Starting from n_bar=1
        # leaves a raw relative residual of 1, even though a 1e-6 relaxed step
        # moves by less than the 1e-4 tolerance.
        result = solve_nbar_loop(
            P_read_uev_per_ns=0.0,
            Q_c=1e5,
            omega_0=20.0,
            solve_f=lambda n_bar: np.array([n_bar]),
            compute_Q_i=lambda f: 1e6,
            n_bar_initial=1.0,
            tol=1e-4,
            max_iter=1,
            under_relaxation=1e-6,
        )

        assert not result.converged

    def test_underrelaxation_still_converges(self) -> None:
        # Same Q_i(n̄) roll-off model, but with α = 0.1 — forces more
        # iterations but still converges to the same fixed point.
        Q_0, n_sat, Q_c, omega_0, P_read = 1e6, 1e5, 5e5, 20.0, 1.0e7

        def solve_f(n_bar: float) -> np.ndarray:
            return np.array([n_bar])
        def compute_Q_i(f: np.ndarray) -> float:
            return Q_0 / (1.0 + float(f[0]) / n_sat)

        result = solve_nbar_loop(
            P_read_uev_per_ns=P_read, Q_c=Q_c, omega_0=omega_0,
            solve_f=solve_f, compute_Q_i=compute_Q_i,
            tol=1e-10, max_iter=200, under_relaxation=0.1,
        )
        assert result.converged
        assert result.iterations > 5  # under-relaxation slows things down

        # Spot-check the fixed point is still correct.
        Q_tot = 1.0 / (1.0 / compute_Q_i(result.f) + 1.0 / Q_c)
        n_bar_expected = 2.0 * HBAR_UEV_NS * Q_tot**2 * P_read / (omega_0**2 * Q_c)
        assert result.n_bar == pytest.approx(n_bar_expected, rel=1e-6)
