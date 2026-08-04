# The module name carries the review packet id, which N999 wants lowercase.
# ruff: noqa: N999
"""Regression gates for the 2026-08-03 review, packet P12.

Covers the degenerate ``calibrate_gap`` bracket endpoint and the Dynes
coverage contract of the ``qpsim.observables.density`` integrals.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid
from qpsim.observables.density import qp_fraction_paper
from qpsim.physics.gap_equation import calibrate_gap
from qpsim.physics.spectral import (
    SpectralContext,
    dynes_density_of_states,
    fermi_dirac_occupation,
)
from scipy.integrate import quad

_GAP_UEV = 180.0


class TestCalibrateGapDegenerateEndpoint:
    """At T_bath ≪ T_c the residual at Δ0_bcs is pure trapezoid roundoff."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"n_quadrature": 16},
            {"n_quadrature": 64},
            {"omega_D_over_Tc": 1000.0},
        ],
    )
    def test_non_negative_endpoint_residual_returns_delta_0_bcs(
        self, overrides: dict[str, float]
    ) -> None:
        # Every 1 − 2f sample has rounded to exactly 1.0, so the endpoint
        # residual is the summation roundoff of the trapezoid; a positive
        # ULP used to abort the calibration with the scipy bracket error
        # "f(a) and f(b) must have different signs".
        calibration = calibrate_gap(T_c=1.2, T_bath=0.012, **overrides)
        assert calibration.delta_eq == calibration.delta_0_bcs

    def test_default_cold_path_is_unchanged(self) -> None:
        # The shipped defaults land on an exactly-zero endpoint residual,
        # which brentq accepted by returning the endpoint.
        calibration = calibrate_gap(T_c=1.2, T_bath=0.012)
        assert calibration.delta_eq == calibration.delta_0_bcs

    def test_resolved_thermal_suppression_still_solves_the_root(self) -> None:
        # Where the thermal correction is representable the bracketed root
        # solve is untouched; pinned against the pre-fix implementation.
        calibration = calibrate_gap(T_c=1.2, T_bath=0.96)
        assert calibration.delta_eq < calibration.delta_0_bcs
        assert calibration.delta_eq == pytest.approx(
            129.68017556998916, rel=1e-13
        )


def _dynes_context(min_factor: float, *, gamma_ratio: float = 1e-3) -> SpectralContext:
    E, dE = build_energy_grid(_GAP_UEV, min_factor, 10.0, 2000)
    return SpectralContext(
        E,
        np.full_like(E, dE),
        _GAP_UEV,
        dynes_gamma=gamma_ratio * _GAP_UEV,
    )


def _continuum_x_qp_paper(gamma_ratio: float, T: float) -> float:
    """Adaptive-quadrature reference for ``2/Δ_0 ∫_0^{10Δ} ρ(E) f(E) dE``."""
    gamma = gamma_ratio * _GAP_UEV

    def integrand(energy: float) -> float:
        rho = float(dynes_density_of_states(np.array([energy]), _GAP_UEV, gamma)[0])
        occupation = float(fermi_dirac_occupation(np.array([energy]), T)[0])
        return rho * occupation

    below, _ = quad(integrand, 0.0, _GAP_UEV, epsabs=1e-14, epsrel=1e-11)
    above, _ = quad(
        integrand,
        _GAP_UEV,
        10.0 * _GAP_UEV,
        points=[_GAP_UEV],
        epsabs=1e-14,
        epsrel=1e-11,
    )
    return 2.0 * (below + above) / _GAP_UEV


class TestDynesCoverageContract:
    """A Dynes DOS is occupied on [0, Δ); that region is most of n_qp."""

    def test_grid_starting_at_the_gap_is_rejected(self) -> None:
        ctx = _dynes_context(1.0)
        f = fermi_dirac_occupation(ctx.E, 0.20)
        with pytest.raises(ValueError, match="Dynes sub-gap band"):
            qp_fraction_paper(f, ctx, delta_0=_GAP_UEV)

    def test_full_domain_matches_a_continuum_reference(self) -> None:
        ctx = _dynes_context(0.0)
        f = fermi_dirac_occupation(ctx.E, 0.20)
        value = qp_fraction_paper(f, ctx, delta_0=_GAP_UEV)
        assert value == pytest.approx(
            _continuum_x_qp_paper(1e-3, 0.20), rel=1e-2
        )

    def test_truncated_grid_would_have_lost_most_of_the_density(self) -> None:
        # Why the contract is not optional: the rejected grid returns a
        # value ~7.5x too small at Γ/Δ = 1e-3, and worse at larger Γ.
        full = _continuum_x_qp_paper(1e-3, 0.20)
        ctx = _dynes_context(1.0)
        f = fermi_dirac_occupation(ctx.E, 0.20)
        truncated = float(np.sum(f * ctx.cell_weights)) * 2.0 / _GAP_UEV
        assert truncated < 0.2 * full
