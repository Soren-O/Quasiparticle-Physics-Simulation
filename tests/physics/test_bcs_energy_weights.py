"""The energy-weighted BCS cell quadrature is exact, and shares its band
rules with the number weights.

``bcs_energy_cell_weights`` integrates ``E · E/sqrt(E² − Δ²)`` over each
cell's part of the band. It is checked against adaptive numerical quadrature
cell by cell -- including the first cell, where the integrand is singular and
a cell-centred product ``E_i w_i`` would be wrong by a factor that never
converges away -- and the number weights are checked to be unchanged by the
refactor that made the two share their band selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights, bcs_energy_cell_weights
from scipy.integrate import quad

GAP = 180.0


def _grid(n: int, lo: float = 1.0, hi: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo * GAP, hi * GAP, n + 1)
    return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)


def _numeric(lo: float, hi: float, moment: int) -> float:
    value, _err = quad(
        lambda e: e**moment * e / np.sqrt((e - GAP) * (e + GAP)), lo, hi,
        limit=200,
    )
    return value


@pytest.mark.parametrize("n", [8, 33, 96])
def test_energy_weights_match_numerical_quadrature_cell_by_cell(n: int) -> None:
    E, dE = _grid(n)
    edges = np.concatenate([[E[0] - 0.5 * dE[0]], E + 0.5 * dE])
    weights = bcs_energy_cell_weights(E, dE, GAP)
    expected = np.array([_numeric(edges[i], edges[i + 1], 1) for i in range(n)])
    np.testing.assert_allclose(weights, expected, rtol=1e-9)


def test_the_first_cell_is_where_a_centred_product_would_be_wrong() -> None:
    """Guards the reason for an exact integral: at the gap edge the product
    E_i w_i differs from the true moment by a fixed factor."""
    E, dE = _grid(33)
    number = bcs_dos_cell_weights(E, dE, GAP)
    energy = bcs_energy_cell_weights(E, dE, GAP)
    centred = E * number
    first_error = abs(centred[0] - energy[0]) / energy[0]
    later_error = abs(centred[10] - energy[10]) / energy[10]
    assert first_error > 1e-3
    assert later_error < 1e-4


def test_mean_energy_of_a_boltzmann_tail_lies_above_the_gap_by_order_kT() -> None:
    """Two moments of one quadrature give a mean energy that sits where a
    thermal quasiparticle's must: just above Δ, by an amount of order k_B T."""
    E, dE = _grid(400, 1.0, 6.0)
    kT = 86.17 * 0.3  # 0.3 K in μeV (k_B = 86.17 μeV/K)
    f = np.exp(-(E - GAP) / kT)
    number = bcs_dos_cell_weights(E, dE, GAP)
    energy = bcs_energy_cell_weights(E, dE, GAP)
    mean_e = float(np.sum(energy * f) / np.sum(number * f))
    assert GAP < mean_e < GAP + 3.0 * kT
    # Against the two continuum integrals, done adaptively; the tail the grid
    # truncates at 6 Δ is e^-36 of the peak and does not register.
    top = 6.0 * GAP

    def boltzmann(e: float, moment: int) -> float:
        return e**moment * e / np.sqrt((e - GAP) * (e + GAP)) * np.exp(-(e - GAP) / kT)

    num = quad(lambda e: boltzmann(e, 1), GAP, top, limit=200)[0]
    den = quad(lambda e: boltzmann(e, 0), GAP, top, limit=200)[0]
    # The cell-constant f is the only approximation left; 400 cells over 5 Δ
    # at kT = 0.14 Δ resolve the tail to better than a part in a thousand.
    assert mean_e == pytest.approx(num / den, rel=2e-3)


def test_number_weights_are_unchanged_by_sharing_the_band_logic() -> None:
    E, dE = _grid(48)
    edges = np.concatenate([[E[0] - 0.5 * dE[0]], E + 0.5 * dE])
    weights = bcs_dos_cell_weights(E, dE, GAP)
    expected = np.array([_numeric(edges[i], edges[i + 1], 0) for i in range(48)])
    np.testing.assert_allclose(weights, expected, rtol=1e-9)
    # An explicit band splits a cell for both moments identically.
    number = bcs_dos_cell_weights(E, dE, GAP, lower_bound=1.5 * GAP, upper_bound=2.5 * GAP)
    energy = bcs_energy_cell_weights(E, dE, GAP, lower_bound=1.5 * GAP, upper_bound=2.5 * GAP)
    assert np.array_equal(number > 0, energy > 0)
    assert number.sum() == pytest.approx(_numeric(1.5 * GAP, 2.5 * GAP, 0), rel=1e-9)
    assert energy.sum() == pytest.approx(_numeric(1.5 * GAP, 2.5 * GAP, 1), rel=1e-9)


def test_gap_zero_is_the_plain_energy_measure() -> None:
    E, dE = _grid(10, 0.0, 2.0)
    np.testing.assert_allclose(bcs_energy_cell_weights(E, dE, 0.0), E * dE)
