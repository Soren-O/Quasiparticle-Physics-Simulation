# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Regression tests for the qpsim.collisions._uniform_grid success cache.

The cache added in 8fdc289 was keyed on ``id(E)``/``id(dE)`` plus size and
endpoint values.  Those terms do not identify a grid: ``SpectralContext.E``
and ``.dE`` mint a fresh read-only view per access, so the ids are freed and
recycled, and no key term constrained the interior spacings the guard exists
to check.  A cached success could therefore be replayed for a *different*,
non-uniform grid, silently handing the fixed-index-offset photon partner map
a grid it is invalid on.  These tests pin the content-addressed key.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions._uniform_grid import (
    _VALIDATED_GRIDS,
    _clear_validated_grids,
    uniform_grid_spacing,
)
from qpsim.collisions.pair_breaking_photon import validate_pair_breaking_photon_grid
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext

GAP = 175.0
SPACING = 10.0
NE = 41


def _uniform_centers() -> np.ndarray:
    return 180.0 + SPACING * np.arange(NE)


@pytest.fixture(autouse=True)
def _hermetic_cache():
    """No test may inherit or leak a remembered validation."""
    _clear_validated_grids()
    yield
    _clear_validated_grids()


class TestCacheIdentifiesTheGrid:
    def test_hit_returns_the_same_spacing_from_one_entry(self) -> None:
        E = _uniform_centers()
        dE = np.full(NE, SPACING)
        first = uniform_grid_spacing(E, dE, "probe")
        second = uniform_grid_spacing(E.copy(), dE.copy(), "probe")
        assert first == second == SPACING
        # Equal content is one entry regardless of how many array objects
        # carried it; under the id key this grew with every fresh view.
        assert len(_VALIDATED_GRIDS) == 1

    def test_in_place_interior_mutation_still_raises(self) -> None:
        # Deterministic form: the caller owns the array whose content the
        # cache remembered, and mutates its interior between two calls.
        E = _uniform_centers()
        dE = np.full(NE, SPACING)
        contract = validate_pair_breaking_photon_grid(
            E, dE, gap=GAP, omega_PB=40 * SPACING
        )
        assert contract.spacing == SPACING
        E[10] += 3.0  # size, E[0], E[-1], dE[0], dE[-1] all unchanged
        with pytest.raises(ValueError, match="uniform energy grid"):
            validate_pair_breaking_photon_grid(E, dE, gap=GAP, omega_PB=40 * SPACING)

    def test_distinct_grid_with_matching_endpoints_still_raises(self) -> None:
        # Recycled-id form: a fresh non-uniform array whose size and
        # endpoints match the cached uniform grid.
        uniform_grid_spacing(_uniform_centers(), np.full(NE, SPACING), "probe")
        kinked = _uniform_centers()
        kinked[20] += 3.0
        for _ in range(50):
            with pytest.raises(ValueError, match="uniform energy grid"):
                uniform_grid_spacing(kinked.copy(), np.full(NE, SPACING), "probe")

    def test_nonuniform_widths_with_matching_end_widths_still_raise(self) -> None:
        E = _uniform_centers()
        uniform_grid_spacing(E, np.full(NE, SPACING), "probe")
        graded = np.full(NE, SPACING)
        graded[10] = 1.4 * SPACING
        graded[11] = 0.6 * SPACING
        with pytest.raises(ValueError, match="uniform integration widths"):
            uniform_grid_spacing(E, graded, "probe")

    def test_interior_nonfinite_width_still_raises(self) -> None:
        E = _uniform_centers()
        uniform_grid_spacing(E, np.full(NE, SPACING), "probe")
        bad = np.full(NE, SPACING)
        bad[15] = np.nan
        with pytest.raises(ValueError, match="finite positive grid values"):
            uniform_grid_spacing(E, bad, "probe")

    def test_failures_are_never_cached(self) -> None:
        kinked = _uniform_centers()
        kinked[20] += 3.0
        for _ in range(5):
            with pytest.raises(ValueError, match="uniform energy grid"):
                uniform_grid_spacing(kinked, np.full(NE, SPACING), "probe")
        assert _VALIDATED_GRIDS == {}

    def test_eviction_keeps_the_cache_bounded(self) -> None:
        for k in range(40):
            spacing = SPACING + k
            uniform_grid_spacing(
                180.0 + spacing * np.arange(NE), np.full(NE, spacing), "probe"
            )
        assert len(_VALIDATED_GRIDS) <= 16


class TestShippedKernelKeepsTheGuard:
    def test_nonuniform_context_is_rejected_after_a_uniform_solve(self) -> None:
        E = _uniform_centers()
        ctx_ok = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=GAP
        )
        kinked = E.copy()
        kinked[20] += 3.0
        ctx_bad = SpectralContext(
            E_bins=kinked,
            dE_bins=integration_widths_from_centers(kinked),
            gap=GAP,
        )
        f = np.zeros(NE)
        for _ in range(100):
            sub_gap_photon_collision_rates(
                f, ctx_ok, omega_0=5 * SPACING, n_bar=0.5, c_phot=1.0
            )
            with pytest.raises(ValueError, match="uniform energy grid"):
                sub_gap_photon_collision_rates(
                    f, ctx_bad, omega_0=5 * SPACING, n_bar=0.5, c_phot=1.0
                )

    def test_cached_uniform_grid_returns_bit_identical_rates(self) -> None:
        E = _uniform_centers()
        ctx = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=GAP
        )
        f = np.full(NE, 0.01)
        cold = sub_gap_photon_collision_rates(
            f, ctx, omega_0=5 * SPACING, n_bar=0.5, c_phot=1.0
        )
        warm = sub_gap_photon_collision_rates(
            f, ctx, omega_0=5 * SPACING, n_bar=0.5, c_phot=1.0
        )
        np.testing.assert_array_equal(cold[0], warm[0])
        np.testing.assert_array_equal(cold[1], warm[1])
