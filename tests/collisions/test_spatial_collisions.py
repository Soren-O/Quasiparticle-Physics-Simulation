"""Tests for qpsim.collisions.spatial."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.spatial import (
    _GAP_COUNT_WARN,
    _MAX_CACHED_OPERATORS,
    SpatialCollisions,
)
from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext

D0 = 6.0
DT = 0.05
T_BATH = 0.15


def _context(ne: int = 32):
    material = load_material("Al")
    energies, _ = build_energy_grid(
        gap=material.Delta_0, energy_min_factor=1.0,
        energy_max_factor=5.0, num_energy_bins=ne,
    )
    spectral = SpectralContext(
        E_bins=energies,
        dE_bins=integration_widths_from_centers(energies),
        gap=material.Delta_0,
        diffusion_coefficient=D0,
    )
    return material, spectral


def _occupations(spectral, material, ncells, seed=11):
    rng = np.random.default_rng(seed)
    shape = (spectral.E.size, ncells)
    profile = 0.05 * np.exp(
        -((spectral.E - 2 * material.Delta_0) / 50.0) ** 2
    )
    return np.clip(
        profile[:, None] * np.ones((1, ncells)) + rng.uniform(0, 1e-3, shape),
        0.0, 1.0,
    )


def _collisions(spectral, material, gap_per_cell, **kwargs):
    return SpatialCollisions(
        spectral, gap_per_cell, tau_0=material.tau_0, T_c=material.T_c,
        T_bath=T_BATH, **kwargs,
    )




class TestDimensionAgnostic:
    def test_a_two_dimensional_cell_set_steps(self):
        """Cells are a flat axis; the geometry's shape never enters."""
        material, spectral = _context(ne=16)
        ncells = 12                       # e.g. a 3x4 mask, flattened
        f0 = _occupations(spectral, material, ncells)
        collisions = _collisions(
            spectral, material, np.full(ncells, material.Delta_0),
        )
        got = collisions.apply(f0.copy(), DT)
        assert got.shape == f0.shape
        assert np.all(got >= 0.0) and np.all(got <= 1.0)

    def test_a_single_cell_is_the_zero_dimensional_case(self):
        material, spectral = _context(ne=16)
        f0 = _occupations(spectral, material, 1)
        got = _collisions(
            spectral, material, np.array([material.Delta_0]),
        ).apply(f0.copy(), DT)
        assert got.shape == (spectral.E.size, 1)

    def test_each_gap_group_sees_only_its_own_cells(self):
        material, spectral = _context(ne=16)
        gaps = np.array([material.Delta_0, material.Delta_0, 260.0, 260.0])
        collisions = _collisions(spectral, material, gaps)
        assert collisions.distinct_gap_count == 2
        f0 = _occupations(spectral, material, 4)
        got = collisions.apply(f0.copy(), DT)
        # The high-gap columns have no states at the low-gap edge, so their
        # occupation there must not evolve into something the group cannot hold.
        high_operator = collisions.local_operator(260.0)
        unsupported = ~high_operator[4]
        assert np.array_equal(got[unsupported, 2], f0[unsupported, 2])


class TestTermSwitches:
    def test_switching_a_channel_off_changes_the_step(self):
        material, spectral = _context(ne=16)
        per_cell = np.full(4, material.Delta_0)
        f0 = _occupations(spectral, material, 4)
        both = _collisions(spectral, material, per_cell).apply(f0.copy(), DT)
        no_scat = _collisions(
            spectral, material, per_cell, enable_scattering=False,
        ).apply(f0.copy(), DT)
        no_rec = _collisions(
            spectral, material, per_cell, enable_recombination=False,
        ).apply(f0.copy(), DT)
        assert not np.array_equal(both, no_scat)
        assert not np.array_equal(both, no_rec)

    def test_both_channels_off_leaves_the_occupation_alone(self):
        material, spectral = _context(ne=16)
        f0 = _occupations(spectral, material, 3)
        got = _collisions(
            spectral, material, np.full(3, material.Delta_0),
            enable_scattering=False, enable_recombination=False,
        ).apply(f0.copy(), DT)
        assert np.array_equal(got, f0)


class TestOperatorCache:
    def test_repeated_lookups_reuse_one_operator(self):
        material, spectral = _context(ne=16)
        collisions = _collisions(
            spectral, material, np.full(3, material.Delta_0),
        )
        first = collisions.local_operator(material.Delta_0)
        assert collisions.local_operator(material.Delta_0) is first

    def test_the_cache_is_bounded(self):
        material, spectral = _context(ne=16)
        collisions = _collisions(
            spectral, material, np.full(3, material.Delta_0),
        )
        for k in range(_MAX_CACHED_OPERATORS + 3):
            collisions.local_operator(material.Delta_0 + 10.0 * k)
        # Each entry owns three dense (NE, NE) matrices.
        assert len(collisions._cache) <= _MAX_CACHED_OPERATORS


class TestGuards:
    def test_a_continuously_varying_gap_is_warned_about(self):
        """Grouping is by EXACT gap, so this is the pathological case.

        Tolerable when a 1-D strip had tens of cells; a 2-D mask has
        thousands, and each distinct gap carries its own dense kernels.
        """
        material, spectral = _context(ne=8)
        gaps = material.Delta_0 + np.arange(_GAP_COUNT_WARN + 1) * 0.5
        with pytest.warns(RuntimeWarning, match="distinct local gaps"):
            _collisions(spectral, material, gaps)

    def test_a_broadened_density_of_states_is_refused(self):
        material, spectral = _context(ne=8)
        broadened = SpectralContext(
            E_bins=spectral.E, dE_bins=spectral.dE, gap=material.Delta_0,
            dynes_gamma=1.0, diffusion_coefficient=D0,
        )
        with pytest.raises(ValueError, match="pure-BCS"):
            _collisions(broadened, material, np.full(2, material.Delta_0))

    def test_a_non_positive_step_is_refused(self):
        material, spectral = _context(ne=8)
        collisions = _collisions(
            spectral, material, np.full(2, material.Delta_0),
        )
        f0 = _occupations(spectral, material, 2)
        with pytest.raises(ValueError, match="dt must be positive"):
            collisions.apply(f0, 0.0)
