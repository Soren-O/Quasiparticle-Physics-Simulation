"""One finite-volume frequency lattice for both phonon channels.

These pin the three structural properties the deposit rewiring depends on:
the lattice exists at all (the union-grid question), the deposit conserves
events, and deposit and read are exact adjoints -- the last being the one that
decides whether detailed balance can survive the change.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.omega_lattice import (
    build_unified_omega_lattice,
    effective_occupation,
)
from qpsim.collisions.pair_split import PairSplitUnavailable
from qpsim.grid.energy_grid import build_energy_grid

GAP = 180.0


def _grid(num_bins: int, *, max_factor: float = 10.0):
    E, _ = build_energy_grid(
        gap=GAP,
        energy_min_factor=1.0,
        energy_max_factor=max_factor,
        num_energy_bins=num_bins,
    )
    return E


class TestOneLatticeServesBothChannels:
    """The union-grid question (decision D3), resolved by construction.

    The pair channel needs bin faces at 2Δ + m·h; the scattering channel needs
    them at k·h. Those coincide exactly when 2Δ/h is an integer -- which is the
    commensurability the grid validator already enforces for the two channels
    to share bins at all. So the answer needs no new restriction.
    """

    @pytest.mark.parametrize(
        ("num_bins", "max_factor"), [(405, 10.0), (1620, 10.0), (66, 4.0)],
    )
    def test_the_shipped_grids_admit_it(self, num_bins: int, max_factor: float) -> None:
        lattice = build_unified_omega_lattice(_grid(num_bins, max_factor=max_factor), GAP)
        offset = 2.0 * GAP / lattice.h
        assert offset == pytest.approx(round(offset), abs=1e-9)
        # 2Δ lands on a bin FACE, which is what the pair split requires.
        assert lattice.pair_lower.min() == round(offset)

    def test_an_incommensurate_grid_is_refused_with_the_reason(self) -> None:
        """The old 400-bin default, which put the two channels on disjoint
        sublattices -- a scattering phonon above 2Δ could not break a pair."""
        E, _ = build_energy_grid(
            gap=GAP, energy_min_factor=1.0, energy_max_factor=10.0,
            num_energy_bins=400,
        )
        with pytest.raises(PairSplitUnavailable, match=r"2Δ/h"):
            build_unified_omega_lattice(E, GAP)

    def test_both_channels_index_the_same_array(self) -> None:
        lattice = build_unified_omega_lattice(_grid(405), GAP)
        assert lattice.pair_lower.max() + 1 < lattice.n_omega
        assert lattice.scatter_lower.max() + 1 < lattice.n_omega
        # Overlapping ranges are the point: a scattering-emitted phonon above
        # 2Δ must be able to land in a bin the pair channel reads.
        assert lattice.scatter_lower.max() > lattice.pair_lower.min()


class TestTheDepositConservesEvents:
    @pytest.mark.parametrize("channel", ["pair", "scatter"])
    def test_nothing_is_created_or_lost(self, channel: str) -> None:
        """Structural, not numerical: the two fractions are a partition of
        unity per cell, so the total survives whatever the quadrature error."""
        lattice = build_unified_omega_lattice(_grid(180), GAP)
        rng = np.random.default_rng(0)
        weights = rng.random((180, 180))

        deposited = lattice.deposit(weights, channel=channel)

        assert deposited.sum() == pytest.approx(weights.sum(), rel=1e-14)

    def test_a_deposit_is_not_vacuously_conserved(self) -> None:
        """Guard the guard: a deposit that put everything in one bin would
        also conserve the total, so check it actually spread."""
        lattice = build_unified_omega_lattice(_grid(180), GAP)
        deposited = lattice.deposit(np.ones((180, 180)), channel="pair")
        assert int(np.count_nonzero(deposited)) > 100


class TestDepositAndReadAreAdjoint:
    """The property the whole rewiring rests on.

    The phonon equation deposits an event across two bins; the quasiparticle
    equation reads a phonon occupation back for the same event. If the read
    takes one bin while the deposit spreads over two, the two discrete
    operators are not transposes and detailed balance breaks at the 1e-2 level.
    Reading with the deposit's own weights makes them exact transposes.
    """

    @pytest.mark.parametrize("channel", ["pair", "scatter"])
    def test_inner_products_agree(self, channel: str) -> None:
        lattice = build_unified_omega_lattice(_grid(180), GAP)
        rng = np.random.default_rng(1)
        weights = rng.random((180, 180))
        occupation = rng.random(lattice.n_omega)

        deposited = float(lattice.deposit(weights, channel=channel) @ occupation)
        read_back = float(
            np.sum(weights * effective_occupation(lattice, occupation, channel=channel))
        )

        assert deposited == pytest.approx(read_back, rel=1e-13)

    def test_a_single_bin_read_would_fail_this(self) -> None:
        """Non-vacuity: the naive read really does break the identity.

        Without this, the test above would pass for a lattice that never split
        anything, and the property it is protecting would be untested.
        """
        lattice = build_unified_omega_lattice(_grid(180), GAP)
        # Concentrated on the GAP-CORNER cell and read against an occupation
        # that differs sharply between its two bins. Random weights over the
        # whole matrix average the discrepancy away -- most cells split near
        # half -- so a diffuse probe would understate it to ~2e-4 and this
        # guard would pass for the wrong reason.
        weights = np.zeros((180, 180))
        weights[0, 0] = 1.0
        occupation = np.zeros(lattice.n_omega)
        occupation[lattice.pair_lower[0, 0]] = 1.0

        deposited = float(lattice.deposit(weights, channel="pair") @ occupation)
        naive = float(np.sum(weights * occupation[lattice.pair_lower]))

        # The corner splits ~0.78/0.22, so the naive read overstates by ~28%.
        assert naive == pytest.approx(1.0)
        assert deposited == pytest.approx(lattice.pair_split[0, 0], rel=1e-12)
        assert abs(deposited - naive) > 0.2 * abs(naive)
