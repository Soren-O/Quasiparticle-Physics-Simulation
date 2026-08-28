"""The exact two-bin cut-cell split for the phonon pair channel.

These tests pin the finite-volume geometry itself. The shipped phonon equation
is point-collocated instead: its per-frequency Kaplan correction and this area
split are two different, convergent discretizations of the same continuum
source, not two halves of one discrete operation.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from qpsim.collisions.omega_lattice import build_unified_omega_lattice
from qpsim.collisions.pair_split import PairSplitUnavailable, pair_split_fractions
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext
from scipy.integrate import quad

GAP = 180.0


def _grid(num_bins: int, *, max_factor: float = 10.0, min_factor: float = 1.0):
    E, _ = build_energy_grid(
        gap=GAP,
        energy_min_factor=min_factor,
        energy_max_factor=max_factor,
        num_energy_bins=num_bins,
    )
    return E, float(E[1] - E[0])


def _reference(i: int, j: int, h: float) -> float:
    """Adaptive-quadrature split fraction for one cell, independent of the module."""
    def xi(energy: float) -> float:
        return float(np.sqrt(max(energy * energy - GAP * GAP, 0.0)))

    lo_i, hi_i = xi(GAP + i * h), xi(GAP + (i + 1) * h)
    lo_j, hi_j = xi(GAP + j * h), xi(GAP + (j + 1) * h)
    face = 2.0 * GAP + (i + j + 1) * h

    def height(x: float) -> float:
        remainder = face - np.sqrt(GAP * GAP + x * x)
        if remainder <= GAP:
            return 0.0
        top = float(np.sqrt(remainder * remainder - GAP * GAP))
        return min(max(top - lo_j, 0.0), hi_j - lo_j)

    area = quad(height, lo_i, hi_i, limit=400)[0]
    return area / ((hi_i - lo_i) * (hi_j - lo_j))


class TestTheGeometryItClaims:
    def test_the_corner_cell_converges_to_pi_over_four(self) -> None:
        """The whole 4/π defect is this one number.

        In the state-counting coordinate ξ the corner cell is a square and the
        first frequency strip is the inscribed quarter disk, so the fraction is
        an area ratio that tends to π/4. Anything else means the split is not
        the geometry it claims to be.
        """
        errors = []
        for num_bins in (180, 405, 1620, 6480):
            E, _h = _grid(num_bins)
            errors.append(abs(pair_split_fractions(E, GAP)[0, 0] - np.pi / 4))

        assert errors[-1] < 2e-4, f"corner is {errors[-1]:.2e} from π/4 at NE=6480"
        # First order in h: each refinement should shrink the gap roughly in
        # proportion. Loose bound -- the point is that it CONVERGES, where the
        # defect it replaces does not.
        for coarse, fine in pairwise(errors):
            assert fine < coarse

    def test_cells_far_from_the_gap_split_evenly(self) -> None:
        """Away from the singularity the pushforward is symmetric.

        A cell whose two energies are both far above Δ sees an almost flat
        density, so its frequency support is almost symmetric about the bin
        face and the split must approach one half. A scheme that returned π/4
        everywhere would be applying a threshold correction to the whole grid.
        """
        E, _h = _grid(405)
        split = pair_split_fractions(E, GAP)
        assert split[200, 200] == pytest.approx(0.5, abs=1e-3)
        assert split[380, 380] == pytest.approx(0.5, abs=1e-3)
        assert split[0, 0] > 0.75, "the corner must NOT be near one half"

    def test_it_is_exactly_symmetric(self) -> None:
        """(i, j) and (j, i) are mirror images of one cell, not two cells.

        Exact equality rather than approximate: the implementation gets it by
        construction, and the reason matters -- the two orientations are not
        equally well conditioned, so a merely-approximate symmetry would mean
        it had integrated the badly conditioned one somewhere.
        """
        E, _h = _grid(180)
        split = pair_split_fractions(E, GAP)
        assert np.array_equal(split, split.T)

    def test_every_fraction_is_a_fraction(self) -> None:
        E, _h = _grid(180)
        split = pair_split_fractions(E, GAP)
        assert np.all(split >= 0.0) and np.all(split <= 1.0)


class TestItIsAccurate:
    @pytest.mark.parametrize(
        ("i", "j"),
        [(0, 0), (1, 0), (0, 1), (2, 5), (50, 50), (404, 0), (200, 204)],
    )
    def test_matches_adaptive_quadrature(self, i: int, j: int) -> None:
        """Checked against a different rule, not against itself."""
        E, h = _grid(405)
        got = pair_split_fractions(E, GAP)[i, j]
        assert got == pytest.approx(_reference(i, j, h), abs=5e-13)

    def test_the_corner_is_not_the_weak_cell(self) -> None:
        """It is the cell that carries the defect, so it gets its own rule.

        Both of its energies reach the gap, so the integrand meets a
        square-root endpoint whichever way round it is integrated -- the
        orientation choice that makes every other cell exact cannot help here.
        Without the substitution this cell sits at ~5e-07 while its neighbours
        are at 1e-16.
        """
        E, h = _grid(405)
        corner = pair_split_fractions(E, GAP)[0, 0]
        assert corner == pytest.approx(_reference(0, 0, h), abs=1e-13)


class TestRawWholeCellDeposit:
    @staticmethod
    def _deposit(num_bins: int):
        """Pair weight deposited on a lattice with FACES at 2Δ + m·h."""
        E, h = _grid(num_bins)
        n = E.size
        faces = GAP + np.arange(n + 1, dtype=float) * h
        cell_measure = np.diff(np.sqrt(np.maximum((faces - GAP) * (faces + GAP), 0.0)))
        weight = cell_measure[:, None] * cell_measure[None, :]
        split = pair_split_fractions(E, GAP)
        label = np.add.outer(np.arange(n), np.arange(n))
        size = 2 * n
        lower = np.bincount(label.ravel(), weights=(weight * split).ravel(), minlength=size)
        upper = np.bincount((label + 1).ravel(), weights=(weight * (1.0 - split)).ravel(), minlength=size)
        whole = np.bincount(label.ravel(), weights=weight.ravel(), minlength=size)
        return lower + upper, whole, weight.sum()

    @pytest.mark.parametrize("num_bins", [180, 405, 1620])
    def test_the_books_close_exactly(self, num_bins: int) -> None:
        """The property a finite-volume split has by construction.

        The split is a partition of unity per cell, so the deposited total is
        the matrix total to the last bit -- no event is created or lost, on any
        grid, for any occupation. This is not a comparison with the shipped
        point-collocation rule, whose source is a line value rather than a
        finite-volume total.
        """
        deposited, _whole, total = self._deposit(num_bins)
        assert deposited.sum() == pytest.approx(total, rel=0.0, abs=1e-9 * total)

    @pytest.mark.parametrize("num_bins", [180, 405, 1620])
    def test_raw_midpoint_has_four_over_pi_threshold_error(self, num_bins: int) -> None:
        """The raw whole-cell deposit overcounts the first finite-volume bin.

        The ratio of the uncorrected midpoint deposit to the split deposit in
        the threshold bin tends to 4/π = 1.2732 and does not shrink under
        refinement. The shipped point-collocation scheme is not this raw rule:
        it applies the Kaplan line-quadrature correction tested below.
        """
        deposited, whole, _total = self._deposit(num_bins)
        ratio = whole[0] / deposited[0]
        assert ratio > 1.27, f"expected the 4/π defect, got {ratio:.4f}"
        assert ratio == pytest.approx(4.0 / np.pi, abs=0.01)


class TestShippedPointCollocationConvergesWithTheSplit:
    def test_fixed_physical_window_converges(self) -> None:
        """The two representations approach the same continuum source.

        This uses the shipped corrected point-collocation path and a genuinely
        fixed physical window. Omitting the correction and holding a fixed
        number of cells produces the spurious 1.164 limit that originally
        motivated wiring the split.
        """
        errors = []
        window_uev = 24.0
        for num_bins in (45, 90, 180, 360):
            E, h = _grid(num_bins, max_factor=4.0)
            ctx = SpectralContext(
                E_bins=E,
                dE_bins=integration_widths_from_centers(E),
                gap=GAP,
            )
            f = 1e-3 * np.exp(-E / (0.2 * GAP))
            K_r = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=0.28)

            omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(E)
            point_source, _ = compute_phonon_source_sink(
                f,
                ctx,
                None,
                None,
                idx_diff,
                idx_sum,
                sign,
                omega.size,
                enable_scattering=False,
                K_r0_phonon_side=K_r,
            )

            lattice = build_unified_omega_lattice(E, GAP)
            n_qp = ctx.cell_density * f
            base = ctx.dE * (n_qp[:, None] * K_r * n_qp[None, :])
            volume_source = lattice.deposit(base, channel="pair")

            cut = 2.0 * GAP + window_uev + 1e-9 * h
            point_total = float(point_source[omega <= cut].sum())
            volume_total = float(
                volume_source[lattice.omega_bins <= cut].sum()
            )
            errors.append(abs(point_total / volume_total - 1.0))

        assert errors[-1] < 0.015
        assert all(
            fine < coarse
            for coarse, fine in pairwise(errors)
        )


class TestItRefusesGridsItCannotSplit:
    def test_a_grid_not_starting_at_the_gap_is_refused(self) -> None:
        """Off-face the support straddles three bins, not two.

        Returning a two-bin split anyway would be a plausible number computed
        for a geometry the cell does not have.
        """
        E, _h = _grid(180, min_factor=0.8)
        with pytest.raises(PairSplitUnavailable, match="lowest energy face"):
            pair_split_fractions(E, GAP)

    def test_a_non_uniform_grid_is_refused(self) -> None:
        E, _h = _grid(180)
        stretched = E.copy()
        stretched[-1] += 1.0
        with pytest.raises(PairSplitUnavailable, match="UNIFORM"):
            pair_split_fractions(stretched, GAP)
