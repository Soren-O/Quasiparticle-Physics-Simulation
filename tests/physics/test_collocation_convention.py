"""The two populations are discretised differently, and that is on purpose.

Quasiparticles are stored as CELL AVERAGES: what is held for a cell is the
occupation averaged across it, and the count is the density of states
*integrated* over the cell. Phonons are stored as POINT SAMPLES: every
frequency bin is an exact event frequency and holds the occupation number of
that one mode, carrying no measure.

Both are right for their own equation -- the quasiparticle equation integrates
over states, the phonon equation is stated per state -- and the hazard is code
that forgets which one it is holding. These tests pin the two conventions and
the one place that mixes them.

See "Quasiparticles are stored as cell averages, phonons as point samples" in
``docs/Phonon_Model_Decisions.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext, bcs_density_of_states

GAP = 180.0


def _grid(num_bins: int, *, min_factor: float = 1.0) -> SpectralContext:
    E, _ = build_energy_grid(
        gap=GAP,
        energy_min_factor=min_factor,
        energy_max_factor=10.0,
        num_energy_bins=num_bins,
    )
    return SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=GAP
    )


class TestQuasiparticlesAreCellAverages:
    def test_gap_edge_point_sample_is_wrong_by_exactly_root_two(self) -> None:
        """Refining the mesh does not fix it -- it converges to the wrong answer.

        This is the same disease as the pair-marginal 4/pi, and it is worth
        having the exact constant rather than an observation. On a grid whose
        lowest face sits on the gap, the first cell is [D, D+h] and

            cell integral  = sqrt((D+h)^2 - D^2) = sqrt(2Dh + h^2) -> sqrt(2Dh)
            point sample   = N(D + h/2) * h                        -> sqrt(Dh)

        so their ratio tends to 1/sqrt(2) and STAYS there: the point sample
        undercounts the gap-edge cell by 29.3% at every resolution. An error
        that survives refinement is in the representation, not the mesh, which
        is why no convergence study catches it.
        """
        ratios = []
        exact = []
        for num_bins in (180, 720, 2880, 11520):
            ctx = _grid(num_bins)
            weights = bcs_dos_cell_weights(ctx.E, ctx.dE, ctx.gap, lower_bound=ctx.gap)
            live = int(np.argmax(weights > 0.0))
            cell_integral = float(weights[live])
            point_sample = (
                float(bcs_density_of_states(ctx.E[live : live + 1], GAP)[0])
                * float(ctx.dE[live])
            )
            exact.append(cell_integral)
            ratios.append(point_sample / cell_integral)

        # The cell integral shrinks like sqrt(h), as a measure across a
        # square-root singularity must.
        assert exact[0] > exact[1] > exact[2] > exact[3]
        for cell_integral, finer in zip(exact, exact[1:]):
            np.testing.assert_allclose(cell_integral / finer, 2.0, rtol=0.02)

        # And the point sample converges -- to the wrong constant.
        assert ratios[-1] < ratios[0], f"ratio should approach 1/sqrt(2), got {ratios}"
        np.testing.assert_allclose(ratios[-1], 1.0 / np.sqrt(2.0), rtol=1e-3)

    def test_cell_weights_sum_to_the_analytic_measure(self) -> None:
        """Summing the cells reproduces the closed-form integrated DOS."""
        ctx = _grid(900)
        weights = bcs_dos_cell_weights(ctx.E, ctx.dE, ctx.gap, lower_bound=ctx.gap)
        upper = float(ctx.E[-1] + 0.5 * ctx.dE[-1])
        # \int N(E) dE = sqrt(E^2 - gap^2) for the BCS density of states.
        analytic = np.sqrt(max(upper * upper - GAP * GAP, 0.0))
        np.testing.assert_allclose(weights.sum(), analytic, rtol=1e-10)


class TestPhononsArePointSamples:
    def test_every_frequency_bin_is_an_exact_event_frequency(self) -> None:
        """No bin is an interpolation -- each one is a frequency something emits.

        This is what makes the phonon occupation a per-mode number rather than
        a cell average, and therefore why it carries no density of states.
        """
        ctx = _grid(60)
        omega_bins, idx_diff, idx_sum, _ = build_phonon_frequency_map(ctx.E)

        events = np.concatenate(
            [
                np.abs(ctx.E[:, None] - ctx.E[None, :]).ravel(),
                (ctx.E[:, None] + ctx.E[None, :]).ravel(),
            ]
        )
        for omega in omega_bins:
            assert np.min(np.abs(events - omega)) <= 1e-9 * max(1.0, abs(omega)), (
                f"omega bin {omega:g} is not an event frequency"
            )

        # And the maps land every pair on the bin holding its own frequency.
        np.testing.assert_allclose(
            omega_bins[idx_diff], np.abs(ctx.E[:, None] - ctx.E[None, :]), atol=1e-9
        )
        np.testing.assert_allclose(
            omega_bins[idx_sum], ctx.E[:, None] + ctx.E[None, :], atol=1e-9
        )


class TestTheOnePlaceThatMixesThem:
    """Junction band weights fall back to a point sample under broadening.

    Accurate only while the broadening is at least a cell wide. Unreachable in
    shipped runs -- transport rejects a Dynes context outright -- so this pins
    the guard, not the physics.
    """

    def _weights(self, gamma: float, num_bins: int) -> None:
        from qpsim.devices.m25_junction import _spectral_band_weights

        E, _ = build_energy_grid(
            gap=GAP, energy_min_factor=0.8, energy_max_factor=10.0,
            num_energy_bins=num_bins,
        )
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=GAP,
            dynes_gamma=gamma,
        )
        # The band must start no lower than the grid does, or the weights
        # refuse rather than silently dropping the uncovered support.
        _spectral_band_weights(ctx, lower_bound=float(E[0] - 0.5 * ctx.dE[0]))

    def test_warns_when_broadening_is_narrower_than_a_cell(self) -> None:
        with pytest.warns(RuntimeWarning, match="only when the broadening"):
            self._weights(gamma=0.5, num_bins=180)

    def test_silent_when_the_broadening_resolves_the_grid(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self._weights(gamma=40.0, num_bins=180)

    def test_this_one_is_repaired_by_refinement_unlike_the_bcs_case(self) -> None:
        """The distinction that makes this branch tolerable and the other not.

        Broadening leaves a smooth function, so the point sample is merely
        under-resolved and refining the mesh fixes it -- a grid that warns
        becomes a grid that is quiet. Contrast
        ``test_gap_edge_point_sample_is_wrong_by_exactly_root_two``, where no
        amount of refinement helps because the target is singular.
        """
        with pytest.warns(RuntimeWarning):
            self._weights(gamma=2.0, num_bins=180)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self._weights(gamma=2.0, num_bins=2880)

    def test_pure_bcs_path_never_warns(self) -> None:
        """The branch that actually ships uses the exact cell measure."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            self._weights(gamma=0.0, num_bins=2880)
