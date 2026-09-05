# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Regression tests for review packet P16 (2026-08-03).

Pins, in order:

* the centers-mode gap-edge reconstruction domain guard, and its bit-exact
  no-op property on a smooth occupation (``qpsim/observables/gap_suppression``);
* the documented zero-drive quadrature bias of ``compute_gap_suppression``;
* the documented absolute-floor behaviour of the Picard change test
  (``qpsim/services/steady_state``);
* the phonon ω-lattice commensurability condition now documented in
  ``qpsim/phonon_models/local``;
* the newline-independence of the recorded Fig. 6 P0 oracle digests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.grid.energy_grid import build_energy_grid
from qpsim.observables.gap_suppression import (
    compute_gap_suppression,
    edge_samples_from_centers,
    gap_integral_from_distribution_direct,
)
from qpsim.physics import fermi_dirac_occupation
from qpsim.physics.gap_equation import calibrate_gap
from qpsim.services.steady_state import _picard_convergence_ratio
from validation.source_provenance import source_sha256

GAP_UEV = 180.0


def _sharp_gap_edge_occupation() -> tuple[np.ndarray, np.ndarray]:
    """Near-saturated first cell followed by a sharp drop; f stays in [0, 1]."""
    h = 0.6
    E = GAP_UEV + 0.5 * h + h * np.arange(64)
    f = 0.3 * np.exp(-np.arange(64) / 5.0)
    f[0] = 1.0
    f[1] = 0.3
    return f, E


class TestDirectGapIntegralEdgeDomain:
    """The gap-edge node is reconstructed, and the reconstruction can overshoot.

    ``1.5*f[0] - 0.5*f[1] = 1.35`` is not an occupation, and it multiplies the
    BCS square-root endpoint weight of the most singular cell. Both routes to
    it now project onto the Pauli bound rather than raise: the overshoot is
    truncation error of the reconstruction, not a property of the caller's
    data, and raising left the headline observable undefined for the whole
    smooth saturated-edge class. The ``[0, 1]`` contract on INPUT data is a
    separate check and stays strict.
    """

    def test_centers_mode_projects_the_super_unity_reconstruction(self) -> None:
        f, E = _sharp_gap_edge_occupation()

        with pytest.warns(RuntimeWarning, match="projected onto the Pauli bound"):
            integral = gap_integral_from_distribution_direct(
                f, E, gap=GAP_UEV, samples="centers"
            )

        # Re-derived on this code, NOT copied from the prepared patch, which
        # proposed 0.212047 for this fixture -- 1.9% away from what the
        # implementation actually computes.
        assert integral == pytest.approx(0.21609093087522843, rel=1e-12)

    def test_edge_producer_and_consumer_agree_on_the_same_array(self) -> None:
        """The two reconstructions of one quantity must not disagree.

        This is the test that failed while the clip was applied to only one of
        them: the producer projected onto [0, 1] and the centers branch still
        raised, so the same f gave a number one way and an exception the other,
        and the exception's own advice -- "pass samples='edges'" -- routed the
        caller to the number it had just refused to compute.
        """
        f, E = _sharp_gap_edge_occupation()

        with pytest.warns(RuntimeWarning, match="projected onto the Pauli bound"):
            edge_samples = edge_samples_from_centers(f, E)
        assert edge_samples[0] == 1.0, "the extrapolate was not projected"

        via_edges = gap_integral_from_distribution_direct(
            edge_samples, E, gap=GAP_UEV, samples="edges"
        )
        with pytest.warns(RuntimeWarning):
            via_centers = gap_integral_from_distribution_direct(
                f, E, gap=GAP_UEV, samples="centers"
            )
        # Bit-identical, not approx: one shared projection means there is no
        # arithmetic left for the two paths to differ in.
        assert via_edges == via_centers

    def test_out_of_range_input_data_is_still_refused(self) -> None:
        """The projection is for the reconstruction, not for caller data.

        Clipping what a caller hands in would turn a genuine unit or sign error
        into a plausible number, which is the opposite of what the projection
        is for.
        """
        f, E = _sharp_gap_edge_occupation()
        bad = f.copy()
        bad[3] = 1.4

        with pytest.raises(ValueError, match="physical occupations"):
            gap_integral_from_distribution_direct(
                bad, E, gap=GAP_UEV, samples="edges"
            )
        with pytest.raises(ValueError, match="physical occupations"):
            gap_integral_from_distribution_direct(
                bad, E, gap=GAP_UEV, samples="centers"
            )

    @pytest.mark.parametrize(
        ("samples", "expected"),
        [
            ("centers", 2.9943396195485556e-05),
            ("edges", 2.753851349183186e-05),
        ],
    )
    def test_smooth_occupation_is_unchanged_by_the_domain_guard(
        self,
        samples: str,
        expected: float,
    ) -> None:
        # Values recorded from c269af2, i.e. before the guard was added: the
        # guard must never bite on a physically smooth driven distribution.
        E, _ = build_energy_grid(
            gap=GAP_UEV,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=400,
        )
        f = fermi_dirac_occupation(E, 0.2) + 2e-5 * np.exp(
            -(((E - 3.0 * GAP_UEV) / (0.3 * GAP_UEV)) ** 2)
        )

        integral = gap_integral_from_distribution_direct(
            f, E, gap=GAP_UEV, samples=samples
        )

        assert integral == pytest.approx(expected, rel=1e-12)


class TestComputeGapSuppressionQuadratureBias:
    @pytest.mark.parametrize("T_bath", [0.15, 0.3, 0.5])
    def test_thermal_input_bias_stays_below_the_documented_bound(
        self,
        T_bath: float,
    ) -> None:
        # Exactly thermal f, so the true suppression is zero; what is left is
        # the calibrate_gap-vs-solve_gap discretization mismatch documented on
        # compute_gap_suppression. Bound it against the physical scale of the
        # thermal suppression itself.
        T_c = 1.18
        calibration = calibrate_gap(T_c=T_c, T_bath=T_bath)
        E, _ = build_energy_grid(
            gap=calibration.delta_0_bcs,
            energy_min_factor=0.9,
            energy_max_factor=10.0,
            num_energy_bins=400,
        )
        f = fermi_dirac_occupation(E, T_bath)

        result = compute_gap_suppression(f, E, T_c=T_c, T_bath=T_bath)

        thermal_suppression = calibration.delta_0_bcs - calibration.delta_eq
        assert abs(result.delta_suppression) < 0.02 * thermal_suppression


class TestPicardConvergenceRatioIsAnAbsoluteFloor:
    def test_sub_floor_bin_is_certified_regardless_of_relative_change(self) -> None:
        # Documents the corrected docstring: at the shipped defaults every bin
        # with n < atol/rtol = 0.1 is tested absolutely, so a 5e4-relative
        # change of a sub-floor above-gap bin still certifies.
        n_ph = np.array([1.0, 1e-16])
        n_ph_new = n_ph.copy()
        n_ph_new[1] += 5e-12

        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=1e-10, atol=1e-11
        ) <= 1.0
        assert _picard_convergence_ratio(n_ph, n_ph_new, rtol=1e-10, atol=0.0) > 1.0


class TestPhononOmegaLatticeCommensurability:
    @pytest.mark.parametrize(
        ("num_energy_bins", "commensurate"),
        [(400, False), (405, True)],
    )
    def test_pair_difference_and_pair_sum_lattices_are_disjoint_unless_commensurate(
        self,
        num_energy_bins: int,
        commensurate: bool,
    ) -> None:
        # Characterizes the unguarded behaviour described in the local
        # module docstring: update this test together with any commensurability
        # guard or rate-preserving deposition that fixes it.
        E, _ = build_energy_grid(
            gap=GAP_UEV,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=num_energy_bins,
        )
        omega, idx_diff, idx_sum, _ = build_phonon_frequency_map(E)

        dE = float(E[1] - E[0])
        ratio = 2.0 * GAP_UEV / dE
        assert (abs(ratio - round(ratio)) < 1e-9) is commensurate

        used_diff = np.unique(idx_diff)
        used_sum = np.unique(idx_sum)
        shared = np.intersect1d(used_diff, used_sum)
        diff_only = np.setdiff1d(used_diff, used_sum)
        pair_breaking_orphans = omega[diff_only] > 2.0 * GAP_UEV

        if commensurate:
            assert shared.size > 0
            assert not np.any(pair_breaking_orphans)
        else:
            assert shared.size == 0
            assert np.any(pair_breaking_orphans)


class TestFig6OracleDigestsAreContentDefined:
    """The recorded P0 digests must stay bound to the LF (git-blob) content.

    ``validation/paper_parity.py`` hashes raw working-tree bytes, so on a CRLF
    checkout the Fig. 6 chain refuses to load at all.  The tempting local
    "fix" is to rebind the manifest to the CRLF digests, which would destroy
    the provenance chain and turn CI red; these two assertions fail loudly if
    anyone does.  They pass on LF and CRLF checkouts alike.
    """

    @staticmethod
    def _manifest() -> dict:
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "validation"
            / "paper_data"
            / "fischer_2023"
            / "fig6"
            / "oracle.json"
        )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_digitizer_source_digest_is_newline_independent(self) -> None:
        manifest = self._manifest()
        repository_root = Path(__file__).resolve().parents[2]
        script = repository_root / manifest["extraction"]["script"]

        assert source_sha256(script) == manifest["extraction"]["script_sha256"]

    def test_points_digest_is_bound_to_the_lf_bytes(self) -> None:
        manifest = self._manifest()
        points = (
            Path(__file__).resolve().parents[2]
            / "validation"
            / "paper_data"
            / "fischer_2023"
            / "fig6"
            / manifest["data"]["path"]
        )
        canonical = hashlib.sha256(
            points.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()

        assert canonical == manifest["data"]["sha256"]
