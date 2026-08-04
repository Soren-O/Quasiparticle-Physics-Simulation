# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review 2026-08-03, packet P05 regressions.

Covers the transport-operator cache ordering and the gap de-duplication in
``T3Spatial1DBackend``'s per-cell spectral helpers.  Both changes are
numerically neutral, so every assertion here is either bit-exact or a
structural one about which work runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial_1d import (
    T3Spatial1DBackend,
    T3Spatial1DState,
    _bcs_support_fraction,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel


def _stepped_state(
    *,
    NE: int = 32,
    NX: int = 13,
    interface_conductance: float | None = None,
) -> T3Spatial1DState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=4.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=60.0,
    )
    x = np.linspace(0.0, 100.0, NX)
    profile = np.where(np.arange(NX) < NX // 2, gap, gap * 200.0 / 180.0)
    kT = KB_UEV_PER_K * 0.15
    f0 = np.repeat(
        (1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0))[:, None], NX, axis=1
    )
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=0.15,
        diffusion_model=DiffusionModel.A1,
        gap_profile=profile,
        interface_conductance=interface_conductance,
    )


class TestPerCellSpectralDeduplication:
    def test_n1_per_cell_matches_per_column_construction(self) -> None:
        state = _stepped_state()
        backend = T3Spatial1DBackend()
        E, dE = state.spectral.E, state.spectral.dE

        from qpsim.backends.t3_spatial_1d import _represented_bcs_weights

        expected = np.column_stack(
            [_represented_bcs_weights(E, dE, float(g)) / dE for g in state.gap_profile]
        )

        np.testing.assert_array_equal(backend._n1_per_cell(state), expected)

    def test_support_fraction_matches_per_column_construction(self) -> None:
        state = _stepped_state()
        backend = T3Spatial1DBackend()
        E, dE = state.spectral.E, state.spectral.dE

        expected = np.column_stack(
            [_bcs_support_fraction(E, dE, float(g)) for g in state.gap_profile]
        )

        np.testing.assert_array_equal(
            backend._support_fraction_per_cell(state), expected
        )

    def test_smooth_profile_keeps_every_distinct_gap(self) -> None:
        """De-duplication must not merge cells that differ in gap."""
        state = _stepped_state()
        gap = float(state.spectral.gap)
        NX = state.f.shape[1]
        state.gap_profile = gap * (1.0 + 0.1 * np.linspace(0.0, 1.0, NX) ** 2)
        backend = T3Spatial1DBackend()

        support = backend._support_fraction_per_cell(state)

        assert np.unique(support, axis=1).shape[1] == NX


class TestTransportOperatorCacheOrdering:
    def test_cache_hit_skips_the_per_cell_spectral_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _stepped_state(interface_conductance=2.0)
        backend = T3Spatial1DBackend()
        first = backend._build_transport_operators(state, 0.5)

        def _boom(*args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("cache hit recomputed per-cell spectral data")

        monkeypatch.setattr(T3Spatial1DBackend, "_n1_per_cell", _boom)
        monkeypatch.setattr(T3Spatial1DBackend, "_support_fraction_per_cell", _boom)
        monkeypatch.setattr(
            "qpsim.backends.t3_spatial_1d._kl_interface_cell_average", _boom
        )

        assert backend._build_transport_operators(state, 0.5) is first

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda state: setattr(state, "interface_conductance", 3.0),
                id="interface_conductance",
            ),
            pytest.param(
                lambda state: state.gap_profile.__setitem__(0, 190.0),
                id="gap_profile",
            ),
            pytest.param(
                lambda state: setattr(state, "diffusion_model", DiffusionModel.A2),
                id="diffusion_model",
            ),
        ],
    )
    def test_cache_key_still_separates_operator_inputs(
        self, mutate: object
    ) -> None:
        state = _stepped_state(interface_conductance=2.0)
        backend = T3Spatial1DBackend()
        first = backend._build_transport_operators(state, 0.5)

        mutate(state)  # type: ignore[operator]

        assert backend._build_transport_operators(state, 0.5) is not first
