# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review 2026-08-03, packet P05 regressions.

Covers the transport-operator cache ordering and the gap de-duplication in the
per-cell spectral helpers. Both changes are numerically neutral, so every
assertion here is either bit-exact or a structural one about which work runs.

Written against ``T3Spatial1DBackend`` and moved to ``T3SpatialBackend`` when
that backend was retired. The properties are the unified backend's too -- it
groups cells by exact gap and caches transport operators on the same inputs --
so the packet's regressions stay covered rather than lapsing with the class
they were first found in.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.bcs_quadrature import (
    bcs_support_fraction,
    represented_bcs_weights,
)
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel

NE = 18
NX = 9
LENGTH_UM = 80.0
T_BATH = 0.15
D0 = 6.0


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _stepped_state(interface_conductance: float | None = None) -> T3SpatialState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.0, energy_max_factor=5.0, num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    profile = np.where(np.arange(NX) < NX // 2, gap, 1.25 * gap).astype(float)
    return T3SpatialState(
        f=np.repeat(_fermi_dirac(E, T_BATH)[:, None], NX, axis=1),
        geometry=strip(NX, mesh_size=LENGTH_UM / NX),
        spectral=spectral,
        material=material,
        T_bath=T_BATH,
        gap_per_cell=profile,
        interface_conductance=interface_conductance,
    )


class TestPerCellSpectralDeduplication:
    def test_n1_per_cell_matches_per_column_construction(self) -> None:
        state = _stepped_state()
        backend = T3SpatialBackend()
        E, dE = state.spectral.E, state.spectral.dE

        expected = np.column_stack(
            [represented_bcs_weights(E, dE, float(g)) / dE for g in state.gaps()]
        )

        np.testing.assert_array_equal(
            backend._per_cell(state.spectral, state.gaps(), "density"), expected
        )

    def test_support_fraction_matches_per_column_construction(self) -> None:
        state = _stepped_state()
        backend = T3SpatialBackend()
        E, dE = state.spectral.E, state.spectral.dE

        expected = np.column_stack(
            [bcs_support_fraction(E, dE, float(g)) for g in state.gaps()]
        )

        np.testing.assert_array_equal(
            backend._per_cell(state.spectral, state.gaps(), "support"), expected
        )

    def test_smooth_profile_keeps_every_distinct_gap(self) -> None:
        """De-duplication must not merge cells that differ in gap."""
        state = _stepped_state()
        gap = float(state.spectral.gap)
        state.gap_per_cell = gap * (1.0 + 0.1 * np.linspace(0.0, 1.0, NX) ** 2)
        backend = T3SpatialBackend()

        support = backend._per_cell(state.spectral, state.gaps(), "support")

        assert np.unique(support, axis=1).shape[1] == NX


class TestTransportOperatorCacheOrdering:
    @pytest.mark.xfail(
        reason=(
            "OPEN REGRESSION (performance, not correctness), found by moving "
            "this packet onto the unified backend. P05's whole point was that "
            "a transport-operator cache HIT skips the per-cell spectral "
            "quadratures. T3SpatialBackend cannot: _transport_weights runs "
            "first and its output IS the cache key "
            "(build(..., cache_key=(weights.tobytes(), ...))), so the work the "
            "cache exists to avoid is redone in order to look the cache up. "
            "Measured on this 2-gap, 9-cell, NE=18 fixture: 664 us per "
            "cache-hit call, of which 637 us -- 96% -- is _transport_weights. "
            "Un-xfail when the key is derived from the cheap inputs (gaps, "
            "diffusion model, conductance) rather than from the weights."
        ),
    )
    def test_cache_hit_skips_the_per_cell_spectral_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _stepped_state(interface_conductance=2.0)
        backend = T3SpatialBackend()
        first = backend._build_transport_operators(state, 0.5)

        def _boom(*args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("cache hit recomputed per-cell spectral data")

        monkeypatch.setattr(T3SpatialBackend, "_per_cell", staticmethod(_boom))

        assert backend._build_transport_operators(state, 0.5) is first

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda state: state.gap_per_cell.__setitem__(0, 190.0),
                id="gap_per_cell",
            ),
            pytest.param(
                lambda state: setattr(state, "diffusion_model", DiffusionModel.A2),
                id="diffusion_model",
            ),
        ],
    )
    def test_cache_key_still_separates_operator_inputs(self, mutate: object) -> None:
        state = _stepped_state(interface_conductance=2.0)
        backend = T3SpatialBackend()
        first = backend._build_transport_operators(state, 0.5)

        mutate(state)  # type: ignore[operator]

        assert backend._build_transport_operators(state, 0.5) is not first

    def test_the_barrier_conductance_is_in_the_operator_key(self) -> None:
        """Separated from the parametrised cases deliberately.

        ``_build_transport_operators`` does not apply the Kupriyanov-Lukichev
        overrides -- ``_transport_ops`` does -- so a conductance change has to
        be checked on the path that reads it, or the test passes while
        asserting nothing about the barrier.
        """
        state = _stepped_state(interface_conductance=2.0)
        backend = T3SpatialBackend()
        _transport, first = backend._transport_ops(state, 0.5)

        state.interface_conductance = 3.0
        _transport, second = backend._transport_ops(state, 0.5)

        assert second is not first
