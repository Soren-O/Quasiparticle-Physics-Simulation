"""The 1-D reduction of the unified spatial backend: the stopping residual.

The unified :class:`~qpsim.backends.spatial.SpatialBackend` solves the strip
as a ``(1, N)`` mask, and its driver is
:meth:`~qpsim.backends.spatial.SpatialBackend.run`.

The property under test: the convergence certificate is the RAW endpoint
operator residual -- transport and collision/source summed pointwise BEFORE
the norm, and evaluated before any occupation clipping. The backend assembles
both halves in :meth:`~qpsim.backends.spatial.SpatialBackend.rates` and the
driver takes one ``max|.|`` of what comes back, so the first test constructs
the cancellation physically (an external gain that annihilates the summed
rate) rather than monkeypatching two halves that are not separate seams.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.spatial import SpatialBackend, SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import Geometry, strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _strip_geometry(NX: int) -> Geometry:
    """The strip grid: ``x = linspace(0, 100, NX)``, so the spacing is
    ``100/(NX-1)`` -- read off the array rather than assumed."""
    x = np.linspace(0.0, 100.0, NX)
    return strip(NX, mesh_size=float(x[1] - x[0]))


def _build_state(
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
    NX: int = 11,
) -> SpatialState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=5.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    geometry = _strip_geometry(NX)
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return SpatialState(
        f=f0,
        geometry=geometry,
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


def _collision_only_rates(state: SpatialState) -> np.ndarray:
    """``df/dt`` from collisions alone, cell by cell.

    Collisions are strictly local, so one cell of the strip evaluated on its own
    ``1x1`` mask sees exactly the collision half of that column's rate: a single
    reflective cell has no acting device face, so ``rates`` skips the transport
    contribution entirely (``spatial.py``, same guard as ``apply_transport``).
    This is only used to show that BOTH halves of the residual are individually
    live in the test below -- it is not itself the quantity under test.
    """
    cell = strip(1, mesh_size=state.geometry.mesh_size)
    columns = []
    for j in range(state.f.shape[1]):
        one = SpatialState(
            f=state.f[:, j : j + 1].copy(),
            geometry=cell,
            spectral=state.spectral,
            material=state.material,
            T_bath=state.T_bath,
        )
        # A fresh backend per column: the collision layer caches on the gap, and
        # sharing one would only obscure what this helper is for.
        columns.append(SpatialBackend().rates(one)[:, 0])
    return np.column_stack(columns)


class TestStripRunResidual:
    """The 1-D reduction of the unified backend's stopping residual."""

    def test_transport_and_source_rates_cancel_before_norm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A spatially perturbed thermal strip: the transport half of the
        # residual is nonzero because of the gradient, the collision half
        # because the perturbed column is off-equilibrium.
        state = _build_state(T_bath=0.1, NE=8, NX=5)
        f = state.f.copy()
        f[:, 0] = np.clip(f[:, 0] + 0.05, 0.0, 1.0)
        state = replace(state, f=f)
        backend = SpatialBackend()

        # Freeze the state so the residual is measured at the configuration
        # that was constructed rather than at wherever one step happens to
        # land.
        monkeypatch.setattr(backend, "step", lambda state_arg, _dt, **_kwargs: state_arg)

        bare = backend.rates(state)
        baseline = float(np.max(np.abs(bare)))
        assert baseline > 0.0

        # Both halves must be individually large, or the cancellation below
        # would be trivially satisfied by a residual that was one-sided anyway.
        collision_only = _collision_only_rates(state)
        transport_only = bare - collision_only
        collision_norm = float(np.max(np.abs(collision_only)))
        transport_norm = float(np.max(np.abs(transport_only)))
        assert collision_norm > 0.0
        assert transport_norm > 0.0

        # An external source that annihilates the SUM pointwise. It enters
        # `combined_group_rates` additively as a gain, so the total endpoint
        # rate is `bare + external_gain` = 0 -- while transport and collisions
        # each stay at their full size. Norming the two separately and adding
        # would report `transport_norm + collision_norm` and never converge.
        norm_then_add = transport_norm + collision_norm
        stop_tol = 1.0e-6 * baseline
        # A driver that normed the two halves separately could not pass this
        # tolerance, so `converged` below is the whole assertion.
        assert stop_tol < 1.0e-3 * norm_then_add

        result = backend.run(
            state,
            dt=1.0,
            max_time=2.0,
            stop_tol=stop_tol,
            external_gain=-bare,
            snapshot_interval=2.0,
        )

        assert result.converged
        assert result.n_steps == 1
        assert result.snapshots[-1].max_rate < stop_tol
        # The cancellation here is real arithmetic rather than two exactly
        # opposite arrays, so what survives is that the reported residual is
        # rounding dust against the size of the two halves that cancelled --
        # measured at 2.8e-17 of the baseline, against 1e-12 here.
        assert result.snapshots[-1].max_rate < 1.0e-12 * baseline
        # The residual that a norm-then-add driver would have reported is
        # orders of magnitude above the one actually reported.
        assert norm_then_add > 1.0e3 * max(result.snapshots[-1].max_rate, 0.0)

    def test_constant_thermal_state_stops_on_raw_operator_residual(self) -> None:
        state = _build_state(T_bath=0.1, NE=12, NX=5)
        result = SpatialBackend().run(
            state,
            dt=1.0,
            max_time=10.0,
            stop_tol=1.0e-6,
            snapshot_interval=10.0,
        )

        assert result.converged
        assert result.n_steps == 1
        assert result.snapshots[-1].max_rate < 1.0e-6

    def test_saturated_positive_source_does_not_claim_convergence(self) -> None:
        state = _build_state(T_bath=0.0, NE=8, NX=3)
        state.f[:] = 1.0
        # A pure gain source: gain=..., loss_rate=zeros.
        gain = np.full_like(state.f, 1.0e6)

        result = SpatialBackend().run(
            state,
            dt=0.1,
            max_time=0.3,
            stop_tol=1.0,
            snapshot_interval=0.3,
            external_gain=gain,
            external_loss=None,
        )

        assert not result.converged
        assert result.n_steps == 3
        np.testing.assert_allclose(result.state.f, 1.0, rtol=0.0, atol=1.0e-15)
        assert result.snapshots[-1].max_rate > 1.0
