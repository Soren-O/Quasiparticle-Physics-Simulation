"""Transport on the 1-D reduction of the unified spatial backend.

Translated from ``TestT3Spatial1DTransport`` in
``tests/backends/test_t3_spatial_1d.py``, which tested the now-retired
``T3Spatial1DBackend`` / ``T3Spatial1DState``. The same strip is a
``T3SpatialState`` on a ``strip(N, mesh_size=dx)`` geometry -- a mask of shape
``(1, N)`` -- so every property here is a statement about the unified backend
solving the degenerate one-cell-wide case.

Two guards the retired backend carried have no counterpart on the unified
backend and are held as strict xfails rather than dropped, because the
property they assert is physics, not an implementation detail of the class
that is going away:

* the ``[0, 1]`` clip escalation (warn, then fail) -- ``SpatialTransport.apply``
  still MEASURES the clipped density, but ``T3SpatialBackend.apply_transport``
  discards the diagnostics dict, so nothing acts on it;
* the pure-BCS transport contract -- a Dynes-broadened context is accepted and
  then silently ignored by the transport dressings.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
    NX: int = 11,
    dx: float | None = None,
) -> T3SpatialState:
    """The retired ``_build_state`` on a strip geometry.

    The 1-D tests laid cells on ``x = np.linspace(0.0, 100.0, NX)``, so the
    spacing is ``100/(NX-1)``, not ``100/NX``. ``dx`` overrides it for the
    tests that replaced the mesh outright.
    """
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
    x = np.linspace(0.0, 100.0, NX)
    mesh_size = float(x[1] - x[0]) if dx is None else float(dx)
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=strip(NX, mesh_size=mesh_size),
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


class _FixedSolve:
    """A stand-in LU factor that returns a prescribed solve."""

    def __init__(self, result: np.ndarray) -> None:
        self.result = np.asarray(result, dtype=float)

    def solve(self, _rhs: np.ndarray) -> np.ndarray:
        return self.result


def _fixed_op(result: np.ndarray) -> tuple:
    """One unified ``EnergyTransportOp`` whose CN solve is prescribed.

    The retired backend's operator tuple was
    ``(B, LU, idx, rho_p, n_substeps)``; the unified one is
    ``(B, LU, idx, rho_p, n_substeps, forcing, operator)`` -- the per-substep
    boundary forcing and the raw generator were appended so a caller can form
    the endpoint residual. Only the width changed, not the meaning of the
    first five entries.
    """
    return (
        np.eye(2),
        _FixedSolve(result),
        np.array([0, 1]),
        np.ones(2),
        1,
        np.zeros(2),   # no inhomogeneous boundary forcing
        None,          # raw generator, unused by SpatialTransport.apply
    )


class TestStripTransport:
    """Unified-backend transport on a one-cell-wide ``(1, N)`` mask.

    The 1-D reduction of :class:`qpsim.backends.t3_spatial.T3SpatialBackend`.
    Translated from ``TestT3Spatial1DTransport``, which exercised the retired
    :class:`qpsim.backends.t3_spatial_1d.T3Spatial1DBackend`.
    """

    def test_large_finite_diffusivity_remains_finite(self) -> None:
        """The public transport path must not overflow its face mean."""
        state = _build_state(D0=1e200, T_bath=0.0, NE=2, NX=2, dx=1.0)
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2

        out = T3SpatialBackend().apply_transport(state, dt=1e-200)

        assert np.all(np.isfinite(out.f))
        assert out.f[-1, 1] > 0.0
        assert float(np.sum(out.f[-1])) == pytest.approx(0.2)

    def test_reflective_transport_preserves_uniform_field(self) -> None:
        state = _build_state()
        out = T3SpatialBackend().apply_transport(state, dt=2.0)
        np.testing.assert_allclose(out.f, state.f, atol=1e-13)

    def test_reflective_transport_spreads_and_conserves_pulse(self) -> None:
        state = _build_state(T_bath=0.0)
        f = np.zeros_like(state.f)
        energy_idx = -1
        f[energy_idx, 0] = 0.2
        state.f = f

        out = T3SpatialBackend().apply_transport(state, dt=5.0)

        assert out.f[energy_idx, 0] < state.f[energy_idx, 0]
        assert out.f[energy_idx, 1] > 0.0
        np.testing.assert_allclose(
            np.sum(out.f[energy_idx]),
            np.sum(state.f[energy_idx]),
            atol=1e-13,
        )

    def test_transport_conserves_declared_cell_center_measure(self) -> None:
        # The retired state carried an explicit cell-center array plus a
        # derived ``cell_widths``; the geometry carries one uniform
        # ``mesh_size`` instead, and the declared measure is that pitch times
        # the cell count. The original mesh -- (arange(NX)+0.5)*100/NX -- IS
        # the unified cell-center convention, so this is the same strip.
        length_um = 100.0
        NX = 21
        state = _build_state(T_bath=0.0, NX=NX, dx=length_um / NX)
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2
        before = float(np.sum(state.geometry.mesh_size * state.f[-1]))

        backend = T3SpatialBackend()
        out = state
        for _ in range(20):
            out = backend.apply_transport(out, dt=0.5)
        after = float(np.sum(out.geometry.mesh_size * out.f[-1]))

        declared = state.geometry.mesh_size * state.geometry.cell_count
        assert declared == pytest.approx(length_um)
        assert after == pytest.approx(before, abs=1e-12)

    def test_clip_accounting_uses_absolute_mass_not_cancelable_signed_net(
        self,
    ) -> None:
        # Opposite-sign, individually small clips cancel in the signed sum.
        # The absolute diagnostic must still register them.
        #
        # The retired backend did this accounting inside its own
        # ``apply_transport``; on the unified backend it lives one layer down,
        # in ``SpatialTransport.apply``, which is where the loop over substeps
        # now is. This is the reachable half of the original test -- the
        # escalation on top of it is held as an xfail below.
        state = _build_state(T_bath=0.0, NE=2, NX=2, dx=1.0)
        state.f[:] = 0.5
        transport = T3SpatialBackend()._transport_for(state)
        ops = [
            _fixed_op(np.array([1.000001, 0.5])),
            _fixed_op(np.array([-0.000001, 0.5])),
        ]

        f_new, diagnostics = transport.apply(state.f, ops)

        # Two clips of 1e-6 against 2.0 of represented density.
        assert diagnostics["clip_absolute_fraction"] == pytest.approx(1e-6, rel=1e-6)
        assert abs(diagnostics["clip_signed_fraction"]) < 1e-12
        np.testing.assert_array_equal(f_new[0], np.array([1.0, 0.5]))
        np.testing.assert_array_equal(f_new[1], np.array([0.0, 0.5]))

    def test_clip_warning_uses_absolute_mass_not_cancelable_signed_net(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=2, NX=2, dx=1.0)
        state.f[:] = 0.5
        backend = T3SpatialBackend()
        transport = backend._transport_for(state)

        # Opposite-sign, individually small clips cancel in the signed sum.
        # The absolute diagnostic must still fire.
        ops = [
            _fixed_op(np.array([1.000001, 0.5])),
            _fixed_op(np.array([-0.000001, 0.5])),
        ]
        # apply_transport goes through _transport_ops, not
        # _build_transport_operators, so that is the seam to hold fixed.
        monkeypatch.setattr(
            backend, "_transport_ops", lambda _state, _dt: (transport, ops),
        )

        with pytest.warns(UserWarning, match="absolute conserved density"):
            out = backend.apply_transport(state, dt=1.0)
        np.testing.assert_array_equal(out.f[0], np.array([1.0, 0.5]))
        np.testing.assert_array_equal(out.f[1], np.array([0.0, 0.5]))

    def test_large_clip_corruption_fails_instead_of_converging(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=1, NX=2, dx=1.0)
        state.f[:] = 0.5
        backend = T3SpatialBackend()
        transport = backend._transport_for(state)
        ops = [_fixed_op(np.array([1.1, 0.5]))]
        monkeypatch.setattr(
            backend, "_transport_ops", lambda _state, _dt: (transport, ops),
        )

        with pytest.raises(RuntimeError, match="absolute conserved density"):
            backend.apply_transport(state, dt=1.0)

    def test_huge_dt_subcycles_before_cn_can_swap_two_cell_pulse(self) -> None:
        state = _build_state(T_bath=0.0, NE=2, NX=2, dx=1.0)
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2
        backend = T3SpatialBackend()

        ops = backend._build_transport_operators(state, dt=100.0)
        assert ops[-1] is not None
        # The unified operator tuple appends (forcing, generator), so the
        # substep count is entry 4 rather than the last one.
        assert ops[-1][4] > 1

        out = backend.apply_transport(state, dt=100.0)
        pulse = out.f[-1]
        # Diffusion damps the antisymmetric mode without changing its sign:
        # the initially occupied cell cannot become less occupied than its
        # neighbour (the phase-flipped, nearly swapped full-step CN result).
        assert pulse[0] >= pulse[1] - 1.0e-14
        assert abs(pulse[0] - pulse[1]) < 1.0e-12
        np.testing.assert_allclose(np.sum(pulse), 0.2, atol=2.0e-13)

    def test_cn_subcycle_work_guard_fails_loudly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The cap moved with the subcycling itself: it is now a module
        # constant of qpsim.transport.spatial_transport, which is where the
        # unified backend builds its per-energy operators.
        import qpsim.transport.spatial_transport as spatial_module

        state = _build_state(T_bath=0.0, NE=1, NX=2, dx=1.0)
        monkeypatch.setattr(spatial_module, "_MAX_CN_SUBSTEPS", 2)

        with pytest.raises(RuntimeError, match="monotonicity would require"):
            T3SpatialBackend().apply_transport(state, dt=100.0)

    @staticmethod
    def _dynes_state() -> T3SpatialState:
        state = _build_state()
        dynes = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            dynes_gamma=0.05,
            diffusion_coefficient=state.spectral.diffusion_coefficient,
        )
        return replace(state, spectral=dynes)

    def test_dynes_context_rejected_by_collisions(self) -> None:
        # The collision kernels use ideal-BCS coherence functions, so swapping
        # only the DOS is not a self-consistent broadened model and must fail
        # loudly (paper's Dynes footnote).
        with pytest.raises(ValueError, match="pure-BCS"):
            T3SpatialBackend().apply_collisions(self._dynes_state(), dt=1.0)

    def test_dynes_context_rejected_by_transport(self) -> None:
        # The transport dressings are clean-BCS traces (D_L indicator, KL
        # weight from real N_1/N_2, identity N_1^2 - N_2^2 = 1); a
        # Dynes-broadened context invalidates them and must fail loudly.
        with pytest.raises(ValueError, match="pure-BCS"):
            T3SpatialBackend().apply_transport(self._dynes_state(), dt=1.0)
