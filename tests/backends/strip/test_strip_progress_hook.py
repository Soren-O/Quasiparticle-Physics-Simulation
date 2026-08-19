"""The 1-D reduction of the unified spatial backend: the driver's progress hook.

Translated from ``TestRunUntilSteadyStateProgressHook`` in
``tests/backends/test_t3_spatial_1d.py``, which covered the retired
``T3Spatial1DBackend`` / ``T3Spatial1DState``. The unified
:class:`~qpsim.backends.t3_spatial.T3SpatialBackend` solves the same strip as a
``(1, N)`` mask and exposes the same driver as
:meth:`~qpsim.backends.t3_spatial.T3SpatialBackend.run`, so the hook contract
carries over verbatim: reporting only, plus a clean cooperative early stop when
it returns ``False``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
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
    """The retired tests' grid: ``x = linspace(0, 100, NX)``, so ``dx`` is
    ``100/(NX-1)`` -- taken off the array rather than assumed."""
    x = np.linspace(0.0, 100.0, NX)
    return strip(NX, mesh_size=float(x[1] - x[0]))


def _build_state(
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
    NX: int = 11,
) -> T3SpatialState:
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
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], geometry.cell_count, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=geometry,
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


class TestStripRunProgressHook:
    """The driver hook is physics-neutral: reporting only, plus a clean
    cooperative early stop when it returns False."""

    @staticmethod
    def _perturbed_state() -> T3SpatialState:
        state = _build_state(NE=12, NX=7)
        f = state.f.copy()
        f[:, 0] = np.clip(f[:, 0] + 0.05, 0.0, 1.0)
        state.f = f
        return state

    def test_none_hook_bit_for_bit_unchanged(self) -> None:
        backend = T3SpatialBackend()
        baseline = backend.run(
            self._perturbed_state(), dt=1.0, max_time=6.0, stop_tol=0.0,
        )
        hooked = backend.run(
            self._perturbed_state(), dt=1.0, max_time=6.0, stop_tol=0.0,
            progress_hook=lambda _t, _total: True,
        )
        np.testing.assert_array_equal(hooked.state.f, baseline.state.f)
        assert hooked.n_steps == baseline.n_steps
        # ``total_time`` on the retired result object is ``elapsed`` here.
        assert hooked.elapsed == baseline.elapsed

    def test_hook_called_once_per_step_with_times(self) -> None:
        calls: list[tuple[float, float]] = []

        def hook(t: float, total: float) -> bool:
            calls.append((t, total))
            return True

        result = T3SpatialBackend().run(
            self._perturbed_state(), dt=1.0, max_time=5.0, stop_tol=0.0,
            progress_hook=hook,
        )
        assert len(calls) == result.n_steps == 5
        assert all(total == pytest.approx(5.0) for _, total in calls)

    def test_false_return_stops_early(self) -> None:
        result = T3SpatialBackend().run(
            self._perturbed_state(), dt=1.0, max_time=100.0, stop_tol=0.0,
            snapshot_interval=50.0,
            progress_hook=lambda t, _total: t < 3.0,
        )
        assert result.n_steps == 3
        assert result.elapsed == pytest.approx(3.0)
        assert not result.converged
        assert result.snapshots[-1].t == pytest.approx(3.0)

    def test_dt_larger_than_interval_emits_every_snapshot(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A coarse dt against a fine cadence must not drop or duplicate frames.

        The retired driver answered this with dense output: it interpolated
        between the split-step endpoints and emitted a frame at every crossed
        cadence boundary, so ``dt=5`` against ``interval=2`` gave
        ``[0, 2, 4, 6, 8, 10]``. The unified driver deliberately does not --
        see :meth:`T3SpatialBackend.run`: a frame now carries the whole
        ``(NE, Ncells)`` field plus the phonon map, so interpolating one would
        be both expensive and a claim about a state that was never computed.
        Frames are taken at step boundaries at or after each requested time and
        every boundary already overtaken is advanced past, so the same setup
        emits one frame per step -- and, still, no dropped run segment and no
        duplicate times.
        """
        backend = T3SpatialBackend()
        monkeypatch.setattr(
            backend,
            "step",
            lambda state, _dt, **_kwargs: state,
        )

        result = backend.run(
            self._perturbed_state(),
            dt=5.0,
            max_time=10.0,
            stop_tol=0.0,
            snapshot_interval=2.0,
        )

        np.testing.assert_allclose(
            [snapshot.t for snapshot in result.snapshots],
            [0.0, 5.0, 10.0],
        )
        # The intent the retired assertion encoded, restated on this contract:
        # both ends of the run are present and no time is emitted twice.
        times = [snapshot.t for snapshot in result.snapshots]
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(result.elapsed)
        assert times == sorted(times)
        assert len(times) == len(set(times))

    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            ("dt", {"dt": float("nan")}),
            ("max_time", {"max_time": float("inf")}),
            ("stop_tol", {"stop_tol": float("nan")}),
            ("snapshot_interval", {"snapshot_interval": float("inf")}),
        ],
    )
    def test_nonfinite_driver_controls_are_rejected(
        self, field: str, kwargs: dict[str, float],
    ) -> None:
        values = {"dt": 1.0, "max_time": 2.0, "stop_tol": 0.0}
        values.update(kwargs)
        with pytest.raises(ValueError, match=field):
            T3SpatialBackend().run(
                self._perturbed_state(),
                **values,
            )
