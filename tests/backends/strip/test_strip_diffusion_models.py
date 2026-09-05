"""Diffusion-closure family on the 1-D reduction of the unified backend.

Every property asserted here is a statement about the ``(p, q)`` operator
family itself -- which conserved density each member carries, which members
coincide and which separate -- checked on ``SpatialBackend`` with a
one-cell-wide ``strip`` geometry.
"""

from __future__ import annotations

import numpy as np
from qpsim.backends.spatial import SpatialBackend, SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel


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
    # The state meshes with x = linspace(0, 100, NX), i.e. a spacing of
    # 100/(NX-1) -- NOT 100/NX. The geometry's mesh_size is that spacing.
    x = np.linspace(0.0, 100.0, NX)
    dx = float(x[1] - x[0])
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return SpatialState(
        f=f0,
        geometry=strip(NX, mesh_size=dx),
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


def _centres(state: SpatialState) -> np.ndarray:
    """Cell-centre coordinates of the strip, the unified mesh's own x."""
    mesh = float(state.geometry.mesh_size)
    return (np.arange(state.geometry.cell_count, dtype=float) + 0.5) * mesh


def _model_state(
    base: SpatialState, f: np.ndarray, model: DiffusionModel,
) -> SpatialState:
    return SpatialState(
        f=f,
        geometry=base.geometry,
        spectral=base.spectral,
        material=base.material,
        T_bath=0.0,
        diffusion_model=model,
    )


class TestStripDiffusionModels:
    """1-D reduction of ``SpatialBackend``: the ``(p, q)`` closure family.

    The geometry is ``strip(NX, mesh_size=dx)``.
    """

    def test_default_model_is_a1(self) -> None:
        assert _build_state().diffusion_model is DiffusionModel.A1

    def test_each_closure_conserves_weighted_density(self) -> None:
        backend = SpatialBackend()
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        x = _centres(base)
        packet = np.tile(0.3 * np.exp(-((x - 50.0) / 15.0) ** 2), (NE, 1))
        for model in DiffusionModel:
            state = _model_state(base, packet.copy(), model)
            weight = backend._n1_per_cell(state) ** model.p
            before = float(np.sum(weight * state.f))
            evolving = state
            for _ in range(30):
                evolving = backend.apply_transport(evolving, 1.0)
            after = float(np.sum(weight * evolving.f))
            assert abs(after - before) / abs(before) < 1e-12, model

    def test_c_path_matches_finite_volume_modal_step(self) -> None:
        base = _build_state(T_bath=0.0)
        NE, NX = base.f.shape
        x = _centres(base)
        f0 = np.tile(0.3 * np.exp(-((x - 50.0) / 15.0) ** 2), (NE, 1))
        state = _model_state(base, f0.copy(), DiffusionModel.C)
        new = SpatialBackend().apply_transport(state, dt=2.0).f

        # Modal Crank-Nicolson step with the mass-lumped finite-volume
        # D_E = D0/N1_bar closure.
        dx = float(state.geometry.mesh_size)
        main = -2.0 * np.ones(NX)
        main[0] = -1.0
        main[-1] = -1.0
        lap = (
            np.diag(main)
            + np.diag(np.ones(NX - 1), 1)
            + np.diag(np.ones(NX - 1), -1)
        ) / dx**2
        w, V = np.linalg.eigh(lap)
        D_E_fv = (
            state.spectral.diffusion_coefficient / state.spectral.cell_density
        )
        alpha = 0.5 * 2.0 * D_E_fv[:, None] * w[None, :]
        old = np.clip(((f0 @ V) * (1.0 + alpha) / (1.0 - alpha)) @ V.T, 0.0, 1.0)
        np.testing.assert_allclose(new, old, atol=1e-12)

    def test_a1_and_c_coincide_at_uniform_gap(self) -> None:
        # A1 (conserves N1 f, undressed flux) and C (bare f, 1/N1-dressed
        # flux) share the uniform-gap rate D0/N1, so with N1 x-independent
        # their f-dynamics are identical -- the dirty-Usadel and clean/BRT
        # reductions agree at a uniform gap.
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        x = _centres(base)
        f0 = np.tile(0.3 * np.exp(-((x - 50.0) / 15.0) ** 2), (NE, 1))
        out = {
            model: SpatialBackend().apply_transport(
                _model_state(base, f0.copy(), model), 2.0
            ).f
            for model in (DiffusionModel.A1, DiffusionModel.C)
        }
        np.testing.assert_allclose(
            out[DiffusionModel.A1], out[DiffusionModel.C], atol=1e-12
        )

    def test_a1_and_a1p_dynamics_differ(self) -> None:
        # The diagnostic A1P carries the transverse N1^2 dressing and
        # separates from A1 already at a uniform gap.
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        x = _centres(base)
        f0 = np.tile(0.3 * np.exp(-((x - 50.0) / 15.0) ** 2), (NE, 1))
        out = {
            model: SpatialBackend().apply_transport(
                _model_state(base, f0.copy(), model), 2.0
            ).f
            for model in (DiffusionModel.A1, DiffusionModel.A1P)
        }
        assert np.max(np.abs(out[DiffusionModel.A1] - out[DiffusionModel.A1P])) > 1e-3
