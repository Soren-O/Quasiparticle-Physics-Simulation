"""Varying-gap transport on a strip, i.e. the 1-D reduction of the unified
backend.

Translated from ``TestT3Spatial1DVaryingGap`` in
``tests/backends/test_t3_spatial_1d.py``, which exercised the retired
``T3Spatial1DBackend``. The retired backend solved a strip directly; the
unified :class:`qpsim.backends.t3_spatial.T3SpatialBackend` solves the same
problem as the degenerate case of a ``(1, N)`` mask, so every physical property
here -- current continuity across a gap ramp, the Kupriyanov-Lukichev
interface, the A1/A2/C closure separations -- must still hold cell for cell.

Mesh note: the retired tests meshed with ``x = np.linspace(0.0, 100.0, NX)``,
so the pitch is ``100/(NX - 1)``, NOT ``100/NX``. The geometry carries that
pitch and places cells at ``(arange(NX) + 0.5) * mesh_size``.
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
from qpsim.transport.diffusion.base import DiffusionModel


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _uniform_state(
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
    NX: int = 11,
    mesh_size: float | None = None,
) -> T3SpatialState:
    """The retired ``_build_state``: a thermal strip at a single scalar gap."""
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
    pitch = float(x[1] - x[0]) if mesh_size is None else mesh_size
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=strip(NX, mesh_size=pitch),
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


def _varying_gap_setup(
    *, NE: int = 16, NX: int = 21, interface: bool = False,
) -> tuple[object, SpectralContext, Geometry, float, np.ndarray]:
    material = load_material("Al")
    base_gap = material.Delta_0
    gap_max = 1.6 * base_gap
    E, _ = build_energy_grid(
        gap=gap_max, energy_min_factor=1.05, energy_max_factor=4.0, num_energy_bins=NE
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap_max,
        diffusion_coefficient=6.0,
    )
    x = np.linspace(0.0, 100.0, NX)
    geometry = strip(NX, mesh_size=float(x[1] - x[0]))
    if interface:
        profile = np.where(np.arange(NX) < NX // 2, gap_max, base_gap).astype(float)
    else:
        profile = np.linspace(base_gap, gap_max, NX)
    return material, spectral, geometry, gap_max, profile


def _centers(geometry: Geometry) -> np.ndarray:
    """Cell-centre coordinates of a strip, the unified stand-in for ``state.x``."""
    return (np.arange(geometry.cell_count) + 0.5) * geometry.mesh_size


class TestStripVaryingGap:
    """1-D reduction of the unified backend under a spatially varying gap.

    From ``TestT3Spatial1DVaryingGap`` (retired ``T3Spatial1DBackend``).
    """

    def test_gap_profile_shape_validation(self) -> None:
        # A gap profile that does not have one entry per cell must not be
        # silently broadcast or truncated.  The unified backend has no explicit
        # `gap_per_cell` shape check the way the retired one had: the mismatch
        # is caught incidentally, by the assembler's boolean cell index, so the
        # exception is an IndexError naming the two sizes rather than a
        # ValueError naming the field.  The *rejection* is what this test is
        # for; the message quality is recorded as a finding.
        material, spectral, geometry, _gap_max, _ = _varying_gap_setup()
        NE, NX = spectral.E.size, geometry.cell_count
        bad = T3SpatialState(
            f=np.zeros((NE, NX)),
            geometry=geometry,
            spectral=spectral,
            material=material,
            T_bath=0.1,
            gap_per_cell=np.ones(NX + 1),
        )
        with pytest.raises((ValueError, IndexError), match=str(NX + 1)):
            T3SpatialBackend().apply_transport(bad, 1.0)

    def test_degenerate_and_descending_mesh_rejected(self) -> None:
        # A constant mesh (dx=0) and a descending mesh (dx<0) both pass an
        # equal-diffs test yet corrupt every dx-divided flux downstream.  On a
        # uniform geometry the whole mesh is the single pitch `mesh_size`, so
        # the two cases are mesh_size == 0 and mesh_size < 0.
        pitch = _uniform_state().geometry.mesh_size
        for bad_pitch in (0.0, -pitch):
            state = _uniform_state(mesh_size=bad_pitch)
            with pytest.raises(ValueError, match="positive and finite"):
                T3SpatialBackend().apply_transport(state, 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_mesh_is_rejected(self, bad: float) -> None:
        state = _uniform_state(mesh_size=bad)
        with pytest.raises(ValueError, match="positive and finite"):
            T3SpatialBackend().apply_transport(state, 1.0)

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(
                float("nan"),
                marks=pytest.mark.xfail(
                    reason=(
                        "FINDING: a NaN bath temperature is NOT rejected by the "
                        "unified collision path.  qpsim/collisions/phonon.py:208 "
                        "and :235 branch on `if T_bath > 0`, which is False for "
                        "NaN, so the T = 0 branch runs and the run silently "
                        "reports a zero-temperature answer (measured 2.3e-07 "
                        "from the T = 0.1 K result on this setup, i.e. the 0 K "
                        "answer to within the 0.1 K correction).  The retired "
                        "backend rejected it with 'state.T_bath must be finite "
                        "and non-negative.'"
                    ),
                ),
            ),
            float("inf"),
        ],
    )
    def test_nonfinite_bath_is_rejected(self, bad: float) -> None:
        state = _uniform_state()
        state.T_bath = bad
        # The unified path raises from the shared temperature helper, so the
        # message names 'temperature' rather than the state field 'T_bath'.
        with pytest.raises(ValueError, match=r"(?i)T_bath|temperature"):
            T3SpatialBackend().apply_collisions(state, 1.0)

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(
                -1.0,
                marks=pytest.mark.xfail(
                    reason=(
                        "FINDING: a finite NEGATIVE interface conductance is "
                        "accepted.  The unified backend has no "
                        "interface_conductance validation at all "
                        "(t3_spatial.py:338-345 only asks whether it is None), "
                        "so g_N = -1 builds a negative face conductance and the "
                        "step runs uphill: on the retired class's own step-gap "
                        "fixture one dt=0.5 transport step takes the N1-weighted "
                        "density from 77.0783 to 77.8124 (+0.95%) and lifts the "
                        "source cell from f=0.4 to f=0.4406.  Transport is "
                        "manufacturing quasiparticles."
                    ),
                ),
            ),
            -np.inf,
            np.inf,
            np.nan,
        ],
    )
    def test_negative_interface_conductance_rejected(self, bad: float) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup(
            interface=True,
        )
        NE, NX = spectral.E.size, geometry.cell_count
        state = T3SpatialState(
            f=np.zeros((NE, NX)), geometry=geometry, spectral=spectral,
            material=material, T_bath=0.1, gap_per_cell=profile,
            interface_conductance=bad,
        )
        # The non-finite cases are still rejected, but downstream of the field:
        # -inf trips the Laplacian finiteness check (ValueError) and +-inf/NaN
        # trip the stiffness estimate (RuntimeError).  The retired backend
        # rejected all four up front with a message naming
        # 'interface_conductance'.
        with pytest.raises((ValueError, RuntimeError)):
            T3SpatialBackend().apply_transport(state, 0.5)

    @pytest.mark.xfail(
        reason=(
            "FINDING: interface_conductance without a gap profile is SILENTLY "
            "IGNORED.  t3_spatial.py:343 reads `if conductance is not None and "
            "state.gap_per_cell is not None`, so with gap_per_cell=None the "
            "barrier is dropped and the step is bit-for-bit identical to "
            "interface_conductance=None (verified with np.array_equal).  This "
            "is the exact failure the retired backend's own error text names: "
            "'without it the conductance would be silently unused.'"
        ),
    )
    def test_interface_conductance_without_profile_is_rejected(self) -> None:
        state = _uniform_state()
        state.interface_conductance = 2.0

        with pytest.raises(ValueError, match="requires a gap_p"):
            T3SpatialBackend().apply_transport(state, 0.5)

    def test_zero_interface_conductance_is_an_opaque_interface(self) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup(
            interface=True,
        )
        NE, NX = spectral.E.size, geometry.cell_count
        f0 = np.zeros((NE, NX))
        face = NX // 2 - 1
        f0[:, : face + 1] = 0.4
        state = T3SpatialState(
            f=f0.copy(), geometry=geometry, spectral=spectral,
            material=material, T_bath=0.1, diffusion_model=DiffusionModel.A1,
            gap_per_cell=profile, interface_conductance=0.0,
        )

        out = T3SpatialBackend().apply_transport(state, 0.5)

        # G_N = 0 is the physical opaque-interface limit.  It is distinct
        # from None, which leaves the step face on the bulk-diffusion path.
        np.testing.assert_array_equal(out.f[:, face + 1 :], 0.0)
        np.testing.assert_allclose(out.f[:, : face + 1], f0[:, : face + 1])

    def test_varying_gap_conserves_weighted_density(self) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup()
        NE = spectral.E.size
        x = _centers(geometry)
        f0 = np.tile(0.2 * np.exp(-((x - 30.0) / 18.0) ** 2), (NE, 1))
        state = T3SpatialState(
            f=f0.copy(), geometry=geometry, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_per_cell=profile,
        )
        N1 = T3SpatialBackend()._n1_per_cell(state)
        before = float(np.sum(N1 * state.f))  # p = 1 for A1
        backend = T3SpatialBackend()
        evolving = state
        for _ in range(30):
            evolving = backend.apply_transport(evolving, 1.0)
        after = float(np.sum(N1 * evolving.f))
        assert abs(after - before) / abs(before) < 1e-12

    def test_a1_and_c_differ_in_gap_ramp(self) -> None:
        # A1 and C coincide at a uniform gap but separate once the gap
        # varies: A1 has no DOS-gradient drift (the spectral factor sits
        # outside the divergence), C dresses the flux inside it.
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup()
        NE = spectral.E.size
        x = _centers(geometry)
        f0 = np.tile(0.2 * np.exp(-((x - 30.0) / 18.0) ** 2), (NE, 1))

        def step(model: DiffusionModel) -> np.ndarray:
            state = T3SpatialState(
                f=f0.copy(), geometry=geometry, spectral=spectral,
                material=material, T_bath=0.1, diffusion_model=model,
                gap_per_cell=profile,
            )
            out = state
            backend = T3SpatialBackend()
            for _ in range(10):
                out = backend.apply_transport(out, 1.0)
            return out.f

        assert np.max(np.abs(step(DiffusionModel.A1) - step(DiffusionModel.C))) > 1e-4

    def test_interface_conserves_and_jumps(self) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup(
            interface=True,
        )
        NE, NX = spectral.E.size, geometry.cell_count
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4
        state = T3SpatialState(
            f=f0.copy(), geometry=geometry, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_per_cell=profile,
            interface_conductance=2.0,
        )
        N1 = T3SpatialBackend()._n1_per_cell(state)
        before = float(np.sum(N1 * state.f))
        out = T3SpatialBackend().apply_transport(state, 0.5)
        after = float(np.sum(N1 * out.f))
        assert abs(after - before) / abs(before) < 1e-12  # current continuity
        k, e = NX // 2 - 1, NE - 1
        assert out.f[e, k] > out.f[e, k + 1] + 1e-6  # f-discontinuity at the interface

    def test_interface_differs_from_bulk(self) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup(
            interface=True,
        )
        NE, NX = spectral.E.size, geometry.cell_count
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4

        def step(conductance: float | None) -> np.ndarray:
            state = T3SpatialState(
                f=f0.copy(), geometry=geometry, spectral=spectral,
                material=material, T_bath=0.1,
                diffusion_model=DiffusionModel.A1, gap_per_cell=profile,
                interface_conductance=conductance,
            )
            return T3SpatialBackend().apply_transport(state, 0.5).f

        assert np.max(np.abs(step(0.1) - step(None))) > 1e-3

    def test_a1_a2_distinct_under_interface_relaxation(self) -> None:
        material, spectral, geometry, _gap_max, profile = _varying_gap_setup(
            interface=True,
        )
        NE, NX = spectral.E.size, geometry.cell_count
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4
        backend = T3SpatialBackend()

        def relax(model: DiffusionModel) -> np.ndarray:
            state = T3SpatialState(
                f=f0.copy(), geometry=geometry, spectral=spectral,
                material=material, T_bath=0.1, diffusion_model=model,
                gap_per_cell=profile, interface_conductance=2.0,
            )
            for _ in range(1500):
                state = backend.apply_transport(state, 2.0)
            return state.f

        diff = np.max(np.abs(relax(DiffusionModel.A1) - relax(DiffusionModel.A2)))
        assert diff > 1e-3
