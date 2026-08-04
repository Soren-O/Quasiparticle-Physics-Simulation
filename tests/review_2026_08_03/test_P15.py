# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review packet P15 regressions: transient snapshot cadence + stop_tol certificate.

Two defects are pinned here:

* ``run_time_dependent`` emits interior snapshots by linear dense output
  between ETD step endpoints. That is only first-order consistent with the
  exponential relaxation the ETD steps solve, so a cadence finer than ``dt``
  reports O(1)-wrong ``f(E, t)`` while the endpoints stay exact — the driver
  now warns instead of advertising the regime as free.
* The finite-difference convergence fallback (used when a backend does not
  own ``apply_collisions_with_diagnostics``) disqualified itself on any bin
  below ``32·eps``. A cold thermal tail is legitimately below that in most
  bins, so early stopping was dead for every mK state; only an *active* clip
  onto a bound may veto the certificate.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.transient import run_time_dependent


def _build_state(T_bath: float = 0.1, num_energy: int = 12) -> T3DiffusionState:
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    return T3DiffusionState(
        f=1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0),
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


class _DecayBackend:
    """``f → f·exp(−μ·dt)`` — the exact map an ETD step solves at frozen rates.

    A ``NoOpBackend`` cannot detect a dense-output defect: with
    ``prev_f == current.f`` the interpolation is the identity.
    """

    def __init__(self, mu: float) -> None:
        self.mu = mu

    def apply_collisions(
        self, state_arg: T3DiffusionState, dt: float, **kwargs: object,
    ) -> T3DiffusionState:
        return replace(state_arg, f=state_arg.f * np.exp(-self.mu * dt))


class _FrozenBackend(T3DiffusionBackend):
    """Subclass overriding only ``apply_collisions``: an exact fixed point.

    Overriding ``apply_collisions`` alone routes the driver onto the
    finite-difference certificate, because the inherited
    ``apply_collisions_with_diagnostics`` would call this override and find no
    populated diagnostics.
    """

    def apply_collisions(
        self, state_arg: T3DiffusionState, dt: float, **kwargs: object,
    ) -> T3DiffusionState:
        return state_arg


class TestSnapshotDenseOutput:
    def test_cadence_finer_than_dt_warns(self) -> None:
        state = _build_state()
        with pytest.warns(UserWarning, match="finer than"):
            run_time_dependent(
                state,
                dt=40.0,
                total_time=80.0,
                snapshot_interval=2.0,
                backend=_DecayBackend(0.1),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("snapshot_interval", [1.0, 2.0])
    def test_cadence_at_or_above_dt_is_silent(self, snapshot_interval: float) -> None:
        state = _build_state()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_time_dependent(
                state,
                dt=1.0,
                total_time=4.0,
                snapshot_interval=snapshot_interval,
                backend=_DecayBackend(0.1),  # type: ignore[arg-type]
            )

    def test_step_endpoints_stay_exact_under_a_coarse_driver_step(self) -> None:
        # Endpoints carry the integrator's own accuracy whatever the cadence
        # is; that is exactly why a coarse dt looks free until an interior
        # snapshot is read.
        state = _build_state()
        mu = 0.1
        with pytest.warns(UserWarning, match="finer than"):
            result = run_time_dependent(
                state,
                dt=40.0,
                total_time=80.0,
                snapshot_interval=2.0,
                backend=_DecayBackend(mu),  # type: ignore[arg-type]
            )

        by_time = {round(snap.t, 9): snap.f for snap in result.snapshots}
        for endpoint in (40.0, 80.0):
            np.testing.assert_allclose(
                by_time[endpoint],
                state.f * np.exp(-mu * endpoint),
                rtol=1e-14,
                atol=0.0,
            )

    def test_interpolated_snapshots_are_faithful_in_the_recommended_regime(
        self,
    ) -> None:
        # With dt inside the relaxation time the dense-output error is bounded
        # by |f_n − f_inf|·(μ·dt)²/8 = 1.25e-3 relative here, so an interior
        # cadence point is still usable. The snapshot times 1.5, 3.0, 4.5 …
        # never coincide with a step endpoint, so every one is interpolated.
        state = _build_state()
        mu = 0.1
        dt = 1.0
        result = run_time_dependent(
            state,
            dt=dt,
            total_time=6.0,
            snapshot_interval=1.5,
            backend=_DecayBackend(mu),  # type: ignore[arg-type]
        )

        worst = 0.0
        for snap in result.snapshots[1:]:
            exact = state.f * np.exp(-mu * snap.t)
            worst = max(worst, float(np.max(np.abs(snap.f - exact) / exact)))
        assert worst < 2.0e-3
        assert worst > 1.0e-4  # genuinely interpolated, not endpoint-only


class TestFiniteDifferenceCertificate:
    def test_cold_thermal_fixed_point_is_certified(self) -> None:
        # 100 mK on the standard grid: most bins sit far below 32·eps, which
        # used to force the fallback rate to inf and run the whole total_time
        # on an exact fixed point.
        state = _build_state(T_bath=0.1, num_energy=40)
        assert np.count_nonzero(state.f <= 32.0 * np.finfo(float).eps) > 0
        assert float(np.min(state.f)) > 0.0

        result = run_time_dependent(
            state,
            dt=1.0,
            total_time=1000.0,
            snapshot_interval=1000.0,
            stop_tol=1e-6,
            backend=_FrozenBackend(),
        )

        assert result.converged
        assert result.n_steps == 1

    def test_saturated_upper_bound_still_vetoes_the_certificate(self) -> None:
        state = replace(_build_state(num_energy=8), f=np.ones(8))
        result = run_time_dependent(
            state,
            dt=0.1,
            total_time=0.2,
            snapshot_interval=0.2,
            stop_tol=1.0,
            backend=_FrozenBackend(),
        )

        assert not result.converged
        assert result.n_steps == 2

    def test_active_low_clip_still_vetoes_the_certificate(self) -> None:
        # A bin driven onto zero moves f by 1e-30, far below stop_tol, so only
        # the saturation test can catch it. Only the step that saturates is
        # vetoed: a bin that was already zero and stays zero is not a clip,
        # because a non-negative gain can never be clipped downward at f = 0,
        # so its finite difference *is* the residual.
        state = _build_state(num_energy=8)
        state.f = np.full(8, 1.0e-30)

        class _ClipOneBin(T3DiffusionBackend):
            def apply_collisions(
                self, state_arg: T3DiffusionState, dt: float, **kwargs: object,
            ) -> T3DiffusionState:
                f = state_arg.f.copy()
                f[3] = 0.0
                return replace(state_arg, f=f)

        result = run_time_dependent(
            state,
            dt=0.1,
            total_time=0.1,
            snapshot_interval=0.1,
            stop_tol=1e-6,
            backend=_ClipOneBin(),
        )

        assert not result.converged
        assert result.n_steps == 1
