"""Tests for the self-consistent-gap path in T3DiffusionBackend.steady_state."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import (
    SelfConsistentGapCollapseError,
    T3DiffusionBackend,
    T3DiffusionState,
)
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics import acoustic_escape_tau_l, calibrate_gap
from qpsim.physics.bcs_quadrature import cell_edges_from_widths
from qpsim.physics.gap_equation import solve_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _build_state(
    *,
    T_bath: float = 0.3,
    num_energy: int = 40,
    tau_l_mode: str = "constant",
) -> T3DiffusionState:
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=0.9,
        energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega, _, _, _ = build_phonon_frequency_map(E)
    omega_2d = omega.reshape(1, -1)
    if tau_l_mode == "constant":
        tau_l = np.full((1, omega.size), 0.25)
    elif tau_l_mode == "acoustic":
        material = replace(material, film_thickness=63.0, substrate_transmission_eta=0.2)
        tau_l = acoustic_escape_tau_l(omega_2d, material)
    else:
        raise ValueError(f"Unknown tau_l_mode {tau_l_mode!r}.")

    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega_2d,
        tau_l=tau_l,
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_init,
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


class TestSelfConsistentGapPath:
    def test_gap_controls_are_ignored_for_fixed_gap_solve(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.3, num_energy=20)
        backend = T3DiffusionBackend()
        monkeypatch.setattr(
            backend,
            "_steady_state_fixed_gap",
            lambda state_arg, **_kwargs: state_arg,
        )

        returned = backend.steady_state(
            state,
            use_thermal_phonons=True,
            self_consistent_gap=False,
            gap_tol=float("nan"),
            gap_max_iter=0,
            gap_under_relaxation=float("nan"),
        )

        assert returned is state

    @pytest.mark.parametrize("gap_tol", [float("nan"), float("inf"), 0.0, -1.0])
    def test_rejects_nonfinite_or_nonpositive_gap_tolerance(
        self,
        gap_tol: float,
    ) -> None:
        state = _build_state(T_bath=0.3, num_energy=20)
        with pytest.raises(ValueError, match="gap_tol must be finite and positive"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_tol=gap_tol,
            )

    @pytest.mark.parametrize("gap_max_iter", [True, 0, -1, 1.5])
    def test_rejects_nonpositive_or_noninteger_gap_iteration_cap(
        self,
        gap_max_iter: object,
    ) -> None:
        state = _build_state(T_bath=0.3, num_energy=20)
        with pytest.raises(ValueError, match="gap_max_iter must be a positive integer"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_max_iter=gap_max_iter,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "gap_under_relaxation",
        [float("nan"), float("inf"), 0.0, -1.0, 1.01],
    )
    def test_rejects_invalid_gap_under_relaxation(
        self,
        gap_under_relaxation: float,
    ) -> None:
        state = _build_state(T_bath=0.3, num_energy=20)
        with pytest.raises(ValueError, match="gap_under_relaxation must be finite"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_under_relaxation=gap_under_relaxation,
            )

    def test_thermal_fixed_point_matches_equilibrium_gap(self) -> None:
        state = _build_state(T_bath=0.3, num_energy=80)
        calibration = calibrate_gap(
            T_c=state.material.T_c,
            T_bath=state.T_bath,
            Delta_0=state.material.Delta_0,
        )

        new_state = T3DiffusionBackend().steady_state(
            state,
            use_thermal_phonons=True,
            self_consistent_gap=True,
            gap_tol=1e-6,
        )

        assert new_state.gap == pytest.approx(calibration.delta_eq, rel=3e-3)
        assert new_state.spectral.gap == pytest.approx(new_state.gap, rel=1e-12)
        lower_edge = float(
            cell_edges_from_widths(
                new_state.spectral.E,
                new_state.spectral.dE,
            )[0]
        )
        assert new_state.gap >= lower_edge
        constrained_gap = solve_gap(
            calibration,
            new_state.f,
            new_state.spectral.E,
            dE_bins=new_state.spectral.dE,
            reference_gap=new_state.gap,
            xtol=1e-12,
        )
        assert abs(constrained_gap - new_state.gap) / new_state.gap < 1e-6

    def test_preserves_spectral_configuration(self) -> None:
        state = _build_state(T_bath=0.25, num_energy=60)
        state.spectral = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            diffusion_coefficient=7.0,
            rebuild_tolerance=1e-8,
            active_margin_factor=0.5,
        )

        new_state = T3DiffusionBackend().steady_state(
            state,
            use_thermal_phonons=True,
            self_consistent_gap=True,
            gap_tol=1e-6,
        )

        assert new_state.spectral.diffusion_coefficient == pytest.approx(7.0)
        assert new_state.spectral.rebuild_tolerance == pytest.approx(1e-8)
        assert new_state.spectral.active_margin_factor == pytest.approx(0.5)

    def test_acoustic_escape_picard_path_runs(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=30, tau_l_mode="acoustic")
        dE = float(state.spectral.dE[0])
        photon_params = {
            "omega_0": 2.0 * dE,
            "n_bar": 1.0e6,
            "c_phot": 1.0e-9,
        }

        new_state = T3DiffusionBackend().steady_state(
            state,
            method="picard",
            photon_params=photon_params,
            self_consistent_gap=True,
            gap_tol=1e-6,
            gap_max_iter=20,
            picard_tol=1e-8,
            picard_max_iter=500,
            picard_mixing=0.3,
        )

        assert np.isfinite(new_state.gap)
        assert new_state.gap > 0.0
        assert new_state.spectral.gap == pytest.approx(new_state.gap, rel=1e-12)
        assert new_state.phonon.n_ph.shape[1] == new_state.phonon.omega_bins.shape[1]

    def test_rejects_bath_above_tc(self) -> None:
        state = _build_state(T_bath=1.3, num_energy=30)
        with pytest.raises(ValueError, match="T_bath < T_c"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
            )

    def test_gap_collapse_raises_instead_of_drifting(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force solve_gap to report collapse. Without the explicit
        # delta_raw <= 0 guard, under-relaxation would drift toward a
        # spurious tiny Δ > 0 instead of raising.
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=40)
        monkeypatch.setattr(t3_mod, "solve_gap", lambda *args, **kwargs: 0.0)
        with pytest.raises(
            SelfConsistentGapCollapseError,
            match="gap collapsed",
        ) as caught:
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_tol=1e-6,
                gap_max_iter=4,
            )
        assert caught.value.iteration == 1
        assert caught.value.max_occupation >= 0.0

    def test_gap_under_relaxation_does_not_scale_convergence_residual(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=12)
        backend = T3DiffusionBackend()
        solved_gaps: list[float] = []
        reference_gaps: list[float] = []

        def fixed_gap_identity(
            state_arg: T3DiffusionState,
            **kwargs: object,
        ) -> T3DiffusionState:
            solved_gaps.append(state_arg.gap)
            return state_arg

        monkeypatch.setattr(backend, "_steady_state_fixed_gap", fixed_gap_identity)

        def doubled_gap(*args: object, **kwargs: object) -> float:
            reference_gaps.append(float(kwargs["reference_gap"]))
            return 2.0 * solved_gaps[-1]

        monkeypatch.setattr(
            t3_mod,
            "solve_gap",
            doubled_gap,
        )

        # The raw gap-map residual is 1.0. A 1e-6 relaxed update is smaller
        # than gap_tol, but must not be reported as a converged gap equation.
        with pytest.raises(RuntimeError, match=r"Final \|Δ_raw - Δ\| / Δ = 1.00e\+00"):
            backend.steady_state(
                state,
                self_consistent_gap=True,
                use_thermal_phonons=True,
                gap_under_relaxation=1e-6,
                gap_tol=1e-4,
                gap_max_iter=1,
            )
        assert reference_gaps == [state.gap]

    def test_returns_only_the_kinetic_state_whose_gap_map_was_checked(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=12)
        backend = T3DiffusionBackend()
        initial_gap = state.gap
        gap_tol = 1e-4
        solved_gaps: list[float] = []

        def kinetic_state_at_gap(
            state_arg: T3DiffusionState,
            **_kwargs: object,
        ) -> T3DiffusionState:
            solved_gaps.append(state_arg.gap)
            f = np.zeros_like(state_arg.f)
            # A post-update kinetic solve changes the occupation branch.  The
            # old implementation returned that second state without checking
            # its gap map.
            f[0] = 0.0 if state_arg.gap == initial_gap else 0.25
            return replace(state_arg, f=f)

        def occupation_dependent_gap(
            _calibration: object,
            f: np.ndarray,
            *_args: object,
            **_kwargs: object,
        ) -> float:
            if f[0] == 0.0:
                return initial_gap * (1.0 + 0.5 * gap_tol)
            return 0.95 * initial_gap

        monkeypatch.setattr(backend, "_steady_state_fixed_gap", kinetic_state_at_gap)
        monkeypatch.setattr(t3_mod, "solve_gap", occupation_dependent_gap)

        solved = backend.steady_state(
            state,
            self_consistent_gap=True,
            use_thermal_phonons=True,
            gap_tol=gap_tol,
            gap_under_relaxation=0.5,
            gap_max_iter=2,
        )
        checked_gap = occupation_dependent_gap(None, solved.f)

        assert solved_gaps == [initial_gap]
        assert solved.gap == initial_gap
        assert abs(checked_gap - solved.gap) / solved.gap < gap_tol

    def test_rejects_candidate_gap_below_represented_energy_face(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=20)
        backend = T3DiffusionBackend()
        lower_edge = float(
            cell_edges_from_widths(
                state.spectral.E,
                state.spectral.dE,
            )[0]
        )
        monkeypatch.setattr(
            backend,
            "_steady_state_fixed_gap",
            lambda state_arg, **_kwargs: state_arg,
        )
        monkeypatch.setattr(
            t3_mod,
            "solve_gap",
            lambda *_args, **_kwargs: lower_edge - 1.0,
        )

        with pytest.raises(
            ValueError,
            match="below the first represented energy-cell face",
        ):
            backend.steady_state(
                state,
                self_consistent_gap=True,
                use_thermal_phonons=True,
            )


class TestBelowSupportCollapseClassification:
    """2026-07-19 audit: on every shipped BCS grid (first face > 0) solve_gap
    signals collapse via GapBelowGridSupportError, never a <= 0 return — the
    classified chain must catch it."""

    def test_below_support_solve_gap_raises_classified_collapse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.backends import t3_diffusion as t3_mod
        from qpsim.physics.gap_equation import GapBelowGridSupportError

        state = _build_state(T_bath=0.3, num_energy=40)

        def raise_below_support(*args: object, **kwargs: object) -> float:
            raise GapBelowGridSupportError(
                "solve_gap: candidate gap Δ=0 μeV lies below the "
                "reconstructed energy-grid support edge (test double).",
                candidate_gap=0.0,
            )

        monkeypatch.setattr(t3_mod, "solve_gap", raise_below_support)
        with pytest.raises(
            SelfConsistentGapCollapseError, match="gap collapsed"
        ) as caught:
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_tol=1e-6,
                gap_max_iter=4,
            )
        assert isinstance(caught.value.__cause__, GapBelowGridSupportError)

    def test_positive_root_below_support_is_not_classified_as_collapse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """2026-07-20 review: a POSITIVE superconducting root the grid
        cannot represent (e.g. root 155.85 μeV under a 162.06 μeV first
        face) is grid under-resolution, not collapse — it must propagate
        as the domain error, never fold to the collapse/NaN path."""
        from qpsim.backends import t3_diffusion as t3_mod
        from qpsim.physics.gap_equation import GapBelowGridSupportError

        state = _build_state(T_bath=0.3, num_energy=40)

        def raise_positive_below_support(*args: object, **kwargs: object) -> float:
            raise GapBelowGridSupportError(
                "solve_gap: candidate gap Δ=155.85 μeV lies below the "
                "reconstructed energy-grid support edge 162.06 μeV "
                "(test double).",
                candidate_gap=155.85,
            )

        monkeypatch.setattr(t3_mod, "solve_gap", raise_positive_below_support)
        with pytest.raises(GapBelowGridSupportError) as caught:
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_tol=1e-6,
                gap_max_iter=4,
            )
        assert not isinstance(caught.value, SelfConsistentGapCollapseError)
        assert caught.value.candidate_gap == 155.85

    def test_solve_gap_normal_state_on_truncated_grid_is_classified(self) -> None:
        """End-to-end at the gap_equation level: a saturating hot occupation
        on a grid whose first face sits above zero must raise the classified
        GapBelowGridSupportError (which is still a ValueError), not return
        0.0 and not raise a bare unclassified error."""
        import numpy as np
        from qpsim.physics.gap_equation import (
            GapBelowGridSupportError,
            calibrate_gap,
            solve_gap,
        )

        calibration = calibrate_gap(T_c=1.2, T_bath=0.2)
        delta_eq = calibration.delta_eq
        # Grid starting well above zero (first face ~0.89*delta_eq), like the
        # shipped fig6 grid; f = 0.5 saturates pair-breaking.
        E = np.linspace(
            0.9 * delta_eq,
            1.01 * calibration._omega_D,
            2000,
        )
        f_sat = np.full_like(E, 0.5)
        with pytest.raises(GapBelowGridSupportError) as caught:
            solve_gap(calibration, f_sat, E, reference_gap=delta_eq)
        assert isinstance(caught.value, ValueError)  # subclass contract
        # Saturated f admits no superconducting solution: this is the
        # genuine normal-state decision, the ONLY case sweeps may
        # classify as collapse (2026-07-20 review).
        assert caught.value.candidate_gap == 0.0
