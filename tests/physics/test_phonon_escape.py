"""Tests for qpsim.physics.phonon_escape."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.materials.database import Material
from qpsim.physics.phonon_escape import (
    acoustic_escape_tau_l,
    build_tau_l,
    constant_tau_l,
)


class TestConstantTauL:
    def test_tiles_scalar_across_grid(self) -> None:
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        tau_l = constant_tau_l(omega, value=0.17)
        assert tau_l.shape == (1, 5)
        np.testing.assert_allclose(tau_l, 0.17)

    def test_multi_branch_shape(self) -> None:
        omega = np.tile(np.linspace(0.1, 1.0, 4), (2, 1))
        tau_l = constant_tau_l(omega, value=0.3)
        assert tau_l.shape == (2, 4)
        np.testing.assert_allclose(tau_l, 0.3)

    def test_rejects_non_positive(self) -> None:
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        with pytest.raises(ValueError, match="must be positive"):
            constant_tau_l(omega, value=0.0)


class TestAcousticEscapeTauL:
    def _mat_al_like(self) -> Material:
        return Material(
            name="X", Delta_0=180.0, T_c=1.2, tau_0=1.0,
            sound_velocity_longitudinal=6000.0,
            sound_velocity_transverse=3000.0,
            film_thickness=63.0,
            substrate_transmission_eta=0.2,
        )

    def test_fischer_like_value(self) -> None:
        # For d = 63 nm, η = 0.2, s_D ≈ 3527 m/s (from our L/T pair):
        # τ_l = 4 · 63 / (0.2 · 3527) ≈ 357 ps.
        mat = self._mat_al_like()
        omega = np.linspace(0.1, 1.0, 10).reshape(1, 10)
        tau_l = acoustic_escape_tau_l(omega, mat)
        assert tau_l.shape == (1, 10)
        # All entries equal (frequency-independent in v1).
        np.testing.assert_allclose(tau_l, tau_l[0, 0])
        # Check the formula: 4 · 63 / (0.2 · s_D)
        assert mat.sound_velocity_debye is not None
        expected = 4.0 * 63.0 / (0.2 * mat.sound_velocity_debye)
        assert tau_l[0, 0] == pytest.approx(expected)

    def test_longitudinal_branch(self) -> None:
        mat = self._mat_al_like()
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        tau_l = acoustic_escape_tau_l(omega, mat, branch="longitudinal")
        expected = 4.0 * 63.0 / (0.2 * 6000.0)
        np.testing.assert_allclose(tau_l, expected)

    def test_transverse_branch(self) -> None:
        mat = self._mat_al_like()
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        tau_l = acoustic_escape_tau_l(omega, mat, branch="transverse")
        expected = 4.0 * 63.0 / (0.2 * 3000.0)
        np.testing.assert_allclose(tau_l, expected)

    def test_rejects_unknown_branch(self) -> None:
        mat = self._mat_al_like()
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        with pytest.raises(ValueError, match="branch must be"):
            acoustic_escape_tau_l(omega, mat, branch="bogus")

    def test_rejects_missing_film_thickness(self) -> None:
        mat = Material(name="X", Delta_0=180.0, T_c=1.2, tau_0=1.0,
                       substrate_transmission_eta=0.2,
                       sound_velocity_longitudinal=6000.0,
                       sound_velocity_transverse=3000.0)
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        with pytest.raises(ValueError, match="film_thickness"):
            acoustic_escape_tau_l(omega, mat)

    def test_rejects_missing_eta(self) -> None:
        mat = Material(name="X", Delta_0=180.0, T_c=1.2, tau_0=1.0,
                       film_thickness=50.0,
                       sound_velocity_longitudinal=6000.0,
                       sound_velocity_transverse=3000.0)
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        with pytest.raises(ValueError, match="substrate_transmission_eta"):
            acoustic_escape_tau_l(omega, mat)

    def test_rejects_missing_sound_velocity(self) -> None:
        mat = Material(name="X", Delta_0=180.0, T_c=1.2, tau_0=1.0,
                       film_thickness=50.0,
                       substrate_transmission_eta=0.2)
        # s_D derivation requires both s_L and s_T; without them, it's None.
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        with pytest.raises(ValueError, match="sound_velocity"):
            acoustic_escape_tau_l(omega, mat)


class TestBuildTauL:
    def _mat(self, *, tau_0_pb_ns: float | None = 0.255) -> Material:
        return Material(
            name="X", Delta_0=180.0, T_c=1.2, tau_0=1.0,
            sound_velocity_longitudinal=6000.0,
            sound_velocity_transverse=3000.0,
            film_thickness=63.0, substrate_transmission_eta=0.2,
            tau_0_pb_ns=tau_0_pb_ns,
        )

    def _omega(self) -> np.ndarray:
        return np.linspace(0.1, 1.0, 6).reshape(1, 6)

    def test_acoustic_escape_matches_builder(self) -> None:
        mat, omega = self._mat(), self._omega()
        np.testing.assert_allclose(
            build_tau_l("acoustic_escape", omega, mat),
            acoustic_escape_tau_l(omega, mat),
        )

    def test_constant_uses_value(self) -> None:
        mat, omega = self._mat(), self._omega()
        np.testing.assert_allclose(
            build_tau_l("constant", omega, mat, value=0.42), 0.42
        )

    def test_constant_requires_value(self) -> None:
        mat, omega = self._mat(), self._omega()
        with pytest.raises(ValueError, match="requires an explicit value"):
            build_tau_l("constant", omega, mat)

    def test_tau_0_pb_pins_to_material(self) -> None:
        # Paper convention τ_l = τ_0^PB; here 255 ps, distinct from the
        # ~357 ps acoustic-escape value, so the two models are separable.
        mat, omega = self._mat(tau_0_pb_ns=0.255), self._omega()
        np.testing.assert_allclose(build_tau_l("tau_0_pb", omega, mat), 0.255)

    def test_tau_0_pb_requires_material_field(self) -> None:
        mat, omega = self._mat(tau_0_pb_ns=None), self._omega()
        with pytest.raises(ValueError, match="tau_0_pb_ns"):
            build_tau_l("tau_0_pb", omega, mat)

    def test_rejects_unknown_model(self) -> None:
        mat, omega = self._mat(), self._omega()
        with pytest.raises(ValueError, match="unknown tau_l model"):
            build_tau_l("bogus", omega, mat)
