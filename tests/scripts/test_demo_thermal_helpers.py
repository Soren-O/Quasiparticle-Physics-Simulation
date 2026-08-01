from __future__ import annotations

import numpy as np
import pytest

from scripts import demo_kid_pulse_response as kid_demo
from scripts import demo_materials_pb_response as materials_demo


def test_kid_demo_delegates_thermal_seed_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy = np.array([kid_demo.DELTA_0, 2.0 * kid_demo.DELTA_0])
    expected = np.array([0.25, 0.125])

    def fake_fermi_dirac(actual_energy: np.ndarray, temperature: float) -> np.ndarray:
        assert actual_energy is energy
        assert temperature == kid_demo.T_BATH
        return expected

    monkeypatch.setattr(kid_demo, "fermi_dirac_occupation", fake_fermi_dirac)

    assert kid_demo._fermi_dirac(energy) is expected


def test_material_demo_delegates_positive_temperature_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy = np.array([100.0, 200.0])
    temperature = 0.2
    expected = np.array([0.2, 0.1])

    def fake_fermi_dirac(
        actual_energy: np.ndarray,
        actual_temperature: float,
    ) -> np.ndarray:
        assert actual_energy is energy
        assert actual_temperature == temperature
        return expected

    monkeypatch.setattr(
        materials_demo,
        "fermi_dirac_occupation",
        fake_fermi_dirac,
    )

    assert materials_demo._fermi_dirac(energy, temperature) is expected


def test_material_demo_preserves_zero_temperature_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy = np.array([100.0, 200.0])

    def unexpected_call(*_args: object, **_kwargs: object) -> np.ndarray:
        pytest.fail("The shared positive-temperature helper must not be called.")

    monkeypatch.setattr(
        materials_demo,
        "fermi_dirac_occupation",
        unexpected_call,
    )

    np.testing.assert_array_equal(
        materials_demo._fermi_dirac(energy, 0.0),
        np.zeros_like(energy),
    )
