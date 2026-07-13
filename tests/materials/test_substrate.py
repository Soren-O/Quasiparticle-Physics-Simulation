"""Tests for qpsim.materials.substrate."""

from __future__ import annotations

import pytest
from qpsim.materials.substrate import Substrate


class TestSubstrate:
    def test_minimal_construction(self) -> None:
        sub = Substrate(name="Si")
        assert sub.name == "Si"
        assert sub.density is None
        assert sub.sound_velocity is None

    def test_full_construction(self) -> None:
        sub = Substrate(name="Al2O3", density=3980.0, sound_velocity=10800.0)
        assert sub.density == 3980.0
        assert sub.sound_velocity == 10800.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("density", 0.0),
            ("density", float("nan")),
            ("sound_velocity", -1.0),
            ("sound_velocity", float("inf")),
        ],
    )
    def test_rejects_nonphysical_acoustic_fields(
        self, field: str, value: float,
    ) -> None:
        with pytest.raises(ValueError, match=field):
            Substrate(name="X", **{field: value})
