"""Tests for qpsim.materials.database."""

from __future__ import annotations

from pathlib import Path

import pytest
from qpsim.materials.database import Material, list_materials, load_material


class TestMaterialDataclass:
    def test_minimal_required_fields(self) -> None:
        mat = Material(name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0)
        assert mat.name == "X"
        assert mat.Delta_0 == 200.0
        # tau_s and tau_r default to tau_0 via __post_init__.
        assert mat.tau_s == 1.0
        assert mat.tau_r == 1.0

    def test_debye_velocity_derivation(self) -> None:
        # s_D⁻³ = (1/3)(s_L⁻³ + 2 s_T⁻³).
        mat = Material(
            name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0,
            sound_velocity_longitudinal=6000.0,
            sound_velocity_transverse=3000.0,
        )
        inv_sD3 = (1.0 / 6000.0 ** 3 + 2.0 / 3000.0 ** 3) / 3.0
        expected = inv_sD3 ** (-1.0 / 3.0)
        assert mat.sound_velocity_debye == pytest.approx(expected)

    def test_debye_velocity_preserved_when_given(self) -> None:
        mat = Material(
            name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0,
            sound_velocity_longitudinal=6000.0,
            sound_velocity_transverse=3000.0,
            sound_velocity_debye=4200.0,
        )
        assert mat.sound_velocity_debye == 4200.0

    def test_debye_velocity_none_when_branches_missing(self) -> None:
        # With zero s_L and s_T, the Debye average is left as None so
        # callers can detect an incomplete spec.
        mat = Material(name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0)
        assert mat.sound_velocity_debye is None

    def test_explicit_tau_overrides(self) -> None:
        mat = Material(
            name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0,
            tau_s=5.0, tau_r=10.0,
        )
        assert mat.tau_s == 5.0
        assert mat.tau_r == 10.0


class TestLoadMaterial:
    def test_loads_aluminum_from_builtin_database(self) -> None:
        mat = load_material("Al")
        assert mat.name == "Al"
        assert mat.Delta_0 == pytest.approx(180.0)
        assert mat.T_c == pytest.approx(1.18)
        assert mat.substrate is not None
        assert mat.substrate.name == "Al2O3"

    def test_loads_niobium(self) -> None:
        mat = load_material("Nb")
        assert mat.name == "Nb"
        # Δ(0) for Nb is ~1.5 meV = 1500 μeV.
        assert mat.Delta_0 == pytest.approx(1500.0)

    def test_loads_tin(self) -> None:
        mat = load_material("TiN")
        assert mat.name == "TiN"
        assert 4.0 < mat.T_c < 5.0

    def test_every_numeric_field_loads_as_float(self) -> None:
        # YAML 1.1 resolves unsigned-exponent notation ("1.74e28") as a
        # string — the shipped files write v_F/rho_F that way, and the
        # dataclass must coerce so arithmetic on loaded materials works.
        for name in ("Al", "Nb", "TiN"):
            mat = load_material(name)
            for field_name, value in vars(mat).items():
                if field_name in ("name", "substrate"):
                    continue
                assert value is None or isinstance(value, float), (
                    f"{name}.{field_name} loaded as {type(value).__name__}: {value!r}"
                )
        al = load_material("Al")
        assert al.rho_F == pytest.approx(1.74e28)
        assert al.v_F == pytest.approx(1.36e6)

    def test_unsigned_exponent_substrate_fields_coerced(self, tmp_path: Path) -> None:
        (tmp_path / "Weird.yaml").write_text(
            "name: Weird\nDelta_0: 42.0\nT_c: 0.5\ntau_0: 1.0\n"
            "substrate:\n  name: Sub\n  density: 2.3e3\n  sound_velocity: 8.4e3\n"
        )
        mat = load_material("Weird", database_dir=tmp_path)
        assert mat.substrate is not None
        assert mat.substrate.density == pytest.approx(2300.0)
        assert mat.substrate.sound_velocity == pytest.approx(8400.0)

    def test_raises_on_missing(self) -> None:
        with pytest.raises(FileNotFoundError, match="Material 'Nope' not found"):
            load_material("Nope")

    def test_custom_database_dir(self, tmp_path: Path) -> None:
        (tmp_path / "Custom.yaml").write_text(
            "name: Custom\nDelta_0: 42.0\nT_c: 0.5\ntau_0: 1.0\n"
        )
        mat = load_material("Custom", database_dir=tmp_path)
        assert mat.name == "Custom"
        assert mat.Delta_0 == 42.0

    def test_rejects_non_mapping_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "BadList.yaml").write_text("- not-a-dict\n- still-not\n")
        with pytest.raises(ValueError, match="must parse to a mapping"):
            load_material("BadList", database_dir=tmp_path)

    def test_rejects_non_mapping_substrate(self, tmp_path: Path) -> None:
        (tmp_path / "Bad.yaml").write_text(
            "name: Bad\nDelta_0: 42.0\nT_c: 0.5\ntau_0: 1.0\nsubstrate: 'just a string'\n"
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            load_material("Bad", database_dir=tmp_path)

    def test_loads_with_no_substrate(self, tmp_path: Path) -> None:
        (tmp_path / "NoSub.yaml").write_text(
            "name: NoSub\nDelta_0: 42.0\nT_c: 0.5\ntau_0: 1.0\n"
        )
        mat = load_material("NoSub", database_dir=tmp_path)
        assert mat.substrate is None


class TestListMaterials:
    def test_lists_builtin(self) -> None:
        names = list_materials()
        assert "Al" in names
        assert "Nb" in names
        assert "TiN" in names

    def test_lists_custom(self, tmp_path: Path) -> None:
        (tmp_path / "A.yaml").touch()
        (tmp_path / "B.yaml").touch()
        (tmp_path / "notyaml.txt").touch()  # ignored
        assert list_materials(database_dir=tmp_path) == ["A", "B"]
