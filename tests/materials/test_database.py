"""Tests for qpsim.materials.database."""

from __future__ import annotations

import warnings
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
        # Every shipped material, not a hand-listed three: a new entry with a
        # string-typed exponent would otherwise ship unnoticed.
        for name in list_materials():
            mat = load_material(name)
            for field_name, value in vars(mat).items():
                # Provenance fields are deliberately not numbers.
                if field_name in (
                    "name", "substrate", "notes", "references", "D_0_range",
                ):
                    continue
                assert value is None or isinstance(value, float), (
                    f"{name}.{field_name} loaded as {type(value).__name__}: {value!r}"
                )
        al = load_material("Al")
        # Conventional single-spin DOS in eV⁻¹ m⁻³.
        assert al.rho_F == pytest.approx(1.74e28)
        assert al.v_F == pytest.approx(1.36e6)

    @pytest.mark.parametrize("rho_F", [float("nan"), float("inf"), -1.0])
    def test_rejects_invalid_rho_F(self, rho_F: float) -> None:
        with pytest.raises(ValueError, match="rho_F"):
            Material(name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0, rho_F=rho_F)

    def test_rejects_legacy_per_uev_rho_f(self) -> None:
        with pytest.raises(ValueError, match="implausibly small"):
            Material(
                name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0, rho_F=1.74e22
            )

    def test_rejects_other_tiny_rho_f_unit_mistakes(self) -> None:
        with pytest.raises(ValueError, match="implausibly small"):
            Material(
                name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0, rho_F=1.74e10
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("Delta_0", 0.0),
            ("T_c", -1.0),
            ("tau_0", float("nan")),
            ("tau_s", 0.0),
            ("D_0", -1.0),
            ("sound_velocity_longitudinal", float("inf")),
            ("film_thickness", -1.0),
            ("substrate_transmission_eta", 1.01),
        ],
    )
    def test_rejects_nonphysical_material_fields(
        self, field: str, value: float,
    ) -> None:
        kwargs = {"Delta_0": 200.0, "T_c": 1.0, "tau_0": 1.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            Material(name="X", **kwargs)

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


class TestProvenance:
    """A bare number cannot be chosen with; the band and the citation can.

    D_0 spans a factor of five or more across film qualities of the same
    material, so "which value should I use for my film" is answered by the
    reference class, not by the stored scalar.
    """

    def test_every_material_carries_its_sources(self) -> None:
        for name in list_materials():
            mat = load_material(name)
            assert mat.references, f"{name} has no references"
            assert mat.notes.strip(), f"{name} has no film-class note"
            assert mat.D_0_range is not None, f"{name} has no D_0 band"

    def test_a_stored_value_outside_its_own_band_warns(self) -> None:
        """The inconsistency has to be visible at load, not only in a comment."""
        with pytest.warns(UserWarning, match="outside its own sourced band"):
            Material(
                name="X", Delta_0=1.0, T_c=1.0, tau_0=1.0,
                D_0=60.0, D_0_range=(2.25, 10.0),
            )

    def test_a_value_inside_its_band_is_quiet(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Material(
                name="X", Delta_0=1.0, T_c=1.0, tau_0=1.0,
                D_0=3.0, D_0_range=(2.0, 4.0),
            )

    def test_an_inverted_band_is_refused(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            Material(
                name="X", Delta_0=1.0, T_c=1.0, tau_0=1.0,
                D_0=1.0, D_0_range=(4.0, 2.0),
            )
