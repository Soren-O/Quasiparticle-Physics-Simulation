"""Material dataclass + YAML-backed database.

The :class:`Material` dataclass captures everything the kinetics
framework needs about a superconducting material: gap, critical
temperature, e-ph timescales, normal-state transport, phonon branches
(per Phonon_Model_Decisions.md D5, all three sound velocities are
carried so the v3 multi-branch extension is purely additive), film
thickness, and a :class:`Substrate` descriptor.

YAML files live in ``qpsim/materials/data/``. Load one with
``load_material("Al")``. Pass ``database_dir`` to point at a custom
directory for user-defined materials.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from qpsim.materials.substrate import Substrate

_LEGACY_RHO_F_MAX = 1.0e25


def validate_rho_F_eV(rho_F: float, *, allow_zero: bool) -> float:
    """Validate the eV-based DOS contract and catch legacy micro-eV values."""
    value = float(rho_F)
    if not isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(
            f"rho_F must be finite and {qualifier} (eV^-1 m^-3); got {value}."
        )
    if 0.0 < value < _LEGACY_RHO_F_MAX:
        raise ValueError(
            f"rho_F={value:g} eV^-1 m^-3 is implausibly small for a "
            "volumetric electronic DOS and likely carries legacy per-"
            "micro-eV or per-micrometre^3 units. Convert to eV^-1 m^-3 "
            "(for legacy Al: 1.74e22 -> 1.74e28)."
        )
    return value


@dataclass
class Material:
    """Superconducting-material parameters.

    Required: ``name``, ``Delta_0``, ``T_c``, ``tau_0``. All other
    fields default to zero (or ``None`` for derived/optional values)
    so a minimal YAML works. ``tau_s`` and ``tau_r`` default to
    ``tau_0`` after construction via ``__post_init__``; the Debye
    sound velocity ``sound_velocity_debye`` is derived from the
    longitudinal + transverse pair when not supplied explicitly.
    """

    name: str

    # Superconducting parameters.
    Delta_0: float              # T=0 gap (μeV)
    T_c: float                  # critical temperature (K)

    # Electron-phonon timescales (all ns).
    tau_0: float                # characteristic e-ph time
    tau_s: float | None = None  # scattering time (defaults to tau_0)
    tau_r: float | None = None  # recombination time (defaults to tau_0)
    # Phonon-side characteristic time (Kaplan 1976 Eq. 30; Table II
    # values). Distinct from ``tau_0`` (which is the QP side). Used by
    # :func:`qpsim.physics.kaplan_pair_breaking.tau_PB_inverse_Hz` to
    # build the frequency-resolved pair-breaking rate. Optional —
    # materials that don't supply it will error if the Kaplan evaluator
    # is called with that material's ``tau_0_phonon``.
    tau_0_phonon: float | None = None  # τ_0^ph (ns)
    # Pair-breaking phonon characteristic time (F&C 2023 Eq. 13;
    # appears as ``τ_0^PB`` in the paper's phonon kinetic equation
    # Eq. 12). For Al, F&C 2023 quotes 0.255 ns from the parameters
    # in Table I. Used by the opt-in phonon-side F&C Eq. 12 kernels:
    # scattering ``2K⁻/(π Δ τ_0^PB)`` and pair-breaking/recombination
    # ``K⁺/(π Δ τ_0^PB)``. Distinct from ``tau_0`` (QP-side e-ph
    # characteristic time): F&C Eq. 13 relates the two via a
    # material-specific ratio (≈ 1.7×10³ for Al). Dynamic Ph0 backend
    # solves use these phonon-side kernels by default; callers can explicitly
    # opt into the pre-fix QP-side-kernel behavior for legacy reproduction.
    tau_0_pb_ns: float | None = None  # τ_0^PB (ns)

    # Normal-state transport.
    D_0: float = 0.0            # normal-state diffusion (μm²/ns)
    v_F: float = 0.0            # Fermi velocity (m/s)
    rho_F: float = 0.0          # single-spin DOS (eV⁻¹ m⁻³)

    # Phonon branches (D5 commits to carrying all three; the Debye
    # average is the scalar-s default for the Ph0 single-branch model).
    sound_velocity_longitudinal: float = 0.0  # s_L (m/s)
    sound_velocity_transverse: float = 0.0    # s_T (m/s)
    sound_velocity_debye: float | None = None  # s_D; derived if omitted

    # Film geometry and film-substrate interface.
    film_thickness: float = 0.0  # nm
    substrate: Substrate | None = None
    substrate_transmission_eta: float = 0.0

    def __post_init__(self) -> None:
        # YAML 1.1 resolves unsigned-exponent scientific notation
        # ("1.74e28") as a *string*, not a float — the shipped Al/Nb/TiN
        # files write v_F and rho_F that way. Coerce every numeric field
        # so loaded materials are usable in arithmetic regardless of
        # notation.
        for field_name in _MATERIAL_FLOAT_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                setattr(self, field_name, float(value))

        if self.tau_s is None:
            self.tau_s = self.tau_0
        if self.tau_r is None:
            self.tau_r = self.tau_0

        required_positive = ("Delta_0", "T_c", "tau_0", "tau_s", "tau_r")
        optional_positive = ("tau_0_phonon", "tau_0_pb_ns")
        optional_nonnegative = (
            "D_0",
            "v_F",
            "sound_velocity_longitudinal",
            "sound_velocity_transverse",
            "sound_velocity_debye",
            "film_thickness",
        )
        for field_name in required_positive:
            value = getattr(self, field_name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive; got {value}.")
        for field_name in optional_positive:
            value = getattr(self, field_name)
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError(f"{field_name} must be finite and positive when set; got {value}.")
        for field_name in optional_nonnegative:
            value = getattr(self, field_name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative; got {value}.")
        if not isfinite(self.substrate_transmission_eta) or not (
            0.0 <= self.substrate_transmission_eta <= 1.0
        ):
            raise ValueError(
                "substrate_transmission_eta must be finite and lie in [0, 1]; "
                f"got {self.substrate_transmission_eta}."
            )

        self.rho_F = validate_rho_F_eV(self.rho_F, allow_zero=True)

        if (
            self.sound_velocity_debye is None
            and self.sound_velocity_longitudinal > 0
            and self.sound_velocity_transverse > 0
        ):
            # s_D⁻³ = (1/3)(s_L⁻³ + 2 s_T⁻³)  — standard Debye average.
            inv_sL3 = 1.0 / self.sound_velocity_longitudinal ** 3
            inv_sT3 = 1.0 / self.sound_velocity_transverse ** 3
            inv_sD3 = (inv_sL3 + 2.0 * inv_sT3) / 3.0
            self.sound_velocity_debye = inv_sD3 ** (-1.0 / 3.0)


_MATERIAL_FLOAT_FIELDS = (
    "Delta_0",
    "T_c",
    "tau_0",
    "tau_s",
    "tau_r",
    "tau_0_phonon",
    "tau_0_pb_ns",
    "D_0",
    "v_F",
    "rho_F",
    "sound_velocity_longitudinal",
    "sound_velocity_transverse",
    "sound_velocity_debye",
    "film_thickness",
    "substrate_transmission_eta",
)


def _default_database_dir() -> Path:
    return Path(__file__).parent / "data"


def load_material(
    name: str,
    *,
    database_dir: Path | None = None,
) -> Material:
    """Load a material by name from a YAML database.

    Reads ``{database_dir}/{name}.yaml``. Defaults to the built-in
    database at ``qpsim/materials/data/``. The YAML may include a
    nested ``substrate:`` mapping; its contents are passed to
    :class:`Substrate`.
    """
    dir_path = _default_database_dir() if database_dir is None else database_dir
    yaml_path = dir_path / f"{name}.yaml"
    # Guard against path traversal in the material name (e.g. "../secret"): the
    # resolved path must stay inside the database directory. Material names are
    # simple file stems, never paths.
    if not yaml_path.resolve().is_relative_to(Path(dir_path).resolve()):
        raise ValueError(
            f"Material name {name!r} resolves outside the database directory "
            f"{dir_path}; material names must be simple file stems."
        )
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Material '{name}' not found at {yaml_path}."
        )
    with yaml_path.open(encoding="utf-8") as fp:
        data: dict[str, Any] = yaml.safe_load(fp)

    if not isinstance(data, dict):
        raise ValueError(
            f"Material YAML at {yaml_path} must parse to a mapping, "
            f"got {type(data).__name__}."
        )

    substrate_data = data.pop("substrate", None)
    substrate = None
    if substrate_data is not None:
        if not isinstance(substrate_data, dict):
            raise ValueError(
                f"'substrate:' in {yaml_path} must be a mapping, "
                f"got {type(substrate_data).__name__}."
            )
        substrate = Substrate(**substrate_data)

    return Material(substrate=substrate, **data)


def list_materials(*, database_dir: Path | None = None) -> list[str]:
    """Return the names of all materials in the database directory."""
    dir_path = _default_database_dir() if database_dir is None else database_dir
    return sorted(p.stem for p in dir_path.glob("*.yaml"))
