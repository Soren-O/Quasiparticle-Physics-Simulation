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
from pathlib import Path
from typing import Any

import yaml

from qpsim.materials.substrate import Substrate


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

    # Normal-state transport.
    D_0: float = 0.0            # normal-state diffusion (μm²/ns)
    v_F: float = 0.0            # Fermi velocity (m/s)
    rho_F: float = 0.0          # single-spin DOS (J⁻¹ m⁻³)

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
        if self.tau_s is None:
            self.tau_s = self.tau_0
        if self.tau_r is None:
            self.tau_r = self.tau_0

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
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Material '{name}' not found at {yaml_path}."
        )
    with yaml_path.open() as fp:
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
