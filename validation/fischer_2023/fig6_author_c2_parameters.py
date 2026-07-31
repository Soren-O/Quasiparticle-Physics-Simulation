"""Explicit Figure 6 parameter plumbing for the C2 substitution stage.

This module deliberately does not load a material YAML record.  C2 must
separate two questions that an implicit ``load_material("Al")`` would mix:

``C2a``
    Can the authenticated author values be represented in qpsim's native
    micro-eV / nanosecond units without changing the values delivered to the
    still-author-equivalent numerical operator?

``C2b``
    What changes when those values are replaced, one controlled field at a
    time, by the values explicitly declared by qpsim's Fischer Figure 6
    configuration?

Energy conversion deserves special care.  The authenticated author literal
``180 * 10**-6`` is one binary64 value in eV.  Multiplying that value by
``1e6`` produces ``179.99999999999997`` micro-eV, one ulp below qpsim's
explicit ``180.0`` micro-eV literal.  :class:`EnergySpaceMapping` therefore
retains the source-space value as the authoritative value for the C2a author
operator while exposing a separate qpsim-native view.  It never silently
substitutes qpsim's rounded display literal into the author equations.

The records here are parameter evidence, not a completed ladder score.  A
strict frozen-state operator differential can complete the parameter stage.
The C2b values nevertheless change the nonlinear root, so a later solver
stage must independently re-solve them before assigning C2b a root or plotted
ordinate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.gap_equation import calibrate_gap

from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
)
from validation.source_provenance import source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c2-parameter-plan.v1"

EV_TO_UEV = 1.0e6
UEV_TO_EV = 1.0e-6
SECONDS_PER_NS = 1.0e-9
EQ35_COEFFICIENT = 105.0 / 64.0
FINITE_CUTOFF_XTOL_UEV = 1.0e-12

# Reviewed expectations for validation.fischer_2023.fig6_solve.  These
# literals make a change to the scientific endpoint an explicit review event;
# :func:`qpsim_fig6_configuration_endpoint` also extracts the live explicit
# declarations and grid configuration, and C2 refuses to run if either side
# drifts. It deliberately never loads the generic Al material/YAML.
QPSIM_FIG6_DELTA0_UEV = 180.0
QPSIM_FIG6_TAU0_NS = 438.0
QPSIM_FIG6_TAU0_PB_NS = 255.0 / 1000.0
QPSIM_FIG6_TAU_L_NS = 255.0 / 1000.0
QPSIM_FIG6_C_PHOTON_NS_INV = 1.0e-9

_PROVENANCE_PATHS = (
    REPOSITORY_ROOT / "qpsim" / "constants.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "gap_equation.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_solve.py",
    REPOSITORY_ROOT / "validation" / "reference_models" / "fischer_2023" / "fig6_author_c0.py",
)


class C2ParameterError(ValueError):
    """A C2 parent parameter or controlled mapping is invalid."""


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise C2ParameterError(f"{label} must be a finite positive scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C2ParameterError(f"{label} must be a finite positive scalar.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise C2ParameterError(f"{label} must be a finite positive scalar.")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise C2ParameterError(f"{label} must be a positive integer.")
    return value


def _require_exact_binding(label: str, expected: object, live: object) -> None:
    """Require an exact semantic-endpoint match, without tolerance masking."""

    if live != expected:
        raise C2ParameterError(
            f"Figure 6 endpoint drift for {label}: expected {expected!r}, "
            f"live configuration provides {live!r}. Review and re-pin the "
            "C2 parameter ladder before accepting this change."
        )


@dataclass(frozen=True)
class Fig6ConfigurationEndpoint:
    """Live explicit inputs extracted from the current Figure 6 solve.

    The full qpsim grid has twenty sub-gap cells that the author C0/C2
    operator intentionally does not adopt until C3.  C2 binds the shared
    grid semantics instead: uniform ``dE``, the integer photon shift, and
    their resulting photon energy.  The remaining grid fields are retained
    here so tests can prove that the extractor is reading the current solve
    rather than another handwritten copy.
    """

    delta0_ueV: float
    tau_0_ns: float
    tau_0_pb_ns: float
    tau_l_ns: float
    c_photon_ns_inv: float
    omega_0_ueV: float
    grid_spacing_ueV: float
    photon_bin: int
    grid_num_bins: int
    energy_min_factor: float
    energy_max_factor: float
    T_c_K: float
    finite_cutoff_ratio: float
    gap_solve_xtol_ueV: float
    tau_l_model: str

    def __post_init__(self) -> None:
        for field_name in (
            "delta0_ueV",
            "tau_0_ns",
            "tau_0_pb_ns",
            "tau_l_ns",
            "c_photon_ns_inv",
            "omega_0_ueV",
            "grid_spacing_ueV",
            "energy_min_factor",
            "energy_max_factor",
            "T_c_K",
            "finite_cutoff_ratio",
            "gap_solve_xtol_ueV",
        ):
            _finite_positive(getattr(self, field_name), field_name)
        _positive_integer(self.photon_bin, "photon_bin")
        _positive_integer(self.grid_num_bins, "grid_num_bins")
        if not self.tau_l_model:
            raise C2ParameterError("tau_l_model must be a non-empty string.")


@dataclass(frozen=True)
class EnergySpaceMapping:
    """One source-eV scalar and its non-authoritative qpsim-native view."""

    name: str
    source_eV: float
    native_ueV: float
    scale_ueV_per_eV: float = EV_TO_UEV

    @classmethod
    def from_eV(cls, name: str, source_eV: float) -> EnergySpaceMapping:
        """Create a mapping without feeding a lossy round trip back to C2a."""

        source = _finite_positive(source_eV, f"{name}.source_eV")
        return cls(
            name=name,
            source_eV=source,
            native_ueV=source * EV_TO_UEV,
        )

    @property
    def author_effective_eV(self) -> float:
        """The exact source-space value that C2a must keep using."""

        return self.source_eV

    @property
    def native_round_trip_eV(self) -> float:
        """Diagnostic only: the native value converted back to eV."""

        return self.native_ueV / self.scale_ueV_per_eV

    @property
    def round_trip_bit_exact(self) -> bool:
        return self.native_round_trip_eV == self.source_eV

    def as_record(self) -> dict[str, object]:
        return {
            "author_effective_eV": self.author_effective_eV,
            "author_effective_eV_hex": self.author_effective_eV.hex(),
            "name": self.name,
            "native_round_trip_eV": self.native_round_trip_eV,
            "native_round_trip_eV_hex": self.native_round_trip_eV.hex(),
            "native_ueV": self.native_ueV,
            "native_ueV_hex": self.native_ueV.hex(),
            "round_trip_bit_exact": self.round_trip_bit_exact,
            "scale_ueV_per_eV": self.scale_ueV_per_eV,
            "source_eV": self.source_eV,
            "source_eV_hex": self.source_eV.hex(),
        }


@dataclass(frozen=True)
class TimeSpaceMapping:
    """One source-second scalar and its qpsim-native nanosecond view."""

    name: str
    source_s: float
    native_ns: float

    @classmethod
    def from_seconds(cls, name: str, source_s: float) -> TimeSpaceMapping:
        source = _finite_positive(source_s, f"{name}.source_s")
        return cls(name=name, source_s=source, native_ns=source / SECONDS_PER_NS)

    @property
    def native_round_trip_s(self) -> float:
        return self.native_ns * SECONDS_PER_NS

    @property
    def round_trip_bit_exact(self) -> bool:
        return self.native_round_trip_s == self.source_s

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "native_ns": self.native_ns,
            "native_ns_hex": self.native_ns.hex(),
            "native_round_trip_s": self.native_round_trip_s,
            "native_round_trip_s_hex": self.native_round_trip_s.hex(),
            "round_trip_bit_exact": self.round_trip_bit_exact,
            "source_s": self.source_s,
            "source_s_hex": self.source_s.hex(),
        }


@dataclass(frozen=True)
class RateSpaceMapping:
    """One source ``s^-1`` scalar and its qpsim-native ``ns^-1`` view."""

    name: str
    source_s_inv: float
    native_ns_inv: float

    @classmethod
    def from_per_second(cls, name: str, source_s_inv: float) -> RateSpaceMapping:
        source = _finite_positive(source_s_inv, f"{name}.source_s_inv")
        return cls(
            name=name,
            source_s_inv=source,
            native_ns_inv=source * SECONDS_PER_NS,
        )

    @property
    def native_round_trip_s_inv(self) -> float:
        return self.native_ns_inv / SECONDS_PER_NS

    @property
    def round_trip_bit_exact(self) -> bool:
        return self.native_round_trip_s_inv == self.source_s_inv

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "native_ns_inv": self.native_ns_inv,
            "native_ns_inv_hex": self.native_ns_inv.hex(),
            "native_round_trip_s_inv": self.native_round_trip_s_inv,
            "native_round_trip_s_inv_hex": self.native_round_trip_s_inv.hex(),
            "round_trip_bit_exact": self.round_trip_bit_exact,
            "source_s_inv": self.source_s_inv,
            "source_s_inv_hex": self.source_s_inv.hex(),
        }


@dataclass(frozen=True)
class NativeFig6Parameters:
    """The parameter vector consumed by qpsim-unit Figure 6 arithmetic."""

    gap_ueV: float
    delta0_ueV: float
    h_ueV: float
    temperature_K: float
    T_c_K: float
    tau_0_ns: float
    tau_0_pb_ns: float
    tau_l_ns: float
    photon_bin: int
    n_bar: float
    c_photon_ns_inv: float
    kB_ueV_per_K: float

    def __post_init__(self) -> None:
        for field_name in (
            "gap_ueV",
            "delta0_ueV",
            "h_ueV",
            "temperature_K",
            "T_c_K",
            "tau_0_ns",
            "tau_0_pb_ns",
            "tau_l_ns",
            "n_bar",
            "c_photon_ns_inv",
            "kB_ueV_per_K",
        ):
            _finite_positive(getattr(self, field_name), field_name)
        _positive_integer(self.photon_bin, "photon_bin")

    @property
    def omega_0_ueV(self) -> float:
        """Solved photon energy on the frozen author grid."""

        return self.photon_bin * self.h_ueV

    @property
    def eq35_t_star_over_delta(self) -> float:
        """Figure 6 Eq. 35 coordinate using qpsim's native-unit arithmetic."""

        A_ueV6 = (
            EQ35_COEFFICIENT
            * (self.kB_ueV_per_K * self.T_c_K) ** 3
            * self.c_photon_ns_inv
            * self.tau_0_ns
            * self.omega_0_ueV**2
            * self.gap_ueV
        )
        return float((A_ueV6 * self.n_bar) ** (1.0 / 6.0) / self.delta0_ueV)

    def as_record(self) -> dict[str, object]:
        values = asdict(self)
        values["omega_0_ueV"] = self.omega_0_ueV
        values["eq35_t_star_over_delta"] = self.eq35_t_star_over_delta
        values["hex"] = {
            key: value.hex() for key, value in values.items() if isinstance(value, float)
        }
        return values


@dataclass(frozen=True)
class ControlledParameterStep:
    """One diagnostic substep inside the single ``parameters`` component."""

    step_id: str
    changed_fields: tuple[str, ...]
    parameters: NativeFig6Parameters
    qualification: str

    def as_record(self) -> dict[str, object]:
        return {
            "changed_fields": list(self.changed_fields),
            "parameters": self.parameters.as_record(),
            "qualification": self.qualification,
            "step_id": self.step_id,
        }


@dataclass(frozen=True)
class C2ParameterPlan:
    """C2a exact plumbing plus the controlled path to the C2b endpoint."""

    energy_mappings: tuple[EnergySpaceMapping, ...]
    time_mappings: tuple[TimeSpaceMapping, ...]
    photon_rate_mapping: RateSpaceMapping
    author_boltzmann_constant_J_per_K: float
    author_electron_charge_C: float
    parent_author_parameters: AuthorSolveParameters
    c2a: NativeFig6Parameters
    c2b_steps: tuple[ControlledParameterStep, ...]

    @property
    def c2b(self) -> NativeFig6Parameters:
        return self.c2b_steps[-1].parameters

    @property
    def c2a_author_effective(self) -> AuthorSolveParameters:
        """C2a carrier with every authenticated parent bit preserved."""

        return author_solve_parameters_from_native(
            self.c2a,
            parent=self.parent_author_parameters,
            preserve_parent_source_bits=True,
        )

    def c2b_author_effective_steps(
        self,
    ) -> tuple[tuple[str, AuthorSolveParameters], ...]:
        """Return every C2b native step in the unchanged author's units."""

        return tuple(
            (
                step.step_id,
                author_solve_parameters_from_native(
                    step.parameters,
                    parent=self.parent_author_parameters,
                ),
            )
            for step in self.c2b_steps
        )

    def as_record(self) -> dict[str, object]:
        c2b_carriers = {
            step_id: _author_solve_parameters_record(parameters)
            for step_id, parameters in self.c2b_author_effective_steps()
        }
        return {
            "author_constant_mapping": {
                "boltzmann_constant_J_per_K": (self.author_boltzmann_constant_J_per_K),
                "electron_charge_C": self.author_electron_charge_C,
                "kB_ueV_per_K": self.c2a.kB_ueV_per_K,
                "kB_ueV_per_K_hex": self.c2a.kB_ueV_per_K.hex(),
            },
            "c2a": {
                "author_operator_carrier": _author_solve_parameters_record(
                    self.c2a_author_effective
                ),
                "parameters": self.c2a.as_record(),
                "qualification": (
                    "The qpsim-native view is emitted, but the unchanged "
                    "author operator continues to receive each exact "
                    "source-space scalar. No native energy round trip is "
                    "fed back into C2a."
                ),
            },
            "c2b": {
                "author_operator_carriers": c2b_carriers,
                "controlled_steps": [step.as_record() for step in self.c2b_steps],
                "endpoint": self.c2b.as_record(),
                "frozen_operator_differential_can_complete_parameter_stage": True,
                "nonlinear_resolve_required_before_root_or_ordinate_claim": True,
                "nonlinear_resolve_stage": "later nonlinear-solver stage",
                "qualification": (
                    "C2 may complete as a provenance-bound frozen-state "
                    "operator differential. These values change the nonlinear "
                    "root, so a later solver stage must perform a same-seed "
                    "certified re-solve before claiming a C2b root or ordinate."
                ),
            },
            "energy_space_mappings": [mapping.as_record() for mapping in self.energy_mappings],
            "no_implicit_material_or_yaml_defaults": True,
            "photon_rate_mapping": self.photon_rate_mapping.as_record(),
            "schema": SCHEMA,
            "sources": parameter_source_manifest(),
            "time_space_mappings": [mapping.as_record() for mapping in self.time_mappings],
        }


def parameter_source_manifest() -> dict[str, str]:
    """Hash the qpsim sources that define the C2b endpoint."""

    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): source_sha256(path)
        for path in _PROVENANCE_PATHS
    }


@lru_cache(maxsize=1)
def qpsim_fig6_finite_cutoff_ratio() -> float:
    """Return qpsim's finite-cutoff ``Delta(0)/(k_B T_c)`` calibration."""

    calibration = calibrate_gap(
        T_c=1.0,
        T_bath=0.0,
        xtol=FINITE_CUTOFF_XTOL_UEV,
    )
    return float(calibration.delta_0_bcs / KB_UEV_PER_K)


def qpsim_fig6_critical_temperature(delta0_ueV: float) -> float:
    """Map the explicit Figure 6 gap to qpsim's finite-cutoff ``T_c``."""

    delta0 = _finite_positive(delta0_ueV, "delta0_ueV")
    return float(delta0 / (qpsim_fig6_finite_cutoff_ratio() * KB_UEV_PER_K))


@lru_cache(maxsize=1)
def qpsim_fig6_configuration_endpoint() -> Fig6ConfigurationEndpoint:
    """Extract the current Figure 6 declarations, drive, and grid endpoint.

    This import is deliberately lazy: the C2 module remains cheap to inspect,
    while every actual plan construction is checked against the scientific
    configuration that ``fig6_solve`` executes.  Reading the built grid also
    catches drift between the declarations and the grid factory.

    Deliberately do **not** call ``fig6_solve._fischer_material()`` here.  It
    starts from ``load_material("Al")`` and would violate C2's explicit
    no-YAML/no-implicit-material contract.  The C2 endpoint binds only the
    values explicitly declared in ``fig6_solve``; its source hash separately
    makes any material-factory edit invalidate the evidence.
    """

    from validation.fischer_2023 import fig6_solve

    E, dE, spectral = fig6_solve._build_grid_and_spectral()
    if E.size < 1 or dE.size != E.size:
        raise C2ParameterError(
            "The live Figure 6 grid builder returned inconsistent energy "
            "centers and integration widths."
        )
    grid_spacing = _finite_positive(float(dE[0]), "fig6 grid spacing")
    if any(float(width) != grid_spacing for width in dE):
        raise C2ParameterError(
            "The live Figure 6 grid is not exactly uniform; C2's integer "
            "photon-bin mapping no longer describes its numerical endpoint."
        )

    _require_exact_binding(
        "grid center count",
        int(fig6_solve.NUM_BINS),
        int(E.size),
    )
    _require_exact_binding(
        "grid spectral gap",
        float(fig6_solve.DELTA_0),
        float(spectral.gap),
    )
    _require_exact_binding(
        "grid lower face",
        float(fig6_solve.E_MIN_FACTOR * fig6_solve.DELTA_0),
        float(E[0] - 0.5 * grid_spacing),
    )
    _require_exact_binding(
        "grid upper face",
        float(fig6_solve.E_MAX_FACTOR * fig6_solve.DELTA_0),
        float(E[-1] + 0.5 * grid_spacing),
    )
    tau_l_model = str(fig6_solve.TAU_L_MODEL).lower()
    if tau_l_model not in ("tau_0_pb", "tau0_pb", "pair_breaking", "paper"):
        raise C2ParameterError(
            "The live Figure 6 tau_l model is not the paper endpoint "
            f"tau_l=tau_0_pb: got {tau_l_model!r}."
        )
    tau_0_pb = _finite_positive(
        fig6_solve.PAPER_TAU_0_PB_PS / 1000.0,
        "fig6 tau_0_pb",
    )

    omega_0 = _finite_positive(fig6_solve.OMEGA_0, "fig6 omega_0")
    photon_bin = round(omega_0 / grid_spacing)
    _positive_integer(photon_bin, "fig6 photon_bin")
    _require_exact_binding(
        "photon energy/grid commensurability",
        omega_0,
        photon_bin * grid_spacing,
    )

    return Fig6ConfigurationEndpoint(
        delta0_ueV=float(fig6_solve.DELTA_0),
        tau_0_ns=float(fig6_solve.TAU_0),
        tau_0_pb_ns=tau_0_pb,
        tau_l_ns=tau_0_pb,
        c_photon_ns_inv=float(fig6_solve.C_PHOT),
        omega_0_ueV=omega_0,
        grid_spacing_ueV=grid_spacing,
        photon_bin=photon_bin,
        grid_num_bins=int(E.size),
        energy_min_factor=float(fig6_solve.E_MIN_FACTOR),
        energy_max_factor=float(fig6_solve.E_MAX_FACTOR),
        T_c_K=float(fig6_solve.T_C),
        finite_cutoff_ratio=float(fig6_solve.FINITE_CUTOFF_DELTA0_OVER_KBTC),
        gap_solve_xtol_ueV=float(fig6_solve.GAP_SOLVE_XTOL_UEV),
        tau_l_model=tau_l_model,
    )


def _require_current_fig6_endpoint(
    c2a: NativeFig6Parameters,
) -> Fig6ConfigurationEndpoint:
    """Bind the controlled C2 endpoint to the live Figure 6 configuration."""

    live = qpsim_fig6_configuration_endpoint()
    reviewed_bindings = (
        ("Delta_0", QPSIM_FIG6_DELTA0_UEV, live.delta0_ueV),
        ("tau_0", QPSIM_FIG6_TAU0_NS, live.tau_0_ns),
        ("tau_0_pb", QPSIM_FIG6_TAU0_PB_NS, live.tau_0_pb_ns),
        ("tau_l", QPSIM_FIG6_TAU_L_NS, live.tau_l_ns),
        (
            "photon coupling",
            QPSIM_FIG6_C_PHOTON_NS_INV,
            live.c_photon_ns_inv,
        ),
        (
            "finite-cutoff ratio",
            qpsim_fig6_finite_cutoff_ratio(),
            live.finite_cutoff_ratio,
        ),
        ("gap-solve xtol", FINITE_CUTOFF_XTOL_UEV, live.gap_solve_xtol_ueV),
        (
            "critical temperature",
            qpsim_fig6_critical_temperature(live.delta0_ueV),
            live.T_c_K,
        ),
        ("C2 inherited tau_0", live.tau_0_ns, c2a.tau_0_ns),
        ("C2 inherited tau_l", live.tau_l_ns, c2a.tau_l_ns),
        ("C2 inherited grid spacing", live.grid_spacing_ueV, c2a.h_ueV),
        ("C2 inherited photon bin", live.photon_bin, c2a.photon_bin),
        ("C2 inherited photon energy", live.omega_0_ueV, c2a.omega_0_ueV),
    )
    for label, expected, actual in reviewed_bindings:
        _require_exact_binding(label, expected, actual)
    return live


def _parent_value(parent: Mapping[str, object], key: str) -> float:
    if key not in parent:
        raise C2ParameterError(f"C2 parent parameters are missing {key!r}.")
    return _finite_positive(parent[key], f"parent.{key}")


def _author_solve_parameters_record(
    parameters: AuthorSolveParameters,
) -> dict[str, object]:
    """Emit the effective author-unit carrier, including every float bit."""

    values = {
        "T_c_K": parameters.T_c_K,
        "boltzmann_constant_J_per_K": (parameters.constants.boltzmann_constant_J_per_K),
        "c_photon_s_inv": parameters.c_photon_s_inv,
        "delta0_eV": parameters.delta0_eV,
        "electron_charge_C": parameters.constants.electron_charge_C,
        "gap_eV": parameters.gap_eV,
        "h_eV": parameters.h_eV,
        "max_newton_steps": parameters.max_newton_steps,
        "n_bar": parameters.n_bar,
        "photon_bin": parameters.photon_bin,
        "relative_step_threshold": parameters.relative_step_threshold,
        "tau_0_pb_s": parameters.tau_0_pb_s,
        "tau_0_s": parameters.tau_0_s,
        "tau_l_s": parameters.tau_l_s,
        "temperature_K": parameters.temperature_K,
        "thermal_gap_eV": parameters.thermal_gap_eV,
    }
    return {
        "hex": {key: value.hex() for key, value in values.items() if isinstance(value, float)},
        "values": values,
    }


def author_solve_parameters_from_native(
    native: NativeFig6Parameters,
    *,
    parent: AuthorSolveParameters,
    preserve_parent_source_bits: bool = False,
) -> AuthorSolveParameters:
    """Carry one native parameter step into the unchanged author operator.

    C2a sets ``preserve_parent_source_bits=True``: the native-unit view is
    evidence only and every effective author-space scalar remains the exact
    parent value.  C2b steps instead treat the native values as authoritative
    and use explicit ``micro-eV * 1e-6`` / ``ns * 1e-9`` carriers.

    The author operator expresses thermal energies as ``k_B/e``.  qpsim has
    no electron-charge parameter, so the carrier deliberately retains the
    authenticated parent charge and constructs
    ``k_B[J/K] = k_B[µeV/K] * 1e-6 * e[C]``.  The inverse quotient is required
    to recover the native qpsim constant bit-for-bit.
    """

    if preserve_parent_source_bits:
        return parent

    electron_charge = parent.constants.electron_charge_C
    boltzmann_constant = native.kB_ueV_per_K * UEV_TO_EV * electron_charge
    recovered_kB = boltzmann_constant / electron_charge * EV_TO_UEV
    if recovered_kB != native.kB_ueV_per_K:
        raise C2ParameterError(
            "The qpsim Boltzmann constant cannot be carried through the "
            "author k_B/e representation without changing its binary64 value."
        )
    return AuthorSolveParameters(
        gap_eV=native.gap_ueV * UEV_TO_EV,
        h_eV=native.h_ueV * UEV_TO_EV,
        temperature_K=native.temperature_K,
        T_c_K=native.T_c_K,
        tau_0_s=native.tau_0_ns * SECONDS_PER_NS,
        tau_0_pb_s=native.tau_0_pb_ns * SECONDS_PER_NS,
        tau_l_s=native.tau_l_ns * SECONDS_PER_NS,
        photon_bin=native.photon_bin,
        n_bar=native.n_bar,
        c_photon_s_inv=native.c_photon_ns_inv / SECONDS_PER_NS,
        delta0_eV=native.delta0_ueV * UEV_TO_EV,
        # This is a frozen-state parameter differential. The thermal gap is
        # an inherited parent observable, not a new solved endpoint.
        thermal_gap_eV=parent.thermal_gap_eV,
        max_newton_steps=parent.max_newton_steps,
        relative_step_threshold=parent.relative_step_threshold,
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=boltzmann_constant,
            electron_charge_C=electron_charge,
        ),
    )


def build_c2_parameter_plan(
    parent: Mapping[str, object],
) -> C2ParameterPlan:
    """Build C2a and the controlled C2b path from accepted C1/C0 scalars.

    ``parent`` is intentionally a plain mapping so a later strict score
    generator can supply the already authenticated ``C0.parameters`` object
    without this plumbing module becoming another evidence reader.
    """

    gap = EnergySpaceMapping.from_eV("gap", _parent_value(parent, "gap_eV"))
    delta0 = EnergySpaceMapping.from_eV(
        "delta0",
        _parent_value(parent, "delta0_eV"),
    )
    h = EnergySpaceMapping.from_eV("h", _parent_value(parent, "h_eV"))
    tau_0 = TimeSpaceMapping.from_seconds(
        "tau_0",
        _parent_value(parent, "tau_0_s"),
    )
    tau_0_pb = TimeSpaceMapping.from_seconds(
        "tau_0_pb",
        _parent_value(parent, "tau_0_pb_s"),
    )
    tau_l = TimeSpaceMapping.from_seconds(
        "tau_l",
        _parent_value(parent, "tau_l_s"),
    )
    c_photon = RateSpaceMapping.from_per_second(
        "c_photon",
        _parent_value(parent, "c_photon_s_inv"),
    )
    author_k_B = _parent_value(parent, "boltzmann_constant_J_per_K")
    author_e = _parent_value(parent, "electron_charge_C")
    author_kB_ueV_per_K = author_k_B / author_e * EV_TO_UEV

    photon_bin_raw = parent.get("photon_bin")
    photon_bin = _positive_integer(photon_bin_raw, "parent.photon_bin")
    c2a = NativeFig6Parameters(
        gap_ueV=gap.native_ueV,
        delta0_ueV=delta0.native_ueV,
        h_ueV=h.native_ueV,
        temperature_K=_parent_value(parent, "temperature_K"),
        T_c_K=_parent_value(parent, "T_c_K"),
        tau_0_ns=tau_0.native_ns,
        tau_0_pb_ns=tau_0_pb.native_ns,
        tau_l_ns=tau_l.native_ns,
        photon_bin=photon_bin,
        n_bar=_parent_value(parent, "n_bar"),
        c_photon_ns_inv=c_photon.native_ns_inv,
        kB_ueV_per_K=author_kB_ueV_per_K,
    )
    parent_author_parameters = AuthorSolveParameters(
        gap_eV=gap.source_eV,
        h_eV=h.source_eV,
        temperature_K=_parent_value(parent, "temperature_K"),
        T_c_K=_parent_value(parent, "T_c_K"),
        tau_0_s=tau_0.source_s,
        tau_0_pb_s=tau_0_pb.source_s,
        tau_l_s=tau_l.source_s,
        photon_bin=photon_bin,
        n_bar=_parent_value(parent, "n_bar"),
        c_photon_s_inv=c_photon.source_s_inv,
        delta0_eV=delta0.source_eV,
        thermal_gap_eV=_parent_value(parent, "thermal_gap_eV"),
        max_newton_steps=_positive_integer(
            parent.get("max_newton_steps"),
            "parent.max_newton_steps",
        ),
        relative_step_threshold=_parent_value(
            parent,
            "relative_step_threshold",
        ),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=author_k_B,
            electron_charge_C=author_e,
        ),
    )

    live_endpoint = _require_current_fig6_endpoint(c2a)
    explicit_energies = replace(
        c2a,
        gap_ueV=live_endpoint.delta0_ueV,
        delta0_ueV=live_endpoint.delta0_ueV,
    )
    modern_kB = replace(
        explicit_energies,
        kB_ueV_per_K=KB_UEV_PER_K,
    )
    literal_coupling = replace(
        modern_kB,
        c_photon_ns_inv=live_endpoint.c_photon_ns_inv,
    )
    declared_pair_time = replace(
        literal_coupling,
        tau_0_pb_ns=live_endpoint.tau_0_pb_ns,
    )
    qpsim_endpoint = replace(
        declared_pair_time,
        T_c_K=live_endpoint.T_c_K,
    )

    steps = (
        ControlledParameterStep(
            step_id="C2b1-explicit-energy-literals",
            changed_fields=("delta0_ueV", "gap_ueV"),
            parameters=explicit_energies,
            qualification=(
                "Adopt qpsim's explicit 180.0 micro-eV gap literals instead "
                "of the one-ulp-lower native view of the author eV value."
            ),
        ),
        ControlledParameterStep(
            step_id="C2b2-modern-kB",
            changed_fields=("kB_ueV_per_K",),
            parameters=modern_kB,
            qualification=(
                "Use qpsim's current Boltzmann constant; retain every material "
                "and drive value from the preceding native-energy step."
            ),
        ),
        ControlledParameterStep(
            step_id="C2b3-literal-photon-coupling",
            changed_fields=("c_photon_ns_inv",),
            parameters=literal_coupling,
            qualification=(
                "Replace the author volume-derived 1.000035... Hz coupling "
                "with qpsim's explicit 1 Hz declaration."
            ),
        ),
        ControlledParameterStep(
            step_id="C2b4-declared-pair-breaking-time",
            changed_fields=("tau_0_pb_ns",),
            parameters=declared_pair_time,
            qualification=(
                "Replace the author's derived 0.255336... ns phonon time "
                "with the explicitly declared Fischer value 0.255 ns."
            ),
        ),
        ControlledParameterStep(
            step_id="C2b5-finite-cutoff-critical-temperature",
            changed_fields=("T_c_K",),
            parameters=qpsim_endpoint,
            qualification=(
                "Derive qpsim's Figure 6 critical temperature from the "
                "explicit 180 micro-eV gap and finite-cutoff BCS calibration."
            ),
        ),
    )
    return C2ParameterPlan(
        energy_mappings=(gap, delta0, h),
        time_mappings=(tau_0, tau_0_pb, tau_l),
        photon_rate_mapping=c_photon,
        author_boltzmann_constant_J_per_K=author_k_B,
        author_electron_charge_C=author_e,
        parent_author_parameters=parent_author_parameters,
        c2a=c2a,
        c2b_steps=steps,
    )
