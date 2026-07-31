"""Isolated single-point adapter for the authenticated Fig. 6 author source.

The attachment's ``examples/Figure_6.py`` must never be imported: its
top-level ``if True`` launches all 300 dense solves.  This module extracts
only authenticated solver modules into a temporary directory and drives one
point in a subprocess. Exact modes do not modify the author equations; the
separately labelled ``runtime_compatibility`` mode applies one recorded patch
to an undefined masked ``np.divide`` result that produces NaNs on modern
NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from validation.author_source import authenticate_author_source
from validation.reproduction_ladder import describe_array
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_MANIFEST = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "author-source.json"
)
ADAPTER_DEPENDENCIES = (
    REPOSITORY_ROOT / "validation" / "author_source.py",
    REPOSITORY_ROOT / "validation" / "reproduction_ladder.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
)
_MODULE_SOURCE_BYTES_AT_IMPORT = canonical_source_bytes(Path(__file__))
_MODULE_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    _MODULE_SOURCE_BYTES_AT_IMPORT
).hexdigest()
_DEPENDENCY_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in ADAPTER_DEPENDENCIES
}
_DEPENDENCY_SHA256S_AT_IMPORT = {
    path: hashlib.sha256(content).hexdigest()
    for path, content in _DEPENDENCY_SOURCE_BYTES_AT_IMPORT.items()
}
ADAPTER_SCHEMA = "qpsim.fischer2023.fig6-author-point.v1"
AUTHOR_EQUIVALENT_NUM_ENERGY = 1620
AUTHOR_EQUIVALENT_NEWTON_STEPS = 10
AUTHOR_EQUIVALENT_THRESHOLD = 1e-7
SMOKE_NUM_ENERGY = 81
SOURCE_ROLES = (
    "array_helpers",
    "material_parameters",
    "quasiparticle_solver",
    "coupled_solver",
)
SOURCE_ROLE_FILENAMES = {
    "array_helpers": "array_operations.py",
    "material_parameters": "materials.py",
    "quasiparticle_solver": "quasiparticle_solver.py",
    "coupled_solver": "quasiparticle_and_phonon_solver.py",
}
SUBPROCESS_ISOLATION_FLAGS = ("-I", "-B")
ADAPTER_SOURCE_HASH_KIND = "canonical_sha256_import_time_disk_snapshot"
ADAPTER_SOURCE_EXECUTION_BINDING = (
    "The adapter and dependency digests identify immutable source snapshots "
    "read at module import. The live files are required to match those bytes "
    "before and after evidence collection. This guards source drift but does "
    "not claim an independent interpreter-bytecode attestation."
)
_TAU_PB_UNDEFINED_FRAGMENT = (
    b"        tau_PB=np.divide(1.,self.const_phon*np.sum(c_plus,axis=1),"
    b"where=(np.arange(1,self.qp.N_dis)>=2*self.qp.a_Delta))"
)
_ENTRYPOINT_CONTRACT_FRAGMENTS = (
    b"e=1.60217662e-19",
    b"h_bar=6.626070e-34/(2*np.pi)",
    b"k_B=1.38064852e-23",
    b"n_list=np.geomspace(10**4,10**8,100)",
    b"T_list=[0.1,0.15,0.2]",
    b"param['discretization steps']=9*180",
    b"param['E_max']=10*param[\"zero temperature gap\"]",
    b"param['photon energy']=20*10**-6",
    b'param["size superconducting volume"]= 1000*10**-18*1.544',
    b"param['gap']=180*1e-6",
    b"param['tau_l']=255e-12",
    b"param['alpha']=1.",
    b"sim.newton(10,threshold=1e-7)",
    b"gaps[i_n,i_T]=sim.qp.delta_not_consistent(",
    b"gaps_thermal[i_n,i_T]=sim.qp.delta_not_consistent(",
    b"T_star_list=(105./64.",
    b"(gaps[:,2]-gaps_thermal[:,2])/(sim.qp.Delta-gaps_thermal[:,2])",
)


@dataclass(frozen=True)
class AuthorPointConfig:
    """Inputs for one isolated author-source point."""

    temperature_K: float
    target_t_star_over_delta: float | None = None
    sweep_index: int | None = None
    num_energy_cells: int = AUTHOR_EQUIVALENT_NUM_ENERGY
    max_newton_steps: int = AUTHOR_EQUIVALENT_NEWTON_STEPS
    relative_step_threshold: float = AUTHOR_EQUIVALENT_THRESHOLD
    rng_seed: int | None = None
    mode: str = "author_equivalent"

    def validate(self) -> None:
        if (
            isinstance(self.temperature_K, bool)
            or not isinstance(self.temperature_K, Real)
            or not math.isfinite(self.temperature_K)
            or self.temperature_K <= 0.0
        ):
            raise ValueError("temperature_K must be finite and positive.")
        if (self.target_t_star_over_delta is None) == (self.sweep_index is None):
            raise ValueError(
                "Select exactly one input: target_t_star_over_delta or sweep_index."
            )
        if self.target_t_star_over_delta is not None and (
            isinstance(self.target_t_star_over_delta, bool)
            or not isinstance(self.target_t_star_over_delta, Real)
            or not math.isfinite(self.target_t_star_over_delta)
            or self.target_t_star_over_delta <= 0.0
        ):
            raise ValueError("target_t_star_over_delta must be finite and positive.")
        if self.sweep_index is not None and (
            not isinstance(self.sweep_index, int)
            or isinstance(self.sweep_index, bool)
            or not 0 <= self.sweep_index < 100
        ):
            raise ValueError("sweep_index must be an integer in [0, 99].")
        if (
            not isinstance(self.num_energy_cells, int)
            or isinstance(self.num_energy_cells, bool)
            or self.num_energy_cells < 2
        ):
            raise ValueError("num_energy_cells must be an integer >= 2.")
        if (
            not isinstance(self.max_newton_steps, int)
            or isinstance(self.max_newton_steps, bool)
            or self.max_newton_steps < 1
        ):
            raise ValueError("max_newton_steps must be a positive integer.")
        if (
            isinstance(self.relative_step_threshold, bool)
            or not isinstance(self.relative_step_threshold, Real)
            or not math.isfinite(self.relative_step_threshold)
            or self.relative_step_threshold <= 0.0
        ):
            raise ValueError("relative_step_threshold must be finite and positive.")
        if self.rng_seed is not None and (
            not isinstance(self.rng_seed, int) or isinstance(self.rng_seed, bool)
        ):
            raise ValueError("rng_seed must be null or an integer.")
        if self.mode not in {
            "author_equivalent",
            "compatibility_smoke",
            "runtime_compatibility",
        }:
            raise ValueError(
                "mode must be 'author_equivalent', 'compatibility_smoke', "
                "or 'runtime_compatibility'."
            )
        if self.mode in {"author_equivalent", "runtime_compatibility"} and (
            self.num_energy_cells != AUTHOR_EQUIVALENT_NUM_ENERGY
            or self.max_newton_steps != AUTHOR_EQUIVALENT_NEWTON_STEPS
            or self.relative_step_threshold != AUTHOR_EQUIVALENT_THRESHOLD
            or self.rng_seed is not None
            or self.sweep_index is None
            or self.target_t_star_over_delta is not None
            or self.temperature_K not in {0.1, 0.15, 0.2}
        ):
            raise ValueError(
                f"{self.mode} mode requires N=1620, ten Newton steps, "
                "threshold=1e-7, the author's unseeded RNG semantics, one "
                "original sweep index, and an original bath temperature."
            )


@dataclass(frozen=True)
class AuthorPointResult:
    """Normalized output captured from one isolated author-source point."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    stdout: str
    stderr: str

    def provenance_record(self) -> dict[str, Any]:
        """Return scalar metadata plus exact native-array descriptors."""

        return {
            "arrays": {
                name: {
                    "dtype": descriptor.dtype,
                    "npy_sha256": descriptor.npy_sha256,
                    "shape": list(descriptor.shape),
                }
                for name, value in sorted(self.arrays.items())
                for descriptor in (describe_array(value),)
            },
            "metadata": self.metadata,
            "stderr_sha256": hashlib.sha256(self.stderr.encode("utf-8")).hexdigest(),
            "stdout_sha256": hashlib.sha256(self.stdout.encode("utf-8")).hexdigest(),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SUBPROCESS_WRAPPER = r'''
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

source_dir = Path(sys.argv[1]).resolve()
config = json.loads(sys.argv[2])
output_dir = Path(sys.argv[3]).resolve()
source_contract = json.loads(sys.argv[4])

source_snapshots = {}
source_origins = {}
for module_name, descriptor in source_contract.items():
    if (
        not isinstance(module_name, str)
        or not module_name.isidentifier()
        or not isinstance(descriptor, dict)
        or set(descriptor) != {"filename", "sha256"}
    ):
        raise RuntimeError("Invalid authenticated-source execution contract.")
    filename = descriptor["filename"]
    expected_sha256 = descriptor["sha256"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise RuntimeError("Invalid authenticated-source module descriptor.")
    content = (source_dir / filename).read_bytes()
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Authenticated source {filename!r} changed before child snapshot."
        )
    source_snapshots[module_name] = content
    source_origins[module_name] = (
        f"authenticated-author-snapshot://{filename}@{actual_sha256}"
    )


class SnapshotSourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import authenticated author modules only from immutable byte snapshots."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in source_snapshots:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=source_origins[fullname],
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__spec__.name
        origin = source_origins[name]
        code = compile(source_snapshots[name], origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)


sys.meta_path.insert(0, SnapshotSourceLoader())

from materials import Material
from quasiparticle_and_phonon_solver import SimulationPhononsQuasiparticles

e = 1.60217662e-19
h_bar = 6.626070e-34 / (2.0 * np.pi)
k_B = 1.38064852e-23

if config["rng_seed"] is not None:
    np.random.seed(config["rng_seed"])

material = Material(material="Al")
param = material.return_paras()
param["path"] = str(output_dir)
param["discretization steps"] = config["num_energy_cells"]
param["temperature phonons"] = config["temperature_K"]
param["E_max"] = 10.0 * param["zero temperature gap"]
param["photon energy"] = 20.0e-6
param["size superconducting volume"] = 1000.0e-18 * 1.544
param["gap"] = 180.0e-6
param["tau_l"] = 255.0e-12
param["name"] = "qpsim_author_adapter_point"
param["alpha"] = 1.0

h = (param["E_max"] - param["gap"]) / float(config["num_energy_cells"])
delta_grid = round(param["gap"] / h) * h
dos = param["fermi density of states"] * param["size superconducting volume"]
c_photon = (
    param["photon energy"]
    * e
    * param["alpha"]
    / (2.0 * np.pi * dos * h_bar * param["gap"])
)
denominator = (
    (105.0 / 64.0)
    * (param["critical temperature"] * k_B / e) ** 3
    * c_photon
    * param["tau_0"]
    * param["photon energy"] ** 2
    * param["gap"]
)
if config["sweep_index"] is not None:
    n_bar = float(np.geomspace(10.0**4, 10.0**8, 100)[config["sweep_index"]])
else:
    n_bar = (
        config["target_t_star_over_delta"] * delta_grid
    ) ** 6 / denominator
param["average photon number"] = n_bar

started = time.perf_counter()
simulation = SimulationPhononsQuasiparticles(param)
constructor_seconds = time.perf_counter() - started
newton_started = time.perf_counter()
simulation.newton(
    config["max_newton_steps"],
    threshold=config["relative_step_threshold"],
)
newton_seconds = time.perf_counter() - newton_started

N = simulation.qp.N_dis
initial_state = np.asarray(simulation.newton_list[0])
final_state = np.asarray(simulation.newton_list[-1])
f = final_state[:N]
n_ph = final_state[N:]
residual = np.asarray(simulation.X(f, n_ph))
qp_photon_residual = np.asarray(simulation.qp.st_phot(f))
qp_phonon_residual = np.asarray(simulation.qp.st_phon(f, n_ph))
phonon_residual = np.asarray(simulation.N(f, n_ph))
gap_driven = float(simulation.qp.delta_not_consistent(f))
thermal_f = simulation.qp.fermi_distribution(np.arange(0, N))
gap_thermal = float(simulation.qp.delta_not_consistent(thermal_f))
observable = float(
    (gap_driven - gap_thermal) / (simulation.qp.Delta - gap_thermal)
)
if len(simulation.newton_list) >= 2:
    previous_f = np.asarray(simulation.newton_list[-2][:N])
    last_relative_qp_step = float(
        np.linalg.norm(f - previous_f) / np.linalg.norm(f)
    )
else:
    last_relative_qp_step = math.inf

P_abs = float(simulation.absorbed_power(f))
P_bath = float(simulation.cooling_power(n_ph))
P_qp = float(simulation.quasiparticle_phonon_power_qp(f, n_ph))
P_phon = float(simulation.quasiparticle_phonon_power_phon(f, n_ph))
if P_abs == 0.0:
    max_power_mismatch = math.inf
else:
    max_power_mismatch = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        P_abs - P_bath,
                        P_abs + P_qp,
                        P_bath + P_qp,
                        P_phon,
                    ]
                )
                / P_abs
            )
        )
    )

E_qp = simulation.qp.Delta + np.arange(N) * simulation.qp.h
omega_phonon = np.arange(1, N) * simulation.qp.h
np.savez(
    output_dir / "captured_arrays.npz",
    E_qp=E_qp,
    f_final=f,
    initial_state=initial_state,
    n_phonon_final=n_ph,
    omega_phonon=omega_phonon,
    phonon_residual_author_s_inv=phonon_residual,
    qp_phonon_residual_author_s_inv=qp_phonon_residual,
    qp_photon_residual_author_s_inv=qp_photon_residual,
    residual=residual,
)

def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

def finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None

all_finite = bool(
    np.all(np.isfinite(initial_state))
    and np.all(np.isfinite(final_state))
    and np.all(np.isfinite(residual))
)
metadata = {
    "_executed_module_sha256s": {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(source_snapshots.items())
    },
    "actual_t_star_over_delta": float(simulation.qp.T_star / simulation.qp.Delta),
    "all_finite": all_finite,
    "array_origins": {
        "E_qp": "adapter-derived from author Delta + integer index * h",
        "f_final": "author final Newton state view",
        "initial_state": "author initial Newton state",
        "n_phonon_final": "author final Newton state view",
        "omega_phonon": "adapter-derived from author integer index * h",
        "phonon_residual_author_s_inv": (
            "author SimulationPhononsQuasiparticles.N evaluation"
        ),
        "qp_phonon_residual_author_s_inv": (
            "author SimulationQuasiparticles.st_phon evaluation"
        ),
        "qp_photon_residual_author_s_inv": (
            "author SimulationQuasiparticles.st_phot evaluation"
        ),
        "residual": "author SimulationPhononsQuasiparticles.X evaluation",
    },
    "c_photon_per_second": float(c_photon),
    "computed_n_bar": float(n_bar),
    "constructor_seconds": float(constructor_seconds),
    "f_max": finite_or_none(np.max(f)),
    "f_min": finite_or_none(np.min(f)),
    "gap_driven_eV": finite_or_none(gap_driven),
    "gap_thermal_eV": finite_or_none(gap_thermal),
    "h_eV": float(simulation.qp.h),
    "author_numerical_constants": {
        "boltzmann_constant_J_per_K": float(k_B),
        "electron_charge_C": float(e),
        "reduced_planck_constant_J_s": float(h_bar),
    },
    "material_parameters": {
        "T_c_K": float(simulation.qp.T_c),
        "debye_energy_eV": float(simulation.omega_D),
        "fermi_density_of_states_per_eV_m3": float(simulation.qp.fermi_dos),
        "ion_density_per_m3": float(simulation.ion_dens),
        "tau_0_s": float(simulation.qp.tau_0),
        "tau_0_pb_s": float(simulation.t_phon),
        "tau_l_s": float(simulation.tau_l),
        "zero_temperature_gap_eV": float(simulation.qp.Delta_0),
    },
    "input": config,
    "last_relative_qp_step": finite_or_none(last_relative_qp_step),
    "max_power_mismatch": finite_or_none(max_power_mismatch),
    "n_ph_max": finite_or_none(np.max(n_ph)),
    "n_ph_min": finite_or_none(np.min(n_ph)),
    "newton_iterations": len(simulation.newton_list) - 1,
    "newton_seconds": float(newton_seconds),
    "normalized_numerical_observable": finite_or_none(observable),
    "photon_bin": int(simulation.qp.x_phot),
    "rounded_gap_eV": float(simulation.qp.Delta),
    "runtime": {
        "matplotlib": package_version("matplotlib"),
        "numba": package_version("numba"),
        "numpy": package_version("numpy"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scipy": package_version("scipy"),
    },
    "schema": "qpsim.fischer2023.fig6-author-point.v1",
    "scientific_status": "finite_state" if all_finite else "nonfinite_author_state",
    "solved_photon_energy_eV": float(simulation.qp.x_phot * simulation.qp.h),
}
(output_dir / "captured_metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8",
)
'''
SUBPROCESS_WRAPPER_SHA256 = hashlib.sha256(
    _SUBPROCESS_WRAPPER.encode("utf-8")
).hexdigest()


def _verify_entrypoint_contract(
    content: bytes,
    *,
    member_path: str,
) -> dict[str, object]:
    """Bind the manual single-point driver to authenticated Figure_6.py."""

    missing = [
        fragment.decode("ascii", errors="replace")
        for fragment in _ENTRYPOINT_CONTRACT_FRAGMENTS
        if fragment not in content
    ]
    if missing:
        raise ValueError(
            "The authenticated Figure_6.py no longer has the exact driver "
            f"contract expected by the adapter: {missing!r}."
        )
    contract_bytes = b"\n".join(_ENTRYPOINT_CONTRACT_FRAGMENTS)
    return {
        "entry_point_path": member_path,
        "entry_point_sha256": hashlib.sha256(content).hexdigest(),
        "fragment_count": len(_ENTRYPOINT_CONTRACT_FRAGMENTS),
        "fragments_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "selection": (
            "temperature from T_list and n_bar from the original 100-point "
            "np.geomspace by integer sweep_index"
        ),
    }


def _runtime_compatibility_patch(
    content: bytes,
    *,
    member_path: str,
) -> tuple[bytes, dict[str, str]]:
    """Initialize the author's masked ``tau_PB`` array deterministically.

    The original ``np.divide(..., where=mask)`` omits ``out``. NumPy leaves
    the false-mask entries uninitialized; those entries are subsequently
    multiplied by zero, so a stray NaN contaminates the entire solve. Infinity
    is the intended below-threshold lifetime and makes the later fraction
    exactly zero. The exact source mode deliberately retains the original
    behavior; this patch exists only in the named compatibility stage.
    """

    if content.count(_TAU_PB_UNDEFINED_FRAGMENT) != 1:
        raise ValueError(
            "The authenticated coupled solver no longer contains exactly one "
            "expected undefined tau_PB expression."
        )
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    replacement = (
        b"        tau_PB=np.full(self.qp.N_dis-1,np.inf)"
        + newline
        + b"        np.divide(1.,self.const_phon*np.sum(c_plus,axis=1),"
        b"where=(np.arange(1,self.qp.N_dis)>=2*self.qp.a_Delta),out=tau_PB)"
    )
    patched = content.replace(_TAU_PB_UNDEFINED_FRAGMENT, replacement)
    return patched, {
        "executed_member_sha256": hashlib.sha256(patched).hexdigest(),
        "member_path": member_path,
        "original_fragment_sha256": hashlib.sha256(
            _TAU_PB_UNDEFINED_FRAGMENT
        ).hexdigest(),
        "replacement_fragment_sha256": hashlib.sha256(replacement).hexdigest(),
        "transform_id": "initialize-subthreshold-tau-pb-to-infinity-v1",
    }


def _write_authenticated_sources(
    archive_path: Path,
    destination: Path,
    *,
    runtime_compatibility: bool,
) -> tuple[
    str,
    str,
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, str]],
    list[dict[str, str]],
    dict[str, object],
]:
    authenticated = authenticate_author_source(AUTHOR_MANIFEST, archive_path)
    by_role = {member.role: member for member in authenticated.members}
    missing = {*SOURCE_ROLES, "entry_point"} - set(by_role)
    if missing:
        raise ValueError(f"Author archive lacks required solver roles: {sorted(missing)!r}.")
    entry_point = by_role["entry_point"]
    driver_contract = _verify_entrypoint_contract(
        entry_point.content,
        member_path=entry_point.path,
    )
    original_member_hashes: dict[str, str] = {}
    executed_member_hashes: dict[str, str] = {}
    execution_contract: dict[str, dict[str, str]] = {}
    transformations: list[dict[str, str]] = []
    original_member_hashes[entry_point.path] = entry_point.sha256
    destination_names: set[str] = set()
    for role in SOURCE_ROLES:
        member = by_role[role]
        filename = Path(member.path).name
        expected_filename = SOURCE_ROLE_FILENAMES[role]
        if filename != expected_filename:
            raise ValueError(
                f"Author source role {role!r} must map to "
                f"{expected_filename!r}; got {filename!r}."
            )
        if filename in destination_names:
            raise ValueError(
                f"Author source roles collide at destination {filename!r}."
            )
        destination_names.add(filename)
        content = member.content
        if runtime_compatibility and role == "coupled_solver":
            content, transformation = _runtime_compatibility_patch(
                content,
                member_path=member.path,
            )
            transformation["original_member_sha256"] = member.sha256
            transformations.append(transformation)
        executed_sha256 = hashlib.sha256(content).hexdigest()
        (destination / filename).write_bytes(content)
        original_member_hashes[member.path] = member.sha256
        executed_member_hashes[member.path] = executed_sha256
        execution_contract[filename.removesuffix(".py")] = {
            "filename": filename,
            "sha256": executed_sha256,
        }
    return (
        authenticated.source_id,
        authenticated.archive_sha256,
        original_member_hashes,
        executed_member_hashes,
        execution_contract,
        transformations,
        driver_contract,
    )


def _assert_adapter_source_snapshot() -> None:
    """Fail if adapter/dependency files differ from their import snapshots."""

    current_module = canonical_source_bytes(Path(__file__))
    current_dependencies = {
        path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
        for path in ADAPTER_DEPENDENCIES
    }
    if (
        current_module != _MODULE_SOURCE_BYTES_AT_IMPORT
        or current_dependencies != _DEPENDENCY_SOURCE_BYTES_AT_IMPORT
    ):
        raise RuntimeError(
            "Adapter or provenance dependency changed after this module was "
            "imported; launch a fresh Python process before collecting evidence."
        )


def run_author_point(
    config: AuthorPointConfig,
    *,
    archive_path: Path | None = None,
    timeout_seconds: float = 3600.0,
    scratch_root: Path | None = None,
) -> AuthorPointResult:
    """Run one author-source point in an isolated subprocess."""

    config.validate()
    # Bind the adapter bytes before the long-running child starts. Reading
    # this file only after the solve would let a concurrent edit label old
    # in-memory code with a newer on-disk hash.
    _assert_adapter_source_snapshot()
    adapter_sha256 = _MODULE_SOURCE_SHA256_AT_IMPORT
    adapter_dependency_sha256s = dict(_DEPENDENCY_SHA256S_AT_IMPORT)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, Real)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be finite and positive.")
    if archive_path is None:
        configured = os.environ.get("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE")
        if not configured:
            raise ValueError(
                "Set QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE or pass archive_path."
            )
        archive_path = Path(configured)
    if scratch_root is not None:
        scratch_root = scratch_root.resolve()
        if not scratch_root.is_dir():
            raise ValueError("scratch_root must be an existing directory.")

    with tempfile.TemporaryDirectory(
        prefix="qpsim-f23-fig6-author-",
        dir=scratch_root,
    ) as temporary:
        temp_root = Path(temporary)
        source_dir = temp_root / "source"
        output_dir = temp_root / "output"
        cache_dir = temp_root / "cache"
        source_dir.mkdir()
        output_dir.mkdir()
        (cache_dir / "mpl").mkdir(parents=True)
        (cache_dir / "numba").mkdir()
        (
            source_id,
            archive_sha,
            member_hashes,
            executed_member_hashes,
            execution_contract,
            source_transformations,
            author_driver_contract,
        ) = _write_authenticated_sources(
            archive_path.resolve(),
            source_dir,
            runtime_compatibility=config.mode == "runtime_compatibility",
        )
        config_payload = {
            "max_newton_steps": config.max_newton_steps,
            "mode": config.mode,
            "num_energy_cells": config.num_energy_cells,
            "relative_step_threshold": config.relative_step_threshold,
            "rng_seed": config.rng_seed,
            "sweep_index": config.sweep_index,
            "target_t_star_over_delta": config.target_t_star_over_delta,
            "temperature_K": config.temperature_K,
        }
        config_json = json.dumps(config_payload, sort_keys=True, allow_nan=False)
        execution_contract_json = json.dumps(
            execution_contract,
            sort_keys=True,
            allow_nan=False,
        )
        # The attachment is trusted scientific source, not sandboxed code.
        # Still, do not hand it API keys or arbitrary user configuration via
        # the environment. Retain only Windows/Python process essentials.
        environment = {
            key: os.environ[key]
            for key in (
                "APPDATA",
                "COMSPEC",
                "HOME",
                "LOCALAPPDATA",
                "PATH",
                "PATHEXT",
                "PROGRAMDATA",
                "SYSTEMROOT",
                "TEMP",
                "TMP",
                "USERPROFILE",
                "WINDIR",
            )
            if key in os.environ
        }
        environment.update(
            {
                "MPLCONFIGDIR": str(cache_dir / "mpl"),
                "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MPLBACKEND": "Agg",
            }
        )
        completed = subprocess.run(
            [
                sys.executable,
                *SUBPROCESS_ISOLATION_FLAGS,
                "-c",
                _SUBPROCESS_WRAPPER,
                str(source_dir),
                config_json,
                str(output_dir),
                execution_contract_json,
            ],
            cwd=temp_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Author-source point failed.\n"
                f"stdout:\n{completed.stdout[-4000:]}\n"
                f"stderr:\n{completed.stderr[-4000:]}"
            )
        metadata_bytes = (output_dir / "captured_metadata.json").read_bytes()
        arrays_bytes = (output_dir / "captured_arrays.npz").read_bytes()
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        with np.load(io.BytesIO(arrays_bytes), allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name], copy=True) for name in payload.files}

    child_module_sha256s = metadata.pop("_executed_module_sha256s", None)
    expected_module_sha256s = {
        module_name: descriptor["sha256"]
        for module_name, descriptor in execution_contract.items()
    }
    if child_module_sha256s != expected_module_sha256s:
        raise RuntimeError(
            "Child-reported executed author-module snapshots do not match the "
            "authenticated execution contract."
        )
    if metadata.get("input") != config_payload:
        raise RuntimeError(
            "Child-reported scientific input differs from the parent execution "
            "contract."
        )
    metadata["adapter"] = {
        "dependency_sha256s": adapter_dependency_sha256s,
        "execution_binding": ADAPTER_SOURCE_EXECUTION_BINDING,
        "hash_kind": ADAPTER_SOURCE_HASH_KIND,
        "path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": adapter_sha256,
        "subprocess_wrapper_sha256": SUBPROCESS_WRAPPER_SHA256,
    }
    metadata["author_source"] = {
        "archive_sha256": archive_sha,
        "member_sha256s": member_hashes,
        "source_id": source_id,
    }
    metadata["executed_source"] = {
        "loading": "immutable_child_byte_snapshots_v1",
        "member_sha256s": executed_member_hashes,
        "module_sha256s": child_module_sha256s,
        "transformations": source_transformations,
    }
    metadata["author_driver_contract"] = author_driver_contract
    metadata["execution_security"] = (
        "Trusted external Python source executed with normal user filesystem "
        "and network permissions in a temporary working directory. Python "
        "isolated mode (-I) disables PYTHON* environment configuration and "
        "the user site; -B disables bytecode-cache writes. Authenticated "
        "author modules are compiled from child-retained byte snapshots, and "
        "selected non-Python environment variables are scrubbed. This is not "
        "an OS sandbox."
    )
    metadata["subprocess_isolation"] = {
        "flags": list(SUBPROCESS_ISOLATION_FLAGS),
        "isolated_python": True,
        "os_sandbox": False,
        "user_site_enabled": False,
    }
    if config.mode == "compatibility_smoke":
        metadata["qualification"] = (
            "compatibility smoke only; reduced grid and/or seeded RNG; "
            "author source bytes unchanged"
        )
    elif config.mode == "runtime_compatibility":
        metadata["qualification"] = (
            "author-resolution single point with one recorded runtime "
            "compatibility patch: below-threshold tau_PB is initialized to "
            "infinity before the author's masked divide; all other source "
            "and the unseeded RNG semantics are unchanged"
        )
    else:
        metadata["qualification"] = (
            "author-resolution single-point adapter; the source's unseeded RNG "
            "means exact replay requires retaining the captured initial state; "
            "author source bytes unchanged"
        )
    _assert_adapter_source_snapshot()
    return AuthorPointResult(
        metadata=metadata,
        arrays=arrays,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def write_author_point_bundle(result: AuthorPointResult, output_dir: Path) -> Path:
    """Persist native arrays and a provenance manifest without coercion."""

    resolved = output_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, object]] = {}
    origins = result.metadata.get("array_origins", {})
    if not isinstance(origins, dict):
        raise ValueError("metadata.array_origins must be a mapping when supplied.")
    for name, value in sorted(result.arrays.items()):
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or Path(name).name != name
        ):
            raise ValueError(
                "Author bundle array names must be safe Python identifiers; "
                f"got {name!r}."
            )
        path = resolved / f"{name}.npy"
        with path.open("wb") as handle:
            np.lib.format.write_array(handle, np.asarray(value), version=(3, 0), allow_pickle=False)
        origin = origins.get(name, "unspecified array origin")
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"metadata.array_origins[{name!r}] must be a nonempty string.")
        files[path.name] = {
            "role": f"array:{name}:{origin}",
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    for name, text in (("stdout.txt", result.stdout), ("stderr.txt", result.stderr)):
        path = resolved / name
        path.write_text(text, encoding="utf-8", newline="\n")
        files[name] = {
            "role": "author_process_log",
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = result.provenance_record()
    manifest["files"] = files
    manifest_path = resolved / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--t-star-over-delta", type=float)
    selection.add_argument("--sweep-index", type=int)
    parser.add_argument(
        "--mode",
        choices=(
            "author_equivalent",
            "compatibility_smoke",
            "runtime_compatibility",
        ),
        default="author_equivalent",
    )
    parser.add_argument("--num-energy-cells", type=int, default=AUTHOR_EQUIVALENT_NUM_ENERGY)
    parser.add_argument("--max-newton-steps", type=int, default=AUTHOR_EQUIVALENT_NEWTON_STEPS)
    parser.add_argument(
        "--relative-step-threshold",
        type=float,
        default=AUTHOR_EQUIVALENT_THRESHOLD,
    )
    parser.add_argument("--rng-seed", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    """CLI entry point for an explicit single-point author-source run."""

    args = _parse_args()
    config = AuthorPointConfig(
        temperature_K=args.temperature,
        target_t_star_over_delta=args.t_star_over_delta,
        sweep_index=args.sweep_index,
        num_energy_cells=args.num_energy_cells,
        max_newton_steps=args.max_newton_steps,
        relative_step_threshold=args.relative_step_threshold,
        rng_seed=args.rng_seed,
        mode=args.mode,
    )
    result = run_author_point(
        config,
        archive_path=args.archive,
        timeout_seconds=args.timeout_seconds,
    )
    print(write_author_point_bundle(result, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
