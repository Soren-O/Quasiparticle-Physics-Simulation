from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() not in ("false", "0", "no", "")
    return bool(val)


from .models import (
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    ExternalGenerationSpec,
    GeometryData,
    InitialConditionSpec,
    SetupData,
    SimulationParameters,
    TestGeometryGroupData,
    SimulationResultData,
    TestCaseResultData,
    TestSuiteData,
    utc_now_iso,
)
from .initial_conditions import canonicalize_initial_condition
from .paths import SETUPS_DIR, SIMULATIONS_DIR, TEST_CASES_DIR, ensure_data_dirs

TEST_SUITE_FORMAT_VERSION = 3


def slugify_name(name: str, fallback: str = "item") -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return value or fallback


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_data_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_to_jsonable(frame: np.ndarray) -> list[list[float | None]]:
    payload: list[list[float | None]] = []
    for row in frame:
        payload.append([None if np.isnan(v) else float(v) for v in row])
    return payload


def frame_from_jsonable(frame: list[list[float | None]]) -> np.ndarray:
    return np.array([[np.nan if v is None else float(v) for v in row] for row in frame], dtype=float)


def serialize_setup(setup: SetupData) -> dict[str, Any]:
    return asdict(setup)


def _deserialize_external_generation(raw: dict[str, Any] | None) -> ExternalGenerationSpec:
    if raw is None:
        return ExternalGenerationSpec()
    return ExternalGenerationSpec(
        mode=str(raw.get("mode", "none")),
        rate=float(raw.get("rate", 0.0)),
        pulse_start=float(raw.get("pulse_start", 0.0)),
        pulse_duration=float(raw.get("pulse_duration", 10.0)),
        pulse_rate=float(raw.get("pulse_rate", 0.0)),
        custom_body=str(raw.get("custom_body", "return 0.0")),
        custom_params=dict(raw.get("custom_params", {})),
    )


def deserialize_setup(payload: dict[str, Any]) -> SetupData:
    geometry_raw = payload["geometry"]
    edges: list[EdgeSegment] = []
    for edge in geometry_raw["edges"]:
        faces = [BoundaryFace(**face) for face in edge["faces"]]
        edges.append(
            EdgeSegment(
                edge_id=edge["edge_id"],
                x0=edge["x0"],
                y0=edge["y0"],
                x1=edge["x1"],
                y1=edge["y1"],
                normal=edge["normal"],
                faces=faces,
            )
        )

    geometry = GeometryData(
        name=geometry_raw["name"],
        source_path=geometry_raw["source_path"],
        layer=int(geometry_raw["layer"]),
        mesh_size=float(geometry_raw["mesh_size"]),
        mask=geometry_raw["mask"],
        edges=edges,
        bounds=geometry_raw.get("bounds"),
    )

    bc_map = {
        edge_id: BoundaryCondition(
            kind=bc_raw["kind"],
            value=bc_raw.get("value"),
            aux_value=bc_raw.get("aux_value"),
        )
        for edge_id, bc_raw in payload.get("boundary_conditions", {}).items()
    }

    params_raw = payload["parameters"]
    params = SimulationParameters(
        diffusion_coefficient=float(params_raw["diffusion_coefficient"]),
        dt=float(params_raw["dt"]),
        total_time=float(params_raw["total_time"]),
        mesh_size=float(params_raw["mesh_size"]),
        store_every=int(params_raw.get("store_every", 1)),
        energy_gap=float(params_raw.get("energy_gap", 0.0)),
        energy_min_factor=float(params_raw.get("energy_min_factor", 1.0)),
        energy_max_factor=float(params_raw.get("energy_max_factor", 10.0)),
        num_energy_bins=int(params_raw.get("num_energy_bins", 50)),
        dynes_gamma=float(params_raw.get("dynes_gamma", 0.0)),
        gap_expression=str(params_raw.get("gap_expression", "")),
        collision_solver=str(params_raw.get("collision_solver") or "fischer_catelani_local"),
        enable_diffusion=_to_bool(params_raw.get("enable_diffusion", True)),
        enable_recombination=_to_bool(params_raw.get("enable_recombination", False)),
        enable_scattering=_to_bool(params_raw.get("enable_scattering", False)),
        tau_0=float(params_raw.get("tau_0", 440.0)),
        tau_s=(
            float(params_raw["tau_s"])
            if params_raw.get("tau_s") is not None
            else None
        ),
        tau_r=(
            float(params_raw["tau_r"])
            if params_raw.get("tau_r") is not None
            else None
        ),
        T_c=float(params_raw.get("T_c", 1.2)),
        bath_temperature=float(params_raw.get("bath_temperature", 0.1)),
        export_phonon_history=_to_bool(params_raw.get("export_phonon_history", False)),
        external_generation=_deserialize_external_generation(params_raw.get("external_generation")),
    )

    ic_raw = payload.get("initial_condition", {})
    initial_condition_raw = InitialConditionSpec(
        kind=ic_raw.get("kind", "gaussian"),
        params=ic_raw.get("params", {}),
        custom_body=ic_raw.get(
            "custom_body",
            "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)",
        ),
        custom_params=ic_raw.get("custom_params", {}),
        spatial_kind=ic_raw.get("spatial_kind", ""),
        spatial_params=ic_raw.get("spatial_params", {}),
        spatial_custom_body=ic_raw.get(
            "spatial_custom_body",
            "",
        ),
        spatial_custom_params=ic_raw.get("spatial_custom_params", {}),
        energy_kind=ic_raw.get("energy_kind", ""),
        energy_params=ic_raw.get("energy_params", {}),
        energy_custom_body=ic_raw.get("energy_custom_body", ""),
        energy_custom_params=ic_raw.get("energy_custom_params", {}),
        qp_full_custom_enabled=_to_bool(ic_raw.get("qp_full_custom_enabled", False)),
        qp_full_custom_body=ic_raw.get("qp_full_custom_body", ""),
        qp_full_custom_params=ic_raw.get("qp_full_custom_params", {}),
        phonon_spatial_kind=ic_raw.get("phonon_spatial_kind", ""),
        phonon_spatial_params=ic_raw.get("phonon_spatial_params", {}),
        phonon_spatial_custom_body=ic_raw.get("phonon_spatial_custom_body", ""),
        phonon_spatial_custom_params=ic_raw.get("phonon_spatial_custom_params", {}),
        phonon_energy_kind=ic_raw.get("phonon_energy_kind", ""),
        phonon_energy_params=ic_raw.get("phonon_energy_params", {}),
        phonon_energy_custom_body=ic_raw.get("phonon_energy_custom_body", ""),
        phonon_energy_custom_params=ic_raw.get("phonon_energy_custom_params", {}),
        phonon_full_custom_enabled=_to_bool(ic_raw.get("phonon_full_custom_enabled", False)),
        phonon_full_custom_body=ic_raw.get("phonon_full_custom_body", ""),
        phonon_full_custom_params=ic_raw.get("phonon_full_custom_params", {}),
    )
    initial_condition = canonicalize_initial_condition(initial_condition_raw)

    return SetupData(
        setup_id=payload["setup_id"],
        name=payload["name"],
        created_at=payload.get("created_at", utc_now_iso()),
        geometry=geometry,
        boundary_conditions=bc_map,
        parameters=params,
        initial_condition=initial_condition,
    )


def precompute_npz_path(setup_path: Path) -> Path:
    """Return the .precompute.npz sidecar path for a setup JSON file."""
    return setup_path.with_suffix(".precompute.npz")


def save_precomputed(setup_path: Path, arrays: dict) -> Path:
    """Save precomputed arrays to .npz sidecar file."""
    npz_path = precompute_npz_path(setup_path)
    np.savez(str(npz_path), **arrays)
    return npz_path


def load_precomputed(setup_path: Path) -> dict:
    """Load precomputed arrays from .npz sidecar file."""
    npz_path = precompute_npz_path(setup_path)
    data = np.load(str(npz_path), allow_pickle=False)
    return dict(data)


def precomputed_exists(setup_path: Path) -> bool:
    """Check if a precomputed .npz sidecar file exists."""
    return precompute_npz_path(setup_path).exists()


def save_setup(setup: SetupData, path: Path | None = None) -> Path:
    if path is None:
        slug = slugify_name(setup.name, "setup")
        filename = f"{slug}_{setup.setup_id}.json"
        path = SETUPS_DIR / filename
    return _write_json(path, serialize_setup(setup))


def load_setup(path: str | Path) -> SetupData:
    return deserialize_setup(_read_json(Path(path)))


def create_setup_id() -> str:
    return uuid.uuid4().hex[:12]


def serialize_simulation(result: SimulationResultData) -> dict[str, Any]:
    return asdict(result)


def deserialize_simulation(payload: dict[str, Any]) -> SimulationResultData:
    return SimulationResultData(
        simulation_id=payload["simulation_id"],
        setup_id=payload["setup_id"],
        setup_name=payload["setup_name"],
        created_at=payload.get("created_at", utc_now_iso()),
        times=[float(v) for v in payload["times"]],
        frames=payload["frames"],
        mass_over_time=[float(v) for v in payload["mass_over_time"]],
        color_limits=[float(v) for v in payload["color_limits"]],
        metadata=payload.get("metadata", {}),
        energy_frames=payload.get("energy_frames"),
        energy_bins=[float(v) for v in payload["energy_bins"]] if payload.get("energy_bins") else None,
        phonon_frames=payload.get("phonon_frames"),
        phonon_energy_frames=payload.get("phonon_energy_frames"),
        phonon_energy_bins=[
            float(v) for v in payload["phonon_energy_bins"]
        ] if payload.get("phonon_energy_bins") else None,
        phonon_metadata=payload.get("phonon_metadata"),
    )


def save_simulation(result: SimulationResultData, path: Path | None = None) -> Path:
    if path is None:
        slug = slugify_name(result.setup_name, "simulation")
        filename = f"{slug}_{result.simulation_id}.json"
        path = SIMULATIONS_DIR / filename
    return _write_json(path, serialize_simulation(result))


def load_simulation(path: str | Path) -> SimulationResultData:
    return deserialize_simulation(_read_json(Path(path)))


def list_simulation_files() -> list[Path]:
    ensure_data_dirs()
    return sorted(SIMULATIONS_DIR.glob("*.json"))


def create_simulation_id() -> str:
    return uuid.uuid4().hex[:12]


def _deserialize_test_case(case: dict[str, Any]) -> TestCaseResultData:
    return TestCaseResultData(
        case_id=case["case_id"],
        title=case["title"],
        boundary_label=case["boundary_label"],
        formula_latex=case["formula_latex"],
        initial_condition_latex=case["initial_condition_latex"],
        description=case["description"],
        x=[float(v) for v in case.get("x", [])],
        times=[float(v) for v in case["times"]],
        # Keep raw nested arrays for speed; viewers convert to numpy at render time.
        simulated=case["simulated"],
        analytic=case["analytic"],
        metadata=case.get("metadata", {}),
    )


def _deserialize_group_inline(group: dict[str, Any]) -> TestGeometryGroupData:
    group_cases = [_deserialize_test_case(case) for case in group.get("cases", [])]
    preview = [[int(v) for v in row] for row in group.get("preview_mask", [])]
    return TestGeometryGroupData(
        geometry_id=group["geometry_id"],
        title=group["title"],
        description=group.get("description", ""),
        view_mode=group.get("view_mode", "line1d"),
        preview_mask=preview,
        cases=group_cases,
        case_count=int(group.get("case_count", len(group_cases))),
        group_file=group.get("group_file"),
    )


def _resolve_group_sidecar_path(manifest_path: Path, group_file: str) -> Path:
    suite_dir = manifest_path.with_suffix("")
    group_rel = Path(group_file)
    if group_rel.is_absolute():
        raise ValueError(f"Geometry group sidecar must be a relative path, got '{group_file}'.")

    suite_root = suite_dir.resolve()
    group_path = (suite_dir / group_rel).resolve()
    try:
        group_path.relative_to(suite_root)
    except ValueError as exc:
        raise ValueError(
            f"Geometry group sidecar '{group_file}' escapes suite directory '{suite_dir}'."
        ) from exc
    return group_path


def load_test_geometry_group(manifest_path: str | Path, geometry_id: str) -> TestGeometryGroupData:
    manifest_path = Path(manifest_path)
    payload = _read_json(manifest_path)
    groups = payload.get("geometry_groups", [])
    raw_group = next((g for g in groups if g.get("geometry_id") == geometry_id), None)
    if raw_group is None:
        raise ValueError(f"Geometry group '{geometry_id}' not found in suite manifest.")

    inline_cases = raw_group.get("cases", [])
    if inline_cases:
        return _deserialize_group_inline(raw_group)

    group_file = raw_group.get("group_file")
    if not group_file:
        raise ValueError(f"Geometry group '{geometry_id}' has no group file reference.")

    group_path = _resolve_group_sidecar_path(manifest_path, str(group_file))
    group_payload = _read_json(group_path)
    raw = group_payload.get("group", group_payload)
    group = _deserialize_group_inline(raw)
    if group.case_count <= 0:
        group.case_count = int(raw_group.get("case_count", len(group.cases)))
    if not group.preview_mask:
        group.preview_mask = [[int(v) for v in row] for row in raw_group.get("preview_mask", [])]
    group.group_file = group_file
    if group.case_count <= 0:
        group.case_count = len(group.cases)
    return group


def deserialize_test_suite(
    payload: dict[str, Any],
    manifest_path: Path | None = None,
    load_group_cases: bool = True,
) -> TestSuiteData:
    geometry_groups_raw = payload.get("geometry_groups")
    if not geometry_groups_raw:
        raise ValueError(
            "Test suite manifest missing 'geometry_groups'. "
            "Legacy flat-case suite format is no longer supported."
        )

    geometry_groups: list[TestGeometryGroupData] = []
    for group in geometry_groups_raw:
        parsed_group = _deserialize_group_inline(group)
        if (
            load_group_cases
            and not parsed_group.cases
            and manifest_path is not None
            and parsed_group.group_file
        ):
            try:
                parsed_group = load_test_geometry_group(manifest_path, parsed_group.geometry_id)
            except Exception as exc:
                raise ValueError(
                    f"Failed to load geometry group '{parsed_group.geometry_id}' "
                    f"from sidecar '{parsed_group.group_file}'."
                ) from exc
        geometry_groups.append(parsed_group)
    cases: list[TestCaseResultData] = []
    for group in geometry_groups:
        cases.extend(group.cases)

    return TestSuiteData(
        suite_id=payload["suite_id"],
        created_at=payload.get("created_at", utc_now_iso()),
        cases=cases,
        geometry_groups=geometry_groups,
        metadata=payload.get("metadata", {}),
    )


def save_test_suite(suite: TestSuiteData, path: Path | None = None) -> Path:
    if path is None:
        filename = f"test_suite_{suite.suite_id}.json"
        path = TEST_CASES_DIR / filename
    if not suite.geometry_groups:
        raise ValueError("Test suite must contain at least one geometry group.")

    suite_dir = path.with_suffix("")

    groups_summary: list[dict[str, Any]] = []
    for group in suite.geometry_groups:
        group_file = f"{slugify_name(group.geometry_id, 'group')}.json"
        group_path = suite_dir / group_file
        full_group = TestGeometryGroupData(
            geometry_id=group.geometry_id,
            title=group.title,
            description=group.description,
            view_mode=group.view_mode,
            preview_mask=group.preview_mask,
            cases=list(group.cases),
            case_count=len(group.cases),
            group_file=group_file,
        )
        _write_json(
            group_path,
            {
                "suite_id": suite.suite_id,
                "group": asdict(full_group),
            },
        )
        groups_summary.append(
            {
                "geometry_id": group.geometry_id,
                "title": group.title,
                "description": group.description,
                "view_mode": group.view_mode,
                "preview_mask": group.preview_mask,
                "cases": [],
                "case_count": len(group.cases),
                "group_file": group_file,
            }
        )

    metadata = dict(suite.metadata or {})
    metadata["format_version"] = max(TEST_SUITE_FORMAT_VERSION, int(metadata.get("format_version", 0)))
    manifest = {
        "suite_id": suite.suite_id,
        "created_at": suite.created_at,
        "cases": [],
        "geometry_groups": groups_summary,
        "metadata": metadata,
    }
    return _write_json(path, manifest)


def load_test_suite(path: str | Path, load_group_cases: bool = True) -> TestSuiteData:
    path = Path(path)
    return deserialize_test_suite(_read_json(path), manifest_path=path, load_group_cases=load_group_cases)


def latest_test_suite_file() -> Path | None:
    files = list_test_suite_files()
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def list_test_suite_files() -> list[Path]:
    ensure_data_dirs()
    return sorted(TEST_CASES_DIR.glob("*.json"))
