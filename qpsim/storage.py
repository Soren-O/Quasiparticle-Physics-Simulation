from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import (
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    GeometryData,
    InitialConditionSpec,
    SetupData,
    SimulationParameters,
    SimulationResultData,
    TestCaseResultData,
    TestSuiteData,
    utc_now_iso,
)
from .paths import SETUPS_DIR, SIMULATIONS_DIR, TEST_CASES_DIR, ensure_data_dirs


def slugify_name(name: str, fallback: str = "item") -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return value or fallback


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_data_dirs()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_setup(setup: SetupData) -> dict[str, Any]:
    return asdict(setup)


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
    )

    ic_raw = payload.get("initial_condition", {})
    initial_condition = InitialConditionSpec(
        kind=ic_raw.get("kind", "gaussian"),
        params=ic_raw.get("params", {}),
        custom_body=ic_raw.get(
            "custom_body",
            "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)",
        ),
        custom_params=ic_raw.get("custom_params", {}),
    )

    return SetupData(
        setup_id=payload["setup_id"],
        name=payload["name"],
        created_at=payload.get("created_at", utc_now_iso()),
        geometry=geometry,
        boundary_conditions=bc_map,
        parameters=params,
        initial_condition=initial_condition,
    )


def save_setup(setup: SetupData, path: Path | None = None) -> Path:
    if path is None:
        slug = slugify_name(setup.name, "setup")
        filename = f"{slug}_{setup.setup_id}.json"
        path = SETUPS_DIR / filename
    return _write_json(path, serialize_setup(setup))


def load_setup(path: str | Path) -> SetupData:
    return deserialize_setup(_read_json(Path(path)))


def list_setup_files() -> list[Path]:
    ensure_data_dirs()
    return sorted(SETUPS_DIR.glob("*.json"))


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


def serialize_test_suite(suite: TestSuiteData) -> dict[str, Any]:
    return asdict(suite)


def deserialize_test_suite(payload: dict[str, Any]) -> TestSuiteData:
    cases = [
        TestCaseResultData(
            case_id=case["case_id"],
            title=case["title"],
            boundary_label=case["boundary_label"],
            formula_latex=case["formula_latex"],
            initial_condition_latex=case["initial_condition_latex"],
            description=case["description"],
            x=[float(v) for v in case["x"]],
            times=[float(v) for v in case["times"]],
            simulated=[[float(v) for v in row] for row in case["simulated"]],
            analytic=[[float(v) for v in row] for row in case["analytic"]],
            metadata=case.get("metadata", {}),
        )
        for case in payload.get("cases", [])
    ]
    return TestSuiteData(
        suite_id=payload["suite_id"],
        created_at=payload.get("created_at", utc_now_iso()),
        cases=cases,
    )


def save_test_suite(suite: TestSuiteData, path: Path | None = None) -> Path:
    if path is None:
        filename = f"test_suite_{suite.suite_id}.json"
        path = TEST_CASES_DIR / filename
    return _write_json(path, serialize_test_suite(suite))


def load_test_suite(path: str | Path) -> TestSuiteData:
    return deserialize_test_suite(_read_json(Path(path)))


def latest_test_suite_file() -> Path | None:
    files = list_test_suite_files()
    return files[-1] if files else None


def list_test_suite_files() -> list[Path]:
    ensure_data_dirs()
    return sorted(TEST_CASES_DIR.glob("*.json"))
