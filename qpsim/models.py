from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


BOUNDARY_KINDS = {
    "reflective",
    "neumann",
    "dirichlet",
    "absorbing",
    "robin",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BoundaryCondition:
    kind: str
    value: float | None = None
    aux_value: float | None = None

    def normalized_kind(self) -> str:
        return self.kind.strip().lower()

    def validate(self) -> None:
        kind = self.normalized_kind()
        if kind not in BOUNDARY_KINDS:
            raise ValueError(f"Unsupported boundary condition kind: {self.kind}")
        if kind in {"reflective", "absorbing"}:
            return
        if kind in {"neumann", "dirichlet", "robin"} and self.value is None:
            raise ValueError(f"Boundary condition '{kind}' requires a numeric value")


@dataclass
class BoundaryFace:
    row: int
    col: int
    direction: str


@dataclass
class EdgeSegment:
    edge_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    normal: str
    faces: list[BoundaryFace]


@dataclass
class GeometryData:
    name: str
    source_path: str
    layer: int
    mesh_size: float
    mask: list[list[int]]
    edges: list[EdgeSegment]
    bounds: list[float] | None = None


@dataclass
class InitialConditionSpec:
    kind: str = "gaussian"
    params: dict[str, float] = field(default_factory=dict)
    custom_body: str = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)"
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationParameters:
    diffusion_coefficient: float
    dt: float
    total_time: float
    mesh_size: float
    store_every: int = 1


@dataclass
class SetupData:
    setup_id: str
    name: str
    created_at: str
    geometry: GeometryData
    boundary_conditions: dict[str, BoundaryCondition]
    parameters: SimulationParameters
    initial_condition: InitialConditionSpec


@dataclass
class SimulationResultData:
    simulation_id: str
    setup_id: str
    setup_name: str
    created_at: str
    times: list[float]
    frames: list[list[list[float | None]]]
    mass_over_time: list[float]
    color_limits: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCaseResultData:
    case_id: str
    title: str
    boundary_label: str
    formula_latex: str
    initial_condition_latex: str
    description: str
    x: list[float]
    times: list[float]
    simulated: list[list[float]]
    analytic: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteData:
    suite_id: str
    created_at: str
    cases: list[TestCaseResultData]

