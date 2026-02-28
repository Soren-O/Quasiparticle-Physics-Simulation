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
COLLISION_SOLVERS = {"fischer_catelani_local"}
EXTERNAL_GENERATION_MODES = {"none", "constant", "pulse", "custom"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_collision_solver_name(value: str) -> str:
    solver = str(value).strip().lower()
    if solver not in COLLISION_SOLVERS:
        allowed = ", ".join(sorted(COLLISION_SOLVERS))
        raise ValueError(
            f"Unsupported collision solver '{value}'. Supported values: {allowed}."
        )
    return solver


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
    # Split representation for quasiparticle and phonon initialization.
    spatial_kind: str = ""
    spatial_params: dict[str, Any] = field(default_factory=dict)
    spatial_custom_body: str = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)"
    spatial_custom_params: dict[str, Any] = field(default_factory=dict)
    energy_kind: str = ""  # dos / fermi_dirac / uniform / custom
    energy_params: dict[str, Any] = field(default_factory=dict)
    energy_custom_body: str = "return np.ones_like(E)"
    energy_custom_params: dict[str, Any] = field(default_factory=dict)
    # Optional non-separable QP initializer F_qp(x, y, E).
    qp_full_custom_enabled: bool = False
    qp_full_custom_body: str = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) * np.exp(-E / 500.0)"
    qp_full_custom_params: dict[str, Any] = field(default_factory=dict)
    # Phonon initializer (mirrors split spatial/energy structure).
    phonon_spatial_kind: str = ""  # gaussian / uniform / point / custom
    phonon_spatial_params: dict[str, Any] = field(default_factory=dict)
    phonon_spatial_custom_body: str = "return 1.0"
    phonon_spatial_custom_params: dict[str, Any] = field(default_factory=dict)
    phonon_energy_kind: str = ""  # bose_einstein / uniform / custom
    phonon_energy_params: dict[str, Any] = field(default_factory=dict)
    phonon_energy_custom_body: str = "return np.ones_like(E)"
    phonon_energy_custom_params: dict[str, Any] = field(default_factory=dict)
    # Optional non-separable phonon initializer F_ph(x, y, omega).
    phonon_full_custom_enabled: bool = False
    phonon_full_custom_body: str = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) * np.exp(-E / 500.0)"
    phonon_full_custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalGenerationSpec:
    mode: str = "none"                    # "none" / "constant" / "pulse" / "custom"
    rate: float = 0.0                     # generation rate for constant mode (μeV⁻¹ μm⁻² ns⁻¹)
    pulse_start: float = 0.0             # ns — pulse onset time
    pulse_duration: float = 10.0         # ns — pulse width
    pulse_rate: float = 0.0              # generation rate during pulse
    custom_body: str = "return 0.0"      # expression g(E, x, y, t, params)
    custom_params: dict[str, Any] = field(default_factory=dict)

    def normalized_mode(self) -> str:
        return self.mode.strip().lower()

    def validate(self) -> None:
        mode = self.normalized_mode()
        if mode not in EXTERNAL_GENERATION_MODES:
            allowed = ", ".join(sorted(EXTERNAL_GENERATION_MODES))
            raise ValueError(
                f"Unsupported external generation mode '{self.mode}'. Supported: {allowed}."
            )
        if self.rate < 0:
            raise ValueError("External generation constant rate must be non-negative.")
        if self.pulse_rate < 0:
            raise ValueError("External generation pulse rate must be non-negative.")
        if self.pulse_duration < 0:
            raise ValueError("External generation pulse_duration must be non-negative.")


@dataclass
class SimulationParameters:
    diffusion_coefficient: float     # D₀ in μm²/ns
    dt: float                        # ns
    total_time: float                # ns
    mesh_size: float                 # μm (spatial grid spacing)
    store_every: int = 1
    # Energy grid fields (energy_gap=0 disables energy dimension / scalar mode)
    energy_gap: float = 0.0          # Δ in μeV (0 = no energy dimension)
    energy_min_factor: float = 1.0   # E_min = factor × Δ (must be ≥ 1.0)
    energy_max_factor: float = 10.0  # E_max = factor × Δ
    num_energy_bins: int = 50        # number of energy bins
    dynes_gamma: float = 0.0         # Dynes broadening Γ in μeV (0 = pure BCS)
    gap_expression: str = ""         # spatial gap Δ(x,y) expression (empty = uniform energy_gap)
    collision_solver: str = "fischer_catelani_local"
    # --- physics process toggles ---
    enable_diffusion: bool = True
    enable_recombination: bool = False
    enable_scattering: bool = False
    # --- collision parameters ---
    # tau_0 acts as a convenience default for tau_s and tau_r.
    tau_0: float = 440.0
    tau_s: float | None = None
    tau_r: float | None = None
    T_c: float = 1.2                      # critical temperature in K
    bath_temperature: float = 0.1         # phonon bath temperature in K
    export_phonon_history: bool = False   # store fixed-temperature phonon arrays in simulation output
    external_generation: ExternalGenerationSpec = field(default_factory=ExternalGenerationSpec)

    def __post_init__(self) -> None:
        self.collision_solver = normalize_collision_solver_name(self.collision_solver)
        if self.tau_s is None:
            self.tau_s = float(self.tau_0)
        if self.tau_r is None:
            self.tau_r = float(self.tau_0)
        # Keep tau_0 synchronized with tau_s/tau_r defaults.
        self.tau_0 = float(0.5 * (self.tau_s + self.tau_r))
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.total_time <= 0:
            raise ValueError("total_time must be positive.")
        if self.mesh_size <= 0:
            raise ValueError("mesh_size must be positive.")
        if self.bath_temperature < 0:
            raise ValueError("bath_temperature must be non-negative.")
        if self.enable_recombination or self.enable_scattering:
            if self.T_c <= 0:
                raise ValueError("T_c must be positive when recombination or scattering is enabled.")
            if self.tau_s <= 0:
                raise ValueError("tau_s must be positive when recombination or scattering is enabled.")
            if self.tau_r <= 0:
                raise ValueError("tau_r must be positive when recombination or scattering is enabled.")
        if self.energy_gap > 0:
            if self.energy_min_factor < 1.0:
                raise ValueError("energy_min_factor must be >= 1.0 when energy_gap > 0.")
            if self.energy_max_factor <= self.energy_min_factor:
                raise ValueError("energy_max_factor must be > energy_min_factor when energy_gap > 0.")
            if self.num_energy_bins < 2:
                raise ValueError("num_energy_bins must be >= 2 when energy_gap > 0.")
        self.external_generation.validate()


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
    frames: list[list[list[float | None]]]       # energy-integrated 2D snapshots (for viewer)
    mass_over_time: list[float]
    color_limits: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    energy_frames: list[list[list[list[float | None]]]] | None = None  # full 3D [time][energy][y][x]
    phonon_frames: list[list[list[float | None]]] | None = None  # fixed-temperature phonon map [time][y][x]
    phonon_energy_frames: list[list[list[list[float | None]]]] | None = None  # full 3D [time][omega][y][x]
    phonon_energy_bins: list[float] | None = None
    phonon_metadata: dict[str, Any] | None = None
    energy_bins: list[float] | None = None        # energy bin centers in μeV


@dataclass
class TestCaseResultData:
    __test__ = False  # Prevent pytest from collecting this dataclass as a test class.
    case_id: str
    title: str
    boundary_label: str
    formula_latex: str
    initial_condition_latex: str
    description: str
    x: list[float]
    times: list[float]
    simulated: list[Any]
    analytic: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestGeometryGroupData:
    __test__ = False  # Prevent pytest from collecting this dataclass as a test class.
    geometry_id: str
    title: str
    description: str
    view_mode: str
    preview_mask: list[list[int]]
    cases: list[TestCaseResultData] = field(default_factory=list)
    case_count: int = 0
    group_file: str | None = None


@dataclass
class TestSuiteData:
    __test__ = False  # Prevent pytest from collecting this dataclass as a test class.
    suite_id: str
    created_at: str
    cases: list[TestCaseResultData] = field(default_factory=list)
    geometry_groups: list[TestGeometryGroupData] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
