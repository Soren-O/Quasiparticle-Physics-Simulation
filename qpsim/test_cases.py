from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import special
from scipy.optimize import brentq

from .geometry import extract_edge_segments
from .models import (
    BoundaryCondition,
    TestCaseResultData,
    TestGeometryGroupData,
    TestSuiteData,
    utc_now_iso,
)
from .solver import (
    _bcs_density_of_states,
    build_energy_grid,
    recombination_kernel,
    run_2d_crank_nicolson,
    scattering_kernel,
    thermal_qp_weights,
)
from .storage import TEST_SUITE_FORMAT_VERSION, frame_to_jsonable, save_test_suite


@dataclass
class _StripCaseDefinition:
    case_id: str
    title: str
    boundary_label: str
    left_bc: BoundaryCondition
    right_bc: BoundaryCondition
    init_fn: Callable[[np.ndarray, float, float], np.ndarray]
    analytic_fn: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]
    formula_latex: str
    initial_latex: str
    description: str


def _find_root(func: Callable[[float], float], intervals: list[tuple[float, float]]) -> float:
    for a, b in intervals:
        try:
            fa, fb = func(a), func(b)
        except Exception:
            continue
        if np.isnan(fa) or np.isnan(fb):
            continue
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb < 0:
            return float(brentq(func, a, b))
    raise ValueError("Could not find root in provided intervals.")


def _build_strip_case_definitions(length: float) -> list[_StripCaseDefinition]:
    h = 0.02
    eps = 1e-6

    def even_eq(mu: float) -> float:
        return mu * np.tan(mu * length / 2.0) - h

    def odd_eq(mu: float) -> float:
        return mu / np.tan(mu * length / 2.0) + h

    m_even = _find_root(even_eq, [(eps, np.pi / length - eps)])
    m_odd = _find_root(
        odd_eq,
        [
            (np.pi / length + eps, 2 * np.pi / length - eps),
            (3 * np.pi / length + eps, 4 * np.pi / length - eps),
        ],
    )

    reflective = BoundaryCondition(kind="reflective")
    dirichlet0 = BoundaryCondition(kind="dirichlet", value=0.0)
    absorbing = BoundaryCondition(kind="absorbing")
    robin = BoundaryCondition(kind="robin", value=h, aux_value=0.0)

    q1 = 0.02
    q2 = -0.015
    neumann1_left = BoundaryCondition(kind="neumann", value=-q1)
    neumann1_right = BoundaryCondition(kind="neumann", value=q1)
    neumann2_left = BoundaryCondition(kind="neumann", value=-q2)
    neumann2_right = BoundaryCondition(kind="neumann", value=q2)

    return [
        _StripCaseDefinition(
            case_id="reflective_mode1",
            title="Reflective BC - Cosine Mode 1",
            boundary_label="Reflective / Insulated (zero flux)",
            left_bc=reflective,
            right_bc=reflective,
            init_fn=lambda x, l, d: 1.0 + 0.4 * np.cos(np.pi * x / l),
            analytic_fn=lambda x, t, l, d: 1.0
            + 0.4 * np.cos(np.pi * x[None, :] / l) * np.exp(-d * (np.pi / l) ** 2 * t[:, None]),
            formula_latex=r"u(x,t)=1+0.4\cos\left(\frac{\pi x}{L}\right)e^{-D(\pi/L)^2t}",
            initial_latex=r"u(x,0)=1+0.4\cos\left(\frac{\pi x}{L}\right)",
            description="Single Neumann cosine mode decay with conserved average.",
        ),
        _StripCaseDefinition(
            case_id="reflective_mode2",
            title="Reflective BC - Cosine Mode 2",
            boundary_label="Reflective / Insulated (zero flux)",
            left_bc=reflective,
            right_bc=reflective,
            init_fn=lambda x, l, d: 0.8 + 0.3 * np.cos(2 * np.pi * x / l),
            analytic_fn=lambda x, t, l, d: 0.8
            + 0.3 * np.cos(2 * np.pi * x[None, :] / l) * np.exp(-d * (2 * np.pi / l) ** 2 * t[:, None]),
            formula_latex=r"u(x,t)=0.8+0.3\cos\left(\frac{2\pi x}{L}\right)e^{-D(2\pi/L)^2t}",
            initial_latex=r"u(x,0)=0.8+0.3\cos\left(\frac{2\pi x}{L}\right)",
            description="Higher Neumann cosine mode decay with insulated boundaries.",
        ),
        _StripCaseDefinition(
            case_id="neumann_flux_mode1",
            title="Neumann Flux BC - Linear + Mode 1",
            boundary_label="Neumann (non-zero flux)",
            left_bc=neumann1_left,
            right_bc=neumann1_right,
            init_fn=lambda x, l, d: q1 * x + 0.25 * np.cos(np.pi * x / l),
            analytic_fn=lambda x, t, l, d: q1 * x[None, :]
            + 0.25 * np.cos(np.pi * x[None, :] / l) * np.exp(-d * (np.pi / l) ** 2 * t[:, None]),
            formula_latex=r"u(x,t)=qx+0.25\cos\left(\frac{\pi x}{L}\right)e^{-D(\pi/L)^2t},\ q=0.02",
            initial_latex=r"u(x,0)=qx+0.25\cos\left(\frac{\pi x}{L}\right)",
            description="Non-zero equal-slope derivative boundaries via homogeneous-mode reduction.",
        ),
        _StripCaseDefinition(
            case_id="neumann_flux_mode2",
            title="Neumann Flux BC - Linear + Mode 2",
            boundary_label="Neumann (non-zero flux)",
            left_bc=neumann2_left,
            right_bc=neumann2_right,
            init_fn=lambda x, l, d: q2 * x + 0.2 * np.cos(2 * np.pi * x / l),
            analytic_fn=lambda x, t, l, d: q2 * x[None, :]
            + 0.2 * np.cos(2 * np.pi * x[None, :] / l) * np.exp(-d * (2 * np.pi / l) ** 2 * t[:, None]),
            formula_latex=r"u(x,t)=qx+0.2\cos\left(\frac{2\pi x}{L}\right)e^{-D(2\pi/L)^2t},\ q=-0.015",
            initial_latex=r"u(x,0)=qx+0.2\cos\left(\frac{2\pi x}{L}\right)",
            description="Second non-zero flux validation case with a higher spatial mode.",
        ),
        _StripCaseDefinition(
            case_id="dirichlet_mode1",
            title="Dirichlet BC - Sine Mode 1",
            boundary_label="Dirichlet (fixed zero boundary value)",
            left_bc=dirichlet0,
            right_bc=dirichlet0,
            init_fn=lambda x, l, d: np.sin(np.pi * x / l),
            analytic_fn=lambda x, t, l, d: np.sin(np.pi * x[None, :] / l) * np.exp(
                -d * (np.pi / l) ** 2 * t[:, None]
            ),
            formula_latex=r"u(x,t)=\sin\left(\frac{\pi x}{L}\right)e^{-D(\pi/L)^2t}",
            initial_latex=r"u(x,0)=\sin\left(\frac{\pi x}{L}\right)",
            description="Classical first Dirichlet eigenmode decay.",
        ),
        _StripCaseDefinition(
            case_id="dirichlet_mode2",
            title="Dirichlet BC - Sine Mode 2",
            boundary_label="Dirichlet (fixed zero boundary value)",
            left_bc=dirichlet0,
            right_bc=dirichlet0,
            init_fn=lambda x, l, d: 0.7 * np.sin(2 * np.pi * x / l),
            analytic_fn=lambda x, t, l, d: 0.7 * np.sin(2 * np.pi * x[None, :] / l) * np.exp(
                -d * (2 * np.pi / l) ** 2 * t[:, None]
            ),
            formula_latex=r"u(x,t)=0.7\sin\left(\frac{2\pi x}{L}\right)e^{-D(2\pi/L)^2t}",
            initial_latex=r"u(x,0)=0.7\sin\left(\frac{2\pi x}{L}\right)",
            description="Second Dirichlet eigenmode decay benchmark.",
        ),
        _StripCaseDefinition(
            case_id="absorbing_mode1",
            title="Absorbing BC - Sine Mode 1",
            boundary_label="Absorbing (implemented as zero-value sink)",
            left_bc=absorbing,
            right_bc=absorbing,
            init_fn=lambda x, l, d: 0.6 * np.sin(np.pi * x / l),
            analytic_fn=lambda x, t, l, d: 0.6 * np.sin(np.pi * x[None, :] / l) * np.exp(
                -d * (np.pi / l) ** 2 * t[:, None]
            ),
            formula_latex=r"u(x,t)=0.6\sin\left(\frac{\pi x}{L}\right)e^{-D(\pi/L)^2t}",
            initial_latex=r"u(x,0)=0.6\sin\left(\frac{\pi x}{L}\right)",
            description="Absorbing boundary replay using the same analytic mode as zero Dirichlet sink.",
        ),
        _StripCaseDefinition(
            case_id="absorbing_mode3",
            title="Absorbing BC - Sine Mode 3",
            boundary_label="Absorbing (implemented as zero-value sink)",
            left_bc=absorbing,
            right_bc=absorbing,
            init_fn=lambda x, l, d: 0.5 * np.sin(3 * np.pi * x / l),
            analytic_fn=lambda x, t, l, d: 0.5 * np.sin(3 * np.pi * x[None, :] / l) * np.exp(
                -d * (3 * np.pi / l) ** 2 * t[:, None]
            ),
            formula_latex=r"u(x,t)=0.5\sin\left(\frac{3\pi x}{L}\right)e^{-D(3\pi/L)^2t}",
            initial_latex=r"u(x,0)=0.5\sin\left(\frac{3\pi x}{L}\right)",
            description="Higher absorbing mode for sink-boundary validation.",
        ),
        _StripCaseDefinition(
            case_id="robin_even_mode",
            title="Robin BC - Even Eigenmode",
            boundary_label="Robin (mixed flux-value)",
            left_bc=robin,
            right_bc=robin,
            init_fn=lambda x, l, d: np.cos(m_even * (x - l / 2.0)),
            analytic_fn=lambda x, t, l, d: np.cos(m_even * (x[None, :] - l / 2.0)) * np.exp(
                -d * m_even * m_even * t[:, None]
            ),
            formula_latex=rf"u(x,t)=\cos(\mu_1(x-L/2))e^{{-D\mu_1^2 t}},\ \mu_1\tan(\mu_1L/2)=h,\ h={h}",
            initial_latex=r"u(x,0)=\cos(\mu_1(x-L/2))",
            description="First symmetric Robin eigenmode with root from transcendental condition.",
        ),
        _StripCaseDefinition(
            case_id="robin_odd_mode",
            title="Robin BC - Odd Eigenmode",
            boundary_label="Robin (mixed flux-value)",
            left_bc=robin,
            right_bc=robin,
            init_fn=lambda x, l, d: np.sin(m_odd * (x - l / 2.0)),
            analytic_fn=lambda x, t, l, d: np.sin(m_odd * (x[None, :] - l / 2.0)) * np.exp(
                -d * m_odd * m_odd * t[:, None]
            ),
            formula_latex=rf"u(x,t)=\sin(\mu_2(x-L/2))e^{{-D\mu_2^2 t}},\ \mu_2\cot(\mu_2L/2)=-h,\ h={h}",
            initial_latex=r"u(x,0)=\sin(\mu_2(x-L/2))",
            description="First antisymmetric Robin eigenmode benchmark.",
        ),
    ]


def _uniform_edge_conditions(edges, bc: BoundaryCondition) -> dict[str, BoundaryCondition]:
    return {edge.edge_id: bc for edge in edges}


def _generate_strip_geometry_group(
    nx: int,
    dx: float,
    diffusion_coefficient: float,
    dt: float,
    total_time: float,
    store_every: int,
) -> TestGeometryGroupData:
    length = nx * dx
    cases: list[TestCaseResultData] = []
    definitions = _build_strip_case_definitions(length)

    for case_def in definitions:
        x_cell_centers = (np.arange(nx, dtype=float) + 0.5) * dx
        u0 = case_def.init_fn(x_cell_centers, length, diffusion_coefficient)
        mask = np.ones((1, nx), dtype=bool)
        edges = extract_edge_segments(mask)

        edge_conditions: dict[str, BoundaryCondition] = {}
        for edge in edges:
            if edge.normal == "left":
                edge_conditions[edge.edge_id] = case_def.left_bc
            elif edge.normal == "right":
                edge_conditions[edge.edge_id] = case_def.right_bc
            else:
                edge_conditions[edge.edge_id] = BoundaryCondition(kind="reflective")

        initial_field = np.zeros((1, nx), dtype=float)
        initial_field[0, :] = u0
        times, frames, _, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial_field,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            total_time=total_time,
            dx=dx,
            store_every=store_every,
        )

        t_arr = np.asarray(times, dtype=float)
        simulated = np.asarray([frame[0, :] for frame in frames], dtype=float)
        analytic = case_def.analytic_fn(x_cell_centers, t_arr, length, diffusion_coefficient)

        cases.append(
            TestCaseResultData(
                case_id=case_def.case_id,
                title=case_def.title,
                boundary_label=case_def.boundary_label,
                formula_latex=case_def.formula_latex,
                initial_condition_latex=case_def.initial_latex,
                description=case_def.description,
                x=x_cell_centers.tolist(),
                times=t_arr.tolist(),
                simulated=simulated.tolist(),
                analytic=np.asarray(analytic, dtype=float).tolist(),
                metadata={
                    "geometry_id": "strip_1d_effective",
                    "view_mode": "line1d",
                    "diffusion_coefficient": diffusion_coefficient,
                    "dx": dx,
                    "dt": dt,
                    "total_time": total_time,
                },
            )
        )

    preview = np.zeros((14, nx + 8), dtype=int)
    preview[6:8, 4:-4] = 1
    return TestGeometryGroupData(
        geometry_id="strip_1d_effective",
        title="Effective 1D Strip",
        description="One-cell-thick strip solved with full 2D engine; 10 boundary-condition validation cases.",
        view_mode="line1d",
        preview_mask=preview.tolist(),
        cases=cases,
    )


def _generate_rectangle_geometry_group(
    dx: float,
    diffusion_coefficient: float,
    dt: float,
    total_time: float,
    store_every: int,
) -> TestGeometryGroupData:
    nx, ny = 56, 36
    lx = nx * dx
    ly = ny * dx
    x_centers = (np.arange(nx, dtype=float) + 0.5) * dx
    y_centers = (np.arange(ny, dtype=float) + 0.5) * dx
    gx, gy = np.meshgrid(x_centers, y_centers)

    mask = np.ones((ny, nx), dtype=bool)
    edges = extract_edge_segments(mask)
    dirichlet_zero = BoundaryCondition(kind="dirichlet", value=0.0)
    reflective = BoundaryCondition(kind="reflective")
    edge_conditions = _uniform_edge_conditions(edges, dirichlet_zero)
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]

    def _edge_conditions_by_normal(overrides: dict[str, BoundaryCondition]) -> dict[str, BoundaryCondition]:
        return {
            edge.edge_id: overrides.get(edge.normal, reflective)
            for edge in edges
        }

    def _append_case(
        *,
        case_id: str,
        title: str,
        boundary_label: str,
        formula_latex: str,
        initial_condition_latex: str,
        description: str,
        mode_m: int,
        mode_n: int,
        phi: np.ndarray,
        lam_sq: float,
        current_edge_conditions: dict[str, BoundaryCondition],
    ) -> None:
        initial_field = phi.copy()
        times, frames, _, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=current_edge_conditions,
            initial_field=initial_field,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            total_time=total_time,
            dx=dx,
            store_every=store_every,
        )
        t_arr = np.asarray(times, dtype=float)
        analytic_frames = [phi * np.exp(-diffusion_coefficient * lam_sq * t) for t in t_arr]
        cases.append(
            TestCaseResultData(
                case_id=case_id,
                title=title,
                boundary_label=boundary_label,
                formula_latex=formula_latex,
                initial_condition_latex=initial_condition_latex,
                description=description,
                x=[],
                times=t_arr.tolist(),
                simulated=[frame_to_jsonable(frame) for frame in frames],
                analytic=[frame_to_jsonable(frame) for frame in analytic_frames],
                metadata={
                    "geometry_id": "rectangle_2d",
                    "view_mode": "heatmap2d",
                    "grid_shape": [ny, nx],
                    "mode_m": mode_m,
                    "mode_n": mode_n,
                    "diffusion_coefficient": diffusion_coefficient,
                    "dx": dx,
                    "dt": dt,
                    "total_time": total_time,
                },
            )
        )

    cases: list[TestCaseResultData] = []
    for mode_index, (m, n) in enumerate(modes, start=1):
        phi = np.sin(m * np.pi * gx / lx) * np.sin(n * np.pi * gy / ly)
        lam_sq = (m * np.pi / lx) ** 2 + (n * np.pi / ly) ** 2
        _append_case(
            case_id=f"rectangle_mode_{m}_{n}",
            title=f"Rectangle Mode ({m}, {n})",
            boundary_label="Dirichlet zero on all rectangle edges",
            formula_latex=(
                rf"u(x,y,t)=\sin\left(\frac{{{m}\pi x}}{{L_x}}\right)"
                rf"\sin\left(\frac{{{n}\pi y}}{{L_y}}\right)"
                rf"e^{{-D[(\frac{{{m}\pi}}{{L_x}})^2+(\frac{{{n}\pi}}{{L_y}})^2]t}}"
            ),
            initial_condition_latex=(
                rf"u(x,y,0)=\sin\left(\frac{{{m}\pi x}}{{L_x}}\right)"
                rf"\sin\left(\frac{{{n}\pi y}}{{L_y}}\right)"
            ),
            description=f"2D rectangular Dirichlet eigenmode benchmark case {mode_index}.",
            mode_m=m,
            mode_n=n,
            phi=phi,
            lam_sq=lam_sq,
            current_edge_conditions=edge_conditions,
        )

    mixed_profiles = [
        {
            "case_id": "rectangle_mix_dirichlet_x_neumann_y_1_1",
            "title": "Rectangle Mixed BC (D/N) Mode (1, 1)",
            "boundary_label": "Dirichlet on left/right, reflective on top/bottom",
            "formula_latex": (
                r"u(x,y,t)=\sin\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right)"
                r"e^{-D[(\frac{\pi}{L_x})^2+(\frac{\pi}{L_y})^2]t}"
            ),
            "initial_condition_latex": (
                r"u(x,y,0)=\sin\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right)"
            ),
            "description": "Mixed-boundary rectangle benchmark with Dirichlet-x and Neumann-y constraints.",
            "mode_m": 1,
            "mode_n": 1,
            "phi": np.sin(np.pi * gx / lx) * np.cos(np.pi * gy / ly),
            "edge_conditions": _edge_conditions_by_normal(
                {"left": dirichlet_zero, "right": dirichlet_zero, "up": reflective, "down": reflective}
            ),
        },
        {
            "case_id": "rectangle_mix_neumann_x_dirichlet_y_1_1",
            "title": "Rectangle Mixed BC (N/D) Mode (1, 1)",
            "boundary_label": "Reflective on left/right, Dirichlet on top/bottom",
            "formula_latex": (
                r"u(x,y,t)=\cos\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right)"
                r"e^{-D[(\frac{\pi}{L_x})^2+(\frac{\pi}{L_y})^2]t}"
            ),
            "initial_condition_latex": (
                r"u(x,y,0)=\cos\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right)"
            ),
            "description": "Mixed-boundary rectangle benchmark with Neumann-x and Dirichlet-y constraints.",
            "mode_m": 1,
            "mode_n": 1,
            "phi": np.cos(np.pi * gx / lx) * np.sin(np.pi * gy / ly),
            "edge_conditions": _edge_conditions_by_normal(
                {"left": reflective, "right": reflective, "up": dirichlet_zero, "down": dirichlet_zero}
            ),
        },
        {
            "case_id": "rectangle_reflective_mode_1_1",
            "title": "Rectangle Reflective Mode (1, 1)",
            "boundary_label": "Reflective on all rectangle edges",
            "formula_latex": (
                r"u(x,y,t)=\cos\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right)"
                r"e^{-D[(\frac{\pi}{L_x})^2+(\frac{\pi}{L_y})^2]t}"
            ),
            "initial_condition_latex": (
                r"u(x,y,0)=\cos\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right)"
            ),
            "description": "Fully reflective rectangle benchmark with zero-flux boundaries on all sides.",
            "mode_m": 1,
            "mode_n": 1,
            "phi": np.cos(np.pi * gx / lx) * np.cos(np.pi * gy / ly),
            "edge_conditions": _uniform_edge_conditions(edges, reflective),
        },
    ]
    for profile in mixed_profiles:
        m = int(profile["mode_m"])
        n = int(profile["mode_n"])
        lam_sq = (m * np.pi / lx) ** 2 + (n * np.pi / ly) ** 2
        _append_case(
            case_id=str(profile["case_id"]),
            title=str(profile["title"]),
            boundary_label=str(profile["boundary_label"]),
            formula_latex=str(profile["formula_latex"]),
            initial_condition_latex=str(profile["initial_condition_latex"]),
            description=str(profile["description"]),
            mode_m=m,
            mode_n=n,
            phi=np.asarray(profile["phi"], dtype=float),
            lam_sq=lam_sq,
            current_edge_conditions=profile["edge_conditions"],
        )

    preview = np.pad(mask.astype(int), pad_width=3, mode="constant", constant_values=0)
    return TestGeometryGroupData(
        geometry_id="rectangle_2d",
        title="2D Rectangle",
        description="Non-1D rectangular diffusion with Dirichlet, mixed, and reflective analytic eigenmode solutions.",
        view_mode="heatmap2d",
        preview_mask=preview.tolist(),
        cases=cases,
    )


def _regular_polygon(
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    phase: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False) + phase
    if clockwise:
        angles = angles[::-1]
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


def _polygon_donut_mask(nx: int, ny: int) -> tuple[np.ndarray, float, float, float, float]:
    x_centers = np.arange(nx, dtype=float) + 0.5
    y_centers = np.arange(ny, dtype=float) + 0.5
    gx, gy = np.meshgrid(x_centers, y_centers)
    points = np.column_stack([gx.ravel(), gy.ravel()])

    cx = nx / 2.0
    cy = ny / 2.0
    outer_r = 0.42 * min(nx, ny)
    inner_r = 0.19 * min(nx, ny)
    # Concentric regular polygons with matching orientation/phase keep the
    # donut visually symmetric while still remaining polygonal (not circular).
    outer_poly = _regular_polygon(cx, cy, outer_r, sides=20, phase=0.0, clockwise=False)
    inner_poly = _regular_polygon(cx, cy, inner_r, sides=20, phase=0.0, clockwise=True)

    inside_outer = MplPath(outer_poly).contains_points(points)
    inside_inner = MplPath(inner_poly).contains_points(points)
    mask = (inside_outer & ~inside_inner).reshape((ny, nx))
    return mask, cx, cy, inner_r, outer_r


def _annulus_eigenvalue(inner_r: float, outer_r: float, mode_index: int) -> float:
    return _annulus_radial_eigenvalue(
        inner_r=inner_r,
        outer_r=outer_r,
        mode_index=mode_index,
        inner_boundary="dirichlet",
        outer_boundary="dirichlet",
    )


def _annulus_radial_eigenvalue(
    inner_r: float,
    outer_r: float,
    mode_index: int,
    inner_boundary: str,
    outer_boundary: str,
) -> float:
    valid = {"dirichlet", "reflective", "neumann"}
    if inner_boundary not in valid or outer_boundary not in valid:
        raise ValueError("Annulus radial eigenvalue supports dirichlet/reflective(neumann) boundaries only.")

    def _row_values(lam: float, radius: float, boundary: str) -> tuple[float, float]:
        if boundary in {"reflective", "neumann"}:
            return float(special.j1(lam * radius)), float(special.y1(lam * radius))
        return float(special.j0(lam * radius)), float(special.y0(lam * radius))

    def f(lam: float) -> float:
        i0, i1 = _row_values(lam, inner_r, inner_boundary)
        o0, o1 = _row_values(lam, outer_r, outer_boundary)
        return i0 * o1 - i1 * o0

    roots: list[float] = []
    left = 1e-4
    f_left = f(left)
    for right in np.linspace(0.01, 4.0, 5000):
        f_right = f(right)
        if np.isfinite(f_left) and np.isfinite(f_right) and f_left * f_right < 0:
            try:
                root = float(brentq(f, left, right))
            except Exception:
                root = None
            if root is not None and (not roots or abs(root - roots[-1]) > 1e-4):
                roots.append(root)
                if len(roots) >= mode_index:
                    return roots[mode_index - 1]
        left, f_left = right, f_right
    raise ValueError("Failed to find annulus eigenvalue root.")


def _annulus_radial_mode(
    r: np.ndarray,
    lam: float,
    inner_r: float,
    inner_boundary: str,
) -> np.ndarray:
    if inner_boundary in {"reflective", "neumann"}:
        coeff_j = special.y1(lam * inner_r)
        coeff_y = -special.j1(lam * inner_r)
    else:
        coeff_j = special.y0(lam * inner_r)
        coeff_y = -special.j0(lam * inner_r)
    return coeff_j * special.j0(lam * r) + coeff_y * special.y0(lam * r)


def _annulus_edge_conditions(
    edges,
    cx: float,
    cy: float,
    inner_r: float,
    outer_r: float,
    inner_bc: BoundaryCondition,
    outer_bc: BoundaryCondition,
) -> dict[str, BoundaryCondition]:
    split_radius = 0.5 * (inner_r + outer_r)
    out: dict[str, BoundaryCondition] = {}
    for edge in edges:
        mx = 0.5 * (edge.x0 + edge.x1)
        my = 0.5 * (edge.y0 + edge.y1)
        radius = float(np.hypot(mx - cx, my - cy))
        out[edge.edge_id] = inner_bc if radius < split_radius else outer_bc
    return out


def _generate_polygon_donut_geometry_group(
    dx: float,
    diffusion_coefficient: float,
    dt: float,
    total_time: float,
    store_every: int,
) -> TestGeometryGroupData:
    nx, ny = 64, 64
    mask, cx, cy, inner_r, outer_r = _polygon_donut_mask(nx=nx, ny=ny)
    edges = extract_edge_segments(mask)
    dirichlet_zero = BoundaryCondition(kind="dirichlet", value=0.0)
    reflective = BoundaryCondition(kind="reflective")

    y_idx, x_idx = np.indices(mask.shape, dtype=float)
    x = x_idx + 0.5
    y = y_idx + 0.5
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    cases: list[TestCaseResultData] = []
    profiles = [
        {
            "case_id": "donut_radial_dd_mode_1",
            "title": "Donut Radial D/D Mode 1",
            "inner_boundary": "dirichlet",
            "outer_boundary": "dirichlet",
            "boundary_label": "Dirichlet on inner and outer polygon boundaries",
            "mode_index": 1,
            "boundary_latex": r"\phi_k(a)=0,\quad \phi_k(b)=0",
        },
        {
            "case_id": "donut_radial_dn_mode_1",
            "title": "Donut Radial D/N Mode 1",
            "inner_boundary": "dirichlet",
            "outer_boundary": "reflective",
            "boundary_label": "Dirichlet inner boundary, reflective outer boundary",
            "mode_index": 1,
            "boundary_latex": r"\phi_k(a)=0,\quad \partial_r\phi_k(b)=0",
        },
        {
            "case_id": "donut_radial_nd_mode_1",
            "title": "Donut Radial N/D Mode 1",
            "inner_boundary": "reflective",
            "outer_boundary": "dirichlet",
            "boundary_label": "Reflective inner boundary, Dirichlet outer boundary",
            "mode_index": 1,
            "boundary_latex": r"\partial_r\phi_k(a)=0,\quad \phi_k(b)=0",
        },
        {
            "case_id": "donut_radial_nn_mode_1",
            "title": "Donut Radial N/N Mode 1",
            "inner_boundary": "reflective",
            "outer_boundary": "reflective",
            "boundary_label": "Reflective inner and outer polygon boundaries",
            "mode_index": 1,
            "boundary_latex": r"\partial_r\phi_k(a)=0,\quad \partial_r\phi_k(b)=0",
        },
    ]
    for profile in profiles:
        mode_index = int(profile["mode_index"])
        inner_boundary = str(profile["inner_boundary"])
        outer_boundary = str(profile["outer_boundary"])
        lam = _annulus_radial_eigenvalue(
            inner_r=inner_r,
            outer_r=outer_r,
            mode_index=mode_index,
            inner_boundary=inner_boundary,
            outer_boundary=outer_boundary,
        )
        phi = _annulus_radial_mode(r=r, lam=lam, inner_r=inner_r, inner_boundary=inner_boundary)
        phi[~mask] = 0.0
        amp = np.max(np.abs(phi[mask]))
        if amp > 0:
            phi = phi / amp

        edge_conditions = _annulus_edge_conditions(
            edges=edges,
            cx=cx,
            cy=cy,
            inner_r=inner_r,
            outer_r=outer_r,
            inner_bc=dirichlet_zero if inner_boundary == "dirichlet" else reflective,
            outer_bc=dirichlet_zero if outer_boundary == "dirichlet" else reflective,
        )
        initial_field = phi.copy()
        times, frames, _, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial_field,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            total_time=total_time,
            dx=dx,
            store_every=store_every,
        )
        t_arr = np.asarray(times, dtype=float)
        analytic_frames: list[np.ndarray] = []
        for t in t_arr:
            frame = phi * np.exp(-diffusion_coefficient * lam * lam * t)
            frame = frame.copy()
            frame[~mask] = np.nan
            analytic_frames.append(frame)

        frames_with_nan = []
        for frame in frames:
            copy = frame.copy()
            copy[~mask] = np.nan
            frames_with_nan.append(copy)

        cases.append(
            TestCaseResultData(
                case_id=str(profile["case_id"]),
                title=str(profile["title"]),
                boundary_label=str(profile["boundary_label"]),
                formula_latex=(
                    r"u(r,t)=\phi_k(r)e^{-D\lambda_k^2 t},\ "
                    + str(profile["boundary_latex"])
                ),
                initial_condition_latex=r"u(r,0)=\phi_k(r)",
                description=(
                    "Polygon annulus benchmark using radial Bessel eigenmodes "
                    f"with {profile['boundary_label'].lower()} (k={mode_index})."
                ),
                x=[],
                times=t_arr.tolist(),
                simulated=[frame_to_jsonable(frame) for frame in frames_with_nan],
                analytic=[frame_to_jsonable(frame) for frame in analytic_frames],
                metadata={
                    "geometry_id": "polygon_donut",
                    "view_mode": "heatmap2d",
                    "grid_shape": [ny, nx],
                    "mode_index": mode_index,
                    "inner_boundary": inner_boundary,
                    "outer_boundary": outer_boundary,
                    "lambda": float(lam),
                    "inner_radius": float(inner_r),
                    "outer_radius": float(outer_r),
                    "diffusion_coefficient": diffusion_coefficient,
                    "dx": dx,
                    "dt": dt,
                    "total_time": total_time,
                },
            )
        )

    preview = np.pad(mask.astype(int), pad_width=3, mode="constant", constant_values=0)
    return TestGeometryGroupData(
        geometry_id="polygon_donut",
        title="Polygon Donut",
        description="Polygonal annulus geometry with Dirichlet/reflective boundary variants and radial Bessel analytic solutions.",
        view_mode="heatmap2d",
        preview_mask=preview.tolist(),
        cases=cases,
    )


def _generate_recombination_test_group() -> TestGeometryGroupData:
    """Generate recombination test cases comparing simulation to analytic ODE solutions."""
    cases: list[TestCaseResultData] = []

    # ── Case 1: Pure 1/t Recombination Decay (T_bath=0) ──
    tau_0_1 = 440.0  # ns
    T_c_1 = 1.2      # K
    gap_1 = 180.0     # μeV
    T_bath_1 = 0.0
    E_bin_1 = np.array([1.5 * gap_1])  # single energy bin
    dE_1 = 1.0  # arbitrary for single bin; cancels with n₀ units

    K_r_1 = recombination_kernel(E_bin_1, gap_1, tau_0_1, T_c_1, T_bath_1)
    R_1 = 2.0 * float(K_r_1[0, 0]) * dE_1  # effective rate constant

    n0_1 = 0.5
    dt_1 = 0.5
    total_time_1 = 2000.0
    store_every_1 = 4

    # Run simulation: 1×1 mask, reflective BCs, diffusion OFF, recombination ON
    mask_1 = np.ones((1, 1), dtype=bool)
    edges_1 = extract_edge_segments(mask_1)
    edge_conds_1 = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges_1}
    init_field_1 = np.full((1, 1), n0_1, dtype=float)

    times_1, _, _, _, energy_frames_1, _ = run_2d_crank_nicolson(
        mask=mask_1, edges=edges_1, edge_conditions=edge_conds_1,
        initial_field=init_field_1, diffusion_coefficient=1.0,
        dt=dt_1, total_time=total_time_1, dx=1.0, store_every=store_every_1,
        energy_gap=gap_1, energy_min_factor=1.5, energy_max_factor=1.5,
        num_energy_bins=1, energy_weights=np.array([1.0]),
        enable_diffusion=False, enable_recombination=True,
        tau_0=tau_0_1, T_c=T_c_1, bath_temperature=T_bath_1,
    )

    t_arr_1 = np.asarray(times_1, dtype=float)
    # Extract simulated n(t) from energy frames: single bin, single spatial point
    sim_n_1 = np.array([ef[0][0, 0] for ef in energy_frames_1], dtype=float) * dE_1
    ana_n_1 = n0_1 / (1.0 + R_1 * n0_1 * t_arr_1)

    cases.append(TestCaseResultData(
        case_id="recomb_pure_1_over_t",
        title="Pure 1/t Recombination Decay",
        boundary_label="Reflective (single cell, no diffusion)",
        formula_latex=r"n(t) = \frac{n_0}{1 + R\,n_0\,t},\quad R = 2\,K^r\,\Delta E",
        initial_condition_latex=r"n(0) = 0.5",
        description=(
            "Single energy bin at E=1.5\u0394, T_bath=0. "
            "Two-body recombination gives dn/dt = -Rn\u00b2 "
            "with the classic 1/t power-law solution."
        ),
        x=t_arr_1.tolist(),
        times=[0.0],
        simulated=[sim_n_1.tolist()],
        analytic=[ana_n_1.tolist()],
        metadata={
            "geometry_id": "recombination",
            "view_mode": "timeseries",
            "tau_0": tau_0_1, "T_c": T_c_1, "gap": gap_1,
            "T_bath": T_bath_1, "R": R_1, "n0": n0_1,
        },
    ))

    # ── Case 2: Equilibrium Stationarity ──
    tau_0_2 = 10.0   # ns (fast for practical timing)
    T_c_2 = 1.2
    gap_2 = 180.0
    T_bath_2 = 0.8
    num_bins_2 = 15
    E_bins_2, dE_2 = build_energy_grid(gap_2, 1.0, 3.0, num_bins_2)

    n_eq_2 = thermal_qp_weights(E_bins_2, gap_2, T_bath_2)
    total_n_eq_2 = float(np.sum(n_eq_2) * dE_2)

    # Initial field = total equilibrium density
    mask_2 = np.ones((1, 1), dtype=bool)
    edges_2 = extract_edge_segments(mask_2)
    edge_conds_2 = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges_2}
    init_field_2 = np.full((1, 1), total_n_eq_2, dtype=float)

    dt_2 = 0.1
    total_time_2 = 200.0
    store_every_2 = 10

    times_2, _, _, _, energy_frames_2, _ = run_2d_crank_nicolson(
        mask=mask_2, edges=edges_2, edge_conditions=edge_conds_2,
        initial_field=init_field_2, diffusion_coefficient=1.0,
        dt=dt_2, total_time=total_time_2, dx=1.0, store_every=store_every_2,
        energy_gap=gap_2, energy_min_factor=1.0, energy_max_factor=3.0,
        num_energy_bins=num_bins_2, energy_weights=n_eq_2,
        enable_diffusion=False, enable_recombination=True,
        tau_0=tau_0_2, T_c=T_c_2, bath_temperature=T_bath_2,
    )

    t_arr_2 = np.asarray(times_2, dtype=float)
    sim_n_2 = np.array([
        float(np.sum(np.array([ef_bin[0, 0] for ef_bin in ef])) * dE_2)
        for ef in energy_frames_2
    ], dtype=float)
    ana_n_2 = np.full_like(t_arr_2, total_n_eq_2)

    cases.append(TestCaseResultData(
        case_id="recomb_equilibrium_stationarity",
        title="Equilibrium Stationarity",
        boundary_label="Reflective (single cell, no diffusion)",
        formula_latex=r"n(t) = n_{\mathrm{eq}} = \mathrm{const}",
        initial_condition_latex=r"n(0) = n_{\mathrm{eq}}(T_{\mathrm{bath}})",
        description=(
            "15 energy bins, T_bath=0.8 K, \u03c4\u2080=10 ns. "
            "Initial state is exact thermal equilibrium. "
            "Thermal generation exactly balances recombination, "
            "so total QP density remains constant."
        ),
        x=t_arr_2.tolist(),
        times=[0.0],
        simulated=[sim_n_2.tolist()],
        analytic=[ana_n_2.tolist()],
        metadata={
            "geometry_id": "recombination",
            "view_mode": "timeseries",
            "tau_0": tau_0_2, "T_c": T_c_2, "gap": gap_2,
            "T_bath": T_bath_2, "n_eq": total_n_eq_2,
        },
    ))

    # ── Case 3: Decay to Thermal Equilibrium ──
    tau_0_3 = 10.0
    T_c_3 = 1.2
    gap_3 = 180.0
    T_bath_3 = 0.8
    E_bin_3 = np.array([1.5 * gap_3])
    dE_3 = 1.0

    K_r_3 = recombination_kernel(E_bin_3, gap_3, tau_0_3, T_c_3, T_bath_3)
    R_3 = 2.0 * float(K_r_3[0, 0]) * dE_3

    n_eq_w3 = thermal_qp_weights(E_bin_3, gap_3, T_bath_3)
    G_therm_3 = 2.0 * n_eq_w3[0] * dE_3 * float(K_r_3[0, 0]) * n_eq_w3[0]
    # n_eq for the single-bin ODE: R n_eq² = G_therm → n_eq = sqrt(G/R)
    n_eq_3 = np.sqrt(float(G_therm_3) / R_3)

    n0_3 = 0.5
    dt_3 = 0.05
    total_time_3 = 50.0
    store_every_3 = 4

    mask_3 = np.ones((1, 1), dtype=bool)
    edges_3 = extract_edge_segments(mask_3)
    edge_conds_3 = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges_3}
    init_field_3 = np.full((1, 1), n0_3, dtype=float)

    times_3, _, _, _, energy_frames_3, _ = run_2d_crank_nicolson(
        mask=mask_3, edges=edges_3, edge_conditions=edge_conds_3,
        initial_field=init_field_3, diffusion_coefficient=1.0,
        dt=dt_3, total_time=total_time_3, dx=1.0, store_every=store_every_3,
        energy_gap=gap_3, energy_min_factor=1.5, energy_max_factor=1.5,
        num_energy_bins=1, energy_weights=np.array([1.0]),
        enable_diffusion=False, enable_recombination=True,
        tau_0=tau_0_3, T_c=T_c_3, bath_temperature=T_bath_3,
    )

    t_arr_3 = np.asarray(times_3, dtype=float)
    sim_n_3 = np.array([ef[0][0, 0] for ef in energy_frames_3], dtype=float) * dE_3
    # Analytic: n(t) = n_eq * coth(R n_eq t + arccoth(n0/n_eq))
    arccoth_arg = n0_3 / n_eq_3
    arccoth_val = 0.5 * np.log((arccoth_arg + 1.0) / (arccoth_arg - 1.0))
    ana_n_3 = n_eq_3 / np.tanh(R_3 * n_eq_3 * t_arr_3 + arccoth_val)

    cases.append(TestCaseResultData(
        case_id="recomb_decay_to_equilibrium",
        title="Decay to Thermal Equilibrium",
        boundary_label="Reflective (single cell, no diffusion)",
        formula_latex=r"n(t) = n_{\mathrm{eq}}\,\coth\!\left(R\,n_{\mathrm{eq}}\,t + \mathrm{arccoth}\!\left(\frac{n_0}{n_{\mathrm{eq}}}\right)\right)",
        initial_condition_latex=r"n(0) = 0.5 \gg n_{\mathrm{eq}}",
        description=(
            "Single energy bin at E=1.5\u0394, T_bath=0.8 K, \u03c4\u2080=10 ns. "
            "Elevated initial density decays toward thermal equilibrium "
            "via dn/dt = R(n_eq\u00b2 - n\u00b2)."
        ),
        x=t_arr_3.tolist(),
        times=[0.0],
        simulated=[sim_n_3.tolist()],
        analytic=[ana_n_3.tolist()],
        metadata={
            "geometry_id": "recombination",
            "view_mode": "timeseries",
            "tau_0": tau_0_3, "T_c": T_c_3, "gap": gap_3,
            "T_bath": T_bath_3, "R": R_3, "n0": n0_3, "n_eq": n_eq_3,
        },
    ))

    preview = np.zeros((8, 12), dtype=int)
    preview[3:5, 5:7] = 1
    return TestGeometryGroupData(
        geometry_id="recombination",
        title="Recombination Dynamics",
        description="Quasiparticle recombination test cases comparing simulated dynamics to analytic ODE solutions.",
        view_mode="timeseries",
        preview_mask=preview.tolist(),
        cases=cases,
    )


def _generate_scattering_test_group() -> TestGeometryGroupData:
    """Generate scattering test cases comparing simulation to analytic predictions."""
    cases: list[TestCaseResultData] = []

    # ── Case 1: Top-Bin Scattering Out (Exponential Decay) ──
    tau_0_1 = 10.0   # ns
    T_c_1 = 1.2      # K
    gap_1 = 180.0     # μeV
    T_bath_1 = 0.3    # K
    num_bins_1 = 10
    E_bins_1, dE_1 = build_energy_grid(gap_1, 1.0, 3.0, num_bins_1)

    # Precompute analytic decay rate for the top bin
    K_s_1 = scattering_kernel(E_bins_1, gap_1, tau_0_1, T_c_1, T_bath_1)
    rho_1 = _bcs_density_of_states(E_bins_1, gap_1)
    # Γ_top = ΔE × Σ_j K^s_{top,j} × ρ_j  (with 1-f_j ≈ 1 at low occupation)
    top_idx = num_bins_1 - 1
    Gamma_top = dE_1 * float(np.sum(K_s_1[top_idx, :] * rho_1))

    # Initial: only top bin populated with small density, all others zero
    n0_top = 0.01
    # Build energy weights: zero everywhere except top bin
    e_weights_1 = np.zeros(num_bins_1)
    e_weights_1[top_idx] = 1.0

    mask_1 = np.ones((1, 1), dtype=bool)
    edges_1 = extract_edge_segments(mask_1)
    edge_conds_1 = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges_1}
    init_field_1 = np.full((1, 1), n0_top, dtype=float)

    dt_1 = 0.002
    total_time_1 = 4.0
    store_every_1 = 20

    times_1, _, _, _, energy_frames_1, _ = run_2d_crank_nicolson(
        mask=mask_1, edges=edges_1, edge_conditions=edge_conds_1,
        initial_field=init_field_1, diffusion_coefficient=1.0,
        dt=dt_1, total_time=total_time_1, dx=1.0, store_every=store_every_1,
        energy_gap=gap_1, energy_min_factor=1.0, energy_max_factor=3.0,
        num_energy_bins=num_bins_1, energy_weights=e_weights_1,
        enable_diffusion=False, enable_recombination=False, enable_scattering=True,
        tau_0=tau_0_1, T_c=T_c_1, bath_temperature=T_bath_1,
    )

    t_arr_1 = np.asarray(times_1, dtype=float)
    # Extract top-bin density over time (spectral density × dE)
    sim_n_1 = np.array([ef[top_idx][0, 0] for ef in energy_frames_1], dtype=float) * dE_1
    ana_n_1 = n0_top * np.exp(-Gamma_top * t_arr_1)

    cases.append(TestCaseResultData(
        case_id="scat_top_bin_decay",
        title="Top-Bin Scattering Out (Exponential Decay)",
        boundary_label="Reflective (single cell, no diffusion)",
        formula_latex=r"n_{\mathrm{top}}(t)=n_0 e^{-\Gamma t},\quad \Gamma=\Delta E\sum_j K^s_{\mathrm{top},j}\rho_j",
        initial_condition_latex=r"n_{\mathrm{top}}(0)=0.01,\quad n_{j\neq \mathrm{top}}(0)=0",
        description=(
            "10 energy bins, T_bath=0.3 K, \u03c4\u2080=10 ns. "
            "Only the highest bin is populated (low density, Pauli blocking \u2248 0). "
            "No density above \u2192 nothing scatters in. "
            "Pure exponential decay at rate \u0393."
        ),
        x=t_arr_1.tolist(),
        times=[0.0],
        simulated=[sim_n_1.tolist()],
        analytic=[ana_n_1.tolist()],
        metadata={
            "geometry_id": "scattering",
            "view_mode": "timeseries",
            "tau_0": tau_0_1, "T_c": T_c_1, "gap": gap_1,
            "T_bath": T_bath_1, "Gamma_top": Gamma_top, "n0": n0_top,
        },
    ))

    # ── Case 2: Scattering Equilibrium Stationarity ──
    tau_0_2 = 10.0
    T_c_2 = 1.2
    gap_2 = 180.0
    T_bath_2 = 0.8
    num_bins_2 = 15
    E_bins_2, dE_2 = build_energy_grid(gap_2, 1.0, 3.0, num_bins_2)

    n_eq_2 = thermal_qp_weights(E_bins_2, gap_2, T_bath_2)
    total_n_eq_2 = float(np.sum(n_eq_2) * dE_2)

    mask_2 = np.ones((1, 1), dtype=bool)
    edges_2 = extract_edge_segments(mask_2)
    edge_conds_2 = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges_2}
    init_field_2 = np.full((1, 1), total_n_eq_2, dtype=float)

    dt_2 = 0.1
    total_time_2 = 200.0
    store_every_2 = 10

    times_2, _, _, _, energy_frames_2, _ = run_2d_crank_nicolson(
        mask=mask_2, edges=edges_2, edge_conditions=edge_conds_2,
        initial_field=init_field_2, diffusion_coefficient=1.0,
        dt=dt_2, total_time=total_time_2, dx=1.0, store_every=store_every_2,
        energy_gap=gap_2, energy_min_factor=1.0, energy_max_factor=3.0,
        num_energy_bins=num_bins_2, energy_weights=n_eq_2,
        enable_diffusion=False, enable_recombination=False, enable_scattering=True,
        tau_0=tau_0_2, T_c=T_c_2, bath_temperature=T_bath_2,
    )

    t_arr_2 = np.asarray(times_2, dtype=float)
    sim_n_2 = np.array([
        float(np.sum(np.array([ef_bin[0, 0] for ef_bin in ef])) * dE_2)
        for ef in energy_frames_2
    ], dtype=float)
    ana_n_2 = np.full_like(t_arr_2, total_n_eq_2)

    cases.append(TestCaseResultData(
        case_id="scat_equilibrium_stationarity",
        title="Scattering Equilibrium Stationarity",
        boundary_label="Reflective (single cell, no diffusion)",
        formula_latex=r"n(t) = n_{\mathrm{eq}} = \mathrm{const}",
        initial_condition_latex=r"n(0) = n_{\mathrm{eq}}(T_{\mathrm{bath}})",
        description=(
            "15 energy bins, T_bath=0.8 K, \u03c4\u2080=10 ns. "
            "Initial state is exact thermal equilibrium. "
            "Detailed balance ensures scattering in = scattering out "
            "at every energy, so total QP density remains constant."
        ),
        x=t_arr_2.tolist(),
        times=[0.0],
        simulated=[sim_n_2.tolist()],
        analytic=[ana_n_2.tolist()],
        metadata={
            "geometry_id": "scattering",
            "view_mode": "timeseries",
            "tau_0": tau_0_2, "T_c": T_c_2, "gap": gap_2,
            "T_bath": T_bath_2, "n_eq": total_n_eq_2,
        },
    ))

    preview = np.zeros((8, 12), dtype=int)
    preview[3:5, 5:7] = 1
    return TestGeometryGroupData(
        geometry_id="scattering",
        title="Scattering Dynamics",
        description="Quasiparticle-phonon scattering test cases verifying exponential decay and detailed balance.",
        view_mode="timeseries",
        preview_mask=preview.tolist(),
        cases=cases,
    )


def generate_test_suite(
    nx: int = 100,
    dx: float = 1.0,
    diffusion_coefficient: float = 25.0,
    dt: float = 0.05,
    total_time: float = 8.0,
    store_every: int = 2,
) -> TestSuiteData:
    if nx < 8:
        raise ValueError("nx must be at least 8 for test generation.")
    if abs(dx - 1.0) > 1e-9:
        raise ValueError("Test suite expects mesh_size (dx) = 1.0.")

    strip_group = _generate_strip_geometry_group(
        nx=nx,
        dx=dx,
        diffusion_coefficient=diffusion_coefficient,
        dt=dt,
        total_time=total_time,
        store_every=store_every,
    )
    rectangle_group = _generate_rectangle_geometry_group(
        dx=dx,
        diffusion_coefficient=diffusion_coefficient,
        dt=dt,
        total_time=total_time,
        store_every=store_every,
    )
    donut_group = _generate_polygon_donut_geometry_group(
        dx=dx,
        diffusion_coefficient=diffusion_coefficient,
        dt=dt,
        total_time=total_time,
        store_every=store_every,
    )

    recombination_group = _generate_recombination_test_group()
    scattering_group = _generate_scattering_test_group()
    geometry_groups = [strip_group, rectangle_group, donut_group, recombination_group, scattering_group]
    return TestSuiteData(
        suite_id=uuid.uuid4().hex[:12],
        created_at=utc_now_iso(),
        cases=[],
        geometry_groups=geometry_groups,
        metadata={"format_version": TEST_SUITE_FORMAT_VERSION},
    )


def generate_and_save_test_suite() -> tuple[TestSuiteData, str]:
    suite = generate_test_suite()
    path = save_test_suite(suite)
    return suite, str(path)
