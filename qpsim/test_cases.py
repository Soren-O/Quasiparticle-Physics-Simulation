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
from .solver import run_2d_crank_nicolson
from .storage import save_test_suite


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


def _frame_to_jsonable(frame: np.ndarray) -> list[list[float | None]]:
    payload: list[list[float | None]] = []
    for row in frame:
        payload.append([None if np.isnan(v) else float(v) for v in row])
    return payload


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
        times, frames, _, _ = run_2d_crank_nicolson(
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
    edge_conditions = _uniform_edge_conditions(edges, dirichlet_zero)
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]

    cases: list[TestCaseResultData] = []
    for mode_index, (m, n) in enumerate(modes, start=1):
        phi = np.sin(m * np.pi * gx / lx) * np.sin(n * np.pi * gy / ly)
        initial_field = phi.copy()
        times, frames, _, _ = run_2d_crank_nicolson(
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
        lam_sq = (m * np.pi / lx) ** 2 + (n * np.pi / ly) ** 2
        t_arr = np.asarray(times, dtype=float)
        analytic_frames = [phi * np.exp(-diffusion_coefficient * lam_sq * t) for t in t_arr]

        cases.append(
            TestCaseResultData(
                case_id=f"rectangle_mode_{m}_{n}",
                title=f"Rectangle Mode ({m}, {n})",
                boundary_label="Dirichlet zero on all rectangle edges",
                formula_latex=(
                    rf"u(x,y,t)=\sin\left(\frac{{{m}\pi x}}{{L_x}}\right)"
                    rf"\sin\left(\frac{{{n}\pi y}}{{L_y}}\right)"
                    rf"e^{{-D[(\frac{{{m}\pi}}{{L_x}})^2+(\frac{{{n}\pi}}{{L_y}})^2]t}}"
                ),
                initial_condition_latex=rf"u(x,y,0)=\sin\left(\frac{{{m}\pi x}}{{L_x}}\right)\sin\left(\frac{{{n}\pi y}}{{L_y}}\right)",
                description=f"2D rectangular eigenmode benchmark case {mode_index}.",
                x=[],
                times=t_arr.tolist(),
                simulated=[_frame_to_jsonable(frame) for frame in frames],
                analytic=[_frame_to_jsonable(frame) for frame in analytic_frames],
                metadata={
                    "geometry_id": "rectangle_2d",
                    "view_mode": "heatmap2d",
                    "grid_shape": [ny, nx],
                    "mode_m": m,
                    "mode_n": n,
                    "diffusion_coefficient": diffusion_coefficient,
                    "dx": dx,
                    "dt": dt,
                    "total_time": total_time,
                },
            )
        )

    preview = np.pad(mask.astype(int), pad_width=3, mode="constant", constant_values=0)
    return TestGeometryGroupData(
        geometry_id="rectangle_2d",
        title="2D Rectangle",
        description="Non-1D rectangular diffusion with analytic sine-mode solutions.",
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
    def f(lam: float) -> float:
        return special.j0(lam * inner_r) * special.y0(lam * outer_r) - special.y0(lam * inner_r) * special.j0(
            lam * outer_r
        )

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
    edge_conditions = _uniform_edge_conditions(edges, dirichlet_zero)

    y_idx, x_idx = np.indices(mask.shape, dtype=float)
    x = x_idx + 0.5
    y = y_idx + 0.5
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    cases: list[TestCaseResultData] = []
    for mode_index in [1, 2, 3, 4]:
        lam = _annulus_eigenvalue(inner_r=inner_r, outer_r=outer_r, mode_index=mode_index)
        phi = special.j0(lam * r) * special.y0(lam * inner_r) - special.y0(lam * r) * special.j0(lam * inner_r)
        phi[~mask] = 0.0
        amp = np.max(np.abs(phi[mask]))
        if amp > 0:
            phi = phi / amp

        initial_field = phi.copy()
        times, frames, _, _ = run_2d_crank_nicolson(
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
                case_id=f"donut_radial_mode_{mode_index}",
                title=f"Donut Radial Mode {mode_index}",
                boundary_label="Dirichlet zero on inner and outer polygon boundaries",
                formula_latex=(
                    r"u(r,t)=\phi(r)e^{-D\lambda_k^2 t},\ "
                    r"\phi(r)=J_0(\lambda_k r)Y_0(\lambda_k a)-Y_0(\lambda_k r)J_0(\lambda_k a)"
                ),
                initial_condition_latex=r"u(r,0)=\phi(r)",
                description=f"Polygon annulus benchmark using radial Bessel mode k={mode_index}.",
                x=[],
                times=t_arr.tolist(),
                simulated=[_frame_to_jsonable(frame) for frame in frames_with_nan],
                analytic=[_frame_to_jsonable(frame) for frame in analytic_frames],
                metadata={
                    "geometry_id": "polygon_donut",
                    "view_mode": "heatmap2d",
                    "grid_shape": [ny, nx],
                    "mode_index": mode_index,
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
        description="Polygonal annulus geometry compared against radial Bessel-mode analytic solutions.",
        view_mode="heatmap2d",
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

    geometry_groups = [strip_group, rectangle_group, donut_group]
    return TestSuiteData(
        suite_id=uuid.uuid4().hex[:12],
        created_at=utc_now_iso(),
        cases=[],
        geometry_groups=geometry_groups,
        metadata={"format_version": 2},
    )


def generate_and_save_test_suite() -> tuple[TestSuiteData, str]:
    suite = generate_test_suite()
    path = save_test_suite(suite)
    return suite, str(path)
