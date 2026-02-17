from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .geometry import extract_edge_segments
from .models import BoundaryCondition, TestCaseResultData, TestSuiteData, utc_now_iso
from .solver import run_2d_crank_nicolson
from .storage import save_test_suite


@dataclass
class _CaseDefinition:
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


def _build_case_definitions(length: float) -> list[_CaseDefinition]:
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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
        _CaseDefinition(
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


def generate_test_suite(
    nx: int = 100,
    dx: float = 1.0,
    diffusion_coefficient: float = 1.0,
    dt: float = 0.25,
    total_time: float = 40.0,
    store_every: int = 2,
) -> TestSuiteData:
    if nx < 8:
        raise ValueError("nx must be at least 8 for test generation.")
    if abs(dx - 1.0) > 1e-9:
        # Keep this visible because the user requested 1D mesh_size = 1 cases.
        raise ValueError("Test suite expects mesh_size (dx) = 1.0.")

    length = nx * dx
    cases: list[TestCaseResultData] = []

    definitions = _build_case_definitions(length)
    for case_def in definitions:
        x_cell_centers = (np.arange(nx, dtype=float) + 0.5) * dx
        u0 = case_def.init_fn(x_cell_centers, length, diffusion_coefficient)
        # Effective-1D test geometry: one-cell-thick strip solved by the full 2D engine.
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
        x = x_cell_centers
        simulated = np.asarray([frame[0, :] for frame in frames], dtype=float)

        t_arr = np.asarray(times, dtype=float)
        analytic = case_def.analytic_fn(x, t_arr, length, diffusion_coefficient)

        cases.append(
            TestCaseResultData(
                case_id=case_def.case_id,
                title=case_def.title,
                boundary_label=case_def.boundary_label,
                formula_latex=case_def.formula_latex,
                initial_condition_latex=case_def.initial_latex,
                description=case_def.description,
                x=x.tolist(),
                times=t_arr.tolist(),
                simulated=simulated.tolist(),
                analytic=np.asarray(analytic, dtype=float).tolist(),
                metadata={
                    "diffusion_coefficient": diffusion_coefficient,
                    "dx": dx,
                    "dt": dt,
                    "total_time": total_time,
                },
            )
        )

    return TestSuiteData(
        suite_id=uuid.uuid4().hex[:12],
        created_at=utc_now_iso(),
        cases=cases,
    )


def generate_and_save_test_suite() -> tuple[TestSuiteData, str]:
    suite = generate_test_suite()
    path = save_test_suite(suite)
    return suite, str(path)
