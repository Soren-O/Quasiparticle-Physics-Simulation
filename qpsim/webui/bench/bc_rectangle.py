"""A rectangle with all four sides absorbing: the ``sin sin`` mode decays.

The two-dimensional counterpart of the strip case. With ``f = 0`` on every
face the eigenmodes are ``sin(m pi x / L_x) sin(n pi y / L_y)`` with
``k^2 = (m pi / L_x)^2 + (n pi / L_y)^2``, and, as on the strip, the
cell-centred product is an exact eigenvector of the discrete absorbing
operator, so the residual is the anisotropic 5-point stencil error the
diffusion benchmark documents -- at the same 16x32 mesh it is the same
2.7e-3, now arising from the absorbing rim rather than the reflective one.

Unlike the strip cases, the rim default IS the condition here: a 2-D mask
has no long sides to leak through, so ``boundary.kind = absorbing`` applies
to all four segments and no per-edge override is needed. This is also the
only case where the rim default's behaviour on every edge of a real 2-D
device is checked against a closed form.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.bench._transport import decay_rate_curve
from qpsim.webui.benchmarks import Benchmark, Curve, register

CASE_OVERRIDES: dict[str, Any] = {
    "material.name": "Al",
    "material.Delta_0": 180.0,
    "material.T_c": 1.18,
    "material.tau_0": 438.0,
    "material.D_0": 60.0,
    "material.dynes_gamma": 0.0,
    "T_bath": 0.1,
    "grid.min_factor": 1.0,
    "grid.max_factor": 4.0,
    "grid.num_bins": 32,
    "geometry.kind": "rectangle",
    "geometry.rows": 16,
    "geometry.cols": 32,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "absorbing",
    "diffusion_model": "A1",
    "collisions.scattering": False,
    "collisions.recombination": False,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "thermal_bath",
    "self_consistent_gap": False,
    "initial.kind": "absolute",
    "initial.expression": (
        "params['a'] * np.sin(params['m'] * np.pi * x) * np.sin(params['n'] * np.pi * y)"
    ),
    "initial.params": {"a": 0.2, "m": 1.0, "n": 1.0},
    # A corner cell has two absorbing faces on top of its two neighbours, so
    # its exit rate is twice the interior's; dt is halved against the
    # diffusion case to keep the Crank-Nicolson substep count at 1.
    "dt": 0.025,
    "max_time": 12.0,
    "stop_tol": 0.0,
    "snapshot_interval": 1.0,
}


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    if str(setup.boundary.kind) != "absorbing" or setup.boundary.per_edge:
        raise ValueError(
            "The rim must be absorbing on every side with no per-edge override: "
            "the sin·sin product is an eigenmode only when all four faces hold f = 0."
        )
    mask = np.asarray(arrays["mask"]).astype(bool)
    if not mask.all():
        raise ValueError("The product mode is an eigenmode of a FULL rectangle only.")
    params = dict(getattr(setup.initial, "params", {}) or {})
    m, n = float(params.get("m", 0.0)), float(params.get("n", 0.0))
    if m <= 0.0 or n <= 0.0:
        raise ValueError("initial.params needs positive mode numbers m and n.")
    rows, cols = np.nonzero(mask)
    nrow, ncol = mask.shape
    dx = float(setup.geometry.mesh_size_um)
    phi = np.sin(m * np.pi * (cols + 0.5) / ncol) * np.sin(n * np.pi * (rows + 0.5) / nrow)
    k2 = (m * np.pi / (ncol * dx)) ** 2 + (n * np.pi / (nrow * dx)) ** 2
    return decay_rate_curve(
        setup, arrays, phi, k2, baseline=0.0, exact_shape=True,
        residual_label=(
            "The residual is the anisotropic 5-point stencil error "
            "|k_h² − k²|/k² with k_h² = (4/dx²)[sin²(k_x dx/2) + sin²(k_y dx/2)]."
        ),
    )


register(Benchmark(
    name="bc-rectangle",
    title="Absorbing rectangle: sin·sin mode decay",
    tier="T1",
    formula_latex=(
        r"f=A(E)\,\sin\!\frac{m\pi x}{L_x}\sin\!\frac{n\pi y}{L_y}\,e^{-\lambda(E)t},"
        r"\qquad f=0\ \text{on every face},\qquad"
        r"\boxed{\lambda(E)=D_{\rm eff}(E)\Big[\Big(\frac{m\pi}{L_x}\Big)^{2}"
        r"+\Big(\frac{n\pi}{L_y}\Big)^{2}\Big]}"
    ),
    headline_latex=r"\lambda(E)=D_{\rm eff}(E)\,[(m\pi/L_x)^{2}+(n\pi/L_y)^{2}]",
    reason=(
        "With f = 0 on all four faces the separable sine product is an "
        "eigenmode of the Laplacian and, cell-centred, an exact eigenvector of "
        "the discrete absorbing operator; it decays at D_eff(E) k². This is the "
        "rim DEFAULT checked on every edge of a 2-D device, which no other "
        "case does."
    ),
    rel_tol=3.5e-3,
    convergence=(
        "Headline case: 16x32 cells at 4 μm (128x64 μm), every face absorbing, "
        "32 energy bins, dt = 0.025 ns, 480 steps, 13 frames over 12 ns, 11 s. "
        "Measured max relative error 2.7275e-03 -- the diffusion benchmark's "
        "number at the same mesh, because k is the same (m, n) = (1, 1) and the "
        "anisotropic stencil error does not know which rim produced the mode."
        "\n\nSPACE (2026-09-02): dx halved at fixed 128x64 μm, T = 12 ns, "
        "NE = 16, dt as dx².\n"
        "    8x16  dx=8 μm  dt=0.1 ns      1.0869e-02\n"
        "   16x32  dx=4 μm  dt=0.025 ns    2.7275e-03   order 1.995\n"
        "   32x64  dx=2 μm  dt=0.00625 ns  6.8250e-04   order 1.999\n"
        "Second order. Single-exponential residual over the trace: 5e-15."
    ),
    modes=("kinetics",),
    build=_build,
    activity=(
        "Switch the rim to reflective and the product of sines is no eigenmode: "
        "the projected amplitude leaks into the cosines and the single-exponential "
        "residual reports it."
    ),
    caveat=(
        "Same stencil residual as the diffusion benchmark at the same mesh; "
        "the rim changes the mode, not the discretisation error."
    ),
))
