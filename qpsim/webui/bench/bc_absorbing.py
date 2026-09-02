"""Absorbing rim: a sine mode on a strip decays at ``D_eff(E) (m pi / L)^2``.

The transport equation on a strip with both ends absorbing (``f = 0`` at the
end faces) has eigenmodes ``sin(m pi x / L)``. The engine's absorbing face is
a mirrored ghost cell of opposite sign, which places ``f = 0`` exactly on the
face half a cell beyond the end centre -- and the cell-centred sine is then an
exact eigenvector of the discrete operator, with the same stencil eigenvalue
``k_h^2 = (4/dx^2) sin^2(k dx/2)`` the diffusion benchmark documents. So the
residual is the stencil's ``|k_h^2 - k^2|/k^2 ~ (k dx)^2/12`` and nothing
else, and it halves four times when the mesh halves.

Why a strip and not the rim default: on a one-row mask a rim-wide
``absorbing`` also absorbs through the two long sides, so the 1-D reduction
leaks. The case therefore keeps the rim reflective and overrides the two END
segments by their direction aliases -- which is also the per-edge machinery
this benchmark is the first closed-form check of.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.bench._transport import decay_rate_curve, strip_axis
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
    "geometry.rows": 1,
    "geometry.cols": 32,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "boundary.per_edge": {
        "left": {"kind": "absorbing", "value": 0.0, "aux_value": None},
        "right": {"kind": "absorbing", "value": 0.0, "aux_value": None},
    },
    "diffusion_model": "A1",
    "collisions.scattering": False,
    "collisions.recombination": False,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "thermal_bath",
    "self_consistent_gap": False,
    # f = a sin(m pi x): non-negative for m = 1, so the [0, 1] clip never fires.
    "initial.kind": "absolute",
    "initial.expression": "params['a'] * np.sin(params['m'] * np.pi * x)",
    "initial.params": {"a": 0.2, "m": 1.0},
    "dt": 0.05,
    "max_time": 40.0,
    "stop_tol": 0.0,
    "snapshot_interval": 2.0,
}


def _require_absorbing_ends(setup: Any) -> None:
    if str(setup.boundary.kind) != "reflective":
        raise ValueError(
            "The rim default must be reflective: on a one-row strip a rim-wide "
            "absorbing condition also absorbs through the long sides, and the "
            "sine is then not an eigenmode of what is being solved."
        )
    ends = dict(setup.boundary.per_edge)
    if set(ends) != {"left", "right"} or any(str(v.kind) != "absorbing" for v in ends.values()):
        raise ValueError(
            "Both ends, and only the ends, must be absorbing: "
            "boundary.per_edge = {left: absorbing, right: absorbing}."
        )


def _mode_number(setup: Any) -> float:
    params = dict(getattr(setup.initial, "params", {}) or {})
    if "m" not in params or float(params["m"]) <= 0.0:
        raise ValueError("initial.params needs a positive mode number m.")
    return float(params["m"])


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    _require_absorbing_ends(setup)
    mask = np.asarray(arrays["mask"]).astype(bool)
    i = strip_axis(mask)
    n = i.size
    m = _mode_number(setup)
    dx = float(setup.geometry.mesh_size_um)
    length = n * dx
    phi = np.sin(m * np.pi * (i + 0.5) / n)
    k2 = (m * np.pi / length) ** 2
    return decay_rate_curve(
        setup, arrays, phi, k2, baseline=0.0, exact_shape=True,
        residual_label=(
            "The residual is the 3-point stencil's eigenvalue error "
            "|k_h² − k²|/k² with k_h² = (4/dx²) sin²(k dx/2): the cell-centred "
            "sine is an exact eigenvector of the discrete absorbing operator."
        ),
    )


register(Benchmark(
    name="bc-absorbing",
    title="Absorbing ends: sine-mode decay on a strip",
    tier="T1",
    formula_latex=(
        r"f(E,x,t)=A(E)\,\sin\!\frac{m\pi x}{L}\,e^{-\lambda(E)t},\qquad "
        r"f(E,0,t)=f(E,L,t)=0,\qquad"
        r"\boxed{\lambda(E)=D_{\rm eff}(E)\Big(\frac{m\pi}{L}\Big)^{2}},\qquad "
        r"D_{\rm eff}=D_0\frac{\sqrt{E^{2}-\Delta^{2}}}{E}\ \ (A1)"
    ),
    headline_latex=r"\lambda(E)=D_{\rm eff}(E)\,(\frac{m\pi}{L})^{2}",
    reason=(
        "With both end faces held at f = 0 the strip's eigenmodes are sines, "
        "and the engine's absorbing face (a mirrored ghost of opposite sign) "
        "puts f = 0 exactly on the face half a cell beyond the end centre -- so "
        "the cell-centred sine is an exact eigenvector of the discrete operator "
        "and decays as one exponential at D_eff(E) k². The energy dependence of "
        "that rate is the N₁ dressing; the rim only sets k."
    ),
    # 1.2e-3 against a measured 8.0293e-4 at the shipped mesh: the headroom
    # covers a mesh that is not exactly this one, not noise -- the residual is
    # deterministic to 13 digits.
    rel_tol=1.2e-3,
    convergence=(
        "Headline case: 32 cells at 4 μm (L = 128 μm), 32 energy bins, "
        "dt = 0.05 ns, 800 steps, 21 frames over 40 ns, 7 s. Measured max "
        "relative error 8.0293e-04; the stencil prediction "
        "|k_h² − k²|/k² = (4/dx²)sin²(k dx/2)/k² − 1 at k = π/128 μm⁻¹ is "
        "8.0294e-04.\n\n"
        "SPACE (2026-09-02): dx halved at fixed L = 128 μm, T = 40 ns, NE = 16, "
        "dt scaled as dx² to hold the CN substep count at 1.\n"
        "   16 cells  dx=8 μm  dt=0.2 ns     3.2085e-03\n"
        "   32 cells  dx=4 μm  dt=0.05 ns    8.0293e-04   order 1.999\n"
        "   64 cells  dx=2 μm  dt=0.0125 ns  2.0078e-04   order 2.000\n"
        "Second order, as the 3-point stencil with an exactly-placed absorbing "
        "face must be. Single-exponential residual over the trace: 2e-14."
    ),
    modes=("kinetics",),
    build=_build,
    activity=(
        "Switch the ends back to reflective and the same sine is NOT an "
        "eigenmode: its projection decays at a different rate and leaks into "
        "the cosines, which the single-exponential residual reports."
    ),
    caveat=(
        "Reads the recorded frames and the per-edge overrides. The rim default "
        "must stay reflective (see the module docstring). The residual is the "
        "stencil's known (k dx)²/12 offset and is not noise."
    ),
))
