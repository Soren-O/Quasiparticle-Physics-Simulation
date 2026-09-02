"""Dirichlet ends held at ``g != 0``: the sine mode about ``g`` decays.

The inhomogeneous branch of the transport layer -- a face held at a fixed
non-zero value -- is the least exercised code in it. With both ends of a strip
held at ``g``, the steady state is ``f = g`` everywhere and every departure
from it is a sum of sines; ``f - g = A sin(m pi x / L) e^{-lambda t}`` with the
same rate as the absorbing case. The engine's Dirichlet face is a ghost cell
``2g - f_end``, which places ``f = g`` exactly on the face, so the cell-centred
sine is again an exact discrete eigenvector and the residual is the stencil's.

What is checked beyond the absorbing case: that the baseline the mode decays
towards is the value the rim was given, not zero. The benchmark subtracts the
SETUP's ``g`` before projecting, never a mean read from the run; a Dirichlet
face that silently acted as absorbing would then leave a non-decaying offset
in the projection and fail the single-exponential residual.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.bench._transport import decay_rate_curve, strip_axis
from qpsim.webui.benchmarks import Benchmark, Curve, register

_G = 0.05

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
        "left": {"kind": "dirichlet", "value": _G, "aux_value": None},
        "right": {"kind": "dirichlet", "value": _G, "aux_value": None},
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
    "initial.kind": "absolute",
    "initial.expression": "params['g'] + params['a'] * np.sin(params['m'] * np.pi * x)",
    "initial.params": {"g": _G, "a": 0.2, "m": 1.0},
    "dt": 0.05,
    "max_time": 40.0,
    "stop_tol": 0.0,
    "snapshot_interval": 2.0,
}


def _dirichlet_value(setup: Any) -> float:
    if str(setup.boundary.kind) != "reflective":
        raise ValueError(
            "The rim default must be reflective: a rim-wide Dirichlet condition "
            "on a one-row strip also acts through the long sides."
        )
    ends = dict(setup.boundary.per_edge)
    if set(ends) != {"left", "right"} or any(str(v.kind) != "dirichlet" for v in ends.values()):
        raise ValueError(
            "Both ends, and only the ends, must be Dirichlet: "
            "boundary.per_edge = {left: dirichlet g, right: dirichlet g}."
        )
    values = {float(v.value) for v in ends.values()}
    if len(values) != 1:
        raise ValueError("Both ends must hold the same value g for f = g to be the steady state.")
    g = values.pop()
    params = dict(getattr(setup.initial, "params", {}) or {})
    if "g" not in params or abs(float(params["g"]) - g) > 1e-12:
        raise ValueError(
            "initial.params['g'] must equal the Dirichlet value, so the "
            "prepared state is g + mode and the mode decays towards g."
        )
    return g


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    g = _dirichlet_value(setup)
    params = dict(getattr(setup.initial, "params", {}) or {})
    m = float(params.get("m", 0.0))
    if m <= 0.0:
        raise ValueError("initial.params needs a positive mode number m.")
    mask = np.asarray(arrays["mask"]).astype(bool)
    i = strip_axis(mask)
    n = i.size
    dx = float(setup.geometry.mesh_size_um)
    phi = np.sin(m * np.pi * (i + 0.5) / n)
    k2 = (m * np.pi / (n * dx)) ** 2
    return decay_rate_curve(
        setup, arrays, phi, k2, baseline=g, exact_shape=True,
        residual_label=(
            "The residual is the 3-point stencil's eigenvalue error, exactly as "
            "for the absorbing rim: the ghost 2g − f_end puts f = g on the face "
            "and the cell-centred sine stays an exact discrete eigenvector."
        ),
    )


register(Benchmark(
    name="bc-dirichlet",
    title="Dirichlet ends at g ≠ 0: sine-mode decay towards g",
    tier="T1",
    formula_latex=(
        r"f(E,x,t)=g+A(E)\,\sin\!\frac{m\pi x}{L}\,e^{-\lambda(E)t},\qquad "
        r"f(E,0,t)=f(E,L,t)=g,\qquad"
        r"\boxed{\lambda(E)=D_{\rm eff}(E)\Big(\frac{m\pi}{L}\Big)^{2}}"
    ),
    headline_latex=r"f-g \propto \sin\frac{m\pi x}{L}\,e^{-D_{\rm eff}(E)(m\pi/L)^{2}t}",
    reason=(
        "A face held at g makes f = g the steady state and the sines the modes "
        "of the departure from it. The benchmark subtracts the rim's own g "
        "before projecting, so a Dirichlet face that acted as absorbing, or "
        "held the wrong value, would leave an offset that does not decay and "
        "fail the single-exponential check."
    ),
    rel_tol=1.2e-3,
    convergence=(
        "Headline case: 32 cells at 4 μm, g = 0.05 at both ends, 32 energy "
        "bins, dt = 0.05 ns, 21 frames over 40 ns, 7 s. Measured max relative "
        "error 8.0293e-04 -- identical to the absorbing case to ten digits, as "
        "it must be: the operator is the same and only the source differs, and "
        "the source is exact for a constant.\n\n"
        "The refinement is the absorbing case's (order 1.999, 2.000 for 16, 32, "
        "64 cells at fixed L), because the discrete operators are identical; "
        "what this case adds is the check that the mode decays towards the "
        "rim's g and not towards zero, which a Dirichlet face that acted as "
        "absorbing would fail through the single-exponential residual."
    ),
    modes=("kinetics",),
    build=_build,
    activity=(
        "Set g = 0 and this is the absorbing case; the non-zero g is what "
        "exercises the inhomogeneous source the Dirichlet face adds."
    ),
    caveat="Reads the recorded frames and the per-edge overrides; rim default reflective.",
))
