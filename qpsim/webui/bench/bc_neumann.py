"""Neumann ends with a prescribed flux: the steady state is a line.

Hold the outward normal derivative at ``q`` on one end and ``-q`` on the
other and the strip carries a constant current: as much enters one end as
leaves the other, the mean is conserved, and the steady state is the linear
profile ``f(E, x) = <f(E)> + s (x - L/2)`` with slope ``s = q_R = -q_L``
(``value`` is the OUTWARD normal derivative at the face, in 1/um). The slope
does not depend on the diffusivity; the time to reach it does, at
``D_eff(E) (pi/L)^2``, so the bins near the gap edge, where ``D_eff`` goes to
zero, are still relaxing when the run ends and are excluded by that
prediction rather than by looking at the answer.

This checks the inhomogeneous Neumann branch -- the source the engine adds
for a prescribed flux -- against an exact profile. The discrete steady state
is exactly linear (the 3-point stencil annihilates linear functions and the
flux source is exact for one), so the only residual is the unfinished
transient of the slowest included bin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.bench._transport import (
    MIN_DECAY_EXPONENT,
    d_eff_per_bin,
    require_frames,
    require_no_collisions,
    require_uniform_gap,
    strip_axis,
)
from qpsim.webui.benchmarks import Benchmark, Curve, register

_Q = 0.005          # outward normal derivative at the right end (1/um)
_SETTLED = 16.0     # e-foldings of the slowest mode before a bin counts as steady

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
    "geometry.cols": 16,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "boundary.per_edge": {
        "left": {"kind": "neumann", "value": -_Q, "aux_value": None},
        "right": {"kind": "neumann", "value": _Q, "aux_value": None},
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
    # A flat start at f0 well inside [0, 1], so the line +-q L/2 fits without
    # the clip ever firing.
    "initial.kind": "absolute",
    "initial.expression": "params['f0'] + 0.0 * x",
    "initial.params": {"f0": 0.5},
    "dt": 0.1,
    "max_time": 160.0,
    "stop_tol": 0.0,
    "snapshot_interval": 16.0,
}


def _slope(setup: Any) -> float:
    if str(setup.boundary.kind) != "reflective":
        raise ValueError("The rim default must be reflective on a one-row strip.")
    ends = dict(setup.boundary.per_edge)
    if set(ends) != {"left", "right"} or any(str(v.kind) != "neumann" for v in ends.values()):
        raise ValueError(
            "Both ends, and only the ends, must be Neumann: "
            "boundary.per_edge = {left: neumann -q, right: neumann q}."
        )
    q_left, q_right = float(ends["left"].value), float(ends["right"].value)
    if abs(q_left + q_right) > 1e-12 * max(abs(q_left), abs(q_right), 1e-300) or q_right == 0.0:
        raise ValueError(
            "The two fluxes must be equal and opposite (right = -left, both "
            "non-zero) for a steady state to exist: with a net inflow the mean "
            "grows without bound and there is no profile to compare against."
        )
    return q_right


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    slope = _slope(setup)
    times, frames = require_frames(arrays, minimum=2)
    require_uniform_gap(setup, arrays)
    require_no_collisions(setup)
    mask = np.asarray(arrays["mask"]).astype(bool)
    i = strip_axis(mask)
    n = i.size
    dx = float(setup.geometry.mesh_size_um)
    length = n * dx
    x_um = (i + 0.5) * dx
    energies = np.asarray(arrays["E_bins"], dtype=float)
    gap = float(setup.material.Delta_0)

    # Which bins have had time to settle: the slowest departure from a flat
    # start under equal-and-opposite fluxes is the m = 1 cosine, at
    # D_eff (pi/L)^2. Predicted from the setup, not judged from the answer.
    span = float(times[-1] - times[0])
    rate = d_eff_per_bin(setup, energies) * (np.pi / length) ** 2
    settled = rate * span >= _SETTLED
    if not settled.any():
        raise ValueError(
            "No energy bin is predicted to reach its steady state over this "
            f"run (needs {_SETTLED:g} e-foldings of D_eff (π/L)²); lengthen "
            "max_time or shorten the strip."
        )
    if float(np.max(rate)) * span < MIN_DECAY_EXPONENT:
        raise ValueError("The closed form predicts no relaxation over this run.")

    final = frames[-1][settled]                              # (k, N)
    mean0 = frames[0][settled].mean(axis=1)                  # conserved per bin
    exact = mean0[:, None] + slope * (x_um - 0.5 * length)[None, :]
    worst_unsettled = (
        float(np.max(np.abs(frames[-1][~settled] - exact.mean(axis=0)[None, :])))
        if (~settled).any() else 0.0
    )
    note = (
        f"{int(settled.sum())} of {energies.size} energy bins are predicted to "
        f"have settled ({_SETTLED:g} e-foldings of D_eff (π/L)² over {span:g} ns) "
        f"and are compared; the {int((~settled).sum())} nearest the gap edge, "
        f"where D_eff → 0, are still relaxing and excluded by that prediction. "
        f"The profile is compared cell by cell against the exact line "
        f"⟨f⟩ + q (x − L/2) with ⟨f⟩ the conserved mean of the first frame. "
        f"The discrete steady state is exactly linear, so the residual is the "
        f"unfinished transient of the slowest included bin."
        + (
            f" Worst excluded-bin departure from the line: {worst_unsettled:.2e}."
            if (~settled).any() else ""
        )
    )
    labels = tuple(f"E = {e / gap:.3g} Δ" for e in energies[settled])
    return Curve(
        x=x_um,
        y_sim=final,
        y_analytic=exact,
        x_label="x (μm)",
        y_label="f(E, x) at the end of the run",
        series_labels=labels,
        note=note,
    )


register(Benchmark(
    name="bc-neumann",
    title="Neumann ends with a prescribed flux: the linear steady state",
    tier="T1",
    formula_latex=(
        r"\partial_n f\big|_{x=0}=-q,\quad \partial_n f\big|_{x=L}=q"
        r"\ \Rightarrow\ \boxed{f_\infty(E,x)=\langle f(E)\rangle+q\,(x-\tfrac{L}{2})},"
        r"\qquad \frac{d}{dt}\langle f\rangle=0"
    ),
    headline_latex=r"f_\infty(E,x)=\langle f(E)\rangle+q\,(x-L/2)",
    reason=(
        "Equal and opposite fluxes through the two ends conserve the mean and "
        "drive the strip to a line of slope q, independent of D_eff; only the "
        "time to get there depends on energy. The compared profile is the "
        "run's last frame against that exact line, bin by bin, for every bin "
        "predicted to have settled."
    ),
    # 1e-6 against a measured 2.85e-08: the residual is an unfinished
    # transient, e^-16 of the initial departure, not a discretisation error.
    rel_tol=1e-6,
    convergence=(
        "Headline case: 16 cells at 4 μm (L = 64 μm), q = ±0.005 μm⁻¹ on a "
        "flat start f0 = 0.5, 32 energy bins, dt = 0.1 ns, 1600 steps over "
        "160 ns, 13 s. 28 of 32 bins predicted settled (16 e-foldings of "
        "D_eff (π/L)²) and compared; measured max relative error 2.8495e-08 "
        "against the exact line.\n\n"
        "SETTLING (2026-09-02), NE = 16, same strip: the error is the unfinished "
        "transient of the slowest included bin and falls with the run length "
        "until it hits the solver floor.\n"
        "   T=160 ns   1.8037e-08\n"
        "   T=320 ns   1.2565e-12\n"
        "   T=640 ns   1.2571e-12   (floor)\n"
        "There is no spatial refinement to report: the discrete steady state "
        "under a prescribed flux is EXACTLY the line, at any mesh -- the "
        "3-point stencil annihilates a linear profile and the flux source is "
        "exact for one -- which is what the 1e-12 floor shows."
    ),
    modes=("kinetics",),
    build=_build,
    activity=(
        "Set q = 0 and the strip stays flat: the whole signal is the flux "
        "source the Neumann face adds, so a face that dropped it would fail "
        "at |q| L/2 / f0 ≈ 6%."
    ),
    caveat=(
        "A pointwise check of a converged profile. Bins near the gap edge are "
        "excluded by a prediction, not by inspection; their number is in the "
        "note. Not a check of the transient rate -- the homogeneous Neumann "
        "operator is the reflective one, which the diffusion benchmark covers."
    ),
))
