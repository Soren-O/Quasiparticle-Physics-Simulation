"""An annulus with both rims absorbing: the Bessel cross-product mode.

A device that is not a rectangle, stated as two polygons (an outer circle
and an inner one wound the other way), with ``f = 0`` on both rims. The
radially symmetric eigenmodes are

    phi(r) = J0(k r) Y0(k b) - J0(k b) Y0(k r),

with ``k`` the first root of the cross product ``J0(k a) Y0(k b) - J0(k b)
Y0(k a) = 0``, and the fundamental decays at ``D_eff(E) k^2``. The closed
form reads no engine array: ``k`` is a root of Bessel functions of the two
radii, and the mode is evaluated at the cell centres from the mask's own
centroid.

Two things are different from the rectangle cases and both are said plainly.
The mask is a STAIRCASE approximation of two circles, so the discrete rim
sits within a cell of the true one and the eigenvalue error is first order
in ``dx / (b - a)`` rather than the stencil's second order -- the tolerance
is set by measurement at the shipped mesh, with the refinement recorded. And
the prepared state is a smooth radial bump, not the mode itself (Bessel
functions are not available to the initial-condition expression), so the
projection onto ``phi`` carries a little of the higher radial modes at first;
those decay at least four times faster, the first frames are skipped, and the
fit residual reports what remains.

This is the case the geometry waves were for: a non-rectangular device from
typed outlines, its rim addressed by the mask, checked against a closed form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, y0

from qpsim.webui.bench._transport import decay_rate_curve
from qpsim.webui.benchmarks import Benchmark, Curve, register

_A = 10.0       # inner radius (um)
_B = 40.0       # outer radius (um)
_DX = 2.5
_SIDES = 64


def _circle(radius: float, centre: float, reverse: bool) -> list[list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, _SIDES, endpoint=False)
    if reverse:
        angles = angles[::-1]
    return [
        [round(centre + radius * float(np.cos(t)), 6), round(centre + radius * float(np.sin(t)), 6)]
        for t in angles
    ]


def wavenumber(a: float, b: float, root: int = 1) -> float:
    """The ``root``-th root of J0(ka)Y0(kb) - J0(kb)Y0(ka): radial mode ``root``.

    The roots sit near ``root * pi / (b - a)``; the bracket is the half-width
    window about that, across which the cross product changes sign exactly
    once for the radii this benchmark uses (brentq checks the sign change).
    """
    cross = lambda k: j0(k * a) * y0(k * b) - j0(k * b) * y0(k * a)  # noqa: E731
    centre = root * np.pi / (b - a)
    lo, hi = centre - 0.5 * np.pi / (b - a), centre + 0.5 * np.pi / (b - a)
    return float(brentq(cross, lo, hi))


# The outline is centred at (b, b) in layout units. The rasteriser pads the
# window by one cell, so in MASK coordinates (what x_um, y_um are) the centre
# sits at (b + dx, b + dx); the benchmark itself never uses that number -- it
# takes the centre from the mask's centroid, which is exact by symmetry.
_CENTRE_MASK = _B + _DX

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
    "grid.num_bins": 16,
    "geometry.kind": "polygons",
    "geometry.mesh_size_um": _DX,
    "geometry.polygons": [_circle(_B, _B, reverse=False), _circle(_A, _B, reverse=True)],
    "geometry.require_connected": True,
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
    # A radial bump that vanishes at both rims: a * cos(pi (r - r_mid)/(b - a)),
    # clipped at zero for the few cell centres the staircase rim lets sit a
    # hair outside the true ring.
    "initial.kind": "absolute",
    "initial.expression": (
        "params['a'] * np.maximum(np.cos(np.pi * (np.sqrt((x_um - params['cx'])**2 "
        "+ (y_um - params['cy'])**2) - params['rmid']) / params['width']), 0.0)"
    ),
    "initial.params": {
        "a": 0.2, "cx": _CENTRE_MASK, "cy": _CENTRE_MASK,
        "rmid": 0.5 * (_A + _B), "width": _B - _A,
    },
    # Long enough for the bump's leakage into the second radial mode to decay
    # below the fit's target (see _transport.LEAK_TARGET) in every bin above
    # ~1.5 Delta -- that takes ~5 ns at the gap-edge end -- with a fitting
    # window left over. The fundamental has fallen to ~1% by 6.5 ns, which a
    # log fit resolves without difficulty.
    "dt": 0.01,
    "max_time": 6.5,
    "stop_tol": 0.0,
    "snapshot_interval": 0.125,
}


def _radii(setup: Any) -> tuple[float, float]:
    polys = setup.geometry.polygons or []
    if str(setup.geometry.kind) != "polygons" or len(polys) != 2:
        raise ValueError(
            "This benchmark needs geometry.kind = 'polygons' with exactly two "
            "outlines: the outer and the inner circle."
        )
    radii = []
    for poly in polys:
        pts = np.asarray(poly, dtype=float)
        centre = pts.mean(axis=0)
        r = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
        if np.ptp(r) > 1e-6 * r.mean():
            raise ValueError("Each outline must be a circle (equal radii about its centre).")
        radii.append(float(r.mean()))
    b, a = max(radii), min(radii)
    if a <= 0.0 or b <= a:
        raise ValueError("The inner radius must be positive and smaller than the outer.")
    return a, b


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    if str(setup.boundary.kind) != "absorbing" or setup.boundary.per_edge:
        raise ValueError(
            "Both rims must be absorbing: boundary.kind = 'absorbing' with no override."
        )
    a, b = _radii(setup)
    mask = np.asarray(arrays["mask"]).astype(bool)
    rows, cols = np.nonzero(mask)
    dx = float(setup.geometry.mesh_size_um)
    x = (cols + 0.5) * dx
    y = (rows + 0.5) * dx
    cx, cy = float(x.mean()), float(y.mean())          # exact by symmetry
    r = np.hypot(x - cx, y - cy)
    k = wavenumber(a, b)
    k_next = wavenumber(a, b, root=2)
    phi = j0(k * r) * y0(k * b) - j0(k * b) * y0(k * r)
    return decay_rate_curve(
        setup, arrays, phi, k * k, baseline=0.0, exact_shape=False,
        transient_k2=k_next * k_next,
        residual_label=(
            "The residual is first order in dx/(b − a): the staircase rim sits "
            "within a cell of the true circles."
        ),
    )


register(Benchmark(
    name="bc-annulus",
    title="Absorbing annulus: Bessel cross-product mode decay",
    tier="T1",
    formula_latex=(
        r"\phi(r)=J_0(kr)\,Y_0(kb)-J_0(kb)\,Y_0(kr),\qquad "
        r"J_0(ka)\,Y_0(kb)-J_0(kb)\,Y_0(ka)=0,\qquad"
        r"\boxed{\lambda(E)=D_{\rm eff}(E)\,k^{2}}"
    ),
    headline_latex=r"J_0(ka)Y_0(kb)-J_0(kb)Y_0(ka)=0,\qquad \lambda(E)=D_{\rm eff}(E)\,k^{2}",
    reason=(
        "On a ring with both rims absorbing the fundamental radial mode is the "
        "Bessel cross product with k the first root of its boundary equation; "
        "it decays at D_eff(E) k². The device is two typed outlines rasterised "
        "onto the mask, so this is the geometry machinery -- polygons, holes, "
        "an inner rim -- checked against a closed form."
    ),
    # 3e-2 covers the FINER mesh's error, not the shipped one's: at 2.5 μm the
    # staircase-rim error (positive) and the stencil error (negative) nearly
    # cancel and the measured 5.2e-3 is luck, not accuracy; at 1.25 μm the
    # error is 2.2e-2. A tolerance set at the shipped mesh alone would be a
    # claim the refinement contradicts.
    rel_tol=3.0e-2,
    convergence=(
        "Headline case: a = 10 μm, b = 40 μm as two 64-gons at 2.5 μm (760 cells), "
        "both rims absorbing, 16 energy bins, dt = 0.01 ns, 650 steps, 53 frames over "
        "6.5 ns, 10 s. 15 of 16 bins fitted (one gap-edge bin's transient outlasts "
        "the run); measured max relative error 5.1562e-03, single-exponential "
        "residual 5.4e-07; the spread of λ_sim/λ_exact across energy is 2e-06, so "
        "the whole error is one number -- a wavenumber, i.e. the rim.\n\n"
        "SPACE (2026-09-02): dx halved at fixed radii, T = 4-6.5 ns, NE = 8-16, dt as "
        "dx², SIGNED mean of λ_sim/λ_exact − 1 (uniform across energy to 1e-4):\n"
        "   dx=5 μm      −6.50e-02   (ring 6 cells wide)\n"
        "   dx=2.5 μm    +5.15e-03   (sign change: NOT convergence)\n"
        "   dx=1.25 μm   +2.20e-02\n"
        "   dx=0.625 μm  +6.80e-03   order 1.69 over the last halving\n"
        "The staircase rim error is POSITIVE (the effective ring is narrower than "
        "the circles, k_eff > k) and the 5-point stencil error is negative, ~(k dx)²/12; "
        "at 2.5 μm they nearly cancel, which is why the shipped case reads 5e-3 and "
        "the tolerance is nonetheless 3e-2: it is set from the finer mesh, not from "
        "the lucky one. From 1.25 μm down the error falls at better than first order; "
        "the asymptotic order of a staircase rim is one."
    ),
    modes=("kinetics",),
    build=_build,
    activity=(
        "Make the inner rim reflective and the fundamental changes to the "
        "mixed-rim mode with a visibly smaller k; the projection onto the "
        "absorbing-rim mode then decays at the wrong rate."
    ),
    caveat=(
        "FIRST ORDER in dx/(b − a) from the staircase rim, not second; the "
        "tolerance is a measured statement at the shipped mesh. The prepared "
        "state is a bump, not the mode: its leakage into the second radial mode "
        "is measured at t = 0 and each bin's fit starts once that has decayed "
        "at D_eff(E)(k₂² − k²); bins whose transient outlasts the run are "
        "excluded and counted in the note."
    ),
))
