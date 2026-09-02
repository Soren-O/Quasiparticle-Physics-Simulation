"""Robin ends ``d_n f + beta f = gamma``: transcendental modes, first order.

The finite-transparency contact. With the same Robin condition on both ends
the steady state is ``f = gamma / beta`` and the departures are cosines and
sines about the strip's centre whose wavenumbers solve

    even:  k tan(k L / 2) = beta          f - f_inf ~ cos k (x - L/2)
    odd:   k cot(k L / 2) = -beta         f - f_inf ~ sin k (x - L/2)

Both are registered, as two benchmarks, because they exercise the two
symmetries of the same operator and the plan recorded them separately.

The engine applies ``beta`` to the CELL-CENTRE value of the end cell rather
than to an extrapolated face value (``grid/spatial_grid.py``, the robin
branch: diagonal ``-D beta/dx``). That is a first-order discretisation of
the condition, so unlike the absorbing and Dirichlet cases the prepared
continuum mode is NOT the exact discrete eigenvector, the rate error is
O(dx) rather than O(dx^2), and the plan's instruction is followed: the
tolerance is a first-order statement at the shipped mesh, and the measured
halving sequence is recorded rather than a second-order claim being made.

``gamma != 0`` on purpose: the inhomogeneous Robin source is the
least-exercised branch of the transport layer, and a baseline of
``gamma / beta`` that the mode must decay towards checks it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq

from qpsim.webui.bench._transport import decay_rate_curve, strip_axis
from qpsim.webui.benchmarks import Benchmark, Curve, register

_CELLS = 32
_DX = 4.0
_L = _CELLS * _DX
_BETA = 2.0 / _L            # beta L / 2 = 1: neither the reflective nor the absorbing limit
_F_INF = 0.3
_GAMMA = _BETA * _F_INF


def wavenumber(parity: str, beta: float, length: float) -> float:
    """First positive root of the even or odd Robin eigenvalue equation."""
    half = 0.5 * length
    if parity == "even":
        # k tan(k L/2) - beta: rises from -beta through zero before pi/L.
        return float(brentq(
            lambda k: k * np.tan(k * half) - beta, 1e-9, np.pi / length * (1.0 - 1e-9),
        ))
    if parity == "odd":
        # k cos(k L/2) + beta sin(k L/2): +beta at pi/L, negative at 2 pi/L.
        return float(brentq(
            lambda k: k * np.cos(k * half) + beta * np.sin(k * half),
            np.pi / length * (1.0 + 1e-9), 2.0 * np.pi / length * (1.0 - 1e-9),
        ))
    raise ValueError(f"parity must be 'even' or 'odd', not {parity!r}")


def _overrides(parity: str) -> dict[str, Any]:
    k = wavenumber(parity, _BETA, _L)
    shape = "np.cos" if parity == "even" else "np.sin"
    return {
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
        "geometry.cols": _CELLS,
        "geometry.mesh_size_um": _DX,
        "boundary.kind": "reflective",
        "boundary.per_edge": {
            "left": {"kind": "robin", "value": _BETA, "aux_value": _GAMMA},
            "right": {"kind": "robin", "value": _BETA, "aux_value": _GAMMA},
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
        # f_inf + a * mode(k (x_um - L/2)); the wavenumber is a transcendental
        # root, so it is carried as a number and CHECKED by the benchmark
        # against the root it re-derives from beta and L.
        "initial.kind": "absolute",
        "initial.expression": (
            f"params['finf'] + params['a'] * {shape}(params['k'] * (x_um - params['half']))"
        ),
        "initial.params": {"finf": _F_INF, "a": 0.2, "k": k, "half": 0.5 * _L},
        # The interior exit rate is 2 D/dx^2 = 7.5/ns and the Robin face adds
        # only beta D/dx = 0.23/ns, so dt = 0.1 keeps the CN substep count at
        # 1; the temporal error is three orders below the first-order rim.
        "dt": 0.1,
        "max_time": 100.0,
        "stop_tol": 0.0,
        "snapshot_interval": 5.0,
    }


CASE_OVERRIDES_EVEN: dict[str, Any] = _overrides("even")
CASE_OVERRIDES_ODD: dict[str, Any] = _overrides("odd")


def _robin_ends(setup: Any) -> tuple[float, float]:
    if str(setup.boundary.kind) != "reflective":
        raise ValueError("The rim default must be reflective on a one-row strip.")
    ends = dict(setup.boundary.per_edge)
    if set(ends) != {"left", "right"} or any(str(v.kind) != "robin" for v in ends.values()):
        raise ValueError(
            "Both ends, and only the ends, must be Robin: "
            "boundary.per_edge = {left: robin (β, γ), right: robin (β, γ)}."
        )
    betas = {float(v.value) for v in ends.values()}
    gammas = {float(v.aux_value or 0.0) for v in ends.values()}
    if len(betas) != 1 or len(gammas) != 1:
        raise ValueError("Both ends must carry the same β and γ for the modes below to apply.")
    beta, gamma = betas.pop(), gammas.pop()
    if beta <= 0.0:
        raise ValueError("β must be positive: β = 0 is the reflective rim and has no Robin mode.")
    return beta, gamma


def _make_build(parity: str) -> Any:
    def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
        beta, gamma = _robin_ends(setup)
        params = dict(getattr(setup.initial, "params", {}) or {})
        mask = np.asarray(arrays["mask"]).astype(bool)
        i = strip_axis(mask)
        n = i.size
        dx = float(setup.geometry.mesh_size_um)
        length = n * dx
        k = wavenumber(parity, beta, length)
        if "k" in params and abs(float(params["k"]) - k) > 1e-9 * k:
            raise ValueError(
                f"initial.params['k'] = {float(params['k']):.9g} is not the {parity} Robin "
                f"root {k:.9g} for β = {beta:g}, L = {length:g}: the prepared state is "
                "not the mode whose rate is predicted."
            )
        f_inf = gamma / beta
        if "finf" in params and abs(float(params["finf"]) - f_inf) > 1e-12:
            raise ValueError("initial.params['finf'] must equal γ/β, the steady state.")
        centred = (i + 0.5) * dx - 0.5 * length
        phi = np.cos(k * centred) if parity == "even" else np.sin(k * centred)
        return decay_rate_curve(
            setup, arrays, phi, k * k, baseline=f_inf, exact_shape=True,
            residual_label=(
                "The residual is FIRST order: the engine applies β to the end "
                "cell's centre value rather than to the face, so the continuum "
                f"{parity} mode is not the exact discrete eigenvector and its "
                "projection leaks slightly into the neighbouring discrete modes."
            ),
        )
    return _build


for _parity, _overrides_dict in (("even", CASE_OVERRIDES_EVEN), ("odd", CASE_OVERRIDES_ODD)):
    register(Benchmark(
        name=f"bc-robin-{_parity}",
        title=f"Robin ends (∂ₙf + βf = γ): {_parity} mode decay on a strip",
        tier="T1",
        formula_latex=(
            r"\partial_n f+\beta f=\gamma\ \text{at both ends},\qquad "
            r"f=\frac{\gamma}{\beta}+A(E)\,\phi_k(x)\,e^{-\lambda(E)t},\qquad "
            r"\boxed{\lambda(E)=D_{\rm eff}(E)\,k^{2}}\\[4pt]"
            + (
                r"\phi_k=\cos k(x-\tfrac{L}{2}),\qquad k\tan\tfrac{kL}{2}=\beta"
                if _parity == "even" else
                r"\phi_k=\sin k(x-\tfrac{L}{2}),\qquad k\cot\tfrac{kL}{2}=-\beta"
            )
        ),
        headline_latex=(
            r"k\tan\frac{kL}{2}=\beta" if _parity == "even" else r"k\cot\frac{kL}{2}=-\beta"
        ) + r",\qquad \lambda(E)=D_{\rm eff}(E)\,k^{2}",
        reason=(
            "A finite-transparency contact on both ends fixes the mode's "
            "wavenumber through a transcendental equation in β L; the mode about "
            "the steady state γ/β decays at D_eff(E) k². The engine discretises "
            "the condition at first order (β on the centre value, not the face), "
            "so the tolerance is a first-order statement and the halving "
            "sequence is recorded, as the plan asked."
        ),
        # First order: the tolerance is the measured error at the shipped
        # 32-cell mesh with ~30% headroom, and the halving sequence below is
        # the claim. Halve the mesh and the tolerance could be halved with it.
        rel_tol=(3.0e-2 if _parity == "even" else 1.2e-2),
        convergence=(
            "Headline case: 32 cells at 4 μm (L = 128 μm), β = 2/L (β L/2 = 1), "
            "γ = 0.3 β, 32 energy bins, dt = 0.1 ns, 1000 steps, 21 frames over "
            "100 ns, ~9 s.\n\n"
            "SPACE (2026-09-02): dx halved at fixed L, T = 100 ns, NE = 16, dt as "
            "dx². The engine applies β to the end cell's CENTRE value, so this "
            "is FIRST order by construction, and the sequence says so:\n"
            + (
                "   16 cells  dx=8 μm  dt=0.4 ns     4.6031e-02\n"
                "   32 cells  dx=4 μm  dt=0.1 ns     2.2918e-02   order 1.006\n"
                "   64 cells  dx=2 μm  dt=0.025 ns   1.1432e-02   order 1.003\n"
                "Cleanly first order: the even mode has the smaller k (1.72/L), "
                "and a boundary shift of half a cell is a larger fraction of its "
                "wavelength."
                if _parity == "even" else
                "   16 cells  dx=8 μm  dt=0.4 ns     1.4116e-02\n"
                "   32 cells  dx=4 μm  dt=0.1 ns     8.6688e-03   order 0.703\n"
                "   64 cells  dx=2 μm  dt=0.025 ns   4.7261e-03   order 0.875\n"
                "Approaching first order from below: at coarse meshes the "
                "second-order stencil error of the odd mode's larger k (4.06/L) "
                "partly cancels the first-order rim error, and the cancellation "
                "shrinks as the stencil term does."
            )
            + "\n\nThe plan asked for a first-order convergence STATEMENT rather "
            "than a fixed tolerance; a Benchmark carries a tolerance, so it is "
            "set at the shipped mesh and the order is recorded here. To tighten "
            "it, refine the mesh -- or extrapolate β to the face in "
            "grid/spatial_grid.py, which would make these second order."
        ),
        modes=("kinetics",),
        build=_make_build(_parity),
        activity=(
            "β → 0 is the reflective rim (k → 0 for the even mode, k → π/L for the "
            "odd) and β → ∞ the absorbing one (k → π/L, 2π/L); β L/2 = 1 sits "
            "between them, where the rate is far from either limit."
        ),
        caveat=(
            "FIRST ORDER in dx by construction of the engine's Robin face. The "
            "tolerance holds at the shipped mesh; refining the mesh halves the "
            "error, coarsening it doubles it. γ ≠ 0 exercises the inhomogeneous "
            "Robin source; the mode decays towards γ/β, which is a setup "
            "quantity, not a fitted one."
        ),
    ))
