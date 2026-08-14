"""Self-consistent gap: the BCS closure re-solved from the occupation.

The closure is the one term here that is not a rate. Every other switch adds
or removes a contribution to ``df/dt``; this one takes whatever ``f`` the
kinetics produced and re-solves

    1/lambda = int_Delta^{omega_D} (1 - 2 f(E)) / sqrt(E^2 - Delta^2) dE

for the gap, with ``omega_D = 100 k_B T_c`` and ``1/lambda`` fixed by the
linearised equation at ``T_c`` (:func:`qpsim.physics.gap_equation.calibrate_gap`).
So the observable is a scalar per cell, and the only way to turn it into a
curve is to give the closure a different occupation in each cell.

That is what this case does. Every kinetic term is switched off and transport
is switched off (``D_0 = 0``), so each cell is an independent 0-D problem
frozen at its prepared occupation; the prepared state is a Fermi-Dirac
distribution whose temperature ramps linearly across the columns. The gap
solved in cell *j* is then the BCS gap at ``T_eff(x_j)``, and the abscissa is
a temperature sweep executed as a spatial map: 24 independent gap solves in
one run, spanning ``T_eff/T_c = 0.39 .. 0.90`` and ``Delta = 177 .. 96 ueV``.

Why not a 0-D temperature sweep. A benchmark builds from *one* payload, and
``steady_state_0d`` returns one gap per run. The audited 41-point ``Delta(T)``
sweep therefore has no home in this framework; worse, it cannot be run through
the shipped schema at all, because it needs ``solver.gap_under_relaxation`` and
``solver.gap_solve_xtol``, and ``SolverOptions`` exposes neither (confirmed
against the current schema, not the one the derivation was written against).
The 2-D route needs no new schema field, and it exercises the closure at a
*prepared* occupation rather than at the bath equilibrium, which the audited
0-D case explicitly could not do.

The analytic side solves the same finite-cutoff model by a route that shares
nothing with the engine: the exact Matsubara identity
``tanh(E/2T)/E = 4T sum_n (omega_n^2 + E^2)^{-1}`` plus the substitution
``xi = sqrt(E^2 - Delta^2)`` turns the integral into
``4t sum_n arctan(U/a_n)/a_n``, whose truncation remainder is closed form via
Hurwitz zeta. No quadrature rule and no engine array appear in it. The engine
instead calibrates ``1/lambda`` by adaptive quadrature and integrates a
cell-constant ``f`` against exact per-cell ``acosh`` weights.

What the residual is. The reference is the *continuum* solution; the engine
solves the finite-volume one. The whole disagreement is therefore the
cell-constant representation of ``f`` in the cell that straddles the gap edge,
which the ``solve_gap`` docstring already puts at ``O((dE/Delta)^{3/2})``. It
converges, it is measured below, and it is what sets the tolerance.

Cost. 34-38 s wall clock for the shipped case, measured on a box running thirty
other jobs; about 0.85 s per cell plus 9 s fixed (12 cells: 19 s, 16: 25 s,
20: 31 s, 24: 36 s). The cost is the outer gap relaxation -- 20 damped
iterations x 24 cells of ``solve_gap`` -- not the 1250-bin grid, so trading
curve points for wall clock is the only lever, and 24 points was judged worth
the seconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import zeta as _hurwitz_zeta

from qpsim.constants import KB_UEV_PER_K
from qpsim.webui.benchmarks import Benchmark, Curve, register

# The engine's cutoff convention: `solve_gaps` calls `calibrate_gap` without
# `omega_D_over_Tc`, so 100 is what the model uses. Stated as a constant rather
# than imported, because the reference must not inherit the engine's choice
# silently -- but see `caveat`: this curve is very nearly BLIND to the cutoff,
# so the constant is a model declaration, not something the check pins down.
_OMEGA_D_OVER_TC = 100.0

# T_eff(x) = _T_OFFSET_K + _T_SLOPE_K * x over the normalised column centre
# x = (j + 1/2)/ncols, giving T_eff/T_c = 0.3923 .. 0.8958 across 24 columns.
# Both ends are deliberate. Below T/T_c ~ 0.15 the thermal correction to the
# gap is O(e^-35), so both sides return Delta(0) to machine precision and cold
# points are one test repeated rather than several. At the hot end the gap
# reaches the grid's first face, 90 ueV, at T/T_c = 0.918 -- past which
# `solve_gap` extrapolates f flat below the face -- and the audit of the 0-D
# form measured several roots of the finite-volume residual above ~0.96 on a
# comparable grid. 0.896 clears both with room.
_T_OFFSET_K = 0.45
_T_SLOPE_K = 0.62

# One source of truth: the case string and the abscissa are built from the same
# two constants, so they cannot drift apart. `build` refuses any other seed.
_SEED_EXPRESSION = (
    f"1.0/(np.exp(E/({KB_UEV_PER_K}*({_T_OFFSET_K} + {_T_SLOPE_K}*x))) + 1.0)"
)

CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material": {
        # Delta_0 is a GRID ANCHOR and the outer loop's starting guess, not
        # physics: the closure fixes the pairing scale from T_c alone, and this
        # model's own Delta(0) is 179.345 ueV. It is set well below every gap
        # on the curve so the solved gap always sits strictly inside the grid
        # -- below the grid's first face `solve_gap` extrapolates f flat, which
        # is a modelling error that does NOT vanish under refinement.
        "Delta_0": 90.0,
        "T_c": 1.18,
        # Transport off: each cell must be an independent 0-D closure problem,
        # or the prepared per-cell occupations would diffuse into each other.
        "D_0": 0.0,
        # Spatial modes require a pure-BCS spectral context, and the closure is
        # only defined against the BCS density of states.
        "dynes_gamma": 0.0,
    },
    # Unused by the closure (the prepared occupation replaces the thermal seed
    # outright) but it does set `calibrate_gap`'s brentq tolerance, 1e-6*Delta_eq
    # ~ 1.8e-4 ueV here, so a cold bath keeps that floor at ~1e-6 relative.
    "T_bath": 0.1,
    # dE = (16 - 1)*90/1250 = 1.08 ueV, and the top face sits at 1440 ueV. The
    # engine takes f = 0 above that face while the reference keeps f_FD all the
    # way to omega_D; extending the grid to 1800 ueV at identical cell edges
    # moves the answer by 8.8e-8, so that truncation is four orders below the
    # residual. The tolerance IS grid-locked to dE -- see `convergence`.
    "grid": {"min_factor": 1.0, "max_factor": 16.0, "num_bins": 1250},
    # 1 x 24 rectangle: the 1-D reduction of the 2-D core, 24 independent cells.
    "geometry": {"kind": "rectangle", "rows": 1, "cols": 24, "mesh_size_um": 4.0},
    "collisions": {
        "scattering": False,
        "recombination": False,
        "phonon_scattering_source": False,
        "phonon_recombination_source": False,
    },
    "initial": {"kind": "absolute", "expression": _SEED_EXPRESSION},
    "self_consistent_gap": True,
    # The gap is snapped to this fraction of dE before use. The shipped default
    # 0.1 would quantise Delta to 0.108 ueV, i.e. 1e-3 relative -- larger than
    # the residual being measured. 1e-3 puts the quantum at 1.08e-3 ueV (~1e-5
    # relative), below every other floor here. It costs nothing because a
    # 24-cell ramp already gives 24 distinct gaps whatever the quantum.
    "gap_quantum_over_dE": 1e-3,
    # df/dt is identically zero, so the run converges on its first step; the
    # gap is relaxed to its fixed point inside that step.
    "dt": 1.0,
    "max_time": 1.0,
    "stop_tol": 2e-10,
}


def _tail_power_sums(n_start: int, t: float, orders: int = 5) -> np.ndarray:
    """``sum_{n>=n_start} omega_n^{-2m}`` for m = 1..orders, in closed form.

    ``omega_n = (2n+1) pi t``, so the sum is a Hurwitz zeta at ``n_start + 1/2``.
    This is what keeps the reference free of any quadrature rule: the Matsubara
    series converges as ``1/n^2`` and truncating it without a remainder would
    put a ~1e-4 bias on the answer.
    """
    q = n_start + 0.5
    return np.array([
        float(_hurwitz_zeta(2 * (m + 1), q)) / (2.0 * np.pi * t) ** (2 * (m + 1))
        for m in range(orders)
    ])


def _gap_integral(d: float, t: float, w: float = _OMEGA_D_OVER_TC) -> float:
    """``int_d^w tanh(E/2t)/sqrt(E^2 - d^2) dE`` in units of ``k_B T_c``.

    Exact: ``tanh(E/2t)/E = 4t sum_n (omega_n^2 + E^2)^{-1}`` and
    ``xi = sqrt(E^2 - d^2)`` give ``4t sum_n arctan(U/a_n)/a_n`` term by term.
    The head is summed until ``U/a_n <= 2e-3``; beyond that the arctan is
    expanded to fifth order and each power sum closed with :func:`_tail_power_sums`,
    so the retained third term is already below one ULP of the total.
    """
    if d >= w:
        return 0.0
    u = float(np.sqrt((w - d) * (w + d)))
    n_terms = int(np.ceil(u / (2e-3 * 2.0 * np.pi * t))) + 1000
    n = np.arange(n_terms, dtype=float)
    omega_n = (2.0 * n + 1.0) * np.pi * t
    a = np.sqrt(omega_n * omega_n + d * d)
    head = float(np.sum(np.arctan(u / a) / a))
    z = _tail_power_sums(n_terms, t)
    d2 = d * d
    # sum a_n^{-2k} expanded in (d/omega_n)^2, k = 1, 2, 3.
    s1 = z[0] - d2 * z[1] + d2 * d2 * z[2]
    s2 = z[1] - 2.0 * d2 * z[2] + 3.0 * d2 * d2 * z[3]
    s3 = z[2] - 3.0 * d2 * z[3] + 6.0 * d2 * d2 * z[4]
    return 4.0 * t * (head + u * s1 - (u**3 / 3.0) * s2 + (u**5 / 5.0) * s3)


def _inverse_coupling(w: float = _OMEGA_D_OVER_TC) -> float:
    """``1/lambda`` from the linearised equation at ``T_c``.

    The same integral at ``d = 0, t = 1``: the coupling is anchored exactly the
    way :func:`qpsim.physics.gap_equation.calibrate_gap` anchors it, which is
    the whole reason the two sides are comparable at all. Note the engine
    reaches this number by scipy adaptive quadrature and this series does not.
    """
    return _gap_integral(0.0, 1.0, w)


def _reduced_gap(t: float, inv_lambda: float, w: float = _OMEGA_D_OVER_TC) -> float:
    """``Delta(t)/k_B T_c`` for the finite-cutoff model, ``t = T/T_c``.

    ``Delta(0) = w/cosh(1/lambda)`` is analytic and needs no root find, so it
    doubles as the upper bracket.
    """
    d0 = float(w / np.cosh(inv_lambda))
    # Deep in the cold limit the thermal correction underflows and the residual
    # at d0 is exactly +0.0; brentq would reject that as an invalid bracket
    # rather than return the endpoint the answer already sits on.
    if _gap_integral(d0, t, w) - inv_lambda >= 0.0:
        return d0
    return float(brentq(
        lambda d: _gap_integral(d, t, w) - inv_lambda,
        1e-12 * d0, d0, xtol=1e-15 * d0, rtol=8.9e-16, maxiter=200,
    ))


def _analytic_gap_ueV(T_eff_K: np.ndarray, T_c: float) -> np.ndarray:
    """The closed-form gap in micro-eV at each prepared temperature.

    Raises outside ``0 < T < T_c`` rather than returning the trivial endpoint:
    the audit of the original derivation found that the delivered snippet
    silently returned an all-zero curve when it was handed an energy grid
    instead of temperatures, and an all-zero analytic side is indistinguishable
    from a broken engine.
    """
    t = np.asarray(T_eff_K, dtype=float).reshape(-1) / float(T_c)
    if t.size == 0:
        raise ValueError(
            "The self-consistent-gap reference was handed no temperatures."
        )
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0) or np.any(t >= 1.0):
        raise ValueError(
            "The self-consistent-gap reference needs prepared temperatures "
            f"strictly inside (0, T_c); got T/T_c spanning "
            f"[{np.nanmin(t):g}, {np.nanmax(t):g}]."
        )
    inv_lambda = _inverse_coupling()
    scale = KB_UEV_PER_K * float(T_c)
    return np.array([_reduced_gap(float(x), inv_lambda) for x in t]) * scale


def _cell_temperatures(setup: Any) -> np.ndarray:
    """``T_eff`` in each cell, in the mask order the payload arrays use.

    ``cell_coordinates`` normalises column centres over the mask's bounding
    box, so column ``j`` of an ``ncols``-wide mask sits at ``(j + 1/2)/ncols``.
    A one-row full rectangle is required precisely so that this mapping is the
    whole story: on a ragged mask, cell order is still row-major but the
    columns present are not ``0..ncols-1``.
    """
    geometry = setup.geometry
    if geometry.kind != "rectangle" or geometry.rows != 1:
        raise ValueError(
            "This benchmark's abscissa is the column index of a one-row "
            f"rectangle; this run is a {geometry.kind!r} geometry with "
            f"rows={geometry.rows}."
        )
    x_norm = (np.arange(geometry.cols, dtype=float) + 0.5) / float(geometry.cols)
    return _T_OFFSET_K + _T_SLOPE_K * x_norm


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    initial = setup.initial
    if initial.kind != "absolute" or initial.expression != _SEED_EXPRESSION:
        raise ValueError(
            "This benchmark reads the prepared temperature ramp out of the "
            "initial condition, so it only applies to the case that declares "
            f"it: initial.kind='absolute' with expression {_SEED_EXPRESSION!r}."
        )

    T_eff = _cell_temperatures(setup)
    T_c = float(setup.material.T_c)
    analytic = _analytic_gap_ueV(T_eff, T_c)

    sim = np.asarray(arrays["gap_per_cell"], dtype=float).reshape(-1)
    if sim.shape != analytic.shape:
        raise ValueError(
            f"The run has {sim.size} cells but the abscissa has {analytic.size}; "
            "the mask is not the full one-row rectangle this case declares."
        )

    # The closed form describes the gap of a state, so it is only a statement
    # about this run if the state the closure saw is the prepared one. With
    # every kinetic term and transport off it is, to the bit -- and if a term
    # were switched back on, f would have moved and the reference would be
    # describing a different experiment. Checked rather than assumed.
    energies = np.asarray(arrays["E_bins"], dtype=float)
    seeded = 1.0 / (np.exp(energies[:, None] / (KB_UEV_PER_K * T_eff[None, :])) + 1.0)
    seed_err = float(np.max(np.abs(np.asarray(arrays["f_final"], float) - seeded)))
    if seed_err > 1e-12:
        raise ValueError(
            f"The final occupation differs from the prepared Fermi-Dirac ramp "
            f"by {seed_err:.3g}, so something moved f: this reduction requires "
            "every collision channel off, no drive, and D_0 = 0."
        )

    return Curve(
        x=T_eff / T_c,
        y_sim=sim,
        y_analytic=analytic,
        x_label="T_eff / T_c  (temperature of the occupation prepared in each cell)",
        y_label="Delta  [micro-eV]",
        series_labels=("self-consistent gap",),
        note=(
            f"{sim.size} independent cells, one BCS closure each. The prepared "
            f"occupation is reproduced to {seed_err:.1e} (bit-exact when nothing "
            f"else acts). Delta falls {analytic[0]:.2f} -> {analytic[-1]:.2f} "
            f"micro-eV, a factor {analytic[0] / analytic[-1]:.2f}. Note the curve "
            f"does NOT start at material.Delta_0 = {setup.material.Delta_0:g} "
            "micro-eV: that value is the energy-grid anchor and the closure's "
            "starting guess, never a physical input. This model's own Delta(0) "
            f"is {_OMEGA_D_OVER_TC / np.cosh(_inverse_coupling()) * KB_UEV_PER_K * T_c:.3f}"
            " micro-eV, set by T_c alone."
        ),
    )


register(Benchmark(
    name="self-consistent-gap",
    title="Self-consistent gap: BCS closure against the finite-cutoff Delta(T)",
    tier="T1",
    formula_latex=(
        r"\frac{1}{\lambda}=\int_{\Delta}^{\hbar\omega_D}"
        r"\frac{1-2f(E)}{\sqrt{E^2-\Delta^2}}\,dE,\qquad "
        r"\hbar\omega_D=100\,k_BT_c,\qquad "
        r"\frac{1}{\lambda}=\int_0^{\hbar\omega_D}\frac{\tanh(E/2k_BT_c)}{E}\,dE"
        r"\\[6pt]"
        r"f=f_{FD}(E,T_{\rm eff})\Rightarrow 1-2f=\tanh\!\big(E/2k_BT_{\rm eff}\big),"
        r"\quad\text{so with }d=\Delta/k_BT_c,\ t=T_{\rm eff}/T_c,\ w=100:"
        r"\\[6pt]"
        r"\boxed{\;G(d,t)=4t\sum_{n\ge0}\frac{\arctan(U/a_n)}{a_n}=\frac{1}{\lambda},"
        r"\quad U=\sqrt{w^2-d^2},\ a_n=\sqrt{\omega_n^2+d^2},\ "
        r"\omega_n=(2n+1)\pi t\;}"
        r"\\[6pt]"
        r"\Delta(0)=\frac{\hbar\omega_D}{\cosh(1/\lambda)}=1.7637398\,k_BT_c"
        r"=179.345\ \mu\mathrm{eV}\ \text{at}\ T_c=1.18\,\mathrm{K}"
    ),
    reason=(
        "With every kinetic term off and D_0 = 0 each cell is frozen at the "
        "occupation it was prepared with, so the closure alone acts and its "
        "root is the finite-cutoff BCS gap at that cell's temperature. The "
        "Matsubara identity for tanh(E/2T)/E makes that root an explicit "
        "arctangent series with a Hurwitz-zeta remainder -- no quadrature "
        "rule, no engine array, no root of an engine residual."
    ),
    headline_latex=(
        r"\frac{1}{N_0V} \;=\; \int_\Delta^{\hbar\omega_D}\frac{1-2f(E)}{\sqrt{E^2-\Delta^2}}\,dE"
    ),
    rel_tol=1.5e-3,
    convergence=(
        "MEASURED ON THIS CASE. Abscissa T_eff/T_c = 0.3923 .. 0.8958 over 24 "
        "cells; grid [1, 16] x 90 micro-eV, so dE = 1350/num_bins micro-eV. "
        "Refinement ladder on the same 24 points, driving solve_gap directly on "
        "the case grid so the run's outer relaxation cap and the gap quantum "
        "cannot contaminate the convergence statement. They do not contaminate "
        "the case either: at the shipped grid the full web route gives "
        "3.1528e-04 against the direct 3.1963e-04, a 4.4e-06 difference, which "
        "is the quantum.\n"
        "  num_bins   dE [ueV]    rms|rel|    max|rel|\n"
        "       160     8.4375    2.370e-03   7.254e-03\n"
        "       320     4.2188    5.081e-04   1.127e-03\n"
        "       625     2.1600    2.171e-04   7.449e-04\n"
        "      1250     1.0800    1.138e-04   3.196e-04   <- the shipped case\n"
        "      2500     0.5400    2.425e-05   8.881e-05\n"
        "rms falls 97.7x for a 16x finer grid (observed order 1.65) and the max "
        "81.7x (order 1.59), consistent with the O((dE/Delta)^{3/2}) gap-edge "
        "cell error solve_gap documents. Successive orders scatter (2.22, 1.27, "
        "0.93, 2.23) because the residual OSCILLATES IN SIGN with the grid: the "
        "phase of Delta inside its containing cell moves with num_bins. That is "
        "why the tolerance comes from a dense envelope, not from these 24 "
        "points, which are one sample of that oscillation.\n"
        "DENSE ENVELOPE, shipped grid, over exactly the shipped abscissa range "
        "(the 24 case points are one sample of it, so this is the number the "
        "tolerance is set from): 5.587e-04 on a 400-point scan, 6.245e-04 on a "
        "1200-point scan, the peak sitting at T_eff/T_c = 0.890. Refining the "
        "sampling 3x moved the peak by 12%, so it is converging rather than "
        "hiding a spike; 0 of those 1200 points exceed 1.0e-03.\n"
        "TOLERANCE. rel_tol = 1.5e-3 is 2.4x that envelope. This DEPARTS from "
        "the audit's recommended 5.0e-3, deliberately and by the audit's own "
        "argument: 5.0e-3 was set for an abscissa running to T/T_c = 0.95, "
        "where the envelope climbs steeply (the audit measured 1.81e-3 there on "
        "a 1201-point scan) and the audit offered 'or stop the abscissa at "
        "0.94' as the alternative remedy. This abscissa stops at 0.896 -- the "
        "remedy, not the loosening. 5.0e-3 here would be 8x the measured "
        "envelope, and it would no longer fail a 1e-4 relative error in the "
        "pairing anchor 1/lambda (which moves the curve 2.1e-3): that is the "
        "one quantity this curve really pins, so the tolerance is held below "
        "it on purpose. 2.4x rather than the audit's 2.76x is the price of "
        "keeping that discrimination, and it is a margin over a measured, "
        "converging envelope rather than over a single lucky sample.\n"
        "GRID-LOCKED. Valid at num_bins >= 1250 on this grid. At 625 the "
        "residual reaches 7.4e-4 on these very points -- and the envelope off "
        "them is larger again -- so a coarser run needs a looser tolerance, not "
        "this one. The case pins num_bins, which is why that is safe.\n"
        "OTHER FLOORS, all far below the residual: the gap quantum is 1.08e-3 "
        "micro-eV (~1e-5 relative); solve_gap's brentq xtol defaults to "
        "1e-6*Delta_eq = 1.8e-4 micro-eV (~1e-6 relative); the run's outer gap "
        "relaxation (mixing 0.5, capped at 20 iterations) leaves at most 0.5^20 "
        "of the initial 90 -> Delta offset, ~8e-5 micro-eV; and the engine's "
        "f = 0 truncation above the top cell face costs 8.8e-8, measured by "
        "extending the grid from 1440 to 1800 micro-eV at identical cell edges "
        "(1260 gives 7.3e-7 and 1080 gives 6.1e-6, so 1440 has room).\n"
        "ANALYTIC SIDE, self-checks -- these are checks OF the transcription, "
        "not the reference: the series matches an adaptive quadrature of the "
        "same integral to 7.5e-16 worst case over d in [0, 1.76] x t in "
        "[0.02, 0.99]; brentq and a 200-step bisection on the same residual "
        "agree to 1.6e-15; and 1/lambda = 4.730803145600169 from the series is "
        "bit-identical to the engine's scipy-quad value, which is the one "
        "number the two sides must share for the comparison to mean anything."
    ),
    modes=("spatial_2d",),
    build=_build,
    caveat=(
        "IT DOES NOT CONSTRAIN THE DEBYE CUTOFF. The original derivation "
        "claimed this check would detect 'a wrong cutoff convention'; the "
        "audit disproved it and the claim is struck. Re-anchoring 1/lambda at "
        "T_c for each cutoff and re-solving moves this whole curve by 3.3e-4 "
        "(w = 50), 8.2e-5 (w = 200) and 1.0e-4 (w = 400) relative -- all inside "
        "the tolerance. Delta(T)/k_B T_c is cutoff-independent to O(1/w^2), so "
        "halving or quadrupling omega_D is invisible here, and the 100 written "
        "into this module is a declaration of the model, not something the "
        "green result establishes. What the curve DOES pin is the pairing "
        "anchor: a 1e-4 relative error in 1/lambda moves it by 2.1e-3, 1.4x "
        "the tolerance, and 1e-3 moves it by 2.1e-2. Below about 7e-5 in "
        "1/lambda it would go unnoticed.\n"
        "The reference is the CONTINUUM solution of the model the engine "
        "defines. It cannot detect an error shared by the model definition and "
        "this transcription of it -- if the finite-cutoff, T_c-anchored "
        "weak-coupling kernel is the wrong physics for a real film, both sides "
        "are wrong together. What it does pin without touching engine code is "
        "1/lambda, the endpoint Delta(0) = 1.7637 k_B T_c and the shape of "
        "Delta(T).\n"
        "It exercises the closure at a THERMAL-SHAPED occupation, prepared "
        "cell by cell rather than inherited from the bath. So it does test the "
        "closure away from its own equilibrium fixed point -- which the 0-D "
        "form of this benchmark could not -- but it still says nothing about "
        "branch selection for a genuinely non-thermal f (a pumped or "
        "population-inverted spectrum), about GapBelowGridSupportError "
        "handling, or about the coupling between a moving gap and the "
        "collision kernels: with collisions off the gap never feeds back into "
        "anything.\n"
        "The abscissa is truncated at both ends on purpose, and neither end is "
        "covered. Below T/T_c ~ 0.15 the thermal correction to the gap is "
        "O(e^-35): both sides return Delta(0) to machine precision, so cold "
        "points are one test repeated rather than 24 tests (the audit raised "
        "this against the original 41-point sweep, a third of which sat "
        "there). At the hot end this case has a hard ceiling of its own: the "
        "gap reaches the grid's first face, 90 micro-eV, at T/T_c = 0.918, and "
        "below that face solve_gap extrapolates f flat -- a modelling error "
        "that does not shrink under refinement. Independently, the audit "
        "measured multiple roots of the finite-volume residual above "
        "T/T_c ~ 0.96 on a comparable grid. 0.896 keeps clear of both.\n"
        "Requires dynes_gamma = 0 (the closure is BCS-only, and spatial modes "
        "reject a broadened spectrum anyway), and says nothing about the gap "
        "quantisation itself beyond that a 1e-3*dE quantum is negligible."
    ),
    activity=(
        "SWITCH TEST. Same case with self_consistent_gap = False: the gap is "
        "pinned at material.Delta_0 = 90 micro-eV in every cell (asserted "
        "exactly -- np.unique of the 24 returns a single value), and this "
        "benchmark then reports max|rel| = 4.913e-01, rms 3.995e-01: 328x the "
        "tolerance, with all 24 points outside it, the closest being 6.4e-02. "
        "Switching the closure on moves the gap by +86.9 micro-eV at the cold "
        "end and +6.2 at the hot end.\n"
        "THE CURVE MOVES. Delta falls 176.92 -> 96.16 micro-eV across the 24 "
        "cells, a factor 1.84 against a 1.5e-3 tolerance. This is a shape "
        "statement, not a constancy statement dressed up as a curve.\n"
        "NOTHING ELSE IS ACTING. The build refuses to score the run unless the "
        "final occupation still equals the prepared Fermi-Dirac ramp to 1e-12; "
        "in the shipped case it is bit-exact (0.0). Falsified rather than "
        "asserted: adding one gain drive of amplitude 1e-6/ns moves f by 1e-6 "
        "and the benchmark then declines to return a verdict at all, instead "
        "of comparing a closed form against a state it does not describe. So a "
        "green result cannot have come from some other term quietly reshaping "
        "f. This reduction also has no inert-default trap of the "
        "subgap_drive.n_bar = 0.0 kind -- the closure has no amplitude knob "
        "that can be silently zero."
    ),
))
