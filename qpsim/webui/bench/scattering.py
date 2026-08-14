r"""Quasiparticle-phonon scattering — the Kaplan out-scattering rate, in closed form.

At ``T_bath -> 0`` the phonon occupation vanishes, so the scattering operator can
only carry a quasiparticle DOWN in energy. The **highest occupied cell of a
column** therefore has no source above it: its gain term is identically zero for
all time and its equation collapses from the full integro-differential form to

    df_i/dt = -f_i * SUM_{j<i} K0s[i,j] * w_j * (1 - f_j)

exactly, not asymptotically. Once the prepared excess is small enough that
``f_j << 1``, the bracket is a constant and the decay is a pure exponential whose
rate is the continuum Kaplan scattering-out integral — elementary in
``s = sqrt(E^2-D^2)`` and ``L = arccosh(E/D)``.

What makes this a *curve* rather than a rate. A single decaying exponential is
one number wearing 321 points, and the adversarial audit of the original
derivation said so: ~91% of its residual is the single scalar Gamma. The fix
uses the 2-D core as five independent 0-D experiments. The mask is 1x5, ``D_0``
is 0 so no cell can talk to another, and the non-separable initial condition
seeds a DIFFERENT single energy cell in each column. Five columns, five
different highest-occupied bins, five independently predicted rates spanning a
factor 7.4 in Gamma — from one run, on one abscissa. That is the energy sweep
the audit called "the stronger evidence" and could only get from 16 separate
runs.

Authored against the AUDITED position. The audit returned "sound" with six minor
findings, all in the write-up; each correction is marked ``AUDIT`` below and the
recommended tolerance (1.0e-3, not the original 1.5e-3) is the one that ships.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# Boltzmann's constant, written out rather than imported from qpsim.constants.
# The analytic side of a T1 must not share a constant table with the engine it
# checks: a shifted k_B would cancel out of the comparison instead of showing up.
_KB_UEV_PER_K = 86.17333262145

# Near the gap edge J ~ (E/D - 1)^(7/2) while its four terms are individually
# O((E/D - 1)^(1/2)): the first three orders cancel identically and float64
# loses digits. Measured against an independent quadrature in the theta =
# arccosh(E/D) variable, which has no endpoint singularity at all --
#   E/D:   1.001      1.01      1.05      1.10      >= 1.5
#   err:   2.6e-05   1.5e-08   1.2e-11   3.6e-12   4.4e-16
# -- so 1.05 still leaves eight orders of margin under the tolerance. The
# derivation quoted 1.1 without a measured cost; this is the measurement.
_MIN_E_OVER_GAP = 1.05

# The reduction linearises in f. Measured cost of NOT being in that limit, same
# case with only the seed amplitude changed (curve residual):
#   1e-8 -> 3.7000e-4 | 1e-6 -> 3.7003e-4 | 1e-4 -> 3.7294e-4
#   1e-3 -> 3.9946e-4 | 1e-2 -> 6.6463e-4
# So 1e-3 costs 8% of the residual (still 0.40 of the tolerance) and 1e-2 costs
# 80% (0.66 of it, i.e. it would still pass while measuring something else).
# Refuse above 1e-3: a benchmark that quietly reports Pauli blocking as a kernel
# error is worse than one that declines.
_MAX_INITIAL_OCCUPATION = 1e-3

# The closed form uses N_p = 1 (spontaneous emission only). AUDIT finding 4 is
# right that T -> 0 is not the knife edge the original derivation implied -- but
# it is a bound, not a free parameter. Same case, T_bath swept, curve residual:
#   0.001 K -> 3.7003e-4 | 0.02 -> 3.6998e-4 | 0.05 -> 3.6916e-4
#   0.1     -> 3.6347e-4 | 0.2  -> 3.9400e-4 | 0.5 -> 1.9292e-2 | 0.9 -> 1.46
# i.e. flat to 6% of itself while the neglected Bose factor n_BE(dE) runs from
# 4e-21 to 3.8, then failing by 19x the tolerance at 0.5 K and by 1500x at
# 0.9 K. The controlling ratio is k_B*T/Delta: 0.096 at 0.2 K (passes), 0.24 at
# 0.5 K (fails). The bound below sits just above the last measured pass.
_MAX_KT_OVER_GAP = 0.1

# The case this closed form applies to. Every other term is off, so 100% of the
# motion in the run is this one (see ``activity``).
#
# The numbers that are not free:
#   geometry 1x5 + D_0 = 0. Five INDEPENDENT 0-D cells: the flux weight is
#     D_0*N_1**q, so D_0 = 0 gives an identically zero transport operator and
#     T3SpatialBackend.apply_transport moves nothing between columns. That is
#     what lets five different energies be measured in one run.
#   initial.expression. A one-bin excess per column at 987.975 / 1190.475 /
#     1392.975 / 1595.475 / 1797.975 ueV = cells 199/249/299/349/399 of this
#     grid (the top cell and every 50th below it), each 1e-6 in occupation. The
#     half-widths are 0.4*dE = 1.62 ueV, so each term selects exactly one cell
#     and each column starts with exactly one populated bin. x is the normalised
#     cell centre, (col+0.5)/5 = 0.1, 0.3, 0.5, 0.7, 0.9.
#   T_bath = 1 mK. The T -> 0 limit: n_BE(dE) = 3.9e-21 and the thermal f
#     underflows to exactly 0 in every bin, so the limit is exact in float64.
#     Every web run used to start thermal, which at this temperature is
#     identically zero -- this case was unreachable until Spatial2DSetup.initial
#     existed.
#   num_bins = 400 and dt. AUDIT finding 5: the tolerance is a measured
#     discretisation floor and is valid ONLY at this pair. dt = total/320 gives
#     Gamma_top*dt = 0.025 (inside ETD2's max_loss_step = 0.25, so no
#     subcycling) and max_time = 8/Gamma_top gives the fastest series 8
#     e-foldings and the slowest 1.08. At num_bins = 50 the residual is
#     9.4e-3 = 9.4x this tolerance, and at Gamma*dt = 0.2 it is 3.5e-3 = 3.5x;
#     both are inside the schema's legal range for those fields, so editing
#     either one invalidates the verdict rather than degrading it gracefully.
#   stop_tol = 0. This is a relaxation measurement, not a steady-state search:
#     the residual falls with the population it is measuring, so ANY positive
#     stop_tol truncates the curve at the moment it becomes most informative.
#     0 is the only value that never fires. The run then reports "did not reach
#     stop_tol", which is the truth.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material.name": "Al",
    "material.Delta_0": 180,
    "material.T_c": 1.18,
    "material.tau_0": 438,
    "material.D_0": 0,
    "material.dynes_gamma": 0,
    "T_bath": 0.001,
    "grid.min_factor": 1,
    "grid.max_factor": 10,
    "grid.num_bins": 400,
    "geometry.kind": "rectangle",
    "geometry.rows": 1,
    "geometry.cols": 5,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "diffusion_model": "A1",
    "collisions.scattering": True,
    "collisions.recombination": False,
    "collisions.phonon_scattering_source": True,
    "collisions.phonon_recombination_source": True,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "initial.kind": "excess",
    "initial.amplitude": 0.0,
    "initial.expression": (
        "1e-06 * ("
        "(abs(x - 0.1) < 0.1) * (abs(E - 987.975) < 1.62)"
        " + (abs(x - 0.3) < 0.1) * (abs(E - 1190.475) < 1.62)"
        " + (abs(x - 0.5) < 0.1) * (abs(E - 1392.975) < 1.62)"
        " + (abs(x - 0.7) < 0.1) * (abs(E - 1595.475) < 1.62)"
        " + (abs(x - 0.9) < 0.1) * (abs(E - 1797.975) < 1.62))"
    ),
    "drives": [],
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "thermal_bath",
    "self_consistent_gap": False,
    "dt": 0.006757025481804307,
    "max_time": 2.1622481541773784,
    "stop_tol": 0.0,
    "snapshot_interval": 0.006757025481804307,
}


def _kaplan_out_rate(E: np.ndarray, gap: float, tau_0: float, T_c: float) -> np.ndarray:
    r"""Kaplan T=0 scattering-out rate ``Gamma(E)``, in 1/ns. Closed form.

    ``Gamma = J/(tau_0 (k_B T_c)^3)`` with

        J(E) = int_D^E dE' (E-E')^2 (1 - D^2/(E E')) E'/sqrt(E'^2-D^2)
             = s^3/3 + (5/2) D^2 s - 2 E D^2 L - D^4 L/(2E)

    from the elementary antiderivatives ``I0 = L``, ``I1 = s``,
    ``I2 = E s/2 + D^2 L/2``, ``I3 = s^3/3 + D^2 s`` (all vanishing at E' = D)
    applied to the expansion of ``(E-u)^2 (u - D^2/E)``.

    Reads only ``Delta_0``, ``T_c``, ``tau_0`` and the cell centre. Nothing from
    the engine: no kernel array, no spectral context, not even its k_B.
    """
    s = np.sqrt(E * E - gap * gap)
    L = np.arccosh(E / gap)
    J = (
        s**3 / 3.0
        + 2.5 * gap**2 * s
        - 2.0 * E * gap**2 * L
        - gap**4 * L / (2.0 * E)
    )
    return J / (tau_0 * (_KB_UEV_PER_K * T_c) ** 3)


def _refuse_unless_reduced(setup: Any, n_cells: int) -> None:
    """Raise unless the run is the reduction this closed form describes.

    Note the deliberate asymmetry: ``collisions.scattering = False`` is NOT
    refused. There the closed form still applies and is violated, and a numeric
    FAIL is the honest result — that null run is this benchmark's falsification
    test and it has to produce a verdict, not an error.
    """
    if setup.collisions.recombination:
        raise ValueError(
            "recombination is on, so scattering is not the only channel on the "
            "right-hand side. (At this seed amplitude recombination is in fact "
            "numerically inert -- it is two-body, so its rate is ~1e-6 of the "
            "scattering rate here, and the measured residual only moves from "
            "3.7003e-4 to 3.6991e-4. The refusal is a statement of scope, not a "
            "numerical necessity: at a larger amplitude it would be neither.)"
        )
    if setup.subgap_drive.enabled or setup.pb_drive.enabled:
        raise ValueError(
            "a photon drive is enabled; the observed cell then has a source "
            "term and its decay is not the scattering-out rate."
        )
    if setup.injection.enabled or any(d.enabled for d in setup.drives):
        raise ValueError(
            "an injection or prescribed drive is enabled; the observed cell "
            "then has a source term and does not decay freely."
        )
    if setup.phonons.mode != "thermal_bath":
        raise ValueError(
            f"phonons.mode={setup.phonons.mode!r} solves the phonon population, "
            "which feeds absorption back into the quasiparticles. This closed "
            "form is the pinned-bath reduction."
        )
    if setup.self_consistent_gap:
        raise ValueError(
            "a self-consistent gap moves Delta during the run, so the rate is "
            "not constant and the decay is not a single exponential."
        )
    if setup.gap_regions.kind != "uniform":
        raise ValueError(
            "a non-uniform gap gives each column its own Delta; this closed "
            "form is written for the material gap everywhere."
        )
    if n_cells > 1 and setup.material.D_0 > 0.0:
        raise ValueError(
            f"D_0 = {setup.material.D_0:g} um^2/ns couples the columns, so a "
            "column can receive quasiparticles above its own highest occupied "
            "cell and the gain term is no longer zero. This case needs D_0 = 0, "
            "which is the transport off-switch (the flux weight is D_0*N_1**q)."
        )
    kt_over_gap = _KB_UEV_PER_K * setup.T_bath / setup.material.Delta_0
    if kt_over_gap > _MAX_KT_OVER_GAP:
        raise ValueError(
            f"k_B*T_bath/Delta = {kt_over_gap:.3f} exceeds {_MAX_KT_OVER_GAP}: "
            "the closed form carries N_p = 1 (spontaneous emission only) and "
            "never reads T_bath, so a warm bath would be silently absorbed into "
            "the residual. Measured: 0.096 passes at 3.94e-4, 0.24 fails at "
            "1.93e-2."
        )


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    """One decay curve per column, against the closed-form rate at its energy.

    The analytic side consumes exactly: ``Delta_0``, ``T_c``, ``tau_0``, the grid
    triple, and the run's own snapshot times. The only thing read from the run's
    field is WHICH row to observe -- see the caveat.
    """
    gap = float(setup.material.Delta_0)
    ne = int(setup.grid.num_bins)
    min_factor = float(setup.grid.min_factor)
    dE = (float(setup.grid.max_factor) - min_factor) * gap / ne

    if "snap_f" not in arrays or "snap_t_ns" not in arrays:
        raise ValueError(
            "this benchmark needs the recorded field: set snapshot_interval so "
            "the run emits frames. Without them the only observable is the "
            "endpoint, which cannot distinguish a rate from a final value."
        )
    t = np.asarray(arrays["snap_t_ns"], dtype=float)
    f = np.asarray(arrays["snap_f"], dtype=float)          # (frames, NE, Ncells)
    n_cells = int(f.shape[2])
    _refuse_unless_reduced(setup, n_cells)

    # Rebuild the cell centres from the setup and check the run is on the grid
    # the closed form assumes. Reading E_bins and calling it "the run's grid"
    # would import the uniformity assumption instead of testing it.
    E = min_factor * gap + (np.arange(ne, dtype=float) + 0.5) * dE
    engine_E = np.asarray(arrays["E_bins"], dtype=float)
    if engine_E.shape != E.shape or not np.allclose(engine_E, E, rtol=1e-12, atol=0.0):
        raise ValueError(
            "the run's energy grid is not the uniform cell-centred grid this "
            "closed form assumes, so a cell centre is not the energy its rate "
            "was computed at."
        )

    f0 = f[0]
    if float(np.max(f0)) > _MAX_INITIAL_OCCUPATION:
        raise ValueError(
            f"the prepared state reaches f = {float(np.max(f0)):.3g}, above "
            f"{_MAX_INITIAL_OCCUPATION:g}. This closed form drops the Pauli "
            "factor (1-f) in the rate; at 1e-2 that costs 80% of the residual "
            "while still passing, which is a wrong measurement wearing a green "
            "verdict."
        )

    # The observable is the highest cell of each column carrying a prepared
    # EXCESS, because that is the one cell whose gain term is identically zero:
    # nothing above it is populated and nothing can be scattered up to it at
    # T -> 0. Excess, not occupation: the Fermi factor underflows to exactly 0
    # at 1 mK but is merely small (1e-24 at the lowest cell observed here) at
    # the warmest bath this closed form allows, and "highest nonzero cell" would
    # then silently pick the top of the grid in every column and measure one
    # rate five times.
    #
    # Taken from the prepared state rather than from a pinned index so that
    # editing the seed in the catalogue changes which rate is measured, not
    # whether the measurement is right.
    kT = _KB_UEV_PER_K * float(setup.T_bath)
    # e^-x/(1+e^-x), not 1/(e^x+1): the second overflows loudly at 1 mK, where
    # E/kT reaches 1e4 and the answer is a silent, correct zero.
    boltzmann = np.exp(-E / kT)
    thermal = boltzmann / (1.0 + boltzmann)
    excess = f0 - thermal[:, None]

    rows: list[int] = []
    for cell in range(n_cells):
        column = excess[:, cell]
        peak = float(np.max(column))
        if peak <= 0.0:
            raise ValueError(
                f"column {cell} starts at (or below) equilibrium, so there is "
                "nothing to relax and no rate to measure. Prepare an excess "
                "with Spatial2DSetup.initial."
            )
        # 1e-6 of the peak, not "nonzero": a Gaussian seed has a tail, and a
        # cell whose excess is a millionth of the peak perturbs the cell below
        # it by at most that fraction of its own rate -- four orders under the
        # tolerance -- while being the wrong cell to call the top of the
        # distribution.
        rows.append(int(np.flatnonzero(column > 1e-6 * peak)[-1]))

    E_obs = E[rows]
    if float(np.min(E_obs)) < _MIN_E_OVER_GAP * gap:
        raise ValueError(
            f"an observed cell sits at E/Delta = {float(np.min(E_obs)) / gap:.3f}, "
            f"below {_MIN_E_OVER_GAP}. The four terms of J(E) cancel to their "
            "first three orders near the gap edge and float64 loses the answer; "
            "use the arccosh-substituted quadrature there instead."
        )

    rate = _kaplan_out_rate(E_obs, gap, float(setup.material.tau_0),
                            float(setup.material.T_c))
    y_sim = np.array([f[:, row, cell] / f[0, row, cell]
                      for cell, row in enumerate(rows)])
    y_analytic = np.exp(-np.outer(rate, t))

    labels = tuple(
        f"E = {e / gap:.3f}Δ  (Γ = {g:.4f} /ns)" for e, g in zip(E_obs, rate, strict=True)
    )
    decades = np.log10(y_sim[:, 0] / np.maximum(y_sim[:, -1], 1e-300))
    return Curve(
        x=t,
        y_sim=y_sim,
        y_analytic=y_analytic,
        x_label="t (ns)",
        y_label="f(t) / f(0) of the highest occupied cell",
        series_labels=labels,
        log_y=True,
        note=(
            f"{n_cells} columns, D_0 = 0, so {n_cells} independent 0-D "
            f"experiments in one run. Observed cells {rows} at E/Δ = "
            f"{np.array2string(E_obs / gap, precision=5)}; predicted rates "
            f"{np.array2string(rate, precision=6)} /ns spanning a factor "
            f"{rate.max() / rate.min():.2f}; the curves fall by "
            f"{np.array2string(decades, precision=2)} decades. Scattering "
            "conserves quasiparticle number, so x_qp is blind to this term and "
            "would be a flat line whether or not it were switched on -- the "
            "per-cell occupation is the observable that moves."
        ),
    )


register(
    Benchmark(
        name="scattering",
        title="e-ph scattering: Kaplan out-scattering rate at five energies",
        tier="T1",
        formula_latex=(
            r"\begin{aligned}"
            r"&T_{\rm bath}\!\to\!0\ \Rightarrow\ N_p=1\ \text{for }E_i>E_j"
            r"\text{ (emission) and }0\text{ for }E_i<E_j\text{ (absorption).}\\"
            r"&\text{The highest occupied cell of a column then has no gain, and"
            r"\ for } f\ll1 \text{ its equation is exactly}"
            r"\quad \dot f_i=-f_i\!\!\sum_{j:E_j<E_i}\!\!K^{s}_0[i,j]\,w_j,\\[4pt]"
            r"&\qquad\boxed{\;\frac{f_i(t)}{f_i(0)}=e^{-\Gamma(E_i)t},\qquad"
            r"\Gamma(E)=\frac{J(E)}{\tau_0\,(k_BT_c)^3}\;}\\[4pt]"
            r"&J(E)=\int_{\Delta}^{E}\!dE'\,(E-E')^2"
            r"\Big(1-\frac{\Delta^2}{EE'}\Big)\frac{E'}{\sqrt{E'^2-\Delta^2}}"
            r"=\frac{s^3}{3}+\frac{5}{2}\Delta^2 s-2E\Delta^2 L-\frac{\Delta^4 L}{2E},\\"
            r"&\qquad s=\sqrt{E^2-\Delta^2},\qquad L=\operatorname{arccosh}(E/\Delta),"
            r"\qquad E_i=E_{\min}+\big(i+\tfrac12\big)\,\delta E,\\[4pt]"
            r"&\text{one series per column, } i=\text{its highest cell carrying a"
            r"\ prepared excess.}"
            r"\end{aligned}"
        ),
        reason=(
            "At T_bath -> 0 the phonon occupation vanishes and the scattering "
            "operator can only move a quasiparticle down in energy, so the "
            "highest occupied cell of a column has no source above it: its gain "
            "term is identically zero and, once the excess is small enough for "
            "Pauli blocking to be negligible, its equation is a linear decay at "
            "a constant rate. That rate's continuum limit is the Kaplan "
            "scattering-out integral, which is elementary. Five columns seeded "
            "at five energies with D_0 = 0 keeps them independent, so one run "
            "measures five different rates against five closed forms."
        ),
        headline_latex=(
            r"\frac{f_{\rm top}(t)}{f_{\rm top}(0)} \;=\; e^{-\Gamma(E)\,t}, \qquad \Gamma(E) "
            r"\;=\; \frac{J(E)}{\tau_0\,(k_BT_c)^3}"
        ),
        rel_tol=1e-3,
        convergence=(
            "Measured residual on the shipped case: max 3.7003e-04, rms "
            "1.6670e-04 over 5 series x 321 times. rel_tol = 1.0e-3 is 2.70x "
            "that. AUDIT finding 5: the original 1.5e-3 was 4.05x and was "
            "justified by a cross-environment drift argument worth ~1e-9 "
            "relative; the audit's recommended 1.0e-3 is what ships.\n\n"
            "The residual has exactly two sources and both were refined "
            "separately.\n\n"
            "(A) ENERGY QUADRATURE -- the engine's discrete row sum "
            "SUM_j K0s[i,j]*w_j against the closed form, at the five observed "
            "cells. No time stepping, so this isolates the energy "
            "discretisation (relative rate error, lowest observed cell first):\n"
            "   NE    dE     E/D=5.49    6.61        7.74        8.86        9.99      order\n"
            "   50  32.400  -3.7171e-03 -2.5941e-03 -1.9195e-03 -1.4228e-03 -1.1788e-03    -\n"
            "  100  16.200  -1.3675e-03 -9.5037e-04 -6.8507e-04 -5.2895e-04 -4.2131e-04  1.484\n"
            "  200   8.100  -4.9562e-04 -3.3893e-04 -2.4728e-04 -1.9071e-04 -1.5041e-04  1.486\n"
            "* 400   4.050  -1.7954e-04 -1.2242e-04 -8.9138e-05 -6.7957e-05 -5.3602e-05  1.489\n"
            "  800   2.025  -6.4132e-05 -4.3695e-05 -3.1800e-05 -2.4234e-05 -1.9067e-05  1.491\n"
            " 1600   1.013  -2.2830e-05 -1.5548e-05 -1.1312e-05 -8.6184e-06 -6.7720e-06  1.493\n"
            " 3200   0.506  -8.1087e-06 -5.5208e-06 -4.0159e-06 -3.0593e-06 -2.4023e-06  1.495\n"
            " 6400   0.253  -2.8759e-06 -1.9577e-06 -1.4239e-06 -1.0846e-06 -8.5140e-07  1.496\n"
            "(* marks the shipped point, here and below.)\n"
            "Monotone, single-signed, order -> 1.500 over eight halvings and 3.6 "
            "decades. 3/2 is the expected midpoint order against the gap-edge "
            "inverse-square-root singularity, and it has TWO sources of the same "
            "order, not one. AUDIT finding 1: the original derivation attributed "
            "it entirely to centre-sampling of the smooth factor (E-E')^2 across "
            "the gap-cut first cell, and stated the engine's coherence factor as "
            "the pointwise K- = 1 - D^2/(E E'). That is not the code path. At "
            "dynes_gamma = 0 the engine builds K- = 1 - r_i r_j from CELL-AVERAGED "
            "N2/N1 ratios (physics/spectral.py:426-448), which differs from the "
            "pointwise form by up to 7.3e-03 across the matrix at NE = 400 "
            "(0.90063148 vs 0.90100111 on the top row's gap-edge cell). Same "
            "continuum limit, so the benchmark is unaffected -- but substituting "
            "the pointwise factor with the SAME exact cell weights halves the "
            "error (-9.3775e-05 vs -1.7954e-04 at the lowest observed cell, "
            "-2.7657e-05 vs -5.3602e-05 at the top), so roughly half the "
            "quadrature error is the cell-averaged coherence factor and half is "
            "the smooth-factor sampling. Both are order 3/2. What is NOT half "
            "and half: replacing the exact BCS cell integrals w_j with midpoint "
            "rho*dE weights gives -2.69e-02 at NE = 400 falling at order 1/2, "
            "1.5 decades worse -- the exact cell measure is load-bearing.\n\n"
            "(B) TIME INTEGRATION -- measured endpoint rate against the row sum "
            "(which contains no dt), NE = 400 fixed, top series:\n"
            "  n_steps  G*dt    (G_end/G_row - 1)   order   curve max   curve rms\n"
            "       40  0.2000   +4.687551e-04         -    3.4556e-03  1.5160e-03\n"
            "       80  0.1000   +1.175714e-04     1.995    7.3001e-04  2.9712e-04\n"
            "      160  0.0500   +2.942518e-05     1.998    2.6774e-04  1.2463e-04\n"
            "*     320  0.0250   +7.357095e-06     2.000    3.7003e-04  1.6670e-04\n"
            "      640  0.0125   +1.836906e-06     2.002    4.1420e-04  1.8099e-04\n"
            "     1280  0.0063   +4.565117e-07     2.009    4.2525e-04  1.8470e-04\n"
            "Order 2.00 to three digits (ETD2 with the weighted-balance "
            "projection, qpsim.solvers.etd). The curve column is NON-MONOTONE "
            "because the two error sources have opposite sign and cancel near "
            "G*dt = 0.05; the shipped case sits at 0.025, where the dt half "
            "(+7.4e-6) is 7.3x smaller than the dE half (-5.4e-5) on the top "
            "series, so no cancellation is being relied on.\n\n"
            "(C) FULL CURVE vs GRID at the shipped G*dt = 0.025. AUDIT finding "
            "2: the original stopped this table at NE = 800 and read it as "
            "convergence to zero. It is not, and saying so would misfire on the "
            "next maintainer who refines the grid:\n"
            "     NE    curve max   curve rms   top: dE part    dt part     total\n"
            "     50   9.4158e-03  4.1393e-03  -1.1788e-03  +7.3202e-06  -1.1715e-03\n"
            "    100   3.3173e-03  1.4513e-03  -4.2131e-04  +7.3423e-06  -4.1397e-04\n"
            "    200   1.1452e-03  5.0368e-04  -1.5041e-04  +7.3524e-06  -1.4306e-04\n"
            "*   400   3.7003e-04  1.6670e-04  -5.3602e-05  +7.3571e-06  -4.6245e-05\n"
            "    800   1.0451e-04  4.8893e-05  -1.9067e-05  +7.3593e-06  -1.1708e-05\n"
            "   1600   3.7230e-05  1.6933e-05  -6.7720e-06  +7.3604e-06  +5.8836e-07\n"
            "   3200   4.8628e-05  2.0343e-05  -2.4023e-06  +7.3610e-06  +4.9587e-06\n"
            "The dt part is NE-independent to four digits, so at FIXED G*dt the "
            "total rate error crosses zero near NE ~ 1600 and the curve residual "
            "floors at about 8*7.36e-6 = 5.9e-5 rather than converging to zero. "
            "Each half converges separately and cleanly; the sum at fixed G*dt "
            "does not. The shipped point is unaffected -- the dE half dominates "
            "7.3:1 there -- but refining the grid alone past ~1600 makes this "
            "benchmark worse, and dt has to come down with it.\n\n"
            "(D) SIZE. NE = 400 x 5 cells x 320 steps runs in 3.0 s and emits "
            "321 frames of a (400, 5) field. NE = 800 costs 5.2 s and 1600 costs "
            "11.8 s for a residual that is no longer telling the truth about the "
            "grid alone (see C), so refining was rejected in favour of stating "
            "the floor.\n\n"
            "(E) SHAPE. Dividing each series by its own best-fit exponential "
            "leaves max |sim/fit - 1| = 6.8e-08 / 6.4e-07 / 3.4e-06 / 1.2e-05 / "
            "3.5e-05 against curve residuals of 1.93e-04 / 2.49e-04 / 3.00e-04 / "
            "3.43e-04 / 3.70e-04. So 0.04% to 9.3% of each series' residual is "
            "structure beyond a single rate error: the trajectories ARE "
            "exponential, and essentially the whole residual is the rate, i.e. "
            "the kernel."
        ),
        modes=("spatial_2d",),
        build=_build,
        caveat=(
            "1. IT TESTS ROW SUMS, NOT THE DISTRIBUTION OF THE OUTGOING FLUX. "
            "Gamma is SUM_{j<i} K0s[i,j]*w_j. A kernel with correct row sums but a "
            "wrong distribution of final states within the row passes unchanged. "
            "Five rows spanning a factor 7.4 in Gamma widen this but do not "
            "close it.\n"
            "2. THE GAIN SIDE IS NOT TESTED AT ALL. The observed cell is chosen "
            "precisely because its gain term is identically zero. The original "
            "derivation offered number conservation as evidence tying the gain "
            "side to the loss side; AUDIT finding 3 is right that it carries no "
            "such information, and it is worth being precise about why. With "
            "gain = (1-f)*(K.T @ (w*f)) and loss = K @ (w*(1-f)) "
            "(collisions/spatial.py:367-368), SUM_i w_i*(gain_i - loss_i*f_i) = "
            "SUM_ij w_i w_j K_ji (1-f_i) f_j - SUM_ij w_i w_j K_ij f_i (1-f_j), "
            "and relabelling i<->j in the first sum makes the two identical. It "
            "is zero for ANY matrix whatsoever, including a wrong one. The other "
            "use of number conservation -- that x_qp cannot see this term, so it "
            "is the wrong observable to check it with -- is correct and stands.\n"
            "3. THE ABSORPTION BRANCH IS SWITCHED OFF BY CONSTRUCTION. At 1 mK "
            "N_p is exactly 1 for E_i > E_j and 0 for E_i < E_j, so nothing here "
            "sees the 1+n_BE / n_BE branch selection or the Bose factor itself. "
            "AUDIT finding 4: that blindness is STRUCTURAL and cannot be "
            "repaired by running warmer. The residual is flat to 6% of itself "
            "from 1 mK to 0.2 K while n_BE(dE) goes from 4e-21 to 3.8, because "
            "the (E_i-E_j)^2 weight suppresses exactly the small transfers whose "
            "Bose factor is large; and by 0.5 K, where the factor would finally "
            "matter, the closed form does not start testing that branch, it "
            "fails outright (1.93e-2, 19x the tolerance). Detailed balance "
            "(validation/analytic/test_detailed_balance.py) covers that branch "
            "and conversely cannot see a rate normalisation.\n"
            "4. PAULI BLOCKING IS DELIBERATELY NEGLIGIBLE. Max occupation "
            "anywhere over the whole run is the 1e-6 seed itself. The nonlinear "
            "(1-f) factor is not tested; the amplitude guard is what keeps it "
            "that way.\n"
            "5. THE KERNEL DIAGONAL is zeroed by construction and never probed.\n"
            "6. TIME-INTEGRATION SENSITIVITY IS WEAK. ETD2 solves df/dt = -Gf "
            "exactly for a frozen rate, so the only dt error is coefficient "
            "drift plus the weighted-balance projection. Section (B) isolates it "
            "and finds clean order 2, but a bug that preserved exactness for a "
            "linear autonomous problem would be invisible here.\n"
            "7. THE OBSERVED ROW IS TAKEN FROM THE PREPARED STATE, so this "
            "benchmark does NOT test that the initial-condition machinery seeded "
            "the cell that was asked for. It tests the decay of whichever cell "
            "was in fact prepared highest. The alternative -- a pinned index -- "
            "would observe the wrong row silently if the catalogue's seed "
            "expression were edited. The ``note`` reports the observed cells and "
            "energies so the choice is visible in the payload.\n"
            "8. A FACTOR 7.4 IN GAMMA IS NOT THE 3.74-DECADE SWEEP. AUDIT "
            "finding 6 correctly said the original single 321-point curve "
            "overstated its discriminating power (~91% of its residual is one "
            "scalar). This ships five independent rates instead, but they span "
            "E/Delta from 5.49 to 9.99, not the 16-point, 3.74-decade sweep, "
            "which needs 16 runs at different grid tops and is not a single-run "
            "UI curve. The span is bounded by the shared abscissa. A rate error "
            "dG/G shows up as a pointwise relative error (G*t)*(dG/G), largest "
            "at the end of the window, so the tolerance rejects dG/G above "
            "1.25e-4 on the fastest series (G*T = 8) and 9.3e-4 on the slowest "
            "(G*T = 1.08). Widening the window buys span, but slowly and not "
            "far: the fastest series' own residual grows as Gamma_err*t and "
            "reaches the tolerance at about 21 e-foldings -- 2.7x the shipped "
            "window, so a factor ~20 in Gamma rather than 7.4 -- and past that "
            "the framework's 1e-9 relative floor starts dropping its points. "
            "Three decades in Gamma is not reachable on one abscissa at this "
            "tolerance.\n"
            "9. THE TIER'S OWN BOUNDARY. T1 means the analytic side reads no "
            "engine array, and it does not -- Gamma is rebuilt from Delta_0, "
            "T_c, tau_0, the grid triple and a literal k_B. What no T1 of this "
            "construction can exclude is a CONVENTION shared by the derivation "
            "and the engine: if both misread the definition of tau_0 in the "
            "prefactor 1/(tau_0 (k_B T_c)^3), neither catches it. That prefactor "
            "is Kaplan et al., PRB 14, 4854 (1976) Eq. 8, and tau_0 = 438 ns is "
            "Kaplan's own aluminium number, but a physicist should confirm it "
            "against the paper rather than against this benchmark.\n"
            "10. SCOPE. The 0-D reduction of the 2-D core: D_0 = 0, so spatial "
            "transport is switched off and untested, and each column is solved "
            "by the same collision layer a single cell would use. Frozen gap, "
            "phonons pinned at the Bose bath, one material, ideal BCS "
            "(dynes_gamma = 0), a uniform cell-centred grid whose lower edge "
            "sits exactly at Delta. Nothing about diffusion, gap "
            "self-consistency or the phonon-side kernel is exercised. The Kaplan "
            "S+ pair-breaking correction is not on this path (it applies to the "
            "phonon-side recombination kernel), so this benchmark is unaffected "
            "by, and cannot adjudicate, the pending re-baseline of that "
            "quantity."
        ),
        activity=(
            "1. THE NULL RUN IS BIT-EXACT AND THE BENCHMARK FAILS ON IT. With "
            "collisions.scattering = false and everything else already off, the "
            "right-hand side is empty: max |f(t) - f(0)| over all 321 frames x "
            "400 bins x 5 columns is 0.000e+00 -- bit-identical to the prepared "
            "state, not merely close. Scored against the same closed form that "
            "null run gives 2.9800e+03 = 2.98e+06 x rel_tol. This is not a check "
            "that passes either way.\n"
            "2. THE MOTION IS THE TERM. The five curves fall by 0.47 / 0.89 / "
            "1.51 / 2.36 / 3.47 decades (f_end/f_0 = 3.394e-01, 1.275e-01, "
            "3.063e-02, 4.342e-03, 3.356e-04) in 2.16 ns. With the term off they "
            "are flat at 1. There is no baseline and no other process on the "
            "right-hand side that could produce a 0.27 ns lifetime.\n"
            "3. FIVE RATES, NOT ONE, AND THEY ARE MEASURED SEPARATELY. Best-fit "
            "rates 0.49968920 / 0.95239565 / 1.61211369 / 2.51566505 / "
            "3.69967009 /ns against closed forms 0.49977861 / 0.95251080 / "
            "1.61225320 / 2.51582712 / 3.69985285 -- errors -1.79e-04 / "
            "-1.21e-04 / -8.65e-05 / -6.44e-05 / -4.94e-05, smooth and monotone "
            "in energy with no outlier. One global rate error cannot produce "
            "five values that track a closed form across a factor 7.4.\n"
            "4. FALSIFIABILITY -- three deliberate kernel corruptions with the "
            "analytic side untouched, all rejected, and each with a DIFFERENT "
            "energy signature that only a multi-energy curve can see:\n"
            "     tau_0 low by 1% (pure normalisation)  max 7.7285e-02 (77x tol); "
            "rate errors +0.0099 +0.0100 +0.0100 +0.0100 +0.0101 -- flat in E, "
            "as a normalisation must be.\n"
            "     coherence factor K- dropped (set to 1)  max 3.3892e-01 (339x); "
            "rate errors +0.1322 +0.0986 +0.0771 +0.0624 +0.0517 -- FALLING with "
            "energy, because 1 - D^2/(E E') -> 1.\n"
            "     (E_i-E_j)^2 -> (E_i-E_j)^2.02  max 3.3469e-01 (335x); rate "
            "errors +0.0367 +0.0413 +0.0450 +0.0482 +0.0509 -- RISING with "
            "energy. Opposite slope to the previous one at a similar magnitude: "
            "the shipped curve separates them, a single-energy curve could not.\n"
            "     unmutated control  max 3.7003e-04, accepted.\n"
            "5. TRAP #4 EXPLICITLY AVOIDED. Scattering conserves quasiparticle "
            "number identically, so x_qp -- the observable this mode's summary "
            "leads with -- is blind to this term and would show a flat line "
            "whether it were on or off. The per-cell occupation is the "
            "observable the term actually moves."
        ),
        extra={
            "case_overrides": CASE_OVERRIDES,
            # Pinned because everything above is a measurement, and a
            # measurement without the tree it was taken on is an anecdote.
            "verified_at_commit": "41ee7f8",
        },
    )
)
