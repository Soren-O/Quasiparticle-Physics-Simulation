"""Recombination: a one-cell excess decays by the exact two-body ``1/t`` law.

Switch scattering off, both photon drives off, pin the phonons at the bath and
hold the gap fixed, and the whole right-hand side of the kinetic equation is

    df_i/dt = (1 - f_i) sum_j K^r_ij w_j N^abs_ij (1 - f_j)
            -        f_i sum_j K^r_ij w_j N^emit_ij f_j.

Two facts then collapse this to one scalar equation. First, the loss term
removes a quasiparticle from the bin it is already in and books its partner in
the partner's own bin -- the Kaplan Eq. (8) per-quasiparticle normalisation,
no leading 2 -- so **no quasiparticle is moved between bins**. Second, the
pair-breaking gain is proportional to n_B(E_i + E_j), which on this grid at
60 mK is at most n_B(2 E_i0) = 3.2e-31. So an excess prepared in a single
energy cell stays in that cell and obeys

    dg/dt = -R g^2,    R = w_i0 K^r(E_i0, E_i0) [1 + n_B(2 E_i0, T_bath)],

whose solution is the two-body law. In the published observable
``x_qp = (1/Delta_0) sum_i w_i f_i`` that is

    x_qp(t) = x_th + X_0 / (1 + R_x X_0 t),   R_x = Delta_0 K^r,  X_0 = w_i0 A / Delta_0.

Zero free parameters: A is the amplitude the setup asks for, and w_i0, r_i0 and
K^r are elementary integrals of the BCS density of states over one cell.

Why the seed sits in the GAP-EDGE cell
--------------------------------------
An earlier version of this benchmark seeded the cell at 2*Delta. An adversarial
audit then showed that the case could not tell the engine's finite-volume
convention apart from the obvious alternative: replacing the exact cell
integrals with point-sampled ``rho(E_c) dE`` and ``Delta/E_c`` moved the
prediction by only 3.1e-06, sixteen times BELOW the tolerance, so the case
passed either way and the convergence table offered as proof of the convention
was a statement about two formulas rather than about the engine.

Seeding the first cell fixes that, because that is where the finite-volume
average of the integrable ``1/sqrt`` singularity is nothing like its centre
value. On the shipped grid the same two substitutions now move the prediction
by 4.11e-01 and 2.72e-03 -- 8226x and 54x the tolerance (measured; see
``caveat``). The check is therefore a real test of which discretisation the
engine implements, which is the strongest claim available here and is NOT the
same as a test of the continuum limit. See the tier note below.

Where the residual comes from
-----------------------------
All of it is time discretisation, and its order is derived rather than fitted.
ETD2's predictor is the exponential ETD1 step and its corrector applies the
trapezoidal number balance, which is the projection the published observable
is read from. On ``dN/dt = -K N^2`` with ``s = K N h`` that composition gives

    N_new/N = 1 - (s/2)(1 + e^(-2s)) = 1 - s + s^2 - s^3 + (2/3) s^4

against the exact ``1/(1+s) = 1 - s + s^2 - s^3 + s^4``. The local error is
``-(1/3) s^4`` and the global order is 3 -- which the ladder in ``convergence``
reproduces to three digits, ratio 8 per halving converging to 8 from above.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.constants import KB_UEV_PER_K
from qpsim.webui.benchmarks import Benchmark, Curve, register

# The case this benchmark is written for, as web-schema override paths. Every
# field below is reachable through Spatial2DSetup today; the run in
# ``convergence`` was produced by validating exactly this dict and handing it
# to qpsim.webui.execute.execute_setup.
#
# The ones that are not free choices:
#
#   geometry 1x1 is the 0-D reduction of the 2-D mode. A single cell has no
#     faces, so the transport half of the split step is structurally absent
#     rather than merely small -- T3SpatialBackend.apply_transport returns the
#     state untouched. That is a stronger statement than D_0 = 0, which would
#     still be a claim about an operator that got built.
#   grid.min_factor = 1.0 puts the first cell edge exactly at E = Delta, so the
#     seeded cell is the gap-edge cell [Delta, Delta + dE] and its weight is the
#     full singular integral. This is what gives the case its grip on the
#     finite-volume convention (module docstring).
#   T_bath = 0.06 K. The neglected term is the thermal PARTNERS the excess
#     recombines against, and the audited 0-D construction measured its cost
#     over 0.06-0.20 K as x_th/x_excess(t_end) times an O(1) coefficient. Here
#     that ratio is 4.2e-13, seven orders of magnitude below the discretisation
#     floor. Raising the bath to 0.20 K takes it to 3.7e-03 -- 74x the gate --
#     on a window nine times SHORTER than the shipped one, and _build refuses
#     the run rather than reporting a physics error as a discretisation one.
#   initial.energy.width = dE/12 makes the "monoenergetic" Gaussian a SINGLE
#     CELL: the neighbouring cell receives exp(-72) = 5e-32 of the peak, and the
#     measured excess number outside the seeded cell is 6.8e-30 of the total. A
#     single cell is not a physical state, it is the device that makes the law
#     exact rather than asymptotic -- a spread seed distorts as it decays,
#     because the per-quasiparticle rate depends on energy. Measured here:
#     widening to sigma = dE/2 puts 5.4e-02 of the excess outside the cell, and
#     sigma = 0.05 Delta (three cells) misses the one-cell law by 1.04, i.e.
#     100%. The gap edge is unforgiving about this because w and r change by
#     tens of percent from one cell to the next there.
#   dt = 12 ns is part of the case, not a performance knob. The gate is a
#     discretisation floor measured at this step (see ``convergence``), and the
#     ETD2 rate limiter starts subcycling above 16.4 ns.
#   stop_tol = 0.0 so the run always reaches max_time. This is a relaxation
#     measurement, not a steady-state search: the run is correctly reported as
#     "not converged" and that note is expected.
#   snapshot_interval = 144 ns = 12*dt, so every recorded time lands exactly on
#     an integration endpoint (measured: max |t/dt - round(t/dt)| = 0.0). No
#     dense output is interpolated into the comparison.
#   phonon_scattering_source / phonon_recombination_source are pinned at their
#     defaults for completeness only: run_spatial_2d does not forward them to
#     the backend, and with phonons.mode = "thermal_bath" there is no phonon
#     equation for them to remove a term from. They are inert here in two
#     independent ways, and the case says so rather than leaving them unstated.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material.name": "Al",
    "material.Delta_0": 180.0,
    "material.T_c": 1.18,
    "material.tau_0": 438.0,
    "material.dynes_gamma": 0.0,
    "T_bath": 0.06,
    "grid.min_factor": 1.0,
    "grid.max_factor": 4.0,
    "grid.num_bins": 180,
    "geometry.kind": "rectangle",
    "geometry.rows": 1,
    "geometry.cols": 1,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "collisions.scattering": False,
    "collisions.recombination": True,
    "collisions.phonon_scattering_source": True,
    "collisions.phonon_recombination_source": True,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "thermal_bath",
    "self_consistent_gap": False,
    "initial.kind": "excess",
    "initial.amplitude": 0.2,
    "initial.energy.kind": "monoenergetic",
    "initial.energy.E_0": 181.5,   # centre of cell 0 = Delta + dE/2
    "initial.energy.width": 0.25,  # dE/12; see the note above
    "initial.space.kind": "uniform",
    "drives": [],
    "dt": 12.0,
    "max_time": 27000.0,
    "stop_tol": 0.0,
    "snapshot_interval": 144.0,
}

# The prepared excess must sit in ONE cell for the closed form to be exact.
# This bounds the fraction of the excess QUASIPARTICLE NUMBER (weight x
# occupation, not occupation alone) that landed anywhere else. The shipped case
# measures 6.8e-30 against it, so the threshold is not load-bearing -- it exists
# to reject a Gaussian somebody widened without re-deriving the law.
_MAX_OFF_CELL_FRACTION = 1e-6
# The thermal background is a constant in the closed form, but the thermal
# PARTNERS it recombines against are dropped. The measured penalty for that is
# x_th/x_excess(t_end) times an O(1) coefficient, so this caps the dropped term
# at a fifth of the tolerance. Above it the pure two-body law is not the
# solution and reporting a verdict against it would be reporting a physics
# error as a discretisation one.
_MAX_THERMAL_RATIO = 1e-5
# The closed form must predict real decay over the window, or the check reduces
# to "the constant X_0 equals the constant X_0". R_x X_0 t_end below this is a
# run too short (or a rate too small) to measure anything.
_MIN_DECAY = 1.0


def _fermi(energies: np.ndarray, temperature: float) -> np.ndarray:
    """f_FD, written so the deep-subgap tail stays representable.

    ``exp(-z)/(1+exp(-z))`` rather than ``1/(exp(z)+1)``: at 60 mK the top of
    this grid has z = 139, where the second form overflows to inf and returns
    an exact zero instead of the 6e-61 that is actually there.
    """
    z = np.asarray(energies, dtype=float) / (KB_UEV_PER_K * float(temperature))
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(-z)
        return e / (1.0 + e)


def _cell_integrals(
    energies: np.ndarray, gap: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(w_i, r_i, dE)`` -- the exact BCS integrals over each grid cell.

    ``w_i = int_cell rho dE`` and ``r_i = (int_cell rho Delta/E dE) / w_i``,
    both elementary:

        int E/sqrt(E^2 - D^2) dE = sqrt(E^2 - D^2),
        int D/sqrt(E^2 - D^2) dE = D arccosh(E/D).

    Written out here rather than read from ``ctx.cell_weights`` /
    ``ctx.K_plus``. That is deliberate but it is not, on its own, independence:
    these ARE the integrals the engine's spectral context evaluates, and they
    agree with it to 0.0. What the comparison buys is stated in ``caveat``.
    """
    e = np.asarray(energies, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("The energy grid must be a 1-D grid of at least 2 cells.")
    widths = np.diff(e)
    d_e = float(widths[0])
    if np.max(np.abs(widths - d_e)) > 1e-9 * d_e:
        raise ValueError(
            "The energy grid is not uniform, so the cell edges assumed below "
            "(centre +/- dE/2) are not this run's cells and every integral "
            "would be over the wrong interval."
        )
    edges = np.concatenate((e - 0.5 * d_e, [e[-1] + 0.5 * d_e]))
    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], lo)
    # (E - D)(E + D) rather than E^2 - D^2: at the gap edge the latter is a
    # cancellation of two nearly equal numbers and loses half the digits in
    # exactly the cell this case seeds.
    w = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0)) - np.sqrt(
        np.maximum((lo - gap) * (lo + gap), 0.0)
    )
    anomalous = gap * (
        np.arccosh(np.maximum(hi / gap, 1.0)) - np.arccosh(np.maximum(lo / gap, 1.0))
    )
    r = np.zeros_like(w)
    live = w > 0.0
    r[live] = np.clip(anomalous[live] / w[live], 0.0, 1.0)
    return w, r, d_e


def _require_reduction(setup: Any, summary: dict[str, Any]) -> None:
    """Refuse a case the closed form does not describe.

    Note what is NOT checked here: ``collisions.recombination``. A benchmark
    that raised on the term being off would report "could not be evaluated"
    instead of a verdict, and the whole point of this entry is that switching
    the term off makes it FAIL by six orders of magnitude.
    """
    if int(summary.get("cells", 0)) != 1:
        raise ValueError(
            f"This closed form is the 0-D reduction and the run has "
            f"{summary.get('cells')} cells. On more than one cell the same law "
            "holds cell by cell only while transport is exactly inert, which is "
            "a separate claim this benchmark does not make. Use a 1x1 mask."
        )
    if setup.collisions.scattering:
        raise ValueError(
            "collisions.scattering is on. Scattering redistributes the excess "
            "over energy, so it leaves the single cell the two-body law is "
            "derived for and the reduction does not close."
        )
    if setup.phonons.mode != "thermal_bath":
        raise ValueError(
            f"phonons.mode is {setup.phonons.mode!r}. The closed form uses the "
            "bath occupation n_B(E_i + E_j) for every emitted phonon; with a "
            "solved phonon sector the emission factor grows as the population "
            "builds up and the decay is no longer a pure two-body law."
        )
    if setup.self_consistent_gap:
        raise ValueError(
            "self_consistent_gap is on. Every coefficient below is an integral "
            "over cells cut by a FIXED gap; a moving gap changes the weights, "
            "the coherence factor and the rate during the run."
        )
    if setup.injection.enabled or any(d.enabled for d in setup.drives):
        raise ValueError(
            "An external source is enabled. The excess must decay freely: a "
            "source adds quasiparticles the closed form does not account for."
        )
    if setup.subgap_drive.enabled or setup.pb_drive.enabled:
        raise ValueError(
            "A photon drive is enabled. Both photon channels move quasiparticles "
            "in energy or create them, neither of which the reduction contains."
        )
    if setup.gap_regions.kind != "uniform":
        raise ValueError(
            "gap_regions is not uniform. Cells with different gaps have "
            "different spectral weights and different kernels, so there is no "
            "single R to decay with."
        )
    initial = setup.initial
    if initial.kind != "excess" or initial.expression is not None:
        raise ValueError(
            "This benchmark needs initial.kind = 'excess' with a separable "
            "profile: the closed form's X_0 is w_i0 * initial.amplitude / Delta, "
            "which is only what was prepared for an excess over the bath."
        )
    if initial.energy.kind != "monoenergetic" or initial.energy.E_0 is None:
        raise ValueError(
            "initial.energy.kind must be 'monoenergetic' with an E_0: the "
            "reduction closes because the excess occupies one cell, and every "
            "other energy profile spreads it over many."
        )
    if initial.space.kind != "uniform":
        raise ValueError(
            "initial.space.kind must be 'uniform'. On a single cell that is the "
            "only shape there is, and asking for it explicitly keeps the case "
            "readable when somebody widens the mask."
        )


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    if "snap_t_ns" not in arrays or "snap_f" not in arrays:
        raise ValueError(
            "This benchmark reads the recorded frames: set snapshot_interval on "
            "the setup. Without them the only observable is the endpoint, and a "
            "single number says nothing about the RATE, which is the whole "
            "content of the closed form."
        )
    _require_reduction(setup, summary)

    times = np.asarray(arrays["snap_t_ns"], dtype=float)
    sim = np.asarray(arrays["obs_x_qp_mean"], dtype=float)
    energies = np.asarray(arrays["E_bins"], dtype=float)
    if times.size < 8:
        raise ValueError(
            f"Only {times.size} recorded frame(s). A two-body law fitted to a "
            "handful of points is a scalar check wearing a curve's clothes."
        )

    # The reference gap, checked against the gap the run actually carried in
    # every cell and every frame. Every integral below is over cells cut by
    # THIS gap, so a profile that moved during the run (or differs by region)
    # would make all of them integrals of the right function over the wrong
    # interval -- the one error that looks like agreement.
    gap = float(summary["gap_ueV"])
    recorded = np.asarray(arrays.get("snap_gap", np.array([gap])), dtype=float)
    if recorded.size and np.max(np.abs(recorded - gap)) > 1e-9 * gap:
        raise ValueError(
            "The local gap is not the reference gap in every cell and frame, so "
            "the cell weights and coherence factor below are not the ones the "
            "run used."
        )

    w, r, d_e = _cell_integrals(energies, gap)
    i0 = int(np.argmin(np.abs(energies - float(setup.initial.energy.E_0))))
    amplitude = float(setup.initial.amplitude)

    # Is the prepared state the one-cell excess the closed form is derived for?
    # Checked against the run's OWN first frame, so a change in how the builder
    # normalises a profile shows up as this message rather than as a curve that
    # mysteriously misses. The amplitude itself still comes from the setup: it
    # is the number the case asks for, and reading it back off the frame would
    # make the benchmark unable to notice if it were not delivered.
    first = np.asarray(arrays["snap_f"][0], dtype=float)[:, 0]
    excess = first - _fermi(energies, setup.T_bath)
    total = float(np.sum(w * excess))
    if total <= 0.0:
        raise ValueError(
            "The run does not start above the bath: there is no excess to "
            "recombine and the closed form is the constant x_th."
        )
    off_cell = float(np.sum(np.abs(w * excess)) - abs(w[i0] * excess[i0])) / total
    if off_cell > _MAX_OFF_CELL_FRACTION:
        raise ValueError(
            f"{off_cell:.3e} of the prepared excess sits outside cell {i0} "
            f"(E = {energies[i0]:.4g} ueV), past the {_MAX_OFF_CELL_FRACTION:g} "
            "this closed form allows. Recombination moves no quasiparticle "
            "between bins, so a spread seed decays with an energy-dependent "
            "rate and distorts as it goes -- the law is then asymptotic, not "
            "exact. Narrow initial.energy.width."
        )
    if abs(excess[i0] - amplitude) > 1e-9 * amplitude:
        raise ValueError(
            f"The seeded cell holds an excess occupation of {excess[i0]:.6g}, "
            f"not the initial.amplitude = {amplitude:.6g} the closed form's X_0 "
            "is built from (clipped against the ceiling, or peak-normalised "
            "against a different bin)."
        )

    # ---- the closed form -------------------------------------------------
    kb_tc = KB_UEV_PER_K * float(setup.material.T_c)
    pair = 2.0 * float(energies[i0])
    y = pair / (KB_UEV_PER_K * float(setup.T_bath))
    with np.errstate(over="ignore", under="ignore"):
        n_bose = float(np.exp(-y) / (-np.expm1(-y)))
    k_r = (
        (1.0 / float(setup.material.tau_0))
        * (pair / kb_tc) ** 2
        / kb_tc
        * (1.0 + float(r[i0]) ** 2)
    )
    rate = gap * k_r * (1.0 + n_bose)                 # R_x, per ns per unit x_qp
    x_0 = float(w[i0]) * amplitude / gap              # X_0, the excess at t = 0
    x_th = float(np.sum(w * _fermi(energies, setup.T_bath)) / gap)

    decay = rate * x_0 * float(times[-1])
    if decay < _MIN_DECAY:
        raise ValueError(
            f"The closed form predicts R_x X_0 t = {decay:.3g} over this run, so "
            "the excess barely moves and a verdict of 'pass' would certify only "
            "that two constants agree. Run longer, or seed a larger amplitude."
        )
    thermal_ratio = x_th / (x_0 / (1.0 + decay))
    if thermal_ratio > _MAX_THERMAL_RATIO:
        raise ValueError(
            f"The thermal population is {thermal_ratio:.3e} of the excess left "
            f"at t = {times[-1]:g} ns. The dropped thermal PARTNERS then bend "
            "the decay by about that much -- the audited 0-D construction "
            "measured the coefficient as O(1) over 0.06-0.20 K -- which is a "
            "physics error and would be reported here as a discretisation one. "
            "Lower T_bath, or shorten the run."
        )

    analytic = x_th + x_0 / (1.0 + rate * x_0 * times)

    span = float(sim[0] / sim[-1])
    note = (
        f"{times.size} frames over {times[-1]:g} ns on a single cell; the "
        f"dimensionless decay variable R_x X_0 t runs 0 -> {decay:.4g}, so x_qp "
        f"falls {sim[0]:.6e} -> {sim[-1]:.6e}, a factor {span:.2f}. The seed is "
        f"cell {i0} (E = {energies[i0]:.4g} ueV, dE = {d_e:g}), whose exact BCS "
        f"integrals are w = {w[i0]:.6g} ueV and r = {r[i0]:.8f}; that gives "
        f"R_x = {rate:.9g} /ns, X_0 = {x_0:.9g} and a knee at 1/(R_x X_0) = "
        f"{1.0 / (rate * x_0):.4g} ns. The emission factor 1 + n_B(2E) is "
        f"1 + {n_bose:.3g} and the thermal background is {thermal_ratio:.2e} of "
        f"the excess left at the end, so neither is being tested here. The t = 0 "
        f"point is in the comparison and in the verdict; a log abscissa cannot "
        f"draw it."
    )

    return Curve(
        x=times,
        y_sim=sim,
        y_analytic=analytic,
        x_label="t (ns)",
        y_label="x_qp = n_qp / (4 rho_F Delta_0)",
        # Both axes logarithmic. On a linear abscissa the knee at ~266 ns is
        # the first 1% of a 27000 ns run and the whole shape is invisible;
        # log-log shows the flat head and the -1 slope of the 1/t tail, which
        # is what distinguishes this from the exponential a reader expects.
        log_x=True,
        log_y=True,
        note=note,
    )


register(Benchmark(
    name="recombination",
    title="Recombination - two-body decay x_qp(t) = x_th + X0/(1 + R_x X0 t)",
    # T1, with the qualification spelled out because an audit has already shown
    # the unqualified claim to be false.
    #
    # T1 in the sense the framework defines: the reference is written from the
    # physics (BCS cell integrals, the Kaplan tau_0 prefactor, the Bose factor,
    # the exact solution of dg/dt = -R g^2), it reads no kernel or weight array
    # from the engine, and nothing in it is fitted.
    #
    # What that does NOT buy, and what an earlier version of this entry wrongly
    # claimed it did: independence from the engine's DISCRETISATION. The cell
    # integrals here are the same integrals qpsim.physics.spectral evaluates,
    # and they reproduce ctx.cell_weights and ctx.K_plus to 0.0 -- so on the
    # discretisation question this reference is worth exactly what a T2 built
    # from those arrays would be worth, and the previous entry's claim that "a
    # T2 could not make that statement at all" was simply wrong.
    #
    # What it does buy is the two things a T2 cannot: the ABSOLUTE rate (a
    # 0.1% error in the engine's kernel is 20x the gate, measured) and the
    # SELECTION of the discretisation (the plausible point-sampled alternative
    # is 54x and 8226x the gate on this case, measured). The continuum limit of
    # the convention is a T3 question and is not answered here.
    tier="T1",
    formula_latex=(
        r"x_{qp}(t)=x_{qp}^{\rm th}+\frac{X_0}{1+R_xX_0t},\qquad"
        r"X_0=\frac{w_{i_0}A}{\Delta_0},\qquad"
        r"R_x=\Delta_0\,K^{r}(E_{i_0},E_{i_0})\bigl[1+n_B(2E_{i_0},T_b)\bigr]"
        r"\\[6pt]"
        r"K^{r}(E,E')=\frac{1}{\tau_0}\Bigl(\frac{E+E'}{k_BT_c}\Bigr)^{2}"
        r"\frac{1+rr'}{k_BT_c}"
        r"\\[6pt]"
        r"w_i=\!\!\int_{\rm cell}\!\!\rho\,dE=\sqrt{h^2-\Delta^2}-\sqrt{l^2-\Delta^2},"
        r"\qquad"
        r"r_i=\frac{1}{w_i}\!\!\int_{\rm cell}\!\!\rho\frac{\Delta}{E}dE"
        r"=\frac{\Delta\bigl[\cosh^{-1}(h/\Delta)-\cosh^{-1}(l/\Delta)\bigr]}{w_i}"
    ),
    reason=(
        "With scattering, both drives and transport removed and the phonons "
        "pinned at the bath, recombination is the only term left, and it moves "
        "no quasiparticle between energy bins (Kaplan Eq. 8: the partner is "
        "booked in the partner's own bin). A one-cell excess therefore stays in "
        "its cell and the NE-dimensional system collapses to dg/dt = -R g^2, "
        "whose exact solution is the two-body 1/t law. R is written down from "
        "elementary BCS cell integrals with no free parameter, so the height "
        "AND the rate of the whole curve are predicted rather than fitted."
    ),
    headline_latex=(
        r"x_{qp}(t) \;=\; x_{qp}^{\rm th} \;+\; \frac{X_0}{1 + R_x X_0\,t}"
    ),
    rel_tol=5e-05,
    convergence=(
        "SHIPPED CASE: NE = 180 cells over [1, 4]Delta (dE = 3 ueV), a 1x1 mask, "
        "T_bath = 0.06 K, dt = 12 ns, 2250 steps to 27000 ns, 189 recorded "
        "frames. 3.4 s wall end to end through webui.execute. Measured max "
        "pointwise |sim/analytic - 1| = "
        "6.062710e-06, rms 1.138852e-06, worst point at t = 144 ns = 0.54 x the "
        "knee 1/(R_x X_0) = 266.2 ns, which is where a wrong rate constant shows "
        "first. Deterministic: three runs reproduced 6.0627103806e-06 to every "
        "digit.\n"
        "\n"
        "TIME REFINEMENT, everything else fixed (same 189 output times):\n"
        "  dt (ns)  steps    max rel err     rms rel err    ratio   order\n"
        "   12       2250    6.062710e-06    1.138852e-06     -       -     <- SHIPPED\n"
        "    6       4500    7.371296e-07    1.389599e-07    8.225   3.040\n"
        "    3       9000    9.085980e-08    1.715971e-08    8.113   3.020\n"
        "    1.5    18000    1.127782e-08    2.131933e-09    8.057   3.010\n"
        "Ratio 8 per halving, converging to 8 from above, matching the -(1/3)s^4 "
        "local error derived in the module docstring: order 3, not 2, because "
        "the observable is read from ETD2's trapezoidal number balance. The "
        "residual converges to ZERO, i.e. the closed form IS the solution the "
        "scheme converges to, so the engine's assembled rate equals R_x far "
        "tighter than the gate.\n"
        "\n"
        "ENERGY-GRID REFINEMENT at fixed dt/(knee time), which is the honest "
        "version of this table -- it shows the closed form is exact at every NE, "
        "not that it converges to anything:\n"
        "  NE     dE     E_i0     w_i0      R_x (1/ns)    dt (ns)   max rel err\n"
        "   90    6.0    183.00   46.861    0.10357714     8.358    6.127452e-06\n"
        "  180    3.0    181.50   33.000    0.10244250    12.000    6.062710e-06\n"
        "  360    1.5    180.75   23.286    0.10187742    17.100    6.127452e-06\n"
        "Flat to 1% across a 4x grid refinement while w_i0 moves by a factor "
        "2.01 and R_x by 1.7%: the residual is a pure function of dt/tau with no "
        "grid dependence. It does not fall because dt/tau was held fixed on "
        "purpose; the ladder above is what converges.\n"
        "\n"
        "THE GATE PRESUMES dt = 12 ns. dt = 16 ns gives 1.463535e-05 (0.29x the "
        "gate, on the ladder: 1.46e-05/6.06e-06 = 2.41 against the predicted "
        "(16/12)^3 = 2.37). dt = 24 ns gives 2.901002e-05 (0.58x) but is "
        "OFF-LADDER: the largest per-bin loss rate at t = 0 is 1.450111e-02 /ns, "
        "so the ETD2 rate limiter (max_loss_step = 0.25) subcycles above 16.4 ns "
        "and the effective step no longer follows dt. At the shipped 12 ns there "
        "is 1.36x headroom to that bound and the ladder is a clean dt "
        "refinement.\n"
        "\n"
        "TOLERANCE: 5e-05 is the number the 2026-08 audit recommended and is "
        "kept unchanged; the shipped residual sits 8.2x under it. The headroom "
        "absorbs BLAS/numpy variation, not a measured spread -- there is none. "
        "For scale, the gate rejects a rate error above about 0.005%: perturbing "
        "R_x by 0.01% moves the curve 9.8868e-05 = 2.0x the gate."
    ),
    modes=("spatial_2d",),
    build=_build,
    caveat=(
        "WHAT THIS DOES NOT PROVE.\n"
        "\n"
        "1. IT IS NOT INDEPENDENT OF THE DISCRETISATION. The cell integrals in "
        "the reference are the same integrals the engine's spectral context "
        "evaluates; they agree with ctx.cell_weights and ctx.K_plus to 0.0. So "
        "'no engine array is read' is true of the code and buys nothing about "
        "the finite-volume convention. What the case DOES do, which the earlier "
        "2*Delta-seeded version did not, is discriminate that convention against "
        "its obvious alternative. Scoring the SHIPPED RUN against a closed form "
        "rewritten with point-sampled rho(E_c) dE instead of the exact cell "
        "integral gives 4.1129e-01 (8226x the gate), and with point-sampled "
        "Delta/E_c instead of the cell-averaged r, 2.7152e-03 (54x) -- i.e. the "
        "two conventions are that far apart HERE, so an engine implementing the "
        "other one could not pass. At a 2*Delta seed those same "
        "substitutions moved the prediction by 3.1e-06, 16x BELOW the gate, and "
        "an audit demonstrated the case passed either way. Whether the "
        "convention is the right approximation to CONTINUUM BCS is a separate, "
        "T3-class question and is not answered here.\n"
        "\n"
        "2. ONE MATRIX ELEMENT. Only the diagonal K^r(E_i0, E_i0) at the "
        "gap-edge cell is pinned, in absolute units. The off-diagonal entries "
        "and the rest of the diagonal are untouched by this curve. (Companion "
        "constructions exist -- a 30-energy sweep of the diagonal, and a tracer "
        "row whose exponents are the kernel ratios K^r_j,i0/K^r_i0,i0 -- but "
        "they are not what this single-run benchmark ships, and the ratios "
        "cancel tau_0 and k_B T_c in any case.)\n"
        "\n"
        "3. THE BOSE EMISSION FACTOR IS NOT TESTED. 1 + n_B(2 E_i0) = "
        "1 + 3.2e-31 here, so this pins the T = 0 emission rate and nothing "
        "more. It is not reachable by warming the bath either: n_B(2 Delta) "
        "for aluminium is 8.5e-10 at 0.20 K and only clears the 5e-05 gate "
        "above about 0.42 K, and long before that the thermal partners this "
        "reduction drops dominate -- the guard in _build already refuses 0.20 K, "
        "where the thermal ratio is 3.7e-03 even over a window nine times "
        "shorter than the shipped one. Testing the factor needs a small-gap "
        "material or a different construction.\n"
        "\n"
        "4. THE PAIR-BREAKING HALF IS INERT BY PHYSICS, NOT BY CONSTRUCTION. "
        "N_abs = n_B(E_i + E_j) <= 3.2e-31 in this case, so the gain integral is "
        "switched on and contributes nothing. Its correctness is NOT evidenced "
        "here; a direct benchmark of the bath pair-breaking gain is a separate "
        "job.\n"
        "\n"
        "5. IT IS A REDUCTION, NOT THE MODEL. Scattering off, both photon drives "
        "off, Delta frozen, phonons pinned at the bath, dynes_gamma = 0, and a "
        "single cell so no transport operator is built at all. Nothing is said "
        "about how recombination composes with any of those, nor about a broad "
        "non-equilibrium f. Measured on this case: widening the seed to sigma = "
        "0.05 Delta (three cells) makes the one-cell law miss by 1.04, i.e. "
        "100%. The benchmark refuses such a run rather than scoring it.\n"
        "\n"
        "6. THE SINGLE-CELL SEED IS A MATHEMATICAL DEVICE. It is chosen "
        "precisely because the term moves no quasiparticle between bins. It is "
        "not a physical state and the closed form does not describe a realistic "
        "injected distribution.\n"
        "\n"
        "7. THE RESIDUAL IS THE ETD2 + BALANCE-PROJECTION DISCRETISATION OF THIS "
        "REDUCTION. The observed order is 3 rather than ETD2's nominal 2, which "
        "means the case is insensitive to the leading error term: it is a strong "
        "test of the RATE and a weak test of the integrator, and it says nothing "
        "about the stepper on stiff multi-term problems. The subcycling path is "
        "deliberately never entered.\n"
        "\n"
        "8. THE PHONON SIDE IS UNTOUCHED. The phonon-side recombination kernel "
        "(tau_0_pb_ns), the phonon equation itself and the Kaplan S_+ quadrature "
        "correction do not enter this run. Neither does the incommensurate "
        "omega-sublattice case: 2*E_face/dE = 120 here, an integer.\n"
        "\n"
        "9. THE ABSOLUTE NORMALISATION WAS ANCHORED ELSEWHERE, NOT HERE. The "
        "2026-08 audit checked the engine's assembled rate against an "
        "independently coded quadrature of the published Kaplan Eq. (8) "
        "integral (converging: 2.1e-02 at NE = 180 to 1.0e-03 at NE = 1440) and "
        "against the textbook low-T lifetime asymptotic (ratio 1.097 -> 1.047 "
        "over T = 0.30 -> 0.15 K, marching to 1 as an O(k_B T/Delta) expansion "
        "must). That is real external evidence and it is why the physics here is "
        "believed -- but it is NOT re-measured by this benchmark, and a green "
        "verdict below is not a restatement of it."
    ),
    activity=(
        "1. SWITCH TEST, same seed, same 2250 steps, same closed form. With "
        "collisions.recombination = true, x_qp falls 3.666667e-02 -> "
        "3.580097e-04, a factor 102.4, and the check passes at 6.0627e-06. With "
        "collisions.recombination = false (scattering is already off) the "
        "right-hand side is identically zero: x_qp ends at 3.666667e-02, a "
        "relative move of exactly 0.000e+00 -- bit-identical, not merely small "
        "-- and the check FAILS at 1.0142e+02, 2.0e+06 times the gate. The whole "
        "102x decay is attributable to the term under test and to nothing else.\n"
        "\n"
        "2. THE OBSERVABLE IS ONE THE TERM MOVES. x_qp is a pure number "
        "observable and recombination is the only number-changing channel "
        "present; scattering, which conserves number, is off. The excess also "
        "stays where it was put, so the decay is annihilation and not "
        "redistribution.\n"
        "\n"
        "3. NO INERT DEFAULT IS INVOLVED. Both photon drives are disabled, so "
        "photon_params and pb_params are None and neither n_bar default is ever "
        "reached; phonons.mode = 'thermal_bath' passes phonon_escape_time = None, "
        "so the tau_l = 0.0 sentinel is not used and the phonon-side kernel never "
        "enters; a 1x1 mask has no faces, so no transport operator is built. The "
        "comparison is emphatically not at a fixed point -- the thermal start "
        "where this term's two halves cancel exactly is the case that measures "
        "nothing, which is why this one is seeded.\n"
        "\n"
        "4. A CORRUPTED ENGINE IS CAUGHT, AND BY THE PREDICTED AMOUNT. Scaling "
        "build_recombination_kernel_base by 1.001 with the reference untouched "
        "gives 9.8941e-04 = 19.8x the gate, matching the closed form's own "
        "sensitivity to a 0.1% rate error (9.9008e-04) to three digits. Rewiring "
        "the coherence factor K+ -> K- (CoherenceAssignment.PHOTON, a mis-wire "
        "the code makes reachable) gives 6.4600e+01 = 1.3e+06 x the gate."
    ),
    extra={"case_overrides": CASE_OVERRIDES},
))
