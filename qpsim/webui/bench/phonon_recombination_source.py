"""Phonon recombination source: the pair line integral that feeds n_ph(ω).

THE REDUCTION. One cell, no transport, no drives, quasiparticle scattering
off, quasiparticle recombination on, and the phonon sector dynamic with no
substrate escape (τ_l → ∞). The quasiparticle occupation is prepared *flat and
absolute*, ``f(E) ≡ A``, and A is small enough that the quasiparticle side
cannot move it over the run (measured below). At frozen f the phonon equation
is diagonal in ω and affine — ``dn/dt = a_ph + b_ph·n`` — and the engine solves
it as the exact flow of that affine ODE, so the whole trajectory is closed form
and the only physics left in it is the pair line integral

    J(ω) = Σ_{i+j: E_i+E_j=ω} dE · ρ̄_i ρ̄_j K⁰ʳ_ij      (what the engine sums)

against which this benchmark scores an independently written continuum
quadrature of the same integral. That is a **T3** statement in ω: the density
of states, the coherence factor, the pair measure and the τ_0 prefactor are all
written here from the physics and none of them is read back from the engine.

WHY THE COMPARISON IS NOT CIRCULAR — the point the audit turned on. The 0-D and
1-D dynamic-phonon route (``compute_phonon_source_sink`` with an opt-in
``K_r0_phonon_side``) multiplies its pair sum by ``_pair_breaking_quadrature_
correction``, which is *defined* as exact/discrete with exact = the Kaplan
closed form. Wherever that correction is active the sum is forced onto the
closed form, ρ and K⁺ cancel identically between numerator and denominator, and
comparing the two measures the Fermi function and nothing else. The spatial
route does not take that path at all: ``SpatialCollisions.advance_phonons``
calls ``compute_phonon_source_sink`` with the quasiparticle-side kernel and no
``K_r0_phonon_side``, so **no correction is applied at any ω** and the raw
finite-volume quadrature is what is under test. Measured: deleting the
coherence factor entirely (K⁺ → 1) moves the scored residual to 2.7e-02, and a
flat 3× rescale of ρ moves it to 7.9e+00. Both are invisible to a
correction-active window.

TWO CONSEQUENCES OF TAKING THE SPATIAL ROUTE, STATED UP FRONT.

* The kernel is the quasiparticle-side ``K⁰ʳ = (1/τ_0)(E_i+E_j)²K⁺/(k_BT_c)³``,
  not the paper's phonon-side ``K⁺/(πΔτ_0^PB)``. The rate here is set by
  τ_0 = 438 ns and **not** by τ_0^PB = 0.255 ns; those differ by ~1718× and
  confusing them is a documented trap.

  ``phonons.use_phonon_side_kernel`` USED TO BE unread by ``SpatialCollisions``,
  and this case recorded it as ``false`` merely to describe the path it was
  forced onto. As of 2026-08-15 the spatial path reads the flag and DEFAULTS to
  the phonon-side kernel, so that ``false`` is now a deliberate SELECTION of
  the legacy quasiparticle-side path. The closed form below is written for that
  path and is scored against it, which is self-consistent -- but it means the
  DEFAULT path has no coverage for this term: run this case at the schema
  default and it fails at 7.5e-01 against a 3e-07 tolerance, because the
  closed form describes the other kernel. Re-deriving it onto ``K⁺/(πΔτ_0^PB)``
  (as ``phonon_scattering_source`` and ``phonon_escape`` were) is the open work.
* ``collisions.phonon_recombination_source`` — the switch this benchmark is
  named after — WAS inert in spatial_2d, because the collision layer was
  handed the quasiparticle-side ``enable_recombination`` in its place. That was
  fixed on 2026-08-14 (``7e15175``), and the phonon-side kernels added on
  2026-08-15 removed the last coupling to the quasiparticle flags: the named
  switch is now the gate, and flipping it off freezes ``snap_n_ph`` at its seed
  and drives the residual to 1.0. Any statement below that this switch is
  inert, or that ``collisions.recombination`` is what gates this term, is
  describing the pre-fix engine.

WHERE THE CLOSED FORM IS SCORED. Only where the pair interval is bounded away
from the BCS singularity: ω ≥ E_top + 2Δ, so the lower limit E = ω − E_top sits
at least a full gap above Δ. Below that the interval's lower endpoint is the
singular one and the discrete sum is genuinely wrong there — 2.2e-03 at
ω = E_top + Δ, and *flat under refinement* — which is exactly the artifact the
Kaplan endpoint correction exists to repair on the other route. It is excluded,
not hidden, and the excluded bin's non-convergence is reported in
``convergence``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.webui.benchmarks import Benchmark, Curve, register

# μeV per kelvin. Written out rather than imported so the analytic side owns
# every constant in its own prefactor.
_KB_UEV_PER_K = 86.17333262145

# How far above the top of the represented energy band the scored window
# starts, in gaps. The pair interval for ω is [max(Δ, ω−E_top), min(E_top,
# ω−Δ)]; below ω = E_top + Δ its lower endpoint is pinned at Δ, where ρ
# diverges and the product rule ρ̄_i ρ̄_j is not a second-order approximation to
# ρρ. Two gaps of clearance was chosen over one: at ω = E_top + 1.05Δ the
# residual is already inside the tolerance (2.3e-07) but with only 1.3× margin
# and a measured convergence order of 1.88, while from E_top + 2Δ it is
# 5.7e-08 at order 2.00. The audit of the original derivation independently
# recommended the same cut (ω/Δ ≥ 12 on this grid).
_WINDOW_MARGIN_GAPS = 2.0

# Gauss-Legendre nodes for the pair line integral. Both endpoints are interior
# inside the scored window, so the integrand is analytic there and the rule
# converges geometrically: measured against a 512-node reference, 16 nodes give
# 2.1e-11 and 24 already give 8.9e-16. 64 is far more than needed and costs
# nothing (320 bins × 64 nodes, vectorised), and the headroom means the
# analytic side stays machine-exact if the window is ever widened.
_GL_NODES = 64
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_NODES)

# The exact overrides that produce the case this benchmark is measured on.
# Runs in 0.9 s. Every entry is load-bearing except where noted:
#   grid.num_bins  MUST keep 2·E_face/dE an integer or the pair-sum and
#                  pair-difference ω lattices are disjoint (phonon.py:257).
#                  360 bins over [1, 10]Δ give dE = 4.5 μeV and 2Δ/dE = 80;
#                  the shipped default 400 gives 88.889 and is NOT usable.
#   initial        kind="absolute" REPLACES the thermal seed, so f is exactly
#                  the constant A with no Fermi background to carry through
#                  the convolution. amplitude=1e-10 is two decades inside the
#                  frozen-f regime (see `caveat` item 4).
#   stop_tol = 0   f is stationary to 4e-10, so the run's own residual
#                  certificate (2e-20) is under any positive stop_tol and the
#                  run would "converge" after one step and record no
#                  trajectory at all.
#   phonons.tau_l_ns is NOT set: dynamic_closed maps unconditionally to the
#                  engine's tau_l = 0.0 (τ_l → ∞) sentinel and any value here
#                  would be dead weight.
#   material.tau_0_pb_ns is NOT set for the same reason — this path runs on
#                  tau_0.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material": {
        "name": "Al",
        "Delta_0": 180.0,
        "T_c": 1.18,
        "tau_0": 438.0,
        "D_0": 0.0,
        "rho_F": 1.74e28,
        "dynes_gamma": 0.0,
    },
    "T_bath": 0.1,
    "grid": {"min_factor": 1.0, "max_factor": 10.0, "num_bins": 360},
    "geometry": {"kind": "rectangle", "rows": 1, "cols": 1, "mesh_size_um": 1.0},
    "diffusion_model": "A1",
    "collisions": {
        "scattering": False,
        "recombination": True,
        "phonon_scattering_source": False,
        "phonon_recombination_source": True,
    },
    "injection": {"enabled": False},
    "initial": {
        "kind": "absolute",
        "amplitude": 1e-10,
        "energy": {"kind": "flat"},
        "space": {"kind": "uniform"},
    },
    "drives": [],
    "phonons": {"mode": "dynamic_closed", "use_phonon_side_kernel": False},
    "self_consistent_gap": False,
    "dt": 0.025,
    "max_time": 0.5,
    "stop_tol": 0.0,
    "snapshot_interval": 0.025,
}


def _pair_line_integral(
    omega: np.ndarray, gap: float, e_lo: np.ndarray, e_hi: np.ndarray
) -> np.ndarray:
    r"""``∫ ρ(E) ρ(ω−E) K⁺(E, ω−E) dE`` over the represented pair interval.

    ``ρ(E) = E/√(E²−Δ²)`` and ``K⁺ = 1 + Δ²/(E E′)``, so the integrand is
    ``[E(ω−E) + Δ²] / (√(E²−Δ²) √((ω−E)²−Δ²))`` — the Kaplan pair integrand.
    Over the full interval ``[Δ, ω−Δ]`` this is ``Δ·S₊(ω/Δ)`` with
    ``S₊(x) = x·E(1−4/x²)``; the caller's limits are the *truncated* ones the
    finite energy grid actually represents, for which there is no elementary
    form, which is why this is a quadrature rather than an elliptic call.

    Callers must supply endpoints bounded away from ``Δ``: the ``√`` factors
    are handled to no particular accuracy at the singularity itself.
    """
    mid = 0.5 * (e_hi + e_lo)
    half = 0.5 * (e_hi - e_lo)
    e = mid[:, None] + half[:, None] * _GL_X[None, :]
    e_partner = omega[:, None] - e
    # Factored (E−Δ)(E+Δ) rather than E²−Δ²: the same cancellation guard the
    # engine's own BCS quadrature uses.
    integrand = (e * e_partner + gap * gap) / (
        np.sqrt((e - gap) * (e + gap))
        * np.sqrt((e_partner - gap) * (e_partner + gap))
    )
    return half * (integrand @ _GL_W)


def _bose(omega: np.ndarray, temperature: float) -> np.ndarray:
    """``n_B(ω, T)``, returning 0 where the occupation underflows binary64."""
    ratio = omega / (_KB_UEV_PER_K * temperature)
    safe = np.minimum(ratio, 700.0)
    return np.where(ratio > 700.0, 0.0, 1.0 / np.expm1(safe))


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    if "snap_n_ph" not in arrays:
        raise ValueError(
            "no snap_n_ph in the payload: the phonon population is only "
            "recorded when phonons.mode is a dynamic sector and "
            "snapshot_interval is set. This benchmark needs "
            "phonons.mode='dynamic_closed'."
        )
    n_ph = np.asarray(arrays["snap_n_ph"], dtype=float)
    t_ns = np.asarray(arrays["snap_t_ns"], dtype=float)
    if n_ph.shape[2] != 1:
        raise ValueError(
            f"{n_ph.shape[2]} cells: this reduction is the 0-D one (a 1x1 "
            "mask with D_0 = 0). With more cells the closed form would have "
            "to state what transport does to f, and it does not."
        )

    energies = np.asarray(arrays["E_bins"], dtype=float)
    spacing = np.diff(energies)
    d_e = float(spacing[0])
    if not np.allclose(spacing, d_e, rtol=1e-12, atol=0.0):
        raise ValueError(
            "the energy grid is not uniform; the pair sum's dE measure and "
            "this quadrature's interval endpoints both assume it is."
        )
    gaps = np.asarray(arrays["gap_per_cell"], dtype=float)
    gap = float(gaps[0])
    if not np.all(gaps == gap):
        raise ValueError("the local gap is not uniform across the mask.")
    # The top cell EDGE, not the top cell centre. The pair sum over an ω bin is
    # a midpoint rule whose outermost cells reach half a spacing past their
    # centres, and near the top of the lattice the whole interval is only a few
    # spacings wide, so the half-cell is not a refinement — it is the interval.
    # Measured: E[-1] + dE/2 gives 5.7e-08, E[-1] gives 9.97e-01 and
    # E[-1] + 3dE/2 gives 6.66e-01. This benchmark pins the convention.
    e_top = float(energies[-1] + 0.5 * d_e)

    # f is read from the run's own first frame rather than rebuilt from the
    # setup: the premise of the reduction is "f is this constant and stays
    # there", and the frame is the record of what was actually prepared.
    f_snapshots = np.asarray(arrays["snap_f"], dtype=float)
    f_initial = f_snapshots[0][:, 0]
    amplitude = float(f_initial[0])
    if amplitude <= 0.0 or not np.all(f_initial == amplitude):
        raise ValueError(
            "the run did not start from a flat, non-zero occupation. This "
            "reduction needs initial.kind='absolute' with a flat energy "
            "profile and a uniform spatial profile, so that f(E) ≡ A and the "
            "pair convolution factorises into A² times the line integral."
        )
    # Not a precondition — a measured quantity the tolerance has to cover. The
    # quasiparticle side is still on (it is what gates the phonon source at
    # all), so f decays; the case keeps A small enough that it does not move.
    f_drift = float(np.max(np.abs(f_snapshots[-1][:, 0] - f_initial)) / amplitude)

    # The ω lattice is not published for this mode, so it is rebuilt from the
    # run's own energy grid with the engine's own map. That is reading the
    # abscissa, not the answer: row k of snap_n_ph HAS no meaning except "the
    # frequency this map put there", and there is no independent notion of it
    # to substitute. Everything scored against it below is written here.
    omega = build_phonon_frequency_map(energies)[0]
    if omega.size != n_ph.shape[1]:
        raise ValueError(
            f"the rebuilt ω lattice has {omega.size} bins but snap_n_ph has "
            f"{n_ph.shape[1]}; the two do not describe the same run."
        )

    e_lo = np.maximum(gap, omega - e_top)
    e_hi = np.minimum(e_top, omega - gap)
    window = (omega >= e_top + _WINDOW_MARGIN_GAPS * gap) & (e_hi > e_lo)
    if not np.any(window):
        raise ValueError(
            f"no ω bin satisfies ω ≥ E_top + {_WINDOW_MARGIN_GAPS:g}Δ = "
            f"{e_top + _WINDOW_MARGIN_GAPS * gap:g} μeV with a non-empty pair "
            "interval. The energy band must reach past ~2Δ for a truncated "
            "pair interval to exist at all."
        )
    w = omega[window]

    tau_0 = float(setup.material.tau_0)
    k_b_t_c = _KB_UEV_PER_K * float(setup.material.T_c)
    # K⁰ʳ = (1/τ_0)·((E_i+E_j)/k_BT_c)²/k_BT_c · K⁺, and E_i + E_j is exactly ω
    # for every pair the bin collects, so the whole prefactor comes out of the
    # sum and only the coherence/DOS part is integrated.
    rate = (
        (w / k_b_t_c) ** 2
        / (tau_0 * k_b_t_c)
        * _pair_line_integral(w, gap, e_lo[window], e_hi[window])
    )

    # dn/dt = a_ph + b_ph·n, with a_ph = A²J (emission) and
    # b_ph = (A² − (1−A)²)J = −(1−2A)J (emission minus pair-breaking
    # absorption). Only b_ph and the fixed point n* = −a_ph/b_ph are needed,
    # and in n* the line integral cancels outright — which is exactly why the
    # plateau carries no kernel information and why dividing both curves by it
    # launders nothing (caveat item 1).
    b_ph = -(1.0 - 2.0 * amplitude) * rate
    n_star = amplitude * amplitude / (1.0 - 2.0 * amplitude)

    # Frame 0 is the seed, not a measurement, and it is ~89 decades below n*,
    # so a relative comparison there reports the exponent of the Bose function
    # rather than anything the term did. Dropped, and the seed enters the
    # closed form below instead, where it belongs.
    times = t_ns[1:]
    seed = _bose(w, float(setup.T_bath))
    y_analytic = 1.0 + (seed[None, :] / n_star - 1.0) * np.exp(
        b_ph[None, :] * times[:, None]
    )
    y_sim = n_ph[1:, window, 0] / n_star

    return Curve(
        x=w / gap,
        y_sim=y_sim,
        y_analytic=y_analytic,
        x_label="ω / Δ  (phonon frequency, in units of the gap)",
        y_label="n_ph(ω, t) / n*   (approach to the phonon fixed point)",
        series_labels=tuple(f"t = {value:.4g} ns" for value in times),
        metric="pointwise",
        note=(
            f"Scored where the pair interval clears the BCS singularity: "
            f"ω/Δ ∈ [{w[0] / gap:.4g}, {w[-1] / gap:.4g}], "
            f"{w.size} bins × {times.size} frames. "
            f"1/|b_ph| runs {1.0 / np.max(-b_ph):.4g}–{1.0 / np.min(-b_ph):.4g} ns, "
            f"so the frames span {times[0] * np.min(-b_ph):.3g} to "
            f"{times[-1] * np.max(-b_ph):.3g} relaxation times. "
            f"Both curves are divided by n* = A²/(1−2A) = {n_star:.4g}, which "
            "is independent of the kernel and therefore launders nothing: it "
            "removes the one tautological factor and leaves the rate. "
            f"Measured quasiparticle drift over the run: {f_drift:.3e} "
            "relative — the closed form assumes f is frozen and this is by "
            "how much it is not."
        ),
    )


register(Benchmark(
    name="phonon-recombination-source",
    title="Phonon recombination source: the truncated Kaplan pair integral",
    # T3 in ω, which is where all the kernel content is: the scored quantity is
    # the engine's raw pair sum against an independently written continuum
    # quadrature of the same integral, with no endpoint correction on either
    # side. T1 in t: n(t) = n* + (n_0 − n*)e^{b t} is the exact flow of the
    # affine ODE, written from the physics, and it is the statement that the
    # engine's phonon half-step is that exact flow rather than an approximation
    # to it (measured dt-independent to 2% over a 10× change in dt). The
    # registry takes one label; the weaker-sounding one is the honest one,
    # because a reader who takes "T1" from this would take it to mean the
    # frequency dependence is a closed form, and it is not.
    tier="T3",
    formula_latex=(
        r"\begin{aligned}"
        r"&\textbf{represented pair interval:}\quad"
        r"E_-(\omega)=\max(\Delta,\ \omega-E_{\rm top}),\qquad"
        r"E_+(\omega)=\min(E_{\rm top},\ \omega-\Delta)\\[4pt]"
        r"&\textbf{pair line integral:}\quad"
        r"J(\omega)=\frac{1}{\tau_0}\,\frac{\omega^{2}}{(k_BT_c)^{3}}"
        r"\int_{E_-}^{E_+}\!\!dE\;"
        r"\frac{E(\omega-E)+\Delta^{2}}"
        r"{\sqrt{E^{2}-\Delta^{2}}\;\sqrt{(\omega-E)^{2}-\Delta^{2}}}\\[4pt]"
        r"&\qquad\text{(for the untruncated interval this is }"
        r"\Delta\,S_+(\omega/\Delta),\ S_+(x)=x\,E(1-4/x^{2})\text{)}\\[6pt]"
        r"&\textbf{affine phonon equation at frozen }f\equiv A:\quad"
        r"\dot n_{\rm ph}=a_{\rm ph}+b_{\rm ph}n_{\rm ph},\quad"
        r"a_{\rm ph}=A^{2}J,\quad b_{\rm ph}=-(1-2A)\,J\\[6pt]"
        r"&\textbf{trajectory from the Bose seed:}\quad"
        r"\frac{n_{\rm ph}(\omega,t)}{n^{*}}"
        r"=1-\left(1-\frac{n_B(\omega,T_{\rm bath})}{n^{*}}\right)"
        r"e^{-(1-2A)J(\omega)\,t},\qquad"
        r"n^{*}=-\frac{a_{\rm ph}}{b_{\rm ph}}=\frac{A^{2}}{1-2A}"
        r"\end{aligned}"
    ),
    reason=(
        "With f held at a constant A the pair convolution factorises: every "
        "emission weight is A² and every pair-breaking weight is (1−A)², so "
        "a_ph and b_ph are the same line integral J(ω) times two known "
        "numbers, and the phonon equation becomes an affine ODE with constant "
        "coefficients whose solution is an exponential approach to −a/b. J(ω) "
        "is the Kaplan pair integral truncated to the energy band the grid "
        "represents, evaluated here by an independent Gauss-Legendre "
        "quadrature; the engine evaluates it as a midpoint sum over the pair "
        "lattice and, on the spatial route, applies no endpoint correction to "
        "it, so the two are genuinely different computations of the same "
        "number."
    ),
    # The audit of the original derivation recommended 3e-7 for a
    # correction-free tail curve. Kept, and it is now comfortable rather than
    # tight: the shipped case measures 5.69e-08, a 5.3x margin, and unlike the
    # limit-bounded residual the original spec proposed, this one is pure
    # discretisation and shrinks as dE² under refinement. The tolerance is
    # therefore an upper bound that only gets easier, not one that erodes.
    headline_latex=(
        r"a_{\rm rec}(\omega) \;=\; \frac{S_+(\omega/\Delta)}{\pi\,\tau_0^{\rm "
        r"PB}}\,e^{-\omega/k_BT}, \qquad S_+(x) = x\,E\!\left(1-4/x^2\right)"
    ),
    rel_tol=3e-7,
    convergence=(
        "ENERGY GRID, at the shipped dt = 0.025 ns and window ω ≥ E_top + 2Δ. "
        "num_bins kept a multiple of 9 on [1, 10]Δ so the ω lattice stays "
        "commensurate. Columns: max and rms pointwise relative residual over "
        "the whole scored curve.\n"
        "  NE     dE(μeV)   max          rms          ratio  order   runtime\n"
        "   180    9.0000   2.279e-07    3.106e-08      -      -      0.2 s\n"
        "   360    4.5000   5.692e-08    7.622e-09    4.005  2.002    0.9 s\n"
        "   720    2.2500   1.416e-08    1.822e-09    4.019  2.007    4.7 s\n"
        "  1440    1.1250   3.474e-09    4.855e-10    4.077  2.028   20.0 s\n"
        "Clean second order over 8× refinement, which is what a midpoint rule "
        "on a smooth integrand must give and is the evidence that the "
        "residual IS the quadrature error and not something else. NE = 360 is "
        "shipped: 5.69e-08 against rel_tol 3e-07 is 5.3× margin at 0.9 s.\n\n"
        "TIME STEP (NE = 360, dt chosen to keep the snapshot times identical): "
        "dt = 0.025 → 5.692e-08, dt = 0.005 → 5.696e-08, dt = 0.0025 → "
        "5.576e-08. A 2% spread over a 10× change, and the residual to which "
        "it converges is unchanged. The phonon half-step is the exact flow of "
        "the affine ODE (expm1-based φ₁), so there is no time-integration "
        "error to refine away; the 2% that does move is the quasiparticle "
        "drift below, which is dt-dependent at the 1e-9 level. A residual "
        "that moved with dt would mean the phonon update is not exact.\n\n"
        "WINDOW EDGE (NE = 360 / 720), max residual as the lower cut moves. "
        "The cut is on ω − E_top, the distance of the pair interval's lower "
        "endpoint above Δ:\n"
        "  ω/Δ ≥ 11.00 (endpoint AT Δ)  2.213e-03 / 2.212e-03  — does not "
        "converge; this is the singular-endpoint artifact, excluded\n"
        "  ω/Δ ≥ 11.05                  2.309e-07 / 6.289e-08  order 1.88\n"
        "  ω/Δ ≥ 11.50                  8.866e-08 / 2.210e-08  order 2.00\n"
        "  ω/Δ ≥ 12.00 (shipped, = E_top + 2Δ)  5.692e-08 / 1.416e-08  order "
        "2.00\n"
        "  ω/Δ ≥ 13.00                  3.230e-08 / —          \n"
        "The excluded bin at ω = E_top + Δ is flat under refinement to four "
        "digits, so it is a modelling error of the discretisation and not "
        "something a finer grid repairs — the same artifact the Kaplan "
        "endpoint correction was written for on the 0-D route.\n\n"
        "AMPLITUDE (the frozen-f condition, NE = 360): A = 1e-10 → 5.692e-08, "
        "1e-8 → 5.824e-08, 1e-6 → 6.192e-06, 1e-4 → 6.194e-04. Above ~1e-7 "
        "the residual is proportional to A and is quasiparticle drift, not "
        "quadrature. The shipped 1e-10 sits two decades inside the "
        "quadrature-limited regime.\n\n"
        "THE ANALYTIC SIDE'S OWN ERROR, so the residual can be attributed. "
        "(a) The integrand written here, integrated over the UNTRUNCATED "
        "interval by scipy's adaptive rule under a sin² endpoint "
        "substitution, matches Kaplan's x·E(1−4/x²) to 3.2e-14 over "
        "x ∈ [2.5, 40] — the integrand is the pair integrand and not "
        "something that merely resembles it. (b) The fixed 64-node "
        "Gauss-Legendre rule used in the scored window matches "
        "scipy.integrate.quad on the same 320 intervals to 4.0e-15. The "
        "analytic side therefore contributes nothing at this tolerance and "
        "the whole 5.69e-08 is the engine's discretisation.\n\n"
        "BATH TEMPERATURE: 0.10 K → 5.692e-08, 0.12 K → 5.692e-08, 0.15 K → "
        "5.692e-08, 0.20 K → 1.864e-07, 0.30 K → 1.236e-02. The mechanism is "
        "the quasiparticle-side pair-breaking gain, which scales as "
        "e^{−2Δ/k_BT} and repopulates f; it is not the phonon integral. Flat "
        "to 0.15 K, so 0.10 K is not a cliff edge."
    ),
    modes=("spatial_2d",),
    build=_build,
    caveat=(
        "WHAT THIS DOES NOT PROVE.\n\n"
        "1. THE FIXED POINT IS FREE. n* = A²/(1−2A) is the ratio of two sums "
        "that share the same kernel, so it is identical for any ρ, any K and "
        "any grid — it is the occupancy algebra and nothing else. The late "
        "end of every series is therefore a tautology. All the information is "
        "in the rate b_ph, which is why both curves are normalised by n* and "
        "why the frames are placed from 0.003 to 8 relaxation times rather "
        "than at the plateau. A reader should treat a green verdict here as a "
        "statement about J(ω), not about the phonon fixed point.\n\n"
        "2. THE NAMED SWITCH GATES THIS TERM (corrected 2026-08-15). It used "
        "to be inert on the spatial path, because SpatialCollisions received "
        "the quasiparticle-side enable_recombination in its place, and this "
        "caveat said so on the strength of a measurement that was true then. "
        "Both halves of that coupling are now fixed (7e15175, then the "
        "phonon-side kernels), so setting "
        "collisions.phonon_recombination_source = false freezes snap_n_ph at "
        "its seed and drives the residual to 1.0. The claim that "
        "collisions.recombination is what gates this term is likewise "
        "obsolete.\n\n"
        "3. NOT THE PAPER'S PHONON-SIDE KERNEL — BY THIS CASE'S OWN CHOICE. "
        "The spatial route now DEFAULTS to the phonon-side K⁺/(πΔτ_0^PB), and "
        "this case pins phonons.use_phonon_side_kernel = false to select the "
        "legacy quasiparticle-side K⁰ʳ with the (E_i+E_j)²/(τ_0 (k_BT_c)³) "
        "prefactor, which is what the closed form below is written for. That "
        "pairing is self-consistent, and it is also the limit of what this "
        "verdict certifies: the DEFAULT path this term takes in every other "
        "run is not covered here. Scored at the schema default the case fails "
        "at 7.5e-01 against a 3e-07 tolerance — the closed form describing the "
        "other kernel, not the engine being wrong. τ_0^PB, the F&C Eq. 12 "
        "normalisation and _pair_breaking_quadrature_correction are therefore "
        "still untouched by this benchmark, which is why the Kaplan S₊ item in "
        "docs/HELD-BACK-ADJUDICATION-2026-08-11.md can move the 0-D numbers "
        "without moving anything here.\n\n"
        "4. f IS FROZEN BY CONSTRUCTION, NOT BY PHYSICS. The quasiparticle "
        "side is on (it is what enables the term); the case simply chooses an "
        "amplitude at which it cannot move f over 0.5 ns. The run reports the "
        "measured drift in the curve note (4.1e-10 relative at the shipped "
        "case) and the amplitude table above shows where the assumption "
        "fails. Nothing here validates the quasiparticle collision integral.\n\n"
        "5. THE AMPLITUDES ARE MATHEMATICAL, NOT PHYSICAL. n* = 1e-20 and the "
        "seed runs down to 6e-182. All comfortably representable, and every "
        "comparison is relative, but this is a probe of an integral in a deep "
        "freeze, not a resolvable experimental regime.\n\n"
        "6. SCOPE. Ideal BCS only (dynes_gamma = 0; the coherence factor's "
        "cell-averaged form and the ρ used here both assume it). Uniform "
        "energy grid only. Commensurate ω lattice only — the incommensurate "
        "sublattice pathology at phonon.py:257 is avoided here, not tested, "
        "and the shipped default num_bins = 400 is incommensurate. One cell, "
        "no transport, no self-consistent gap.\n\n"
        "7. NOT TESTED: the back-coupling of n_ph into the quasiparticle "
        "equation, finite phonon escape τ_l, the scattering source, "
        "phonon_source_sink_jacobian_f, the coupled-Newton assembly, the "
        "sub-gap cut-cell ω remap, and energy conservation between the two "
        "populations — this reduction books the pair channel on the phonon "
        "side only and therefore violates it deliberately."
    ),
    activity=(
        "1. TERM OFF ⇒ NOTHING MOVES, AND THE CHECK FAILS. With "
        "collisions.recombination = false the phonon population is bitwise "
        "unchanged from its seed at every frame (max|n(t) − n(0)| = 0.0 "
        "exactly), so in the scored window n stays at ~1.4e-109 against an "
        "analytic plateau of 1e-20 and the residual is 1.000e+00 — 3.3e6 × "
        "the tolerance. With the term on, a_ph runs 1.3e-21 to 1.6e-19 /ns "
        "and b_ph runs −1.593e+01 to −1.289e-01 /ns.\n\n"
        "2. THE CURVE CROSSES ITS WHOLE RANGE. From the Bose seed the "
        "population reaches 32.9% of n* by the first scored frame (t = 0.025 "
        "ns) at the fast end and 99.97% by t = 0.5 ns, while the slowest bin "
        "is still at 6.2%; 1/|b_ph| runs 0.063 to 7.76 ns, so the ensemble "
        "spans 0.003 to 8.0 relaxation times. The exponential is pinned at "
        "both ends, not read at its plateau.\n\n"
        "3. THE INTEGRAND IS UNDER TEST — the point the audit made about the "
        "original derivation, checked by sabotage rather than argued. "
        "Perturbing the engine's kernel and re-scoring the same window: "
        "ρ → 3.0ρ (flat rescale) 7.885e+00; ρ → (1 + 0.30 sin(E/Δ))ρ "
        "2.941e-01; K⁺ → 1 (coherence factor deleted) 2.673e-02; "
        "K⁺ → 1 + 0.5Δ²/(EE′) 1.332e-02; K⁺ → K⁻ (photon convention) "
        "5.380e-02. Every one is 4–7 decades above the tolerance. The audit "
        "measured all six of the equivalent perturbations as UNCAUGHT on the "
        "correction-active window the original spec scored.\n\n"
        "4. CONSTANT SENSITIVITY, MEASURED. Perturbing one constant on the "
        "analytic side: τ_0 × 1.000001 → 1.004e-06 (3.3× the tolerance, so "
        "the benchmark bounds τ_0 inside this term to ~3e-7 relative); "
        "τ_0 × 1.001 → 9.984e-04; Δ × 1.001 → 1.447e-04.\n\n"
        "5. THE 2Δ THRESHOLD IS A SEPARATE FACT, not a curve point: every ω "
        "bin below 2Δ carries a_ph exactly 0.0 (checked at the shipped case, "
        "0 of the sub-threshold bins non-zero). Those bins exist and are "
        "populated by the scattering source in the full model.\n\n"
        "6. THE RUN'S OWN PREMISE IS MEASURED, NOT ASSUMED. f is flat to "
        "bitwise equality at frame 0 (the build refuses otherwise) and drifts "
        "by 4.1e-10 relative over the run. The amplitude table in "
        "`convergence` gives the slope from drift to residual as ≈1.5, so "
        "that drift contributes ≈6e-10 — 90× below the quadrature error that "
        "dominates, and 500× below the tolerance. The drift is recomputed and "
        "reported in the curve note on every evaluation rather than quoted "
        "from here."
    ),
))
