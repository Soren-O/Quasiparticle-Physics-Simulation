r"""Sub-gap photon drive — the two-rung ladder is a Riccati equation.

The sub-gap term (:mod:`qpsim.collisions.sub_gap_photon`) is a pure jump
operator in energy: a quasiparticle at bin ``i`` can only move to ``i ± m``,
where ``m = round(ω₀/δE)`` is the grid-commensurate photon step. Nothing is
created or destroyed. A general ``m`` gives a ladder ``i, i±m, i±2m, …`` of
``⌈N_E/m⌉`` rungs coupled through Pauli factors, which has no closed form.

Choosing ``N_E/2 ≤ m < N_E`` closes it. Then no bin can have both an up and a
down partner, the operator decomposes into independent PAIRS ``(a, a+m)`` plus
partnerless bins, and each pair is a scalar Riccati equation whose solution is
elementary. The reduction is exact rather than asymptotic: the Pauli blocking
factors are carried in full, so the closed form holds for arbitrary occupation
and at every grid spacing — the energy grid is not a quadrature here, it is the
lattice the jump operator lives on. What is left in the residual is the ETD2
time step alone, and the measured order is 2.000 (see ``convergence``).

Two curves ship on the same abscissa:

``⟨E⟩(t)/Δ``
    The number-weighted mean quasiparticle energy. This is the observable the
    drive moves: absorption (rate ∝ n̄) pushes population up the ladder,
    stimulated + spontaneous emission (∝ n̄+1) pushes it back, and the balance
    settles on the photon detailed-balance plateau. It swings +21.6% here.
``N(t)/N(0)``
    Total quasiparticle number, identically 1. This is a conservation
    certificate, **not** an activity certificate: it is equally flat with the
    drive switched off. It is not a tautology either — see ``caveat``.

Authored against the AUDITED position, not the original derivation. The
adversarial audit found five defects, all wording or scope rather than numbers,
and each is corrected where it lands: the partnerless bins are not "exactly
frozen" at run level (``caveat`` 3, and the comment in :func:`_build`); the
``N(t)`` curve is not an activity certificate (``caveat`` 4, and the curve's own
note); "a genuine sum of seven distinct exponentials" was oversold (``activity``
5); the kernel's ``supported[j_up]`` / ``supported[j_down]`` asymmetry is a
second untested branch (``caveat`` 2, and the ``min_factor`` guard that keeps
this case out of it); and every number here was re-measured on a clean tree at
the commit recorded in ``extra``, the audit's own having been taken on a dirty
one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# Boltzmann's constant, written out rather than imported from qpsim.physics.
# The analytic side of a T1 must not share a constant table with the engine it
# is checking; a shifted k_B would then cancel out of the comparison.
_KB_UEV_PER_K = 86.17333262145

# The case this closed form applies to. Every other term is off, so 100% of the
# motion in the run is the sub-gap drive (see ``activity``).
#
# The three numbers that are not free:
#   omega_0 = 202.5 = 9*dE  with dE = (3-1)*180/16 = 22.5 μeV. Grid-commensurate
#     (so the kernel's snap warning never fires and n̄ is unambiguous) and
#     N_E/2 = 8 ≤ 9 < 16, which is what buys the two-rung ladder. It is a legal
#     sub-gap drive: 202.5 < 2Δ = 360.
#   dt = 0.015625 and snapshot_interval = 0.25 are exact binary fractions with
#     an exact integer ratio 16, so every snapshot lands on an integration
#     endpoint and no dense-output interpolation enters the comparison.
#   T_bath = 0.5 K is warm for Al, and deliberately so. At the UI default 0.1 K
#     the thermal occupations run down to 2e-27 and ETD2's weighted-balance
#     projection (which restores missing mass as f ← 1 − (1−f)·h) loses every
#     significant digit of them; the residual then GROWS as dt shrinks. That is
#     an integrator property, not a sub-gap-kernel error, and it is reported
#     rather than hidden — see ``caveat``. At 0.5 K the smallest occupation on
#     this grid is 4.7e-06 and the run converges cleanly at order 2.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "transient_0d",
    "material.Delta_0": 180,
    "T_bath": 0.5,
    "grid.min_factor": 1,
    "grid.max_factor": 3,
    "grid.num_bins": 16,
    "phonons.mode": "thermal_bath",
    "collisions.scattering": False,
    "collisions.recombination": False,
    "collisions.phonon_scattering_source": True,
    "collisions.phonon_recombination_source": True,
    "subgap_drive.enabled": True,
    "subgap_drive.omega_0": 202.5,
    "subgap_drive.n_bar": 3,
    "subgap_drive.c_phot": 0.01,
    "pb_drive.enabled": False,
    "dt": 0.015625,
    "total_time": 50,
    "snapshot_interval": 0.25,
    "stop_tol": None,
    "probe.enabled": False,
}


def _bcs_cell_measure(
    gap: float, e_min: float, dE: float, ne: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Finite-volume BCS measure on a uniform grid, in closed form.

    From the exact antiderivatives, over a cell ``[E⁻, E⁺]``::

        ∫ E/√(E²−Δ²) dE = √(E²−Δ²)                    → capacity   w_i
        ∫ Δ/√(E²−Δ²) dE = Δ·arccosh(E/Δ)              → anomalous  a_i

    so ``ρ̄_i = w_i/δE`` is the cell-average DOS and ``r_i = a_i/w_i`` the cell
    average of ``Δ/E`` under that same measure. The photon coherence factor is
    ``K⁺ = 1 + r_a r_b`` — the case-II factor, reversed from the QP-phonon case.

    Written here from the physics, NOT read from ``SpectralContext``. It agrees
    with the engine's arrays bit-for-bit, which is the signature of a T2 in
    disguise and was attacked as such; see ``caveat`` item 5 for why it is not.
    """
    edges = e_min + np.arange(ne + 1, dtype=float) * dE
    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], lo)
    w = np.sqrt(np.maximum(hi**2 - gap**2, 0.0)) - np.sqrt(
        np.maximum(lo**2 - gap**2, 0.0)
    )
    anomalous = gap * (
        np.arccosh(np.maximum(hi / gap, 1.0)) - np.arccosh(np.maximum(lo / gap, 1.0))
    )
    r = np.zeros(ne)
    supported = w > 0.0
    # Clipped because r_i is a cell average of Δ/E ≤ 1; roundoff on a cell whose
    # lower edge sits exactly at Δ can push the ratio a hair past 1.
    r[supported] = np.clip(anomalous[supported] / w[supported], 0.0, 1.0)
    return w, w / dE, 1.0 + r[:, None] * r[None, :]


def _ladder_pairs(ne: int, m: int) -> tuple[list[tuple[int, int]], list[int]]:
    """``(pairs, partnerless)`` for the two-rung ladder; raises if not two-rung.

    Structural, not numerical: this asserts the decomposition the closed form
    assumes, rather than trusting the ``N_E/2 ≤ m < N_E`` arithmetic. A ladder
    of three or more rungs has no elementary solution, so a case that fails
    here must be refused loudly instead of silently compared against a formula
    that does not describe it.
    """
    up = {i: i + m for i in range(ne) if i + m < ne}
    down = {i: i - m for i in range(ne) if i - m >= 0}
    pairs: list[tuple[int, int]] = []
    partnerless: list[int] = []
    for i in range(ne):
        if i in up and i in down:
            raise ValueError(
                f"bin {i} has partners both up and down (N_E={ne}, m={m}): the "
                "ladder is longer than two rungs and has no closed form. This "
                "benchmark needs N_E/2 <= m < N_E."
            )
        if i in up:
            j = up[i]
            if j in up:
                raise ValueError(f"bin {j} continues the ladder upward: not two-rung.")
            pairs.append((i, j))
        elif i not in down:
            partnerless.append(i)
    if not pairs:
        # The recurring failure mode in this codebase is a knob that looks set
        # and does nothing. A photon step past the top of the grid leaves every
        # bin partnerless: the kernel runs, costs time, and moves nothing.
        raise ValueError(
            f"omega_0 is {m} bins on a grid of {ne}, so every partner lands off "
            "the grid and the drive is inert -- the run would look driven and be "
            "undriven. There is nothing for this benchmark to check."
        )
    return pairs, partnerless


def _pair_trajectory(
    f_a0: float, f_b0: float, A: float, B: float, n_bar: float, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Exact ``(f_a(t), f_b(t))`` for one two-rung ladder.

    With ``J = n̄ f_a(1−f_b) − (n̄+1) f_b(1−f_a)`` the kernel gives
    ``ḟ_a = −A·J`` and ``ḟ_b = +B·J``, so ``C = B f_a + A f_b`` is conserved
    for arbitrary occupation (the Pauli factors enter symmetrically). That
    conserved quantity IS the ladder's quasiparticle number, because
    ``B f_a + A f_b = (c_ph K⁺/δE)(w_a f_a + w_b f_b)``.

    Eliminating ``f_a`` leaves the scalar Riccati equation
    ``u̇ = −A u² + P u + Q`` with ``P = C − A n̄ − B(n̄+1)``, ``Q = C n̄``, whose
    exact solution is a Möbius transform of a single decaying exponential.
    """
    C = B * f_a0 + A * f_b0
    if C <= 0.0:
        return np.zeros_like(t), np.zeros_like(t)
    P = C - A * n_bar - B * (n_bar + 1.0)
    discriminant = np.sqrt(P * P + 4.0 * A * C * n_bar)
    # P < 0 and Q >= 0 here, so the small positive root of A u² − P u − Q must
    # be formed as 2Q/(−P+√D). The textbook (P+√D)/(2A) cancels catastrophically
    # and loses ~8 digits at the n̄ and c_phot of this case.
    denominator = -P + discriminant
    u1 = 2.0 * (C * n_bar) / denominator  # positive fixed point f_b(∞)
    u2 = -denominator / (2.0 * A)  # negative root
    k = A * (u1 - u2)  # relaxation rate (1/ns), == √D
    # R(t) is an exact invariant: log-linear in t even with the Pauli
    # nonlinearity active. u2 < 0 <= f_b0, so the ratio never divides by zero.
    R = ((f_b0 - u1) / (f_b0 - u2)) * np.exp(-k * t)
    f_b = (u1 - R * u2) / (1.0 - R)
    return (C - A * f_b) / B, f_b


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    """Both curves, from the setup's scalars and the run's own abscissa.

    The analytic side consumes exactly: Δ₀, the grid triple, T_bath, ω₀, n̄,
    c_phot, and ``arrays['t_ns']``. Nothing simulated enters it.
    """
    gap = float(setup.material.Delta_0)
    ne = int(setup.grid.num_bins)
    min_factor = float(setup.grid.min_factor)
    dE = (float(setup.grid.max_factor) - min_factor) * gap / ne
    n_bar = float(setup.subgap_drive.n_bar)
    c_phot = float(setup.subgap_drive.c_phot)
    m = round(float(setup.subgap_drive.omega_0) / dE)

    # The reduction, not a preference. With another collision channel on, the
    # closed form is not a statement about this run at all, so refuse rather
    # than report a red verdict that a reader would charge to the sub-gap
    # kernel. Note the deliberate asymmetry: `subgap_drive.enabled = False` is
    # NOT refused here. There the closed form does apply and is violated, and a
    # numeric FAIL is the honest result — that null run is the falsification
    # test for this benchmark and it must produce a verdict, not an error.
    if setup.collisions.scattering or setup.collisions.recombination:
        raise ValueError(
            "the sub-gap closed form is the reduction with the electron-phonon "
            "channels off; got scattering="
            f"{setup.collisions.scattering}, recombination="
            f"{setup.collisions.recombination}."
        )
    if setup.pb_drive.enabled:
        raise ValueError(
            "the pair-breaking drive is on, so the sub-gap term is not the only "
            "term on the right-hand side and the two-rung solution does not apply."
        )
    if min_factor < 1.0:
        # Sub-gap cells have zero capacity, and the kernel checks
        # supported[j_down] for the down partner but not supported[j_up] for the
        # up partner (sub_gap_photon.py:141 vs :148). That asymmetry is a code
        # path this closed form does not model, so keep the grid above Δ where
        # active_mask is all-True and the branch cannot differ.
        raise ValueError(
            f"grid.min_factor={min_factor:g} puts cells below the gap, where the "
            "cell capacity vanishes; this closed form assumes every cell is "
            "supported. Use min_factor >= 1."
        )

    t = np.asarray(arrays["t_ns"], dtype=float)
    E_engine = np.asarray(arrays["E_bins"], dtype=float)
    f_sim = np.asarray(arrays["f_snapshots"], dtype=float)

    # Rebuild the cell centres from the setup and check the run is on the grid
    # the closed form assumes. Reading E_bins and calling it "the run's grid"
    # would otherwise import a uniformity assumption without testing it.
    E = min_factor * gap + (np.arange(ne, dtype=float) + 0.5) * dE
    if E_engine.shape != E.shape or not np.allclose(E_engine, E, rtol=1e-12, atol=0.0):
        raise ValueError(
            "the run's energy grid is not the uniform cell-centred grid this "
            "closed form assumes; the jump partner i+m is then not at ω₀."
        )

    pairs, partnerless = _ladder_pairs(ne, m)
    w, rho_bar, K_plus = _bcs_cell_measure(gap, min_factor * gap, dE, ne)

    # The transient always starts at equilibrium: Transient0DSetup has no
    # InitialCondition field (only Spatial2DSetup does), so the thermal state is
    # the only start this mode can express.
    f0 = 1.0 / (np.exp(E / (_KB_UEV_PER_K * float(setup.T_bath))) + 1.0)

    # AUDIT: partnerless bins are frozen for the KERNEL, not for the run. ETD2
    # is called with balance_weights = cell_weights and its weighted-balance
    # projection rescales every supported bin, so bins with no kernel coupling
    # still drift O(dt²) — measured 1.4e-06 and 2.3e-06 relative at the pinned
    # dt, falling by exactly 4.000 per halving. The closed form holds them
    # fixed; that difference is part of the residual, not hidden from it.
    f_ana = np.repeat(f0[None, :], t.size, axis=0)
    for a, b in pairs:
        K = K_plus[a, b]
        f_ana[:, a], f_ana[:, b] = _pair_trajectory(
            f0[a],
            f0[b],
            c_phot * rho_bar[b] * K,  # A: absorption a → b and its reverse
            c_phot * rho_bar[a] * K,  # B: emission   b → a and its reverse
            n_bar,
            t,
        )

    number_sim = f_sim @ w
    number_ana = f_ana @ w
    y_sim = np.vstack([(f_sim @ (w * E)) / number_sim / gap, number_sim / number_sim[0]])
    y_ana = np.vstack([(f_ana @ (w * E)) / number_ana / gap, np.ones(t.size)])

    return Curve(
        x=t,
        y_sim=y_sim,
        y_analytic=y_ana,
        x_label="t (ns)",
        y_label="⟨E⟩/Δ  and  N(t)/N(0)",
        series_labels=("⟨E⟩(t)/Δ", "N(t)/N(0)"),
        note=(
            f"{len(pairs)} two-rung ladders {pairs[0]}…{pairs[-1]} at m={m} "
            f"(ω₀={m * dE:g} μeV = {m}·δE), plus partnerless bins {partnerless}. "
            f"Series 1 is the activity test: ⟨E⟩/Δ moves "
            f"{100.0 * (y_sim[0, -1] / y_sim[0, 0] - 1.0):+.2f}% over this run, and "
            "scoring the drive-off run against this same closed form gives 1.8e-01, "
            "5.9e+05× the tolerance. Series 2 is a conservation certificate ONLY — "
            "it is equally flat (0.0e+00) with the drive off, so on its own it "
            "cannot tell whether the term is active."
        ),
    )


register(
    Benchmark(
        name="subgap-drive",
        title="Sub-gap photon drive: two-rung ladder (exact Riccati)",
        tier="T1",
        formula_latex=(
            r"\begin{aligned}"
            r"&\text{Every other term off and } N_E/2\le m=\omega_0/\delta E<N_E"
            r"\ \Rightarrow\ \text{each bin } a \text{ has at most one partner } b=a+m:\\"
            r"&\qquad \dot f_a=-A\,J,\qquad \dot f_b=+B\,J,\qquad"
            r" J=\bar n\,f_a(1-f_b)-(\bar n+1)\,f_b(1-f_a),\\"
            r"&\qquad A=c_{\rm ph}\,\bar\rho_b\,K^{+}_{ab},\qquad"
            r" B=c_{\rm ph}\,\bar\rho_a\,K^{+}_{ab},\qquad K^{+}_{ab}=1+r_a r_b,\\"
            r"&\qquad w_i=\sqrt{E_i^{+2}-\Delta^2}-\sqrt{E_i^{-2}-\Delta^2},\qquad"
            r" \bar\rho_i=\frac{w_i}{\delta E},\qquad"
            r" r_i=\frac{\Delta\big[\operatorname{arccosh}(E_i^{+}/\Delta)"
            r"-\operatorname{arccosh}(E_i^{-}/\Delta)\big]}{w_i}.\\[4pt]"
            r"&\text{Then } C=B f_a+A f_b=\tfrac{c_{\rm ph}K^{+}_{ab}}{\delta E}"
            r"\big(w_a f_a+w_b f_b\big) \text{ is conserved, and } u=f_b"
            r" \text{ obeys the Riccati equation}\\"
            r"&\qquad \dot u=-A u^{2}+P u+Q,\qquad P=C-A\bar n-B(\bar n+1),"
            r"\qquad Q=C\,\bar n,\\[4pt]"
            r"&\qquad \boxed{\;f_b(t)=\frac{u_1-R(t)\,u_2}{1-R(t)},\qquad"
            r" R(t)=\frac{f_b(0)-u_1}{f_b(0)-u_2}\,e^{-k t},\qquad"
            r" f_a(t)=\frac{C-A\,f_b(t)}{B}\;}\\[2pt]"
            r"&\qquad u_1=\frac{2Q}{-P+\sqrt{P^{2}+4AQ}},\qquad"
            r" u_2=-\frac{-P+\sqrt{P^{2}+4AQ}}{2A},\qquad k=A(u_1-u_2)"
            r"=\sqrt{P^{2}+4AQ},\\[2pt]"
            r"&\qquad f_i(0)=\big[e^{E_i/k_BT_{\rm bath}}+1\big]^{-1},\qquad"
            r" \text{partnerless bins: } \dot f_i=0.\\[4pt]"
            r"&\text{Curves: }\quad"
            r" \frac{\langle E\rangle(t)}{\Delta}"
            r"=\frac{\sum_i w_i E_i f_i(t)}{\Delta\sum_i w_i f_i(t)},"
            r"\qquad \frac{N(t)}{N(0)}"
            r"=\frac{\sum_i w_i f_i(t)}{\sum_i w_i f_i(0)}=1."
            r"\end{aligned}"
        ),
        reason=(
            "The sub-gap term is a jump operator in energy, so with the photon "
            "step chosen above half the grid span (N_E/2 <= m < N_E) no bin has "
            "both an up and a down partner and the operator decomposes into "
            "independent two-rung ladders. Each ladder conserves B·f_a + A·f_b, "
            "which reduces it to one scalar Riccati equation with an elementary "
            "solution. No linearisation is used: the Pauli factors are carried in "
            "full, so the closed form is exact for arbitrary occupation and at "
            "every grid spacing, and the only approximation left in the run is "
            "the ETD2 time step."
        ),
        headline_latex=(
            r"\langle E\rangle(t)/\Delta \quad \mathrm{with} \quad \sum_i \rho_i f_i = "
            r"\mathrm{const}"
        ),
        rel_tol=3e-07,
        convergence=(
            "TIME STEP (N_E=16, m=9, the same 201 snapshot times; max relative "
            "error on <E>(t)/Delta, and elementwise on f(E,t) over 16 bins x 201 "
            "times):\n"
            "     dt (ns)  steps   err <E>   order |  err f(E,t)  order\n"
            "       0.125    400  2.697e-06     -   |   2.482e-04    -\n"
            "      0.0625    800  6.741e-07  2.000  |   6.202e-05  2.001\n"
            "     0.03125   1600  1.685e-07  2.000  |   1.550e-05  2.000\n"
            "  * 0.015625   3200  4.212e-08  2.000  |   3.875e-06  2.000   (* shipped)\n"
            "   0.0078125   6400  1.053e-08  2.000  |   9.687e-07  2.000\n"
            "  0.00390625  12800  2.631e-09  2.001  |   2.423e-07  1.999\n"
            "Order 2.000 over six points: ETD2 is second order, so the residual is "
            "pure time discretisation. rel_tol 3e-07 is 7.1x the shipped residual "
            "and 114x the finest measured one -- it is a discretisation floor, not "
            "a number chosen until the check passed.\n\n"
            "ENERGY GRID (fixed dt=0.015625 ns; m = 9*N_E/16 keeps omega_0 = 202.5 "
            "ueV and the two-rung structure): N_E = 16 / 32 / 64 gives 4.2124e-08 / "
            "6.1323e-08 / 8.0361e-08. It does NOT fall, and that is the point of the "
            "row: the closed form is exact on every grid because the ladder is a "
            "jump operator and not a quadrature, so refining dE cannot reduce a "
            "residual that is purely temporal. The mild 1.9x growth is consistent "
            "with the extra, faster ladders that appear at the gap edge -- the "
            "gap-edge cell-average DOS goes as dE^(-1/2) -- raising the ETD2 "
            "truncation error at fixed dt.\n\n"
            "SIZE. N_E=16 with 3200 steps runs in 6.6-7.4 s. The alternative, N_E=64 "
            "(28 ladders), costs 9.2 s and measures the same 8.0e-08: refining the "
            "grid buys this benchmark nothing, while N_E=16 keeps the seven ladders "
            "and the two partnerless bins individually inspectable. Shortening the "
            "50 ns window is the one thing that would cost something -- it spans "
            "4.7 e-folds of the slowest ladder (tau = 3.50, 6.76, 8.03, 8.90, 9.56, "
            "10.10, 10.54 ns), and the approach to the plateau is where the seven "
            "rates separate.\n\n"
            "NUMBER CONSERVATION: max |N(t)/N(0) - 1| = 3.9e-12 over the 201 points, "
            "at the ETD2 balance tolerance (32*eps ~ 7e-15) accumulated over 3200 "
            "steps.\n\n"
            "WHERE IT DOES NOT CONVERGE, reported not hidden: the identical case at "
            "T_bath = 0.1 K (the UI default) has residual 7.9e-05 and it GROWS as dt "
            "shrinks -- 6.5e-06 / 1.9e-05 / 4.1e-05 / 7.9e-05 / 1.6e-04 for dt = "
            "0.125 down to 0.0078125, i.e. proportional to the step count. Cause, "
            "traced to source: ETD2's weighted-balance projection restores missing "
            "mass as f <- 1 - (1-f)*h (qpsim/solvers/etd.py::"
            "_project_weighted_balance), which carries an ABSOLUTE error of order "
            "eps for f << 1. At 0.1 K the occupations run down to 2e-27 and are "
            "destroyed. This is an integrator finding, not a sub-gap-kernel error, "
            "and it is why the case is pinned at T_bath = 0.5 K where the smallest "
            "occupation is 4.7e-06."
        ),
        modes=("transient_0d",),
        build=_build,
        caveat=(
            "1. THE 'BOTH PARTNERS' KERNEL BRANCH IS NEVER EXECUTED. The two-rung "
            "reduction is bought by giving every bin at most ONE partner, so the "
            "branch where a bin sums an up AND a down contribution is untested "
            "here. That branch is the one that runs for the UI default (omega_0 = "
            "22 ueV on a 400-bin [Delta,10Delta] grid is a ~65-rung ladder, which "
            "has no closed form). The companion that would cover it is the "
            "stationary law f_{i+m}/(1-f_{i+m}) = [n/(n+1)] f_i/(1-f_i) for every "
            "consecutive pair of a ladder of any length, whose stationarity "
            "requires exactly that cancellation; it needs a long run to steady "
            "state and is not shipped here.\n"
            "2. A SECOND UNTESTED BRANCH (audit finding): the kernel checks "
            "supported[j_down] for the down partner but not supported[j_up] for "
            "the up partner (sub_gap_photon.py:141 vs :148). At min_factor = 1.0 "
            "the active mask is all-True and the two branches cannot differ, so "
            "this case can never exercise the asymmetry. A grid with a gap cut or "
            "a zero-capacity cell would; _build refuses such a grid rather than "
            "compare against a formula that does not model it.\n"
            "3. 'PARTNERLESS BINS ARE FROZEN' IS A STATEMENT ABOUT THE KERNEL, NOT "
            "THE RUN (audit finding; the original derivation said 'exactly frozen' "
            "and that is false at run level). ETD2 runs with balance_weights = "
            "cell_weights, and _project_weighted_balance rescales every supported "
            "bin, so bins 7 and 8 -- which have no kernel coupling to anything -- "
            "drift by 1.372e-06 and 2.312e-06 relative at the shipped dt. It is "
            "truncation, not roundoff: the drift falls by exactly 4.000 per "
            "halving of dt.\n"
            "4. THE N(t) CURVE IS A CONSERVATION CERTIFICATE, NOT AN ACTIVITY ONE "
            "(audit finding). It is 1 to 3.9e-12 with the drive on and to 0.0e+00 "
            "with it off -- the null is FLATTER than the signal. It is not a "
            "tautology of the balance projection either, because the projection "
            "targets N + (dt/2)*sum(w*(R_n+R_p)) and a leaking kernel drags its "
            "own target: substituting the partner's POINT DOS for its cell average "
            "(the failure mode the kernel's source comment warns about) drifts N by "
            "4.0e-02 in a real run, and dropping the partner Pauli factor by "
            "2.3e-02. Both are 5 orders above the tolerance.\n"
            "5. BIT-FOR-BIT AGREEMENT WITH SpectralContext MEANS THE FINITE-VOLUME "
            "CONVENTION ITSELF IS NOT UNDER TEST. rho_bar and K+ are re-derived "
            "here from the BCS antiderivatives and match the engine's arrays to "
            "0.0 relative difference, which is the signature of a T2 in disguise. "
            "The audit attacked it by mutation instead of by inspection and the "
            "tier survived; re-measured here at the shipped commit, patching "
            "qpsim.backends.t3_diffusion.sub_gap_photon_collision_rates and "
            "re-running the whole transient against the UNCHANGED closed form "
            "gives: identity re-implementation 4.2124e-08 (control, matches the "
            "baseline exactly), point DOS instead of the cell average 4.0e-02, "
            "swapped n_bar/(n_bar+1) 8.7e-02, K- instead of K+ 5.1e-02, no "
            "coherence factor 2.1e-02, dropped partner Pauli factor 2.3e-02, "
            "deleted down-partner branch 8.8e-01 -- every one caught, by 7e+04 to "
            "3e+06 times the tolerance, and a T2 assembled from the engine's own "
            "kernel could not have caught any of them. (Three are dominated by the "
            "conservation "
            "series; on the headline curve alone the audit measured 9.0e-03, "
            "1.4e-03 and 1.6e-01 for the point DOS, the dropped Pauli factor and "
            "the deleted branch, plus 1.8e-02 for the wrong photon energy m=8.) "
            "What agreement does NOT prove is the "
            "convention: if the engine's cell-average DOS were the wrong measure "
            "for this integral, this benchmark would agree with it anyway. The "
            "continuum check closes most of that gap -- the engine's rate constant "
            "converges to the textbook point-BCS value c*(1+D^2/(Ea*Eb))*[n*rho(Eb)"
            "+(n+1)*rho(Ea)] at order 1.980 in dE, from 5.2e-04 at N_E=16 to "
            "5.4e-07 at N_E=512 -- but that comparison is made on SpectralContext "
            "arrays rather than through a time-stepped run, so it is not part of "
            "the shipped verdict.\n"
            "6. THE DETECTION FLOOR IS ~5e-06 RELATIVE ON THE RATE CONSTANT. "
            "Perturbing c_phot on the ANALYTIC side alone, against an unchanged "
            "run, moves the score to 4.0e-08 (pass) / 1.6e-07 (pass) / 6.2e-07 "
            "(fail) / 6.5e-06 (fail) for perturbations of 1e-6 / 3e-6 / 1e-5 / "
            "1e-4, so the verdict flips between 3 and 10 ppm and a rate error "
            "below ~5 ppm passes. That floor is a consequence of the tolerance, "
            "which is a consequence of ETD2's order -- it is not a hole that a "
            "different closed form would close.\n"
            "7. THE CASE IS A CORNER OF PARAMETER SPACE. omega_0 = 1.125*Delta on a "
            "[Delta,3Delta] grid, far from the schema default 22 ueV; there is no "
            "way to get a two-rung ladder otherwise. c_phot = 0.01 /ns is 1.7e8 "
            "times the schema default, which is a pure time rescaling because the "
            "operator is exactly linear in c_phot, but nothing here validates the "
            "physical magnitude of the coupling. T_bath = 0.5 K is T/T_c = 0.42 for "
            "Al; with no recombination, no scattering and Delta frozen there is no "
            "thermal fixed point, so the bath only sets the initial condition -- do "
            "not read these numbers as a device state.\n"
            "8. SCOPE. 0-D transient only. Delta is frozen, phonons are pinned at "
            "the Bose bath, there is no transport, and nothing is coupled to any "
            "other term. The identical kernel is also called from "
            "qpsim/collisions/spatial.py, qpsim/backends/t3_spatial.py and "
            "qpsim/solvers/newton_steady_state.py; none of those assembly paths is "
            "covered. Also untested: x_qp, the Mattis-Bardeen probe, the "
            "dense-output interpolation (deliberately avoided by making "
            "snapshot_interval/dt an exact integer of exact binary fractions), the "
            "ETD2 rate-subcycling path (max loss rate is 0.235/ns, so "
            "max_loss_step = 0.25 would only bind at dt > 1.07 ns), the "
            "non-commensurate frequency snap, and the Dynes path (the kernel "
            "refuses dynes_gamma > 0 by design).\n"
            "9. The headline is a weighted average, in which errors could in "
            "principle cancel. They do not: f(E,t) checked elementwise over all 16 "
            "bins and 201 times gives 3.875e-06 at the shipped dt, converging at "
            "order 2.000. That stricter check is not the shipped verdict because "
            "its floor is 92x higher, and a 92x looser tolerance would cost most of "
            "two decades of sensitivity on the rate constant."
        ),
        activity=(
            "1. THE NULL RUN IS BIT-EXACT. With subgap_drive.enabled = false and "
            "every other term already off, the engine returns max over 201 "
            "snapshots x 16 bins of |f(t) - f(0)| = 0.000e+00 EXACTLY. There is "
            "nothing else in this case that can move f, so 100% of the motion is "
            "this term.\n"
            "2. THE BENCHMARK FAILS WITH THE TERM OFF. Scoring that null run "
            "against the same closed form gives 1.7786e-01 = 5.93e+05 x the "
            "tolerance. It is not a check that passes either way.\n"
            "3. THE MOTION IS LARGE. f at bin 9 (E = 393.75 ueV, the upper rung of "
            "the fastest ladder) rises 1.074306e-04 -> 7.301062e-03, a factor 68.0; "
            "over all bins max |f_end/f_thermal - 1| = 66.96. The headline curve "
            "moves <E>/Delta from 1.161780 to 1.413111, +21.633%. Measured against "
            "the same curve the verdict is: the swing is 7.2e+05 times the "
            "tolerance and 5.1e+06 times the residual.\n"
            "4. BOTH HALVES OF THE OPERATOR ARE EXERCISED. At n_bar = 3 the final "
            "<E>/Delta is 1.413111; at n_bar = 0 it is 1.158488, BELOW the thermal "
            "1.161780. That is the correct physics and a sharp discriminator: "
            "n_bar = 0 deletes the absorption half (rate ~ n_bar) and leaves "
            "spontaneous emission (~ n_bar+1), which pulls quasiparticles DOWN the "
            "ladder. The schema default n_bar = 0.0 is therefore not literally "
            "inert for this drive, but it removes half the operator. Scored across: "
            "the n_bar = 0 RUN against the n_bar = 3 closed form fails at 1.8019e-01 "
            "(6.0e+05 x the tolerance), while the same run against its OWN closed "
            "form passes at 7.5e-12. The Bose factors are pinned individually, not "
            "just their sum.\n"
            "5. SEVEN RATES ARE PINNED INDIVIDUALLY. Fitting the exact invariant "
            "(f_b-u1)/(f_b-u2) to the engine's time series recovers each ladder's k "
            "to 7.3e-07 / 2.9e-07 / 8.5e-07 / 1.7e-06 / 3.0e-06 / 5.3e-06 / 9.0e-06 "
            "relative (k = 0.285443, 0.147879, 0.124589, 0.112375, 0.104552, "
            "0.099031, 0.094899 /ns). One global rate error could not reproduce "
            "seven independent values. But the curve is only MILDLY "
            "multi-exponential (audit finding; the original derivation called it "
            "'a genuine sum of SEVEN distinct exponentials, not a single decay', "
            "which is oversold): the best single exponential A(1-exp(-k t)), k = "
            "0.190/ns, already reproduces <E>(t) to a max residual of 3.9e-03, "
            "1.55% of the 0.2513 swing -- the audit's differently parameterised fit "
            "gave 6.7%, and the claim is weak either way. Pair (0,9) alone carries "
            "48.0% of the rise and the top two carry 71.5%. What the seven rates "
            "buy is not curve shape, it is that the RATE CONSTANT is pinned bin by "
            "bin rather than on average.\n"
            "6. THE STATIONARY LAW IS APPROACHED. At t = 50 ns every pair satisfies "
            "f_b/(1-f_b) = [n/(n+1)] f_a/(1-f_a) with n/(n+1) = 0.750000 to within "
            "1.411e-02. That is the un-relaxed remainder rather than an error: the "
            "slowest ladder still carries exp(-50*0.0949) = 8.7e-03 of its initial "
            "displacement, and the deviation sits a factor 1.6 above it."
        ),
        extra={
            "case_overrides": CASE_OVERRIDES,
            # Pinned because the numbers above are measurements, and a
            # measurement without the tree it was taken on is an anecdote. The
            # audit's own numbers were taken at b799b9d with a dirty tree in the
            # 2-D spatial region; these were re-measured here on a clean tree.
            "verified_at_commit": "41ee7f8",
        },
    )
)
