r"""Phonon scattering source — the phonon-side booking of quasiparticle scattering.

Every quasiparticle that scatters from ``E_i`` down to ``E_j`` puts a phonon of
``ω = E_i − E_j`` into the phonon field, and every absorption takes one out.
:func:`qpsim.collisions.phonon.compute_phonon_source_sink` books that traffic as
the two coefficients of an affine ODE, ``dn_ω/dt = a_ω + b_ω n_ω``, with ``a_ω``
the emission sum and ``b_ω`` emission minus absorption.

The reduction. Hold the quasiparticle occupation still and that equation is
exactly linear, diagonal in ω, and solved in closed form: an exponential
approach to ``n^∞_ω``, one rate and one fixed point per frequency. Nothing else
in this case can move ``n_ph`` away from its seed — the escape term only pulls
it back towards the bath, and recombination is off, so the sum lattice carries
no source at all — so the whole ω-resolved departure of the phonon field from
that seed *is* this term, and that departure is what is compared, not the
occupation it sits on.

What holds ``f`` still is a timescale separation, not a switch: with the escape
time ``τ_l`` far below the quasiparticle relaxation time the phonons reach
their fixed point while ``f`` has not moved, and the residual is first order in
``τ_l/τ_0``. That is a limit of the *model*, taken along the axis the schema
already documents — ``τ_l → 0`` is the pinned ``thermal_bath`` sector — rather
than a falsified material constant: ``material.tau_0`` here is aluminium's real
438 ns. Only the ratio matters, and the measurement confirms it (see
``convergence``): moving the same factor of 1000 onto ``τ_0`` instead, and
leaving ``τ_l`` at the schema default, gives 4.0072e-05 — the shipped residual
to every digit measured.

The curve is the fractional phonon excess above the bath,

    y(ω, t) = n_ph(ω, t)/n_B(ω, T_bath) − 1,

over 127 driven frequencies and 20 times. It is exactly zero with the term off,
so the falsification is unambiguous — 1.0000 relative error, every point — and
it cannot be flattered by pushing further into the limit, which is the trap the
raw occupation would fall into: ``n_ph`` itself agrees to 1.5e-06 here only
because the excess it carries is at most 4% of it.

Authored against the audited position. The audit's central finding — that the
analytic side is a transcription of the engine's finite-volume discretisation
and is therefore T2-equivalent for the kernel VALUES, whatever it is for the
time law — is accepted, and the tier says so. Points marked ``AUDIT`` below are
its other findings, applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# Written out rather than imported from qpsim.constants: an analytic side that
# shares a constant table with the engine cannot detect a shifted constant,
# because it cancels from both sides of the comparison.
_KB_UEV_PER_K = 86.17333262145

# The case. It is the 0-D reduction of the 2-D mode (a 1x1 mask), because the
# phonon field is per cell and this term needs no transport to act.
#
# NOT SET HERE, deliberately: ``collisions.phonon_scattering_source``. On this
# route it is INERT — run_kinetics does not forward it, T3SpatialBackend does
# not accept it, and SpatialCollisions.advance_phonons calls
# compute_phonon_source_sink with ``enable_scattering`` alone. Writing it into
# the case would state a switch that is not honoured, which is exactly the
# defect the audit charged against the original derivation. The term is
# switched here by ``collisions.scattering``, which on this route removes the
# quasiparticle-side channel with it; see ``activity`` and ``caveat`` item 3.
#
# The three numbers that are not free:
#   tau_l_ns = 1.7e-4 is the schema default 0.170 ns / 1000. The residual is a
#     sum of two terms with opposite slopes in it — the frozen-f error, first
#     order in tau_l, and double-precision cancellation in an excess that
#     shrinks with tau_l — and this sits at their minimum (see ``convergence``).
#   amplitude = f_FD(E_0, 0.6 K) with E_0 = 181.40625 ueV the first cell centre.
#     The energy profile is peak-normalised, so this exact amplitude is what
#     makes the seeded occupation the Fermi function itself rather than a
#     rescaling of its shape; the closed form does not need that (a_w and b_w
#     are explicit sums for any frozen f) but the identity -a_w/b_w =
#     n_B(w, T_eff) does, and it is checked in the note the run carries.
#   max_time = 6*tau_l with snapshot_interval = dt = max_time/20, so every
#     sample is a step endpoint (the 2-D backend never interpolates a frame)
#     and the exponential is resolved from 0.3 to 6 escape times.
# stop_tol = 0 disables the early stop on purpose. The backend's convergence
# residual is max|df/dt|, which says nothing about the phonon field, so a run
# could be certified steady while n_ph was still relaxing. The cost is one
# "did not reach stop_tol" note on a run that was never meant to.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "kinetics",
    "material.name": "Al",
    "material.Delta_0": 180.0,
    "material.T_c": 1.1837,
    "material.tau_0": 438.0,
    "material.D_0": 0.0,
    "material.dynes_gamma": 0.0,
    "T_bath": 0.2,
    "grid.min_factor": 1.0,
    "grid.max_factor": 3.0,
    "grid.num_bins": 128,
    "geometry.kind": "rectangle",
    "geometry.rows": 1,
    "geometry.cols": 1,
    "collisions.scattering": True,
    "collisions.recombination": False,
    "phonons.mode": "dynamic_escape",
    "phonons.tau_l_ns": 1.7e-4,
    "initial.kind": "absolute",
    "initial.amplitude": 0.029069834452214446,
    "initial.energy.kind": "thermal",
    "initial.energy.T_eff": 0.6,
    "initial.space.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "self_consistent_gap": False,
    "dt": 5.1e-5,
    "max_time": 1.02e-3,
    "snapshot_interval": 5.1e-5,
    "stop_tol": 0.0,
}


def _energy_grid(setup: Any) -> tuple[np.ndarray, float, np.ndarray]:
    """Cell centres, spacing and contiguous edges, from the setup's scalars."""
    gap = float(setup.material.Delta_0)
    ne = int(setup.grid.num_bins)
    e_min = float(setup.grid.min_factor) * gap
    dE = (float(setup.grid.max_factor) - float(setup.grid.min_factor)) * gap / ne
    E = e_min + (np.arange(ne, dtype=float) + 0.5) * dE
    edges = (E[0] - 0.5 * dE) + np.arange(ne + 1, dtype=float) * dE
    return E, dE, edges


def _cell_measure(
    gap: float, edges: np.ndarray, dE: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Finite-volume BCS measure on a uniform grid, from the antiderivatives.

    Over a cell ``[E⁻, E⁺]`` above the gap::

        ∫ E/√(E²−Δ²) dE = √(E²−Δ²)          → capacity  w_i,  ρ̄_i = w_i/δE
        ∫ Δ/√(E²−Δ²) dE = Δ·arccosh(E/Δ)    → anomalous a_i,  r_i = a_i/w_i

    ``r_i`` is the cell average of Δ/E under the DOS measure, so the phonon
    coherence factor is ``K⁻ = max(0, 1 − r_i r_j)``: the product form makes the
    double-cell average of ``1 − Δ²/(E_i E_j)`` factorise exactly, which is what
    keeps a cell cut by the gap edge finite.

    Written from the physics, not read from ``SpectralContext`` — and it agrees
    with ``SpectralContext`` to 0.0 relative, which is the fingerprint of a
    transcription and is why the tier is what it is. See ``caveat`` item 1.
    """
    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], lo)
    # Factored, as the engine's quadrature is: E²−Δ² cancels catastrophically
    # in the first cell, whose lower edge sits exactly at Δ.
    w = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0)) - np.sqrt(
        np.maximum((lo - gap) * (lo + gap), 0.0)
    )
    anomalous = gap * (
        np.arccosh(np.maximum(hi / gap, 1.0)) - np.arccosh(np.maximum(lo / gap, 1.0))
    )
    r = np.zeros_like(w)
    supported = w > 0.0
    r[supported] = np.clip(anomalous[supported] / w[supported], 0.0, 1.0)
    return w / dE, r


def _seed_occupation(setup: Any, E: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """The prepared ``f(E)`` the run starts from, rebuilt from the setup.

    Mirrors :func:`qpsim.fields.initial.energy_profile`: the shape is
    peak-normalised over the supported cells and then scaled by ``amplitude``,
    so ``absolute`` with ``amplitude = f_FD(E_0, T_eff)`` is the Fermi function
    itself. Rebuilt rather than read from ``snap_f[0]`` so that a wrong initial
    condition shows up as a failed benchmark instead of being adopted as truth.
    """
    spec = setup.initial
    thermal = 1.0 / (np.exp(E / (_KB_UEV_PER_K * float(setup.T_bath))) + 1.0)
    if spec.kind == "thermal":
        return thermal
    if spec.expression is not None or spec.energy.kind != "thermal":
        raise ValueError(
            "this closed form rebuilds only a thermal-shaped initial energy "
            f"profile; got kind={spec.energy.kind!r}, expression="
            f"{spec.expression!r}."
        )
    if spec.space.kind != "uniform":
        raise ValueError(
            f"the initial condition is {spec.space.kind!r} in space, so f is not "
            "the same in every cell; this closed form is written for one cell."
        )
    supported = rho > 0.0
    shape = 1.0 / (np.exp(E / (_KB_UEV_PER_K * float(spec.energy.T_eff))) + 1.0)
    shape = np.where(supported, shape, 0.0)
    field = float(spec.amplitude) * shape / np.max(np.abs(shape))
    if spec.kind == "excess":
        field = field + thermal
    return np.clip(np.where(supported, field, 0.0), 0.0, 1.0)


def _omega_lattice(E: np.ndarray, dE: float) -> np.ndarray:
    """The phonon frequency lattice: the union of the difference and sum sets.

    Rebuilt here rather than taken from the payload, which does not carry it.
    It is the same construction as
    :func:`qpsim.collisions.phonon.build_phonon_frequency_map` on a uniform
    grid — differences ``k·δE`` and sums ``2E_0 + m·δE``, merged at the
    arithmetic resolution of the energy operands — and ``_build`` checks its
    length against the phonon axis the run actually recorded.
    """
    ne = E.size
    values = np.concatenate([
        dE * np.arange(ne, dtype=float),
        2.0 * E[0] + dE * np.arange(2 * ne - 1, dtype=float),
    ])
    merge_tol = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(E))))
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.empty(ordered.size, dtype=bool)
    starts[0] = True
    starts[1:] = np.diff(ordered) > merge_tol
    cluster = np.cumsum(starts, dtype=np.int64) - 1
    unique = np.bincount(cluster, weights=ordered) / np.bincount(cluster)
    unique[np.abs(unique) <= merge_tol] = 0.0
    return unique


def _bose(omega: np.ndarray, temperature: float) -> np.ndarray:
    """``n_B(ω, T)``, zero at ω = 0 (the bookkeeping mode carries no energy)."""
    out = np.zeros_like(omega)
    positive = omega > 0.0
    with np.errstate(over="ignore", under="ignore"):
        z = omega[positive] / (_KB_UEV_PER_K * temperature)
        # exp(-z)/(-expm1(-z)) rather than 1/(exp(z)-1): the driven bins run to
        # ω/kT ≈ 21, where the direct form has already lost digits to the
        # subtraction and would overflow before the occupation underflows.
        out[positive] = np.exp(-z) / (-np.expm1(-z))
    return out


def _source_sink(
    setup: Any,
    E: np.ndarray,
    dE: float,
    rho: np.ndarray,
    r: np.ndarray,
    f: np.ndarray,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(a_ω, b_ω, driven)`` for the scattering channel at frozen ``f``.

    ``a_ω = δE Σ_l ρ̄_l ρ̄_{l+k} K⁰ˢ f_{l+k}(1−f_l)`` is emission into ω = k·δE
    and the absorption sum is the same expression with the two ends exchanged,
    so ``b_ω`` is emission minus absorption. The Pauli factors cancel
    identically in ``b`` and not in ``a``: that asymmetry is why the fixed point
    has to be part of the check, and it is the mutation the table in
    ``activity`` catches at 3.0e-02.

    ``K⁰ˢ`` is the PHONON-side kernel ``2K⁻/(π Δ τ₀^PB)`` (F&C 2023 Eq. 12),
    which is the one the phonon equation is written on. It used to be the
    quasiparticle-side ``(E_i−E_j)²K⁻/(τ₀(k_BT_c)³)`` because the engine put
    that one in the phonon equation — so this closed form agreed with the
    engine about the wrong equation and scored 4.0e-05 against a 1.2e-04
    tolerance while n_ph was out by a factor of order 30. A T2 check assembles
    from the engine's own kernels and therefore cannot detect that those
    kernels are the wrong ones; only the tier rule warns you, and the fix is
    to correct the engine, not to loosen this.
    """
    ne = E.size
    kBTc = _KB_UEV_PER_K * float(setup.material.T_c)
    coherence = np.maximum(1.0 - r[:, None] * r[None, :], 0.0)
    if setup.phonons.use_phonon_side_kernel:
        tau_0_pb_ns = setup.material.tau_0_pb_ns
        if tau_0_pb_ns is None or not np.isfinite(tau_0_pb_ns) or tau_0_pb_ns <= 0.0:
            raise ValueError(
                "the phonon-side kernel needs a finite positive "
                f"material.tau_0_pb_ns; got {tau_0_pb_ns!r}."
            )
        # This form builds ONE pair-breaking spectrum at the material gap, and
        # _build refuses a self-consistent gap for exactly that reason, so
        # Delta_0 is the gap the engine used.
        gap = float(setup.material.Delta_0)
        kernel = (2.0 / (np.pi * gap * float(tau_0_pb_ns))) * coherence
    else:
        transfer = E[:, None] - E[None, :]
        kernel = (
            (transfer * transfer) * coherence
            / (float(setup.material.tau_0) * kBTc**3)
        )
    np.fill_diagonal(kernel, 0.0)

    # dE·ρ̄_i f_i · K · ρ̄_j(1−f_j): one emission event from i to j. The uniform
    # spacing is what lets a scalar δE stand for the engine's per-column dE
    # array; compute_phonon_source_sink refuses a non-uniform grid outright.
    occupied = rho * f
    vacant = rho * np.maximum(1.0 - f, 0.0)
    events = dE * (occupied[:, None] * kernel * vacant[None, :])

    a_k = np.zeros(ne)
    b_k = np.zeros(ne)
    for k in range(1, ne):
        lower = np.arange(0, ne - k)
        emission = float(np.sum(events[lower + k, lower]))
        absorption = float(np.sum(events[lower, lower + k]))
        a_k[k] = emission
        b_k[k] = emission - absorption

    level = np.rint(omega / dE).astype(int)
    if np.any(np.abs(omega - level * dE) > 1e-9 * dE):
        raise ValueError(
            "the phonon frequency lattice is not a multiple of the energy "
            "spacing, so the difference lattice cannot be indexed by k = ω/δE."
        )
    driven = (level >= 1) & (level <= ne - 1)
    a = np.zeros(omega.size)
    b = np.zeros(omega.size)
    a[driven] = a_k[level[driven]]
    b[driven] = b_k[level[driven]]
    return a, b, driven


def _build(setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> Curve:
    """The ω-resolved phonon excess, simulated and closed form.

    The analytic side consumes exactly: Δ₀, T_c, τ₀, the grid triple, T_bath,
    τ_l, the initial condition, and ``arrays['snap_t_ns']``. No simulated
    quantity enters it.
    """
    if setup.collisions.recombination:
        raise ValueError(
            "recombination is on, so the phonon equation also carries the "
            "pair source on the sum lattice and this closed form — which is "
            "the scattering channel alone — does not describe the run."
        )
    if setup.phonons.mode == "thermal_bath":
        raise ValueError(
            "the thermal-bath sector pins n_ph at the bath: there is no phonon "
            "equation for this term to act on, and no phonon field is recorded."
        )
    if setup.self_consistent_gap:
        raise ValueError(
            "a self-consistent gap moves Δ during the run, so the kernel and "
            "the density of states this closed form is built on move with it."
        )
    if setup.initial.kind == "thermal":
        raise ValueError(
            "the run starts at equilibrium, where this term satisfies detailed "
            "balance per bin (a_ω + b_ω n_B(ω,T_bath) = 0) and the phonon field "
            "is rigorously constant. Both sides of the comparison would be "
            "identically zero, which is a benchmark that cannot fail. Prepare a "
            "quasiparticle occupation away from the bath instead."
        )
    if setup.injection.enabled or setup.subgap_drive.enabled or setup.pb_drive.enabled:
        raise ValueError(
            "a drive is enabled; it moves f on its own timescale and the "
            "frozen-occupation reduction no longer holds."
        )
    if any(drive.enabled for drive in setup.drives):
        raise ValueError("a prescribed drive is enabled; see above.")
    if float(setup.grid.min_factor) < 1.0:
        raise ValueError(
            f"grid.min_factor={float(setup.grid.min_factor):g} puts cells below "
            "the gap. They carry zero capacity and no scattering events, and "
            "the seeded occupation there is discarded rather than prepared."
        )
    if "snap_n_ph" not in arrays or "snap_t_ns" not in arrays:
        raise ValueError(
            "the run recorded no phonon frames; set snapshot_interval, and a "
            "dynamic phonon sector."
        )
    n_ph = np.asarray(arrays["snap_n_ph"], dtype=float)
    if n_ph.shape[2] != 1:
        raise ValueError(
            f"the geometry has {n_ph.shape[2]} cells. The closed form is per "
            "cell at frozen f; use a 1x1 mask (the 0-D reduction) so there is "
            "one occupation and one phonon field to compare."
        )

    t = np.asarray(arrays["snap_t_ns"], dtype=float)
    gap = float(setup.material.Delta_0)
    E, dE, edges = _energy_grid(setup)
    rho, r = _cell_measure(gap, edges, dE)
    f0 = _seed_occupation(setup, E, rho)
    omega = _omega_lattice(E, dE)
    if omega.size != n_ph.shape[1]:
        raise ValueError(
            f"rebuilt an ω lattice of {omega.size} bins; the run recorded "
            f"{n_ph.shape[1]}. The two constructions disagree, so nothing "
            "downstream can be compared bin by bin."
        )
    a, b, driven = _source_sink(setup, E, dE, rho, r, f0, omega)

    # 0.0 is the engine's τ_l → ∞ sentinel, so the escape term is absent in the
    # closed sector rather than infinitely fast.
    inverse_tau_l = (
        0.0 if setup.phonons.mode == "dynamic_closed"
        else 1.0 / float(setup.phonons.tau_l_ns)
    )
    n_bath = _bose(omega, float(setup.T_bath))
    a_eff = a + inverse_tau_l * n_bath
    b_eff = b - inverse_tau_l
    decaying = b_eff != 0.0
    n_inf = np.where(decaying, -a_eff / np.where(decaying, b_eff, 1.0), 0.0)
    analytic = np.where(
        decaying[None, :],
        n_inf[None, :]
        + (n_bath - n_inf)[None, :] * np.exp(b_eff[None, :] * t[:, None]),
        # b_eff = 0 is the undriven bin of a closed sector: linear in t, and
        # with a_eff = 0 there too it is simply the seed held.
        n_bath[None, :] + a_eff[None, :] * t[:, None],
    )

    # AUDIT: the 255 sum-lattice bins are a null channel, not data. Both sides
    # are the same constant there, so including them would dilute the reported
    # rms by ~1.7x and inflate the point count by 3x. They are measured and
    # reported as a leakage check instead.
    simulated = n_ph[:, :, 0]
    quiet = (~driven) & (n_bath > 0.0)
    leak = float(np.max(np.abs(simulated[:, quiet] / n_bath[quiet] - 1.0)))
    seed = float(np.max(np.abs(simulated[0, n_bath > 0.0] / n_bath[n_bath > 0.0] - 1.0)))
    bookkeeping = omega == 0.0
    zero_mode = (
        float(np.max(np.abs(simulated[:, bookkeeping]))) if np.any(bookkeeping) else 0.0
    )

    # The excess above the bath, which is the term's whole contribution: it is
    # identically zero with the term off, and unlike n_ph itself it does not
    # tend to a constant as the reduction is pushed further into its limit.
    scale = n_bath[driven]
    excess_sim = simulated[1:, driven] / scale - 1.0
    excess_ana = analytic[1:, driven] / scale - 1.0
    raw = float(
        np.max(np.abs(simulated[1:, driven] - analytic[1:, driven])
               / analytic[1:, driven])
    )
    # The seeded occupation is Fermi-Dirac iff a_ω/(−b_ω) is the Bose function
    # at T_eff, bin by bin — an identity of the closed form, checked here on the
    # numbers the run was actually built from rather than asserted.
    warm = float(setup.initial.energy.T_eff or setup.T_bath)
    active = driven & (b < 0.0)
    ratio = -a[active] / b[active] / _bose(omega[active], warm)
    fermi = float(np.max(np.abs(ratio - 1.0))) if np.any(active) else float("nan")
    escaping = setup.phonons.mode != "dynamic_closed"
    tau_l = float(setup.phonons.tau_l_ns)

    return Curve(
        x=omega[driven],
        y_sim=excess_sim,
        y_analytic=excess_ana,
        x_label="ω (μeV)",
        y_label="n_ph(ω,t)/n_B(ω,T_bath) − 1",
        series_labels=tuple(
            f"t = {time / tau_l:.1f} τ_l" if escaping else f"t = {time:.3g} ns"
            for time in t[1:]
        ),
        log_x=True,
        log_y=True,
        note=(
            f"{int(np.count_nonzero(driven))} driven ω bins "
            f"({omega[driven][0]:.4g}–{omega[driven][-1]:.4g} μeV) × "
            f"{t.size - 1} times. The t = 0 frame is dropped: the excess is "
            "identically zero on both sides there, and the seed it checks is "
            f"reported instead — max |n_ph(0)/n_B − 1| = {seed:.2e}. Null "
            f"channel: {int(np.count_nonzero(quiet))} sum-lattice bins, which "
            "this term never feeds and recombination is off, drift by "
            f"{leak:.2e}; the ω = 0 bookkeeping bin stays at {zero_mode:.1e}. "
            f"The identity −a_ω/b_ω = n_B(ω, {warm:g} K) holds to {fermi:.1e}, "
            "which is what makes the fixed point a weighted average of two "
            f"Bose functions. Scored on n_ph itself rather than on its excess, "
            f"the same run reads {raw:.2e}: a weaker statement, not a tighter "
            "one, since the excess is at most a few percent of the occupation "
            "it sits on and the excess is the whole of what this term does."
        ),
    )


register(
    Benchmark(
        name="phonon-scattering-source",
        title="Phonon scattering source: the ω-resolved excess above the bath",
        # AUDIT: the original claimed T1 and "no engine array is read into the
        # analytic curve". The second is true and the first does not follow
        # from it. T1 in the time law — the exponential flow of an affine ODE
        # is written from the physics — and T2-equivalent in the per-ω rate
        # constant, which is a transcription of the engine's own finite-volume
        # discretisation and would reproduce a wrong kernel formula unchanged.
        # The weaker of the two labels is the one that ships.
        tier="T2",
        formula_latex=(
            r"\begin{aligned}"
            r"&\text{Frozen } f \Rightarrow \dot n_\omega = a_\omega + b_\omega n_\omega"
            r" + \frac{n_B(\omega,T_b)-n_\omega}{\tau_l}"
            r" \text{ is affine and diagonal in } \omega:\\[2pt]"
            r"&\qquad \boxed{\;n_\omega(t)=n^\infty_\omega"
            r"+\bigl[n_B(\omega,T_b)-n^\infty_\omega\bigr]e^{-\lambda_\omega t},"
            r"\qquad \lambda_\omega=\tfrac{1}{\tau_l}+|b_\omega|,\qquad"
            r" n^\infty_\omega=\frac{a_\omega+n_B(\omega,T_b)/\tau_l}{\lambda_\omega}\;}"
            r"\\[6pt]"
            r"&\text{with, for } \omega=k\,\delta E\ \text{ and } \rho\text{-weighted"
            r" pair sums over } (l,\,l+k):\\"
            r"&\qquad a_\omega=\delta E\sum_l \bar\rho_l\bar\rho_{l+k}"
            r"K^{s}_{0}(l,l+k)\,f_{l+k}\bigl(1-f_l\bigr),\qquad"
            r" b_\omega=\delta E\sum_l \bar\rho_l\bar\rho_{l+k}K^{s}_{0}(l,l+k)"
            r"\bigl(f_{l+k}-f_l\bigr),\\"
            r"&\qquad K^{s}_{0}(i,j)=\frac{(E_i-E_j)^2\,K^-_{ij}}{\tau_0 (k_BT_c)^3},"
            r"\qquad K^-_{ij}=\max\bigl(0,\,1-r_i r_j\bigr),\qquad"
            r" \bar\rho_i=\frac{w_i}{\delta E},\\"
            r"&\qquad w_i=\sqrt{E_i^{+2}-\Delta^2}-\sqrt{E_i^{-2}-\Delta^2},"
            r"\qquad r_i=\frac{\Delta\bigl[\operatorname{arccosh}(E^+_i/\Delta)"
            r"-\operatorname{arccosh}(E^-_i/\Delta)\bigr]}{w_i}.\\[6pt]"
            r"&\text{For } f=f_{FD}(E,T_{qp})\text{ the Fermi factors give}"
            r"\ f_{l+k}(1-f_l)=e^{-\omega/k_BT_{qp}}f_l(1-f_{l+k})\ \text{term by"
            r" term, hence exactly}\\"
            r"&\qquad a_\omega=|b_\omega|\,n_B(\omega,T_{qp}),\qquad"
            r" n^\infty_\omega=\frac{|b_\omega|\,n_B(\omega,T_{qp})"
            r"+n_B(\omega,T_b)/\tau_l}{|b_\omega|+1/\tau_l},\\[4pt]"
            r"&\text{so the source alone would carry the phonon field to the"
            r" quasiparticle temperature.}\\[6pt]"
            r"&\text{Curve: }\quad y(\omega,t)=\frac{n_\omega(t)}{n_B(\omega,T_b)}-1"
            r"\;=\;\bigl[\text{0 with the term off}\bigr]."
            r"\end{aligned}"
        ),
        reason=(
            "TIER, PRECISELY: T1 in the time law and T2-equivalent in the per-ω "
            "rate constant. The exponential flow is written from the physics, "
            "but a_ω and b_ω are rebuilt from the same finite-volume formulas "
            "the engine uses and would reproduce a wrong kernel unchanged, so "
            "the weaker label ships. "
            "With the quasiparticle occupation held still the phonon equation is "
            "exactly affine and diagonal in ω, so it has an elementary solution: "
            "one exponential per frequency, with rate 1/τ_l + |b_ω| and fixed "
            "point (a_ω + n_B/τ_l)/(1/τ_l + |b_ω|). The engine integrates that "
            "same affine flow exactly (advance_phonons composes exp(Bh) with "
            "φ₁(Bh) through expm1), so there is no time-discretisation error to "
            "remove and the whole residual is the frozen-occupation reduction "
            "plus arithmetic. What the curve tests per ω is the combination a_ω "
            "+ b_ω n_B(ω,T_bath) — the phonon-side collision integral evaluated "
            "at the bath occupation — over 127 frequencies at once."
        ),
        # 3.0x the shipped residual, which is itself a minimum: one decade
        # either way in the reduction parameter costs a factor 10 (shallower)
        # or 2.5 (deeper, where cancellation takes over). See ``convergence``.
        headline_latex=(
            r"n_\omega(t) \;=\; n_\omega^\infty + "
            r"\left(n_\omega(0)-n_\omega^\infty\right)e^{-\lambda_\omega t}, \quad \lambda_\omega "
            r"= \frac{1}{\tau_l} + |b_\omega|"
        ),
        rel_tol=1.2e-04,
        convergence=(
            "THE REDUCTION PARAMETER is tau_l/tau_0 — the phonon escape time "
            "against the quasiparticle relaxation time. Only the ratio enters, "
            "and the residual is FIRST ORDER in it. Ladder in tau_l at the "
            "shipped tau_0 = 438 ns (Al), window fixed at 6*tau_l, 20 frames, "
            "NE = 128:\n"
            "    tau_l (ns)   tau_l/tau_l(Al)   max rel err   ratio\n"
            "     1.7e-02          1/10           3.9990e-03      -\n"
            "     1.7e-03          1/100          4.0065e-04   9.98\n"
            "     1.7e-04          1/1000         4.0072e-05   10.00   <- shipped\n"
            "     1.7e-05          1/10000        1.0227e-04   0.39    <- turns\n"
            "The turn is not a failure of the reduction: it is double precision. "
            "The excess being compared shrinks with tau_l, and the engine forms "
            "it as a ratio of two O(1/tau_l) quantities, so its relative "
            "resolution degrades as 1/tau_l. Substituting the exact frozen "
            "(a_w, b_w) into the engine — the control row of the mutation table "
            "in ``activity``, which leaves the phonon equation EXACTLY the "
            "closed form — still reports 1.0226e-05, so that much is arithmetic "
            "and nothing else. The two dominate opposite ends of the frequency "
            "range (the reduction at high omega, cancellation at low omega), so "
            "the reported maximum is the larger of them, and the shipped "
            "4.0072e-05 is the reduction error at the top bin.\n\n"
            "THE SAME LADDER IN tau_0, at the shipped tau_l, confirms that only "
            "the ratio matters and that no material constant is doing secret "
            "work: tau_0 = 43.8 / 438 / 4380 / 4.38e4 ns gives 4.0065e-04 / "
            "4.0072e-05 / 1.0830e-04 / 5.1125e-04 — the same first-order fall on "
            "the way in, matching the tau_l rows to five digits where the "
            "reduction dominates, and the same turn once cancellation does (the "
            "rows past the turn agree only to a factor ~2, because there the "
            "maximum is set by whichever noise-dominated point happens to win). "
            "The sharpest form of the same statement: put the whole factor of "
            "1000 on tau_0 (4.38e5 ns) and leave tau_l at the schema default "
            "0.170 ns, and the residual is 4.0072e-05 — identical to the shipped "
            "case in every digit reported.\n\n"
            "TIME STEP, at fixed window: 10 / 20 / 40 / 80 frames (dt = 1.02e-04 "
            "down to 1.275e-05 ns) gives 4.1064e-05 / 4.0072e-05 / 3.9533e-05 / "
            "4.8068e-05. FLAT, as it must be: the phonon step is the EXACT flow "
            "of the affine ODE, so two half-steps compose to the exact full step "
            "and there is no truncation error to converge away. The rise at 80 "
            "frames is accumulated roundoff over 4x the steps. If a future "
            "change makes the phonon step approximate, this row stops being flat "
            "— treat that as the tripwire it is.\n\n"
            "ENERGY GRID: NE = 32 / 64 / 128 gives 3.8594e-05 / 3.9582e-05 / "
            "4.0072e-05. FLAT, and necessarily so — the analytic side is "
            "evaluated on the same finite-volume grid as the engine, so their "
            "difference is not a discretisation error and refinement cannot "
            "remove it. NE = 128 is shipped because it buys 127 frequencies for "
            "0.14 s; NE = 32 measures the same number with 31.\n\n"
            "WINDOW: 2 / 6 / 18 escape times gives 2.4992e-05 / 4.0072e-05 / "
            "1.3506e-04 — the frozen-occupation error accumulates with elapsed "
            "time while the excess saturates after ~3 tau_l, so a longer run is "
            "strictly worse. Six is where the exponential is fully resolved.\n\n"
            "AGAINST THE CONTINUUM (the ladder the original derivation declared "
            "impossible, and the reason rel_tol here is NOT a physics-error "
            "bound). The engine's discrete b_omega compared with scipy.quad of\n"
            "  b(w) = (w^2/(tau_0 (k_B T_c)^3)) * INT_Delta^{Emax-w} dE rho(E) "
            "rho(E+w) [1 - D^2/(E(E+w))] [f(E+w) - f(E)]\n"
            "at fixed physical w, NE = 32 -> 1024 (relative difference, then the "
            "ratio per doubling):\n"
            "   w=45  ueV: 6.00e-03 2.13e-03 7.59e-04 2.71e-04 9.65e-05 3.43e-05"
            "  (2.82 2.80 2.80 2.81 2.81)\n"
            "   w=90  ueV: 1.11e-02 4.05e-03 1.47e-03 5.26e-04 1.88e-04 6.70e-05"
            "  (2.74 2.77 2.78 2.80 2.81)\n"
            "   w=180 ueV: 1.48e-02 5.44e-03 1.97e-03 7.10e-04 2.54e-04 9.05e-05"
            "  (2.72 2.76 2.78 2.80 2.81)\n"
            "   w=270 ueV: 1.75e-02 6.46e-03 2.35e-03 8.46e-04 3.03e-04 1.08e-04"
            "  (2.70 2.75 2.78 2.79 2.80)\n"
            "Order log2(2.80) = 1.49 at every frequency: the discretisation IS "
            "convergent physics. But at the shipped NE = 128 it still sits 7.6e-04 "
            "to 2.3e-03 away from the continuum integral, which is 20x rel_tol. "
            "STATE BOTH NUMBERS: 1.2e-04 is agreement with the DISCRETE model; "
            "~2e-03 is how well that model represents the integral it stands for."
        ),
        modes=("kinetics",),
        build=_build,
        caveat=(
            "1. THE ω RATE CONSTANT IS NOT INDEPENDENTLY DERIVED. rho_bar, r, K- "
            "and the omega lattice are rebuilt here from the BCS antiderivatives "
            "and agree with SpectralContext and build_phonon_frequency_map to "
            "0.0 relative — bit-identical, which is transcription (same closed "
            "forms, same operation order), not independent derivation. If the "
            "engine's kernel FORMULA is wrong, this benchmark is wrong the same "
            "way and still reports 4e-05. What the transcription cannot hide is "
            "the ASSEMBLY, and that is separately demonstrated by mutating the "
            "ENGINE and leaving the analytic side alone (see ``activity``): six "
            "engine-side corruptions, all caught by 2 to 5 orders. So: assembly "
            "validated, kernel values not. The continuum ladder in "
            "``convergence`` is the closest thing to an independent check of the "
            "values, and it is reported rather than shipped as the verdict "
            "because at NE = 128 it is 20x looser.\n"
            "2. THIS ROUTE NOW USES THE PHONON-SIDE KERNEL, AND SO DOES THIS "
            "CLOSED FORM (corrected 2026-08-15). Until then "
            "SpatialCollisions.advance_phonons called compute_phonon_source_sink "
            "with the raw quasiparticle-side K_s0 = (E_i-E_j)^2 "
            "K-/(tau_0 (k_B T_c)^3), never passed K_s0_phonon_side, and "
            "run_kinetics did not forward phonons.use_phonon_side_kernel at "
            "all — so kinetics solved a different phonon equation from the "
            "0-D routes, with n_ph out by an omega-dependent factor of order "
            "30. This caveat reported that as a finding a benchmark could not "
            "fix; it has since been fixed in the engine. Both sides now run "
            "F&C 2023 Eq. 12's 2K-/(pi Delta tau_0^PB), so the verdict DOES "
            "pin tau_0_pb_ns and the paper-faithful prefactor.\n"
            "   The tier caveat above still stands and is the lesson: this is "
            "a T2, its analytic side was assembled from the engine's own "
            "kernels, and it passed at 4.01e-05 against a 1.2e-04 tolerance "
            "for as long as the wrong kernel was in place. It scores 4.0072e-05 "
            "now, on the corrected kernel — the same number, because a T2 "
            "measures assembly and time integration and those did not change.\n"
            "3. THE TERM IS SWITCHED BY collisions.scattering, NOT BY "
            "collisions.phonon_scattering_source (AUDIT: the original reported "
            "three wiring gaps and this is a fourth of the same kind, on a "
            "different route). On kinetics the phonon-source flags reach "
            "nothing: run_kinetics does not forward them and T3SpatialBackend "
            "does not take them. So the null run removes the quasiparticle-side "
            "scattering channel as well. That is harmless HERE — at frozen f the "
            "quasiparticle side moves nothing measurable, and the phonon field "
            "is the only observable compared — but it means this case cannot "
            "distinguish 'the phonon-side booking is missing' from 'the whole "
            "scattering channel is missing'. The 0-D transient route, where the "
            "flag is live, could; it publishes no phonon field to compare.\n"
            "4. THE RATE IS THE ESCAPE TERM'S, NOT THIS TERM'S. lambda_omega = "
            "1/tau_l + |b_omega| and the reduction requires |b_omega| tau_l << 1 "
            "(it is 8e-11 here), so the exponential's TIME constant is set by "
            "phonon escape and the term under test sets the AMPLITUDE. On this "
            "route that is unavoidable: the same kernel carries both the "
            "quasiparticle relaxation and the phonon source, and the source is "
            "smaller by a factor of order f, so a case with |b| tau_l ~ 1 would "
            "also have f moving faster than n_ph. What is pinned per omega is "
            "therefore the combination a_omega + b_omega n_B(omega, T_bath), not "
            "a_omega and b_omega separately — though both enter it, weighted by "
            "n_B, which is O(1) at the low-omega end.\n"
            "5. THE CASE IS A REDUCTION, NOT A DEVICE. tau_l = 1.7e-4 ns is the "
            "schema default over 1000 and no film escapes phonons in 0.17 ps; "
            "the run lasts 1.02 ps. The seeded occupation is a 0.6 K Fermi "
            "function at a gap frozen at its 0 K value, which no self-consistent "
            "solve would produce (T/T_c = 0.51). Recombination is off, so "
            "nothing sets the quasiparticle number and there is no thermal fixed "
            "point. Read the numbers as a measurement of one operator, not as a "
            "state of aluminium. The physical-parameter point is on the ladder: "
            "at tau_l = 0.17 ns the same comparison is off by 3.9e-02, i.e. the "
            "closed form is simply not valid there and the benchmark says so.\n"
            "6. ONLY 2468 OF 2540 POINTS ARE SCORED. The predicted excess spans "
            "eleven decades over the frequency range — it carries the kernel's "
            "(E_i-E_j)^2, so it vanishes quadratically as omega -> 0 — and the "
            "framework's relative floor (1e-9 of the curve's own peak) drops the "
            "72 points where it falls below what double precision resolves in "
            "the occupation it sits on. Those bins are not evidence either way; "
            "the count is reported in the payload as n_below_floor.\n"
            "7. SCOPE. One cell, one phonon branch, fixed gap, pure BCS "
            "(dynes_gamma = 0, which the collision kernels require). No "
            "transport, no photon drive, no self-consistent gap, no "
            "recombination — so the sum lattice never carries a source and "
            "nothing here tests whether the difference and sum lattices couple "
            "correctly when both channels are live. On this grid they cannot "
            "even meet: the difference lattice tops out at 2Delta - dE and the "
            "sum lattice starts at 2Delta + dE, so the documented "
            "incommensurate-lattice pathology is out of reach by construction "
            "rather than by luck."
        ),
        activity=(
            "1. THE BENCHMARK FAILS WITH THE TERM OFF, AT EXACTLY 1.0000. With "
            "collisions.scattering = false the phonon coefficients are "
            "identically zero (a_ph = b_ph = 0.0 on every bin, not merely "
            "small), n_ph never leaves its seed, and the excess the closed form "
            "predicts is compared against zero: max relative error 1.0000e+00 at "
            "every one of the 2468 scored points, 8.3e+03 x the tolerance. There "
            "is nothing else in this case that can move n_ph.\n"
            "2. THE NULL CHANNEL IS STRUCTURAL. 255 of the 383 omega bins lie on "
            "the sum lattice, which the scattering source never feeds and which "
            "recombination — off here — would. They stay at 1.4e-14 relative "
            "drift for the whole run while the driven bins move, and the omega = "
            "0 bookkeeping bin stays at exactly 0.0. The run carries its own "
            "control, and it is reported in the curve's note.\n"
            "3. SIX ENGINE-SIDE CORRUPTIONS ARE CAUGHT (AUDIT: the original "
            "table perturbed the ANALYTIC side, which measures sensitivity, not "
            "the ability to catch an engine bug; this one patches "
            "compute_phonon_source_sink and leaves the closed form alone, so "
            "each row is a regression the benchmark would have stopped):\n"
            "     control, exact coefficients substituted   1.0226e-05  passes\n"
            "     kernel prefactor x 1.001                  1.0080e-03  CAUGHT\n"
            "     coherence factor K- -> K+                 7.6793e+00  CAUGHT\n"
            "     Pauli blocking (1-f) dropped              2.9940e-02  CAUGHT\n"
            "     absorption half dropped (b := a)          1.8341e+00  CAUGHT\n"
            "     point-sampled DOS instead of cell average 2.9152e-01  CAUGHT\n"
            "     every event shifted one omega bin         4.5463e-01  CAUGHT\n"
            "   The prefactor row is linear: x(1+eps) gives 1.008*eps, so at "
            "rel_tol = 1.2e-04 the check resolves a 1.2e-04 relative error in "
            "the kernel — not the 4e-04 the original claimed by scaling the "
            "wrong number.\n"
            "4. THE MOTION IS THE TERM'S, AND IT IS OMEGA-RESOLVED. The excess "
            "rises from 0 to 4.08e-02 at the top driven bin and to 5.3e-13 at "
            "the bottom one, a spread of eleven decades that follows "
            "|b_omega| n_B(omega,T_qp)/n_B(omega,T_bath) bin by bin. A single "
            "wrong scale factor cannot reproduce 127 values spanning that range, "
            "which is what a fixed-point or scalar check cannot see.\n"
            "5. THE QUASIPARTICLE SIDE IS QUIET, AS THE REDUCTION REQUIRES. Over "
            "the whole run max |f(t) - f(0)| = 8.7e-09 (4.7e-05 relative), and "
            "that drift IS the residual: it is the same 4e-05, and both fall "
            "together by exactly 10 per decade of tau_l/tau_0."
        ),
        extra={
            "case_overrides": CASE_OVERRIDES,
            # The numbers above are measurements, and a measurement without the
            # tree it was taken on is an anecdote. Re-measured on 545355f after
            # the per-edge-boundary and prescribed-gap commits landed: every
            # row reproduced digit for digit.
            "verified_at_commit": "545355f",
        },
    )
)
