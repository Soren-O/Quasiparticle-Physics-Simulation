r"""Phonon escape to the substrate: the ``(n_bath − n)/τ_l`` term in the Ph0 sector.

The term
--------
Per ω bin and per cell the engine's phonon equation is affine,

    dn/dt = a_ph(ω)[f] + b_ph(ω)[f]·n + (n_bath(ω) − n)/τ_l,

and is advanced with the exact map ``n·e^{Bh} + A·φ₁(Bh)·h`` (``φ₁`` through
``expm1``); see ``qpsim/collisions/spatial.py::advance_phonons``, which builds
``a_eff = a_ph + n_bath/τ_l`` and ``b_eff = b_ph − 1/τ_l``. The escape term is
those two additions and nothing else: one rate ``1/τ_l``, the same in every
bin, toward the Bose–Einstein bath at ``T_bath``.

Why the obvious reduction is NOT the one shipped here
-----------------------------------------------------
The textbook check is "switch both e-ph channels off so a_ph = b_ph = 0, start
the phonons away from the bath, watch them relax at 1/τ_l". **That case cannot
be built through today's schema, and if it were scored it would pass with the
escape term switched off.** ``SpatialCollisions.__init__`` seeds
``n_ph = thermal_phonon_occupation(omega_bins, T_bath)`` and ``advance_phonons``
relaxes toward ``thermal_phonon_occupation(omega_bins, T_bath)`` — the same call
with the same arguments — so with the sources off the run starts exactly at the
term's own fixed point and nothing moves. Measured: that case gives
max |n(t)/n(0) − 1| = 3.1888e-16 with escape ON and exactly 0 with escape OFF;
the two trajectories are indistinguishable. ``Spatial2DSetup.initial`` prepares the
QUASIPARTICLE occupation; there is no phonon initial condition on
``PhononSector``, so the departure the reduction needs is unreachable. This is
the audit's blocker-class finding on the original derivation, and it is not
fixed — it is worked around here, and ``_build`` refuses the vacuous case
outright rather than reporting a green verdict for a term that did nothing.

The reduction that IS reachable
-------------------------------
Prepare the EMPTY superconductor: ``initial.kind = "absolute"`` with amplitude
0, i.e. ``f ≡ 0``, and leave both collision channels ON. Count factors of ``f``
in ``compute_phonon_source_sink`` (``qpsim/collisions/phonon.py``): every
scattering term carries ``n_qp = ρf`` and every recombination-emission term
carries it twice, so all of them vanish; the pair-BREAKING term carries
``partner = ρ(1−f)`` twice and survives. Hence, exactly,

    a_ph ≡ 0,      b_ph(ω) = −Γ_pb(ω) ≤ 0,

with Γ_pb supported only on the sum lattice ω ≥ 2Δ. The phonon equation is then
a constant-coefficient scalar affine ODE per bin whose solution can be written
with the source coefficient eliminated:

    n(ω,t)/n_bath(ω) = τ_eff/τ_l + (1 − τ_eff/τ_l)·e^{−t/τ_eff},
    1/τ_eff(ω) = 1/τ_l + Γ_pb(ω).

The escape term supplies **both** explicit appearances of τ_l: the rate it adds
to the total, and — the sharper statement — the level the population settles
at, ``n_∞/n_bath = τ_eff/τ_l``. Read the curve as: *the fraction of the bath
population that survives equals the fraction of the total relaxation rate that
escape contributes.* Get τ_l's magnitude, units or target wrong and that
identity breaks by 9 to 12 orders (falsification table in ``convergence``).

``f ≡ 0`` is not merely a good approximation here, it is exact: max |f| over
every recorded frame of the shipped case is 0.0, so ``a_ph`` and ``b_ph`` are
frozen constants and the exact exponential map composes exactly. (Even without
that, df/dt at f = 0 is 8.551e-21 — the thermal pair-breaking phonon flux — so
the coefficients would drift by ~1e-20 over the run.)

Tier: T2, and precisely where
-----------------------------
Everything belonging to the ESCAPE term is written here from the setup and the
SI defining constants and is genuinely under test — τ_l, its sign, its
frequency independence, and the Bose–Einstein target coded from
k_B = 1.380649e-23/1.602176634e-19 × 1e6 μeV/K. That is demonstrably not
decorative: rescoring the same simulated curve against the engine's own rounded
literal (86.17333262145) drops the residual 118× to 2.9e-14, and the 3.44e-12 it
drops from is exactly ``ω_max/(k_B T_bath) × Δk_B/k_B`` = 165.8 × 2.078e-14, the
amplified constant mismatch. The whole tolerance is that one number.

But ``Γ_pb(ω)`` is read from the engine's own recombination kernel, because the
only reachable non-trivial phonon dynamics run THROUGH pair breaking. A wrong
pair-breaking kernel is invisible here — both sides move together. That makes
the whole benchmark T2 in the framework's sense even though the term under test
is not itself reused, and it is why the audit's recommended T1 is not claimed:
the T1 case is the unreachable one above.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    compute_phonon_source_sink,
)
from qpsim.webui.benchmarks import Benchmark, Curve, register
from qpsim.webui.builders import build_spectral

# The catalogue case.
#
#   initial.kind = "absolute" with amplitude 0 is the whole reduction: an
#     occupation that IS zero, not a zero EXCESS over thermal. The schema
#     rejects amplitude 0 only for kind="excess" (which would prepare exactly
#     the thermal state), and allows it here, which is the one setting that
#     freezes f while leaving a live phonon coefficient.
#   Both collision channels stay ON. Nothing is switched off: the isolation is
#     a property of the prepared state, so the case exercises the shipped
#     physical model rather than a term-stripped variant. Setting
#     collisions.scattering = False gives a bit-identical verdict (residual and
#     rms agree to all 10 printed digits, difference exactly 0.000e+00) — the
#     scattering channel is already identically zero at f = 0.
#   stop_tol = 0.0 is REQUIRED, not tidiness. The convergence certificate is
#     max |df/dt| over the quasiparticles only; it does not see the phonon
#     population at all. At f = 0 it reads 8.6e-21, so any positive stop_tol
#     ends the run after one step and the curve has two points.
#   T_bath = 0.1 K. The frozen-f hypothesis holds to O(n_bath(2Δ)) = O(e^{−2Δ/kT}),
#     so the bath temperature is the small parameter: 0.1 K keeps max |f| at
#     exactly 0.0 and 0.15 K at 1.5e-14, while 0.2 K reaches 2.320e-11 and 0.3 K
#     3.601e-08 — both above the frozen-f bound below, so build() refuses them
#     rather than reporting the resulting 3.1e-09 as a physics failure. Do not
#     raise T_bath to make the occupations look larger.
#   48 bins over [1, 4]Δ gives dE = 11.25 μeV and 2·E_face/dE = 32, an integer,
#     so the phonon difference and sum lattices are commensurate. Escape is
#     diagonal in ω and blind to that, but an incommensurate grid is an
#     unnecessary confound in an artifact shown to a physicist.
#   1×1 geometry. A larger mask adds no discriminating power: transport applied
#     to f ≡ 0 returns zero whether or not it is right, and every cell carries a
#     bit-identical copy of the phonon population. build() scores every cell it
#     is given, so 1×3 and 2×3 produce 285 and 570 series instead of 95 — and
#     the identical residual, 3.439589e-12, with cell-to-cell spread exactly
#     0.000e+00, at 6× the runtime.
#   phonons.use_phonon_side_kernel is deliberately ABSENT. It is an inert knob
#     on this path — SpatialCollisions is never given it and uses the QP-side
#     kernels in the phonon equation — and the audit was right that listing it
#     makes a case read as if a kernel choice were part of the setup when it is
#     not. The two phonon_source flags are absent for the same reason:
#     run_spatial_2d does not forward them to the spatial backend either.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material.name": "Al",
    "material.Delta_0": 180.0,
    "material.T_c": 1.18,
    "material.tau_0": 438.0,
    "material.dynes_gamma": 0.0,
    "T_bath": 0.1,
    "grid.min_factor": 1.0,
    "grid.max_factor": 4.0,
    "grid.num_bins": 48,
    "geometry.kind": "rectangle",
    "geometry.rows": 1,
    "geometry.cols": 1,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "collisions.scattering": True,
    "collisions.recombination": True,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "dynamic_escape",
    "phonons.tau_l_ns": 0.17,
    "self_consistent_gap": False,
    "initial.kind": "absolute",
    "initial.amplitude": 0.0,
    "initial.energy.kind": "flat",
    "initial.space.kind": "uniform",
    "dt": 0.03125,
    "max_time": 1.0,
    "stop_tol": 0.0,
    "snapshot_interval": 0.03125,
}

# k_B in μeV/K from the SI defining constants, NOT from qpsim.constants. The
# engine's literal is the same number rounded at the 13th digit, and that
# difference is what the residual floor is made of (see the module docstring),
# so importing it would silently retire the only part of the bath target this
# curve can still test.
_KB_UEV_PER_K = 1.380649e-23 / 1.602176634e-19 * 1.0e6

# The reduction assumes f ≡ 0. It is not assumed here, it is measured on the
# run's own frames. The bound is set from the observed sensitivity: at
# T_bath = 0.3 K a peak occupation of 3.601e-08 costs 3.149e-09 of residual, so
# the induced error runs about 0.09·max|f| and 1e-11 keeps it near 1e-12, an
# order below the constant-representation floor. Not 1e-9 (that would admit an
# error comparable to rel_tol itself) and not 0.0 (a legitimate run at 0.15 K
# carries 1.5e-14 and is fine).
_MAX_OCCUPATION = 1e-11

# The one failure mode this benchmark exists to prevent: a configuration in
# which the phonons never leave the bath, scored green against a closed form
# that also predicts they do not move. 1e-3 is far above the 3.2e-16 that the
# genuinely inert case produces and far below the 2.4e-1 the shipped case does,
# so no judgement is being exercised in between.
_MIN_DEPARTURE = 1e-3

# Grid-identity tolerance, relative to the top of the energy grid. The analytic
# side rebuilds the spectral context from the setup; if that is not the grid the
# run actually solved on, every ω bin is mislabelled and the comparison is
# meaningless rather than merely wrong. The two agree to 0.0 exactly here.
_GRID_TOL = 1e-12


def _bath_occupation(omega: np.ndarray, T_bath: float) -> np.ndarray:
    """Bose–Einstein ``1/(e^{ω/k_BT} − 1)``, written out rather than imported.

    The ω = 0 entry is the engine's zero-energy bookkeeping mode, which carries
    no energy transfer and is zero by qpsim convention; reproducing that
    convention is not evidence about it, so those bins are dropped from the
    scored curve rather than compared.
    """
    x = np.asarray(omega, dtype=float) / (_KB_UEV_PER_K * float(T_bath))
    out = np.zeros_like(x)
    positive = x > 0.0
    with np.errstate(over="ignore"):
        out[positive] = 1.0 / np.expm1(x[positive])
    return out


def _empty_state_coefficients(
    setup: Any, gap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(E, omega, a_ph, b_ph)`` of the phonon ODE at ``f ≡ 0``.

    Assembled by handing the engine's own kernels a zero occupation — the same
    call ``advance_phonons`` makes — rather than by re-deriving Kaplan pair
    breaking. That choice is what makes this benchmark T2; see the module
    docstring. ``a_ph`` comes back identically zero, which is the structural
    claim of the reduction and is asserted by the caller.
    """
    spectral = build_spectral(setup)
    if float(spectral.gap) != float(gap):
        raise ValueError(
            f"the run solved at a local gap of {gap:g} μeV while the setup's "
            f"material gap is {spectral.gap:g} μeV. This closed form builds one "
            "pair-breaking spectrum, at the material gap; a shifted or "
            "self-consistent gap needs a per-region assembly this benchmark "
            "does not do."
        )
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(spectral.E)
    k_s0 = (
        build_scattering_kernel_base(
            spectral, tau_0=setup.material.tau_0, T_c=setup.material.T_c
        )
        if setup.collisions.scattering
        else None
    )
    k_r0 = (
        build_recombination_kernel_base(
            spectral, tau_0=setup.material.tau_0, T_c=setup.material.T_c
        )
        if setup.collisions.recombination
        else None
    )
    a_ph, b_ph = compute_phonon_source_sink(
        np.zeros(spectral.E.size),
        spectral,
        k_s0,
        k_r0,
        idx_diff,
        idx_sum,
        diff_sign,
        int(omega.size),
        enable_scattering=setup.collisions.scattering,
        enable_recombination=setup.collisions.recombination,
    )
    return spectral.E, omega, a_ph, b_ph


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    if "snap_n_ph" not in arrays:
        raise ValueError(
            "this run recorded no phonon frames, so there is no escape "
            f"trajectory to check. phonons.mode = {setup.phonons.mode!r} pins "
            "n_ph at the bath and removes the phonon equation entirely; set it "
            "to 'dynamic_escape', and set snapshot_interval so frames are kept."
        )
    t = np.asarray(arrays["snap_t_ns"], dtype=float)
    n_sim = np.asarray(arrays["snap_n_ph"], dtype=float)   # (frames, n_omega, cells)
    if t.size < 4:
        raise ValueError(
            f"an exponential needs a curve, and this run recorded {t.size} "
            "frame(s). The usual cause is stop_tol: the convergence "
            "certificate is max|df/dt| over the QUASIPARTICLES and does not see "
            "the phonon population at all, so at f = 0 it reads ~1e-20 and any "
            "positive stop_tol ends the run after one step. Set stop_tol = 0."
        )

    # -- the reduction's hypothesis, measured rather than assumed -------------
    f_snapshots = np.asarray(arrays["snap_f"], dtype=float)
    f_peak = float(np.max(np.abs(f_snapshots)))
    if f_peak > _MAX_OCCUPATION:
        raise ValueError(
            "this reduction is the EMPTY superconductor: it needs f ≡ 0 so "
            "that the phonon source a_ph vanishes and b_ph is a frozen "
            f"constant. The run reached max |f| = {f_peak:.3e}, above the "
            f"{_MAX_OCCUPATION:g} bound, so a_ph is live and the coefficients "
            "drift. Set initial.kind = 'absolute' with amplitude 0, and keep "
            "T_bath low enough that thermal pair breaking does not refill the "
            "quasiparticles (0.1 K holds; 0.3 K does not)."
        )

    gaps = np.unique(np.asarray(arrays["gap_per_cell"], dtype=float))
    if gaps.size != 1 or np.unique(np.asarray(arrays["snap_gap"])).size != 1:
        raise ValueError(
            "the pair-breaking spectrum that sets b_ph is written for ONE gap; "
            f"this run carries {gaps.size} distinct local gap(s) and "
            f"{np.unique(np.asarray(arrays['snap_gap'])).size} over time. Use "
            "gap_regions.kind = 'uniform' and self_consistent_gap = False."
        )
    gap = float(gaps[0])

    E_ana, omega, a_ph, b_ph = _empty_state_coefficients(setup, gap)
    E = np.asarray(arrays["E_bins"], dtype=float)
    if E.shape != E_ana.shape or float(np.max(np.abs(E - E_ana))) > _GRID_TOL * float(
        np.max(np.abs(E))
    ):
        raise ValueError(
            "the analytic ω grid is not the engine's: the energy grid rebuilt "
            "from the setup does not reproduce the run's E_bins."
        )
    if n_sim.shape[1] != omega.size:
        raise ValueError(
            f"phonon frames carry {n_sim.shape[1]} ω bins; the grid rebuilt "
            f"from the setup has {omega.size}."
        )
    # a_ph is f-independent only in the sense that it is identically zero at
    # f = 0. If a kernel ever grows a term that survives an empty state, the
    # closed form below — which eliminates the source using n_∞ = n_bath·τ_eff/τ_l
    # — stops being the solution, silently. Catch that here.
    if float(np.max(np.abs(a_ph))) != 0.0:
        raise ValueError(
            "the phonon source a_ph is not identically zero at f = 0 "
            f"(max {np.max(np.abs(a_ph)):.3e}), so the empty-state reduction "
            "this closed form is written for no longer holds."
        )

    n_bath = _bath_occupation(omega, setup.T_bath)
    # Bins the escape term acts on ALONE (ω below 2Δ, where Γ_pb is exactly
    # zero) are excluded on purpose: there the run starts at the fixed point
    # and stays, so the series is flat at 1 by construction and would dilute
    # the verdict with points that cannot disagree. They are counted in the
    # note instead.
    live = (n_bath > 0.0) & (b_ph != 0.0)
    if not np.any(live):
        raise ValueError(
            "no ω bin has a live phonon coefficient, so the phonon population "
            "starts at the escape term's own fixed point (the engine seeds "
            "n_ph AT the bath it relaxes toward) and cannot move. Scoring this "
            "would report a pass for a run in which the term under test did "
            "nothing. The usual cause is collisions.recombination = False, "
            "which removes the pair-breaking channel that drives the phonons."
        )

    tau_l = float(setup.phonons.tau_l_ns)
    # 1/tau_eff = 1/tau_l + Gamma_pb; the escape term is the 1/tau_l in it and
    # the tau_eff/tau_l that fixes the saturation level.
    tau_eff = -1.0 / (b_ph[live] - 1.0 / tau_l)
    surviving = tau_eff / tau_l                      # = n_infinity / n_bath
    y_bin = (
        surviving[:, None]
        + (1.0 - surviving)[:, None] * np.exp(-t[None, :] / tau_eff[:, None])
    )

    # Normalised by the bath occupation, which is a per-series rescaling and so
    # leaves the pointwise relative error of every point unchanged — but lifts
    # every bin above the harness's 1e-9-of-peak floor. Scored raw, the bath
    # occupation falls from 1.9e-19 at 2Δ to 9.9e-73 at the top of the grid and
    # all but ~16 of the 95 bins would be dropped from the verdict.
    n_live = n_sim[:, live, :]                        # (frames, live, cells)
    y_sim = (n_live / n_bath[live][None, :, None]).transpose(1, 2, 0)
    n_cells = y_sim.shape[1]
    y_sim = y_sim.reshape(-1, t.size)
    y_ana = np.repeat(y_bin, n_cells, axis=0)

    departure = float(np.max(np.abs(y_sim - 1.0)))
    if departure < _MIN_DEPARTURE:
        raise ValueError(
            f"the phonon population moved by {departure:.3e} of the bath over "
            "the whole run, which is indistinguishable from not moving. The "
            "escape term relaxes toward the population the engine seeded, so a "
            "case with no live phonon source measures nothing; this is refused "
            "rather than scored."
        )

    omega_live = omega[live]
    labels = (
        tuple(f"ω = {w:.4g} μeV" for w in omega_live)
        if n_cells == 1
        else tuple(
            f"ω = {w:.4g} μeV, cell {c}" for w in omega_live for c in range(n_cells)
        )
    )
    note = (
        f"{omega_live.size} of {omega.size} ω bins carry a live coefficient "
        f"(ω = {omega_live.min():.4g} to {omega_live.max():.4g} μeV, i.e. the "
        f"sum lattice above 2Δ = {2.0 * gap:g} μeV); the other "
        f"{int(np.count_nonzero(~live))} sit at the escape term's fixed point "
        "and are flat at 1 by construction, so they are excluded rather than "
        f"counted as agreement. Γ_pb·τ_l runs {-b_ph[live].max() * tau_l:.4g} to "
        f"{-b_ph[live].min() * tau_l:.4g} across them, so escape supplies "
        f"{surviving.min():.4g} to {surviving.max():.4g} of the total "
        f"relaxation rate — which is exactly the surviving fraction "
        "n_∞/n_bath the curve saturates at. Peak occupation over the run: "
        f"max |f| = {f_peak:.3e} (the reduction needs 0)."
    )

    return Curve(
        x=t,
        y_sim=y_sim,
        y_analytic=y_ana,
        x_label="time (ns)",
        y_label="n(ω, t) / n_bath(ω)",
        series_labels=labels,
        # Linear in both: the ordinate is a ratio confined to [τ_eff/τ_l, 1] and
        # the abscissa spans a few escape times. Pointwise rather than "scale"
        # because the curve never approaches zero — the smallest value on it is
        # 0.76 of the peak — so dividing by |analytic| is the honest measure and
        # no point is floored (n_below_floor = 0).
        metric="pointwise",
        note=note,
    )


register(
    Benchmark(
        name="phonon-escape",
        title="Phonon escape: relaxation rate and surviving bath fraction",
        tier="T2",
        formula_latex=(
            r"\frac{n(\omega,t)}{n_{\mathrm{bath}}(\omega)}"
            r"=\frac{\tau_{\mathrm{eff}}}{\tau_l}"
            r"+\Bigl(1-\frac{\tau_{\mathrm{eff}}}{\tau_l}\Bigr)"
            r"e^{-t/\tau_{\mathrm{eff}}(\omega)},\qquad "
            r"\frac{1}{\tau_{\mathrm{eff}}(\omega)}=\frac{1}{\tau_l}"
            r"+\Gamma_{\mathrm{pb}}(\omega),\qquad "
            r"n_{\mathrm{bath}}(\omega)=\frac{1}{e^{\omega/k_BT_{\mathrm{bath}}}-1}"
        ),
        reason=(
            "From an empty superconductor (f ≡ 0) every scattering and "
            "recombination-emission contribution to the phonon source carries a "
            "factor of the quasiparticle density ρf and vanishes, while pair "
            "breaking carries ρ(1−f) twice and survives: a_ph ≡ 0 and "
            "b_ph = −Γ_pb exactly. The phonon equation is then a "
            "constant-coefficient affine ODE per ω bin, which the engine "
            "integrates with the exact exponential map, and eliminating the "
            "source leaves a form in which τ_l appears twice — once as the rate "
            "it contributes and once as the surviving bath fraction "
            "n_∞/n_bath = τ_eff/τ_l, which is the escape term's own statement."
        ),
        headline_latex=(
            r"\frac{n_\infty(\omega)}{n_{\rm bath}(\omega)} = \frac{\tau_{\rm eff}}{\tau_l}, "
            r"\qquad \frac{1}{\tau_{\rm eff}} = \frac{1}{\tau_l} + \Gamma_{\rm pb}(\omega)"
        ),
        rel_tol=1e-10,
        convergence=(
            "Every number here is measured through benchmarks.evaluate() on the "
            "Curve this module returns, not on a private reimplementation of it. "
            "That was the audit's problem 5 on the original derivation: its "
            "refinement ladder described a different construction from the one "
            "the harness scored, and the two differed by 2.5×.\n"
            "Case: 1×1 geometry, 48 bins over [1, 4]Δ (128 ω bins, 95 live), "
            "dt = 0.03125 ns, 1.0 ns = 5.88 escape times, 33 frames × 95 series "
            "= 3135 points, 0.15 s. Measured max 3.439589e-12, rms 2.289430e-12, "
            "n_below_floor 0, against rel_tol 1e-10 — 29× headroom.\n"
            "THE RESIDUAL DOES NOT CONVERGE UNDER REFINEMENT AND MUST NOT. With "
            "f frozen the flow is constant-coefficient and the engine advances "
            "it with the exact map n·e^{Bh} + A·φ₁(Bh)h, whose steps compose "
            "exactly, so there is no discretisation error to remove. dt = 1.0 "
            "(each step crossing 5.88 escape times in one shot) / 0.25 / 0.125 / "
            "0.03125 / 0.0078125 ns gives 3.439292e-12 / 3.439162e-12 / "
            "3.439274e-12 / 3.439589e-12 / 3.439027e-12 — flat to 6 digits over "
            "a factor of 128 in step count. That is the sharp form of the same "
            "statement: any fixed-order integrator would be catastrophic taking "
            "one step across six relaxation times.\n"
            "Energy mesh, which changes the ω lattice entirely: NE = 32 / 48 / "
            "96 / 192 (63 / 95 / 191 / 383 live bins) gives 3.439416e-12 / "
            "3.439589e-12 / 3.467575e-12 / 3.467963e-12. Flat to 0.8%. Escape "
            "and pair breaking are both diagonal in ω, so the mesh is not a "
            "discretisation axis for either.\n"
            "WHAT THE FLOOR IS, MEASURED. It is the float64 representation of "
            "k_B in the bath target, amplified by d ln n_BE/d ln k_B = "
            "x e^x/(e^x − 1) ≈ x at the top of the grid. Scoring the identical "
            "simulated curve against a bath built from the engine's rounded "
            "literal 86.17333262145 instead of the SI-derived 86.17333262145179 "
            "drops the residual from 3.439589e-12 to 2.922176e-14, a factor "
            "117.7 — and ω_max/(k_B T_bath) = 165.8 while Δk_B/k_B = 2.0779e-14, "
            "whose product is 3.45e-12. The floor is therefore predictable: it "
            "is ω_max/(k_B T_bath) × 2.08e-14. Measured against that law: "
            "T_bath = 0.08 / 0.10 / 0.12 / 0.15 K gives 4.320724e-12 / "
            "3.439589e-12 / 2.870917e-12 / 2.303034e-12 (a 1/T law to 3 digits); "
            "a shorter grid ([1,3]Δ, NE = 32) gives 2.572276e-12; Δ₀ = 200 μeV "
            "gives 3.808509e-12.\n"
            "TOLERANCE. rel_tol = 1e-10 holds while ω_max/(k_B T_bath) < 4800, "
            "i.e. T_bath above ~0.0035 K on this grid, or a grid out to ω_max ≈ "
            "41 meV at 0.1 K. 1e-11 was rejected: it leaves 2.9× headroom on the "
            "shipped case and expires at T_bath ≈ 0.035 K or ω_max ≈ 4.1 meV — "
            "only 2.9× beyond this case in either direction, and it would then "
            "be reporting constant-representation noise as a physics failure.\n"
            "Invariance in the term's own parameter: τ_l = 0.02 / 0.17 / 2.0 ns "
            "gives 3.439027e-12 / 3.439589e-12 / 3.440446e-12 while the "
            "surviving fraction at the worst bin moves 0.964845 → 0.763639 → "
            "0.292326. Run length 8 ns (47 escape times): 3.439628e-12. Cells: "
            "1×1 / 1×3 / 2×3 all give 3.439589e-12 (95 / 285 / 570 series) with "
            "cell-to-cell spread exactly 0.000e+00.\n"
            "Snapshot cadence is NOT a hazard in this backend, unlike the 0-D "
            "transient the derivation was written against. T3SpatialBackend.run "
            "snapshots at step boundaries and records the time actually reached "
            "rather than the time asked for, so a cadence that is not a multiple "
            "of dt mislabels nothing: snapshot_interval = 1.6·dt gives 21 frames "
            "and 3.439564e-12. On the 0-D path — which interpolates f but takes "
            "n_ph from the end of the step — the same trick produced a false red "
            "of 1.4e-01, and the derivation carried a guard against it. No "
            "cadence guard is needed here and none is imposed.\n"
            "FALSIFICATION — the same simulated curve rescored against "
            "deliberately wrong closed forms, all through evaluate(): escape at "
            "2τ_l → 2.3013e-01; τ_l read as microseconds (×1000) → 3.6708e+00; "
            "τ_l 1% low → 2.3553e-03; bath target 1% too hot → 8.0633e-01. The "
            "nearest wrong model is 6.8e8 × the residual of the right one.\n"
            "DISCRIMINATION FLOOR, both ways. A relative error ε in τ_l reads "
            "2.1390e-11 at 1e-10, 6.8491e-11 at 3e-10, 1.1559e-10 at 5e-10 and "
            "2.3335e-10 at 1e-9, so τ_l is pinned to about 4.3e-10 relative. A "
            "relative error in the bath TEMPERATURE reads 2.0009e-11 at 1e-13, "
            "8.6346e-11 at 5e-13 and 1.6925e-10 at 1e-12, so the target is "
            "pinned to about 6e-13 in T — the amplification x ≈ 166 is what buys "
            "that."
        ),
        modes=("spatial_2d",),
        build=_build,
        caveat=(
            "IT IS NOT THE TERM'S OWN REDUCTION, AND THAT IS A SCHEMA GAP, NOT A "
            "CHOICE. The isolated case — both e-ph channels off, phonons started "
            "away from the bath — cannot be expressed: the engine seeds n_ph at "
            "the same Bose–Einstein bath the escape term relaxes toward, and "
            "Spatial2DSetup.initial prepares the QUASIPARTICLE occupation only. "
            "PhononSector needs an initial-condition field (a thermal-plus-excess "
            "seed would do) before the T1 case exists. Until then the escape term "
            "can only be watched competing with pair breaking, which is why this "
            "is T2 and why build() refuses the isolated case rather than passing "
            "it: measured, that case moves the phonons by 3.2e-16 with escape ON "
            "and 0 with it OFF.\n"
            "IT SAYS NOTHING ABOUT Γ_pb. The pair-breaking rate is read from the "
            "engine's own recombination kernel, so a wrong kernel moves both "
            "sides together and passes untouched. Everything belonging to escape "
            "— the rate 1/τ_l, its sign, its frequency independence, the "
            "Bose–Einstein target — is written here independently and is under "
            "test; nothing else in the phonon equation is.\n"
            "IT TESTS τ_l's PLUMBING, NOT ITS PHYSICS. It proves the number in "
            "phonons.tau_l_ns is used as a relaxation time in nanoseconds, with "
            "the right sign, identically in every bin, toward the bath at "
            "T_bath. It says nothing about whether τ_l ≈ 4d/(η s) "
            "(qpsim/physics/phonon_escape.py) is right for a given film, nor "
            "about the frequency-independent-τ_l modelling assumption — which "
            "this benchmark ASSUMES by verifying it.\n"
            "NO DISCRETISATION ERROR EXISTS TO CONVERGE, so the usual refinement "
            "evidence is absent by construction (see convergence). A green here "
            "is not evidence that the Strang splitting or ETD2 are accurate on "
            "any problem with non-constant coefficients — which is every "
            "coupled run.\n"
            "THE INITIAL PHONON POPULATION IS NOT CHECKED, it is assumed to be "
            "the bath: the closed form uses n(0) = n_bath rather than reading "
            "the run's first frame, because that identity is the only thing "
            "making the source eliminable. A bug in the SEED would therefore "
            "read as a trajectory error, which is a sharper coupling to the "
            "seed than the original derivation had, not a weaker one.\n"
            "SCOPE. One material (Al, Δ₀ = 180 μeV), one bath temperature, one "
            "uniform gap, pure BCS only (spatial collisions reject dynes_gamma > "
            "0), 0-D in space. The 0-D transient's own copy of this update "
            "(qpsim/backends/t3_diffusion.py::advance_phonons) and the "
            "steady-state balance in qpsim/phonon_models/ph0_local.py are "
            "separate implementations of the same term and are untested here — "
            "and run_transient_0d publishes no phonon arrays at all, which is "
            "why this benchmark lives in spatial_2d.\n"
            "THE POPULATIONS ARE PHYSICALLY TINY. n_bath runs 1.9e-19 down to "
            "9.9e-73 over the scored bins, because pair-breaking phonons at "
            "100 mK are exponentially rare — which is exactly why f stays frozen "
            "and the reduction closes. The ODE is linear, so the check is "
            "scale-free and float64 carries full relative precision throughout; "
            "but nobody should read this curve as a statement about a "
            "device-relevant phonon population."
        ),
        activity=(
            "Measured on the shipped case. The escape term removes up to 23.6% "
            "of the bath population: max |n/n_bath − 1| = 0.236361, and the "
            "surviving fraction n_∞/n_bath = τ_eff/τ_l spans 0.763639 to "
            "0.990490 across the 95 scored bins. Per-series motion n(0)/n(T) "
            "runs 1.0096 (min) / 1.1462 (median) / 1.3095 (max); 2 of the 95 "
            "series move by less than 2% and 1 by less than 1% — those are the "
            "near-threshold bins where Γ_pb·τ_l = 9.6e-03, and they are reported "
            "rather than dropped.\n"
            "A/B WITH THE TERM REMOVED: phonons.mode = 'dynamic_closed' is the "
            "engine's τ_l = 0.0 no-substrate sentinel, i.e. escape deleted from "
            "the equation. The same case then scores 7.8820e-01 — a clean "
            "numeric FAIL at 7.9e9 × the tolerance and 11.4 orders above the "
            "residual with the term on — because the population decays toward 0 "
            "at Γ_pb instead of saturating at n_bath·τ_eff/τ_l. The benchmark "
            "does not guard on phonons.mode, precisely so that switching the "
            "term off produces a failure rather than a skipped check.\n"
            "THE VACUOUS CONFIGURATION IS REFUSED, NOT PASSED. With "
            "collisions.recombination = False the phonon coefficients are "
            "identically zero, the seeded population is already the term's fixed "
            "point, and the escape-on and escape-off runs move n_ph by 3.1888e-16 "
            "and exactly 0 — a green verdict there would certify nothing. build() "
            "raises on an empty live set, and again on a total departure below "
            "1e-3, so the two independent ways of reaching that state are both "
            "closed. The frozen-f bound is the third guard and it fires too: "
            "T_bath = 0.2 K and 0.3 K reach max |f| = 2.320e-11 and 3.601e-08 "
            "and are refused rather than scored (0.3 K would otherwise have read "
            "3.1487e-09, a red caused by the reduction lapsing, not by the "
            "engine).\n"
            "THE REDUCTION'S HYPOTHESIS IS ALSO MEASURED, NOT ASSUMED: max |f| "
            "over all 33 frames is exactly 0.0 (df/dt at f = 0 is 8.551e-21, the "
            "thermal pair-breaking flux, and the ETD2 balance projection "
            "rescales occupations so an empty state stays exactly empty), and "
            "build() refuses any run above 1e-11."
        ),
        extra={"case_overrides": CASE_OVERRIDES},
    )
)
