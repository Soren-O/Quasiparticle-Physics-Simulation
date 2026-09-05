r"""Pair-breaking photon drive: the ``f → 0`` pair-generation source spectrum.

The reduction
-------------
Switch both electron-phonon collision channels off, pin the phonons at the
bath, freeze Δ, and leave the pair-breaking photon drive on. The drive's
collision integral is then the whole right-hand side, and it splits into three
blocks (``qpsim/collisions/pair_breaking_photon.py``): K⁺ scattering, K⁻ pair
generation and K⁻ pair recombination. Take ``f → 0`` and count factors of
``f``: both scattering gains carry ``f`` at the partner index, the entire loss
rate multiplies ``f_i``, and the pair-recombination loss carries ``f_j`` on top
of that. Exactly one term survives with no factor of ``f`` anywhere,

    df_i/dt|_{f=0} = c_phot_PB · n̄_PB · ρ̄_{j(i)} · K⁻_{i,j(i)} ≡ S_i,
    E_{j(i)} = ω_PB − E_i,

so ``f_i(t) = f_i(0) + S_i t + O(f²)`` and the measured per-bin slope is the
closed form itself.

The mode
--------
The case runs on the spatial core (``kinetics``) with a 1×1 mask.
Dimensionality is read off the mask extent rather than set by a flag, so one
cell *is* the 0-D reduction — the run's summary reports ``dimensionality = 0``
— and the single-cell transport operator is identically zero, which is why
``material.D_0`` is set to 0 for intent rather than for effect. The closed form
is spatially uniform, so ``build`` scores every cell the mask carries; on the
shipped mask that is the one cell.

What is checked
---------------
The running source rate ``(f_i(t) − f_i(0)) / t`` over all 405 energy bins at
four snapshot times. One object, two statements: across E it is the reflected
BCS density of states times the pair coherence factor — 1.0471e-07 /ns at the
gap-edge bin, whose partner sits at the top of the pair band, rising 9.39× to
9.8313e-07 /ns at E = 1018 μeV, whose partner is the singular gap-edge cell,
and exactly zero on the 195 bins whose partner ω_PB − E lies below Δ. Across t
it is constant, which is the straight-line claim. Contracting the same S_i
against the BCS cell weights gives the x_qp(t) line, reported in the curve's
note as a cross-check.

Independence
------------
The analytic side is built from five numbers — Δ₀, the grid triple, ω_PB, n̄_PB
and c_phot_PB — plus two elementary BCS primitives. It reads no engine array:
not ``SpectralContext.K_minus``, not ``cell_density``, not ``cell_weights``,
not the collision module. ``E_bins`` is touched only to assert that the two
grids are the same grid (they agree to 0.0 exactly here). Subtracting ``f(0)``
also removes the initial condition from the prediction, so not even the
Fermi-Dirac start is an ingredient.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# The catalogue case. dE = (10 − 1)·180/405 = 4 μeV exactly, so ω_PB = 300·dE
# and (ω_PB − 2E_0)/dE = 209 are both integers at machine precision rather than
# through the engine's 1%-of-a-bin snap; at this ω_PB the K⁺ scattering block is
# also live for 210 of the 405 bins, so the case does not quietly switch off two
# thirds of the term under test.
#
# n_bar_PB = 125 rather than the 1000 this benchmark was first measured at. The
# residual here is the O(f) linearisation departure and nothing else, and f is
# proportional to n̄_PB, so the weaker drive buys an 8× tighter tolerance at
# identical runtime (4.5 s) while x_qp still climbs 3.9e4×. The cost is nil; the
# stronger drive was simply 8× less discriminating.
#
# The geometry is the 0-D reduction expressed as a mask, not a separate mode:
#   rows = cols = 1.  One cell, so `summary["dimensionality"]` is 0 and the
#     5-point Laplacian assembles to the identically zero operator. The
#     boundary condition is then unreachable; `reflective` is stated anyway so
#     the case does not depend on a schema default for something a reader would
#     look for.
#   material.D_0 = 0.  Redundant with one cell — there is nowhere for a flux to
#     go — and set for intent: this case measures a source term, not transport.
#   stop_tol = 0.  The residual here GROWS (the population grows without bound),
#     so no positive stop_tol truncates this particular run; 0 states that the
#     run is meant to reach `max_time` rather than to find a steady state, and
#     the run then reports "did not reach stop_tol", which is the truth.
#   snapshot_interval = 0.2.  Without it `kinetics` records no frames at all
#     and there is no slope to measure — only the endpoint, which cannot
#     distinguish "linear in t" from "arrived at the right place along a curve".
#   initial.kind = "thermal".  States the Fermi-Dirac start explicitly.
#     `build` subtracts f(0), so the start is not an ingredient of the
#     prediction either way.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "kinetics",
    "material.D_0": 0.0,
    "T_bath": 0.1,
    "grid.min_factor": 1.0,
    "grid.max_factor": 10.0,
    "grid.num_bins": 405,
    "geometry.kind": "rectangle",
    "geometry.rows": 1,
    "geometry.cols": 1,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "phonons.mode": "thermal_bath",
    "collisions.scattering": False,
    "collisions.recombination": False,
    "gap_regions.kind": "uniform",
    "self_consistent_gap": False,
    "initial.kind": "thermal",
    "injection.enabled": False,
    "drives": [],
    "subgap_drive.enabled": False,
    "pb_drive.enabled": True,
    "pb_drive.omega_PB": 1200.0,
    "pb_drive.n_bar_PB": 125.0,
    "pb_drive.c_phot_PB": 1e-9,
    "dt": 0.1,
    "max_time": 12.0,
    "stop_tol": 0.0,
    "snapshot_interval": 0.2,
}

# Fractions of the run at which the running slope is read. Four times rather
# than one because the endpoint alone cannot distinguish "linear in t" from
# "arrived at the right place along a curve"; more than four only repeats the
# same statement, and every series carries the full 405-bin spectrum.
_SERIES_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# ω_PB must land on the cell lattice for the reflection partner to be an exact
# index reflection. The engine tolerates 1% of a bin and solves the snapped
# frequency; this closed form is written at the exact one, so an inexact case is
# refused rather than compared against a prediction for a different ω_PB.
_LATTICE_TOL = 1e-12


def _bcs_cell_measures(
    gap: float, e_min: float, e_max: float, n_bins: int
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    r"""Exact per-cell BCS measures on a uniform cell-centred grid.

        w_i = ∫_cell E dE / sqrt(E² − Δ²) = [sqrt(E² − Δ²)]
        a_i = ∫_cell Δ dE / sqrt(E² − Δ²) = Δ [arccosh(E/Δ)]

    Both primitives are elementary, which is why S_i below is a closed form and
    not a quadrature. The square root is evaluated factored as
    sqrt((E−Δ)(E+Δ)) because the gap-edge cell otherwise loses most of its
    significant digits to cancellation.
    """
    dE = (e_max - e_min) / n_bins
    E = e_min + (np.arange(n_bins, dtype=float) + 0.5) * dE
    edges = e_min + np.arange(n_bins + 1, dtype=float) * dE
    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], lo)
    w = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0)) - np.sqrt(
        np.maximum((lo - gap) * (lo + gap), 0.0)
    )
    a = gap * (
        np.arccosh(np.maximum(hi / gap, 1.0)) - np.arccosh(np.maximum(lo / gap, 1.0))
    )
    return E, dE, w, a


def _source_spectrum(
    setup: Any,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form per-bin pair-generation source ``S_i`` (1/ns).

    Returns ``(E, dE, w, S, live)``: cell centres, spacing, BCS cell weights,
    the source, and the mask of bins whose reflection partner is a represented
    above-gap cell. ``S`` is zero off that mask, which is a prediction and not a
    gap in coverage — those partners sit below Δ, where the pair channel does
    not exist.
    """
    gap = float(setup.material.Delta_0)
    grid = setup.grid
    n_bins = int(grid.num_bins)
    E, dE, w, a = _bcs_cell_measures(
        gap, float(grid.min_factor) * gap, float(grid.max_factor) * gap, n_bins
    )
    supported = w > 0.0

    pb = setup.pb_drive
    amplitude = float(pb.c_phot_PB) * float(pb.n_bar_PB)
    if amplitude == 0.0:
        # n_bar_PB = 0.0 is the shipped schema default, and at that default an
        # "enabled" drive applies nothing. The predicted curve would then be
        # identically zero and the comparison would report a pass against a
        # simulation that also did nothing — the one failure mode this whole
        # exercise exists to avoid. Refuse instead.
        raise ValueError(
            "pb-drive predicts an identically zero source: "
            f"c_phot_PB={pb.c_phot_PB:g} * n_bar_PB={pb.n_bar_PB:g} = 0. "
            "That comparison would pass without measuring anything; set a "
            "non-zero n_bar_PB (the schema default 0.0 is inert)."
        )

    omega_nominal = float(pb.omega_PB)
    m = round(omega_nominal / dE)
    omega = m * dE
    if abs(omega_nominal - omega) > _LATTICE_TOL * dE:
        raise ValueError(
            f"omega_PB={omega_nominal:g} μeV is {abs(omega_nominal - omega) / dE:.3g} "
            f"bins off the grid lattice (dE={dE:g} μeV). The engine would snap it "
            "and solve a different photon energy than this closed form states."
        )
    if omega <= 2.0 * gap:
        raise ValueError(
            f"pair channel closed: omega_PB={omega:g} μeV is not above "
            f"2Δ={2.0 * gap:g} μeV, so there is no generation source to check."
        )
    steps = float((omega - 2.0 * E[0]) / dE)
    partner_index = round(steps)
    if abs(steps - partner_index) > _LATTICE_TOL:
        raise ValueError(
            "reflection partners are not on the cell lattice: "
            f"(omega_PB - 2*E_min)/dE = {steps!r} is not an integer."
        )

    rho_bar = w / dE
    ratio = np.zeros_like(E)
    ratio[supported] = a[supported] / w[supported]
    # The cell ratio is a Cauchy-Schwarz bound below 1; clipping guards the
    # gap-edge cell against roundoff pushing K⁻ negative.
    ratio = np.clip(ratio, 0.0, 1.0)

    j = partner_index - np.arange(n_bins)  # E_j = ω_PB − E_i, exactly, on lattice
    j_clipped = np.clip(j, 0, n_bins - 1)
    live = supported & (j >= 0) & (j < n_bins) & supported[j_clipped]
    partner = j_clipped[live]
    S = np.zeros(n_bins)
    S[live] = (
        amplitude
        * rho_bar[partner]
        * np.maximum(1.0 - ratio[live] * ratio[partner], 0.0)
    )
    return E, dE, w, S, live


def _refuse_unless_reduced(setup: Any) -> None:
    """Raise unless the run is the reduction this closed form describes.

    Only the knobs that can silently invalidate the closed form are checked
    here: a second source means S_i is not the whole f-independent right-hand
    side, and a gap that is not Δ₀ everywhere makes the cell measures wrong.

    Note the deliberate omissions. ``pb_drive.enabled = False`` is NOT refused:
    the closed form still applies and is violated, and that null run is this
    benchmark's falsification test, so it has to produce a numeric FAIL rather
    than an error. Neither collision channel is refused either — both carry an
    explicit factor of f and are therefore inside the O(f) residual rather than
    outside the reduction (which the caveat says, and which is exactly why this
    benchmark cannot see them).
    """
    if setup.injection.enabled:
        raise ValueError(
            "injection is enabled, so the pair-breaking drive is not the whole "
            "right-hand side: the measured slope would be S_i plus the injected "
            "spectrum, and the excess would be scored as a kernel error."
        )
    if any(drive.enabled for drive in setup.drives):
        raise ValueError(
            "a prescribed drive is enabled, so the measured slope is S_i plus "
            "that drive rather than S_i. Switch it off, or write a closed form "
            "for the sum."
        )
    if setup.gap_regions.kind != "uniform":
        raise ValueError(
            f"gap_regions.kind={setup.gap_regions.kind!r} gives cells different "
            "local gaps; this closed form is written at the material Δ₀ "
            "everywhere, and both the BCS cell measures and the 2Δ pair "
            "threshold move with the local gap."
        )
    if setup.self_consistent_gap:
        raise ValueError(
            "a self-consistent gap moves Δ during the run, so the cell measures "
            "and the reflection partner are not constant and the growth is not "
            "linear in t. This reduction freezes Δ."
        )


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    _refuse_unless_reduced(setup)
    E_ana, dE, w, S, live = _source_spectrum(setup)

    E = np.asarray(arrays["E_bins"], dtype=float)
    if E.shape != E_ana.shape:
        raise ValueError(
            "the analytic grid is not the engine's grid: shapes "
            f"{E_ana.shape} vs {E.shape}."
        )
    grid_gap = float(np.max(np.abs(E - E_ana)))
    if grid_gap > _LATTICE_TOL * dE:
        raise ValueError(
            f"the analytic grid is not the engine's grid: max |ΔE| = {grid_gap:.3g} μeV."
        )

    if "snap_t_ns" not in arrays or "snap_f" not in arrays:
        raise ValueError(
            "this benchmark needs the recorded field: set snapshot_interval so "
            "the run emits frames. Without them the only observable is the "
            "endpoint, which cannot distinguish a rate from a final value."
        )
    t = np.asarray(arrays["snap_t_ns"], dtype=float)
    f = np.asarray(arrays["snap_f"], dtype=float)  # (frames, NE, Ncells)
    if f.ndim != 3 or f.shape[0] != t.size or f.shape[1] != E.size:
        raise ValueError(
            f"snap_f has shape {f.shape}; expected (frames, NE, Ncells) = "
            f"({t.size}, {E.size}, ...)."
        )
    n_cells = int(f.shape[2])
    if t.size < 2:
        raise ValueError(
            f"a slope needs at least two snapshots; this run recorded {t.size}. "
            "Set snapshot_interval."
        )
    last = t.size - 1
    idx = sorted({max(1, round(q * last)) for q in _SERIES_FRACTIONS})
    if any(t[k] <= t[0] for k in idx):
        raise ValueError("snapshot times are not increasing; cannot form a slope.")

    # One series per (time, cell). The source is spatially uniform — the drive
    # is, the gap is, and the thermal start is — so every cell is predicted to
    # carry the same S_i, and on a mask wider than one cell that equality is
    # part of what is being checked rather than an assumption. With the shipped
    # 1×1 mask this collapses to four series, one per read-off frame, at this
    # run's own snapshot times.
    y_sim = np.array([
        (f[k, :, c] - f[0, :, c]) / (t[k] - t[0])
        for k in idx
        for c in range(n_cells)
    ])
    y_ana = np.tile(S, (len(idx) * n_cells, 1))

    # Cross-check, not a second verdict: the same S_i contracted against the BCS
    # cell weights is the slope of x_qp(t), which is what the run reports as an
    # observable. It uses no new physics, so it earns a note and not a curve.
    # The mean over cells is the right reduction because the prediction is
    # uniform; on the 1×1 mask it is the single cell's own value.
    x_qp = np.asarray(arrays["obs_x_qp_mean"], dtype=float)
    predicted = float(np.sum(S * w) / float(setup.material.Delta_0)) * (t[-1] - t[0])
    observed = float(x_qp[-1] - x_qp[0])
    geometry = (
        f"{summary.get('rows', '?')}×{summary.get('cols', '?')} mask, "
        f"{n_cells} cell{'s' if n_cells != 1 else ''}, "
        f"dimensionality {summary.get('dimensionality', '?')}"
    )
    note = (
        f"{geometry}. x_qp cross-check: Δx_qp over {t[-1] - t[0]:g} ns measured "
        f"{observed:.6e} against Σ_i w_i S_i/Δ₀ · t = {predicted:.6e} "
        f"({abs(observed / predicted - 1.0):.2e} relative). "
        f"{int(np.count_nonzero(live))} of {S.size} bins carry a source; the "
        f"other {int(np.count_nonzero(~live))} are predicted to be exactly zero "
        "because their partner ω_PB − E lies below Δ."
    )

    return Curve(
        x=E,
        y_sim=y_sim,
        y_analytic=y_ana,
        x_label="E (μeV)",
        y_label="(f(E,t) − f(E,0)) / t   (1/ns)",
        series_labels=tuple(
            f"t = {t[k]:.3g} ns" + (f", cell {c}" if n_cells > 1 else "")
            for k in idx
            for c in range(n_cells)
        ),
        # "scale" rather than the pointwise default because the closed form is
        # EXACTLY ZERO on 195 of the 405 bins. Pointwise relative error divides
        # by |analytic| and the framework floors that at 1e-9 of the peak, so
        # those bins would be dropped from the verdict entirely — the score
        # reports n_below_floor = 780, which is exactly those 195 bins at each
        # of the 4 times — and they are precisely where a term wired to the
        # wrong partner would show up. Normalising to the peak of the analytic
        # curve gates the whole spectrum; the predicted-zero half currently
        # sits at 8.6e-7 of peak, well inside the tolerance.
        metric="scale",
        note=note,
    )


register(
    Benchmark(
        name="pb-drive",
        title="Pair-breaking drive against its source spectrum",
        tier="T1",
        formula_latex=(
            r"\frac{f_i(t)-f_i(0)}{t}=S_i+\mathcal{O}(f),\quad "
            r"S_i=c^{PB}_{\mathrm{phot}}\,\bar n_{PB}\,\bar\rho_{j}\,K^{-}_{ij},\quad "
            r"E_j=\omega_{PB}-E_i,\quad "
            r"\bar\rho_i=\frac{1}{\delta E}\int_{\mathrm{cell}}\frac{E\,dE}"
            r"{\sqrt{E^2-\Delta^2}},\quad "
            r"K^{-}_{ij}=\max\!\left(1-\frac{a_i}{w_i}\frac{a_j}{w_j},\,0\right),\quad "
            r"a_i=\int_{\mathrm{cell}}\frac{\Delta\,dE}{\sqrt{E^2-\Delta^2}}"
        ),
        reason=(
            "At f = 0 every block of the pair-breaking photon integral except pair "
            "generation carries an explicit factor of f — both scattering gains at "
            "the partner index, the whole loss rate, and the pair-recombination loss "
            "on top of it — so df/dt at zero occupation is the single f-independent "
            "term S_i. With both collision channels off, the phonons pinned and Δ "
            "frozen, S_i is the entire right-hand side, so f is linear in t and the "
            "per-bin slope is the closed form. The reflection partner is an exact "
            "lattice reflection here, j(i) = 209 − i rather than a rounding, and the "
            "cell measures are elementary integrals, so S_i is written rather than "
            "quadratured."
        ),
        headline_latex=(
            r"f_i(t) \;=\; f_i(0) \;+\; S_i\,t \;+\; \mathcal{O}(f^2)"
        ),
        rel_tol=2e-5,
        convergence=(
            "Case: 405 bins over [1, 10]Δ (δE = 4 μeV exactly), a 1×1 mask, "
            "dt = 0.1 ns, 12 ns, 4 snapshot times × 405 bins = 1620 points, "
            "1.0 s. Measured worst error 6.528e-06 against rel_tol 2e-05 "
            "(3.1× headroom).\n"
            "The small parameter is the OCCUPATION, not the mesh or the step, "
            "because the closed form is the f → 0 limit. Halving the drive: "
            "n_bar_PB = 1000/500/250/125/62.5 gives 5.2215e-05 / 2.6108e-05 / "
            "1.3055e-05 / 6.5277e-06 / 3.2642e-06, order 1.00 at every halving. "
            "Doubling the run time doubles it too (3.264e-06 at 6 ns, 6.528e-06 at "
            "12 ns, 1.305e-05 at 24 ns): both knobs move f, and the residual is O(f).\n"
            "Refining dt does not converge, and should not: ETD2 is exact for "
            "constant gain and loss, which is what this reduction is at leading "
            "order. dt = 0.4/0.2/0.1/0.05 ns all read 6.527687e-06 — identical to "
            "seven figures — and move x_qp(12 ns) by 3.3e-15 relative over that "
            "factor of 8. This benchmark therefore says nothing about "
            "time-integration order.\n"
            "Refining the ENERGY mesh makes it WORSE, reported rather than hidden: "
            "the gap-edge cell density sqrt(2ΔδE)/δE grows as δE^(−1/2), so the peak "
            "source and the occupation it drives grow with refinement and walk out "
            "of the linear limit. 810 bins measures 8.943e-06 — a factor 1.37 "
            "against the predicted sqrt(2) = 1.41 — still inside 2e-05 but with 2.2× "
            "headroom instead of 3.1×. Re-measure if num_bins, n_bar_PB, c_phot_PB "
            "or max_time changes.\n"
            "Independence of the analytic cell measure: cell-averaged S converges to "
            "the continuum BCS point value (ω−E)/sqrt((ω−E)²−Δ²)·(1 − Δ²/(E(ω−E))) "
            "at second order in δE — 2.101e-07, 5.142e-08, 1.272e-08, 3.162e-09 at "
            "δE = 4/2/1/0.5 μeV, mid-band at E = 600 μeV. The cell weights really "
            "are the BCS DOS and not a re-coding of engine internals.\n"
            "Snapshot cadence: the spatial core records the step boundary it "
            "actually reached, and its cadence counter drifts by a ulp, so two "
            "of the four read-off frames land at 6.1 and 9.1 ns instead of 6.0 "
            "and 9.0. Those two series are read 1.7% later, and their residual "
            "— which is O(f) and therefore proportional to t — is 1.7% larger, "
            "which puts the rms at 1.4432e-06. The scored maximum is the 12 ns "
            "series, which is read at exactly its requested time, so the score "
            "itself carries no cadence offset."
        ),
        modes=("kinetics",),
        build=_build,
        caveat=(
            "It validates the K⁻ pair-GENERATION block at leading order, and only "
            "that block. The K⁺ scattering block and the K⁻ recombination loss both "
            "carry an explicit factor of f, so from a near-empty start they enter "
            "only the residual: removing the final-state blocking factor (1 − f_j) "
            "from the engine's generation term does NOT break this benchmark — the "
            "mutant reads 5.899e-06 against a 6.528e-06 baseline, so it improves "
            "apparent agreement. Making K⁺ leading-order needs a non-thermal start "
            "and is a different benchmark.\n"
            "The discrimination floor is finite and measured. A systematic relative "
            "error ε in the generation coefficient (c_phot_PB perturbed in the "
            "engine alone) reads ε − 1.6e-06 on this curve, because the O(f) deficit "
            "and the excess partly cancel; ε = 2e-05 still passes and ε = 5e-05 "
            "fails at 4.837e-05. Errors below ~2.2e-05 in that coefficient are not "
            "resolved here.\n"
            "One shared assumption, and it is a convention rather than physics: the "
            "analytic K⁻ is built from the cell ratio a_i/w_i, as the engine builds "
            "it, rather than from the point value 1 − Δ²/(E_i E_j). That does not "
            "make the check blind to it — patching the engine to the point-value K⁻ "
            "makes this curve read 7.807e-04, 39× the tolerance — but if the "
            "convention itself is judged wrong, both sides move together. Over the "
            "210 live pair elements the two conventions differ by at most 7.824e-04, "
            "worst at the gap-edge bin. The large gap sometimes quoted for this "
            "convention — 0.014630 against 0.021857, 49% — is the DIAGONAL element "
            "K⁻(E₀,E₀) of the gap-edge cell, which a reflection partner never is; "
            "it is not the size of the effect here.\n"
            "It is a maximal reduction, not physics: both collision channels off, "
            "phonons pinned at the bath, Δ frozen, no transport. There is no steady "
            "state and the population grows without bound; that is the case, not a "
            "defect. The case also sits at exact grid commensurability, so the "
            "1%-of-a-bin snapping path, the 1e-9 partner-lattice rejection, the "
            "threshold-crossing guard, the off-grid-partner guards and the K⁺ "
            "up-partner truncation at i + m ≥ NE are all untested by it. One "
            "material (Al, Δ₀ = 180 μeV), one bath temperature, one photon energy, "
            "pure BCS only — the term rejects dynes_gamma > 0 outright.\n"
            "SPATIAL SCOPE. The shipped mask is one cell, so the case is the 0-D "
            "reduction of the spatial core: the transport operator is identically "
            "zero and nothing here exercises the Laplacian, a boundary condition, "
            "an interface or a gap step. ``build`` scores whatever cells the mask "
            "carries, which was checked rather than left as an intention — on a "
            "2×3 mask with D_0 = 0 the same case scores 6.5276867346370295e-06 "
            "over 24 series, the identical float, so all six cells carry the same "
            "source bit-for-bit. That widening is nearly free of information, "
            "though, and the reason is worth stating: the drive, the gap and the "
            "thermal start are all uniform, so the six cells are six copies of one "
            "experiment. Turning transport back on (D_0 = 60 μm²/ns, same mask) "
            "moves the score only to 6.5276867372e-06 — the tenth digit — because "
            "a uniform field has no gradient for the Laplacian to act on. This "
            "benchmark cannot be made to test transport by making the mask bigger; "
            "that needs a source that varies in space, which is the diffusion "
            "benchmark's job."
        ),
        activity=(
            "Measured on the shipped case, not asserted. Over 12 ns x_qp goes "
            "2.224322e-10 → 8.599905e-06 (3.87e+04×, 4.59 decades) and the peak "
            "occupation 6.72e-10 → 1.180e-05; 210 of the 405 bins acquire a source, "
            "spanning E = 182 to 1018 μeV, and the source itself rises 9.39× across "
            "them.\n"
            "Switching the drive off is not a small change to this curve: "
            "pb_drive.enabled = False gives error exactly 1.000e+00 against a 2e-05 "
            "tolerance, and max |f(T) − f(0)| is exactly 0.000e+00 — with both "
            "collision channels off, the phonons pinned and a one-cell mask whose "
            "transport operator is identically zero, nothing else in the engine "
            "touches f, so the reduction is exact rather than approximate.\n"
            "The trap this term ships with is n_bar_PB = 0.0, which is the schema "
            "default: an enabled drive at that default is inert (x_qp relative move "
            "0.000e+00, max |f(T) − f(0)| = 2.896e-77). That residue is the "
            "spontaneous (n̄+1) branch of the number-conserving K⁺ block and not "
            "pair recombination — it sits at E = 1382 μeV, index 300, a bin whose "
            "pair partner index would be 209 − 300 = −91 and which therefore has no "
            "pair channel at all, while its K⁺ down-partner is bin 0. build() "
            "refuses to score a case whose predicted source is identically zero "
            "rather than reporting the vacuous zero-against-zero pass."
        ),
        extra={"case_overrides": CASE_OVERRIDES},
    )
)
