"""External injection: a prescribed source, integrated exactly.

The injection term is the one place in the kinetic equation where the
right-hand side does not depend on ``f``. Switch transport off (``D_0 = 0``),
switch both collision channels off, leave the photon drives off, and the whole
equation is

    df(E_i, c)/dt = S(E_i, c),    S = R g(E_i) 1[c in F],

with ``g`` the peak-1 Gaussian line and ``F`` the injected cells. Nothing on
the right depends on the state, so ``f`` is exactly linear in time and the
closed form is not an approximation to be converged towards -- it is what the
stepper computes. That is the value of this benchmark: it measures the source's
absolute amplitude, its units (1/ns), its line centre and width, and which
cells it lands in, with no discretisation error in the way.

What is checked is the occupation itself, one series per (energy bin, cell)
probe against the recorded snapshot times, rather than the reduced ``x_qp``.
Comparing ``f`` keeps the BCS cell measure out of the closed form entirely, so
the analytic side needs only the schema's grid formula, a Fermi-Dirac seed and
the Gaussian -- and the check is about the source term rather than about the
observable's quadrature. The price is stated in the caveat: this says nothing
about the ``x_qp`` weights.

Derived and numerically verified 2026-08-13/14 against the ``spatial_2d``
executor at web-API level; the adversarial audit of the first derivation is
applied here (tolerance tightened to 1e-12, the "2-D has no snapshot cadence"
claim withdrawn -- it does, and this case uses it -- the unverifiable
provenance citation dropped, and the analytic side no longer reads
``injection.enabled``, which is what made the predecessor agree with itself
when the term was switched off).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# k_B/e = 8.617333262145e-5 V/K is exact under the SI definitions of k_B and e,
# so this is a defined constant rather than a measurement. Written out instead
# of imported from qpsim.constants to keep the prediction free of engine
# imports; the two agree, and a disagreement would be a real finding.
_KB_UEV_PER_K = 86.17333262145

# Bins kept in the line probe, in units of the line width. Beyond 3 sigma the
# increment R*g*t drops below the thermal background and the series stops
# saying anything about the line.
_LINE_HALF_WIDTH_SIGMA = 3.0

# The case this closed form describes. Every entry is load-bearing:
#
#   D_0 = 0            the transport off-switch (flux coefficient is D_0*N_1**q,
#                      so every face weight is exactly zero and the CN step is
#                      the identity). This is the reduction.
#   collisions off     scattering and recombination are the only other terms in
#                      the local right-hand side; a disabled channel is a kernel
#                      that is never built, so it contributes exactly zero.
#   phonons pinned     thermal_bath, so there is no phonon equation to feed back.
#   initial thermal    the closed form's t = 0 value IS the thermal seed. Initial
#                      conditions are reachable from the schema now, and any
#                      other seed changes the constant term.
#   drives []          the general DriveSpec path would add a second source.
#   T_bath = 0.3 K     a measurement choice, not a physics one -- see `activity`.
#   3 x 5, odd extents so 'centre_cell' is a cell rather than a four-way tie.
#   stop_tol = 0       nothing removes quasiparticles, so there is no steady
#                      state to stop at; let the clock run out instead.
#   snapshot_interval  a multiple of dt, so frames land exactly on the cadence.
#
# The photon drives (subgap_drive, pb_drive) are off by default; enabling
# either adds a gain term and breaks the closed form.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material": {"D_0": 0.0},
    "T_bath": 0.3,
    "grid": {"min_factor": 1.0, "max_factor": 3.0, "num_bins": 64},
    "geometry": {"kind": "rectangle", "rows": 3, "cols": 5, "mesh_size_um": 4.0},
    "collisions": {"scattering": False, "recombination": False},
    "phonons": {"mode": "thermal_bath"},
    "self_consistent_gap": False,
    "initial": {"kind": "thermal"},
    "drives": [],
    "injection": {
        "enabled": True,
        "center_over_delta": 2.0,
        "sigma_over_delta": 0.1,
        "rate_per_ns": 2e-5,
        "where": "centre_cell",
    },
    "dt": 2.0,
    "max_time": 200.0,
    "stop_tol": 0.0,
    "snapshot_interval": 10.0,
}


def _energy_grid(setup: Any) -> np.ndarray:
    """The schema's uniform cell-centred energy grid (μeV), from the setup.

    Rebuilt from ``grid.min_factor``/``max_factor``/``num_bins`` rather than
    read from ``arrays["E_bins"]``: a grid taken from the payload would agree
    with the engine by construction, and where the line sits on the grid is
    part of what is being checked.
    """
    gap = setup.material.Delta_0
    lo = setup.grid.min_factor * gap
    hi = setup.grid.max_factor * gap
    n = int(setup.grid.num_bins)
    dE = (hi - lo) / n
    return lo + dE * (np.arange(n, dtype=float) + 0.5)


def _thermal_occupation(setup: Any, energies: np.ndarray) -> np.ndarray:
    """Fermi-Dirac seed at the bath temperature — the closed form's t = 0."""
    return 1.0 / (np.exp(energies / (_KB_UEV_PER_K * setup.T_bath)) + 1.0)


def _line_shape(setup: Any, energies: np.ndarray) -> np.ndarray:
    """Injection line ``g(E)``: Gaussian, peak exactly 1, dimensionless.

    Peak-normalised, not area-normalised. That is the shipped convention and
    it has a consequence worth knowing (see the caveat): ``rate_per_ns`` is a
    peak spectral rate, so the total injected current scales with the width.
    """
    gap = setup.material.Delta_0
    centre = setup.injection.center_over_delta * gap
    sigma = setup.injection.sigma_over_delta * gap
    return np.exp(-0.5 * ((energies - centre) / sigma) ** 2)


def _footprint(setup: Any) -> np.ndarray:
    """Boolean mask of injected cells, in the solver's row-major cell order.

    Deliberately does NOT consult ``injection.enabled``. A benchmark asserts
    something about a case, and the case in ``CASE_OVERRIDES`` injects; an
    analytic side that read the switch would flatten alongside the engine when
    the term was disabled and report agreement, which is how a check that
    measures nothing passes. With the switch off this prediction is simply
    wrong about the run, and the verdict says so (measured: 1.0e+12 × tol).
    """
    source = setup.geometry
    if source.kind != "rectangle":
        raise ValueError(
            "The injection benchmark derives the injected cells from the "
            "setup's own extent, so it covers the rectangle geometry only; a "
            "rasterised layout's cell order cannot be reconstructed without "
            "reading the engine's mask."
        )
    n_rows, n_cols = int(source.rows), int(source.cols)
    rows, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
    mask = np.zeros(n_rows * n_cols, dtype=bool)
    where = setup.injection.where
    if where == "uniform":
        mask[:] = True
    elif where == "left_edge":
        mask[cols == cols.min()] = True
    else:  # centre_cell: the cell nearest the mask's bounding-box centre
        centre_row = 0.5 * (n_rows - 1)
        centre_col = 0.5 * (n_cols - 1)
        nearest = int(
            np.argmin((rows - centre_row) ** 2 + (cols - centre_col) ** 2)
        )
        mask[nearest] = True
    return mask


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    """f(E, cell) against time at a probe set spanning energy and space.

    One curve, three axes. The series index carries energy (every bin across
    the line, in the injected cell) and space (the line-centre bin in every
    un-injected cell); the abscissa carries time. The audit of the first
    derivation showed why that matters: the device-mean-vs-time curve is
    invariant under permuting the injected cells, so a source wired to the
    wrong cell passed it unchanged.
    """
    if "snap_f" not in arrays or "snap_t_ns" not in arrays:
        raise ValueError(
            "This benchmark reads the recorded frames, and this run kept only "
            "its endpoint. Set snapshot_interval on the setup."
        )
    times = np.asarray(arrays["snap_t_ns"], dtype=float)
    recorded = np.asarray(arrays["snap_f"], dtype=float)  # (frames, NE, cells)

    energies = _energy_grid(setup)
    injected = _footprint(setup)
    expected = (times.size, energies.size, injected.size)
    if recorded.shape != expected:
        raise ValueError(
            f"snap_f is {recorded.shape}, but the setup describes {expected} "
            "(frames, energy bins, cells)."
        )

    # The closed form. Linear in t with no small parameter and no fitted
    # constant: the slope in each bin is the setup's own rate times the line.
    rate = float(setup.injection.rate_per_ns)
    analytic = (
        _thermal_occupation(setup, energies)[None, :, None]
        + rate
        * times[:, None, None]
        * _line_shape(setup, energies)[None, :, None]
        * injected[None, None, :]
    )

    gap = setup.material.Delta_0
    centre = setup.injection.center_over_delta * gap
    sigma = setup.injection.sigma_over_delta * gap
    band = np.abs(energies - centre) <= _LINE_HALF_WIDTH_SIGMA * sigma
    peak_bin = int(np.argmin(np.abs(energies - centre)))
    # A line narrower than one cell would select no bins at all; keep the
    # nearest one so the curve degrades to a single probe rather than to zero.
    band[peak_bin] = True
    source_cell = int(np.argmax(injected))

    series_sim = [recorded[:, band, source_cell].T]
    series_ana = [analytic[:, band, source_cell].T]
    labels = [
        f"E = {e / gap:.3f}Δ, cell {source_cell} (source)" for e in energies[band]
    ]
    quiet = np.flatnonzero(~injected)
    if quiet.size:
        series_sim.append(recorded[:, peak_bin, quiet].T)
        series_ana.append(analytic[:, peak_bin, quiet].T)
        labels += [
            f"E = {energies[peak_bin] / gap:.3f}Δ, cell {c} (not injected)"
            for c in quiet
        ]

    return Curve(
        x=times,
        y_sim=np.concatenate(series_sim),
        y_analytic=np.concatenate(series_ana),
        x_label="t (ns)",
        y_label="occupation f(E, cell)",
        series_labels=tuple(labels),
        # Pointwise, not scale: the un-injected series are four decades below
        # the injected ones and a range-normalised metric would stop resolving
        # them, which is exactly where a leaked source would hide.
        metric="pointwise",
        note=(
            f"{int(band.sum())} energy bins across the line in the injected "
            f"cell {source_cell}, plus the line-centre bin in each of the "
            f"{int(quiet.size)} un-injected cells, over {times.size} recorded "
            "frames. Slopes are R·g(E) exactly; the un-injected series must "
            "hold their thermal seed."
        ),
    )


register(
    Benchmark(
        name="injection",
        title="External injection — a Gaussian line growing linearly in time",
        # T1 throughout. The prediction reads no engine kernel, weight array,
        # density of states or grid array; it needs the schema's grid formula,
        # a Fermi-Dirac seed and the Gaussian. The one engine-derived input is
        # the abscissa (the recorded frame times), which is a fact about the
        # frames rather than a physics input — see the caveat.
        tier="T1",
        formula_latex=(
            r"\frac{\partial f(E_i,c)}{\partial t}=S_{i,c},\qquad "
            r"S_{i,c}=R\,e^{-(E_i-E_0)^2/2\sigma^2}\,"
            r"\mathbb{1}\!\left[c\in\mathcal{F}\right]"
            r"\\[6pt]"
            r"\Longrightarrow\quad f(E_i,c;t)=f_{\rm FD}(E_i,T_{\rm bath})"
            r"+R\,e^{-(E_i-E_0)^2/2\sigma^2}\,"
            r"\mathbb{1}\!\left[c\in\mathcal{F}\right]\,t"
            r"\\[6pt]"
            r"E_i=\Delta_0\!\left(m_{\min}+(i+\tfrac12)\delta\right),\quad "
            r"E_0=c_\Delta\Delta_0,\quad \sigma=s_\Delta\Delta_0,\quad "
            r"R=\texttt{rate\_per\_ns},\quad "
            r"\mathcal{F}=\{\text{centre cell}\}"
        ),
        reason=(
            "With D₀ = 0, both collision channels off and no photon drives, "
            "the right-hand side is the source alone and does not depend on f, "
            "so f is exactly linear in t. The stepper reproduces that exactly "
            "rather than approximately: loss = 0 collapses the ETD1 weight "
            "(1−e^{−μdt})/μ to dt, the ETD2 corrector averages two identical "
            "constant rates, and the weighted-balance projection finds its "
            "target already met, so each step is f ← f + dt·S."
        ),
        # The audit of the first derivation recommended tightening 1e-11 to
        # 1e-12, and the sweep below is why that is right in both directions.
        headline_latex=(
            r"\partial_t f(E_i,c) \;=\; g(E_i,c) \;\Rightarrow\; x_{qp}(t) \;=\; x_{qp}(0) + \dot "
            r"S\,t"
        ),
        rel_tol=1e-12,
        convergence=(
            "Metric: pointwise |sim − ana|/|ana|, over the 34 series × 21 "
            "frames = 714 points of the shipped case (3×5 rectangle, 64 bins "
            "over [1,3]Δ, dt = 2 ns, 200 ns, 21 frames, 0.4 s). None of the "
            "714 sits below the metric's relative floor, so every point is "
            "informative — unlike the predecessor's 101-point straight line, "
            "which had 2 degrees of freedom, or its 64-bin energy curve, of "
            "which the audit found 23 bins carried signal.\n"
            "SHIPPED CASE: max 2.85e-15, rms 1.45e-15.\n"
            "\n"
            "R1 TIME STEP (200 ns fixed):\n"
            "  dt (ns)  steps   max rel err\n"
            "   8.000      25     1.964e-15\n"
            "   4.000      50     1.964e-15\n"
            "   2.000     100     2.853e-15\n"
            "   1.000     200     4.566e-15\n"
            "   0.500     400     8.128e-15\n"
            "   0.250     800     1.329e-14\n"
            "The residual GROWS as n_steps^0.55, i.e. the random-walk "
            "accumulation of float64 rounding over a sum of equal increments. "
            "There is no order in dt because there is no truncation error to "
            "converge away; that absence is the positive evidence of "
            "exactness, and the tolerance is set from this floor and its "
            "growth law rather than from a fit.\n"
            "\n"
            "R2 ENERGY BINS: NE = 32/64/128/256 (1.60/3.20/6.40/12.80 bins "
            "per σ) give 1.97e-15 / 2.85e-15 / 2.46e-15 / 2.93e-15 — flat at "
            "roundoff. The engine reproduces the closed form on whatever grid "
            "it is handed, so the term is grid-exact and this is not a "
            "convergence.\n"
            "\n"
            "R3 CELLS: 1, 15, 35 and 99 cells all give 2.85e-15, and the "
            "injected cell's x_qp excess is bit-identical across them "
            "(1.159761495227e-03), with N_cells·mean_excess/peak_excess = "
            "1.000000000 exactly. That pins the 1/N_cells source convention "
            "rather than hiding it (caveat 5).\n"
            "\n"
            "R4 RUN LENGTH (dt = 2 ns): 100/200/1000/5000 ns give 1.96e-15 / "
            "2.85e-15 / 9.10e-15 / 5.30e-14, the last reaching f = 9.88e-02 "
            "at the line peak. So at the mode's default max_time = 5000 ns — "
            "25× the shipped run — the residual is still 19× under tolerance.\n"
            "\n"
            "WHY 1e-12. Measured floor 2.85e-15 on the shipped case; worst "
            "measured 5.30e-14 at 25× the run length. 1e-11 was rejected "
            "because it fails to reject a 1-part-in-1e11 amplitude error — "
            "that run scores 9.9995e-12, which passes at 1e-11 and fails at "
            "1e-12. 1e-13 was rejected because a 5000 ns run already reaches "
            "5.3e-14, which would make the verdict depend on the run length "
            "rather than on the physics. At 1e-12 a 1-part-in-1e12 rate error "
            "is still separated (1.0013e-12, fail) while 3 parts in 1e13 are "
            "not (3.02e-13, pass), so this is where the check's resolution "
            "genuinely ends."
        ),
        modes=("spatial_2d",),
        build=_build,
        caveat=(
            "1. NOTHING COUPLED. Transport and both collision channels are "
            "off by construction, so this cannot detect a bug in how the "
            "source interacts with diffusion, recombination or a gap step. It "
            "validates the source's amplitude, units, line shape, footprint "
            "and time integration in isolation.\n"
            "\n"
            "2. NO PAULI BLOCKING TO CHECK. The engine adds external_gain "
            "with no (1−f) factor and no loss partner, and this benchmark "
            "PINS that absence: adding a blocking factor makes it go red (the "
            "audit measured 7.03e-03 on the 1-D twin, ~7e9 × this tolerance). "
            "What it cannot do is adjudicate whether blocking ought to be "
            "there; that is a modelling question this test is silent on. The "
            "exact linearity also fails once f approaches 1 and the stepper's "
            "[0,1] clip engages — here f reaches 3.95e-03, two decades "
            "below, so the clip is provably untouched.\n"
            "\n"
            "3. NOT A PHYSICAL STEADY STATE. With nothing removing "
            "quasiparticles the run cannot converge; stop_tol = 0 is set so "
            "the clock runs out instead, and the payload carries the honest "
            "'did not reach stop_tol' note. The reduction is a diagnostic.\n"
            "\n"
            "4. rate_per_ns IS A PEAK SPECTRAL RATE, not a total injected "
            "current: the line is peak-normalised to 1, so the injected "
            "current scales linearly with sigma_over_delta and a user "
            "sweeping the line width is also sweeping the source strength. "
            "This benchmark pins that convention absolutely: area-normalising "
            "the line would divide the source by σ√(2π) = 45.1 μeV on this "
            "case, a 98% error in every injected series.\n"
            "\n"
            "5. IT PINS A MESH BEHAVIOUR IT DOES NOT ENDORSE. 'centre_cell' "
            "and 'left_edge' inject into a fixed NUMBER of cells whatever the "
            "mesh (build_injection_2d), so the device-total injected current "
            "falls as 1/N_cells — measured exactly, R3 above. If that is ever "
            "changed to a per-area or per-device source, this benchmark will "
            "go red, correctly, and _footprint is where the new convention "
            "must be re-derived rather than patched. (The first derivation "
            "attributed this to a dated review at a specific source line; the "
            "audit could find neither, so the behaviour is stated on the "
            "strength of the measurement alone.)\n"
            "\n"
            "6. ONLY THE NARROW INJECTION KNOB. The general prescribed-drive "
            "path (setup.drives / DriveSpec) is different code and is not "
            "exercised here, and nothing is said about a general S(E).\n"
            "\n"
            "7. GRID AND GEOMETRY CONTRACT. grid.min_factor = 1.0 is required "
            "(validate_setup rejects above 1), which is also what makes the "
            "collision layer's zeroing of unsupported bins inert here. The "
            "footprint is derived from a rectangle's extent; a GDS mask is "
            "refused with a message rather than guessed. Odd extents are used "
            "so 'centre_cell' is a cell and not a four-way tie.\n"
            "\n"
            "8. THE ABSCISSA IS THE ENGINE'S. Frames are recorded at step "
            "boundaries at or after each cadence boundary, so the frame time "
            "is a fact about the frame; predicting it would re-implement the "
            "cadence rule. On this case snap_t_ns − k·snapshot_interval = 0.0 "
            "exactly. The residual still tests the relation between the "
            "recorded time and the recorded state: a stepper advancing twice "
            "per frame would double every slope and fail.\n"
            "\n"
            "9. INDEPENDENCE, PRECISELY. The prediction reads no engine "
            "kernel, weight array, DOS or grid array — the energy grid is "
            "rebuilt from the schema's documented formula and agrees with the "
            "engine's E_bins to 0.0 μeV, which is a result and not a "
            "dependency. But the grid formula and the Gaussian are the "
            "engine's own expressions re-derived, so this checks that the "
            "engine does what the schema says, not that the schema says the "
            "right thing.\n"
            "\n"
            "10. THE x_qp QUADRATURE IS NOT TESTED. Comparing f directly "
            "keeps the BCS cell measure out of the closed form on purpose; a "
            "bug in the published x_qp weights would not show up here."
        ),
        activity=(
            "1. THE OFF RUN IS PROVABLY A NULL. Same case with "
            "injection.enabled = False: the engine holds its thermal seed to "
            "7.59e-19 absolute (3.77e-15 relative) over 100 steps. The "
            "reduced right-hand side really is empty except for the term "
            "under test, so the whole measured signal is that term.\n"
            "\n"
            "2. THE CHECK FAILS WITH THE TERM OFF. The analytic side does not "
            "consult injection.enabled, so scoring that same off-run against "
            "this closed form gives 9.998e-01 — 1.0e+12 × the tolerance. The "
            "audit found exactly this fragility in the predecessor, whose "
            "helpers read the switch and therefore agreed with the engine in "
            "both positions; the only guard was a hand-written on/off ratio "
            "outside the benchmark.\n"
            "\n"
            "3. AMPLITUDE, LINE AND WIRING ALL MOVE THE VERDICT. Each row is "
            "an engine run with one input perturbed, scored against the "
            "UNPERTURBED closed form — perturbing the setup and re-deriving "
            "from it would move both sides and prove nothing:\n"
            "  rate × (1+1e-12)              1.0013e-12   1.0e+00 × tol\n"
            "  rate × (1+1e-09)              9.9982e-10   1.0e+03 × tol\n"
            "  centre 2.0 → 2.000001 Δ       2.9619e-05   3.0e+07 × tol\n"
            "  sigma × (1+1e-06)             8.7931e-06   8.8e+06 × tol\n"
            "  T_bath 0.3 → 0.3000001 K      5.3308e-06   5.3e+06 × tol\n"
            "  footprint → uniform           3.9559e+03   4.0e+15 × tol\n"
            "  footprint → left_edge         3.9559e+03   4.0e+15 × tol\n"
            "The complement holds too: the same closed form re-derived on a "
            "genuinely different case (5×7 cells, 96 bins, left_edge, rate "
            "5e-5) passes at 2.60e-15 over 58 series, so it describes the "
            "term rather than one dict.\n"
            "\n"
            "4. THE SIGNAL IS LARGE WHERE IT SHOULD BE. At the line-peak bin "
            "of the injected cell f runs from a thermal 9.9888e-07 to "
            "3.9525e-03 (×3956) while the 14 un-injected cells hold thermal; "
            "that cell's x_qp is 1.6228e-03 against the thermal 4.6302e-04.\n"
            "\n"
            "5. WHY T_bath = 0.3 K, WHICH IS A MEASUREMENT CHOICE. The "
            "un-injected series are the two-sided half of the footprint check "
            "— they are what catches a source applied in ADDITION to the "
            "right cell, which the injected series alone cannot see. At the "
            "predecessor's 0.1 K the thermal occupation at the line centre is "
            "~6e-19, below the metric's relative floor of 1e-9 × peak, and "
            "those series drop out: measured, the 'inject everywhere' "
            "mutation then scores 2.854e-15, i.e. indistinguishable from a "
            "clean run (314 of 714 points below floor). At 0.3 K nothing is "
            "below the floor and the same mutation scores 3.956e+03. The bath "
            "temperature is therefore part of the instrument, and lowering it "
            "would quietly remove a check."
        ),
    )
)
