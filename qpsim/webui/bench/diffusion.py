"""Diffusion: a cosine eigenmode decays at ``lambda(E) = D_eff(E) k^2``.

Switch every collision channel off and hold the gap uniform, and the kinetic
equation is nothing but the transport term,

    d(N_1^p f)/dt = div( D_0 N_1^q grad f ).

A uniform gap makes ``N_1`` constant in space, so it comes out of the
divergence and every energy bin obeys its own scalar heat equation with
diffusivity ``D_eff(E) = D_0 N_1(E)^(q-p)``. On a rectangle with no-flux rims
the separable cosine is an eigenfunction of the Laplacian, so the initial
modulation keeps its shape and only its amplitude moves -- exponentially, at a
rate that is a closed form.

The curve is taken over ENERGY because the energy dependence of ``D_eff`` is
the entire physical content of the ``(p, q)`` dressing: for the shipped
dirty-limit operator A1 = (1, 0) it is ``D_0 sqrt(E^2 - D^2)/E``, which falls
toward the gap edge, while the diagnostic A2 is flat and A1P rises. A check
that only measured "something decayed" would not tell those apart; this one
rejects the flat law at every point of the curve.

What the reference is, precisely
--------------------------------
``lambda_i = D_0 s_i k^2 / <N_1>_i^p`` (A1), with ``<N_1>_i`` the *exact
finite-volume cell average* of the BCS density over the cell -- closed form,
because the antiderivative is elementary:

    int E dE / sqrt(E^2 - D^2) = sqrt(E^2 - D^2).

That antiderivative is re-derived here; nothing is read from the engine's
kernel arrays, which is what makes this T1 rather than T2. But it is a
discretisation-*aware* T1: it reproduces the engine's cell-average convention
analytically, and that is deliberate, because near the gap edge the cell
average of an integrable ``1/sqrt`` singularity exceeds the centre value by a
factor that does NOT go away under refinement. Comparing the first bin against
the point value would be asserting something false. The independent evidence
that the convention is the average of the RIGHT function is the joint
space-and-energy refinement in ``convergence`` below, which converges at order
2.0 to the pure continuum form with no finite-volume convention anywhere in
it -- and which is, by construction, blind to the convention itself. The
headline pins the convention; the refinement pins the continuum limit. Neither
sentence should be read as the other.

Where the residual comes from
-----------------------------
All of it is the 5-point stencil. The cell-centred cosine is an exact
eigenvector of the discrete no-flux operator with eigenvalue
``k_h^2 = (4/dx^2)[sin^2(k_x dx/2) + sin^2(k_y dx/2)]``, so the reference,
which uses the continuum ``k^2``, sits a known ``|k_h^2 - k^2|/k^2`` away. At
the shipped mesh that is 2.7275e-03 and the measurement lands on it to six
digits. (The often-quoted isotropic shorthand ``(k dx)^2/12`` is NOT the
prediction here -- on a 2:1 rectangle ``k_x != k_y`` and it gives 4.02e-03.
The anisotropic form above is what is confirmed.)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Benchmark, Curve, register

# The case this benchmark is written for, as web-schema override paths.
#
# Every field is reachable through Spatial2DSetup today. Notes on the ones
# that are not free choices:
#
#   grid.min_factor = 1.0 puts the first cell edge exactly at E = Delta, so no
#     bin is cut by the gap and the q = 0 flux indicator averages to exactly 1
#     in every cell. (A cut bin is handled below, but it is one more convention
#     to argue about and there is no reason to buy that argument here.)
#   dt = 0.05 keeps the Crank-Nicolson substep count at exactly 1. Above about
#     0.137 ns the stepper subcycles and the effective step saturates at the
#     stiffness bound, which makes a dt refinement measure nothing.
#   stop_tol = 0.0 so the run always reaches max_time. The rate is fitted
#     against the run's own recorded frame times, so an early stop would not
#     corrupt it, but a run that stops early has fewer frames and less decay.
#   snapshot_interval = 1.0 is REQUIRED, not decorative: the rate is measured
#     from the whole recorded trajectory and the initial mode amplitude is read
#     off the t = 0 frame rather than re-derived from the initial-condition
#     builder. Re-deriving it agrees to roundoff today and silently couples the
#     verdict to the builder it is supposed to be checking.
#   initial.* prepares f = f0 (1 + a cos(m pi x) cos(n pi y)) as an ABSOLUTE
#     occupation on a flat baseline. Three reasons, each of which rules out an
#     alternative: (i) multiplicative rather than additive, so f stays strictly
#     inside [0, 1] and the stepper's clip never fires -- an additive cosine on
#     the 1e-10 thermal background goes negative wherever the cosine does and
#     is silently clipped every substep; (ii) a flat baseline rather than the
#     thermal one, so every energy bin starts with the same mode amplitude and
#     every bin's fit has the same dynamic range; (iii) transport is exactly
#     linear in f and does not couple energies, so the baseline cannot change
#     the measured rate -- the derivation checked this directly (thermal
#     baseline vs flat 0.1 moves every rate by < 3e-13).
#   The mode numbers live in initial.params, not in the expression text, so
#     the analytic side can read them. They are then CHECKED against the run's
#     own first frame, so a mismatch between params and expression is caught
#     rather than believed.
CASE_OVERRIDES: dict[str, Any] = {
    "mode": "spatial_2d",
    "material.name": "Al",
    "material.Delta_0": 180.0,
    "material.T_c": 1.18,
    "material.tau_0": 438.0,
    "material.D_0": 60.0,
    "material.dynes_gamma": 0.0,
    "T_bath": 0.1,
    "grid.min_factor": 1.0,
    "grid.max_factor": 4.0,
    "grid.num_bins": 64,
    "geometry.kind": "rectangle",
    "geometry.rows": 16,
    "geometry.cols": 32,
    "geometry.mesh_size_um": 4.0,
    "boundary.kind": "reflective",
    "diffusion_model": "A1",
    "collisions.scattering": False,
    "collisions.recombination": False,
    "gap_regions.kind": "uniform",
    "injection.enabled": False,
    "subgap_drive.enabled": False,
    "pb_drive.enabled": False,
    "phonons.mode": "thermal_bath",
    "self_consistent_gap": False,
    "initial.kind": "absolute",
    "initial.expression": (
        "params['f0'] * (1.0 + params['a']"
        " * np.cos(params['m'] * np.pi * x) * np.cos(params['n'] * np.pi * y))"
    ),
    "initial.params": {"f0": 0.1, "a": 0.4, "m": 1.0, "n": 1.0},
    "dt": 0.05,
    "max_time": 12.0,
    "stop_tol": 0.0,
    "snapshot_interval": 1.0,
}

# (p, q) per operator name. Deliberately a second copy of what
# qpsim.transport.diffusion.base.DiffusionModel holds rather than an import:
# the dressing exponents ARE the physics under test here, and a reference that
# imported them would follow the engine silently wherever it went. If the two
# ever disagree this benchmark fails, which is the point of it.
_PQ: dict[str, tuple[int, int]] = {
    "A1": (1, 0), "A1P": (1, 2), "A2": (2, 2), "C": (0, -1), "B": (0, -2),
}

# The mode amplitude must be a real modulation, not the residue of a uniform
# start. A thermal (spatially flat) initial condition projects onto the cosine
# at ~1e-26 of the field, and the ratio of two such numbers is a finite,
# entirely fictitious decay rate -- the run looks measured and means nothing.
_MIN_MODE_FRACTION = 1e-6
# The reference must predict real decay over the window, or the check is
# 0 == 0: with D_0 = 0 the closed form is also zero everywhere and a benchmark
# that scored it would report a perfect pass for a run in which the term under
# test did nothing at all. 0.1 e-foldings is a ~10% amplitude change. Not "any
# nonzero decay", because a run with lambda*T = 1e-9 is arithmetically
# scoreable and physically vacuous, and PASS would be read as evidence that
# diffusion works; not 1.0 either, because the fit resolves ln A to 1e-14 and
# refusing a legitimately shorter window would be pure superstition.
_MIN_DECAY_EXPONENT = 0.1
# The cosine must still BE the prepared state: this bounds the t=0 residual
# after removing the cell mean and the mode, relative to the field's own peak.
_SHAPE_TOL = 1e-10


def _mode_numbers(setup: Any) -> tuple[float, float]:
    params = dict(getattr(setup.initial, "params", {}) or {})
    missing = [k for k in ("m", "n") if k not in params]
    if missing:
        raise ValueError(
            f"initial.params is missing {', '.join(missing)}. This benchmark "
            "needs the cosine mode numbers as parameters (and the initial "
            "expression should read them from params too), because the "
            "analytic rate is D_eff k^2 with k^2 built from them. See "
            "CASE_OVERRIDES in this module."
        )
    m, n = float(params["m"]), float(params["n"])
    if m <= 0.0 and n <= 0.0:
        raise ValueError(
            "Mode numbers m = n = 0 give a spatially uniform start, which is "
            "the null mode of a reflective operator: nothing diffuses and "
            "there is no rate to measure."
        )
    return m, n


def _cell_cosine(mask: np.ndarray, m: float, n: float) -> np.ndarray:
    """The mode at CELL CENTRES -- the exact eigenvector of the no-flux stencil.

    Cell centres, not nodes: the mirrored ghost cell then takes the value of
    the end cell identically, so the discrete no-flux condition is satisfied
    exactly and the discrete eigenvalue is the clean
    ``(4/dx^2) sin^2(k dx/2)``. On nodes the boundary condition is only
    approximately satisfied and the whole benchmark quietly drops to first
    order.
    """
    rows, cols = np.nonzero(mask)
    nrow, ncol = mask.shape
    return (
        np.cos(m * np.pi * (cols + 0.5) / ncol)
        * np.cos(n * np.pi * (rows + 0.5) / nrow)
    )


def _analytic_rates(setup: Any, energies: np.ndarray) -> tuple[np.ndarray, float]:
    """``(lambda_i, k^2)`` from the setup alone -- no engine array is read."""
    gap = float(setup.material.Delta_0)
    d0 = float(setup.material.D_0)
    n_bins = int(setup.grid.num_bins)
    model = str(setup.diffusion_model)
    if model not in _PQ:
        raise ValueError(
            f"No (p, q) dressing recorded here for diffusion model {model!r}. "
            f"Known: {', '.join(sorted(_PQ))}. The exponents are written out "
            "in this module rather than imported, so a new operator has to be "
            "stated here before its rate can be predicted."
        )
    p, q = _PQ[model]

    # The energy grid rebuilt from its stated definition (uniform, cell
    # centred, spanning [min_factor, max_factor] x gap) and then checked
    # against the grid the run actually used. Checked rather than trusted:
    # every edge below is an exact integral over a cell, so a different cell
    # would silently be an integral of the right function over the wrong
    # interval -- the one error that looks like agreement.
    width = (float(setup.grid.max_factor) - float(setup.grid.min_factor)) * gap
    d_e = width / n_bins
    edges = float(setup.grid.min_factor) * gap + d_e * np.arange(n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if energies.size != n_bins or np.max(np.abs(energies - centres)) > 1e-9 * gap:
        raise ValueError(
            "The run's energy grid is not the uniform cell-centred grid this "
            "reference integrates over, so the closed-form cell averages "
            "below would be integrals over the wrong cells."
        )

    # Exact cell average of N_1 = E/sqrt(E^2 - D^2) over each cell, from the
    # elementary antiderivative sqrt(E^2 - D^2). Written as (E-D)(E+D) rather
    # than E^2 - D^2 because at E ~ D the latter is a catastrophic
    # cancellation of two nearly equal numbers.
    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], gap)
    n1 = (
        np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0))
        - np.sqrt(np.maximum((lo - gap) * (lo + gap), 0.0))
    ) / d_e
    # q = 0 is the dirty-limit indicator "is there continuum here", whose
    # exact cell average is the supported fraction of the cell, not 1 merely
    # because the cell has some support.
    support = np.maximum(edges[1:] - lo, 0.0) / d_e

    live = n1 > 0.0
    flux = np.zeros_like(n1)
    flux[live] = d0 * (support[live] if q == 0 else n1[live] ** q)
    d_eff = np.zeros_like(n1)
    d_eff[live] = flux[live] / (n1[live] ** p if p else 1.0)

    dx = float(setup.geometry.mesh_size_um)
    m, n = _mode_numbers(setup)
    k2 = (m * np.pi / (setup.geometry.cols * dx)) ** 2 + (
        n * np.pi / (setup.geometry.rows * dx)
    ) ** 2
    return d_eff * k2, float(k2)


def _fit_rates(times: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares decay rate per bin from ln|A(t)|, and the worst residual.

    A least squares over every recorded frame rather than the endpoint ratio:
    Crank-Nicolson multiplies the mode by a fixed factor per step, so ln|A| is
    exactly linear in t and the fit residual is a measurement in its own right
    -- it is how a decay that is NOT a single exponential (a leaked second
    mode, a clipped state, a term that is not actually off) announces itself.
    """
    log_a = np.log(np.abs(amps))
    t_centred = times - times.mean()
    slope = (t_centred[:, None] * (log_a - log_a.mean(axis=0))).sum(axis=0) / (
        t_centred**2
    ).sum()
    intercept = log_a.mean(axis=0) - slope * times.mean()
    model = intercept[None, :] + np.outer(times, slope)
    spread = np.max(np.abs(log_a - log_a.mean(axis=0)), axis=0)
    residual = np.max(np.abs(log_a - model) / np.maximum(spread, 1e-300))
    return -slope, float(residual)


def _build(
    setup: Any, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> Curve:
    if "snap_f" not in arrays or "snap_t_ns" not in arrays:
        raise ValueError(
            "This benchmark reads the recorded frames: set snapshot_interval "
            "on the setup. The rate is fitted over the whole trajectory and "
            "the initial mode amplitude is measured from the t = 0 frame, so "
            "that neither the rate nor its reference comes from the "
            "initial-condition builder being checked."
        )
    times = np.asarray(arrays["snap_t_ns"], dtype=float)
    frames = np.asarray(arrays["snap_f"], dtype=float)      # (n_t, NE, N_cells)
    if times.size < 3:
        raise ValueError(
            f"Only {times.size} recorded frame(s); a decay rate fitted over "
            "fewer than three is not a measurement. Shorten snapshot_interval."
        )

    mask = np.asarray(arrays["mask"]).astype(bool)
    if not mask.all():
        raise ValueError(
            "The cosine is an eigenmode of the no-flux operator on a FULL "
            "rectangle. On a ragged or masked geometry the reference does not "
            "apply -- and the reflective faces the operator invents at a mask "
            "boundary are exactly what it would fail to describe."
        )
    if str(setup.boundary.kind) != "reflective":
        raise ValueError(
            f"boundary.kind is {setup.boundary.kind!r}. The cosine satisfies "
            "the no-flux condition identically at all four walls; under a "
            "Dirichlet or absorbing rim it is not an eigenmode and decays at "
            "an entirely different rate."
        )
    gap = float(setup.material.Delta_0)
    # Every recorded frame's gap map when there is one, so a self-consistent
    # gap that moved mid-run and came back is caught too, not just the final
    # profile.
    gaps = np.asarray(
        arrays["snap_gap"] if "snap_gap" in arrays else arrays["gap_per_cell"],
        dtype=float,
    )
    if gaps.size and np.max(np.abs(gaps - gap)) > 1e-9 * gap:
        raise ValueError(
            "The gap is not uniform over the device (or moved during the run, "
            "with a self-consistent gap). N_1 then depends on position, does "
            "not come out of the divergence, and the separable cosine is no "
            "longer an eigenmode."
        )

    m, n = _mode_numbers(setup)
    phi = _cell_cosine(mask, m, n)
    amps = frames @ phi / (phi @ phi)                        # (n_t, NE)

    # Is the prepared state the mode the parameters claim? Removing the cell
    # mean and the projection onto phi must leave nothing. This is what makes
    # reading (m, n) from params safe: a params/expression mismatch shows up
    # here rather than as a wrong reference nobody can see.
    first = frames[0]
    peak = float(np.max(np.abs(first)))
    residual = first - (
        first.mean(axis=1)[:, None] + np.outer(amps[0], phi)
    )
    if peak <= 0.0 or float(np.max(np.abs(residual))) > _SHAPE_TOL * peak:
        raise ValueError(
            "The initial state is not 'uniform + cosine mode' to roundoff, so "
            "the projected amplitude is not the amplitude of a single "
            "eigenmode and its decay is not a single exponential."
        )
    if float(np.max(np.abs(amps[0]))) <= _MIN_MODE_FRACTION * peak:
        raise ValueError(
            "The initial cosine-mode amplitude is "
            f"{float(np.max(np.abs(amps[0]))):.3e} against a field peak of "
            f"{peak:.3e}: the run starts spatially uniform, so the transport "
            "term has no gradient to act on and is exactly inert. Fitting "
            "that ratio yields a finite, entirely fictitious rate -- prepare "
            "an initial condition with a spatial modulation."
        )

    energies = np.asarray(arrays["E_bins"], dtype=float)
    exact, _k2 = _analytic_rates(setup, energies)
    span = float(times[-1] - times[0])
    if float(np.max(exact)) * span < _MIN_DECAY_EXPONENT:
        raise ValueError(
            "The closed form predicts a decay of at most "
            f"{float(np.max(exact)) * span:.3e} e-foldings over this run "
            f"(needs {_MIN_DECAY_EXPONENT:g}), so there is nothing to measure. "
            "At D_0 = 0, or over a window this short, the reference is zero "
            "or nearly so, the simulation matches it trivially, and a verdict "
            "of 'pass' would certify only that neither moved."
        )

    live = exact > 0.0
    measured, fit_residual = _fit_rates(times, amps[:, live])
    exact = exact[live]
    x = energies[live] / gap

    ratio = measured / exact
    decay = np.abs(amps[-1, live] / amps[0, live])
    note = (
        f"{int(live.sum())} energy bins, {times.size} frames over "
        f"{span:g} ns. Mode amplitude fell to {decay.min():.4f}-"
        f"{decay.max():.4f} of its initial value, i.e. up to a factor "
        f"{1.0 / decay.min():.2f} -- not a perturbation. Worst departure from "
        f"a single exponential over the whole trace: {fit_residual:.2e} of the "
        f"fitted range. The residual is a near-uniform multiplicative offset "
        f"(the 5-point stencil's eigenvalue error): the spread of "
        f"lambda_sim/lambda_exact across the curve is "
        f"{ratio.max() / ratio.min() - 1.0:.2e}, so the SHAPE -- which is the "
        f"whole content of the N_1 dressing -- is reproduced far tighter than "
        f"the scale the verdict is taken on."
        + (
            f" {int((~live).sum())} bin(s) with no continuum weight carry no "
            "transport and are excluded."
            if not live.all() else ""
        )
    )

    return Curve(
        x=x,
        y_sim=measured,
        y_analytic=exact,
        x_label="E / Δ",
        y_label="cosine-mode decay rate λ(E)  (1/ns)",
        note=note,
    )


register(Benchmark(
    name="diffusion",
    title="Diffusion — cosine-mode decay rate λ(E) = D_eff(E) k²",
    # T1: the reference is written from the physics and reads no engine array.
    # It is nonetheless discretisation-aware in energy (it reproduces the
    # engine's finite-volume cell average analytically, because near the gap
    # edge the cell average of an integrable singularity differs from the
    # centre value by a factor that never converges away). So the honest
    # reading is: T1 throughout, continuum-verified in the joint refinement
    # below, with the headline additionally pinning the cell-average
    # convention. It is NOT a T2 -- no kernel array is imported, and a wrong
    # kernel IS detected: measured on this case, a point-value dressing fails
    # at 3.80e-01, an inverted one at 7.41e+00, a 1% error in D_eff at
    # 7.25e-03. A T2 could not see any of them.
    tier="T1",
    formula_latex=(
        r"f(E,x,y,t)=\bar f(E)+A(E)\,"
        r"\cos\!\frac{m\pi x}{L_x}\cos\!\frac{n\pi y}{L_y}\,e^{-\lambda(E)t}"
        r",\qquad \boxed{\lambda(E)=D_{\rm eff}(E)\,k^{2}}\\[4pt]"
        r"k^{2}=\Big(\frac{m\pi}{L_x}\Big)^{2}+\Big(\frac{n\pi}{L_y}\Big)^{2},"
        r"\qquad D_{\rm eff}=D_{0}\,N_{1}^{\,q-p}"
        r"\;\xrightarrow[\;A1:(p,q)=(1,0)\;]{}\;\frac{D_{0}}{N_{1}}"
        r"=D_{0}\frac{\sqrt{E^{2}-\Delta^{2}}}{E}\\[6pt]"
        r"\text{with }N_1\text{ its exact cell average }"
        r"\langle N_{1}\rangle_{i}=\frac{1}{\Delta E}"
        r"\int_{E_{i-1/2}}^{E_{i+1/2}}\!\frac{E\,dE}{\sqrt{E^{2}-\Delta^{2}}}"
        r"=\frac{\sqrt{E_{i+1/2}^{2}-\Delta^{2}}-"
        r"\sqrt{E_{i-1/2}^{2}-\Delta^{2}}}{\Delta E}"
    ),
    reason=(
        "With no collisions and a uniform gap, N₁ leaves the divergence and "
        "each energy bin is an independent scalar heat equation; a cell-"
        "centred cosine is an exact eigenfunction of the Laplacian AND of the "
        "discrete no-flux stencil, so its amplitude decays as a pure "
        "exponential at D_eff(E)k². The energy dependence of that rate is the "
        "whole content of the (p,q) dressing, which is why the curve is taken "
        "over energy rather than time."
    ),
    # 3.5e-3 against a measured 2.7275e-3. The headroom is 28%, and it covers
    # a residual that is deterministic to 13 digits, so it is not slack for
    # noise -- it is slack for a mesh that is not exactly this one. 4e-3 (the
    # original claim) was looser than the evidence needed; 1e-3 would need the
    # 32x64 mesh, which costs 51 s per run instead of 18 s.
    rel_tol=3.5e-3,
    convergence=(
        "Headline case: 16x32 cells at 4 μm (128x64 μm), 64 energy bins, "
        "dt = 0.05 ns, 240 steps, 13 frames, 18 s. Measured max relative "
        "error 2.7275e-03 (rms 2.7262e-03).\n\n"
        "That residual is not noise and not a fitting artefact: it is the "
        "5-point stencil's eigenvalue error. The cell-centred cosine is an "
        "exact eigenvector of the discrete no-flux operator with eigenvalue "
        "k_h² = (4/dx²)[sin²(k_x dx/2) + sin²(k_y dx/2)], so a reference built "
        "on the continuum k² sits |k_h² - k²|/k² = 2.727495e-03 away — and the "
        "measurement lands on it to six digits (2.727457e-03).\n\n"
        "A. SPACE. Halve dx at a fixed 128x64 μm domain, T = 12 ns, NE = 16, "
        "dt scaled as dx² to hold the CN substep count at 1. Reference: this "
        "module's closed form.\n"
        "    8x16  dx=8 μm  dt=0.2 ns     1.0868e-02\n"
        "   16x32  dx=4 μm  dt=0.05 ns    2.7274e-03   order 1.994\n"
        "   32x64  dx=2 μm  dt=0.0125 ns  6.8249e-04   order 1.999\n"
        "  Second order, as a 5-point finite-volume stencil must be.\n\n"
        "B. TIME. Halve dt at a fixed 16x32 mesh, NE = 16. Reference here is "
        "the DISCRETE-space rate D_eff k_h², so the spatial error is removed "
        "and only the integrator is left.\n"
        "   dt=0.05 ns     1.5808e-06\n"
        "   dt=0.025 ns    3.9520e-07   order 2.000\n"
        "   dt=0.0125 ns   9.8799e-08   order 2.000\n"
        "  Second order, as Crank-Nicolson must be. Note the magnitude: the "
        "engine's monotonicity bound (dt x max exit rate <= 1) forces "
        "λτ <= k²dx²/4, so the temporal error is structurally ~1700x below the "
        "spatial one and cannot dominate at any admissible dt.\n\n"
        "C. JOINT, AGAINST THE PURE CONTINUUM. Refine space AND energy "
        "together by 1.5x per level, comparing against "
        "D_0 sqrt(E²-Δ²)/E x k² — point-value density of states, continuum k², "
        "no finite-volume convention anywhere in it — restricted to E >= 1.5Δ.\n"
        "    8x16  NE=32  dE=16.875 μeV  dt=0.2 ns     1.1503e-02\n"
        "   12x24  NE=48  dE=11.250 μeV  dt=0.1 ns     5.1095e-03   order 2.001\n"
        "   18x36  NE=72  dE= 7.500 μeV  dt=0.075 ns   2.2792e-03   order 1.991\n"
        "  This is what establishes that the engine's density dressing is the "
        "cell average of E/sqrt(E²-Δ²) and of nothing else. A pure energy "
        "refinement at a fixed mesh floors out on the (fixed) spatial error "
        "and reports a meaningless order, which is why it is joint. It is "
        "also, by construction, blind to the finite-volume convention itself: "
        "the headline pins the convention, C pins the continuum limit.\n\n"
        "BUDGET at the headline case: stencil 2.727495e-03; integrator "
        "1.584738e-06 against the Crank-Nicolson prediction (λτ)²/12 = "
        "1.584734e-06 with τ = dt/2 (the Strang half-step); remainder after "
        "removing both, 4.5e-12. The tolerance therefore covers a known and "
        "reproducible offset, not an unexplained one.\n\n"
        "DETECTION POWER, measured by mutating the engine's density dressing "
        "by a constant factor: x1.0005 -> 3.23e-03 (missed), x1.0013 -> "
        "4.02e-03 (caught), x1.005 -> 7.69e-03 (caught). A common-mode error "
        "in <N_1> or D_0 is caught from about 0.08% up; see the last caveat "
        "for why the SHAPE is pinned some 1800x tighter than that."
    ),
    modes=("spatial_2d",),
    build=_build,
    caveat=(
        "1. IT CANNOT TELL A1 FROM C. At a uniform gap both give "
        "D_eff = D_0/N_1, and the engine's two runs agree to 1.6e-14 — "
        "measured, not assumed. They differ in the conserved density (N_1 f vs "
        "f) and at a gap gradient, and this case has neither. Separating them "
        "needs a non-uniform gap. It does separate A1 from A2 and A1P sharply; "
        "see `activity`.\n"
        "2. UNIFORM GAP ONLY. The harmonic-mean face weight is exact only "
        "where neighbours share a gap, so the operator's documented "
        "first-order bias at a gap step, and the Kupriyanov-Lukichev "
        "interface (gap_regions.kind='column_step' with a finite "
        "interface_G_N), are invisible here by construction. A non-uniform-gap "
        "run is refused rather than scored: a red verdict there would blame "
        "the engine for a case the reference does not cover.\n"
        "3. REFLECTIVE RIMS ONLY. The Dirichlet/absorbing/Robin edges and the "
        "inhomogeneous boundary-forcing branch are never exercised — with "
        "reflective rims that forcing is identically zero. Also refused, not "
        "scored.\n"
        "4. SOLID RECTANGLE ONLY. The per-energy active region is the whole "
        "mask, so the operator never has to invent a reflective face at a "
        "states boundary. A masked, annular or ragged active set is untested "
        "here.\n"
        "5. NO CLIPPING AND NO SUBCYCLING. Exactly one CN substep, and f stays "
        "in [0.0713, 0.1287], so the subcycling path and the [0,1] clip are "
        "cold.\n"
        "6. THE MONOTONICITY BOUND HIDES TEMPORAL ERROR. λτ <= k²dx²/4 always, "
        "so refinement B has to compare against the discrete-space rate to see "
        "the integrator at all. A time-integration regression would have to be "
        "large to move the headline number.\n"
        "7. ONE MODE, ONE MATERIAL. Only (m,n) = (1,1) on a 2:1 rectangle with "
        "Al parameters. Higher modes probe the stencil harder (larger k dx) "
        "and are not run.\n"
        "8. THE DYNAMIC RANGE IS MOSTLY IN ONE BIN. λ spans a factor 6.40 over "
        "the full grid, but 45.4% of that log range is the single gap-edge "
        "bin — and that bin's reference is the finite-volume cell average of "
        "an integrable 1/sqrt singularity, i.e. exactly the point refinement C "
        "excludes. Over the continuum-verified part (E >= 1.5Δ, 53 bins) the "
        "curve spans 1.273x, and 1.718x over E >= 1.2Δ. It is a real curve — "
        "monotone, 64 points, shape resolved to 1.5e-06 — but it is not the "
        "factor of 6.4 of independently grounded variation that '6.4x' would "
        "suggest.\n"
        "9. THE GAP-EDGE CELL AVERAGE IS NOT THE POINT VALUE, AND NEVER "
        "BECOMES IT. It exceeds the centre value by sqrt(2) there, and that "
        "ratio does not converge away: it is a true property of a "
        "finite-volume scheme on an integrable singularity, not an error. The "
        "headline compares that bin against the exact cell average, where it "
        "agrees like every other bin. Asserting continuum agreement there "
        "would be asserting something false.\n"
        "10. FOR q != 0 (A1P, A2, B) the engine's flux weight is "
        "(cell average of N_1)^q, not the cell average of N_1^q, and this "
        "reference makes the same choice. Only A1 (q = 0) has refinement C "
        "behind it, so the A1P agreement quoted in `activity` is a "
        "convention-locked check, not independent evidence.\n"
        "11. THE VERDICT IS ON THE ABSOLUTE RATE. The residual is an almost "
        "purely multiplicative offset (shape spread 1.5e-06 across the curve), "
        "so the energy dependence — the whole content of the (p,q) dressing — "
        "is verified some 1800x tighter than this tolerance, while a "
        "common-mode error in D_0 or <N_1> is only caught above ~0.08%. A "
        "shape-only check normalised to its own mean would be far more "
        "powerful at ~1e-04; it is not what this verdict is."
    ),
    activity=(
        "1. THE TERM MOVES THE STATE A LOT. Over 12 ns the cosine amplitude "
        "falls to 0.1233 (top bin) .. 0.7209 (gap-edge bin) of its initial "
        "value — a factor 8.11 at the top of the grid, which is not a "
        "perturbation. A(0) = 4.000e-02 in every bin, read off the recorded "
        "t = 0 frame rather than re-derived.\n"
        "2. DISABLING IT FAILS THE CHECK. With the transport half of the "
        "split step replaced by the identity in the engine and the reference "
        "untouched (D_0 still 60 μm²/ns), the measured rate is identically "
        "zero and the benchmark FAILS at 1.000e+00. That is the honest foil: "
        "setting D_0 = 0 instead would zero the reference too, and 0 == 0 "
        "passes — so this module refuses that case rather than scoring it "
        "('the closed form predicts a decay of at most 0.000e+00 "
        "e-foldings').\n"
        "3. IT REFUSES AN INERT MEASUREMENT. A uniform (thermal) start "
        "projects onto the cosine at 7.6e-29 against a field peak of 1.2e-10, "
        "and the fit would report finite, entirely fictitious rates "
        "(-1.90, -0.50, -1.12, ... 1/ns) rather than a clean NaN. The "
        "amplitude guard rejects it instead — this is the 'inert default' "
        "trap, and it is checked rather than assumed away.\n"
        "4. NOTHING ELSE IS MOVING. The full step is transport(dt/2) -> "
        "collisions -> transport(dt/2), run 240 times with both collision "
        "channels off: the mode shape is preserved to 2.8e-14 and the cell "
        "mean (the null mode of a reflective operator) drifts by 6.0e-14. "
        "Nothing is leaking into or out of the eigenmode.\n"
        "5. THE CURVE'S SHAPE IS THE OPERATOR'S DRESSING. Against the flat law "
        "λ = D_0k² (the A2 form) the measured curve is rejected at every one "
        "of the 64 bins: 8.49e-01 at worst and 3.48e-02 at best, still 10x the "
        "tolerance where the foil is closest. And the benchmark tracks the "
        "operator family rather than merely detecting decay — A2 measures flat "
        "(λ_max/λ_min = 1.0000) and A1P reverses the slope "
        "(λ(E_max)/λ(E_min) = 0.303 against 3.302 for A1), each matching its "
        "own closed form to 2.7e-03.\n"
        "6. A WRONG KERNEL FAILS LOUDLY. Mutating the engine: point-value "
        "density of states instead of the cell average 3.80e-01; inverted "
        "dressing sqrt(E²-Δ²)/E 7.41e+00; D_eff off by 1% 7.25e-03; <N_1> "
        "scaled by 1.0013 4.02e-03 — all FAIL."
    ),
))
