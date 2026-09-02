"""Shared machinery for the transport benchmarks.

Every transport benchmark is the same experiment with a different rim: switch
every collision channel off, hold the gap uniform, and each energy bin is an
independent scalar heat equation with diffusivity ``D_eff(E) = D_0 N_1^(q-p)``.
Prepare an eigenmode of the Laplacian *under the rim's boundary condition*,
and its amplitude decays as a pure exponential at ``D_eff(E) k^2`` with ``k``
fixed by the rim. What changes between the cases is only the mode and its
wavenumber; the energy dependence of the rate, the projection, the fit and
the checks that make the measurement honest are the same and live here.

The closed forms are written from the physics and read no engine array.
``d_eff_per_bin`` is discretisation-aware in energy on purpose -- it
reproduces the engine's finite-volume cell average of the BCS density
analytically, because near the gap edge that average differs from the point
value by a factor that never converges away (see ``bench/diffusion.py`` for
the joint refinement that pins the continuum limit independently).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpsim.webui.benchmarks import Curve

# (p, q) per operator name. Deliberately a second copy of what
# qpsim.transport.diffusion.base.DiffusionModel holds rather than an import:
# the dressing exponents ARE the physics under test here, and a reference that
# imported them would follow the engine silently wherever it went. If the two
# ever disagree these benchmarks fail, which is the point of them.
PQ: dict[str, tuple[int, int]] = {
    "A1": (1, 0), "A1P": (1, 2), "A2": (2, 2), "C": (0, -1), "B": (0, -2),
}

# The mode amplitude must be a real modulation, not the residue of a uniform
# start: a flat initial condition projects onto a mode at ~1e-26 of the field,
# and the ratio of two such numbers is a finite, entirely fictitious rate.
MIN_MODE_FRACTION = 1e-6
# The reference must predict real decay over the window, or the check is
# 0 == 0 and a run in which transport did nothing would pass.
MIN_DECAY_EXPONENT = 0.1
# The prepared state must still BE baseline + mode to roundoff, or the
# projected amplitude is not a single eigenmode's and its decay is not one
# exponential. Relaxed only where the case says so (a bump on an annulus).
SHAPE_TOL = 1e-10
# Where the prepared state is NOT the mode, the fit of a bin starts once its
# leakage into the next mode has decayed to this fraction of the mode: a
# residual leak e biases the fitted rate by about e (k2_next/k2 - 1), so 1e-4
# keeps the bias two orders below the first-order rim errors being measured.
LEAK_TARGET = 1e-4


def dressing(setup: Any) -> tuple[int, int]:
    model = str(setup.diffusion_model)
    if model not in PQ:
        raise ValueError(
            f"No (p, q) dressing recorded here for diffusion model {model!r}. "
            f"Known: {', '.join(sorted(PQ))}. The exponents are written out in "
            "bench/_transport.py rather than imported, so a new operator has "
            "to be stated there before its rate can be predicted."
        )
    return PQ[model]


def d_eff_per_bin(setup: Any, energies: np.ndarray) -> np.ndarray:
    """``D_eff(E_i)`` per bin from the setup alone -- no engine array is read.

    Zero where a bin carries no continuum weight (entirely below the gap); a
    caller treats those bins as carrying no transport. The energy grid is
    rebuilt from its stated definition and CHECKED against the run's, because
    every quantity below is an exact integral over a cell and a different cell
    would silently be an integral of the right function over the wrong
    interval -- the one error that looks like agreement.
    """
    gap = float(setup.material.Delta_0)
    d0 = float(setup.material.D_0)
    n_bins = int(setup.grid.num_bins)
    p, q = dressing(setup)

    width = (float(setup.grid.max_factor) - float(setup.grid.min_factor)) * gap
    d_e = width / n_bins
    edges = float(setup.grid.min_factor) * gap + d_e * np.arange(n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if energies.size != n_bins or np.max(np.abs(energies - centres)) > 1e-9 * gap:
        raise ValueError(
            "The run's energy grid is not the uniform cell-centred grid this "
            "reference integrates over, so the closed-form cell averages "
            "would be integrals over the wrong cells."
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
    # exact cell average is the supported fraction of the cell.
    support = np.maximum(edges[1:] - lo, 0.0) / d_e

    live = n1 > 0.0
    flux = np.zeros_like(n1)
    flux[live] = d0 * (support[live] if q == 0 else n1[live] ** q)
    d_eff = np.zeros_like(n1)
    d_eff[live] = flux[live] / (n1[live] ** p if p else 1.0)
    return d_eff


def require_frames(
    arrays: dict[str, np.ndarray], minimum: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """``(times, frames)`` -- the recorded trajectory, or a clear refusal."""
    if "snap_f" not in arrays or "snap_t_ns" not in arrays:
        raise ValueError(
            "This benchmark reads the recorded frames: set snapshot_interval "
            "on the setup. The rate is fitted over the whole trajectory and "
            "the initial mode amplitude is measured from the t = 0 frame, so "
            "that neither comes from the initial-condition builder being "
            "checked."
        )
    times = np.asarray(arrays["snap_t_ns"], dtype=float)
    frames = np.asarray(arrays["snap_f"], dtype=float)      # (n_t, NE, N_cells)
    if times.size < minimum:
        raise ValueError(
            f"Only {times.size} recorded frame(s); a decay rate fitted over "
            f"fewer than {minimum} is not a measurement. Shorten snapshot_interval."
        )
    return times, frames


def require_uniform_gap(setup: Any, arrays: dict[str, np.ndarray]) -> float:
    """The scalar gap, having checked no cell and no frame departed from it."""
    gap = float(setup.material.Delta_0)
    gaps = np.asarray(
        arrays["snap_gap"] if "snap_gap" in arrays else arrays["gap_per_cell"],
        dtype=float,
    )
    if gaps.size and np.max(np.abs(gaps - gap)) > 1e-9 * gap:
        raise ValueError(
            "The gap is not uniform over the device (or moved during the run, "
            "with a self-consistent gap). N_1 then depends on position, does "
            "not come out of the divergence, and the mode is no longer an "
            "eigenfunction."
        )
    return gap


def require_no_collisions(setup: Any) -> None:
    """Transport alone: every other switch must be off, and say so if not."""
    c = setup.collisions
    if c.scattering or c.recombination:
        raise ValueError(
            "This benchmark is transport alone; scattering and recombination "
            "must be off, or the mode also relaxes by collisions at a rate "
            "the closed form does not contain."
        )
    if str(setup.phonons.mode) != "thermal_bath" or bool(setup.self_consistent_gap):
        raise ValueError(
            "This benchmark needs a pinned thermal bath and a fixed gap."
        )
    if setup.injection.enabled or setup.drives:
        raise ValueError("This benchmark needs no injection and no drives.")


def strip_axis(mask: np.ndarray) -> np.ndarray:
    """Cell index along a one-row strip, in mask order; refuses anything else."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or np.ptp(rows) != 0 or np.ptp(cols) + 1 != rows.size:
        raise ValueError(
            "This benchmark is written for a one-row strip with every cell "
            "present: geometry.rows = 1, geometry.kind = 'rectangle'."
        )
    return (cols - cols.min()).astype(float)


def fit_rates(times: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares decay rate per bin from ln|A(t)|, and the worst residual.

    A least squares over every recorded frame rather than the endpoint ratio:
    Crank-Nicolson multiplies an eigenmode by a fixed factor per step, so
    ln|A| is exactly linear in t and the fit residual is a measurement in its
    own right -- it is how a decay that is NOT a single exponential (a leaked
    second mode, a clipped state, a term that is not actually off) announces
    itself.
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


def decay_rate_curve(
    setup: Any,
    arrays: dict[str, np.ndarray],
    phi: np.ndarray,
    k2: float,
    *,
    baseline: float | np.ndarray = 0.0,
    exact_shape: bool = True,
    transient_k2: float | None = None,
    residual_label: str,
) -> Curve:
    """The measured decay rate of the mode ``phi`` against ``D_eff(E) k^2``.

    ``baseline`` is the state the mode decays TOWARDS -- zero for an absorbing
    rim, the Dirichlet value, ``gamma/beta`` for Robin -- and is a setup
    quantity, never read from the run. ``phi`` is the mode at cell centres in
    mask order. With ``exact_shape`` the prepared state must be baseline +
    mode to roundoff (an eigenmode that IS the discrete eigenvector).

    Without it the prepared state is a bump, not the mode, and what it
    carries in the NEXT mode leaks into the projection. ``transient_k2`` is
    that next mode's wavenumber squared: the leaked fraction at t = 0 is
    MEASURED from the part of the first frame orthogonal to ``phi``, and each
    bin's fit starts only once that fraction has decayed, at
    ``D_eff(E) (k2_next - k2)``, below ``LEAK_TARGET``. A bin whose transient
    outlasts the run is excluded and counted, not fitted through.
    """
    times, frames = require_frames(arrays)
    energies = np.asarray(arrays["E_bins"], dtype=float)
    gap = require_uniform_gap(setup, arrays)
    require_no_collisions(setup)

    base = np.broadcast_to(np.asarray(baseline, dtype=float), frames.shape[1:]) \
        if np.ndim(baseline) else float(baseline)
    excess = frames - base                                   # (n_t, NE, N)
    amps = excess @ phi / (phi @ phi)                        # (n_t, NE)

    first = frames[0]
    peak = float(np.max(np.abs(first)))
    if exact_shape:
        residual0 = excess[0] - np.outer(amps[0], phi)
        if peak <= 0.0 or float(np.max(np.abs(residual0))) > SHAPE_TOL * peak:
            raise ValueError(
                "The initial state is not 'baseline + mode' to roundoff, so "
                "the projected amplitude is not the amplitude of a single "
                "eigenmode and its decay is not a single exponential."
            )
    if float(np.max(np.abs(amps[0]))) <= MIN_MODE_FRACTION * max(peak, 1e-300):
        raise ValueError(
            "The initial mode amplitude is "
            f"{float(np.max(np.abs(amps[0]))):.3e} against a field peak of "
            f"{peak:.3e}: the run starts with no modulation along the mode, "
            "so the transport term has no gradient to act on. Fitting that "
            "ratio yields a finite, entirely fictitious rate."
        )

    exact = d_eff_per_bin(setup, energies) * k2
    span = float(times[-1] - times[0])
    if float(np.max(exact)) * span < MIN_DECAY_EXPONENT:
        raise ValueError(
            "The closed form predicts a decay of at most "
            f"{float(np.max(exact)) * span:.3e} e-foldings over this run "
            f"(needs {MIN_DECAY_EXPONENT:g}), so there is nothing to measure."
        )

    live = exact > 0.0
    transient_note = ""
    if transient_k2 is None:
        measured, fit_residual = fit_rates(times, amps[:, live])
    else:
        # Leaked fraction at t = 0, per bin: what the prepared state carries
        # orthogonal to the mode, relative to what it carries along it.
        along = np.outer(amps[0], phi)
        leak0 = np.linalg.norm(excess[0] - along, axis=1) / np.maximum(
            np.linalg.norm(along, axis=1), 1e-300,
        )
        gap_rate = d_eff_per_bin(setup, energies) * (float(transient_k2) - k2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_skip = np.where(
                gap_rate > 0.0,
                np.log(np.maximum(leak0, 1e-300) / LEAK_TARGET) / gap_rate,
                np.inf,
            )
        t_skip = np.maximum(t_skip, 0.0) + times[0]
        enough = np.sum(times[None, :] >= t_skip[:, None], axis=1) >= 3
        fitted = live & enough
        if not fitted.any():
            raise ValueError(
                "The prepared state's leakage into the next mode has not decayed "
                f"below {LEAK_TARGET:g} in any bin before the run ends; lengthen "
                "max_time or prepare a state closer to the mode."
            )
        rates = np.zeros(energies.size)
        worst = 0.0
        for j in np.flatnonzero(fitted):
            keep = times >= t_skip[j]
            r, res = fit_rates(times[keep], amps[keep, j][:, None])
            rates[j] = r[0]
            worst = max(worst, res)
        measured, fit_residual = rates[fitted], worst
        transient_note = (
            f" The prepared state is not the mode: it carried up to "
            f"{leak0[live].max():.2e} of its projection in the next mode at t = 0, "
            f"and each bin's fit starts once that has decayed below {LEAK_TARGET:g} "
            f"at D_eff(E)(k₂² − k²); {int((live & ~fitted).sum())} bin(s) whose "
            f"transient outlasts the run are excluded."
        )
        live = fitted
    exact = exact[live]
    x = energies[live] / gap
    ratio = measured / exact
    decay = np.abs(amps[-1, live] / amps[0, live])
    note = (
        f"{int(live.sum())} energy bins, {times.size} frames over {span:g} ns. "
        f"Mode amplitude fell to {decay.min():.4f}-{decay.max():.4f} of its "
        f"initial value. Worst departure from a single exponential over the "
        f"fitted trace: {fit_residual:.2e} of the fitted range.{transient_note} "
        f"{residual_label} "
        f"The spread of λ_sim/λ_exact across the curve is "
        f"{ratio.max() / ratio.min() - 1.0:.2e}: the energy SHAPE of the rate, "
        f"which is the content of the N₁ dressing, is reproduced far tighter "
        f"than the scale the verdict is taken on."
        + (
            f" {int((~live).sum())} bin(s) with no continuum weight carry no "
            "transport and are excluded."
            if not live.all() else ""
        )
    )
    # The comparison as a FIELD, at the last frame, for the fitted bin with
    # the most signal left: what every cell holds against what the closed
    # form -- the prepared amplitude decayed at the EXACT rate -- puts there.
    # On a rectangle this is the stencil error spread over the mode; on a
    # staircase rim it is where the discrete boundary sits.
    fitted_bins = np.flatnonzero(live)
    j_star = int(fitted_bins[np.argmax(np.abs(amps[-1, fitted_bins]))])
    t_last = float(times[-1] - times[0])
    field_analytic = np.asarray(base, dtype=float).reshape(-1, phi.size)[
        j_star if np.ndim(baseline) else 0
    ] if np.ndim(baseline) else float(baseline)
    field_analytic = field_analytic + amps[0, j_star] * phi * np.exp(
        -float(d_eff_per_bin(setup, energies)[j_star] * k2) * t_last
    )
    return Curve(
        x=x,
        y_sim=measured,
        y_analytic=exact,
        x_label="E / Δ",
        y_label="mode decay rate λ(E)  (1/ns)",
        note=note,
        field_sim=frames[-1, j_star],
        field_analytic=np.asarray(field_analytic, dtype=float),
        field_label=(
            f"f at E = {energies[j_star] / gap:.3g} Δ, t = {times[-1]:.4g} ns: "
            "simulated against the prepared mode decayed at the closed-form rate"
        ),
    )
