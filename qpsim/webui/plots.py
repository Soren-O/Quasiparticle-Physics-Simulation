"""Server-side plot rendering (matplotlib Agg → PNG) and CSV export.

One registry of named figures per run mode, rendered on demand from a
run's NPZ arrays. Uses a fixed-order colorblind-validated categorical
palette (worst adjacent CVD ΔE 24.2), a single-hue blue sequential
ramp for time families and heatmaps, and recessive chart chrome; every
multi-series figure carries a legend plus distinct markers, and every
figure has a CSV table export as the accessible fallback.

matplotlib is forced onto the Agg canvas before pyplot is imported, so
rendering works on headless CI and inside server worker threads.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

# Fixed-order categorical palette (validated; never cycled).
SERIES = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P")

# Chart chrome.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Single-hue blue sequential ramp (light → dark).
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "qpsim_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

import numpy as np  # noqa: E402


def _new_axes(
    xlabel: str, ylabel: str, title: str
) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=130, layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11)
    return fig, ax


def _finish(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _positive(y: np.ndarray) -> np.ndarray:
    """Mask non-positive values for log axes."""
    return np.where(y > 0.0, y, np.nan)




def _plot_occupation(arrays: dict[str, np.ndarray], gap: float) -> bytes:
    fig, ax = _new_axes("E / Δ", "f(E)", "Quasiparticle occupation")
    E_over_gap = arrays["E_bins"] / gap
    ax.semilogy(
        E_over_gap, _positive(arrays["f_thermal"]),
        linestyle="--", color=MUTED, linewidth=2, label="thermal (seed)",
    )
    ax.semilogy(
        E_over_gap, _positive(arrays["f"]),
        color=SERIES[0], linewidth=2, label="steady state",
    )
    _clamp_log_floor(ax, arrays["f"])
    ax.legend()
    return _finish(fig)


def _clamp_log_floor(ax: Any, reference: np.ndarray, decades_below: float = 2.0) -> None:
    """Keep a log y-axis on the decades the reference curve occupies.

    Thermal overlays reach ~1e-90 at 10Δ for mK baths; without a floor
    they stretch the axis until the physics band is one unreadable
    stripe at the top.
    """
    positive = reference[reference > 0.0]
    if positive.size:
        ax.set_ylim(bottom=float(np.min(positive)) * 10.0 ** (-decades_below))


def _plot_phonons(arrays: dict[str, np.ndarray], gap: float) -> bytes:
    fig, ax = _new_axes("ω / Δ", "n_ph(ω)", "Phonon occupation")
    ax.semilogy(
        arrays["omega_bins"] / gap, _positive(arrays["n_ph"]),
        color=SERIES[0], linewidth=2,
    )
    return _finish(fig)














# -- kinetics -------------------------------------------------------


def _plot_xqp_field(arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> bytes:
    """The device as it is shaped, coloured by local x_qp.

    Cells outside the mask are NaN and draw as blank rather than as zero: a
    hole in an annulus is absent material, not cold material, and colouring it
    dark would read as the latter.
    """
    field = np.asarray(arrays["xqp_field"], dtype=float)
    mesh_size = _mesh(summary)
    rows, cols = field.shape
    fig, ax = _new_axes("x (μm)", "y (μm)", "x_qp over the device")

    finite = field[np.isfinite(field)]
    positive = finite[finite > 0.0]
    norm: Normalize
    if positive.size and float(np.max(positive)) > float(np.min(positive)):
        norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))
        label = "x_qp (log scale)"
    else:
        # A uniform field has no decade to spread over; a log norm would need
        # vmin < vmax and there is none.
        value = float(finite[0]) if finite.size else 0.0
        norm = Normalize(vmin=min(0.0, value), vmax=max(value, value + 1e-300))
        label = "x_qp (linear; field is uniform)"

    x_edges = np.arange(cols + 1) * mesh_size
    y_edges = np.arange(rows + 1) * mesh_size
    mesh = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(field),
        cmap=SEQ_BLUE, norm=norm, shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.grid(False)
    return _finish(fig)


def _plot_geometry_mask(arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> bytes:
    """What is actually being solved on -- worth seeing before trusting a run.

    A GDS import that rasterised at too coarse a mesh, or a layer that carried
    a stray polygon, is obvious here and very hard to diagnose from a number.
    """
    mask = np.asarray(arrays["mask"], dtype=float)
    mesh_size = _mesh(summary)
    rows, cols = mask.shape
    cells = int(np.count_nonzero(mask))
    fig, ax = _new_axes(
        "x (μm)", "y (μm)",
        f"Geometry — {cells} cells at {mesh_size:g} μm",
    )
    ax.pcolormesh(
        np.arange(cols + 1) * mesh_size,
        np.arange(rows + 1) * mesh_size,
        np.ma.masked_where(mask <= 0.0, mask),
        cmap=SEQ_BLUE, vmin=0.0, vmax=1.0, shading="flat",
    )
    ax.set_aspect("equal")
    ax.grid(False)
    return _finish(fig)


def _plot_occupation_spectrum_2d(
    arrays: dict[str, np.ndarray], gap: float,
) -> bytes:
    """Occupation against energy, reduced over cells.

    The generic 0-D occupation plot cannot be reused: it wants a single f(E)
    and a thermal reference, while this mode carries f(E, cell). Showing the
    cell mean with the spread around it says what the field actually does
    without pretending there is one curve.
    """
    f = np.asarray(arrays["f_final"], dtype=float)
    energies = np.asarray(arrays["E_bins"], dtype=float) / gap
    fig, ax = _new_axes("E / Δ", "f", "Occupation, reduced over cells")
    ax.semilogy(energies, _positive(f.mean(axis=1)), lw=1.4, label="cell mean")
    if f.shape[1] > 1:
        ax.fill_between(
            energies, _positive(f.min(axis=1)), _positive(f.max(axis=1)),
            alpha=0.25, linewidth=0, label="min–max over cells",
        )
    ax.legend(loc="best", frameon=False)
    return _finish(fig)


def _plot_xqp_profile_2d(arrays: dict[str, np.ndarray]) -> bytes:
    """Cell-ordered x_qp, which is the readable view for a 1-D reduction."""
    profile = np.asarray(arrays["xqp_profile"], dtype=float)
    # Against the recorded distance where the run emits one. `x_um` holds cell
    # CENTRES, (i + 1/2)h; plotting against arange puts every point half a
    # cell to the left of where it was computed, which is invisible in a plot
    # and wrong in a fit -- and a length axis is what a reader wants anyway.
    x = arrays.get("x_um")
    if x is not None and np.asarray(x).size == profile.size:
        fig, ax = _new_axes("x (μm)", "x_qp", "x_qp per cell")
        abscissa = np.asarray(x, dtype=float)
    else:
        fig, ax = _new_axes("cell (mask order)", "x_qp", "x_qp per cell")
        abscissa = np.arange(profile.size, dtype=float)
    ax.semilogy(abscissa, _positive(profile), marker=".", lw=1.0)
    return _finish(fig)


def _convention_label(summary: dict[str, Any]) -> str:
    """The x_qp axis label, taken from the convention the run RECORDED.

    x_qp is a ratio whose value depends on a convention -- this engine's
    n_qp/(4 rho_F Delta_0) is half the paper's -- and a payload carries both
    ``obs_x_qp_mean`` and ``obs_x_qp_mean_paper``. A figure that hardcodes
    one label while plotting the other array is wrong by a factor of two
    with nothing red. So the label is read from ``summary["x_qp_convention"]``,
    which names the convention of the UNSUFFIXED arrays this figure draws,
    and a run that recorded none says so on the axis rather than borrowing a
    default.
    """
    convention = summary.get("x_qp_convention")
    if isinstance(convention, str) and convention.strip():
        return f"x_qp  [{convention.strip()}]"
    return "x_qp  [convention not recorded]"


def _draw_xqp_time_series(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
) -> tuple[Any, Any, Any]:
    """x_qp against time, with the solver residual on a twin log axis.

    Returns the figure and both axes rather than PNG bytes so a test can read
    the plotted values back: bytes cannot show which array a curve came from,
    and this figure exists to show ``obs_x_qp_mean`` itself, not a reduction
    of something adjacent to it. Values go on the line untouched for the same
    reason; the log axis clips non-positives at draw time.
    """
    t = np.asarray(arrays["snap_t_ns"], dtype=float)
    fig, ax = _new_axes("t (ns)", _convention_label(summary), "x_qp over time")
    ax.plot(
        t, np.asarray(arrays["obs_x_qp_mean"], dtype=float),
        color=SERIES[0], marker=MARKERS[0], markersize=3.5, linewidth=1.8,
        label="cell mean",
    )
    if "obs_x_qp_max" in arrays:
        ax.plot(
            t, np.asarray(arrays["obs_x_qp_max"], dtype=float),
            color=SERIES[1], marker=MARKERS[1], markersize=3.5, linewidth=1.4,
            label="cell max",
        )
    # The reference the run is measured AGAINST: "x_qp = 1.1e-5" is unreadable
    # until the reader knows how far above thermal it sits. It is in the same
    # convention as the curves -- execute.py computes it with the same
    # qp_fraction and delta_0.
    thermal = summary.get("x_qp_thermal")
    if isinstance(thermal, (int, float)) and np.isfinite(thermal) and thermal > 0.0:
        ax.axhline(
            float(thermal), color=MUTED, linestyle=":", linewidth=1.4,
            label="thermal",
        )
    ax.set_yscale("log")

    residual = None
    rate = arrays.get("snap_max_rate")
    if rate is not None and np.asarray(rate).shape == t.shape:
        residual = ax.twinx()
        residual.plot(
            t, _positive(np.asarray(rate, dtype=float)),
            color=SERIES[5], linestyle="--", linewidth=1.3,
            label="max rate (residual)",
        )
        residual.set_yscale("log")
        residual.set_ylabel("max |df/dt| (1/ns)", color=INK)
        residual.tick_params(colors=MUTED, labelcolor=INK)
        for spine in residual.spines.values():
            spine.set_color(BASELINE)
    handles, labels = ax.get_legend_handles_labels()
    if residual is not None:
        more, more_labels = residual.get_legend_handles_labels()
        handles, labels = handles + more, labels + more_labels
    ax.legend(handles, labels, loc="best", fontsize=8)
    return fig, ax, residual


def _plot_xqp_time_series(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
) -> bytes:
    fig, _ax, _residual = _draw_xqp_time_series(arrays, summary)
    return _finish(fig)


# -- m25_junction -----------------------------------------------------


def _plot_m25_mu(arrays: dict[str, np.ndarray]) -> bytes:
    fig, ax = _new_axes("T (mK)", "μ_α / Δ_L", "Sub-band chemical potentials")
    labels = [
        ("mu_L_over_Delta_L", "μ_L"),
        ("mu_Rgt_over_Delta_L", "μ_R>"),
        ("mu_Rlt_over_Delta_L", "μ_R<"),
    ]
    for i, (key, label) in enumerate(labels):
        ax.plot(
            arrays["T_mK"], arrays[key],
            color=SERIES[i], marker=MARKERS[i], markersize=4.5,
            linewidth=1.8, label=label,
        )
    ax.legend()
    return _finish(fig)


def _plot_m25_densities(arrays: dict[str, np.ndarray]) -> bytes:
    fig, ax = _new_axes("T (mK)", "x_α", "Sub-band quasiparticle densities")
    labels = [("x_L", "x_L"), ("x_Rgt", "x_R>"), ("x_Rlt", "x_R<")]
    for i, (key, label) in enumerate(labels):
        ax.semilogy(
            arrays["T_mK"], _positive(arrays[key]),
            color=SERIES[i], marker=MARKERS[i], markersize=4.5,
            linewidth=1.8, label=label,
        )
    ax.legend()
    return _finish(fig)


def _plot_m25_p1(arrays: dict[str, np.ndarray]) -> bytes:
    fig, ax = _new_axes("T (mK)", "p₁", "Qubit excited-state probability")
    ax.semilogy(
        arrays["T_mK"], _positive(arrays["p_1"]),
        color=SERIES[0], marker=MARKERS[0], markersize=4.5, linewidth=1.8,
    )
    return _finish(fig)


# -- analytic comparison ----------------------------------------------


def _plot_analytic_comparison(
    arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> bytes:
    """Simulated curve against its closed form, with the residual beneath.

    Two panels rather than one. The overlay answers "do these lie on top of
    each other", which at a glance is nearly always yes; the residual answers
    "by how much, and is the disagreement structured", which is the question
    that actually distinguishes a discretisation floor from a wrong term. A
    residual that sits flat under the tolerance line is convergence; one that
    grows with the abscissa, or changes sign in a pattern, is a bug — and only
    the second panel shows the difference.
    """
    bench = summary.get("benchmark") or {}
    x = arrays["bench_x"]
    sim = np.atleast_2d(arrays["bench_sim"])
    ana = np.atleast_2d(arrays["bench_analytic"])
    labels = list(bench.get("series_labels") or [])

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 5.8), dpi=130, layout="constrained",
        sharex=True, height_ratios=(2.1, 1.0),
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axr):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for spine in a.spines.values():
            spine.set_color(BASELINE)
        a.tick_params(colors=MUTED, labelcolor=INK)

    if bench.get("log_x"):
        ax.set_xscale("log")
        axr.set_xscale("log")
    if bench.get("log_y"):
        ax.set_yscale("log")

    n = sim.shape[0]
    # One series is the common case and reads best with an explicit legend
    # naming which line is which. Many series would drown in one, so they are
    # distinguished by the sequential ramp and labelled by the axis instead.
    for i in range(n):
        colour = SERIES[i % len(SERIES)] if n <= len(SERIES) else SEQ_BLUE(
            0.25 + 0.7 * (i / max(n - 1, 1))
        )
        label = labels[i] if i < len(labels) else (f"series {i + 1}" if n > 1 else "")
        y_s = _positive(sim[i]) if bench.get("log_y") else sim[i]
        y_a = _positive(ana[i]) if bench.get("log_y") else ana[i]
        ax.plot(
            x, y_a, color=colour, linewidth=2.0, zorder=2,
            label=f"{label} — analytic".strip(" —") if label else "analytic",
        )
        # Simulated points sit ON the analytic line as open markers: a filled
        # marker hides the line it is meant to be compared against.
        ax.plot(
            x, y_s, linestyle="none", marker=MARKERS[i % len(MARKERS)],
            markersize=5, markerfacecolor="none", markeredgecolor=colour,
            markeredgewidth=1.3, zorder=3,
            label=f"{label} — simulated".strip(" —") if label else "simulated",
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            peak = float(np.max(np.abs(ana[i]))) if ana[i].size else 0.0
            denom = (
                np.full_like(ana[i], peak)
                if bench.get("metric") == "scale"
                else np.abs(ana[i])
            )
            rel = np.abs(sim[i] - ana[i]) / np.where(denom > 0.0, denom, np.nan)
        axr.plot(x, rel, color=colour, linewidth=1.6, marker=None)

    tol = bench.get("rel_tol")
    if isinstance(tol, (int, float)) and tol > 0:
        axr.axhline(
            float(tol), color=SERIES[5], linewidth=1.4, linestyle="--",
            label=f"tolerance {float(tol):.0e}",
        )
        axr.legend(loc="best", fontsize=8)
    axr.set_yscale("log")

    ax.set_ylabel(bench.get("y_label", ""), color=INK)
    axr.set_ylabel("relative error", color=INK)
    axr.set_xlabel(bench.get("x_label", ""), color=INK)
    tier = bench.get("tier", "")
    verdict = bench.get("verdict", "")
    ax.set_title(
        f"{bench.get('title', 'Analytic comparison')}  [{tier}, {verdict}]",
        color=INK, fontsize=11,
    )
    if n <= 3:
        ax.legend(fontsize=8)
    return _finish(fig)


def _frame_axes(arrays, summary, title):
    """A device-shaped axes in microns, with the mask's aspect preserved."""
    mesh = _mesh(summary)
    rows, cols = np.asarray(arrays["mask"]).shape
    fig, ax = _new_axes("x (μm)", "y (μm)", title)
    ax.grid(False)
    ax.set_aspect("equal")
    return fig, ax, mesh, (0.0, cols * mesh, 0.0, rows * mesh)


class _UniformNorm(Normalize):
    """The colour scale of a field that is ONE number in every frame.

    A norm needs vmin < vmax, so a uniform stack gets a synthetic spread;
    this subclass marks that the spread is synthetic, so the colourbar can
    say "uniform" and print the one value instead of printing the spread.
    """


def _frame_norm(stack: np.ndarray) -> Any:
    """One colour scale across EVERY frame.

    Renormalising per frame is the classic way to make an animation lie: a
    field decaying by three decades looks perfectly steady because each frame
    is rescaled to its own maximum. The scale is therefore global, and the
    colourbar means the same thing in every frame.
    """
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return _UniformNorm(vmin=lo, vmax=lo + max(abs(lo) * 1e-6, 1e-30))
    return Normalize(vmin=lo, vmax=hi)


def _draw_frame(arrays, summary, values, title, label, norm) -> bytes:
    from qpsim.grid.spatial_grid import reconstruct_field
    mask = np.asarray(arrays["mask"]).astype(bool)
    fig, ax, _mesh, extent = _frame_axes(arrays, summary, title)
    image = ax.imshow(
        reconstruct_field(mask, values), origin="lower", extent=extent,
        cmap=SEQ_BLUE, norm=norm, interpolation="nearest",
    )
    bar = fig.colorbar(image, ax=ax, label=label)
    if isinstance(norm, _UniformNorm) and norm.vmin is not None:
        # Left alone, the bar prints the synthetic spread -- "+1.8e2" with
        # ticks at 0.000025 -- which reads as structure in a field that has
        # none. A pinned gap is the everyday case, not a corner.
        value = float(norm.vmin)
        bar.set_ticks([value])
        bar.set_ticklabels([f"{value:.6g}"])
        bar.set_label(f"{label} — uniform field")
    return _finish(fig)


def _plot_field_frame(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int = 0,
) -> bytes:
    """Quasiparticle density over the device at one recorded time."""
    stack = arrays["snap_xqp_profile"]
    t = float(arrays["snap_t_ns"][frame])
    return _draw_frame(
        arrays, summary, stack[frame],
        f"x_qp over the device — t = {t:.4g} ns "
        f"(frame {frame + 1} of {stack.shape[0]})",
        "x_qp [n_qp/(4 rho_F Delta_0)]", _frame_norm(stack),
    )


def _plot_energy_map(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
    frame: int = 0, energy: int = 0,
) -> bytes:
    """Occupation at ONE energy, over the device.

    The integrated density says where the quasiparticles are; this says where
    the HOT ones are, which is a different map and the reason for solving an
    energy-resolved problem in the first place.
    """
    stack = arrays["snap_f"]
    t = float(arrays["snap_t_ns"][frame])
    gap = float(summary.get("gap_ueV", 1.0)) or 1.0
    e_over_gap = float(arrays["E_bins"][energy]) / gap
    return _draw_frame(
        arrays, summary, stack[frame, energy],
        f"occupation f at E = {e_over_gap:.4g} Δ — t = {t:.4g} ns",
        "f (occupation, not density)", _frame_norm(stack[:, energy]),
    )


def _plot_phonon_frame(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
    frame: int = 0, omega: int = 0,
) -> bytes:
    """Phonon occupation at one frequency, over the device."""
    stack = arrays["snap_n_ph"]
    t = float(arrays["snap_t_ns"][frame])
    # The run records the frequency axis its phonon populations are indexed
    # by, so name the frequency rather than its position. "bin 60" is not a
    # physical quantity and cannot be compared between two runs on different
    # grids -- the same index is a different frequency. Falls back to the
    # index for payloads written before the axis was emitted.
    axis = arrays.get("snap_omega_bins")
    if axis is not None and omega < np.asarray(axis).size:
        where = f"ω = {float(np.asarray(axis)[omega]):.4g} μeV"
    else:
        where = f"ω bin {omega}"
    return _draw_frame(
        arrays, summary, stack[frame, omega],
        f"n_ph at {where} — t = {t:.4g} ns",
        "n_ph", _frame_norm(stack[:, omega]),
    )


def _plot_gap_frame(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int = 0,
) -> bytes:
    """The gap over the device at one recorded time.

    Recorded per frame because the gap follows the occupation where the run
    solves it self-consistently. A run with a pinned gap draws the same map
    in every frame, which is a true statement about that run and cheaper
    than a rule deciding when the figure is "interesting".
    """
    stack = np.asarray(arrays["snap_gap"], dtype=float)
    t = float(arrays["snap_t_ns"][frame])
    return _draw_frame(
        arrays, summary, stack[frame],
        f"Δ over the device — t = {t:.4g} ns "
        f"(frame {frame + 1} of {stack.shape[0]})",
        "Δ (μeV)", _frame_norm(stack),
    )


def _phonon_occupation_integral(
    arrays: dict[str, np.ndarray], frame: int,
) -> np.ndarray:
    """``∫ n_ph dω`` per cell at one frame, on the lattice the run recorded.

    Trapezoid over the recorded centres: the phonon state is a POINT SAMPLE
    on its lattice, and that lattice is a clustered union of the difference
    and sum lattices rather than a uniform grid, so neither a bin-count sum
    nor a fixed dω is the integral. The result is an OCCUPATION integrated
    over frequency, in μeV, and deliberately not an energy: weighting by ω
    and a mode density is the quantity this repo once came within inches of
    double counting, and it must not be drawn until someone establishes
    whether the ω lattice already carries a mode density.
    """
    n_ph = np.asarray(arrays["snap_n_ph"][frame], dtype=float)  # (Nω, Ncells)
    omega = np.asarray(arrays["snap_omega_bins"], dtype=float)
    if omega.shape != (n_ph.shape[0],):
        raise ValueError(
            f"the recorded phonon lattice has {omega.shape[0]} frequencies but "
            f"the frame holds {n_ph.shape[0]}; integrating would pair "
            "populations with the wrong frequencies."
        )
    return np.trapezoid(n_ph, omega, axis=0)


def _plot_phonon_occupation_frame(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int = 0,
) -> bytes:
    """Frequency-integrated phonon occupation over the device at one time."""
    frames = int(np.asarray(arrays["snap_t_ns"]).shape[0])
    t = float(arrays["snap_t_ns"][frame])
    # Every frame's integral, because the colour scale is global (see
    # _frame_norm) and needs the run's range, not this frame's.
    integrals = np.stack(
        [_phonon_occupation_integral(arrays, k) for k in range(frames)]
    )
    return _draw_frame(
        arrays, summary, integrals[frame],
        f"∫ n_ph dω over the device — t = {t:.4g} ns "
        f"(frame {frame + 1} of {frames})",
        "∫ n_ph dω (μeV) — occupation, no mode density",
        _frame_norm(integrals),
    )


# -- pre-run preview --------------------------------------------------
#
# The same figures a finished run gets, drawn from a mask and a seed before
# anything runs. They take plain arrays rather than a run's payload so the
# preview endpoint and the run detail cannot drift into two pictures of one
# device.


# Colour per boundary-condition kind on the rim overlay. Fixed, so the same
# kind is the same colour in every preview.
_CONDITION_COLOURS = {
    "reflective": MUTED,
    "absorbing": SERIES[5],
    "dirichlet": SERIES[2],
    "neumann": SERIES[1],
    "robin": SERIES[4],
}
# Where a segment's label sits: just outside the material, along the
# outward normal. Normals are named in MASK-ROW order -- "up" faces
# decreasing row -- and row 0 is drawn at y = 0, so "up" points to -y here.
_NORMAL_OFFSETS = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}


def render_mask_png(
    mask: np.ndarray,
    mesh_size_um: float,
    edges: Sequence[Any] | None = None,
    conditions: dict[str, Any] | None = None,
) -> bytes:
    """The geometry figure, from a mask rather than a run's arrays.

    With ``edges`` (the geometry's segments) each rim segment is drawn in the
    colour of its condition and labelled with its id, which is the picture a
    per-edge override needs: on an annulus "right" names BOTH rims, and only
    a labelled figure shows which id is the inner one.
    """
    if not edges:
        return _plot_geometry_mask(
            {"mask": np.asarray(mask, dtype=float)}, {"mesh_size_um": float(mesh_size_um)},
        )
    field = np.asarray(mask, dtype=float)
    mesh = float(mesh_size_um)
    rows, cols = field.shape
    cells = int(np.count_nonzero(field))
    fig, ax = _new_axes("x (μm)", "y (μm)", f"Geometry — {cells} cells at {mesh:g} μm")
    ax.pcolormesh(
        np.arange(cols + 1) * mesh, np.arange(rows + 1) * mesh,
        np.ma.masked_where(field <= 0.0, field),
        cmap=SEQ_BLUE, vmin=0.0, vmax=2.0, shading="flat",
    )
    seen: dict[str, Any] = {}
    for edge in edges:
        kind = "reflective"
        if conditions and edge.edge_id in conditions:
            kind = str(getattr(conditions[edge.edge_id], "kind", "reflective"))
        colour = _CONDITION_COLOURS.get(kind, INK)
        (line,) = ax.plot(
            [edge.x0 * mesh, edge.x1 * mesh], [edge.y0 * mesh, edge.y1 * mesh],
            color=colour, linewidth=3.0, solid_capstyle="butt",
        )
        seen.setdefault(kind, line)
        dx, dy = _NORMAL_OFFSETS.get(edge.normal, (0.0, 0.0))
        ax.annotate(
            edge.edge_id,
            ((edge.x0 + edge.x1) / 2.0 * mesh + 0.35 * mesh * dx,
             (edge.y0 + edge.y1) / 2.0 * mesh + 0.35 * mesh * dy),
            ha="center", va="center", fontsize=7, color=colour,
            rotation=90 if edge.normal in ("left", "right") else 0,
        )
    ax.legend(
        list(seen.values()), list(seen), title="rim condition",
        loc="upper right", fontsize=8, title_fontsize=8,
    )
    ax.set_aspect("equal")
    ax.grid(False)
    ax.margins(0.08)
    return _finish(fig)


def render_cell_field_png(
    mask: np.ndarray, mesh_size_um: float, values: np.ndarray, title: str, label: str,
) -> bytes:
    """One per-cell field over the device -- a seed, before any run."""
    values = np.asarray(values, dtype=float)
    return _draw_frame(
        {"mask": np.asarray(mask)}, {"mesh_size_um": float(mesh_size_um)},
        values, title, label, _frame_norm(values[None, :]),
    )


def render_phonon_seed_png(
    omega: np.ndarray, seed: np.ndarray, bath: np.ndarray, gap: float,
) -> bytes:
    """The phonon population a run starts from, against the bath it departs from."""
    x = np.asarray(omega, dtype=float) / float(gap)
    fig, ax = _new_axes("ω / Δ", "n_ph(ω)", "Phonon seed against the bath")
    ax.semilogy(
        x, _positive(np.asarray(bath, dtype=float)),
        linestyle="--", color=MUTED, linewidth=2, label="bath (thermal)",
    )
    ax.semilogy(
        x, _positive(np.asarray(seed, dtype=float)),
        color=SERIES[0], linewidth=2, label="seed",
    )
    _clamp_log_floor(ax, np.asarray(seed, dtype=float))
    ax.legend()
    return _finish(fig)


# -- formulas ---------------------------------------------------------


class FormulaRenderError(ValueError):
    """A formula matplotlib's mathtext cannot typeset."""


def render_formula(latex: str, *, fontsize: float = 15.0, dpi: int = 200) -> bytes:
    """Typeset one equation to a transparent PNG.

    Rendered on the server with matplotlib's mathtext, which is already the
    typesetting engine behind every axis label in this app -- so a formula
    costs no new dependency and no build step, matching the frontend's
    no-build design. The cost is mathtext's LaTeX SUBSET: no ``	ext``, no
    ``align`` environments, no ``oxed``. That is a real constraint and it
    is the right one here, because this string is a HEADLINE. A statement
    that needs an align environment is a derivation, and belongs in prose
    where it can be read, not in a one-line banner.

    Raises :class:`FormulaRenderError` rather than emitting a broken image:
    the interface falls back to showing the source, which is honest, where a
    half-typeset equation would not be.
    """
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)
    fig.text(0.0, 0.0, f"${latex}$", fontsize=fontsize, color=INK)
    buf = io.BytesIO()
    try:
        fig.savefig(
            buf, format="png", dpi=dpi, transparent=True,
            bbox_inches="tight", pad_inches=0.05,
        )
    except (ValueError, RuntimeError) as exc:
        raise FormulaRenderError(str(exc).strip().splitlines()[-1]) from exc
    finally:
        plt.close(fig)
    return buf.getvalue()


# -- registry ---------------------------------------------------------


def _required_scale(summary: dict[str, Any], key: str, what: str) -> float:
    """A physical scale the figure cannot be drawn without.

    These used to default to 1.0, which is not a neutral choice: it is a
    specific claim about the device, silently substituted for a missing one.
    Every energy axis is in units of the gap, so a missing gap_ueV drew a
    correctly-labelled E/Delta axis that was wrong by a factor of ~180; a
    missing mesh_size_um did the same to the micron axes and the scale bar.
    Refuse instead -- a figure that fails is recoverable, one that lies is not.
    """
    value = summary.get(key)
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(
            f"this run reports no usable {key} ({value!r}), and {what} is the "
            "scale of the axis being drawn, so the figure would be mislabelled "
            "rather than merely imprecise."
        )
    return float(value)


def _gap(summary: dict[str, Any]) -> float:
    return _required_scale(summary, "gap_ueV", "the gap")


def _mesh(summary: dict[str, Any]) -> float:
    return _required_scale(summary, "mesh_size_um", "the cell size")


@dataclass(frozen=True)
class _PlotSpec:
    """One named figure: its renderer plus an optional required array.

    ``params`` makes the entry a FAMILY of figures rather than one: each
    named integer parameter is bounded by the length of the array it maps to,
    so a caller can ask for frame 7 or energy bin 120 and the server can
    reject an index the run does not have instead of raising deep in numpy.
    """

    render: Callable[..., bytes]
    requires: str | tuple[str, ...] | None = None
    params: dict[str, tuple[str, int]] = field(default_factory=dict)


def _required(requires: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """The arrays an entry needs -- ALL of them, or it is not offered.

    A tuple is for a figure built from two arrays that are only meaningful
    together: phonon populations without the lattice they were recorded on
    is the bin-count defect again, and a figure offered for a run that then
    cannot render it 404s after the user has already clicked.
    """
    if requires is None:
        return ()
    return (requires,) if isinstance(requires, str) else tuple(requires)


def _plot_occupation_either_shape(arrays, summary):
    """`occupation` for either strategy of the merged mode.

    steady_state carries f:(NE,) from the 0-D solver; time_march carries
    f_final:(NE, Ncells). Dispatch on the array present rather than on the
    strategy string: the payload is what the renderer can actually verify, and
    a strategy recorded in the summary could disagree with what was stored.
    """
    if "f_final" in arrays:
        return _plot_occupation_spectrum_2d(arrays, _gap(summary))
    return _plot_occupation(arrays, _gap(summary))


# Single source of truth per mode: the listing endpoint and the render
# dispatch both read this table, so a figure can't be listed without
# being renderable (or vice versa).
_PLOTS: dict[str, dict[str, _PlotSpec]] = {
    # ONE mode, TWO payload shapes. `strategy="steady_state"` runs the 0-D
    # solver and carries f:(NE,) with n_ph; `strategy="time_march"` carries
    # f_final:(NE, Ncells) on a mask. Every entry therefore declares the array
    # it needs, or it gets offered for a run that cannot render it -- which is
    # exactly what happened when the 0-D catalogue cases moved onto this mode:
    # four figures listed, four KeyErrors, and the catalogue gate could not see
    # it because it records arrays and never renders one.
    "kinetics": {
        "xqp_field": _PlotSpec(
            lambda a, s: _plot_xqp_field(a, s), requires="xqp_field",
        ),
        # Figure families: one image per recorded frame, so the interface can
        # scrub through a run instead of showing only where it ended.
        "field_over_time": _PlotSpec(
            _plot_field_frame, requires="snap_xqp_profile",
            params={"frame": ("snap_t_ns", 0)},
        ),
        "gap_over_time": _PlotSpec(
            _plot_gap_frame, requires="snap_gap",
            params={"frame": ("snap_gap", 0)},
        ),
        "energy_resolved_map": _PlotSpec(
            _plot_energy_map, requires="snap_f",
            params={"frame": ("snap_f", 0), "energy": ("snap_f", 1)},
        ),
        "phonon_field_over_time": _PlotSpec(
            _plot_phonon_frame, requires="snap_n_ph",
            params={"frame": ("snap_n_ph", 0), "omega": ("snap_n_ph", 1)},
        ),
        "phonon_occupation_map": _PlotSpec(
            _plot_phonon_occupation_frame,
            requires=("snap_n_ph", "snap_omega_bins"),
            params={"frame": ("snap_n_ph", 0)},
        ),
        "geometry": _PlotSpec(
            lambda a, s: _plot_geometry_mask(a, s), requires="mask",
        ),
        "xqp_profile": _PlotSpec(
            lambda a, s: _plot_xqp_profile_2d(a), requires="xqp_profile",
        ),
        # The run AS A TIME SERIES. The frames above answer "where"; this one
        # answers "when" -- when the density settles, when the residual
        # stops falling -- which no single frame can.
        "xqp_over_time": _PlotSpec(
            _plot_xqp_time_series, requires=("snap_t_ns", "obs_x_qp_mean"),
        ),
        # One name for the question a reader is actually asking -- "show me the
        # occupation" -- dispatching on which field the strategy produced.
        # Splitting it into two names would make the figure a run offers depend
        # on a solver choice rather than on what is being looked at.
        "occupation": _PlotSpec(_plot_occupation_either_shape),
        # 0-D only: the spatial route has a phonon map per cell, which is a
        # different figure (phonon_field_over_time above).
        "phonons": _PlotSpec(
            lambda a, s: _plot_phonons(a, _gap(s)), requires="n_ph",
        ),
    },
    "m25_junction": {
        "chemical_potentials": _PlotSpec(lambda a, s: _plot_m25_mu(a)),
        "densities": _PlotSpec(lambda a, s: _plot_m25_densities(a)),
        "qubit_p1": _PlotSpec(lambda a, s: _plot_m25_p1(a)),
    },
}

# Every mode can carry an analytic comparison and the figure is the same one in
# all of them: a benchmark either wrote its bench_* arrays into the payload or
# it did not, and `requires` decides whether the figure is offered. Registering
# it in a loop keeps that a single fact instead of five copies free to drift.
for _mode_plots in _PLOTS.values():
    _mode_plots["analytic_comparison"] = _PlotSpec(
        _plot_analytic_comparison, requires="bench_analytic"
    )


def available_plots(mode: str, array_names: Collection[str]) -> list[str]:
    """Figure names renderable for this mode given the stored arrays."""
    return [
        name
        for name, spec in _PLOTS.get(mode, {}).items()
        if all(needed in array_names for needed in _required(spec.requires))
    ]


def plot_parameter_arrays(mode: str, name: str) -> set[str]:
    """Which stored arrays a figure family reads its index bounds from."""
    spec = _PLOTS.get(mode, {}).get(name)
    return {array for array, _axis in spec.params.values()} if spec else set()


def plot_parameters(
    mode: str, name: str, shapes: dict[str, tuple[int, ...]]
) -> dict[str, int]:
    """``{parameter: count}`` for a figure family, or ``{}`` for a single figure.

    Takes SHAPES rather than arrays so a caller on a polling path does not
    have to decompress a run's payload to draw a scrubber. The counts come
    from the stored data either way, so the control cannot offer an index the
    run does not have.
    """
    spec = _PLOTS.get(mode, {}).get(name)
    if spec is None:
        return {}
    counts: dict[str, int] = {}
    for param, (array_name, axis) in spec.params.items():
        shape = shapes.get(array_name)
        if shape is None or len(shape) <= axis:
            return {}
        counts[param] = int(shape[axis])
    return counts


def render_plot(
    mode: str,
    name: str,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
    params: dict[str, int] | None = None,
) -> bytes:
    """Render one named figure to PNG bytes; raises KeyError for unknown names."""
    spec = _PLOTS.get(mode, {}).get(name)
    if spec is None or any(needed not in arrays for needed in _required(spec.requires)):
        raise KeyError(f"No plot named {name!r} for mode {mode!r}.")
    if not spec.params:
        return spec.render(arrays, summary)
    counts = plot_parameters(
        mode, name, {k: tuple(v.shape) for k, v in arrays.items()},
    )
    chosen: dict[str, int] = {}
    for param, count in counts.items():
        value = int((params or {}).get(param, 0))
        if not 0 <= value < count:
            raise KeyError(
                f"{name!r} {param}={value} is outside the run's range "
                f"0..{count - 1}."
            )
        chosen[param] = value
    return spec.render(arrays, summary, **chosen)


# -- CSV export -------------------------------------------------------
#
# A table is a FRAME of a run when the run recorded frames: ``frame=None`` is
# the endpoint the run finished on (the arrays every run writes), ``frame=k``
# is the k-th recorded snapshot. Every table that could hold more than one
# time carries a ``t_ns`` column naming the one it holds, so the file states
# its own time instead of leaving that to the URL it was fetched from.

CsvBuilder = Callable[[dict[str, np.ndarray], dict[str, Any], int | None], str]


def _csv_from_columns(header: Iterable[str], columns: list[np.ndarray]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(list(header))
    for row in zip(*columns, strict=True):
        writer.writerow([f"{v:.10g}" for v in row])
    return out.getvalue()


def _frame_index(arrays: dict[str, np.ndarray], frame: int | None) -> int | None:
    """Validate a requested frame against what the run recorded.

    Raises KeyError -- the server's 404 -- rather than IndexError deep in
    numpy, for the same reason the figure families do: a scrubber can outrun
    a run. ``None`` is the endpoint and always valid.
    """
    if frame is None:
        return None
    if "snap_t_ns" not in arrays:
        raise KeyError("this run recorded no frames, so there is no frame to select.")
    count = int(np.asarray(arrays["snap_t_ns"]).shape[0])
    if not 0 <= int(frame) < count:
        raise KeyError(f"frame={frame} is outside the run's range 0..{count - 1}.")
    return int(frame)


def _no_frames(name: str, frame: int | None) -> None:
    """A table with no frame axis refuses a frame rather than ignoring it."""
    if frame is not None:
        raise KeyError(f"{name!r} has no frame axis; 'frame' does not apply.")


def _frame_time(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None,
) -> float:
    """The simulation time a table holds, in ns; NaN if the run did not say."""
    if frame is not None:
        return float(arrays["snap_t_ns"][frame])
    total = summary.get("total_time_ns")
    return float(total) if isinstance(total, (int, float)) else float("nan")


def _csv_ss_occupation(arrays: dict[str, np.ndarray]) -> str:
    return _csv_from_columns(
        ["E_ueV", "f", "f_thermal"],
        [arrays["E_bins"], arrays["f"], arrays["f_thermal"]],
    )


def _csv_ss_phonons(arrays: dict[str, np.ndarray]) -> str:
    return _csv_from_columns(["omega_ueV", "n_ph"], [arrays["omega_bins"], arrays["n_ph"]])


def _csv_m25_sweep(arrays: dict[str, np.ndarray]) -> str:
    keys = [
        "T_mK", "x_L", "x_Rgt", "x_Rlt", "p_1", "residual_Hz",
        "mu_L_over_Delta_L", "mu_Rgt_over_Delta_L", "mu_Rlt_over_Delta_L",
    ]
    return _csv_from_columns(keys, [arrays[k] for k in keys])


def _cell_coordinates(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres in mask order, in cell units.

    Mask order is row-major over ``mask.nonzero()``, which is exactly what the
    solver uses, so this is the mapping from a result column back to a place
    on the device.
    """
    mask = np.asarray(arrays["mask"]).astype(bool)
    rows, cols = np.nonzero(mask)
    return cols.astype(float), rows.astype(float)


def _cell_columns(
    arrays: dict[str, np.ndarray], summary: dict[str, Any],
) -> tuple[list[str], list[np.ndarray]]:
    """``col, row`` in mask order, plus ``x_um, y_um`` when the mesh is known.

    Cell CENTRES, (i + 1/2)h -- the convention ``x_um`` is emitted in on a
    strip and the figures' extents use, where i*h would put every point half
    a cell off. The micron columns are omitted rather than guessed when the
    run did not record its mesh size: ``col``/``row`` stay exact, and a
    length axis invented from a default is the mislabelling this module
    refuses in ``_required_scale``.
    """
    col, row = _cell_coordinates(arrays)
    header, cols = ["col", "row"], [col, row]
    mesh = summary.get("mesh_size_um")
    if isinstance(mesh, (int, float)) and np.isfinite(mesh) and mesh > 0.0:
        header += ["x_um", "y_um"]
        cols += [(col + 0.5) * float(mesh), (row + 0.5) * float(mesh)]
    return header, cols


def _csv_kinetics_profile(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """Per-cell x_qp with the cell's place on the device.

    Without this a 2-D run's numbers cannot leave the browser at all: the
    result was four PNGs and a handful of summary scalars, so replotting,
    fitting, or comparing against another tool was impossible.
    """
    k = _frame_index(arrays, frame)
    header, cols = _cell_columns(arrays, summary)
    n = int(cols[0].size)
    profile = np.asarray(
        arrays["xqp_profile"] if k is None else arrays["snap_xqp_profile"][k],
        dtype=float,
    )
    header += ["t_ns", "x_qp", "x_qp_paper"]
    cols += [np.full(n, _frame_time(arrays, summary, k)), profile, 2.0 * profile]
    gap = arrays.get("gap_per_cell") if k is None else arrays["snap_gap"][k]
    if gap is not None:
        header.append("gap_ueV")
        cols.append(np.asarray(gap, dtype=float))
    return _csv_from_columns(header, cols)


def _csv_kinetics_occupation(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """f(E) per cell, one column per cell, at the endpoint or a recorded frame."""
    k = _frame_index(arrays, frame)
    col, row = _cell_coordinates(arrays)
    f = np.asarray(arrays["f_final"] if k is None else arrays["snap_f"][k], dtype=float)
    energies = np.asarray(arrays["E_bins"], dtype=float)
    return _csv_from_columns(
        ["E_ueV", "t_ns",
         *[f"f_col={int(c)}_row={int(r)}" for c, r in zip(col, row, strict=True)]],
        [energies, np.full(energies.size, _frame_time(arrays, summary, k)),
         *[f[:, j] for j in range(f.shape[1])]],
    )


def _csv_kinetics_occupation_either_shape(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """`occupation.csv` for either strategy -- see the plot of the same name."""
    if "f_final" in arrays:
        return _csv_kinetics_occupation(arrays, summary, frame)
    _frame_index(arrays, frame)
    return _csv_ss_occupation(arrays)


def _csv_kinetics_phonons(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """n_ph(ω) per cell at one recorded frame, one column per cell.

    The spatial route records phonons only as frames -- there is no endpoint
    phonon array -- so the endpoint table is the LAST recorded frame, and the
    ``t_ns`` column says which time that is rather than implying max_time.
    The frequency column is the lattice the run recorded, never one rebuilt
    from the setup (see T3SpatialBackend.phonon_frequency_axis).
    """
    k = _frame_index(arrays, frame)
    if k is None:
        k = int(np.asarray(arrays["snap_t_ns"]).shape[0]) - 1
    n_ph = np.asarray(arrays["snap_n_ph"][k], dtype=float)  # (Nω, Ncells)
    omega = np.asarray(arrays["snap_omega_bins"], dtype=float)
    col, row = _cell_coordinates(arrays)
    return _csv_from_columns(
        ["omega_ueV", "t_ns",
         *[f"n_ph_col={int(c)}_row={int(r)}" for c, r in zip(col, row, strict=True)]],
        [omega, np.full(omega.size, float(arrays["snap_t_ns"][k])),
         *[n_ph[:, j] for j in range(n_ph.shape[1])]],
    )


def _csv_kinetics_phonons_either_shape(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """`phonons.csv` for either payload shape.

    The 0-D solver writes ``n_ph`` on ``omega_bins``; the spatial time march
    writes ``snap_n_ph`` frames on ``snap_omega_bins`` when its phonon sector
    is dynamic. Dispatch on the array present, as the figures do. Before this
    the table was keyed on ``n_ph`` alone, so the phonon sector on a geometry
    was view-only: drawable, never downloadable.
    """
    if "snap_n_ph" in arrays:
        return _csv_kinetics_phonons(arrays, summary, frame)
    _frame_index(arrays, frame)
    return _csv_ss_phonons(arrays)


def _csv_kinetics_time_series(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None = None,
) -> str:
    """Recorded frames reduced to per-time observables.

    Both x_qp conventions, as the profile table carries them. ``n_ph_mean``
    is the UNWEIGHTED mean of n_ph over lattice points and cells -- one
    number for "how much phonon occupation is there", present only when the
    phonon sector was dynamic; it is not an energy and carries no mode
    density (see the occupation map). ``Q_i`` is the readout probe as a
    time series where the run computed one.
    """
    _no_frames("time_series", frame)
    header = ["t_ns", "max_rate_per_ns", "x_qp_mean", "x_qp_max"]
    cols = [arrays["snap_t_ns"], arrays["snap_max_rate"],
            arrays["obs_x_qp_mean"], arrays["obs_x_qp_max"]]
    for name in ("obs_x_qp_mean_paper", "obs_x_qp_max_paper"):
        if name in arrays:
            header.append(name.removeprefix("obs_"))
            cols.append(arrays[name])
    if "snap_n_ph" in arrays:
        header.append("n_ph_mean")
        cols.append(np.asarray(arrays["snap_n_ph"], dtype=float).mean(axis=(1, 2)))
    if "obs_Q_i" in arrays:
        header.append("Q_i")
        cols.append(arrays["obs_Q_i"])
    return _csv_from_columns(header, cols)


def _csv_benchmark(arrays: dict[str, np.ndarray]) -> str:
    """The comparison as a table: abscissa, simulated, analytic, per series."""
    sim = np.atleast_2d(arrays["bench_sim"])
    ana = np.atleast_2d(arrays["bench_analytic"])
    header = ["x"]
    cols: list[np.ndarray] = [arrays["bench_x"]]
    for i in range(sim.shape[0]):
        suffix = "" if sim.shape[0] == 1 else f"_{i}"
        header += [f"simulated{suffix}", f"analytic{suffix}"]
        cols += [sim[i], ana[i]]
    return _csv_from_columns(header, cols)


def _framed_sweep(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None,
) -> str:
    _no_frames("sweep", frame)
    return _csv_m25_sweep(arrays)


def _framed_benchmark(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], frame: int | None,
) -> str:
    _no_frames("analytic_comparison", frame)
    return _csv_benchmark(arrays)


_CSVS: dict[str, dict[str, CsvBuilder]] = {
    "m25_junction": {"sweep": _framed_sweep},
    "kinetics": {
        "profile": _csv_kinetics_profile,
        "occupation": _csv_kinetics_occupation_either_shape,
        "phonons": _csv_kinetics_phonons_either_shape,
        "time_series": _csv_kinetics_time_series,
    },
}

# A table that needs an array the run did not produce must not be offered, or
# the download 404s after the user has already clicked it. Keyed by (mode,
# name), not by name alone: "time_series" means a different array in each mode
# -- kinetics only has one when frames were requested -- and a single global
# entry would hide a table that is always there. A predicate rather than one
# array name because `phonons` is built from EITHER payload shape, and the
# spatial one needs its frequency axis as well as its populations: a phonon
# table without the lattice it was recorded on is the bin-count defect again.
_CSV_OFFERED: dict[tuple[str, str], Callable[[Collection[str]], bool]] = {
    ("kinetics", "time_series"): lambda have: "snap_t_ns" in have,
    ("kinetics", "profile"): lambda have: "xqp_profile" in have,
    ("kinetics", "phonons"): lambda have: (
        "n_ph" in have or {"snap_n_ph", "snap_omega_bins"} <= set(have)
    ),
}

for _mode, _mode_csvs in _CSVS.items():
    _mode_csvs["analytic_comparison"] = _framed_benchmark
    _CSV_OFFERED[(_mode, "analytic_comparison")] = lambda have: "bench_x" in have


def available_csvs(mode: str, array_names: Collection[str]) -> list[str]:
    return [
        name
        for name in _CSVS.get(mode, {})
        if (offered := _CSV_OFFERED.get((mode, name))) is None
        or offered(array_names)
    ]


def render_csv(
    mode: str,
    name: str,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any] | None = None,
    params: dict[str, int | None] | None = None,
) -> str:
    """Render one named table to CSV text; raises KeyError for unknown names.

    ``params["frame"]`` selects a recorded frame for the tables that have a
    frame axis and is a KeyError -- a 404 -- on the ones that do not, or when
    the run has no such frame. Omitted, the table is the run's endpoint.
    """
    builder = _CSVS.get(mode, {}).get(name)
    if builder is None:
        raise KeyError(f"No CSV named {name!r} for mode {mode!r}.")
    frame = (params or {}).get("frame")
    return builder(arrays, summary or {}, None if frame is None else int(frame))
