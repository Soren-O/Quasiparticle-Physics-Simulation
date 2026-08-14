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
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
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


# -- steady_state_0d --------------------------------------------------


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


# -- transient_0d -----------------------------------------------------


def _plot_occupation_evolution(arrays: dict[str, np.ndarray], gap: float) -> bytes:
    fig, ax = _new_axes("E / Δ", "f(E, t)", "Occupation evolution")
    E_over_gap = arrays["E_bins"] / gap
    t = arrays["t_ns"]
    f_snap = arrays["f_snapshots"]
    t_max = float(t[-1]) if float(t[-1]) > 0 else 1.0
    for i in range(f_snap.shape[0]):
        ax.semilogy(
            E_over_gap, _positive(f_snap[i]),
            color=SEQ_BLUE(0.15 + 0.85 * float(t[i]) / t_max), linewidth=1.4,
        )
    ax.semilogy(
        E_over_gap, _positive(arrays["f_thermal"]),
        linestyle="--", color=MUTED, linewidth=2, label="thermal",
    )
    mappable = plt.cm.ScalarMappable(
        cmap=SEQ_BLUE, norm=Normalize(vmin=0.0, vmax=t_max)
    )
    fig.colorbar(mappable, ax=ax, label="t (ns)")
    _clamp_log_floor(ax, f_snap[-1], decades_below=3.0)
    ax.legend()
    return _finish(fig)


def _plot_time_series(
    arrays: dict[str, np.ndarray],
    names: list[tuple[str, str]],
    ylabel: str,
    title: str,
    *,
    logy: bool = False,
) -> bytes:
    fig, ax = _new_axes("t (ns)", ylabel, title)
    t_key = "t_ns" if "t_ns" in arrays else "snap_t_ns"
    t = arrays[t_key]
    plot = ax.semilogy if logy else ax.plot
    for i, (key, label) in enumerate(names):
        y = arrays[key]
        plot(
            t, _positive(y) if logy else y,
            color=SERIES[i % len(SERIES)], linewidth=2,
            marker=MARKERS[i % len(MARKERS)], markersize=4, label=label,
        )
    if len(names) > 1:
        ax.legend()
    return _finish(fig)


# -- spatial_1d -------------------------------------------------------


def _plot_xqp_profile(arrays: dict[str, np.ndarray]) -> bytes:
    fig, ax = _new_axes(
        "x (μm)",
        "x_qp [n_qp/(4 rho_F Delta_0)]",
        "Quasiparticle fraction along the strip",
    )
    ax.plot(arrays["x_um"], arrays["xqp_profile"], color=SERIES[0], linewidth=2)
    return _finish(fig)


def _plot_occupation_heatmap(arrays: dict[str, np.ndarray], gap: float) -> bytes:
    fig, ax = _new_axes("x (μm)", "E / Δ", "Occupation f(E, x)")
    floor = 1e-300
    f = np.maximum(arrays["f_final"], floor)
    vmax = float(np.max(f))
    if vmax <= floor:
        # A schema-valid ultracold equilibrium can underflow every displayed
        # occupation to the plotting floor. LogNorm requires vmin < vmax;
        # give the constant field one harmless display decade instead of
        # constructing the former inverted (1e-12, 1e-300) range.
        norm = Normalize(vmin=0.0, vmax=floor)
        colorbar_label = "f (linear; values at plotting floor)"
    else:
        positive = f[f > floor]
        vmin = max(float(np.min(positive)), vmax * 1e-12, floor)
        if vmin >= vmax:
            # Constant non-floor fields need a finite colour interval too.
            vmin = max(vmax * 0.1, floor)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        colorbar_label = "f (log scale)"
    mesh = ax.pcolormesh(
        arrays["x_um"], arrays["E_bins"] / gap, f,
        cmap=SEQ_BLUE, norm=norm, shading="nearest",
    )
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.grid(False)
    return _finish(fig)


# -- spatial_2d -------------------------------------------------------


def _plot_xqp_field(arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> bytes:
    """The device as it is shaped, coloured by local x_qp.

    Cells outside the mask are NaN and draw as blank rather than as zero: a
    hole in an annulus is absent material, not cold material, and colouring it
    dark would read as the latter.
    """
    field = np.asarray(arrays["xqp_field"], dtype=float)
    mesh_size = float(summary.get("mesh_size_um", 1.0) or 1.0)
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
    mesh_size = float(summary.get("mesh_size_um", 1.0) or 1.0)
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
    fig, ax = _new_axes("cell (mask order)", "x_qp", "x_qp per cell")
    ax.semilogy(np.arange(profile.size), _positive(profile), marker=".", lw=1.0)
    return _finish(fig)


def _plot_convergence(arrays: dict[str, np.ndarray]) -> bytes:
    fig, ax = _new_axes("t (ns)", "max |df/dt| (1/ns)", "Convergence")
    rate = arrays["snap_max_rate"]
    finite = np.isfinite(rate)
    ax.semilogy(
        arrays["snap_t_ns"][finite], _positive(rate[finite]),
        color=SERIES[0], linewidth=2,
    )
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


# -- registry ---------------------------------------------------------


def _gap(summary: dict[str, Any]) -> float:
    return float(summary.get("gap_ueV", 1.0)) or 1.0


@dataclass(frozen=True)
class _PlotSpec:
    """One named figure: its renderer plus an optional required array."""

    render: Callable[[dict[str, np.ndarray], dict[str, Any]], bytes]
    requires: str | None = None


# Single source of truth per mode: the listing endpoint and the render
# dispatch both read this table, so a figure can't be listed without
# being renderable (or vice versa).
_PLOTS: dict[str, dict[str, _PlotSpec]] = {
    "steady_state_0d": {
        "occupation": _PlotSpec(lambda a, s: _plot_occupation(a, _gap(s))),
        "phonons": _PlotSpec(lambda a, s: _plot_phonons(a, _gap(s))),
    },
    "transient_0d": {
        "occupation_evolution": _PlotSpec(lambda a, s: _plot_occupation_evolution(a, _gap(s))),
        "x_qp_vs_t": _PlotSpec(
            lambda a, s: _plot_time_series(
                a,
                [("obs_x_qp", "x_qp (qpsim)")],
                "x_qp [n_qp/(4 rho_F Delta_0)]",
                "Quasiparticle fraction",
                logy=True,
            )
        ),
        "Q_i_vs_t": _PlotSpec(
            lambda a, s: _plot_time_series(
                a, [("obs_Q_i", "Q_i")], "Q_i", "Internal quality factor", logy=True
            ),
            requires="obs_Q_i",
        ),
    },
    "spatial_1d": {
        "xqp_profile": _PlotSpec(lambda a, s: _plot_xqp_profile(a)),
        "occupation_heatmap": _PlotSpec(lambda a, s: _plot_occupation_heatmap(a, _gap(s))),
        "convergence": _PlotSpec(lambda a, s: _plot_convergence(a)),
        "observables_vs_t": _PlotSpec(
            lambda a, s: _plot_time_series(
                a,
                [("obs_x_qp_mean", "x_qp mean"), ("obs_x_qp_max", "x_qp max")],
                "x_qp [n_qp/(4 rho_F Delta_0)]",
                "Strip observables",
                logy=True,
            )
        ),
    },
    "spatial_2d": {
        "xqp_field": _PlotSpec(lambda a, s: _plot_xqp_field(a, s)),
        "geometry": _PlotSpec(lambda a, s: _plot_geometry_mask(a, s)),
        "xqp_profile": _PlotSpec(lambda a, s: _plot_xqp_profile_2d(a)),
        "occupation": _PlotSpec(
            lambda a, s: _plot_occupation_spectrum_2d(a, _gap(s))
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
        if spec.requires is None or spec.requires in array_names
    ]


def render_plot(
    mode: str, name: str, arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> bytes:
    """Render one named figure to PNG bytes; raises KeyError for unknown names."""
    spec = _PLOTS.get(mode, {}).get(name)
    if spec is None or (spec.requires is not None and spec.requires not in arrays):
        raise KeyError(f"No plot named {name!r} for mode {mode!r}.")
    return spec.render(arrays, summary)


# -- CSV export -------------------------------------------------------


def _csv_from_columns(header: Iterable[str], columns: list[np.ndarray]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(list(header))
    for row in zip(*columns, strict=True):
        writer.writerow([f"{v:.10g}" for v in row])
    return out.getvalue()


def _csv_ss_occupation(arrays: dict[str, np.ndarray]) -> str:
    return _csv_from_columns(
        ["E_ueV", "f", "f_thermal"],
        [arrays["E_bins"], arrays["f"], arrays["f_thermal"]],
    )


def _csv_ss_phonons(arrays: dict[str, np.ndarray]) -> str:
    return _csv_from_columns(["omega_ueV", "n_ph"], [arrays["omega_bins"], arrays["n_ph"]])


def _csv_transient_time_series(arrays: dict[str, np.ndarray]) -> str:
    keys = [k for k in arrays if k.startswith("obs_")]
    return _csv_from_columns(
        ["t_ns", *[k.removeprefix("obs_") for k in keys]],
        [arrays["t_ns"], *[arrays[k] for k in keys]],
    )


def _csv_transient_snapshots(arrays: dict[str, np.ndarray]) -> str:
    t = arrays["t_ns"]
    f_snap = arrays["f_snapshots"]
    return _csv_from_columns(
        ["E_ueV", *[f"f_t={ti:.6g}ns" for ti in t]],
        [arrays["E_bins"], *[f_snap[i] for i in range(f_snap.shape[0])]],
    )


def _csv_spatial_profile(arrays: dict[str, np.ndarray]) -> str:
    # Keep the historical ``x_qp`` column for saved-workflow compatibility;
    # its qpsim normalization is now paired with an explicit paper column.
    header = ["x_um", "x_qp", "x_qp_paper"]
    paper_profile = arrays.get("xqp_profile_paper", 2.0 * arrays["xqp_profile"])
    cols = [
        arrays["x_um"],
        arrays["xqp_profile"],
        paper_profile,
    ]
    if "gap_profile" in arrays:
        header.append("gap_ueV")
        cols.append(arrays["gap_profile"])
    return _csv_from_columns(header, cols)


def _csv_spatial_time_series(arrays: dict[str, np.ndarray]) -> str:
    keys = [k for k in arrays if k.startswith("obs_")]
    return _csv_from_columns(
        ["t_ns", "max_rate_per_ns", *[k.removeprefix("obs_") for k in keys]],
        [arrays["snap_t_ns"], arrays["snap_max_rate"], *[arrays[k] for k in keys]],
    )


def _csv_spatial_occupation(arrays: dict[str, np.ndarray]) -> str:
    x = arrays["x_um"]
    f = arrays["f_final"]
    return _csv_from_columns(
        ["E_ueV", *[f"f_x={xi:.6g}um" for xi in x]],
        [arrays["E_bins"], *[f[:, j] for j in range(f.shape[1])]],
    )


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


def _csv_spatial_2d_profile(arrays: dict[str, np.ndarray]) -> str:
    """Per-cell x_qp with the cell's place on the mask.

    Without this a 2-D run's numbers cannot leave the browser at all: the
    result was four PNGs and a handful of summary scalars, so replotting,
    fitting, or comparing against another tool was impossible.
    """
    col, row = _cell_coordinates(arrays)
    header = ["col", "row", "x_qp", "x_qp_paper"]
    cols = [col, row, arrays["xqp_profile"], 2.0 * arrays["xqp_profile"]]
    if "gap_per_cell" in arrays:
        header.append("gap_ueV")
        cols.append(arrays["gap_per_cell"])
    return _csv_from_columns(header, cols)


def _csv_spatial_2d_occupation(arrays: dict[str, np.ndarray]) -> str:
    """f(E) per cell at the final time, one column per cell."""
    col, row = _cell_coordinates(arrays)
    f = arrays["f_final"]
    return _csv_from_columns(
        ["E_ueV", *[f"f_col={int(c)}_row={int(r)}" for c, r in zip(col, row, strict=True)]],
        [arrays["E_bins"], *[f[:, j] for j in range(f.shape[1])]],
    )


def _csv_spatial_2d_time_series(arrays: dict[str, np.ndarray]) -> str:
    """Recorded frames reduced to per-time observables."""
    return _csv_from_columns(
        ["t_ns", "max_rate_per_ns", "x_qp_mean", "x_qp_max"],
        [arrays["snap_t_ns"], arrays["snap_max_rate"],
         arrays["obs_x_qp_mean"], arrays["obs_x_qp_max"]],
    )


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


_CSVS: dict[str, dict[str, Callable[[dict[str, np.ndarray]], str]]] = {
    "steady_state_0d": {"occupation": _csv_ss_occupation, "phonons": _csv_ss_phonons},
    "transient_0d": {
        "time_series": _csv_transient_time_series,
        "snapshots": _csv_transient_snapshots,
    },
    "spatial_1d": {
        "profile": _csv_spatial_profile,
        "time_series": _csv_spatial_time_series,
        "occupation": _csv_spatial_occupation,
    },
    "m25_junction": {"sweep": _csv_m25_sweep},
    "spatial_2d": {
        "profile": _csv_spatial_2d_profile,
        "occupation": _csv_spatial_2d_occupation,
        "time_series": _csv_spatial_2d_time_series,
    },
}

# A table that needs an array the run did not produce must not be offered, or
# the download 404s after the user has already clicked it. Keyed by (mode,
# name), not by name alone: "time_series" means a different array in each mode
# -- transient_0d always has one, spatial_2d only when frames were requested --
# and a single global entry would hide a table that is always there.
_CSV_REQUIRES: dict[tuple[str, str], str] = {
    ("spatial_2d", "time_series"): "snap_t_ns",
}

for _mode, _mode_csvs in _CSVS.items():
    _mode_csvs["analytic_comparison"] = _csv_benchmark
    _CSV_REQUIRES[(_mode, "analytic_comparison")] = "bench_x"


def available_csvs(mode: str, array_names: Collection[str]) -> list[str]:
    return [
        name
        for name in _CSVS.get(mode, {})
        if (required := _CSV_REQUIRES.get((mode, name))) is None
        or required in array_names
    ]


def render_csv(mode: str, name: str, arrays: dict[str, np.ndarray]) -> str:
    """Render one named table to CSV text; raises KeyError for unknown names."""
    builder = _CSVS.get(mode, {}).get(name)
    if builder is None:
        raise KeyError(f"No CSV named {name!r} for mode {mode!r}.")
    return builder(arrays)
