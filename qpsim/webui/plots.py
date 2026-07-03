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
    fig, ax = _new_axes("x (μm)", "x_qp", "Quasiparticle fraction along the strip")
    ax.plot(arrays["x_um"], arrays["xqp_profile"], color=SERIES[0], linewidth=2)
    return _finish(fig)


def _plot_occupation_heatmap(arrays: dict[str, np.ndarray], gap: float) -> bytes:
    fig, ax = _new_axes("x (μm)", "E / Δ", "Occupation f(E, x)")
    f = np.maximum(arrays["f_final"], 1e-300)
    vmax = float(np.max(f))
    vmin = max(float(np.min(f[f > 1e-290])) if np.any(f > 1e-290) else 1e-12, vmax * 1e-12)
    mesh = ax.pcolormesh(
        arrays["x_um"], arrays["E_bins"] / gap, f,
        cmap=SEQ_BLUE, norm=LogNorm(vmin=vmin, vmax=vmax), shading="nearest",
    )
    fig.colorbar(mesh, ax=ax, label="f (log scale)")
    ax.grid(False)
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
                a, [("obs_x_qp", "x_qp")], "x_qp", "Quasiparticle fraction", logy=True
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
                "x_qp",
                "Strip observables",
                logy=True,
            )
        ),
    },
    "m25_junction": {
        "chemical_potentials": _PlotSpec(lambda a, s: _plot_m25_mu(a)),
        "densities": _PlotSpec(lambda a, s: _plot_m25_densities(a)),
        "qubit_p1": _PlotSpec(lambda a, s: _plot_m25_p1(a)),
    },
}


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
    header = ["x_um", "x_qp"]
    cols = [arrays["x_um"], arrays["xqp_profile"]]
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
}


def available_csvs(mode: str, array_names: Collection[str]) -> list[str]:
    return list(_CSVS.get(mode, {}))


def render_csv(mode: str, name: str, arrays: dict[str, np.ndarray]) -> str:
    """Render one named table to CSV text; raises KeyError for unknown names."""
    builder = _CSVS.get(mode, {}).get(name)
    if builder is None:
        raise KeyError(f"No CSV named {name!r} for mode {mode!r}.")
    return builder(arrays)
