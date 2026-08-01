"""Build the manuscript figures from the Q0 diagnostic sweep artifacts.

Figure 1: Figure-6-style curves for three bath temperatures under both
sampling conventions, overlaid on the digitized published numerical points.
Figure 2: grid-refinement of the anchored-point ordinate under both
conventions, showing the convention gap closing as dE -> 0.

Diagnostic-figure builder; reads only committed/provenance-recorded
artifacts and writes PDFs next to the manuscript.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "tmp" / "author-runs"
POINTS_CSV = REPO / "validation/paper_data/fischer_2023/fig6/points.csv"
REPLAY_JSON = REPO / "validation/paper_data/fischer_2023/fig6/author-replay-sweep.json"
OUT = Path(__file__).resolve().parent

# Okabe-Ito trio, fixed identity order by bath temperature (validated:
# lightness band, chroma, CVD separation, normal-vision floor all pass;
# the orange contrast WARN is relieved by direct labels on every series).
COLOR = {0.1: "#0072B2", 0.15: "#E69F00", 0.2: "#009E73"}
LABEL = {0.1: r"$T_B=0.10\,$K", 0.15: r"$T_B=0.15\,$K", 0.2: r"$T_B=0.20\,$K"}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def load_sweep(path: Path) -> dict[float, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_t: dict[float, list[dict]] = {}
    for point in data["points"]:
        by_t.setdefault(point["T_bath_K"], []).append(point)
    for series in by_t.values():
        series.sort(key=lambda p: p["sweep_index"])
    return by_t


def figure_curves(de1_path: Path, extra_paths: list[Path]) -> None:
    by_t = load_sweep(de1_path)
    for extra in extra_paths:
        if extra.is_file():
            for T, series in load_sweep(extra).items():
                by_t[T] = series
    paper: dict[str, list[tuple[float, float, float]]] = {}
    with POINTS_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["curve_kind"] != "paper_numerical":
                continue
            paper.setdefault(row["curve_id"], []).append(
                (
                    float(row["x_value"]),
                    float(row["y_value"]),
                    float(row["y_uncertainty"]),
                )
            )
    curve_id = {0.1: "T_bath_0.10K", 0.15: "T_bath_0.15K", 0.2: "T_bath_0.20K"}
    replay: dict[float, list[dict]] = {}
    if REPLAY_JSON.is_file():
        data = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
        for point in data["points"]:
            if point.get("replay_ok"):
                replay.setdefault(point["T_bath_K"], []).append(point)
        for series in replay.values():
            series.sort(key=lambda p: p["actual_t_star_over_delta"])

    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)
    for T in (0.1, 0.15, 0.2):
        series = [p for p in by_t.get(T, []) if p.get("converged")]
        if not series:
            continue
        x = [p["eq35_t_star_over_delta"] for p in series]
        ax.plot(
            x,
            [p["ordinate_authors"] for p in series],
            "-",
            color=COLOR[T],
            label=f"{LABEL[T]}",
        )
        ax.plot(
            x,
            [p["ordinate_centers"] for p in series],
            "--",
            color=COLOR[T],
            linewidth=1.1,
        )
        replayed = replay.get(T, [])
        if replayed:
            ax.plot(
                [p["actual_t_star_over_delta"] for p in replayed],
                [p["ordinate"] for p in replayed],
                linestyle=(0, (3, 1, 1, 1)),
                color=COLOR[T],
                linewidth=1.0,
            )
        digitized = sorted(paper.get(curve_id[T], []))
        if digitized:
            ax.errorbar(
                [d[0] for d in digitized],
                [d[1] for d in digitized],
                yerr=[d[2] for d in digitized],
                fmt="o",
                markersize=3.2,
                markerfacecolor="white",
                color=COLOR[T],
                linewidth=0.7,
                capsize=0,
            )
    ax.plot([], [], "-", color="#666666", label="qpsim, author sampling")
    ax.plot([], [], "--", color="#666666", label="qpsim, center sampling")
    ax.plot(
        [],
        [],
        linestyle=(0, (3, 1, 1, 1)),
        color="#666666",
        linewidth=1.0,
        label="authors' program, replayed",
    )
    ax.errorbar(
        [],
        [],
        fmt="o",
        markersize=3.2,
        markerfacecolor="white",
        color="#666666",
        label="digitized published points",
    )
    ax.set_xlabel(r"$T^*/\Delta$")
    ax.set_ylabel(r"$(\Delta[f]-\Delta[f_{\rm th}])/(\Delta_0-\Delta[f_{\rm th}])$")
    ax.legend(fontsize=6.5, loc="upper left")
    # Clip to the published figure window (digitized x in [0.25, 0.41])
    # with context margin; the strong-drive collapse beyond x ~ 0.6 is
    # outside the published scope.
    ax.set_xlim(0.15, 0.45)
    ax.set_ylim(-0.005, 0.25)
    fig.savefig(OUT / "fig_q0_curves.pdf")
    fig.savefig(OUT / "fig_q0_curves.png", dpi=300)
    print("wrote fig_q0_curves.{pdf,png}")


def figure_refinement(
    paths: dict[int, list[Path]], anchor_index: int = 49
) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    for T in (0.1, 0.15, 0.2):
        for convention, style in (("authors", "-o"), ("centers", "--s")):
            xs: list[float] = []
            ys: list[float] = []
            for scale, scale_paths in sorted(paths.items()):
                match: list[dict] = []
                for path in scale_paths:
                    if not path.is_file():
                        continue
                    by_t = load_sweep(path)
                    match.extend(
                        p
                        for p in by_t.get(T, [])
                        if p["sweep_index"] == anchor_index
                        and p.get("converged")
                    )
                if match:
                    xs.append(1.0 / scale)
                    ys.append(match[0][f"ordinate_{convention}"])
            if len(xs) >= 2:
                ax.plot(
                    xs,
                    ys,
                    style,
                    color=COLOR[T],
                    markersize=3.5,
                    markerfacecolor="white" if convention == "centers" else None,
                )
    ax.plot([], [], "-o", color="#666666", markersize=3.5, label="author sampling")
    ax.plot(
        [],
        [],
        "--s",
        color="#666666",
        markersize=3.5,
        markerfacecolor="white",
        label="center sampling",
    )
    ax.set_xscale("log")
    ax.set_xticks([1.0, 0.5, 1.0 / 3.0, 0.25, 0.125])
    ax.set_xticklabels(["1", "0.5", "1/3", "0.25", "0.125"])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel(r"grid spacing d$E$ ($\mu$eV)")
    ax.set_ylabel(f"ordinate at sweep index {anchor_index}")
    ax.legend(fontsize=6.5)
    fig.savefig(OUT / "fig_refinement.pdf")
    fig.savefig(OUT / "fig_refinement.png", dpi=300)
    print("wrote fig_refinement.{pdf,png}")


def main() -> int:
    de1 = RUNS / "fig6-Q0-sweep-de1-v1.json"
    # T = 0.10 K is deliberately excluded: neither certificate mode nor the
    # instrumented legacy mode yields a trustworthy curve in float64 (see
    # the manuscript certification subsection).
    t015 = RUNS / "fig6-Q0-sweep-de1-t015-v1.json"
    if de1.is_file():
        figure_curves(de1, [t015])
    else:
        print("de1 sweep artifact missing; skipping curve figure")
    figure_refinement(
        {
            1: [de1],
            2: [RUNS / "fig6-Q0-sweep-de2-v1.json"],
            3: [RUNS / "fig6-Q0-sweep-de3-v1.json"],
            4: [RUNS / "fig6-Q0-sweep-de4-v1.json"],
            8: [RUNS / "fig6-Q0-sweep-de8-v1.json"],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
