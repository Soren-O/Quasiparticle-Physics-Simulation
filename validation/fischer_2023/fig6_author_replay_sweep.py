"""Full-sweep replay of the authenticated Fischer--Catelani Figure 6 program.

Drives the accepted single-point author-source adapter
(:mod:`validation.fischer_2023.fig6_author_adapter`, unmodified) over the
complete original sweep -- ``n_bar = geomspace(1e4, 1e8, 100)`` for
``T_B in {0.10, 0.15, 0.20} K`` -- in author-equivalent exact mode, one
isolated subprocess per point, exactly as the attachment's own driver loop
would.  Every per-point bundle is retained; this artifact additionally
collects the per-point driven/thermal gaps, the published-figure ordinate
``(Delta[f] - Delta[f_th]) / (Delta_0 - Delta[f_th])``, the Eq.~35
coordinate, and the Newton stopping record into one JSON.

This completes the previously pending full 300-point replay of the
authenticated author source (the A0 ``replay_status`` gap).  It is a
provenance-recorded diagnostic artifact: promotion of A0 itself remains a
separate ladder action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-replay-sweep.v1"
DELTA_EV = 180 * 10**-6
T_LIST = (0.1, 0.15, 0.2)
N_SWEEP = 100


def _read_point(bundle: Path) -> dict[str, object]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    driven = float(metadata["gap_driven_eV"])
    thermal = float(metadata["gap_thermal_eV"])
    return {
        "actual_t_star_over_delta": metadata["actual_t_star_over_delta"],
        "computed_n_bar": metadata["computed_n_bar"],
        "gap_driven_eV": driven,
        "gap_thermal_eV": thermal,
        "last_relative_qp_step": metadata["last_relative_qp_step"],
        "manifest_sha256": hashlib.sha256(
            (bundle / "manifest.json").read_bytes()
        ).hexdigest(),
        "newton_iterations": metadata["newton_iterations"],
        "ordinate": (driven - thermal) / (DELTA_EV - thermal),
        "threshold_met": float(metadata["last_relative_qp_step"])
        < float(metadata["input"]["relative_step_threshold"]),
    }


def run_replay(
    *,
    archive: Path,
    output: Path,
    bundle_root: Path,
    temperatures: tuple[float, ...],
    indices: tuple[int, ...],
    python_executable: str,
) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_suffix(output.suffix + ".jsonl")
    records: list[dict[str, object]] = []
    started = time.time()
    with progress_path.open("w", encoding="utf-8") as progress:
        for T_bath in temperatures:
            for index in indices:
                label = f"T{T_bath:.2f}_i{index:03d}"
                bundle = bundle_root / label
                point: dict[str, object] = {
                    "T_bath_K": T_bath,
                    "sweep_index": index,
                }
                wall = time.time()
                if not (bundle / "manifest.json").is_file():
                    completed = subprocess.run(
                        [
                            python_executable,
                            "-m",
                            "validation.fischer_2023.fig6_author_adapter",
                            "--archive",
                            str(archive),
                            "--output",
                            str(bundle),
                            "--temperature",
                            str(T_bath),
                            "--sweep-index",
                            str(index),
                            "--mode",
                            "author_equivalent",
                        ],
                        cwd=REPOSITORY_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=3600,
                    )
                    if completed.returncode != 0:
                        point["replay_ok"] = False
                        point["stderr_tail"] = completed.stderr[-400:]
                        point["wall_seconds"] = time.time() - wall
                        records.append(point)
                        progress.write(json.dumps(point, sort_keys=True) + "\n")
                        progress.flush()
                        continue
                point["replay_ok"] = True
                point["wall_seconds"] = time.time() - wall
                point.update(_read_point(bundle))
                records.append(point)
                progress.write(json.dumps(point, sort_keys=True) + "\n")
                progress.flush()

    artifact = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "delta_eV": DELTA_EV,
        "mode": "author_equivalent",
        "points": records,
        "qualification": (
            "Full-sweep replay of the authenticated author source through "
            "the accepted, unmodified single-point adapter; a "
            "provenance-recorded diagnostic completing the pending A0 "
            "300-point replay, not itself a ladder-stage promotion."
        ),
        "schema": SCHEMA,
        "sweep": {
            "n_bar_geomspace": [1.0e4, 1.0e8, N_SWEEP],
            "temperatures_K": list(temperatures),
        },
        "total_wall_seconds": time.time() - started,
    }
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"{output} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=list(T_LIST)
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=list(range(N_SWEEP))
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_replay(
        archive=args.archive.resolve(),
        output=args.output,
        bundle_root=args.bundle_root,
        temperatures=tuple(args.temperatures),
        indices=tuple(args.indices),
        python_executable=args.python,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
