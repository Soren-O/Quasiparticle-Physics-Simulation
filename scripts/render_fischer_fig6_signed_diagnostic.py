"""Render a noncanonical full-range diagnostic from promoted Fischer Fig. 6.

This command does not run a numerical solve.  It holds the canonical
publication lock while the existing artifact validator independently
recertifies every stored state once and the promotion record is bound to that
same result.  It then renders every finite stored numerical and Eq. 53 sample
on a signed-log axis and records any nonfinite samples that cannot be plotted.
PCHIP curves, when enabled, are visual interpolation only; the markers are the
evidence-bearing stored samples.

The default outputs live under ``tmp/`` and are deliberately outside the
canonical validation bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Direct ``python scripts/...py`` invocation otherwise resolves an
    # unrelated/stale installed ``validation`` package before this checkout.
    sys.path.insert(0, str(_REPO_ROOT))

from validation.fischer_2023 import fig6_paper  # noqa: E402

SCHEMA = "qpsim.fischer2023.fig6_signed_diagnostic.v3"
MODE = "noncanonical-signed-diagnostic"
CLAIM_SCOPE = "qpsim self-regression, no digitized-data parity"
PAPER_X_WINDOW = (0.20, 0.65)
PAPER_Y_WINDOW = (0.00, 0.25)
SYMLOG_LINTHRESH = 1.0e-2
_DEFAULT_PDF = _REPO_ROOT / "tmp" / "fischer_fig6_signed_diagnostic.pdf"


class DiagnosticAuthenticationError(RuntimeError):
    """Raised when the canonical Fig. 6 bundle cannot be trusted."""


class DiagnosticPublicationError(RuntimeError):
    """Raised when a signed-diagnostic PDF/commit-marker pair is invalid."""


@dataclass(frozen=True)
class AuthenticatedCanonical:
    """One reader-certified canonical snapshot and its durable identities."""

    result: fig6_paper.Fig6PaperResult
    csv_path: Path
    promotion_path: Path
    csv_identity: dict[str, object]
    promotion_identity: dict[str, object]
    promotion_record: dict[str, Any]


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _renderer_runtime_identity() -> dict[str, object]:
    """Capture the numerical and plotting runtime that produced the PDF."""

    import matplotlib

    # Select the noninteractive figure backend before recording it.  Capturing
    # the configured default first (for example TkAgg or SVG) and only later
    # switching in ``build_figure`` stamped a backend that did not construct the
    # rendered figure.
    matplotlib.use("Agg")
    runtime = fig6_paper.generation_runtime_identity()
    runtime["matplotlib"] = {
        "backend": str(matplotlib.get_backend()),
        "freetype": getattr(
            getattr(matplotlib, "ft2font", None),
            "__freetype_version__",
            None,
        ),
        "version": matplotlib.__version__,
    }
    return runtime


def _identity_with_path(path: Path, *, recorded_path: Path | None = None) -> dict[str, object]:
    identity = _file_identity(path)
    if recorded_path is not None:
        identity["path"] = str(recorded_path.resolve())
    return identity


def _record_csv_identity(record: dict[str, Any]) -> dict[str, object]:
    try:
        identity = record["artifacts"]["csv"]
        return {
            "sha256": str(identity["sha256"]),
            "size_bytes": int(identity["size_bytes"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticAuthenticationError(
            "The authenticated Fig. 6 promotion record has no valid CSV identity."
        ) from exc


def _read_bound_promotion_record_unlocked(
    *,
    result: fig6_paper.Fig6PaperResult | None,
    csv_path: Path,
    pdf_path: Path,
    promotion_path: Path,
) -> dict[str, Any]:
    """Validate one record against an already re-certified artifact.

    The canonical public helpers each re-certify the complete 66-state
    artifact, so composing three promotion reads with one baseline read used
    to repeat that same work five times.  The caller holds the canonical
    publication lock; reusing ``result`` here preserves every promotion,
    source, row-semantic, PDF, and byte-identity check without another state
    replay.  Passing ``None`` performs all byte/source checks before the
    expensive replay; the caller validates row-semantic digests against the
    resulting state immediately afterward.
    """

    try:
        record = json.loads(promotion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise fig6_paper.ArtifactValidationError(
            "Fig. 6 canonical publication record is missing or invalid."
        ) from exc
    if not isinstance(record, dict) or set(record) != {
        "artifact_schema",
        "artifacts",
        "generation",
        "schema",
    }:
        raise fig6_paper.ArtifactValidationError(
            "Fig. 6 publication record shape is invalid."
        )
    if (
        record["schema"] != fig6_paper.PROMOTION_RECORD_SCHEMA
        or record["artifact_schema"] != fig6_paper.ARTIFACT_SCHEMA
    ):
        raise fig6_paper.ArtifactValidationError(
            "Fig. 6 publication record schema is stale."
        )
    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"csv", "pdf"}:
        raise fig6_paper.ArtifactValidationError(
            "Fig. 6 publication identities are invalid."
        )
    for name, artifact_path in (("csv", csv_path), ("pdf", pdf_path)):
        if artifacts[name] != fig6_paper._file_identity(artifact_path):
            raise fig6_paper.ArtifactValidationError(
                f"Fig. 6 canonical {name.upper()} does not match its "
                "publication record."
            )
    fig6_paper._require_valid_pdf(pdf_path)
    fig6_paper.validate_generation_evidence(
        record["generation"],
        result=result,
    )
    return record


def load_authenticated_canonical() -> AuthenticatedCanonical:
    """Read one locked snapshot with exactly one full state recertification."""

    csv_path = fig6_paper.baseline_path()
    pdf_path = fig6_paper.plot_path()
    promotion_path = fig6_paper.promotion_record_path()
    try:
        with fig6_paper._publication_lock():
            record = _read_bound_promotion_record_unlocked(
                result=None,
                csv_path=csv_path,
                pdf_path=pdf_path,
                promotion_path=promotion_path,
            )
            result, _metadata = fig6_paper._read_artifact(csv_path)
            fig6_paper.validate_generation_evidence(
                record["generation"],
                result=result,
            )
            csv_identity = _file_identity(csv_path)
            promotion_identity = _file_identity(promotion_path)
    except Exception as exc:
        raise DiagnosticAuthenticationError(
            "Canonical Fig. 6 authentication/currentness validation failed."
        ) from exc

    stamped_csv = _record_csv_identity(record)
    measured_csv = {
        "sha256": csv_identity["sha256"],
        "size_bytes": csv_identity["size_bytes"],
    }
    if stamped_csv != measured_csv:
        raise DiagnosticAuthenticationError(
            "Canonical Fig. 6 CSV bytes do not match the authenticated promotion record."
        )
    return AuthenticatedCanonical(
        result=result,
        csv_path=csv_path,
        promotion_path=promotion_path,
        csv_identity=csv_identity,
        promotion_identity=promotion_identity,
        promotion_record=record,
    )


def _require_canonical_current_unlocked(
    authenticated: AuthenticatedCanonical,
) -> None:
    """Refuse publication if the canonical tuple changed after rendering.

    The caller holds :func:`fig6_paper._publication_lock`.  Reusing the
    already recertified ``result`` preserves the semantic worker-row checks
    without decoding or recertifying the 66 persisted states a second time.
    Exact CSV and promotion-record identities close the render-time race:
    a diagnostic may never be published against a canonical snapshot that
    ceased to be current while Matplotlib was running.
    """

    csv_path = fig6_paper.baseline_path()
    pdf_path = fig6_paper.plot_path()
    promotion_path = fig6_paper.promotion_record_path()
    if (
        csv_path.resolve() != authenticated.csv_path.resolve()
        or promotion_path.resolve() != authenticated.promotion_path.resolve()
    ):
        raise DiagnosticAuthenticationError(
            "Canonical Fig. 6 paths changed while rendering the diagnostic."
        )
    try:
        current_csv_identity = _file_identity(csv_path)
        current_promotion_identity = _file_identity(promotion_path)
        if (
            current_csv_identity != authenticated.csv_identity
            or current_promotion_identity != authenticated.promotion_identity
        ):
            raise DiagnosticAuthenticationError(
                "Canonical Fig. 6 bytes changed while rendering the diagnostic."
            )
        current_record = _read_bound_promotion_record_unlocked(
            result=authenticated.result,
            csv_path=csv_path,
            pdf_path=pdf_path,
            promotion_path=promotion_path,
        )
    except DiagnosticAuthenticationError:
        raise
    except Exception as exc:
        raise DiagnosticAuthenticationError(
            "Canonical Fig. 6 currentness changed while rendering the diagnostic."
        ) from exc
    if current_record != authenticated.promotion_record:
        raise DiagnosticAuthenticationError(
            "Canonical Fig. 6 promotion evidence changed while rendering "
            "the diagnostic."
        )


def _curve_stats(
    x: np.ndarray,
    y: np.ndarray,
    temperatures: np.ndarray,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    totals = {
        "finite_samples": 0,
        "visible_in_paper_window": 0,
        "clipped_from_paper_window": 0,
        "nonfinite_samples": 0,
    }
    finite_values: list[np.ndarray] = []
    for index, temperature in enumerate(temperatures):
        finite = np.isfinite(x[index]) & np.isfinite(y[index])
        visible = (
            finite
            & (x[index] >= PAPER_X_WINDOW[0])
            & (x[index] <= PAPER_X_WINDOW[1])
            & (y[index] >= PAPER_Y_WINDOW[0])
            & (y[index] <= PAPER_Y_WINDOW[1])
        )
        values = y[index][finite]
        finite_values.append(values)
        row: dict[str, object] = {
            "temperature_K": float(temperature),
            "finite_samples": int(np.count_nonzero(finite)),
            "visible_in_paper_window": int(np.count_nonzero(visible)),
            "clipped_from_paper_window": int(np.count_nonzero(finite & ~visible)),
            "nonfinite_samples": int(finite.size - np.count_nonzero(finite)),
            "left_of_paper_x": int(np.count_nonzero(finite & (x[index] < PAPER_X_WINDOW[0]))),
            "right_of_paper_x": int(np.count_nonzero(finite & (x[index] > PAPER_X_WINDOW[1]))),
            "below_paper_y": int(np.count_nonzero(finite & (y[index] < PAPER_Y_WINDOW[0]))),
            "above_paper_y": int(np.count_nonzero(finite & (y[index] > PAPER_Y_WINDOW[1]))),
            "minimum": None if values.size == 0 else float(np.min(values)),
            "maximum": None if values.size == 0 else float(np.max(values)),
        }
        rows.append(row)
        for field in totals:
            count = row[field]
            assert isinstance(count, int)
            totals[field] += count

    combined = np.concatenate(finite_values) if finite_values else np.asarray([])
    return {
        **totals,
        "minimum": None if combined.size == 0 else float(np.min(combined)),
        "maximum": None if combined.size == 0 else float(np.max(combined)),
        "by_temperature": rows,
    }


def compute_plot_stats(result: fig6_paper.Fig6PaperResult) -> dict[str, object]:
    """Return JSON-safe clipping counts and extrema for both stored curves."""

    temperatures = np.asarray(result.T_bath, dtype=float)
    x = np.asarray(result.T_star_over_delta, dtype=float)
    numerical = np.asarray(result.paper_observable_num, dtype=float)
    analytic = np.asarray(result.paper_observable_eq53, dtype=float)
    expected = (temperatures.size, np.asarray(result.n_bar).size)
    if (
        temperatures.ndim != 1
        or x.shape != expected
        or numerical.shape != expected
        or analytic.shape != expected
    ):
        raise ValueError("Fig. 6 diagnostic arrays do not share the declared 2-D axes.")
    return {
        "paper_window": {
            "x": list(PAPER_X_WINDOW),
            "y": list(PAPER_Y_WINDOW),
        },
        "numerical": _curve_stats(x, numerical, temperatures),
        "analytic_eq53": _curve_stats(x, analytic, temperatures),
    }


def _format_extreme(value: object) -> str:
    if value is None:
        return "none"
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"Expected a numeric diagnostic extreme, got {type(value).__name__}.")
    return f"{float(value):.4g}"


def clipping_disclosure(stats: dict[str, object]) -> str:
    """Build the text printed on the plot itself."""

    numerical = stats["numerical"]
    analytic = stats["analytic_eq53"]
    assert isinstance(numerical, dict)
    assert isinstance(analytic, dict)
    lines = [
        CLAIM_SCOPE,
        "Markers are stored samples; PCHIP lines are visual interpolation only.",
        (
            "Paper window x=[0.20, 0.65], y=[0, 0.25] hides "
            f"{numerical['clipped_from_paper_window']}/{numerical['finite_samples']} "
            "numerical and "
            f"{analytic['clipped_from_paper_window']}/{analytic['finite_samples']} "
            "Eq. 53 samples."
        ),
        (
            "Full stored extrema: numerical "
            f"[{_format_extreme(numerical['minimum'])}, "
            f"{_format_extreme(numerical['maximum'])}]; Eq. 53 "
            f"[{_format_extreme(analytic['minimum'])}, "
            f"{_format_extreme(analytic['maximum'])}]."
        ),
        (
            "Nonfinite stored samples (recorded, not plottable): "
            f"{numerical['nonfinite_samples']} numerical; "
            f"{analytic['nonfinite_samples']} Eq. 53."
        ),
    ]
    return "\n".join(lines)


def _pchip_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if x.size < 4:
        return None
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    if np.any(np.diff(xs) <= 0.0):
        return None
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:  # pragma: no cover - SciPy is a required dependency
        return None
    dense_x = np.linspace(float(xs[0]), float(xs[-1]), 500)
    dense_y = np.asarray(PchipInterpolator(xs, ys, extrapolate=False)(dense_x))
    return dense_x, dense_y


def build_figure(
    result: fig6_paper.Fig6PaperResult,
    stats: dict[str, object],
    *,
    interpolate: bool = True,
) -> tuple[Any, dict[str, object]]:
    """Build the diagnostic figure without reading artifacts or solving."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    temperatures = np.asarray(result.T_bath, dtype=float)
    x_values = np.asarray(result.T_star_over_delta, dtype=float)
    numerical = np.asarray(result.paper_observable_num, dtype=float)
    analytic = np.asarray(result.paper_observable_eq53, dtype=float)
    colors = {0.10: "C2", 0.15: "C0", 0.20: "C3"}

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    interpolated_curves = 0
    for index, temperature in enumerate(temperatures):
        color = colors.get(float(round(float(temperature), 4)), f"C{index % 10}")
        for values, marker, linestyle, curve_label in (
            (numerical[index], "o", "-", "numerical"),
            (analytic[index], "x", (0, (5, 2)), "Eq. 53"),
        ):
            finite = np.isfinite(x_values[index]) & np.isfinite(values)
            xs = x_values[index][finite]
            ys = values[finite]
            ax.plot(
                xs,
                ys,
                linestyle="none",
                marker=marker,
                markersize=4.0,
                color=color,
                label=rf"{curve_label} samples, $T_B={temperature:g}$ K",
                zorder=5,
            )
            interpolated = _pchip_points(xs, ys) if interpolate else None
            if interpolated is not None:
                ax.plot(
                    interpolated[0],
                    interpolated[1],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    alpha=0.8,
                    label="_nolegend_",
                    zorder=3,
                )
                interpolated_curves += 1

    ax.add_patch(
        Rectangle(
            (PAPER_X_WINDOW[0], PAPER_Y_WINDOW[0]),
            PAPER_X_WINDOW[1] - PAPER_X_WINDOW[0],
            PAPER_Y_WINDOW[1] - PAPER_Y_WINDOW[0],
            fill=False,
            edgecolor="black",
            linestyle=":",
            linewidth=1.1,
            label="canonical paper window",
            zorder=6,
        )
    )
    ax.axhline(0.0, color="black", linewidth=0.6)
    finite_x = x_values[np.isfinite(x_values)]
    if finite_x.size:
        x_lo = min(PAPER_X_WINDOW[0], float(np.min(finite_x)))
        x_hi = max(PAPER_X_WINDOW[1], float(np.max(finite_x)))
        pad = 0.03 * max(x_hi - x_lo, np.finfo(float).eps)
        ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
    ax.set_xlabel(r"$T_*/\Delta$")
    ax.set_ylabel(r"$(\delta\Delta_T-\delta\Delta)/\delta\Delta_T$")
    ax.set_title("Fischer 2023 Fig. 6 — full signed qpsim diagnostic (noncanonical)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(fontsize=7, ncol=2)
    fig.text(
        0.02,
        0.02,
        clipping_disclosure(stats),
        ha="left",
        va="bottom",
        fontsize=7.2,
        family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    return fig, {
        "raw_numerical_markers": int(stats["numerical"]["finite_samples"]),  # type: ignore[index]
        "raw_analytic_markers": int(stats["analytic_eq53"]["finite_samples"]),  # type: ignore[index]
        "pchip_interpolation_requested": bool(interpolate),
        "pchip_curves_rendered": interpolated_curves,
        "symlog_linthresh": SYMLOG_LINTHRESH,
    }


def _expected_pchip_curve_count(
    result: fig6_paper.Fig6PaperResult,
    *,
    requested: bool,
) -> int:
    """Recompute the exact number of PCHIP curves implied by stored samples."""

    if not requested:
        return 0
    x_values = np.asarray(result.T_star_over_delta, dtype=float)
    count = 0
    for values_by_temperature in (
        np.asarray(result.paper_observable_num, dtype=float),
        np.asarray(result.paper_observable_eq53, dtype=float),
    ):
        for index in range(x_values.shape[0]):
            finite = np.isfinite(x_values[index]) & np.isfinite(
                values_by_temperature[index]
            )
            if (
                _pchip_points(
                    x_values[index][finite],
                    values_by_temperature[index][finite],
                )
                is not None
            ):
                count += 1
    return count


def _render_commitment(
    render: dict[str, object],
    runtime: dict[str, object],
    created_utc: str,
) -> str:
    """Commit render claims, creation time, and runtime into PDF metadata."""

    payload = json.dumps(
        {
            "created_utc": created_utc,
            "render": render,
            "runtime": runtime,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "qpsim-fig6-render-" + hashlib.sha256(payload).hexdigest()


def _require_pdf_render_commitment(
    path: Path,
    render: dict[str, object],
    runtime: dict[str, object],
    created_utc: str,
) -> None:
    """Bind sidecar render/runtime claims to the exact PDF."""

    commitment = _render_commitment(
        render,
        runtime,
        created_utc,
    ).encode("ascii")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DiagnosticPublicationError(
            "Signed-diagnostic PDF could not be read for render authentication."
        ) from exc
    if payload.count(commitment) != 1:
        raise DiagnosticPublicationError(
            "Signed-diagnostic PDF does not contain the render commitment "
            "declared by its JSON commit marker."
        )


def _guard_output_paths(pdf_path: Path, json_path: Path) -> None:
    canonical = {
        fig6_paper.baseline_path().resolve(),
        fig6_paper.plot_path().resolve(),
        fig6_paper.promotion_record_path().resolve(),
        fig6_paper.promotion_record_path().with_suffix(".json.lock").resolve(),
    }
    if pdf_path.resolve() in canonical or json_path.resolve() in canonical:
        raise ValueError(
            "The signed diagnostic may not overwrite a canonical Fig. 6 "
            "bundle member or publication lock."
        )
    if pdf_path.resolve() == json_path.resolve():
        raise ValueError("The diagnostic PDF and JSON sidecar paths must be distinct.")
    resources = {pdf_path.resolve(), json_path.resolve()}
    lock_paths = {_diagnostic_resource_lock_path(path).resolve() for path in resources}
    if resources & lock_paths:
        raise ValueError(
            "A diagnostic output path may not alias either output resource lock."
        )


def output_paths(
    pdf_path: Path | None = None,
    json_path: Path | None = None,
) -> tuple[Path, Path]:
    pdf = _DEFAULT_PDF if pdf_path is None else pdf_path
    sidecar = pdf.with_suffix(".json") if json_path is None else json_path
    _guard_output_paths(pdf, sidecar)
    return pdf, sidecar


def _diagnostic_resource_lock_path(resource: Path) -> Path:
    return resource.with_name(f".{resource.name}.qpsim.lock")


@contextmanager
def _diagnostic_publication_lock(
    pdf_path: Path,
    sidecar: Path,
) -> Iterator[None]:
    """Lock both output resources in deterministic order.

    Sidecar-only locking allowed two otherwise valid calls that shared one PDF
    but selected different JSON paths to race over that PDF (and vice versa).
    One lock per resolved resource makes every overlapping pair contend while
    sorted acquisition avoids deadlocks.
    """

    lock_paths = sorted(
        {
            _diagnostic_resource_lock_path(pdf_path).resolve(),
            _diagnostic_resource_lock_path(sidecar).resolve(),
        },
        key=lambda path: os.path.normcase(str(path)),
    )
    streams: list[Any] = []

    def release(stream: Any) -> None:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    stream.fileno(),
                    fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )
        finally:
            stream.close()

    try:
        for lock_path in lock_paths:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            stream = lock_path.open("a+b")
            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(  # type: ignore[attr-defined]
                        stream.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                    )
            except BaseException:
                stream.close()
                raise
            streams.append(stream)
    except BaseException as exc:
        for stream in reversed(streams):
            release(stream)
        if isinstance(exc, OSError):
            raise DiagnosticPublicationError(
                "Another signed-diagnostic publisher or reader holds an "
                f"overlapping output lock ({lock_paths})."
            ) from exc
        raise
    try:
        yield
    finally:
        for stream in reversed(streams):
            release(stream)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticPublicationError(
            f"Signed-diagnostic commit marker {path} is missing or invalid."
        ) from exc
    if not isinstance(value, dict):
        raise DiagnosticPublicationError(
            "Signed-diagnostic commit marker must contain one JSON object."
        )
    return value


def _validate_identity(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise DiagnosticPublicationError(
            f"Signed-diagnostic {label} identity is malformed."
        )
    path = value["path"]
    sha256 = value["sha256"]
    size_bytes = value["size_bytes"]
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise DiagnosticPublicationError(
            f"Signed-diagnostic {label} identity is invalid."
        )
    return value


def _validate_renderer_runtime(value: object) -> dict[str, object]:
    """Validate portable runtime provenance without requiring the same host."""

    expected_keys = {
        "machine",
        "matplotlib",
        "numeric_runtime",
        "numpy",
        "os_family",
        "python",
        "python_implementation",
        "scipy",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise DiagnosticPublicationError(
            "Signed-diagnostic renderer runtime is malformed."
        )
    for key in expected_keys - {"matplotlib", "numeric_runtime"}:
        if not isinstance(value[key], str) or not value[key]:
            raise DiagnosticPublicationError(
                "Signed-diagnostic renderer runtime is invalid."
            )
    matplotlib_record = value["matplotlib"]
    if (
        not isinstance(matplotlib_record, dict)
        or set(matplotlib_record) != {"backend", "freetype", "version"}
        or not isinstance(matplotlib_record["backend"], str)
        or not matplotlib_record["backend"]
        or (
            matplotlib_record["freetype"] is not None
            and not isinstance(matplotlib_record["freetype"], str)
        )
        or not isinstance(matplotlib_record["version"], str)
        or not matplotlib_record["version"]
    ):
        raise DiagnosticPublicationError(
            "Signed-diagnostic Matplotlib runtime is invalid."
        )
    if not isinstance(value["numeric_runtime"], dict):
        raise DiagnosticPublicationError(
            "Signed-diagnostic numerical runtime is invalid."
        )
    try:
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DiagnosticPublicationError(
            "Signed-diagnostic renderer runtime is not canonical JSON."
        ) from exc
    return value


def _expected_canonical_identity(
    authenticated: AuthenticatedCanonical,
) -> dict[str, object]:
    generation = authenticated.promotion_record.get("generation")
    if not isinstance(generation, dict):
        raise DiagnosticPublicationError(
            "Authenticated canonical promotion has no generation evidence."
        )
    return {
        "csv": authenticated.csv_identity,
        "promotion_record": authenticated.promotion_identity,
        "promotion_schema": authenticated.promotion_record.get("schema"),
        "generation_run_identity": generation.get("run_identity"),
    }


def _read_diagnostic_record_unlocked(
    *,
    pdf_path: Path,
    json_path: Path,
    recorded_pdf_path: Path,
    authenticated: AuthenticatedCanonical | None = None,
) -> dict[str, Any]:
    """Validate one PDF against its JSON commit marker and canonical inputs.

    A publisher may pass the one canonical snapshot it authenticated before
    rendering.  External readers omit it and independently re-certify the
    current canonical bundle once.
    """

    record = _read_json(json_path)
    if set(record) != {
        "schema",
        "mode",
        "claim_scope",
        "created_utc",
        "canonical",
        "renderer",
        "runtime",
        "output_pdf",
        "render",
        "stats",
    }:
        raise DiagnosticPublicationError(
            "Signed-diagnostic commit-marker shape is invalid."
        )
    if (
        record["schema"] != SCHEMA
        or record["mode"] != MODE
        or record["claim_scope"] != CLAIM_SCOPE
    ):
        raise DiagnosticPublicationError(
            "Signed-diagnostic commit-marker schema or scope is stale."
        )
    created_utc = record["created_utc"]
    if not isinstance(created_utc, str) or not created_utc:
        raise DiagnosticPublicationError(
            "Signed-diagnostic creation timestamp is invalid."
        )
    try:
        created_at = datetime.fromisoformat(created_utc)
    except ValueError as exc:
        raise DiagnosticPublicationError(
            "Signed-diagnostic creation timestamp is malformed."
        ) from exc
    if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
        raise DiagnosticPublicationError(
            "Signed-diagnostic creation timestamp must be timezone-aware UTC."
        )

    output_identity = _validate_identity(record["output_pdf"], label="PDF")
    measured_output = _identity_with_path(
        pdf_path,
        recorded_path=recorded_pdf_path,
    )
    if output_identity != measured_output:
        raise DiagnosticPublicationError(
            "Signed-diagnostic PDF does not match its JSON commit marker."
        )
    try:
        fig6_paper._require_valid_pdf(pdf_path)
    except Exception as exc:
        raise DiagnosticPublicationError(
            "Signed-diagnostic PDF is not a complete nonempty Matplotlib PDF."
        ) from exc

    renderer = _validate_identity(record["renderer"], label="renderer")
    if renderer != _file_identity(Path(__file__).resolve()):
        raise DiagnosticPublicationError(
            "Signed-diagnostic renderer source has changed."
        )
    runtime = _validate_renderer_runtime(record["runtime"])

    authenticated = (
        load_authenticated_canonical()
        if authenticated is None
        else authenticated
    )
    canonical = record["canonical"]
    if not isinstance(canonical, dict):
        raise DiagnosticPublicationError(
            "Signed-diagnostic canonical identity is malformed."
        )
    _validate_identity(canonical.get("csv"), label="canonical CSV")
    _validate_identity(
        canonical.get("promotion_record"),
        label="canonical promotion record",
    )
    if canonical != _expected_canonical_identity(authenticated):
        raise DiagnosticPublicationError(
            "Signed-diagnostic canonical inputs no longer match their "
            "authenticated identities."
        )

    expected_stats = compute_plot_stats(authenticated.result)
    if record["stats"] != expected_stats:
        raise DiagnosticPublicationError(
            "Signed-diagnostic statistics do not match the canonical samples."
        )
    render = record["render"]
    if not isinstance(render, dict) or set(render) != {
        "raw_numerical_markers",
        "raw_analytic_markers",
        "pchip_interpolation_requested",
        "pchip_curves_rendered",
        "symlog_linthresh",
    }:
        raise DiagnosticPublicationError(
            "Signed-diagnostic render evidence is malformed."
        )
    numerical = expected_stats["numerical"]
    analytic = expected_stats["analytic_eq53"]
    assert isinstance(numerical, dict)
    assert isinstance(analytic, dict)
    if (
        render["raw_numerical_markers"] != numerical["finite_samples"]
        or render["raw_analytic_markers"] != analytic["finite_samples"]
        or render["symlog_linthresh"] != SYMLOG_LINTHRESH
        or not isinstance(render["pchip_interpolation_requested"], bool)
        or isinstance(render["pchip_curves_rendered"], bool)
        or not isinstance(render["pchip_curves_rendered"], int)
        or render["pchip_curves_rendered"] < 0
    ):
        raise DiagnosticPublicationError(
            "Signed-diagnostic render evidence is inconsistent."
        )
    requested = render["pchip_interpolation_requested"]
    assert isinstance(requested, bool)
    expected_pchip_curves = _expected_pchip_curve_count(
        authenticated.result,
        requested=requested,
    )
    if render["pchip_curves_rendered"] != expected_pchip_curves:
        raise DiagnosticPublicationError(
            "Signed-diagnostic PCHIP evidence does not match the canonical samples."
        )
    _require_pdf_render_commitment(
        pdf_path,
        render,
        runtime,
        created_utc,
    )
    return record


def read_diagnostic_record(
    *,
    pdf_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, Any]:
    """Accept a noncanonical diagnostic pair only through its commit marker."""

    pdf, sidecar = output_paths(pdf_path, json_path)
    with _diagnostic_publication_lock(pdf, sidecar):
        return _read_diagnostic_record_unlocked(
            pdf_path=pdf,
            json_path=sidecar,
            recorded_pdf_path=pdf,
        )


def _write_json_stage(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _create_unique_stage_path(target: Path, *, suffix: str) -> Path:
    """Reserve a collision-safe stage beside ``target``.

    A PID-only name aliases across threads in one process and can also be
    selected explicitly as the other final output path.  ``mkstemp`` reserves a
    unique file atomically; the caller holds both output-resource locks before
    invoking this helper.
    """

    descriptor, raw_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(raw_path)


def _guard_stage_paths(
    pdf_path: Path,
    json_path: Path,
    *,
    pdf_stage: Path,
    json_stage: Path,
) -> None:
    """Require stages to be distinct from finals and their resource locks."""

    resources = {pdf_path.resolve(), json_path.resolve()}
    lock_paths = {
        _diagnostic_resource_lock_path(pdf_path).resolve(),
        _diagnostic_resource_lock_path(json_path).resolve(),
    }
    stages = {pdf_stage.resolve(), json_stage.resolve()}
    if (
        len(stages) != 2
        or stages & resources
        or stages & lock_paths
    ):
        raise DiagnosticPublicationError(
            "Signed-diagnostic stage paths alias a final output or resource lock."
        )


def _restore_payload(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    stage = _create_unique_stage_path(path, suffix=".rollback")
    try:
        stage.write_bytes(payload)
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def generate_diagnostic(
    *,
    pdf_path: Path | None = None,
    json_path: Path | None = None,
    interpolate: bool = True,
) -> tuple[Path, Path]:
    """Publish a noncanonical PDF followed by its JSON commit marker.

    Both files are fully staged and authenticated before either final path is
    replaced.  The PDF is promoted first and the JSON sidecar last.  Therefore
    an interruption can at worst leave a PDF that the old/missing commit
    marker rejects; :func:`read_diagnostic_record` never accepts a mixed pair.
    """

    pdf, sidecar = output_paths(pdf_path, json_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with _diagnostic_publication_lock(pdf, sidecar):
        # Hold both output-resource locks before authenticating, rendering, or
        # creating any stage.  This gives every overlapping publication/read
        # the same diagnostic->canonical lock order and prevents same-process
        # threads from racing over pre-commit work.
        authenticated = load_authenticated_canonical()
        stats = compute_plot_stats(authenticated.result)
        runtime = _renderer_runtime_identity()
        renderer_path = Path(__file__).resolve()
        renderer_identity = _file_identity(renderer_path)
        pdf_stage: Path | None = None
        json_stage: Path | None = None
        try:
            pdf_stage = _create_unique_stage_path(pdf, suffix=".stage.pdf")
            json_stage = _create_unique_stage_path(sidecar, suffix=".stage.json")
            _guard_stage_paths(
                pdf,
                sidecar,
                pdf_stage=pdf_stage,
                json_stage=json_stage,
            )
            figure, render = build_figure(
                authenticated.result,
                stats,
                interpolate=interpolate,
            )
            created_utc = datetime.now(UTC).isoformat()
            try:
                figure.savefig(
                    pdf_stage,
                    format="pdf",
                    metadata={
                        "Title": "Fischer 2023 Fig. 6 signed qpsim diagnostic",
                        "Subject": CLAIM_SCOPE,
                        "Keywords": _render_commitment(
                            render,
                            runtime,
                            created_utc,
                        ),
                    },
                )
            finally:
                import matplotlib.pyplot as plt

                plt.close(figure)
            with pdf_stage.open("r+b") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            if _file_identity(renderer_path) != renderer_identity:
                raise DiagnosticPublicationError(
                    "Signed-diagnostic renderer source changed while rendering."
                )
            evidence: dict[str, object] = {
                "schema": SCHEMA,
                "mode": MODE,
                "claim_scope": CLAIM_SCOPE,
                "created_utc": created_utc,
                "canonical": _expected_canonical_identity(authenticated),
                "renderer": renderer_identity,
                "runtime": runtime,
                "output_pdf": _identity_with_path(
                    pdf_stage,
                    recorded_path=pdf,
                ),
                "render": render,
                "stats": stats,
            }
            _write_json_stage(json_stage, evidence)

            with fig6_paper._publication_lock():
                # Rebind the cached, recertified result to the still-current
                # canonical bytes without replaying all 66 states. Holding the
                # canonical lock through final promotion prevents drift.
                _require_canonical_current_unlocked(authenticated)
                # Validate both staged files before exposing either final path.
                _read_diagnostic_record_unlocked(
                    pdf_path=pdf_stage,
                    json_path=json_stage,
                    recorded_pdf_path=pdf,
                    authenticated=authenticated,
                )
                old_payloads = {
                    pdf: pdf.read_bytes() if pdf.is_file() else None,
                    sidecar: sidecar.read_bytes() if sidecar.is_file() else None,
                }
                try:
                    # The JSON is the commit marker and MUST be promoted last.
                    os.replace(pdf_stage, pdf)
                    _read_diagnostic_record_unlocked(
                        pdf_path=pdf,
                        json_path=json_stage,
                        recorded_pdf_path=pdf,
                        authenticated=authenticated,
                    )
                    os.replace(json_stage, sidecar)
                    _read_diagnostic_record_unlocked(
                        pdf_path=pdf,
                        json_path=sidecar,
                        recorded_pdf_path=pdf,
                        authenticated=authenticated,
                    )
                except BaseException:
                    # Restore in the same data-first, commit-marker-last order.
                    _restore_payload(pdf, old_payloads[pdf])
                    _restore_payload(sidecar, old_payloads[sidecar])
                    raise
        finally:
            if pdf_stage is not None:
                pdf_stage.unlink(missing_ok=True)
            if json_stage is not None:
                json_stage.unlink(missing_ok=True)
    return pdf, sidecar


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Noncanonical output PDF path.")
    parser.add_argument("--json", type=Path, help="Evidence JSON sidecar path.")
    parser.add_argument(
        "--no-pchip",
        action="store_true",
        help="Plot stored markers only; omit visual PCHIP interpolation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pdf_path, json_path = generate_diagnostic(
        pdf_path=args.pdf,
        json_path=args.json,
        interpolate=not args.no_pchip,
    )
    print(f"Signed diagnostic PDF: {pdf_path}")
    print(f"Evidence sidecar:      {json_path}")
    print(f"Scope:                 {CLAIM_SCOPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
