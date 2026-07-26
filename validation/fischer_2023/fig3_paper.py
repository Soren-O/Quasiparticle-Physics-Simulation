"""Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction.

Four curves at $\\tau_\\ell / \\tau_0^{PB} \\in \\{0, 0.1, 1, 10\\}$ on the
paper's energy grid, with $\\tau_0^{PB}$ extracted through the phonon-side
Kaplan pair-breaking rate.

* Paper grid: ``NE = 1620``, ``dE = 1 μeV``, integer-commensurate with
  $\\omega_0 = \\Delta_0/9 = 20\\,\\mu$eV.
* Paper legend ratios: $\\{0, 0.1, 1, 10\\}$.
* Continuation through intermediate ratios for stability of the strong-
  bottleneck branch.
* Paper-style axis: $f(E)$ vs $E/\\Delta_0 - 1$ on $[0, 4]$, with photon-
  step markers at $n\\,\\omega_0/\\Delta_0$.

The module uses the F&C/Kaplan phonon-side pair-breaking kernel and the
analytic near-threshold S_+ quadrature correction, giving tau_0^PB ~= 255 ps
for the Table I parameters.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:

    Δ_0     = 180 μeV
    τ_0     = 438 ns
    T_bath  = 0.1 K
    ω_0     = Δ_0 / 9 = 20 μeV
    n̄       = 1 × 10^7
    c_phot  = 1 Hz = 1 × 10^-9 ns^-1

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from qpsim.constants import KB_UEV_PER_K

from validation import source_provenance, sweep_cache
from validation.fischer_2023 import fig3_solve
from validation.fischer_2023 import (
    steady_state_certificate as certificate_module,
)
from validation.fischer_2023.fig3_solve import (
    C_PHOT,
    CONTINUATION_RATIOS,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    N_BAR,
    NUM_BINS,
    OMEGA_0,
    PAPER_RATIOS,
    T_BATH,
    T_C,
    TARGET_BACKWARD_ERROR_LIMIT,
    TAU_0,
    Fig3StepEvent,
    _build_grid_and_spectral,
    _compute_tau_0_pb,
    solve,
    solver_fingerprint,
)

CURVE_REGRESSION_RTOL = 1e-4
CURVE_REGRESSION_ATOL_OVER_PEAK = 1e-6
STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL = 1.5e-2


def curve_regression_rtol(
    ratio: float,
    *,
    pinned_on: str,
    running_on: str | None = None,
) -> float:
    """Return the measured fixed-grid curve envelope for this platform.

    Ratios through one remain stable at the strict regression tolerance on
    Windows and Linux.  The ratio-10 state is a residual-polished,
    near-degenerate strong-bottleneck solution: exact single-thread Linux
    runs differ from the Windows pin by 1.27723% while agreeing with each
    other to about 1e-8 relative.  Keep the wider envelope restricted to that
    curve and only for the Windows/Linux OS-family case calibrated by those
    hosted runs.
    """
    current = sys.platform if running_on is None else running_on
    measured_pair = {pinned_on, current} == {"win32", "linux"}
    if ratio == 10.0 and measured_pair:
        return STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL
    return CURVE_REGRESSION_RTOL


@dataclass(frozen=True)
class Fig3PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    tau_0_pb_ns: float
    ratios: tuple[float, ...]   # paper ratios {0, 0.1, 1, 10}
    f_by_ratio: dict[float, np.ndarray]
    f_FD: np.ndarray            # thermal reference at T_bath
    certificate_maxima: dict[str, float]
    producer_solve_contract_digest: str
    validated_solve_contract_digest: str


def _certificate_maxima(raw: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Reduce per-ratio certificate arrays to compact artifact provenance."""
    ratios = np.asarray(raw["ratios"], dtype=float)
    expected_shape = (ratios.size,)
    zero_ratio = ratios == 0.0
    maxima: dict[str, float] = {}
    for name in certificate_module.NUMBER_CERTIFICATE_FIELDS:
        if name not in raw:
            raise ValueError(
                f"Fig. 3 raw solve payload is missing certificate field {name!r}."
            )
        values = np.asarray(raw[name], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"Fig. 3 certificate field {name!r} has shape {values.shape}; "
                f"expected {expected_shape}."
            )
        if name.startswith("phonon_"):
            if np.any(~np.isnan(values[zero_ratio])):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be NaN exactly "
                    "for the ratio-zero thermal-phonon shortcut."
                )
            dynamic_values = values[~zero_ratio]
            if np.any(~np.isfinite(dynamic_values)) or np.any(dynamic_values < 0.0):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be finite and "
                    "non-negative at every positive ratio."
                )
            maxima[name] = (
                float(np.max(dynamic_values))
                if dynamic_values.size
                else float("nan")
            )
        else:
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be finite and "
                    "non-negative at every ratio."
                )
            maxima[name] = float(np.max(values))
    return maxima


def observables(
    raw: Mapping[str, np.ndarray],
    *,
    producer_solve_contract_digest: str,
    validated_solve_contract_digest: str,
) -> Fig3PaperResult:
    """Repackage a raw :func:`fig3_solve.solve` payload into a Fig3PaperResult.

    Fig. 3's plotted quantity is the converged f(E) per ratio, so there is no
    expensive downstream derivation here — this is a pure unpacking of the cached
    arrays. It exists to keep the ``run() = observables(solve())`` split uniform
    with the other figures and to be the boundary the cache leaves uncached.
    """
    E = np.asarray(raw["E"], dtype=float)
    f_FD = np.asarray(raw["f_FD"], dtype=float)
    ratios = tuple(float(r) for r in raw["ratios"])
    tau_0_pb_ns = float(np.asarray(raw["tau_0_pb_ns"]).reshape(-1)[0])
    f_ratios = np.asarray(raw["f_ratios"], dtype=float)
    f_by_ratio = {r: f_ratios[i] for i, r in enumerate(ratios)}
    return Fig3PaperResult(
        E=E,
        tau_0_pb_ns=tau_0_pb_ns,
        ratios=ratios,
        f_by_ratio=f_by_ratio,
        f_FD=f_FD,
        certificate_maxima=_certificate_maxima(raw),
        producer_solve_contract_digest=producer_solve_contract_digest,
        validated_solve_contract_digest=validated_solve_contract_digest,
    )


def solve_contract_digest() -> str:
    """Return a runtime-neutral digest of the Fig. 3 numerical contract.

    The restart/cache identity deliberately also includes Python, NumPy,
    SciPy, BLAS, platform, and thread controls. Those are useful producer
    provenance but cannot be a cross-platform baseline-currentness gate.
    This digest instead binds only the conservative qpsim solve tree and the
    Fig. 3 solver/certificate sources; the separately stamped configuration
    fields bind the resolved grid and physics parameters.
    """
    digest = hashlib.sha256()
    ingredients = (
        ("qpsim", sweep_cache.solve_source_digest()),
        ("solve_source_digest", inspect.getsource(sweep_cache.solve_source_digest)),
        (
            "canonical_source_bytes",
            inspect.getsource(source_provenance.canonical_source_bytes),
        ),
        ("fig3_solve", inspect.getsource(fig3_solve)),
        ("certificate", inspect.getsource(certificate_module)),
    )
    for label, source in ingredients:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def run(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    on_step: Callable[[Fig3StepEvent], None] | None = None,
    restart_checkpoint_path: Path | None = None,
) -> Fig3PaperResult:
    """Solve Fischer Fig. 3 and repackage — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this (no args) so it always truly recomputes against the pinned
    baseline; the cached dev / regen path is :func:`run_cached`. Restart state
    is likewise opt-in: pass ``restart_checkpoint_path`` explicitly when an
    interruptible long run should resume. A completed checkpoint is retained;
    no automatic cleanup is performed, so the persistence owner should remove
    it explicitly only after the final artifact is durable.
    """
    _kwargs, _fingerprint, _extra_source, identity = _solve_cache_inputs(
        num_bins=num_bins,
        paper_ratios=paper_ratios,
        continuation_ratios=continuation_ratios,
    )
    solve_kwargs: dict[str, Any] = {
        "num_bins": num_bins,
        "paper_ratios": paper_ratios,
        "continuation_ratios": continuation_ratios,
        "on_step": on_step,
    }
    if restart_checkpoint_path is not None:
        solve_kwargs.update(
            checkpoint_path=Path(restart_checkpoint_path),
            checkpoint_identity=identity,
        )
    contract_digest = solve_contract_digest()
    return observables(
        solve(**solve_kwargs),
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def _solve_cache_inputs(
    *,
    num_bins: int,
    paper_ratios: tuple[float, ...],
    continuation_ratios: tuple[float, ...],
) -> tuple[dict[str, object], dict[str, Any], str, str]:
    """Return the one canonical content identity for cache and restart state."""
    kwargs: dict[str, object] = {
        "num_bins": int(num_bins),
        "paper_ratios": [float(r) for r in paper_ratios],
        "continuation_ratios": [float(r) for r in continuation_ratios],
    }
    fingerprint = solver_fingerprint(num_bins=num_bins)
    extra_source = (
        inspect.getsource(fig3_solve)
        + inspect.getsource(certificate_module)
    )
    identity = sweep_cache.cache_key(
        "fischer_2023/fig3",
        fingerprint,
        kwargs,
        extra_source=extra_source,
    )
    return kwargs, fingerprint, extra_source, identity


def _default_restart_checkpoint_path(identity: str) -> Path:
    return (
        sweep_cache.default_cache_dir()
        / "fischer_2023__fig3_restart"
        / f"{identity}.npz"
    )


def run_cached(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    restart_checkpoint_path: Path | None = None,
) -> Fig3PaperResult:
    """Like :func:`run`, but the expensive continuation solve is served from the
    disk cache when nothing solve-relevant has changed (see
    :mod:`validation.sweep_cache`). Used by the regen / ``__main__`` path; editing
    the plotting / observable code here does not invalidate the cached solve.
    Disable with ``QPSIM_SWEEP_CACHE=0``. In that mode the solve is genuinely
    fresh: no default restart state is read or written. Pass
    ``restart_checkpoint_path`` explicitly to opt back into interruption
    recovery while leaving the final-result cache disabled. Completed restart
    files are retained until explicitly removed by the persistence owner.
    """
    kwargs, fingerprint, extra_source, identity = _solve_cache_inputs(
        num_bins=num_bins,
        paper_ratios=paper_ratios,
        continuation_ratios=continuation_ratios,
    )
    checkpoint_path = (
        Path(restart_checkpoint_path)
        if restart_checkpoint_path is not None
        else (
            _default_restart_checkpoint_path(identity)
            if sweep_cache.is_enabled()
            else None
        )
    )

    def solve_current() -> Mapping[str, np.ndarray]:
        solve_kwargs: dict[str, Any] = {
            "num_bins": num_bins,
            "paper_ratios": paper_ratios,
            "continuation_ratios": continuation_ratios,
        }
        if checkpoint_path is not None:
            solve_kwargs.update(
                checkpoint_path=checkpoint_path,
                checkpoint_identity=identity,
            )
        return solve(**solve_kwargs)

    raw = sweep_cache.cached_solve(
        "fischer_2023/fig3",
        solve_current,
        fingerprint=fingerprint,
        kwargs=kwargs,
        extra_source=extra_source,
    )
    contract_digest = solve_contract_digest()
    return observables(
        raw,
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer_fig3_paper.csv`` because the tau_0^PB normalization is
    now pinned to the paper/Kaplan phonon-side pair-breaking rate.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer_fig3_paper.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


_TAU_0_PB_RE = re.compile(r"tau_0_pb_ns=([\deE.+-]+)")
_RATIOS_RE = re.compile(r"ratios=\[([^\]]+)\]")
_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_CERTIFICATE_VERSION_RE = re.compile(r"certificate_metric_version='([^']*)'")
_CERTIFICATE_LIMIT_RE = re.compile(
    r"target_backward_error_limit=([\deE.+-]+)"
)
_CERTIFICATE_MAXIMA_RE = re.compile(r"^# certificate_maxima\s+(.+)$", re.MULTILINE)
_PIN_PLATFORM_RE = re.compile(r"^# pinned_on: ([^\r\n]+)$", re.MULTILINE)
_PRODUCER_SOLVE_CONTRACT_DIGEST_RE = re.compile(
    r"^# producer_solve_contract_digest=([0-9a-f]{64})$",
    re.MULTILINE,
)
_VALIDATED_SOLVE_CONTRACT_DIGEST_RE = re.compile(
    r"^# validated_solve_contract_digest=([0-9a-f]{64})$",
    re.MULTILINE,
)
_CERTIFIED_BACKWARD_ERROR_FIELDS = (
    "qp_backward_error",
    "qp_number_backward_error",
    "phonon_backward_error",
)
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_bath": re.compile(r"T_bath=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "n_bar": re.compile(r"n_bar=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
}


def _read_baseline_text(path: Path) -> str:
    """Read current UTF-8 artifacts and legacy Windows-local CSV output."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Historical ``path.open()`` used the platform default. A baseline
        # generated on Windows therefore encoded the em dash as cp1252 even
        # though Linux-generated pinned artifacts were UTF-8.
        return path.read_text(encoding="cp1252")


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and replace ``path`` on success."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            yield stream
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_certificate_maxima(
    maxima: Mapping[str, float],
    *,
    target: float,
    context: str,
) -> dict[str, float]:
    """Return a normalized maxima map or reject an uncertified artifact."""
    expected = set(certificate_module.NUMBER_CERTIFICATE_FIELDS)
    if set(maxima) != expected:
        missing = sorted(expected.difference(maxima))
        extra = sorted(set(maxima).difference(expected))
        raise RuntimeError(
            f"{context} certificate maxima fields are incomplete; "
            f"missing={missing}, extra={extra}."
        )
    normalized = {
        name: float(maxima[name])
        for name in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for name, value in normalized.items():
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(
                f"{context} certificate maximum {name!r} must be finite "
                "and non-negative."
            )
    for name in _CERTIFIED_BACKWARD_ERROR_FIELDS:
        if normalized[name] > target:
            raise RuntimeError(
                f"{context} certificate maximum {name!r} is "
                f"{normalized[name]:.17g}, above target {target:.17g}."
            )
    return normalized


def _parse_certificate_maxima(text: str, path: Path) -> dict[str, float]:
    matches = _CERTIFICATE_MAXIMA_RE.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Baseline header at {path} must contain exactly one "
            "'# certificate_maxima' record."
        )
    maxima: dict[str, float] = {}
    for item in matches[0].split():
        name, separator, value = item.partition("=")
        if not separator or name in maxima:
            raise RuntimeError(
                f"Baseline header at {path} has malformed certificate maxima."
            )
        try:
            maxima[name] = float(value)
        except ValueError as exc:
            raise RuntimeError(
                f"Baseline header at {path} has a non-numeric certificate maximum."
            ) from exc
    return maxima


def _validate_result_for_artifact(result: Fig3PaperResult) -> None:
    energies = np.asarray(result.E, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise RuntimeError("Fig. 3 artifact energies must be a non-empty 1-D array.")
    if not np.all(np.isfinite(energies)) or np.any(np.diff(energies) <= 0.0):
        raise RuntimeError("Fig. 3 artifact energies must be finite and increasing.")
    ratios = np.asarray(result.ratios, dtype=float)
    if (
        ratios.ndim != 1
        or ratios.size == 0
        or not np.all(np.isfinite(ratios))
        or np.unique(ratios).size != ratios.size
    ):
        raise RuntimeError("Fig. 3 artifact ratios must be finite and unique.")
    if not np.isfinite(result.tau_0_pb_ns) or result.tau_0_pb_ns <= 0.0:
        raise RuntimeError("Fig. 3 artifact tau_0_pb_ns must be finite and positive.")
    thermal = np.asarray(result.f_FD, dtype=float)
    if thermal.shape != energies.shape or not np.all(np.isfinite(thermal)):
        raise RuntimeError("Fig. 3 artifact f_FD must be finite on the energy grid.")
    if set(result.f_by_ratio) != set(result.ratios):
        raise RuntimeError("Fig. 3 artifact curve keys must exactly match its ratios.")
    for ratio in result.ratios:
        values = np.asarray(result.f_by_ratio[ratio], dtype=float)
        if values.shape != energies.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Fig. 3 artifact curve at ratio {ratio:g} must be finite "
                "on the energy grid."
            )
    _validate_certificate_maxima(
        result.certificate_maxima,
        target=TARGET_BACKWARD_ERROR_LIMIT,
        context="Fig. 3 artifact",
    )
    for name, digest in (
        ("producer_solve_contract_digest", result.producer_solve_contract_digest),
        ("validated_solve_contract_digest", result.validated_solve_contract_digest),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"Fig. 3 artifact {name} must be a lowercase 64-character "
                "SHA-256 content identity."
            )


def write_baseline(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Write the four paper-ratio f(E) arrays + thermal reference to CSV."""
    if path is None:
        path = baseline_path()
    _validate_result_for_artifact(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["E_uev", "f_FD"] + [f"f_ratio_{r:g}" for r in result.ratios]
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction"
        ])
        writer.writerow([f"# pinned_on: {sys.platform}"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} "
            f"omega_0={OMEGA_0} n_bar={N_BAR} c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={result.E.size} E_min={E_MIN_FACTOR}*Delta "
            f"E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow([f"# tau_0_pb_ns={result.tau_0_pb_ns} ratios={list(result.ratios)}"])
        writer.writerow([
            "# producer_solve_contract_digest="
            f"{result.producer_solve_contract_digest}"
        ])
        writer.writerow([
            "# validated_solve_contract_digest="
            f"{result.validated_solve_contract_digest}"
        ])
        writer.writerow([
            "# certificate_metric_version="
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r} "
            "target_backward_error_limit="
            f"{TARGET_BACKWARD_ERROR_LIMIT}"
        ])
        writer.writerow([
            "# certificate_maxima "
            + " ".join(
                f"{name}={result.certificate_maxima.get(name, float('nan')):.17e}"
                for name in certificate_module.NUMBER_CERTIFICATE_FIELDS
            )
        ])
        writer.writerow(header_cols)
        n = result.E.size
        for i in range(n):
            row = [f"{result.E[i]:.17e}", f"{result.f_FD[i]:.17e}"]
            row.extend(f"{result.f_by_ratio[r][i]:.17e}" for r in result.ratios)
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig3PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig3PaperResult`."""
    if path is None:
        path = baseline_path()
    metadata = read_baseline_metadata(path)
    rows: list[list[float]] = []
    tau_0_pb: float | None = None
    ratios: tuple[float, ...] = ()
    ratio_records = 0
    columns: tuple[str, ...] | None = None
    reader = csv.reader(_read_baseline_text(path).splitlines())
    for line in reader:
        if not line:
            continue
        first = line[0]
        if first.startswith("# tau_0_pb_ns"):
            ratio_records += 1
            if ratio_records != 1:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has duplicate ratio metadata."
                )
            m_tau = _TAU_0_PB_RE.search(first)
            m_ratios = _RATIOS_RE.search(first)
            if m_tau:
                tau_0_pb = float(m_tau.group(1))
            if m_ratios:
                ratios = tuple(
                    float(x.strip()) for x in m_ratios.group(1).split(",") if x.strip()
                )
            continue
        if first.startswith("#"):
            continue
        if first == "E_uev":
            current_columns = tuple(line)
            expected_columns = (
                "E_uev",
                "f_FD",
                *(f"f_ratio_{ratio:g}" for ratio in metadata.ratios),
            )
            if columns is not None:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has duplicate column headers."
                )
            if current_columns != expected_columns:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has columns {current_columns}; "
                    f"expected {expected_columns}."
                )
            columns = current_columns
            continue
        if columns is None:
            raise RuntimeError(
                f"Fig. 3 baseline at {path} has data before its column header."
            )
        expected_width = 2 + len(metadata.ratios)
        if len(line) != expected_width:
            raise RuntimeError(
                f"Fig. 3 baseline row has {len(line)} columns; expected "
                f"{expected_width}."
            )
        try:
            rows.append([float(x) for x in line])
        except ValueError as exc:
            raise RuntimeError(
                f"Fig. 3 baseline at {path} contains a non-numeric data row."
            ) from exc
    if tau_0_pb is None or not ratios:
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb_ns / ratios metadata.")
    if columns is None:
        raise RuntimeError(f"Fig. 3 baseline at {path} missing its column header.")
    if tau_0_pb != metadata.tau_0_pb_ns or ratios != metadata.ratios:
        raise RuntimeError(f"Fig. 3 baseline at {path} has inconsistent metadata.")
    if len(rows) != metadata.num_bins:
        raise RuntimeError(
            f"Fig. 3 baseline at {path} has {len(rows)} data rows; "
            f"expected NE={metadata.num_bins}."
        )
    data = np.array(rows, dtype=float)
    if data.shape != (metadata.num_bins, 2 + len(ratios)):
        raise RuntimeError(
            f"Fig. 3 baseline at {path} has data shape {data.shape}; expected "
            f"{(metadata.num_bins, 2 + len(ratios))}."
        )
    if not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0.0):
        raise RuntimeError(
            f"Fig. 3 baseline at {path} must contain finite rows on a strictly "
            "increasing energy grid."
        )
    # Column layout: E_uev, f_FD, f_ratio_<r0>, f_ratio_<r1>, ...
    return Fig3PaperResult(
        E=data[:, 0],
        tau_0_pb_ns=tau_0_pb,
        ratios=ratios,
        f_by_ratio={r: data[:, 2 + i] for i, r in enumerate(ratios)},
        f_FD=data[:, 1],
        certificate_maxima=dict(metadata.certificate_maxima),
        producer_solve_contract_digest=metadata.producer_solve_contract_digest,
        validated_solve_contract_digest=metadata.validated_solve_contract_digest,
    )


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config)
    without touching the data rows or running the continuation ladder.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds instead of after the several-minute
    run (see :mod:`fig6_paper` for the same pattern, where the payoff is ~14 h).
    """

    delta_0: float
    tau_0: float
    t_bath: float
    omega_0: float
    n_bar: float
    c_phot: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    tau_0_pb_ns: float
    ratios: tuple[float, ...]
    certificate_metric_version: str
    target_backward_error_limit: float
    certificate_maxima: dict[str, float]
    pinned_on: str
    producer_solve_contract_digest: str
    validated_solve_contract_digest: str


def read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve). Raises
    ``RuntimeError`` if any stamped field is missing — an old/malformed header
    should fail loudly rather than silently skip the check.
    """
    if path is None:
        path = baseline_path()
    text = _read_baseline_text(path)

    def _num(rx: re.Pattern[str], field: str) -> float:
        m = rx.search(text)
        if m is None:
            raise RuntimeError(
                f"Baseline header at {path} missing {field} metadata."
            )
        value = float(m.group(1))
        if not np.isfinite(value):
            raise RuntimeError(
                f"Baseline header at {path} has non-finite {field} metadata."
            )
        return value

    ne_m = _GRID_NE_RE.search(text)
    ratios_m = _RATIOS_RE.search(text)
    versions = _CERTIFICATE_VERSION_RE.findall(text)
    limits = _CERTIFICATE_LIMIT_RE.findall(text)
    pinned_platforms = _PIN_PLATFORM_RE.findall(text)
    producer_solve_contract_digests = (
        _PRODUCER_SOLVE_CONTRACT_DIGEST_RE.findall(text)
    )
    validated_solve_contract_digests = (
        _VALIDATED_SOLVE_CONTRACT_DIGEST_RE.findall(text)
    )
    if (
        ne_m is None
        or ratios_m is None
        or len(versions) != 1
        or len(limits) != 1
        or len(pinned_platforms) != 1
        or not pinned_platforms[0].strip()
        or len(producer_solve_contract_digests) != 1
        or len(validated_solve_contract_digests) != 1
    ):
        raise RuntimeError(
            f"Baseline header at {path} must contain NE / ratios and exactly "
            "one pin-platform / producer contract / validated contract / "
            "certificate metric / target record."
        )
    ratios = tuple(
        float(x.strip()) for x in ratios_m.group(1).split(",") if x.strip()
    )
    if (
        not ratios
        or not np.all(np.isfinite(ratios))
        or len(set(ratios)) != len(ratios)
    ):
        raise RuntimeError(
            f"Baseline header at {path} has empty, non-finite, or duplicate ratios."
        )
    version = versions[0]
    if version != certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION:
        raise RuntimeError(
            f"Baseline header at {path} uses unsupported certificate metric "
            f"{version!r}; expected "
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r}."
        )
    target = _num(_CERTIFICATE_LIMIT_RE, "target_backward_error_limit")
    if target != TARGET_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"Baseline header at {path} has certificate target {target:.17g}; "
            f"expected {TARGET_BACKWARD_ERROR_LIMIT:.17g}."
        )
    maxima = _validate_certificate_maxima(
        _parse_certificate_maxima(text, path),
        target=target,
        context=f"Fig. 3 baseline at {path}",
    )
    num_bins = int(ne_m.group(1))
    if num_bins <= 0:
        raise RuntimeError(f"Baseline header at {path} has non-positive NE.")
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_bath=_num(_HEADER_PARAM_RE["t_bath"], "T_bath"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        n_bar=_num(_HEADER_PARAM_RE["n_bar"], "n_bar"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        num_bins=num_bins,
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        tau_0_pb_ns=_num(_TAU_0_PB_RE, "tau_0_pb_ns"),
        ratios=ratios,
        certificate_metric_version=version,
        target_backward_error_limit=target,
        certificate_maxima=maxima,
        pinned_on=pinned_platforms[0].strip(),
        producer_solve_contract_digest=producer_solve_contract_digests[0],
        validated_solve_contract_digest=validated_solve_contract_digests[0],
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — computed without the (several-minute) continuation run.

    ``tau_0_pb_ns`` is produced by the exact :func:`_compute_tau_0_pb` call
    :func:`run` makes, so it can never drift from a real run; everything else
    is read straight off the module constants.
    """
    _, _, spectral = _build_grid_and_spectral()
    tau_0_pb = _compute_tau_0_pb(spectral)
    contract_digest = solve_contract_digest()
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_bath=T_BATH,
        omega_0=OMEGA_0,
        n_bar=N_BAR,
        c_phot=C_PHOT,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        tau_0_pb_ns=tau_0_pb,
        ratios=PAPER_RATIOS,
        certificate_metric_version=(
            certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
        ),
        target_backward_error_limit=TARGET_BACKWARD_ERROR_LIMIT,
        certificate_maxima={},
        pinned_on=sys.platform,
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def write_plot(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Paper-style plot: log-scale f(E) with all four ratios + thermal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    from validation.fischer_2023._paper_envelope import (
        EnvelopeParams,
        envelope_no_thermal,
        envelope_with_thermal,
        solve_b0,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # Paper Fig. 3 axis: f(E) vs E/Δ_0 - 1 on [0, 4].
    x = result.E / DELTA_0 - 1.0

    # Paper Fig. 3 palette (named matplotlib colors, matching the standalone
    # paper reproduction): solid grayscale numerical, dashed analytical.
    solid_colors = ["k", "dimgray", "gray", "lightgray"]
    dash_colors = ["red", "green", "blue", "blue"]
    Eplot = np.linspace(DELTA_0 + 1e-3, 5.0 * DELTA_0, 4000)
    TB_uev = T_BATH * KB_UEV_PER_K

    for r, sc, dc in zip(result.ratios, solid_colors, dash_colors, strict=True):
        ax.semilogy(
            x, np.maximum(result.f_by_ratio[r], 1e-40),
            color=sc, lw=1.0,
            label=rf"$\tau_\ell/\tau_0^{{\rm PB}}={r:g}$ (num)",
        )
        ep = EnvelopeParams(
            Delta0=DELTA_0,
            Tc_uev=T_C * KB_UEV_PER_K,
            tau0=TAU_0,
            tau0_PB=result.tau_0_pb_ns,
            tau_l=r * result.tau_0_pb_ns,
            TB_uev=TB_uev,
            nbar=N_BAR,
            omega0=OMEGA_0,
            cphot_QP=C_PHOT,
        )
        b0 = solve_b0(ep)
        f_env = (envelope_with_thermal(Eplot, ep, b0) if r == 0.0
                 else envelope_no_thermal(Eplot, ep, b0))
        ax.semilogy(Eplot / DELTA_0 - 1.0, np.maximum(f_env, 1e-40),
                    color=dc, ls="--", lw=0.8)

    ax.set_xlabel(r"$E/\Delta_0 - 1$")
    ax.set_ylabel(r"$f(E)$")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(1e-35, 3e-7)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(
            temporary,
            format=path.suffix.removeprefix("."),
            bbox_inches="tight",
        )
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, T_bath={T_BATH} K, "
        f"ω_0={OMEGA_0:.2f} μeV"
    )
    print(f"  Grid: NE={NUM_BINS}, dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    print(f"  Paper ratios: {list(PAPER_RATIOS)}")
    result = run_cached()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
