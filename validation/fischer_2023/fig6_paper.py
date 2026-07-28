"""Fischer 2023 Fig. 6 — paper-topology qpsim gap-suppression regression.

This is a **structural qpsim regression** on the paper's axes and parameters;
it does not compare against digitized paper-curve data. Gap suppression $\\delta\\Delta$
caused by the nonequilibrium distribution, plotted in the paper's
ordinate

    $\\frac{\\delta\\Delta_T - \\delta\\Delta}{\\delta\\Delta_T}
       = \\frac{\\Delta_\\mathrm{driven} - \\Delta_\\mathrm{eq}(T_B)}
              {\\Delta_0 - \\Delta_\\mathrm{eq}(T_B)},$

against the Eq. 35 drive-equivalent temperature ratio $T_*/\\Delta$, swept
over $\\bar n$ at three bath temperatures $T_B \\in \\{0.10, 0.15, 0.20\\}$ K
on a paper-parameter grid (1640 bins: the original 1620 above-gap cells plus
20 sub-gap guard cells, $dE = 1\\,\\mu$eV). Solid lines: numerical
joint kinetic-equation + self-consistent gap solve. Dashed lines:
analytical Eq. 53.

The ordinate is the paper's normalized form $(\\delta\\Delta_T - \\delta\\Delta)/\\delta\\Delta_T$,
which goes negative on the strong-drive side. The stored arrays resolve the
sign change; the canonical plot retains the paper's nonnegative window, while
the diagnostic ``--direct-gap`` plot expands its axes, uses a signed-log scale,
and marks every finite numerical point.

$\\tau_\\ell$ model
------------------

The paper sets $\\tau_\\ell = \\tau_0^{PB} \\approx 255$ ps for Fig. 6, and
that is the default here (``TAU_L_MODEL = "tau_0_pb"``, overridable via
the ``FISCHER2023_FIG6_TAU_L_MODEL`` environment variable). The
extracted $\\tau_0^{PB}$ diagnostic is pinned to the phonon-side
F&C/Kaplan pair-breaking rate and reproduces the paper-quoted ~255 ps
for the Table I parameters.

For comparison, :func:`qpsim.physics.acoustic_escape_tau_l` with
Fischer's 63 nm film and $\\eta = 0.2$ gives $\\tau_\\ell \\approx 368$ ps
(Debye-averaged sound velocity) — ~44 % longer than the paper's nominal
$\\tau_0^{PB}$. The dimensionless $T_*/\\Delta$ axis from Eq. 35 is
independent of $\\tau_\\ell$, so the x-axis is invariant under the model
choice; the y-axis position of the curves is sensitive to it.

Eq. 53 reads

       $\\delta\\Delta/\\Delta_0 = x_\\mathrm{qp} \\cdot
          \\bigl[1 - 0.42\\,(T_*/\\Delta_0) + 0.22\\,(T_*/\\Delta_0)^2\\bigr]$,

with qpsim's Fischer-convention $x_\\mathrm{qp}
= n_\\mathrm{qp}/(4\\rho_F\\Delta_0)$. The bracketed factor is verified
closed-form from the paper text, and the dashed overlay feeds it the
analytical Eq. 47 + Appendix-E
$x_\\mathrm{qp}$ from :func:`fig5_paper._xqp_analytic_eq47`, using the
same scalar $\\tau_\\ell$ as the numerical solve. The thermal counterpart
$\\delta\\Delta_T$ uses the BCS gap calibration in the denominator,
matching the numerical observable definition.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:
parameters identical to :mod:`fig3_paper` and :mod:`fig5_paper`.

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig6_paper
"""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import tempfile
import time
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import qpsim.observables.density as _density_mod
import qpsim.observables.gap_suppression as _gap_suppression_mod
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.observables.density import qp_fraction
from qpsim.physics import calibrate_gap, solve_gap
from qpsim.physics.spectral import SpectralContext

from validation import sweep_cache
from validation.fischer_2023 import (
    fig5_paper,
    fig5_solve,
    fig6_solve,
)
from validation.fischer_2023 import (
    steady_state_certificate as certificate_module,
)
from validation.fischer_2023.fig6_solve import (
    C_PHOT,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    FILM_THICKNESS_NM,
    FINITE_CUTOFF_DELTA0_OVER_KBTC,
    GAP_FIXED_POINT_ABS_TOL_UEV,
    OMEGA_0,
    SUBSTRATE_ETA,
    T_BATH_VALUES,
    T_C,
    TAU_0,
    TAU_L_MODEL,
    _build_grid_and_spectral,
    _build_state,
    _compute_tau_0_pb,
    _fischer_material,
    solve,
    solver_fingerprint,
)
from validation.fischer_2024 import _pdf as _pdf_validation
from validation.source_provenance import source_manifest, source_sha256

# Output-path suffix (does NOT affect the solve or cache key). ``__main__`` sets
# it for --fast and separately passes the explicit direct-mode flag to the path
# helpers. It also mutates solve-affecting knobs on :mod:`fig6_solve`.
_FAST_SUFFIX: str = ""


@dataclass(frozen=True)
class Fig6PaperResult:
    """Arrays returned by :func:`run`. Shape ``(n_T, n_n)`` for each grid."""

    tau_0_pb_ns: float
    tau_l_ns: float
    T_bath: np.ndarray  # shape (n_T,)
    n_bar: np.ndarray  # shape (n_n,)
    T_star_over_delta: np.ndarray  # shape (n_T, n_n) — Eq. 35
    delta_eq: np.ndarray  # shape (n_T,) — Δ_eq(T_B), thermal
    delta_driven: np.ndarray  # shape (n_T, n_n) — sc-gap solve
    delta_thermal_T_bath: np.ndarray  # shape (n_T,) — Δ_eq(T_B) reissue
    paper_observable_num: np.ndarray  # shape (n_T, n_n) — solid lines
    paper_observable_eq53: np.ndarray  # shape (n_T, n_n) — dashed lines
    x_qp_num: np.ndarray  # shape (n_T, n_n) — diagnostic
    x_qp_eq47: np.ndarray  # shape (n_T, n_n) — Eq. 47 diagnostic
    qp_residual_inf: np.ndarray  # shape (n_T, n_n) — raw balance
    qp_backward_error: np.ndarray  # shape (n_T, n_n) — normwise
    qp_number_backward_error: np.ndarray  # shape (n_T, n_n) — number mode
    phonon_residual_inf: np.ndarray  # shape (n_T, n_n) — raw balance
    phonon_raw_backward_error: np.ndarray  # shape (n_T, n_n) — raw normwise
    phonon_backward_error: np.ndarray  # shape (n_T, n_n) — certified normwise
    gap_fixed_point_abs_error_uev: np.ndarray  # shape (n_T, n_n) — gap map
    state_f: np.ndarray | None = None
    state_n_ph: np.ndarray | None = None


def observables(raw: Mapping[str, np.ndarray]) -> Fig6PaperResult:
    """Repackage a raw :func:`fig6_solve.solve` payload into a Fig6PaperResult.

    Fig. 6's numerical observable gates the solve's seed chain (a fold sits past
    each curve's peak), so it is computed inside the cached solve, not re-derived
    here — this is a pure unpack. Because the gated solve stores outputs of the
    qpsim ``gap_suppression`` (direct mode) and ``density`` modules **and** the
    Eq.47/Eq.53 analytic overlay (via ``fig5_paper._xqp_analytic_eq47``, which
    transitively reads ``fig5_solve``), :func:`run_cached` folds all of those
    sources into the cache key so editing any of them invalidates the cached
    solve.
    """
    delta_eq = np.asarray(raw["delta_eq"], dtype=float)
    return Fig6PaperResult(
        tau_0_pb_ns=float(np.asarray(raw["tau_0_pb_ns"]).reshape(-1)[0]),
        tau_l_ns=float(np.asarray(raw["tau_l_ns"]).reshape(-1)[0]),
        T_bath=np.asarray(raw["T_bath"], dtype=float),
        n_bar=np.asarray(raw["n_bar"], dtype=float),
        T_star_over_delta=np.asarray(raw["T_star_over_delta"], dtype=float),
        delta_eq=delta_eq,
        delta_driven=np.asarray(raw["delta_driven"], dtype=float),
        delta_thermal_T_bath=delta_eq.copy(),
        paper_observable_num=np.asarray(raw["paper_observable_num"], dtype=float),
        paper_observable_eq53=np.asarray(raw["paper_observable_eq53"], dtype=float),
        x_qp_num=np.asarray(raw["x_qp_num"], dtype=float),
        x_qp_eq47=np.asarray(raw["x_qp_eq47"], dtype=float),
        qp_residual_inf=np.asarray(raw["qp_residual_inf"], dtype=float),
        qp_backward_error=np.asarray(raw["qp_backward_error"], dtype=float),
        qp_number_backward_error=np.asarray(
            raw[certificate_module.QP_NUMBER_CERTIFICATE_FIELD],
            dtype=float,
        ),
        phonon_residual_inf=np.asarray(raw["phonon_residual_inf"], dtype=float),
        phonon_raw_backward_error=np.asarray(raw["phonon_raw_backward_error"], dtype=float),
        phonon_backward_error=np.asarray(raw["phonon_backward_error"], dtype=float),
        gap_fixed_point_abs_error_uev=np.asarray(raw["gap_fixed_point_abs_error_uev"], dtype=float),
        state_f=np.asarray(raw["state_f"], dtype=float),
        state_n_ph=np.asarray(raw["state_n_ph"], dtype=float),
    )


def validate_temperature_row_payload(
    raw: Mapping[str, np.ndarray],
    *,
    T_bath: float,
) -> Fig6PaperResult:
    """Reassemble and independently certify one campaign row payload."""
    result = observables(raw)
    _validate_result_for_artifact(
        result,
        expected_t_bath=np.asarray([float(T_bath)], dtype=float),
        expected_n_bar=np.asarray(fig6_solve.N_BAR_VALUES, dtype=float),
    )
    return result


def run(
    *,
    direct_gap_observable: bool = False,
    fixed_gap_kinetics: bool = False,
) -> Fig6PaperResult:
    """Solve Fischer Fig. 6 and repackage — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this so it always truly recomputes against the pinned baseline;
    the cached dev / regen path is :func:`run_cached`.
    """
    return observables(
        solve(
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=fixed_gap_kinetics,
        )
    )


def run_cached(
    *,
    direct_gap_observable: bool = False,
    fixed_gap_kinetics: bool = False,
) -> Fig6PaperResult:
    """Like :func:`run`, but the ~6.04 h serial solve is served from disk when
    nothing solve-relevant has changed (see :mod:`validation.sweep_cache`).

    Used by the regen / ``__main__`` path. Editing the plotting code here keeps
    the cached solve warm; editing :mod:`fig6_solve`, the qpsim solver subtree,
    the qpsim ``gap_suppression`` / ``density`` modules, or ``fig5_paper`` /
    ``fig5_solve`` (the Eq.47 overlay helper) — all of whose outputs the gated
    solve stores — invalidates it. Disable with ``QPSIM_SWEEP_CACHE=0``. The
    grid / n̄ / picard-tol live on :mod:`fig6_solve` (mutated by --fast), so the
    fingerprint distinguishes a --fast run from a paper run.
    """
    fig6_solve._require_supported_mode(
        direct_gap_observable=direct_gap_observable,
        fixed_gap_kinetics=fixed_gap_kinetics,
    )
    extra_source = (
        inspect.getsource(fig6_solve)
        + inspect.getsource(_gap_suppression_mod)
        + inspect.getsource(_density_mod)
        # fig6's solve stores the Eq.47/Eq.53 analytic overlay, computed via
        # fig5_paper._xqp_analytic_eq47 (which transitively reads fig5_solve's
        # Table-I constants / _A_EQ35). Those modules live under validation/ —
        # outside solve_source_digest's qpsim/ scope — so fold their source in,
        # else run_cached() could serve a stale overlay after a fig5 edit.
        + inspect.getsource(fig5_paper)
        + inspect.getsource(fig5_solve)
        + inspect.getsource(certificate_module)
    )
    kwargs = {
        "direct_gap_observable": bool(direct_gap_observable),
        "fixed_gap_kinetics": bool(fixed_gap_kinetics),
    }
    raw = sweep_cache.cached_solve(
        "fischer_2023/fig6",
        lambda: solve(
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=fixed_gap_kinetics,
        ),
        fingerprint=solver_fingerprint(),
        kwargs=kwargs,
        extra_source=extra_source,
    )
    return observables(raw)


def baseline_path(*, direct_gap_observable: bool = False) -> Path:
    """Output CSV path.

    The filename follows the paper figure, while the module docstring records the
    remaining $\\tau_\\ell$ convention gap. ``--fast`` runs append a
    ``_fast`` suffix via :data:`_FAST_SUFFIX` so dev baselines do not
    clobber the canonical qpsim regression CSV. Direct-mode calls always receive a
    ``_direct`` suffix, including programmatic generation outside ``__main__``.
    """
    root = Path(__file__).resolve().parents[2]
    mode_suffix = "_direct" if direct_gap_observable else ""
    return (
        root
        / "validation"
        / "baselines"
        / "ph0_kaplan"
        / f"fischer_fig6_paper{mode_suffix}{_FAST_SUFFIX}.csv"
    )


def plot_path(*, direct_gap_observable: bool = False) -> Path:
    return baseline_path(
        direct_gap_observable=direct_gap_observable,
    ).with_suffix(".pdf")


_TAU_0_PB_RE = re.compile(r"tau_0_pb_ns=([\deE.+-]+)")
_TAU_L_RE = re.compile(r"tau_l_ns=([\deE.+-]+)")
_TAU_L_MODEL_RE = re.compile(r"TAU_L_MODEL='([^']*)'")
_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_FINITE_CUTOFF_RATIO_RE = re.compile(r"finite_cutoff_delta0_over_kBTc=([\deE.+-]+)")
_GAP_FIXED_POINT_ABS_TOL_RE = re.compile(r"gap_fixed_point_abs_tol_ueV=([\deE.+-]+)")
_CERTIFICATE_METRIC_VERSION_RE = re.compile(r"certificate_metric_version=([^\s]+)")
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_c": re.compile(r"T_c=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
    "film_thickness_nm": re.compile(r"film_thickness_nm=([\deE.+-]+)"),
    "eta": re.compile(r"eta=([\deE.+-]+)"),
}

FIG6_BASELINE_COLUMNS: tuple[str, ...] = (
    "T_bath_K",
    "n_bar",
    "T_star_over_delta",
    "delta_eq_T_bath_ueV",
    "delta_driven_ueV",
    "x_qp_num",
    "x_qp_eq47",
    "paper_observable_num",
    "paper_observable_eq53",
    "qp_residual_inf",
    "qp_backward_error",
    "phonon_residual_inf",
    "phonon_raw_backward_error",
    "phonon_backward_error",
    "gap_fixed_point_abs_error_uev",
)
_LEGACY_BASELINE_COLUMNS = FIG6_BASELINE_COLUMNS[:9]
_OLD_CERTIFIED_BASELINE_COLUMNS = (
    *_LEGACY_BASELINE_COLUMNS,
    "qp_residual_inf",
    "qp_backward_error",
    "phonon_residual_inf",
    "phonon_backward_error",
)
_SUPPORTED_BASELINE_COLUMNS = {
    _LEGACY_BASELINE_COLUMNS,
    _OLD_CERTIFIED_BASELINE_COLUMNS,
    FIG6_BASELINE_COLUMNS,
}

ARTIFACT_SCHEMA = "qpsim.fischer2023.fig6_paper.v2"
_LEGACY_CANONICAL_SHA256 = "a1e10233b3898de05244a8812dd71b64bab1634c93eb8172e98109245f313ba9"
_ARTIFACT_CERTIFICATE_FIELDS = fig6_solve.FIG6_CERTIFICATE_FIELDS
_ARTIFACT_COLUMNS: tuple[str, ...] = (
    *FIG6_BASELINE_COLUMNS[:11],
    certificate_module.QP_NUMBER_CERTIFICATE_FIELD,
    *FIG6_BASELINE_COLUMNS[11:],
    "state_f_f64_zlib_base64",
    "state_n_ph_f64_zlib_base64",
    "state_sha256",
)
_CERTIFIED_FIELDS = (
    "qp_backward_error",
    certificate_module.QP_NUMBER_CERTIFICATE_FIELD,
    "phonon_backward_error",
)
_METADATA_KEYS = {
    "certificate_fields",
    "certificate_maxima",
    "certificate_metric_version",
    "certificate_target_backward_error",
    "columns",
    "fingerprint",
    "payload_sha256",
    "row_count",
    "schema",
}


class LegacyArtifactError(RuntimeError):
    """Raised only for the exact known pre-schema canonical Fig. 6 CSV."""


class ArtifactValidationError(RuntimeError):
    """Raised when a current-schema Fig. 6 artifact is incomplete or stale."""


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Write beside the destination, fsync, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _legacy_write_baseline(result: Fig6PaperResult, path: Path | None = None) -> Path:
    """Write a flat row-per-(T_bath, n_bar) CSV with all observables."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["# Fischer 2023 Fig. 6 — paper-topology qpsim regression"])
        writer.writerow(
            [
                f"# Delta_0={DELTA_0:.17g} tau_0={TAU_0:.17g} "
                f"T_c={T_C:.17g} "
                "finite_cutoff_delta0_over_kBTc="
                f"{FINITE_CUTOFF_DELTA0_OVER_KBTC:.17g} "
                f"omega_0={OMEGA_0:.17g} c_phot={C_PHOT:.17g} "
                f"film_thickness_nm={FILM_THICKNESS_NM:.17g} "
                f"eta={SUBSTRATE_ETA:.17g}"
            ]
        )
        writer.writerow(
            [
                f"# Grid: NE={fig6_solve.NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
            ]
        )
        writer.writerow(
            [
                f"# tau_0_pb_ns={result.tau_0_pb_ns}  "
                f"tau_l_ns={result.tau_l_ns}  (TAU_L_MODEL={TAU_L_MODEL!r}) "
                f"gap_fixed_point_abs_tol_ueV={GAP_FIXED_POINT_ABS_TOL_UEV:.17g} "
                "certificate_metric_version="
                f"{certificate_module.CERTIFICATE_METRIC_VERSION}"
            ]
        )
        writer.writerow(FIG6_BASELINE_COLUMNS)
        for i, T_bath in enumerate(result.T_bath):
            for j, n_bar in enumerate(result.n_bar):
                writer.writerow(
                    [
                        f"{T_bath:.17e}",
                        f"{n_bar:.17e}",
                        f"{result.T_star_over_delta[i, j]:.17e}",
                        f"{result.delta_eq[i]:.17e}",
                        f"{result.delta_driven[i, j]:.17e}",
                        f"{result.x_qp_num[i, j]:.17e}",
                        f"{result.x_qp_eq47[i, j]:.17e}",
                        f"{result.paper_observable_num[i, j]:.17e}",
                        f"{result.paper_observable_eq53[i, j]:.17e}",
                        f"{result.qp_residual_inf[i, j]:.17e}",
                        f"{result.qp_backward_error[i, j]:.17e}",
                        f"{result.phonon_residual_inf[i, j]:.17e}",
                        f"{result.phonon_raw_backward_error[i, j]:.17e}",
                        f"{result.phonon_backward_error[i, j]:.17e}",
                        f"{result.gap_fixed_point_abs_error_uev[i, j]:.17e}",
                    ]
                )
    return path


def _legacy_read_baseline(path: Path | None = None) -> Fig6PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig6PaperResult`."""
    if path is None:
        path = baseline_path()
    tau_0_pb: float | None = None
    tau_l: float | None = None
    rows: list[tuple[float, ...]] = []
    data_columns: tuple[str, ...] | None = None
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# tau_0_pb_ns"):
                m_pb = _TAU_0_PB_RE.search(first)
                m_l = _TAU_L_RE.search(first)
                if m_pb:
                    tau_0_pb = float(m_pb.group(1))
                if m_l:
                    tau_l = float(m_l.group(1))
                continue
            if first.startswith("#"):
                continue
            if first == "T_bath_K":
                candidate_columns = tuple(line)
                if candidate_columns not in _SUPPORTED_BASELINE_COLUMNS:
                    raise RuntimeError(
                        f"Fig. 6 baseline has an unsupported data header: {candidate_columns!r}."
                    )
                if data_columns is not None:
                    raise RuntimeError("Fig. 6 baseline contains multiple data headers.")
                data_columns = candidate_columns
                continue
            if data_columns is None:
                raise RuntimeError("Fig. 6 baseline data appeared before a supported header.")
            if len(line) != len(data_columns):
                raise RuntimeError(
                    f"Fig. 6 baseline row has {len(line)} columns but its "
                    f"header declares {len(data_columns)}."
                )
            values = tuple(float(value) for value in line)
            if len(values) == 9:
                # Backward compatibility with pre-certificate baselines.
                values += (float("nan"),) * 6
            elif len(values) == 13:
                # Old certified schema: the four appended columns were
                # qp residual/BE and phonon residual/certified BE. Insert the
                # later raw-phonon and gap-map fields without reinterpreting
                # that final phonon value.
                values = (
                    *values[:12],
                    float("nan"),
                    values[12],
                    float("nan"),
                )
            elif len(values) != 15:
                raise RuntimeError(
                    f"Fig. 6 baseline row has {len(values)} columns; expected "
                    "the legacy 9-column, old-certified 13-column, or "
                    "current 15-column schema."
                )
            rows.append(values)
    if tau_0_pb is None or tau_l is None:
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb_ns / tau_l_ns metadata.")
    if data_columns is None or not rows:
        raise RuntimeError(f"Fig. 6 baseline at {path} has no data rows.")
    coordinates = [(row[0], row[1]) for row in rows]
    if any(not np.all(np.isfinite(coordinate)) for coordinate in coordinates):
        raise RuntimeError("Fig. 6 baseline coordinates must be finite.")
    if len(set(coordinates)) != len(coordinates):
        raise RuntimeError("Fig. 6 baseline contains duplicate (T_bath, n_bar) rows.")
    T_bath_unique = sorted({r[0] for r in rows})
    n_bar_unique = sorted({r[1] for r in rows})
    expected_coordinates = {(T_bath, n_bar) for T_bath in T_bath_unique for n_bar in n_bar_unique}
    missing_coordinates = expected_coordinates - set(coordinates)
    if missing_coordinates:
        raise RuntimeError(
            "Fig. 6 baseline is missing Cartesian (T_bath, n_bar) rows: "
            f"{sorted(missing_coordinates)!r}."
        )
    n_T = len(T_bath_unique)
    n_n = len(n_bar_unique)
    T_idx = {t: i for i, t in enumerate(T_bath_unique)}
    n_idx = {n: i for i, n in enumerate(n_bar_unique)}
    T_star = np.full((n_T, n_n), np.nan)
    delta_driven = np.full((n_T, n_n), np.nan)
    x_qp_num = np.full((n_T, n_n), np.nan)
    x_qp_eq47 = np.full((n_T, n_n), np.nan)
    obs_num = np.full((n_T, n_n), np.nan)
    obs_eq53 = np.full((n_T, n_n), np.nan)
    qp_residual_inf = np.full((n_T, n_n), np.nan)
    qp_backward_error = np.full((n_T, n_n), np.nan)
    phonon_residual_inf = np.full((n_T, n_n), np.nan)
    phonon_raw_backward_error = np.full((n_T, n_n), np.nan)
    phonon_backward_error = np.full((n_T, n_n), np.nan)
    gap_fixed_point_abs_error_uev = np.full((n_T, n_n), np.nan)
    delta_eq_per_T = np.full(n_T, np.nan)
    for (
        T_bath,
        n_bar,
        ts,
        deq,
        ddr,
        xq,
        xq47,
        on,
        oe,
        qp_inf,
        qp_be,
        ph_inf,
        ph_raw_be,
        ph_be,
        gap_abs_error,
    ) in rows:
        i, j = T_idx[T_bath], n_idx[n_bar]
        T_star[i, j] = ts
        delta_driven[i, j] = ddr
        x_qp_num[i, j] = xq
        x_qp_eq47[i, j] = xq47
        obs_num[i, j] = on
        obs_eq53[i, j] = oe
        qp_residual_inf[i, j] = qp_inf
        qp_backward_error[i, j] = qp_be
        phonon_residual_inf[i, j] = ph_inf
        phonon_raw_backward_error[i, j] = ph_raw_be
        phonon_backward_error[i, j] = ph_be
        gap_fixed_point_abs_error_uev[i, j] = gap_abs_error
        delta_eq_per_T[i] = deq

    return Fig6PaperResult(
        tau_0_pb_ns=tau_0_pb,
        tau_l_ns=tau_l,
        T_bath=np.array(T_bath_unique),
        n_bar=np.array(n_bar_unique),
        T_star_over_delta=T_star,
        delta_eq=delta_eq_per_T,
        delta_driven=delta_driven,
        delta_thermal_T_bath=delta_eq_per_T.copy(),
        paper_observable_num=obs_num,
        paper_observable_eq53=obs_eq53,
        x_qp_num=x_qp_num,
        x_qp_eq47=x_qp_eq47,
        qp_residual_inf=qp_residual_inf,
        qp_backward_error=qp_backward_error,
        qp_number_backward_error=np.full_like(qp_backward_error, np.nan),
        phonon_residual_inf=phonon_residual_inf,
        phonon_raw_backward_error=phonon_raw_backward_error,
        phonon_backward_error=phonon_backward_error,
        gap_fixed_point_abs_error_uev=gap_fixed_point_abs_error_uev,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(rows: list[list[str]]) -> str:
    return hashlib.sha256(_canonical_json(rows).encode("ascii")).hexdigest()


def _array_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f8", order="C").tobytes(order="C")


def _encode_state_array(values: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(_array_bytes(values), level=9)).decode("ascii")


def _decode_state_array(text: str, *, size: int, name: str) -> np.ndarray:
    expected_bytes = size * np.dtype("<f8").itemsize
    if len(text) > 2 * expected_bytes + 1024:
        raise ArtifactValidationError(f"Fig. 6 state payload {name!r} is oversized.")
    try:
        compressed = base64.b64decode(text, validate=True)
        decompressor = zlib.decompressobj()
        payload = decompressor.decompress(compressed, expected_bytes + 1)
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise ArtifactValidationError(
            f"Fig. 6 state payload {name!r} is not valid compressed float64 data."
        ) from exc
    if (
        len(payload) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ArtifactValidationError(f"Fig. 6 state payload {name!r} has the wrong decoded size.")
    return np.frombuffer(payload, dtype="<f8", count=size).astype(float, copy=True)


def _state_sha256(
    *,
    T_bath: float,
    n_bar: float,
    gap: float,
    f: np.ndarray,
    n_ph: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "T_bath_K": T_bath,
                "delta_driven_ueV": gap,
                "n_bar": n_bar,
            }
        ).encode("ascii")
    )
    digest.update(b"\0f\0")
    digest.update(_array_bytes(f))
    digest.update(b"\0n_ph\0")
    digest.update(_array_bytes(n_ph))
    return digest.hexdigest()


def result_row_sha256(result: Fig6PaperResult, row_index: int) -> str:
    """Canonical semantic digest of one merged temperature row.

    Unlike the worker NPZ file hash, this digest is independent of ZIP
    metadata/compression.  The publisher recomputes it from the fully
    recertified merged result, binding every worker row's semantic payload to
    the canonical artifact.
    """
    if (
        isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or not 0 <= row_index < np.asarray(result.T_bath).size
    ):
        raise ValueError("row_index is outside the Fig. 6 result.")
    if result.state_f is None or result.state_n_ph is None:
        raise ArtifactValidationError(
            "Fig. 6 semantic row hashing requires persisted solver states."
        )
    arrays = {
        "T_bath": np.asarray(result.T_bath[row_index : row_index + 1]),
        "T_star_over_delta": np.asarray(
            result.T_star_over_delta[row_index : row_index + 1]
        ),
        "delta_driven": np.asarray(
            result.delta_driven[row_index : row_index + 1]
        ),
        "delta_eq": np.asarray(result.delta_eq[row_index : row_index + 1]),
        "delta_thermal_T_bath": np.asarray(
            result.delta_thermal_T_bath[row_index : row_index + 1]
        ),
        "gap_fixed_point_abs_error_uev": np.asarray(
            result.gap_fixed_point_abs_error_uev[row_index : row_index + 1]
        ),
        "n_bar": np.asarray(result.n_bar),
        "paper_observable_eq53": np.asarray(
            result.paper_observable_eq53[row_index : row_index + 1]
        ),
        "paper_observable_num": np.asarray(
            result.paper_observable_num[row_index : row_index + 1]
        ),
        "phonon_backward_error": np.asarray(
            result.phonon_backward_error[row_index : row_index + 1]
        ),
        "phonon_raw_backward_error": np.asarray(
            result.phonon_raw_backward_error[row_index : row_index + 1]
        ),
        "phonon_residual_inf": np.asarray(
            result.phonon_residual_inf[row_index : row_index + 1]
        ),
        "qp_backward_error": np.asarray(
            result.qp_backward_error[row_index : row_index + 1]
        ),
        "qp_number_backward_error": np.asarray(
            result.qp_number_backward_error[row_index : row_index + 1]
        ),
        "qp_residual_inf": np.asarray(
            result.qp_residual_inf[row_index : row_index + 1]
        ),
        "state_f": np.asarray(result.state_f[row_index : row_index + 1]),
        "state_n_ph": np.asarray(result.state_n_ph[row_index : row_index + 1]),
        "tau_0_pb_ns": np.asarray([result.tau_0_pb_ns]),
        "tau_l_ns": np.asarray([result.tau_l_ns]),
        "x_qp_eq47": np.asarray(result.x_qp_eq47[row_index : row_index + 1]),
        "x_qp_num": np.asarray(result.x_qp_num[row_index : row_index + 1]),
    }
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_fingerprint() -> dict[str, Any]:
    """Bind the artifact to current axes, config, solver, and source tree."""
    return {
        "axes": {
            "T_bath_K": [float(value) for value in T_BATH_VALUES],
            "n_bar": [float(value) for value in fig6_solve.N_BAR_VALUES],
        },
        "config": asdict(config_metadata()),
        "mode": {
            "direct_gap_observable": False,
            "fixed_gap_kinetics": False,
        },
        "solver": solver_fingerprint(),
        "source_sha256": source_manifest(
            Path(__file__),
            extra_validation_modules=(
                Path(fig6_solve.__file__),
                Path(fig5_paper.__file__),
                Path(fig5_solve.__file__),
                Path(certificate_module.__file__),
                Path(_pdf_validation.__file__),
                Path(sweep_cache.__file__),
            ),
        ),
    }


def _require_close(
    claimed: np.ndarray | float,
    recomputed: np.ndarray | float,
    *,
    name: str,
) -> None:
    if not np.allclose(
        np.asarray(claimed, dtype=float),
        np.asarray(recomputed, dtype=float),
        rtol=256.0 * np.finfo(float).eps,
        atol=np.finfo(float).tiny,
    ):
        raise ArtifactValidationError(
            f"Fig. 6 artifact {name} does not match its persisted solver state."
        )


def _required_state_array(
    result: Fig6PaperResult,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    upper_bound: float | None = None,
) -> np.ndarray:
    raw = getattr(result, name)
    if raw is None:
        raise ArtifactValidationError(
            f"Fig. 6 artifact is missing returned-state payload {name!r}."
        )
    raw_array = np.asarray(raw)
    if np.issubdtype(raw_array.dtype, np.bool_):
        raise ArtifactValidationError(f"Fig. 6 state payload {name!r} cannot be boolean.")
    values = np.asarray(raw_array, dtype=float)
    if values.shape != expected_shape or np.any(~np.isfinite(values)):
        raise ArtifactValidationError(
            f"Fig. 6 state payload {name!r} must have finite shape "
            f"{expected_shape}. Current-schema artifacts cannot encode an "
            "uncertified/collapsed point as a successful curve row."
        )
    if np.any(values < 0.0) or (upper_bound is not None and np.any(values > upper_bound)):
        raise ArtifactValidationError(
            f"Fig. 6 state payload {name!r} is outside its physical domain."
        )
    return values


def _rebuild_state(
    *,
    f: np.ndarray,
    n_ph: np.ndarray,
    T_bath: float,
    gap: float,
    base_spectral: SpectralContext,
) -> Any:
    point_spectral = SpectralContext(
        E_bins=base_spectral.E,
        dE_bins=base_spectral.dE,
        gap=gap,
    )
    initial = _build_state(
        _fischer_material(),
        base_spectral,
        T_bath,
        f_seed=f,
        n_ph_seed=n_ph,
    )
    return replace(
        initial,
        f=np.asarray(f, dtype=float).copy(),
        gap=gap,
        spectral=point_spectral,
        phonon=replace(
            initial.phonon,
            n_ph=np.asarray(n_ph, dtype=float).reshape(1, -1, 1).copy(),
        ),
    )


def _validate_result_for_artifact(
    result: Fig6PaperResult,
    *,
    expected_t_bath: np.ndarray | None = None,
    expected_n_bar: np.ndarray | None = None,
) -> dict[str, float]:
    """Independently derive every stored curve and certificate from raw state.

    Canonical artifact callers use the default full axes.  The guarded
    parallel regenerator supplies a one-temperature ``expected_t_bath`` while
    validating each atomic row payload; all curve and state recertification is
    otherwise identical to the final canonical validation.
    """
    expected_T = (
        np.asarray(T_BATH_VALUES, dtype=float)
        if expected_t_bath is None
        else np.asarray(expected_t_bath, dtype=float)
    )
    expected_n = (
        np.asarray(fig6_solve.N_BAR_VALUES, dtype=float)
        if expected_n_bar is None
        else np.asarray(expected_n_bar, dtype=float)
    )
    if (
        expected_T.ndim != 1
        or expected_T.size == 0
        or np.any(~np.isfinite(expected_T))
        or expected_n.ndim != 1
        or expected_n.size == 0
        or np.any(~np.isfinite(expected_n))
    ):
        raise ValueError("Expected Fig. 6 validation axes must be finite non-empty vectors.")
    if not np.array_equal(np.asarray(result.T_bath, dtype=float), expected_T):
        raise ArtifactValidationError("Fig. 6 T_bath axis is stale or reordered.")
    if not np.array_equal(np.asarray(result.n_bar, dtype=float), expected_n):
        raise ArtifactValidationError("Fig. 6 n_bar axis is stale or reordered.")
    shape = (expected_T.size, expected_n.size)
    current = config_metadata()
    if not np.isclose(
        result.tau_0_pb_ns,
        current.tau_0_pb_ns,
        rtol=1.0e-14,
        atol=0.0,
    ) or not np.isclose(
        result.tau_l_ns,
        current.tau_l_ns,
        rtol=1.0e-14,
        atol=0.0,
    ):
        raise ArtifactValidationError("Fig. 6 tau_0_pb/tau_l metadata is stale or non-physical.")

    vector_fields = (
        ("delta_eq", result.delta_eq, (expected_T.size,)),
        (
            "delta_thermal_T_bath",
            result.delta_thermal_T_bath,
            (expected_T.size,),
        ),
    )
    matrix_fields = (
        ("T_star_over_delta", result.T_star_over_delta),
        ("delta_driven", result.delta_driven),
        ("paper_observable_num", result.paper_observable_num),
        ("paper_observable_eq53", result.paper_observable_eq53),
        ("x_qp_num", result.x_qp_num),
        ("x_qp_eq47", result.x_qp_eq47),
        ("qp_residual_inf", result.qp_residual_inf),
        ("qp_backward_error", result.qp_backward_error),
        (
            certificate_module.QP_NUMBER_CERTIFICATE_FIELD,
            result.qp_number_backward_error,
        ),
        ("phonon_residual_inf", result.phonon_residual_inf),
        ("phonon_raw_backward_error", result.phonon_raw_backward_error),
        ("phonon_backward_error", result.phonon_backward_error),
        (
            fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD,
            result.gap_fixed_point_abs_error_uev,
        ),
    )
    for name, raw, expected_shape in vector_fields:
        array = np.asarray(raw, dtype=float)
        if array.shape != expected_shape or np.any(~np.isfinite(array)):
            raise ArtifactValidationError(
                f"Fig. 6 artifact {name} must have finite shape {expected_shape}."
            )
    for name, raw in matrix_fields:
        array = np.asarray(raw, dtype=float)
        if array.shape != shape or np.any(~np.isfinite(array)):
            raise ArtifactValidationError(f"Fig. 6 artifact {name} must have finite shape {shape}.")
    if np.any(np.asarray(result.delta_driven) <= 0.0):
        raise ArtifactValidationError("Fig. 6 driven gaps must be positive.")
    if np.any(np.asarray(result.x_qp_num) < 0.0) or np.any(np.asarray(result.x_qp_eq47) < 0.0):
        raise ArtifactValidationError("Fig. 6 quasiparticle fractions must be non-negative.")

    _, _, base_spectral = _build_grid_and_spectral()
    omega, _, _, _ = build_phonon_frequency_map(base_spectral.E)
    states_f = _required_state_array(
        result,
        "state_f",
        (*shape, base_spectral.E.size),
        upper_bound=1.0,
    )
    states_n_ph = _required_state_array(
        result,
        "state_n_ph",
        (*shape, omega.size),
    )

    claimed_certificates = {
        "qp_residual_inf": np.asarray(result.qp_residual_inf, dtype=float),
        "qp_backward_error": np.asarray(result.qp_backward_error, dtype=float),
        certificate_module.QP_NUMBER_CERTIFICATE_FIELD: np.asarray(
            result.qp_number_backward_error,
            dtype=float,
        ),
        "phonon_residual_inf": np.asarray(result.phonon_residual_inf, dtype=float),
        "phonon_raw_backward_error": np.asarray(result.phonon_raw_backward_error, dtype=float),
        "phonon_backward_error": np.asarray(result.phonon_backward_error, dtype=float),
        fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD: np.asarray(
            result.gap_fixed_point_abs_error_uev,
            dtype=float,
        ),
    }
    maxima = {field: float(np.max(values)) for field, values in claimed_certificates.items()}
    for field in _CERTIFIED_FIELDS:
        if maxima[field] > fig6_solve.TARGET_BACKWARD_ERROR_LIMIT:
            raise ArtifactValidationError(f"Fig. 6 certificate field {field!r} exceeds its gate.")
    if maxima[fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD] > GAP_FIXED_POINT_ABS_TOL_UEV:
        raise ArtifactValidationError("Fig. 6 gap fixed-point certificate exceeds its gate.")

    for i, T_bath in enumerate(expected_T):
        calibration = calibrate_gap(
            T_c=T_C,
            T_bath=float(T_bath),
            Delta_0=DELTA_0,
            xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
        )
        delta_eq = float(calibration.delta_eq)
        delta_T = DELTA_0 - delta_eq
        if delta_T <= 0.0:
            raise ArtifactValidationError(
                "Fig. 6 configured bath temperature has undefined suppression."
            )
        _require_close(result.delta_eq[i], delta_eq, name=f"delta_eq[{i}]")
        _require_close(
            result.delta_thermal_T_bath[i],
            delta_eq,
            name=f"delta_thermal_T_bath[{i}]",
        )
        for j, n_bar in enumerate(expected_n):
            gap = float(result.delta_driven[i, j])
            state = _rebuild_state(
                f=states_f[i, j],
                n_ph=states_n_ph[i, j],
                T_bath=float(T_bath),
                gap=gap,
                base_spectral=base_spectral,
            )
            tau_l = float(state.phonon.tau_l[0, 0])
            certificate = certificate_module.steady_state_certificate(
                state,
                photon_params={
                    "omega_0": OMEGA_0,
                    "n_bar": float(n_bar),
                    "c_phot": C_PHOT,
                },
                tau_l=tau_l,
            )
            mapped_gap = solve_gap(
                calibration,
                state.f,
                state.spectral.E,
                dE_bins=state.spectral.dE,
                reference_gap=gap,
                xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
            )
            certificate[fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD] = abs(mapped_gap - gap)
            T_star = fig6_solve._kBTstar_eq35(float(n_bar)) / DELTA_0
            x_num = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)
            x_eq47 = fig5_paper._xqp_analytic_eq47(
                float(T_bath),
                float(n_bar),
                tau_l=tau_l,
                tau_0_pb=result.tau_0_pb_ns,
            )
            obs_num = (gap - delta_eq) / delta_T
            delta_drive_analytic = DELTA_0 * fig6_solve._paper_eq53_analytic_drive(
                x_eq47,
                T_star,
            )
            obs_eq53 = (delta_T - delta_drive_analytic) / delta_T
            for name, claimed, recomputed in (
                ("T_star_over_delta", result.T_star_over_delta[i, j], T_star),
                ("x_qp_num", result.x_qp_num[i, j], x_num),
                ("x_qp_eq47", result.x_qp_eq47[i, j], x_eq47),
                (
                    "paper_observable_num",
                    result.paper_observable_num[i, j],
                    obs_num,
                ),
                (
                    "paper_observable_eq53",
                    result.paper_observable_eq53[i, j],
                    obs_eq53,
                ),
            ):
                _require_close(claimed, recomputed, name=f"{name}[{i},{j}]")
            for field in _ARTIFACT_CERTIFICATE_FIELDS:
                _require_close(
                    claimed_certificates[field][i, j],
                    certificate[field],
                    name=f"{field}[{i},{j}]",
                )
    return maxima


def validate_artifact_result(result: Fig6PaperResult) -> dict[str, float]:
    """Public full-axis raw-state validator used by production regenerators."""
    return _validate_result_for_artifact(result)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactValidationError(f"Fig. 6 metadata contains duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ArtifactValidationError(f"Could not read Fig. 6 artifact at {path}.") from exc
    if not rows or any(not row for row in rows):
        raise ArtifactValidationError(
            "Fig. 6 artifact must be non-empty and contain no blank rows."
        )
    return rows


def _artifact_metadata(path: Path, rows: list[list[str]]) -> dict[str, Any]:
    marker = [f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"]
    if rows[0] != marker:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = ""
        if digest == _LEGACY_CANONICAL_SHA256:
            raise LegacyArtifactError(
                f"The pinned Fig. 6 artifact at {path} predates the "
                f"state-certified {ARTIFACT_SCHEMA} schema."
            )
        raise ArtifactValidationError("Fig. 6 artifact has a missing or unsupported schema marker.")
    if len(rows) < 3 or len(rows[1]) != 1:
        raise ArtifactValidationError("Fig. 6 artifact metadata row is malformed.")
    prefix = "# qpsim_metadata="
    if not rows[1][0].startswith(prefix):
        raise ArtifactValidationError("Fig. 6 artifact metadata marker is missing.")
    try:
        metadata = json.loads(
            rows[1][0][len(prefix) :],
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactValidationError(
                    f"Fig. 6 metadata contains non-finite JSON token {value!r}."
                )
            ),
        )
    except ArtifactValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Fig. 6 artifact metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise ArtifactValidationError(
            "Fig. 6 artifact metadata fields do not match the exact schema."
        )
    if metadata["schema"] != ARTIFACT_SCHEMA:
        raise ArtifactValidationError("Fig. 6 metadata schema is stale.")
    if metadata["columns"] != list(_ARTIFACT_COLUMNS):
        raise ArtifactValidationError("Fig. 6 metadata columns are stale.")
    if metadata["certificate_fields"] != list(_ARTIFACT_CERTIFICATE_FIELDS):
        raise ArtifactValidationError("Fig. 6 certificate fields are stale.")
    if (
        metadata["certificate_metric_version"]
        != certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
    ):
        raise ArtifactValidationError("Fig. 6 certificate metric is stale.")
    target = metadata["certificate_target_backward_error"]
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not np.isfinite(target)
        or float(target) != fig6_solve.TARGET_BACKWARD_ERROR_LIMIT
    ):
        raise ArtifactValidationError("Fig. 6 certificate target is stale.")
    expected_rows = len(T_BATH_VALUES) * fig6_solve.N_BAR_VALUES.size
    if metadata["row_count"] != expected_rows or len(rows) != expected_rows + 3:
        raise ArtifactValidationError("Fig. 6 artifact row count is stale.")
    if rows[2] != list(_ARTIFACT_COLUMNS):
        raise ArtifactValidationError("Fig. 6 CSV header is stale or reordered.")
    if metadata["fingerprint"] != artifact_fingerprint():
        raise ArtifactValidationError(
            "Fig. 6 artifact fingerprint is stale (config, axes, solver, or source)."
        )
    payload_hash = metadata["payload_sha256"]
    if (
        not isinstance(payload_hash, str)
        or len(payload_hash) != 64
        or any(character not in "0123456789abcdef" for character in payload_hash)
        or payload_hash != _payload_sha256(rows[3:])
    ):
        raise ArtifactValidationError("Fig. 6 artifact payload hash does not match.")
    return metadata


def _finite_float(text: str, *, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ArtifactValidationError(f"Fig. 6 row field {name!r} is not numeric.") from exc
    if not np.isfinite(value):
        raise ArtifactValidationError(f"Fig. 6 row field {name!r} is non-finite.")
    return value


def _is_canonical_bundle_path(path: Path) -> bool:
    resolved = path.resolve()
    canonical = baseline_path()
    return resolved in {
        canonical.resolve(),
        plot_path().resolve(),
        canonical.with_suffix(".promotion.json").resolve(),
    }


def write_baseline(result: Fig6PaperResult, path: Path | None = None) -> Path:
    """Write raw states and claims only after independent recertification."""
    path = baseline_path() if path is None else path
    if _is_canonical_bundle_path(path):
        raise ArtifactValidationError(
            "Direct writes to the canonical Fig. 6 bundle are forbidden; "
            "use publish_artifacts() or generate_baseline()."
        )
    maxima = _validate_result_for_artifact(result)
    states_f = np.asarray(result.state_f, dtype=float)
    states_n_ph = np.asarray(result.state_n_ph, dtype=float)
    certificate_arrays = {
        "qp_residual_inf": np.asarray(result.qp_residual_inf, dtype=float),
        "qp_backward_error": np.asarray(result.qp_backward_error, dtype=float),
        certificate_module.QP_NUMBER_CERTIFICATE_FIELD: np.asarray(
            result.qp_number_backward_error, dtype=float
        ),
        "phonon_residual_inf": np.asarray(result.phonon_residual_inf, dtype=float),
        "phonon_raw_backward_error": np.asarray(result.phonon_raw_backward_error, dtype=float),
        "phonon_backward_error": np.asarray(result.phonon_backward_error, dtype=float),
        fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD: np.asarray(
            result.gap_fixed_point_abs_error_uev, dtype=float
        ),
    }
    rows: list[list[str]] = []
    for i, T_bath in enumerate(result.T_bath):
        for j, n_bar in enumerate(result.n_bar):
            f = states_f[i, j]
            n_ph = states_n_ph[i, j]
            gap = float(result.delta_driven[i, j])
            rows.append(
                [
                    f"{T_bath:.17e}",
                    f"{n_bar:.17e}",
                    f"{result.T_star_over_delta[i, j]:.17e}",
                    f"{result.delta_eq[i]:.17e}",
                    f"{gap:.17e}",
                    f"{result.x_qp_num[i, j]:.17e}",
                    f"{result.x_qp_eq47[i, j]:.17e}",
                    f"{result.paper_observable_num[i, j]:.17e}",
                    f"{result.paper_observable_eq53[i, j]:.17e}",
                    f"{certificate_arrays['qp_residual_inf'][i, j]:.17e}",
                    f"{certificate_arrays['qp_backward_error'][i, j]:.17e}",
                    f"{certificate_arrays[certificate_module.QP_NUMBER_CERTIFICATE_FIELD][i, j]:.17e}",
                    f"{certificate_arrays['phonon_residual_inf'][i, j]:.17e}",
                    f"{certificate_arrays['phonon_raw_backward_error'][i, j]:.17e}",
                    f"{certificate_arrays['phonon_backward_error'][i, j]:.17e}",
                    f"{certificate_arrays[fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD][i, j]:.17e}",
                    _encode_state_array(f),
                    _encode_state_array(n_ph),
                    _state_sha256(
                        T_bath=float(T_bath),
                        n_bar=float(n_bar),
                        gap=gap,
                        f=f,
                        n_ph=n_ph,
                    ),
                ]
            )
    metadata = {
        "certificate_fields": list(_ARTIFACT_CERTIFICATE_FIELDS),
        "certificate_maxima": maxima,
        "certificate_metric_version": (certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION),
        "certificate_target_backward_error": (fig6_solve.TARGET_BACKWARD_ERROR_LIMIT),
        "columns": list(_ARTIFACT_COLUMNS),
        "fingerprint": artifact_fingerprint(),
        "payload_sha256": _payload_sha256(rows),
        "row_count": len(rows),
        "schema": ARTIFACT_SCHEMA,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_text_file(path) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"])
        writer.writerow([f"# qpsim_metadata={_canonical_json(metadata)}"])
        writer.writerow(_ARTIFACT_COLUMNS)
        writer.writerows(rows)
    return path


def _read_artifact(path: Path) -> tuple[Fig6PaperResult, dict[str, Any]]:
    rows = _read_csv_rows(path)
    metadata = _artifact_metadata(path, rows)
    T_values = np.asarray(T_BATH_VALUES, dtype=float)
    n_values = np.asarray(fig6_solve.N_BAR_VALUES, dtype=float)
    shape = (T_values.size, n_values.size)
    _, _, spectral = _build_grid_and_spectral()
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    arrays = {
        name: np.empty(shape)
        for name in (
            "T_star_over_delta",
            "delta_driven",
            "x_qp_num",
            "x_qp_eq47",
            "paper_observable_num",
            "paper_observable_eq53",
            *_ARTIFACT_CERTIFICATE_FIELDS,
        )
    }
    delta_eq = np.empty(T_values.size)
    state_f = np.empty((*shape, spectral.E.size))
    state_n_ph = np.empty((*shape, omega.size))
    contexts = [
        (i, j, float(T_bath), float(n_bar))
        for i, T_bath in enumerate(T_values)
        for j, n_bar in enumerate(n_values)
    ]
    for row, (i, j, expected_T, expected_n) in zip(rows[3:], contexts, strict=True):
        if len(row) != len(_ARTIFACT_COLUMNS):
            raise ArtifactValidationError("Fig. 6 data row width is invalid.")
        values = [
            _finite_float(row[offset], name=_ARTIFACT_COLUMNS[offset]) for offset in range(16)
        ]
        if values[0] != expected_T or values[1] != expected_n:
            raise ArtifactValidationError("Fig. 6 data row axes are stale or reordered.")
        (
            _,
            _,
            arrays["T_star_over_delta"][i, j],
            delta_eq[i],
            arrays["delta_driven"][i, j],
            arrays["x_qp_num"][i, j],
            arrays["x_qp_eq47"][i, j],
            arrays["paper_observable_num"][i, j],
            arrays["paper_observable_eq53"][i, j],
            arrays["qp_residual_inf"][i, j],
            arrays["qp_backward_error"][i, j],
            arrays[certificate_module.QP_NUMBER_CERTIFICATE_FIELD][i, j],
            arrays["phonon_residual_inf"][i, j],
            arrays["phonon_raw_backward_error"][i, j],
            arrays["phonon_backward_error"][i, j],
            arrays[fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD][i, j],
        ) = values
        f = _decode_state_array(row[16], size=spectral.E.size, name=f"f[{i},{j}]")
        n_ph = _decode_state_array(row[17], size=omega.size, name=f"n_ph[{i},{j}]")
        expected_hash = _state_sha256(
            T_bath=expected_T,
            n_bar=expected_n,
            gap=float(arrays["delta_driven"][i, j]),
            f=f,
            n_ph=n_ph,
        )
        if row[18] != expected_hash:
            raise ArtifactValidationError("Fig. 6 per-point state hash does not match.")
        state_f[i, j] = f
        state_n_ph[i, j] = n_ph
    config = metadata["fingerprint"]["config"]
    result = Fig6PaperResult(
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
        tau_l_ns=float(config["tau_l_ns"]),
        T_bath=T_values,
        n_bar=n_values,
        T_star_over_delta=arrays["T_star_over_delta"],
        delta_eq=delta_eq,
        delta_driven=arrays["delta_driven"],
        delta_thermal_T_bath=delta_eq.copy(),
        paper_observable_num=arrays["paper_observable_num"],
        paper_observable_eq53=arrays["paper_observable_eq53"],
        x_qp_num=arrays["x_qp_num"],
        x_qp_eq47=arrays["x_qp_eq47"],
        qp_residual_inf=arrays["qp_residual_inf"],
        qp_backward_error=arrays["qp_backward_error"],
        qp_number_backward_error=arrays[certificate_module.QP_NUMBER_CERTIFICATE_FIELD],
        phonon_residual_inf=arrays["phonon_residual_inf"],
        phonon_raw_backward_error=arrays["phonon_raw_backward_error"],
        phonon_backward_error=arrays["phonon_backward_error"],
        gap_fixed_point_abs_error_uev=arrays[fig6_solve.GAP_FIXED_POINT_CERTIFICATE_FIELD],
        state_f=state_f,
        state_n_ph=state_n_ph,
    )
    recomputed_maxima = _validate_result_for_artifact(result)
    stamped_maxima = metadata["certificate_maxima"]
    if not isinstance(stamped_maxima, dict) or set(stamped_maxima) != set(
        _ARTIFACT_CERTIFICATE_FIELDS
    ):
        raise ArtifactValidationError("Fig. 6 certificate maxima are invalid.")
    for field, recomputed in recomputed_maxima.items():
        stamped = stamped_maxima[field]
        if (
            isinstance(stamped, bool)
            or not isinstance(stamped, (int, float))
            or not np.isfinite(stamped)
            or float(stamped) != recomputed
        ):
            raise ArtifactValidationError(
                f"Fig. 6 stamped certificate maximum {field!r} is forged."
            )
    return result, metadata


def read_baseline(path: Path | None = None) -> Fig6PaperResult:
    """Read and independently re-certify an exact current-schema artifact."""
    resolved = baseline_path() if path is None else path
    if resolved.resolve() == baseline_path().resolve():
        with _publication_lock():
            result = _read_artifact(resolved)[0]
            _read_promotion_record_unlocked()
            return result
    return _read_artifact(resolved)[0]


PROMOTION_RECORD_SCHEMA = "qpsim.fischer2023.fig6_promotion.v2"
GENERATION_EVIDENCE_SCHEMA = "qpsim.fischer2023.fig6_generation.v1"
_THREAD_ENVIRONMENT_NAMES = (
    "BLIS_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SINGLE_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def generation_run_identity(
    *,
    mode: str,
    runner: Mapping[str, object],
    runtime: Mapping[str, object],
    single_thread_environment: Mapping[str, object],
) -> str:
    """Content address one Fig. 6 producer before expensive row work."""
    payload = {
        "artifact_fingerprint": artifact_fingerprint(),
        "generation_schema": GENERATION_EVIDENCE_SCHEMA,
        "mode": mode,
        "runner": dict(runner),
        "runtime": dict(runtime),
        "single_thread_environment": dict(single_thread_environment),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def generation_runtime_identity() -> dict[str, object]:
    """Portable numerical-runtime identity for serial publication evidence."""
    import matplotlib
    import scipy

    return {
        "machine": platform.machine().strip().lower() or "unknown",
        "matplotlib": matplotlib.__version__,
        "numeric_runtime": sweep_cache._numeric_runtime_identity(),
        "numpy": np.__version__,
        "os_family": sys.platform,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation().lower(),
        "scipy": scipy.__version__,
    }


def serial_generation_evidence() -> dict[str, object]:
    """Describe the in-process fallback producer used by the legacy CLI."""
    root = Path(__file__).resolve().parents[2]
    runner_path = Path(__file__).resolve()
    runner = {
        "path": runner_path.relative_to(root).as_posix(),
        "sha256": source_sha256(runner_path),
    }
    runtime = generation_runtime_identity()
    environment = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES
    }
    return {
        "campaign": {
            "aggregate_worker_s": 0.0,
            "new_rows": 0,
            "resumed_rows": 0,
            "wall_s": 0.0,
        },
        "mode": "serial-in-process",
        "run_identity": generation_run_identity(
            mode="serial-in-process",
            runner=runner,
            runtime=runtime,
            single_thread_environment=environment,
        ),
        "runner": runner,
        "runtime": runtime,
        "schema": GENERATION_EVIDENCE_SCHEMA,
        "single_thread_environment": environment,
        "worker_payloads": {},
    }


def validate_generation_evidence(
    evidence: Mapping[str, object],
    *,
    result: Fig6PaperResult | None = None,
    require_current_runtime: bool = False,
) -> dict[str, object]:
    """Freeze and fail-closed validate campaign/runner evidence."""
    try:
        frozen = json.loads(
            json.dumps(
                dict(evidence),
                allow_nan=False,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            "Fig. 6 generation evidence is not finite JSON data."
        ) from exc
    if not isinstance(frozen, dict) or set(frozen) != {
        "campaign",
        "mode",
        "run_identity",
        "runner",
        "runtime",
        "schema",
        "single_thread_environment",
        "worker_payloads",
    }:
        raise ArtifactValidationError("Fig. 6 generation evidence shape is invalid.")
    if frozen["schema"] != GENERATION_EVIDENCE_SCHEMA:
        raise ArtifactValidationError("Fig. 6 generation evidence schema is stale.")
    mode = frozen["mode"]
    if mode not in {"parallel-temperature-rows", "serial-in-process"}:
        raise ArtifactValidationError("Fig. 6 generation mode is invalid.")

    root = Path(__file__).resolve().parents[2]
    runner = frozen["runner"]
    if not isinstance(runner, dict) or set(runner) != {"path", "sha256"}:
        raise ArtifactValidationError("Fig. 6 runner evidence is invalid.")
    runner_relative = runner["path"]
    if not isinstance(runner_relative, str):
        raise ArtifactValidationError("Fig. 6 runner path is invalid.")
    runner_path = (root / Path(runner_relative)).resolve()
    try:
        runner_path.relative_to(root)
    except ValueError as exc:
        raise ArtifactValidationError("Fig. 6 runner path escapes the repository.") from exc
    if not runner_path.is_file() or not _is_sha256(runner["sha256"]):
        raise ArtifactValidationError("Fig. 6 runner source identity is invalid.")
    if source_sha256(runner_path) != runner["sha256"]:
        raise ArtifactValidationError("Fig. 6 runner source has changed.")

    runtime = frozen["runtime"]
    environment = frozen["single_thread_environment"]
    if not isinstance(runtime, dict) or not runtime:
        raise ArtifactValidationError("Fig. 6 runtime evidence is invalid.")
    if (
        not isinstance(environment, dict)
        or tuple(sorted(environment)) != tuple(sorted(_THREAD_ENVIRONMENT_NAMES))
        or any(
            value is not None and not isinstance(value, str)
            for value in environment.values()
        )
    ):
        raise ArtifactValidationError(
            "Fig. 6 thread-environment evidence is invalid."
        )
    if require_current_runtime:
        current_runtime = generation_runtime_identity()
        if runtime != current_runtime:
            raise ArtifactValidationError(
                "Fig. 6 numerical runtime changed after producer capture."
            )
        current_environment = {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES
        }
        if environment != current_environment:
            raise ArtifactValidationError(
                "Fig. 6 numerical thread environment changed after producer capture."
            )
    if mode == "parallel-temperature-rows" and environment != _SINGLE_THREAD_ENVIRONMENT:
        raise ArtifactValidationError(
            "Fig. 6 parallel workers were not constrained to one numerical thread."
        )
    expected_identity = generation_run_identity(
        mode=str(mode),
        runner=runner,
        runtime=runtime,
        single_thread_environment=environment,
    )
    if frozen["run_identity"] != expected_identity:
        raise ArtifactValidationError("Fig. 6 generation run identity is forged or stale.")

    campaign = frozen["campaign"]
    if not isinstance(campaign, dict) or set(campaign) != {
        "aggregate_worker_s",
        "new_rows",
        "resumed_rows",
        "wall_s",
    }:
        raise ArtifactValidationError("Fig. 6 campaign summary is invalid.")
    for field in ("aggregate_worker_s", "wall_s"):
        value = campaign[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ArtifactValidationError(
                f"Fig. 6 campaign field {field!r} is invalid."
            )
    for field in ("new_rows", "resumed_rows"):
        value = campaign[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactValidationError(
                f"Fig. 6 campaign field {field!r} is invalid."
            )

    payloads = frozen["worker_payloads"]
    if not isinstance(payloads, dict):
        raise ArtifactValidationError("Fig. 6 worker-payload evidence is invalid.")
    if mode == "parallel-temperature-rows":
        expected_keys = {
            f"t{index:02d}" for index in range(len(T_BATH_VALUES))
        }
        if set(payloads) != expected_keys:
            raise ArtifactValidationError(
                "Fig. 6 worker evidence is missing or has extra temperature rows."
            )
        if campaign["new_rows"] + campaign["resumed_rows"] != len(T_BATH_VALUES):
            raise ArtifactValidationError("Fig. 6 campaign row counts are inconsistent.")
        for index, T_bath in enumerate(T_BATH_VALUES):
            payload = payloads[f"t{index:02d}"]
            if not isinstance(payload, dict) or set(payload) != {
                "semantic_sha256",
                "temperature_K",
            }:
                raise ArtifactValidationError("Fig. 6 worker-payload record is invalid.")
            if (
                payload["temperature_K"] != float(T_bath)
                or not _is_sha256(payload["semantic_sha256"])
            ):
                raise ArtifactValidationError(
                    "Fig. 6 worker-payload identity is invalid."
                )
            if (
                result is not None
                and payload["semantic_sha256"] != result_row_sha256(result, index)
            ):
                raise ArtifactValidationError(
                    "Fig. 6 worker semantic row digest does not match the "
                    "recertified merged artifact."
                )
    elif payloads or campaign["new_rows"] or campaign["resumed_rows"]:
        raise ArtifactValidationError(
            "Serial Fig. 6 evidence cannot claim parallel worker rows."
        )
    return frozen


def promotion_record_path() -> Path:
    return baseline_path().with_suffix(".promotion.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_valid_pdf(path: Path) -> None:
    """Require one structurally valid, nonempty Matplotlib PDF page."""
    try:
        payload = path.read_bytes()
        _pdf_validation.validate_single_nonempty_matplotlib_pdf(
            payload,
            path=path,
        )
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            "Fig. 6 plot is not a complete nonempty Matplotlib PDF."
        ) from exc


@contextmanager
def _publication_lock() -> Iterator[None]:
    """Serialize canonical publishers with a process-scoped OS file lock."""
    lock_path = promotion_record_path().with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
    except OSError as exc:
        stream.close()
        raise RuntimeError(
            f"Another Fig. 6 publisher holds {lock_path}; refusing "
            "concurrent canonical publication."
        ) from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_UN,  # type: ignore[attr-defined]
            )
        stream.close()


def _read_promotion_record_unlocked(
    path: Path | None = None,
    *,
    csv_path: Path | None = None,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the commit marker binding one exact CSV/PDF pair."""
    record_path = promotion_record_path() if path is None else path
    csv_resolved = baseline_path() if csv_path is None else csv_path
    pdf_resolved = plot_path() if pdf_path is None else pdf_path
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            "Fig. 6 canonical publication record is missing or invalid."
        ) from exc
    if not isinstance(record, dict) or set(record) != {
        "artifact_schema",
        "artifacts",
        "generation",
        "schema",
    }:
        raise ArtifactValidationError("Fig. 6 publication record shape is invalid.")
    if (
        record["schema"] != PROMOTION_RECORD_SCHEMA
        or record["artifact_schema"] != ARTIFACT_SCHEMA
    ):
        raise ArtifactValidationError("Fig. 6 publication record schema is stale.")
    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"csv", "pdf"}:
        raise ArtifactValidationError("Fig. 6 publication identities are invalid.")
    for name, artifact_path in (("csv", csv_resolved), ("pdf", pdf_resolved)):
        if artifacts[name] != _file_identity(artifact_path):
            raise ArtifactValidationError(
                f"Fig. 6 canonical {name.upper()} does not match its "
                "publication record."
            )
    _require_valid_pdf(pdf_resolved)
    validated_result, _ = _read_artifact(csv_resolved)
    validate_generation_evidence(
        record["generation"],
        result=validated_result,
    )
    return record


def read_promotion_record(
    path: Path | None = None,
    *,
    csv_path: Path | None = None,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically validate the canonical publication tuple.

    Noncanonical staged paths are an internal publisher concern and use the
    unlocked helper while already holding :func:`_publication_lock`.
    """
    with _publication_lock():
        return _read_promotion_record_unlocked(
            path,
            csv_path=csv_path,
            pdf_path=pdf_path,
        )


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config)
    without touching the data rows or running the sweep.

    Comparing the live config's fingerprint against the pinned baseline's is
    the **cheap preflight** that lets the slow regression test reject a stale
    config/baseline pairing in seconds instead of after the ~6.04 h sweep (the
    failure mode that once burned 9.5 h: baseline pinned at
    ``TAU_L_MODEL='acoustic_escape'`` / 368 ps while the script default is
    ``'tau_0_pb'`` / 255 ps).
    """

    delta_0: float
    finite_cutoff_delta0_over_kbtc: float
    tau_0: float
    t_c: float
    omega_0: float
    c_phot: float
    film_thickness_nm: float
    eta: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    tau_0_pb_ns: float
    tau_l_ns: float
    tau_l_model: str
    gap_fixed_point_abs_tol_uev: float
    certificate_metric_version: str


def _legacy_read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve), so it is cheap
    enough for a preflight. Raises ``RuntimeError`` if any field the writer
    stamps is missing — a malformed/old header should fail loudly, not
    silently skip the check.
    """
    if path is None:
        path = baseline_path()
    text = path.read_text(encoding="utf-8")

    def _num(rx: re.Pattern[str], field: str) -> float:
        m = rx.search(text)
        if m is None:
            raise RuntimeError(f"Baseline header at {path} missing {field} metadata.")
        return float(m.group(1))

    def _optional_num(rx: re.Pattern[str]) -> float:
        """Read post-legacy metadata while retaining old-header readability."""
        match = rx.search(text)
        return float("nan") if match is None else float(match.group(1))

    ne_m = _GRID_NE_RE.search(text)
    model_m = _TAU_L_MODEL_RE.search(text)
    if ne_m is None or model_m is None:
        raise RuntimeError(f"Baseline header at {path} missing NE / TAU_L_MODEL metadata.")
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        finite_cutoff_delta0_over_kbtc=_optional_num(_FINITE_CUTOFF_RATIO_RE),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_c=_num(_HEADER_PARAM_RE["t_c"], "T_c"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        film_thickness_nm=_num(_HEADER_PARAM_RE["film_thickness_nm"], "film_thickness_nm"),
        eta=_num(_HEADER_PARAM_RE["eta"], "eta"),
        num_bins=int(ne_m.group(1)),
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        tau_0_pb_ns=_num(_TAU_0_PB_RE, "tau_0_pb_ns"),
        tau_l_ns=_num(_TAU_L_RE, "tau_l_ns"),
        tau_l_model=model_m.group(1),
        gap_fixed_point_abs_tol_uev=_optional_num(_GAP_FIXED_POINT_ABS_TOL_RE),
        certificate_metric_version=(
            ""
            if (metric_match := _CERTIFICATE_METRIC_VERSION_RE.search(text)) is None
            else metric_match.group(1)
        ),
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — computed without the (~6.04 h serial) sweep.

    ``tau_0_pb_ns`` and ``tau_l_ns`` are produced by the exact same calls
    :func:`run` makes (:func:`_compute_tau_0_pb` and the ``τ_ℓ`` of a freshly
    built state), so this can never drift from what a real run would write;
    everything else is read straight off the module constants.
    """
    material = _fischer_material()
    _, _, spectral = _build_grid_and_spectral()
    tau_0_pb = _compute_tau_0_pb(spectral)
    tau_l = float(_build_state(material, spectral, T_BATH_VALUES[0]).phonon.tau_l[0, 0])
    return BaselineMetadata(
        delta_0=DELTA_0,
        finite_cutoff_delta0_over_kbtc=FINITE_CUTOFF_DELTA0_OVER_KBTC,
        tau_0=TAU_0,
        t_c=T_C,
        omega_0=OMEGA_0,
        c_phot=C_PHOT,
        film_thickness_nm=FILM_THICKNESS_NM,
        eta=SUBSTRATE_ETA,
        num_bins=fig6_solve.NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        tau_0_pb_ns=tau_0_pb,
        tau_l_ns=tau_l,
        tau_l_model=TAU_L_MODEL,
        gap_fixed_point_abs_tol_uev=GAP_FIXED_POINT_ABS_TOL_UEV,
        certificate_metric_version=(certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION),
    )


def read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Return metadata only after the complete state artifact is recertified."""
    resolved = baseline_path() if path is None else path
    if resolved.resolve() == baseline_path().resolve():
        with _publication_lock():
            _, metadata = _read_artifact(resolved)
            _read_promotion_record_unlocked()
    else:
        _, metadata = _read_artifact(resolved)
    config = metadata["fingerprint"]["config"]
    return BaselineMetadata(
        delta_0=float(config["delta_0"]),
        finite_cutoff_delta0_over_kbtc=float(config["finite_cutoff_delta0_over_kbtc"]),
        tau_0=float(config["tau_0"]),
        t_c=float(config["t_c"]),
        omega_0=float(config["omega_0"]),
        c_phot=float(config["c_phot"]),
        film_thickness_nm=float(config["film_thickness_nm"]),
        eta=float(config["eta"]),
        num_bins=int(config["num_bins"]),
        e_min_factor=float(config["e_min_factor"]),
        e_max_factor=float(config["e_max_factor"]),
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
        tau_l_ns=float(config["tau_l_ns"]),
        tau_l_model=str(config["tau_l_model"]),
        gap_fixed_point_abs_tol_uev=float(config["gap_fixed_point_abs_tol_uev"]),
        certificate_metric_version=str(config["certificate_metric_version"]),
    )


def _direct_plot_limits(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand the paper window enough to show every finite direct-mode point."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return (0.20, 0.65), (0.0, 0.25)

    x_lo = min(0.20, float(np.min(x[finite])))
    x_hi = max(0.65, float(np.max(x[finite])))
    y_lo = min(0.0, float(np.min(y[finite])))
    y_hi = max(0.25, float(np.max(y[finite])))
    x_pad = 0.03 * max(x_hi - x_lo, np.finfo(float).eps)
    y_pad = 0.03 * max(y_hi - y_lo, np.finfo(float).eps)
    x_limits = (
        x_lo - x_pad if x_lo < 0.20 else 0.20,
        x_hi + x_pad if x_hi > 0.65 else 0.65,
    )
    y_limits = (
        y_lo - y_pad if y_lo < 0.0 else 0.0,
        y_hi + y_pad if y_hi > 0.25 else 0.25,
    )
    return x_limits, y_limits


def write_plot(
    result: Fig6PaperResult,
    path: Path | None = None,
    *,
    direct_gap_observable: bool = False,
) -> Path:
    """Paper-style plot: paper observable vs $T_*/\\Delta$, three $T_B$ curves.

    Colors match Fischer 2023 Fig. 6:
        T_B = 0.10 K → green
        T_B = 0.15 K → blue
        T_B = 0.20 K → red
    Solid: numerics. Dashed: Eq. 53 (with caveats per docstring).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = (
        plot_path(direct_gap_observable=direct_gap_observable)
        if path is None
        else path
    )
    if _is_canonical_bundle_path(path):
        raise ArtifactValidationError(
            "Direct writes to the canonical Fig. 6 bundle are forbidden; "
            "use publish_artifacts() or generate_baseline()."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paper Fig. 6 palette — matches standalone reproduction figures/fig6.py:
    # T_B = 0.10/0.15/0.20 K → C2 (green) / C0 (blue) / C3 (red).
    PAPER_COLORS = {0.10: "C2", 0.15: "C0", 0.20: "C3"}
    fallback_cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:  # pragma: no cover
        PchipInterpolator = None

    for i, T_bath in enumerate(result.T_bath):
        color = PAPER_COLORS.get(
            float(round(float(T_bath), 4)),
            fallback_cmap(i / max(1, len(result.T_bath) - 1)),
        )
        x = result.T_star_over_delta[i]
        y_num = result.paper_observable_num[i]
        finite = np.isfinite(x) & np.isfinite(y_num)
        xs = x[finite]
        ys = y_num[finite]
        if PchipInterpolator is not None and xs.size >= 4:
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            xd = np.linspace(xs[0], xs[-1], 500)
            yd = PchipInterpolator(xs, ys)(xd)
            ax.plot(xd, yd, color=color, lw=1.8, label=rf"$T_B = {T_bath:g}$ K")
        else:
            ax.plot(xs, ys, color=color, lw=1.8, label=rf"$T_B = {T_bath:g}$ K")
        if direct_gap_observable:
            ax.plot(xs, ys, linestyle="none", marker="o", ms=2.5, color=color)

        # Plot the exact Eq. 53 array stored in and independently re-derived
        # from the canonical artifact.  The former plot-time ratio-2 curve was
        # a separate diagnostic while the configured/certified solve uses
        # tau_l/tau_0^PB = 1; displaying it as the canonical dashed curve made
        # the PDF visually disagree with the CSV and its tests.
        y_analytic = result.paper_observable_eq53[i]
        finite_analytic = np.isfinite(x) & np.isfinite(y_analytic)
        xa = x[finite_analytic]
        ya = y_analytic[finite_analytic]
        if PchipInterpolator is not None and xa.size >= 4:
            order = np.argsort(xa)
            xa, ya = xa[order], ya[order]
            xad = np.linspace(xa[0], xa[-1], 500)
            yad = PchipInterpolator(xa, ya)(xad)
            ax.plot(
                xad,
                yad,
                color=color,
                ls=(0, (5, 2)),
                lw=1.6,
                alpha=0.95,
                zorder=4,
                label=(r"stored Eq. 53 ($\tau_\ell=\tau_0^{PB}$)" if i == 0 else None),
            )
        else:
            ax.plot(
                xa,
                ya,
                color=color,
                ls=(0, (5, 2)),
                lw=1.6,
                alpha=0.95,
                zorder=4,
                label=(r"stored Eq. 53 ($\tau_\ell=\tau_0^{PB}$)" if i == 0 else None),
            )

    ax.axhline(0.0, color="k", lw=0.4)
    if direct_gap_observable:
        x_limits, y_limits = _direct_plot_limits(
            result.T_star_over_delta,
            result.paper_observable_num,
        )
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    else:
        # Preserve the published paper window for the canonical default plot.
        ax.set_xlim(0.20, 0.65)
        ax.set_ylim(0.00, 0.25)
    ax.set_xlabel(r"$T_*/\Delta$")
    ax.set_ylabel(r"$(\delta\Delta_T - \delta\Delta)/\delta\Delta_T$")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _restore_payload(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.rollback.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_artifacts(
    result: Fig6PaperResult,
    *,
    generation_evidence: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    """Stage, verify, and publish PDF -> CSV -> commit marker under a lock.

    The commit marker authenticates not only the artifact pair but also the
    exact producer/runtime.  Parallel callers must supply their row campaign
    evidence.  Callers must capture serial evidence before solving; silently
    inventing it here after receiving a result could stamp an old in-memory
    solve with post-edit source identity.
    """
    with _publication_lock():
        csv_path = baseline_path()
        pdf_path = plot_path()
        record_path = promotion_record_path()
        stages = {
            "csv": csv_path.with_name(f".{csv_path.stem}.{os.getpid()}.stage.csv"),
            "pdf": pdf_path.with_name(f".{pdf_path.stem}.{os.getpid()}.stage.pdf"),
            "record": record_path.with_name(
                f".{record_path.stem}.{os.getpid()}.stage.json"
            ),
        }
        for stage in stages.values():
            stage.unlink(missing_ok=True)
        source_before = artifact_fingerprint()
        generation = validate_generation_evidence(
            generation_evidence,
            result=result,
            require_current_runtime=True,
        )
        try:
            write_plot(result, stages["pdf"])
            _require_valid_pdf(stages["pdf"])
            pdf_identity = _file_identity(stages["pdf"])
            write_baseline(result, stages["csv"])
            if _file_identity(stages["pdf"]) != pdf_identity:
                raise ArtifactValidationError(
                    "Fig. 6 staged PDF changed while the CSV was written."
                )
            _read_artifact(stages["csv"])
            if artifact_fingerprint() != source_before:
                raise ArtifactValidationError(
                    "Fig. 6 source/configuration changed during publication."
                )
            record = {
                "artifact_schema": ARTIFACT_SCHEMA,
                "artifacts": {
                    "csv": _file_identity(stages["csv"]),
                    "pdf": pdf_identity,
                },
                "generation": generation,
                "schema": PROMOTION_RECORD_SCHEMA,
            }
            with _atomic_text_file(stages["record"]) as stream:
                stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
            _read_promotion_record_unlocked(
                stages["record"],
                csv_path=stages["csv"],
                pdf_path=stages["pdf"],
            )
            if artifact_fingerprint() != source_before:
                raise ArtifactValidationError(
                    "Fig. 6 source/configuration changed before promotion."
                )
            old_payloads = {
                path: path.read_bytes() if path.is_file() else None
                for path in (pdf_path, csv_path, record_path)
            }
            try:
                os.replace(stages["pdf"], pdf_path)
                os.replace(stages["csv"], csv_path)
                os.replace(stages["record"], record_path)
                _read_artifact(csv_path)
                _read_promotion_record_unlocked()
            except BaseException:
                for artifact_path, payload in old_payloads.items():
                    _restore_payload(artifact_path, payload)
                raise
            return csv_path, pdf_path, record_path
        finally:
            for stage in stages.values():
                stage.unlink(missing_ok=True)


def generate_baseline(
    *,
    direct_gap_observable: bool = False,
    fixed_gap_kinetics: bool = False,
) -> tuple[Path, Path]:
    generation_started = time.perf_counter()
    # Freeze producer/source/runtime BEFORE the expensive serial solve.  A
    # post-solve validation below refuses to stamp an old in-memory result
    # with source or runtime identities captured after a mid-run edit.
    generation = serial_generation_evidence()
    print("Fischer 2023 Fig. 6 paper-topology qpsim regression ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, "
        f"omega_0={OMEGA_0:.2f} micro-eV, c_phot={C_PHOT:.0e} ns^-1"
    )
    print(f"  Acoustic-escape geometry: d={FILM_THICKNESS_NM:.0f} nm, eta={SUBSTRATE_ETA:.2f}")
    print(
        f"  Grid: NE={fig6_solve.NUM_BINS}, "
        f"dE={(E_MAX_FACTOR - E_MIN_FACTOR) * DELTA_0 / fig6_solve.NUM_BINS:.3f} micro-eV"
    )
    print(f"  T_B values: {list(T_BATH_VALUES)} K")
    print(
        f"  nbar values: {fig6_solve.N_BAR_VALUES.size} pts in "
        f"[{fig6_solve.N_BAR_VALUES[0]:.0e}, {fig6_solve.N_BAR_VALUES[-1]:.0e}]"
    )
    result = run_cached(
        direct_gap_observable=direct_gap_observable,
        fixed_gap_kinetics=fixed_gap_kinetics,
    )
    generation = json.loads(json.dumps(generation))
    campaign = generation["campaign"]
    if not isinstance(campaign, dict):  # serial_generation_evidence invariant
        raise AssertionError("Serial Fig. 6 generation evidence lost its campaign.")
    campaign["wall_s"] = time.perf_counter() - generation_started
    validate_generation_evidence(generation, require_current_runtime=True)
    if direct_gap_observable:
        # The authenticated canonical publication is intentionally the
        # self-consistent default. Direct-gap output remains a separately
        # named diagnostic and cannot overwrite that evidence pair.
        csv_path = _legacy_write_baseline(
            result,
            baseline_path(direct_gap_observable=True),
        )
        pdf_path = write_plot(
            result,
            plot_path(direct_gap_observable=True),
            direct_gap_observable=True,
        )
    else:
        csv_path, pdf_path, _ = publish_artifacts(
            result,
            generation_evidence=generation,
        )
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fischer 2023 Fig. 6 paper-topology qpsim regression "
            "(no digitized-paper curve comparison). "
            "Default settings use a 1640-bin paper-parameter grid "
            "(dE=1 micro-eV plus sub-gap guard cells), 22 nbar pts, "
            "picard_tol=1e-12) and historically took ~6.04 h serial. "
            "Pass --fast for a dev-speed "
            "knob (~30 min/run)."
        )
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Dev mode: 410-bin grid, 8 nbar pts, picard_tol=1e-9. Output "
            "paths gain a '_fast' suffix so the canonical qpsim regression "
            "is not clobbered. Use during iteration; switch back to the "
            "default for the final ship run."
        ),
    )
    parser.add_argument(
        "--direct-gap",
        action="store_true",
        help=(
            "Use the author-style fixed-Delta kinetic solve and direct "
            "Delta[f] gap observable. Output paths gain a '_direct' suffix."
        ),
    )
    args = parser.parse_args()

    if args.direct_gap:
        print(
            "--direct-gap mode: fixed-gap kinetics, direct Delta[f] observable, "
            "output suffix '_direct'."
        )

    if args.fast:
        # Mutate fig6_solve's solve-affecting globals (read by solve() and
        # solver_fingerprint(), so the cache key reflects --fast) before
        # generate_baseline() runs. NUM_BINS=410 keeps OMEGA_0/dE = 5
        # commensurate (dE = 4 micro-eV) while retaining the same 20-micro-eV
        # sub-gap guard as the full-resolution grid. Tighter tolerances stay
        # canonical paper-parameter regression —
        # the gap-precision fix for the low-T_B observable is not the bottleneck.
        fig6_solve.NUM_BINS = 410
        fig6_solve.N_BAR_VALUES = np.logspace(4.0, 8.2, 8)
        fig6_solve.PICARD_TOL = 1e-9
        _FAST_SUFFIX = "_fast"
        print("--fast mode: 410-bin grid, 8 nbar pts, picard_tol=1e-9, output suffix '_fast'.")

    generate_baseline(
        direct_gap_observable=args.direct_gap,
        fixed_gap_kinetics=args.direct_gap,
    )
