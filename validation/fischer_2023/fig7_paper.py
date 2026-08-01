"""Fischer 2023 Fig. 7 paper-topology qpsim Q_i(T_B) regression.

Uses parameter values transcribed from Tables II/III:

* Delta_0 = 189 micro-eV, tau_0 = 63 ns, omega_0 = 22 micro-eV
* c_phot = 0.06 Hz, tau_l = 170 ps, alpha = 0.13
* h = Delta_T / 189, E_max = 10 Delta_T
* nbar set by the Table III Tstar,0/Delta values
* plotted quality factor capped by Eq. (65) with Table III Q_i,ext

The quasiparticle kinetic solve is done by the qpsim T3 backend with the
finite-tau_l phonon field. The intrinsic quasiparticle Q_i is evaluated
with the leading Fischer Eq. (57) form used by the standalone
transcribed-formula implementation:

    Q_i,qp = pi Delta / (omega_0 alpha sigma_1).

This validates qpsim's internal paper-topology calculation and pinned
formula transcription. It does not compare the simulated curves with
digitized data from the published Fig. 7 and therefore is not evidence of
quantitative agreement with the plotted experimental data.

Usage:

    python -m validation.fischer_2023.fig7_paper
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO, cast

import numpy as np
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.ac_conductivity import compute_ac_conductivity

from validation import source_provenance, sweep_cache
from validation.fischer_2023 import fig7_solve
from validation.fischer_2023 import steady_state_certificate as certificate_module
from validation.fischer_2023.fig7_solve import (
    C_PHOT,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    NUM_BINS,
    OMEGA_0,
    P_READ_DBM,
    T_BATH_VALUES,
    T_C,
    TARGET_BACKWARD_ERROR_LIMIT,
    TAU_0,
    TAU_0_PB,
    TAU_L,
    TSTAR_OVER_DELTA,
    _build_grid,
    _nbar_from_table_iii,
    _validated_sweep_request,
    solve,
    solver_fingerprint,
)
from validation.fischer_2024 import _pdf as _pdf_validation

_CACHE_ROOT = Path(tempfile.gettempdir()) / "qpsim-cache"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

# Observable-only constants — they do NOT affect the kinetic solve, so they live
# here rather than in fig7_solve: editing them keeps a cached solve warm. ALPHA_KI
# (kinetic-inductance fraction) enters only Q_i,qp = pi Delta / (omega_0 alpha
# sigma_1); Q_EXT only the parallel-loss cap (Eq. 65). Every solve-affecting
# Table II/III constant (Delta_0, omega_0, tau_l, grid, P_read, T*/Delta, T_bath,
# and the solver knobs) lives in fig7_solve and is imported above.
ALPHA_KI = 0.13
Q_EXT_BY_DBM: dict[float, float] = {
    -100.0: 2.5e6,
    -90.0: 2.5e6,
    -80.0: 2.5e6,
    -72.0: 1.3e6,
    -68.0: 0.9e6,
    -64.0: 0.7e6,
}

#: Table 2 coupling quality factor (verified against the paper source).
#: The former overlay passed Q_EXT (~0.7-2.5e6) as Eq. 65's Q_c, inflating
#: the dashed curves by up to ~55x at 0.30 K (2026-07-20 round-4 review).
Q_C_PAPER = 20100.0
ARTIFACT_SCHEMA = "qpsim-fischer-2023-fig7-summary-v2"
LIVE_CERTIFICATE_SCOPE = "independently-reassembled-live-state"
SUMMARY_CERTIFICATE_SCOPE = (
    "producer-asserted-summary-only; solved-f-and-n_ph-state-omitted"
)

_GAMMA0_EQ63 = 19.3  # gamma_0 = pi 2^{1/3} 5 sqrt(7) (2.1)^2/(2.3 * 3^{3/2})


def _fig7_x_th_paper(T_K: float) -> float:
    """x_th = N_qp^{T_B}/(2 rho_F Delta) = sqrt(2 pi T_B/Delta) e^{-Delta/T_B}."""
    T_uev = T_K * KB_UEV_PER_K
    if T_uev <= 0:
        return 0.0
    return float(np.sqrt(2.0 * np.pi * T_uev / DELTA_0) * np.exp(-DELTA_0 / T_uev))


def _fig7_Q_thermal(T_K: float) -> float:
    """Thermal-equilibrium Q_i (paper Eq. 113 / ref. [31] form)."""
    from scipy.special import k0 as _k0

    T_uev = T_K * KB_UEV_PER_K
    if T_uev <= 0:
        return np.inf
    x = OMEGA_0 / (2.0 * T_uev)
    return float(
        np.pi / (4.0 * ALPHA_KI * np.sinh(x) * _k0(x)) * np.exp(DELTA_0 / T_uev)
    )


def _fig7_Q_i0_eq63(p_dbm: float) -> float:
    """Transcribed-formula pin for the low-T_B plateau in paper Eq. 63."""
    kBTs = TSTAR_OVER_DELTA[float(p_dbm)] * DELTA_0
    if TAU_L <= 0:
        return np.inf
    arg = float(np.sqrt(14.0 / 5.0)) * (DELTA_0 / kBTs) ** 3
    if arg > 700.0:
        return np.inf
    return (
        (_GAMMA0_EQ63 * DELTA_0 / (ALPHA_KI * OMEGA_0))
        * (TAU_0_PB / TAU_L)
        * (DELTA_0 / kBTs) ** 3
        * np.exp(arg)
    )


def _fig7_Q_i_eq65(p_dbm: float, T_K: float, Q_c: float) -> float:
    """Transcribed-formula pin for the driven high-T_B branch in paper Eq. 65."""
    x_th = _fig7_x_th_paper(T_K)
    if x_th <= 0.0:
        return np.inf
    amplitude = 2.1 * np.sqrt(2.0) * np.pi * DELTA_0 / (2.3 * ALPHA_KI * OMEGA_0)
    ts3 = TSTAR_OVER_DELTA[float(p_dbm)] ** 3
    return float(
        np.sqrt((Q_c / 2.0) ** 2 + amplitude**2 * ts3 / x_th**2) - Q_c / 2.0
    )


def _fig7_dashed_Q_tot(p_dbm: float, T_K: float) -> float:
    """Transcribed-formula pin: Eqs. 63+65 (or thermal at -100 dBm).

    The transcription uses Q_c = 20100 from Table 2, combined with the
    extrinsic loss as an effective coupling
    ``1/Q_c_eff = 1/Q_c + 1/Q_ext``. The -100 dBm branch uses the thermal
    equilibrium expression because ``T_{*,0} ~ omega_0`` there. The Eq. 63
    plateau and Eq. 65 branch add harmonically, then parallel with Q_ext.
    These helpers pin the implemented formulas; they are not a comparison
    with digitized Fig. 7 curve data.
    """
    Q_ext = Q_EXT_BY_DBM[float(p_dbm)]
    if float(p_dbm) == -100.0:
        Q_i = _fig7_Q_thermal(T_K)
    else:
        Q_c_eff = 1.0 / (1.0 / Q_C_PAPER + 1.0 / Q_ext)
        Q_i = 1.0 / (
            1.0 / max(_fig7_Q_i_eq65(p_dbm, T_K, Q_c_eff), 1.0)
            + 1.0 / max(_fig7_Q_i0_eq63(p_dbm), 1.0)
        )
    return 1.0 / (1.0 / max(Q_i, 1.0) + 1.0 / Q_ext)


# Same-system regression limits remain deliberately strict.  Exact
# single-thread hosted-Linux runs of the Windows pin expose a larger, stable
# same-root portability envelope: QP-loss drift reached 4.633e-3 and Q_tot
# drift 1.6342e-4 while every independently assembled balance certificate
# remained below 2e-8.  A tighter inner Newton gate was rejected because it
# selects a different low-occupation branch at T=0.14 K.  Keep the measured
# wider limits restricted to Windows/Linux comparisons; the absolute loss
# floor still covers only the numerically tiny cold-tail points.
#
# Floor calibration: the former 2e-19 floor had ZERO margin against
# measured hosted host-to-host jitter — CI run 29706352205 (Linux 3.14,
# identical code/library versions to earlier green runs) drifted the
# coldest -64 dBm point from the pinned 4.608e-19 to 6.737e-19, i.e.
# an absolute deviation of 2.129e-19 that breached the floor by 6%.
# Losses at this scale correspond to Q ~ 2e18 and are numerically
# indistinguishable from zero; 1e-18 keeps the floor ~5x above the
# worst observed jitter while remaining eight orders below the tightest
# physically meaningful pinned loss.
QP_LOSS_REGRESSION_RTOL = 4e-3
QP_LOSS_REGRESSION_ATOL = 1e-18
Q_TOTAL_REGRESSION_RTOL = 1e-4
QP_LOSS_CROSS_PLATFORM_RTOL = 8e-3
Q_TOTAL_CROSS_PLATFORM_RTOL = 2e-4


def fig7_regression_tolerances(
    generator_platform: str,
    *,
    running_system: str | None = None,
) -> tuple[float, float]:
    """Return ``(QP-loss rtol, Q_tot rtol)`` for a pinned comparison.

    ``generator_platform`` is the full :func:`platform.platform` record stored
    in the baseline.  Comparing only its operating-system prefix keeps
    same-system reruns strict without pretending that Windows and Linux
    floating-point solver paths are interchangeable at the same tolerance.
    The wider envelope is limited to that measured OS pair; unmeasured pairs
    retain the strict limits.
    """
    current_system = platform.system() if running_system is None else running_system
    pinned_system = generator_platform.split("-", maxsplit=1)[0]
    if pinned_system == "macOS":
        pinned_system = "Darwin"
    measured_pair = {pinned_system, current_system} == {"Windows", "Linux"}
    if measured_pair:
        return QP_LOSS_CROSS_PLATFORM_RTOL, Q_TOTAL_CROSS_PLATFORM_RTOL
    return QP_LOSS_REGRESSION_RTOL, Q_TOTAL_REGRESSION_RTOL


@dataclass(frozen=True)
class Fig7PaperResult:
    T_bath: np.ndarray
    p_read_dbm: tuple[float, ...]
    n_bar_by_dbm: dict[float, float]
    Q_qp_by_dbm: dict[float, np.ndarray]
    Q_tot_by_dbm: dict[float, np.ndarray]
    sigma1_by_dbm: dict[float, np.ndarray]
    qp_residual_inf: np.ndarray
    qp_backward_error: np.ndarray
    phonon_residual_inf: np.ndarray
    phonon_raw_backward_error: np.ndarray
    phonon_backward_error: np.ndarray
    qp_number_backward_error: np.ndarray
    certificate_scope: str = LIVE_CERTIFICATE_SCOPE


@dataclass(frozen=True)
class Fig7ProducerIdentity:
    """Source and numerical-runtime identity frozen before a Fig. 7 solve."""

    solve_contract_digest: str
    observable_contract_digest: str
    runner_sha256: str
    runtime: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRecord:
    """Hash and size of one fully written artifact."""

    sha256: str
    size_bytes: int


def file_sha256(path: Path) -> str:
    """Hash one file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_pdf_record(path: Path) -> ArtifactRecord:
    """Validate one nonempty Matplotlib PDF page and return its content record."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read Fig. 7 PDF at {path}: {exc}") from exc
    try:
        _pdf_validation.validate_single_nonempty_matplotlib_pdf(
            payload,
            path=path,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Fig. 7 artifact at {path} is not a complete nonempty "
            "one-page Matplotlib PDF."
        ) from exc
    return ArtifactRecord(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def runner_source_sha256(path: Path) -> str:
    """Portable hash of the exact campaign/generation runner source."""
    return hashlib.sha256(source_provenance.canonical_source_bytes(path)).hexdigest()


def capture_producer_identity(runner_path: Path) -> Fig7ProducerIdentity:
    """Freeze source, runner, and runtime identities before expensive work."""
    import matplotlib

    matplotlib.use("Agg")
    provenance = reproduction_provenance()
    runtime = {
        key: value
        for key, value in provenance.items()
        if key
        not in {
            "observable_contract_digest",
            "powers_dbm",
            "solve_contract_digest",
            "solver_fingerprint",
            "temperatures",
        }
    }
    # JSON round trip freezes numpy scalars/containers into one canonical,
    # mutation-independent record suitable for worker payloads and attestations.
    frozen_runtime = cast(
        dict[str, Any],
        json.loads(json.dumps(runtime, allow_nan=False, sort_keys=True)),
    )
    return Fig7ProducerIdentity(
        solve_contract_digest=str(provenance["solve_contract_digest"]),
        observable_contract_digest=str(
            provenance["observable_contract_digest"]
        ),
        runner_sha256=runner_source_sha256(runner_path),
        runtime=frozen_runtime,
    )


def assert_producer_identity_current(
    producer: Fig7ProducerIdentity,
    runner_path: Path,
) -> None:
    """Refuse persistence after source, runner, or runtime drift."""
    current = capture_producer_identity(runner_path)
    if current != producer:
        raise RuntimeError(
            "Fischer Fig. 7 producer identity changed during generation; "
            "discard the candidate and restart from a stable source/runtime."
        )


def solve_contract_digest() -> str:
    """Content digest for a portable Fig. 7 reproduction report.

    The disk cache already keys on the same ingredients.  Exposing one digest
    here lets an uncached cross-platform run state exactly which solver tree,
    Fig. 7 driver, and independent certificate implementation it exercised.
    """
    digest = hashlib.sha256()
    ingredients = (
        ("qpsim", sweep_cache.solve_source_digest()),
        ("solve_source_digest", inspect.getsource(sweep_cache.solve_source_digest)),
        (
            "canonical_source_bytes",
            inspect.getsource(source_provenance.canonical_source_bytes),
        ),
        ("fig7_solve", inspect.getsource(fig7_solve)),
        ("certificate", inspect.getsource(certificate_module)),
    )
    for label, source in ingredients:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def observable_contract_digest() -> str:
    """Portable content digest for the Fig. 7 observable/artifact layer.

    The kinetic solve digest deliberately excludes downstream observables so
    plot-only edits do not invalidate expensive raw-solve caches.  A promoted
    CSV/PDF pair still needs to identify the exact code that converted those
    raw occupations into observables and serialized the artifact.  Hash this
    module plus the complete :mod:`qpsim.observables` source tree with
    newline-canonical bytes so Windows and POSIX checkouts agree.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        Path(_pdf_validation.__file__).resolve(),
    ]
    paths.extend(sorted((root / "qpsim" / "observables").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_provenance.canonical_source_bytes(path))
        digest.update(b"\0")
    return digest.hexdigest()


def reproduction_provenance() -> dict[str, object]:
    """Return environment + source provenance for an uncached reproduction."""
    import matplotlib
    import scipy

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "python_implementation": platform.python_implementation(),
        "python_byteorder": sys.byteorder,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": {
            "backend": str(matplotlib.get_backend()),
            "version": matplotlib.__version__,
        },
        "numeric_runtime": sweep_cache._numeric_runtime_identity(),
        "solve_contract_digest": solve_contract_digest(),
        "observable_contract_digest": observable_contract_digest(),
        "solver_fingerprint": solver_fingerprint(),
        "temperatures": list(T_BATH_VALUES),
        "powers_dbm": list(P_READ_DBM),
    }


def _parallel_quality_factor(Q_qp: float, Q_ext: float) -> float:
    if not np.isfinite(Q_qp):
        return float(Q_ext)
    return float(1.0 / (1.0 / Q_qp + 1.0 / Q_ext))


def _validated_raw_payload(
    raw: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[float, ...], np.ndarray, int, np.ndarray, dict[str, np.ndarray]]:
    """Validate an uncached or disk-cached solve before deriving observables."""
    required = {
        "f_solved",
        "temperatures",
        "powers_dbm",
        "n_bar",
        "num_bins",
        *certificate_module.NUMBER_CERTIFICATE_FIELDS,
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Fig. 7 raw solve payload is missing fields {missing}.")

    num_bins_raw = np.asarray(raw["num_bins"])
    if num_bins_raw.shape != (1,):
        raise ValueError(
            "Fig. 7 raw num_bins must have exact shape (1,); "
            f"got {num_bins_raw.shape}."
        )
    num_bins_value = num_bins_raw[0]
    if (
        isinstance(num_bins_value, (bool, np.bool_))
        or not np.isfinite(num_bins_value)
        or float(num_bins_value) != int(num_bins_value)
    ):
        raise ValueError("Fig. 7 raw num_bins must contain one finite integer.")

    T_values, powers_array, num_bins = _validated_sweep_request(
        np.asarray(raw["temperatures"], dtype=float),  # type: ignore[arg-type]
        np.asarray(raw["powers_dbm"], dtype=float),  # type: ignore[arg-type]
        int(num_bins_value),
    )
    powers = tuple(float(power) for power in powers_array)
    n_bar = np.asarray(raw["n_bar"], dtype=float)
    if n_bar.shape != powers_array.shape or np.any(~np.isfinite(n_bar)) or np.any(n_bar < 0.0):
        raise ValueError(
            "Fig. 7 raw n_bar must be finite, non-negative, and match the power axis."
        )
    expected_n_bar = np.asarray([_nbar_from_table_iii(power) for power in powers])
    if not np.allclose(n_bar, expected_n_bar, rtol=1e-14, atol=0.0):
        raise ValueError("Fig. 7 raw n_bar is inconsistent with the Table-III power axis.")

    f_solved = np.asarray(raw["f_solved"], dtype=float)
    expected_f_shape = (len(powers), T_values.size, num_bins)
    if f_solved.shape != expected_f_shape:
        raise ValueError(
            f"Fig. 7 raw f_solved has shape {f_solved.shape}; "
            f"expected {expected_f_shape}."
        )
    if np.any(~np.isfinite(f_solved)) or np.any((f_solved < 0.0) | (f_solved > 1.0)):
        raise ValueError("Fig. 7 raw f_solved must contain finite occupations in [0, 1].")

    certificate_shape = (len(powers), T_values.size)
    certificate_arrays = {
        field: np.asarray(raw[field], dtype=float)
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for field, values in certificate_arrays.items():
        if values.shape != certificate_shape:
            raise ValueError(
                f"Fig. 7 certificate field {field!r} has shape {values.shape}; "
                f"expected {certificate_shape}."
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"Fig. 7 certificate field {field!r} must be finite and non-negative."
            )
    for field in _CERTIFIED_BACKWARD_ERROR_FIELDS:
        if np.any(certificate_arrays[field] > TARGET_BACKWARD_ERROR_LIMIT):
            raise ValueError(
                f"Fig. 7 raw certificate field {field!r} exceeds "
                f"{TARGET_BACKWARD_ERROR_LIMIT:g}."
            )
    return T_values, powers, n_bar, num_bins, f_solved, certificate_arrays


def observables(raw: Mapping[str, np.ndarray]) -> Fig7PaperResult:
    """Derive sigma_1 / Q_i,qp / Q_i,tot from a raw :func:`fig7_solve.solve` payload.

    The cheap downstream half of Fig. 7: rebuilds the (deterministic) spectral
    grid and evaluates the Mattis-Bardeen sigma_1 and the quality factors per
    (power, T_bath) point. A pure function of ``raw`` — this is what the cache
    leaves uncached, so editing it never triggers a re-solve.
    """
    T_values, powers, n_bar, num_bins, f_solved, certificate_arrays = (
        _validated_raw_payload(raw)
    )

    spectral, _omega = _build_grid(num_bins)

    n_bar_by_dbm = {p: float(n_bar[pi]) for pi, p in enumerate(powers)}
    Q_qp_by_dbm = {p: np.zeros_like(T_values) for p in powers}
    Q_tot_by_dbm = {p: np.zeros_like(T_values) for p in powers}
    sigma1_by_dbm = {p: np.zeros_like(T_values) for p in powers}

    for pi, p in enumerate(powers):
        for i in range(T_values.size):
            sigma1, _ = compute_ac_conductivity(f_solved[pi, i], spectral, OMEGA_0)
            Q_qp = (
                np.pi * DELTA_0 / (OMEGA_0 * ALPHA_KI * sigma1)
                if sigma1 > 0.0
                else float("inf")
            )
            Q_qp_by_dbm[p][i] = Q_qp
            Q_tot_by_dbm[p][i] = _parallel_quality_factor(Q_qp, Q_EXT_BY_DBM[p])
            sigma1_by_dbm[p][i] = sigma1

    return Fig7PaperResult(
        T_bath=T_values,
        p_read_dbm=powers,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
        qp_residual_inf=certificate_arrays["qp_residual_inf"],
        qp_backward_error=certificate_arrays["qp_backward_error"],
        phonon_residual_inf=certificate_arrays["phonon_residual_inf"],
        phonon_raw_backward_error=certificate_arrays[
            "phonon_raw_backward_error"
        ],
        phonon_backward_error=certificate_arrays["phonon_backward_error"],
        qp_number_backward_error=certificate_arrays[
            certificate_module.QP_NUMBER_CERTIFICATE_FIELD
        ],
        certificate_scope=LIVE_CERTIFICATE_SCOPE,
    )


def run(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> Fig7PaperResult:
    """Solve the sweep and derive observables — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this so it always truly recomputes against the pinned baseline;
    the cached dev / regen path is :func:`run_cached`.
    """
    return observables(
        solve(temperatures=temperatures, powers_dbm=powers_dbm, num_bins=num_bins)
    )


def run_cached(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> Fig7PaperResult:
    """Like :func:`run`, but the expensive solve is served from the disk cache
    when nothing solve-relevant has changed (see :mod:`validation.sweep_cache`).

    Used by the regen / ``__main__`` path. Editing the observables or plotting
    code in this module does not invalidate the cached solve; any edit to
    :mod:`fig7_solve` or to the ``qpsim`` solver subtree does. Disable entirely
    with ``QPSIM_SWEEP_CACHE=0``.
    """
    kwargs = {
        "temperatures": [float(x) for x in temperatures],
        "powers_dbm": [float(x) for x in powers_dbm],
        "num_bins": int(num_bins),
    }
    raw = sweep_cache.cached_solve(
        "fischer_2023/fig7",
        lambda: solve(
            temperatures=temperatures, powers_dbm=powers_dbm, num_bins=num_bins
        ),
        fingerprint=solver_fingerprint(num_bins=num_bins),
        kwargs=kwargs,
        # The raw payload now includes independently reassembled balance
        # certificates. Their helper lives outside qpsim/ and fig7_solve, so it
        # must participate explicitly in cache invalidation.
        extra_source=(
            inspect.getsource(fig7_solve)
            + inspect.getsource(certificate_module)
        ),
    )
    return observables(raw)


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig7_paper.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_CERTIFICATE_VERSION_RE = re.compile(r"certificate_metric_version='([^']*)'")
_CERTIFICATE_LIMIT_RE = re.compile(
    r"target_backward_error_limit=([\deE.+-]+)"
)
_SOLVE_CONTRACT_RE = re.compile(r"solve_contract_digest=([0-9a-f]{64})")
_OBSERVABLE_CONTRACT_RE = re.compile(
    r"observable_contract_digest=([0-9a-f]{64})"
)
_RUNNER_SHA_RE = re.compile(r"runner_sha256=([0-9a-f]{64})")
_COMPANION_PDF_SHA_RE = re.compile(
    r"companion_pdf_sha256=(none|[0-9a-f]{64})"
)
_COMPANION_PDF_SIZE_RE = re.compile(r"companion_pdf_size_bytes=(\d+)")
_ARTIFACT_SCHEMA_RE = re.compile(r"artifact_schema='([^']*)'")
_CERTIFICATE_SCOPE_RE = re.compile(r"certificate_scope='([^']*)'")
_GENERATOR_RE = {
    "platform": re.compile(r"generator_platform='([^']*)'"),
    "python": re.compile(r"generator_python='([^']*)'"),
    "numpy": re.compile(r"generator_numpy='([^']*)'"),
    "scipy": re.compile(r"generator_scipy='([^']*)'"),
}
_BASELINE_COLUMNS = (
    "T_bath_K",
    "P_read_dbm",
    "n_bar",
    "Q_qp",
    "Q_tot",
    "sigma1",
    *certificate_module.NUMBER_CERTIFICATE_FIELDS,
)
_CERTIFIED_BACKWARD_ERROR_FIELDS = (
    "qp_backward_error",
    "phonon_backward_error",
    certificate_module.QP_NUMBER_CERTIFICATE_FIELD,
)
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_c": re.compile(r"T_c=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "alpha": re.compile(r"alpha=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
    "tau_l": re.compile(r"tau_l=([\deE.+-]+)"),
    "tau_0_pb": re.compile(r"tau_0_pb=([\deE.+-]+)"),
}


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and replace ``path`` on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield cast(TextIO, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_certificate_arrays(
    result: Fig7PaperResult,
    *,
    context: str,
) -> None:
    """Reject incomplete or failed persisted steady-state certificates."""
    expected_shape = (len(result.p_read_dbm), result.T_bath.size)
    for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
        values = np.asarray(getattr(result, field), dtype=float)
        if values.shape != expected_shape:
            raise RuntimeError(
                f"{context} certificate field {field!r} has shape "
                f"{values.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(
                f"{context} certificate field {field!r} must contain only "
                "finite non-negative values."
            )
    for field in _CERTIFIED_BACKWARD_ERROR_FIELDS:
        maximum = float(np.max(np.asarray(getattr(result, field), dtype=float)))
        if maximum > TARGET_BACKWARD_ERROR_LIMIT:
            raise RuntimeError(
                f"{context} certificate field {field!r} has maximum "
                f"{maximum:.17g}, above target "
                f"{TARGET_BACKWARD_ERROR_LIMIT:.17g}."
            )


def _validate_result_for_artifact(result: Fig7PaperResult) -> None:
    """Validate the rectangular, finite Fig. 7 payload before serialization."""
    temperatures = np.asarray(result.T_bath, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise RuntimeError("Fig. 7 artifact temperatures must be a non-empty 1-D array.")
    if (
        not np.all(np.isfinite(temperatures))
        or np.any(temperatures <= 0.0)
        or np.unique(temperatures).size != temperatures.size
    ):
        raise RuntimeError("Fig. 7 artifact temperatures must be finite, positive, and unique.")
    powers = np.asarray(result.p_read_dbm, dtype=float)
    if powers.ndim != 1 or powers.size == 0:
        raise RuntimeError("Fig. 7 artifact powers must be a non-empty 1-D tuple.")
    if not np.all(np.isfinite(powers)) or np.unique(powers).size != powers.size:
        raise RuntimeError("Fig. 7 artifact powers must be finite and unique.")
    unsupported = sorted({float(power) for power in powers} - set(Q_EXT_BY_DBM))
    if unsupported:
        raise RuntimeError(
            f"Fig. 7 artifact contains unsupported readout powers {unsupported}."
        )

    expected_keys = set(result.p_read_dbm)
    for name, mapping in (
        ("n_bar", result.n_bar_by_dbm),
        ("Q_qp", result.Q_qp_by_dbm),
        ("Q_tot", result.Q_tot_by_dbm),
        ("sigma1", result.sigma1_by_dbm),
    ):
        if set(mapping) != expected_keys:
            raise RuntimeError(
                f"Fig. 7 artifact {name} power keys do not exactly match "
                f"{result.p_read_dbm}."
            )
        for power in result.p_read_dbm:
            values = np.asarray(mapping[power], dtype=float)
            expected_shape = () if name == "n_bar" else temperatures.shape
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Fig. 7 artifact {name} at P={power:g} must have finite "
                    f"shape {expected_shape}; got {values.shape}."
                )
            if name in {"n_bar", "Q_qp", "Q_tot"} and np.any(values <= 0.0):
                raise RuntimeError(f"Fig. 7 artifact {name} at P={power:g} must be positive.")
            if name == "sigma1" and np.any(values <= 0.0):
                raise RuntimeError(f"Fig. 7 artifact sigma1 at P={power:g} must be positive.")

    for power in result.p_read_dbm:
        expected_n_bar = _nbar_from_table_iii(power)
        if not np.isclose(
            result.n_bar_by_dbm[power],
            expected_n_bar,
            rtol=1e-14,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Fig. 7 artifact n_bar at P={power:g} is inconsistent with Table III."
            )
        expected_q_qp = (
            np.pi
            * DELTA_0
            / (OMEGA_0 * ALPHA_KI * np.asarray(result.sigma1_by_dbm[power]))
        )
        if not np.allclose(
            result.Q_qp_by_dbm[power],
            expected_q_qp,
            rtol=1e-14,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Fig. 7 artifact Q_qp at P={power:g} is not bound to sigma1."
            )
        expected_q_tot = 1.0 / (
            1.0 / expected_q_qp + 1.0 / Q_EXT_BY_DBM[power]
        )
        if not np.allclose(
            result.Q_tot_by_dbm[power],
            expected_q_tot,
            rtol=1e-14,
            atol=0.0,
        ):
            raise RuntimeError(
                f"Fig. 7 artifact Q_tot at P={power:g} is not bound to Q_qp and Q_ext."
            )
    _validate_certificate_arrays(result, context="Fig. 7 artifact")


def write_baseline(
    result: Fig7PaperResult,
    path: Path | None = None,
    *,
    producer: Fig7ProducerIdentity | None = None,
    companion_pdf: ArtifactRecord | None = None,
) -> Path:
    if path is None:
        path = baseline_path()
    if path.resolve() == baseline_path().resolve():
        raise RuntimeError(
            "Direct writes to the canonical Fig. 7 CSV are forbidden; "
            "use publish_baseline_pair() or generate_baseline()."
        )
    _validate_result_for_artifact(result)
    if result.certificate_scope != LIVE_CERTIFICATE_SCOPE:
        raise RuntimeError(
            "Fig. 7 publication requires independently certified live-state "
            f"evidence; got certificate_scope={result.certificate_scope!r}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if producer is None:
        producer = capture_producer_identity(Path(__file__).resolve())
    if (
        companion_pdf is not None
        and (
            len(companion_pdf.sha256) != 64
            or any(char not in "0123456789abcdef" for char in companion_pdf.sha256)
            or companion_pdf.size_bytes <= 0
        )
    ):
        raise ValueError("Fig. 7 companion PDF record is malformed.")
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Fischer 2023 Fig. 7 paper-topology qpsim regression; "
            "no digitized-data comparison"
        ])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.17g} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} tau_l={TAU_L} tau_0_pb={TAU_0_PB}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow([
            "# certificate_metric_version="
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r} "
            "target_backward_error_limit="
            f"{TARGET_BACKWARD_ERROR_LIMIT}"
        ])
        writer.writerow([
            f"# artifact_schema={ARTIFACT_SCHEMA!r} "
            f"certificate_scope={SUMMARY_CERTIFICATE_SCOPE!r}"
        ])
        writer.writerow([
            "# solve_contract_digest="
            f"{producer.solve_contract_digest} "
            "observable_contract_digest="
            f"{producer.observable_contract_digest} "
            f"generator_platform={producer.runtime['platform']!r} "
            f"generator_python={producer.runtime['python']!r} "
            f"generator_numpy={producer.runtime['numpy']!r} "
            f"generator_scipy={producer.runtime['scipy']!r}"
        ])
        writer.writerow([
            f"# runner_sha256={producer.runner_sha256} "
            "companion_pdf_sha256="
            f"{companion_pdf.sha256 if companion_pdf is not None else 'none'} "
            "companion_pdf_size_bytes="
            f"{companion_pdf.size_bytes if companion_pdf is not None else 0}"
        ])
        writer.writerow([f"# p_read_dbm={','.join(f'{p:g}' for p in result.p_read_dbm)}"])
        writer.writerow(_BASELINE_COLUMNS)
        for pi, p in enumerate(result.p_read_dbm):
            for i, T_bath in enumerate(result.T_bath):
                writer.writerow([
                    f"{T_bath:.17e}",
                    f"{p:.17e}",
                    f"{result.n_bar_by_dbm[p]:.17e}",
                    f"{result.Q_qp_by_dbm[p][i]:.17e}",
                    f"{result.Q_tot_by_dbm[p][i]:.17e}",
                    f"{result.sigma1_by_dbm[p][i]:.17e}",
                    f"{result.qp_residual_inf[pi, i]:.17e}",
                    f"{result.qp_backward_error[pi, i]:.17e}",
                    f"{result.phonon_residual_inf[pi, i]:.17e}",
                    f"{result.phonon_raw_backward_error[pi, i]:.17e}",
                    f"{result.phonon_backward_error[pi, i]:.17e}",
                    f"{result.qp_number_backward_error[pi, i]:.17e}",
                ])
    return path


def read_baseline(
    path: Path | None = None,
    *,
    companion_pdf_path: Path | None = None,
    require_companion_pdf: bool = False,
    accept_producer_certificate_claims: bool = False,
    _validate_canonical_attestation: bool = True,
    _publication_lock_held: bool = False,
) -> Fig7PaperResult:
    """Read a persisted Fig. 7 summary.

    The CSV omits the solved quasiparticle and phonon states. Its certificate
    scalars are authenticated producer assertions, not quantities a reader can
    independently reassemble from the persisted summary. Every summary read
    therefore requires an explicit opt-in; regenerate the sweep when independent
    live-state certification is required. Canonical reads additionally
    authenticate the guarded CSV/PDF promotion record.
    """
    if path is None:
        path = baseline_path()
    canonical_read = path.resolve() == baseline_path().resolve()
    if canonical_read and not _publication_lock_held:
        with _publication_lock(promotion_attestation_path(path)):
            return read_baseline(
                path,
                companion_pdf_path=companion_pdf_path,
                require_companion_pdf=require_companion_pdf,
                accept_producer_certificate_claims=(
                    accept_producer_certificate_claims
                ),
                _validate_canonical_attestation=_validate_canonical_attestation,
                _publication_lock_held=True,
            )
    if not accept_producer_certificate_claims:
        raise RuntimeError(
            "Fig. 7 summary artifacts omit solved f(E) and n_ph states, so "
            "its persisted certificate scalars are producer assertions only. "
            "Pass accept_producer_certificate_claims=True to read the summary, "
            "or regenerate for independent live-state certification."
        )
    metadata = read_baseline_metadata(
        path,
        _publication_lock_held=_publication_lock_held,
    )
    if canonical_read:
        require_companion_pdf = True
        if companion_pdf_path is None:
            companion_pdf_path = plot_path()
    if metadata.companion_pdf_sha256 is None:
        if require_companion_pdf:
            raise LegacyArtifactError(
                f"Fig. 7 baseline at {path} does not authenticate its companion PDF."
            )
    else:
        if companion_pdf_path is None:
            if require_companion_pdf:
                raise RuntimeError(
                    f"Fig. 7 baseline at {path} requires a companion PDF path."
                )
        else:
            record = complete_pdf_record(companion_pdf_path)
            if (
                record.sha256 != metadata.companion_pdf_sha256
                or record.size_bytes != metadata.companion_pdf_size_bytes
            ):
                raise RuntimeError(
                    f"Fig. 7 baseline at {path} does not authenticate "
                    f"companion PDF {companion_pdf_path}."
                )
    if (
        metadata.certificate_metric_version
        != certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
    ):
        raise RuntimeError(
            f"Fig. 7 baseline at {path} uses unsupported certificate metric "
            f"{metadata.certificate_metric_version!r}; expected "
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r}."
        )
    if metadata.target_backward_error_limit != TARGET_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} has certificate target "
            f"{metadata.target_backward_error_limit:.17g}; expected "
            f"{TARGET_BACKWARD_ERROR_LIMIT:.17g}."
        )
    if metadata.artifact_schema != ARTIFACT_SCHEMA:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} uses unsupported artifact schema "
            f"{metadata.artifact_schema!r}; expected {ARTIFACT_SCHEMA!r}."
        )
    if metadata.certificate_scope != SUMMARY_CERTIFICATE_SCOPE:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} has unsupported certificate scope "
            f"{metadata.certificate_scope!r}; expected "
            f"{SUMMARY_CERTIFICATE_SCOPE!r}."
        )
    rows: list[list[float]] = []
    powers: list[float] = []
    power_records = 0
    columns: tuple[str, ...] | None = None
    with path.open(encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# p_read_dbm"):
                power_records += 1
                if power_records != 1:
                    raise RuntimeError(
                        f"Fig. 7 baseline at {path} has duplicate power metadata."
                    )
                try:
                    powers = [float(x) for x in first.split("=", 1)[1].split(",")]
                except (IndexError, ValueError) as exc:
                    raise RuntimeError(
                        f"Fig. 7 baseline at {path} has malformed power metadata."
                    ) from exc
                continue
            if first.startswith("#"):
                continue
            if first == "T_bath_K":
                current_columns = tuple(line)
                if columns is not None:
                    raise RuntimeError(
                        f"Fig. 7 baseline at {path} has duplicate column headers."
                    )
                if current_columns != _BASELINE_COLUMNS:
                    raise RuntimeError(
                        f"Fig. 7 baseline at {path} has columns "
                        f"{current_columns}; expected current certified schema "
                        f"{_BASELINE_COLUMNS}."
                    )
                columns = current_columns
                continue
            if columns is None:
                raise RuntimeError(
                    f"Fig. 7 baseline at {path} has data before its column header."
                )
            if len(line) != len(_BASELINE_COLUMNS):
                raise RuntimeError(
                    f"Fig. 7 baseline row has {len(line)} columns; expected "
                    f"the current {len(_BASELINE_COLUMNS)}-column certified schema."
                )
            try:
                values = [float(x) for x in line]
            except ValueError as exc:
                raise RuntimeError(
                    f"Fig. 7 baseline at {path} contains a non-numeric data row."
                ) from exc
            rows.append(values)
    if not powers:
        raise RuntimeError(f"Baseline at {path} missing '# p_read_dbm=' metadata.")
    if columns is None:
        raise RuntimeError(f"Fig. 7 baseline at {path} missing its column header.")
    power_array = np.asarray(powers, dtype=float)
    if not np.all(np.isfinite(power_array)) or np.unique(power_array).size != len(powers):
        raise RuntimeError(
            f"Fig. 7 baseline at {path} has non-finite or duplicate power metadata."
        )
    if not rows:
        raise RuntimeError(f"Fig. 7 baseline at {path} contains no data rows.")
    data = np.array(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            f"Fig. 7 baseline at {path} contains non-finite data or certificates."
        )
    temps = np.unique(data[:, 0])
    actual_keys = [(float(row[1]), float(row[0])) for row in data]
    if len(set(actual_keys)) != len(actual_keys):
        raise RuntimeError(
            f"Fig. 7 baseline at {path} contains duplicate (P_read, T_bath) rows."
        )
    expected_keys = {(p, float(T)) for p in powers for T in temps}
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys.difference(actual_keys))
        extra = sorted(set(actual_keys).difference(expected_keys))
        raise RuntimeError(
            f"Fig. 7 baseline at {path} is not the exact Cartesian "
            f"(P_read, T_bath) grid; missing={missing}, extra={extra}."
        )
    p_tuple = tuple(powers)
    n_bar_by_dbm: dict[float, float] = {}
    Q_qp_by_dbm: dict[float, np.ndarray] = {}
    Q_tot_by_dbm: dict[float, np.ndarray] = {}
    sigma1_by_dbm: dict[float, np.ndarray] = {}
    certificate_arrays = {
        field: np.full((len(p_tuple), temps.size), np.nan, dtype=float)
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for pi, p in enumerate(p_tuple):
        subset = data[data[:, 1] == p]
        if subset.shape[0] != temps.size:
            raise RuntimeError(f"Baseline has {subset.shape[0]} rows for P={p}, expected {temps.size}.")
        order = np.argsort(subset[:, 0])
        subset = subset[order]
        if not np.all(subset[:, 2] == subset[0, 2]):
            raise RuntimeError(
                f"Fig. 7 baseline at {path} has inconsistent n_bar values "
                f"for P={p:g}."
            )
        n_bar_by_dbm[p] = float(subset[0, 2])
        Q_qp_by_dbm[p] = subset[:, 3]
        Q_tot_by_dbm[p] = subset[:, 4]
        sigma1_by_dbm[p] = subset[:, 5]
        for field_index, field in enumerate(
            certificate_module.NUMBER_CERTIFICATE_FIELDS
        ):
            certificate_arrays[field][pi] = subset[:, 6 + field_index]
    result = Fig7PaperResult(
        T_bath=temps,
        p_read_dbm=p_tuple,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
        qp_residual_inf=certificate_arrays["qp_residual_inf"],
        qp_backward_error=certificate_arrays["qp_backward_error"],
        phonon_residual_inf=certificate_arrays["phonon_residual_inf"],
        phonon_raw_backward_error=certificate_arrays[
            "phonon_raw_backward_error"
        ],
        phonon_backward_error=certificate_arrays["phonon_backward_error"],
        qp_number_backward_error=certificate_arrays[
            certificate_module.QP_NUMBER_CERTIFICATE_FIELD
        ],
        certificate_scope=SUMMARY_CERTIFICATE_SCOPE,
    )
    _validate_result_for_artifact(result)
    if canonical_read and _validate_canonical_attestation:
        attestation_path = promotion_attestation_path(path)
        if not attestation_path.is_file():
            raise LegacyArtifactError(
                f"Fig. 7 canonical baseline at {path} has no current "
                f"promotion attestation at {attestation_path}."
            )
        validate_promotion_attestation(
            attestation_path,
            csv_path=path,
            pdf_path=cast(Path, companion_pdf_path),
            _publication_lock_held=_publication_lock_held,
        )
    return result


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config) without
    touching the data rows or running the kinetic sweep.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds (see :mod:`fig6_paper` for the same
    pattern). Note Fig. 7's ``τ_l`` / ``τ_0^PB`` are fixed Table II/III scalars
    (``TAU_L`` / ``TAU_0_PB``), not phonon-side extractions. The ``p_read_dbm``
    powers and per-power ``n_bar`` are compared separately against the baseline
    rows.
    """

    delta_0: float
    tau_0: float
    t_c: float
    omega_0: float
    alpha: float
    c_phot: float
    tau_l: float
    tau_0_pb: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    certificate_metric_version: str
    target_backward_error_limit: float
    artifact_schema: str
    certificate_scope: str
    solve_contract_digest: str
    observable_contract_digest: str
    runner_sha256: str
    companion_pdf_sha256: str | None
    companion_pdf_size_bytes: int
    generator_platform: str
    generator_python: str
    generator_numpy: str
    generator_scipy: str


class LegacyArtifactError(RuntimeError):
    """A pre-provenance Fig. 7 pin that must not validate current code."""


def read_baseline_metadata(
    path: Path | None = None,
    *,
    _publication_lock_held: bool = False,
) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve). Raises
    ``RuntimeError`` if any stamped field is missing — an old/malformed header
    should fail loudly rather than silently skip the check.
    """
    if path is None:
        path = baseline_path()
    canonical_read = path.resolve() == baseline_path().resolve()
    if canonical_read and not _publication_lock_held:
        with _publication_lock(promotion_attestation_path(path)):
            return read_baseline_metadata(
                path,
                _publication_lock_held=True,
            )
    text = path.read_text(encoding="utf-8")

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
    versions = _CERTIFICATE_VERSION_RE.findall(text)
    limits = _CERTIFICATE_LIMIT_RE.findall(text)
    contracts = _SOLVE_CONTRACT_RE.findall(text)
    observable_contracts = _OBSERVABLE_CONTRACT_RE.findall(text)
    runner_hashes = _RUNNER_SHA_RE.findall(text)
    companion_hashes = _COMPANION_PDF_SHA_RE.findall(text)
    companion_sizes = _COMPANION_PDF_SIZE_RE.findall(text)
    artifact_schemas = _ARTIFACT_SCHEMA_RE.findall(text)
    certificate_scopes = _CERTIFICATE_SCOPE_RE.findall(text)
    if ne_m is None or len(versions) != 1 or len(limits) != 1:
        raise RuntimeError(
            f"Baseline header at {path} must contain NE and exactly one "
            "certificate metric / target record."
        )
    if not contracts:
        raise LegacyArtifactError(
            f"Fig. 7 baseline at {path} predates solve-contract provenance."
        )
    if len(contracts) != 1:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} must contain exactly one solve-contract digest."
        )
    if not observable_contracts:
        raise LegacyArtifactError(
            f"Fig. 7 baseline at {path} predates observable-contract digest "
            "provenance."
        )
    if len(observable_contracts) != 1:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} must contain exactly one "
            "observable-contract digest."
        )
    if (
        len(runner_hashes) != 1
        or len(companion_hashes) != 1
        or len(companion_sizes) != 1
    ):
        raise LegacyArtifactError(
            f"Fig. 7 baseline at {path} has no unique runner/PDF provenance; "
            "regenerate it through the guarded Fig. 7 publisher."
        )
    if not artifact_schemas or not certificate_scopes:
        raise LegacyArtifactError(
            f"Fig. 7 baseline at {path} predates explicit artifact schema / "
            "producer-certificate scope metadata; regenerate it through the "
            "guarded Fig. 7 publisher."
        )
    if len(artifact_schemas) != 1 or len(certificate_scopes) != 1:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} must contain exactly one artifact "
            "schema and certificate-scope record."
        )
    companion_sha = (
        None if companion_hashes[0] == "none" else companion_hashes[0]
    )
    companion_size = int(companion_sizes[0])
    if (companion_sha is None) != (companion_size == 0):
        raise RuntimeError(
            f"Fig. 7 baseline at {path} has inconsistent companion PDF provenance."
        )
    generator_values: dict[str, str] = {}
    for field, pattern in _GENERATOR_RE.items():
        values = pattern.findall(text)
        if len(values) != 1 or not values[0]:
            raise RuntimeError(
                f"Fig. 7 baseline at {path} must contain exactly one non-empty "
                f"generator_{field} record."
            )
        generator_values[field] = values[0]
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_c=_num(_HEADER_PARAM_RE["t_c"], "T_c"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        alpha=_num(_HEADER_PARAM_RE["alpha"], "alpha"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        tau_l=_num(_HEADER_PARAM_RE["tau_l"], "tau_l"),
        tau_0_pb=_num(_HEADER_PARAM_RE["tau_0_pb"], "tau_0_pb"),
        num_bins=int(ne_m.group(1)),
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        certificate_metric_version=versions[0],
        target_backward_error_limit=_num(
            _CERTIFICATE_LIMIT_RE,
            "target_backward_error_limit",
        ),
        artifact_schema=artifact_schemas[0],
        certificate_scope=certificate_scopes[0],
        solve_contract_digest=contracts[0],
        observable_contract_digest=observable_contracts[0],
        runner_sha256=runner_hashes[0],
        companion_pdf_sha256=companion_sha,
        companion_pdf_size_bytes=companion_size,
        generator_platform=generator_values["platform"],
        generator_python=generator_values["python"],
        generator_numpy=generator_values["numpy"],
        generator_scipy=generator_values["scipy"],
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — pure constants, so effectively instant (Fig. 7's
    ``τ_l`` / ``τ_0^PB`` are fixed Table II/III scalars, not extracted)."""
    provenance = reproduction_provenance()
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_c=T_C,
        omega_0=OMEGA_0,
        alpha=ALPHA_KI,
        c_phot=C_PHOT,
        tau_l=TAU_L,
        tau_0_pb=TAU_0_PB,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        certificate_metric_version=(
            certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
        ),
        target_backward_error_limit=TARGET_BACKWARD_ERROR_LIMIT,
        artifact_schema=ARTIFACT_SCHEMA,
        certificate_scope=SUMMARY_CERTIFICATE_SCOPE,
        solve_contract_digest=str(provenance["solve_contract_digest"]),
        observable_contract_digest=str(
            provenance["observable_contract_digest"]
        ),
        runner_sha256=runner_source_sha256(Path(__file__).resolve()),
        companion_pdf_sha256=None,
        companion_pdf_size_bytes=0,
        generator_platform=str(provenance["platform"]),
        generator_python=str(provenance["python"]),
        generator_numpy=str(provenance["numpy"]),
        generator_scipy=str(provenance["scipy"]),
    )


def write_plot(result: Fig7PaperResult, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    if path.resolve() == plot_path().resolve():
        raise RuntimeError(
            "Direct writes to the canonical Fig. 7 PDF are forbidden; "
            "use publish_baseline_pair() or generate_baseline()."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paper Fig. 7 palette per readout power, matching standalone reproduction
    # figures/fig7.py PAPER_COLORS.
    PAPER_COLORS = {
        -100.0: "firebrick", -90.0: "dimgray", -80.0: "black",
        -72.0: "red", -68.0: "green", -64.0: "blue",
    }
    fallback_cmap = matplotlib.colormaps["viridis_r"]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    T_dense = np.linspace(float(np.min(result.T_bath)),
                          float(np.max(result.T_bath)), 200)
    for i, p in enumerate(result.p_read_dbm):
        color = PAPER_COLORS.get(
            float(p), fallback_cmap(i / max(1, len(result.p_read_dbm) - 1))
        )
        ax.semilogy(
            result.T_bath, result.Q_tot_by_dbm[p],
            "o-", lw=1.4, ms=3.0, color=color, label=f"{p:g} dBm",
        )
        if float(p) in TSTAR_OVER_DELTA:
            Qa = np.array([_fig7_dashed_Q_tot(float(p), T) for T in T_dense])
            ax.semilogy(T_dense, Qa, color=color, ls="--", lw=1.0)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$Q_{i,\mathrm{tot}}$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        fig.savefig(temporary, format=path.suffix.removeprefix("."))
        complete_pdf_record(temporary)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        plt.close(fig)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def promotion_attestation_path(csv_path: Path | None = None) -> Path:
    """Return the durable promotion-attestation path for a Fig. 7 CSV."""
    if csv_path is None:
        csv_path = baseline_path()
    return csv_path.with_suffix(".promotion.json")


@contextmanager
def _publication_lock(
    attestation_path: Path | None = None,
) -> Iterator[None]:
    """Serialize readers/publishers of one canonical Fig. 7 artifact triple."""
    if attestation_path is None:
        attestation_path = promotion_attestation_path()
    lock_path = attestation_path.with_suffix(attestation_path.suffix + ".lock")
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
            f"Another Fig. 7 publisher or reader holds {lock_path}; "
            "refusing a mixed canonical artifact snapshot."
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


def _assert_same_result(
    expected: Fig7PaperResult,
    actual: Fig7PaperResult,
) -> None:
    np.testing.assert_array_equal(actual.T_bath, expected.T_bath)
    if actual.p_read_dbm != expected.p_read_dbm:
        raise RuntimeError("Fig. 7 CSV round trip changed the power axis.")
    if actual.n_bar_by_dbm != expected.n_bar_by_dbm:
        raise RuntimeError("Fig. 7 CSV round trip changed the n_bar mapping.")
    for power in expected.p_read_dbm:
        np.testing.assert_array_equal(
            actual.Q_qp_by_dbm[power],
            expected.Q_qp_by_dbm[power],
        )
        np.testing.assert_array_equal(
            actual.Q_tot_by_dbm[power],
            expected.Q_tot_by_dbm[power],
        )
        np.testing.assert_array_equal(
            actual.sigma1_by_dbm[power],
            expected.sigma1_by_dbm[power],
        )
    for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(actual, field),
            getattr(expected, field),
        )


def _result_point_hashes(
    result: Fig7PaperResult,
    payload_hashes: Mapping[tuple[int, int], str] | None,
) -> list[dict[str, object]]:
    """Hash every exact persisted point, optionally binding worker payloads."""
    expected_keys = {
        (pi, ti)
        for pi in range(len(result.p_read_dbm))
        for ti in range(result.T_bath.size)
    }
    if payload_hashes is not None and set(payload_hashes) != expected_keys:
        raise RuntimeError(
            "Fig. 7 worker-payload hashes do not cover the exact point grid."
        )
    records: list[dict[str, object]] = []
    for pi, power in enumerate(result.p_read_dbm):
        for ti, temperature in enumerate(result.T_bath):
            values = [
                float(temperature),
                float(power),
                float(result.n_bar_by_dbm[power]),
                float(result.Q_qp_by_dbm[power][ti]),
                float(result.Q_tot_by_dbm[power][ti]),
                float(result.sigma1_by_dbm[power][ti]),
                *[
                    float(getattr(result, field)[pi, ti])
                    for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
                ],
            ]
            point_payload = "\n".join(value.hex() for value in values).encode(
                "ascii"
            )
            record: dict[str, object] = {
                "power_index": pi,
                "temperature_index": ti,
                "power_dbm": float(power),
                "temperature_K": float(temperature),
                "result_sha256": hashlib.sha256(point_payload).hexdigest(),
            }
            if payload_hashes is not None:
                payload_hash = payload_hashes[(pi, ti)]
                if (
                    len(payload_hash) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in payload_hash
                    )
                ):
                    raise RuntimeError(
                        f"Fig. 7 point ({pi}, {ti}) has a malformed payload hash."
                    )
                record["worker_payload_sha256"] = payload_hash
            records.append(record)
    return records


def _write_json_durable(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write canonical finite JSON with an fsync before replace."""
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with _atomic_text_file(path) as stream:
        stream.write(serialized)
        stream.write("\n")


def _artifact_record(path: Path) -> ArtifactRecord:
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError(f"Fig. 7 artifact at {path} is empty.")
    return ArtifactRecord(sha256=file_sha256(path), size_bytes=size)


def _restore_artifact(path: Path, previous: bytes | None) -> None:
    """Best-effort atomic restoration used after a promotion exception."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.rollback.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_PROMOTION_ATTESTATION_KEYS = {
    "artifacts",
    "campaign",
    "certificate_maxima",
    "certificate_scope",
    "created_utc",
    "observable_contract_digest",
    "point_hashes",
    "producer_runtime",
    "runner_path",
    "runner_sha256",
    "schema",
    "solve_contract_digest",
}


def _repo_relative_runner_path(runner_path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    resolved = runner_path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Fig. 7 producer runner {resolved} is outside repository {root}."
        ) from exc


def _current_attested_producer(
    payload: Mapping[str, object],
) -> tuple[Fig7ProducerIdentity, Path]:
    runner_relative = payload.get("runner_path")
    if not isinstance(runner_relative, str) or not runner_relative:
        raise RuntimeError("Fig. 7 promotion attestation has malformed runner_path.")
    root = Path(__file__).resolve().parents[2]
    runner_path = (root / runner_relative).resolve()
    try:
        runner_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "Fig. 7 promotion attestation runner escapes the repository."
        ) from exc
    if not runner_path.is_file():
        raise RuntimeError(
            f"Fig. 7 promotion attestation runner is missing: {runner_path}."
        )
    producer = capture_producer_identity(runner_path)
    expected = {
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Fig. 7 promotion attestation is stale for current {field}."
            )
    return producer, runner_path


def _stable_file_sha256(path: Path, *, label: str) -> str:
    """Hash a file while refusing a concurrently changing byte stream."""
    try:
        before = path.stat()
        first = file_sha256(path)
        middle = path.stat()
        second = file_sha256(path)
        after = path.stat()
    except OSError as exc:
        raise RuntimeError(f"Cannot authenticate Fig. 7 {label} at {path}.") from exc
    signatures = {
        (record.st_size, record.st_mtime_ns)
        for record in (before, middle, after)
    }
    if len(signatures) != 1 or first != second:
        raise RuntimeError(
            f"Fig. 7 {label} at {path} changed while it was authenticated."
        )
    return first


def _validate_parallel_run_evidence(
    attestation: Mapping[str, object],
    *,
    run_dir: Path,
    status_path: Path,
    csv_path: Path,
    pdf_path: Path,
    attestation_path: Path,
) -> None:
    """Independently bind a completed parallel run to its promoted artifacts."""
    resolved_run_dir = run_dir.resolve()
    if status_path.resolve() != (resolved_run_dir / "status.json"):
        raise RuntimeError(
            "Fig. 7 post-run status must be the status.json inside its run directory."
        )
    campaign = attestation.get("campaign")
    if not isinstance(campaign, dict):
        raise RuntimeError("Fig. 7 parallel attestation has malformed campaign data.")
    expected_campaign_fields = {
        "mode",
        "run_identity",
        "wall_s",
        "aggregate_worker_s",
        "resumed_points",
        "new_points",
    }
    if set(campaign) != expected_campaign_fields:
        raise RuntimeError(
            "Fig. 7 parallel attestation has incomplete campaign identity."
        )
    run_identity = campaign.get("run_identity")
    if (
        not isinstance(run_identity, str)
        or len(run_identity) != 64
        or any(char not in "0123456789abcdef" for char in run_identity)
        or resolved_run_dir.name != f"fig7-{run_identity}"
        or campaign.get("mode") != "parallel-independent-points"
    ):
        raise RuntimeError("Fig. 7 parallel attestation has invalid run identity.")
    expected_count = len(P_READ_DBM) * len(T_BATH_VALUES)
    for field in ("resumed_points", "new_points"):
        value = campaign.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"Fig. 7 parallel attestation has invalid {field}."
            )
    if int(campaign["resumed_points"]) + int(campaign["new_points"]) != expected_count:
        raise RuntimeError("Fig. 7 parallel campaign point counts are inconsistent.")
    for field in ("wall_s", "aggregate_worker_s"):
        value = campaign.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise RuntimeError(
                f"Fig. 7 parallel attestation has invalid {field}."
            )

    points = attestation.get("point_hashes")
    if not isinstance(points, list):
        raise RuntimeError("Fig. 7 parallel attestation has malformed point hashes.")
    stamped_worker_hashes: dict[str, str] = {}
    for point in points:
        if not isinstance(point, dict) or "worker_payload_sha256" not in point:
            raise RuntimeError(
                "Fig. 7 parallel attestation must authenticate every worker payload."
            )
        pi = int(point["power_index"])
        ti = int(point["temperature_index"])
        stamped_worker_hashes[f"p{pi:02d}-t{ti:02d}"] = str(
            point["worker_payload_sha256"]
        )

    expected_point_paths = {
        f"p{pi:02d}-t{ti:02d}": resolved_run_dir
        / f"point-p{pi:02d}-t{ti:02d}.npz"
        for pi in range(len(P_READ_DBM))
        for ti in range(len(T_BATH_VALUES))
    }
    actual_point_paths = {path.resolve() for path in resolved_run_dir.glob("*.npz")}
    if actual_point_paths != {
        path.resolve() for path in expected_point_paths.values()
    }:
        raise RuntimeError(
            "Fig. 7 run directory has missing or unexpected worker payloads."
        )
    actual_worker_hashes = {
        key: _stable_file_sha256(path, label=f"worker payload {key}")
        for key, path in expected_point_paths.items()
    }
    if stamped_worker_hashes != actual_worker_hashes:
        raise RuntimeError(
            "Fig. 7 promotion attestation does not authenticate its worker payloads."
        )

    status_digest = _stable_file_sha256(status_path, label="run status")
    try:
        status_bytes = status_path.read_bytes()
        status = json.loads(status_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse Fig. 7 run status at {status_path}.") from exc
    if hashlib.sha256(status_bytes).hexdigest() != status_digest:
        raise RuntimeError("Fig. 7 run status changed while it was parsed.")
    common_status_fields = {
        "schema",
        "state",
        "run_identity",
        "solve_contract_digest",
        "observable_contract_digest",
        "runner_sha256",
        "producer_runtime",
        "single_thread_environment",
        "started_utc",
        "updated_utc",
        "attempt",
        "completed_points",
        "total_points",
        "completion",
    }
    if not isinstance(status, dict) or set(status) != common_status_fields:
        raise RuntimeError("Fig. 7 completed run status has the wrong schema fields.")
    expected_status = {
        "schema": "qpsim-fischer-2023-fig7-run-v1",
        "state": "complete",
        "run_identity": run_identity,
        "solve_contract_digest": attestation["solve_contract_digest"],
        "observable_contract_digest": attestation["observable_contract_digest"],
        "runner_sha256": attestation["runner_sha256"],
        "producer_runtime": attestation["producer_runtime"],
        "single_thread_environment": {
            "BLIS_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "completed_points": expected_count,
        "total_points": expected_count,
    }
    for field, expected in expected_status.items():
        if status.get(field) != expected:
            raise RuntimeError(f"Fig. 7 completed run status has wrong {field}.")
    attempt = status.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeError("Fig. 7 completed run status has invalid attempt.")
    for field in ("started_utc", "updated_utc"):
        try:
            datetime.fromisoformat(str(status[field]))
        except ValueError as exc:
            raise RuntimeError(
                f"Fig. 7 completed run status has invalid {field}."
            ) from exc

    completion = status.get("completion")
    completion_fields = {
        "csv",
        "pdf",
        "attestation",
        "csv_sha256",
        "pdf_sha256",
        "attestation_sha256",
        "wall_s",
        "aggregate_worker_s",
        "certificate_maxima",
        "point_payload_sha256",
    }
    if not isinstance(completion, dict) or set(completion) != completion_fields:
        raise RuntimeError("Fig. 7 completed run has malformed completion evidence.")
    for field, expected_path in (
        ("csv", csv_path),
        ("pdf", pdf_path),
        ("attestation", attestation_path),
    ):
        stamped_path = completion.get(field)
        if not isinstance(stamped_path, str) or Path(stamped_path).resolve() != (
            expected_path.resolve()
        ):
            raise RuntimeError(
                f"Fig. 7 completed run points to the wrong {field} artifact."
            )
    expected_artifact_hashes = {
        "csv_sha256": _stable_file_sha256(csv_path, label="promoted CSV"),
        "pdf_sha256": _stable_file_sha256(pdf_path, label="promoted PDF"),
        "attestation_sha256": _stable_file_sha256(
            attestation_path, label="promotion attestation"
        ),
    }
    for field, expected in expected_artifact_hashes.items():
        if completion.get(field) != expected:
            raise RuntimeError(f"Fig. 7 completed run has wrong {field}.")
    if completion.get("certificate_maxima") != attestation.get("certificate_maxima"):
        raise RuntimeError(
            "Fig. 7 completed run certificate maxima differ from the attestation."
        )
    if completion.get("point_payload_sha256") != actual_worker_hashes:
        raise RuntimeError(
            "Fig. 7 completed run worker hashes differ from the payload bytes."
        )
    for field in ("wall_s", "aggregate_worker_s"):
        value = completion.get(field)
        if value != campaign.get(field):
            raise RuntimeError(
                f"Fig. 7 completed run and attestation disagree on {field}."
            )


def validate_promotion_attestation(
    path: Path,
    *,
    csv_path: Path,
    pdf_path: Path,
    expected_producer: Fig7ProducerIdentity | None = None,
    run_dir: Path | None = None,
    status_path: Path | None = None,
    _publication_lock_held: bool = False,
) -> dict[str, object]:
    """Strictly validate the durable record for a promoted artifact pair."""
    if not _publication_lock_held:
        with _publication_lock(path):
            return validate_promotion_attestation(
                path,
                csv_path=csv_path,
                pdf_path=pdf_path,
                expected_producer=expected_producer,
                run_dir=run_dir,
                status_path=status_path,
                _publication_lock_held=True,
            )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read Fig. 7 promotion attestation at {path}."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != _PROMOTION_ATTESTATION_KEYS:
        raise RuntimeError(
            f"Fig. 7 promotion attestation at {path} has the wrong schema fields."
        )
    if payload["schema"] != "qpsim-fischer-2023-fig7-promotion-v2":
        raise RuntimeError(f"Fig. 7 promotion attestation at {path} is stale.")
    if payload["certificate_scope"] != SUMMARY_CERTIFICATE_SCOPE:
        raise RuntimeError(
            "Fig. 7 promotion attestation has the wrong producer-certificate "
            "scope."
        )
    for field in (
        "solve_contract_digest",
        "observable_contract_digest",
        "runner_sha256",
    ):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise RuntimeError(
                f"Fig. 7 promotion attestation has malformed {field}."
            )
    if not isinstance(payload["producer_runtime"], dict):
        raise RuntimeError("Fig. 7 promotion attestation has malformed runtime.")
    if not isinstance(payload["campaign"], dict):
        raise RuntimeError("Fig. 7 promotion attestation has malformed campaign.")
    try:
        datetime.fromisoformat(str(payload["created_utc"]))
    except ValueError as exc:
        raise RuntimeError(
            "Fig. 7 promotion attestation has malformed creation time."
        ) from exc
    current_producer, _runner_path = _current_attested_producer(payload)
    if expected_producer is not None and current_producer != expected_producer:
        raise RuntimeError(
            "Fig. 7 promotion attestation has the wrong current producer identity."
        )
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"csv", "pdf"}:
        raise RuntimeError(f"Fig. 7 promotion attestation at {path} is malformed.")
    actual_records = {
        "csv": _artifact_record(csv_path),
        "pdf": complete_pdf_record(pdf_path),
    }
    for name, actual in actual_records.items():
        stamped = artifacts[name]
        if not isinstance(stamped, dict) or set(stamped) != {
            "name",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError(
                f"Fig. 7 promotion attestation at {path} has malformed {name}."
            )
        if stamped != {
            "name": (csv_path.name if name == "csv" else pdf_path.name),
            "sha256": actual.sha256,
            "size_bytes": actual.size_bytes,
        }:
            raise RuntimeError(
                f"Fig. 7 promotion attestation at {path} does not "
                f"authenticate {name}."
            )
    maxima = payload["certificate_maxima"]
    if (
        not isinstance(maxima, dict)
        or set(maxima) != set(certificate_module.NUMBER_CERTIFICATE_FIELDS)
    ):
        raise RuntimeError("Fig. 7 promotion attestation has malformed maxima.")
    for field, value in maxima.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise RuntimeError(
                f"Fig. 7 promotion attestation has invalid maximum {field}."
            )
    restored = read_baseline(
        csv_path,
        companion_pdf_path=pdf_path,
        require_companion_pdf=True,
        accept_producer_certificate_claims=True,
        _validate_canonical_attestation=False,
        _publication_lock_held=_publication_lock_held,
    )
    expected_maxima = {
        field: float(np.max(np.asarray(getattr(restored, field))))
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    if maxima != expected_maxima:
        raise RuntimeError(
            "Fig. 7 promotion attestation maxima do not match its CSV."
        )
    points = payload["point_hashes"]
    expected_count = len(P_READ_DBM) * len(T_BATH_VALUES)
    if not isinstance(points, list) or len(points) != expected_count:
        raise RuntimeError(
            "Fig. 7 promotion attestation has an incomplete point-hash set."
        )
    expected_indices = {
        (pi, ti)
        for pi in range(len(P_READ_DBM))
        for ti in range(len(T_BATH_VALUES))
    }
    actual_indices: set[tuple[int, int]] = set()
    for point in points:
        if not isinstance(point, dict):
            raise RuntimeError("Fig. 7 promotion point hash is malformed.")
        required = {
            "power_index",
            "temperature_index",
            "power_dbm",
            "temperature_K",
            "result_sha256",
        }
        actual_fields = set(point)
        if (
            actual_fields != required
            and actual_fields != required | {"worker_payload_sha256"}
        ):
            raise RuntimeError("Fig. 7 promotion point hash has wrong fields.")
        index = (int(point["power_index"]), int(point["temperature_index"]))
        actual_indices.add(index)
        for hash_field in ("result_sha256", "worker_payload_sha256"):
            if hash_field in point:
                value = point[hash_field]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)
                ):
                    raise RuntimeError(
                        f"Fig. 7 promotion point has malformed {hash_field}."
                    )
    if actual_indices != expected_indices or len(actual_indices) != len(points):
        raise RuntimeError(
            "Fig. 7 promotion attestation has duplicate or missing point hashes."
        )
    has_worker_hashes = all("worker_payload_sha256" in point for point in points)
    if has_worker_hashes:
        worker_hashes = {
            (int(point["power_index"]), int(point["temperature_index"])): str(
                point["worker_payload_sha256"]
            )
            for point in points
        }
    else:
        if any("worker_payload_sha256" in point for point in points):
            raise RuntimeError(
                "Fig. 7 promotion attestation has a partial worker-hash set."
            )
        worker_hashes = None
    if points != _result_point_hashes(restored, worker_hashes):
        raise RuntimeError(
            "Fig. 7 promotion attestation point hashes do not match its CSV."
        )
    if (run_dir is None) != (status_path is None):
        raise RuntimeError(
            "Fig. 7 post-run validation requires both run_dir and status_path."
        )
    if run_dir is not None and status_path is not None:
        _validate_parallel_run_evidence(
            payload,
            run_dir=run_dir,
            status_path=status_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            attestation_path=path,
        )
    return cast(dict[str, object], payload)


def publish_baseline_pair(
    result: Fig7PaperResult,
    *,
    producer: Fig7ProducerIdentity,
    runner_path: Path,
    csv_path: Path | None = None,
    pdf_path: Path | None = None,
    attestation_path: Path | None = None,
    worker_payload_hashes: Mapping[tuple[int, int], str] | None = None,
    campaign: Mapping[str, object] | None = None,
    replace_file: Callable[[Path, Path], object] | None = None,
    _publication_lock_held: bool = False,
) -> tuple[Path, Path, Path]:
    """Crash-safe, rollback-capable publication of one certified Fig. 7 pair.

    The PDF and CSV are completed and cross-validated off-path. The PDF moves
    first and its authenticating CSV moves last; a crash between those steps is
    fail-closed because the old CSV cannot authenticate the new PDF. Ordinary
    exceptions roll all canonical files back to their exact prior bytes. The
    durable attestation is promoted last and binds source/runtime, every point,
    certificate maxima, and both final artifact byte streams.
    """
    if csv_path is None:
        csv_path = baseline_path()
    if pdf_path is None:
        pdf_path = csv_path.with_suffix(".pdf")
    if attestation_path is None:
        attestation_path = promotion_attestation_path(csv_path)
    if not _publication_lock_held:
        with _publication_lock(attestation_path):
            return publish_baseline_pair(
                result,
                producer=producer,
                runner_path=runner_path,
                csv_path=csv_path,
                pdf_path=pdf_path,
                attestation_path=attestation_path,
                worker_payload_hashes=worker_payload_hashes,
                campaign=campaign,
                replace_file=replace_file,
                _publication_lock_held=True,
            )
    _validate_result_for_artifact(result)
    assert_producer_identity_current(producer, runner_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    attestation_path.parent.mkdir(parents=True, exist_ok=True)
    replace = os.replace if replace_file is None else replace_file

    csv_stage: Path | None = None
    pdf_stage: Path | None = None
    attestation_stage: Path | None = None
    canonical_paths = (pdf_path, csv_path, attestation_path)
    previous = {
        path: path.read_bytes() if path.exists() else None
        for path in canonical_paths
    }
    try:
        for destination, suffix in (
            (pdf_path, ".pdf"),
            (csv_path, ".csv"),
            (attestation_path, ".json"),
        ):
            with NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=suffix,
                delete=False,
            ) as stream:
                stage = Path(stream.name)
            if destination == pdf_path:
                pdf_stage = stage
            elif destination == csv_path:
                csv_stage = stage
            else:
                attestation_stage = stage
        assert pdf_stage is not None
        assert csv_stage is not None
        assert attestation_stage is not None

        write_plot(result, pdf_stage)
        pdf_record = complete_pdf_record(pdf_stage)
        write_baseline(
            result,
            csv_stage,
            producer=producer,
            companion_pdf=pdf_record,
        )
        round_trip = read_baseline(
            csv_stage,
            companion_pdf_path=pdf_stage,
            require_companion_pdf=True,
            accept_producer_certificate_claims=True,
        )
        _assert_same_result(result, round_trip)
        metadata = read_baseline_metadata(csv_stage)
        if (
            metadata.solve_contract_digest != producer.solve_contract_digest
            or metadata.observable_contract_digest
            != producer.observable_contract_digest
            or metadata.runner_sha256 != producer.runner_sha256
            or metadata.companion_pdf_sha256 != pdf_record.sha256
            or metadata.companion_pdf_size_bytes != pdf_record.size_bytes
        ):
            raise RuntimeError("Staged Fig. 7 CSV has inconsistent provenance.")

        csv_record = _artifact_record(csv_stage)
        point_hashes = _result_point_hashes(result, worker_payload_hashes)
        attestation: dict[str, object] = {
            "schema": "qpsim-fischer-2023-fig7-promotion-v2",
            "certificate_scope": SUMMARY_CERTIFICATE_SCOPE,
            "created_utc": datetime.now(UTC).isoformat(),
            "solve_contract_digest": producer.solve_contract_digest,
            "observable_contract_digest": producer.observable_contract_digest,
            "runner_path": _repo_relative_runner_path(runner_path),
            "runner_sha256": producer.runner_sha256,
            "producer_runtime": producer.runtime,
            "artifacts": {
                "csv": {
                    "name": csv_path.name,
                    "sha256": csv_record.sha256,
                    "size_bytes": csv_record.size_bytes,
                },
                "pdf": {
                    "name": pdf_path.name,
                    "sha256": pdf_record.sha256,
                    "size_bytes": pdf_record.size_bytes,
                },
            },
            "certificate_maxima": {
                field: float(np.max(np.asarray(getattr(result, field))))
                for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
            },
            "point_hashes": point_hashes,
            "campaign": dict(campaign or {}),
        }
        _write_json_durable(attestation_stage, attestation)
        # The durable writer atomically replaced the originally reserved stage
        # path; parse it before any canonical path changes.
        staged_payload = json.loads(attestation_stage.read_text(encoding="utf-8"))
        if staged_payload != attestation:
            raise RuntimeError("Staged Fig. 7 promotion attestation changed.")
        assert_producer_identity_current(producer, runner_path)

        replace(pdf_stage, pdf_path)
        pdf_stage = None
        promoted_pdf = complete_pdf_record(pdf_path)
        if promoted_pdf != pdf_record:
            raise RuntimeError("Promoted Fig. 7 PDF changed during replacement.")
        replace(csv_stage, csv_path)
        csv_stage = None
        promoted = read_baseline(
            csv_path,
            companion_pdf_path=pdf_path,
            require_companion_pdf=True,
            accept_producer_certificate_claims=True,
            # The new attestation is staged but deliberately not canonical
            # yet. Validate it after its final atomic replacement below.
            _validate_canonical_attestation=False,
            _publication_lock_held=_publication_lock_held,
        )
        _assert_same_result(result, promoted)
        replace(attestation_stage, attestation_path)
        attestation_stage = None
        assert_producer_identity_current(producer, runner_path)
        validate_promotion_attestation(
            attestation_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            expected_producer=producer,
            _publication_lock_held=_publication_lock_held,
        )
        return csv_path, pdf_path, attestation_path
    except BaseException:
        for path in reversed(canonical_paths):
            _restore_artifact(path, previous[path])
        raise
    finally:
        for remaining_stage in (csv_stage, pdf_stage, attestation_stage):
            if remaining_stage is not None:
                remaining_stage.unlink(missing_ok=True)


def generate_baseline() -> tuple[Path, Path]:
    print(
        "Fischer 2023 Fig. 7 paper-topology qpsim Q_i(T_B) regression "
        "(no digitized-data comparison) ..."
    )
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, omega_0={OMEGA_0} micro-eV, "
        f"alpha={ALPHA_KI}, tau_l={TAU_L} ns"
    )
    print(f"  P_read (dBm): {list(P_READ_DBM)}")
    print(f"  T_B: {list(T_BATH_VALUES)}")
    runner_path = Path(__file__).resolve()
    producer = capture_producer_identity(runner_path)
    result = run_cached()
    csv_path, pdf_path, attestation_path = publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner_path,
        campaign={"mode": "generic-cached-generate-baseline"},
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {pdf_path}")
    print(f"  wrote {attestation_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
