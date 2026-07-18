"""Fischer 2023 Fig. 7 paper-facing Q_i(T_B) validation.

Uses the experimental-comparison parameters from Tables II/III:

* Delta_0 = 189 micro-eV, tau_0 = 63 ns, omega_0 = 22 micro-eV
* c_phot = 0.06 Hz, tau_l = 170 ps, alpha = 0.13
* h = Delta_T / 189, E_max = 10 Delta_T
* nbar set by the Table III Tstar,0/Delta values
* plotted quality factor capped by Eq. (65) with Table III Q_i,ext

The quasiparticle kinetic solve is done by the qpsim T3 backend with the
finite-tau_l phonon field.  The intrinsic quasiparticle Q_i is evaluated
with the same leading Fischer Eq. (57) form used by the standalone paper
reproduction:

    Q_i,qp = pi Delta / (omega_0 alpha sigma_1).

Usage:

    python -m validation.fischer_2023.fig7_paper
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import os
import platform
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

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

# Cross-platform regression envelope measured on exact single-thread 48-point
# tight solves: Windows/Python 3.14/NumPy 2.4/SciPy 1.17 versus
# Linux/Python 3.13/NumPy 2.5/SciPy 1.18.  Meaningful loss drift peaked at
# 2.440e-3 relative.  The larger relative tail scatter occurred only at
# T=0.06 K, where the largest absolute loss difference was 1.0941e-19 and
# Q_tot moved by <8e-14 relative.  Runtime provenance reports BLAS variables,
# and CI pins them to one thread; the current CSV header does not serialize
# that runtime field, so the workflow contract remains part of this envelope.
QP_LOSS_REGRESSION_RTOL = 4e-3
QP_LOSS_REGRESSION_ATOL = 2e-19
Q_TOTAL_REGRESSION_RTOL = 1e-4


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


def reproduction_provenance() -> dict[str, object]:
    """Return environment + source provenance for an uncached reproduction."""
    import scipy

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "solve_contract_digest": solve_contract_digest(),
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
        *certificate_module.CERTIFICATE_FIELDS,
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
        for field in certificate_module.CERTIFICATE_FIELDS
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
    *certificate_module.CERTIFICATE_FIELDS,
)
_CERTIFIED_BACKWARD_ERROR_FIELDS = (
    "qp_backward_error",
    "phonon_backward_error",
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            yield stream
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_certificate_arrays(
    result: Fig7PaperResult,
    *,
    context: str,
) -> None:
    """Reject incomplete or failed persisted steady-state certificates."""
    expected_shape = (len(result.p_read_dbm), result.T_bath.size)
    for field in certificate_module.CERTIFICATE_FIELDS:
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


def write_baseline(result: Fig7PaperResult, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    _validate_result_for_artifact(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance = reproduction_provenance()
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(["# Fischer 2023 Fig. 7 paper-facing Q_i(T_B); pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.17g} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} tau_l={TAU_L} tau_0_pb={TAU_0_PB}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow([
            "# certificate_metric_version="
            f"{certificate_module.CERTIFICATE_METRIC_VERSION!r} "
            "target_backward_error_limit="
            f"{TARGET_BACKWARD_ERROR_LIMIT}"
        ])
        writer.writerow([
            "# solve_contract_digest="
            f"{provenance['solve_contract_digest']} "
            f"generator_platform={provenance['platform']!r} "
            f"generator_python={provenance['python']!r} "
            f"generator_numpy={provenance['numpy']!r} "
            f"generator_scipy={provenance['scipy']!r}"
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
                ])
    return path


def read_baseline(path: Path | None = None) -> Fig7PaperResult:
    if path is None:
        path = baseline_path()
    metadata = read_baseline_metadata(path)
    if metadata.certificate_metric_version != certificate_module.CERTIFICATE_METRIC_VERSION:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} uses unsupported certificate metric "
            f"{metadata.certificate_metric_version!r}; expected "
            f"{certificate_module.CERTIFICATE_METRIC_VERSION!r}."
        )
    if metadata.target_backward_error_limit != TARGET_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"Fig. 7 baseline at {path} has certificate target "
            f"{metadata.target_backward_error_limit:.17g}; expected "
            f"{TARGET_BACKWARD_ERROR_LIMIT:.17g}."
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
        for field in certificate_module.CERTIFICATE_FIELDS
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
        for field_index, field in enumerate(certificate_module.CERTIFICATE_FIELDS):
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
    )
    _validate_result_for_artifact(result)
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
    solve_contract_digest: str
    generator_platform: str
    generator_python: str
    generator_numpy: str
    generator_scipy: str


class LegacyArtifactError(RuntimeError):
    """A pre-provenance Fig. 7 pin that must not validate current code."""


def read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve). Raises
    ``RuntimeError`` if any stamped field is missing — an old/malformed header
    should fail loudly rather than silently skip the check.
    """
    if path is None:
        path = baseline_path()
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
        solve_contract_digest=contracts[0],
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
            certificate_module.CERTIFICATE_METRIC_VERSION
        ),
        target_backward_error_limit=TARGET_BACKWARD_ERROR_LIMIT,
        solve_contract_digest=str(provenance["solve_contract_digest"]),
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
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paper Fig. 7 palette per readout power, matching standalone reproduction
    # figures/fig7.py PAPER_COLORS.
    PAPER_COLORS = {
        -100.0: "firebrick", -90.0: "dimgray", -80.0: "black",
        -72.0: "red", -68.0: "green", -64.0: "blue",
    }
    fallback_cmap = matplotlib.colormaps["viridis_r"]

    # Analytical Q_i overlay (paper-repro figures/fig7.py): combine Eq. F1
    # thermal-equilibrium Q with Eq. 62 low-T non-equilibrium plateau in
    # parallel with Q_ext (Eq. 65). Closed-form, no kinetic solve needed.
    from scipy.special import k0 as _k0

    GAMMA0 = 19.3

    def _Qth(T_K: float) -> float:
        T_uev = T_K * KB_UEV_PER_K
        if T_uev <= 0:
            return np.inf
        x = OMEGA_0 / (2.0 * T_uev)
        return np.pi / (4.0 * ALPHA_KI * np.sinh(x) * _k0(x)) * np.exp(DELTA_0 / T_uev)

    def _Q_neq_lowT(p_dbm: float) -> float:
        # Eq. 62: depends only on (Δ, ω0, α, τ_l, τ_0^PB, T*).
        kBTs = (TSTAR_OVER_DELTA[float(p_dbm)] * DELTA_0)
        if TAU_L <= 0:
            return np.inf
        arg = float(np.sqrt(14.0 / 5.0)) * (DELTA_0 / kBTs) ** 3
        if arg > 700.0:
            return np.inf
        return (GAMMA0 * DELTA_0 / (ALPHA_KI * OMEGA_0)) * (TAU_0_PB / TAU_L) \
            * (DELTA_0 / kBTs) ** 1.5 * np.exp(arg)

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
            Q_neq = _Q_neq_lowT(float(p))
            Q_ext = Q_EXT_BY_DBM[float(p)]
            Qa = np.array([
                1.0 / (1.0 / _Qth(T) + 1.0 / max(Q_neq, 1.0)) for T in T_dense
            ])
            Qa = 1.0 / (1.0 / np.maximum(Qa, 1.0) + 1.0 / Q_ext)
            ax.semilogy(T_dense, Qa, color=color, ls="--", lw=1.0)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$Q_{i,\mathrm{tot}}$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(temporary, format=path.suffix.removeprefix("."))
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig. 7 paper-facing Q_i(T_B) ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, omega_0={OMEGA_0} micro-eV, "
        f"alpha={ALPHA_KI}, tau_l={TAU_L} ns"
    )
    print(f"  P_read (dBm): {list(P_READ_DBM)}")
    print(f"  T_B: {list(T_BATH_VALUES)}")
    result = run_cached()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  wrote {csv_path}")
    print(f"  wrote {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
