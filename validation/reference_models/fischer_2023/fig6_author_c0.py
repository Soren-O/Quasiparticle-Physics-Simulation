"""Minimal qpsim-free numerical port of the Fischer--Catelani Fig. 6 solver.

This module contains only the author-equivalent numerical model.  It has no
bundle, provenance, plotting, or :mod:`qpsim` dependency.  Hybrid replacement
stages may inject alternative spectral coefficients, but the residual,
historical Newton matrix, explicit-inverse update, stopping rule, and direct
gap observable remain defined here.

Three authenticated source quirks are deliberate:

* the photon residual omits transitions touching the terminal QP bin while
  the Newton matrix includes them;
* photon off-diagonal matrix entries use the partner occupation;
* the phonon-pair ``D_n N`` diagonal uses ``(first + 1, second - 1)`` with
  zero-filled out-of-range values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class AuthorC0Error(ValueError):
    """The clean-room author-equivalent numerical input is invalid."""


@dataclass(frozen=True)
class AuthorNumericalConstants:
    """Numerical constants copied from the authenticated author source."""

    boltzmann_constant_J_per_K: float = 1.38064852e-23
    electron_charge_C: float = 1.60217662e-19


@dataclass(frozen=True)
class AuthorSolveParameters:
    """Scalar inputs held fixed during one author-equivalent solve."""

    gap_eV: float
    h_eV: float
    temperature_K: float
    T_c_K: float
    tau_0_s: float
    tau_0_pb_s: float
    tau_l_s: float
    photon_bin: int
    n_bar: float
    c_photon_s_inv: float
    delta0_eV: float
    thermal_gap_eV: float
    max_newton_steps: int
    relative_step_threshold: float
    constants: AuthorNumericalConstants


@dataclass(frozen=True)
class SpectralCoefficients:
    """Explicit fixed-grid coefficients supplied to the numerical core."""

    rho: np.ndarray
    K_plus: np.ndarray
    K_minus: np.ndarray
    pair_frequency_offset_bins: int = 0


@dataclass(frozen=True)
class AuthorOperator:
    """Precomputed fixed-grid coefficients for one numerical convention."""

    parameters: AuthorSolveParameters
    E_left_eV: np.ndarray
    coefficients: SpectralCoefficients
    n_thermal: np.ndarray
    a_delta: int
    qp_prefactor_s_inv: float
    phonon_prefactor_per_eV_s: float

    @property
    def rho(self) -> np.ndarray:
        return self.coefficients.rho

    @property
    def K_plus(self) -> np.ndarray:
        return self.coefficients.K_plus

    @property
    def K_minus(self) -> np.ndarray:
        return self.coefficients.K_minus

    @property
    def pair_frequency_offset_bins(self) -> int:
        return self.coefficients.pair_frequency_offset_bins


@dataclass(frozen=True)
class ChannelBalance:
    """Gain/loss evidence plus the source-order author residual.

    On physical states the gain and loss arrays are non-negative.  ``net_s_inv``
    is stored separately instead of recomputed so the Newton path retains the
    authenticated source's floating-point expression order.
    """

    gain_s_inv: np.ndarray
    loss_s_inv: np.ndarray
    net_s_inv: np.ndarray


@dataclass(frozen=True)
class SystemEvaluation:
    """Gain/loss decomposition and optional authenticated Newton matrix."""

    qp_photon: ChannelBalance
    qp_scattering: ChannelBalance
    qp_pair: ChannelBalance
    phonon_scattering: ChannelBalance
    phonon_pair: ChannelBalance
    phonon_escape: ChannelBalance
    residual_s_inv: np.ndarray
    update_matrix_s_inv: np.ndarray | None

    # Compatibility net views used by the existing staged diagnostics.
    @property
    def qp_photon_s_inv(self) -> np.ndarray:
        return self.qp_photon.net_s_inv

    @property
    def qp_scattering_s_inv(self) -> np.ndarray:
        return self.qp_scattering.net_s_inv

    @property
    def qp_pair_s_inv(self) -> np.ndarray:
        return self.qp_pair.net_s_inv

    @property
    def phonon_scattering_s_inv(self) -> np.ndarray:
        return self.phonon_scattering.net_s_inv

    @property
    def phonon_pair_s_inv(self) -> np.ndarray:
        return self.phonon_pair.net_s_inv

    @property
    def phonon_escape_s_inv(self) -> np.ndarray:
        return self.phonon_escape.net_s_inv


@dataclass(frozen=True)
class NewtonSolveResult:
    """One complete author-policy Newton path."""

    state_history: np.ndarray
    newton_deltas: np.ndarray
    relative_qp_steps: np.ndarray
    linear_solve_backward_errors: np.ndarray
    converged: bool
    termination: str
    final_evaluation: SystemEvaluation


def _finite_positive(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AuthorC0Error(f"{label} must be finite and positive.")
    return result


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AuthorC0Error(f"{label} must be finite and non-negative.")
    return result


def _uniform_author_grid(
    E_left_eV: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
) -> np.ndarray:
    raw = np.asarray(E_left_eV)
    if (
        raw.ndim != 1
        or raw.size < 4
        or np.iscomplexobj(raw)
        or not np.issubdtype(raw.dtype, np.floating)
    ):
        raise AuthorC0Error("E_left_eV must be a real floating one-dimensional grid.")
    E = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(E)):
        raise AuthorC0Error("E_left_eV must contain only finite values.")
    expected = gap_eV + h_eV * np.arange(E.size)
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        abs(gap_eV),
        abs(float(E[-1])),
    )
    if not np.allclose(E, expected, rtol=0.0, atol=tolerance):
        raise AuthorC0Error("E_left_eV is not the author grid Delta+i*h.")
    return E


def author_cell_average_bcs_dos(
    E_left_eV: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
) -> np.ndarray:
    """Return the author's exact BCS DOS cell average over ``[E_i,E_i+h]``."""

    E = np.asarray(E_left_eV, dtype=float)
    lower = np.sqrt(np.maximum(E**2 - gap_eV**2, 0.0))
    upper = np.sqrt((E + h_eV) ** 2 - gap_eV**2)
    return (upper - lower) / h_eV


def author_bose_occupation(
    omega_eV: np.ndarray,
    temperature_K: float,
    constants: AuthorNumericalConstants,
) -> np.ndarray:
    """Return the Bose occupation using the author's numerical constants."""

    exponent = (
        np.asarray(omega_eV, dtype=float)
        * constants.electron_charge_C
        / (constants.boltzmann_constant_J_per_K * temperature_K)
    )
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return 1.0 / np.expm1(exponent)


def author_fermi_occupation(
    E_left_eV: np.ndarray,
    temperature_K: float,
    constants: AuthorNumericalConstants,
) -> np.ndarray:
    """Return the author's left-edge thermal Fermi occupation."""

    exponent = (
        np.asarray(E_left_eV, dtype=float)
        * constants.electron_charge_C
        / (constants.boltzmann_constant_J_per_K * temperature_K)
    )
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return 1.0 / (np.exp(exponent) + 1.0)


def author_direct_gap_integral(
    f: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
) -> float:
    """Transcribe the attachment's left-edge ``delta_not_consistent`` integral."""

    occupation = np.asarray(f, dtype=float)
    if (
        occupation.ndim != 1
        or occupation.size < 2
        or np.any(~np.isfinite(occupation))
        or np.any((occupation < 0.0) | (occupation > 1.0))
    ):
        raise AuthorC0Error("f must be a finite physical one-dimensional occupation.")
    gap = _finite_positive(gap_eV, "gap_eV")
    h = _finite_positive(h_eV, "h_eV")
    indices = np.arange(occupation.size, dtype=float)
    a_lo = np.arcsinh(np.sqrt(indices * h / (2.0 * gap)))
    a_hi = np.arcsinh(np.sqrt((indices + 1.0) * h / (2.0 * gap)))
    constant = float(np.sum(4.0 * occupation * (a_hi - a_lo)))

    left = h * np.arange(occupation.size - 1, dtype=float)
    da = a_hi[:-1] - a_lo[:-1]
    linear_weight = (
        np.sqrt((left + h) * (left + h + 2.0 * gap))
        - np.sqrt(left * (left + 2.0 * gap))
        - 2.0 * (left + gap) * da
    )
    linear = float(
        np.sum(2.0 * np.diff(occupation) / h * linear_weight)
    )
    return constant + linear


def author_direct_gap(
    f: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
    delta0_eV: float,
) -> float:
    """Return ``Delta_0 exp(-I[f])`` under exact author sampling."""

    delta0 = _finite_positive(delta0_eV, "delta0_eV")
    integral = author_direct_gap_integral(f, gap_eV=gap_eV, h_eV=h_eV)
    return delta0 * float(np.exp(-integral))


def build_author_coefficients(
    E_left_eV: np.ndarray,
    parameters: AuthorSolveParameters,
) -> SpectralCoefficients:
    """Build the clean-room author left-edge spectral coefficients."""

    E = _uniform_author_grid(
        E_left_eV,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    rho = author_cell_average_bcs_dos(
        E,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    product = parameters.gap_eV**2 / (E[:, None] * E[None, :])
    return SpectralCoefficients(
        rho=rho,
        K_plus=1.0 + product,
        K_minus=np.maximum(1.0 - product, 0.0),
        pair_frequency_offset_bins=0,
    )


def build_author_operator(
    E_left_eV: np.ndarray,
    parameters: AuthorSolveParameters,
    *,
    coefficients: SpectralCoefficients | None = None,
) -> AuthorOperator:
    """Build one fixed-grid operator, optionally with injected coefficients."""

    gap = _finite_positive(parameters.gap_eV, "gap_eV")
    h = _finite_positive(parameters.h_eV, "h_eV")
    _finite_positive(parameters.temperature_K, "temperature_K")
    _finite_positive(parameters.T_c_K, "T_c_K")
    _finite_positive(parameters.tau_0_s, "tau_0_s")
    _finite_positive(parameters.tau_0_pb_s, "tau_0_pb_s")
    _finite_positive(parameters.tau_l_s, "tau_l_s")
    _finite_nonnegative(parameters.n_bar, "n_bar")
    _finite_nonnegative(parameters.c_photon_s_inv, "c_photon_s_inv")
    _finite_positive(parameters.delta0_eV, "delta0_eV")
    _finite_positive(parameters.thermal_gap_eV, "thermal_gap_eV")
    _finite_positive(parameters.relative_step_threshold, "relative_step_threshold")
    if (
        isinstance(parameters.max_newton_steps, bool)
        or not isinstance(parameters.max_newton_steps, (int, np.integer))
        or parameters.max_newton_steps < 1
    ):
        raise AuthorC0Error("max_newton_steps must be a positive integer.")

    E = _uniform_author_grid(E_left_eV, gap_eV=gap, h_eV=h)
    size = E.size
    if (
        isinstance(parameters.photon_bin, bool)
        or not isinstance(parameters.photon_bin, (int, np.integer))
        or parameters.photon_bin < 1
        or 2 * parameters.photon_bin >= size - 1
    ):
        raise AuthorC0Error(
            "The authenticated author photon loop requires 2*photon_bin < N-1."
        )
    a_delta = round(gap / h)
    if not math.isclose(
        a_delta * h,
        gap,
        rel_tol=0.0,
        abs_tol=1e-14 * gap,
    ):
        raise AuthorC0Error("The author gap is not an integer multiple of h.")

    selected = build_author_coefficients(E, parameters) if coefficients is None else coefficients
    rho = np.asarray(selected.rho, dtype=float)
    K_plus = np.asarray(selected.K_plus, dtype=float)
    K_minus = np.asarray(selected.K_minus, dtype=float)
    if (
        rho.shape != (size,)
        or K_plus.shape != (size, size)
        or K_minus.shape != (size, size)
        or np.any(~np.isfinite(rho))
        or np.any(~np.isfinite(K_plus))
        or np.any(~np.isfinite(K_minus))
        or np.any(rho < 0.0)
        or np.any(K_plus < 0.0)
        or np.any(K_minus < 0.0)
    ):
        raise AuthorC0Error("Injected spectral coefficients are malformed.")
    if selected.pair_frequency_offset_bins not in {0, 1}:
        raise AuthorC0Error("Pair-frequency offset must be zero or one bin.")
    selected = SpectralCoefficients(
        rho=rho.copy(),
        K_plus=K_plus.copy(),
        K_minus=K_minus.copy(),
        pair_frequency_offset_bins=selected.pair_frequency_offset_bins,
    )

    k_B = parameters.constants.boltzmann_constant_J_per_K
    electron_charge = parameters.constants.electron_charge_C
    qp_prefactor = (
        h**3
        * electron_charge**3
        / (parameters.tau_0_s * (k_B * parameters.T_c_K) ** 3)
    )
    phonon_prefactor = 1.0 / (np.pi * gap * parameters.tau_0_pb_s)
    omega = h * np.arange(1, size)
    return AuthorOperator(
        parameters=parameters,
        E_left_eV=E.copy(),
        coefficients=selected,
        n_thermal=author_bose_occupation(
            omega,
            parameters.temperature_K,
            parameters.constants,
        ),
        a_delta=a_delta,
        qp_prefactor_s_inv=qp_prefactor,
        phonon_prefactor_per_eV_s=phonon_prefactor,
    )


def _balance(
    gain: np.ndarray,
    loss: np.ndarray,
    net: np.ndarray,
) -> ChannelBalance:
    return ChannelBalance(
        gain_s_inv=np.asarray(gain, dtype=float),
        loss_s_inv=np.asarray(loss, dtype=float),
        net_s_inv=np.asarray(net, dtype=float),
    )


def evaluate_author_system(
    operator: AuthorOperator,
    f: np.ndarray,
    n_phonon: np.ndarray,
    *,
    build_update_matrix: bool,
) -> SystemEvaluation:
    """Evaluate all author residual channels and optional historical matrix."""

    f_value = np.asarray(f, dtype=float)
    n_value = np.asarray(n_phonon, dtype=float)
    size = operator.E_left_eV.size
    phonon_size = size - 1
    if f_value.shape != (size,) or n_value.shape != (phonon_size,):
        raise AuthorC0Error("State vector shapes do not match the operator.")
    if np.any(~np.isfinite(f_value)) or np.any(~np.isfinite(n_value)):
        raise AuthorC0Error("Newton state contains non-finite values.")

    params = operator.parameters
    rho = operator.rho
    K_plus = operator.K_plus
    K_minus = operator.K_minus
    photon_step = params.photon_bin
    n_bar = params.n_bar
    c_photon = params.c_photon_s_inv
    offset = operator.pair_frequency_offset_bins

    qp_photon_gain = np.zeros(size)
    qp_photon_loss = np.zeros(size)
    qp_photon_net = np.zeros(size)
    # Exact residual endpoint policy: no transition touching QP bin N-1.
    upper_i = np.arange(0, size - 1 - photon_step)
    upper_j = upper_i + photon_step
    U_upper = K_plus[upper_i, upper_j] * rho[upper_j]
    qp_photon_gain[upper_i] += (
        U_upper
        * f_value[upper_j]
        * (1.0 - f_value[upper_i])
        * (1.0 + n_bar)
    )
    qp_photon_loss[upper_i] += (
        U_upper
        * f_value[upper_i]
        * (1.0 - f_value[upper_j])
        * n_bar
    )
    qp_photon_net[upper_i] += U_upper * (
        f_value[upper_j] * (1.0 - f_value[upper_i]) * (1.0 + n_bar)
        - f_value[upper_i] * (1.0 - f_value[upper_j]) * n_bar
    )
    lower_i = np.arange(photon_step, size - 1)
    lower_j = lower_i - photon_step
    U_lower = K_plus[lower_i, lower_j] * rho[lower_j]
    qp_photon_gain[lower_i] += (
        U_lower
        * f_value[lower_j]
        * (1.0 - f_value[lower_i])
        * n_bar
    )
    qp_photon_loss[lower_i] += (
        U_lower
        * f_value[lower_i]
        * (1.0 - f_value[lower_j])
        * (1.0 + n_bar)
    )
    qp_photon_net[lower_i] += U_lower * (
        f_value[lower_j] * (1.0 - f_value[lower_i]) * n_bar
        - f_value[lower_i]
        * (1.0 - f_value[lower_j])
        * (1.0 + n_bar)
    )
    qp_photon_gain *= c_photon
    qp_photon_loss *= c_photon
    qp_photon_net *= c_photon

    qp_scattering_gain = np.zeros(size)
    qp_scattering_loss = np.zeros(size)
    qp_scattering_net = np.zeros(size)
    qp_pair_gain = np.zeros(size)
    qp_pair_loss = np.zeros(size)
    qp_pair_net = np.zeros(size)
    phonon_scattering_gain = np.zeros(phonon_size)
    phonon_scattering_loss = np.zeros(phonon_size)
    phonon_scattering_net = np.zeros(phonon_size)
    phonon_pair_gain = np.zeros(phonon_size)
    phonon_pair_loss = np.zeros(phonon_size)
    phonon_pair_net = np.zeros(phonon_size)
    phonon_escape_gain = operator.n_thermal / params.tau_l_s
    phonon_escape_loss = n_value / params.tau_l_s
    phonon_escape_net = (operator.n_thermal - n_value) / params.tau_l_s

    state_size = 2 * size - 1
    update_matrix = (
        np.zeros((state_size, state_size)) if build_update_matrix else None
    )

    if update_matrix is not None:
        jac_upper_i = np.arange(0, size - photon_step)
        jac_upper_j = jac_upper_i + photon_step
        jac_U_upper = K_plus[jac_upper_i, jac_upper_j] * rho[jac_upper_j]
        update_matrix[jac_upper_i, jac_upper_i] += (
            -c_photon * jac_U_upper * (n_bar + f_value[jac_upper_j])
        )
        update_matrix[jac_upper_i, jac_upper_j] += (
            c_photon
            * jac_U_upper
            * (1.0 + n_bar - f_value[jac_upper_j])
        )
        jac_lower_i = np.arange(photon_step, size)
        jac_lower_j = jac_lower_i - photon_step
        jac_U_lower = K_plus[jac_lower_i, jac_lower_j] * rho[jac_lower_j]
        update_matrix[jac_lower_i, jac_lower_i] += (
            c_photon
            * jac_U_lower
            * (f_value[jac_lower_j] - (1.0 + n_bar))
        )
        update_matrix[jac_lower_i, jac_lower_j] += (
            c_photon * jac_U_lower * (n_bar + f_value[jac_lower_j])
        )

    for transfer in range(1, size):
        low = np.arange(size - transfer)
        high = low + transfer
        n_transfer = n_value[transfer - 1]
        K_diag = K_minus[low, high]
        transfer_squared = float(transfer * transfer)
        qp_weight_low = (
            operator.qp_prefactor_s_inv
            * transfer_squared
            * K_diag
            * rho[high]
        )
        qp_weight_high = (
            operator.qp_prefactor_s_inv
            * transfer_squared
            * K_diag
            * rho[low]
        )
        qp_scattering_gain[low] += (
            qp_weight_low
            * f_value[high]
            * (1.0 + n_transfer - f_value[low])
        )
        qp_scattering_loss[low] += qp_weight_low * n_transfer * f_value[low]
        qp_scattering_gain[high] += (
            qp_weight_high * f_value[low] * n_transfer
        )
        qp_scattering_loss[high] += (
            qp_weight_high
            * (1.0 - f_value[low] + n_transfer)
            * f_value[high]
        )
        qp_scattering_net[low] += qp_weight_low * (
            -n_transfer * f_value[low]
            + f_value[high] * (1.0 + n_transfer - f_value[low])
        )
        qp_scattering_net[high] += qp_weight_high * (
            f_value[low] * n_transfer
            - (1.0 - f_value[low] + n_transfer) * f_value[high]
        )

        phonon_weight = (
            2.0
            * operator.phonon_prefactor_per_eV_s
            * params.h_eV
            * rho[low]
            * rho[high]
            * K_diag
        )
        phonon_scattering_gain[transfer - 1] = np.sum(
            phonon_weight
            * f_value[high]
            * (1.0 - f_value[low])
            * (1.0 + n_transfer)
        )
        phonon_scattering_loss[transfer - 1] = np.sum(
            phonon_weight
            * f_value[low]
            * (1.0 - f_value[high])
            * n_transfer
        )
        phonon_scattering_net[transfer - 1] = np.sum(
            phonon_weight
            * (
                f_value[high]
                * (1.0 - f_value[low])
                * (1.0 + n_transfer)
                - f_value[low]
                * (1.0 - f_value[high])
                * n_transfer
            )
        )

        if update_matrix is not None:
            phonon_column = size + transfer - 1
            update_matrix[low, low] += (
                -qp_weight_low * (n_transfer + f_value[high])
            )
            update_matrix[low, high] += (
                qp_weight_low * (1.0 + n_transfer - f_value[low])
            )
            update_matrix[low, phonon_column] += (
                qp_weight_low * (f_value[high] - f_value[low])
            )
            update_matrix[high, high] += (
                qp_weight_high * (f_value[low] - (1.0 + n_transfer))
            )
            update_matrix[high, low] += (
                qp_weight_high * (n_transfer + f_value[high])
            )
            update_matrix[high, phonon_column] += (
                qp_weight_high * (f_value[low] - f_value[high])
            )
            phonon_row = size + transfer - 1
            update_matrix[phonon_row, low] += (
                -phonon_weight * (n_transfer + f_value[high])
            )
            update_matrix[phonon_row, high] += (
                phonon_weight * (1.0 + n_transfer - f_value[low])
            )
            update_matrix[phonon_row, phonon_column] += np.sum(
                phonon_weight * (f_value[high] - f_value[low])
            )

    partner = np.arange(size)
    for i in range(size):
        levels = i + partner + 2 * operator.a_delta + offset
        represented = levels < size
        n_pair = np.zeros(size)
        n_pair[represented] = n_value[levels[represented] - 1]
        weight = (
            operator.qp_prefactor_s_inv
            * levels.astype(float) ** 2
            * K_plus[i]
            * rho
        )
        qp_pair_gain[i] = np.sum(
            weight * (1.0 - f_value[i]) * (1.0 - f_value) * n_pair
        )
        qp_pair_loss[i] = np.sum(
            weight * f_value[i] * f_value * (1.0 + n_pair)
        )
        qp_pair_net[i] = np.sum(
            weight
            * (
                (1.0 - f_value[i] - f_value) * n_pair
                - f_value[i] * f_value
            )
        )
        if update_matrix is not None:
            update_matrix[i, :size] += -weight * (n_pair + f_value[i])
            update_matrix[i, i] += np.sum(-weight * (n_pair + f_value))
            represented_levels = levels[represented]
            update_matrix[i, size + represented_levels - 1] += (
                weight[represented]
                * (1.0 - f_value[i] - f_value[represented])
            )

    for level in range(1, size):
        pair_sum = level - 2 * operator.a_delta - offset
        if pair_sum < 0:
            continue
        first = np.arange(pair_sum + 1)
        second = pair_sum - first
        n_level = n_value[level - 1]
        pair_weight = (
            operator.phonon_prefactor_per_eV_s
            * params.h_eV
            * rho[first]
            * rho[second]
            * K_plus[first, second]
        )
        phonon_pair_gain[level - 1] = np.sum(
            pair_weight
            * (1.0 + n_level)
            * f_value[first]
            * f_value[second]
        )
        phonon_pair_loss[level - 1] = np.sum(
            pair_weight
            * (1.0 - f_value[first])
            * (1.0 - f_value[second])
            * n_level
        )
        phonon_pair_net[level - 1] = np.sum(
            pair_weight
            * (
                (1.0 + n_level) * f_value[first] * f_value[second]
                - (1.0 - f_value[first])
                * (1.0 - f_value[second])
                * n_level
            )
        )
        if update_matrix is not None:
            phonon_row = size + level - 1
            np.add.at(
                update_matrix[phonon_row, :size],
                first,
                pair_weight * (f_value[second] + n_level),
            )
            np.add.at(
                update_matrix[phonon_row, :size],
                second,
                pair_weight * (f_value[first] + n_level),
            )
            author_first = np.zeros(first.shape, dtype=float)
            author_first_index = first + 1
            author_first_valid = author_first_index < size
            author_first[author_first_valid] = f_value[
                author_first_index[author_first_valid]
            ]
            author_second = np.zeros(second.shape, dtype=float)
            author_second_index = second - 1
            author_second_valid = author_second_index >= 0
            author_second[author_second_valid] = f_value[
                author_second_index[author_second_valid]
            ]
            update_matrix[phonon_row, phonon_row] += np.sum(
                pair_weight * (author_first + author_second - 1.0)
            )

    if update_matrix is not None:
        phonon_diagonal = size + np.arange(phonon_size)
        update_matrix[phonon_diagonal, phonon_diagonal] -= 1.0 / params.tau_l_s

    balances = (
        _balance(qp_photon_gain, qp_photon_loss, qp_photon_net),
        _balance(qp_scattering_gain, qp_scattering_loss, qp_scattering_net),
        _balance(qp_pair_gain, qp_pair_loss, qp_pair_net),
        _balance(
            phonon_scattering_gain,
            phonon_scattering_loss,
            phonon_scattering_net,
        ),
        _balance(phonon_pair_gain, phonon_pair_loss, phonon_pair_net),
        _balance(phonon_escape_gain, phonon_escape_loss, phonon_escape_net),
    )
    qp_total = balances[0].net_s_inv + balances[1].net_s_inv + balances[2].net_s_inv
    phonon_total = (
        balances[3].net_s_inv + balances[4].net_s_inv + balances[5].net_s_inv
    )
    return SystemEvaluation(
        qp_photon=balances[0],
        qp_scattering=balances[1],
        qp_pair=balances[2],
        phonon_scattering=balances[3],
        phonon_pair=balances[4],
        phonon_escape=balances[5],
        residual_s_inv=np.concatenate((qp_total, phonon_total)),
        update_matrix_s_inv=update_matrix,
    )


def author_newton_delta(evaluation: SystemEvaluation) -> np.ndarray:
    """Apply the authenticated explicit-inverse Newton transition."""

    matrix = evaluation.update_matrix_s_inv
    if matrix is None:
        raise AuthorC0Error("The Newton update matrix was not built.")
    return -np.matmul(np.linalg.inv(matrix), evaluation.residual_s_inv)


def solve_author_system(
    operator: AuthorOperator,
    initial_state: np.ndarray,
) -> NewtonSolveResult:
    """Run the author's explicit-inverse, undamped Newton policy."""

    size = operator.E_left_eV.size
    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (2 * size - 1,) or np.any(~np.isfinite(state)):
        raise AuthorC0Error("Initial Newton state has an invalid shape or value.")
    history = [state.copy()]
    deltas: list[np.ndarray] = []
    relative_steps: list[float] = []
    linear_errors: list[float] = []
    converged = False
    termination = "maximum_steps_reached"

    for _ in range(operator.parameters.max_newton_steps):
        evaluation = evaluate_author_system(
            operator,
            state[:size],
            state[size:],
            build_update_matrix=True,
        )
        matrix = evaluation.update_matrix_s_inv
        if matrix is None:  # pragma: no cover - guaranteed above.
            raise AssertionError("Newton update matrix was not built.")
        delta = author_newton_delta(evaluation)
        deltas.append(delta.copy())
        denominator = (
            float(np.linalg.norm(matrix, ord=np.inf))
            * float(np.linalg.norm(delta, ord=np.inf))
            + float(np.linalg.norm(evaluation.residual_s_inv, ord=np.inf))
        )
        linear_error = float(
            np.linalg.norm(
                matrix @ delta + evaluation.residual_s_inv,
                ord=np.inf,
            )
        )
        linear_errors.append(
            linear_error / denominator if denominator > 0.0 else 0.0
        )
        next_state = state + delta
        if np.any(~np.isfinite(next_state)):
            raise AuthorC0Error("The author Newton policy produced a non-finite state.")
        qp_denominator = float(np.linalg.norm(next_state[:size]))
        relative_step = float(
            np.linalg.norm(next_state[:size] - state[:size]) / qp_denominator
        )
        relative_steps.append(relative_step)
        state = next_state
        history.append(state.copy())
        if relative_step < operator.parameters.relative_step_threshold:
            converged = True
            termination = "relative_qp_step_threshold"
            break

    final = evaluate_author_system(
        operator,
        state[:size],
        state[size:],
        build_update_matrix=False,
    )
    state_size = 2 * size - 1
    return NewtonSolveResult(
        state_history=np.stack(history),
        newton_deltas=(
            np.stack(deltas)
            if deltas
            else np.empty((0, state_size), dtype=state.dtype)
        ),
        relative_qp_steps=np.asarray(relative_steps),
        linear_solve_backward_errors=np.asarray(linear_errors),
        converged=converged,
        termination=termination,
        final_evaluation=final,
    )


def qp_number_diagnostics(
    evaluation: SystemEvaluation,
    operator: AuthorOperator,
) -> dict[str, float]:
    """Return DOS-weighted QP-number residual diagnostics."""

    weights = operator.parameters.h_eV * operator.rho
    channel_nets = {
        "photon": evaluation.qp_photon_s_inv,
        "scattering": evaluation.qp_scattering_s_inv,
        "pair": evaluation.qp_pair_s_inv,
    }
    result = {
        f"{name}_weighted_number_rate_eV_s_inv": float(weights @ value)
        for name, value in channel_nets.items()
    }
    result["total_weighted_number_rate_eV_s_inv"] = float(
        weights @ evaluation.residual_s_inv[: weights.size]
    )
    return result
