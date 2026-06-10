"""Ph0 local-phonon tier: steady-state ``n_ph(ω)`` from QP distribution.

At phonon steady state with a local bath of escape time ``τ_l``, the
Ph0 equation is

    0 = a_ph[f] + b_ph[f] · n_ph + (n_th − n_ph) / τ_l,

where ``(a_ph, b_ph)`` are the affine coefficients of the e-ph
source-sink on the QP distribution (see
:func:`qpsim.collisions.phonon.compute_phonon_source_sink`). Solving:

* ``τ_l = 0`` — **sentinel for no substrate coupling**: the escape term
  is dropped entirely and ``n_ph = −a_ph / b_ph``. Physically this is
  the ``τ_l → ∞`` limit of the escape term, *not* the instantaneous-
  escape limit the literal value suggests.
* ``τ_l > 0``: ``n_ph = (a_ph + n_th / τ_l) / (1/τ_l − b_ph)``.

Naming trap (audited 2026-06-10): the *opposite* limit — phonons pinned
at the bath, Fischer's instantaneous-thermalization ``τ_l → 0`` — is
spelled ``phonon_escape_time=None`` in
:func:`qpsim.services.steady_state.solve_steady_state` and
``use_thermal_phonons=True`` on the backend; it never reaches this
module. Passing the float ``0.0`` here therefore means the opposite of
passing ``None`` upstream.

Moved here from the private ``_phonon_steady_state`` helper that
previously lived in :mod:`qpsim.services.steady_state` (Gate 2
task 11).
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import compute_phonon_source_sink
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

_SINGULAR_TOL = 1e-30
_NEGATIVE_TOL = 1e-12


def _solve_affine_balance(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Solve ``denominator * n = numerator`` and reject unphysical states."""
    singular = np.abs(denominator) <= _SINGULAR_TOL
    inconsistent = singular & (np.abs(numerator) > _SINGULAR_TOL)
    if np.any(inconsistent):
        bad = np.flatnonzero(inconsistent)[:5].tolist()
        raise RuntimeError(
            f"Ph0 phonon steady state has no finite solution in {label}; "
            f"singular balance at omega indices {bad}."
        )

    n_ph = np.zeros_like(numerator, dtype=float)
    regular = ~singular
    n_ph[regular] = numerator[regular] / denominator[regular]

    invalid = (~np.isfinite(n_ph)) | (n_ph < -_NEGATIVE_TOL)
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:5].tolist()
        raise RuntimeError(
            f"Ph0 phonon steady state is unphysical in {label}; "
            f"computed negative or non-finite occupation at omega indices {bad}. "
            "This indicates phonon runaway or no non-negative fixed point."
        )

    n_ph[n_ph < 0.0] = 0.0
    return n_ph


def phonon_steady_state(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_bins: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    T_bath: float,
    tau_l: float,
    *,
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the Ph0 phonon-balance equation for ``n_ph(ω)`` given ``f``.

    Parameters
    ----------
    f
        QP occupation, shape ``(NE,)``.
    ctx
        SpectralContext with current Δ.
    K_s0, K_r0
        Base e-ph kernels (shape ``(NE, NE)``) or ``None`` to disable
        the corresponding channel.
    omega_bins, omega_idx_diff, omega_idx_sum, diff_sign
        Outputs of :func:`qpsim.collisions.phonon.build_phonon_frequency_map`
        for ``ctx.E``.
    T_bath
        Substrate bath temperature (K), used to compute the thermal
        ``n_th(ω)``.
    tau_l
        Phonon bath-escape time (ns). ``0.0`` is the **no-substrate
        sentinel** (escape term dropped — the ``τ_l → ∞`` physics
        limit); ``> 0`` means finite escape time. For bath-pinned
        phonons (the physical ``τ_l → 0`` limit) use the upstream
        thermal-phonon path instead of this solver (see module
        docstring).
    K_s0_phonon_side, K_r0_phonon_side
        **Opt-in** phonon-side scattering and recombination/pair-breaking
        kernels ``2K⁻/(π Δ τ_0^PB)`` and ``K⁺/(π Δ τ_0^PB)`` (build via
        :func:`qpsim.collisions.phonon.build_scattering_kernel_phonon_side` and
        :func:`qpsim.collisions.phonon.build_recombination_kernel_phonon_side`).
        Forwarded to :func:`compute_phonon_source_sink`; when supplied,
        the phonon-equation rates use the F&C 2023 Eq. 12 prefactors
        instead of the QP-side ``K_s0`` / ``K_r0``. ``None`` (default)
        preserves legacy behavior bit-for-bit.

    Returns
    -------
    n_ph
        1D array of shape ``(len(omega_bins),)``, clipped to ``≥ 0``.
    """
    n_omega = len(omega_bins)
    a_ph, b_ph = compute_phonon_source_sink(
        f, ctx, K_s0, K_r0,
        omega_idx_diff, omega_idx_sum, diff_sign, n_omega,
        K_s0_phonon_side=K_s0_phonon_side,
        K_r0_phonon_side=K_r0_phonon_side,
    )

    if tau_l < 0.0:
        raise ValueError("tau_l must be non-negative.")

    if tau_l == 0.0:
        n_ph = _solve_affine_balance(
            -a_ph, b_ph, label="tau_l=0 no-bath branch"
        )
    else:
        inv_tau_l = 1.0 / tau_l
        n_th = thermal_phonon_occupation(omega_bins, T_bath)
        denom = inv_tau_l - b_ph
        n_ph = _solve_affine_balance(
            a_ph + inv_tau_l * n_th,
            denom,
            label="finite-tau_l branch",
        )

    return n_ph
