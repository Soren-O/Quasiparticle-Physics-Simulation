"""Electron–phonon collision integral (J_1^eph).

Ported from the old ``qpsim/numerics/collision_phonon.py`` at Gate 2
with the numba acceleration path dropped. Provides:

- ``phonon_collision_rates``: the main ``(gain, loss_rate)`` computation
  for a quasiparticle distribution ``f(E)`` at one spatial pixel.
- ``CoherenceAssignment``, ``build_scattering_kernel_base``,
  ``build_recombination_kernel_base``: coherence-factor wiring (swap
  K⁺ ↔ K⁻ under the photon convention).
- ``build_scattering_kernel_phonon_side`` and
  ``build_recombination_kernel_phonon_side``: opt-in phonon-side
  kernels for the F&C 2023 Eq. 12 paper-faithful prefactors; pass via
  ``compute_phonon_source_sink(K_s0_phonon_side=…,
  K_r0_phonon_side=…)``.
- ``build_phonon_frequency_map``,
  ``phonon_occupation_matrices_from_state``,
  ``compute_phonon_source_sink``: the dynamic-phonon coupling pieces
  used when ``n_ph(ω, t)`` is a live dynamical variable.
- ``apply_phonon_collision``: ETD2 step that combines the above for the
  thermal-bath case. Both steppers (ETD1 and ETD2) live in
  ``qpsim.solvers.etd``; this wrapper uses the second-order form per
  the Build Handoff's committed port-time upgrade.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.physics.kaplan_pair_breaking import kaplan_S_plus
from qpsim.physics.kernels import recombination_kernel_base as _recombination_kernel_base
from qpsim.physics.kernels import scattering_kernel_base as _scattering_kernel_base
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.etd import etd2_step


class CoherenceAssignment(Enum):
    """Selects which coherence factor pairs with scattering vs recombination.

    * ``PHONON`` — standard QP-phonon: scattering uses K⁻, recombination K⁺.
    * ``PHOTON`` — QP-photon (Fischer 2023 sub-gap): swap K⁺ ↔ K⁻.
    """

    PHONON = "phonon"
    PHOTON = "photon"


def build_scattering_kernel_base(
    ctx: SpectralContext,
    tau_0: float,
    T_c: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    """Base scattering kernel K₀ˢ(E_i, E_j) with coherence wiring, shape (NE, NE).

    Thin wrapper around ``qpsim.physics.kernels.scattering_kernel_base`` that
    pulls the precomputed coherence matrix from ``ctx`` and honors the
    photon-convention swap.
    """
    coh = ctx.K_minus if coherence is CoherenceAssignment.PHONON else ctx.K_plus
    return _scattering_kernel_base(ctx.E, ctx.gap, tau_0, T_c, coherence_factor=coh)


def build_scattering_kernel_phonon_side(
    ctx: SpectralContext,
    tau_0_pb_ns: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    r"""Phonon-side scattering kernel ``2 K⁻/(π Δ τ_0^PB)``.

    Used by :func:`compute_phonon_source_sink` for the **phonon
    equation** qp-conserving term (F&C 2023 Eq. 12, first integral).
    This is distinct from :func:`build_scattering_kernel_base`, whose
    ``ω²/(τ₀ T_c³)`` prefactor belongs to the QP equation.
    """
    coh = ctx.K_minus if coherence is CoherenceAssignment.PHONON else ctx.K_plus
    if tau_0_pb_ns <= 0.0:
        raise ValueError(f"tau_0_pb_ns must be positive; got {tau_0_pb_ns}")
    if ctx.gap <= 0.0:
        raise ValueError(f"ctx.gap must be positive; got {ctx.gap}")
    return (2.0 / (np.pi * ctx.gap * tau_0_pb_ns)) * np.asarray(coh, dtype=float)


def build_recombination_kernel_base(
    ctx: SpectralContext,
    tau_0: float,
    T_c: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    """Base recombination kernel K₀ʳ(E_i, E_j), shape (NE, NE)."""
    coh = ctx.K_plus if coherence is CoherenceAssignment.PHONON else ctx.K_minus
    return _recombination_kernel_base(ctx.E, ctx.gap, tau_0, T_c, coherence_factor=coh)


def build_recombination_kernel_phonon_side(
    ctx: SpectralContext,
    tau_0_pb_ns: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    r"""Phonon-side recombination/pair-breaking kernel K⁺/(π Δ τ_0^PB).

    Used by :func:`compute_phonon_source_sink` for the **phonon
    equation** rate (F&C 2023 Eq. 12, second integral). Distinct from
    :func:`build_recombination_kernel_base`, which carries the
    QP-side prefactor ``(E_sum/k_BT_c)²/(τ₀ k_BT_c)`` appropriate for
    the QP collision integrals (Eqs. 10, 11).

    The matching condition derives from the Kaplan zero-T pair-breaking
    rate ``1/τ_PB(ω, 0) = (1/(π Δ₀ τ_0^PB)) ∫ dE ρ(E) ρ(ω−E) K⁺(E, ω−E)``.
    Discretized as ``-b_ph(ω) ≈ Σ_{i,j: E_i+E_j=ω} dE · ρ_i ρ_j K(i,j)``,
    this fixes ``K(E_i, E_j) = K⁺/(π Δ τ_0^PB)`` with Δ = current gap
    (≈ Δ₀ at the temperatures of F&C 2023 Fig 3).

    Parameters
    ----------
    ctx
        SpectralContext with current Δ; supplies ``K_plus``/``K_minus``.
    tau_0_pb_ns
        ``τ_0^PB`` (ns); F&C 2023 Eq. 13 + Table I gives 0.255 ns for
        Al. Distinct from the Kaplan ``τ_0^{ph}`` (Eq. 30) of
        :mod:`qpsim.physics.kaplan_pair_breaking`.
    coherence
        Selects K⁺ (phonon convention, default) or K⁻ (photon swap).

    Returns
    -------
    np.ndarray
        ``(NE, NE)`` kernel with units ``1/(μeV · ns)``. Pair with
        ``compute_phonon_source_sink(..., K_r0_phonon_side=...)`` to
        opt into the paper-faithful phonon-side prefactor; the QP-side
        ``K_r0`` argument is unchanged.
    """
    coh = ctx.K_plus if coherence is CoherenceAssignment.PHONON else ctx.K_minus
    if tau_0_pb_ns <= 0.0:
        raise ValueError(f"tau_0_pb_ns must be positive; got {tau_0_pb_ns}")
    if ctx.gap <= 0.0:
        raise ValueError(f"ctx.gap must be positive; got {ctx.gap}")
    return np.asarray(coh, dtype=float) / (np.pi * ctx.gap * tau_0_pb_ns)


def _thermal_phonon_scattering_occupation(
    E: np.ndarray, T_bath: float,
) -> np.ndarray:
    """(NE, NE) scattering phonon occupation at bath temperature T_bath.

    * Emission (E_i > E_j): ``1 + n_BE(E_i − E_j)``
    * Absorption (E_i < E_j): ``n_BE(E_j − E_i)``
    * Diagonal: 0.
    """
    E_diff = E[:, None] - E[None, :]
    kBT = _KB_UEV_PER_K * T_bath if T_bath > 0 else 0.0
    N_p = np.zeros_like(E_diff)

    if kBT > 0:
        emission = E_diff > 0
        absorption = E_diff < 0
        exp_em = np.minimum(E_diff[emission] / kBT, 500.0)
        exp_abs = np.minimum(-E_diff[absorption] / kBT, 500.0)
        N_p[emission] = 1.0 + 1.0 / (np.exp(exp_em) - 1.0)
        N_p[absorption] = 1.0 / (np.exp(exp_abs) - 1.0)
    else:
        N_p[E_diff > 0] = 1.0
        N_p[E_diff < 0] = 0.0

    return N_p


def _thermal_phonon_recombination_occupations(
    E: np.ndarray, T_bath: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N_emit, N_abs)`` at bath temperature ``T_bath``.

    * ``N_emit = 1 + n_BE(E_i + E_j)``: phonon emission from recombination.
    * ``N_abs  = n_BE(E_i + E_j)``: pair-breaking absorption.
    """
    E_sum = E[:, None] + E[None, :]
    kBT = _KB_UEV_PER_K * T_bath if T_bath > 0 else 0.0

    if kBT > 0:
        exp_sum = np.minimum(E_sum / kBT, 500.0)
        N_BE = 1.0 / (np.exp(exp_sum) - 1.0)
    else:
        N_BE = np.zeros_like(E_sum)

    return 1.0 + N_BE, N_BE


def build_phonon_frequency_map(
    E_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an ω-grid and (i, j) → ω-index maps for QP–phonon coupling.

    For every ``(E_i, E_j)`` pair, identifies the corresponding scattering
    phonon frequency ``|E_i − E_j|`` and the recombination phonon
    frequency ``E_i + E_j``. Returns:

    * ``omega_bins``: unique-sorted 1D ω grid.
    * ``omega_idx_diff``, ``omega_idx_sum``: ``(NE, NE)`` int indices
      into ``omega_bins``.
    * ``diff_sign``: ``(NE, NE)`` int8, ``sign(E_i − E_j)``. Used to
      distinguish emission (+) from absorption (−) in the dynamic-n_ph
      projection.
    """
    E = np.asarray(E_bins, dtype=float)
    if E.ndim != 1:
        raise ValueError("E_bins must be a 1D array.")
    E_diff_abs = np.abs(E[:, None] - E[None, :])
    E_sum = E[:, None] + E[None, :]
    all_vals = np.concatenate([E_diff_abs.ravel(), E_sum.ravel()])
    omega_bins, inverse = np.unique(np.round(all_vals, 12), return_inverse=True)
    n_pairs = E.size * E.size
    omega_idx_diff = inverse[:n_pairs].reshape((E.size, E.size))
    omega_idx_sum = inverse[n_pairs:].reshape((E.size, E.size))
    diff_sign = np.sign(E[:, None] - E[None, :]).astype(np.int8)
    return omega_bins, omega_idx_diff, omega_idx_sum, diff_sign


def phonon_occupation_matrices_from_state(
    n_ph: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a non-equilibrium ``n_ph(ω)`` onto the three (NE, NE) matrices.

    * ``N_p``: scattering occupation (emission/absorption by sign of i − j).
    * ``N_emit``: recombination emission, ``1 + n_ph(E_i + E_j)``.
    * ``N_abs``: pair-breaking absorption, ``n_ph(E_i + E_j)``.
    """
    n_diff = n_ph[omega_idx_diff]
    n_sum = n_ph[omega_idx_sum]

    N_p = np.where(diff_sign > 0, 1.0 + n_diff, n_diff)
    np.fill_diagonal(N_p, 0.0)

    N_emit = 1.0 + n_sum
    N_abs = n_sum.copy()

    return N_p, N_emit, N_abs


def compute_phonon_source_sink(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_omega: int,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine-ODE coefficients ``(a_ph, b_ph)`` for the phonon distribution.

    From the current QP occupation ``f(E)``, produces ``(a_ph, b_ph)`` of
    shape ``(n_omega,)`` such that ``dn_ph/dt = a_ph + b_ph · n_ph``.
    Used by the dynamic-phonon backend when ``n_ph`` evolves
    self-consistently (finite-τ_l regime).

    Parameters
    ----------
    K_s0_phonon_side, K_r0_phonon_side
        **Opt-in** phonon-side scattering and recombination kernels for
        paper-faithful F&C 2023 Eq. 12 prefactors. When supplied, the
        phonon-equation sums use these kernels instead of the QP-side
        ``K_s0`` / ``K_r0`` kernels that carry ``ω²/(τ₀ T_c³)``.
        Build via :func:`build_scattering_kernel_phonon_side`, which
        returns ``2K⁻/(π Δ τ_0^PB)``, and
        :func:`build_recombination_kernel_phonon_side`, which returns
        ``K⁺/(π Δ τ_0^PB)`` per F&C Eq. 12 / Kaplan 1976.

        When ``None`` (default), the legacy behavior is preserved: the
        QP-side ``K_s0`` / ``K_r0`` are used for both QP-equation and
        phonon-equation rates. **The QP-side path is unchanged**;
        those kernels continue to drive the QP collision integral via
        :func:`phonon_collision_rates`. Backend callers opt into this
        path with ``use_phonon_side_kernel=True``.
    """
    rho = ctx.rho
    dE = ctx.dE
    n_qp = rho * f
    one_minus_f = np.maximum(1.0 - f, 0.0)
    partner = rho * one_minus_f

    a_ph = np.zeros(n_omega)
    b_ph = np.zeros(n_omega)

    K_sc = K_s0_phonon_side if K_s0_phonon_side is not None else K_s0
    if enable_scattering and K_sc is not None:
        base_sc = dE * (n_qp[:, None] * K_sc * (rho[None, :] * one_minus_f[None, :]))
        emit_mask = diff_sign > 0
        abs_mask = diff_sign < 0
        if np.any(emit_mask):
            emit = np.bincount(
                omega_idx_diff[emit_mask].ravel(),
                weights=base_sc[emit_mask].ravel(),
                minlength=n_omega,
            )
            a_ph += emit
            b_ph += emit
        if np.any(abs_mask):
            absor = np.bincount(
                omega_idx_diff[abs_mask].ravel(),
                weights=base_sc[abs_mask].ravel(),
                minlength=n_omega,
            )
            b_ph -= absor

    K_rec = K_r0_phonon_side if K_r0_phonon_side is not None else K_r0
    if enable_recombination and K_rec is not None:
        base_rec = dE * (n_qp[:, None] * K_rec * n_qp[None, :])
        rec = np.bincount(
            omega_idx_sum.ravel(),
            weights=base_rec.ravel(),
            minlength=n_omega,
        )
        base_pb = dE * (partner[:, None] * K_rec * partner[None, :])
        pb = np.bincount(
            omega_idx_sum.ravel(),
            weights=base_pb.ravel(),
            minlength=n_omega,
        )
        if K_r0_phonon_side is not None:
            correction = _pair_breaking_quadrature_correction(
                ctx, K_r0_phonon_side, omega_idx_sum, n_omega,
            )
            rec *= correction
            pb *= correction
        a_ph += rec
        b_ph += rec
        b_ph -= pb

    return a_ph, b_ph


def _pair_breaking_quadrature_correction(
    ctx: SpectralContext,
    K_r0_phonon_side: np.ndarray,
    omega_idx_sum: np.ndarray,
    n_omega: int,
) -> np.ndarray:
    """Scale phonon-side pair-breaking bins to the Kaplan S_+ total weight.

    A midpoint sum of the BCS endpoint singularity underestimates the
    pair-breaking sink immediately above ``2Δ`` by ``2/π``.  The standalone
    Fischer reproduction avoids that artifact with the analytic Kaplan
    ``S_+(ω/Δ)`` total.  Apply the same per-ω correction only for callers
    that opted into the phonon-side ``K⁺/(π Δ τ_0^PB)`` kernel; legacy
    QP-side behavior remains unchanged.
    """
    K = np.asarray(K_r0_phonon_side, dtype=float)
    if K.shape != ctx.K_plus.shape:
        return np.ones(n_omega)

    valid = (ctx.K_plus > 0.0) & np.isfinite(ctx.K_plus) & np.isfinite(K)
    if not np.any(valid):
        return np.ones(n_omega)
    prefactor = float(np.median(K[valid] / ctx.K_plus[valid]))
    if prefactor <= 0.0 or not np.isfinite(prefactor):
        return np.ones(n_omega)

    tau_0_pb = 1.0 / (np.pi * ctx.gap * prefactor)
    if tau_0_pb <= 0.0 or not np.isfinite(tau_0_pb):
        return np.ones(n_omega)

    rho = ctx.rho
    dE = ctx.dE
    empty_weights = dE * (rho[:, None] * K * rho[None, :])
    discrete = np.bincount(
        omega_idx_sum.ravel(),
        weights=empty_weights.ravel(),
        minlength=n_omega,
    )

    omega = np.zeros(n_omega)
    np.maximum.at(omega, omega_idx_sum.ravel(), (ctx.E[:, None] + ctx.E[None, :]).ravel())
    exact = np.array(
        [
            kaplan_S_plus(float(w / ctx.gap)) / (np.pi * tau_0_pb)
            if w > 2.0 * ctx.gap
            else 0.0
            for w in omega
        ],
        dtype=float,
    )

    correction = np.ones(n_omega)
    mask = (discrete > 0.0) & (exact > 0.0)
    correction[mask] = exact[mask] / discrete[mask]
    return correction


def phonon_source_sink_jacobian_f(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_omega: int,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Analytical ``(∂a_ph/∂f, ∂b_ph/∂f)`` for :func:`compute_phonon_source_sink`.

    Returns two ``(n_omega, NE)`` arrays — the exact derivatives of the
    affine-ODE coefficients ``(a_ph, b_ph)`` with respect to the QP
    occupation ``f``. Mirrors :func:`compute_phonon_source_sink`
    term-for-term: identical kernel selection (phonon-side ``K_*_phonon_side``
    when supplied, else QP-side ``K_s0``/``K_r0``) and the same f-independent
    :func:`_pair_breaking_quadrature_correction`, so the two stay in lockstep.

    Used to build the analytical ``J_nf = ∂R_ph/∂f`` cross-block of
    :func:`qpsim.solvers.coupled_newton.coupled_newton_solve` via
    ``J_nf = ∂a_ph/∂f + n_ph · ∂b_ph/∂f``.

    Derivation. The scattering weight
    ``base_sc[i,j] = dE ρ_i ρ_j f_i (1−f_j) K_sc[i,j]`` is bilinear in f, so
    ``∂base_sc/∂f_i = dE ρ_i ρ_j (1−f_j) K_sc`` (column i) and
    ``∂base_sc/∂f_j = −dE ρ_i ρ_j f_i K_sc`` (column j). For
    recombination/pair-breaking the ``b_ph`` combination collapses because
    ``f_i f_j − (1−f_i)(1−f_j) = f_i + f_j − 1``, so ``∂b_ph^rec/∂f`` is
    f-independent (``= dE ρ_i ρ_j K_rec`` in both columns i and j), while
    ``∂a_ph^rec/∂f_i = dE ρ_i ρ_j f_j K_rec`` and ``…/∂f_j = … f_i K_rec``.
    """
    NE = int(f.size)
    rho = ctx.rho
    dE = ctx.dE
    one_minus_f = np.maximum(1.0 - f, 0.0)
    pref = dE * (rho[:, None] * rho[None, :])  # dE ρ_i ρ_j, shape (NE, NE)

    da_df = np.zeros((n_omega, NE))
    db_df = np.zeros((n_omega, NE))

    # Per-pair (i, j) contributions deposit into column i or column j of the
    # row given by the pair's ω-bin index.
    col_i = np.broadcast_to(np.arange(NE)[:, None], (NE, NE))
    col_j = np.broadcast_to(np.arange(NE)[None, :], (NE, NE))

    K_sc = K_s0_phonon_side if K_s0_phonon_side is not None else K_s0
    if enable_scattering and K_sc is not None:
        d_i = pref * K_sc * one_minus_f[None, :]   # ∂base_sc/∂f_i, into column i
        d_j = -pref * K_sc * f[:, None]            # ∂base_sc/∂f_j, into column j
        emit = diff_sign > 0   # i > j: +base_sc into a_ph and b_ph
        absb = diff_sign < 0   # i < j: −base_sc into b_ph only
        rows = omega_idx_diff
        np.add.at(da_df, (rows[emit], col_i[emit]), d_i[emit])
        np.add.at(da_df, (rows[emit], col_j[emit]), d_j[emit])
        np.add.at(db_df, (rows[emit], col_i[emit]), d_i[emit])
        np.add.at(db_df, (rows[emit], col_j[emit]), d_j[emit])
        np.add.at(db_df, (rows[absb], col_i[absb]), -d_i[absb])
        np.add.at(db_df, (rows[absb], col_j[absb]), -d_j[absb])

    K_rec = K_r0_phonon_side if K_r0_phonon_side is not None else K_r0
    if enable_recombination and K_rec is not None:
        rows = omega_idx_sum
        base = pref * K_rec  # dE ρ_i ρ_j K_rec[i,j]
        if K_r0_phonon_side is not None:
            corr = _pair_breaking_quadrature_correction(
                ctx, K_r0_phonon_side, omega_idx_sum, n_omega,
            )
            base = base * corr[rows]   # per-ω, f-independent scaling
        # a_ph^rec[w] = Σ base f_i f_j  →  ∂/∂f_i = base f_j, ∂/∂f_j = base f_i.
        np.add.at(da_df, (rows, col_i), base * f[None, :])
        np.add.at(da_df, (rows, col_j), base * f[:, None])
        # b_ph^rec[w] = Σ base (f_i + f_j − 1)  →  ∂/∂f_i = ∂/∂f_j = base.
        np.add.at(db_df, (rows, col_i), base)
        np.add.at(db_df, (rows, col_j), base)

    return da_df, db_df


def phonon_collision_rates(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
    N_p_override: np.ndarray | None = None,
    N_emit_override: np.ndarray | None = None,
    N_abs_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(gain, loss_rate)`` for the e–phonon collision integral.

    Returns arrays of shape ``(NE,)`` such that ``df/dt = gain − loss_rate · f``.
    Works on the occupation ``f`` (not the spectral density ``n = ρ·f``).
    The ``gain`` includes the ``(1 − f_i)`` Pauli factor already.

    When ``N_*_override`` arguments are supplied, they replace the thermal
    Bose-Einstein factors that would otherwise be computed from ``T_bath``.
    Used by the dynamic-phonon backend when ``n_ph`` is evolved
    self-consistently.
    """
    E = ctx.E
    rho = ctx.rho
    dE = ctx.dE
    one_minus_f = np.maximum(1.0 - f, 0.0)

    gain = np.zeros_like(f)
    loss_rate = np.zeros_like(f)

    if enable_scattering and K_s0 is not None:
        N_p = (
            N_p_override
            if N_p_override is not None
            else _thermal_phonon_scattering_occupation(E, T_bath)
        )
        K_s_eff = K_s0 * N_p
        n_qp = rho * f
        gain += one_minus_f * (K_s_eff.T @ (n_qp * dE))
        loss_rate += K_s_eff @ (rho * one_minus_f * dE)

    if enable_recombination and K_r0 is not None:
        if N_emit_override is not None and N_abs_override is not None:
            N_emit, N_abs = N_emit_override, N_abs_override
        else:
            N_emit, N_abs = _thermal_phonon_recombination_occupations(E, T_bath)
        partner = rho * one_minus_f
        # Kaplan Eq. (8) normalization: the per-QP recombination loss is
        # 1/τ_r(E) = Σ_j K_r0[i,j] N_emit ρ_j f_j dE_j — no leading 2.
        # Each recombination event removes one QP *at this energy*; the
        # partner is counted in its own bin, so the total density already
        # loses two per event. (A legacy 2× "pair convention" here doubled
        # the absolute rate against Kaplan's τ_r and the F&C Eq. 47/E2
        # envelope; it cancelled out of detailed balance and of
        # thermal-dominated steady states, which is why figure-level
        # validations never caught it.)
        loss_rate += (K_r0 * N_emit) @ (rho * f * dE)
        gain += one_minus_f * ((K_r0 * N_abs) @ (partner * dE))

    return gain, loss_rate


def phonon_collision_jacobian_nph(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_omega: int,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
) -> np.ndarray:
    r"""Analytical ``∂(gain − loss·f)/∂n_ph`` for :func:`phonon_collision_rates`.

    Returns an ``(NE, n_omega)`` array — the exact Jacobian of the e–phonon
    collision residual ``R_f = gain − loss·f`` with respect to the phonon
    occupation ``n_ph``. ``R_f`` depends on ``n_ph`` only *linearly*, through
    the occupation matrices ``N_p`` (scattering) and ``N_emit``/``N_abs``
    (recombination/pair-breaking) of
    :func:`phonon_occupation_matrices_from_state`, so this block does not
    depend on ``n_ph`` itself. Used for the analytical ``J_fn = ∂R_f/∂n_ph``
    cross-block in
    :func:`qpsim.solvers.coupled_newton.coupled_newton_solve`.

    Mirrors :func:`phonon_collision_rates`. Scattering (``K_s0``) carries the
    zeroed self-scattering diagonal of ``N_p``; recombination (``K_r0``) the
    Kaplan Eq. (8) per-QP normalization (no pair factor). With
    ``∂N_p[a,b]/∂n_ph,m = [omega_idx_diff[a,b] = m]`` (``a ≠ b``) and
    ``∂N_emit[i,j]/∂n_ph,m = ∂N_abs[i,j]/∂n_ph,m = [omega_idx_sum[i,j] = m]``:

    * scattering: ``∂R_i/∂n_ph,m = (1−f_i) Σ_{j: idx_diff=m} K_s0[j,i] ρ_j f_j dE
      − f_i Σ_{j: idx_diff=m} K_s0[i,j] ρ_j (1−f_j) dE``  (``j ≠ i``);
    * recombination: ``∂R_i/∂n_ph,m = Σ_{j: idx_sum=m} K_r0[i,j] ρ_j dE
      [(1−f_i)(1−f_j) − f_i f_j]``.
    """
    NE = int(f.size)
    rho = ctx.rho
    dE = ctx.dE
    one_minus_f = np.maximum(1.0 - f, 0.0)
    w_j = rho * dE  # ρ_j dE, shape (NE,)

    J_fn = np.zeros((NE, n_omega))
    row_i = np.broadcast_to(np.arange(NE)[:, None], (NE, NE))

    if enable_scattering and K_s0 is not None:
        # [i,j] = (1−f_i) K_s0[j,i] ρ_j f_j dE  (gain) and
        #         f_i K_s0[i,j] ρ_j (1−f_j) dE  (loss); deposit at col idx_diff.
        gain_w = one_minus_f[:, None] * K_s0.T * (w_j * f)[None, :]
        loss_w = f[:, None] * K_s0 * (w_j * one_minus_f)[None, :]
        off = ~np.eye(NE, dtype=bool)   # N_p diagonal is zero ⇒ no self term
        np.add.at(J_fn, (row_i[off], omega_idx_diff[off]), (gain_w - loss_w)[off])

    if enable_recombination and K_r0 is not None:
        rec_w = (
            K_r0 * w_j[None, :]
            * (one_minus_f[:, None] * one_minus_f[None, :] - f[:, None] * f[None, :])
        )
        np.add.at(J_fn, (row_i, omega_idx_sum), rec_w)

    return J_fn


def apply_phonon_collision(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    dt: float,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
) -> np.ndarray:
    """One ETD2 collision step at a single spatial pixel (thermal bath).

    Convenience wrapper: defines a ``rhs(f) = phonon_collision_rates(f)``
    closure and runs one Heun-type second-order exponential step
    (:func:`qpsim.solvers.etd.etd2_step`). For the dynamic-phonon case,
    callers assemble the step explicitly using ``phonon_collision_rates``
    with ``N_*_override`` arguments.
    """
    def rhs(f_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return phonon_collision_rates(
            f_state, ctx, K_s0, K_r0, T_bath,
            enable_scattering=enable_scattering,
            enable_recombination=enable_recombination,
        )

    return etd2_step(f, rhs, dt)
