r"""Qubit, QubitState, QubitTransitionChannel, JunctionQubitCoupling.

Phase 4 of the Device Architecture: a discrete two-level (or N-level)
system optionally coupled to junction tunneling events. The qubit
state space is ``(level, parity)``: a 2-level transmon with parity
tracking has 4 discrete states |0,e>, |0,o>, |1,e>, |1,o>, addressed
by ``QubitState.p`` of shape ``(n_levels, 2)``.

Junction implementations that couple to a qubit return a list of
:class:`QubitTransitionChannel` records on every ``evaluate`` call.
Each channel tags itself ``flips_parity`` so the master-equation
evolver knows whether to advance the parity axis along with the
level axis. This separates the M25-style ``eo`` (parity-flipping
QP tunneling) channels from ``ee`` (parity-preserving photon-driven)
channels cleanly.

The master-equation evolver
:func:`solve_qubit_master_equation_steady_state` takes a list of
channels and a Qubit configuration and returns the steady-state
populations satisfying ``M @ p = 0`` with ``sum(p) = 1`` — a small
linear-algebra problem (4×4 for the parity-tracking 2-level case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Qubit:
    """N-level discrete two-level (or higher) system with optional parity.

    Parameters
    ----------
    n_levels
        Number of logical levels. v1 ships with 2 (transmon ground
        + first excited); 3 (qutrit) and higher are accepted by the
        master-equation evolver but not currently exercised.
    track_parity
        When True, the state space is ``(n_levels, 2)`` — every
        Junction tunneling event flips parity, so this is needed for
        any QP-coupled qubit. When False, drops the parity axis.
    omega_kelvin
        Per-level energies (length ``n_levels``) in Kelvin. Used for
        the transition-frequency labeling and detailed-balance
        relations between channels. The level-0 entry is conventionally
        zero (ground reference).
    E_J_kelvin, E_C_kelvin
        Transmon Josephson and charging energies (Kelvin). Optional
        metadata; the master-equation evolver does not use them
        directly. Carried so a ``JunctionQubitCoupling`` (which
        derives its matrix elements from the transmon parameters)
        can pull them off the Qubit.

    Raises
    ------
    ValueError
        On non-positive ``n_levels``, mismatched ``omega_kelvin``
        length, or non-monotone level energies.
    """

    n_levels: int = 2
    track_parity: bool = True
    omega_kelvin: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0])
    )
    E_J_kelvin: float = 0.0
    E_C_kelvin: float = 0.0

    def __post_init__(self) -> None:
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be ≥ 1; got {self.n_levels}")
        omega = np.asarray(self.omega_kelvin, dtype=float)
        if omega.shape != (self.n_levels,):
            raise ValueError(
                f"omega_kelvin must have shape ({self.n_levels},); "
                f"got {omega.shape}."
            )
        if not np.all(np.diff(omega) >= 0.0):
            raise ValueError(
                f"omega_kelvin must be non-decreasing (level 0 = ground "
                f"reference); got {omega}."
            )
        object.__setattr__(self, "omega_kelvin", omega)


@dataclass
class QubitState:
    """Probabilities over the (level, parity) state space.

    Parameters
    ----------
    p
        Shape ``(n_levels, 2)`` when parity is tracked, else
        ``(n_levels,)``. Parity-axis indices: ``[0]`` = even,
        ``[1]`` = odd. Entries must be non-negative; ``np.sum(p) = 1``
        within ``rtol = 1e-12``.
    t_ns
        Lab-frame time when this state was reached (ns). Defaults to
        zero. Used by transient evolution (Phase 5+); steady-state
        results leave it at zero.

    Raises
    ------
    ValueError
        On wrong shape, negative entry, non-finite, or
        ``|sum(p) − 1| > 1e-12``.
    """

    p: np.ndarray
    t_ns: float = 0.0

    def __post_init__(self) -> None:
        p = np.asarray(self.p, dtype=float)
        if p.ndim not in (1, 2):
            raise ValueError(
                f"QubitState.p must be 1D (no parity) or 2D shape "
                f"(n_levels, 2); got shape {p.shape}."
            )
        if p.ndim == 2 and p.shape[1] != 2:
            raise ValueError(
                f"QubitState.p with parity must have shape "
                f"(n_levels, 2); got {p.shape}."
            )
        if not np.all(np.isfinite(p)):
            raise ValueError("QubitState.p has non-finite entries.")
        if np.any(p < 0.0):
            raise ValueError(
                f"QubitState.p must be non-negative; got min {float(p.min())}."
            )
        total = float(p.sum())
        if abs(total - 1.0) > 1e-12:
            raise ValueError(
                f"QubitState.p must sum to 1; got {total}."
            )
        object.__setattr__(self, "p", p)


@dataclass
class QubitTransitionChannel:
    """One addressable qubit transition produced by a Junction.

    Parameters
    ----------
    level_from, level_to
        Initial and final logical levels (0-indexed). May be equal
        when the transition is purely parity-flipping (``M25``-style
        photon-conserving channel).
    rate_per_ns
        Transition rate ``Γ_{level_from → level_to}`` in 1/ns. Must
        be non-negative. State-independent at the qubit-master-
        equation level: any density-dependence on the source region's
        f(E) is computed inside the Junction's ``evaluate``.
    flips_parity
        ``True`` for parity-flipping (``eo``) channels — every QP
        tunneling event flips junction parity. ``False`` for parity-
        preserving (``ee``) channels — ``M25``-style photon-driven
        relaxation/excitation that does not move charge across the
        junction.
    label
        Optional diagnostic label, e.g. ``"ph_00"``, ``"ee_10"``,
        ``"Γ̃^L_10"``. Not consumed by the evolver.

    Raises
    ------
    ValueError
        On negative rate or non-finite values.
    """

    level_from: int
    level_to: int
    rate_per_ns: float
    flips_parity: bool
    label: str = ""

    def __post_init__(self) -> None:
        if self.level_from < 0 or self.level_to < 0:
            raise ValueError(
                f"level_from and level_to must be non-negative; "
                f"got level_from={self.level_from}, level_to={self.level_to}."
            )
        if self.rate_per_ns < 0.0:
            raise ValueError(
                f"rate_per_ns must be non-negative; got {self.rate_per_ns}."
            )
        if not np.isfinite(self.rate_per_ns):
            raise ValueError(f"rate_per_ns must be finite; got {self.rate_per_ns}.")


@dataclass
class JunctionQubitCoupling:
    r"""How a Junction drives the Qubit.

    Holds the transmon matrix elements (M25 SI Eqs. S25–S28) that
    split the tunneling rate between ``sin(φ̂/2)`` (parity-flipping,
    logical-changing — the ``s²_{ii'}`` factors) and ``cos(φ̂/2)``
    (parity-flipping, logical-conserving — the ``c²_{ii'}`` factors)
    contributions. Parity-preserving (``ee``) channels live in a
    separate optional rate matrix.

    Parameters
    ----------
    sin_matrix_elements
        Shape ``(n_levels, n_levels)`` — ``s²_{ii'}`` for the
        ``sin(φ̂/2)`` matrix elements. Single-junction transmon
        approximation (E_J ≫ E_C): only the off-diagonal
        ``s²_{10}`` and ``s²_{01}`` are nonzero; diagonals
        exponentially suppressed by charge dispersion.
    cos_matrix_elements
        Shape ``(n_levels, n_levels)`` — ``c²_{ii'}`` for the
        ``cos(φ̂/2)`` matrix elements. Single-junction transmon
        approximation: only the diagonals ``c²_{ii}`` are nonzero
        and ≈ 1 at leading order in ``E_C / E_J``.
    parity_preserving_rates
        Optional ``(n_levels, n_levels)`` matrix of parity-preserving
        ``ee``-channel rates in 1/ns. ``None`` (default) means no
        ``ee`` channels are emitted. Used by Junctions that include
        e.g. M25's bulk parity-preserving photon channel.

    Raises
    ------
    ValueError
        On wrong shape, negative entry, or non-finite.
    """

    sin_matrix_elements: np.ndarray
    cos_matrix_elements: np.ndarray
    parity_preserving_rates: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("sin_matrix_elements", "cos_matrix_elements"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"{name} must be a square 2D array; got shape {arr.shape}."
                )
            if np.any(arr < 0.0):
                raise ValueError(
                    f"{name} must be non-negative; got min {float(arr.min())}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} has non-finite entries.")
            object.__setattr__(self, name, arr)
        if self.sin_matrix_elements.shape != self.cos_matrix_elements.shape:
            raise ValueError(
                f"sin_matrix_elements shape {self.sin_matrix_elements.shape} "
                f"and cos_matrix_elements shape "
                f"{self.cos_matrix_elements.shape} must match."
            )
        if self.parity_preserving_rates is not None:
            arr = np.asarray(self.parity_preserving_rates, dtype=float)
            if arr.shape != self.sin_matrix_elements.shape:
                raise ValueError(
                    f"parity_preserving_rates shape {arr.shape} must match "
                    f"sin/cos matrix elements shape "
                    f"{self.sin_matrix_elements.shape}."
                )
            if np.any(arr < 0.0):
                raise ValueError(
                    f"parity_preserving_rates must be non-negative; "
                    f"got min {float(arr.min())}."
                )
            object.__setattr__(self, "parity_preserving_rates", arr)


# ═══════════════════════════════════════════════════════════════════════
#  Master-equation evolver
# ═══════════════════════════════════════════════════════════════════════


def _state_index(level: int, parity: int, n_par: int) -> int:
    """Flat index into the (level, parity) state space."""
    return level * n_par + parity


def build_rate_matrix(
    channels: list[QubitTransitionChannel],
    n_levels: int,
    *,
    track_parity: bool = True,
) -> np.ndarray:
    r"""Assemble the master-equation rate matrix from a channel list.

    The flat state vector has shape ``(n_levels × n_par,)`` with
    ``n_par = 2 if track_parity else 1``. The state index ordering
    is ``level * n_par + parity`` so a 2-level parity-tracking qubit
    has flat states ``[|0,e>, |0,o>, |1,e>, |1,o>]``.

    The returned matrix ``M`` satisfies ``dp/dt = M @ p`` with
    ``M[to, from]`` for off-diagonal entries (incoming rates) and
    ``M[k, k] = -Σ_{j ≠ k} M[j, k]`` for diagonals (outgoing).

    Parameters
    ----------
    channels
        List of :class:`QubitTransitionChannel`. Each contributes
        rate to ``M[to_state, from_state]``; if ``flips_parity``
        and ``track_parity``, the parity axis is advanced by 1 (mod 2).
    n_levels
        Number of qubit levels.
    track_parity
        Whether the state space includes the 2-state parity axis.

    Returns
    -------
    np.ndarray
        Shape ``(n_levels × n_par, n_levels × n_par)`` rate matrix.
    """
    n_par = 2 if track_parity else 1
    n_states = n_levels * n_par
    M = np.zeros((n_states, n_states))

    for ch in channels:
        if ch.level_from >= n_levels or ch.level_to >= n_levels:
            raise ValueError(
                f"Channel {ch.label!r}: level_from={ch.level_from}, "
                f"level_to={ch.level_to} exceeds n_levels={n_levels}."
            )
        for sigma_from in range(n_par):
            sigma_to = (
                (1 - sigma_from) if (track_parity and ch.flips_parity)
                else sigma_from
            )
            from_idx = _state_index(ch.level_from, sigma_from, n_par)
            to_idx = _state_index(ch.level_to, sigma_to, n_par)
            # Off-diagonal: incoming flux at to_idx FROM from_idx.
            M[to_idx, from_idx] += ch.rate_per_ns
            # Diagonal: outgoing flux at from_idx (loss term).
            M[from_idx, from_idx] -= ch.rate_per_ns
    return M


def solve_qubit_master_equation_steady_state(
    channels: list[QubitTransitionChannel],
    qubit: Qubit,
) -> QubitState:
    """Solve ``M @ p = 0`` with ``sum(p) = 1`` for the qubit steady state.

    The rate matrix from :func:`build_rate_matrix` has a one-
    dimensional null space spanned by the (unique up to normalization)
    steady-state distribution. We replace one row with the
    normalization constraint and solve the resulting square system.

    Parameters
    ----------
    channels
        Active channels driving the qubit.
    qubit
        Qubit configuration (n_levels, track_parity).

    Returns
    -------
    QubitState
        Converged populations summing to 1 with shape
        ``(n_levels, 2)`` when parity is tracked, else
        ``(n_levels,)``.

    Raises
    ------
    RuntimeError
        If the assembled rate matrix is singular in a way that
        prevents a unique steady state (no channels passed, or
        disconnected state space).
    """
    if not channels:
        raise RuntimeError(
            "No channels supplied; the qubit master equation has no "
            "transitions and the steady state is undefined. Check that "
            "the device's junctions emit qubit_channels."
        )

    n_par = 2 if qubit.track_parity else 1
    n_states = qubit.n_levels * n_par
    M = build_rate_matrix(channels, qubit.n_levels, track_parity=qubit.track_parity)

    # Replace last row with the normalization constraint ∑ p = 1.
    A = M.copy()
    A[-1, :] = 1.0
    b = np.zeros(n_states)
    b[-1] = 1.0

    try:
        p_flat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as err:
        raise RuntimeError(
            "Qubit rate matrix is singular even after the normalization "
            "row replacement. The state space may be disconnected by "
            "the supplied channels."
        ) from err

    if np.any(p_flat < -1e-12):
        raise RuntimeError(
            f"Qubit steady-state solve produced negative populations "
            f"(min {float(p_flat.min()):.3e}). The channel rates may "
            "be inconsistent (e.g. negative effective rate after sign "
            "convention)."
        )
    p_flat = np.maximum(p_flat, 0.0)
    p_flat = p_flat / p_flat.sum()  # normalize away rounding

    p = p_flat.reshape(qubit.n_levels, 2) if qubit.track_parity else p_flat
    return QubitState(p=p)
