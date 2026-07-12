r"""Self-consistent n̄(P_read) loop for the sub-gap photon drive.

At fixed read-out power, the on-resonance photon number inside the
resonator depends on its loaded quality factor, which in turn depends
on the quasiparticle distribution — creating a coupled scalar loop
the outer driver has to close.

From [F23] Eqs. 59-60, with Q_tot ≡ (Q_i⁻¹ + Q_c⁻¹)⁻¹:

.. math::

    \bar n = \frac{2\,\hbar}{\omega_0^2}\,\frac{Q_\mathrm{tot}^2}{Q_c}\,P_\mathrm{read}

(``ℏ`` enters because ``ω_0`` is stored as an energy in μeV and time
in ns — the photon-energy-per-second ``P_read`` has units μeV/ns, so
``P_read/ω_0`` gives photons/ns and an extra ``ℏ/ω_0`` gives
photons.)

This service is tier-agnostic: it calls user-supplied ``solve_f`` and
``compute_Q_i`` callables, so the same loop serves T3, T2, T1 and any
phonon model. The Fischer Fig 7-with-drive + Figs 9-13 Q_i(P_read)
reproductions are the two primary callers.

Ported from the legacy ``SimulationService.run_steady_state_nbar_loop``
at Gate 3; the legacy entry-point was bundled with the whole
``SetupData`` orchestration and is not appropriate for direct reuse
in the greenfield services layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qpsim.constants import HBAR_UEV_NS

# 1 W = 1e-3 × 6.241509074e15 μeV/ns (1 eV = 1.602176634e-19 J).
_MW_TO_UEV_PER_NS = 6.241509074e12


def dbm_to_uev_per_ns(dbm: float) -> float:
    """Convert microwave drive power from dBm to μeV/ns (the code units).

    ``P[μeV/ns] = 10^(dBm/10) · 1 mW · (6.241509e12 μeV/ns / mW)``.
    """
    return (10.0 ** (float(dbm) / 10.0)) * _MW_TO_UEV_PER_NS


@dataclass(frozen=True)
class NbarLoopIteration:
    """Per-iteration diagnostics emitted by :func:`solve_nbar_loop`."""

    iteration: int
    n_bar: float        # occupancy at the start of this iteration
    Q_qp: float         # Q_i(n_bar) from the converged f
    Q_tot: float        # (Q_qp⁻¹ + Q_c⁻¹)⁻¹
    n_bar_raw: float    # prefactor·Q_tot²·P_read  (pre-relaxation fixed-point)
    n_bar_next: float   # n_bar after under-relaxation


@dataclass(frozen=True)
class NbarLoopResult:
    """Outputs of :func:`solve_nbar_loop`."""

    n_bar: float               # converged occupancy
    f: np.ndarray              # converged f(E) at the final n_bar
    Q_i: float                 # Q_i(n_bar) at the converged state
    Q_tot: float               # loaded Q factor
    iterations: int            # number of fixed-point iterations taken
    converged: bool            # True iff the relative-change tolerance was met
    history: list[NbarLoopIteration] = field(default_factory=list)


def solve_nbar_loop(
    *,
    P_read_uev_per_ns: float,
    Q_c: float,
    omega_0: float,
    solve_f: Callable[[float], np.ndarray],
    compute_Q_i: Callable[[np.ndarray], float],
    n_bar_initial: float | None = None,
    tol: float = 1e-4,
    max_iter: int = 50,
    under_relaxation: float = 1.0,
) -> NbarLoopResult:
    r"""Fixed-point iterate on ``n̄`` until ``|Δn̄/n̄| < tol``.

    Each iteration:

    1. Solve the steady-state ``f(E)`` at the current ``n̄`` via the
       injected ``solve_f(n_bar)`` callable.
    2. Evaluate ``Q_i(n̄)`` from that ``f`` via the injected
       ``compute_Q_i(f)`` callable.
    3. Compute ``Q_tot = (Q_i⁻¹ + Q_c⁻¹)⁻¹``.
    4. Update: ``n̄ ← (1−α)·n̄ + α · 2·ℏ·Q_tot²·P_read/(ω₀²·Q_c)``,
       where ``α = under_relaxation``.

    Parameters
    ----------
    P_read_uev_per_ns
        Drive power in code units (μeV/ns). Use
        :func:`dbm_to_uev_per_ns` to convert from dBm.
    Q_c
        Coupling quality factor of the resonator.
    omega_0
        Photon frequency in μeV.
    solve_f
        ``n_bar -> f(E)``. Should internally wire a photon channel
        at occupancy ``n_bar``, frequency ``omega_0`` and solve for
        the steady-state QP distribution.
    compute_Q_i
        ``f(E) -> Q_i``. Typically wraps :func:`compute_ac_conductivity`
        and :func:`compute_quality_factor` at the same ``omega_0`` /
        ``alpha`` the caller cares about.
    n_bar_initial
        Initial ``n̄`` guess. Defaults to the ``Q_i → ∞`` limit
        ``2·ℏ·Q_c·P_read/ω₀²``, which is a safe upper bound on the
        true ``n̄`` (since ``Q_tot ≤ Q_c``).
    tol
        Relative-change tolerance on ``n̄``.
    max_iter
        Iteration cap.
    under_relaxation
        Mixing factor ``α ∈ (0, 1]``. ``1.0`` means pure fixed-point
        iteration; lower values help at high-``Q_i`` plateaus where
        the map is sensitive to small ``n̄`` changes.

    Returns
    -------
    NbarLoopResult
        Converged ``n̄``, ``f``, ``Q_i`` at that state, the loaded
        ``Q_tot``, iteration count, and the per-iteration history.

    Raises
    ------
    ValueError
        On any non-physical input (``P_read < 0``, ``Q_c ≤ 0``, etc.).
    """
    if P_read_uev_per_ns < 0:
        raise ValueError("P_read_uev_per_ns must be non-negative.")
    if Q_c <= 0:
        raise ValueError("Q_c must be positive.")
    if omega_0 <= 0:
        raise ValueError("omega_0 must be positive.")
    if not (0.0 < under_relaxation <= 1.0):
        raise ValueError("under_relaxation must be in (0, 1].")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    omega_sq = float(omega_0) ** 2
    prefactor = 2.0 * HBAR_UEV_NS / (omega_sq * float(Q_c))

    if n_bar_initial is not None:
        if n_bar_initial < 0:
            raise ValueError("n_bar_initial must be non-negative.")
        n_bar = float(n_bar_initial)
    else:
        # Q_i → ∞ limit: Q_tot → Q_c ⇒ n_bar ≈ 2·ℏ·Q_c·P_read/ω₀².
        n_bar = prefactor * float(Q_c) ** 2 * float(P_read_uev_per_ns)

    history: list[NbarLoopIteration] = []
    f_converged: np.ndarray | None = None
    q_qp = float("nan")
    q_tot = float("nan")
    converged = False

    for it in range(max_iter):
        f_converged = solve_f(n_bar)
        q_qp = float(compute_Q_i(f_converged))
        if np.isnan(q_qp) or q_qp <= 0.0:
            # A NaN or non-positive internal Q_i is a failed / unphysical
            # observable, not the physical zero-loss limit. Folding it into
            # Q_tot = Q_c (the "Q_i -> inf" stand-in) silently masked a broken
            # solve and could carry a failed observable into a pinned baseline.
            # Fail loudly instead, matching phonon_steady_state's policy.
            raise RuntimeError(
                f"nbar_loop: compute_Q_i returned a non-physical quality "
                f"factor Q_i = {q_qp} at iteration {it} (n_bar = {n_bar:g}); "
                "expected a positive value or +inf (the zero-loss limit)."
            )
        # Physical Q_i -> inf (vanishing quasiparticle loss) gives Q_tot -> Q_c;
        # a finite positive Q_i combines with Q_c in parallel.
        q_tot = (
            float(Q_c)
            if np.isinf(q_qp)
            else 1.0 / (1.0 / q_qp + 1.0 / float(Q_c))
        )

        n_bar_raw = prefactor * q_tot**2 * float(P_read_uev_per_ns)
        n_bar_next = (
            (1.0 - under_relaxation) * n_bar + under_relaxation * n_bar_raw
        )

        history.append(
            NbarLoopIteration(
                iteration=it,
                n_bar=float(n_bar),
                Q_qp=q_qp,
                Q_tot=float(q_tot),
                n_bar_raw=float(n_bar_raw),
                n_bar_next=float(n_bar_next),
            )
        )

        denom = max(abs(n_bar), 1e-300)
        rel_change = abs(n_bar_next - n_bar) / denom
        n_bar = n_bar_next
        if rel_change < tol:
            converged = True
            break

    # Re-solve once at the final n_bar so the returned f matches the
    # reported n_bar. (After the break, n_bar has already advanced to
    # n_bar_next; the last solve inside the loop used the old n_bar.)
    f_converged = solve_f(n_bar)
    q_qp_raw = compute_Q_i(f_converged)
    q_qp = float(q_qp_raw) if np.isfinite(q_qp_raw) else float("nan")
    q_tot = (
        1.0 / (1.0 / q_qp + 1.0 / float(Q_c))
        if np.isfinite(q_qp) and q_qp > 0
        else float(Q_c)
    )

    return NbarLoopResult(
        n_bar=float(n_bar),
        f=f_converged,
        Q_i=q_qp,
        Q_tot=float(q_tot),
        iterations=len(history),
        converged=converged,
        history=history,
    )
