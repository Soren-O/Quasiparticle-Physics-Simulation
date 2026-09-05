"""Layer-1 validation — closed-form checks on the single-term reductions.

The existing analytic layer pins *equilibria*: the collision integral
vanishes at thermal (``test_detailed_balance``), the gap solver lands on
``Δ_eq`` (``test_gap_equation_equilibrium``), σ reduces to Mattis-Bardeen
(``test_mattis_bardeen_thermal``). Each says "the right answer is
stationary".

This module covers what those leave out when a *single* term is active. Two
things only:

1. **Per-channel detailed balance.** The existing test evaluates the
   aggregate e-ph integral. Aggregate cancellation is also satisfied by two
   channels that are individually wrong and happen to cancel, so each channel
   is pinned separately here.
2. **The driven phonon fixed point against its closed form.** With ``f``
   held fixed the phonon equation is exactly affine, ``dn/dt = a + b n``, so
   adding escape gives a root in closed form:
   ``n* = (a + n_B/τ) / (1/τ - b)``. Comparing that against
   ``phonon_steady_state`` over a range of ``τ`` tests the solver against
   algebra rather than against a stored number.
   ``tests/phonon_models/test_local.py`` already does the *thermal* case,
   where the answer collapses to ``n_BE``; this is the driven case, where the
   root actually depends on ``τ``.

Deliberately NOT duplicated here, because the existing tests are stronger:

* scattering conserving quasiparticle number —
  ``tests/collisions/test_phonon.py`` asserts the number moment vanishes AND
  that the quasiparticle and phonon energy rates cancel;
* the sub-gap photon channel conserving capacity —
  ``tests/collisions/test_sub_gap_photon.py`` asserts it and additionally
  reproduces the point-rate failure mode it guards against;
* thermal ``f`` giving ``n_BE`` —
  ``tests/phonon_models/test_local.py``, through the solver.

Also deliberately absent: the two-quasiparticles-per-phonon identity. It
needs like-for-like measures across the two grids, and
``docs/CODE-REVIEW-FALSE-POSITIVES.md`` entry 35 already characterises it
properly. A naive ratio here would restate it wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.phonon_models.local import phonon_steady_state
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation

DELTA_0 = 180.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.2
TAU_0 = 438.0


def _ctx(num: int = 205) -> SpectralContext:
    """Commensurate grid: 2*E_min/dE integer, as the phonon lattice requires."""
    E, _ = build_energy_grid(
        gap=DELTA_0, energy_min_factor=1.0, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    return SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)


def _kernels(ctx: SpectralContext) -> tuple[np.ndarray, np.ndarray]:
    return (
        build_scattering_kernel_base(ctx, tau_0=TAU_0, T_c=T_C),
        build_recombination_kernel_base(ctx, tau_0=TAU_0, T_c=T_C),
    )


def _number_moment(ctx: SpectralContext, dfdt: np.ndarray) -> float:
    """d/dt of the represented quasiparticle number.

    The conserved measure is ``cell_weights`` -- the integrated DOS per cell,
    not ``rho * dE`` -- so this is the measure the backend conserves.
    """
    return float(np.sum(np.asarray(ctx.cell_weights) * dfdt))


def _turnover(ctx: SpectralContext, gain: np.ndarray, loss: np.ndarray,
              f: np.ndarray) -> float:
    """Scale to judge a moment against: total events, not net."""
    w = np.asarray(ctx.cell_weights)
    return float(np.sum(w * gain) + np.sum(w * loss * f))


def _driven_f(ctx: SpectralContext, boost: float = 1.0) -> np.ndarray:
    """A clearly non-thermal occupation, bounded well inside [0, 1]."""
    E = np.asarray(ctx.E)
    f = 3.0e-3 * np.exp(-(E - ctx.gap) / (6.0 * KB_UEV_PER_K * T_BATH))
    f += 5.0e-4 * np.exp(-(((E - 1.7 * ctx.gap) / (0.35 * ctx.gap)) ** 2))
    return np.clip(f * boost, 0.0, 0.4)


class TestEachChannelVanishesSeparatelyAtThermal:
    """Per-channel detailed balance.

    ``test_detailed_balance`` evaluates the aggregate integral. Two channels
    that are individually wrong can still cancel in the aggregate, so pin them
    one at a time. The both-on row is kept as the control: without it, a
    per-channel failure could not be distinguished from a bad setup.
    """

    @pytest.mark.parametrize(
        ("scattering", "recombination"),
        [(True, False), (False, True), (True, True)],
    )
    def test_thermal_is_a_fixed_point_channel_by_channel(
        self, scattering: bool, recombination: bool
    ) -> None:
        ctx = _ctx()
        K_s0, K_r0 = _kernels(ctx)
        f = fermi_dirac_occupation(ctx.E, T_BATH)
        gain, loss = phonon_collision_rates(
            f, ctx, K_s0, K_r0, T_BATH,
            enable_scattering=scattering, enable_recombination=recombination,
        )
        dfdt = gain - loss * f
        scale = _turnover(ctx, gain, loss, f)
        assert scale > 0.0
        assert float(np.max(np.abs(dfdt))) / scale < 1e-12


class TestDrivenPhononFixedPointMatchesClosedForm:
    """With f fixed the phonon equation is affine, so its root is algebraic.

    ``dn/dt = a + b n - (n - n_B)/τ`` has the root
    ``n* = (a + n_B/τ) / (1/τ - b)``. This is the driven case: unlike the
    thermal case already covered in ``test_local``, the root genuinely
    moves with τ, sweeping from the bath occupation at fast escape to the
    quasiparticle-driven root at slow escape.
    """

    def _affine(self, ctx: SpectralContext, f: np.ndarray):
        K_s0, K_r0 = _kernels(ctx)
        omega, idx_d, idx_s, sgn = build_phonon_frequency_map(ctx.E)
        a, b = compute_phonon_source_sink(
            f, ctx, K_s0, K_r0, idx_d, idx_s, sgn, omega.size,
        )
        return omega, idx_d, idx_s, sgn, K_s0, K_r0, a, b

    def test_decay_coefficient_is_negative_so_a_root_exists(self) -> None:
        """b < 0 is what makes the root stable rather than a runaway."""
        ctx = _ctx()
        _omega, _d, _s, _g, _Ks, _Kr, a, b = self._affine(ctx, _driven_f(ctx))
        live = b != 0.0
        # exactly one inert row: omega = 0, where there is no phonon at all
        assert int(np.count_nonzero(~live)) == 1
        assert float(np.max(np.abs(a[~live]))) == 0.0
        assert float(np.max(b[live])) < 0.0

    @pytest.mark.parametrize("tau_l", [0.05, 0.25, 2.5, 25.0])
    def test_solver_matches_the_closed_form_root(self, tau_l: float) -> None:
        ctx = _ctx()
        f = _driven_f(ctx, boost=8.0)
        omega, idx_d, idx_s, sgn, K_s0, K_r0, a, b = self._affine(ctx, f)
        n_bath = thermal_phonon_occupation(omega, T_BATH)

        solved = phonon_steady_state(
            f, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_BATH, tau_l=tau_l,
        )
        closed = (a + n_bath / tau_l) / (1.0 / tau_l - b)

        live = (b != 0.0) & np.isfinite(closed) & (closed > 0.0)
        assert int(np.count_nonzero(live)) > 50
        rel = np.abs(solved[live] - closed[live]) / closed[live]
        assert float(np.max(rel)) < 1e-12, float(np.max(rel))

    def test_root_moves_with_tau_between_the_two_limits(self) -> None:
        """Discriminating power: the root is not tau-independent here.

        At thermal f the root is n_BE for every tau, so a tau-independent
        answer would pass the existing thermal test. Driven, the two limits
        must differ, or the parametrised comparison above proves nothing.
        """
        ctx = _ctx()
        f = _driven_f(ctx, boost=8.0)
        omega, idx_d, idx_s, sgn, K_s0, K_r0, _a, b = self._affine(ctx, f)
        fast = phonon_steady_state(
            f, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_BATH, tau_l=1e-3 / float(np.max(np.abs(b))),
        )
        slow = phonon_steady_state(
            f, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_BATH, tau_l=1e3 / float(np.min(np.abs(b[b != 0.0]))),
        )
        live = (fast > 0.0) & (slow > 0.0)
        assert int(np.count_nonzero(live)) > 50
        # The root must move by orders of magnitude between the two limits,
        # or the parametrised comparison above would pass on a solver that
        # ignored tau entirely. Only the ratio is asserted: in the high-omega
        # tail the bath occupation underflows to ~1e-54 while the driven
        # contribution does not, so the population there is driven rather than
        # thermal at ANY escape rate, and "pinned to the bath" is not a
        # relative statement that survives out there.
        assert float(np.max(slow[live] / fast[live])) > 10.0


class TestClosedLoopEnergyLedger:
    """A mirrored channel and its partner conserve energy exactly.

    ``I_sc`` with ``Σ_sc``, and ``I_r`` with ``Σ_r``, are the same events
    counted from either side. With no escape the energy leaving the
    quasiparticles must arrive in the phonons, exactly -- this is the
    invariant the two-equation structure exists to preserve, and the one the
    Kaplan S+ one-sidedness breaks, so it doubles as an acceptance gate for
    that repair.

    MEASURE TRAP, and the reason this is easy to get wrong. The phonon source
    is assembled by bincounting the full ``(i, j)`` matrix. On the DIFFERENCE
    lattice that scattering uses, ``(i, j)`` and ``(j, i)`` are distinct
    events -- an emission and an absorption -- so the plain sum is correct.
    On the SUM lattice that recombination uses they are the SAME event, so
    the plain sum counts every unordered pair twice and ``omega @ a`` comes
    out at exactly twice the quasiparticle energy loss. The one half below is
    that double count, not a fudge factor;
    ``test_the_unordered_pair_halving_is_load_bearing`` pins it, so nobody
    can quietly drop it or copy it onto the scattering channel where it does
    not belong.
    """

    def _rates(self, ctx: SpectralContext, f: np.ndarray, *, scattering: bool):
        K_s0, K_r0 = _kernels(ctx)
        omega, idx_d, idx_s, sgn = build_phonon_frequency_map(ctx.E)
        # zero-temperature phonons, so the ledger is a clean one-way transfer
        N_p, _emit, _abs = phonon_occupation_matrices_from_state(
            np.zeros(omega.size), idx_d, idx_s, sgn,
        )
        gain, loss = phonon_collision_rates(
            f, ctx, K_s0, K_r0, T_bath=0.0, N_p_override=N_p,
            enable_scattering=scattering, enable_recombination=not scattering,
        )
        a, _b = compute_phonon_source_sink(
            f, ctx, K_s0, K_r0, idx_d, idx_s, sgn, omega.size,
            enable_scattering=scattering, enable_recombination=not scattering,
        )
        E = np.asarray(ctx.E)
        qp = float(np.asarray(ctx.cell_weights) @ (E * (gain - loss * f)))
        phonon_sum = float(float(ctx.dE[0]) * (omega @ a))
        return qp, phonon_sum

    @pytest.mark.parametrize("num", [105, 205, 305])
    def test_scattering_loop_conserves_energy(self, num: int) -> None:
        ctx = _ctx(num)
        qp, ph = self._rates(ctx, _driven_f(ctx, boost=8.0), scattering=True)
        assert qp < 0.0 and ph > 0.0
        assert abs(qp + ph) / (abs(qp) + abs(ph)) < 1e-13, (num, qp, ph)

    @pytest.mark.parametrize("num", [105, 205, 305])
    def test_recombination_loop_conserves_energy(self, num: int) -> None:
        ctx = _ctx(num)
        qp, ph_sum = self._rates(ctx, _driven_f(ctx, boost=8.0), scattering=False)
        ph = 0.5 * ph_sum  # unordered pairs; see the class docstring
        assert qp < 0.0 and ph > 0.0
        assert abs(qp + ph) / (abs(qp) + abs(ph)) < 1e-13, (num, qp, ph)

    def test_the_unordered_pair_halving_is_load_bearing(self) -> None:
        """Without the one half the recombination ledger is out by exactly 2.

        If this ever stops being 2, the assembly convention has changed and
        the factor above needs re-deriving rather than re-tuning.
        """
        ctx = _ctx()
        qp, ph_sum = self._rates(ctx, _driven_f(ctx, boost=8.0), scattering=False)
        assert ph_sum / abs(qp) == pytest.approx(2.0, rel=1e-12)

    def test_scattering_needs_no_halving(self) -> None:
        """The same factor must NOT be applied to the difference lattice."""
        ctx = _ctx()
        qp, ph = self._rates(ctx, _driven_f(ctx, boost=8.0), scattering=True)
        assert ph / abs(qp) == pytest.approx(1.0, rel=1e-12)


class TestNumberChangingChannels:
    """Which single terms can change the quasiparticle count, and which sign.

    Only the pair channel and the pair-breaking drive touch the count; that
    asymmetry is what makes the steady state unique when they are present and
    degenerate when they are not.
    """

    def test_recombination_alone_removes_quasiparticles(self) -> None:
        ctx = _ctx()
        K_s0, K_r0 = _kernels(ctx)
        f = _driven_f(ctx, boost=30.0)
        gain, loss = phonon_collision_rates(
            f, ctx, K_s0, K_r0, T_BATH, enable_scattering=False,
        )
        moment = _number_moment(ctx, gain - loss * f)
        scale = _turnover(ctx, gain, loss, f)
        assert scale > 0.0
        assert moment / scale < -1e-6, (moment, scale)

    def test_pair_breaking_photon_alone_creates_quasiparticles(self) -> None:
        ctx = _ctx()
        f = _driven_f(ctx)
        dE = float(ctx.dE[0])
        m = int(np.ceil(2.2 * ctx.gap / dE))
        omega_pb = m * dE
        assert omega_pb > 2.0 * ctx.gap
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=omega_pb, n_bar_PB=1.0e3, c_phot_PB=1.0,
        )
        moment = _number_moment(ctx, gain - loss * f)
        scale = _turnover(ctx, gain, loss, f)
        assert scale > 0.0
        assert moment / scale > 1e-6, (moment, scale)
