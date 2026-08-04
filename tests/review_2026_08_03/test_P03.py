"""Regression tests for qpsim/solvers/newton_steady_state.py (packet P03)."""

# ruff: noqa: N999  (review packet file names are fixed by the review process)

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.solvers.newton_steady_state import _jacobian_analytical, newton_solve_f


def _setup(T_c: float = 1.2, num: int = 30):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    return ctx, K_s0, K_r0


class TestSubnormalRowScaling:
    def test_overflowed_row_scaling_falls_back_to_the_raw_system(self) -> None:
        """A subnormal cold tail must degrade to the raw Newton direction.

        ``_row_scaled_newton_system`` raises when a row whose physical
        turnover has decayed into the subnormal band cannot be normalized
        inside the float64 range.  That RuntimeError used to escape the
        ``except np.linalg.LinAlgError`` guard, so the deliberately built raw
        fallback was unreachable and a recoverable 15 mK solve aborted.
        """
        ctx, K_s0, K_r0 = _setup()
        T_bath = 0.015
        f_thermal = fermi_dirac_occupation(ctx.E, T_bath)
        # A live gap-edge population above a subnormal thermal tail: the
        # iterate the solver walks itself into on a cold continuation.
        seed = f_thermal.copy()
        seed[:6] = 1e-3

        out = newton_solve_f(
            ctx, seed, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, max_iter=40
        )

        assert np.all(np.isfinite(out))
        # The root is the thermal occupation, recovered across ~70 decades.
        np.testing.assert_allclose(out, f_thermal, rtol=1e-6)


class TestPhononOccupationOverrideContract:
    @pytest.mark.parametrize(
        "supplied", ["N_emit_override", "N_abs_override"]
    )
    def test_unpaired_pair_bath_override_is_rejected(self, supplied: str) -> None:
        """A lone pair-bath matrix must fail loud, not be silently replaced.

        ``N_abs_override`` alone used to be overwritten by the thermal matrix
        (the default keys on ``N_emit`` only), so the solver answered the
        thermal question and the discarded array never reached the collision
        layer's shape/finiteness checks either.
        """
        ctx, K_s0, K_r0 = _setup(num=20)
        T_bath = 0.3
        shape = (ctx.E.size, ctx.E.size)
        f0 = fermi_dirac_occupation(ctx.E, T_bath)

        with pytest.raises(
            ValueError,
            match="N_emit_override and N_abs_override must be supplied together",
        ):
            newton_solve_f(
                ctx,
                f0,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                **{supplied: np.zeros(shape)},
            )

    def test_paired_zero_absorption_override_still_reaches_the_solver(self) -> None:
        """The honest loss-only pair bath is unaffected by the pairing guard."""
        ctx, _K_s0, K_r0 = _setup(num=20)
        shape = (ctx.E.size, ctx.E.size)
        vacuum = np.zeros_like(ctx.E)

        out = newton_solve_f(
            ctx,
            vacuum,
            K_r0=K_r0,
            T_bath=0.3,
            N_emit_override=np.ones(shape),
            N_abs_override=np.zeros(shape),
        )

        np.testing.assert_array_equal(out, vacuum)


class TestEffectiveKernelPrecomputeContract:
    @pytest.mark.parametrize("name", ["K_s0", "K_r0"])
    def test_mis_shaped_kernel_names_the_parameter(self, name: str) -> None:
        """The precompute must not degrade the named contract error."""
        ctx, K_s0, K_r0 = _setup(num=20)
        kwargs = {"K_s0": K_s0, "K_r0": K_r0}
        kwargs[name] = np.zeros((ctx.E.size + 1, ctx.E.size + 1))

        with pytest.raises(
            ValueError,
            match=rf"{name} must have shape \({ctx.E.size}, {ctx.E.size}\)",
        ):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                T_bath=0.3,
                max_iter=2,
                **kwargs,
            )

    @pytest.mark.parametrize(
        "name", ["N_p_override", "N_emit_override", "N_abs_override"]
    )
    def test_mis_shaped_occupation_override_names_the_parameter(
        self, name: str
    ) -> None:
        ctx, K_s0, K_r0 = _setup(num=20)
        T_bath = 0.3
        shape = (ctx.E.size, ctx.E.size)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        kwargs: dict[str, np.ndarray] = {
            "N_p_override": _thermal_phonon_scattering_occupation(ctx.E, T_bath),
            "N_emit_override": N_emit,
            "N_abs_override": N_abs,
        }
        kwargs[name] = np.ones((shape[0] + 1, shape[1] + 1))

        with pytest.raises(
            ValueError, match=rf"{name} must have shape \({shape[0]}, {shape[1]}\)"
        ):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                max_iter=2,
                **kwargs,
            )

    def test_non_finite_kernel_reports_the_contract_error_under_errstate(
        self,
    ) -> None:
        """``inf * 0`` in the precompute must not pre-empt the named error."""
        ctx, K_s0, K_r0 = _setup(num=20)
        bad = K_s0.copy()
        bad[0, 0] = np.inf

        with np.errstate(invalid="raise"), pytest.raises(
            ValueError, match="K_s0 must contain only finite non-negative rates"
        ):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                K_s0=bad,
                K_r0=K_r0,
                T_bath=0.3,
                max_iter=2,
            )

    def test_reduced_precision_inputs_are_widened_like_the_unassisted_path(
        self,
    ) -> None:
        """float32 kernels/occupations must give the float64 product.

        The collision layer widens both factors before multiplying, so the
        precomputed loop invariants have to be built from widened arrays too;
        a raw ``float32 * float32`` product rounds every kernel element to a
        24-bit mantissa.
        """
        ctx, K_s0, K_r0 = _setup(num=20)
        T_bath = 0.3
        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        # A seed that is not already the root, so the products are exercised
        # through real Newton steps.
        f0 = fermi_dirac_occupation(ctx.E, T_bath) * (
            1.0 + 0.5 * np.sin(ctx.E / ctx.gap)
        )
        narrow = {
            "K_s0": K_s0.astype(np.float32),
            "K_r0": K_r0.astype(np.float32),
            "N_p_override": N_p.astype(np.float32),
            "N_emit_override": N_emit.astype(np.float32),
            "N_abs_override": N_abs.astype(np.float32),
        }
        widened = {k: np.asarray(v, dtype=float) for k, v in narrow.items()}

        out_narrow = newton_solve_f(ctx, f0, T_bath=T_bath, tol=1e-12, **narrow)
        out_widened = newton_solve_f(ctx, f0, T_bath=T_bath, tol=1e-12, **widened)

        np.testing.assert_array_equal(out_narrow, out_widened)


class TestPairBreakingPartnerIndex:
    def test_out_of_int64_partner_index_does_not_raise_on_cast(self) -> None:
        """An off-grid partner is rejected, not cast out of the int64 range.

        The former scalar ``round()`` produced an arbitrary-precision int the
        range test discarded silently; ``np.rint(...).astype(np.int64)`` set
        the FP-invalid flag instead, which is a hard failure under
        ``errstate(invalid="raise")`` or ``-W error``.
        """
        E = np.linspace(1.0, 3.0, 21)
        dE = np.full(21, 0.1)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=1.0)
        f = np.full(21, 0.2)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            J = _jacobian_analytical(
                f,
                ctx,
                None,
                None,
                None,
                {"omega_PB": 1e30, "n_bar_PB": 1.0, "c_phot_PB": 1.0},
                None,
                None,
                None,
            )

        # Every partner is off grid, so the pair block contributes nothing.
        assert not np.any(J)
