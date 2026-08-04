# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Packet P01: qpsim/collisions/phonon.py review fixes.

Covers the pair-breaking quadrature memo (identity/liveness/content guards),
the direction of the Kaplan endpoint correction on the production grid family,
and the shape contract of the effective-kernel fast-path overrides.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    _PB_QUAD_CORR_CACHE,
    _pair_breaking_quadrature_correction,
    _pair_breaking_quadrature_correction_impl,
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    phonon_collision_rates,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext

TAU_0_PB = 0.255


def _memo_setup() -> tuple[SpectralContext, np.ndarray, np.ndarray, int]:
    """Small active grid whose Kaplan correction is far from unity."""
    gap = 1.0
    E = np.linspace(gap, 3.0 * gap, 21)
    ctx = SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap
    )
    omega, _, idx_sum, _ = build_phonon_frequency_map(E)
    K = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=TAU_0_PB)
    return ctx, K, idx_sum, int(omega.size)


@pytest.fixture(autouse=True)
def _clean_memo() -> None:
    _PB_QUAD_CORR_CACHE.clear()
    yield
    _PB_QUAD_CORR_CACHE.clear()


class TestPairBreakingQuadratureMemo:
    def test_repeated_identical_call_is_served_from_the_cache(self) -> None:
        # The memo must still memoize: a hit returns the very same array.
        ctx, K, idx_sum, n_omega = _memo_setup()
        first = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)
        second = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)

        assert second is first
        assert np.any(first != 1.0)  # the correction is genuinely active here

    def test_entry_holds_identity_guards_for_all_three_inputs(self) -> None:
        # id() is not identity for objects the cache does not own: CPython
        # reuses a freed array's address. Every id in the key must therefore
        # be backed by a weakref that is re-verified on a hit.
        ctx, K, idx_sum, n_omega = _memo_setup()
        _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)

        (entry,) = _PB_QUAD_CORR_CACHE.values()
        ctx_ref, kernel_ref, idx_ref, _ = entry
        assert ctx_ref() is ctx
        assert kernel_ref() is K
        assert idx_ref() is idx_sum

    def test_in_place_kernel_edit_is_a_documented_limitation(self) -> None:
        # DOCUMENTED LIMITATION, pinned deliberately.  The memo key is O(1) by
        # design: identity comes from the weakrefs, and only three sampled
        # entries of the kernel are compared.  Hashing all NE**2 entries would
        # detect an in-place edit, but costs ~10 ms/call at the NE=1620
        # production grid against a 0.006 ms hit, i.e. it would erase the memo
        # (this helper is ~21% of a fig5-class solve when recomputed).
        #
        # So: editing a supported interior entry of a STILL-LIVE kernel between
        # calls is served from cache, and the round-5 canonicity detector does
        # not re-run.  No shipped caller mutates a kernel in place, and every
        # ctx-derived array is a read-only view, so this is unreachable in the
        # tree as it stands.
        #
        # This test pins the limitation rather than hiding it: if someone later
        # adds full content hashing, it fails and this comment plus the docstring
        # in phonon.py must be updated together.  The genuine defect this memo
        # was fixed for -- a recycled id serving a DEAD array's correction -- is
        # covered by test_recycled_kernel_address_does_not_serve_the_dead_array.
        ctx, K, idx_sum, n_omega = _memo_setup()
        before = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)
        assert np.any(before != 1.0)

        flat_edit = 1  # K[0, 1]; none of the sampled fingerprint slots
        assert flat_edit not in {0, K.size - 1, K.size // 2}
        assert ctx.active_mask[0] and ctx.K_plus.flat[flat_edit] > 0.0
        K.flat[flat_edit] = K.flat[flat_edit] * 3.7 + 1.0

        after = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)
        np.testing.assert_array_equal(after, before)  # served from cache

        # ...and the uncached path still disables the rescale, proving the
        # detector itself is intact and only the memo lookup short-circuits it.
        direct = _pair_breaking_quadrature_correction_impl(
            ctx, K, idx_sum, n_omega
        )
        np.testing.assert_array_equal(direct, 1.0)

    def test_in_place_omega_map_edit_is_a_documented_limitation(self) -> None:
        # Same deliberate trade-off as above: omega_idx_sum is keyed by shape
        # plus weakref identity, not by content, because summing it is O(NE**2).
        ctx, K, idx_sum, n_omega = _memo_setup()
        before = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)
        assert np.any(before != 1.0)

        idx_sum[...] = 0  # collapse every pair into one ω bin
        after = _pair_breaking_quadrature_correction(ctx, K, idx_sum, n_omega)
        np.testing.assert_array_equal(after, before)  # served from cache

        expected = _pair_breaking_quadrature_correction_impl(
            ctx, K, idx_sum, n_omega
        )
        assert not np.array_equal(expected, before)  # uncached path differs

    def test_recycled_kernel_address_does_not_serve_the_dead_array(self) -> None:
        # No mutation anywhere: build a canonical kernel, drop it, then build
        # a genuinely different non-canonical one. The allocator hands the new
        # array the freed address, so an id-keyed entry with no liveness guard
        # returns the dead kernel's correction.
        ctx, canonical, idx_sum, n_omega = _memo_setup()
        template = canonical.copy()
        template.flat[1] *= 1.5  # supported interior entry: breaks canonicity

        _pair_breaking_quadrature_correction(ctx, canonical, idx_sum, n_omega)
        dead_address = id(canonical)
        del canonical

        misses = []  # keep misses alive so the freed slot stays the LIFO head
        recycled = False
        for _ in range(64):
            perturbed = template.copy()
            correction = _pair_breaking_quadrature_correction(
                ctx, perturbed, idx_sum, n_omega
            )
            np.testing.assert_array_equal(correction, 1.0)
            if id(perturbed) == dead_address:
                recycled = True
                break
            misses.append(perturbed)
        assert recycled, "address never recycled; guard not exercised"

    def test_eviction_survives_more_live_entries_than_the_cap(self) -> None:
        ctx, K, idx_sum, n_omega = _memo_setup()
        kernels = [K * scale for scale in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)]
        for kernel in kernels:
            _pair_breaking_quadrature_correction(ctx, kernel, idx_sum, n_omega)

        assert len(_PB_QUAD_CORR_CACHE) <= 4


class TestKaplanCorrectionDirection:
    def test_face_aligned_production_grid_correction_never_exceeds_one(
        self,
    ) -> None:
        # The docstring used to claim a 2/π under-count (a π/2 ≈ 1.57 rescale).
        # On the energy_min_factor=1.0 production family the finite-volume sum
        # OVER-counts the collapsed threshold interval by 4/π, so the applied
        # factor tends to π/4 ≈ 0.785 and no bin is rescaled upward.
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=405,
        )
        ctx = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap
        )
        omega, _, idx_sum, _ = build_phonon_frequency_map(E)
        K = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=TAU_0_PB)

        correction = _pair_breaking_quadrature_correction(
            ctx, K, idx_sum, int(omega.size)
        )
        changed = correction != 1.0

        assert np.any(changed)
        assert np.all(correction[changed] < 1.0)
        threshold_idx = int(np.argmax(changed))
        assert correction[threshold_idx] == pytest.approx(np.pi / 4.0, rel=5e-3)


class TestEffectiveKernelOverrideContract:
    def _setup(self) -> tuple[np.ndarray, SpectralContext, np.ndarray, np.ndarray]:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=4.0,
            num_energy_bins=24,
        )
        ctx = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap
        )
        K_s0 = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        K_r0 = build_recombination_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        return np.full(E.size, 0.01), ctx, K_s0, K_r0

    def test_rejects_broadcastable_wrong_shape_eff_override(self) -> None:
        # A 1-D (NE,) override broadcasts through both contractions and
        # returns a silently wrong loss rate; the sibling N_*_override
        # arguments have rejected exactly this shape all along.
        f, ctx, K_s0, K_r0 = self._setup()
        with pytest.raises(ValueError, match=r"K_s_eff_override must have shape"):
            phonon_collision_rates(
                f, ctx, K_s0, K_r0, 0.2,
                K_s_eff_override=np.ones(ctx.E.size),
            )

    def test_rejects_unpaired_recombination_eff_overrides(self) -> None:
        f, ctx, K_s0, K_r0 = self._setup()
        with pytest.raises(ValueError, match=r"supplied together"):
            phonon_collision_rates(
                f, ctx, K_s0, K_r0, 0.2,
                K_r_emit_override=K_r0.copy(),
            )

    def test_correct_overrides_stay_bit_identical(self) -> None:
        f, ctx, K_s0, K_r0 = self._setup()
        T_bath = 0.2
        gain, loss = phonon_collision_rates(f, ctx, K_s0, K_r0, T_bath)

        from qpsim.collisions.phonon import (
            _thermal_phonon_recombination_occupations,
            _thermal_phonon_scattering_occupation,
        )

        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        gain_eff, loss_eff = phonon_collision_rates(
            f, ctx, K_s0, K_r0, T_bath,
            K_s_eff_override=K_s0 * N_p,
            K_r_emit_override=K_r0 * N_emit,
            K_r_abs_override=K_r0 * N_abs,
        )

        np.testing.assert_array_equal(gain_eff, gain)
        np.testing.assert_array_equal(loss_eff, loss)
