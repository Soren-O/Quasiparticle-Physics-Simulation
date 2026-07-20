"""Tests for M25GapAsymmetricJJ — Layer-2 wrap of Stage A M25 physics.

Phases 5-5c of the Device Architecture: the M25 gap-asymmetric
Josephson junction is a first-class Device-level Junction subclass
that wraps the Stage A coefficient evaluators inside the Layer-2
``Junction.evaluate(state_a, state_b, qubit_state) -> JunctionResult``
contract.

Coverage:

* Construction / composition / API contract tests
  (TestM25JunctionConstruction, TestM25JunctionEvaluate,
  TestM25JunctionInDevice)
* Defensive validation: gap mismatches, missing R sub-bands,
  missing L band (TestM25JunctionEvaluateValidation)
* Phase 5b architectural pin: M25 owns dissipation, Device routes
  external_dissipation_only, multiple-owner rejection
  (TestM25NoDoubleCounting)
* Phase 5c quantitative pin: M25 Fig 3a x_L / x_R< / x_R> / p_1
  match the correctly-normalized qpsim root to ≤10% rtol via the
  cached moment-solver fixed point (test_fig3a_quantitative_match);
  the reference values are qpsim roots consistent with the paper's
  figure-reading precision, not digitized paper data

Per the design doc §6.1 caveat the underlying Stage A evaluators
remain a moment closure (Fermi-Dirac per-sub-band ansatz). A
``KineticJunction`` operating directly on f(E) would drop that
assumption; deferred unless / until M25 Fig 4/5 reproduction
requires it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import (
    Device,
    M25GapAsymmetricJJ,
    Qubit,
    Region,
    solve_device_steady_state,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.spectral import SpectralContext
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
)

_H_OVER_KB = 4.799243e-11   # K / Hz


# ═══════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════


def _build_region_state(
    *, T_bath: float, gap_kelvin: float,
    num_energy: int = 30, energy_max_factor: float = 6.0,
    energy_min_factor: float = 1.0,
    second_gap_kelvin: float | None = None,
) -> T3DiffusionState:
    """Build a Fermi-Dirac thermal state on an energy grid above ``gap_kelvin``.

    If ``second_gap_kelvin`` is given (and exceeds ``gap_kelvin``),
    the grid is built piecewise: half the bins resolve the
    ``[gap, second_gap]`` window (the M25 R< band) and the rest cover
    ``[second_gap, energy_max_factor·gap]``. This is required for the R
    electrode in M25 setups where the L/R gap asymmetry is small enough
    that a uniform grid would put zero bins in R<.
    """
    gap_uev = gap_kelvin * KB_UEV_PER_K
    if second_gap_kelvin is not None and second_gap_kelvin > gap_kelvin:
        split_uev = second_gap_kelvin * KB_UEV_PER_K
        E_max = energy_max_factor * gap_uev
        n_lo = num_energy // 2
        n_hi = num_energy - n_lo
        # Cell-centered uniform sub-grids in each sub-band.
        lo_min = gap_uev
        dE_lo = (split_uev - lo_min) / float(n_lo)
        E_lo = lo_min + (np.arange(n_lo, dtype=float) + 0.5) * dE_lo
        dE_hi = (E_max - split_uev) / float(n_hi)
        E_hi = split_uev + (np.arange(n_hi, dtype=float) + 0.5) * dE_hi
        E = np.concatenate([E_lo, E_hi])
        # Preserve the intended finite-volume faces, especially the explicit
        # R</R> face at Delta_L. Reconstructing widths from the abruptly
        # changing centers assigns a huge transition cell to R< and shifts its
        # normalization by O(1).
        dE = np.concatenate([
            np.full(n_lo, dE_lo),
            np.full(n_hi, dE_hi),
        ])
    else:
        E, _ = build_energy_grid(
            gap=gap_uev, energy_min_factor=energy_min_factor,
            energy_max_factor=energy_max_factor, num_energy_bins=num_energy,
        )
        dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap_uev)
    omega_bins, _, _, _ = build_phonon_frequency_map(spectral.E)
    phonon = PhononState(
        n_ph=np.zeros((1, omega_bins.size, 1)),
        omega_bins=omega_bins.reshape(1, -1),
        tau_l=np.full((1, omega_bins.size), 0.25),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    # Cheap thermal Fermi-Dirac initial guess.
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    # Material: use Al as the carrier; we just need T_c, tau_0.
    mat = load_material("Al")
    # Override the gap on a copy of the material to match this region.
    custom_mat = replace(mat, Delta_0=gap_uev)
    return T3DiffusionState(
        f=f_init, gap=gap_uev, spectral=spectral, phonon=phonon,
        material=custom_mat, T_bath=T_bath,
    )


def _fig3a_setup() -> tuple[M25PhysicalParameters, M25PhotonDrive]:
    """M25 Fig 3a (small gap asymmetry) parameters + back-solved drive."""
    params = M25PhysicalParameters(
        Delta_L_kelvin=49.5e9 * _H_OVER_KB,
        Delta_R_kelvin=49.0e9 * _H_OVER_KB,
        omega_10_kelvin=5.5e9 * _H_OVER_KB,
        T_kelvin=0.020,
        E_J_kelvin=14.5e9 * _H_OVER_KB,
        E_C_kelvin=290e6 * _H_OVER_KB,
        R_T_Hz=8.0 * 14.5e9 * (49.25 / 49.5),
        r_L_Hz=6.25e6,
        r_Rlt_Hz=6.25e6,
        Gamma_ee_10_Hz=100e3,
    )
    drive_template = M25PhotonDrive(
        omega_nu_kelvin=119e9 * _H_OVER_KB,
        Gamma_nu_scale_Hz=1.0,
        nu_0_per_J_per_m3=0.73e47,
        volume_m3=506e-6 * 240e-6 * 0.028e-6,  # 3.4e-15 m³
    )
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, drive_template, 300.0,
    )
    drive = replace(drive_template, Gamma_nu_scale_Hz=scale)
    return params, drive


# ═══════════════════════════════════════════════════════════════════════
#  Construction + composition
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionConstruction:
    def test_rejects_self_loop(self) -> None:
        params, drive = _fig3a_setup()
        with pytest.raises(ValueError, match="couple two different regions"):
            M25GapAsymmetricJJ(
                name="J", region_a="X", region_b="X",
                m25_params=params, m25_drive=drive,
            )

    def test_constructs_with_valid_inputs(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        assert j.name == "JJ"
        # Coefficients are not computed until first evaluate.
        assert j._coefficients is None


# ═══════════════════════════════════════════════════════════════════════
#  Validation: gap mismatch + missing R sub-band
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionEvaluateValidation:
    def test_rejects_region_temperature_mismatch_with_closure(self) -> None:
        params, drive = _fig3a_setup()
        state_L = _build_region_state(
            T_bath=params.T_kelvin + 0.01,
            gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=params.T_kelvin,
            gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        junction = M25GapAsymmetricJJ(
            name="JJ",
            region_a="L",
            region_b="R",
            m25_params=params,
            m25_drive=drive,
        )

        with pytest.raises(ValueError, match="temperature used to build"):
            junction.evaluate(state_L, state_R)

    def test_bcs_flux_normalization_uses_exact_band_measures(self) -> None:
        from qpsim.devices.m25_junction import _spectral_band_weights

        params, _drive = _fig3a_setup()
        Delta_L_uev = params.Delta_L_kelvin * KB_UEV_PER_K
        Delta_R_uev = params.Delta_R_kelvin * KB_UEV_PER_K
        state_L = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020,
            gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )

        gain_L = 123.0
        ef_L = M25GapAsymmetricJJ._build_per_region_flux(
            state_L,
            gain_L,
            7.0,
            gap_uev=Delta_L_uev,
            lower_bound=Delta_L_uev,
            label="L",
        )
        weights_L = _spectral_band_weights(
            state_L.spectral, lower_bound=Delta_L_uev,
        )
        recovered_L = (
            2.0 / Delta_L_uev * np.sum(weights_L * ef_L.gain) * 1e9
        )
        assert recovered_L == pytest.approx(gain_L, rel=1e-14)

        # Reproduce the old midpoint normalization offset on the same grid.
        midpoint_L = np.sum(
            state_L.spectral.rho * state_L.spectral.dE
        )
        assert np.sum(weights_L) / midpoint_L - 1.0 == pytest.approx(
            0.03002234724567643,
            rel=1e-12,
        )

        gain_Rlt = 17.0
        gain_Rgt = 29.0
        ef_R = M25GapAsymmetricJJ._build_per_region_flux_two_band(
            state_R,
            gain_Rlt=gain_Rlt,
            loss_rate_Rlt=3.0,
            gain_Rgt=gain_Rgt,
            loss_rate_Rgt=5.0,
            Delta_R_uev=Delta_R_uev,
            Delta_L_uev=Delta_L_uev,
        )
        weights_Rlt = _spectral_band_weights(
            state_R.spectral,
            lower_bound=Delta_R_uev,
            upper_bound=Delta_L_uev,
        )
        weights_Rgt = _spectral_band_weights(
            state_R.spectral,
            lower_bound=Delta_L_uev,
        )
        recovered_Rlt = (
            2.0 / Delta_R_uev * np.sum(weights_Rlt * ef_R.gain) * 1e9
        )
        recovered_Rgt = (
            2.0 / Delta_R_uev * np.sum(weights_Rgt * ef_R.gain) * 1e9
        )
        assert recovered_Rlt == pytest.approx(gain_Rlt, rel=1e-14)
        assert recovered_Rgt == pytest.approx(gain_Rgt, rel=1e-14)

    def test_rejects_R_band_boundary_inside_a_cell(self) -> None:
        params, _drive = _fig3a_setup()
        Delta_L_uev = params.Delta_L_kelvin * KB_UEV_PER_K
        Delta_R_uev = params.Delta_R_kelvin * KB_UEV_PER_K
        state_R = _build_region_state(
            T_bath=0.020,
            gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        # Replace the finite-volume widths with Voronoi widths from centers.
        # Across the abrupt resolution change this creates one cell that
        # straddles Delta_L and cannot carry separate R</R> rates.
        bad_spectral = SpectralContext(
            E_bins=state_R.spectral.E,
            dE_bins=integration_widths_from_centers(state_R.spectral.E),
            gap=state_R.spectral.gap,
        )
        state_R = replace(state_R, spectral=bad_spectral)

        with pytest.raises(ValueError, match="must coincide with an energy-cell face"):
            M25GapAsymmetricJJ._build_per_region_flux_two_band(
                state_R,
                gain_Rlt=1.0,
                loss_rate_Rlt=1.0,
                gain_Rgt=1.0,
                loss_rate_Rgt=1.0,
                Delta_R_uev=Delta_R_uev,
                Delta_L_uev=Delta_L_uev,
            )

    def test_rejects_state_gap_mismatch_with_params(self) -> None:
        # If a region's spectral gap drifts away from the cached
        # coefficients' m25_params, evaluate must reject — otherwise
        # we mix rates for one junction with moments from another.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        # Build state_L 10% off the params Δ_L; state_R correct.
        wrong_L_K = 1.10 * params.Delta_L_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=wrong_L_K)
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))
        with pytest.raises(ValueError, match="does not match"):
            j.evaluate(state_L, state_R, qstate)

    def test_rejects_state_gap_vs_spectral_gap_drift(self) -> None:
        # T3DiffusionState carries `gap` and `spectral.gap` separately
        # and they must stay in sync. If only `spectral.gap` matches
        # m25_params, the per-bin DOS / kinetic kernels (which read
        # state.gap) silently disagree with the moment normalization.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        state_L_good = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        # Punch state.gap to 1.10·Δ_L while leaving spectral.gap alone.
        from dataclasses import replace as dc_replace
        state_L = dc_replace(
            state_L_good,
            gap=1.10 * float(state_L_good.spectral.gap),
        )
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))
        with pytest.raises(ValueError, match=r"state\.gap.*disagrees"):
            j.evaluate(state_L, state_R, qstate)

    def test_spectral_context_rejects_L_grid_with_no_active_band(self) -> None:
        from qpsim.physics.spectral import SpectralContext

        params, _drive = _fig3a_setup()
        Delta_L_uev = params.Delta_L_kelvin * KB_UEV_PER_K
        E_L_below = np.linspace(0.5 * Delta_L_uev, 0.95 * Delta_L_uev, 30)
        dE_L_below = integration_widths_from_centers(E_L_below)
        # The shared spectral invariant now catches this before an invalid
        # state can reach the M25-specific positive-support guard.
        with pytest.raises(ValueError, match="positive represented spectral capacity"):
            SpectralContext(
                E_bins=E_L_below, dE_bins=dE_L_below, gap=Delta_L_uev,
            )

    def test_L_spread_uses_band_mask_under_dynes_dos(self) -> None:
        # Without an explicit L band mask the spread normalizes over
        # the full grid. On a pure-BCS context that's harmless because
        # rho==0 below the gap; on a Dynes-broadened context the
        # subgap rho weight steals part of the normalization and the
        # recovered above-gap M25 gain falls below gain_moment_Hz.
        # Pin the moment-integral identity:
        #   (2/Δ_L) × ∫ rho × gain × dE  ==  gain_moment_Hz × 1e-9
        # (the 1e-9 converts the per-bin ExternalFlux from 1/ns to
        # 1/s for comparison against the Hz diagnostic).
        from dataclasses import replace as dc_replace

        from qpsim.physics.spectral import SpectralContext

        params, drive = _fig3a_setup()
        # Build an L state with a Dynes-broadened spectral context and
        # a grid that extends below Δ_L so subgap bins exist.
        Delta_L_uev = params.Delta_L_kelvin * KB_UEV_PER_K
        E_L = np.concatenate([
            np.linspace(0.5 * Delta_L_uev, 0.99 * Delta_L_uev, 8),
            np.linspace(1.01 * Delta_L_uev, 6.0 * Delta_L_uev, 22),
        ])
        from qpsim.grid.energy_grid import integration_widths_from_centers
        dE_L = integration_widths_from_centers(E_L)
        # Strong Dynes broadening so subgap rho is non-trivial.
        spec_dynes = SpectralContext(
            E_bins=E_L, dE_bins=dE_L, gap=Delta_L_uev,
            dynes_gamma=0.05 * Delta_L_uev,
        )
        state_L_normal = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_L = dc_replace(
            state_L_normal,
            f=np.zeros_like(E_L), spectral=spec_dynes,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))

        result = j.evaluate(state_L, state_R, qstate)
        ef_L = result.external_flux_a
        gain_moment_Hz = ef_L.diagnostics["gain_moment_Hz"]
        assert isinstance(gain_moment_Hz, float)
        # Recover the moment from the per-bin ExternalFlux. Multiply
        # by 1e9 to undo the Hz → 1/ns conversion in the spread.
        from qpsim.devices.m25_junction import _spectral_band_weights

        band_weights = _spectral_band_weights(
            spec_dynes, lower_bound=Delta_L_uev,
        )
        recovered_Hz = (
            2.0 / Delta_L_uev
            * float(np.sum(band_weights * ef_L.gain))
            * 1e9
        )
        np.testing.assert_allclose(recovered_Hz, gain_moment_Hz, rtol=1e-12)

    def test_spectral_context_accepts_L_grid_with_gap_cut_boundary_cell(self) -> None:
        from qpsim.physics.spectral import SpectralContext

        params, _drive = _fig3a_setup()
        Delta_L_uev = params.Delta_L_kelvin * KB_UEV_PER_K
        E_L = np.concatenate([
            np.linspace(0.5 * Delta_L_uev, 0.95 * Delta_L_uev, 30),
            np.array([Delta_L_uev]),
        ])
        dE_L = integration_widths_from_centers(E_L)
        spectral = SpectralContext(
            E_bins=E_L, dE_bins=dE_L, gap=Delta_L_uev,
        )
        assert spectral.rho[-1] == 0.0
        assert spectral.cell_weights[-1] > 0.0
        assert spectral.active_mask[-1]

    def test_rejects_R_grid_missing_Rgt(self) -> None:
        # If the R grid never reaches Δ_L, the R> sub-band is empty
        # and the M25 channels involving x_R> would silently zero.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=params.Delta_L_kelvin)
        # Cap the R grid below Δ_L → mask_Rgt is all-False.
        # Δ_L/Δ_R ≈ 1.0102 → max factor 1.005 < that ratio.
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            energy_min_factor=1.0, energy_max_factor=1.005,
        )
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))
        with pytest.raises(ValueError, match="R> sub-band"):
            j.evaluate(state_L, state_R, qstate)

    def test_rejects_R_grid_missing_Rlt(self) -> None:
        # Conversely, if the R grid starts above Δ_L the R< sub-band
        # is empty and the silent-zero would hide x_R<-driven channels.
        # Δ_L/Δ_R ≈ 1.0102, so energy_min_factor=1.05 puts every bin
        # above Δ_L while still using Δ_R as the declared gap.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=params.Delta_L_kelvin)
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            energy_min_factor=1.05,
        )
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))
        with pytest.raises(ValueError, match=r"does not cover.*BCS lower bound"):
            j.evaluate(state_L, state_R, qstate)


# ═══════════════════════════════════════════════════════════════════════
#  Junction.evaluate produces sensible outputs
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionEvaluate:
    def test_evaluate_returns_correct_structure(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        # Build region states matching the M25 gaps.
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K, second_gap_kelvin=Delta_L_K)
        # Need a qubit_state for the channels' p_0/p_1 weighting.
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[0.5, 0.5], [0.0, 0.0]]))

        result = j.evaluate(state_L, state_R, qstate)

        # Per-region fluxes have correct shape.
        assert result.external_flux_a.gain.shape == state_L.f.shape
        assert result.external_flux_b.gain.shape == state_R.f.shape
        # Non-empty qubit channels.
        assert len(result.qubit_channels) > 0
        # All channel rates are non-negative and finite.
        for ch in result.qubit_channels:
            assert ch.rate_per_ns >= 0.0
            assert np.isfinite(ch.rate_per_ns)

    def test_evaluate_caches_coefficients(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K, second_gap_kelvin=Delta_L_K)
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))

        # First call computes and caches.
        assert j._coefficients is None
        j.evaluate(state_L, state_R, qstate)
        assert j._coefficients is not None
        cached = j._coefficients
        # Second call reuses the same object.
        j.evaluate(state_L, state_R, qstate)
        assert j._coefficients is cached

    def test_cached_coefficient_arrays_are_read_only(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )

        coefficients = j._ensure_coefficients_cached()

        assert not coefficients.gammas_L.flags.writeable
        assert not coefficients.g_ph_Rlt_per_state.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            coefficients.gammas_L[0, 0] = 0.0

    @pytest.mark.parametrize(
        ("input_name", "mutate", "rebuilds_coefficients"),
        [
            (
                "m25_params",
                lambda j: setattr(
                    j,
                    "m25_params",
                    replace(j.m25_params, T_kelvin=1.01 * j.m25_params.T_kelvin),
                ),
                True,
            ),
            (
                "m25_drive",
                lambda j: setattr(
                    j,
                    "m25_drive",
                    replace(
                        j.m25_drive,
                        Gamma_nu_scale_Hz=1.1 * j.m25_drive.Gamma_nu_scale_Hz,
                    ),
                ),
                True,
            ),
            (
                "branch_picker_mode",
                lambda j: setattr(j, "branch_picker_mode", "lock_to_preferred"),
                False,
            ),
            (
                "expected_ordering",
                lambda j: setattr(j, "expected_ordering", ("x_L", "x_Rlt")),
                False,
            ),
        ],
    )
    def test_mutating_cache_input_rebuilds_affected_physics(
        self,
        monkeypatch: pytest.MonkeyPatch,
        input_name: str,
        mutate,
        rebuilds_coefficients: bool,
    ) -> None:
        """Every mutable cache input invalidates coefficients and/or moments."""
        import qpsim.devices.m25_junction as m25_junction_module

        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )

        coefficient_calls = 0
        moment_calls = 0

        def coefficient_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal coefficient_calls
            coefficient_calls += 1
            return object()

        def moment_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal moment_calls
            moment_calls += 1
            return object()

        monkeypatch.setattr(
            m25_junction_module,
            "coefficients_from_physical_parameters_with_photon_drive",
            coefficient_spy,
        )
        monkeypatch.setattr(
            m25_junction_module,
            "solve_rate_equation_steady_state_multi_seed",
            moment_spy,
        )

        j._ensure_moment_solution_cached()
        first_coefficients = j._coefficients
        first_moment = j._moment_solution

        mutate(j)
        j._ensure_moment_solution_cached()

        expected_coefficient_calls = 2 if rebuilds_coefficients else 1
        assert coefficient_calls == expected_coefficient_calls, input_name
        assert moment_calls == 2, input_name
        if rebuilds_coefficients:
            assert j._coefficients is not first_coefficients
        else:
            assert j._coefficients is first_coefficients
        assert j._moment_solution is not first_moment

        # Once rebuilt for the new values, an unchanged third call is a cache hit.
        rebuilt_coefficients = j._coefficients
        rebuilt_moment = j._moment_solution
        j._ensure_moment_solution_cached()
        assert coefficient_calls == expected_coefficient_calls
        assert moment_calls == 2
        assert j._coefficients is rebuilt_coefficients
        assert j._moment_solution is rebuilt_moment

    def test_value_fingerprint_detects_in_place_bundle_tampering(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The key stores values, not a reference to the nominally frozen bundle."""
        import qpsim.devices.m25_junction as m25_junction_module

        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        built = []

        def coefficient_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = object()
            built.append(result)
            return result

        monkeypatch.setattr(
            m25_junction_module,
            "coefficients_from_physical_parameters_with_photon_drive",
            coefficient_spy,
        )

        assert j._ensure_coefficients_cached() is not None
        object.__setattr__(
            drive,
            "Gamma_nu_scale_Hz",
            1.1 * drive.Gamma_nu_scale_Hz,
        )
        assert j._ensure_coefficients_cached() is not None
        assert len(built) == 2
        assert built[0] is not built[1]

    def test_emits_eo_and_ee_channels(self) -> None:
        # M25 has both parity-flipping (eo, from QP tunneling and
        # photon-assisted) and parity-preserving (ee) channels.
        # Verify at least one of each appears in the output.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K, second_gap_kelvin=Delta_L_K)
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))

        result = j.evaluate(state_L, state_R, qstate)
        eo_channels = [c for c in result.qubit_channels if c.flips_parity]
        ee_channels = [c for c in result.qubit_channels if not c.flips_parity]
        assert len(eo_channels) > 0
        assert len(ee_channels) > 0  # Γ̃^ee_10 = 100 kHz at Fig 3 caption


# ═══════════════════════════════════════════════════════════════════════
#  Composition with Device + Qubit (architectural contract)
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionInDevice:
    def test_composes_in_device_with_qubit(self) -> None:
        # Build full Device(L, R, M25Junction, Qubit) and verify it
        # composes structurally (no validation errors).
        params, drive = _fig3a_setup()
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        omega_10_K = params.omega_10_kelvin

        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K, second_gap_kelvin=Delta_L_K)
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                M25GapAsymmetricJJ(
                    name="JJ", region_a="L", region_b="R",
                    m25_params=params, m25_drive=drive,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=True,
                omega_kelvin=np.array([0.0, omega_10_K]),
                E_J_kelvin=params.E_J_kelvin,
                E_C_kelvin=params.E_C_kelvin,
            ),
        )
        # Just constructing should not raise.
        assert "L" in device.regions
        assert "R" in device.regions
        assert device.qubit is not None
        assert len(device.junctions) == 1

    def test_rejects_another_junction_sharing_an_m25_region(self) -> None:
        # M25 evaluate() owns an isolated two-electrode moment fixed point and
        # intentionally ignores evolving state.f. A second junction changing L
        # would therefore never feed back into that fixed point.
        from qpsim.devices import SymmetricGapTunnelingJunction

        params, drive = _fig3a_setup()
        state_L = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        state_X = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )

        with pytest.raises(ValueError, match="requires exclusive regions"):
            Device(
                regions={
                    "L": Region(name="L", state=state_L),
                    "R": Region(name="R", state=state_R),
                    "X": Region(name="X", state=state_X),
                },
                junctions=[
                    M25GapAsymmetricJJ(
                        name="M25", region_a="L", region_b="R",
                        m25_params=params, m25_drive=drive,
                    ),
                    SymmetricGapTunnelingJunction(
                        name="extra", region_a="L", region_b="X",
                        alpha_per_ns=1e-3,
                    ),
                ],
            )

    def test_device_solve_converges_with_finite_outputs(self) -> None:
        # Smoke test that the M25 junction actually drives the
        # composed Device → Qubit solve to convergence with finite
        # f(E) and a populated qubit_state. This pins the transcript
        # claim that Phase 5 v1 composes end-to-end; quantitative Fig
        # 3 reproduction is Phase 5c (see module docstring).
        params, drive = _fig3a_setup()
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        omega_10_K = params.omega_10_kelvin

        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=Delta_R_K, second_gap_kelvin=Delta_L_K,
        )
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                M25GapAsymmetricJJ(
                    name="JJ", region_a="L", region_b="R",
                    m25_params=params, m25_drive=drive,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=True,
                omega_kelvin=np.array([0.0, omega_10_K]),
                E_J_kelvin=params.E_J_kelvin,
                E_C_kelvin=params.E_C_kelvin,
            ),
        )
        sol = solve_device_steady_state(device, outer_tol=1e-9)
        assert sol.final_max_delta_f < 1e-6
        for region_name in ("L", "R"):
            f = sol.states[region_name].f
            assert np.all(np.isfinite(f))
            assert np.all(f >= 0.0)
            assert np.all(f <= 1.0)
        assert sol.qubit_state is not None
        p = sol.qubit_state.p
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0.0)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 5b: no-double-counting between e-ph kernel and M25's r_α/g^{pn}_α
# ═══════════════════════════════════════════════════════════════════════


class TestM25NoDoubleCounting:
    def test_owns_region_dissipation_is_set(self) -> None:
        # Architectural sentinel: the Device solver routes
        # external_dissipation_only=True only when this flag is set.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        assert j.owns_region_dissipation is True

    def test_device_drives_x_L_well_above_thermal(self) -> None:
        # Before Phase 5b, the inner T3 e-ph kernel forced f(E) to the
        # bath Fermi-Dirac, so x_L collapsed to ~exp(-Δ_L/T_bath) ≈
        # 1e-52 at Δ_L ≈ 2.4 K, T_bath = 20 mK regardless of the M25
        # photon drive. With the no-double-counting wiring, the M25
        # ExternalFlux owns dissipation and x_L sits at the M25 fixed
        # point — orders of magnitude above the thermal floor.
        # This is the architectural pin, not a quantitative Fig 3
        # match (that's Phase 5c, blocked on moment-coupled Picard).
        params, drive = _fig3a_setup()
        state_L = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                M25GapAsymmetricJJ(
                    name="JJ", region_a="L", region_b="R",
                    m25_params=params, m25_drive=drive,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=True,
                omega_kelvin=np.array([0.0, params.omega_10_kelvin]),
                E_J_kelvin=params.E_J_kelvin,
                E_C_kelvin=params.E_C_kelvin,
            ),
        )
        sol = solve_device_steady_state(device, outer_tol=1e-9)

        from qpsim.devices.m25_junction import _moment_x_M25
        f_L = sol.states["L"].f
        spec_L = sol.states["L"].spectral
        x_L = _moment_x_M25(
            f_L, spec_L, gap_alpha_uev=spec_L.gap,
        )
        # Thermal floor exp(-Δ_L/T) ≈ 1e-52. Phase 5b lifts x_L to
        # the photon-driven M25 fixed point, ~5e-18 at Fig 3a inputs
        # (cross-tunneling cycle quiescent). 1e-30 is comfortably
        # above thermal and below the M25 floor — proves the e-ph
        # kernel is not crushing x_L back to thermal.
        assert x_L > 1e-30, (
            f"x_L = {x_L:.3e} is at or below the thermal floor — "
            "e-ph kernel may be running inside the inner solve."
        )

    def test_fig3a_quantitative_match(self) -> None:
        # Phase 5c: end-to-end M25 Fig 3a reproduction. The Junction
        # caches a moment-solver fixed point on first evaluate and
        # uses it (instead of state-derived moments) to build the
        # per-bin (gain, loss_rate) — sidestepping the cross-tunneling
        # bootstrap problem that stalled the Phase 5b Picard.
        #
        # Reference values come from running the M25 4-unknown moment
        # solver at Fig 3a parameters (Δ_L = 49.5 GHz, Δ_R = 49.0 GHz,
        # ω_10 = 5.5 GHz, T = 20 mK, Γ̃^ph_00 = 300 Hz). They sit on
        # the M25 paper's Fig 3a curve at the photon-power cited in
        # the caption.
        params, drive = _fig3a_setup()
        state_L = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_R_kelvin,
            second_gap_kelvin=params.Delta_L_kelvin,
        )
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                M25GapAsymmetricJJ(
                    name="JJ", region_a="L", region_b="R",
                    m25_params=params, m25_drive=drive,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=True,
                omega_kelvin=np.array([0.0, params.omega_10_kelvin]),
                E_J_kelvin=params.E_J_kelvin,
                E_C_kelvin=params.E_C_kelvin,
            ),
        )
        sol = solve_device_steady_state(device, outer_tol=1e-9)

        from qpsim.devices.m25_junction import _moment_x_M25
        Delta_L_uev = float(state_L.spectral.gap)
        Delta_R_uev = float(state_R.spectral.gap)
        f_L = sol.states["L"].f
        spec_L = sol.states["L"].spectral
        x_L = _moment_x_M25(f_L, spec_L, gap_alpha_uev=Delta_L_uev)

        spec_R = sol.states["R"].spectral
        f_R = sol.states["R"].f
        x_Rlt = _moment_x_M25(
            f_R,
            spec_R,
            gap_alpha_uev=Delta_R_uev,
            lower_bound=Delta_R_uev,
            upper_bound=Delta_L_uev,
        )
        x_Rgt = _moment_x_M25(
            f_R,
            spec_R,
            gap_alpha_uev=Delta_R_uev,
            lower_bound=Delta_L_uev,
        )
        p = sol.qubit_state.p
        p_per_level = p.sum(axis=1) if p.ndim == 2 else p
        p_1 = float(p_per_level[1])

        # Reference values: the unique physical root of the correctly
        # normalized M25 system (Γ̄ = Γ̃ / N_CP(R) in the density
        # equations, M25 text below Eq. 6) at Fig 3a, T = 20 mK, with
        # the exact S48/S49 tau_R quadrature (2026-07-19 audit H4; the
        # old 5.583/2.018/4.696e-8 pins were calibrated to the
        # out-of-domain S50 series).
        # These match the paper's own small-asymmetry description
        # (Sec. II.4: x_L ≈ x_R> + x_R< ≈ √(g^ph_R / r^L) ≈ 6e-8).
        # The historical pins (x_L = 1.5e-5 etc.) were sub-1-Hz
        # pseudo-roots on the recombination slope, admitted by the
        # legacy Γ̄ = Γ̃ normalization plus the max-x_L picker; they
        # are gone with the normalization fix. 10% tolerance covers
        # the moment-extraction discretization of the region grid.
        np.testing.assert_allclose(x_L,   5.313e-08, rtol=1e-1)
        np.testing.assert_allclose(x_Rgt, 1.802e-08, rtol=1e-1)
        np.testing.assert_allclose(x_Rlt, 5.130e-08, rtol=1e-1)
        np.testing.assert_allclose(p_1,   8.330e-04, rtol=1e-1)

        # Plotted observable check: μ_L/Δ_L at T = 20 mK. The paper
        # figure (arXiv 2408.17218 Fig 3a) shows the merged μ_α
        # curves passing ≈ 0.87 at 20 mK on their way from ≈ 0.94
        # (10 mK) to 0 at T̄ ≈ 0.146 K. With the leading-order
        # inversion μ_L = Δ_L + T·log(x_L) (paper footnote 5) the
        # root gives 0.864; the full Eq. (11) inversion gives 0.872.
        T_kelvin = 0.020
        from qpsim.constants import KB_UEV_PER_K
        mu_L_uev = Delta_L_uev + KB_UEV_PER_K * T_kelvin * np.log(x_L)
        mu_L_over_Delta_L = mu_L_uev / Delta_L_uev
        np.testing.assert_allclose(mu_L_over_Delta_L, 0.864, atol=0.015)

    def test_two_dissipation_owners_per_region_rejected(self) -> None:
        # The Device solver enforces "at most one Junction per region
        # claims dissipation ownership" — multiple owners would each
        # supply a complete dissipation flux and the sum over-counts.
        from qpsim.devices.external_flux import ExternalFlux
        from qpsim.devices.junction import Junction, JunctionResult

        class _StubOwningJunction(Junction):
            owns_region_dissipation = True

            def __init__(self, name: str, region_a: str, region_b: str) -> None:
                self.name = name
                self.region_a = region_a
                self.region_b = region_b

            def evaluate(self, state_a, state_b, qubit_state=None):  # type: ignore[no-untyped-def, override]
                ef = ExternalFlux(
                    gain=np.zeros_like(state_a.f),
                    loss_rate=np.full_like(state_a.f, 1e-3),
                )
                return JunctionResult(external_flux_a=ef, external_flux_b=ef)

        params, _ = _fig3a_setup()
        state_L = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        state_R = _build_region_state(
            T_bath=0.020, gap_kelvin=params.Delta_L_kelvin,
        )
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                _StubOwningJunction("J1", "L", "R"),
                _StubOwningJunction("J2", "L", "R"),
            ],
        )
        with pytest.raises(ValueError, match="two junctions claiming"):
            solve_device_steady_state(device, outer_tol=1e-6)
