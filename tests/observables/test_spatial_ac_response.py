from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.observables.spatial_ac_response import compute_current_weighted_ac_response
from qpsim.physics.spectral import SpectralContext


def _ctx() -> SpectralContext:
    gap = 180.0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=5.0,
        num_energy_bins=32,
    )
    return SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
    )


def test_uniform_spatial_response_matches_lumped_observables() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    f = 1e-3 * np.exp(-((ctx.E - 2.0 * ctx.gap) / (0.25 * ctx.gap)) ** 2)
    x_um = np.linspace(0.0, 10.0, 5)
    weights = np.ones_like(x_um)
    omega_0 = 22.0
    alpha = 0.08

    response = compute_current_weighted_ac_response(
        np.repeat(f[:, None], x_um.size, axis=1),
        f_ref,
        x_um,
        ctx,
        omega_0,
        alpha=alpha,
        current_weights=weights,
        full_current_integral_um=10.0,
    )

    assert response.frac_freq_shift == pytest.approx(compute_frequency_shift(
        f,
        f_ref,
        ctx,
        omega_0,
        alpha,
    ))
    assert response.qi_qp == pytest.approx(
        compute_quality_factor(f, ctx, omega_0, alpha)
    )


def test_full_current_integral_dilutes_short_strip_response() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    f = 1e-3 * np.exp(-((ctx.E - 2.0 * ctx.gap) / (0.25 * ctx.gap)) ** 2)
    x_um = np.linspace(0.0, 10.0, 5)
    weights = np.ones_like(x_um)

    strip_only = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        22.0,
        alpha=0.08,
        current_weights=weights,
        full_current_integral_um=10.0,
    )
    diluted = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        22.0,
        alpha=0.08,
        current_weights=weights,
        full_current_integral_um=20.0,
    )

    assert diluted.frac_freq_shift == 0.5 * strip_only.frac_freq_shift
    assert diluted.inverse_qi_qp == 0.5 * strip_only.inverse_qi_qp


def test_negative_sigma2_preserves_signed_active_response() -> None:
    ctx = _ctx()
    omega_0 = 5.0
    alpha = 0.2
    f_ref = np.zeros_like(ctx.E)
    f = 0.75 * np.exp(-(ctx.E - ctx.gap) / (10.0 * omega_0))
    x_um = np.array([0.5, 1.5])
    widths_um = np.ones_like(x_um)

    response = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        omega_0,
        alpha=alpha,
        current_weights=np.ones_like(x_um),
        full_current_integral_um=2.0,
        spatial_cell_widths_um=widths_um,
    )
    lumped_q = compute_quality_factor(f, ctx, omega_0, alpha)

    assert lumped_q < 0.0
    assert response.inverse_qi_qp == pytest.approx(1.0 / lumped_q)
    assert response.qi_qp == pytest.approx(lumped_q)


def test_negative_reference_sigma2_preserves_signed_frequency_shift() -> None:
    ctx = _ctx()
    omega_0 = 5.0
    alpha = 0.2
    f_ref = 0.75 * np.exp(-(ctx.E - ctx.gap) / (10.0 * omega_0))
    f = np.zeros_like(ctx.E)
    x_um = np.array([0.5, 1.5])
    widths_um = np.ones_like(x_um)

    response = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        omega_0,
        alpha=alpha,
        current_weights=np.ones_like(x_um),
        full_current_integral_um=2.0,
        spatial_cell_widths_um=widths_um,
    )
    _, sigma2 = compute_ac_conductivity(f, ctx, omega_0)
    _, sigma2_ref = compute_ac_conductivity(f_ref, ctx, omega_0)
    expected_relative = (sigma2 - sigma2_ref) / sigma2_ref

    assert sigma2_ref < 0.0
    assert response.relative_sigma2_change_current_weighted == pytest.approx(
        expected_relative
    )
    assert response.frac_freq_shift == pytest.approx(
        0.5 * alpha * expected_relative
    )


def test_zero_reference_sigma2_with_change_fails_loudly() -> None:
    ctx = _ctx()
    f = np.zeros_like(ctx.E)
    f_ref = np.zeros_like(ctx.E)
    f_ref[0] = 0.5

    with pytest.raises(ValueError, match="reference sigma2 is zero"):
        compute_current_weighted_ac_response(
            f,
            f_ref,
            np.array([0.5, 1.5]),
            ctx,
            5.0,
            alpha=0.2,
            current_weights=np.ones(2),
            full_current_integral_um=2.0,
            spatial_cell_widths_um=np.ones(2),
        )


def test_zero_weight_point_does_not_trigger_local_sigma2_failure() -> None:
    ctx = _ctx()
    f_ref = np.zeros((ctx.E.size, 2))
    f = np.zeros_like(f_ref)
    f_ref[0, 1] = 0.5

    response = compute_current_weighted_ac_response(
        f,
        f_ref,
        np.array([0.5, 1.5]),
        ctx,
        5.0,
        alpha=0.2,
        current_weights=np.array([1.0, 0.0]),
        full_current_integral_um=1.0,
        spatial_cell_widths_um=np.ones(2),
    )

    assert response.frac_freq_shift == pytest.approx(0.0)
    assert response.inverse_qi_qp == pytest.approx(0.0)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), -0.1, 1.1))
@pytest.mark.parametrize("argument", ("f", "f_ref"))
def test_invalid_occupations_fail_loudly(
    bad_value: float,
    argument: str,
) -> None:
    ctx = _ctx()
    arrays = {
        "f": np.zeros((ctx.E.size, 2)),
        "f_ref": np.zeros((ctx.E.size, 2)),
    }
    arrays[argument][0, 0] = bad_value

    with pytest.raises(ValueError, match=r"finite|occupations in \[0, 1\]"):
        compute_current_weighted_ac_response(
            arrays["f"],
            arrays["f_ref"],
            np.array([0.5, 1.5]),
            ctx,
            5.0,
            alpha=0.2,
            current_weights=np.ones(2),
            full_current_integral_um=2.0,
            spatial_cell_widths_um=np.ones(2),
        )


def test_zero_sigma2_with_nonzero_sigma1_fails_loudly() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    # At omega=5 micro-eV, the first stored occupation supplies f=1/2 over
    # the complete sub-gap sigma2 partner interval, so sigma2 is exactly zero.
    # Its step down into the next cell still produces finite sigma1.
    f = np.zeros_like(ctx.E)
    f[0] = 0.5
    x_um = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match="sigma2 is zero"):
        compute_current_weighted_ac_response(
            f,
            f_ref,
            x_um,
            ctx,
            5.0,
            alpha=0.2,
            current_weights=np.ones_like(x_um),
            full_current_integral_um=2.0,
            spatial_cell_widths_um=np.ones_like(x_um),
        )


def test_explicit_cell_measure_integrates_full_cell_center_domain() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    f = 1e-3 * np.exp(-((ctx.E - 2.0 * ctx.gap) / (0.25 * ctx.gap)) ** 2)
    n_x = 5
    dx_um = 2.0
    x_um = (np.arange(n_x, dtype=float) + 0.5) * dx_um
    widths = np.full(n_x, dx_um)

    response = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        22.0,
        alpha=0.08,
        current_weights=np.ones(n_x),
        full_current_integral_um=10.0,
        spatial_cell_widths_um=widths,
    )

    assert response.strip_current_integral_um == pytest.approx(10.0)
    assert response.current_participation == pytest.approx(1.0)
    assert response.frac_freq_shift == pytest.approx(
        compute_frequency_shift(f, f_ref, ctx, 22.0, 0.08)
    )


@pytest.mark.parametrize(
    ("x_um", "weights", "message"),
    [
        (np.array([0.0, np.nan]), np.ones(2), "x_um"),
        (np.array([0.0 + 1.0j, 1.0]), np.ones(2), "x_um"),
        (np.array([1.0, 0.0]), np.ones(2), "x_um"),
        (np.array([0.0, 1.0]), np.array([1.0, np.nan]), "current_weights"),
        (
            np.array([0.0, 1.0]),
            np.array([1.0 + complex(0.0, np.nan), 1.0]),
            "current_weights",
        ),
        (np.array([0.0, 1.0]), np.array([2.0, -1.0]), "current_weights"),
        (np.array([0.0, 1.0]), np.zeros(2), "current_weights"),
    ],
)
def test_invalid_spatial_measure_fails_loudly(
    x_um: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    ctx = _ctx()
    f = np.zeros_like(ctx.E)

    with pytest.raises(ValueError, match=message):
        compute_current_weighted_ac_response(
            f,
            f,
            x_um,
            ctx,
            22.0,
            alpha=0.08,
            current_weights=weights,
            full_current_integral_um=10.0,
        )


def test_complex_occupation_and_cell_measure_fail_loudly() -> None:
    ctx = _ctx()
    f = np.zeros_like(ctx.E)
    x_um = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match="f must be real-valued"):
        compute_current_weighted_ac_response(
            f.astype(complex) + 1.0j,
            f,
            x_um,
            ctx,
            22.0,
            alpha=0.08,
            current_weights=np.ones(2),
            full_current_integral_um=2.0,
        )

    with pytest.raises(
        ValueError,
        match="spatial_cell_widths_um must be real-valued",
    ):
        compute_current_weighted_ac_response(
            f,
            f,
            x_um,
            ctx,
            22.0,
            alpha=0.08,
            current_weights=np.ones(2),
            full_current_integral_um=2.0,
            spatial_cell_widths_um=np.array([1.0 + 2.0j, 1.0]),
        )
