"""Fast invariants for the gap-gradient drift benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators import (
    BENCHMARK_MODELS,
    D0_DEFAULT,
    drift_coefficient,
    exact_initial_drift,
)
from validation.diffusion_operators.gap_gradient_drift import run


def test_drift_coefficient_reduces_to_own_moment_law() -> None:
    # Reading each operator on its own conserved density (s = p) recovers
    # the operator-moment law D_N q N_1^{q-p-1} d_x N_1.
    for model in BENCHMARK_MODELS:
        assert drift_coefficient(model.p, model.q, s=model.p) == model.q
    # On the common quasiparticle density N_1 f (s = 1) the prefactor is
    # q + 2(1 - p): A1, A2, B null; C +1; A1P +2.
    by_name = {m.name: m for m in BENCHMARK_MODELS}
    assert drift_coefficient(by_name["A1"].p, by_name["A1"].q) == 0
    assert drift_coefficient(by_name["A2"].p, by_name["A2"].q) == 0
    assert drift_coefficient(by_name["B"].p, by_name["B"].q) == 0
    assert drift_coefficient(by_name["C"].p, by_name["C"].q) == 1
    assert drift_coefficient(by_name["A1P"].p, by_name["A1P"].q) == 2


def test_exact_initial_drift_reduces_to_own_moment_law() -> None:
    # For s = p the shape term vanishes and the exact initial rate is the
    # closed form <q D_N N_1^{q-p-1} d_x N_1> averaged over the conserved
    # density -- the supplement's own-moment law.
    x = np.linspace(0.0, 100.0, 201)
    gap = np.linspace(1.0, 1.6, x.size)
    E = np.array([1.7, 2.5, 4.0])
    N1 = E[:, None] / np.sqrt(E[:, None] ** 2 - gap[None, :] ** 2)
    f0 = np.tile(np.exp(-(((x - 50.0) / 6.0) ** 2)), (E.size, 1))
    dN1 = np.gradient(N1, x, axis=1)
    for model in BENCHMARK_MODELS:
        p, q = model.p, model.q
        exact = exact_initial_drift(f0, N1, x, p, q, s=p, D0=D0_DEFAULT)
        u0 = np.power(N1, p) * f0
        closed = np.sum(
            D0_DEFAULT * q * np.power(N1, q - p - 1) * dN1 * u0, axis=1
        ) / np.sum(u0, axis=1)
        assert np.allclose(exact, closed, rtol=1e-3, atol=1e-9), model.name


def test_quasiparticle_density_drift_splits_by_readout_law() -> None:
    # Read on N_1 f: A1P (+2) and the legacy C (+1) drift up the gap
    # gradient; A1, A2 and B (0) carry no net drift beyond the
    # finite-packet/discretization residual.
    result = run(NE=12, NX=31, n_steps=8)
    assert np.all(result.drift_measured["A1P"] > 0.0)
    assert np.all(result.drift_measured["C"] > 0.0)
    a1p = float(np.max(np.abs(result.drift_measured["A1P"])))
    for name in ("A1", "A2", "B"):
        residual = float(np.max(np.abs(result.drift_measured[name])))
        assert residual < 0.05 * a1p, (name, residual, a1p)


def test_a1p_drift_exceeds_c() -> None:
    # v_A1P / v_C = 2 N_1^2 > 1 at every energy.
    result = run(NE=12, NX=31)
    assert np.all(result.drift_measured["A1P"] > result.drift_measured["C"])


def test_drift_matches_analytic_velocity() -> None:
    result = run(NE=12, NX=31)
    a1p = float(np.max(np.abs(result.drift_analytic["A1P"])))
    # A1: its undressed flux integrates to a boundary term -- null to round-off.
    assert float(np.max(np.abs(result.drift_analytic["A1"]))) < 1e-12 * a1p
    # A2, B: null at leading order; the exact rate keeps only a small shape term.
    for name in ("A2", "B"):
        assert float(np.max(np.abs(result.drift_analytic[name]))) < 0.05 * a1p, name
    for name in ("A1P", "C"):
        measured = result.drift_measured[name]
        analytic = result.drift_analytic[name]
        mask = np.abs(analytic) > 1e-3
        rel = np.abs(measured[mask] - analytic[mask]) / np.abs(analytic[mask])
        assert np.max(rel) < 0.15, (name, float(np.max(rel)))
