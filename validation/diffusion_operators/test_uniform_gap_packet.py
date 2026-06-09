"""Fast invariants for the uniform-gap packet benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators.uniform_gap_packet import run


def test_deff_matches_analytic_per_model() -> None:
    result = run(NE=14, NX=21, dt=2.0, conservation_steps=10)
    for name in ("A1", "A2", "C", "B"):
        measured = result.deff_over_dn[name]
        analytic = result.analytic_over_dn[name]
        rel = np.max(np.abs(measured - analytic) / analytic)
        assert rel < 1e-9, (name, rel)


def test_gap_edge_ordering() -> None:
    # Nearest-gap bin has N_1 > 1, so N_1^{q-p} orders A1 > A2 = 1 > C > B.
    result = run(NE=14, NX=21)
    i = 0
    a1 = result.deff_over_dn["A1"][i]
    a2 = result.deff_over_dn["A2"][i]
    c = result.deff_over_dn["C"][i]
    b = result.deff_over_dn["B"][i]
    assert a1 > a2 > c > b
    assert abs(a2 - 1.0) < 1e-6


def test_n_qp_conserved_all_models() -> None:
    result = run(NE=14, NX=21, conservation_steps=30)
    for name in ("A1", "A2", "C", "B"):
        assert result.n_qp_rel_drift[name] < 1e-12, (name, result.n_qp_rel_drift[name])
