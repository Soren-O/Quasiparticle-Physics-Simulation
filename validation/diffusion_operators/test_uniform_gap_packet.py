"""Fast invariants for the uniform-gap packet benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators.uniform_gap_packet import run


def test_deff_matches_analytic_per_model() -> None:
    result = run(NE=14, NX=21, dt=2.0, conservation_steps=10)
    # Exercise the subcycled inversion rather than accidentally reducing this
    # integration check to the historical one-step CN formula.
    assert max(np.max(v) for v in result.transport_substeps.values()) > 1
    for name in ("A1", "A1P", "A2", "C", "B"):
        measured = result.deff_over_dn[name]
        analytic = result.analytic_over_dn[name]
        rel = np.max(np.abs(measured - analytic) / analytic)
        assert rel < 1e-9, (name, rel)


def test_gap_edge_ordering() -> None:
    # Nearest-gap bin is fully represented and has cell-average N_1 > 1,
    # so N_1^{q-p} orders A1P > A2 = 1 > A1 = C > B.
    result = run(NE=14, NX=21)
    i = 0
    a1 = result.deff_over_dn["A1"][i]
    a1p = result.deff_over_dn["A1P"][i]
    a2 = result.deff_over_dn["A2"][i]
    c = result.deff_over_dn["C"][i]
    b = result.deff_over_dn["B"][i]
    assert a1p > a2 > a1 > b
    assert abs(a2 - 1.0) < 1e-6
    # A1 and C share the uniform-gap rate D0/N1.
    np.testing.assert_allclose(a1, c, rtol=1e-12)


def test_n_qp_conserved_all_models() -> None:
    result = run(NE=14, NX=21, conservation_steps=30)
    for name in ("A1", "A1P", "A2", "C", "B"):
        assert result.n_qp_rel_drift[name] < 1e-12, (name, result.n_qp_rel_drift[name])
