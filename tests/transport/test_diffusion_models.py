"""Unit tests for the DiffusionModel operator family.

Covers ``qpsim.transport.diffusion.base``: the ``(p, q)`` dressing table,
the legacy name aliases (including the A2-not-A1 "usadel" correction), the
density/flux/effective-rate weights, and the consistency of closure ``C``
with the legacy ``SpectralContext.D_E`` closure.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.physics.spectral import bcs_density_of_states
from qpsim.transport.diffusion.base import (
    DEFAULT_DIFFUSION_MODEL,
    DiffusionModel,
    density_weight,
    dressing_exponents,
    effective_rate,
    flux_weight,
    from_name,
)


def _N1(gap: float = 180.0, num: int = 40) -> np.ndarray:
    E = np.linspace(1.02 * gap, 4.0 * gap, num)
    return bcs_density_of_states(E, gap)


class TestDressingExponents:
    def test_pq_table(self) -> None:
        assert dressing_exponents(DiffusionModel.A1) == (1, 0)
        assert dressing_exponents(DiffusionModel.A1P) == (1, 2)
        assert dressing_exponents(DiffusionModel.A2) == (2, 2)
        assert dressing_exponents(DiffusionModel.C) == (0, -1)
        assert dressing_exponents(DiffusionModel.B) == (0, -2)

    def test_p_q_properties_match(self) -> None:
        for model in DiffusionModel:
            assert (model.p, model.q) == dressing_exponents(model)

    def test_default_is_a1(self) -> None:
        assert DEFAULT_DIFFUSION_MODEL is DiffusionModel.A1


class TestFromName:
    def test_member_names_case_insensitive(self) -> None:
        assert from_name("A1") is DiffusionModel.A1
        assert from_name("a1p") is DiffusionModel.A1P
        assert from_name(" c ") is DiffusionModel.C
        assert from_name("b") is DiffusionModel.B

    def test_legacy_aliases(self) -> None:
        assert from_name("LEGACY") is DiffusionModel.C
        assert from_name("boltzmann") is DiffusionModel.B
        # The old "usadel" is the rejected diagnostic A2, NOT the dirty
        # Usadel operator (which is A1).
        assert from_name("usadel") is DiffusionModel.A2

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown diffusion model"):
            from_name("not-a-model")


class TestWeights:
    def test_density_weight(self) -> None:
        N1 = _N1()
        np.testing.assert_allclose(density_weight(N1, 0), np.ones_like(N1))
        np.testing.assert_allclose(density_weight(N1, 1), N1)
        np.testing.assert_allclose(density_weight(N1, 2), N1**2)

    def test_effective_rate_table(self) -> None:
        N1 = _N1()
        D0 = 6.0
        np.testing.assert_allclose(effective_rate(D0, N1, DiffusionModel.A1), D0 / N1, rtol=1e-12)
        np.testing.assert_allclose(
            effective_rate(D0, N1, DiffusionModel.A1P), D0 * N1, rtol=1e-12
        )
        np.testing.assert_allclose(
            effective_rate(D0, N1, DiffusionModel.A2), D0 * np.ones_like(N1), rtol=1e-12
        )
        np.testing.assert_allclose(effective_rate(D0, N1, DiffusionModel.C), D0 / N1, rtol=1e-12)
        np.testing.assert_allclose(
            effective_rate(D0, N1, DiffusionModel.B), D0 / N1**2, rtol=1e-12
        )

    def test_a1_and_c_share_uniform_gap_rate(self) -> None:
        # The dirty-limit A1 and the clean/BRT closure C agree on the
        # uniform-gap rate D0/N1; they differ in conserved density and
        # inhomogeneous-gap structure, not in the rate.
        N1 = _N1()
        np.testing.assert_allclose(
            effective_rate(6.0, N1, DiffusionModel.A1),
            effective_rate(6.0, N1, DiffusionModel.C),
            rtol=1e-12,
        )

    def test_closure_C_matches_legacy_DE(self) -> None:
        # Legacy SpectralContext.D_E/D0 = sqrt(1-(gap/E)^2) = 1/N1 = closure C.
        gap = 180.0
        E = np.linspace(1.02 * gap, 4.0 * gap, 40)
        N1 = bcs_density_of_states(E, gap)
        legacy = np.sqrt(1.0 - (gap / E) ** 2)
        np.testing.assert_allclose(effective_rate(1.0, N1, DiffusionModel.C), legacy, rtol=1e-12)

    def test_a1_falls_and_a1p_rises_toward_gap(self) -> None:
        # E[0] is nearest the gap (N1 largest): the A1 rate falls there
        # (same suppression as C); the diagnostic A1P rises.
        N1 = _N1()
        a1 = effective_rate(6.0, N1, DiffusionModel.A1)
        a1p = effective_rate(6.0, N1, DiffusionModel.A1P)
        c = effective_rate(6.0, N1, DiffusionModel.C)
        assert a1[0] < a1[-1]
        assert a1p[0] > a1p[-1]
        assert c[0] < c[-1]

    def test_subgap_guard_keeps_finite(self) -> None:
        N1 = np.array([0.0, 2.0])  # first bin sub-gap (no states)
        for model in DiffusionModel:
            assert np.all(np.isfinite(effective_rate(1.0, N1, model)))
        # Zero and negative exponents go to zero (no states -> no
        # transport), not 1 or inf.
        assert effective_rate(1.0, N1, DiffusionModel.C)[0] == 0.0
        assert flux_weight(1.0, N1, -2)[0] == 0.0
        # q = 0 is the dirty-limit indicator D_L: 1 above the local gap
        # edge, 0 below it.
        np.testing.assert_allclose(flux_weight(3.0, N1, 0), [0.0, 3.0])
