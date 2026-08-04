"""Fast invariants for the Kupriyanov-Lukichev interface-trap benchmark."""

from __future__ import annotations

import pytest

from validation.diffusion_operators.interface_trap import InterfaceResult, run


@pytest.fixture(scope="module")
def interface_result() -> InterfaceResult:
    """Share the expensive linear relaxation across its independent gates."""
    return run(NE=12, NX=25, relax_steps=2000)


def test_current_continuous_across_interface(
    interface_result: InterfaceResult,
) -> None:
    result = interface_result
    for name, (f_left, f_int, f_right) in result.currents.items():
        assert abs(f_left - f_int) / abs(f_int) < 1e-8, (name, f_left, f_int)
        assert abs(f_right - f_int) / abs(f_int) < 1e-8, (name, f_right, f_int)


def test_f_discontinuity_dominates_bulk_and_matches_KL(
    interface_result: InterfaceResult,
) -> None:
    result = interface_result
    for name in result.interface_jump:
        # Resistive interface: the jump dwarfs a single bulk-cell drop ...
        assert result.interface_jump[name] > 5.0 * result.bulk_drop[name]
        # ... and equals the bulk current / interface conductance.
        assert abs(result.interface_jump[name] - result.jump_predicted[name]) < 1e-7


def test_a1_a2_distinct_closed_equilibria(
    interface_result: InterfaceResult,
) -> None:
    # The driven steady state cannot see p; the closed relaxation can.
    #
    # "They differ by > 1e-3" alone is not a gate on p: at this energy the
    # equilibrium p separation (2.13e-3) is no larger than the *transient*
    # difference between two operators that share p (A1 vs A1P at 200 relax
    # steps: 2.17e-3), so an unconverged relaxation passes a bare
    # difference check for the wrong reason. Certify each equilibrium
    # against its own closed form instead.
    result = interface_result
    NX = result.x.size
    n_hi = NX // 2  # cells at gap_hi; see interface_trap.run
    n_lo = NX - n_hi
    f_inject = 0.4  # f0[ed, :half] in interface_trap.run
    for name, p in (("A1", 1), ("A2", 2)):
        profile = result.relax_profiles[name]
        # (a) Zero-flux equilibrium is f uniform in x. This is what
        # certifies the relaxation actually converged.
        assert float(profile.max() - profile.min()) < 1e-4, name
        # (b) ... at the constant fixed by the conserved measure N_1^p f,
        # f_eq = Σ_j N1_j^p f0_j / Σ_j N1_j^p, which is where the p
        # dressing enters: 0.1941255 (A1) vs 0.1962523 (A2). N1 is
        # piecewise constant, so the two cell values suffice.
        w_hi = n_hi * result.N1_left**p
        w_lo = n_lo * result.N1_right**p
        f_eq = f_inject * w_hi / (w_hi + w_lo)
        assert float(profile.mean()) == pytest.approx(f_eq, abs=1e-5), name
    # Distinct equilibria: the p separation is ~200x the tolerance above.
    assert result.relax_a1_a2_maxdiff > 1e-3
