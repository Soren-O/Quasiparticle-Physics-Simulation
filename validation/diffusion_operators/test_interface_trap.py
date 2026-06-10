"""Fast invariants for the Kupriyanov-Lukichev interface-trap benchmark."""

from __future__ import annotations

from validation.diffusion_operators.interface_trap import run


def test_current_continuous_across_interface() -> None:
    result = run(NE=12, NX=25, relax_steps=1500)
    for name, (f_left, f_int, f_right) in result.currents.items():
        assert abs(f_left - f_int) / abs(f_int) < 1e-8, (name, f_left, f_int)
        assert abs(f_right - f_int) / abs(f_int) < 1e-8, (name, f_right, f_int)


def test_f_discontinuity_dominates_bulk_and_matches_KL() -> None:
    result = run(NE=12, NX=25, relax_steps=1500)
    for name in result.interface_jump:
        # Resistive interface: the jump dwarfs a single bulk-cell drop ...
        assert result.interface_jump[name] > 5.0 * result.bulk_drop[name]
        # ... and equals the bulk current / interface conductance.
        assert abs(result.interface_jump[name] - result.jump_predicted[name]) < 1e-7


def test_a1_a2_distinct_closed_equilibria() -> None:
    # The driven steady state cannot see p; the closed relaxation can.
    result = run(NE=12, NX=25, relax_steps=2000)
    assert result.relax_a1_a2_maxdiff > 1e-3
