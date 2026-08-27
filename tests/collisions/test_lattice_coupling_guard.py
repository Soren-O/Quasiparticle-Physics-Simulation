"""The commensurability constraint is a property of the MODEL, not of the UI.

The phonon frequency grid is the union of two families: the differences
``|E_i - E_j|`` that scattering emits and absorbs on, and the sums
``E_i + E_j`` that pair breaking and recombination live on. One ``n(omega)``
is shared between them, but nothing downstream re-couples the bins. When the
two families share no bins above the pair threshold, scattering phonons there
can never break a pair and recombination phonons can never be reabsorbed, so
the two halves of the kinetic equation evolve on disjoint sublattices and
converge to DIFFERENT limits under refinement. That is an inconsistent
discretisation returning a confident number, not a coarse one.

This was checked only in the web-UI submission validator, which guards one
entry point rather than the model: a script, a validation campaign or a
notebook that builds the collision layer directly ran a decoupled grid to
completion. These tests pin the check at the layer, and pin the two ways it
must NOT overreach -- a commensurate grid must pass, and a pinned phonon bath
never couples the channels at all, so the constraint does not apply to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpsim.webui.execute import run_kinetics
from qpsim.webui.schemas import KineticsSetup

# 2*E_face/dE is an integer at 90 bins over [Delta, 4*Delta] and is not at 91.
COMMENSURATE = 90
DECOUPLED = 91


def _setup(num_bins: int, phonon_mode: str) -> KineticsSetup:
    setup = KineticsSetup()
    setup.geometry.rows = 1
    setup.geometry.cols = 1
    setup.phonons.mode = phonon_mode
    setup.grid.num_bins = num_bins
    setup.max_time = 0.05
    setup.snapshot_interval = 0.05
    return setup


def _run(setup: KineticsSetup) -> None:
    run_kinetics(setup, lambda *a, **k: None, lambda: False)


def test_a_decoupled_grid_is_refused_by_the_engine_not_only_the_form() -> None:
    """The path a script takes. This ran to completion before."""
    with pytest.raises(ValueError, match="decouples its two channels"):
        _run(_setup(DECOUPLED, "dynamic_closed"))


def test_the_refusal_says_which_bin_counts_would_work() -> None:
    """A rejection that does not tell you the fix gets worked around."""
    with pytest.raises(ValueError) as excinfo:
        _run(_setup(DECOUPLED, "dynamic_closed"))
    message = str(excinfo.value)
    assert "2*E_face/dE is an integer" in message
    assert str(COMMENSURATE) in message


def test_a_commensurate_grid_still_runs() -> None:
    """The guard must not reject the grids the constraint is satisfied on."""
    _run(_setup(COMMENSURATE, "dynamic_closed"))


def test_a_pinned_bath_is_not_subject_to_the_constraint() -> None:
    """No phonon equation is solved, so the two channels never share n_ph.

    Enforcing commensurability here would reject grids that are perfectly
    sound for the run being asked for -- the guard has to be gated on a LIVE
    phonon sector, and this is what holds that gate open.
    """
    _run(_setup(DECOUPLED, "thermal_bath"))


def test_the_two_bin_counts_really_do_differ_in_commensurability() -> None:
    """Guard the premise, so the tests above cannot pass for a stale reason."""
    from qpsim.grid.energy_grid import build_energy_grid

    ratios = {}
    for num_bins in (COMMENSURATE, DECOUPLED):
        E, _ = build_energy_grid(
            gap=180.0,
            energy_min_factor=1.0,
            energy_max_factor=4.0,
            num_energy_bins=num_bins,
        )
        spacing = float(E[1] - E[0])
        face = float(E[0]) - 0.5 * spacing
        ratios[num_bins] = 2.0 * face / spacing

    assert ratios[COMMENSURATE] == pytest.approx(round(ratios[COMMENSURATE]))
    assert abs(ratios[DECOUPLED] - round(ratios[DECOUPLED])) > 1e-6
