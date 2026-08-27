"""A recorded phonon population is meaningless without its frequency axis.

The scattering-source benchmark rebuilds the frequency lattice from the setup
rather than reading it from the run, and used to bind its copy to the run's by
comparing BIN COUNTS. That is not a binding. The unified lattice being built
for the pair-marginal work sits exactly half a bin away from today's -- cell
centres against nodes -- and carries the SAME count on every commensurate
grid, which is every grid this repo allows. On the shipped default both are
450 bins: the count check passes, every frequency differs, and the benchmark
compares the engine against the wrong frequencies while reporting agreement.

So the run now records the axis it actually used, and the benchmark compares
VALUES. These tests pin both halves, and the last one is the point: it asserts
that the check which used to stand here would NOT have caught this.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpsim.webui.bench.phonon_scattering_source import _build
from qpsim.webui.execute import run_kinetics
from qpsim.webui.schemas import KineticsSetup


def _setup() -> KineticsSetup:
    """A 0-D run the scattering benchmark will actually accept.

    Its preconditions are all about keeping the closed form honest: no
    recombination (that adds the pair channel on the sum lattice), no
    self-consistent gap (that moves the kernel mid-run), and a non-thermal
    start (at equilibrium both sides are identically zero, which would be a
    benchmark that cannot fail).
    """
    setup = KineticsSetup()
    setup.geometry.rows = 1
    setup.geometry.cols = 1
    setup.phonons.mode = "dynamic_closed"
    setup.collisions.recombination = False
    setup.collisions.phonon_recombination_source = False
    setup.initial.kind = "excess"
    setup.initial.amplitude = 1e-3
    setup.initial.energy.kind = "thermal"
    setup.initial.energy.T_eff = 0.5
    setup.max_time = 0.2
    setup.snapshot_interval = 0.05
    return setup


@pytest.fixture(scope="module")
def run():
    setup = _setup()
    payload = run_kinetics(setup, lambda *a, **k: None, lambda: False)
    return setup, payload


def test_the_run_records_the_axis_its_populations_live_on(run) -> None:
    _, payload = run
    assert "snap_n_ph" in payload.arrays
    assert "snap_omega_bins" in payload.arrays, (
        "phonon populations were recorded without the frequencies they are "
        "indexed by, which leaves every reader to guess"
    )

    omega = np.asarray(payload.arrays["snap_omega_bins"], dtype=float)
    n_ph = np.asarray(payload.arrays["snap_n_ph"], dtype=float)
    assert omega.ndim == 1
    assert omega.size == n_ph.shape[1]
    # A frequency axis is sorted and non-negative, or it is not one.
    assert omega[0] >= 0.0
    assert np.all(np.diff(omega) > 0.0)


def test_the_benchmark_accepts_the_axis_the_run_reports(run) -> None:
    setup, payload = run
    curve = _build(setup, dict(payload.arrays), payload.summary)
    assert curve is not None


def test_a_half_bin_offset_is_caught_though_the_count_is_unchanged(run) -> None:
    """The whole point: same number of bins, every frequency wrong.

    Half a bin is not an arbitrary perturbation -- it is the exact offset
    between the two lattice conventions, so this is the failure that will
    actually occur if the engine's lattice is swapped and this rebuild is not
    swapped with it.
    """
    setup, payload = run
    arrays = dict(payload.arrays)
    true_axis = np.asarray(arrays["snap_omega_bins"], dtype=float)
    spacing = float(true_axis[1] - true_axis[0])
    shifted = true_axis + 0.5 * spacing
    arrays["snap_omega_bins"] = shifted

    # The check this replaced compared sizes. Assert here that it would have
    # been satisfied, so this test cannot quietly become a size test.
    assert shifted.size == true_axis.size
    assert shifted.size == np.asarray(arrays["snap_n_ph"]).shape[1]

    with pytest.raises(ValueError, match="different frequencies"):
        _build(setup, arrays, payload.summary)


def test_a_resized_axis_is_still_caught(run) -> None:
    """The weaker fault the old check did cover must stay covered."""
    setup, payload = run
    arrays = dict(payload.arrays)
    arrays["snap_omega_bins"] = np.asarray(
        arrays["snap_omega_bins"], dtype=float
    )[:-1]

    with pytest.raises(ValueError, match="bins"):
        _build(setup, arrays, payload.summary)
