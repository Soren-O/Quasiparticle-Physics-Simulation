"""Bind the term statement to what the engine actually does.

``qpsim.webui.terms`` claims which terms a setup solves, and the interface
draws that claim. A claim nobody checks is how this broke: the panel had its
own rules, they disagreed with the engine in three separate ways, and every
disagreement was invisible because the NUMBERS were right -- only the
statement about them was false.

So the claim is measured. Each term's switch is flipped and the two runs
compared:

* ``on`` / ``off`` -> flipping MUST change the result. The term is in the
  model, so switching it has to do something. A term reported as present that
  changes nothing is the inert-default trap this module exists to catch.
* ``absent``       -> flipping must change NOTHING, because the term is not
  in the model at all. This is the case the interface was getting wrong.

Two things this harness has to get right, both learned by getting them wrong:

The observable covers BOTH populations. Comparing only ``f`` reports the
phonon-source terms as inert, because they act on ``n_ph`` and reach ``f``
only through a collision integral that is already at its fixed point.

The runs have to be able to move. A uniform field makes transport inert
whatever ``D_0`` is, and a thermal start makes the collision terms inert
because thermal is their fixed point -- so the setups carry a spatial
gradient and a quasiparticle excess, and the assertions are worthless
without them.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="term tests need the qpsim[ui] extra")

from qpsim.webui import execute
from qpsim.webui.schemas import Spatial2DSetup
from qpsim.webui.terms import ABSENT, OFF, ON, term_status

# One grid on which every flip below is a LEGAL setup, so the test measures
# physics rather than tripping a validator. The constraints fight each other:
# the pair-breaking drive needs its reflection partners on the lattice, which
# forces 2*min_factor*num_bins/(max_factor-min_factor) to be an integer, while
# a self-consistent gap needs sub-gap cells, which forces min_factor < 1.
# These values satisfy both, and both photon energies are grid-commensurate.
_BINS, _MIN_FACTOR, _MAX_FACTOR = 64, 0.75, 2.75
_OMEGA_SUBGAP = 45.0        # 8 * dE. Small ON PURPOSE: the ladder steps by
                            # omega_0, so a photon near 2*Delta leaves almost
                            # every bin without a partner inside a grid this
                            # narrow, and the drive is then inert for a reason
                            # that has nothing to do with the switch.
_OMEGA_PB = 365.625         # 65 * dE, above 2*Delta
_TOLERANCE = 1e-12


def _base(rows: int = 2, cols: int = 3) -> Spatial2DSetup:
    """A setup where every term has something to act on.

    The excess and the gradient are the load-bearing part: without them the
    'on' assertions pass vacuously for terms that are genuinely inert.
    """
    setup = Spatial2DSetup()
    setup.T_bath = 0.2
    setup.grid.num_bins = _BINS
    setup.grid.min_factor = _MIN_FACTOR
    setup.grid.max_factor = _MAX_FACTOR
    setup.geometry.rows, setup.geometry.cols = rows, cols
    setup.material.D_0 = 6.0
    # A gradient, so transport has somewhere to move things.
    setup.initial.kind = "excess"
    setup.initial.amplitude = 2e-3
    setup.initial.space.kind = "gaussian"
    setup.initial.space.sigma = 0.25
    # Parameters for the drives are set here, so a flip toggles only the
    # switch -- exactly what the interface's control does.
    setup.subgap_drive.omega_0 = _OMEGA_SUBGAP
    setup.subgap_drive.n_bar = 1e7
    setup.subgap_drive.c_phot = 1e-9
    setup.pb_drive.omega_PB = _OMEGA_PB
    setup.pb_drive.n_bar_PB = 1e3
    setup.pb_drive.c_phot_PB = 1e-9
    setup.dt = 1.0
    setup.max_time = 8.0
    setup.stop_tol = 0.0
    setup.snapshot_interval = 4.0     # so the phonon field is in the payload
    return setup


def _with_live_phonons(setup: Spatial2DSetup) -> Spatial2DSetup:
    setup.phonons.mode = "dynamic_escape"
    # Away from the bath, or escape starts at its own fixed point and the
    # whole phonon sector looks inert.
    setup.phonons.initial.kind = "scaled"
    setup.phonons.initial.factor = 3.0
    return setup


def _observable(setup: Spatial2DSetup) -> dict[str, np.ndarray]:
    """Everything a term could move: the quasiparticles AND the phonons."""
    payload = execute.execute_setup(setup, lambda *a, **k: None, lambda: False)
    out = {"f": np.asarray(payload.arrays["f_final"], dtype=float)}
    if "snap_n_ph" in payload.arrays:
        out["n_ph"] = np.asarray(payload.arrays["snap_n_ph"], dtype=float)
    if "gap_per_cell" in payload.arrays:
        out["gap"] = np.asarray(payload.arrays["gap_per_cell"], dtype=float)
    return out


def _change(before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for key in before.keys() & after.keys():
        a, b = before[key], after[key]
        if a.shape != b.shape:
            return float("inf")     # a changed shape is certainly a change
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
        worst = max(worst, float(np.max(np.abs(a - b))) / scale)
    return worst


def _flip(setup: Spatial2DSetup, term: str) -> Spatial2DSetup:
    """Toggle one term the way the interface's own control does."""
    clone = Spatial2DSetup.model_validate(setup.model_dump())
    collisions, drives = clone.collisions, clone
    if term == "diff":
        clone.material.D_0 = 0.0 if clone.material.D_0 > 0 else 6.0
    elif term == "scat":
        collisions.scattering = not collisions.scattering
    elif term == "recomb":
        collisions.recombination = not collisions.recombination
    elif term == "src":
        clone.injection.enabled = not clone.injection.enabled
    elif term == "psc":
        collisions.phonon_scattering_source = not collisions.phonon_scattering_source
    elif term == "prc":
        collisions.phonon_recombination_source = (
            not collisions.phonon_recombination_source
        )
    elif term == "photsg":
        drives.subgap_drive.enabled = not drives.subgap_drive.enabled
    elif term == "photpb":
        drives.pb_drive.enabled = not drives.pb_drive.enabled
    elif term == "pesc":
        clone.phonons.mode = (
            "dynamic_closed" if clone.phonons.mode == "dynamic_escape"
            else "dynamic_escape"
        )
    elif term == "gapeq":
        clone.self_consistent_gap = not clone.self_consistent_gap
    else:  # pragma: no cover - a new term needs a flip rule
        raise AssertionError(f"no flip rule for {term!r}")
    return clone


ALL_TERMS = ("diff", "scat", "recomb", "src", "psc", "prc",
             "photsg", "photpb", "pesc", "gapeq")

# `pesc`'s switch IS the phonon-sector selector, so when the sector is pinned
# there is no term-level control to flip: any change to that field brings the
# whole sector in, which is a different edit. Its absence is asserted
# separately, and its presence is exercised in the live-phonon setup.
_NO_TERM_LEVEL_FLIP_WHEN_ABSENT = {"pesc"}


class TestTheClaimMatchesTheEngine:
    @pytest.mark.parametrize("term", ALL_TERMS)
    @pytest.mark.parametrize("phonons", ["pinned", "live"])
    def test_flipping_a_term_does_what_the_status_says(
        self, term: str, phonons: str
    ) -> None:
        setup = _base()
        if phonons == "live":
            setup = _with_live_phonons(setup)
        status = term_status(setup)[term]
        if status.state == ABSENT and term in _NO_TERM_LEVEL_FLIP_WHEN_ABSENT:
            pytest.skip(
                f"{term}'s field selects the sector; flipping it is a "
                "different edit, not a toggle of this term"
            )

        change = _change(_observable(setup), _observable(_flip(setup, term)))
        if status.state == ABSENT:
            assert change <= _TOLERANCE, (
                f"[{phonons} phonons] {term} is reported ABSENT -- not in the "
                f"model -- but flipping it moved the result by {change:.3e}."
            )
        else:
            assert change > _TOLERANCE, (
                f"[{phonons} phonons] {term} is reported "
                f"{status.state.upper()} -- in the model -- but flipping it "
                f"moved the result by only {change:.3e}. Either it is inert "
                f"and should be reported ABSENT, or this setup gives it "
                f"nothing to do, which would make the assertion vacuous."
            )


class TestTheThreeDisagreementsThatPromptedThis:
    """Each was a term the panel showed as present while it did nothing."""

    def test_a_pinned_phonon_bath_has_no_phonon_terms(self) -> None:
        setup = _base()
        assert setup.phonons.mode == "thermal_bath"
        status = term_status(setup)
        for term in ("psc", "prc", "pesc"):
            assert status[term].state == ABSENT, term
            assert "pinned" in status[term].reason, term

    def test_a_single_cell_has_no_transport(self) -> None:
        setup = _base(rows=1, cols=1)
        setup.material.D_0 = 60.0
        status = term_status(setup)["diff"]
        assert status.state == ABSENT, "absence here is structural, not D_0 = 0"
        assert "no faces" in status.reason
        # And the engine agrees: a large D_0 changes nothing on one cell.
        before = _observable(setup)
        setup.material.D_0 = 0.0
        assert _change(before, _observable(setup)) <= _TOLERANCE

    def test_a_drive_with_no_photons_is_still_on(self) -> None:
        """n_bar = 0 is NOT an off-switch, and this test used to say it was.

        The kernels carry ``(n_bar + 1)``, so zero photons removes the
        stimulated term and leaves spontaneous emission acting at full
        strength. Asserted by measurement, because the previous version of
        this test asserted the panel's claim against nothing at all and so
        certified it while it was false.
        """
        setup = _base()
        setup.subgap_drive.enabled = True
        setup.subgap_drive.n_bar = 0.0
        setup.subgap_drive.c_phot = 1e-4
        status = term_status(setup)["photsg"]
        assert status.state == ON

        before = _observable(setup)
        setup.subgap_drive.enabled = False
        assert _change(before, _observable(setup)) > _TOLERANCE

    def test_the_coupling_is_the_off_switch(self) -> None:
        """c_phot = 0 multiplies every term of the channel by zero."""
        setup = _base()
        setup.subgap_drive.enabled = True
        setup.subgap_drive.n_bar = 1e7
        setup.subgap_drive.c_phot = 0.0
        status = term_status(setup)["photsg"]
        assert status.state == OFF
        assert "c_phot = 0" in status.reason

        before = _observable(setup)
        setup.subgap_drive.enabled = False
        assert _change(before, _observable(setup)) <= _TOLERANCE

    def test_the_same_drive_with_photons_is_on(self) -> None:
        setup = _base()
        setup.subgap_drive.enabled = True
        assert term_status(setup)["photsg"].state == ON


class TestTheStatusIsTotal:
    """Every term the panel can draw gets a verdict, on every mode."""

    def test_every_panel_term_is_accounted_for(self) -> None:
        assert set(term_status(_base())) == set(ALL_TERMS)

    def test_a_mode_without_a_field_reports_absent_not_missing(self) -> None:
        """A silently missing key would render as 'off', which is a claim."""
        from qpsim.webui.schemas import M25JunctionSetup
        status = term_status(M25JunctionSetup())
        assert set(status) == set(ALL_TERMS)
        assert all(s.state == ABSENT for s in status.values())
