"""Which terms of the kinetic equations a setup actually solves.

One authority, because there were two. The interface had its own rules for
reading a setup and deciding what to draw, and the engine had the gates it
actually branches on. Two implementations of the same question drift, and
these had: the panel showed the phonon-source terms as present while the
phonons were pinned (so the phonon equation was never advanced), showed
diffusion as present on a single cell inside a reflective rim (which has
nothing to transport between, so the transport operator is never built), and
showed a photon drive as present with ``n_bar = 0`` (which applies nothing).
In all three the numbers were right and the statement about them was false.

So the statement is computed here, next to the engine, from the same
conditions the engine branches on, and the interface renders what it is told.
Each entry cites the gate it mirrors so the two can be checked against each
other by reading, and ``tests/webui/test_terms.py`` binds them by measurement:
a term reported absent or off must not change the result when it is toggled,
and a term reported on must.

Three states, and the difference between the last two is the whole point:

``on``      the model contains this term and it is acting.
``off``     the model contains it and it is switched off.
``absent``  the model does not contain it at all, so it can be neither
            switched on nor off -- a single cell inside a reflective rim has
            no transport, and a pinned phonon bath has no phonon equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qpsim.webui.builders import acting_rim_kinds_2d, build_geometry_2d

ON = "on"
OFF = "off"
ABSENT = "absent"
# Not a fourth kind of physics -- the absence of an answer. Some facts about a
# setup are only settled when the run starts (a layout is rasterised then), and
# reporting one of the three states anyway is how "we have not looked" became
# "there is no transport here".
UNKNOWN = "unknown"


@dataclass(frozen=True)
class TermStatus:
    """What a term is doing in this setup, and why."""

    state: str
    reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"state": self.state, "reason": self.reason}


def _get(setup: Any, path: str, default: Any = None) -> Any:
    node: Any = setup
    for key in path.split("."):
        if node is None:
            return default
        node = getattr(node, key, None)
    return default if node is None else node


def _cell_count(setup: Any) -> int | None:
    """Cells in the mask, or None when the setup genuinely does not say.

    None means UNKNOWN and is reported as such. It used to mean "assume it has
    neighbours", which turned a missing answer into a definite claim -- a GDS
    layout that rasterises to one cell was told it was diffusing. Guessing the
    less-bad direction is still guessing, and the panel exists to stop the
    interface asserting things the engine has not agreed to.
    """
    geometry = getattr(setup, "geometry", None)
    if geometry is None:
        # Every mode carries a `geometry`, and `SAVED_SETUP_UPGRADES`
        # supplies one for any saved setup that stores its extent as
        # `num_cells` instead. This branch is a backstop for an object that
        # reached here without going through the envelope: reading only
        # `geometry` would report a 31-cell strip as a single cell with no
        # transport at all.
        cells = getattr(setup, "num_cells", None)
        if cells is not None:
            return int(cells)
        return 1  # genuinely 0-D: no geometry and no extent field
    if getattr(geometry, "kind", "rectangle") != "rectangle":
        # A layout is rasterised when the run starts, so the setup alone
        # cannot say how many cells it has.
        return None
    return int(getattr(geometry, "rows", 1)) * int(getattr(geometry, "cols", 1))


def _phonons_solved(setup: Any) -> bool:
    """Is a phonon equation advanced at all?

    ``thermal_bath`` pins n_ph and the executors pass
    ``phonon_escape_time=None``, on which both the 0-D and the spatial
    collision layers skip the phonon advance outright.
    """
    mode = _get(setup, "phonons.mode")
    return mode is not None and mode != "thermal_bath"


def _flag(setup: Any, path: str) -> TermStatus:
    return TermStatus(ON if bool(_get(setup, path, False)) else OFF)


def _acting_rim(setup: Any) -> list[str] | None:
    """Kinds of the rim conditions that act, or None when they cannot be read.

    The engine's own rule, evaluated on the conditions it will build
    (:func:`qpsim.webui.builders.acting_rim_kinds_2d`): a reflective face
    contributes nothing, and every other kind is a term in the equation. A
    setup with no rim declared is a reflective device, which is what
    ``SpatialState.boundary`` defaults to.
    """
    if getattr(setup, "geometry", None) is None or getattr(setup, "boundary", None) is None:
        return []
    try:
        return acting_rim_kinds_2d(setup, build_geometry_2d(setup))
    except ValueError:
        # A rim override naming an edge this geometry does not have. The run
        # refuses it with the same message; the panel does not guess.
        return None


def _transport(setup: Any) -> TermStatus:
    # Gate: SpatialBackend.apply_transport (spatial.py:496-498) returns the
    # state untouched, and SpatialBackend.rates (spatial.py:624-626) builds no
    # operator, only when `state.f.shape[1] == 1` AND
    # `SpatialTransport.has_acting_device_faces()` is false. A rim of any kind
    # but reflective is a term in the equation even on one cell, and the flux
    # weight carries D_0.
    cells = _cell_count(setup)
    if cells is None:
        return TermStatus(
            UNKNOWN,
            "this layout is rasterised when the run starts, so whether it has "
            "more than one cell -- and therefore whether it transports at all "
            "-- is not known yet",
        )
    if cells <= 1:
        rim = _acting_rim(setup)
        if _steady_state(setup):
            # DiffusionBackend.steady_state carries f(E) with no cell axis and
            # no rim; `validate_setup` refuses a rim that acts on this route.
            declared = (
                f"; the declared {' and '.join(rim)} rim is not part of that "
                "solver, and the run refuses it"
                if rim else ""
            )
            return TermStatus(
                ABSENT,
                "the steady-state strategy is a root find for f(E) on one cell "
                "with no rim, so there is nothing to transport" + declared,
            )
        if rim is None:
            return TermStatus(
                UNKNOWN,
                "a rim override names an edge this geometry does not have, so "
                "whether the rim acts -- and therefore whether this one cell "
                "transports at all -- cannot be read; the run refuses it too",
            )
        if not rim:
            return TermStatus(
                ABSENT,
                "a single cell inside a reflective rim has no neighbour to "
                "exchange with and no face that lets density out, so there is "
                "nothing to transport",
            )
        if float(_get(setup, "material.D_0", 0.0)) <= 0.0:
            return TermStatus(OFF, "D_0 = 0 makes the flux coefficient vanish")
        return TermStatus(
            ON,
            f"one cell, but its rim is {' and '.join(rim)}: those faces "
            "exchange density with the outside, so the transport operator is "
            "built and D_0 scales it",
        )
    if float(_get(setup, "material.D_0", 0.0)) <= 0.0:
        return TermStatus(OFF, "D_0 = 0 makes the flux coefficient vanish")
    return TermStatus(ON)


def _phonon_side(setup: Any, path: str) -> TermStatus:
    if not _phonons_solved(setup):
        return TermStatus(
            ABSENT,
            "the phonons are pinned at the bath, so no phonon equation is "
            "solved and this term is not in the model",
        )
    return _flag(setup, path)


def _escape(setup: Any) -> TermStatus:
    if not _phonons_solved(setup):
        return TermStatus(
            ABSENT,
            "the phonons are pinned at the bath, so there is no phonon "
            "population to couple to the substrate",
        )
    return TermStatus(ON if _get(setup, "phonons.mode") == "dynamic_escape" else OFF)


def _photon(setup: Any, prefix: str, occupancy: str, coupling: str) -> TermStatus:
    """A photon drive applies nothing at zero COUPLING. Not at zero n_bar.

    This function used to test the photon number first and report
    "enabled, but n_bar = 0, so the drive applies nothing". That is false, and
    measurably so: both kernels carry ``(n_bar + 1)``
    (``sub_gap_photon.py:143,151``, ``pair_breaking_photon.py:299,306,326``),
    so n_bar = 0 removes only the STIMULATED term and leaves spontaneous
    emission at full strength. Measured on a 1x1 kinetics at n_bar = 0: the
    drive moves the final f by 19% at c_phot = 1e-4 and 80% at 1e-2, while the
    panel said "off". The coupling is the real switch -- every term is
    multiplied by it -- and c_phot = 0 with n_bar = 1e7 is inert to 0.000e+00.

    So this is the same defect the module exists to prevent, inverted: not a
    switch shown on that does nothing, but a channel shown off that is doing
    most of the work.
    """
    # NOT gated on the steady-state strategy: that executor passes both
    # photon drives to the solver, and they move the answer (measured on the
    # catalogue cases, 2026-09-01: pb_drive on/off changes x_qp 151-fold).
    # Reporting them absent there was the panel stating something untrue.
    if not bool(_get(setup, f"{prefix}.enabled", False)):
        return TermStatus(OFF)
    if float(_get(setup, f"{prefix}.{coupling}", 0.0)) <= 0.0:
        return TermStatus(
            OFF, f"enabled, but {coupling} = 0, so the drive applies nothing"
        )
    if float(_get(setup, f"{prefix}.{occupancy}", 0.0)) <= 0.0:
        # On, and worth saying which half is acting: with no photons present
        # the absorption/stimulated channel is gone and what remains is
        # spontaneous emission, which relaxes rather than drives.
        return TermStatus(
            ON,
            f"{occupancy} = 0, so only the spontaneous term acts -- the "
            "channel relaxes quasiparticles rather than driving them",
        )
    return TermStatus(ON)


def _steady_state(setup: Any) -> bool:
    """Does this setup select the 0-D steady-state solver?

    That route reads the material, the phonon sector, the probe, the two
    PHOTON drives and a STATIC injection -- and no prescribed
    (time-dependent) drives and no initial condition, which a root find has
    no axis for. Those two are `absent` there in the precise sense this
    module means: not in the model, and therefore neither on nor off.
    """
    return getattr(setup, "strategy", None) == "steady_state"


_STEADY_STATE_REASON = (
    "the steady-state strategy is a root find with no time axis and no initial "
    "condition; a prescribed drive or a prepared start is not part of that model"
)


def _injection(setup: Any) -> TermStatus:
    # Not gated on the steady-state strategy: a static injection
    # is folded into the solver's ExternalFlux there, and the measured answer
    # moves with it (tests/webui/test_wave7_deferred.py).
    if getattr(setup, "injection", None) is None:
        return TermStatus(ABSENT, "this mode has no injection source")
    if not bool(_get(setup, "injection.enabled", False)):
        return TermStatus(OFF)
    if float(_get(setup, "injection.rate_per_ns", 0.0)) <= 0.0:
        return TermStatus(OFF, "enabled, but the rate is zero")
    return TermStatus(ON)


def _absent_without(setup: Any, path: str, reason: str) -> TermStatus | None:
    """ABSENT when the setup does not carry the field at all."""
    node: Any = setup
    parts = path.split(".")
    for key in parts[:-1]:
        node = getattr(node, key, None)
        if node is None:
            return TermStatus(ABSENT, reason)
    if not hasattr(node, parts[-1]):
        return TermStatus(ABSENT, reason)
    return None


def term_status(setup: Any) -> dict[str, TermStatus]:
    """Every term's state for this setup, keyed by the panel's term id."""
    mode = getattr(setup, "mode", "this mode")
    missing = f"not part of the {mode} model"

    def guarded(path: str, compute) -> TermStatus:
        absent = _absent_without(setup, path, missing)
        return absent if absent is not None else compute()

    return {
        "diff": guarded("material.D_0", lambda: _transport(setup)),
        "scat": guarded(
            "collisions.scattering", lambda: _flag(setup, "collisions.scattering")
        ),
        "recomb": guarded(
            "collisions.recombination",
            lambda: _flag(setup, "collisions.recombination"),
        ),
        "src": guarded("injection.enabled", lambda: _injection(setup)),
        "psc": guarded(
            "collisions.phonon_scattering_source",
            lambda: _phonon_side(setup, "collisions.phonon_scattering_source"),
        ),
        "prc": guarded(
            "collisions.phonon_recombination_source",
            lambda: _phonon_side(setup, "collisions.phonon_recombination_source"),
        ),
        "photsg": guarded(
            "subgap_drive.enabled",
            lambda: _photon(setup, "subgap_drive", "n_bar", "c_phot"),
        ),
        "photpb": guarded(
            "pb_drive.enabled",
            lambda: _photon(setup, "pb_drive", "n_bar_PB", "c_phot_PB"),
        ),
        "pesc": guarded("phonons.mode", lambda: _escape(setup)),
        "gapeq": guarded(
            "self_consistent_gap", lambda: _flag(setup, "self_consistent_gap")
        ),
    }


def term_status_payload(setup: Any) -> dict[str, dict[str, str]]:
    """JSON-serialisable form for the API."""
    return {term: status.as_dict() for term, status in term_status(setup).items()}
