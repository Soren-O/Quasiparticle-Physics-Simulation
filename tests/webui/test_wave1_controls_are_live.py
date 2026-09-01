"""Wave 1: the controls added to the browser form actually reach the engine.

Two kinds of check, and both are needed.

The first is structural and covers EVERY field in the form, not just the new
ones: a control whose `path` does not resolve on the setup model writes into
nothing. It renders, it accepts typing, and it is discarded -- the exact defect
this repo keeps producing, and a typo in a form definition is the cheapest way
to reintroduce it.

The second measures. For each capability the wave exposes, a number has to
MOVE when the control changes. Asserting that a field posts, or that a run
completes, would pass just as well with the value ignored. The runs here are
also DRIVEN on purpose: the audit found that on an undriven setup every solver
and every phonon sector returns bit-identical results, because the thermal
state is the fixed point -- so a binding test built on one is vacuous.
"""

from __future__ import annotations

import io
import pathlib
import re

import numpy as np
import pytest

from qpsim.webui.builders import build_gap_per_cell_2d, build_geometry_2d
from qpsim.webui.execute import run_kinetics
from qpsim.webui.schemas import KineticsSetup, M25JunctionSetup

_APP_JS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "qpsim" / "webui" / "static" / "app.js"
)


def _form_paths() -> list[str]:
    """Every dotted path the form binds a control to."""
    text = io.open(_APP_JS, encoding="utf-8").read()
    return sorted(set(re.findall(r'F\("([A-Za-z_][A-Za-z_0-9.]*)"', text)))


def _resolves(model: object, path: str) -> bool:
    node = model
    for part in path.split("."):
        fields = getattr(type(node), "model_fields", None)
        if fields is None or part not in fields:
            return False
        node = getattr(node, part)
        if node is None:
            # Optional sub-model: the field exists, which is what is asserted.
            return True
    return True


def test_every_form_control_binds_to_a_real_field() -> None:
    """A path that resolves nowhere is a control that silently does nothing."""
    kinetics, junction = KineticsSetup(), M25JunctionSetup()
    orphans = [
        p for p in _form_paths()
        if not _resolves(kinetics, p) and not _resolves(junction, p)
    ]
    assert orphans == [], f"form controls bound to nothing: {orphans}"


def _driven(**over: object) -> KineticsSetup:
    """A small DRIVEN strip. Undriven, none of these settings can be seen."""
    setup = KineticsSetup()
    setup.geometry.rows = 1
    setup.geometry.cols = 4
    setup.strategy = "time_march"
    setup.injection.enabled = True
    setup.injection.rate_per_ns = 1e-2
    setup.max_time = 0.2
    setup.snapshot_interval = 0.1
    for path, value in over.items():
        node: object = setup
        parts = path.split("__")
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    return setup


def _run(setup: KineticsSetup) -> dict[str, np.ndarray]:
    payload = run_kinetics(setup, lambda *a, **k: None, lambda: False)
    return payload.arrays


class TestTheGapExpressionReachesTheEngine:
    def test_a_typed_expression_builds_the_gap_it_describes(self) -> None:
        setup = _driven()
        setup.geometry.cols = 6
        setup.gap_regions.kind = "expression"
        setup.gap_regions.expression = "gap*(1.0 + 0.5*x)"
        geometry = build_geometry_2d(setup)
        gap = build_gap_per_cell_2d(setup, geometry)
        assert gap is not None
        # The recorded profile for this expression on a 6-cell strip.
        np.testing.assert_allclose(
            np.asarray(gap).ravel(),
            [187.5, 202.5, 217.5, 232.5, 247.5, 262.5],
            rtol=1e-9,
        )

    def test_a_uniform_gap_is_not_that(self) -> None:
        """Guards the premise: the assertion above must not hold by default."""
        setup = _driven()
        setup.geometry.cols = 6
        gap = build_gap_per_cell_2d(setup, build_geometry_2d(setup))
        assert gap is None or len(set(np.asarray(gap).ravel())) == 1


class TestThePhononSectorReachesTheEngine:
    def test_a_dynamic_sector_records_phonons_and_a_pinned_one_does_not(self) -> None:
        dynamic = _run(_driven(phonons__mode="dynamic_escape"))
        pinned = _run(_driven(phonons__mode="thermal_bath"))
        assert "snap_n_ph" in dynamic
        assert "snap_n_ph" not in pinned

    def test_the_escape_time_changes_the_answer(self) -> None:
        """tau_l is a control, so it must move a number."""
        fast = _run(_driven(phonons__mode="dynamic_escape", phonons__tau_l_ns=0.017))
        slow = _run(_driven(phonons__mode="dynamic_escape", phonons__tau_l_ns=1.7))
        assert not np.allclose(fast["snap_n_ph"], slow["snap_n_ph"])


class TestTheInitialConditionReachesTheEngine:
    def test_a_non_thermal_start_moves_the_first_frame(self) -> None:
        setup = _driven()
        setup.initial.kind = "excess"
        setup.initial.amplitude = 1e-3
        setup.initial.energy.kind = "thermal"
        setup.initial.energy.T_eff = 0.6
        excess = _run(setup)
        thermal = _run(_driven())
        assert not np.allclose(excess["snap_f"][0], thermal["snap_f"][0])

    def test_a_spatial_profile_makes_the_start_non_uniform(self) -> None:
        setup = _driven()
        setup.initial.kind = "excess"
        setup.initial.amplitude = 1e-3
        setup.initial.energy.kind = "thermal"
        setup.initial.energy.T_eff = 0.6
        setup.initial.space.kind = "point"
        first = _run(setup)["snap_f"][0]
        per_cell = first.sum(axis=0)
        assert per_cell.max() > 2.0 * per_cell.min()


class TestTheRobinBoundaryReachesTheEngine:
    """∂ₙφ + βφ = γ. `value` is β, `aux_value` is γ.

    β is the transparency and the reason this condition exists: β = 0 is
    reflective, β → ∞ is absorbing, and every real contact sits between them.
    So β is what a test should sweep. γ is a source term, and driving one into
    a reflective wall (β = 0, large γ) is not a weak-versus-strong contact --
    it pumps the device and the transport guard rightly refuses the step.
    """

    def _robin(self, beta: float) -> KineticsSetup:
        setup = _driven()
        setup.boundary.kind = "robin"
        setup.boundary.value = beta
        setup.boundary.aux_value = 0.0
        return setup

    def test_beta_moves_the_answer_towards_absorbing(self) -> None:
        nearly_reflective = _run(self._robin(1e-3))["snap_f"][-1]
        nearly_absorbing = _run(self._robin(10.0))["snap_f"][-1]
        assert not np.allclose(nearly_reflective, nearly_absorbing)
        # More transparent must hold FEWER quasiparticles, not merely differ.
        assert nearly_absorbing.sum() < nearly_reflective.sum()
