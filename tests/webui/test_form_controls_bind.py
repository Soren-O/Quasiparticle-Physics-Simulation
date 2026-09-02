"""The browser's form controls bind to the setup they claim to edit.

Every Python test of a "Wave 1 control" so far exercised the ENGINE with a
setup built in Python. None executed the JavaScript control. That is where
the last two instances of this repo's signature defect lived: a field type
declared and never dispatched, so every string control discarded typing; and
`boundary.per_edge` declared as a flat {name: number} "params" box, so the
nested override its own placeholder showed was rejected on every change and
the box resynced -- it rendered, took typing, and threw it away.

So these tests run the shipped app.js under node against a fake DOM
(``form_harness.js``), operate the control the way a browser would, read
back ``state.setup``, and then hand that setup to the engine and require a
NUMBER to move. A control that posts but changes nothing fails here.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qpsim.webui.execute import run_kinetics
from qpsim.webui.schemas import (
    DriveSpec,
    EdgeCondition,
    KineticsSetup,
    M25JunctionSetup,
)

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"
_APP_JS = _STATIC / "app.js"
_HARNESS = pathlib.Path(__file__).with_name("form_harness.js")
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(
    _NODE is None, reason="binding the browser controls needs node on PATH",
)


# -- structural: every control is bound, every type is dispatched -------------


def _app_text() -> str:
    return _APP_JS.read_text(encoding="utf-8")


def _resolves(model: object, path: str) -> bool:
    node = model
    for part in path.split("."):
        fields = getattr(type(node), "model_fields", None)
        if fields is None or part not in fields:
            return False
        node = getattr(node, part)
        if node is None:
            return True
    return True


def test_every_entry_control_binds_to_a_field_of_its_entry_model() -> None:
    """I(...) paths are relative to one list entry, not to the setup."""
    paths = sorted(set(re.findall(r'\bI\("([A-Za-z_][A-Za-z_0-9.]*)"', _app_text())))
    assert paths, "no entry controls found -- the list controls are gone?"
    entry_models = (DriveSpec(), EdgeCondition())
    orphans = [p for p in paths if not any(_resolves(m, p) for m in entry_models)]
    assert orphans == [], f"entry controls bound to nothing: {orphans}"


def test_every_field_type_used_is_one_the_renderer_dispatches() -> None:
    """The Wave 1 defect, made structural: a type the renderer never checks
    falls through to the numeric branch and discards non-numeric input."""
    text = _app_text()
    used = set(re.findall(r'\b[FI]\("[^"]+",\s*"[^"]*",\s*"([a-z_]+)"', text))
    dispatched = set(re.findall(r'field\.type === "([a-z_]+)"', text))
    # `number` and `int` are the else-branch, by design.
    undispatched = used - dispatched - {"number", "int"}
    assert undispatched == set(), f"field types with no renderer branch: {undispatched}"


def test_a_params_box_is_only_ever_bound_to_a_flat_number_map() -> None:
    """A "params" control accepts a flat {name: number} map and nothing else,
    so binding it to a field of any other shape is a control that discards
    every value -- which is what boundary.per_edge was."""
    text = _app_text()
    bound = re.findall(r'\bF\("([A-Za-z_][A-Za-z_0-9.]*)",\s*"[^"]*",\s*"params"', text)
    assert bound, "no params controls found"
    for path in bound:
        for model in (KineticsSetup(), M25JunctionSetup()):
            if not _resolves(model, path):
                continue
            node: Any = model
            *parents, leaf = path.split(".")
            for part in parents:
                node = getattr(node, part)
            annotation = type(node).model_fields[leaf].annotation
            assert annotation == dict[str, float], (
                f"{path} is a params box but the model field is {annotation}"
            )


# -- behavioural: drive the shipped control under node ------------------------


def _run_scenario(script: str) -> dict[str, Any]:
    """Execute a scenario against app.js in the harness; return RESULT."""
    proc = subprocess.run(
        [_NODE, str(_HARNESS), str(_APP_JS)],
        input=script, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


_PRELUDE = '''
const fieldAt = (path) => {
  for (const section of FORMS.kinetics) {
    for (const f of section.fields) if (f.path === path) return f;
  }
  throw new Error(`no shipped control bound to ${path}`);
};
const control = (path) => renderField(fieldAt(path));
const inputOf = (root, path) => {
  const el = root.querySelector(`[data-path="${path}"]`);
  if (!el) throw new Error(`no input bound to ${path}`);
  return el;
};
const type = (root, path, text) => { const el = inputOf(root, path); el.value = text; el.dispatch("change"); };
const tick = (root, path, on) => { const el = inputOf(root, path); el.checked = on; el.dispatch("change"); };
const click = (root, selector) => { const el = root.querySelector(selector); if (!el) throw new Error(`no ${selector}`); el.click(); };
const keyInput = (root, path) => root.querySelector(`[data-key="${path}"]`);
'''


def _setup_json() -> str:
    """A driven strip, as the browser would hold it after loading defaults."""
    setup = KineticsSetup()
    setup.geometry.rows = 1
    setup.geometry.cols = 4
    setup.strategy = "time_march"
    setup.injection.enabled = True
    setup.injection.rate_per_ns = 1e-2
    # Drives are sampled at each step's midpoint, so a step must be short
    # against the shortest pulse a scenario switches off.
    setup.dt = 0.01
    setup.max_time = 0.2
    setup.snapshot_interval = 0.1
    setup.stop_tol = 0.0
    return json.dumps(setup.model_dump(mode="json"))


def _profile(setup_dict: dict[str, Any]) -> np.ndarray:
    setup = KineticsSetup.model_validate(setup_dict)
    payload = run_kinetics(setup, lambda *a, **k: None, lambda: False)
    return np.asarray(payload.arrays["xqp_profile"], dtype=float)


class TestThePerEdgeTable:
    def test_add_edit_rename_remove_all_reach_the_setup(self) -> None:
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("boundary.per_edge");
const steps = {{}};
click(box, ".list-add");
steps.added = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
type(box, "boundary.per_edge.up.value", "3");
inputOf(box, "boundary.per_edge.up.kind").value = "robin";
inputOf(box, "boundary.per_edge.up.kind").dispatch("change");
type(box, "boundary.per_edge.up.aux_value", "0.5");
steps.edited = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
const key = keyInput(box, "boundary.per_edge.up");
key.value = "left"; key.dispatch("change");
steps.renamed = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
const bad = keyInput(box, "boundary.per_edge.left");
bad.value = ""; bad.dispatch("change");
steps.emptyKeyRefused = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
bad.value = "a.b"; bad.dispatch("change");
steps.dottedKeyRefused = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
click(box, ".list-add");
steps.secondAdded = Object.keys(state.setup.boundary.per_edge);
click(box, ".list-remove");
steps.afterRemove = Object.keys(state.setup.boundary.per_edge);
RESULT = steps;
''')
        assert result["added"] == {"up": {"kind": "absorbing", "value": 0, "aux_value": None}}
        assert result["edited"] == {"up": {"kind": "robin", "value": 3, "aux_value": 0.5}}
        assert result["renamed"] == {"left": {"kind": "robin", "value": 3, "aux_value": 0.5}}
        assert result["emptyKeyRefused"] == result["renamed"]
        assert result["dottedKeyRefused"] == result["renamed"]
        # The next free suggested id, not a duplicate of the one in use.
        assert result["secondAdded"] == ["left", "up"]
        assert result["afterRemove"] == ["up"]

    def test_an_edge_authored_in_the_browser_moves_the_engine(self) -> None:
        """The setup the control produced, run: an absorbing left end must
        hold fewer quasiparticles than the untouched reflective rim."""
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("boundary.per_edge");
click(box, ".list-add");
const key = keyInput(box, "boundary.per_edge.up");
key.value = "left"; key.dispatch("change");
RESULT = state.setup;
''')
        assert result["boundary"]["per_edge"] == {
            "left": {"kind": "absorbing", "value": 0, "aux_value": None},
        }
        trapped = _profile(result)
        untouched = _profile(json.loads(_setup_json()))
        assert not np.allclose(trapped, untouched)
        assert trapped.sum() < untouched.sum()


class TestTheDrivesList:
    def test_a_new_drive_is_enabled_and_refused_until_it_has_an_amplitude(self) -> None:
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("drives");
click(box, ".list-add");
RESULT = state.setup;
''')
        assert len(result["drives"]) == 1
        assert result["drives"][0]["enabled"] is True
        assert result["drives"][0]["amplitude"] == 0
        with pytest.raises(ValidationError, match="look driven and be undriven"):
            KineticsSetup.model_validate(result)

    def test_a_pulse_authored_in_the_browser_moves_the_engine(self) -> None:
        def authored(t_off: float) -> dict[str, Any]:
            return _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
state.setup.injection.enabled = false;   // the pulse is the only drive
const box = control("drives");
click(box, ".list-add");
type(box, "drives.0.amplitude", "5e-2");
inputOf(box, "drives.0.time.kind").value = "pulse";
inputOf(box, "drives.0.time.kind").dispatch("change");
type(box, "drives.0.time.t_off", "{t_off}");
inputOf(box, "drives.0.space.kind").value = "point";
inputOf(box, "drives.0.space.kind").dispatch("change");
type(box, "drives.0.space.x_0", "1.0");
RESULT = state.setup;
''')
        short = authored(0.05)
        assert short["drives"][0]["time"] == {
            "kind": "pulse", "t_on": 0, "t_off": 0.05, "tau": None, "expression": None,
        }
        assert short["drives"][0]["space"]["kind"] == "point"
        assert short["drives"][0]["space"]["x_0"] == 1.0
        assert KineticsSetup.model_validate(short).drives[0].amplitude == 5e-2

        quiet = json.loads(_setup_json())
        quiet["injection"]["enabled"] = False
        undriven = _profile(quiet)
        pulsed_short = _profile(short)
        pulsed_long = _profile(authored(0.2))
        # The drive acts, and its window is honoured: a longer pulse deposits
        # more, on the end it is aimed at.
        assert not np.allclose(pulsed_short, undriven)
        assert pulsed_long.sum() > pulsed_short.sum() > undriven.sum()
        assert pulsed_long[-1] > pulsed_long[0]

    def test_remove_and_a_second_entry_index_correctly(self) -> None:
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("drives");
click(box, ".list-add");
click(box, ".list-add");
type(box, "drives.1.amplitude", "7");
inputOf(box, "drives.1.channel").value = "loss";
inputOf(box, "drives.1.channel").dispatch("change");
const before = JSON.parse(JSON.stringify(state.setup.drives));
click(box, ".list-remove");   // removes entry 0; entry 1 becomes entry 0
const after = JSON.parse(JSON.stringify(state.setup.drives));
type(box, "drives.0.amplitude", "9");
RESULT = {{ before, after, final: state.setup.drives }};
''')
        assert [d["amplitude"] for d in result["before"]] == [0, 7]
        assert result["after"] == [result["before"][1]]
        assert result["final"][0]["amplitude"] == 9
        assert result["final"][0]["channel"] == "loss"


class TestTheOlderControlsStillBind:
    """The two shapes of the Wave 1 defect, now executed rather than read."""

    def test_a_string_control_keeps_what_is_typed(self) -> None:
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("geometry.gds_path");
type(box, "geometry.gds_path", "device.gds");
RESULT = state.setup.geometry.gds_path;
''')
        assert result == "device.gds"

    def test_a_params_box_keeps_a_flat_map_and_refuses_a_nested_one(self) -> None:
        result = _run_scenario(_PRELUDE + f'''
state.setup = {_setup_json()};
const box = control("initial.params");
type(box, "initial.params", '{{"a": 1.5, "b": 2}}');
const flat = JSON.parse(JSON.stringify(state.setup.initial.params));
type(box, "initial.params", '{{"a": {{"b": 1}}}}');
RESULT = {{ flat, afterNested: state.setup.initial.params }};
''')
        assert result["flat"] == {"a": 1.5, "b": 2}
        assert result["afterNested"] == {"a": 1.5, "b": 2}
