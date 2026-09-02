"""Wave 3: the pre-run preview says what a run WOULD do, and says it right.

A preview is only worth having if its numbers are the run's numbers. So the
central test here runs the same setup and requires the preview's x_qp at
t = 0 to equal the run's recorded ``x_qp_initial`` -- not "close", equal,
because both are produced by one call. The clip note is required to arrive
WITHOUT a run being created; the edge list is required to be the rim the
engine assembles on; and a setup the builders refuse must come back as a
message, not a traceback.
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import shutil
import subprocess
import time
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi", reason="server tests need the qpsim[ui] extra")
pytest.importorskip("matplotlib", reason="preview figures need matplotlib")

from fastapi.testclient import TestClient  # noqa: E402
from qpsim.webui.preview import build_preview  # noqa: E402
from qpsim.webui.schemas import KineticsSetup, M25JunctionSetup  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"
_HARNESS = pathlib.Path(__file__).with_name("form_harness.js")
_NODE = shutil.which("node")


@pytest.fixture
def client(tmp_path: pathlib.Path) -> TestClient:
    with TestClient(create_app(tmp_path)) as c:
        yield c


def _wait_done(client: TestClient, run_id: str, timeout_s: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.2)
    pytest.fail(f"run {run_id} did not finish within {timeout_s}s")


def _setup(**over: Any) -> dict[str, Any]:
    setup = KineticsSetup()
    setup.geometry.rows = 3
    setup.geometry.cols = 5
    setup.geometry.mesh_size_um = 2.0
    setup.grid.num_bins = 24
    setup.T_bath = 0.2
    setup.strategy = "time_march"
    setup.dt = 0.1
    setup.max_time = 0.2
    body = setup.model_dump(mode="json")
    for key, value in over.items():
        node = body
        parts = key.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return body


def _preview(client: TestClient, setup: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/api/preview", json={"name": "p", "setup": setup})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _png(data_uri: str) -> bytes:
    assert data_uri.startswith("data:image/png;base64,")
    png = base64.b64decode(data_uri.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return png


class TestTheSeedIsReportedWithoutRunning:
    def test_a_clipping_seed_returns_the_note_and_creates_no_run(
        self, client: TestClient
    ) -> None:
        body = _preview(client, _setup(
            initial__kind="excess", initial__amplitude=5.0,
        ))
        assert body["ok"] is True
        assert any("clipped to [0, 1]" in n for n in body["notes"]), body["notes"]
        assert body["seed"]["clipped"] is True
        assert client.get("/api/runs").json() == []

    def test_a_modest_seed_does_not_clip(self, client: TestClient) -> None:
        """Guards the premise: the note must come from the amplitude."""
        body = _preview(client, _setup(initial__kind="excess", initial__amplitude=1e-3))
        assert body["notes"] == []
        assert body["seed"]["clipped"] is False

    def test_the_previewed_start_is_the_run_s_recorded_start(
        self, client: TestClient
    ) -> None:
        """Equal, not close: one call produces both numbers."""
        setup = _setup(
            initial__kind="excess", initial__amplitude=1e-3,
            initial__space__kind="gaussian", initial__space__x_0=0.2,
        )
        body = _preview(client, setup)
        run_id = client.post("/api/runs", json={"name": "r", "setup": setup}).json()["id"]
        manifest = _wait_done(client, run_id)
        assert manifest["status"] == "done", manifest.get("error")
        assert body["seed"]["x_qp_initial"] == manifest["summary"]["x_qp_initial"]
        assert body["seed"]["x_qp_thermal"] == pytest.approx(
            manifest["summary"]["x_qp_thermal"], rel=1e-12,
        )
        assert body["seed"]["x_qp_convention"] == manifest["summary"]["x_qp_convention"]
        # And the seed is a departure: the whole point of previewing it.
        assert body["seed"]["x_qp_initial"] > body["seed"]["x_qp_thermal"]
        assert body["seed"]["x_qp_initial_max"] > body["seed"]["x_qp_initial"]

    def test_a_thermal_start_sits_on_the_bath(self, client: TestClient) -> None:
        body = _preview(client, _setup())
        assert body["seed"]["kind"] == "thermal"
        assert body["seed"]["x_qp_initial"] == body["seed"]["x_qp_thermal"]


class TestTheGeometryIsTheOneTheRunSolvesOn:
    def test_a_rectangle_s_rim_is_four_segments(self, client: TestClient) -> None:
        body = _preview(client, _setup())
        g = body["geometry"]
        assert (g["rows"], g["cols"], g["cells"], g["dimensionality"]) == (3, 5, 15, 2)
        assert (g["mesh_size_um"], g["width_um"], g["height_um"]) == (2.0, 10.0, 6.0)
        by_normal = {e["normal"]: e for e in g["edges"]}
        assert set(by_normal) == {"up", "down", "left", "right"}
        assert by_normal["up"]["faces"] == by_normal["down"]["faces"] == 5
        assert by_normal["left"]["faces"] == by_normal["right"]["faces"] == 3
        # Extents in microns, on the grid lines the mask figure draws.
        assert (by_normal["left"]["x0_um"], by_normal["left"]["x1_um"]) == (0.0, 0.0)
        assert (by_normal["left"]["y0_um"], by_normal["left"]["y1_um"]) == (0.0, 6.0)
        assert (by_normal["right"]["x0_um"], by_normal["right"]["x1_um"]) == (10.0, 10.0)
        # Every direction alias resolves to exactly its segment.
        for direction, edge in by_normal.items():
            assert g["directions"][direction] == [edge["id"]]

    def test_the_segments_are_the_ids_the_engine_accepts(self, client: TestClient) -> None:
        """An override keyed on a previewed id must run; one the preview did
        not list must be refused with the preview's list in the message."""
        body = _preview(client, _setup())
        left = next(e["id"] for e in body["geometry"]["edges"] if e["normal"] == "left")
        accepted = _preview(client, _setup(
            boundary__per_edge={left: {"kind": "absorbing", "value": 0.0, "aux_value": None}},
        ))
        assert accepted["ok"] is True
        refused = _preview(client, _setup(
            boundary__per_edge={"edge_9999": {"kind": "absorbing", "value": 0.0, "aux_value": None}},
        ))
        assert refused["ok"] is False
        assert "edge_9999" in refused["errors"][0]
        for edge in body["geometry"]["edges"]:
            assert edge["id"] in refused["errors"][0]

    def test_a_strip_has_no_up_or_down_in_its_directions_only_when_it_lacks_them(
        self, client: TestClient
    ) -> None:
        """A 1-row strip still has up/down faces (its long sides); the aliases
        must report what the mask has, not what a name suggests."""
        body = _preview(client, _setup(geometry__rows=1))
        directions = body["geometry"]["directions"]
        assert all(len(ids) == 1 for ids in directions.values()), directions
        assert body["geometry"]["dimensionality"] == 1

    def test_the_mask_figure_is_the_device(self, client: TestClient) -> None:
        """Rendered twice with different masks: the bytes must differ."""
        wide = _png(_preview(client, _setup())["images"]["mask"])
        narrow = _png(_preview(client, _setup(geometry__cols=2))["images"]["mask"])
        assert wide != narrow


class TestTheSeedFigures:
    def test_the_seed_map_follows_the_seed(self, client: TestClient) -> None:
        thermal = _png(_preview(client, _setup())["images"]["seed_xqp"])
        excess = _png(_preview(client, _setup(
            initial__kind="excess", initial__amplitude=1e-3,
            initial__space__kind="point", initial__space__x_0=0.0,
        ))["images"]["seed_xqp"])
        assert thermal != excess

    def test_the_phonon_seed_is_shown_only_when_there_is_one(
        self, client: TestClient
    ) -> None:
        pinned = _preview(client, _setup())
        assert pinned["phonons"] == {"mode": "thermal_bath", "seeded": False}
        assert pinned["images"]["phonon_seed"] is None
        at_bath = _preview(client, _setup(phonons__mode="dynamic_escape"))
        assert at_bath["phonons"]["seeded"] is False
        hot = _preview(client, _setup(
            phonons__mode="dynamic_escape",
            phonons__initial__kind="thermal_at", phonons__initial__T_eff=0.6,
        ))
        assert hot["phonons"]["seeded"] is True
        assert hot["phonons"]["n_ph_seed_mean"] > hot["phonons"]["n_ph_bath_mean"]
        _png(hot["images"]["phonon_seed"])


class TestRefusalsAreMessages:
    def test_a_gds_setup_that_cannot_be_read_is_a_message(self, client: TestClient) -> None:
        """Without gdstk the message names the package; with it, the file."""
        body = _preview(client, _setup(geometry__kind="gds", geometry__gds_path="no-such.gds"))
        assert body["ok"] is False
        assert body["geometry"] is None
        assert body["errors"] and (
            "gdstk" in body["errors"][0] or "no-such.gds" in body["errors"][0]
        )

    def test_the_junction_mode_has_no_geometry(self, client: TestClient) -> None:
        body = _preview(client, M25JunctionSetup().model_dump(mode="json"))
        assert body["ok"] is False
        assert "no geometry" in body["errors"][0]

    def test_a_bad_expression_is_a_message(self, client: TestClient) -> None:
        body = _preview(client, _setup(
            initial__kind="excess", initial__expression="E +",
        ))
        assert body["ok"] is False
        assert body["errors"]

    def test_a_schema_error_is_still_422(self, client: TestClient) -> None:
        resp = client.post("/api/preview", json={"name": "p", "setup": {"mode": "kinetics", "T_bath": -1}})
        assert resp.status_code == 422

    def test_build_preview_is_the_same_function_the_route_calls(self) -> None:
        setup = KineticsSetup()
        setup.geometry.rows, setup.geometry.cols = 1, 1
        body = build_preview(setup)
        assert body["ok"] and body["geometry"]["dimensionality"] == 0
        assert json.dumps(body)  # JSON-serialisable: no numpy scalars leak


class TestTheBrowserReachesIt:
    def test_the_page_has_the_button_and_the_panel(self) -> None:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        js = (_STATIC / "app.js").read_text(encoding="utf-8")
        assert 'id="btn-preview"' in html and 'id="preview"' in html
        assert "/api/preview" in js
        assert re.search(r'\$\("#btn-preview"\)\.addEventListener\("click", doPreview\)', js)

    @pytest.mark.skipif(_NODE is None, reason="needs node on PATH")
    def test_previewed_edge_ids_are_offered_by_the_per_edge_table(self) -> None:
        setup_json = json.dumps(KineticsSetup().model_dump(mode="json"))
        script = '''
const fieldAt = (path) => {
  for (const section of FORMS.kinetics) for (const f of section.fields) if (f.path === path) return f;
  throw new Error(`no control bound to ${path}`);
};
state.setup = SETUP;
state.edgeIds = ["edge_0001", "edge_0002", "left"];
const box = renderField(fieldAt("boundary.per_edge"));
box.querySelector(".list-add").click();
const list = box.querySelector("datalist");
RESULT = list.children.map((o) => o.value);
'''.replace("SETUP", setup_json)
        proc = subprocess.run(
            [_NODE, str(_HARNESS), str(_STATIC / "app.js")],
            input=script, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        offered = json.loads(proc.stdout)
        assert offered == ["up", "down", "left", "right", "edge_0001", "edge_0002"]
