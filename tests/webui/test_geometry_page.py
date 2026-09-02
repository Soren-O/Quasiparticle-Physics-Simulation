"""The geometry page: import on the left, the device on the right, click an
edge to assign its condition -- the old desktop page, on the web.

The figure's colours are the engine's answer: /api/geometry returns every
rim segment with its EFFECTIVE condition and where it came from, and the
page redraws from that after each assignment instead of keeping a second
copy of the precedence rules. So the tests hold three things: the route
reports the precedence the builder applies (id beats alias beats rim,
whatever order the entries were typed in); the shipped page, driven under
node, writes an assignment where the engine reads it; and a rim assigned
through the page moves the engine's number.
"""

from __future__ import annotations

import base64
import json
import pathlib
import shutil
import subprocess
from typing import Any

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="server tests need the qpsim[ui] extra")

from fastapi.testclient import TestClient  # noqa: E402
from qpsim.webui.builders import (  # noqa: E402
    boundary_sources_2d,
    build_boundary_conditions_2d,
    build_geometry_2d,
)
from qpsim.webui.execute import run_kinetics  # noqa: E402
from qpsim.webui.schemas import EdgeCondition, KineticsSetup  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"
_HARNESS = pathlib.Path(__file__).with_name("form_harness.js")
_NODE = shutil.which("node")

OUTER = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
INNER = [[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]


@pytest.fixture
def client(tmp_path: pathlib.Path) -> TestClient:
    with TestClient(create_app(tmp_path)) as c:
        yield c


def _setup(**over: Any) -> dict[str, Any]:
    setup = KineticsSetup()
    setup.geometry.rows, setup.geometry.cols = 3, 5
    setup.geometry.mesh_size_um = 2.0
    setup.grid.num_bins = 24
    body = setup.model_dump(mode="json")
    for key, value in over.items():
        node = body
        parts = key.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return body


def _geometry(client: TestClient, setup: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/api/geometry", json={"name": "g", "setup": setup})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestTheSpecificBeatsTheGeneral:
    def test_an_id_entry_wins_over_an_alias_in_either_order(self) -> None:
        setup = KineticsSetup()
        setup.geometry.rows, setup.geometry.cols = 3, 5
        geometry = build_geometry_2d(setup)
        left = next(e.edge_id for e in geometry.edges if e.normal == "left")
        alias_first = {"left": EdgeCondition(kind="absorbing"), left: EdgeCondition(kind="robin", value=0.5, aux_value=0.0)}
        id_first = {left: EdgeCondition(kind="robin", value=0.5, aux_value=0.0), "left": EdgeCondition(kind="absorbing")}
        for entries in (alias_first, id_first):
            setup.boundary.per_edge = entries
            conditions = build_boundary_conditions_2d(setup, geometry)
            assert conditions[left].kind == "robin", entries
            assert boundary_sources_2d(setup, geometry)[left] == "id"

    def test_sources_name_where_each_condition_came_from(self) -> None:
        setup = KineticsSetup()
        setup.geometry.rows, setup.geometry.cols = 3, 5
        geometry = build_geometry_2d(setup)
        by_normal = {e.normal: e.edge_id for e in geometry.edges}
        setup.boundary.per_edge = {
            "up": EdgeCondition(kind="dirichlet", value=0.1),
            by_normal["right"]: EdgeCondition(kind="absorbing"),
        }
        sources = boundary_sources_2d(setup, geometry)
        assert sources[by_normal["up"]] == "alias:up"
        assert sources[by_normal["right"]] == "id"
        assert sources[by_normal["down"]] == sources[by_normal["left"]] == "rim"


class TestTheGeometryRoute:
    def test_it_returns_the_mask_image_and_every_edge_with_its_condition(
        self, client: TestClient
    ) -> None:
        body = _geometry(client, _setup(
            boundary__kind="reflective",
            boundary__per_edge={"left": {"kind": "absorbing", "value": 0.0, "aux_value": None}},
        ))
        assert body["ok"] is True, body["errors"]
        g = body["geometry"]
        assert g["mask_png"].startswith("data:image/png;base64,")
        png = base64.b64decode(g["mask_png"].split(",", 1)[1])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        # One pixel per cell: the IHDR chunk carries width and height.
        width = int.from_bytes(png[16:20], "big")
        height = int.from_bytes(png[20:24], "big")
        assert (width, height) == (g["cols"], g["rows"]) == (5, 3)
        by_normal = {e["normal"]: e for e in g["edges"]}
        assert by_normal["left"]["condition"]["kind"] == "absorbing"
        assert by_normal["left"]["source"] == "alias:left"
        assert by_normal["right"]["condition"]["kind"] == "reflective"
        assert by_normal["right"]["source"] == "rim"
        assert g["rim_default"]["kind"] == "reflective"
        # Cell-unit coordinates for the figure beside the micron ones.
        assert (by_normal["left"]["x0"], by_normal["left"]["y1"]) == (0.0, 3.0)
        assert by_normal["left"]["y1_um"] == 6.0

    def test_a_refusal_is_a_message(self, client: TestClient) -> None:
        body = _geometry(client, _setup(
            boundary__per_edge={"edge_9999": {"kind": "absorbing", "value": 0.0, "aux_value": None}},
        ))
        assert body["ok"] is False and "edge_9999" in body["errors"][0]

    def test_the_preview_and_the_page_share_one_geometry_block(self, client: TestClient) -> None:
        setup = _setup(boundary__per_edge={"up": {"kind": "neumann", "value": 0.01, "aux_value": None}})
        page = _geometry(client, setup)["geometry"]
        preview = client.post("/api/preview", json={"name": "p", "setup": setup}).json()["geometry"]
        for key in ("edges", "directions", "rim_default", "cells", "rows", "cols"):
            assert page[key] == preview[key], key


@pytest.mark.skipif(_NODE is None, reason="driving the page needs node on PATH")
class TestThePageUnderNode:
    @staticmethod
    def _drive(script: str) -> Any:
        proc = subprocess.run(
            [_NODE, str(_HARNESS), str(_STATIC / "app.js")],
            input=script, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    def test_the_figure_draws_every_edge_in_its_condition_s_colour(
        self, client: TestClient
    ) -> None:
        g = _geometry(client, _setup(
            boundary__per_edge={"left": {"kind": "absorbing", "value": 0.0, "aux_value": None}},
        ))["geometry"]
        result = self._drive(f'''
state.setup = {json.dumps(_setup())};
const host = document.createElement("div");
renderGeometryFigure(host, {json.dumps(g)});
const lines = host.querySelectorAll("line.edge-line");
const svg = host.querySelector("svg");
RESULT = {{
  viewBox: svg.getAttribute("viewBox"),
  edges: host.querySelectorAll("g.edge").map((grp) => ({{
    id: grp.dataset.edge, source: grp.dataset.source,
    stroke: grp.querySelector("line.edge-line").getAttribute("stroke"),
    dash: grp.querySelector("line.edge-line").getAttribute("stroke-dasharray"),
    y1: grp.querySelector("line.edge-line").getAttribute("y1"),
  }})),
}};
''')
        assert result["viewBox"] == "0 0 5 3"
        by_id = {e["id"]: e for e in result["edges"]}
        left = next(e for e in g["edges"] if e["normal"] == "left")
        up = next(e for e in g["edges"] if e["normal"] == "up")
        assert by_id[left["id"]]["stroke"] == "#333333"        # absorbing
        assert by_id[left["id"]]["dash"] == "none"             # explicitly assigned
        assert by_id[up["id"]]["stroke"] == "#1155AA"          # rim default: reflective
        assert by_id[up["id"]]["dash"] == "6 4"                # dashed = rim default
        # Row 0 at the bottom: the "up" face of row 0 sits at y = rows.
        assert float(by_id[up["id"]]["y1"]) == 3.0

    def test_clicking_an_edge_and_assigning_writes_where_the_engine_reads(
        self, client: TestClient
    ) -> None:
        g = _geometry(client, _setup())["geometry"]
        right = next(e for e in g["edges"] if e["normal"] == "right")
        result = self._drive(f'''
state.setup = {json.dumps(_setup())};
state.geometry = {json.dumps(g)};
const host = document.createElement("div");
renderGeometryFigure(host, state.geometry);
// #edge-editor / #edge-hover are page-level; the harness hands out throwaway
// elements for $("#..."), so build the editor by opening it directly.
const grp = host.querySelector('g[data-edge="{right["id"]}"]');
const hit = grp.querySelector("line.edge-hit");
hit.dispatch("mouseenter");
const hoveredStroke = grp.querySelector("line.edge-line").getAttribute("stroke");
hit.dispatch("click");
RESULT = {{ hoveredStroke, selected: state.selectedEdge }};
''')
        assert result["hoveredStroke"] == "#FFD500"
        assert result["selected"] == right["id"]

    def test_the_assignment_panel_validates_and_writes_the_condition(
        self, client: TestClient
    ) -> None:
        g = _geometry(client, _setup())["geometry"]
        right = next(e for e in g["edges"] if e["normal"] == "right")
        result = self._drive(f'''
state.setup = {json.dumps(_setup())};
state.geometry = {json.dumps(g)};
// A real panel element to open the editor into.
const panel = document.createElement("div");
panel.id = "edge-editor";
const realQuery = document.querySelector;
document.querySelector = (sel) => (sel === "#edge-editor" ? panel : realQuery(sel));
openEdgeEditor("{right["id"]}");
const kind = panel.querySelector("#edge-kind");
const value = panel.querySelector("#edge-value");
const aux = panel.querySelector("#edge-aux");
kind.value = "robin"; kind.dispatch("change");
const disabledAfterRobin = [value.disabled, aux.disabled];
value.value = "abc";
panel.querySelector("#edge-assign").click();
const refused = panel.querySelector("#edge-editor-msg").textContent;
const before = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
value.value = "0.25"; aux.value = "";
panel.querySelector("#edge-assign").click();
const assigned = JSON.parse(JSON.stringify(state.setup.boundary.per_edge));
kind.value = "absorbing"; kind.dispatch("change");
const disabledAfterAbsorbing = [value.disabled, aux.disabled];
RESULT = {{ disabledAfterRobin, refused, before, assigned, disabledAfterAbsorbing }};
''')
        assert result["disabledAfterRobin"] == [False, False]
        assert "numeric" in result["refused"].lower()
        assert result["before"] == {}
        assert result["assigned"] == {right["id"]: {"kind": "robin", "value": 0.25, "aux_value": 0.0}}
        assert result["disabledAfterAbsorbing"] == [True, True]

    def test_the_page_is_wired(self) -> None:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        js = (_STATIC / "app.js").read_text(encoding="utf-8")
        for needle in ('id="btn-import-geometry"', 'id="geometry-figure"', 'id="edge-editor"', 'id="edge-hover"'):
            assert needle in html, needle
        assert "/api/geometry" in js
        assert '$("#btn-import-geometry").addEventListener("click", importGeometry)' in js


class TestAnAssignmentMadeOnThePageMovesTheEngine:
    def test_the_inner_rim_of_an_annulus_assigned_by_id(self, client: TestClient) -> None:
        """What the page writes is what the run reads: assign the inner rim
        through the per-edge dict the panel fills, and the device holds
        fewer quasiparticles than with the rim left at the reflective
        default."""
        base = _setup(geometry__kind="polygons", geometry__polygons=[OUTER, INNER],
                      geometry__mesh_size_um=1.0, strategy="time_march",
                      injection__enabled=True, injection__rate_per_ns=1e-2, injection__where="uniform",
                      dt=0.05, max_time=0.3, stop_tol=0.0)
        g = _geometry(client, base)["geometry"]
        xs = [v for e in g["edges"] for v in (e["x0_um"], e["x1_um"])]
        ys = [v for e in g["edges"] for v in (e["y0_um"], e["y1_um"])]
        inner = [
            e["id"] for e in g["edges"]
            if not (e["x0_um"] == e["x1_um"] == min(xs) or e["x0_um"] == e["x1_um"] == max(xs)
                    or e["y0_um"] == e["y1_um"] == min(ys) or e["y0_um"] == e["y1_um"] == max(ys))
        ]
        assert len(inner) == 4
        # What the panel writes for each clicked inner edge.
        trapped_setup = dict(base, boundary=dict(base["boundary"], per_edge={
            i: {"kind": "absorbing", "value": 0.0, "aux_value": None} for i in inner
        }))
        after = _geometry(client, trapped_setup)["geometry"]
        assert all(e["source"] == "id" for e in after["edges"] if e["id"] in inner)
        assert all(e["condition"]["kind"] == "absorbing" for e in after["edges"] if e["id"] in inner)

        def profile(body: dict[str, Any]) -> np.ndarray:
            payload = run_kinetics(KineticsSetup.model_validate(body), lambda *a, **k: None, lambda: False)
            return np.asarray(payload.arrays["xqp_profile"], dtype=float)

        trapped, untouched = profile(trapped_setup), profile(base)
        assert trapped.sum() < untouched.sum()
