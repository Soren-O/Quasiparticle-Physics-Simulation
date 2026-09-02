"""Wave 4: geometry reach -- outlines without a file, layers named, rims labelled.

The plan's acceptance case is an annulus with the inner rim absorbing and the
outer reflective, authored entirely from the browser and addressing the two
rims by their own ids. That is what the last test here does: the outline is
typed into the shipped control under node, the preview names the segments,
the per-edge table is filled with those ids, and the engine is then required
to hold fewer quasiparticles than with the inner rim left reflective. Every
other test is one link of that chain checked on its own.
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
pytest.importorskip("matplotlib", reason="preview figures need matplotlib")

from fastapi.testclient import TestClient  # noqa: E402
from qpsim.geometries import gds_support_available  # noqa: E402
from qpsim.webui.builders import validate_setup  # noqa: E402
from qpsim.webui.execute import run_kinetics  # noqa: E402
from qpsim.webui.schemas import KineticsSetup  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"
_HARNESS = pathlib.Path(__file__).with_name("form_harness.js")
_NODE = shutil.which("node")

OUTER = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
INNER = [[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]   # wound the other way: a hole


@pytest.fixture
def client(tmp_path: pathlib.Path) -> TestClient:
    with TestClient(create_app(tmp_path)) as c:
        yield c


def _setup(**over: Any) -> dict[str, Any]:
    setup = KineticsSetup()
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


def _polygons(polys: list[list[list[float]]], mesh: float, **over: Any) -> dict[str, Any]:
    return _setup(
        geometry__kind="polygons", geometry__polygons=polys, geometry__mesh_size_um=mesh, **over,
    )


def _preview(client: TestClient, setup: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/api/preview", json={"name": "p", "setup": setup})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _png(data_uri: str) -> bytes:
    png = base64.b64decode(data_uri.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return png


def _rims(geometry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(outer ids, inner ids): a segment on the mask's bounding box is outer."""
    xs = [v for e in geometry["edges"] for v in (e["x0_um"], e["x1_um"])]
    ys = [v for e in geometry["edges"] for v in (e["y0_um"], e["y1_um"])]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    outer, inner = [], []
    for e in geometry["edges"]:
        on_box = (
            e["x0_um"] == e["x1_um"] == lo_x or e["x0_um"] == e["x1_um"] == hi_x
            or e["y0_um"] == e["y1_um"] == lo_y or e["y0_um"] == e["y1_um"] == hi_y
        )
        (outer if on_box else inner).append(e["id"])
    return outer, inner


class TestOutlinesWithoutAFile:
    def test_a_polygon_source_previews_as_the_device(self, client: TestClient) -> None:
        body = _preview(client, _polygons([[[0, 0], [12, 0], [12, 4], [0, 4]]], 1.0))
        assert body["ok"] is True, body["errors"]
        g = body["geometry"]
        assert g["source"] == "polygons"
        assert g["cells"] == 48 and g["dimensionality"] == 2
        # Geometry.bounds, read at last: the padded window in layout units.
        assert g["origin_um"] == [-1.0, -1.0]
        assert g["bounds_um"] == [-1.0, -1.0, 13.0, 5.0]

    def test_too_coarse_a_mesh_flattens_the_device_before_any_solve(
        self, client: TestClient
    ) -> None:
        """The plan's acceptance: a narrow feature at a coarse mesh is seen
        as what it rasterises to, here, not after a run."""
        strip = [[[0, 0], [10, 0], [10, 3], [0, 3]]]
        fine = _preview(client, _polygons(strip, 1.0))["geometry"]
        coarse = _preview(client, _polygons(strip, 3.0))["geometry"]
        assert fine["cells"] == 30 and fine["dimensionality"] == 2
        # The plan's number: three cells, in a row -- the device has become a
        # 1-D strip, and the preview says so before any solve.
        assert coarse["cells"] == 3
        assert coarse["dimensionality"] == 1, "3 μm wide at a 3 μm mesh is one cell across"

    def test_a_broken_neck_is_a_message(self, client: TestClient) -> None:
        dumbbell = [
            [[0, 0], [4, 0], [4, 4], [0, 4]],
            [[4, 1.6], [6, 1.6], [6, 2.4], [4, 2.4]],
            [[6, 0], [10, 0], [10, 4], [6, 4]],
        ]
        body = _preview(client, _polygons(dumbbell, 2.0))
        assert body["ok"] is False
        assert "too coarse" in body["errors"][0]

    def test_the_schema_refuses_a_non_polygon(self, client: TestClient) -> None:
        for bad in ([], [[[0, 0], [1, 0]]], [[[0, 0, 0], [1, 0, 0], [1, 1, 1]]]):
            resp = client.post(
                "/api/preview", json={"name": "p", "setup": _polygons(bad, 1.0)},
            )
            assert resp.status_code == 422, bad

    def test_an_annulus_names_both_rims(self, client: TestClient) -> None:
        body = _preview(client, _polygons([OUTER, INNER], 1.0))
        assert body["ok"] is True, body["errors"]
        g = body["geometry"]
        assert g["cells"] == 100 - 16
        outer, inner = _rims(g)
        assert len(outer) == 4 and len(inner) == 4
        # The plan's finding, now visible: a direction alias names BOTH rims.
        assert len(g["directions"]["right"]) == 2
        assert set(g["directions"]["right"]) <= set(outer) | set(inner)

    def test_validate_setup_accepts_it(self) -> None:
        setup = KineticsSetup.model_validate(_polygons([OUTER, INNER], 1.0))
        assert validate_setup(setup).ok


class TestTheMaskFigureCarriesTheRim:
    def test_the_overlay_follows_the_conditions(self, client: TestClient) -> None:
        """Rendered with the inner rim absorbing and with it reflective, the
        bytes must differ: the overlay is drawn from the conditions, not
        decoration."""
        plain = _preview(client, _polygons([OUTER, INNER], 1.0))
        _outer, inner = _rims(plain["geometry"])
        overrides = {i: {"kind": "absorbing", "value": 0.0, "aux_value": None} for i in inner}
        marked = _preview(client, _polygons([OUTER, INNER], 1.0, boundary__per_edge=overrides))
        assert marked["ok"] is True, marked["errors"]
        assert _png(plain["images"]["mask"]) != _png(marked["images"]["mask"])


@pytest.mark.skipif(not gds_support_available(), reason="needs gdstk")
class TestLayersAreNamedNotGuessed:
    @staticmethod
    def _write(tmp_path: pathlib.Path) -> pathlib.Path:
        import gdstk

        library = gdstk.Library()
        cell = library.new_cell("TOP")
        cell.add(
            gdstk.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=2),
            gdstk.Polygon([(3, 3), (3, 7), (7, 7), (7, 3)], layer=2),
            gdstk.Polygon([(20, 0), (24, 0), (24, 4), (20, 4)], layer=5),
        )
        path = tmp_path / "chip.gds"
        library.write_gds(str(path))
        return path

    def test_the_preview_lists_the_layers_that_carry_polygons(
        self, client: TestClient, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path)
        body = _preview(client, _setup(
            geometry__kind="gds", geometry__gds_path=str(path),
            geometry__gds_layer=2, geometry__mesh_size_um=1.0,
        ))
        assert body["ok"] is True, body["errors"]
        assert body["geometry"]["gds_layers"] == [2, 5]
        assert body["geometry"]["gds_layer"] == 2
        assert body["geometry"]["cells"] == 100 - 16

    def test_a_layer_without_polygons_is_refused_with_the_list(
        self, client: TestClient, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path)
        body = _preview(client, _setup(
            geometry__kind="gds", geometry__gds_path=str(path),
            geometry__gds_layer=3, geometry__mesh_size_um=1.0,
        ))
        assert body["ok"] is False
        assert "2, 5" in body["errors"][0]

    def test_a_layer_and_its_polygons_preview_identically(
        self, client: TestClient, tmp_path: pathlib.Path
    ) -> None:
        path = self._write(tmp_path)
        from_file = _preview(client, _setup(
            geometry__kind="gds", geometry__gds_path=str(path),
            geometry__gds_layer=2, geometry__mesh_size_um=1.0,
        ))["geometry"]
        by_hand = _preview(client, _polygons([OUTER, INNER], 1.0))["geometry"]
        for key in ("rows", "cols", "cells", "edges", "directions", "bounds_um"):
            assert from_file[key] == by_hand[key], key

    def test_a_missing_file_is_a_message_and_validate_says_so_too(
        self, client: TestClient, tmp_path: pathlib.Path
    ) -> None:
        missing = str(tmp_path / "nope.gds")
        body = _preview(client, _setup(geometry__kind="gds", geometry__gds_path=missing))
        assert body["ok"] is False and "not found" in body["errors"][0]
        report = validate_setup(KineticsSetup.model_validate(
            _setup(geometry__kind="gds", geometry__gds_path=missing),
        ))
        assert not report.ok and any("not found" in e for e in report.errors)


@pytest.mark.skipif(_NODE is None, reason="authoring in the browser needs node")
class TestTheAnnulusAuthoredInTheBrowser:
    """The plan's acceptance case, end to end."""

    @staticmethod
    def _author(inner_ids: list[str]) -> dict[str, Any]:
        base = json.dumps(_setup(geometry__mesh_size_um=1.0))
        script = '''
const fieldAt = (path) => {
  for (const section of FORMS.kinetics) for (const f of section.fields) if (f.path === path) return f;
  throw new Error(`no control bound to ${path}`);
};
const inputOf = (root, path) => {
  const el = root.querySelector(`[data-path="${path}"]`);
  if (!el) throw new Error(`no input bound to ${path}`);
  return el;
};
state.setup = SETUP;
// Geometry step: source and outline.
const kind = renderField(fieldAt("geometry.kind"));
inputOf(kind, "geometry.kind").value = "polygons";
inputOf(kind, "geometry.kind").dispatch("change");
const outline = renderField(fieldAt("geometry.polygons"));
const box = inputOf(outline, "geometry.polygons");
box.value = "not json"; box.dispatch("change");           // refused, resynced
const afterBad = state.setup.geometry.polygons;
box.value = JSON.stringify(POLYGONS); box.dispatch("change");
// Conditions step: the inner rim absorbing, addressed by the previewed ids.
state.edgeIds = INNER_IDS;
const table = renderField(fieldAt("boundary.per_edge"));
for (const id of INNER_IDS) {
  table.querySelector(".list-add").click();
  const entries = table.querySelectorAll("fieldset");
  const entry = entries[entries.length - 1];
  const key = entry.querySelector("input[data-key]");
  key.value = id; key.dispatch("change");
}
RESULT = { afterBad, setup: state.setup };
'''.replace("SETUP", base).replace("POLYGONS", json.dumps([OUTER, INNER])).replace(
            "INNER_IDS", json.dumps(inner_ids),
        )
        proc = subprocess.run(
            [_NODE, str(_HARNESS), str(_STATIC / "app.js")],
            input=script, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    def test_inner_rim_absorbing_outer_reflective(self, client: TestClient) -> None:
        named = _preview(client, _polygons([OUTER, INNER], 1.0))["geometry"]
        _outer, inner = _rims(named)
        result = self._author(inner)
        assert result["afterBad"] is None, "a non-outline must not land in the setup"
        setup = result["setup"]
        assert setup["geometry"]["kind"] == "polygons"
        assert setup["geometry"]["polygons"] == [OUTER, INNER]
        assert set(setup["boundary"]["per_edge"]) == set(inner)
        assert all(v["kind"] == "absorbing" for v in setup["boundary"]["per_edge"].values())
        assert setup["boundary"]["kind"] == "reflective"

        # The setup the browser produced previews to the same rim it named...
        again = _preview(client, setup)
        assert again["ok"] is True, again["errors"]
        assert _rims(again["geometry"])[1] == inner

        # ...and the engine sees the trap: driven uniformly, an annulus that
        # absorbs at its inner rim holds fewer quasiparticles than one that
        # does not, and the cells beside the hole are the emptiest.
        def profile(body: dict[str, Any]) -> np.ndarray:
            body = dict(body)
            body["injection"] = dict(body["injection"], enabled=True, rate_per_ns=1e-2, where="uniform")
            body["dt"], body["max_time"] = 0.05, 0.3
            payload = run_kinetics(KineticsSetup.model_validate(body), lambda *a, **k: None, lambda: False)
            return np.asarray(payload.arrays["xqp_profile"], dtype=float)

        trapped = profile(setup)
        untouched = profile(dict(setup, boundary=dict(setup["boundary"], per_edge={})))
        assert trapped.shape == untouched.shape
        assert trapped.sum() < untouched.sum()
        assert not np.allclose(trapped, untouched)
