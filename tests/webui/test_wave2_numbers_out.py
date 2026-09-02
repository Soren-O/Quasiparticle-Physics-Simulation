"""Wave 2: the numbers get OUT of a run on a geometry.

Before this wave a spatial run's result was a set of PNGs, a few summary
scalars, and tables of its endpoint only. The phonon sector on a geometry was
view-only; the recorded frames could be scrubbed through but not downloaded;
and the arrays nobody had drawn a figure for could not leave the server at all.

Every test here compares NUMBERS against the arrays the run wrote, never PNG
bytes or "the request succeeded": a table that is offered, downloads, and
holds the endpoint when a frame was asked for would pass a status-code test
and be exactly the defect this repo keeps finding. The run is DRIVEN for the
same reason -- undriven, every frame is the thermal fixed point and frame 0
equals frame N, so a frame test would be vacuous.
"""

from __future__ import annotations

import csv
import io
import pathlib
import re
import time
from typing import Any

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="server tests need the qpsim[ui] extra")
pytest.importorskip("matplotlib", reason="plot endpoints need matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from qpsim.webui import plots  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402
from qpsim.webui.store import Workspace  # noqa: E402

_APP_JS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "qpsim" / "webui" / "static" / "app.js"
)


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


def _driven_setup(client: TestClient, **over: Any) -> dict[str, Any]:
    """A small DRIVEN strip with a dynamic phonon sector and recorded frames."""
    setup = client.get("/api/defaults/kinetics").json()
    setup["T_bath"] = 0.2
    setup["grid"]["num_bins"] = 24
    setup["geometry"]["rows"] = 1
    setup["geometry"]["cols"] = 4
    setup["strategy"] = "time_march"
    setup["injection"]["enabled"] = True
    setup["injection"]["rate_per_ns"] = 1e-2
    setup["dt"] = 0.1
    setup["max_time"] = 0.4
    setup["snapshot_interval"] = 0.1
    setup["stop_tol"] = 0.0
    setup["phonons"]["mode"] = "dynamic_escape"
    for key, value in over.items():
        node = setup
        parts = key.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return setup


def _recorded(client: TestClient, **over: Any) -> dict[str, Any]:
    run_id = client.post(
        "/api/runs", json={"name": "wave2", "setup": _driven_setup(client, **over)}
    ).json()["id"]
    manifest = _wait_done(client, run_id)
    assert manifest["status"] == "done", manifest.get("error")
    return manifest


def _arrays(client: TestClient, run_id: str) -> dict[str, np.ndarray]:
    """The run's arrays, fetched through the route under test."""
    resp = client.get(f"/api/runs/{run_id}/result.npz")
    assert resp.status_code == 200, resp.text
    with np.load(io.BytesIO(resp.content), allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _table(text: str) -> tuple[list[str], np.ndarray]:
    """A CSV as (header, float matrix); parsed with csv, not genfromtxt,
    because the per-cell column names carry ``=`` signs."""
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    return header, np.array([[float(v) for v in row] for row in body], dtype=float)


def _column(text: str, name: str) -> np.ndarray:
    header, data = _table(text)
    return data[:, header.index(name)]


def _csv(client: TestClient, run_id: str, name: str, frame: int | None = None) -> str:
    query = "" if frame is None else f"?frame={frame}"
    resp = client.get(f"/api/runs/{run_id}/csv/{name}.csv{query}")
    assert resp.status_code == 200, resp.text
    return resp.text


class TestTheArraysLeaveTheServerWhole:
    def test_result_npz_is_the_file_the_run_wrote(
        self, client: TestClient, tmp_path: pathlib.Path
    ) -> None:
        manifest = _recorded(client)
        resp = client.get(f"/api/runs/{manifest['id']}/result.npz")
        assert resp.status_code == 200
        disposition = resp.headers.get("content-disposition", "")
        assert disposition.startswith("attachment")
        assert re.search(r'filename="?[^";]+\.npz"?', disposition), disposition
        assert "max-age" in resp.headers.get("cache-control", "")
        on_disk = Workspace(tmp_path).run_dir(manifest["id"]) / "result.npz"
        assert resp.content == on_disk.read_bytes()

    def test_it_holds_the_arrays_no_figure_draws(self, client: TestClient) -> None:
        """The point of the route: what the interface never reduced."""
        arrays = _arrays(client, _recorded(client)["id"])
        assert {"snap_f", "snap_n_ph", "snap_omega_bins", "snap_gap",
                "obs_x_qp_mean", "obs_x_qp_max", "x_um"} <= set(arrays)

    def test_a_missing_run_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/no-such-run/result.npz").status_code == 404

    def test_a_run_id_cannot_escape_the_workspace(self, client: TestClient) -> None:
        assert client.get("/api/runs/..%2F..%2Fetc/result.npz").status_code == 404


class TestPhononsLeaveAGeometryRun:
    def test_a_dynamic_sector_offers_the_phonon_table(self, client: TestClient) -> None:
        manifest = _recorded(client)
        assert "phonons" in manifest["csvs"]

    def test_a_pinned_sector_does_not(self, client: TestClient) -> None:
        """Guards the premise: the table must come from recorded phonons."""
        manifest = _recorded(client, phonons__mode="thermal_bath")
        assert "phonons" not in manifest["csvs"]

    def test_the_table_is_the_recorded_frame_on_the_recorded_lattice(
        self, client: TestClient
    ) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        last = arrays["snap_t_ns"].size - 1
        header, data = _table(_csv(client, manifest["id"], "phonons"))
        assert header[:2] == ["omega_ueV", "t_ns"]
        assert len(header) == 2 + arrays["mask"].sum()
        np.testing.assert_allclose(
            data[:, 0], arrays["snap_omega_bins"], rtol=1e-9,
        )
        # The endpoint table is the LAST recorded frame and says so.
        np.testing.assert_allclose(data[:, 1], arrays["snap_t_ns"][last], rtol=1e-9)
        np.testing.assert_allclose(data[:, 2:], arrays["snap_n_ph"][last], rtol=1e-9)

    def test_it_is_not_offered_without_its_lattice(self) -> None:
        """Populations without the axis they live on is the bin-count defect."""
        have = {"snap_n_ph", "snap_t_ns", "mask"}
        assert "phonons" not in plots.available_csvs("kinetics", have)
        assert "phonons" in plots.available_csvs("kinetics", have | {"snap_omega_bins"})
        assert "phonons" in plots.available_csvs("kinetics", {"n_ph", "omega_bins"})


class TestTheFrameParameterSelectsAFrame:
    def test_frame_0_and_the_last_frame_hold_different_numbers(
        self, client: TestClient
    ) -> None:
        manifest = _recorded(client)
        last = manifest["plot_params"]["field_over_time"]["frame"] - 1
        assert last >= 1, "need at least two frames to tell them apart"
        for table in ("profile", "phonons", "occupation"):
            first = _table(_csv(client, manifest["id"], table, 0))[1]
            final = _table(_csv(client, manifest["id"], table, last))[1]
            assert first.shape == final.shape
            assert not np.allclose(first, final), table

    def test_a_frame_is_the_matching_snapshot(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        for k in range(arrays["snap_t_ns"].size):
            profile = _csv(client, manifest["id"], "profile", k)
            np.testing.assert_allclose(
                _column(profile, "x_qp"), arrays["snap_xqp_profile"][k], rtol=1e-9,
            )
            np.testing.assert_allclose(
                _column(profile, "gap_ueV"), arrays["snap_gap"][k], rtol=1e-9,
            )
            np.testing.assert_allclose(
                _column(profile, "t_ns"), arrays["snap_t_ns"][k], rtol=1e-9,
            )
            occupation = _csv(client, manifest["id"], "occupation", k)
            np.testing.assert_allclose(
                _table(occupation)[1][:, 2:], arrays["snap_f"][k], rtol=1e-9,
            )
            phonons = _csv(client, manifest["id"], "phonons", k)
            np.testing.assert_allclose(
                _table(phonons)[1][:, 2:], arrays["snap_n_ph"][k], rtol=1e-9,
            )

    def test_omitting_the_frame_is_the_endpoint(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        profile = _csv(client, manifest["id"], "profile")
        np.testing.assert_allclose(_column(profile, "x_qp"), arrays["xqp_profile"], rtol=1e-9)
        np.testing.assert_allclose(
            _column(profile, "t_ns"), manifest["summary"]["total_time_ns"], rtol=1e-9,
        )
        _header, data = _table(_csv(client, manifest["id"], "occupation"))
        np.testing.assert_allclose(data[:, 2:], arrays["f_final"], rtol=1e-9)

    def test_the_filename_names_the_frame(self, client: TestClient) -> None:
        manifest = _recorded(client)
        resp = client.get(f"/api/runs/{manifest['id']}/csv/profile.csv?frame=1")
        assert "-frame1.csv" in resp.headers["content-disposition"]

    def test_a_frame_the_run_does_not_have_is_a_404(self, client: TestClient) -> None:
        manifest = _recorded(client)
        assert client.get(
            f"/api/runs/{manifest['id']}/csv/profile.csv?frame=9999"
        ).status_code == 404

    def test_a_frame_on_a_run_without_frames_is_a_404(self, client: TestClient) -> None:
        """Not the endpoint with the parameter ignored."""
        setup = _driven_setup(client)
        setup["snapshot_interval"] = None
        run_id = client.post("/api/runs", json={"name": "no frames", "setup": setup}).json()["id"]
        manifest = _wait_done(client, run_id)
        assert manifest["status"] == "done", manifest.get("error")
        assert client.get(f"/api/runs/{run_id}/csv/profile.csv").status_code == 200
        assert client.get(f"/api/runs/{run_id}/csv/profile.csv?frame=0").status_code == 404

    def test_a_table_without_a_frame_axis_refuses_one(self, client: TestClient) -> None:
        manifest = _recorded(client)
        assert client.get(
            f"/api/runs/{manifest['id']}/csv/time_series.csv?frame=0"
        ).status_code == 404

    def test_micron_columns_are_cell_centres(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        profile = _csv(client, manifest["id"], "profile")
        mesh = manifest["summary"]["mesh_size_um"]
        np.testing.assert_allclose(
            _column(profile, "x_um"), (_column(profile, "col") + 0.5) * mesh, rtol=1e-9,
        )
        np.testing.assert_allclose(
            _column(profile, "y_um"), (_column(profile, "row") + 0.5) * mesh, rtol=1e-9,
        )
        # And the same coordinate the strip route records for its plots.
        np.testing.assert_allclose(_column(profile, "x_um"), arrays["x_um"], rtol=1e-9)


class TestTheTimeSeriesFigure:
    def test_it_plots_obs_x_qp_mean_element_wise(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        assert np.ptp(arrays["obs_x_qp_mean"]) > 0.0, "undriven: the test would be vacuous"
        fig, ax, residual = plots._draw_xqp_time_series(arrays, manifest["summary"])
        try:
            lines = {line.get_label(): line for line in ax.get_lines()}
            np.testing.assert_array_equal(lines["cell mean"].get_xdata(), arrays["snap_t_ns"])
            np.testing.assert_array_equal(lines["cell mean"].get_ydata(), arrays["obs_x_qp_mean"])
            np.testing.assert_array_equal(lines["cell max"].get_ydata(), arrays["obs_x_qp_max"])
            assert residual is not None
            (rate_line,) = residual.get_lines()
            np.testing.assert_array_equal(
                rate_line.get_ydata(), plots._positive(arrays["snap_max_rate"]),
            )
        finally:
            plt.close(fig)

    def test_the_axis_names_the_recorded_convention(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        convention = manifest["summary"]["x_qp_convention"]
        fig, ax, _ = plots._draw_xqp_time_series(arrays, manifest["summary"])
        try:
            assert convention in ax.get_ylabel()
        finally:
            plt.close(fig)
        fig, ax, _ = plots._draw_xqp_time_series(arrays, {})
        try:
            assert "not recorded" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_it_is_offered_and_renders_over_http(self, client: TestClient) -> None:
        manifest = _recorded(client)
        assert "xqp_over_time" in manifest["plots"]
        assert "xqp_over_time" not in manifest["plot_params"], "a single figure, not a family"
        resp = client.get(f"/api/runs/{manifest['id']}/plots/xqp_over_time.png")
        assert resp.status_code == 200
        assert resp.content[:4] == b"\x89PNG"

    def test_it_needs_a_time_series(self) -> None:
        have = {"snap_t_ns", "mask"}
        assert "xqp_over_time" not in plots.available_plots("kinetics", have)
        assert "xqp_over_time" in plots.available_plots("kinetics", have | {"obs_x_qp_mean"})


class TestTheTimeSeriesTable:
    def test_it_carries_the_phonon_mean_and_both_conventions(
        self, client: TestClient
    ) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        text = _csv(client, manifest["id"], "time_series")
        header, _ = _table(text)
        assert {"t_ns", "x_qp_mean", "x_qp_mean_paper", "n_ph_mean"} <= set(header)
        np.testing.assert_allclose(
            _column(text, "n_ph_mean"), arrays["snap_n_ph"].mean(axis=(1, 2)), rtol=1e-9,
        )
        np.testing.assert_allclose(
            _column(text, "x_qp_mean_paper"), arrays["obs_x_qp_mean_paper"], rtol=1e-9,
        )
        assert np.ptp(_column(text, "n_ph_mean")) > 0.0

    def test_a_pinned_sector_has_no_phonon_column(self, client: TestClient) -> None:
        manifest = _recorded(client, phonons__mode="thermal_bath")
        header, _ = _table(_csv(client, manifest["id"], "time_series"))
        assert "n_ph_mean" not in header


class TestThePhononOccupationMap:
    def test_it_integrates_over_the_recorded_lattice(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        omega = arrays["snap_omega_bins"]
        for k in range(arrays["snap_t_ns"].size):
            expected = np.trapezoid(arrays["snap_n_ph"][k], omega, axis=0)
            np.testing.assert_allclose(
                plots._phonon_occupation_integral(arrays, k), expected, rtol=1e-12,
            )
        first = plots._phonon_occupation_integral(arrays, 0)
        last = plots._phonon_occupation_integral(arrays, arrays["snap_t_ns"].size - 1)
        assert not np.allclose(first, last), "the map must move with the frame"

    def test_a_lattice_of_the_wrong_length_is_refused(self, client: TestClient) -> None:
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        arrays["snap_omega_bins"] = arrays["snap_omega_bins"][:-1]
        with pytest.raises(ValueError, match="wrong frequencies"):
            plots._phonon_occupation_integral(arrays, 0)

    def test_it_is_a_family_offered_with_its_lattice(self, client: TestClient) -> None:
        manifest = _recorded(client)
        frames = manifest["plot_params"]["field_over_time"]["frame"]
        assert manifest["plot_params"]["phonon_occupation_map"] == {"frame": frames}
        have = {"snap_n_ph", "snap_t_ns", "mask"}
        assert "phonon_occupation_map" not in plots.available_plots("kinetics", have)
        assert "phonon_occupation_map" in plots.available_plots(
            "kinetics", have | {"snap_omega_bins"}
        )

    def test_each_frame_renders(self, client: TestClient) -> None:
        manifest = _recorded(client)
        for k in range(manifest["plot_params"]["phonon_occupation_map"]["frame"]):
            resp = client.get(
                f"/api/runs/{manifest['id']}/plots/phonon_occupation_map.png?frame={k}"
            )
            assert resp.status_code == 200, resp.text
            assert resp.content[:4] == b"\x89PNG"


class TestTheGapMap:
    def test_it_is_a_family_over_the_recorded_gap(self, client: TestClient) -> None:
        manifest = _recorded(client)
        frames = manifest["plot_params"]["field_over_time"]["frame"]
        assert manifest["plot_params"]["gap_over_time"] == {"frame": frames}

    def test_the_figure_reads_snap_gap(self, client: TestClient) -> None:
        """Rendered twice, with the recorded gap and with a scaled one, and
        the output must change -- PNG bytes cannot show which array was used,
        but they can show that the array was used at all."""
        manifest = _recorded(client)
        arrays = _arrays(client, manifest["id"])
        summary = manifest["summary"]
        recorded = plots.render_plot("kinetics", "gap_over_time", arrays, summary, {"frame": 0})
        scaled = dict(arrays)
        scaled["snap_gap"] = arrays["snap_gap"] * 1.5
        assert plots.render_plot("kinetics", "gap_over_time", scaled, summary, {"frame": 0}) != recorded

    def test_a_gap_that_never_moves_is_called_uniform(self) -> None:
        """A pinned gap is the everyday case. Its colour scale is a synthetic
        spread (a norm needs vmin < vmax), and the bar must say so rather
        than print 0.000025 μeV of structure that is not there."""
        pinned = np.full((3, 5), 180.0)
        assert isinstance(plots._frame_norm(pinned), plots._UniformNorm)
        moving = pinned.copy()
        moving[-1, 0] -= 1.0
        norm = plots._frame_norm(moving)
        assert not isinstance(norm, plots._UniformNorm)
        assert (norm.vmin, norm.vmax) == (179.0, 180.0)


class TestTheBrowserOffersWhatTheServerServes:
    """Reachability, the lesson of Wave 1: a route the page never links to is
    a capability that does not exist for the person using it."""

    def test_the_page_links_the_new_routes(self) -> None:
        text = _APP_JS.read_text(encoding="utf-8")
        assert "/result.npz" in text
        assert "?frame=" in text
        assert 'download="' in text or "download=" in text

    def test_every_frame_csv_the_page_offers_is_a_table_with_a_frame_axis(self) -> None:
        text = _APP_JS.read_text(encoding="utf-8")
        block = re.search(r"const FRAME_CSV = \{(.*?)\};", text, re.S)
        assert block, "FRAME_CSV map missing from app.js"
        pairs = re.findall(r"(\w+):\s*\"(\w+)\"", block.group(1))
        assert pairs
        kinetics_plots = plots._PLOTS["kinetics"]
        for family, table in pairs:
            assert "frame" in kinetics_plots[family].params, family
            assert table in plots._CSVS["kinetics"], table
            assert table != "time_series", "time_series has no frame axis"
