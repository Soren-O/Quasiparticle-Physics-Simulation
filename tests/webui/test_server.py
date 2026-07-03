"""End-to-end API tests: create a run over HTTP, poll it, fetch artifacts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi", reason="server tests need the qpsim[ui] extra")
pytest.importorskip("matplotlib", reason="plot endpoints need matplotlib")

from fastapi.testclient import TestClient  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402

TINY_SETUP: dict[str, Any] = {
    "mode": "steady_state_0d",
    "grid": {"num_bins": 48},
}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(tmp_path)) as c:
        yield c


def _wait_done(client: TestClient, run_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.2)
    pytest.fail(f"run {run_id} did not finish within {timeout_s}s")


class TestMetaAndMaterials:
    def test_meta_lists_modes(self, client: TestClient) -> None:
        body = client.get("/api/meta").json()
        assert set(body["modes"]) == {
            "steady_state_0d", "transient_0d", "spatial_1d", "m25_junction",
        }

    def test_materials_carry_autofill_params(self, client: TestClient) -> None:
        mats = client.get("/api/materials").json()
        names = {m["name"] for m in mats}
        assert {"Al", "Nb", "TiN"} <= names
        al = next(m for m in mats if m["name"] == "Al")
        assert al["params"]["Delta_0"] == al["Delta_0"]

    def test_defaults_endpoint(self, client: TestClient) -> None:
        body = client.get("/api/defaults/spatial_1d").json()
        assert body["mode"] == "spatial_1d"
        assert client.get("/api/defaults/nope").status_code == 404

    def test_index_serves_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<title>qpsim</title>" in resp.text


class TestValidationEndpoint:
    def test_valid_setup_passes(self, client: TestClient) -> None:
        body = client.post(
            "/api/validate", json={"name": "t", "setup": TINY_SETUP}
        ).json()
        assert body["ok"] is True

    def test_cross_field_error_reported(self, client: TestClient) -> None:
        setup = {
            "mode": "spatial_1d",
            "material": {"dynes_gamma": 1.0},
        }
        body = client.post("/api/validate", json={"name": "t", "setup": setup}).json()
        assert body["ok"] is False
        assert any("pure-BCS" in e for e in body["errors"])

    def test_schema_error_is_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/validate",
            json={"name": "t", "setup": {"mode": "steady_state_0d", "T_bath": -1.0}},
        )
        assert resp.status_code == 422


class TestSetupsApi:
    def test_save_load_delete(self, client: TestClient) -> None:
        resp = client.post("/api/setups", json={"name": "Tiny", "setup": TINY_SETUP})
        slug = resp.json()["slug"]
        assert [s["slug"] for s in client.get("/api/setups").json()] == [slug]
        body = client.get(f"/api/setups/{slug}").json()
        assert body["setup"]["grid"]["num_bins"] == 48
        client.delete(f"/api/setups/{slug}")
        assert client.get(f"/api/setups/{slug}").status_code == 404


class TestRunLifecycle:
    def test_run_to_completion_with_plots_and_csv(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json={"name": "tiny run", "setup": TINY_SETUP})
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["id"]

        body = _wait_done(client, run_id)
        assert body["status"] == "done", body.get("error")
        assert body["summary"]["x_qp"] > 0.0
        assert set(body["plots"]) == {"occupation", "phonons"}

        png = client.get(f"/api/runs/{run_id}/plots/occupation.png")
        assert png.status_code == 200
        assert png.content[:8] == b"\x89PNG\r\n\x1a\n"

        csv_resp = client.get(f"/api/runs/{run_id}/csv/occupation.csv")
        assert csv_resp.status_code == 200
        header = csv_resp.text.splitlines()[0]
        assert header == "E_ueV,f,f_thermal"

        assert client.get(f"/api/runs/{run_id}/plots/nope.png").status_code == 404

        assert client.delete(f"/api/runs/{run_id}").json()["deleted"] is True
        assert client.get(f"/api/runs/{run_id}").status_code == 404

    def test_invalid_setup_rejected_with_errors(self, client: TestClient) -> None:
        setup = dict(TINY_SETUP, T_bath=2.0)  # above Al T_c
        resp = client.post("/api/runs", json={"name": "bad", "setup": setup})
        assert resp.status_code == 400
        assert any("T_c" in e for e in resp.json()["errors"])
        assert client.get("/api/runs").json() == []
