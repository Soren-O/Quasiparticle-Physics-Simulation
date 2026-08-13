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
    def test_foreign_host_is_rejected(self, client: TestClient) -> None:
        resp = client.get("/api/meta", headers={"host": "attacker.example"})
        assert resp.status_code == 400
        assert resp.text == "Invalid host header"

    @pytest.mark.parametrize(
        "host",
        ["localhost:8000", "127.0.0.1:8000", "127.0.0.2:8000", "[::1]:8000"],
    )
    def test_local_hosts_are_allowed(self, client: TestClient, host: str) -> None:
        assert client.get("/api/meta", headers={"host": host}).status_code == 200

    def test_meta_lists_modes(self, client: TestClient) -> None:
        body = client.get("/api/meta").json()
        assert set(body["modes"]) == {
            "steady_state_0d", "transient_0d", "spatial_1d", "spatial_2d",
            "m25_junction",
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

    def test_equal_gap_interface_is_rejected_before_run(self, client: TestClient) -> None:
        setup = {
            "mode": "spatial_1d",
            "gap_profile": {
                "kind": "step",
                "gap_left": 180.0,
                "gap_right": 180.0,
                "interface_G_N": 1.0,
            },
        }
        resp = client.post("/api/runs", json={"name": "bad interface", "setup": setup})
        assert resp.status_code == 400
        assert any("interface_G_N requires distinct" in e for e in resp.json()["errors"])
        assert client.get("/api/runs").json() == []


class TestReviewFixes:
    """Server-visible behavior pinned by the 2026-07-03 review fixes."""

    def test_permanent_terminal_manifest_failure_does_not_strand_run(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qpsim.webui.runner import JobState

        workspace = client.app.state.workspace
        runner = client.app.state.runner
        run_id = workspace.new_run_id()
        disk_manifest: dict[str, Any] = {
            "id": run_id,
            "name": "write failure",
            "mode": "steady_state_0d",
            "status": "running",
            "created": "2026-07-15T00:00:00",
            "setup": {},
            "summary": {},
            "notes": [],
            "error": None,
            "elapsed_s": None,
        }
        workspace.write_manifest(run_id, disk_manifest)

        terminal_manifest = dict(disk_manifest)
        terminal_manifest["status"] = "failed"
        terminal_manifest["error"] = "synthetic failure"
        job = JobState(run_id=run_id, status="failed")
        runner._jobs[run_id] = job

        write_attempts = 0

        def permanently_unwritable(*_args: object, **_kwargs: object) -> None:
            nonlocal write_attempts
            write_attempts += 1
            raise PermissionError("synthetic permanent manifest failure")

        monkeypatch.setattr(workspace, "write_manifest", permanently_unwritable)
        runner._write_manifest_or_stash(job, terminal_manifest)
        job.worker_finished.set()
        assert job.pending_manifest == terminal_manifest

        # Polling retries and fails again, reproducing the old permanent-live
        # state, while still serving the terminal in-memory snapshot.
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "failed"
        assert write_attempts == 2

        captured_retry = dict(terminal_manifest)
        resp = client.delete(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        assert runner.live_state(run_id) is None
        assert not workspace.run_dir(run_id).exists()

        # An overlay request may have copied the retry just before deletion.
        # Closed writes make that stale copy a no-op rather than recreating a
        # one-file zombie directory.
        runner._write_manifest_or_stash(job, captured_retry)
        assert write_attempts == 2
        assert not workspace.run_dir(run_id).exists()

    def test_terminal_run_cannot_delete_before_worker_finishes(
        self, client: TestClient
    ) -> None:
        from qpsim.webui.runner import JobState

        workspace = client.app.state.workspace
        runner = client.app.state.runner
        run_id = workspace.new_run_id()
        job = JobState(run_id=run_id, status="failed")
        runner._jobs[run_id] = job

        resp = client.delete(f"/api/runs/{run_id}")

        assert resp.status_code == 409
        assert "still finalizing" in resp.json()["detail"]
        runner._jobs.pop(run_id)

    def test_autofill_params_do_not_carry_dynes_gamma(self, client: TestClient) -> None:
        al = next(m for m in client.get("/api/materials").json() if m["name"] == "Al")
        # Γ is not a database property; shipping it would zero a
        # user-entered value on material pick.
        assert "dynes_gamma" not in al["params"]

    def test_creation_warnings_persist_as_run_notes(self, client: TestClient) -> None:
        # T > 0.5 Tc is supported but carries a resolution/BCS-regime warning.
        setup = dict(TINY_SETUP, T_bath=0.7)
        resp = client.post("/api/runs", json={"name": "warned", "setup": setup})
        assert resp.status_code == 200
        assert any("T_c/2" in warning for warning in resp.json()["warnings"])
        run_id = resp.json()["id"]
        # The warning is already on the manifest before completion and
        # survives it — the browser switches views on submit, so a
        # transient banner alone would be lost.
        body = _wait_done(client, run_id)
        assert any("T_c/2" in note for note in body["notes"])

    def test_missing_npz_degrades_detail_and_404s_artifacts(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json={"name": "tiny", "setup": TINY_SETUP})
        run_id = resp.json()["id"]
        body = _wait_done(client, run_id)
        assert body["status"] == "done"

        workspace = client.app.state.workspace
        (workspace.run_dir(run_id) / "result.npz").unlink()

        body = client.get(f"/api/runs/{run_id}").json()
        assert body["plots"] == []
        assert "artifacts_error" in body
        assert client.get(f"/api/runs/{run_id}/plots/occupation.png").status_code == 404
        assert client.get(f"/api/runs/{run_id}/csv/occupation.csv").status_code == 404

    def test_plot_and_csv_responses_are_cacheable(self, client: TestClient) -> None:
        resp = client.post("/api/runs", json={"name": "tiny", "setup": TINY_SETUP})
        run_id = resp.json()["id"]
        assert _wait_done(client, run_id)["status"] == "done"
        png = client.get(f"/api/runs/{run_id}/plots/occupation.png")
        assert "max-age" in png.headers.get("cache-control", "")
        csv_resp = client.get(f"/api/runs/{run_id}/csv/occupation.csv")
        assert "max-age" in csv_resp.headers.get("cache-control", "")


class TestPathTraversal:
    def test_setup_slug_cannot_escape_workspace(self, client: TestClient) -> None:
        # Plant a setup-shaped file one level above setups/. On Windows the
        # encoded-backslash form (%5C) survives Starlette's routing, so this
        # must be rejected at the store, not merely by the '/' router guard.
        workspace = client.app.state.workspace
        secret = workspace.setups_dir.parent / "SECRET.json"
        workspace.setups_dir.mkdir(parents=True, exist_ok=True)
        secret.write_text('{"name":"x","setup":{"mode":"steady_state_0d"}}', encoding="utf-8")
        for evil in ("..%5CSECRET", "..%2FSECRET", "foo%5C..%5C..%5CSECRET"):
            assert client.get(f"/api/setups/{evil}").status_code == 404

    def test_run_id_cannot_escape_workspace(self, client: TestClient) -> None:
        assert client.get("/api/runs/..%5C..%5Csetups").status_code == 404
        assert client.delete("/api/runs/..%5C..%5Csetups").status_code == 404
