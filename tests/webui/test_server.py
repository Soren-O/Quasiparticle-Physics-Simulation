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
    "mode": "kinetics",
    "strategy": "steady_state",
    "geometry": {"rows": 1, "cols": 1},
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


class TestEveryFormControlBindsToAField:
    """A control the editor shows must set something the engine reads.

    The wizard's field paths are written by hand in `app.js` and resolved
    against the setup dict at run time, so a typo or a renamed field does not
    fail — the control renders, the user sets it, and the value lands nowhere.
    That is the defect this repo keeps finding, in the one place no Python test
    was looking.

    It also catches the reverse of the mode collapse: `strategy`, `solver.*`,
    `probe.*` and `snapshot_interval` all existed on the merged model for a
    while with no way to reach them from the browser.
    """

    @staticmethod
    def _form_paths(mode: str) -> set[str]:
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "qpsim" / "webui" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        forms = source[source.index("const FORMS = {"):]
        start = forms.index(f"  {mode}: [")
        later = [
            forms.index(marker)
            for marker in (f"  {m}: [" for m in ("kinetics", "m25_junction", "materials"))
            if marker in forms and forms.index(marker) > start
        ]
        block = forms[start: min(later) if later else len(forms)]
        paths = set(re.findall(r'F\("([a-zA-Z0-9_.]+)"', block))
        # Shared groups are referenced by name (PROBE_FIELDS, GRID_FIELDS, ...),
        # so a parser that only reads inline F() calls would silently skip
        # whole sections and report a form as fully bound while never having
        # looked at half of it.
        for name in set(re.findall(r"\b([A-Z][A-Z_]*_FIELDS)\b", block)):
            definition = source[source.index(f"const {name}"):]
            definition = definition[: definition.index("\n};") + 3] if "\n};" in definition[:4000] else definition[:4000]
            paths |= set(re.findall(r'F\("([a-zA-Z0-9_.]+)"', definition))
        return paths

    @pytest.mark.parametrize("mode", ["kinetics", "m25_junction"])
    def test_every_control_resolves_in_the_setup_model(self, mode: str) -> None:
        from qpsim.webui.schemas import MODE_CLASSES

        defaults = MODE_CLASSES[mode]().model_dump()
        paths = self._form_paths(mode)
        assert paths, f"no controls found for {mode} — the parser missed the block"

        unmapped = []
        for path in sorted(paths):
            node = defaults
            for part in path.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    unmapped.append(path)
                    break
        assert not unmapped, (
            f"{mode}: these controls set nothing the engine reads: {unmapped}"
        )

    def test_the_merged_mode_exposes_its_strategy(self) -> None:
        """Reachability, not just binding.

        `strategy` decides which solver runs, so a model that has it and a form
        that does not means half the mode is unreachable from the browser --
        which was true for several commits.
        """
        paths = self._form_paths("kinetics")
        assert "strategy" in paths
        assert "snapshot_interval" in paths
        assert any(p.startswith("solver.") for p in paths)
        assert any(p.startswith("probe.") for p in paths)


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
        # Two modes, not five: the 0-D and 1-D ones were geometries of this
        # one all along, and now say so.
        assert set(body["modes"]) == {"kinetics", "m25_junction"}

    def test_materials_carry_autofill_params(self, client: TestClient) -> None:
        mats = client.get("/api/materials").json()
        names = {m["name"] for m in mats}
        assert {"Al", "Nb", "TiN"} <= names
        al = next(m for m in mats if m["name"] == "Al")
        assert al["params"]["Delta_0"] == al["Delta_0"]

    def test_defaults_endpoint(self, client: TestClient) -> None:
        body = client.get("/api/defaults/kinetics").json()
        assert body["mode"] == "kinetics"
        assert client.get("/api/defaults/nope").status_code == 404

    def test_defaults_endpoint_answers_a_retired_mode_name(
        self, client: TestClient,
    ) -> None:
        """A bookmarked URL or an older client can still name one.

        404ing here while /api/validate accepts the same name inside a setup
        would be an inconsistency the caller cannot act on.
        """
        body = client.get("/api/defaults/spatial_1d").json()
        assert body["mode"] == "kinetics"

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
        # Posted under the RETIRED name on purpose: the guard has to survive
        # the upgrade, not just exist on the mode that replaced it.
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


class TestFigureFamilies:
    """A run replayed, rather than the single frame it happened to end on.

    A family is one registry entry rendering one image per index. The counts
    come from the stored arrays, so the interface cannot offer a frame the
    run does not have.
    """

    @staticmethod
    def _recorded_run(client: TestClient) -> dict[str, Any]:
        setup = client.get("/api/defaults/kinetics").json()
        setup["T_bath"] = 0.2
        setup["grid"]["num_bins"] = 24
        setup["geometry"]["rows"] = 4
        setup["geometry"]["cols"] = 4
        setup["dt"] = 2.0
        setup["max_time"] = 24.0
        setup["snapshot_interval"] = 8.0
        setup["stop_tol"] = 0.0
        setup["phonons"]["mode"] = "dynamic_escape"
        run_id = client.post(
            "/api/runs", json={"name": "frames", "setup": setup}
        ).json()["id"]
        return _wait_done(client, run_id)

    def test_the_families_are_offered_with_their_index_counts(
        self, client: TestClient
    ) -> None:
        manifest = self._recorded_run(client)
        families = manifest["plot_params"]
        assert set(families) == {
            "field_over_time", "gap_over_time", "energy_resolved_map",
            "phonon_field_over_time", "phonon_occupation_map",
        }
        assert families["field_over_time"] == {"frame": 4}
        assert families["gap_over_time"] == {"frame": 4}
        assert families["phonon_occupation_map"] == {"frame": 4}
        assert families["energy_resolved_map"]["frame"] == 4
        assert families["energy_resolved_map"]["energy"] == 24

    def test_each_index_renders_its_own_image(self, client: TestClient) -> None:
        manifest = self._recorded_run(client)
        run_id = manifest["id"]
        seen = set()
        for frame in range(manifest["plot_params"]["field_over_time"]["frame"]):
            resp = client.get(
                f"/api/runs/{run_id}/plots/field_over_time.png?frame={frame}"
            )
            assert resp.status_code == 200
            assert resp.content[:4] == b"\x89PNG"
            seen.add(resp.content)
        # Identical bytes across frames would mean the index is ignored and
        # the "animation" is one still image repeated.
        assert len(seen) == manifest["plot_params"]["field_over_time"]["frame"]

    def test_an_index_the_run_does_not_have_is_a_404(
        self, client: TestClient
    ) -> None:
        manifest = self._recorded_run(client)
        resp = client.get(
            f"/api/runs/{manifest['id']}/plots/field_over_time.png?frame=9999"
        )
        assert resp.status_code == 404

    def test_a_run_without_frames_offers_no_families(
        self, client: TestClient
    ) -> None:
        """snapshot_interval is opt-in, so most runs have nothing to replay."""
        setup = client.get("/api/defaults/kinetics").json()
        setup["grid"]["num_bins"] = 24
        setup["geometry"]["rows"] = 3
        setup["geometry"]["cols"] = 3
        setup["dt"] = 2.0
        setup["max_time"] = 8.0
        run_id = client.post(
            "/api/runs", json={"name": "endpoint only", "setup": setup}
        ).json()["id"]
        manifest = _wait_done(client, run_id)
        assert manifest["plot_params"] == {}
        assert "field_over_time" not in manifest["plots"]


class TestIterationLoop:
    """A finished run has to be a starting point, not a dead end."""

    def test_a_stored_setup_is_re_postable_unchanged(
        self, client: TestClient
    ) -> None:
        """What "Open in editor" relies on: the manifest's setup is valid input.

        Without this the only way to iterate on a run is to read its setup
        out of a JSON block and retype it.
        """
        setup = client.get("/api/defaults/kinetics").json()
        setup["grid"]["num_bins"] = 24
        setup["geometry"]["rows"] = 3
        setup["geometry"]["cols"] = 3
        setup["dt"] = 2.0
        setup["max_time"] = 8.0
        first = _wait_done(
            client,
            client.post(
                "/api/runs", json={"name": "first", "setup": setup}
            ).json()["id"],
        )
        again = client.post(
            "/api/runs", json={"name": "reopened", "setup": first["setup"]}
        )
        assert again.status_code == 200
        second = _wait_done(client, again.json()["id"])
        assert second["status"] == "done"
        assert second["setup"] == first["setup"]


class TestFormulaRendering:
    """A closed form is typeset, not shown as LaTeX source.

    The formulas are 470-1600 characters of align environments; dropped into
    a banner as text they are unreadable, which is what prompted this.
    """

    def test_every_benchmark_headline_typesets(self, client: TestClient) -> None:
        catalogue = client.get("/api/benchmarks").json()
        assert catalogue, "no benchmarks registered"
        for name in catalogue:
            resp = client.get(f"/api/benchmarks/{name}/formula.png")
            assert resp.status_code == 200, f"{name}: {resp.text[:200]}"
            assert resp.content[:4] == b"\x89PNG"

    def test_the_headline_is_short_enough_to_be_one(self, client: TestClient) -> None:
        """The bug was a derivation in a field rendered as a banner."""
        for name, entry in client.get("/api/benchmarks").json().items():
            headline = entry["headline_latex"]
            assert headline, f"{name} has no headline"
            assert len(headline) < 200, (
                f"{name}: headline is {len(headline)} chars -- that is a "
                "derivation, and belongs in the full statement"
            )

    def test_an_unknown_benchmark_is_a_404_not_a_broken_image(
        self, client: TestClient
    ) -> None:
        assert client.get("/api/benchmarks/nope/formula.png").status_code == 404


class TestStaticAssetsRevalidate:
    """The UI must not be served from cache after it changes.

    Starlette's StaticFiles sends etag and last-modified but no Cache-Control,
    so a browser applies heuristic freshness and keeps serving a stale app.js
    across ordinary reloads. For an app whose interface is edited in place
    that is indistinguishable from the change not working -- which is exactly
    how it presented.
    """

    def test_the_frontend_script_must_revalidate(self, client: TestClient) -> None:
        resp = client.get("/static/app.js")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"
        # no-cache means "revalidate", not "do not store": the validator has
        # to still be there or every reload is a full re-download.
        assert resp.headers.get("etag")

    def test_the_shell_must_revalidate(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"

    def test_an_unchanged_asset_still_answers_304(self, client: TestClient) -> None:
        """Revalidating must stay cheap, or this trades one problem for another."""
        first = client.get("/static/app.js")
        again = client.get(
            "/static/app.js", headers={"If-None-Match": first.headers["etag"]}
        )
        assert again.status_code == 304


class TestCaseSummariesAreNotTheirOwnReason:
    """The card printed the same paragraph twice.

    I generated the benchmark cases with summary = benchmark.reason, so the
    interface showed it as the case summary and again as the expectation's
    reason. The summary says what the case IS; the reason says why the closed
    form holds.
    """

    def test_no_case_summary_repeats_its_benchmark_reason(
        self, client: TestClient
    ) -> None:
        import json
        from pathlib import Path
        catalogue = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "qpsim" / "webui" / "static" / "catalogue.json"
            ).read_text(encoding="utf-8")
        )
        entries = client.get("/api/benchmarks").json()
        for category in catalogue["categories"]:
            for item in category["items"]:
                for case in item.get("cases", []):
                    name = case.get("benchmark")
                    if not name:
                        continue
                    assert case["summary"] != entries[name]["reason"], (
                        f"{case['id']} shows the same text twice"
                    )
