"""Wave 5: the catalogue as one action, and a check a person can ask for.

The batch route queues cases through the ordinary runner and the report
scores their stored runs; the CLI scores in-process. Both read one module,
so the tests here compare the two paths on the same cases and require the
same verdicts. The scalar expectations have a JavaScript evaluator as well
(the single-case view); it is run under node on the same inputs as the
Python one and the verdicts must agree.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import time
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi", reason="server tests need the qpsim[ui] extra")

from fastapi.testclient import TestClient  # noqa: E402
from qpsim.webui import verdicts  # noqa: E402
from qpsim.webui.schemas import KineticsSetup  # noqa: E402
from qpsim.webui.server import create_app  # noqa: E402

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"
_HARNESS = pathlib.Path(__file__).with_name("form_harness.js")
_NODE = shutil.which("node")

# Cheap cases spanning the three verdict sources: two benchmarks measured
# well under a second, one scalar expectation, one case with no claim.
QUICK = ["src-benchmark", "pesc-benchmark", "scat-only"]


@pytest.fixture
def client(tmp_path: pathlib.Path) -> TestClient:
    with TestClient(create_app(tmp_path)) as c:
        yield c


def _wait_settled(client: TestClient, timeout_s: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get("/api/catalogue/report").json()
        if not body["active"]:
            return body
        time.sleep(0.3)
    pytest.fail("the batch did not settle in time")


def _by_case(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["case"]: row for row in report["rows"]}


class TestTheCatalogueModule:
    def test_every_case_is_listed_with_its_source(self) -> None:
        cases = verdicts.catalogue_cases()
        assert len(cases) == 38
        sources = {c.source for c in cases}
        assert sources == {"benchmark", "expectation", "none"}
        assert sum(1 for c in cases if c.benchmark) == 10
        assert sum(1 for c in cases if c.expect) == 5

    def test_a_case_resolves_to_the_setup_the_browser_would_run(self) -> None:
        case = next(c for c in verdicts.catalogue_cases() if c.id == "src-benchmark")
        setup = verdicts.build_case_setup(case)
        assert isinstance(setup, KineticsSetup)
        # The overrides landed: a benchmark case is never the bare defaults.
        assert setup.model_dump() != KineticsSetup().model_dump()
        assert verdicts.envelope_for(case).benchmark == "injection"

    def test_an_override_that_resolves_nowhere_is_an_error(self) -> None:
        case = verdicts.CaseRef(
            category="c", item="i", id="bad", title="bad", mode="kinetics",
            overrides={"geometry.no_such_field": 1},
        )
        with pytest.raises(ValueError, match="does not resolve"):
            verdicts.build_case_setup(case)

    def test_an_unrun_case_reports_its_claim_status(self) -> None:
        cases = {c.id: c for c in verdicts.catalogue_cases()}
        assert verdicts.score_case(cases["src-benchmark"], None)["verdict"] == "not run"
        none = next(c for c in cases.values() if c.source == "none")
        assert verdicts.score_case(none, None)["verdict"] == "none"

    def test_a_failed_run_is_a_row_not_an_exception(self) -> None:
        case = next(c for c in verdicts.catalogue_cases() if c.id == "src-benchmark")

        def explode(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("boom")

        row = verdicts.score_case(case, verdicts.run_case(case, execute=explode))
        assert row["verdict"] == "failed"
        assert "boom" in row["detail"]


class TestTheBatchOverHttp:
    def test_run_all_queues_tagged_runs_and_the_report_scores_them(
        self, client: TestClient
    ) -> None:
        resp = client.post("/api/catalogue/run-all", json={"only": [*QUICK, "no-such-case"]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert sorted(q["case"] for q in body["queued"]) == sorted(QUICK)
        assert body["skipped"] == []
        assert body["unknown"] == ["no-such-case"]
        # Each run carries its case, which is how the report finds it.
        for q in body["queued"]:
            manifest = client.get(f"/api/runs/{q['run_id']}").json()
            assert manifest["case"]["id"] == q["case"]

        report = _wait_settled(client)
        rows = _by_case(report)
        assert rows["src-benchmark"]["verdict"] == "pass"
        assert rows["src-benchmark"]["source"] == "benchmark"
        assert rows["src-benchmark"]["tier"] == "T1"
        assert rows["pesc-benchmark"]["verdict"] == "pass"
        assert rows["scat-only"]["source"] == "expectation"
        assert rows["scat-only"]["verdict"] == "pass", rows["scat-only"]["detail"]
        # The cases not queued are still rows -- the table is the catalogue.
        assert len(report["rows"]) == 38
        assert rows["diff-benchmark"]["verdict"] == "not run"
        assert report["counts"]["pass"] == 3
        assert report["checkable"] == 15

    def test_the_report_scores_what_the_cli_scores(self, client: TestClient) -> None:
        """Two paths, one module: the queued run and the in-process run of
        the same case must produce the same error to the digit."""
        client.post("/api/catalogue/run-all", json={"only": ["src-benchmark"]})
        queued = _by_case(_wait_settled(client))["src-benchmark"]
        case = next(c for c in verdicts.catalogue_cases() if c.id == "src-benchmark")
        direct = verdicts.score_case(case, verdicts.run_case(case))
        assert queued["error"] == direct["error"]
        assert queued["verdict"] == direct["verdict"] == "pass"

    def test_an_empty_body_means_every_case(self, client: TestClient) -> None:
        """Queued only -- not waited for: 38 runs is minutes. The point is
        that the whole catalogue is one request, with nothing skipped."""
        body = client.post("/api/catalogue/run-all", json={}).json()
        assert len(body["queued"]) == 38, body["skipped"]
        assert body["skipped"] == []
        for q in body["queued"]:
            client.post(f"/api/runs/{q['run_id']}/cancel")


class TestACheckAPersonCanAskFor:
    def test_a_user_setup_posted_with_a_benchmark_is_scored(self, client: TestClient) -> None:
        """The envelope's benchmark works for a setup nobody catalogued."""
        case = next(c for c in verdicts.catalogue_cases() if c.id == "src-benchmark")
        setup = verdicts.build_case_setup(case).model_dump(mode="json")
        run_id = client.post(
            "/api/runs", json={"name": "mine", "setup": setup, "benchmark": "injection"},
        ).json()["id"]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            manifest = client.get(f"/api/runs/{run_id}").json()
            if manifest["status"] in ("done", "failed", "cancelled"):
                break
            time.sleep(0.2)
        assert manifest["status"] == "done", manifest.get("error")
        assert manifest["summary"]["benchmark"]["verdict"] == "pass"
        assert manifest["benchmark"] == "injection"

    def test_a_saved_setup_returns_its_check(self, client: TestClient) -> None:
        setup = KineticsSetup().model_dump(mode="json")
        slug = client.post(
            "/api/setups", json={"name": "checked", "setup": setup, "benchmark": "injection"},
        ).json()["slug"]
        assert client.get(f"/api/setups/{slug}").json()["benchmark"] == "injection"

    def test_the_page_sends_the_selector_in_the_envelope(self) -> None:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        js = (_STATIC / "app.js").read_text(encoding="utf-8")
        assert 'id="run-benchmark"' in html
        body = re.search(r"function envelope\(\) \{(.*?)\n\}", js, re.S)
        assert body and "benchmark:" in body.group(1)
        assert "refreshBenchmarkOptions(body.benchmark" in js, "loading a setup must restore its check"
        assert "/api/catalogue/run-all" in js and "/api/catalogue/report" in js


@pytest.mark.skipif(_NODE is None, reason="the JavaScript evaluator needs node")
class TestTheTwoExpectationEvaluatorsAgree:
    """The browser scores a single case in JavaScript; the report scores it
    in Python. Same statements, same summaries, same verdicts -- or one of
    them is wrong and nothing would otherwise say which."""

    def test_on_every_catalogue_expectation(self) -> None:
        expects = [c.expect for c in verdicts.catalogue_cases() if c.expect]
        assert len(expects) == 5
        summaries = [
            {"x_qp_mean": 1.0e-6, "x_qp_thermal": 1.0e-6, "x_qp_initial": 1.0e-6,
             "gap_ueV": 180.0, "delta_eq_ueV": 180.0},
            {"x_qp_mean": 1.5e-5, "x_qp_thermal": 1.0e-6, "x_qp_initial": 1.0e-6,
             "gap_ueV": 180.0, "delta_eq_ueV": 179.0},
            {"x_qp_mean": 1.0e-6 * (1 + 5e-10), "x_qp_thermal": 1.0e-6,
             "x_qp_initial": 1.0e-6 * (1 + 5e-5), "gap_ueV": 180.0,
             "delta_eq_ueV": 180.0 * (1 + 5e-6)},
            {"x_qp_mean": 9.0e-6, "x_qp_thermal": 1.0e-6},
            {},
        ]
        script = '''
RESULT = EXPECTS.flatMap((e) => SUMMARIES.map((s) => evaluateExpectation(e, s).verdict));
'''.replace("EXPECTS", json.dumps(expects)).replace("SUMMARIES", json.dumps(summaries))
        proc = subprocess.run(
            [_NODE, str(_HARNESS), str(_STATIC / "app.js")],
            input=script, capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        from_js = json.loads(proc.stdout)
        from_py = [
            verdicts.evaluate_expectation(e, s)["verdict"] for e in expects for s in summaries
        ]
        assert from_js == from_py
        # And the inputs exercised every branch, or the agreement is vacuous.
        assert {"pass", "fail", "unknown"} <= set(from_py)


class TestTheCommandLine:
    def test_main_prints_a_table_and_reports_the_outcome(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = verdicts.main(["--only", "src-benchmark", "--only", "pesc-benchmark"])
        out = capsys.readouterr().out
        assert code == 0
        assert "src-benchmark" in out and "PASS" in out
        assert "2/2 checkable cases pass" in out

    def test_an_unknown_case_id_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            verdicts.main(["--only", "no-such-case"])
