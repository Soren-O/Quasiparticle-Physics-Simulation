"""The catalogue as a batch: every case run, every verdict in one table.

The catalogue (``static/catalogue.json``) is 38 cases. Ten name an analytic
benchmark and five state a scalar expectation; exercising them used to be 38
clicks, and the ten recorded pass numbers came from a session-local script
that no CI ever ran. This module is that script, promoted: it knows how to
turn a case into the setup the interface would run, how to score a finished
run against what the case claims, and how to lay the results out. Three
callers share it and therefore cannot disagree about a verdict:

* ``python -m qpsim.webui.verdicts`` runs the cases in-process and prints the
  table (exit 1 on any failure) -- what the scratchpad script did;
* ``tests/webui/test_catalogue_verdicts.py`` runs every benchmark case in CI
  and pins its error to the recorded value;
* the server's ``/api/catalogue/run-all`` and ``/api/catalogue/report`` queue
  the cases through the ordinary runner and aggregate the stored runs.

A case's verdict comes from one of two sources. A ``benchmark`` is a whole
curve checked pointwise (:mod:`qpsim.webui.benchmarks`); its score is written
into the run's summary by the runner. An ``expect`` is a scalar statement --
this observable equals that reference, or exceeds it -- evaluated here from
the summary. The browser's single-case view evaluates the same statement in
JavaScript; a test holds the two evaluators to the same verdicts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qpsim.webui import benchmarks
from qpsim.webui.execute import execute_setup
from qpsim.webui.schemas import MODE_CLASSES, AnySetup, SetupEnvelope, canonical_mode

CATALOGUE_PATH = Path(__file__).parent / "static" / "catalogue.json"


@dataclass(frozen=True)
class CaseRef:
    """One catalogue case and where it sits."""

    category: str
    item: str
    id: str
    title: str
    mode: str
    overrides: dict[str, Any] = field(default_factory=dict)
    benchmark: str | None = None
    expect: dict[str, Any] | None = None

    @property
    def source(self) -> str:
        """Where a verdict for this case comes from, if anywhere."""
        if self.benchmark:
            return "benchmark"
        if self.expect:
            return "expectation"
        return "none"

    def tag(self) -> dict[str, Any]:
        """What a run records so the report can find it again."""
        return {
            "id": self.id, "title": self.title,
            "category": self.category, "item": self.item,
            "source": self.source, "benchmark": self.benchmark,
        }


def load_catalogue(path: Path = CATALOGUE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def catalogue_cases(catalogue: dict[str, Any] | None = None) -> list[CaseRef]:
    """Every case, in catalogue order."""
    cat = catalogue if catalogue is not None else load_catalogue()
    cases: list[CaseRef] = []
    for category in cat.get("categories", []):
        for item in category.get("items", []):
            for case in item.get("cases", []) or []:
                cases.append(CaseRef(
                    category=str(category.get("id", "")),
                    item=str(item.get("id", "")),
                    id=str(case["id"]),
                    title=str(case.get("title", case["id"])),
                    mode=canonical_mode(str(case["mode"])),
                    overrides=dict(case.get("overrides") or {}),
                    benchmark=case.get("benchmark") or None,
                    expect=copy.deepcopy(case.get("expect")) or None,
                ))
    return cases


def build_case_setup(case: CaseRef) -> AnySetup:
    """Defaults for the case's mode with its overrides applied.

    The same resolution the browser performs (``buildCaseSetup``): a case
    states only what it changes. An override naming a path the model does
    not have is an error rather than a silently dropped key, because a case
    that claims to test something it does not set is worse than no case.
    """
    data = MODE_CLASSES[case.mode]().model_dump()
    for path, value in case.overrides.items():
        node: Any = data
        parts = path.split(".")
        for key in parts[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"{case.id}: override path {path!r} does not resolve.")
            node = node[key]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ValueError(f"{case.id}: override path {path!r} does not resolve.")
        node[parts[-1]] = value
    return MODE_CLASSES[case.mode](**data)


def envelope_for(case: CaseRef) -> SetupEnvelope:
    return SetupEnvelope(name=case.title, setup=build_case_setup(case), benchmark=case.benchmark)


# -- scoring ------------------------------------------------------------


def _finite(value: Any) -> float | None:
    """A finite number, or None -- the JavaScript ``Number.isFinite`` gate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def evaluate_expectation(expect: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Score a scalar expectation against a run's summary.

    Mirrors ``evaluateExpectation`` in ``static/app.js`` statement for
    statement; ``tests/webui/test_wave5_batch.py`` runs both on the same
    inputs and requires the same verdicts, so neither can drift alone.
    """
    got = _finite(summary.get(expect["observable"]))
    want = _finite(summary.get(expect["reference"]))
    if got is None or want is None:
        missing = expect["observable"] if got is None else expect["reference"]
        return {"verdict": "unknown", "detail": f"the run reported no {missing}"}
    if expect.get("comparison") == "greater":
        factor = float(expect.get("factor", 1.0))
        return {
            "verdict": "pass" if got > want * factor else "fail",
            "got": got, "want": want,
            "detail": f"{got:.6g} vs {want:.6g} × {factor:g} = {want * factor:.6g}",
        }
    scale = abs(want) if abs(want) > 0.0 else 1.0
    rel = abs(got - want) / scale
    tol = float(expect.get("rel_tol", 1e-6))
    return {
        "verdict": "pass" if rel <= tol else "fail",
        "got": got, "want": want, "rel": rel, "rel_tol": tol,
        "detail": f"relative difference {rel:.2e} against a tolerance of {tol:.0e}",
    }


def score_case(case: CaseRef, manifest: dict[str, Any] | None) -> dict[str, Any]:
    """One row of the report: what the case claims, and whether its run agrees.

    ``manifest`` is the run to score (its ``status``, ``summary``, ``notes``)
    or None when the case has not been run. Verdicts: ``pass``/``fail`` from
    the source; ``unknown`` when the run finished but produced nothing to
    score (a benchmark that did not apply, a missing observable); ``failed``
    when the run itself did not finish; ``not run``; and ``none`` for a case
    that makes no checkable claim -- which is reported as such rather than
    hidden, because "23 of 38 cases produce no verdict" is a fact about the
    catalogue worth seeing.
    """
    row: dict[str, Any] = {
        "case": case.id, "title": case.title, "category": case.category,
        "item": case.item, "mode": case.mode, "source": case.source,
        "benchmark": case.benchmark,
        "run_id": None, "status": "not run", "verdict": "not run",
        "tier": None, "error": None, "rel_tol": None, "detail": "",
    }
    if manifest is None:
        if case.source == "none":
            row["verdict"] = "none"
            row["detail"] = "this case states no closed form or expectation"
        return row
    row["run_id"] = manifest.get("id")
    status = str(manifest.get("status", "?"))
    row["status"] = status
    if status in ("queued", "running"):
        row["verdict"] = status
        row["detail"] = str(manifest.get("progress_message") or "")
        return row
    if status != "done":
        row["verdict"] = "failed"
        row["detail"] = str(manifest.get("error") or status)
        return row
    summary = manifest.get("summary") or {}
    if case.source == "benchmark":
        score = summary.get("benchmark")
        if not isinstance(score, dict):
            row["verdict"] = "unknown"
            row["detail"] = "; ".join(
                n for n in (manifest.get("notes") or []) if "benchmark" in str(n).lower()
            ) or "the run reported no benchmark score"
            return row
        row.update({
            "verdict": str(score.get("verdict", "unknown")),
            "tier": score.get("tier"),
            "error": score.get("error"),
            "rel_tol": score.get("rel_tol"),
            "detail": (
                f"{score.get('metric', 'pointwise')} error "
                f"{float(score.get('error', float('nan'))):.3e} against "
                f"{float(score.get('rel_tol', float('nan'))):.1e} over "
                f"{score.get('n_points', '?')} points"
            ),
        })
        return row
    if case.source == "expectation" and case.expect is not None:
        result = evaluate_expectation(case.expect, summary)
        row["verdict"] = result["verdict"]
        row["detail"] = result["detail"]
        row["error"] = result.get("rel")
        row["rel_tol"] = result.get("rel_tol")
        return row
    row["verdict"] = "none"
    row["detail"] = "this case states no closed form or expectation"
    return row


def report(cases: Iterable[CaseRef], manifests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Every case against its most recent run, newest manifest first."""
    rows = []
    for case in cases:
        latest = next(
            (m for m in manifests if (m.get("case") or {}).get("id") == case.id), None,
        )
        rows.append(score_case(case, latest))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    return {
        "rows": rows,
        "counts": counts,
        "active": any(r["verdict"] in ("queued", "running") for r in rows),
        "checkable": sum(1 for r in rows if r["source"] != "none"),
    }


# -- in-process batch (CLI and CI) ---------------------------------------


def run_case(
    case: CaseRef,
    execute: Callable[..., Any] = execute_setup,
) -> dict[str, Any]:
    """Run one case in-process and return a finished manifest for it.

    The same steps the runner takes -- execute, then attach the benchmark --
    without the workspace, so the CLI and CI score exactly what a queued run
    would have scored.
    """
    started = time.perf_counter()
    manifest: dict[str, Any] = {"id": None, "case": case.tag(), "summary": {}, "notes": []}
    try:
        setup = build_case_setup(case)
        payload = execute(setup, lambda *a, **k: None, lambda: False)
        if case.benchmark:
            payload.notes.extend(
                benchmarks.attach(case.benchmark, setup, payload.arrays, payload.summary)
            )
        manifest.update(status="done", summary=payload.summary, notes=list(payload.notes))
    except Exception as exc:  # a failed case is a row, not the end of the table
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    manifest["elapsed_s"] = time.perf_counter() - started
    return manifest


def run_all(
    cases: Iterable[CaseRef],
    execute: Callable[..., Any] = execute_setup,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        manifest = run_case(case, execute)
        row = score_case(case, manifest)
        row["elapsed_s"] = manifest.get("elapsed_s")
        rows.append(row)
    return rows


def format_table(rows: Sequence[dict[str, Any]]) -> str:
    head = (
        f"{'case':<30s} {'source':<12s} {'tier':<5s} {'verdict':<9s} "
        f"{'error':>12s} {'tol':>10s} {'time':>7s}"
    )
    lines = [head, "-" * len(head)]
    for row in rows:
        error = row.get("error")
        tol = row.get("rel_tol")
        elapsed = row.get("elapsed_s")
        error_s = f"{error:12.4e}" if isinstance(error, (int, float)) else f"{'-':>12s}"
        tol_s = f"{tol:10.1e}" if isinstance(tol, (int, float)) else f"{'-':>10s}"
        time_s = f"{elapsed:6.1f}s" if isinstance(elapsed, (int, float)) else f"{'-':>7s}"
        tier = row.get("tier") or "-"
        verdict = str(row["verdict"]).upper()
        lines.append(
            f"{row['case']:<30s} {row['source']:<12s} {tier:<5s} {verdict:<9s} "
            f"{error_s} {tol_s} {time_s}"
        )
        if row["verdict"] != "pass" and row.get("detail"):
            lines.append(f"    {row['detail'][:110]}")
    lines.append("-" * len(head))
    passed = sum(1 for r in rows if r["verdict"] == "pass")
    checkable = sum(1 for r in rows if r["source"] != "none")
    lines.append(
        f"{passed}/{checkable} checkable cases pass; {len(rows) - checkable} state no claim"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qpsim.webui.verdicts",
        description="Run the catalogue's cases in-process and print their verdicts.",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="CASE_ID",
        help="run only this case (repeatable)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="include cases that state no closed form or expectation",
    )
    args = parser.parse_args(argv)
    cases = catalogue_cases()
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in wanted]
    elif not args.all:
        cases = [c for c in cases if c.source != "none"]
    rows = run_all(cases)
    print(format_table(rows))
    return 0 if all(r["verdict"] in ("pass", "none") for r in rows) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
