"""FastAPI application: REST API + static single-page frontend.

``create_app(workspace_root)`` wires a :class:`~qpsim.webui.store.Workspace`
and a :class:`~qpsim.webui.runner.JobRunner` into a FastAPI app. All
engine work happens through the runner's worker thread; request
handlers only validate, persist, and render.

The browser UI is a no-build vanilla HTML/JS page served from
``qpsim/webui/static/``.
"""

from __future__ import annotations

import dataclasses
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from qpsim import __version__
from qpsim.materials import list_materials, load_material
from qpsim.webui import benchmarks
from qpsim.webui.builders import validate_setup
from qpsim.webui.hosts import is_loopback_host
from qpsim.webui.plots import (
    FormulaRenderError,
    available_csvs,
    available_plots,
    plot_parameter_arrays,
    plot_parameters,
    render_csv,
    render_formula,
    render_plot,
)
from qpsim.webui.runner import JobRunner
from qpsim.webui.schemas import (
    MODE_CLASSES,
    MODE_LABELS,
    SetupEnvelope,
    canonical_mode,
)
from qpsim.webui.store import Workspace
from qpsim.webui.terms import term_status_payload

STATIC_DIR = Path(__file__).parent / "static"

# ``testserver`` is TestClient's reserved in-process host. The CLI refuses a
# non-loopback bind, while this Host check is defence in depth against browser
# DNS rebinding. It is not authentication: a direct HTTP client can choose its
# Host header, which is why the bind restriction is the primary boundary.
_TEST_HOSTS = frozenset({"testserver"})

# A finished run's arrays never change and a benchmark's formula is fixed by
# its module, so rendered artifacts can be cached — this also stops the
# browser re-fetching plot PNGs every time the detail view re-renders.
_CACHE_HEADERS = {"Cache-Control": "private, max-age=3600"}


def _hostname_from_header(host_header: str | None) -> str | None:
    """Parse an HTTP Host header, including a bracketed IPv6 literal."""
    if not host_header:
        return None
    try:
        parsed = urlsplit(f"//{host_header}")
        # Accessing ``port`` validates malformed/non-numeric port suffixes.
        _ = parsed.port
    except ValueError:
        return None
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname.lower() if parsed.hostname is not None else None


def _material_payload(name: str) -> dict[str, Any]:
    mat = dataclasses.asdict(load_material(name))
    # Frontend autofill slice, in MaterialParams field names. No
    # dynes_gamma here: Γ is not a database property, and including it
    # would silently zero a value the user already entered when they
    # pick a material.
    mat["params"] = {
        "name": mat["name"],
        "Delta_0": mat["Delta_0"],
        "T_c": mat["T_c"],
        "tau_0": mat["tau_0"],
        "tau_0_pb_ns": mat["tau_0_pb_ns"],
        "D_0": mat["D_0"],
        "rho_F": mat["rho_F"],
    }
    return mat


def create_app(workspace_root: Path | str) -> FastAPI:
    """Build the application against a workspace directory."""
    workspace = Workspace(Path(workspace_root))
    runner = JobRunner(workspace)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runner.shutdown()

    app = FastAPI(
        title="qpsim frontend",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.workspace = workspace
    app.state.runner = runner

    @app.middleware("http")
    async def require_local_host(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        hostname = _hostname_from_header(request.headers.get("host"))
        if hostname not in _TEST_HOSTS and (
            hostname is None or not is_loopback_host(hostname)
        ):
            return PlainTextResponse("Invalid host header", status_code=400)
        return await call_next(request)

    # -- pages --------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # Same reasoning as the static mount: the shell must not be served
        # from cache after it changes, or the app boots against stale markup.
        return FileResponse(
            STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"},
        )

    # -- meta ---------------------------------------------------------

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "qpsim_version": __version__,
            "workspace": str(workspace.root),
            "modes": MODE_LABELS,
        }

    @app.get("/api/materials")
    def materials() -> list[dict[str, Any]]:
        return [_material_payload(name) for name in list_materials()]

    @app.get("/api/benchmarks")
    def benchmarks_list() -> dict[str, dict[str, Any]]:
        """Every analytic benchmark, so a case can name one without restating it.

        The catalogue names a benchmark; the formula, tier, tolerance and
        convergence evidence all come from here. That keeps one source of
        truth: the statement shown before a run and the statement checked
        after it are literally the same string, so they cannot drift apart.
        """
        return {
            name: {
                "title": b.title,
                "tier": b.tier,
                "formula_latex": b.formula_latex,
                "headline_latex": b.headline_latex,
                "reason": b.reason,
                "rel_tol": b.rel_tol,
                "convergence": b.convergence,
                "caveat": b.caveat,
                "activity": b.activity,
                "modes": list(b.modes),
            }
            for name, b in ((n, benchmarks.get(n)) for n in benchmarks.names())
            if b is not None
        }

    @app.get("/api/benchmarks/{name}/formula.png")
    def benchmark_formula(name: str) -> Response:
        """The headline equation, typeset.

        Rendered here rather than in the browser because mathtext is already
        the typesetting engine behind every figure in this app, so a formula
        costs no JavaScript dependency and no build step. A benchmark with no
        headline, or one mathtext cannot parse, is a 404: the interface then
        shows the LaTeX source, which is honest, where a half-typeset
        equation would not be.
        """
        bench = benchmarks.get(name)
        if bench is None or not bench.headline_latex:
            raise HTTPException(404, f"No headline formula for {name!r}.")
        try:
            png = render_formula(bench.headline_latex)
        except FormulaRenderError as exc:
            raise HTTPException(404, f"Formula is not typesettable: {exc}") from exc
        return Response(png, media_type="image/png", headers=_CACHE_HEADERS)

    @app.get("/api/defaults/{mode}")
    def defaults(mode: str) -> dict[str, Any]:
        # A bookmarked URL or an older client can still name a retired mode,
        # and 404ing it here while /api/validate accepts the same name in a
        # setup would be an inconsistency the caller cannot act on.
        setup_cls = MODE_CLASSES.get(canonical_mode(mode))
        if setup_cls is None:
            raise HTTPException(404, f"Unknown mode {mode!r}.")
        return setup_cls().model_dump()

    # -- validation ---------------------------------------------------

    @app.post("/api/validate")
    def validate(envelope: SetupEnvelope) -> dict[str, Any]:
        report = validate_setup(envelope.setup)
        return {"ok": report.ok, "errors": report.errors, "warnings": report.warnings}

    @app.post("/api/terms")
    def terms(envelope: SetupEnvelope) -> dict[str, dict[str, str]]:
        """Which kinetic-equation terms this setup actually solves.

        The interface used to work this out for itself, which meant two
        implementations of one question and, inevitably, three places where
        they disagreed. This is the authority; the panel renders what it is
        told. See :mod:`qpsim.webui.terms`.
        """
        return term_status_payload(envelope.setup)

    # -- setups -------------------------------------------------------

    @app.get("/api/setups")
    def setups_list() -> list[dict[str, Any]]:
        return workspace.list_setups()

    @app.post("/api/setups")
    def setups_save(envelope: SetupEnvelope) -> dict[str, str]:
        return {"slug": workspace.save_setup(envelope)}

    @app.get("/api/setups/{slug}")
    def setups_get(slug: str) -> dict[str, Any]:
        try:
            envelope = workspace.load_setup(slug)
        except (FileNotFoundError, ValueError) as exc:  # ValueError: unsafe slug
            raise HTTPException(404, f"No setup {slug!r}.") from exc
        return {"name": envelope.name, "setup": envelope.setup.model_dump()}

    @app.delete("/api/setups/{slug}")
    def setups_delete(slug: str) -> dict[str, bool]:
        try:
            workspace.delete_setup(slug)
        except ValueError as exc:  # unsafe slug
            raise HTTPException(404, f"No setup {slug!r}.") from exc
        return {"deleted": True}

    # -- runs ---------------------------------------------------------

    @app.post("/api/runs")
    def runs_create(envelope: SetupEnvelope) -> JSONResponse:
        report = validate_setup(envelope.setup)
        if not report.ok:
            return JSONResponse(
                status_code=400,
                content={"errors": report.errors, "warnings": report.warnings},
            )
        # Warnings persist as the run's first notes — the browser
        # switches views on submit, so a transient banner would vanish.
        run_id = runner.submit(envelope, warnings=report.warnings)
        return JSONResponse({"id": run_id, "warnings": report.warnings})

    @app.get("/api/runs")
    def runs_list() -> list[dict[str, Any]]:
        return [runner.overlay(m) for m in workspace.list_runs()]

    @app.get("/api/runs/{run_id}")
    def runs_get(run_id: str) -> dict[str, Any]:
        try:
            manifest = workspace.read_manifest(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, f"No run {run_id!r}.") from exc
        except ValueError as exc:  # corrupt manifest (JSONDecodeError included)
            raise HTTPException(404, f"Run {run_id!r} has an unreadable manifest.") from exc
        manifest = runner.overlay(manifest)
        if manifest["status"] == "done":
            try:
                # Zip namelist only — no array decompression on a poll.
                names = workspace.array_names(run_id)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                manifest["plots"] = []
                manifest["csvs"] = []
                manifest["artifacts_error"] = f"result arrays unavailable: {exc}"
            else:
                manifest["plots"] = available_plots(manifest["mode"], names)
                manifest["csvs"] = available_csvs(manifest["mode"], names)
                # Parameter counts come from the stored data, so a scrubber
                # cannot offer an index this run does not have. Only the
                # arrays the families name are read: this is a polling path.
                wanted: set[str] = set()
                for plot in manifest["plots"]:
                    wanted |= plot_parameter_arrays(manifest["mode"], plot)
                shapes = workspace.array_shapes(run_id, wanted) if wanted else {}
                manifest["plot_params"] = {
                    plot: params
                    for plot in manifest["plots"]
                    if (params := plot_parameters(manifest["mode"], plot, shapes))
                }
        return manifest

    @app.post("/api/runs/{run_id}/cancel")
    def runs_cancel(run_id: str) -> dict[str, bool]:
        return {"cancelled": runner.cancel(run_id)}

    @app.delete("/api/runs/{run_id}")
    def runs_delete(run_id: str) -> dict[str, bool]:
        live = runner.live_state(run_id)
        if live is not None and live.status in ("queued", "running"):
            raise HTTPException(409, "Run is active — cancel it first.")
        if live is not None and not runner.release_terminal_for_delete(run_id):
            # Status becomes terminal just before the worker makes its final
            # persistence attempt. Do not race that attempt, but allow a
            # finished job with a permanently unwritable manifest to discard
            # its pending snapshot and be deleted without a server restart.
            raise HTTPException(409, "Run is still finalizing; retry shortly.")
        try:
            workspace.delete_run(run_id)
        except ValueError as exc:  # unsafe run_id
            raise HTTPException(404, f"No run {run_id!r}.") from exc
        except OSError as exc:
            raise HTTPException(
                409,
                "Run artifacts are busy (for example, an active download); "
                "retry deletion shortly.",
            ) from exc
        return {"deleted": True}

    def _load_run_artifacts(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            manifest = workspace.read_manifest(run_id)
            arrays = workspace.open_arrays(run_id)
        except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(404, "Run or its arrays are missing/unreadable.") from exc
        return manifest, arrays

    @app.get("/api/runs/{run_id}/plots/{name}.png")
    def runs_plot(
        run_id: str,
        name: str,
        frame: int = 0,
        energy: int = 0,
        omega: int = 0,
    ) -> Response:
        manifest, arrays = _load_run_artifacts(run_id)
        try:
            # A figure family takes an index; a single figure ignores these.
            # Out-of-range is a 404 from render_plot's KeyError rather than a
            # traceback, because a scrubber can outrun a deleted run.
            png = render_plot(
                manifest["mode"], name, arrays, manifest.get("summary", {}),
                {"frame": frame, "energy": energy, "omega": omega},
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=png, media_type="image/png", headers=_CACHE_HEADERS)

    @app.get("/api/runs/{run_id}/csv/{name}.csv")
    def runs_csv(run_id: str, name: str, frame: int | None = None) -> Response:
        manifest, arrays = _load_run_artifacts(run_id)
        try:
            # `frame` selects a recorded snapshot on the tables that have a
            # frame axis; omitted, the table is the run's endpoint. A frame
            # the run does not have, or on a table without that axis, is a
            # 404 from render_csv's KeyError rather than a silent endpoint.
            text = render_csv(
                manifest["mode"], name, arrays, manifest.get("summary", {}),
                {"frame": frame},
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        stem = f"{run_id}-{name}" if frame is None else f"{run_id}-{name}-frame{frame}"
        return Response(
            content=text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}.csv"',
                **_CACHE_HEADERS,
            },
        )

    @app.get("/api/runs/{run_id}/result.npz")
    def runs_arrays(run_id: str) -> FileResponse:
        """The run's arrays, whole and unrendered.

        Every figure and table above is one reduction of this file, chosen
        here. The arrays nobody has written a figure for -- and there are
        several -- were unreachable until this route: a reader with numpy
        and a question this interface did not anticipate needs the payload,
        not a PNG of it. Served as the file on disk, byte for byte, so what
        is downloaded is what the run wrote.
        """
        try:
            workspace.read_manifest(run_id)
            path = workspace.run_dir(run_id) / "result.npz"
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(404, f"No run {run_id!r}.") from exc
        if not path.is_file():
            raise HTTPException(
                404, "Run has no result arrays: not finished, or they were not writable.",
            )
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{run_id}-result.npz",
            headers=_CACHE_HEADERS,
        )

    class _RevalidatingStatic(StaticFiles):
        """Serve the frontend with must-revalidate rather than heuristically.

        Starlette's StaticFiles sends etag and last-modified but no
        Cache-Control, so a browser applies HEURISTIC freshness and will keep
        serving a cached app.js across ordinary reloads. For an app whose UI
        is edited in place that is a persistent trap: the server has the new
        file, the user reloads, and still sees the old interface -- which is
        indistinguishable from the change not working.

        `no-cache` does not mean "do not store": the browser keeps the file
        and revalidates, so an unchanged asset still costs one conditional
        request answered 304 by the etag that is already being sent.
        """

        def is_not_modified(self, response_headers, request_headers):  # type: ignore[override]
            response_headers["Cache-Control"] = "no-cache"
            return super().is_not_modified(response_headers, request_headers)

        async def get_response(self, path: str, scope):  # type: ignore[override]
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache"
            return response

    app.mount("/static", _RevalidatingStatic(directory=STATIC_DIR), name="static")
    return app
