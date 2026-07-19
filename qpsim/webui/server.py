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
from qpsim.webui.builders import validate_setup
from qpsim.webui.hosts import is_loopback_host
from qpsim.webui.plots import available_csvs, available_plots, render_csv, render_plot
from qpsim.webui.runner import JobRunner
from qpsim.webui.schemas import MODE_CLASSES, MODE_LABELS, SetupEnvelope
from qpsim.webui.store import Workspace

STATIC_DIR = Path(__file__).parent / "static"

# ``testserver`` is TestClient's reserved in-process host. The CLI refuses a
# non-loopback bind, while this Host check is defence in depth against browser
# DNS rebinding. It is not authentication: a direct HTTP client can choose its
# Host header, which is why the bind restriction is the primary boundary.
_TEST_HOSTS = frozenset({"testserver"})


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
        return FileResponse(STATIC_DIR / "index.html")

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

    @app.get("/api/defaults/{mode}")
    def defaults(mode: str) -> dict[str, Any]:
        setup_cls = MODE_CLASSES.get(mode)
        if setup_cls is None:
            raise HTTPException(404, f"Unknown mode {mode!r}.")
        return setup_cls().model_dump()

    # -- validation ---------------------------------------------------

    @app.post("/api/validate")
    def validate(envelope: SetupEnvelope) -> dict[str, Any]:
        report = validate_setup(envelope.setup)
        return {"ok": report.ok, "errors": report.errors, "warnings": report.warnings}

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
            arrays = workspace.read_arrays(run_id)
        except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
            raise HTTPException(404, "Run or its arrays are missing/unreadable.") from exc
        return manifest, arrays

    # A finished run's arrays never change, so rendered artifacts can
    # be cached — this also stops the browser re-fetching plot PNGs
    # when the detail view re-renders.
    _CACHE_HEADERS = {"Cache-Control": "private, max-age=3600"}

    @app.get("/api/runs/{run_id}/plots/{name}.png")
    def runs_plot(run_id: str, name: str) -> Response:
        manifest, arrays = _load_run_artifacts(run_id)
        try:
            png = render_plot(manifest["mode"], name, arrays, manifest.get("summary", {}))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(content=png, media_type="image/png", headers=_CACHE_HEADERS)

    @app.get("/api/runs/{run_id}/csv/{name}.csv")
    def runs_csv(run_id: str, name: str) -> Response:
        manifest, arrays = _load_run_artifacts(run_id)
        try:
            text = render_csv(manifest["mode"], name, arrays)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return Response(
            content=text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}-{name}.csv"',
                **_CACHE_HEADERS,
            },
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
