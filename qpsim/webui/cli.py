"""``qpsim-ui`` console entry point.

Starts the local server and (by default) opens the browser on it.
Workspace resolution: ``--workspace`` flag, else the
``QPSIM_WORKSPACE`` environment variable, else ``~/qpsim-workspace``.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path


def default_workspace() -> Path:
    env = os.environ.get("QPSIM_WORKSPACE")
    return Path(env) if env else Path.home() / "qpsim-workspace"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qpsim-ui",
        description="Local web frontend for the qpsim superconductor kinetics framework.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8756, help="port (default: %(default)s)")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="setups/runs directory (default: $QPSIM_WORKSPACE or ~/qpsim-workspace)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab on startup"
    )
    args = parser.parse_args(argv)

    import uvicorn

    from qpsim.webui.server import create_app

    workspace = args.workspace if args.workspace is not None else default_workspace()
    app = create_app(workspace)

    url = f"http://{args.host}:{args.port}/"
    print(f"qpsim frontend — workspace: {workspace}")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
