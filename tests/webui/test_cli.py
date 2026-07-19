"""Command-line host-policy tests for the local qpsim web UI."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from qpsim.webui.cli import main


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "example.com", ""])
def test_non_loopback_bind_fails_before_server_start(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *_args, **kwargs: calls.append(kwargs)),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--host", host, "--no-browser"])

    assert exc_info.value.code == 2
    assert "must be localhost or a loopback IP address" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.2", "::1"])
def test_loopback_bind_is_forwarded(
    host: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *_args, **kwargs: calls.append(kwargs)),
    )

    main(["--host", host, "--workspace", str(tmp_path), "--no-browser"])

    assert calls == [{"host": host, "port": 8756, "log_level": "warning"}]


def test_ipv6_browser_url_is_bracketed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    class ImmediateTimer:
        def __init__(
            self,
            _delay: float,
            callback: Any,
            args: tuple[str, ...],
        ) -> None:
            self._callback = callback
            self._args = args

        def start(self) -> None:
            self._callback(*self._args)

    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr("qpsim.webui.cli.threading.Timer", ImmediateTimer)
    monkeypatch.setattr("qpsim.webui.cli.webbrowser.open", opened.append)

    main(["--host", "::1", "--port", "9000", "--workspace", str(tmp_path)])

    assert opened == ["http://[::1]:9000/"]
