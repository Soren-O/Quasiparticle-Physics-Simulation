from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import pytest
from matplotlib.figure import Figure

from validation import rasterize_pdf


def _write_one_page_pdf(path: Path) -> None:
    fig = Figure(figsize=(2.0, 1.5))
    ax = fig.subplots()
    ax.plot([0.0, 1.0], [1.0, 0.0])
    fig.savefig(path)


def test_rasterize_first_page_works_without_poppler_png_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "source.pdf"
    png = tmp_path / "render.png"
    _write_one_page_pdf(pdf)

    executable = str(tmp_path / "fake-pdftoppm")
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append((command, check))
        # P6 PPM: two pixels, red then blue.  This emulates the portable
        # default output supported even by Poppler builds without ``-png``.
        Path(command[-1]).with_suffix(".ppm").write_bytes(
            b"P6\n2 1\n255\n\xff\x00\x00\x00\x00\xff"
        )
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(
        rasterize_pdf.shutil,
        "which",
        lambda name: executable if name == "pdftoppm" else None,
    )
    monkeypatch.setattr(rasterize_pdf.subprocess, "run", fake_run)

    rasterize_pdf.rasterize_first_page(pdf, png, dpi=72)

    assert len(calls) == 1
    command, check = calls[0]
    assert check is True
    assert command[:5] == [
        executable,
        "-singlefile",
        "-r",
        "72",
        str(pdf),
    ]
    assert Path(command[-1]).name == "page"
    assert "-png" not in command
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = mpimg.imread(png)
    assert image.ndim == 3
    assert image.shape[:2] == (1, 2)


def test_rasterize_first_page_requires_poppler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rasterize_pdf.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="pdftoppm not found"):
        rasterize_pdf.rasterize_first_page(
            tmp_path / "missing.pdf",
            tmp_path / "render.png",
        )


@pytest.mark.parametrize("dpi", [0, -1, 1.5, True])
def test_rasterize_first_page_rejects_invalid_dpi(
    tmp_path: Path,
    dpi: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        rasterize_pdf.rasterize_first_page(  # type: ignore[arg-type]
            tmp_path / "source.pdf",
            tmp_path / "render.png",
            dpi=dpi,  # type: ignore[arg-type]
        )
