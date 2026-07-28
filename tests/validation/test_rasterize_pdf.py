from __future__ import annotations

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
) -> None:
    pdf = tmp_path / "source.pdf"
    png = tmp_path / "render.png"
    _write_one_page_pdf(pdf)

    rasterize_pdf.rasterize_first_page(pdf, png, dpi=72)

    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = mpimg.imread(png)
    assert image.ndim == 3
    assert image.shape[0] > 0
    assert image.shape[1] > 0


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
