"""Portable first-page PDF rasterization for manual validation aids.

The Poppler version available on some supported Windows hosts predates the
``pdftoppm -png`` option.  Its default PPM output is older and portable, so
use that lossless format as the interchange and let Matplotlib encode the
final PNG.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.image as mpimg


def rasterize_first_page(pdf: Path, out_png: Path, *, dpi: int = 200) -> None:
    """Rasterize page one of ``pdf`` to a real PNG at ``out_png``."""
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer; got {dpi!r}.")
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "pdftoppm not found on PATH; install Poppler to enable PDF "
            "rasterization."
        )

    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-singlefile",
                "-r",
                str(dpi),
                str(pdf),
                str(prefix),
            ],
            check=True,
        )
        ppm = prefix.with_suffix(".ppm")
        if not ppm.is_file() or ppm.stat().st_size == 0:
            raise RuntimeError(
                "pdftoppm completed without producing a nonempty PPM image."
            )
        image = mpimg.imread(ppm)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(out_png, image)
