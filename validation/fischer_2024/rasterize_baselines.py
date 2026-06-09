"""Rasterize qpsim baseline PDFs into output/ as PNGs.

The figN_*.py modules in this package write PDFs under
``validation/baselines/ph0_constant/``. This helper rasterizes them via
``pdftoppm`` so make_comparison.py (which expects PNGs in output/) has
fresh inputs.

Run after regenerating any baseline:

    python -m validation.fischer_2024.rasterize_baselines
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
QPSIM_ROOT = HERE.parents[1]
BASELINES = QPSIM_ROOT / "validation" / "baselines"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

PDFS: dict[str, str] = {
    "ph0_constant/fischer2024_fig5_qpsim_native.pdf": "fig5_paper.png",
    "ph0_constant/fischer2024_fig8_qpsim_native.pdf": "fig8_paper.png",
    "ph0_constant/f24_fig8_xqp_pb.pdf":               "fig8_xqp_pb.png",
    "ph0_constant/f24_figs_5_7_fe_pb.pdf":            "figs_5_7_fe_pb.png",
}


def rasterize(pdf: Path, out_png: Path, dpi: int = 200) -> None:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError(
            "pdftoppm not found on PATH; install poppler "
            "(brew install poppler) to enable PDF rasterization."
        )
    with tempfile.TemporaryDirectory() as td:
        prefix = Path(td) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1",
             str(pdf), str(prefix)],
            check=True,
        )
        produced = next(Path(td).glob("page-*.png"))
        shutil.copyfile(produced, out_png)


def main() -> None:
    for rel, png_name in PDFS.items():
        src = BASELINES / rel
        dst = OUT_DIR / png_name
        if not src.exists():
            print(f"[skip] missing baseline: {src}")
            continue
        rasterize(src, dst)
        print(f"[ok] {dst.name}")


if __name__ == "__main__":
    main()
