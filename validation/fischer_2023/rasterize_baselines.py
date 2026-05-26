"""Rasterize qpsim baseline PDFs into output/ as PNGs.

The figN_*.py modules in this package write PDFs under
``validation/baselines/{ph0_constant,ph0_kaplan}/``. This helper
rasterizes them via ``pdftoppm`` so make_comparison.py (which expects
PNGs in output/) has fresh inputs.

Run after regenerating any baseline:

    python -m validation.fischer_2023.rasterize_baselines
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

# baseline PDF (relative to BASELINES) -> output PNG name (in output/)
PDFS: dict[str, str] = {
    "ph0_constant/fischer_fig3_paper.pdf":               "fig3_paper.png",
    "ph0_constant/fischer_fig5_paper_a.pdf":             "fig5_paper_a.png",
    "ph0_constant/fischer_fig5_paper_b.pdf":             "fig5_paper_b.png",
    "ph0_kaplan/fischer_fig6_paper.pdf":                 "fig6_paper.png",
    "ph0_constant/fischer_fig7_paper.pdf":               "fig7_paper.png",
    "ph0_constant/fischer_figs_9_13_qi_vs_pread.pdf":    "figs_9_13_qi_vs_pread.png",
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
