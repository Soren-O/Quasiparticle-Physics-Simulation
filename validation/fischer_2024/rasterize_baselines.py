"""Rasterize qpsim baseline PDFs into output/ as PNGs.

The figN_*.py modules in this package write PDFs under
``validation/baselines/ph0_constant/``. This helper rasterizes them via
``pdftoppm`` so make_comparison.py (which expects PNGs in output/) has
fresh inputs.

Run after regenerating any baseline:

    python -m validation.fischer_2024.rasterize_baselines
"""

from __future__ import annotations

from pathlib import Path

from validation.rasterize_pdf import rasterize_first_page

HERE = Path(__file__).parent
QPSIM_ROOT = HERE.parents[1]
BASELINES = QPSIM_ROOT / "validation" / "baselines"
OUT_DIR = HERE / "output"

PDFS: dict[str, str] = {
    "ph0_constant/fischer2024_fig5_qpsim_native.pdf": "fig5_paper.png",
    "ph0_constant/fischer2024_fig8_qpsim_native.pdf": "fig8_paper.png",
    "ph0_constant/f24_fig8_xqp_pb.pdf": "fig8_xqp_pb.png",
    "ph0_constant/f24_figs_5_7_fe_pb.pdf": "figs_5_7_fe_pb.png",
}


def rasterize(pdf: Path, out_png: Path, dpi: int = 200) -> None:
    rasterize_first_page(pdf, out_png, dpi=dpi)


def main() -> None:
    created = 0
    for rel, png_name in PDFS.items():
        src = BASELINES / rel
        dst = OUT_DIR / png_name
        if not src.exists():
            print(f"[skip] missing baseline: {src}")
            continue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rasterize(src, dst)
        created += 1
        print(f"[ok] {dst.name}")
    if created == 0:
        raise SystemExit("No qpsim baseline PDFs were rasterized; every mapped input is missing.")


if __name__ == "__main__":
    main()
