"""Manual raster side-by-side: local paper image vs qpsim output.

Edit MAPPING to pin which paper-figure panel each output should pair with.
Any output without a paper-figure match is skipped with a notice.
This helper performs no digitization, curve extraction, alignment, or
quantitative comparison and is not part of an automated parity gate.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "qpsim-fischer2023-mpl"))
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
PAPER_DIR = ROOT / "paper_figures"
OUT_DIR = ROOT / "output"
CMP_DIR = ROOT / "comparisons"

# paper figure filename (in paper_figures/) -> output filename (in output/)
MAPPING = {
    "Fischer2023Fig3.png": "fig3_paper.png",
    "Fischer2023Fig5a.png": "fig5_paper_a.png",
    "Fischer2023Fig5b.png": "fig5_paper_b.png",
    "Fischer2023Fig6.png": "fig6_paper.png",
    "Fischer2023Fig7a.png": "fig7_paper.png",
}

PAPER_LABEL = "Paper"
REPRO_LABEL = "qpsim"


def make_pair(paper_path: Path, repro_path: Path, out_path: Path) -> None:
    paper = mpimg.imread(paper_path)
    repro = mpimg.imread(repro_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(paper)
    axes[0].set_title(f"{PAPER_LABEL}: {paper_path.name}")
    axes[1].imshow(repro)
    axes[1].set_title(f"{REPRO_LABEL}: {repro_path.name}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    created = 0
    for paper_name, out_name in MAPPING.items():
        paper_path = PAPER_DIR / paper_name
        out_path = OUT_DIR / out_name
        if not paper_path.exists():
            print(f"[skip] missing paper figure: {paper_name}")
            continue
        if not out_path.exists():
            print(f"[skip] missing output: {out_name}")
            continue
        CMP_DIR.mkdir(parents=True, exist_ok=True)
        cmp_path = CMP_DIR / f"{paper_path.stem}_sidebyside.png"
        make_pair(paper_path, out_path, cmp_path)
        created += 1
        print(f"[ok] {cmp_path.name}")
    if created == 0:
        raise SystemExit(
            "No side-by-side comparisons were created. Supply at least one "
            "mapped local paper raster and its qpsim output; this helper does "
            "not download or digitize paper data."
        )


if __name__ == "__main__":
    main()
