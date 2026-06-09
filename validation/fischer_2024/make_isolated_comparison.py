"""Side-by-side: a paper figure (left) vs a single reproduction output (right).

Usage:
    python make_isolated_comparison.py fig8_xqp_pb.png Fischer2024Fig8.png \
        comparisons/Fischer2024Fig8_xqp_pb_sidebyside.png

The first arg is the file in output/ to use for the right panel. The
second arg is the paper-figure file (in paper_figures/) to use for the
left panel. The third arg is the output sidebyside path.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "qpsim-fischer2024-mpl")
)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "output"
PAPER_DIR = ROOT / "paper_figures"


def make_pair(paper_path: Path, repro_path: Path, out_path: Path) -> None:
    paper = mpimg.imread(paper_path)
    repro = mpimg.imread(repro_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(paper)
    axes[0].set_title(f"Paper: {paper_path.name}")
    axes[1].imshow(repro)
    axes[1].set_title(f"qpsim: {repro_path.name}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    repro = OUT_DIR / argv[1]
    paper = PAPER_DIR / argv[2]
    out = ROOT / argv[3]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paper.exists():
        print(f"[err] missing paper figure: {paper}")
        return 1
    if not repro.exists():
        print(f"[err] missing output: {repro}")
        return 1
    make_pair(paper, repro, out)
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
