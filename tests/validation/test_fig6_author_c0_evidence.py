from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY,
    SUMMARY_SCHEMA,
    C0SummaryError,
    build_c0_summary,
    canonical_summary_bytes,
    load_c0_summary,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_PATH = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "author-point-T020-sweep049-exact-anchor.json"
)


def test_checked_c0_summary_is_accepted_and_one_point_scoped() -> None:
    summary = load_c0_summary()
    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["acceptance"]["accepted"] is True
    assert summary["stage"] == {
        "comparison_stage_id": "A1",
        "parent_stage_id": None,
        "stage_id": "C0",
        "status": "completed",
    }
    assert summary["solver_verification"]["verified_every_transition"] is True
    assert summary["solver_verification"]["iterations"] == 4
    assert summary["comparison"]["a1_final_state_relative_l2"] < 1e-12
    assert summary["comparison"]["figure6_ordinate_signed_difference"] == 0.0
    assert summary["limitations"]["scope"].startswith("one authenticated point")


def test_c0_import_path_does_not_import_qpsim() -> None:
    script = r"""
import importlib.abc
import sys

class BlockQpsim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "qpsim" or fullname.startswith("qpsim."):
            raise RuntimeError("qpsim import attempted")
        return None

sys.meta_path.insert(0, BlockQpsim())
import validation.fischer_2023.fig6_author_c0_bundle  # noqa: F401
assert not any(name == "qpsim" or name.startswith("qpsim.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_c0_core_source_has_no_qpsim_import() -> None:
    source = (
        REPOSITORY_ROOT / "validation" / "reference_models" / "fischer_2023" / "fig6_author_c0.py"
    )
    spec = importlib.util.spec_from_file_location("_c0_ast_probe", source)
    assert spec is not None
    text = source.read_text(encoding="utf-8")
    assert "import qpsim" not in text
    assert "from qpsim" not in text


def _external_bundle_paths() -> tuple[Path, Path]:
    c0 = os.environ.get("QPSIM_FISCHER2023_FIG6_C0_BUNDLE")
    a1 = os.environ.get("QPSIM_FISCHER2023_FIG6_A1_BUNDLE")
    if c0 is None or a1 is None:
        pytest.skip(
            "Set QPSIM_FISCHER2023_FIG6_C0_BUNDLE and "
            "QPSIM_FISCHER2023_FIG6_A1_BUNDLE for raw C0 checks."
        )
    return Path(c0), Path(a1)


def test_c0_checked_summary_regenerates_byte_exactly() -> None:
    c0, a1 = _external_bundle_paths()
    regenerated = build_c0_summary(
        c0,
        a1_bundle_dir=a1,
        anchor_path=ANCHOR_PATH,
    )
    assert canonical_summary_bytes(regenerated) == DEFAULT_SUMMARY.read_bytes()


def test_c0_summary_rejects_rehashed_forged_history(tmp_path: Path) -> None:
    c0, a1 = _external_bundle_paths()
    copied = tmp_path / "c0"
    shutil.copytree(c0, copied)
    history_path = copied / "state_history.npy"
    raw = bytearray(history_path.read_bytes())
    # Change one payload byte without damaging the NPY header, then rebind the
    # transport hash. The semantic array descriptor/history checks must still
    # reject the forged transition.
    raw[-8] ^= 1
    history_path.write_bytes(raw)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["state_history.npy"]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C0SummaryError, match=r"descriptor|history|transition"):
        build_c0_summary(
            copied,
            a1_bundle_dir=a1,
            anchor_path=ANCHOR_PATH,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_c0_summary_requires_exact_raw_source_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    c0, a1 = _external_bundle_paths()
    copied = tmp_path / "c0"
    shutil.copytree(c0, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["metadata"]["sources"]
    if mutation == "missing":
        sources.pop("validation/fischer_2023/fig6_author_frozen_state.py")
    else:
        sources["tests/validation/not_a_producer_dependency.py"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(C0SummaryError, match="explicit producer dependency set"):
        build_c0_summary(
            copied,
            a1_bundle_dir=a1,
            anchor_path=ANCHOR_PATH,
        )
