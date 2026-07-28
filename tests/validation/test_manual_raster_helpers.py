"""Fail-closed contracts for the optional manual raster helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "validation.fischer_2023.make_comparison",
        "validation.fischer_2024.make_comparison",
    ),
)
def test_comparison_cli_refuses_zero_complete_pairs(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "PAPER_DIR", tmp_path / "paper")
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "output")
    monkeypatch.setattr(module, "CMP_DIR", tmp_path / "comparisons")
    monkeypatch.setattr(module, "MAPPING", {"paper.png": "qpsim.png"})

    with pytest.raises(SystemExit, match="No side-by-side comparisons"):
        module.main()


@pytest.mark.parametrize(
    "module_name",
    (
        "validation.fischer_2023.rasterize_baselines",
        "validation.fischer_2024.rasterize_baselines",
    ),
)
def test_rasterizer_cli_refuses_zero_existing_pdfs(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "BASELINES", tmp_path / "baselines")
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "output")
    monkeypatch.setattr(module, "PDFS", {"missing.pdf": "missing.png"})

    with pytest.raises(SystemExit, match="No qpsim baseline PDFs"):
        module.main()


@pytest.mark.parametrize(
    "module_name",
    (
        "validation.fischer_2023.make_comparison",
        "validation.fischer_2024.make_comparison",
    ),
)
def test_comparison_cli_accepts_one_complete_pair(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = importlib.import_module(module_name)
    paper_dir = tmp_path / "paper"
    output_dir = tmp_path / "output"
    comparison_dir = tmp_path / "comparisons"
    paper_dir.mkdir()
    output_dir.mkdir()
    comparison_dir.mkdir()
    (paper_dir / "paper.png").write_bytes(b"paper")
    (output_dir / "qpsim.png").write_bytes(b"qpsim")
    generated: list[Path] = []
    monkeypatch.setattr(module, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(module, "OUT_DIR", output_dir)
    monkeypatch.setattr(module, "CMP_DIR", comparison_dir)
    monkeypatch.setattr(module, "MAPPING", {"paper.png": "qpsim.png"})
    monkeypatch.setattr(
        module,
        "make_pair",
        lambda _paper, _qpsim, destination: generated.append(destination),
    )

    module.main()

    assert generated == [comparison_dir / "paper_sidebyside.png"]
