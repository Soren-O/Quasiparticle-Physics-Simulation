"""Every committed canonical validation artifact must exist on disk.

Some per-figure pinned-baseline tests ``pytest.skip`` when their CSV is
missing (a deliberate convenience for regenerating a single figure in a
scratch tree). The 2026-07-19 audit found that this let a deleted or
renamed canonical artifact silently green-wash 19 pinned regressions:
``fischer_2024`` asserted artifact existence, but the fischer_2023,
marchegiani_2025 and transient families did not.

This manifest closes that hole. It lists every required member of each
canonical bundle explicitly: numerical tables, authenticated plot
companions, and validation/promotion records. If one goes missing the
suite fails loudly here even though the per-figure test skipped. Adding a
new canonical baseline means adding its complete bundle to this list —
that is the point (the manifest is the existence contract).

Legacy/archive artifacts are excluded from the live manifest; retained
quarantine evidence has a separate existence-only list below. Lock files,
resumable campaign directories, and explicitly diagnostic/direct outputs are
excluded because they are workflow state or review evidence, not canonical
artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

#: Live committed artifact bundles, relative to validation/baselines.
LIVE_CANONICAL_ARTIFACTS = (
    "marchegiani_2025/m25_fig3_chemical_potentials.artifact.json",
    "marchegiani_2025/m25_fig3_chemical_potentials.pdf",
    "marchegiani_2025/m25_fig3_crossover_temperature.artifact.json",
    "marchegiani_2025/m25_fig3_crossover_temperature.csv",
    "marchegiani_2025/m25_fig3_crossover_temperature.pdf",
    "marchegiani_2025/m25_fig3_paper.artifact.json",
    "marchegiani_2025/m25_fig3_paper.pdf",
    "marchegiani_2025/m25_fig3a_chemical_potentials.csv",
    "marchegiani_2025/m25_fig3a_paper.csv",
    "marchegiani_2025/m25_fig3b_chemical_potentials.csv",
    "marchegiani_2025/m25_fig3b_paper.csv",
    "marchegiani_2025/m25_fig4_paper.artifact.json",
    "marchegiani_2025/m25_fig4_paper.csv",
    "marchegiani_2025/m25_fig4_paper.pdf",
    "marchegiani_2025/m25_fig4_parity_rates.artifact.json",
    "marchegiani_2025/m25_fig4_parity_rates.pdf",
    "marchegiani_2025/m25_fig4a_parity_rates.csv",
    "marchegiani_2025/m25_fig4b_parity_rates.csv",
    "ph0_constant/f24_fig8_xqp_pb.csv",
    "ph0_constant/f24_fig8_xqp_pb.pdf",
    "ph0_constant/f24_figs_5_7_fe_pb.csv",
    "ph0_constant/f24_figs_5_7_fe_pb.pdf",
    "ph0_constant/fischer2024_fig5_qpsim_native.csv",
    "ph0_constant/fischer2024_fig5_qpsim_native.pdf",
    "ph0_constant/fischer2024_fig8_qpsim_native.csv",
    "ph0_constant/fischer2024_fig8_qpsim_native.pdf",
    "ph0_constant/fischer_fig3_paper.csv",
    "ph0_constant/fischer_fig3_paper.pdf",
    "ph0_constant/fischer_fig3_paper.validation.json",
    "ph0_constant/fischer_fig5_paper.campaign.json",
    "ph0_constant/fischer_fig5_paper.csv",
    "ph0_constant/fischer_fig5_paper.promotion.json",
    "ph0_constant/fischer_fig5_paper_a.pdf",
    "ph0_constant/fischer_fig5_paper_b.pdf",
    "ph0_constant/fischer_fig7_paper.csv",
    "ph0_constant/fischer_fig7_paper.pdf",
    "ph0_constant/fischer_fig7_paper.promotion.json",
    "ph0_kaplan/fischer_fig6_paper.csv",
    "ph0_kaplan/fischer_fig6_paper.pdf",
    "ph0_kaplan/fischer_fig6_paper.promotion.json",
    "transient/photon_kick_response.csv",
    "transient/photon_kick_response.pdf",
)

#: Deliberately retained historical evidence that is not an accepted live pin.
QUARANTINED_ARCHIVES = (
    "ph0_constant/fischer_figs_9_13_qi_vs_pread.csv",
    "ph0_constant/fischer_figs_9_13_qi_vs_pread.pdf",
)


@pytest.mark.parametrize("relative", LIVE_CANONICAL_ARTIFACTS)
def test_live_canonical_artifact_exists(relative: str) -> None:
    path = _BASELINES_DIR / relative
    assert path.is_file(), (
        f"Live canonical artifact missing: {path}. The per-figure test "
        "will silently skip without it — restore the artifact (or, for a "
        "deliberate retirement, remove it from LIVE_CANONICAL_ARTIFACTS with a "
        "documented rationale)."
    )
    assert path.stat().st_size > 0, f"Live canonical artifact is empty: {path}"


@pytest.mark.parametrize("relative", QUARANTINED_ARCHIVES)
def test_quarantined_archive_is_preserved(relative: str) -> None:
    path = _BASELINES_DIR / relative
    assert path.is_file(), f"Quarantined historical artifact missing: {path}"
    assert path.stat().st_size > 0, f"Quarantined historical artifact is empty: {path}"
