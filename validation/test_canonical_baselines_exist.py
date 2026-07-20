"""Every committed canonical baseline must exist on disk.

The per-figure pinned-baseline tests ``pytest.skip`` when their CSV is
missing (a deliberate convenience for regenerating a single figure in a
scratch tree). The 2026-07-19 audit found that this let a deleted or
renamed canonical artifact silently green-wash 19 pinned regressions:
``fischer_2024`` asserted artifact existence, but the fischer_2023,
marchegiani_2025 and transient families did not.

This manifest closes that hole. It lists the committed canonical
baselines explicitly; if one goes missing the suite fails loudly here
even though the per-figure test skipped. Adding a new canonical baseline
means adding it to this list — that is the point (the manifest is the
existence contract).

Legacy/archive artifacts are intentionally excluded: they are quarantined
history, not live pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

#: Committed canonical (live-pin) baselines, relative to validation/baselines.
CANONICAL_BASELINES = (
    "marchegiani_2025/m25_fig3_crossover_temperature.csv",
    "marchegiani_2025/m25_fig3a_chemical_potentials.csv",
    "marchegiani_2025/m25_fig3a_paper.csv",
    "marchegiani_2025/m25_fig3b_chemical_potentials.csv",
    "marchegiani_2025/m25_fig3b_paper.csv",
    "marchegiani_2025/m25_fig4_paper.csv",
    "marchegiani_2025/m25_fig4a_parity_rates.csv",
    "marchegiani_2025/m25_fig4b_parity_rates.csv",
    "ph0_constant/f24_fig8_xqp_pb.csv",
    "ph0_constant/f24_figs_5_7_fe_pb.csv",
    "ph0_constant/fischer2024_fig5_qpsim_native.csv",
    "ph0_constant/fischer2024_fig8_qpsim_native.csv",
    "ph0_constant/fischer_fig3_paper.csv",
    "ph0_constant/fischer_fig5_paper.csv",
    "ph0_constant/fischer_fig7_paper.csv",
    "ph0_constant/fischer_figs_9_13_qi_vs_pread.csv",
    "ph0_kaplan/fischer_fig6_paper.csv",
    "transient/photon_kick_response.csv",
)


@pytest.mark.parametrize("relative", CANONICAL_BASELINES)
def test_canonical_baseline_exists(relative: str) -> None:
    path = _BASELINES_DIR / relative
    assert path.is_file(), (
        f"Committed canonical baseline missing: {path}. The per-figure test "
        "will silently skip without it — restore the artifact (or, for a "
        "deliberate retirement, remove it from CANONICAL_BASELINES with a "
        "documented rationale)."
    )
    assert path.stat().st_size > 0, f"Canonical baseline is empty: {path}"
