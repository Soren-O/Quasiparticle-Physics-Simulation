"""One-sided-regeneration guard: duplicated pinned quantities must agree.

The same branch-tracked steady states are pinned in more than one
baseline CSV:

* ``m25_fig3{a,b}_paper.csv`` and
  ``m25_fig3{a,b}_chemical_potentials.csv`` pin the identical
  ``(T, x_α, p_1, μ_α)`` sweeps (both modules call
  ``solve_panel_branch_sweep`` on the same grid and apply the same
  SI Eqs. S2–S5 inversion);
* the full-model rows of ``m25_fig4_paper.csv`` and
  ``m25_fig4{a,b}_parity_rates.csv`` pin the identical
  ``(T, Γ_P, Γ^eo_01/Γ^eo_10)`` curves.

Regenerating one module's baseline without the others would let the
duplicated numbers drift apart silently — the per-module pin tests
each compare only against their own CSV. This test parses the pinned
files themselves (no solves) and asserts numeric equality between
every genuinely shared column, so a one-sided regeneration fails
loudly forever. Equality is exact: the pinned pairs are written from
the same deterministic computation, and since the sweep memoization in
``fig3_chemical_potentials.solve_panel_branch_sweep`` they come from
the same in-process result object.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025 import (
    fig3_chemical_potentials,
    fig3_paper,
    fig4_paper,
    fig4_parity_rates,
)

_FIG3_SHARED_FIELDS = (
    "T_kelvin", "x_L", "x_Rgt", "x_Rlt", "p_1",
    "mu_L_GHz", "mu_Rgt_GHz", "mu_Rlt_GHz",
)
_FIG4_SHARED_FIELDS = ("T_kelvin", "Gamma_P_Hz", "ratio_eo_01_over_10")


def test_fig3_paper_and_chemical_potentials_baselines_agree() -> None:
    paths = (
        fig3_paper.baseline_path_a(), fig3_paper.baseline_path_b(),
        fig3_chemical_potentials.baseline_path_a(),
        fig3_chemical_potentials.baseline_path_b(),
    )
    if not all(p.exists() for p in paths):
        pytest.skip("Fig 3 baselines missing; generate both fig3 modules.")

    paper = fig3_paper.read_baseline()
    chem = fig3_chemical_potentials.read_baseline()
    for panel_name, panel_paper, panel_chem in (
        ("panel_a", paper.panel_a, chem.panel_a),
        ("panel_b", paper.panel_b, chem.panel_b),
    ):
        for field in _FIG3_SHARED_FIELDS:
            np.testing.assert_array_equal(
                getattr(panel_paper, field),
                getattr(panel_chem, field),
                err_msg=(
                    f"{panel_name}: {field} differs between "
                    "m25_fig3*_paper.csv and "
                    "m25_fig3*_chemical_potentials.csv — one-sided "
                    "baseline regeneration"
                ),
            )


def test_fig4_paper_full_model_and_parity_rates_baselines_agree() -> None:
    paths = (
        fig4_paper.baseline_path(),
        fig4_parity_rates.baseline_path_a(),
        fig4_parity_rates.baseline_path_b(),
    )
    if not all(p.exists() for p in paths):
        pytest.skip("Fig 4 baselines missing; generate both fig4 modules.")

    paper = fig4_paper.read_baseline(
        accept_producer_certificate_claims=True
    )
    parity = fig4_parity_rates.read_baseline(
        accept_producer_certificate_claims=True
    )
    for omega_LR_GHz, panel_parity in (
        (0.5, parity.panel_a),
        (5.0, parity.panel_b),
    ):
        # Only the full model is shared: the global-quasiequilibrium
        # and renormalized reductions exist in m25_fig4_paper.csv
        # alone, so they are not compared here.
        panel_full = paper.get(omega_LR_GHz, fig4_paper.MODEL_FULL)
        for field in _FIG4_SHARED_FIELDS:
            np.testing.assert_array_equal(
                getattr(panel_full, field),
                getattr(panel_parity, field),
                err_msg=(
                    f"ω_LR={omega_LR_GHz} GHz: {field} differs between "
                    "m25_fig4_paper.csv (full model) and "
                    "m25_fig4*_parity_rates.csv — one-sided baseline "
                    "regeneration"
                ),
            )
