"""Analytic spectral-flow fixtures from the paper.

Pins the two exact properties of the conservative spectral-flow operator
(paper eq:full_kinetic_conservative) that the BCS DOS-continuity identity
(eq:dos_continuity) makes available as machine checks:

* **Frozen-shell exactness** — with collisions off and a uniform Δ(t),
  any smooth shell function ``f = G(ξ)``, ``ξ = √(E² − Δ²(t))``, is an
  exact solution: each shell carries its occupation unchanged while the
  gap moves.
* **Discrete DOS continuity** — the same step applied to ``f ≡ 1``
  (so ``u = N₁``) must reproduce ``∂_t N₁ + ∂_E[(Δ/E)Δ̇ N₁] = 0`` to
  scheme order: ``N₁(Δ_old)`` advects onto ``N₁(Δ_new)``.

The tests replicate ``T3DiffusionBackend.apply_gap_update``'s composition
(advect ``u = N₁ f`` → rebuild DOS at the new gap → recover
``f = u/N₁``, clip) with a *prescribed* linear gap ramp, so they probe
the operator without coupling to ``solve_gap``.

Tolerances follow the audited error structure (2026-06-10): the bare
advection conserves ``Σ u·dE`` to round-off; the recovery step drops
edge-cell residue at first order; the zero-flux top face starves /
piles the top ~``|ΔΔ|·(Δ/E_max)/dE`` cells, which the interior masks
exclude. Grids leave sub-gap room (``energy_min_factor < 1``) so a
falling gap's spectral inflow band stays on-grid.
"""

from __future__ import annotations

import numpy as np
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import bcs_density_of_states
from qpsim.solvers.spectral_flow_tvd import advect_spectral_flow

GAP0 = 1.0
E_MIN_FACTOR = 0.70
E_MAX_FACTOR = 6.0


def _grid(NE: int) -> tuple[np.ndarray, np.ndarray, float]:
    E, dE_u = build_energy_grid(
        gap=GAP0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NE,
    )
    return E, integration_widths_from_centers(E), dE_u


def _run_ramp(
    NE: int,
    gap_final: float,
    n_steps: int,
    f_init: np.ndarray,
    E: np.ndarray,
    dE: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """apply_gap_update's advect → DOS rebuild → f-recovery composition."""
    gap_dot_total = gap_final - GAP0
    dt = 1.0 / n_steps
    gap = GAP0
    N1 = bcs_density_of_states(E, gap)
    f = f_init.copy()
    mass0 = float(np.sum(N1 * f * dE))
    advect_err = 0.0
    for _ in range(n_steps):
        u = N1 * f
        before = float(np.sum(u * dE))
        u_new = advect_spectral_flow(
            u, E, dE, gap=gap, gap_dot=gap_dot_total, dt=dt,
        )
        advect_err = max(advect_err, abs(float(np.sum(u_new * dE)) - before))
        gap = gap + gap_dot_total * dt
        N1 = bcs_density_of_states(E, gap)
        keep = N1 > 1e-30
        f = np.zeros_like(u_new)
        f[keep] = u_new[keep] / N1[keep]
        f = np.clip(f, 0.0, 1.0)
    mass1 = float(np.sum(N1 * f * dE))
    return f, {
        "mass0": mass0,
        "mass1": mass1,
        "advect_err": advect_err,
        "gap": gap,
    }


def _interior_mask(
    E: np.ndarray, dE_u: float, gap_final: float, delta_gap: float
) -> np.ndarray:
    """Bulk region: away from the edge's traversal wake and the top face.

    The moving edge leaves a first-order limiter wake across the whole
    band it traversed, ``[min(Δ), max(Δ)]``, in both ramp directions;
    the scheme's numerical diffusion smears that wake a further
    ``O(√n_steps)`` cells (≈10 at the 100-step ramps used here) — exclude
    the band plus a 16-cell margin. The zero-flux top face starves/piles
    the top ``~|ΔΔ|·(Δ/E_max)/dE`` cells — exclude those too (documented
    caveats of :func:`advect_spectral_flow`).
    """
    edge_margin = max(gap_final, GAP0) + 16.0 * dE_u
    top_starve = abs(delta_gap) * GAP0 / (E_MAX_FACTOR * GAP0)
    top_margin = E_MAX_FACTOR * GAP0 - top_starve - 10.0 * dE_u
    return (E > edge_margin) & (E < top_margin)


class TestFrozenShellExactness:
    """f = G(ξ) is an exact solution; shells are invariant under the ramp."""

    @staticmethod
    def _G(xi: np.ndarray) -> np.ndarray:
        return 0.5 * np.exp(-((xi / (2.0 * GAP0)) ** 2))

    def _shell_error(self, NE: int, gap_final: float, n_steps: int) -> dict[str, float]:
        E, dE, dE_u = _grid(NE)
        xi0 = np.sqrt(np.maximum(E**2 - GAP0**2, 0.0))
        f_init = np.where(E > GAP0, self._G(xi0), 0.0)
        f_fin, diag = _run_ramp(NE, gap_final, n_steps, f_init, E, dE)
        xi1 = np.sqrt(np.maximum(E**2 - gap_final**2, 0.0))
        f_exact = np.where(E > gap_final, self._G(xi1), 0.0)
        bulk = _interior_mask(E, dE_u, gap_final, gap_final - GAP0)
        band = (E > gap_final + 2.0 * dE_u) & (E < GAP0 - 2.0 * dE_u)
        # One edge cell of the final DOS holds ~√(2Δ·dE)·G(0) of mass —
        # the dominant, sampling-parity-sensitive recovery drop. Bound
        # the deficit by that scale rather than a fixed percentage.
        edge_cell_frac = (
            np.sqrt(2.0 * gap_final * dE_u) * float(self._G(np.zeros(1))[0])
            / diag["mass0"]
        )
        return {
            "bulk_err": float(np.max(np.abs(f_fin[bulk] - f_exact[bulk]))),
            "band_err": (
                float(np.mean(np.abs(f_fin[band] - f_exact[band])))
                if band.any()
                else 0.0
            ),
            "advect_err": diag["advect_err"],
            "mass_deficit": (diag["mass0"] - diag["mass1"]) / diag["mass0"],
            "deficit_bound": 2.0 * edge_cell_frac + 0.005,
        }

    def test_falling_ramp_preserves_shells(self) -> None:
        r = self._shell_error(NE=800, gap_final=0.9, n_steps=100)
        assert r["advect_err"] < 1e-12          # bare advection: exact
        assert r["bulk_err"] < 2e-3             # measured ~2e-4; 10x margin
        assert r["band_err"] < 0.05             # newly opened shells fill
        # measured 8.3% at NE=800 vs bound ~12.9% (edge-cell scale)
        assert abs(r["mass_deficit"]) < r["deficit_bound"]

    def test_rising_ramp_preserves_shells(self) -> None:
        r = self._shell_error(NE=800, gap_final=1.1, n_steps=100)
        assert r["advect_err"] < 1e-12
        assert r["bulk_err"] < 2e-3
        assert abs(r["mass_deficit"]) < r["deficit_bound"]

    def test_bulk_error_converges_with_resolution(self) -> None:
        coarse = self._shell_error(NE=400, gap_final=0.9, n_steps=50)
        fine = self._shell_error(NE=800, gap_final=0.9, n_steps=100)
        assert fine["bulk_err"] < 0.65 * coarse["bulk_err"]


class TestDiscreteDOSContinuity:
    """The step applied to f ≡ 1 reproduces ∂_tN₁ + ∂_E[(Δ/E)Δ̇N₁] = 0."""

    def _dos_error(self, gap_final: float) -> float:
        NE = 800
        E, dE, dE_u = _grid(NE)
        f_init = np.ones(NE)
        f_fin, _ = _run_ramp(NE, gap_final, 100, f_init, E, dE)
        bulk = _interior_mask(E, dE_u, gap_final, gap_final - GAP0)
        return float(np.max(np.abs(f_fin[bulk] - 1.0)))

    def test_f_one_invariant_under_falling_gap(self) -> None:
        assert self._dos_error(gap_final=0.9) < 2e-3

    def test_f_one_invariant_under_rising_gap(self) -> None:
        assert self._dos_error(gap_final=1.1) < 2e-3

    def test_single_step_advects_N1_onto_new_DOS(self) -> None:
        # One conservative step on u = N₁(Δ_old) lands on N₁(Δ_new) in
        # the interior — the discrete image of eq:dos_continuity.
        NE = 800
        E, dE, dE_u = _grid(NE)
        gap_dot, dt = -2.0, 0.001          # CFL ≈ 0.4 at the grid bottom
        u0 = bcs_density_of_states(E, GAP0)
        u1 = advect_spectral_flow(u0, E, dE, gap=GAP0, gap_dot=gap_dot, dt=dt)
        gap_new = GAP0 + gap_dot * dt
        N1_new = bcs_density_of_states(E, gap_new)
        bulk = _interior_mask(E, dE_u, gap_new, gap_new - GAP0)
        rel = np.abs(u1[bulk] - N1_new[bulk]) / N1_new[bulk]
        assert float(np.max(rel)) < 1e-3
        # and the step is exactly conservative on the full grid
        assert abs(float(np.sum((u1 - u0) * dE))) < 1e-12
