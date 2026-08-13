"""Benchmark 2: DOS-gradient drift in a fixed gap ramp.

With a spatially-varying gap the conserved density ``u = N_1^p f`` obeys
``d_t u = d_x(D_eff d_x u) - d_x(v_d u)`` with ``D_eff = D_N N_1^{q-p}`` and
``v_d = D_N p N_1^{q-p-1} d_x N_1``. The first moment of ``u`` then drifts at

    v_com = <d_x D_eff> + <v_d> = D_N q N_1^{q-p-1} d_x N_1 ,

so the drift is controlled by ``q``: the dirty-limit operator A1
(``q = 0``) has *no* DOS-gradient drift, the diagnostics A1P/A2
(``q = 2``) drift *up* the gap gradient (differing by one power of
``N_1``, ratio ``1/N_1``), and C/B (``q < 0``) drift *down* it. We launch
a narrow packet at every energy in a
fixed gap ramp (no collisions) and read each energy's center-of-mass drift.

Oracle measure (corrected 2026-08-11)
-------------------------------------
Both the tracked moment and the analytic velocity are built from the
represented cell-average measure ``bcs_dos_cell_weights(E, dE, g) / dE``,
which is what ``T3SpatialBackend`` conserves. The helper is imported from
the sibling benchmark rather than re-derived so the two cannot diverge.

This completes the July-2026 validation-oracle correction recorded in
``docs/Diffusion_Operators.md``, which repaired the other three benchmarks
and missed this one. Reading the state in the *point-sampled* DOS instead
defines a different conserved density: along the ramp the two weights
differ by a spatially varying factor (bin 0 ratio runs 0.9969 -> 0.9334
across the ramp at ``NE = 12``, 0.9996 -> 0.9780 at ``NE = 40``, so the
artifact is mesh-convergent rather than a fixed offset), which manufactured
center-of-mass motion for the undressed ``q = 0`` flux. The A1 residual it
reported was that weighting artifact, not transport: -2.6e-3 um/ns at
``NE = 12`` and -4.9e-4 um/ns at the published ``NE = 40``, against
~1.3e-8 um/ns -- Crank-Nicolson round-off -- in the correct measure.

Because the artifact was negative while the A1P drift is positive, the old
*relative* A1 gate also admitted a genuine positive leak of up to
+6.2e-3 um/ns on a quantity whose true value is ~1e-8. The gate is now
absolute; see ``test_gap_gradient_drift.py``.

Run ``python -m validation.diffusion_operators.gap_gradient_drift`` to write
the CSV + figure under ``outputs/diffusion_operators/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext

from validation.diffusion_operators import (
    BENCHMARK_MODELS,
    D0_DEFAULT,
    results_dir,
    write_csv,
)
from validation.diffusion_operators.self_consistent_feedback import _n1_columns


@dataclass(frozen=True)
class DriftResult:
    """Outcome of the gap-gradient drift benchmark."""

    E: np.ndarray
    N1_at_packet: np.ndarray
    gap_max: float
    drift_measured: dict[str, np.ndarray]
    drift_analytic: dict[str, np.ndarray]
    center_index: int


def run(
    *,
    NE: int = 40,
    NX: int = 41,
    length_um: float = 100.0,
    gap_lo_factor: float = 1.0,
    gap_hi_factor: float = 1.6,
    packet_sigma_um: float = 8.0,
    D0: float = D0_DEFAULT,
    dt: float = 0.5,
    n_steps: int = 8,
) -> DriftResult:
    """Measure per-energy center-of-mass drift in a gap ramp, all models."""
    base_gap = float(load_material("Al").Delta_0)
    gap_max = gap_hi_factor * base_gap
    E, _ = build_energy_grid(
        gap=gap_max, energy_min_factor=1.05, energy_max_factor=4.0, num_energy_bins=NE
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(
        E_bins=E, dE_bins=dE, gap=gap_max, diffusion_coefficient=D0
    )
    material = load_material("Al")
    x = np.linspace(0.0, length_um, NX)

    ramp = np.linspace(gap_lo_factor * base_gap, gap_max, NX)
    # The represented cell-average BCS measure that T3SpatialBackend actually
    # conserves. Shared with benchmark 4 rather than re-derived, so the two
    # benchmarks cannot drift into different measures. Both the tracked moment
    # and the analytic velocity below are built from this one array: weighting
    # the centre of mass by the cell average while leaving the analytic
    # velocity on the point DOS is a mixed-measure comparison, and it degrades
    # the A1P agreement from 0.025 to 0.104.
    N1 = _n1_columns(E, dE, ramp)  # (NE, NX)

    center = NX // 2
    x0 = float(x[center])
    packet = np.exp(-((x - x0) / packet_sigma_um) ** 2)
    f_seed = np.tile(0.3 * packet, (NE, 1))

    # Analytic COM drift v_com = D_N q N_1^{q-p-1} d_x N_1 at the packet centre.
    dN1_dx = np.gradient(N1, x, axis=1)[:, center]
    N1_center = N1[:, center]

    backend = T3SpatialBackend()
    drift_measured: dict[str, np.ndarray] = {}
    drift_analytic: dict[str, np.ndarray] = {}

    for model in BENCHMARK_MODELS:
        p, q = model.p, model.q
        state = T3SpatialState(
            f=f_seed.copy(),
            geometry=strip(
                len(x),
                mesh_size=float(x[1] - x[0]) if len(x) > 1 else 1.0,
            ),
            spectral=spectral,
            material=material,
            T_bath=0.1,
            diffusion_model=model,
            gap_per_cell=ramp,
        )
        com0 = _center_of_mass(state.f, N1, p, x)
        evolving = state
        for _ in range(n_steps):
            evolving = backend.apply_transport(evolving, dt)
        com1 = _center_of_mass(evolving.f, N1, p, x)
        drift_measured[model.name] = (com1 - com0) / (n_steps * dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = D0 * q * np.power(N1_center, q - p - 1) * dN1_dx
        drift_analytic[model.name] = v

    return DriftResult(
        E=E,
        N1_at_packet=N1_center,
        gap_max=gap_max,
        drift_measured=drift_measured,
        drift_analytic=drift_analytic,
        center_index=center,
    )


def _center_of_mass(
    f: np.ndarray, N1: np.ndarray, p: int, x: np.ndarray
) -> np.ndarray:
    """Per-energy first moment of ``u = N_1^p f`` for the supplied ``N_1``.

    ``run`` supplies the point-sampled BCS DOS, which is *not* the backend's
    conserved cell-average measure -- see the module docstring's oracle note.
    """
    u = np.power(N1, p) * f
    weight = np.sum(u, axis=1)
    moment = np.sum(x[None, :] * u, axis=1)
    return moment / weight


def main() -> None:
    """Write the CSV + figure for the gap-gradient drift benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run()
    out = results_dir()

    header = ["E_over_gapmax", "N1_at_packet"]
    for name in result.drift_measured:
        header += [f"{name}_measured", f"{name}_analytic"]
    rows = []
    for i, e in enumerate(result.E):
        row: list[object] = [e / result.gap_max, result.N1_at_packet[i]]
        for name in result.drift_measured:
            row += [result.drift_measured[name][i], result.drift_analytic[name][i]]
        rows.append(row)
    write_csv(out / "gap_gradient_drift.csv", header, rows)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = {"A1": "C0", "A1P": "C4", "A2": "C1", "C": "C2", "B": "C3"}
    for name in result.drift_measured:
        ax.plot(
            result.E / result.gap_max,
            result.drift_analytic[name],
            "-",
            color=colors.get(name),
            label=f"{name} analytic",
        )
        ax.plot(
            result.E / result.gap_max,
            result.drift_measured[name],
            "o",
            color=colors.get(name),
            markersize=3,
        )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$E / \Delta_{\max}$")
    ax.set_ylabel(r"$v_{\rm com}$  ($\mu$m/ns)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gap_gradient_drift.pdf")
    plt.close(fig)
    print(f"Wrote {out / 'gap_gradient_drift.csv'} and .pdf")


if __name__ == "__main__":
    main()
