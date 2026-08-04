"""Benchmark 1: energy-dependent effective diffusivity at a uniform gap.

At a uniform gap each energy diffuses independently with effective rate
``D_eff(E) = D_N N_1(E)^{q-p}`` on fully represented cells (see
:mod:`qpsim.transport.diffusion.base`). We seed the fundamental reflective
spatial mode ``cos(pi (j+1/2)/NX)``, take one caller-visible transport step,
read the mode's geometric decay factor ``amp = c_1^{n+1}/c_1^n``, and invert
the internally subcycled Crank--Nicolson amplification exactly. If the
backend uses ``m`` equal substeps, the per-substep factor is
``r = amp**(1/m)`` and

    D_eff = (2m/dt) (1 - r)/(1 + r) / lambda_1 ,

with ``lambda_1 = (2/dx^2)(1 - cos(pi/NX))`` the discrete fundamental
eigenvalue. Here ``N_1`` is the same finite-volume cell-average DOS used by
the backend, not the point-sampled DOS at the cell center.
``D_eff(E)/D_N`` then traces ``N_1^{q-p}``: falling toward the
gap edge for A1 and C (which share the uniform-gap rate ``D_N/N_1``),
rising for the transverse-dressed diagnostic A1P, flat for A2, steeply
falling for B. ``n_qp`` is conserved to round-off for every model.

Run ``python -m validation.diffusion_operators.uniform_gap_packet`` to write
the CSV + figure under ``outputs/diffusion_operators/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.bcs_quadrature import cell_edges_from_widths
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import (
    DiffusionModel,
    density_weight,
    flux_weight,
)

from validation.diffusion_operators import (
    BENCHMARK_MODELS,
    D0_DEFAULT,
    results_dir,
    write_csv,
)


@dataclass(frozen=True)
class PacketResult:
    """Outcome of the uniform-gap packet benchmark."""

    E: np.ndarray
    N1: np.ndarray
    gap: float
    deff_over_dn: dict[str, np.ndarray]
    analytic_over_dn: dict[str, np.ndarray]
    n_qp_rel_drift: dict[str, float]
    transport_substeps: dict[str, np.ndarray]


def run(
    *,
    gap: float | None = None,
    NE: int = 40,
    NX: int = 41,
    length_um: float = 100.0,
    D0: float = D0_DEFAULT,
    dt: float = 2.0,
    conservation_steps: int = 30,
    energy_min_factor: float = 1.02,
) -> PacketResult:
    """Measure ``D_eff(E)/D_N`` per model and the ``n_qp`` drift.

    ``energy_min_factor`` places the grid floor at ``energy_min_factor·Δ``.
    The default ``1.02`` leaves every cell fully above the gap. A floor below
    one makes the first cell gap-cut, which is the only configuration that
    separates the ``q = 0`` dirty-limit face weight ``D_N·support_fraction``
    from the uncut ``D_N``; keep it high enough that the cut cell still holds
    a represented (positive-capacity) sliver.
    """
    if gap is None:
        gap = float(load_material("Al").Delta_0)
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=4.0,
        num_energy_bins=NE,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(
        E_bins=E, dE_bins=dE, gap=gap, diffusion_coefficient=D0
    )
    material = load_material("Al")
    x = np.linspace(0.0, length_um, NX)
    dx = float(x[1] - x[0])
    # Transport is finite-volume: its capacity is the exact represented BCS
    # cell measure divided by dE. Comparing against point-sampled ``rho``
    # spuriously reports an O(dE) diffusivity error near the gap edge.
    N1 = spectral.cell_density.copy()
    edges = cell_edges_from_widths(E, dE)
    support_fraction = (
        np.maximum(edges[1:] - np.maximum(edges[:-1], gap), 0.0) / dE
    )

    j = np.arange(NX)
    mode = np.cos(np.pi * (j + 0.5) / NX)
    lambda_1 = (2.0 / dx**2) * (1.0 - np.cos(np.pi / NX))
    cell_weights = spectral.cell_weights

    backend = T3Spatial1DBackend()
    f_seed = np.tile(0.5 + 0.3 * mode, (NE, 1))

    deff_over_dn: dict[str, np.ndarray] = {}
    analytic_over_dn: dict[str, np.ndarray] = {}
    n_qp_rel_drift: dict[str, float] = {}
    transport_substeps: dict[str, np.ndarray] = {}

    for model in BENCHMARK_MODELS:
        state = _state(f_seed.copy(), x, gap, spectral, material, model)
        operators = backend._build_transport_operators(state, dt)
        substeps = np.asarray(
            [1 if op is None else op[4] for op in operators], dtype=np.int64
        )
        c0 = state.f @ mode
        c1 = backend.apply_transport(state, dt).f @ mode
        amp = c1 / c0
        if np.any(~np.isfinite(amp)) or np.any(amp <= 0.0):
            raise RuntimeError(
                "Uniform-gap CN benchmark requires a finite, positive "
                "fundamental-mode amplification."
            )
        # Recover the amplification of one internal step.  expm1 keeps the
        # inversion accurate when weak diffusion makes r extremely close to
        # one; forming ``1 - amp**(1/m)`` directly loses significant digits.
        log_r = np.log(amp) / substeps
        r_minus_one = np.expm1(log_r)
        cn_ratio = -r_minus_one / (2.0 + r_minus_one)
        deff = (2.0 * substeps / dt) * cn_ratio / lambda_1
        deff_over_dn[model.name] = deff / D0
        capacity = density_weight(N1, model.p)
        face_weight = flux_weight(D0, N1, model.q)
        if model.q == 0:
            # Dirty-limit D_L is the above-gap indicator. Its exact average
            # in a gap-cut cell is the represented support fraction rather
            # than one merely because the cell has positive capacity.
            face_weight = D0 * support_fraction
        analytic_over_dn[model.name] = face_weight / capacity / D0
        transport_substeps[model.name] = substeps

        evolving = state
        n_qp0 = float(np.sum(cell_weights[:, None] * evolving.f))
        for _ in range(conservation_steps):
            evolving = backend.apply_transport(evolving, dt)
        n_qp1 = float(np.sum(cell_weights[:, None] * evolving.f))
        n_qp_rel_drift[model.name] = abs(n_qp1 - n_qp0) / abs(n_qp0)

    return PacketResult(
        E=E,
        N1=N1,
        gap=gap,
        deff_over_dn=deff_over_dn,
        analytic_over_dn=analytic_over_dn,
        n_qp_rel_drift=n_qp_rel_drift,
        transport_substeps=transport_substeps,
    )


def _state(
    f: np.ndarray,
    x: np.ndarray,
    gap: float,
    spectral: SpectralContext,
    material: object,
    model: DiffusionModel,
) -> T3Spatial1DState:
    return T3Spatial1DState(
        f=f,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,  # type: ignore[arg-type]
        T_bath=0.1,
        diffusion_model=model,
    )


def main() -> None:
    """Write the CSV + figure for the uniform-gap packet benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run()
    out = results_dir()

    header = ["E_over_gap", "N1"]
    for name in result.deff_over_dn:
        header += [f"{name}_measured", f"{name}_analytic"]
    rows = []
    for i, e in enumerate(result.E):
        row: list[object] = [e / result.gap, result.N1[i]]
        for name in result.deff_over_dn:
            row += [result.deff_over_dn[name][i], result.analytic_over_dn[name][i]]
        rows.append(row)
    write_csv(out / "uniform_gap_packet.csv", header, rows)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    colors = {"A1": "C0", "A1P": "C4", "A2": "C1", "C": "C2", "B": "C3"}
    # A1 and C coincide exactly at uniform gap: draw A1 wide and C dashed so
    # the coincidence is visible rather than an overplot.
    styles = {"A1": {"ls": "-", "lw": 2.4}, "C": {"ls": "--", "lw": 1.2}}
    for name in result.deff_over_dn:
        ax.plot(
            result.E / result.gap,
            result.analytic_over_dn[name],
            color=colors.get(name),
            label=f"{name} analytic",
            **styles.get(name, {"ls": "-", "lw": 1.2}),
        )
        ax.plot(
            result.E / result.gap,
            result.deff_over_dn[name],
            "o",
            color=colors.get(name),
            markersize=3,
        )
    ax.set_xlabel(r"$E / \Delta$")
    ax.set_ylabel(r"$D_{\rm eff}(E) / D_N$")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "uniform_gap_packet.pdf")
    plt.close(fig)

    worst = max(result.n_qp_rel_drift.values())
    print(f"Wrote {out / 'uniform_gap_packet.csv'} and .pdf")
    print(f"  max n_qp relative drift across models: {worst:.2e}")


if __name__ == "__main__":
    main()
