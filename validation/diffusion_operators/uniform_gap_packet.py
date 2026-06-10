"""Benchmark 1: energy-dependent effective diffusivity at a uniform gap.

At a uniform gap each energy diffuses independently with effective rate
``D_eff(E) = D_N N_1(E)^{q-p}`` (see
:mod:`qpsim.transport.diffusion.base`). We seed the fundamental reflective
spatial mode ``cos(pi (j+1/2)/NX)``, take one Crank-Nicolson step, read the
mode's geometric decay factor ``amp = c_1^{n+1}/c_1^n``, and invert the CN
amplification exactly,

    D_eff = (2/dt) (1 - amp)/(1 + amp) / lambda_1 ,

with ``lambda_1 = (2/dx^2)(1 - cos(pi/NX))`` the discrete fundamental
eigenvalue. ``D_eff(E)/D_N`` then traces ``N_1^{q-p}``: falling toward the
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
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel, effective_rate

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


def run(
    *,
    gap: float | None = None,
    NE: int = 40,
    NX: int = 41,
    length_um: float = 100.0,
    D0: float = D0_DEFAULT,
    dt: float = 2.0,
    conservation_steps: int = 30,
) -> PacketResult:
    """Measure ``D_eff(E)/D_N`` per model and the ``n_qp`` drift."""
    if gap is None:
        gap = float(load_material("Al").Delta_0)
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.02, energy_max_factor=4.0, num_energy_bins=NE
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(
        E_bins=E, dE_bins=dE, gap=gap, diffusion_coefficient=D0
    )
    material = load_material("Al")
    x = np.linspace(0.0, length_um, NX)
    dx = float(x[1] - x[0])
    N1 = spectral.rho.copy()

    j = np.arange(NX)
    mode = np.cos(np.pi * (j + 0.5) / NX)
    lambda_1 = (2.0 / dx**2) * (1.0 - np.cos(np.pi / NX))
    rho_dE = spectral.rho * spectral.dE

    backend = T3Spatial1DBackend()
    f_seed = np.tile(0.5 + 0.3 * mode, (NE, 1))

    deff_over_dn: dict[str, np.ndarray] = {}
    analytic_over_dn: dict[str, np.ndarray] = {}
    n_qp_rel_drift: dict[str, float] = {}

    for model in BENCHMARK_MODELS:
        state = _state(f_seed.copy(), x, gap, spectral, material, model)
        c0 = state.f @ mode
        c1 = backend.apply_transport(state, dt).f @ mode
        amp = c1 / c0
        deff = (2.0 / dt) * (1.0 - amp) / (1.0 + amp) / lambda_1
        deff_over_dn[model.name] = deff / D0
        analytic_over_dn[model.name] = effective_rate(D0, N1, model) / D0

        evolving = state
        n_qp0 = float(np.sum(rho_dE[:, None] * evolving.f))
        for _ in range(conservation_steps):
            evolving = backend.apply_transport(evolving, dt)
        n_qp1 = float(np.sum(rho_dE[:, None] * evolving.f))
        n_qp_rel_drift[model.name] = abs(n_qp1 - n_qp0) / abs(n_qp0)

    return PacketResult(
        E=E,
        N1=N1,
        gap=gap,
        deff_over_dn=deff_over_dn,
        analytic_over_dn=analytic_over_dn,
        n_qp_rel_drift=n_qp_rel_drift,
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
    styles = {"A1": dict(ls="-", lw=2.4), "C": dict(ls="--", lw=1.2)}
    for name in result.deff_over_dn:
        ax.plot(
            result.E / result.gap,
            result.analytic_over_dn[name],
            color=colors.get(name),
            label=f"{name} analytic",
            **styles.get(name, dict(ls="-", lw=1.2)),
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
