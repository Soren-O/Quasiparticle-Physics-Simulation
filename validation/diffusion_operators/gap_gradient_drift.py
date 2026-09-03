"""Benchmark 2: DOS-gradient drift in a fixed gap ramp.

With a spatially-varying gap the conserved density ``u = N_1^p f`` obeys
``d_t u = d_x(D_eff d_x u) - d_x(v_d u)`` with ``D_eff = D_N N_1^{q-p}`` and
``v_d = D_N p N_1^{q-p-1} d_x N_1``. For a narrow packet the first moment
of a readout weight ``w = N_1^s f`` then drifts at

    v_com = D_N [q + 2(s - p)] N_1^{q-p-1} d_x N_1 .

With ``s = p`` this is each operator's own conserved-density moment,
``D_N q N_1^{q-p-1} d_x N_1``, a structural diagnostic of ``q != 0``. The
benchmark instead reads every operator on the *same* physical quantity,
the quasiparticle density ``N_1 f`` (``s = 1``, whose energy integral is
``n_qp``), so the prefactor is ``q + 2(1 - p)``: the dirty-limit operator
A1 (1, 0) has *no* DOS-gradient drift, the legacy placement C (0, -1)
drifts *up* the gap gradient at ``D_N N_1^{-2} d_x N_1`` (the same
magnitude as, and the opposite sign to, its own-moment drift), the A1P
diagnostic (1, 2) drifts up at ``2 D_N d_x N_1``, and A2 (2, 2) and
B (0, -2) carry no net drift of ``N_1 f``. We launch a narrow packet at
every energy in a fixed gap ramp (no collisions) and read each energy's
center-of-mass drift. The analytic curves are the exact initial rate of
the ``N_1 f`` centre of mass (:func:`exact_initial_drift`), which on the
linear ramp agrees with the narrow-packet form to about 2%.

Run ``python -m validation.diffusion_operators.gap_gradient_drift`` to write
the CSV + figure under ``outputs/diffusion_operators/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext, bcs_density_of_states

from validation.diffusion_operators import (
    BENCHMARK_MODELS,
    D0_DEFAULT,
    READOUT_WEIGHT,
    exact_initial_drift,
    results_dir,
    write_csv,
)


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
    N1 = np.column_stack([bcs_density_of_states(E, float(g)) for g in ramp])  # (NE, NX)

    center = NX // 2
    x0 = float(x[center])
    packet = np.exp(-((x - x0) / packet_sigma_um) ** 2)
    f_seed = np.tile(0.3 * packet, (NE, 1))

    N1_center = N1[:, center]

    backend = T3Spatial1DBackend()
    drift_measured: dict[str, np.ndarray] = {}
    drift_analytic: dict[str, np.ndarray] = {}

    for model in BENCHMARK_MODELS:
        p, q = model.p, model.q
        state = T3Spatial1DState(
            f=f_seed.copy(),
            x=x,
            gap=gap_max,
            spectral=spectral,
            material=material,
            T_bath=0.1,
            diffusion_model=model,
            gap_profile=ramp,
        )
        com0 = _center_of_mass(state.f, N1, x)
        evolving = state
        for _ in range(n_steps):
            evolving = backend.apply_transport(evolving, dt)
        com1 = _center_of_mass(evolving.f, N1, x)
        drift_measured[model.name] = (com1 - com0) / (n_steps * dt)
        # Exact initial rate of the N_1 f centre of mass on the ramp
        # (narrow-packet form: D_N [q + 2(1 - p)] N_1^{q-p-1} d_x N_1).
        drift_analytic[model.name] = exact_initial_drift(
            f_seed, N1, x, p, q, D0=D0
        )

    return DriftResult(
        E=E,
        N1_at_packet=N1_center,
        gap_max=gap_max,
        drift_measured=drift_measured,
        drift_analytic=drift_analytic,
        center_index=center,
    )


def _center_of_mass(
    f: np.ndarray, N1: np.ndarray, x: np.ndarray, s: int = READOUT_WEIGHT
) -> np.ndarray:
    """Per-energy first moment of the readout density ``w = N_1^s f``.

    The default ``s = READOUT_WEIGHT = 1`` is the quasiparticle density,
    the same physical quantity for every operator.
    """
    w = np.power(N1, s) * f
    weight = np.sum(w, axis=1)
    moment = np.sum(x[None, :] * w, axis=1)
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
