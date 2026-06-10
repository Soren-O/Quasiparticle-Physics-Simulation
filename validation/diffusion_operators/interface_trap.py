"""Benchmark 3: Kupriyanov-Lukichev two-gap interface trap.

Two superconducting regions of unequal gap are joined at one face by a
finite interface conductance carrying the energy-channel current
``F = G_N (N_1^L N_1^R - N_2^L N_2^R)(f_L - f_R)`` (dx-independent) -- the
coherence-factor (Maki-Griffin) weight -- matched to the bulk flux
``-D_N N_1^q d_x f`` of the selected operator.

* Driven steady state (source on the high-gap end, sink on the low-gap
  end): the spatial current is continuous across the interface, while the
  bare occupation ``f`` is *discontinuous* there, with the jump fixed by
  ``f_L - f_R = F / (G_N [N_1^L N_1^R - N_2^L N_2^R])`` -- a resistive
  interface. Because the steady state depends only on ``q``, the
  equal-``q`` diagnostics A1P and A2 coincide here while A1 and C differ.
* Closed relaxation (inject the high-gap side, reflective ends): A1
  (conserves ``N_1 f``) and A2 (conserves ``N_1^2 f``) relax to *distinct*
  equilibria -- the ``p`` dressing that the driven steady state cannot see.

Run ``python -m validation.diffusion_operators.interface_trap`` to write the
CSV + figure under ``outputs/diffusion_operators/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel, flux_weight

from validation.diffusion_operators import D0_DEFAULT, results_dir, write_csv

_DRIVEN_MODELS = (
    DiffusionModel.A1,
    DiffusionModel.A1P,
    DiffusionModel.A2,
    DiffusionModel.C,
)


@dataclass(frozen=True)
class InterfaceResult:
    """Outcome of the interface-trap benchmark."""

    x: np.ndarray
    gap_profile: np.ndarray
    interface_face: int
    drive_energy: float
    G_N: float
    N1_left: float
    N1_right: float
    driven_profiles: dict[str, np.ndarray]
    interface_jump: dict[str, float]
    bulk_drop: dict[str, float]
    currents: dict[str, tuple[float, float, float]]
    jump_predicted: dict[str, float]
    relax_profiles: dict[str, np.ndarray]
    relax_a1_a2_maxdiff: float


def run(
    *,
    NE: int = 16,
    NX: int = 31,
    length_um: float = 100.0,
    gap_hi_factor: float = 1.6,
    gap_lo_factor: float = 1.0,
    G_N: float = 0.1,
    f_source: float = 0.6,
    dt_driven: float = 1.0,
    dt_relax: float = 2.0,
    relax_steps: int = 3000,
    max_iter: int = 200_000,
    tol: float = 1e-12,
    D0: float = D0_DEFAULT,
) -> InterfaceResult:
    """Run the driven steady state and the closed relaxation."""
    base_gap = float(load_material("Al").Delta_0)
    gap_hi = gap_hi_factor * base_gap
    gap_lo = gap_lo_factor * base_gap
    E, _ = build_energy_grid(
        gap=gap_hi, energy_min_factor=1.05, energy_max_factor=4.0, num_energy_bins=NE
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(
        E_bins=E, dE_bins=dE, gap=gap_hi, diffusion_coefficient=D0
    )
    material = load_material("Al")
    x = np.linspace(0.0, length_um, NX)
    dx = float(x[1] - x[0])

    half = NX // 2
    gap_profile = np.where(np.arange(NX) < half, gap_hi, gap_lo).astype(float)
    interface_face = half - 1  # between cell half-1 (hi) and half (lo)
    ed = NE - 1  # highest energy: both regions active
    drive_energy = float(E[ed])

    backend = T3Spatial1DBackend()
    probe = _make_state(
        np.zeros((NE, NX)), x, gap_hi, spectral, material, DiffusionModel.A1, gap_profile
    )
    n1_full = backend._n1_per_cell(probe)
    n2_full = backend._n2_per_cell(probe)
    n1_cell = n1_full[ed]
    n1_left = float(n1_cell[interface_face])
    n1_right = float(n1_cell[interface_face + 1])
    n2_left = float(n2_full[ed][interface_face])
    n2_right = float(n2_full[ed][interface_face + 1])
    # Kupriyanov-Lukichev energy-channel interface weight (coherence factor).
    w_kl = n1_left * n1_right - n2_left * n2_right

    driven_profiles: dict[str, np.ndarray] = {}
    interface_jump: dict[str, float] = {}
    bulk_drop: dict[str, float] = {}
    currents: dict[str, tuple[float, float, float]] = {}
    jump_predicted: dict[str, float] = {}

    k = interface_face
    for model in _DRIVEN_MODELS:
        state = _make_state(
            np.zeros((NE, NX)), x, gap_hi, spectral, material, model, gap_profile, G_N
        )
        steady = _drive_to_steady(backend, state, ed, f_source, dt_driven, max_iter, tol)
        f = steady.f[ed].copy()
        driven_profiles[model.name] = f

        w_cell = flux_weight(D0, n1_cell, model.q)
        w_face = _harmonic(w_cell)
        f_left = w_face[k - 1] * (f[k - 1] - f[k]) / dx
        f_int = G_N * w_kl * (f[k] - f[k + 1])
        f_right = w_face[k + 1] * (f[k + 1] - f[k + 2]) / dx
        currents[model.name] = (float(f_left), float(f_int), float(f_right))
        interface_jump[model.name] = float(f[k] - f[k + 1])
        bulk_drop[model.name] = float(abs(f[k - 1] - f[k]))
        # K-L cross-check: the observed jump should equal the *bulk* current
        # (computed from bulk f-differences) divided by the interface
        # conductance times the coherence-factor weight --
        # f_L - f_R = F_bulk / (G_N [N_1^L N_1^R - N_2^L N_2^R]).
        jump_predicted[model.name] = float(f_left / (G_N * w_kl))

    relax_profiles: dict[str, np.ndarray] = {}
    for model in (DiffusionModel.A1, DiffusionModel.A2):
        f0 = np.zeros((NE, NX))
        f0[ed, :half] = 0.4  # inject the high-gap side
        state = _make_state(f0, x, gap_hi, spectral, material, model, gap_profile, G_N)
        evolving = state
        for _ in range(relax_steps):
            evolving = backend.apply_transport(evolving, dt_relax)
        relax_profiles[model.name] = evolving.f[ed].copy()
    relax_diff = float(
        np.max(np.abs(relax_profiles["A1"] - relax_profiles["A2"]))
    )

    return InterfaceResult(
        x=x,
        gap_profile=gap_profile,
        interface_face=interface_face,
        drive_energy=drive_energy,
        G_N=G_N,
        N1_left=n1_left,
        N1_right=n1_right,
        driven_profiles=driven_profiles,
        interface_jump=interface_jump,
        bulk_drop=bulk_drop,
        currents=currents,
        jump_predicted=jump_predicted,
        relax_profiles=relax_profiles,
        relax_a1_a2_maxdiff=relax_diff,
    )


def _make_state(
    f: np.ndarray,
    x: np.ndarray,
    gap: float,
    spectral: SpectralContext,
    material: object,
    model: DiffusionModel,
    gap_profile: np.ndarray,
    G_N: float | None = None,
) -> T3Spatial1DState:
    return T3Spatial1DState(
        f=f,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,  # type: ignore[arg-type]
        T_bath=0.1,
        diffusion_model=model,
        gap_profile=gap_profile,
        interface_conductance=G_N,
    )


def _drive_to_steady(
    backend: T3Spatial1DBackend,
    state: T3Spatial1DState,
    ed: int,
    f_source: float,
    dt: float,
    max_iter: int,
    tol: float,
) -> T3Spatial1DState:
    """Pin source (cell 0) / sink (cell -1) at energy ``ed`` until steady."""
    current = state
    for _ in range(max_iter):
        current.f[ed, 0] = f_source
        current.f[ed, -1] = 0.0
        nxt = backend.apply_transport(current, dt)
        nxt.f[ed, 0] = f_source
        nxt.f[ed, -1] = 0.0
        if float(np.max(np.abs(nxt.f[ed] - current.f[ed]))) < tol:
            return nxt
        current = nxt
    return current


def _harmonic(w_cell: np.ndarray) -> np.ndarray:
    w_left = w_cell[:-1]
    w_right = w_cell[1:]
    denom = w_left + w_right
    out = np.zeros(w_cell.size - 1, dtype=float)
    nz = denom > 0.0
    out[nz] = 2.0 * w_left[nz] * w_right[nz] / denom[nz]
    return out


def main() -> None:
    """Write the CSV + figure for the interface-trap benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run()
    out = results_dir()

    header = ["x_um", "gap_profile"]
    for name in result.driven_profiles:
        header.append(f"driven_{name}")
    for name in result.relax_profiles:
        header.append(f"relax_{name}")
    rows = []
    for j, xj in enumerate(result.x):
        row: list[object] = [xj, result.gap_profile[j]]
        for name in result.driven_profiles:
            row.append(result.driven_profiles[name][j])
        for name in result.relax_profiles:
            row.append(result.relax_profiles[name][j])
        rows.append(row)
    write_csv(out / "interface_trap.csv", header, rows)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    xf = result.x[result.interface_face] + 0.5 * (result.x[1] - result.x[0])
    for name, prof in result.driven_profiles.items():
        ax0.plot(result.x, prof, "-o", markersize=2, label=name)
    ax0.axvline(xf, color="k", ls="--", lw=0.7, label="interface")
    ax0.set_xlabel(r"$x$ ($\mu$m)")
    ax0.set_ylabel(r"$f(x)$ at drive energy")
    ax0.set_title("Driven steady state (continuity + f-jump)")
    ax0.legend(fontsize=8)
    for name, prof in result.relax_profiles.items():
        ax1.plot(result.x, prof, "-o", markersize=2, label=name)
    ax1.axvline(xf, color="k", ls="--", lw=0.7)
    ax1.set_xlabel(r"$x$ ($\mu$m)")
    ax1.set_ylabel(r"$f(x)$ at drive energy")
    ax1.set_title("Closed relaxation: A1 vs A2 equilibria")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "interface_trap.pdf")
    plt.close(fig)

    print(f"Wrote {out / 'interface_trap.csv'} and .pdf")
    for name, (fl, fi, fr) in result.currents.items():
        print(
            f"  {name}: jump={result.interface_jump[name]:.4f} "
            f"currents=({fl:.4f},{fi:.4f},{fr:.4f})"
        )
    print(f"  A1 vs A2 closed-equilibria max|df| = {result.relax_a1_a2_maxdiff:.3e}")


if __name__ == "__main__":
    main()
