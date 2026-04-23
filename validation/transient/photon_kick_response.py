"""Transient demo: photon-kick response of ``f(E, t)``.

Start at thermal equilibrium at ``T_B = 0.1 K``, turn on a
pair-breaking photon drive at ``t = 0``, and evolve ``f(E)`` under
the collision operator with ``n_ph`` frozen at the bath
Bose-Einstein. The time series shows ``f(E)`` climbing from the
thermal floor up to the driven steady state as pair breaking builds
a non-equilibrium QP population.

Uses the :func:`qpsim.services.transient.run_time_dependent` driver
from Gate 3 — the companion to the nbar-loop and steady-state
services that provides transient capability for the v1 T3 backend.

Parameters (Fischer 2024 Sec. IV pair-breaking configuration):

    Δ₀      = 189 μeV
    τ_0     = 63 ns
    T_B     = 0.1 K
    ω_PB    = 2.8 Δ₀
    c·n̄     = 1e-3 ns⁻¹   (strong drive — visible transient)

Grid: 810 bins so ω_PB/dE = 252 is integer-commensurate. Total
evolution time 120 ns at dt = 0.1 ns captures both the e-ph
relaxation (τ_0 = 63 ns) and the drive ramp-up to its steady state.

Produces a two-panel figure: ``f(E)`` snapshots overlaid on the
driven steady state (upper) and ``x_qp(t)`` trajectory (lower).

Usage::

    python -m validation.transient.photon_kick_response
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.transient import run_time_dependent

# ── Fischer 2024 Sec. IV parameters (pair-breaking photon drive) ────
DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1
OMEGA_PB = 2.8 * DELTA_0  # 529.2 μeV, above 2Δ → pair-breaking
N_BAR_PB = 1e6
C_PHOT_PB = 1e-9  # c·n̄ = 1e-3 ns⁻¹ (strong drive, visible transient)

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810  # ω_PB/dE = 252 integer commensurate

# Transient schedule.
DT = 0.1            # ns
TOTAL_TIME = 120.0  # ns  — resolves the τ_0 = 63 ns timescale
SNAPSHOT_INTERVAL = 6.0  # ns  →  21 snapshots including t=0


@dataclass(frozen=True)
class PhotonKickResult:
    E: np.ndarray
    snapshot_times: np.ndarray
    f_snapshots: np.ndarray     # shape (n_snapshots, NE)
    x_qp_snapshots: np.ndarray  # shape (n_snapshots,)
    f_steady_state: np.ndarray  # shape (NE,)
    x_qp_steady_state: float


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state(material: Material) -> T3DiffusionState:
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_BATH).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_BATH,
    )


def run() -> PhotonKickResult:
    material = _material()
    backend = T3DiffusionBackend()
    state = _build_state(material)

    pb_photon_params = {
        "omega_PB": OMEGA_PB,
        "n_bar_PB": N_BAR_PB,
        "c_phot_PB": C_PHOT_PB,
    }

    transient = run_time_dependent(
        state,
        dt=DT,
        total_time=TOTAL_TIME,
        snapshot_interval=SNAPSHOT_INTERVAL,
        pb_photon_params=pb_photon_params,
        observables={
            "x_qp": lambda s: qp_fraction(s.f, s.spectral, delta_0=DELTA_0),
        },
        backend=backend,
    )

    # Reference steady state (τ_l = 0 Newton solve) for overlay.
    driven_ss = backend.steady_state(
        state,
        use_thermal_phonons=True,
        pb_photon_params=pb_photon_params,
        newton_tol=1e-14,
        newton_max_iter=500,
    )
    x_qp_ss = float(qp_fraction(driven_ss.f, driven_ss.spectral, delta_0=DELTA_0))

    times = np.array([snap.t for snap in transient.snapshots], dtype=float)
    f_arr = np.stack([snap.f for snap in transient.snapshots], axis=0)
    x_qp_arr = np.array(
        [snap.observables["x_qp"] for snap in transient.snapshots], dtype=float,
    )

    return PhotonKickResult(
        E=state.spectral.E,
        snapshot_times=times,
        f_snapshots=f_arr,
        x_qp_snapshots=x_qp_arr,
        f_steady_state=driven_ss.f,
        x_qp_steady_state=x_qp_ss,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "transient" / "photon_kick_response.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: PhotonKickResult, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Photon-kick response — qpsim transient-driver demo"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} "
            f"omega_PB={OMEGA_PB} n_bar_PB={N_BAR_PB} c_phot_PB={C_PHOT_PB}"
        ])
        writer.writerow([
            f"# dt={DT} ns  total_time={TOTAL_TIME} ns  "
            f"snapshot_interval={SNAPSHOT_INTERVAL} ns"
        ])
        writer.writerow([
            f"# x_qp_steady_state={result.x_qp_steady_state:.17e}"
        ])
        header = ["E_uev", "f_steady_state"] + [
            f"f_t={t:g}ns" for t in result.snapshot_times
        ]
        writer.writerow(header)
        for i in range(result.E.size):
            row = [f"{result.E[i]:.17e}", f"{result.f_steady_state[i]:.17e}"]
            row.extend(f"{result.f_snapshots[j, i]:.17e}"
                       for j in range(result.snapshot_times.size))
            writer.writerow(row)
    return path


def write_plot(result: PhotonKickResult, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_f, ax_x) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # Upper panel: f(E) snapshots (plasma colormap from t=0 dark →
    # t=TOTAL_TIME bright) + driven steady-state reference.
    x = result.E / DELTA_0
    n_snaps = result.snapshot_times.size
    cmap = matplotlib.colormaps["plasma"]
    colors = cmap(np.linspace(0.05, 0.85, n_snaps))

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(result.E / kT, 500.0)) + 1.0)
    ax_f.semilogy(x, f_FD, "k:", lw=1.0, alpha=0.7, label=r"$f_\mathrm{FD}(T_B)$")

    plot_idxs = np.linspace(0, n_snaps - 1, min(n_snaps, 8)).astype(int)
    for idx in plot_idxs:
        ax_f.semilogy(
            x, result.f_snapshots[idx], "-", lw=1.2,
            color=colors[idx], alpha=0.85,
            label=rf"$t = {result.snapshot_times[idx]:.0f}$ ns",
        )
    ax_f.semilogy(
        x, result.f_steady_state, "k--", lw=1.8,
        label=r"driven steady state ($t \to \infty$)",
    )

    # Pair-breaking threshold: photons at ω_PB break pairs into QPs
    # at E and ω_PB - E, so QPs accumulate in a band [Δ, ω_PB - Δ].
    ax_f.axvline(
        (OMEGA_PB - DELTA_0) / DELTA_0, color="red", ls="--", lw=0.6,
        alpha=0.5, label=r"$\omega_\mathrm{PB}-\Delta$",
    )

    ax_f.set_xlabel(r"$E / \Delta_0$", fontsize=12)
    ax_f.set_ylabel(r"$f(E, t)$", fontsize=12)
    ax_f.set_xlim(1.0, 5.0)
    # Clip y-range to focus on where the transient actually evolves:
    # floor at the steady-state minimum, ceiling at the steady-state
    # peak × 10. The exponentially-suppressed f_FD tail at high E
    # extends orders of magnitude below this floor but carries no
    # story.
    ss_max = float(result.f_steady_state.max())
    ss_nonzero = result.f_steady_state[result.f_steady_state > 0]
    ss_min = float(ss_nonzero.min()) if ss_nonzero.size > 0 else 1e-12
    ax_f.set_ylim(max(ss_min / 10.0, 1e-14), ss_max * 10.0)
    ax_f.legend(fontsize=8, loc="lower left")
    ax_f.grid(True, which="both", ls=":", alpha=0.3)
    ax_f.set_title(
        "Photon-kick response: $f(E, t)$ climbs toward the driven steady state\n"
        rf"$T_B={T_BATH}$ K, $\omega_\mathrm{{PB}}=2.8\,\Delta_0$, "
        rf"$c\cdot\bar n={C_PHOT_PB * N_BAR_PB:.0e}$ ns$^{{-1}}$, "
        rf"$\tau_0={TAU_0:.0f}$ ns",
        fontsize=10,
    )

    # Lower panel: x_qp(t) — how fast we approach steady state.
    ax_x.plot(result.snapshot_times, result.x_qp_snapshots, "o-", lw=1.5, ms=4,
              color="tab:blue", label=r"$x_\mathrm{qp}(t)$")
    ax_x.axhline(result.x_qp_steady_state, color="k", ls="--", lw=1.0,
                 label=rf"$x_\mathrm{{qp}}^\mathrm{{ss}}={result.x_qp_steady_state:.3e}$")
    ax_x.set_xlabel(r"$t$ [ns]", fontsize=12)
    ax_x.set_ylabel(r"$x_\mathrm{qp}$", fontsize=12)
    ax_x.grid(True, which="both", ls=":", alpha=0.3)
    ax_x.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("qpsim transient-driver demo — PB-photon kick response")
    print(f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, T_B={T_BATH} K")
    print(
        f"  Drive: ω_PB={OMEGA_PB:.2f} μeV, n̄_PB={N_BAR_PB:.0e}, "
        f"c_phot_PB={C_PHOT_PB:.0e}"
    )
    print(f"  dt={DT} ns, total_time={TOTAL_TIME} ns, "
          f"snapshot_interval={SNAPSHOT_INTERVAL} ns")
    result = run()
    print(f"  n_snapshots = {result.snapshot_times.size}")
    print(f"  x_qp: {result.x_qp_snapshots[0]:.3e} (t=0) → "
          f"{result.x_qp_snapshots[-1]:.3e} (t={TOTAL_TIME:.0f} ns)")
    print(f"  x_qp_steady_state = {result.x_qp_steady_state:.3e}")
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
