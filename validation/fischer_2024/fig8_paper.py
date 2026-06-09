"""Fischer & Catelani 2024 Fig. 8 — qpsim-native characterization at paper topology.

This is the **structural** Fig. 8 reproduction: $x_{\\rm qp}$ vs $T_B$
in the low-heating regime at three fixed pair-breaking drive products
$c_{\\rm phot,PB}\\bar n_{\\rm PB}\\in\\{10^{-2},10^{-4},10^{-6}\\}$~Hz.
Solid: kinetic-equation numerics. Dashed: analytical steady-state
density prediction (Eq.~placeholder; see paper-parity gap below).

The companion script :mod:`fig8_xqp_pb` already exists but (a) bakes
the drive products in as ns$^{-1}$ rather than Hz (a 9-decade unit
slip — see audit below), and (b) uses five drive levels rather than
the paper's three. This script enforces the published Hz units and
the paper's three-curve stratification, with naming honesty so a
downstream consumer is not misled until both gaps below close.

Outstanding paper-parity gaps (load-bearing)
--------------------------------------------

1. **Hz ↔ ns$^{-1}$ unit conversion.** The paper quotes drive
   products $c_{\\rm phot,PB}\\bar n_{\\rm PB}$ in Hz; the simulator's
   :mod:`qpsim.collisions.pair_breaking_photon` works in ns$^{-1}$.
   The conversion is pinned at module load via :data:`HZ_TO_NS_INV`
   and asserted at the top of :func:`run`. The companion
   :mod:`fig8_xqp_pb` script silently elides this conversion — its
   curves are at drive products $10^9$× higher than the paper's, so
   the two scripts are **not** comparable.

2. **Analytical density-equation dashed overlay.** The paper plots
   solid (numerical) + dashed (analytic steady-state density). The
   analytic curve is **not** implemented here yet —
   :func:`_xqp_analytic_pb` currently returns the photon-driven
   steady-state estimate $x_{\\rm qp} \\approx \\sqrt{c_{\\rm phot,PB}
   \\bar n_{\\rm PB}\\,\\tau_0\\,U(\\omega_{\\rm PB})}$, which captures
   the $T_B$-independent floor at low $T_B$ but is **not** the paper
   formula. Replace the function body with the verified analytic
   density expression once it has been hand-checked against the
   paper text (likely an algebraic rearrangement of the recombination
   ↔ pair-breaking detailed-balance condition at fixed $\\bar n_{\\rm PB}$).

Once both tickets land, this script becomes the paper-faithful Fig. 8
reproduction. At that point: rename the artifacts to
``fischer2024_fig8_paper.{csv,pdf}``, drop the "qpsim-native" wording
from the plot title, and remove the warning text from the run banner.

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV:

    Δ_0     = 189 μeV
    τ_0     = 63 ns      (faster than F23 — different Al film)
    ω_PB    = 2.8 · Δ_0  (above 2Δ, pair-breaking active)
    τ_ℓ     = 0          (thermal-phonon shortcut)
    n̄_PB    = 1 × 10^6   (factored arbitrarily; only the product matters)
    c·n̄ ∈ {1e-2, 1e-4, 1e-6} Hz   (paper Fig. 8 stratification)

Usage --- generate baseline + PDF::

    python -m validation.fischer_2024.fig8_paper
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

# ── Fischer & Catelani 2024, Sec. IV parameters ──────────────────────

DELTA_0 = 189.0            # μeV
TAU_0 = 63.0               # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_PB = 2.8 * DELTA_0   # 529.2 μeV  (pair-breaking active above 2Δ)
N_BAR_PB = 1e6             # factored arbitrarily; only c·n̄ matters

# Paper grid: 810 bins so ω_PB / dE = 252 is integer-commensurate.
# Same grid as :mod:`fig8_xqp_pb` and :mod:`figs_5_7_fe_pb`.
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810

# ── Hz ↔ ns⁻¹ unit audit ─────────────────────────────────────────────
#
# Paper quotes c_phot_PB · n̄_PB in Hz (cycles per second).
# qpsim's pair_breaking_photon kernel multiplies c_phot_PB into the
# kinetic equation in ns⁻¹.  Conversion is exact:
#
#     1 Hz = 1 cycle / s = 1e-9 cycles / ns = 1e-9 ns⁻¹
#
# Pinned and asserted at the top of run() to prevent the 9-decade
# slip baked into the companion fig8_xqp_pb script.
HZ_TO_NS_INV = 1e-9

# Paper-target drive products (Hz). F24 Fig. 5 caption pins these
# as ``c_phot_PB · n̄_PB ∈ {10⁻², 10⁻⁴, 10⁻⁶} Hz``. The Fig. 8 caption
# only states "fixed numbers of pair-breaking photons (see parameter
# in the inset)" without explicit values, so we use the Fig. 5 set.
PAPER_DRIVES_HZ: tuple[float, ...] = (1e-2, 1e-4, 1e-6)

# T_B sweep: paper Fig. 8 plots T_B/T_c on the x-axis; we sweep T_B in
# kelvin and convert at plot time. Range chosen to cover the paper's
# typical [0.04, 0.20] range of T_B/T_c (≈ 0.05–0.25 K for T_c ≈ 1.244 K).
T_BATH_VALUES: np.ndarray = np.linspace(0.05, 0.30, 12)

# ── ρ_F for absolute-density (left-axis) labelling ───────────────────
#
# Paper Fig. 8 plots N_qp [1/μm³] on the left axis and a dimensionless
# x_qp on the right axis. F24 doesn't redefine ρ_F; we use the F23
# appendix value:
#
#     ρ_F = 1.74 × 10⁴ / (μeV · μm³)  [single-spin DOS at the Fermi level for Al]
#
# qpsim's ``qp_fraction`` returns ``x_qp_qpsim = N_qp / (4 ρ_F Δ)``;
# the paper's ``x_qp`` is ``N_qp / (2 ρ_F Δ)``. So:
#
#     x_qp_paper = 2 · x_qp_qpsim
#     N_qp [1/μm³]   = 4 · ρ_F · Δ_0 · x_qp_qpsim
#
RHO_F_INV_UEV_UM3 = 1.74e4
NQP_PER_X_QP_QPSIM = 4.0 * RHO_F_INV_UEV_UM3 * DELTA_0  # ≈ 1.32e7 [1/μm³]


@dataclass(frozen=True)
class Fig8PaperResult:
    """Arrays returned by :func:`run`."""

    T_bath: np.ndarray                          # shape (NT,) in K
    drives_hz: tuple[float, ...]                # paper drive products in Hz
    drives_ns_inv: tuple[float, ...]            # converted to ns⁻¹
    x_qp_thermal: np.ndarray                    # shape (NT,)
    x_qp_num_by_drive: dict[float, np.ndarray]  # drive (Hz) → (NT,) numerical
    x_qp_ana_by_drive: dict[float, np.ndarray]  # drive (Hz) → (NT,) analytic


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state(material: Material, T_bath: float) -> T3DiffusionState:
    """Build a fresh thermal-seed state at the paper grid + given T_B."""
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
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),  # τ_ℓ = 0 throughout F24 Fig. 8
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _xqp_analytic_pb(drive_ns_inv: float) -> float:
    """Placeholder for the F24 analytical density equation overlay.

    TODO(paper-parity): replace with the paper's analytic steady-state
    density formula. For now this returns

        x_qp ≈ sqrt(c·n̄ · τ_0 · U(ω_PB))

    where ``U(ω_PB) = (ω_PB / Δ_0)`` is a crude Kaplan-style
    coherence-factor stand-in. This captures the $T_B$-independent
    photon-driven floor (horizontal curves in the paper figure) but
    is **not** the paper formula and should not be used as a
    quantitative acceptance test.
    """
    if drive_ns_inv <= 0.0:
        return 0.0
    U_eff = OMEGA_PB / DELTA_0
    return float(np.sqrt(drive_ns_inv * TAU_0 * U_eff))


def _assert_unit_audit() -> None:
    """Sanity-check the Hz → ns⁻¹ conversion before any kernel call.

    Pin the conversion factor (``HZ_TO_NS_INV = 1e-9``) and verify
    that the paper-target drive products land in the expected ns⁻¹
    range when funneled through the conversion. The companion
    :mod:`fig8_xqp_pb` script silently uses ns⁻¹ values that look
    numerically identical to the paper's Hz values, which is a
    9-decade slip; this assertion makes that slip impossible here.
    """
    assert HZ_TO_NS_INV == 1e-9, (
        f"Hz → ns⁻¹ conversion mis-pinned: {HZ_TO_NS_INV} != 1e-9. "
        "Refusing to run a paper-target script with a corrupted unit factor."
    )
    # Independently verify via dimensional analysis:
    # 1 Hz = 1 / s = 1e-9 / ns
    derived = 1.0 / 1e9
    assert abs(HZ_TO_NS_INV - derived) < 1e-30, (
        f"HZ_TO_NS_INV = {HZ_TO_NS_INV} disagrees with dimensional "
        f"derivation 1/1e9 = {derived}."
    )
    # Sanity-check that paper drive products convert into the expected
    # ns⁻¹ window for the PB-photon kernel.
    for d_hz in PAPER_DRIVES_HZ:
        d_ns = d_hz * HZ_TO_NS_INV
        assert 1e-20 < d_ns < 1e-5, (
            f"Drive {d_hz} Hz → {d_ns} ns⁻¹ is outside the expected "
            f"PB-photon kernel range; suspect unit slip."
        )


def run() -> Fig8PaperResult:
    """Solve Fischer 2024 Fig. 8 — sweep T_B at three paper drive products."""
    _assert_unit_audit()

    material = _material()
    backend = T3DiffusionBackend()

    # Convert Hz → ns⁻¹ once, up front, so the kernel only ever sees
    # ns⁻¹ values.
    drives_ns_inv: tuple[float, ...] = tuple(
        d * HZ_TO_NS_INV for d in PAPER_DRIVES_HZ
    )

    # Commensurability probe (all T_B share the same grid).
    probe_state = _build_state(material, float(T_BATH_VALUES[0]))
    dE_scalar = float(probe_state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"ω_PB={OMEGA_PB} not integer-commensurate with dE={dE_scalar:.4f} "
            f"(frac_err={frac_err:.2e}). Choose NUM_BINS so ω_PB/dE is integer."
        )

    nT = T_BATH_VALUES.size
    x_thermal = np.zeros(nT)
    x_num: dict[float, np.ndarray] = {d: np.zeros(nT) for d in PAPER_DRIVES_HZ}
    x_ana: dict[float, np.ndarray] = {d: np.zeros(nT) for d in PAPER_DRIVES_HZ}

    for i, T in enumerate(T_BATH_VALUES):
        state = _build_state(material, float(T))
        # Thermal reference (no PB drive); state.f already = f_FD(T_B).
        x_thermal[i] = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)

        for d_hz, d_ns in zip(PAPER_DRIVES_HZ, drives_ns_inv, strict=True):
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                "c_phot_PB": d_ns / N_BAR_PB,  # factor c out of the c·n̄ product
            }
            driven = backend.steady_state(
                state,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                # 1e-14 trips the Newton line-search at very weak drives
                # where f ≈ f_FD already; 1e-10 is well within tier per
                # NFP §6.4.1 and avoids the no-progress failure mode.
                newton_tol=1e-10,
                newton_max_iter=500,
            )
            x_num[d_hz][i] = qp_fraction(
                driven.f, driven.spectral, delta_0=DELTA_0,
            )
            x_ana[d_hz][i] = _xqp_analytic_pb(d_ns)
            print(
                f"  T_B={T:.3f} K  c·n̄={d_hz:.0e} Hz "
                f"({d_ns:.2e} ns⁻¹)  x_qp(num)={x_num[d_hz][i]:.3e}  "
                f"x_qp(ana)={x_ana[d_hz][i]:.3e}",
                flush=True,
            )

    return Fig8PaperResult(
        T_bath=T_BATH_VALUES.copy(),
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=drives_ns_inv,
        x_qp_thermal=x_thermal,
        x_qp_num_by_drive=x_num,
        x_qp_ana_by_drive=x_ana,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer2024_fig8_qpsim_native.csv`` to flag that the
    analytic dashed overlay is the placeholder
    :func:`_xqp_analytic_pb` estimate, not the paper's analytic
    density formula. Rename to ``fischer2024_fig8_paper.csv`` once the
    Eq.-placeholder ticket lands and this becomes a paper-faithful
    reproduction.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer2024_fig8_qpsim_native.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig8PaperResult, path: Path | None = None) -> Path:
    """Write the T_B grid + thermal + per-drive (num, analytic) curves."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "# Fischer & Catelani 2024 Fig. 8 — qpsim-native characterization at paper topology"
        ])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_PB={OMEGA_PB} "
            f"n_bar_PB={N_BAR_PB} tau_l=0"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
        ])
        drives_hz_csv = ",".join(f"{d:g}" for d in result.drives_hz)
        drives_ns_csv = ",".join(f"{d:g}" for d in result.drives_ns_inv)
        writer.writerow([f"# drives_hz={drives_hz_csv}"])
        writer.writerow([f"# drives_ns_inv={drives_ns_csv}"])
        writer.writerow([f"# hz_to_ns_inv={HZ_TO_NS_INV}"])
        header: list[str] = ["T_bath_K", "x_qp_thermal"]
        for d in result.drives_hz:
            header.append(f"x_qp_num_drive_{d:g}_hz")
            header.append(f"x_qp_ana_drive_{d:g}_hz")
        writer.writerow(header)
        for i in range(result.T_bath.size):
            row = [f"{result.T_bath[i]:.17e}", f"{result.x_qp_thermal[i]:.17e}"]
            for d in result.drives_hz:
                row.append(f"{result.x_qp_num_by_drive[d][i]:.17e}")
                row.append(f"{result.x_qp_ana_by_drive[d][i]:.17e}")
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig8PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig8PaperResult`."""
    if path is None:
        path = baseline_path()
    drives_hz: tuple[float, ...] = ()
    drives_ns_inv: tuple[float, ...] = ()
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# drives_hz"):
                drives_hz = tuple(
                    float(x) for x in first.split("=", 1)[1].split(",")
                )
                continue
            if first.startswith("# drives_ns_inv"):
                drives_ns_inv = tuple(
                    float(x) for x in first.split("=", 1)[1].split(",")
                )
                continue
            if first.startswith("# hz_to_ns_inv"):
                # Audit: pin must agree with module-level constant.
                pinned = float(first.split("=", 1)[1])
                if abs(pinned - HZ_TO_NS_INV) > 1e-30:
                    raise RuntimeError(
                        f"Baseline at {path} pins hz_to_ns_inv={pinned}, "
                        f"but module-level HZ_TO_NS_INV={HZ_TO_NS_INV}. "
                        "Unit-audit drift — refusing to load."
                    )
                continue
            if first.startswith("#") or first == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    if not drives_hz or not drives_ns_inv:
        raise RuntimeError(
            f"Baseline at {path} missing drives_hz / drives_ns_inv metadata."
        )
    if len(drives_hz) != len(drives_ns_inv):
        raise RuntimeError(
            f"Baseline at {path}: drives_hz / drives_ns_inv length mismatch."
        )
    data = np.array(rows, dtype=float)
    # Column layout: T_bath, x_qp_thermal,
    #                (x_qp_num_d0, x_qp_ana_d0,
    #                 x_qp_num_d1, x_qp_ana_d1, ...).
    x_num: dict[float, np.ndarray] = {}
    x_ana: dict[float, np.ndarray] = {}
    for i, d in enumerate(drives_hz):
        x_num[d] = data[:, 2 + 2 * i]
        x_ana[d] = data[:, 2 + 2 * i + 1]
    return Fig8PaperResult(
        T_bath=data[:, 0],
        drives_hz=drives_hz,
        drives_ns_inv=drives_ns_inv,
        x_qp_thermal=data[:, 1],
        x_qp_num_by_drive=x_num,
        x_qp_ana_by_drive=x_ana,
    )


def write_plot(result: Fig8PaperResult, path: Path | None = None) -> Path:
    """Paper-style log-log plot: x_qp vs T_B with three drive curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Paper Fig. 8 axes: x = T_B/T_c, left y = N_qp [1/μm³], right y = x_qp.
    # qpsim's x_qp = N_qp/(4 ρ_F Δ); paper's x_qp = N_qp/(2 ρ_F Δ).
    # Convert at plot time so the right axis matches the paper convention.
    x = result.T_bath / T_C
    nqp_thermal = NQP_PER_X_QP_QPSIM * result.x_qp_thermal

    # Thermal floor (no PB drive).
    ax.loglog(x, nqp_thermal, "k--", lw=1.5, label=r"thermal (no PB drive)")

    # Drive curves: paper uses {green, blue, red} for low → high drive.
    paper_rgb = ["#1f7a3f", "#1f4e9c", "#c83737"]  # low → high in PAPER_DRIVES_HZ ascending
    # PAPER_DRIVES_HZ is ordered high→low (1e-2, 1e-4, 1e-6); reverse so the
    # brightest red sits at the highest drive.
    drives_high_to_low = list(result.drives_hz)
    color_by_drive = {d: paper_rgb[2 - i] for i, d in enumerate(drives_high_to_low)}
    for d_hz in result.drives_hz:
        color = color_by_drive[d_hz]
        nqp_num = NQP_PER_X_QP_QPSIM * result.x_qp_num_by_drive[d_hz]
        nqp_ana = NQP_PER_X_QP_QPSIM * result.x_qp_ana_by_drive[d_hz]
        ax.loglog(
            x, nqp_num, "-", color=color, lw=1.6,
            label=rf"$c\,\bar n_{{PB}} = {d_hz:g}$ Hz",
        )
        ax.loglog(x, nqp_ana, "--", color=color, lw=1.0, alpha=0.85)

    ax.set_xlabel(r"$T_B / T_c$", fontsize=13)
    ax.set_ylabel(r"$N_{\mathrm{qp}}$ $[1/\mu\mathrm{m}^3]$", fontsize=13)

    # Twin right-axis carries the paper-convention dimensionless x_qp.
    # The conversion is N_qp = 2 ρ_F Δ x_qp_paper, so right_y/left_y =
    # 1/(2 ρ_F Δ).
    ax_r = ax.twinx()
    ax_r.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    nqp_per_xqp_paper = 2.0 * RHO_F_INV_UEV_UM3 * DELTA_0  # ≈ 6.58e6
    ax_r.set_ylim(ymin / nqp_per_xqp_paper, ymax / nqp_per_xqp_paper)
    ax_r.set_ylabel(r"$x_{\mathrm{qp}} = N_{\mathrm{qp}} / (2\rho_F\Delta)$",
                    fontsize=12)

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.grid(True, which="major", ls="-", lw=0.5, color="0.85")
    ax.grid(True, which="minor", ls=":", lw=0.5, color="0.92")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95, edgecolor="0.7")

    paper_ratio_note = (
        f"qpsim-native (axes converted to paper convention)\n"
        rf"$\rho_F=1.74\!\times\!10^4 / (\mu eV\!\cdot\!\mu m^3)$, "
        rf"$T_c={T_C:.3f}$ K; dashed = placeholder analytic"
    )
    ax.text(
        0.01, 0.02, paper_ratio_note,
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=7, color="0.45",
    )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Fig. 8 paper-target reproduction ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, ω_PB={OMEGA_PB:.2f} μeV, "
        f"n̄_PB={N_BAR_PB:.0e}, τ_ℓ=0"
    )
    print(f"  Grid: NE={NUM_BINS}")
    print(
        f"  Drive products: {list(PAPER_DRIVES_HZ)} Hz "
        f"(× HZ_TO_NS_INV={HZ_TO_NS_INV} → "
        f"{[f'{d * HZ_TO_NS_INV:.2e}' for d in PAPER_DRIVES_HZ]} ns⁻¹)"
    )
    print(
        f"  T_B sweep: {T_BATH_VALUES.size} points, "
        f"{T_BATH_VALUES[0]:.3f} → {T_BATH_VALUES[-1]:.3f} K"
    )
    print(
        "  ⚠ Analytic dashed overlay is the _xqp_analytic_pb placeholder, "
        "NOT the paper formula.\n"
        "    See module docstring; rename artifacts to *_paper.{csv,pdf} "
        "once the analytic formula lands."
    )
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
