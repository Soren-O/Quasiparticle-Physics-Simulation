"""Measure what the area split changes, WITHOUT changing any answer.

Builds the phonon source/sink two ways on the same physical state:

  today   whole-cell deposit onto the union lattice (bincount at one label),
          with the corr(omega) rescale applied to the pair channel
  split   two-bin area deposit onto the unified finite-volume lattice,
          no rescale -- pi/4 is supposed to fall out of the geometry

The two live on DIFFERENT frequency lattices, so a bin-by-bin diff is
meaningless. Compare physical quantities instead: total event rate, the first
and second frequency moments, and the threshold behaviour that is the whole
point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpsim.collisions.omega_lattice import build_unified_omega_lattice
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext

GAP = 180.0
TAU0_PB = 0.28


def context(num_bins: int, max_factor: float = 4.0) -> SpectralContext:
    E, _ = build_energy_grid(
        gap=GAP,
        energy_min_factor=1.0,
        energy_max_factor=max_factor,
        num_energy_bins=num_bins,
    )
    return SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=GAP
    )


def occupation(ctx: SpectralContext, kind: str) -> np.ndarray:
    """Three profiles, because a defect can hide in any one of them."""
    x = (ctx.E - GAP) / GAP
    if kind == "flat":
        return np.full(ctx.E.shape, 1e-4)
    if kind == "thermal":
        return 1e-3 * np.exp(-ctx.E / (0.2 * GAP))
    if kind == "steep":
        return 1e-3 * np.exp(-x / 0.05)
    raise ValueError(kind)


def today(ctx: SpectralContext, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(ctx.E)
    K_s = build_scattering_kernel_phonon_side(ctx, TAU0_PB)
    K_r = build_recombination_kernel_phonon_side(ctx, TAU0_PB)
    a, b = compute_phonon_source_sink(
        f, ctx, None, None, idx_diff, idx_sum, sign, omega.size,
        K_s0_phonon_side=K_s, K_r0_phonon_side=K_r,
    )
    return omega, a


def with_split(ctx: SpectralContext, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same assembly, deposited by area onto the unified lattice."""
    lat = build_unified_omega_lattice(E_bins=ctx.E, gap=ctx.gap)
    K_s = build_scattering_kernel_phonon_side(ctx, TAU0_PB)
    K_r = build_recombination_kernel_phonon_side(ctx, TAU0_PB)

    rho, dE = ctx.cell_density, ctx.dE
    n_qp = rho * f
    one_minus_f = np.maximum(1.0 - f, 0.0)

    a = np.zeros(lat.n_omega)
    base_sc = dE * (n_qp[:, None] * K_s * (rho[None, :] * one_minus_f[None, :]))
    emit = np.where(lat.diff_sign > 0, base_sc, 0.0)
    a += lat.deposit(emit, channel="scatter")

    base_rec = dE * (n_qp[:, None] * K_r * n_qp[None, :])
    a += lat.deposit(base_rec, channel="pair")
    return lat.omega_bins, a


def moments(omega: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
    total = float(a.sum())
    m1 = float((omega * a).sum())
    m2 = float((omega**2 * a).sum())
    return total, m1, m2


def pair_only(ctx: SpectralContext, f: np.ndarray, *, split: bool):
    """The PAIR channel alone -- where 4*Delta vs pi*Delta lives."""
    K_r = build_recombination_kernel_phonon_side(ctx, TAU0_PB)
    rho, dE = ctx.cell_density, ctx.dE
    n_qp = rho * f
    base = dE * (n_qp[:, None] * K_r * n_qp[None, :])
    if split:
        lat = build_unified_omega_lattice(E_bins=ctx.E, gap=ctx.gap)
        return lat.omega_bins, lat.deposit(base, channel="pair")
    omega, _, idx_sum, _ = build_phonon_frequency_map(ctx.E)
    return omega, np.bincount(idx_sum.ravel(), weights=base.ravel(),
                              minlength=omega.size)


def threshold_table() -> None:
    """Cumulative pair source below a FIXED physical cutoff above 2*Delta.

    Lattice-independent by construction, so the two schemes are comparable
    even though their bins differ. The window is held fixed in physical units
    while the mesh refines -- that is what separates a representation error
    (ratio stays put) from a resolution error (ratio goes to 1).
    """
    print()
    print("PAIR SOURCE within a fixed window above threshold")
    print(f"{'profile':>8} {'window':>10} {'NE':>6} "
          f"{'today':>13} {'split':>13} {'today/split':>12}")
    print("-" * 68)
    # The window is m*h, an exact multiple of the SHARED spacing. Both
    # lattices are aligned to that grid (today's pair nodes at 2D + m*h, the
    # unified faces at multiples of h), so a window of m*h admits exactly m
    # nodes from one and m cells from the other. A window fixed in absolute
    # units instead admits a different count from each, and the ratio then
    # measures the bin edges rather than the physics.
    for kind in ("thermal", "steep"):
        for m in (2, 4, 8):
            for nb in (45, 90, 180, 360, 720):
                ctx = context(nb)
                h = float(ctx.E[1] - ctx.E[0])
                f = occupation(ctx, kind)
                w_o, a_o = pair_only(ctx, f, split=False)
                w_n, a_n = pair_only(ctx, f, split=True)
                cut = 2.0 * GAP + m * h + 1e-9 * h
                o = float(a_o[w_o <= cut].sum())
                n = float(a_n[w_n <= cut].sum())
                ratio = o / n if n else float("nan")
                print(f"{kind:>8} {f'{m}h':>9} {nb:>6} "
                      f"{o:>13.5e} {n:>13.5e} {ratio:>12.4f}")
            print()


def main() -> None:
    print(f"{'profile':>8} {'NE':>6} {'quantity':>10} "
          f"{'today':>14} {'split':>14} {'rel. change':>12}")
    print("-" * 72)
    for kind in ("flat", "thermal", "steep"):
        # Over [Delta, 4*Delta], 2*Delta/h = 2*NE/3, so NE must be a multiple
        # of 3 for either lattice to be well posed at all.
        for nb in (45, 90, 180, 360):
            ctx = context(nb)
            f = occupation(ctx, kind)
            w_old, a_old = today(ctx, f)
            w_new, a_new = with_split(ctx, f)
            mo, mn = moments(w_old, a_old), moments(w_new, a_new)
            for label, o, n in zip(("total", "1st moment", "2nd moment"), mo, mn):
                rel = (n - o) / o if o != 0.0 else float("nan")
                print(f"{kind:>8} {nb:>6} {label:>10} "
                      f"{o:>14.6e} {n:>14.6e} {rel:>+11.3%}")
            print()


if __name__ == "__main__":
    main()
    threshold_table()
