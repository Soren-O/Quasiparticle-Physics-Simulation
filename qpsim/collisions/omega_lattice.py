"""One finite-volume frequency lattice for both phonon channels.

THE UNION-GRID QUESTION, RESOLVED. ``phonon.py``'s frequency map is the union
of a difference lattice ``k·h`` (scattering) and a sum lattice
``2Δ + (m+1)·h`` (pair), and its own docstring flags the two as living on
disjoint, dynamically decoupled sublattices whenever ``2Δ/h`` is not an
integer — a scattering-emitted phonon above 2Δ then cannot break a pair. It
calls the proper fix a physics change (decision D3) rather than a patch.

The fix is smaller than that framing suggests, because both channels want the
SAME thing once each is read as a finite volume rather than a point:

    a cell (i, j) does not have A frequency, it has a RANGE of one, width 2h

        pair         ω = E + E'   spans  [2Δ + (i+j)h,  2Δ + (i+j+2)h]
        scattering   ω = E − E'   spans  [(i−j−1)h,     (i−j+1)h]

The pair support needs bin faces at ``2Δ + m·h``; the scattering support needs
them at ``k·h``. **They coincide exactly when 2Δ/h is an integer** — which is
the commensurability condition the grid validator already enforces and every
shipped default already satisfies (405 bins over [Δ,10Δ] gives 90; 1620 gives
360; 66 over [Δ,4Δ] gives 44). So D3 resolves to:

    ONE uniform lattice of spacing h with faces at multiples of h,
    available precisely on the grids the engine already requires.

No new restriction, and the condition that was imposed to make the two channels
SHARE bins turns out to be the same one that lets them be integrated properly.

THE STRUCTURE IS THE SAME IN BOTH CHANNELS. In the state-counting coordinate
``ξ = √(E² − Δ²)`` the BCS measure is flat, so a cell is a rectangle of uniform
measure and each deposit is an area ratio. In both channels the level curve of
the interior bin face passes through OPPOSITE CORNERS of that rectangle — at
ξ = ξ_i it takes the value ξ_{j+1} and at ξ = ξ_{i+1} it takes ξ_j (pair), or
the mirror of that (scattering) — so no clamping is ever active, the integrand
is smooth and monotone, and the split is well conditioned.

ADJOINTNESS IS THE WHOLE RISK, and it is why the split weights are returned
rather than applied. The phonon equation DEPOSITS event (i, j) across bins
(m, m+1) with weights (S, 1−S). The quasiparticle equation READS a phonon
occupation for the same event. If the read takes a single bin while the deposit
spreads over two, the two discrete operators stop being adjoint and detailed
balance breaks — not subtly, at the 1e-2 level. Reading with the SAME weights,

    N_eff(i, j) = S·n[m] + (1 − S)·n[m+1]

restores it exactly, because deposit and read are then transposes of one
matrix. `effective_occupation` below is that read, and it is not optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpsim.collisions.pair_split import PairSplitUnavailable, pair_split_fractions

__all__ = ["UnifiedOmegaLattice", "build_unified_omega_lattice", "effective_occupation"]

_INT_TOL = 1e-9


@dataclass(frozen=True)
class UnifiedOmegaLattice:
    """A shared finite-volume ω grid plus each channel's two-bin deposit.

    ``omega_bins`` are cell CENTRES at ``(k + 1/2)·h``; bin ``k`` covers
    ``[k·h, (k+1)·h]``. Both index arrays name the LOWER bin of the two a cell
    straddles, and the matching split array gives the fraction belonging to it.
    """

    omega_bins: np.ndarray          # (K,) centres
    h: float                        # bin width, the energy spacing
    pair_lower: np.ndarray          # (NE, NE) int, lower bin of the pair pair
    pair_split: np.ndarray          # (NE, NE) fraction in `pair_lower`
    scatter_lower: np.ndarray       # (NE, NE) int, lower bin for |E - E'|
    scatter_split: np.ndarray       # (NE, NE) fraction in `scatter_lower`
    diff_sign: np.ndarray           # (NE, NE) int8, sign(E_i - E_j)

    @property
    def n_omega(self) -> int:
        return int(self.omega_bins.size)

    def deposit(
        self, weights: np.ndarray, *, channel: str,
    ) -> np.ndarray:
        """Spread per-cell weights onto the lattice, conserving their sum.

        Exact by construction: the two fractions are a partition of unity, so
        the deposited total equals the input total to the last bit whatever the
        quadrature error in the split itself.
        """
        lower, split = self._channel(channel)
        weights = np.asarray(weights, dtype=float)
        size = self.n_omega
        out = np.bincount(
            lower.ravel(), weights=(weights * split).ravel(), minlength=size,
        )
        out += np.bincount(
            (lower + 1).ravel(),
            weights=(weights * (1.0 - split)).ravel(),
            minlength=size,
        )
        return out[:size]

    def _channel(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        if channel == "pair":
            return self.pair_lower, self.pair_split
        if channel == "scatter":
            return self.scatter_lower, self.scatter_split
        raise ValueError(f"channel must be 'pair' or 'scatter'; got {channel!r}.")


def effective_occupation(
    lattice: UnifiedOmegaLattice, n_omega: np.ndarray, *, channel: str,
) -> np.ndarray:
    """The phonon occupation an event sees, read with the DEPOSIT's weights.

    The adjoint of :meth:`UnifiedOmegaLattice.deposit`. Using it is what keeps
    detailed balance exact: deposit and read are then the two directions of one
    matrix, so an equilibrium that is a fixed point of the continuum problem
    stays a fixed point of the discrete one.
    """
    lower, split = lattice._channel(channel)
    n = np.asarray(n_omega, dtype=float)
    return split * n[lower] + (1.0 - split) * n[lower + 1]


def build_unified_omega_lattice(
    E_bins: np.ndarray, gap: float,
) -> UnifiedOmegaLattice:
    """Build the shared lattice and both channels' two-bin deposits."""
    E = np.asarray(E_bins, dtype=float)
    gap = float(gap)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("E_bins must be a 1D array with at least two cells.")

    spacing = np.diff(E)
    h = float(spacing[0])
    if h <= 0.0 or not np.allclose(spacing, h, rtol=1e-12, atol=1e-12 * h):
        raise PairSplitUnavailable("The unified lattice needs a uniform energy grid.")

    face0 = float(E[0]) - 0.5 * h
    if abs(face0 - gap) > 64.0 * np.finfo(float).eps * max(1.0, gap):
        raise PairSplitUnavailable(
            f"The unified lattice needs the lowest energy face at Δ; got "
            f"{face0:g} against Δ = {gap:g}."
        )

    ratio = 2.0 * gap / h
    if abs(ratio - round(ratio)) > _INT_TOL:
        raise PairSplitUnavailable(
            f"2Δ/h = {ratio:.6f} is not an integer, so no single lattice can "
            "carry a face at 2Δ (the pair channel needs one) and at every "
            "multiple of h (the scattering channel needs those). Choose a bin "
            "count that makes it integral — this is the same commensurability "
            "the phonon frequency map already requires for the two channels to "
            "share bins at all."
        )
    pair_offset = int(round(ratio))

    n = E.size
    index = np.arange(n)
    # Pair: cell (i, j) spans bins (2Δ/h + i + j) and one above.
    pair_lower = (np.add.outer(index, index) + pair_offset).astype(np.int64)
    pair_split = pair_split_fractions(E, gap)

    # Scattering: |E_i - E_j| spans [(|i-j|-1)h, (|i-j|+1)h], so the lower bin
    # is |i-j|-1, except on the diagonal where the support straddles zero and
    # folds onto bin 0.
    delta = np.subtract.outer(index, index)
    abs_delta = np.abs(delta)
    scatter_lower = np.maximum(abs_delta - 1, 0).astype(np.int64)
    scatter_split = _scatter_split_fractions(E, gap, abs_delta)
    diff_sign = np.sign(delta).astype(np.int8)

    highest = int(max(pair_lower.max(), scatter_lower.max())) + 2
    omega_bins = (np.arange(highest, dtype=float) + 0.5) * h
    return UnifiedOmegaLattice(
        omega_bins=omega_bins,
        h=h,
        pair_lower=pair_lower,
        pair_split=pair_split,
        scatter_lower=scatter_lower,
        scatter_split=scatter_split,
        diff_sign=diff_sign,
    )


def _scatter_split_fractions(
    E: np.ndarray, gap: float, abs_delta: np.ndarray,
) -> np.ndarray:
    """Fraction of each cell's |E − E'| measure in its LOWER bin.

    Off the diagonal the geometry mirrors the pair channel: the face curve runs
    corner to corner, so the fraction depends only on |i − j| and on where the
    cell sits, and it is computed the same way.

    ON the diagonal the support is ``[−h, h]``, straddling ω = 0. Folding by
    ``|ω|`` puts all of it in bin 0, so the split is 1 and the emission /
    absorption structure is carried by ``diff_sign`` as before — the fold is
    exact, not an approximation.
    """
    n = E.size
    h = float(E[1] - E[0])
    faces = gap + np.arange(n + 1, dtype=float) * h
    xi = np.sqrt(np.maximum((faces - gap) * (faces + gap), 0.0))
    d_xi = np.diff(xi)

    x_gl, w_gl = np.polynomial.legendre.leggauss(64)
    split = np.ones((n, n), dtype=float)

    i_idx = np.arange(n)
    for offset in range(1, n):
        rows = i_idx[offset:]
        cols = rows - offset
        mid = 0.5 * (xi[rows] + xi[rows + 1])[:, None]
        half = 0.5 * d_xi[rows][:, None]
        x = mid + half * x_gl[None, :]
        E_of_x = np.sqrt(gap * gap + x * x)
        # Interior face of |E - E'| for this cell is at offset*h, so the
        # partner energy along it is E(xi) - offset*h.
        partner_E = E_of_x - offset * h
        partner_xi = np.sqrt(
            np.maximum((partner_E - gap) * (partner_E + gap), 0.0)
        )
        # Below the face means E - E' < offset*h, i.e. E' > partner_E.
        height = np.clip(
            xi[cols + 1][:, None] - partner_xi, 0.0, d_xi[cols][:, None],
        )
        area = np.einsum("rp,p->r", height, w_gl) * half[:, 0]
        value = area / (d_xi[rows] * d_xi[cols])
        split[rows, cols] = value
        split[cols, rows] = value

    return np.clip(split, 0.0, 1.0)
