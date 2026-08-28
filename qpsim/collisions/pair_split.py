"""Exact two-bin cut-cell split for the phonon pair channel.

THE GEOMETRY. At phonon frequency ω the pair source is a line integral along
``E + E' = ω`` with two endpoint singularities of the BCS density of states
that COALESCE as ω → 2Δ. The exact threshold value is Kaplan's ``Δ·S₊ = πΔ``.
The raw anti-diagonal midpoint rule gives ``4Δ`` there. The shipped point-
collocation path corrects that line value with Kaplan's integral; this module
instead constructs a finite-volume average over frequency strips.

In the state-counting coordinate ``ξ = √(E² − Δ²)`` the BCS
measure is flat — ``ρ(E) dE = dξ`` exactly — so a product cell is a RECTANGLE
of uniform measure, and the deposit is a pure area question. Near threshold
``ω ≈ 2Δ + (ξ² + ξ'²)/2Δ``, so the first frequency strip is a QUARTER DISK
inscribed in the corner cell's square: area ratio ``π/4``.

The same ``π/4`` therefore appears in two valid discretizations: as this overlap
fraction for a finite volume, and as the threshold correction from the raw
midpoint value to the exact collocated line integral. Equality of the numbers
does not make the operations interchangeable. On a fixed 24 μeV window the
shipped corrected point rule divided by this split is 0.9667, 0.9771, 0.9862,
0.9922 as the 90/180/360/720-bin grid refines: they converge to one.

THE GEOMETRY IS UNUSUALLY CLEAN, and worth stating because it makes the
quadrature easy. On a uniform E grid of spacing ``h`` whose lowest face sits at
Δ, cell ``(i, j)`` has ω-support ``[2Δ + (i+j)h, 2Δ + (i+j+2)h]`` — exactly two
bins of a lattice with faces at ``2Δ + m·h``, straddling the single interior
face ``ω_f = 2Δ + (i+j+1)h``. Along that face,

    ξ'_max(ξ) = √((ω_f − √(Δ² + ξ²))² − Δ²)

and at the cell's own ξ-endpoints it takes the values

    ξ'_max(ξ_i) = ξ_{j+1}          ξ'_max(ξ_{i+1}) = ξ_j

i.e. the curve enters and leaves through OPPOSITE CORNERS of the rectangle.
So no clamping is ever active in the interior, the integrand is smooth and
monotone, and the split fraction is a plain area under a curve.

WHAT THIS MODULE IS AND IS NOT. It computes finite-volume fractions. It does
not authorize changing the shipped point-sample phonon representation. In
particular, composing its deposit with the adjoint read produces a tridiagonal
absorption loss whose off-diagonal terms drive an empty bin negative when its
neighbour is occupied. Exact transpose bookkeeping is therefore not a valid
kinetic closure. Wiring this module requires a new, positivity-preserving
phonon representation, not merely replacing the frequency map.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pair_split_fractions", "PairSplitUnavailable"]

# Gauss-Legendre order. With the orientation choice and the corner
# substitution below, every cell's integrand is smooth, so this is spectral
# and 64 nodes is generous: measured against adaptive quadrature the worst cell
# on the 180/405/1620-bin shipped grids sits at 5.6e-16 / 6.7e-16 / 2.9e-15,
# i.e. machine precision, against the O(h) finite-volume mesh error of
# 4.8e-03 / 2.2e-03 / 5.4e-04. Twelve orders of headroom.
_GL_NODES = 64

# A cell's support spans exactly two ω bins only when the lowest energy face
# sits at Δ. Admit coordinate roundoff, not a user-scale tolerance.
_FACE_TOL = 64.0 * np.finfo(float).eps


class PairSplitUnavailable(RuntimeError):
    """The grid does not admit the exact two-bin split.

    Raised rather than returning an approximate split: a fraction computed for
    a geometry the cell does not have is a plausible number with no meaning,
    and the caller can fall back deliberately.
    """


def pair_split_fractions(
    E_bins: np.ndarray,
    gap: float,
    *,
    nodes: int = _GL_NODES,
    chunk: int = 64,
) -> np.ndarray:
    """Fraction of each pair cell's weight belonging to its LOWER ω bin.

    ``S[i, j]`` is the ρ-weighted fraction of cell ``(i, j)`` whose pair
    frequency ``E + E'`` falls below the interior face ``2Δ + (i+j+1)h``. The
    complement ``1 - S[i, j]`` belongs to the upper bin, so the pair is a
    partition of unity BY CONSTRUCTION and the event count ``φ = 1`` is
    preserved exactly whatever the quadrature error.

    Returns ``(NE, NE)``. Cells with no spectral support above the gap get
    ``0.0``; they carry no pair weight, so the value is never read.
    """
    E = np.asarray(E_bins, dtype=float)
    gap = float(gap)
    if E.ndim != 1 or E.size < 2:
        raise ValueError("E_bins must be a 1D array with at least two cells.")
    if not np.all(np.isfinite(E)) or not np.isfinite(gap) or gap <= 0.0:
        raise ValueError("E_bins must be finite and gap must be finite and positive.")

    spacing = np.diff(E)
    h = float(spacing[0])
    if h <= 0.0 or not np.allclose(spacing, h, rtol=1e-12, atol=1e-12 * h):
        raise PairSplitUnavailable(
            "The two-bin split needs a UNIFORM energy grid: a cell's ω-support "
            "is two bins wide only when every cell is. Got spacings spanning "
            f"[{spacing.min():g}, {spacing.max():g}]."
        )

    face0 = float(E[0]) - 0.5 * h
    if abs(face0 - gap) > _FACE_TOL * max(1.0, gap):
        raise PairSplitUnavailable(
            f"The two-bin split needs the lowest energy face at Δ; this grid "
            f"starts at {face0:g} against Δ = {gap:g}. Off-face, a cell's "
            "ω-support straddles three bins rather than two and the split is a "
            "different tensor."
        )

    n = E.size
    # Cell faces in the state-counting coordinate. rho(E) dE = d xi EXACTLY, so
    # in xi the cell is a rectangle of uniform measure and the split is an area
    # ratio -- no density enters the quadrature at all.
    faces = gap + np.arange(n + 1, dtype=float) * h
    xi = np.sqrt(np.maximum((faces - gap) * (faces + gap), 0.0))
    d_xi = np.diff(xi)

    x_gl, w_gl = np.polynomial.legendre.leggauss(int(nodes))
    S = np.zeros((n, n), dtype=float)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        # Quadrature nodes in xi across each cell of this chunk: (rows, nodes)
        mid = 0.5 * (xi[lo:hi] + xi[lo + 1:hi + 1])[:, None]
        half = 0.5 * d_xi[lo:hi][:, None]
        x = mid + half * x_gl[None, :]
        E_of_x = np.sqrt(gap * gap + x * x)

        # omega face of cell (i, j) is 2*gap + (i+j+1)*h, so the partner energy
        # bound is omega_f - E(xi) = gap + (i + j + 1)*h - (E(xi) - gap).
        i_index = np.arange(lo, hi, dtype=float)[:, None, None]
        j_index = np.arange(n, dtype=float)[None, :, None]
        partner_E = (
            2.0 * gap
            + (i_index + j_index + 1.0) * h
            - E_of_x[:, None, :]
        )
        partner_xi = np.sqrt(
            np.maximum((partner_E - gap) * (partner_E + gap), 0.0)
        )
        # The curve enters and leaves through opposite corners, so this is
        # already inside [xi_j, xi_{j+1}] up to roundoff; the clip only guards
        # the endpoints against a negative-zero radicand.
        height = np.clip(
            partner_xi - xi[None, :n, None],
            0.0,
            d_xi[None, :, None],
        )
        area = np.einsum("ijp,p->ij", height, w_gl) * half
        with np.errstate(invalid="ignore", divide="ignore"):
            S[lo:hi] = area / (d_xi[lo:hi][:, None] * d_xi[None, :])
        del partner_E, partner_xi, height

    # Cells with no represented support above the gap carry no pair weight.
    S[~np.isfinite(S)] = 0.0
    S = np.clip(S, 0.0, 1.0)

    # SYMMETRY IS NOT COSMETIC HERE, IT PICKS THE CONDITIONING. S[i, j] and
    # S[j, i] describe mirror-image cells and must be equal, but the two
    # orientations are not equally well conditioned: the integrand meets a
    # square-root endpoint exactly when the PARTNER cell reaches the gap, i.e.
    # when j is small. Integrating over the low cell with the far partner
    # avoids it entirely -- measured S[0,1] exact to 2e-16 against S[1,0] at
    # 4e-07 on the same grid, from the same rule.
    #
    # Both orientations are already computed, so taking the upper triangle and
    # mirroring it costs nothing and removes the error rather than averaging
    # it down.
    upper = np.triu(S)
    S = upper + np.triu(S, 1).T

    # THE CORNER CELL IS THE ONE THE ORIENTATION TRICK CANNOT SAVE, because
    # there BOTH cells reach the gap and the square-root endpoint is present
    # whichever way round it is integrated. It is also the cell that matters
    # most -- it carries the raw midpoint-to-volume 4/π mismatch -- so it gets
    # its own rule rather than a tolerance.
    #
    # Substituting ξ = ξ₁(1 − u²) maps the endpoint onto u = 0 and cancels it:
    # ξ'_max ~ C·√(ξ₁ − ξ) = C·√ξ₁·u there, while dξ = −2ξ₁u du, so the
    # integrand goes as u² and is smooth on the whole interval. For the exact
    # circle this is the ξ = ξ₁ sin θ substitution that makes ∫√(ξ₁² − ξ²)
    # elementary; here it does the same job without assuming the circle.
    S[0, 0] = _corner_fraction(gap, h, float(xi[1]), x_gl, w_gl)
    return S


def _corner_fraction(
    gap: float, h: float, xi_1: float, x_gl: np.ndarray, w_gl: np.ndarray,
) -> float:
    """Split fraction of the gap-corner cell, endpoint singularity removed."""
    if xi_1 <= 0.0:
        return 0.0
    # Gauss-Legendre nodes mapped onto u in [0, 1].
    u = 0.5 * (x_gl + 1.0)
    w = 0.5 * w_gl
    x = xi_1 * (1.0 - u * u)
    partner_E = 2.0 * gap + h - np.sqrt(gap * gap + x * x)
    partner_xi = np.sqrt(np.maximum((partner_E - gap) * (partner_E + gap), 0.0))
    # dξ = 2 ξ₁ u du, and the cell's measure is ξ₁², so the ξ₁ cancels once.
    return float(np.sum(w * partner_xi * 2.0 * u) / xi_1)
