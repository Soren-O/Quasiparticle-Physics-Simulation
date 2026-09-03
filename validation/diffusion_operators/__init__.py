"""§7.5 diffusion-operator benchmarks (paper "Benchmark problems").

Four transport-only tests that separate the dirty-limit Usadel reduction
A1 = (p, q) = (1, 0) from the diagnostics A1P = (1, 2) (transverse-dressed
flux) and A2 = (2, 2), and the scalar-Boltzmann closures C = (0, -1) and
B = (0, -2):

* :mod:`uniform_gap_packet` -- the energy-dependent effective diffusivity
  ``D_eff(E)/D_N`` traces ``N_1^{q-p}``: falling (A1 and C, which share
  the uniform-gap rate ``D_N/N_1``), rising (A1P), flat (A2), steeply
  falling (B); ``n_qp`` conserved.
* :mod:`gap_gradient_drift` -- the DOS-gradient drift of the quasiparticle
  density ``N_1 f`` (the common readout for every operator),
  ``v = D_N [q + 2(1 - p)] N_1^{q-p-1} d_x N_1``: A1 has *no* drift, the
  legacy C (``+1``) and the A1P diagnostic (``+2``) drift up the gap, and
  A2 and B (``0``) carry no net drift of ``N_1 f``.
* :mod:`interface_trap` -- a Kupriyanov-Lukichev two-gap interface:
  current continuity + ``f``-discontinuity with the coherence-factor
  weight, and A1-vs-A2 distinct closed equilibria.
* :mod:`self_consistent_feedback` -- a gap well dug self-consistently by
  the occupation through the direct gap closure: the quasiparticle density
  of a passive probe packet drifts *away* from the well under the legacy
  C and under A1P, not at all under A1, and only by the finite-packet
  residual of the well's curvature under A2 and B.

Each module exposes a ``run()`` returning structured results (used by the
co-located fast tests) and a ``main()`` writing CSV + a figure under
``outputs/diffusion_operators/`` (gitignored run artifacts).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from qpsim.transport.diffusion.base import DiffusionModel

#: Normal-state diffusivity D_N (um^2/ns), Al-strip scale.
D0_DEFAULT = 6.0

#: Power ``s`` of the readout weight ``N_1**s f`` whose centre of mass the
#: drift benchmarks track. ``s = 1`` is the quasiparticle density ``N_1 f``
#: (whose energy integral is ``n_qp``), the same physical quantity for every
#: operator; ``s = p`` would recover each operator's own conserved density.
READOUT_WEIGHT: int = 1


def drift_coefficient(p: int, q: int, s: int = READOUT_WEIGHT) -> int:
    """Prefactor of the first-moment drift law for ``L_{p,q}``.

    For a narrow packet on a static profile the centre of mass of
    ``N_1**s f`` moves at ``D_N [q + 2(s - p)] N_1**(q - p - 1) d_x N_1``
    (leading order in the gap gradient). With ``s = p`` this is the
    operator's own conserved-density moment, ``D_N q ...``, a structural
    diagnostic of ``q != 0``. With ``s = 1`` it is the drift of the
    quasiparticle density: ``0`` for A1 (1, 0), A2 (2, 2) and B (0, -2),
    ``+1`` for the legacy placement C (0, -1), ``+2`` for the (1, 2)
    diagnostic.
    """
    return q + 2 * (s - p)


def exact_initial_drift(
    f0: np.ndarray,
    N1: np.ndarray,
    x: np.ndarray,
    p: int,
    q: int,
    s: int = READOUT_WEIGHT,
    D0: float = D0_DEFAULT,
) -> np.ndarray:
    """Exact initial rate of the ``N_1**s f`` centre of mass under ``L_{p,q}``.

    On a static profile with reflective ends and the packet away from the
    walls, two integrations by parts give, energy by energy,

        d<x>_s/dt = [ int N_1^(s-p) J dx
                      + (s - p) int (x - <x>_s) N_1^(s-p-1) d_x N_1 J dx ]
                    / int N_1^s f dx ,          J = -D_N N_1^q d_x f ,

    with ``J`` the flux of the conserved density ``N_1^p f``. For ``s = p``
    the second (shape) term vanishes and the first is the closed form
    ``<q D_N N_1^(q-p-1) d_x N_1>`` over the conserved density; for a packet
    narrow on the scale over which ``d_x N_1`` varies the two terms combine
    to ``D_N [q + 2(s - p)] N_1^(q-p-1) d_x N_1`` (:func:`drift_coefficient`).
    Arrays are ``(NE, NX)`` on the uniform grid ``x``; the grid spacing
    cancels between numerator and denominator.
    """
    J = -D0 * np.power(N1, q) * np.gradient(f0, x, axis=1)
    dN1_dx = np.gradient(N1, x, axis=1)
    w = np.power(N1, s) * f0
    weight = np.sum(w, axis=1)
    xbar = np.sum(x[None, :] * w, axis=1) / weight
    flux_term = np.sum(np.power(N1, s - p) * J, axis=1)
    shape_term = (s - p) * np.sum(
        (x[None, :] - xbar[:, None]) * np.power(N1, s - p - 1) * dN1_dx * J,
        axis=1,
    )
    return (flux_term + shape_term) / weight

#: The operators the §7.5 benchmarks compare.
BENCHMARK_MODELS: tuple[DiffusionModel, ...] = (
    DiffusionModel.A1,
    DiffusionModel.A1P,
    DiffusionModel.A2,
    DiffusionModel.C,
    DiffusionModel.B,
)


def results_dir() -> Path:
    """``outputs/diffusion_operators/`` (created on demand; gitignored)."""
    directory = Path(__file__).resolve().parents[2] / "outputs" / "diffusion_operators"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write ``header`` + ``rows`` to ``path`` with stable line endings."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(header))
        writer.writerows([list(row) for row in rows])
