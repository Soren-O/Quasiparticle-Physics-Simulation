"""§7.5 diffusion-operator benchmarks (paper "Benchmark problems").

Three transport-only tests that separate the dirty-limit Usadel reduction
A1 = (p, q) = (1, 0) from the diagnostics A1P = (1, 2) (transverse-dressed
flux) and A2 = (2, 2), and the scalar-Boltzmann closures C = (0, -1) and
B = (0, -2):

* :mod:`uniform_gap_packet` -- the energy-dependent effective diffusivity
  ``D_eff(E)/D_N`` traces ``N_1^{q-p}``: falling (A1 and C, which share
  the uniform-gap rate ``D_N/N_1``), rising (A1P), flat (A2), steeply
  falling (B); ``n_qp`` conserved.
* :mod:`gap_gradient_drift` -- the DOS-gradient drift velocity
  ``v = D_N q N_1^{q-p-1} d_x N_1``: A1 (``q = 0``) has *no* drift, the
  A1P/A2 diagnostics (``q = 2``) drift up the gap, C/B (``q < 0``) down.
* :mod:`interface_trap` -- a Kupriyanov-Lukichev two-gap interface:
  current continuity + ``f``-discontinuity with the coherence-factor
  weight, and A1-vs-A2 distinct closed equilibria.

Each module exposes a ``run()`` returning structured results (used by the
co-located fast tests) and a ``main()`` writing CSV + a figure under
``outputs/diffusion_operators/`` (gitignored run artifacts).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from qpsim.transport.diffusion.base import DiffusionModel

#: Normal-state diffusivity D_N (um^2/ns), Al-strip scale.
D0_DEFAULT = 6.0

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
