"""The registered analytic benchmarks, one module per kinetic-equation term.

Importing this package registers every benchmark. One module per term, and
each owns its whole statement: the closed form, the case it applies to, the
tolerance and the refinement evidence behind it. A term's benchmark can then
be read, argued with, or replaced without touching the other nine.

The framework they register into is :mod:`qpsim.webui.benchmarks`.
"""

from __future__ import annotations

# Imported for their registration side effects. The registry rejects
# duplicate names, so a copy-pasted module fails loudly at import rather
# than silently shadowing the term it was copied from.
from qpsim.webui.bench import (  # noqa: F401
    diffusion,
    injection,
    pb_drive,
    phonon_escape,
    phonon_recombination_source,
    phonon_scattering_source,
    recombination,
    scattering,
    self_consistent_gap,
    subgap_drive,
)
