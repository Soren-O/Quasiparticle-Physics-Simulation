r"""ExternalFlux — boundary source/sink contract on a Region's f-equation.

Decomposes a region-level external coupling into the same
``(gain, loss_rate)`` shape every collision channel uses:

.. math::

    \frac{\partial f}{\partial t}
    = \dotsb_{\text{collisions, transport, gap}}
    + \mathrm{gain}(E)
    - \mathrm{loss\_rate}(E)\,f(E).

Signed fluxes are explicitly rejected — extraction (flow OUT of
the region across a junction) is encoded as ``loss_rate * f``,
NOT a negative gain. This preserves the positivity-preserving
ETD / Newton machinery the existing T3 stack relies on.

Units are ``1/ns`` to match the rest of the T3 stack (every
collision evaluator returns rates in ``1/ns``). Junction
implementations are responsible for converting from Stage A's
``Hz`` rates and applying the boundary-current normalization
``gain(E) = I_J(E) / (4 ρ_F ρ(E) V)`` documented in
``docs/Device_Architecture.md`` §3.2.1.

v1 shape is ``(NE,)`` because Gate-2 T3 is lumped-0D. Optional
``(NE, 1)`` is squeezed transparently. ``target_cells`` is
reserved for Gate 5 spatial T3 and IGNORED in v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExternalFlux:
    """A positivity-preserving boundary source/sink for a Region's ``f(E)``.

    Parameters
    ----------
    gain
        Additive source rate ``∂_t f += gain``. Shape ``(NE,)``;
        ``(NE, 1)`` is accepted and squeezed. All entries must be
        non-negative. Units: ``1/ns``.
    loss_rate
        Damping coefficient ``∂_t f -= loss_rate * f``. Same shape
        as ``gain``. All entries must be non-negative. Units: ``1/ns``.
    target_cells
        Reserved for Gate 5 spatial T3. IGNORED in v1 (lumped 0D
        regions have one cell). Pass ``None`` until spatial T3
        lands.
    diagnostics
        Free-form dict for solver-log diagnostics: junction name,
        total injected current, etc. Not consumed by the residual.

    Raises
    ------
    ValueError
        On wrong shape, mismatched shapes, negative entries, or
        non-finite values.
    """

    gain: np.ndarray
    loss_rate: np.ndarray
    target_cells: np.ndarray | None = None
    diagnostics: dict[str, str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        gain = np.asarray(self.gain, dtype=float)
        loss_rate = np.asarray(self.loss_rate, dtype=float)

        # Squeeze (NE, 1) → (NE,) transparently; reject anything else.
        if gain.ndim == 2 and gain.shape[1] == 1:
            gain = gain[:, 0]
        if loss_rate.ndim == 2 and loss_rate.shape[1] == 1:
            loss_rate = loss_rate[:, 0]

        if gain.ndim != 1:
            raise ValueError(
                f"ExternalFlux.gain must be 1D (shape (NE,)) in v1; "
                f"got shape {gain.shape}. Spatial T3 with (NE, NR) "
                f"shape lands at Gate 5."
            )
        if loss_rate.ndim != 1:
            raise ValueError(
                f"ExternalFlux.loss_rate must be 1D (shape (NE,)) in v1; "
                f"got shape {loss_rate.shape}."
            )
        if gain.shape != loss_rate.shape:
            raise ValueError(
                f"ExternalFlux.gain and loss_rate shapes must match; "
                f"got gain {gain.shape}, loss_rate {loss_rate.shape}."
            )
        if not np.all(np.isfinite(gain)):
            raise ValueError("ExternalFlux.gain has non-finite entries.")
        if not np.all(np.isfinite(loss_rate)):
            raise ValueError("ExternalFlux.loss_rate has non-finite entries.")
        if np.any(gain < 0.0):
            raise ValueError(
                f"ExternalFlux.gain must be non-negative everywhere; "
                f"got min {float(gain.min())}. Extraction flows are "
                f"encoded as loss_rate*f, not as a negative gain."
            )
        if np.any(loss_rate < 0.0):
            raise ValueError(
                f"ExternalFlux.loss_rate must be non-negative everywhere; "
                f"got min {float(loss_rate.min())}."
            )

        # Copy so caller mutating the original arrays after construction
        # doesn't bypass our non-negative/finite validation. Then mark
        # the stored copies as read-only so ``ef.gain[0] = -1`` raises
        # rather than silently corrupting solver state.
        gain = gain.copy()
        loss_rate = loss_rate.copy()
        gain.flags.writeable = False
        loss_rate.flags.writeable = False

        # frozen=True forbids direct attribute rebinding; use
        # object.__setattr__ to populate during __post_init__.
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "loss_rate", loss_rate)

    @classmethod
    def zero(cls, NE: int) -> ExternalFlux:
        """A zero-flux instance shaped for ``NE`` energy bins."""
        return cls(gain=np.zeros(NE), loss_rate=np.zeros(NE))

    def _validate_for_NE(self, NE: int) -> None:
        """Reject if ``gain``/``loss_rate`` length doesn't match the grid.

        The default constructor only checks 1D-ness and matching
        sizes, so a length-``M`` flux passed to a length-``NE`` grid
        sneaks through if ``M ≠ NE`` AND the offending solver site
        relies on NumPy broadcasting. The pathological case is ``M = 1``,
        which broadcasts silently across every bin and turns a
        single-bin contract into an all-bin one. Solver entry points
        invoke this method before doing any addition.
        """
        if self.gain.size != NE:
            raise ValueError(
                f"ExternalFlux is sized for {self.gain.size} energy bins "
                f"but the solver grid has {NE}. Pass an ExternalFlux of "
                f"shape ({NE},) or use ExternalFlux.zero({NE})."
            )
