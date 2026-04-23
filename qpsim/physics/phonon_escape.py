"""Phonon-bath escape time ``τ_l(ω)`` builders.

Two first-class modes per D1 in ``docs/Phonon_Model_Decisions.md``;
the caller picks at ``PhononState`` construction time:

* :func:`constant_tau_l` — scalar ``τ_l`` tiled across the frequency
  grid. Matches the Fischer–Catelani 2023 convention and the current
  reference implementation.
* :func:`acoustic_escape_tau_l` — ``τ_l ≈ 4 d / (η s)`` computed from
  a :class:`Material`'s film thickness, substrate transmission
  coefficient, and sound velocity. See
  ``docs/Phonon_Escape_Time.md`` for the derivation.

Distinct from the pair-breaking lifetime ``τ_PB(ω)`` (Kaplan 1976),
which lives in the collision integral rather than the transport term.
"""

from __future__ import annotations

import numpy as np

from qpsim.materials.database import Material


def constant_tau_l(omega_bins: np.ndarray, value: float) -> np.ndarray:
    """Tile a scalar ``τ_l`` across the phonon frequency grid.

    Parameters
    ----------
    omega_bins
        Phonon frequency grid of shape ``(N_branch, N_omega)`` (matches
        :attr:`qpsim.phonon_models.state.PhononState.omega_bins`).
    value
        Constant ``τ_l`` in nanoseconds.

    Returns
    -------
    tau_l
        Array of the same shape as ``omega_bins`` with every entry
        equal to ``value``.
    """
    if value <= 0:
        raise ValueError("tau_l value must be positive.")
    return np.full_like(omega_bins, float(value), dtype=float)


def acoustic_escape_tau_l(
    omega_bins: np.ndarray,
    material: Material,
    *,
    branch: str = "debye",
) -> np.ndarray:
    r"""Compute ``τ_l ≈ 4 d / (η s)`` from a :class:`Material`.

    The AMM closed-form is frequency-independent in ``η``; the returned
    array tiles the same scalar across the phonon grid. Frequency
    dependence in ``η(ω)`` can be added later without changing this
    signature.

    Parameters
    ----------
    omega_bins
        Phonon frequency grid, shape ``(N_branch, N_omega)``.
    material
        Source of ``film_thickness`` (nm), ``substrate_transmission_eta``
        (dimensionless), and one of the sound velocities (m/s) per
        ``branch``.
    branch
        Which sound velocity to use: ``"debye"`` (Debye average,
        v1 default), ``"longitudinal"``, or ``"transverse"``.

    Returns
    -------
    tau_l
        Array of shape ``omega_bins.shape`` with ``τ_l`` in nanoseconds.
    """
    if material.film_thickness <= 0:
        raise ValueError(
            "material.film_thickness must be positive to compute acoustic-escape τ_l."
        )
    if material.substrate_transmission_eta <= 0:
        raise ValueError(
            "material.substrate_transmission_eta must be positive."
        )

    if branch == "debye":
        s = material.sound_velocity_debye
        missing_msg = "material.sound_velocity_debye"
    elif branch == "longitudinal":
        s = material.sound_velocity_longitudinal or None
        missing_msg = "material.sound_velocity_longitudinal"
    elif branch == "transverse":
        s = material.sound_velocity_transverse or None
        missing_msg = "material.sound_velocity_transverse"
    else:
        raise ValueError(
            f"branch must be 'debye', 'longitudinal', or 'transverse'; got {branch!r}."
        )

    if s is None or s <= 0:
        raise ValueError(
            f"{missing_msg} is not set; cannot compute acoustic-escape τ_l."
        )

    # τ_l [ns] = 4 · d[nm] · (1e-9 m/nm) / (η · s[m/s]) · 1e9 ns/s
    #          = 4 · d[nm] / (η · s[m/s])    — units cancel as derived.
    tau_l_ns = 4.0 * material.film_thickness / (
        material.substrate_transmission_eta * s
    )
    return np.full_like(omega_bins, tau_l_ns, dtype=float)
