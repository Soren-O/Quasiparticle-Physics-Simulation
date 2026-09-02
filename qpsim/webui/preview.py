"""Pre-run preview: what a setup WOULD solve on, before anything runs.

Everything here is what :func:`qpsim.webui.execute.run_kinetics` does in its
first two progress ticks -- build the geometry, the thermal state and the
seed -- stopped before the first step. It answers three questions a person
otherwise learns only from a finished run:

* is this the device I meant -- a GDS layer rasterised at too coarse a mesh
  is obvious as a picture and invisible as a number;
* where does the run start -- a seed that clipped is a different experiment,
  and the note saying so used to arrive only in a finished run's manifest;
* what are the rim's segments called -- so a per-edge condition can be
  addressed by an id the geometry actually has, rather than guessed.

The numbers reported here are computed by the SAME calls the run uses, so
``x_qp_initial`` in a preview equals ``summary.x_qp_initial`` of the run that
follows; the tests hold that equality.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.webui.builders import (
    _DIRECTIONS,
    _edges_facing,
    build_initial_state_2d,
    build_phonon_seed_2d,
    build_state_2d,
)
from qpsim.webui.execute import X_QP_CONVENTION, _xqp_profile_2d
from qpsim.webui.plots import (
    render_cell_field_png,
    render_mask_png,
    render_phonon_seed_png,
)
from qpsim.webui.schemas import AnySetup, KineticsSetup


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _refused(*errors: str) -> dict[str, Any]:
    return {
        "ok": False, "errors": list(errors), "notes": [],
        "geometry": None, "seed": None, "phonons": None, "images": {},
    }


def build_preview(setup: AnySetup) -> dict[str, Any]:
    """The geometry, seed and rim of ``setup`` as the run would build them.

    A setup the builders refuse -- a GDS path with no gdstk, an expression
    that does not evaluate, an override on an edge the geometry lacks --
    comes back as ``ok=False`` with the builder's message, the same words a
    run would fail with. It is not a 500: the person asked what would happen,
    and "it would fail, like this" is the answer.
    """
    if not isinstance(setup, KineticsSetup):
        return _refused(
            f"The {setup.mode!r} mode has no geometry to preview: it is a "
            "rate-equation model over a temperature sweep, not a device.",
        )
    try:
        thermal = build_state_2d(setup)
        seeded, notes = build_initial_state_2d(setup, thermal)
        phonon_seed = build_phonon_seed_2d(setup, thermal.geometry)
    except (ValueError, ArithmeticError) as exc:
        return _refused(str(exc))

    geometry = thermal.geometry
    mesh = float(geometry.mesh_size)
    rows, cols = geometry.shape
    delta_0 = float(setup.material.Delta_0)
    # The same call run_kinetics makes for summary.x_qp_initial and for the
    # thermal reference, so the preview's numbers are the run's numbers.
    profile = _xqp_profile_2d(seeded, delta_0)
    thermal_profile = _xqp_profile_2d(thermal, delta_0)

    images: dict[str, str | None] = {
        "mask": _data_uri(render_mask_png(geometry.mask, mesh)),
        "seed_xqp": _data_uri(render_cell_field_png(
            geometry.mask, mesh, profile,
            f"x_qp at t = 0 — {setup.initial.kind} seed",
            f"x_qp  [{X_QP_CONVENTION}]",
        )),
        "phonon_seed": None,
    }
    phonons: dict[str, Any] = {
        "mode": setup.phonons.mode,
        "seeded": phonon_seed is not None,
    }
    if phonon_seed is not None:
        # The seed is a profile over frequency, independent of the mask (the
        # engine broadcasts it across cells), so the honest picture is
        # n_ph(ω) against the bath -- not a device map that would be uniform.
        omega, _, _, _ = build_phonon_frequency_map(thermal.spectral.E)
        bath = thermal_phonon_occupation(omega, setup.T_bath)
        images["phonon_seed"] = _data_uri(render_phonon_seed_png(
            omega, phonon_seed, bath, float(thermal.spectral.gap),
        ))
        phonons["n_ph_seed_mean"] = float(np.mean(phonon_seed))
        phonons["n_ph_bath_mean"] = float(np.mean(bath))

    edges = [
        {
            "id": edge.edge_id,
            "normal": edge.normal,
            "faces": len(edge.faces),
            # Segment coordinates are recorded in cell units on integer grid
            # lines; in microns they are the extents the mask figure draws.
            "x0_um": edge.x0 * mesh, "y0_um": edge.y0 * mesh,
            "x1_um": edge.x1 * mesh, "y1_um": edge.y1 * mesh,
        }
        for edge in geometry.edges
    ]
    return {
        "ok": True,
        "errors": [],
        "notes": list(notes),
        "geometry": {
            "name": geometry.name,
            "rows": rows, "cols": cols,
            "cells": geometry.cell_count,
            "dimensionality": geometry.dimensionality,
            "mesh_size_um": mesh,
            "width_um": cols * mesh, "height_um": rows * mesh,
            "edges": edges,
            # The direction aliases the per-edge table accepts, resolved to
            # the segments they would address on THIS mask. A direction that
            # resolves to nothing is one the engine will refuse.
            "directions": {d: _edges_facing(geometry, d) for d in _DIRECTIONS},
        },
        "seed": {
            "kind": setup.initial.kind,
            "x_qp_initial": float(np.mean(profile)),
            "x_qp_initial_max": float(np.max(profile)),
            "x_qp_thermal": float(np.mean(thermal_profile)),
            "x_qp_convention": X_QP_CONVENTION,
            "clipped": any("clipped" in note for note in notes),
        },
        "phonons": phonons,
        "images": images,
    }
