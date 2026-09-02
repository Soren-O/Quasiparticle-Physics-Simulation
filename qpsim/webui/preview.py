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
from pathlib import Path
from typing import Any

import numpy as np

from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.geometries import discover_gds_layers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.webui.builders import (
    _DIRECTIONS,
    _edges_facing,
    boundary_sources_2d,
    build_boundary_conditions_2d,
    build_geometry_2d,
    build_initial_state_2d,
    build_phonon_seed_2d,
    build_state_2d,
)
from qpsim.webui.execute import X_QP_CONVENTION, _xqp_profile_2d
from qpsim.webui.plots import (
    render_cell_field_png,
    render_mask_png,
    render_mask_raw_png,
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


def geometry_block(setup: KineticsSetup) -> dict[str, Any]:
    """The geometry as the run would build it, and nothing else.

    What the geometry page needs on every import and after every edge
    assignment: the mask as a one-pixel-per-cell image, every rim segment
    with its EFFECTIVE condition and where that condition came from (the
    rim default, a direction alias, or the segment's own id), the layout
    origin, and the aliases resolved on this mask. Cheap on purpose -- no
    spectral context, no seed -- so a click can re-ask it. Raises the
    builders' ValueError for a setup they refuse; the route turns that into
    a message.
    """
    geometry = build_geometry_2d(setup)
    conditions = build_boundary_conditions_2d(setup, geometry)
    sources = boundary_sources_2d(setup, geometry)
    mesh = float(geometry.mesh_size)
    rows, cols = geometry.shape
    edges = []
    for edge in geometry.edges:
        bc = conditions[edge.edge_id]
        edges.append({
            "id": edge.edge_id,
            "normal": edge.normal,
            "faces": len(edge.faces),
            # Cell units on integer grid lines (what the page draws in), and
            # microns (what a person reads).
            "x0": float(edge.x0), "y0": float(edge.y0),
            "x1": float(edge.x1), "y1": float(edge.y1),
            "x0_um": edge.x0 * mesh, "y0_um": edge.y0 * mesh,
            "x1_um": edge.x1 * mesh, "y1_um": edge.y1 * mesh,
            "condition": {
                "kind": str(bc.kind),
                "value": None if bc.value is None else float(bc.value),
                "aux_value": None if bc.aux_value is None else float(bc.aux_value),
            },
            "source": sources[edge.edge_id],
        })
    bounds = geometry.bounds or (0.0, 0.0, cols * mesh, rows * mesh)
    block: dict[str, Any] = {
        "name": geometry.name,
        "source": geometry.source or "rectangle",
        "origin_um": [float(bounds[0]), float(bounds[1])],
        "bounds_um": [float(b) for b in bounds],
        "rows": rows, "cols": cols,
        "cells": geometry.cell_count,
        "dimensionality": geometry.dimensionality,
        "mesh_size_um": mesh,
        "width_um": cols * mesh, "height_um": rows * mesh,
        "edges": edges,
        "directions": {d: _edges_facing(geometry, d) for d in _DIRECTIONS},
        "rim_default": {
            "kind": str(setup.boundary.kind),
            "value": float(setup.boundary.value),
            "aux_value": setup.boundary.aux_value,
        },
        "mask_png": _data_uri(render_mask_raw_png(geometry.mask)),
    }
    if setup.geometry.kind == "gds" and setup.geometry.gds_path:
        block["gds_layers"] = discover_gds_layers(Path(setup.geometry.gds_path))
        block["gds_layer"] = int(setup.geometry.gds_layer)
    return block


def build_geometry(setup: AnySetup) -> dict[str, Any]:
    """The geometry block, or a refusal in the builders' own words."""
    if not isinstance(setup, KineticsSetup):
        return {"ok": False, "errors": [
            f"The {setup.mode!r} mode has no geometry: it is a rate-equation "
            "model over a temperature sweep, not a device.",
        ], "geometry": None}
    try:
        block = geometry_block(setup)
    except (ValueError, ArithmeticError, OSError) as exc:
        return {"ok": False, "errors": [str(exc)], "geometry": None}
    return {"ok": True, "errors": [], "geometry": block}


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
    except (ValueError, ArithmeticError, OSError) as exc:
        return _refused(str(exc))

    geometry = thermal.geometry
    mesh = float(geometry.mesh_size)
    delta_0 = float(setup.material.Delta_0)
    # The same call run_kinetics makes for summary.x_qp_initial and for the
    # thermal reference, so the preview's numbers are the run's numbers.
    profile = _xqp_profile_2d(seeded, delta_0)
    thermal_profile = _xqp_profile_2d(thermal, delta_0)

    images: dict[str, str | None] = {
        # The rim drawn on the mask, each segment in its condition's colour
        # and labelled by id -- the picture a per-edge override is written
        # against.
        "mask": _data_uri(render_mask_png(
            geometry.mask, mesh, geometry.edges,
            thermal.conditions or geometry.conditions(),
        )),
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

    # The geometry block is the geometry page's own payload (see
    # geometry_block): the same edges, conditions, aliases and layout
    # origin, built by the same call, so the two views cannot disagree.
    return {
        "ok": True,
        "errors": [],
        "notes": list(notes),
        "geometry": geometry_block(setup),
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
