---
title: Frontend (qpsim.webui)
description: Local web frontend — design decisions, architecture, and how to run it.
status: shipped 2026-07-03
---

# Frontend

A local web application for driving the shipped engine surfaces from a
browser. Install the optional extra and launch:

```bash
pip install -e ".[ui]"
qpsim-ui                 # or: python -m qpsim.webui
```

`qpsim-ui` starts a local server (default `127.0.0.1:8756`) and opens
a browser tab. Setups and finished runs persist in a workspace
directory (`--workspace`, else `$QPSIM_WORKSPACE`, else
`~/qpsim-workspace`). `--no-browser` for headless use; interactive API
docs at `/api/docs`.

## What it drives

Two run modes, discriminated on `mode` in the saved setup — the
frontend calls the engine API and adds **no physics**:

| Mode | Engine surface | Outputs |
|---|---|---|
| `kinetics`, `strategy="time_march"` | `SpatialBackend.run` on the cell mask (operator family A1/A1P/A2/C/B, per-face boundary conditions, optional two-gap step + Kupriyanov–Lukichev interface, prescribed drives, Gaussian-in-E injection) | geometry mask, x_qp field and profile, f(E, cell) map, recorded frames of the field / gap / phonon map, x_qp(t) and QP-energy time series, effective phonon T field |
| `kinetics`, `strategy="steady_state"` | `DiffusionBackend.steady_state` on a one-cell mask (Newton thermal / Picard+Anderson / coupled-Newton) | f(E), n_ph(ω), x_qp, n_qp, gap suppression, effective phonon T |
| `m25_junction` | `services.rate_equation` moment layer over a T sweep with continuation seeding | x_α(T), μ_α(T)/Δ_L, p₁(T) |

The mask sets the dimensionality: a 1×1 mask is a single cell, a one-row
mask is a strip, and anything else is a 2-D device. It is built from a
rectangle, a rasterised GDS layer, or polygons authored in the browser
(`GeometrySource`), and the rim carries a default boundary condition with
per-edge overrides by id (`EdgeConditions`), picked on the geometry page.
`strategy` selects the entry point rather than the algorithm:
`steady_state` routes to the 0-D root find, whose state has no cell axis
and therefore needs a one-cell mask, while a multi-cell device reaches
steady state by time-marching to `stop_tol`.

Both photon drives (sub-gap and pair-breaking) are available on the
`kinetics` mode on any geometry. The Mattis–Bardeen probe
(σ₁/σ₂ · Q_i · δω/ω, and Q_i(t) when frames are recorded) runs on a
single cell only: σ(f) is nonlinear, so a probe of a cell average and an
average of the probe are different claims. The materials browser mirrors
the YAML database (`qpsim/materials/data/`); picking a material copies
its values into the editable setup.

## Design decisions

* **Local web app, not a desktop toolkit.** Cross-platform
  (Windows + macOS), and the whole stack — request handlers, run
  execution, plot rendering — is exercised headlessly on CI through
  FastAPI's test client. The legacy tkinter app informed the UX
  (setup editor → background run with progress/cancel → persisted,
  browsable results). The spatial backend runs on a mask of any
  dimensionality, so the frontend drives a single cell, a strip, or a
  full 2-D mask, with a geometry page that draws the rasterised device
  beside the form and lets each edge segment be clicked and given its
  own boundary condition.
* **Server-rendered matplotlib (Agg) → PNG.** Same plotting stack as
  the validation suite; no JS charting dependency, no node build. The
  browser page is no-build vanilla HTML/CSS/JS served from
  `qpsim/webui/static/`.
* **UI dependencies are an optional extra.** `fastapi`, `uvicorn`,
  `matplotlib` live in `qpsim[ui]`; the core library's imports are
  untouched, and nothing in `qpsim` outside `qpsim/webui/` imports the
  extra. `tests/webui/` skips cleanly when the extra is absent.
* **Validation before launch.** Pydantic schemas catch shape/bound
  errors; `webui.builders.validate_setup` reports cross-field physics
  problems with engine-derived context — drive frequencies vs 2Δ,
  probe ω₀ vs Δ, Dynes Γ × spatial transport (the engine's own
  guard, surfaced early), photon-grid commensurability with the
  actual dE and the value the engine would snap to.
* **Runs are background jobs.** A single-worker thread pool executes
  runs (serialized on purpose — solves are CPU-bound), with live
  progress and cooperative cancellation, manifest + compressed-NPZ
  persistence, and per-plot PNG / per-table CSV endpoints rendered on
  demand.

## The frontend's only footprint on the engine

`services.transient.run_time_dependent` and `SpatialBackend.run` each
take an optional physics-neutral
`progress_hook: Callable[[float, float], bool]` — called after each
substep with `(t, total_time)`; returning `False` stops the run cleanly
at the current time. `None` (the default) leaves both time loops
bit-for-bit identical to a run without the hook (regression-tested).
That hook is the whole progress/cancel affordance: no backend, kernel,
or solver logic is frontend-specific.

## Layout

```
qpsim/webui/
    schemas.py    # pydantic setup models (one per mode, discriminated on "mode")
    builders.py   # setup → engine objects; cross-field validation
    execute.py    # mode executors → RunPayload {arrays, summary, notes}
    runner.py     # background job manager (progress, cancel)
    store.py      # workspace persistence (setups JSON, runs manifest+NPZ)
    plots.py      # named matplotlib figures + CSV tables per mode
    server.py     # FastAPI app factory (create_app)
    cli.py        # qpsim-ui entry point
    static/       # index.html, app.css, app.js (no build step)
```

Known limits: a multi-cell run reports x_qp profiles but not
current-weighted resonator response (that needs a mode profile —
`observables.spatial_ac_response` exists for scripted use); the M25
sweep uses continuation seeding, not true bifurcation tracking, so
multi-stable branch points can fail to converge (reported per point);
`nbar_loop` and full Device/Qubit composition are not exposed.
