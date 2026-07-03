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

One run mode per shipped engine surface — the frontend calls the
existing API and adds **no physics**:

| Mode | Engine surface | Outputs |
|---|---|---|
| 0-D steady state | `T3DiffusionBackend.steady_state` (Newton thermal / Picard+Anderson / coupled-Newton) | f(E), n_ph(ω), x_qp, n_qp, Mattis–Bardeen σ₁/σ₂ · Q_i · δω/ω, gap suppression, effective phonon T |
| 0-D transient | `services.transient.run_time_dependent` (ETD2) | f(E, t) snapshot family, x_qp(t), Q_i(t) |
| 1D strip | `T3Spatial1DBackend.run_until_steady_state` (operator family A1/A1P/A2/C/B, optional two-gap step + Kupriyanov–Lukichev interface, Gaussian-in-E injection) | f(E, x) heatmap, x_qp(x) profile, convergence trace |
| M25 junction | `services.rate_equation` moment layer over a T sweep with continuation seeding | x_α(T), μ_α(T)/Δ_L, p₁(T) |

Both photon drives (sub-gap and pair-breaking) are available on the
0-D modes. The materials browser mirrors the YAML database
(`qpsim/materials/data/`); picking a material copies its values into
the editable setup.

## Design decisions

* **Local web app, not a desktop toolkit.** Cross-platform
  (Windows + macOS), and the whole stack — request handlers, run
  execution, plot rendering — is exercised headlessly on CI through
  FastAPI's test client. The legacy tkinter app informed the UX
  (setup editor → background run with progress/cancel → persisted,
  browsable results) but none of its 2D-engine scope: the new engine
  has no 2D spatial backend, so the frontend does not pretend to.
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

## The one engine change

`services.transient.run_time_dependent` and
`T3Spatial1DBackend.run_until_steady_state` gained an optional
physics-neutral `progress_hook: Callable[[float, float], bool]` —
called after each substep with `(t, total_time)`; returning `False`
stops the run cleanly at the current time. `None` (the default)
leaves both time loops bit-for-bit unchanged (regression-tested).
This is the progress/cancel affordance the frontend needs; no
backend, kernel, or solver logic changed.

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

Known v1 limits: the 1D strip reports x_qp profiles but not
current-weighted resonator response (needs a mode profile —
`observables.spatial_ac_response` exists for scripted use); the M25
sweep uses continuation seeding, not true bifurcation tracking, so
multi-stable branch points can fail to converge (reported per point);
`nbar_loop` and full Device/Qubit composition are not yet exposed.
