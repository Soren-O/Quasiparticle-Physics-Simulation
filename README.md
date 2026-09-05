# qpsim

Nonequilibrium superconductor kinetics framework. Solves the Keldysh
kinetic equation for the quasiparticle distribution f(E, r) in the
isotropic dirty limit, resolved in energy and space, coupled to a local
phonon bath with acoustic escape to the substrate.

## Status

The diffusion and spatial backends, the local phonon bath, the M25
rate-equation service, and the Region/Junction/Device/Qubit composition
layer are shipped and numerically tested. Fischer 2023/2024 paper-topology
runs pin against qpsim-generated CSV/PDF baselines under `validation/`.
Those pins test solver certificates, artifact provenance/currentness,
formula helpers, and qpsim-to-qpsim regression. A separate independent
paper-data oracle now checks Fischer-2023 Fig. 6: the digitized dashed
analytic controls agree, while all three solid numerical traces show a
diagnostic 33–39% maximum relative mismatch over seven points on the
visible rising branch (`T*/Delta ≈ 0.250–0.410`). That result is not yet a
release gate because parameter and discretization uncertainty remain
unbounded. The M25 rate-equation engine and its junction UI mode are
covered by unit and integration tests under `tests/`. See
`docs/STATUS.md` for the running status tracker and test count.

### Capabilities

| Capability                                        | Status |
|---------------------------------------------------|--------|
| Energy- and space-resolved kinetics               | ✅ diffusion and spatial backends |
| Local phonon bath with acoustic escape            | ✅ model decisions D1–D5: `docs/Phonon_Model_Decisions.md` |
| Fischer paper-topology numerical regressions      | ⚠️ Round-8 current-source Figs. 3/5/6/7 complete; Fig. 6 has a provenance-bound paper-data diagnostic mismatch; Figs. 9–13 remain quarantined |
| Audit chain                                       | ✅ current-tree default and bounded non-manual slow gates green locally; hosted CI is separate post-push evidence |
| Kaplan phonon-bath characterization               | ✅ |
| M25 junction engine (rate eq + PDE)               | ✅ engine and UI covered by unit tests |

## Install

```bash
pip install -e ".[dev]"
```

Python 3.13+.

## Frontend

A local web UI drives the shipped surfaces — quasiparticle kinetics on
a geometry (a single cell, a strip, or a full 2-D mask built from a
rectangle, a GDS layer or polygons) and the M25 junction sweep — with a
materials browser, background runs with progress/cancel, and
server-rendered plots + CSV export:

```bash
pip install -e ".[ui]"
qpsim-ui
```

See `docs/Frontend.md` for the design and API.

## Layout

- `qpsim/` — the library (physics, collisions, solvers, services,
  devices, observables, materials, grids, backends, phonon models;
  `qpsim/webui/` is the optional web frontend)
- `docs/` — physics and numerics references; phonon-sector decisions
- `validation/` — analytic checks, diffusion-operator benchmarks, Fischer
  2023/2024 paper-topology numerical/artifact audits,
  qpsim-generated pinned CSV/PDF baselines, and independent paper-data
  oracles under `validation/paper_data/`
- `tests/` — pytest suite mirroring the library layout
- `scripts/` — user-facing run scripts

## License

MIT. See `LICENSE`.
