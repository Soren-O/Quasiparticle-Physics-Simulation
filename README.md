# qpsim

Nonequilibrium superconductor kinetics framework. Solves the Keldysh
kinetic equations over a hierarchy of specializations — from the full
two-component (T1) description, through the scalar-kinetic-with-angle
form (T2), down to the isotropic dirty-limit diffusion form (T3).
Phonon dynamics are orthogonal to the electronic tier: Ph0 (local with
escape), Ph1 (ballistic), Ph2 (diffusive substrate).

## Status

T3 + Ph0 backend, the M25 (Marchegiani–Catelani 2025) rate-equation
service, and the Region/Junction/Device/Qubit composition layer are
shipped and numerically tested. Fischer 2023/2024 and
Marchegiani–Catelani 2025 paper-topology runs pin against
qpsim-generated CSV/PDF baselines under `validation/`. Those pins test
solver certificates, artifact provenance/currentness, formula helpers,
and qpsim-to-qpsim regression. A separate independent paper-data oracle
now checks Fischer-2023 Fig. 6: the digitized dashed analytic controls
agree, while all three solid numerical traces show a diagnostic 33–39%
maximum relative mismatch over seven points on the visible rising branch
(`T*/Delta ≈ 0.250–0.410`). That result is not yet a release gate
because parameter and discretization uncertainty remain unbounded.
T2/T1 backends and Ph1/Ph2 phonon transport are not implemented. See
`docs/STATUS.md` for the running gate tracker and test count.

### Gate roadmap

| Gate | Deliverable                                   | Status |
|------|-----------------------------------------------|--------|
| 0    | Phonon-model decisions (D1–D5)                | ✅ `docs/Phonon_Model_Decisions.md` |
| 1    | Repo skeleton                                 | ✅ |
| 2    | Ported physics + T3 diffusion backend         | ✅ |
| 3    | Fischer paper-topology numerical regressions  | ⚠️ Round-8 current-source Figs. 3/5/6/7 complete; Fig. 6 has a provenance-bound paper-data diagnostic mismatch; Figs. 9–13 remain quarantined |
| 4    | Layer-4 audit chain (1e-12 / 1e-6 / 1e-4)     | ✅ current-tree default and bounded non-manual slow gates green locally; hosted CI is separate post-push evidence |
| 4.5  | Characterization tier (Ph0-Kaplan)            | ✅ |
| 5    | Ph1 phonon spatial transport                  | ❌ not started |
| 6    | T2 kinetic scalar backend                     | ❌ not started |
| 7    | T1 two-component backend                      | ❌ not started |
| 8    | Marchegiani junction (rate eq + PDE)          | ✅ via Device Architecture composition layer |

## Install

```bash
pip install -e ".[dev]"
```

Python 3.13+.

## Frontend

A local web UI drives the shipped surfaces (0-D steady state and
transients, the 1D spatial strip, the M25 junction sweep) with a
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
- `validation/` — analytic checks, tier reductions, paper-topology
  numerical/artifact audits (Fischer 2023/2024, Marchegiani 2025),
  qpsim-generated pinned CSV/PDF baselines, and independent paper-data
  oracles under `validation/paper_data/`
- `tests/` — pytest suite mirroring the library layout
- `scripts/` — user-facing run scripts

## License

MIT. See `LICENSE`.
