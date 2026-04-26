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
shipped and validated. Eight Fischer 2023/2024 reproductions and three
Marchegiani 2025 figures pin against self-checked CSV baselines under
`validation/`. T2/T1 backends and Ph1/Ph2 phonon transport are not
implemented. See `docs/STATUS.md` for the running gate tracker and
test count.

### Gate roadmap

| Gate | Deliverable                                   | Status |
|------|-----------------------------------------------|--------|
| 0    | Phonon-model decisions (D1–D5)                | ✅ `docs/Phonon_Model_Decisions.md` |
| 1    | Repo skeleton                                 | ✅ |
| 2    | Ported physics + T3 diffusion backend         | ✅ |
| 3    | Fischer paper-reproduction parity (8 figures) | ✅ |
| 4    | Layer-4 audit chain (1e-12 / 1e-6 / 1e-4)     | ✅ |
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

## Layout

- `qpsim/` — the library (physics, collisions, solvers, services,
  devices, observables, materials, grids, backends, phonon models)
- `docs/` — physics and numerics references; phonon-sector decisions
- `validation/` — analytic checks, tier reductions, paper-reproduction
  audit (Fischer 2023/2024, Marchegiani 2025), pinned CSV/PDF baselines
- `tests/` — pytest suite mirroring the library layout
- `scripts/` — user-facing run scripts

## License

MIT. See `LICENSE`.
