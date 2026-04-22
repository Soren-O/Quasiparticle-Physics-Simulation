# qpsim

Nonequilibrium superconductor kinetics framework. Solves the Keldysh
kinetic equations over a hierarchy of specializations — from the full
two-component (T1) description, through the scalar-kinetic-with-angle
form (T2), down to the isotropic dirty-limit diffusion form (T3).
Phonon dynamics are orthogonal to the electronic tier: Ph0 (local with
escape), Ph1 (ballistic), Ph2 (diffusive substrate).

## Status

**Gate 1 skeleton — no physics implemented yet.** The reference
implementation that this repo supersedes lives at
`~/Documents/Quasiparticle Simulation/Active Code/qpsim/` and stays
frozen as the source of parity baselines. Build plan at
`~/Documents/Quasiparticle Simulation/Documentation/Current/New Framework Plan.md`.

### Gate roadmap (New Framework Plan §7)

| Gate | Deliverable                                 | Effort |
|------|---------------------------------------------|--------|
| 0    | Phonon physics decisions                    | ✓ committed |
| 1    | Repo skeleton                               | ✓ this commit |
| 2    | Ported physics + T3 diffusion backend       | 5 d    |
| 3    | Fig. 3 τ_l=0 bit-identical vs baseline      | 3 d    |
| 3.5  | Generate missing parity baselines (old repo)| 2 d    |
| 4    | Full Layer-4 audit chain passes             | 7 d    |
| 5    | Ph1 phonon spatial transport                | 10 d   |
| 6    | T2 kinetic scalar backend                   | 14 d   |
| 7    | T1 two-component backend (new derivation)   | 21 d   |
| 8    | Marchegiani junction (rate eq + PDE)        | 14 d   |

## Install

```bash
pip install -e ".[dev]"
```

Python 3.13+.

## Layout

- `qpsim/` — the library (see New Framework Plan §5)
- `docs/` — physics and numerics references; Gate 0 phonon decisions
- `validation/` — tier-reduction tests, paper-reproduction audit
- `tests/` — pytest suite mirroring the library layout
- `scripts/` — user-facing run scripts

## License

MIT. See `LICENSE`.
