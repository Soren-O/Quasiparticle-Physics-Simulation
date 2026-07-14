# G1 collision-measure unification — attempted and rejected (2026-07-14)

**Status: DO NOT MERGE. The naive exact-cell-weight swap is physically wrong in
the driven solve.** Code preserved on branch `g1-measure-unification` (commit
`5408b7b`, parent `562c1f4` on `audit-fixes-2026-07-14`); a physicist should
diagnose the root cause before any re-attempt.

## The finding (G1)

The QP-side phonon collision operators (`collisions/phonon.py`) contract with the
midpoint DOS measure `ρ·dE`, while the observables (`observables/density.py`,
`qp_fraction`) integrate the exact singular BCS DOS per cell
(`bcs_dos_cell_weights = ∫_cell ρ dE`). Because the gap edge is sqrt-integrable,
midpoint converges only ~1/√NE there, so a strictly number-conserving scattering
relaxation drifts the *reported* `x_qp` by ~10–15% at the default `NE=400`
low-T grid. The GPT-audit reconciliation rated this the highest-value finding and
recommended "route both onto one SpectralContext cell-weight vector."

## What was implemented

Added `SpectralContext.cell_weights` (exact `∫_cell ρ dE` for pure-BCS covering
the gap; midpoint fallback for Dynes / gap-uncovered grids) and swapped the
phonon scattering + recombination rates and both Jacobians (Newton analytic +
n_ph cross-term) from `ρ·dE` to `cell_weights`. Photon channels are point-partner
(no `dE` quadrature) and were correctly left untouched.

## What validated (and gave false confidence)

- **Detailed balance preserved**: `max|df/dt|/scale = 4.8e-11` at the thermal
  Fermi–Dirac fixed point. (This is *measure-agnostic by KMS* — the thermal
  balance holds term-by-term for any consistent measure, so it could never have
  caught the bug.)
- **Conservation "fixed"**: scattering conserves the exact/observable number to
  `6e-17` (was 2.34% drift) — the stated goal of G1.
- **Jacobian consistent**: analytic vs finite-difference `3e-9`.
- **Fast suite: 1060 passed.** But the Fischer figure reproductions are
  **slow-marked** and not in the fast suite, so this proved nothing about them.

## What the physics review caught (decisive)

A 5-agent Fable physics review compared the G1 numerical output against the
papers' closed-form analytic overlays (Fischer Eq. 47 / 53 / envelope Eqs. 24–51,
which are *G1-invariant* and reproduce identically) and the pre-G1 baselines:

| Reproduction | Verdict | G1 vs paper analytic |
|---|---|---|
| Fischer Fig. 3 f(E) | **degraded** | 6–8 orders below the envelope; **all four τ_l legend curves become byte-identical** — the drive/τ_l dependence vanishes |
| Fischer Fig. 5 x_qp | **degraded** | 5–14 orders below Eq. 47; x_qp *decreases* with drive (unphysical); pre-G1 tracked it to <1.5× |
| Fischer Fig. 6 gap suppression | **degraded** | gap suppression *disappears* (`Δ_driven = 180.0000`, no suppression); x_qp 4–6 orders below the thermal floor |
| Marchegiani M25 Fig. 3 | **unaffected** | roundoff-only (`≤1e-13`) — M25 uses the moment-based `rate_equation` service, not `phonon_collision_rates` |

(The Fig. 7 agent timed out mid-run; Q_i tracks the collapsed x_qp so it would
degrade too.)

## Root-cause evidence (the A/B isolation)

The Fig. 6 agent's smoking gun: **on the G1 tree, monkeypatching
`SpectralContext.cell_weights` back to midpoint `ρ·dE` recovers the physical
suppressed gap** (`solve_gap` → `Δ = 179.99999992 µeV`, matching the CSV), while
the exact weights leave `Δ = 180` (no suppression). So the measure change *is* the
cause, isolated to the driven collision balance.

**Physics diagnosis:** the exact singular cell weight at the sqrt-singular first
cell is ~1.4× larger than `ρ·dE`. This is harmless at the thermal fixed point
(detailed balance is measure-agnostic), but in the **driven, self-consistent-gap,
dynamic-n_ph solve** it over-weights recombination/loss at the gap edge relative
to the generation/drive, collapsing the driven QP population to a spurious
near-empty fixed point 4–7 orders below even the thermal floor. The Fig. 3
signature — all τ_l curves collapsing onto one — indicates the drive/source term
effectively stops entering the balance under the exact measure. This is a real
physics interaction, not a solver glitch (verified via cold- and warm-seeded
single-point solves and the module `run()` paths).

## Lessons

1. **Fast suite + detailed balance are NOT sufficient to validate a
   collision-measure change.** The failure lives entirely in the slow, driven,
   self-consistent reproductions. Any future collision-operator change must be
   gated on the slow Fischer figure reproductions (or a fast reduced-grid proxy
   of the *driven* solve), not just unit + detailed-balance tests.
2. **The G1 "just swap ρ·dE → exact cell weights" recommendation was too naive.**
   Making the collision *conserve* the exact number is not the same as making the
   *driven balance* correct. The exact singular edge weight must be reconciled
   with the drive/generation and recombination terms consistently — likely the
   center-evaluated kernel and the singular measure interact and need a matched
   treatment (or the observable/collision inconsistency should be resolved a
   different way, e.g. a documented convergence budget rather than an
   edge-over-weighting measure swap).

## For the physicist

- Start from `g1-measure-unification @ 5408b7b`; the diff is small and localized
  (`spectral.py` `cell_weights`, `phonon.py` scattering/recombination, the two
  Jacobians).
- Reproduce with `PYTHONUTF8=1 python -m validation.fischer_2023.fig6_paper` (or
  the fast `test_fig6_paper_eq53.py`) and the A/B monkeypatch above.
- The question to resolve: how should the exact gap-edge DOS weight enter the
  *driven* scattering/recombination balance so the driven branch survives and the
  reproductions stay faithful, while still removing the collision-vs-observable
  number inconsistency G1 identified?
- Until then, `qpsim` ships with the documented (midpoint-collision) behavior; the
  G1 inconsistency is a known ~1/√NE convergence budget, not a shipped bug.
