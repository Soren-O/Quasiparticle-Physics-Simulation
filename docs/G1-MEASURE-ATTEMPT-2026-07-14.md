# G1 collision-measure unification — attempted and rejected (2026-07-14)

**Status: HISTORICAL REJECTED EXPERIMENT; ROOT CAUSE RESOLVED 2026-07-16.** Do
not merge commit `5408b7b` itself. It changed only one leg of a coupled discrete
measure. The matched finite-volume repair now in the audit working tree changes
capacity, transition density, coherence factors, photon drive, phonon-side line
measure, Jacobians, support, transport, remap, and observables together, and is
gated by a reduced driven regression.

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

The historical branch added `SpectralContext.cell_weights` (exact
`∫_cell ρ dE` for pure-BCS covering
the gap; midpoint fallback for Dynes / gap-uncovered grids) and swapped the
phonon scattering + recombination rates and both Jacobians (Newton analytic +
n_ph cross-term) from `ρ·dE` to `cell_weights`. Photon channels and phonon-side
source terms were left point-sampled. That was the defect: although a photon
partner lookup has no explicit quadrature sum, its transition density must be
the cell-average density matched to the target cell's conserved capacity.

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

That A/B test correctly isolated the historical patch, but its original
interpretation was incomplete: exact weights do not intrinsically over-weight
physical loss. The patch mixed two discrete measures.

For a uniform energy lattice define the represented QP capacity and its
cell-average density by

```
w_i = integral_cell_i rho(E) dE
rho_bar_i = w_i / dE
```

A number-conserving fixed-mode transition requires the weighted event flux to
be symmetric: `w_i T_ij = w_j T_ji`. The legacy midpoint scheme happened to
satisfy this because `w_i^mid = rho_i*dE` and `T_ij` used `rho_j`. Commit
`5408b7b` changed `w_i` to the exact integral but retained `T_ij proportional
to rho_j`, so generally `w_i*rho_j != w_j*rho_i`. The driven photon channel
therefore became a spurious net drain. The same mismatch existed between the
QP event measure `w_i*w_j` and the phonon-side line measure
`dE*rho_i*rho_j`.

The matched repair uses `rho_bar` on photon partner legs and in phonon line
sources/Jacobians, giving

```
w_i * rho_bar_j = w_i*w_j/dE = w_j * rho_bar_i
dE * rho_bar_i * rho_bar_j = w_i*w_j/dE
```

after accounting for the phonon-bin energy width. Pure-BCS coherence factors
are also averaged under the same product measure. If
`r_i = integral_cell_i N2(E)dE / w_i`, then
`Kbar_plus/minus[i,j] = 1 plus/minus r_i*r_j`. Remaining smooth energy and
frequency factors use the existing cell-center mass-lumped quadrature.

## Resolution evidence (2026-07-16)

- On an 81-cell Fischer Fig. 3 proxy with thermal phonons, legacy midpoint gave
  peak `7.6458e-11`; the rejected exact-QP/point-photon hybrid gave
  `1.5204e-17`; the matched finite-volume operator gives about `9.69e-11`.
- At `tau_l/tau_0^PB = 0.1`, the matched reduced solve retains a larger
  bottleneck state (currently about `2.60e-10` with the stricter Picard
  certificate), rather than becoming byte-identical to the ratio-zero curve.
- A six-cell adversarial drive has exact-capacity number drift below `2e-15`;
  recreating the historical hybrid produces drift above `1e-3` (about
  `4.2e-2` in the manufactured state).
- QP scattering energy loss and phonon energy creation agree to roundoff on a
  gap-cut cell, and analytical QP/phonon Jacobians agree with finite
  differences.
- `validation/fischer_2023/test_fig3_finite_volume_reduced.py` is a fast driven
  branch regression, so thermal detailed balance can no longer provide false
  confidence by itself.

## Lessons

1. **Fast suite + detailed balance are NOT sufficient to validate a
   collision-measure change.** The failure lives entirely in the slow, driven,
   self-consistent reproductions. Any future collision-operator change must be
   gated on the slow Fischer figure reproductions (or a fast reduced-grid proxy
   of the *driven* solve), not just unit + detailed-balance tests.
2. **A discrete measure is a cross-operator contract.** Changing conserved
   capacity without changing partner transition density, phonon delta-line
   measure, Jacobians, sources, support, remap, transport, and observables is not
   a conservative finite-volume discretization, even if one isolated operator
   passes a conservation test.

## For the physicist

- Start from `g1-measure-unification @ 5408b7b`; the diff is small and localized
  (`spectral.py` `cell_weights`, `phonon.py` scattering/recombination, the two
  Jacobians).
- Reproduce with `PYTHONUTF8=1 python -m validation.fischer_2023.fig6_paper` (or
  the fast `test_fig6_paper_eq53.py`) and the A/B monkeypatch above.
- Use the historical commit only to reproduce the mixed-measure failure. The
  repair must retain the algebraic identities above and the driven regression;
  a local `rho*dE -> cell_weights` edit is still prohibited.
- Full-grid paper validation remains a separate release gate. The reduced
  regression proves that this numerical-measure repair no longer destroys the
  driven branch; it does not by itself establish paper agreement.
