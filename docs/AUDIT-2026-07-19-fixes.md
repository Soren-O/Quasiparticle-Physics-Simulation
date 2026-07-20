# 2026-07-19 deep audit — findings and applied fixes

Branch `codex/audit-fixes-2026-07-19` (on top of `codex/qpsim-deep-audit-fixes`
@ `03ee1df`). Audit method: 123-agent multi-stage workflow — 16 independent
finders, dedup, two-lens adversarial verification per finding
(refute-by-execution + reachability/intent), completeness critic, gap finders.
Tally: 64 confirmed (5 high / 19 medium / 40 low), 3 plausible, 5 refuted.
The audited tree's gates were independently reproduced before fixing (fast
1549 passed; slow gate 14 passed + 2 documented xfails; ruff/mypy clean; CI
run 29667989929 green).

Headline: the core engine survived execution-based attack (matched measure,
detailed balance ~4e-14, Jacobians to FD precision, photon-channel absolute
normalization newly anchored to Mattis–Bardeen, certificates hard-gating,
robustness and FP hygiene strong), and `CURRENT-STATUS.md` verified accurate
on every mechanically checkable claim. The real defects sat one layer out:
`scripts/` campaign drivers (zero prior audit coverage), the paper-anchored
analytic-overlay layer, and the M25 coefficient layer.

## High-severity findings — all fixed on this branch

| # | Finding | Fix commit |
|---|---|---|
| H1 | Prelim finite-phonon runner used legacy QP-side kernels in the PHONON equation (phonon rates 4–17× low; every number in `prelim_experiment_simulation_notes.md` inherits it; δf_r 65%, Q_i −40%, n_ph ×307 at the nominal case). Equilibrium tests structurally blind (both variants thermally exact). | `80499b2` — runner now builds phonon-side kernels from `material.tau_0_pb_ns`; notes doc carries a STALE banner pending regeneration (hours of compute, deliberately not run here). |
| H2 | Readout-heating scripts crashed on every driven case: this branch made the sub-gap kernel raise on >1%-of-a-cell incommensurate ω, and the 5.142857 GHz mode is 1.64% off at NE=101 (81/108 overnight cases silently `status=failed`; old main-branch numbers used a silently snapped ω). | `cb06e29` — explicit snap-to-harmonic at the drive boundary with recorded nominal/used/shift, corrected false metadata claim, real-kernel regression tests (the old test monkeypatched the kernel). Smoke preset verified end-to-end. |
| H3 | Fig. 6 "Eq. 53" stored overlay fed qpsim x_qp = N/(4ρΔ₀) into the paper formula requiring N/(2ρΔ₀): pinned `paper_observable_eq53` exactly ×2 too small (off the paper's chart window), entrenched by its own pin test. Same family: F24 Fig. 8 axis. | `3ad904e` — helpers own the ×2 conversion; CSV column regenerated closed-form (old column reconstruction verified bit-exact first); chain pins doubled; fig8 plots paper convention with honest axis; dashed-curve τ_l=2τ₀ᴾᴮ deviation now labeled, not silent. |
| H4 | `_tau_R_inverse` evaluated the T≪ω_LR series at T/ω_LR up to 6.25 (shipped coefficient 0.86×→0.20× vs the exact S48/S49 integral); M25 pins and the junction anchor were calibrated to the wrong coefficient. | `57a964a` — exact I(a,b) quadrature (erfcx-normalized, stable at all T) is now the evaluation path; series kept as in-domain test reference with regression guards; M25 baselines re-pinned. Corrected Fig. 4a dip 1505 Hz, dip/10 mK = 0.902 — inside the paper's 0.85–0.91 band (was 0.936). |

## Medium fixes

- **Validation honesty** (`f8bf359`): canonical-baseline existence manifest
  (19 tests silently skipped on missing CSVs); CI slow-step existence guard;
  gap-suppression test pins the 99.9% collapse magnitude; fabricated
  "M25 Fig 3(a) ≈70 mK" paper anchor removed; `TestReferenceValue` rewritten
  at the correct per-Cooper-pair scale (was ensemble-scale inputs in a
  3.7-decade window); prelim spatial overnight no longer marks non-converged
  runs `completed` (which also blocked resume); convergence-checks script now
  applies an explicit 2% acceptance budget and exits nonzero; AUDIT-2026-07-15
  Fig. 7 certificate maxima corrected to the promoted canonical's values;
  stale "~14 h" Fig. 6 estimates replaced with measured 6.04 h serial.
- **Core contracts** (`347af41`): `SpectralContext._rebuild` exception-safe
  (was torn-state on failure); `GapBelowGridSupportError` classifies a solved
  below-support gap and T3 folds it into `SelfConsistentGapCollapseError`
  (the designed collapse→NaN chain was dead code on every physical grid — a
  genuine collapse aborted sweeps unclassified); near-T_c resolution guard
  (silent +4.7% gap error at dE=0.31Δ_eq now warns); pair-breaking photon
  partners above the grid top fail loud (silent skip lost ~42% of pair
  generation in a realistic case).

## F24 fingerprint rebinds

Artifact source hashes were rebound (rows/certificates untouched) for the
presentation-only fig8 change and for the three core-module edits. Evidence
of physics-neutrality: the full `fischer_2024` suite including the slow
live-recompute pins passes (63/63 with `-m ""`). Live full regeneration was
deliberately NOT used to replace pinned rows: it reproduces them only to
~1e-14 on this environment (documented OS/env envelope drift), and the pins
are the certified artifacts.

## Deliberately not done here (needs decisions or heavy compute)

1. **Prelim campaign regeneration** (H1/H2 downstream): rerun the 7 mK /
   temp / convergence / readout sweeps with the corrected runner and rewrite
   `prelim_experiment_simulation_notes.md`. Hours of compute; the notes are
   banner-flagged STALE until then.
2. **Fig. 6 canonical PDF** regen (legend label for the dashed overlay lands
   with the next full sweep; the stored-column fix does not alter the shipped
   PDF, whose dashed curves are plot-time re-derived).
3. **Plausible/human-physics calls — adjudicated 2026-07-20:**
   (a) moving-gap rising-gap (recovery) asymmetry → RESOLVED permissive:
   the persistent path now tolerates a stranded finite-E_max tail up to the
   sibling's 1e-3 fraction (warn above 1e-9) instead of refusing recovery at
   ~5e-12; the tail stays frozen in the persistent representation (zero
   overlap, collisionless) and re-enters if the gap falls — strictly better
   conservation than the sibling's irreversible top-cell deposit, and
   bitwise-identical for all previously-green trajectories.
   (b) marchegiani win32-only strict pins → ACCEPTED as-is and documented
   (CI exercises the 1e-3 fallback; strict 1e-6 runs on the Windows dev
   machine; Linux-stamped twin declined for now).
   (c) Fig. 5 low-drive atol vacuousness → deferred to the Fig. 5
   regeneration campaign, with signal-scaled tolerances recorded as a
   required part of that re-pin (comment at the gate + NEXT-AUDIT-BRIEF).
4. Pre-existing known-open items (Fig. 5 regeneration, Fig. 6 canonical
   sweep, Figs. 9–13 refinement) — unchanged.

## Full audit record

Complete findings (incl. 40 lows and 5 newly refuted false positives):
machine-readable results in the audit session; human summary at
`C:\tmp\qpsim-audit-2026-07-19-fable-report.md` (off-repo).
