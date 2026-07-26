# 2026-07-19 deep audit — findings and applied fixes

Branch `codex/audit-fixes-2026-07-19` (on top of `codex/qpsim-deep-audit-fixes`
@ `03ee1df`). Audit method: 123-agent multi-stage workflow — 16 independent
finders, dedup, two-lens adversarial verification per finding
(refute-by-execution + reachability/intent), completeness critic, gap finders.
Tally: 64 confirmed items (5 high / 19 medium / 40 low), 3 plausible,
5 refuted. **Errata (2026-07-20 external review):** the confirmed tally
double-counts duplicate roots that entered after the dedup stage (the Fig. 6
factor-2 finding twice at high; the dashed-curve τ_l finding twice at medium;
the M25 S50 domain issue at both high and low) — there are **four distinct
high-severity defects**, as the fix table below records. The "19
missing-baseline tests" figure was an estimate; the existence manifest
guards the 18 committed canonical CSVs. The Fig. 3 amplitude-certificate
criticism is narrower than filed: the certificate itself is not
amplitude-bound, but a fast reduced-grid test does pin a nonzero physical
amplitude. The machine-readable findings are committed at
[`audit-2026-07-19-findings.json`](audit-2026-07-19-findings.json).
The audited tree's gates were independently reproduced before fixing (fast
1549 passed; slow gate 14 passed + 2 documented xfails; ruff/mypy clean; CI
run 29667989929 green).

**Superseded headline for the original 2026-07-19 tree:** the core engine
appeared to survive execution-based attack (matched measure,
detailed balance ~4e-14, Jacobians to FD precision, photon-channel absolute
normalization newly anchored to Mattis–Bardeen, certificates hard-gating,
robustness and FP hygiene strong), and `CURRENT-STATUS.md` verified accurate
on every mechanically checkable claim. The real defects sat one layer out:
`scripts/` campaign drivers (zero prior audit coverage), the paper-anchored
analytic-overlay layer, and the M25 coefficient layer. **Round 5 overturned
the core-engine part of that conclusion** by reproducing an amplitude-blind
acceptance in `newton_solve_f`; retain this paragraph only as audit history.

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

## Historical F24 fingerprint rewrites (not execution provenance)

Artifact source hashes were rebound (rows/certificates untouched) for the
presentation-only fig8 change and for the three core-module edits. Evidence
of physics-neutrality: the full `fischer_2024` suite including the slow
live-recompute pins passes (63/63 with `-m ""`). Live full regeneration was
deliberately NOT used to replace pinned rows: it reproduces them only to
~1e-14 on this environment (documented OS/env envelope drift), and the pins
are the certified artifacts. This paragraph records what was done historically;
the later provenance audit rejected the interpretation that such a metadata
rewrite makes an artifact current. Exact regeneration is required for
summary-only F24 artifacts after a contract change. A full-state artifact may
be re-certified under later equations only if its original producer identity
is retained separately from the validator identity.

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

## 2026-07-20 external-review round (GPT 5.6 Sol)

An independent external review of the fix branch confirmed the core H1–H4
diagnoses and the H3 correction, and filed six findings. Adjudication:

- **CONFIRMED + fixed:** (1) hosted CI flake — PR #5's docs-only run
  29706352205 breached the Fig. 7 near-zero loss floor by 6% on a Linux
  3.14 host with identical code/library versions (host-to-host jitter with
  zero floor margin); `QP_LOSS_REGRESSION_ATOL` recalibrated 2e-19 → 1e-18
  with the measured evidence inline. (2) Collapse conflation — the
  2026-07-19 repair folded a POSITIVE superconducting root below the grid
  face (grid under-resolution; reproduced root 155.85 μeV under a
  162.06 μeV face) into the collapse/NaN path;
  `GapBelowGridSupportError.candidate_gap` now distinguishes the cases and
  only the genuine normal-state decision (0.0) classifies as collapse.
  (3) Pair-breaking lower boundary — partners in the physical window
  `(Δ, first_face)` on grids starting above Δ were still silently dropped
  (reproduced 26.8% loss); a mirror fail-loud guard closes it, with focused
  regression tests for BOTH guards. (4) Campaign resume safety — run ids
  now carry a `_PHYSICS_REV` token (bumped to rev2 for the H1/H2 physics
  change) so stale rows are never accepted as complete; `--no-resume`
  truncates aggregate CSVs; "resume-safe" doc claims qualified. Plus the
  H1 kernel-wiring regression test the fix round lacked. (5) The confirmed
  Windows `sweep_cache.store()` failure mode is now fixed (a failed cache
  write warns and returns the computed payload instead of destroying an
  hours-long solve; the provenance sidecar is written atomically).
  (6) `Moving_Gap_Time_Integration.md` updated to the adopted 1e-3 frozen-
  tail contract with the number-vs-energy caveat stated.
- **`(Δ_R/Δ̄)³` normalization — reviewer RIGHT, first adjudication WRONG
  (corrected in the follow-up round):** the paper's equation source reads
  `r^{R>} ≃ r^{<>} ≃ r^{R<} ≃ 8π b_R Δ̄³` (average gap; MathML alt text
  verified). The first-pass refutation relied on a text extraction that
  truncated D.3 exactly before the `Δ̄³` tail, compounded by this repo's
  coefficient doc having transcribed `Δ_R³`. The conversion now carries
  `(Δ_R/Δ̄)³` (≈0.985 Fig 3a, ≈0.861 Fig 3b), the M25 baselines were
  re-pinned again, the coefficient doc's transcription is fixed, and the
  absolute-normalization pin test pins the corrected values.
- **Stale at review time:** the "no clean committed revision" state was
  mid-session; the branch was subsequently committed and pushed, and a
  draft PR now provides hosted 3.13/3.14 CI.

## 2026-07-20 second external-review round (GPT 5.6 Sol, round 3)

Seven further findings, adjudicated and fixed on this branch:

- **HIGH, confirmed:** the multi-region device outer loop (undamped
  simultaneous updates, absolute 1e-8 tolerance) certified period-2
  orbits and any cold-temperature state — at 100 mK the whole occupation
  signal is ~8e-10, below the tolerance. Fixed with damped iteration
  (θ=0.5 default) plus scale-aware relative fixed-point-defect
  certification; the headline detailed-balance test now asserts 0.1%
  agreement on the resolved head (its former atol exceeded the entire
  signal), and the mismatched-T test's true physics (junction-dominated
  cold region, ~1e5× above its bath FD) surfaced and is now asserted.
- **Gap-cut sub-2Δ pairs, split adjudication:** the PHOTON pair block is
  hard-gated at the pair threshold (current semantics are strictly
  `ω > 2Δ` with a roundoff band; the initial repair used `ω ≥ 2Δ`). A
  commensurate 1.6Δ photon produced finite
  pair generation through cut-cell partners — unphysical, fixed, tested.
  The PHONON kernel case remains a DOCUMENTED ω-labeling approximation:
  a supported cut cell's pair rate is physical (capacity exists only
  ≥ Δ) and only its emitted-ω label is off by ≤ one dE, with
  emission/absorption sharing the bin (discrete detailed balance
  exact). Masking those pairs was implemented and reverted. **Erratum
  (round-4 review):** the ~21% Fig. 6 τ₀ᴾᴮ shift originally cited as
  the revert evidence was an artifact — the experimental mask broke the
  canonical-kernel detection behind the Kaplan endpoint correction, and
  the same shift reproduces on grids with zero subthreshold pairs. The
  principled revert reason stands (deletion removes physical rate); the
  proper fix is a rate-preserving gap-aware ω-remap of cut-cell pair
  events across QP rates, phonon rates, and all Jacobians (designed
  TODO, deferred).
- **Accepted photon-frequency snaps** are now disclosed (RuntimeWarning
  above 1e-6 bins) with the contract stated: occupancies chosen for the
  nominal ω (thermal Bose factors especially) must be evaluated at the
  snapped m·dE. No shipped caller pairs a thermal occupancy with a
  snapped ω today.
- **Fig. 7 dashed analytics rewritten to the paper's Eqs. 63 + 65**
  (verified against the arXiv math source): the old overlay used
  (Δ/T*)^{3/2} where Eq. 63 has power 3, and substituted an equilibrium
  expression for the driven Eq. 65 branch — dashed curves were off by up
  to ~6× at 0.30 K. Plot-time only; certified numerics untouched.
- **Eq. 47 trapping correction** (paper Eq. 112 leading order) added to
  the R̄ linear term in `_paper_envelope` and `fig5_paper` — the Fig. 6
  plot derivation already applied it; the overlays were 1.5–7.4% low.
  The fig6 CSV analytic columns were regenerated closed-form (input
  reconstruction verified to 4e-16 first) and the previously
  self-referential standalone-repro pins updated.
- **Campaign runners:** all five finite-phonon scripts now gate
  convergence on BOTH max|df/dt| and max|dn_ph/dt| (phonon residuals
  lagged up to 8.7×); the readout runner writes shift rows before the
  summary commit marker; appends refuse stale headers and re-write
  headers over zero-byte files.
- **Lows:** material YAML files now fold into `solve_source_digest`
  (a sound-velocity edit invalidates cached solves); the cache writes
  the provenance sidecar before promoting the payload (an accepted
  payload always has provenance); Fig. 7's fixed-Δ₀/single-grid
  convention is documented at `_build_grid` (~0.17% gap mismatch at
  0.34 K); the findings JSON carries schema-caveat metadata.
- **Deferred, documented:** WebUI run provenance, exhaustive public-API
  input validation, M25 stamp-disappearance fallback semantics, and a
  moving-gap recovery error-budget study.

## 2026-07-20 third external-review round (GPT 5.6 Sol, round 4)

- **HIGH, confirmed — deeper than the round-3 repair:** the device solver
  still falsely certified the COMMON mode (both regions at ``c*f_FD`` for
  any ``c``, defect exactly zero): the inner Newton's backward error was
  normalized by the large but exactly balanced junction exchange, and the
  frozen-flux Picard splitting pins each region to the other's old state
  (drainage ~ collision/exchange per iteration — practically never).
  Two-part repair: (1) the inner Newton certificate now normalizes by
  INTERNAL turnover only (external flux stays in the residual; flux-only
  systems keep the full normalization); (2) the outer loop certifies the
  GLOBAL conserved-number mode at the accepted states — pair-channel
  (number-changing) normalized, since number-conserving scattering
  buried the signal under e^{-Δ/kT} — with measured calibration
  (manifold 0.60 at any temperature, converged mismatched-T fixture
  1.0e-3, thermal ~1e-11; fixed 5% limit) and a LOUD refusal when the
  manifold is detected. Scoped to symmetric ratio-1 matched-weight
  junctions where bin-wise cancellation is provable; other junction sets
  get a once-per-solve warning that this certification gap remains open
  (needs per-junction number-flux accounting / a coupled solve).
  Common-mode regression tests assert the refusal; device controls are
  validated (finite tolerances, damping in (0,1]).
- **Fig. 7 dashed overlay Q_c:** Eq. 65 was fed Q_EXT (~1e6) as its
  coupling factor; the paper's Table 2 gives Q_c = 20100 (verified), with
  the effective coupling 1/Q_c + 1/Q_ext, and the −100 dBm curve uses the
  thermal-equilibrium expression (T*,0 ≈ ω₀). Rewritten with
  module-level, unit-tested helpers; the corrected values agree with the
  reviewer's independent computation EXACTLY at all four cross-checked
  points (19286/24308/47344/115950 at 0.30 K).
- **PB Jacobian gate:** the analytic Jacobian kept sub-2Δ pair
  derivatives the rates now zero (0.74% J-vs-FD mismatch on cut grids) —
  the same threshold gate was applied (subsequently tightened everywhere
  to strict `ω > 2Δ`). The round-3 regression test was VACUOUS
  (face-aligned gap, no supported cut cell; its quoted pre-fix value was
  wrong) — rewritten with a genuine cut grid (gap inside a cell,
  ω = 1.9 < 2Δ = 1.94 with supported cut-cell partners).
- **Threshold-crossing snaps:** both photon kernels now fail loud when an
  accepted snap would move ω across 2Δ (both directions reproduced by
  the reviewer; a warning is not sufficient when snapping decides whether
  a channel exists).
- **Campaign persistence:** re-runs now PURGE stale rows for the run id
  first (retry idempotence verified: two consecutive smoke runs leave
  exactly one attempt per id); the shifts schema comes from current code
  with a loud header check instead of adopting the on-disk header
  (which silently dropped new columns via extrasaction="ignore").
- **Erratum accepted:** the ~21% τ₀ᴾᴮ masking penalty cited in round 3
  was an artifact (the experimental mask broke the canonical-kernel
  detection behind the Kaplan endpoint correction; same shift on grids
  with zero subthreshold pairs). Corrected at the kernel docstring, the
  tests, and the round-3 record; the principled revert reason (deletion
  removes physical rate) stands, and the rate-preserving gap-aware
  ω-remap remains the designed TODO.
- **Still open, documented:** sub-2Δ dynamic-phonon remap (above); cache
  payload/sidecar pair atomicity (sidecar-first prevents provenance-less
  payloads; a same-key stale-pair window remains); the mismatched-T
  representative case needs non-default budgets (~1200 iterations);
  WebUI provenance and exhaustive API validation.

## 2026-07-21 fourth external-review round (GPT 5.6 Sol, round 5)

Six findings were confirmed.  The bullets below describe the **initial
`cd4fb81` repair**, not the final state: a same-day follow-up reproduced
additional false acceptances and false rejections in the new certificates,
plus incomplete resume and threshold guards.  The follow-up adjudication and
repairs are recorded in the next section; do not cite this section alone as
evidence that Round 5 was closed.

- **HIGH — the core standalone solver was amplitude-blind at cold
  temperatures:** `newton_solve_f` certified `c·f_FD` unchanged at
  50–80 mK even with `tol=1e-30` — number-conserving scattering
  dominates every aggregate metric while the number-changing pair
  channel scales as e^{−2Δ/kT} (numerically unresolvable below ~80 mK in
  float64). A dedicated conserved-QP-number certificate (full number
  residual over pair-channel turnover, 1% limit) rejected the original
  wrong-number counterexamples. Its initial channel decomposition and scope
  were incomplete; see the follow-up below.
- **HIGH — initial repair for three device-certificate holes:** the threshold
  initially scaled as `min(0.05, 1e3·outer_tol)` (a fixed detector accepted
  5%-wrong modes at any tolerance); the certified sum is COLLISION-ONLY
  (adding ~1e-12 balanced junction terms to a ~1e-38 cold collision
  residual absorbed it in floating point — the exchange cancels
  analytically, so it is omitted, not computed); certification is per
  CONNECTED conservation component (one signed device-wide sum let
  disconnected components cancel to ~4e-30). All three reviewer
  counterexamples are regression tests.
- Finite-phonon device solves were no longer checked with a thermal
  surrogate, but `cd4fb81` merely warned and still returned an uncertified
  solution. Same-family junction misconfiguration
  (symmetric with ratio ≠ 1 / mismatched weights) now REFUSES instead of
  warn-then-answer.
- **2Δ threshold became strictly-above with a roundoff margin in the rates,
  crossing guard, and analytic Jacobian:** the K⁻ pair rate is zero AT
  equality; the shared `pair_channel_open` predicate governs all three paths.
  Before the repair, a gap-cut grid emitted finite generation at exact
  threshold and a snap-to-threshold case came out ~7.4× large.
- **Resume completion received an initial presence check:** a `completed`
  summary row also needed at least one shift row and paths to existing
  trace/profile artifacts. The follow-up found that partial/malformed shift
  sets, blank references, directories, and zero-byte artifacts still passed.
- **The Kaplan-correction detector ignores zero-capacity entries** (they
  cannot affect rates, so they can no longer disable the correction —
  the same fragility that produced the round-3 "21%" artifact).

## 2026-07-21–22 Round-6 repair (working tree)

The read-only follow-up against exact head `cd4fb81` reproduced the initial
certificate as unsound in both directions and found incomplete campaign and
threshold guards. A further adversarial pass then exercised the coupled
finite-phonon solver, spatial finite-volume measure, campaign resume
semantics, and low-temperature special functions. The following repair is in
the **uncommitted working tree based on `cd4fb81`**. An earlier
pre-regeneration partitioned snapshot produced **1830 passes** and exactly
five expected stale-provenance failures (Fig. 7 plus four F24 artifacts).
After the final code changes and real regeneration, the authoritative complete
default collection produced **1866 passes, 17 intentional deselections, and
13 warnings in 657.40 s**. Ruff, mypy over 75 qpsim source files, bytecode
compilation, and `git diff --check` are clean.

The four F24 artifacts were regenerated through their real solve paths; the
parallel wall time was 57 s (the longest individual solve), their four strict
promoted-currentness tests pass, and four live solve comparisons pass in
87.74 s. Metadata-only rebinding was not used. Fig. 7 was then regenerated
from 48 independent fresh thermal seeds under six isolated workers, cache
disabled, with solve-source
and downstream-observable digests checked before and after the sweep. The
point solves consumed 13582.0 aggregate worker/solver-seconds and 3421.3 s
wall time; the hardest `−68 dBm / 0.06 K` point took 3415.6 s. All 48 temporary
payloads were read back with exact axes/digests before a candidate CSV round trip and
atomic per-file CSV and PDF promotion. The Round-6 solve-contract digest was
`71d2730e43b41fba106e3066de156eb6d4f69a23ea4aac91c64de6bd552d0503`
(downstream digest
`b784cdc66a17acf5aa032900346bd3127e71ac53fea3f6c8d00c70f68583fc8a`);
certificate maxima are `3.7007192e-10` (QP backward error),
`9.8294188e-9` (phonon backward error), `7.9783465e-15` (QP residual), and
`5.9491540e-11` (phonon residual), all within their artifact contracts. All
five strict currentness tests pass in 2.25 s. The Round-6 campaign driver was
`scripts/regenerate_fischer_fig7_parallel.py`. The exact refinement slow
gate passes in 4906.90 s, and the transient slow battery passes 4/4 in
850.77 s. The Fig. 5 high-drive slow check and the repinned, independently
number-certified reduced Fig. 6 continuation also pass; the two legacy
baseline nodes remain documented xfails. A durable commit identifier and
completion of the full 1620-bin Fig. 3 pinned-baseline slow node remain
pending. The latter was started on 2026-07-22 and interrupted by the
host/session loss during the `tau_l/tau_0^PB = 1` target. Its durable log
preserves a
certified full-grid `0.1` target (QP residual `2.068e-25`, QP backward error
`1.218e-16`, phonon residual `1.408e-13`, raw/scaled phonon backward errors
`2.589e-6`/`1.321e-6`) and successful continuation returns at `0.3` and
`0.5`; this partial evidence is deliberately not reported as a passing
baseline node:

- **Newton number-mode decomposition and polish:** certification now assembles only
  channels that change QP number: phonon recombination/pair breaking, the
  K⁻ pair part of an above-gap photon, and local external sources/sinks.
  Number-conserving e-ph scattering, sub-gap photon scattering, and PB K⁺
  scattering are excluded. A zero `ExternalFlux` is equivalent to `None`,
  and PB-only solves remain covered when `K_r0` is absent. When the ordinary
  residual reaches its floating-point floor before the number mode, a bounded
  scalar amplitude polish solves the signed pair-number balance without
  loosening `backward_error_tol`; representable wrong-amplitude seeds are
  corrected instead of merely rejected. Cold Fig. 7 then exposed another
  Newton globalization failure. The resulting path assembles
  number-conserving shape channels directly, scales each row by its local
  physical turnover without a global floor, and—once both dimensional
  residuals are below tolerance—uses normalized balance rather than
  meaningless raw-dimensional ordering as the line-search merit. It tries a
  raw-Newton direction only when the scaled direction cannot improve, and
  enters a bracketed log-amplitude solve only after both directions fail; the
  amplitude candidate is clipped against a one-ULP overshoot of the physical
  occupation interval. A fraction-to-boundary alternative was tested rather
  than assumed: at `−64 dBm / 0.06 K`, fractions from `0.001` through `0.99`
  worsened the balance metric from `0.011645` to `0.011693–0.99956`, which is
  why amplitude globalization follows failed Newton directions instead.
  An intermediate slow-gate run on 2026-07-22 exposed one further merit mismatch:
  the refined Fig. 3 continuation reached absolute and aggregate errors of
  `1.29e-26` and `2.63e-18`, yet spent 500 iterations at number error
  `1.09e-10` against the unchanged `1e-10` contract because the line search
  could accept progress on an already-passing aggregate gate. When total
  number is the only failed certificate, trial steps now must preserve the
  aggregate contract and improve number error. A trial that clears the number
  gate returns immediately; if no trial clears it, the full 20-step ladder is
  scanned and the feasible trial with the best number error is selected rather
  than the first one-ULP improvement. The non-vacuous synthetic regression
  passes. The exact source-bound original/refined continuation comparison then
  passed in **4906.90 s**, with every refined point through ratio 10 certified
  and no tolerance or baseline change (down from **6100.62 s** before the safe
  gate-clearing early exit). At the refined ratio-10 endpoint the final
  certificate was `|R_qp|∞=4.136e-25`, `η_qp=6.393e-17`,
  `|R_ph|∞=1.800e-18`, raw `η_ph=7.738e-12`, and scaled
  `η_ph=3.803e-20`.
  The final reduced Fig. 6 continuation then exposed a stale absolute pin,
  not a new solver defect or platform drift. Under the identical runtime the
  untouched `cd4fb81` tree reproduced the old pin, while substituting only the
  repaired Newton number gate reproduced the new endpoint bit-for-bit. The old
  state had independent pair-number backward error `7.4462e-6`, above the
  advertised `1e-6` inner contract; the new state measures `4.4959e-7` and
  improves the aggregate QP/phonon errors as well. The gap and observable were
  re-pinned to the certified root without loosening either tolerance, and the
  test now independently reassembles and asserts the decisive pair-number
  balance from the returned `(f, n_ph)` snapshot. A configured channel whose
  turnover is truly unrepresentable still refuses, except for an exact absorbing vacuum
  with no actual or declared number-generating source. Exact full thermal
  arrays pass at 20, 30, 40, 45, 50, and 100 mK. This total-number certificate
  applies only to the complete represented domain (`active == ctx.active_mask`):
  every nonempty proper subset now refuses because its artificial boundary
  has unaccounted transition flux. An all-false mask remains an explicit
  no-op, and a custom mask may never enable a zero-capacity row outside
  `ctx.active_mask`.
  Prescribed `ExternalFlux` uses full turnover in the gain/loss normalizer;
  only Device regions explicitly identified as conservative exchange exclude
  that transfer from the scale while keeping it in the residual.
- **Cold finite-phonon production evidence:** the old late Anderson-collapse
  guard admitted an exact 200-iteration cycle because recovery was reached
  only after the outer convergence check. Recovery now happens immediately
  after the collapsed inner solve: restore the last matched `(f,n_ph)`, disable
  Anderson once, and finish with plain Picard. In an earlier exact
  `−68 dBm / 0.06 K` Fig. 7 run, inner call 6 collapsed to
  `xqp=3.23e-106`, call 7 restored the physical state, and calls 8–251
  converged monotonically in 2291.77 s to `xqp=2.049328797e-9`. Independent
  certificate errors were `5.65e-17` (QP) and `9.81e-9` (phonon), below the
  `2e-8` artifact limit under that run's recorded source digest. The other two
  reproduced conditioning targets also pass: `−64 dBm / 0.06 K` at
  `(9.91e-17, 9.51e-9)` and `−64 dBm / 0.10 K` at
  `(6.20e-17, 9.42e-9)` for `(QP, phonon)` backward error.
  The subsequent complete 48-target production regeneration passed with the
  maxima and digest recorded above. The collapse reference now follows the
  last matched physical iterate, so a zero/tiny seed may grow across many
  decades without being mistaken for a collapse before a later genuine
  branch loss is recovered.
- **Device contract:** `outer_tol` is the actual capacity-weighted component
  number-error limit (the hidden `1000×` detector is gone). A public Junction
  contract supplies `C_a/C_b` for active conservative transfer edges; every
  declared edge is checked against its evaluated capacity-weighted flux.
  Consistent unequal ratios are supported and zero-rate edges are true no-ops
  that do not join components. The finite-phonon-specific restriction applies
  when an active conservative cross-region component needs the still-missing
  phonon-aware certificate; junction-free and disabled-edge Devices remain
  valid. In either phonon mode, unknown nonzero state-dependent junction
  fluxes without a safety contract refuse, while exclusive dissipation-owning
  M25 closures remain locally certified. Exact zero-temperature vacuum
  components are accepted. The solver returns only a state at which the
  reported fixed-point defect was
  actually evaluated: a quiet initial snapshot first promotes the complete
  backend map output (including phonons), then certifies that same snapshot.
  Injected backends are pure-map boundaries: mutating the input state refuses
  instead of erasing the measured defect. Returned gap, spectral grid, bath
  temperature, occupation, and phonon state are validated as one coherent T3
  state; value-equivalent deep copies are accepted (object identity is not a
  physical static-field invariant), while invalid qubit probabilities fail
  before defect arithmetic. Stored `n_ph` participates in the undamped
  fixed-point defect and damping; a quiet `f` cannot return a stale phonon
  state.
  Declared conservative transfer is normalized before multiplying rates by
  finite-volume weights, closing the former finite-overflow -> `inf/inf` ->
  NaN-pass. The edge check uses `min(1e-12, outer_tol)`, so a caller's tighter
  contract is not silently weakened; capacity-ratio cycle consistency uses
  the same tolerance-aware rule rather than a fixed `1e-12` allowance.
  Complex or string-valued Device controls now raise the documented
  `ValueError` instead of leaking a comparison `TypeError`. Tiny complex
  components in `f`, `n_ph`,
  phonon frequencies, or escape times refuse before any float cast. Exact
  signatures include derived `SpectralContext` caches (`cell_weights`,
  densities, kernels, diffusion values, and active support), catching both
  corrupt returned copies and in-place mutation. The new phonon diagnostic was
  appended after the legacy `DeviceSolution` fields, preserving positional
  constructor compatibility.
- **Strict 2Δ consistency:** the final reflection-partner guard now uses
  `pair_channel_open`; threshold-crossing errors precede informational snap
  warnings; docs state the strict `ω > 2Δ` pair condition. Channel-resolved
  PB rates prevent certificate reuse from reintroducing K⁺ scattering.
  Exact-threshold, roundoff-band, snap-to-equality, component-sum, and Kaplan
  unsupported-entry controls are regression-tested. A zero-coupling or
  zero-frequency photon dictionary is equivalent to no channel through both
  rates and the analytical Jacobian (including nonuniform grids), while its
  scalar fields are still validated.
- **Campaign resume integrity:** both runners validate the current summary
  and shift schemas before skipping. A completed run requires the latest
  attempt to be converged/error-free, exactly six unique expected resonator
  rows with matching labels and finite numeric data, and run-local contained
  trace/profile CSVs with exact headers and finite numeric rows. Revision 6
  run ids include exact-value point fingerprints plus a config digest over the
  numerical configuration, module physics constants, all `qpsim`
  Python/material-YAML sources, relevant campaign scripts, and the numerical
  runtime (Python/platform, NumPy/SciPy, BLAS/LAPACK, and thread settings).
  Distinct points sharing a lossy display label cannot collide, and a duplicate-id gate
  runs before output mutation. Readout source center and width tuples are real
  Cartesian sweep dimensions rather than silently using element zero.
  Config-addressed metadata is written atomically
  only after resume/header validation, allowing multiple presets to coexist
  without a rejected invocation rewriting provenance. Summaries record `NX`;
  profiles must contain exactly `NX` rows on the expected 0–100 μm grid with
  non-negative `xqp`; traces must begin at zero, be nondecreasing, and end at
  the committed `total_time_ns` within an accumulated-roundoff CSV tolerance.
  The committed step count, `dt_ns`, elapsed time, and `max_time_ns` must also
  satisfy the backend's fixed-step/one-shortened-final-step invariant. The
  readout integrator itself now takes that shortened final step rather than
  overshooting a non-divisible horizon; a `dt=1`, `t_max=2.5` regression lands
  at exactly 2.5 in three steps and produces a resumable artifact. Invalid
  integration controls refuse before state construction: `dt_ns` must be a
  finite positive real value and `max_time_ns` a finite non-negative real
  value, preventing both false `t=0` completion and zero-step infinite loops.
  Summary parameters and endpoint residual/density values are cross-checked against
  the shift rows, trace endpoint, and profile rather than trusting the status
  string. The spatial writer maps only the backend's unevaluated
  `t=0, max_rate=+inf` sentinel to
  finite zero and rejects every other non-finite value; the readout producer
  always writes its endpoint. Malformed attributable attempts remain
  purge-healable; ambiguous rows fail loudly with recovery guidance.
- **Coupled finite-phonon certificate:** aggregate residual norms were also
  amplitude-blind when number-conserving QP scattering dominated the cold
  pair mode. Coupled Newton now certifies the QP number-changing balance and
  the complete phonon residual against pair/escape turnover before every
  return. The thermal exact-state shortcut additionally requires the actual
  aggregate balance and the exact QP-derived phonon frequency/index map, so
  neither an asymmetric custom kernel nor a shape/range-valid malformed map
  can bypass the solve merely because the initial arrays are thermal-shaped.
  The reproduced all-zero recombination map had QP-number backward error
  `1.0` and slow-phonon error `1.088` before this guard. Kernel/map inputs and
  returned ancillary state are finite/domain validated. A configured number
  channel with zero representable finite-temperature turnover now fails
  closed in coupled Newton as it does in the standalone solver; an exact
  `T=0`, loss-only absorbing vacuum remains an accepted physical root.
  Complex QP/phonon states, frequency axes, and all four optional kernel
  families are rejected before any float cast, including an imaginary NaN
  that the old conversion silently discarded. Exact cold thermal QP/Bose roots
  pass; the reproduced `0.5·f_FD` state at 55 mK refuses.
- **Spatial finite-volume consistency:** the campaign mesh is now uniformly
  represented by cell centers `x_i=(i+1/2)L/NX` with `dx=L/NX`; source-cell
  volume, CFL selection, profiles, current weighting, and spatial AC
  observables all use that same measure. Both diffusion-Laplacian builders
  share boundary/segment validation. Spatial AC input now requires finite,
  strictly increasing coordinates and finite non-negative `I²` weights with
  positive support, preventing NaN observables and signed-weight artifacts.
  Boundary values, Robin coefficients, `dx`, and variable diffusivity must be
  finite; geometry masks must be genuinely boolean rather than truthy integer
  or NaN arrays, and diffusivity must be real-valued before conversion. The
  harmonic mean is overflow-safe for very large finite `D`, and assembled
  source/operator arrays are checked before return.
- **Additional numerical return/boundary fixes:** Picard returns the exact
  snapshot whose fixed-point defect it reports on both convergence and budget
  exhaustion; moving-gap tail comparisons scale with actual QP mass rather
  than a unit floor; photon rate/Jacobian threshold and disabled-channel
  semantics share one strict `ω>2Δ` contract; phonon public builders reject
  non-finite times; and the WebUI heatmap has a valid linear normalization for
  an all-underflow occupation payload.
- **Ultracold M25 coefficients:** `_K_incomplete` now uses Gaussian-scaled
  endpoint quadrature and log-domain full/incomplete/lower-tail forms;
  cancellation-prone consumers use `kve` and `erfcx`. At `z≈1199.81` the
  branching fraction is the representable `2.11583916322907e-106` instead of
  `0/0`; genuinely subnormal ratios become exact zero while the complete
  coefficient bundle remains finite. Normal-temperature values change only
  at about `1e-14` relative.
- **Audit-ledger process:** `CODE-REVIEW-FALSE-POSITIVES.md` now distinguishes
  refuted findings from real accepted limitations and overturned verdicts.
  The `cd4fb81` certificate-scope entry moved to the overturned section; a
  code/assumption change automatically reopens any historical adjudication.

## 2026-07-22 Round-7 audit and isolated repair (working tree)

Round 7 reviewed the post-Round-6 working tree while the expensive Fig. 3
node continued in the original checkout. Repairs were made in an isolated
worktree on branch `codex/audit-round7-fixes`, based on `cd4fb81` plus the
copied Round-6 changes. The running Fig. 3 process therefore neither reads
these edits nor certifies them; its eventual result remains evidence for the
source snapshot from which it was launched. No Round-7 commit or hosted CI
result exists yet.

- **Gap and threshold physics:** `solve_gap` no longer treats an
  above-`T_c` equilibrium calibration as proof that an arbitrary supplied
  occupation is normal. An unanchored nonequilibrium solve scans the full
  physically admissible interval through the zero-temperature gap: it returns
  a unique sign-changing root, but multiple roots fail loudly and require a
  `reference_gap` continuation anchor instead of silently selecting a branch.
  A genuine normal root still passes through the grid-support contract. The
  Kaplan phonon-side right limit at exactly `2Δ` is finite (`S_+(2)=π`) in
  both analytic and numerical paths, and finite-volume phonon correction
  includes that endpoint. The photon K⁻ pair channel deliberately retains its
  separate strict `ω > 2Δ` contract.
- **WebUI/runtime contract parity:** spatial WebUI grids now use the same
  cell-center measure and `dx=L/NX` as the backend; ideal-BCS kinetic modes
  reject Dynes broadening; thermal-phonon modes do not require a dynamic
  phonon lifetime; and pair-breaking preflight shares the runtime's
  lightweight grid-contract validator. Disabled zero-coupling drives remain
  true no-ops instead of failing an irrelevant frequency preflight.
- **Public numerical boundaries:** overflow-safe harmonic face weights cover
  very large finite diffusivities; Picard controls are finite/domain checked;
  and complex arrays, including imaginary-NaN values, are rejected before
  float conversion across spatial flux, external flux, Newton, and shared
  collision validation paths.
- **Campaign durability:** trace/profile CSV replacement uses temporary-file,
  flush/fsync, and atomic-replace discipline; readout traces are promoted only
  on completion and preserve an earlier final artifact if promotion fails.
  Resume expectations bind the nominal and snapped readout frequency fields,
  preventing a stale row from satisfying a changed snap contract. At this
  checkpoint the runners still relied on a documented single-writer workflow;
  the later hardening below adds an enforced OS advisory lock. Neither change
  claims absolute host-power-loss transactionality.
- **Documentation calibration:** Fig. 7's near-zero QP-loss floor is recorded
  as `1e-18`, matching the measured host-jitter calibration rather than the
  superseded `2e-19` value.

Focused regression slices for these paths passed. One adversarial review found
no blocker in its reviewed scope; that was a snapshot statement, not a durable
conclusion. The 2026-07-25 Fig. 3 follow-up below subsequently found additional
baseline, performance, restart, and acceleration-boundary defects. The
pre-Fig. 3-follow-up un-rebound default
collection produced **1922 passed, 5 failed, 17 deselected, and 13 warnings in
545.55 s**. All five failures were the expected strict provenance preflights
(Fig. 7 plus four F24 artifacts) rejecting Round-7 source hashes; there were
no non-provenance failures in that selected collection. It did not execute or
validate the full 1620-bin Fig. 3 baseline node, which later failed. Bypassing
only those hash equalities while retaining normal decoding, schema/value
checks, and current-code certificate reassembly/checks passed all five
payloads. Exact grid probes also established
that the new Kaplan `2Δ` endpoint has no sum-pair assignment on the Fig. 3 or
Fig. 7 grids and is not called by the thermal-phonon F24 paths; pre/post
correction arrays and the Fig. 3 extracted `tau_0_pb` were bit-identical.
That evidence supports a compatibility diagnosis only; it does not show that
the replacement source contract executed the stored solves. Fig. 7's digest
explicitly identifies the tree exercised, and summary-only F24 artifacts lack
the state needed for current-equation reassembly. They therefore require exact
regeneration, not a metadata-only provenance rebind. Full-state F24 artifacts
may be re-certified under a later contract only with producer and validator
identities kept distinct. The later ratio-zero numerical mismatch independently
required a certified Fig. 3 numerical regeneration and re-pin. Ruff, mypy over
75 source files, bytecode compilation, and `git diff --check` were clean. At
that snapshot, exact artifact regeneration, hosted CI, and a durable commit
remained pending. Fig. 3, all four F24 families, and the exact current-source
Fig. 7 campaign subsequently completed and were promoted. Final local
aggregate and attestation evidence appears below.

The audit also refuted the claim that the existing hosted 180-minute slow
timeout was already insufficient: run 29845574296 at
`cd4fb81` completed its Python 3.13/3.14 slow steps in 39m39s/40m09s. That
pre-Round-6 evidence does not predict the runtime of the enlarged current
tree; the scoped adjudication is recorded in
[`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md).

## 2026-07-25 Round-7 Fig. 3 performance/restart follow-up

A later full 1620-bin validation consumed `177440.15 s` (49.29 h) and then
failed at the first post-solve curve comparison, ratio zero. This was a real
baseline defect, not platform drift: the old pin has independently reassembled
pair-number backward error about `3.659e-3`; the corrected state is about
`3.2e-18` and is approximately `1.0036657949566 ×` the old amplitude with
negligible normalized shape change. The numerical baseline therefore had to be
regenerated; neither tolerance widening nor a metadata-only rebind was
justified.
The source-frozen replacement subsequently completed all 14 continuation
steps in `10671.777 s` and was promoted after independent readback and visual
inspection. Its intermediate checkpoints had already sharpened the diagnosis:
ratios `0.1` and `1` are `1.2907872626 ×` and `1.0256500876 ×` their old
curves, while their peak-normalized shape residuals are only `5.8e-9` and
`3.9e-8`.

- **Scalar number-mode routing:** the generic conservative routing default is
  unchanged. Fig. 3 explicitly opts into its measured `1e-8` shape-routing
  gate; every polished candidate still passes the unchanged `1e-10` return
  certificates. In the routing-only A/B, ratio zero fell from about
  `32.8 s`/16 dense Jacobians to `5.628 s`/3 Jacobians with the returned
  occupation unchanged to the comparison precision.
- **Continuation acceleration:** high-ratio predictors now use 15% mixing and
  depth-3 Anderson, with forward peak/population checks and retry from the
  untouched seed under the historical plain-Picard policy. Acceleration-only
  arithmetic failures have a dedicated exception; solver `RuntimeError`
  triggers fallback, while configuration `ValueError` remains visible.
  Complex and imaginary-NaN Anderson inputs are rejected before float casting,
  and depth is validated directly. On the 81-bin A/B, historical plain Picard
  took `314.2773 s` and the safeguarded policy `17.9573 s`; all target
  certificates passed and normalized full-array differences were
  `[0, 1.727e-7, 3.745e-6, 7.176e-9]`.
- **Fail-fast validation and restart:** every completed target can be compared
  immediately instead of after the entire ladder. Opt-in restart state is
  content-bound, atomically replaced after every continuation step, replays
  completed targets, and records callback acknowledgement separately.
  A complete checkpoint remains until the outer persistence owner has made its
  final artifact durable. No code automatically deletes a completed
  checkpoint; explicit owner cleanup is required. Callback delivery is
  at-least-once; uncached runs remain fresh unless restart is explicitly
  requested. Restart loading now requires the exact current certificate field
  set and finite/non-negative values (with the intentional thermal-phonon
  `NaN` fields only at ratio zero).
- **Scoped refutations:** the explicit routing gate does not weaken acceptance,
  and the reduced-grid A/B does not show an Anderson branch change. The
  49.29-hour local Windows run still does not prove that the exact hosted
  180-minute CI step timed out.
- **Accepted limitations:** the reduced-grid A/B did not itself substitute
  for the later completed 1620-bin result; restart is not multi-process locking or a
  transaction across all output files.
- **Remaining measured cost:** a reduced-ladder profile attributed most time
  to 1,758 inner Newton solves and repeated collision/channel assembly. The
  safest unshipped optimization—precomputing the fixed Kaplan quadrature
  correction—was 7.8% faster in a read-only A/B with bitwise-identical target
  arrays. Larger likely wins require an explicit immutable collision workspace
  (estimated 25–40% reduced-path potential) and a Schur/block treatment of the
  high-ratio coupled-Newton system; those broader refactors are intentionally
  deferred rather than introduced after the full-grid source snapshot began.
- **Amplitude-sensitive artifact certification:** the independent Fischer
  certificate now measures the signed, capacity-weighted QP-number residual
  against number-changing pair turnover, so a correctly shaped but wrongly
  scaled cold state no longer passes. Fig. 3 persists and requires the extended
  certificate. Its strict schema keeps
  `producer_solve_contract_digest` (the contract that generated stored `f`)
  separate from `validated_solve_contract_digest` (the contract that most
  recently re-certified it). For a finite-escape row, the artifact stores `f`
  but not the producer's `n_ph`; validation reconstructs the affine Ph0 root
  implied by `f` and the validation-time equations. That proves root membership
  of the reconstructed `(f, n_ph)` under those equations. It neither recovers
  the producer's original `n_ph` nor proves that the current solver algorithm
  executed or converged to the persisted state. Figs. 5--7 apply the number
  gate to every live solve but retain
  their legacy summary schemas because those CSVs do not contain the raw
  states needed for honest reconstruction. Their existing artifacts are not
  artifact-level number certificates; upgrading them requires genuine
  regeneration, not invented fields or metadata-only rebinding.

Final source-frozen full-grid evidence:

- wall time: **`10671.777 s` (2 h 57 m 51.8 s)** for all 14 continuation
  steps and four paper targets, versus `177440.15 s` (49.29 h) on the
  superseded route;
- producer cache identity:
  **`8594979aff9c3bc946b2bc341184bbff3e05a2b0b7a6d3e7abae85aab36d366f`**;
  frozen qpsim/extra-source/runtime-neutral contract digests:
  **`dde9c86287b5ab44effdda04fa1f832917c47b74ec1ac8191a67484dab2af8b2`**,
  **`1fee56e5e2a262039e14a76c61df3dd9a8cde865a476877c0b2fc1e9d5397b9b`**,
  and
  **`522539fc7cf1ce148fdf3797e269c7d64dcb15642d0ca6802fbf1e9b9c670685`**;
- raw payload SHA-256:
  **`aafe9103d453b48ba0e09cd917c98c89288d1aef2d8d9eb5c9a1a0b53bb1bedc`**;
  completed backup-checkpoint SHA-256:
  **`2eef7f175891b5d5c9138cd31c780ea3ac1cab113bd8ed1e115f3b4c342499ff`**;
- exact producer runtime: Python `3.14.3`, NumPy `2.5.1`, SciPy `1.18.0`.
  The apparent alternate cache key was traced to a validator accidentally
  using NumPy `2.4.2`/SciPy `1.17.0`, not source drift;
- independently reassembled current-certificate maxima: QP backward
  `8.363e-11`, QP-number backward `2.260e-9`, raw phonon backward
  `2.977e-7`, and phonon residual `5.503e-15`, all below the unchanged
  `1e-5` artifact limit. The producer NPZ did not retain finite-ratio
  `n_ph`, so this reassembly certifies the unique Ph0 affine fixed point
  implied by stored `f(E)` rather than authenticating the producer's original
  phonon array;
- promoted CSV/PDF/validation-record SHA-256:
  **`0cd844124f9afbaebc15f77a14bc14a56aafe325910df446285970ea8093bd9d`**,
  **`668d2fe985a26c747a17637cc28c970c5c5f2e5c311c0d9aa2271cf56c3679ce`**,
  and
  **`b08aebceab3001167173e8f2642dd70148c141e569c5924c387e43d5bd5b54c4`**.
  Strict readback/configuration tests passed after promotion, and the
  one-page PDF passed visual inspection.
This subsection is complete for the frozen producer snapshot and promoted
artifact. The resulting payload certifies its frozen source snapshot only;
later checkpoint, certificate-schema, and API hardening in the edit worktree
requires separate gates and must not be retroactively attributed to that run.

Two final public-input guards later advanced the deliberately conservative
whole-tree digest without changing any valid Fig. 3 path: complex/non-1-D
pair-breaking grids and complex rate-equation temperature grids now fail
before casting. The preserved raw Fig. 3 state was therefore independently
reassembled under the final equations rather than header-rebound. Producer
digest `522539fc…` remains the identity that generated `f`; validated digest
`34fd48de…` identifies the final verifier. Numerical rows and certificate
fields were unchanged. The current CSV/PDF/validation-record SHA-256 values
are `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
`7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
and `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`;
54 focused checks (3 slow deselections) and a fresh visual inspection passed.

All four Fischer-2024 families were also freshly regenerated through their
real solve paths and promoted as strict-v3 CSV/PDF pairs; their focused
collection passes 69 tests with 4 slow deselections. They were regenerated
again after the final conservative source-digest change, with unchanged
numerical payloads/certificate maxima and new current matched-pair hashes
recorded in `STATUS.md`. The exact current-source Fig. 7 campaign also
completed under the hardened content-addressed driver; its final evidence is
recorded below.

### Late Round-7 campaign/input hardening

A lightweight review during the first hardened Fig. 7 attempt found three
additional issues. The attempt was deliberately stopped at 10/48 rather than
allowing it to publish under a source digest that was about to change.

- `validate_pair_breaking_photon_grid()` flattened 2-D grids and cast complex
  `E`/`dE` values to float before validation; the public M25 branch sweep had
  the analogous complex-temperature cast. Shipped callers already supplied
  real 1-D arrays, so no stored numerical result was shown corrupt, but both
  public contracts now reject these inputs before conversion. The focused
  collision/branch collection passes 66 tests and Ruff.
- Both overnight campaign runners described config-addressed coexistence in a
  shared output directory, yet `--no-resume` unlinked the shared aggregate
  CSVs and purged every current-config artifact before applying `--max-runs`.
  A limited restart could therefore destroy unrelated preset rows and
  unselected work. Restarts now purge only the selected exact run IDs, invalid
  limits fail before mutation, aggregate I/O is consistently UTF-8, and
  rewrites use unique fsynced temporaries plus atomic replacement. Both runners
  hold one nonblocking OS advisory lock (`msvcrt` byte-range lock on Windows,
  `flock` on POSIX) for the full output-directory campaign; it auto-releases on
  process exit/crash, while the separate owner JSON is diagnostic only and
  never grants authority. Artifact IDs are fully validated before any delete,
  and unattributable aggregate rows fail before a selective restart mutates
  files. Lock-file initialization occurs only after ownership is held, avoiding
  Windows mandatory-lock first-open races. Release diagnostics are also wholly
  enclosed by unconditional descriptor cleanup, so even a clock/metadata
  failure cannot retain ownership. The focused campaign collection passes 99
  tests and Ruff, including simultaneous cross-runner contenders and
  hard-crash release.

Because the qpsim validation fixes participate in the conservative Fig. 7
source digest, the final 48-target campaign restarted under content/runtime
identity
`82ef6da816fedbe89d6920b51cdcbd3d1dabe40d8b265f38bbaf997d0639f320`
and solve digest
`5d66e4de331acaa73c1d190e71b40cb05c503789efbd94f84ee4d9ec37d86502`.
All 48 fresh targets completed. The six-worker campaign consumed
`3642.094 s` wall time (`13292.818 s` aggregate worker time), then
transactionally promoted the matched CSV, PDF, and promotion attestation with
SHA-256 values
`3298d00bc82d90c6d7b1df6835286262deb38116d9e7e3608298e2b7bbbf8628`,
`3ad2153644b2e2fa865b43a0e55a6656880c301139bf4de60bf7b228d6d0cb9b`,
and `a5d31ac42131d13ebc9c57fa3c60b5145c5678be0865f33800496674ee743357`.
Maximum QP, QP-number, and representability-aware phonon backward errors were
`3.701e-10`, `8.006e-9`, and `9.687e-9`, below the `2e-8` gate. The raw
direct-form phonon diagnostic reached `0.429269` at a sub-ULP bath correction;
it is retained to expose binary64 representability loss and is not the
acceptance metric. Strict currentness/promotion/driver tests passed 74 with
2 slow deselections, an independent attestation audit passed 9 additional
checks, and the one-page PDF passed visual inspection.

After final Fig. 3 recertification and fresh F24 regeneration, the consolidated
non-slow aggregate passed **2188 tests, 18 intentional deselections, and
12 warnings with 0 failures in 716.22 s**.

### Post-push hosted-CI runtime correction

The first hosted run on the final Round-7 tree exposed one test-fixture false
positive: the Fig. 7 lock fallback test changed `sys.platform` but left the
Linux runner's real `/proc` creation-time interfaces visible. Production
correctly selected the stronger available identity; the fixture now hides the
capability it intends to test, and the adjudication is recorded in
`CODE-REVIEW-FALSE-POSITIVES.md`.

The follow-up run then exposed real CI classification errors. The exact
`NE=1620` Fig. 3 pin test directly launches the producer measured at
`10671.777 s`, yet it carried only `slow`, so the pull-request matrix started
the same roughly three-hour solve on both Python versions. Both jobs passed
all preceding gates and ordinary pytest, then were deliberately cancelled
after 54/52 minutes rather than waste the remaining compute. The Fig. 7
full-pin test had the same problem: its serial `run()` recomputes 48
independent points whose hardened campaign measured `13292.818` aggregate
worker-seconds, exceeding the hosted step budget before allowing for runner
variance. Both exact nodes now also carry `manual_slow`, alongside the
existing roughly six-hour Fig. 6 wrapper.

PR CI retains 15 bounded slow tests: reduced Fig. 3 branch and ladder solves,
strict Fig. 3/Fig. 7 artifact/configuration/certificate checks in the default
suite, a live Fig. 7 low-temperature point, the remaining live paper pins, and
transient validation. A fast repository-level guard prevents either
multi-hour marker from silently drifting back into PR CI. This changes test
scheduling only; production code, canonical artifacts, and their source
identities are unchanged. A separate fast Fig. 3 gate authenticates the
promoted CSV and PDF hashes and sizes against the passing validation record,
then binds that record to the live solve-contract digest. Thus the canonical
CSV/PDF/record triple cannot drift independently while the exact regeneration
remains a release/manual gate.

## Full audit record

Original 2026-07-19 workflow findings (including its 40 lows and five
subsequently refuted false positives): machine-readable record committed at
[`audit-2026-07-19-findings.json`](audit-2026-07-19-findings.json); human
summary at `C:\tmp\qpsim-audit-2026-07-19-fable-report.md` (off-repo). Later
external-review and Round-6/7 adjudications are recorded in this document and
[`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md), not in
that original workflow JSON.
