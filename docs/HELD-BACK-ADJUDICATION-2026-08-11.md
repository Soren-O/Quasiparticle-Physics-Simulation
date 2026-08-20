# Adjudication of the held-back review set — 2026-08-11

Every item in `REVIEW-2026-08-03-HELD-BACK.md` was held back pending a physics
decision. All of them, plus one item that was never filed there
(`qpsim/services/transient.py:327`), were adjudicated on 2026-08-11 and are
decided here. This document is the decision of record; the held-back doc remains
the description of record.

Method: 13 independent adjudicators plus a reconciling synthesis pass, each
running read-only against this tree, instructed to verify claims by execution
rather than accept them, and to check every verdict against
`CODE-REVIEW-FALSE-POSITIVES.md` before filing.

## Summary

- **35 decisions** — 27 APPLY, 8 REJECT.
- **10 APPLY items touch nothing under `qpsim/`** and cost no recertification.
- **17 APPLY items touch `qpsim/`** and share ONE recertification.
  Only 2 of them earn that recert alone; the rest ride along.
- **24 of 35 prepared patches in the held-back doc are
  wrong or only partial as written.** Re-derive before applying; do not copy-paste.

### The cost that is not obvious

Applying the Kaplan S+ item makes the ~2.5 h fig3 republish a **re-baseline, not
a bitwise check**: it moves published figure numbers, not just provenance. It is
the only deliberate physics change in the set, and it is what the
recertification is actually for.

Exposure was checked per producer rather than assumed. The correction lives in
the phonon equation (`qpsim/collisions/phonon.py:449`), so any producer that
solves for a dynamic phonon occupation is affected:

| producer | how it is exposed | affected |
| --- | --- | --- |
| `fig3_solve.py` | sets `use_phonon_side_kernel=True` (2 call sites) | yes |
| `fig5_solve.py` | finite `tau_l = 1.0 * tau_0_pb` | yes |
| `fig7_solve.py` | finite `tau_l` | yes |
| fig6 / C6 / C7 | dynamic Ph0 | yes |
| F24 fig8, figs 9-13 | `use_thermal_phonons=True`, `tau_l = 0` | **no** |

So the blast radius is fig3, fig5, fig6 and fig7 — expect rate-level shifts of
order a few percent, with fixed-point shifts smaller and thermal anchors
unchanged. Budget for regenerating those baselines, not merely re-binding them.

---

## Rejected — do not re-file

These were examined and found not to be defects, or to be defects whose repair
costs more than it buys. Check here before filing any of them again.

### P07 sigma_1 super-gap quadrature patch

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 222
- **Touches `qpsim/`:** yes — recert: n/a
- **Prepared patch correct as written:** no

**Rationale.** The diagnosis (one-signed low sigma_1, correctly documented in the docstring) is right, but the prepared patch was MEASURED to move sigma_1 AWAY from truth in 4 of 5 cases — the E_partner shift collapses (f - f_partner) and overwhelms the legitimate centroid improvement, and there is no separable 'analytic-factor half' (exact-analytic/cell-constant-f lands BELOW the shipped scheme). This is the exact item the task brief pre-warned about ('the sigma_1 entry's patch is known not to produce its stated effect') and the adjudicator's refutation is decisive and quantitative. Current behaviour is a documented approximation; the real fix is a designed convention change.

**Measured.** My checks: the task brief's own warning about this patch corroborated; docstring-accuracy claim consistent with the doc's 'applied item 4'. Variant-isolation numbers (patch -15.84%/-27.71%/-46.39% vs shipped -12.40%/-26.65%/-47.07% at NE=40) inherited; accepted as the refutation.

**Risk.** Certain harm for no gain: production-temperature sigma_1 gets worse while every Fig.7/Figs.9-13/prelim Q pin moves, plus the full recert and a re-pin campaign — to land numbers further from truth.

**Note.** REJECT PERMANENTLY AS FILED — never re-apply any variant of this patch. The surviving work item (NOT part of this ledger, physicist-signed design): per-cell Gauss-Legendre in xi with edge-aware clipped-linear reconstruction of BOTH occupation factors, a dE/kT resolution gate, the SAME decision applied to density.py:73 (identical pairing, n_qp low by up to ~39% on coarse grids), and a coordinated Fig.7/figs_9_13/prelim re-pin with a convergence table. Keep the accurate docstring as the status quo.

### Item 35 (not in doc): transient.py:327 early-stop veto — keep or revert a47bad3

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 327
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** no

**Rationale.** REJECT the revert; keep a47bad3's saturation-based veto. The old smallness test had no physical meaning for occupations (permanently armed on all mK states, killing the documented early-stop feature); the new complementarity condition is correct (untouched f=0 hides nothing under the gain>=0 contract), the veto gates only the stop decision (trajectory bitwise identical), the stop is measured honest against the exact residual, and every shipped caller uses the exact-diagnostics path anyway. The 'behaviour-neutral' commit label was wrong for this hunk — a process failure, remedied in docs.

**Measured.** My checks: the remedy is ALREADY EXECUTED — CODE-REVIEW-FALSE-POSITIVES.md section 3 item 36 (read in full, dated 2026-08-11) records exactly this adjudication with the adjudicator's measurements and the test_P15 pin. FD-run and bitwise-truncation measurements inherited; accepted.

**Risk.** Reverting would break three shipped regression tests, resurrect a dead-feature defect, and burn a full recert for strictly negative value.

**Note.** No code action, ever. The ledger entry (section 3.36) already exists in the working tree — confirm it is committed/pushed with the branch; add nothing else. Out-of-repo custom-backend callers now stop earlier by documented design.

### P12 solve_gap gap-edge resolution threshold

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 403
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** no

**Rationale.** The proposed estimator assumes a smooth (dE/Delta)^1.5 law the solved gap measurably does not obey (alignment-phase non-monotonicity: 10x LARGER error at dE/Delta=0.056 than at 0.1125), so a 'warning that presents itself as an error estimate' would be quantitatively wrong by orders — strictly worse than the shipped honest message. The 0.03 fixed ratio is a measured regression (fires at true error 1e-10). Exposure is narrow (min_factor>=1 grids fail closed via GapBelowGridSupportError) and the published-number exposure is removed by the 297 APPLY.

**Measured.** My checks: 297-dependency verified consistent (297 is APPLY, so the exposure-removal premise holds); the near-T_c limitation is already a ledger section-2 entry (solve_gap near-T_c grid bias), so nothing new needs recording. Convergence-sweep numbers inherited; accepted.

**Risk.** False warnings on valid fine grids plus false quantitative confidence near T_c, warning-surface churn during recert, and a qpsim/ digest advance for a diagnostic-only change.

**Note.** REJECT PERMANENTLY AS FILED. Reopen ONLY with a validated phase-envelope error model: measure the upper envelope of |rel gap error| over grid offset at each (dE/Delta_ref, T/T_c), fit amplification from the calibration's residual slope, require the estimator to bracket the envelope within ~10x across the sweep before wiring any warning. The already-applied honest message is the correct end state.

### P04 large-asymmetry seed rows (_default_seed_grid)

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 532
- **Touches `qpsim/`:** yes — recert: n/a
- **Prepared patch correct as written:** no

**Rationale.** Fixes a non-failure: the current 24-seed grid plus residual picker already returns both SI-S69 orderings at every probed shipped point, agreeing with analytic-seeded solves to <=5e-15. Adding 24 hand-tuned seeds (with a second magic ratio the verifier explicitly rejected) can change behavior only through residual ties (branch-switch risk on multi-root families) while advancing the M25 fingerprint and breaking the packet's own anti-drift pin — provenance cost for zero measured benefit. Principled tools (analytic_low_T_seed, expected_ordering) already exist for branch-sensitive callers.

**Measured.** My checks: no cross-item interaction (the M25 wave W4 does not need this). Ordering/agreement measurements inherited; accepted.

**Risk.** Branch switches on residual ties, M25 republication, and a defeated regression pin.

**Note.** REJECT PERMANENTLY AS FILED. If a REAL large-asymmetry M25 bundle ever stalls or lands on the wrong branch, implement option (b): analytic_low_T_seed-derived preferred_seed from m25_junction.py:340-344 with lock_to_preferred or expected_ordering, bundled with a planned regeneration. The applied docstring correction is the complete fix until then.

### P03 componentwise return gate in newton_solve_f

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 645
- **Touches `qpsim/`:** yes — recert: n/a
- **Prepared patch correct as written:** no

**Rationale.** The proposed row scale is legitimate (identical to _row_scaled_newton_system's, not an aggregate-turnover violation), but the gate fails on float64 representability: at 15 mK the componentwise error at the best representable root is 0.175, tracking deep-subnormal ULPs row-by-row, and f==0 rows with underflowed fixed points read exactly 1.000 forever — the patch's named exemption cannot rescue either, and its own text forbids the turnover-band cutoff that could. Its two requirements are jointly unsatisfiable in float64. The blind zone it targets is real but moves no observable and is already a scoped ledger limitation with a reopen trigger.

**Measured.** My checks: ledger section-2 F24-tail entry confirmed verbatim including the reopen trigger ('when a tail-sensitive observable is added'); CURRENT-STATUS 0.10 K 'not certifiable in double precision' corroborated by the repo-state memory's Q0 record. Subnormal-ladder and bit-identical-tail-perturbation measurements inherited; accepted.

**Risk.** Cold thermal solves become un-returnable, currently-passing pinned points inherit the 0.10 K failure mode, no published number improves, plus a full recert.

**Note.** REJECT AS FILED; the future design stays on the books under the existing ledger reopen trigger: componentwise error as a RECORDED diagnostic first, then a gate with an explicitly documented subnormal-band exemption calibrated on Fig-7 cold/high-drive — a physicist-signed study. OPTIONAL zero-risk rider for Bundle R (owner's discretion, not required): record componentwise error in the two RuntimeError strings and success-path diagnostics while the digest is advancing anyway.

### P06 author-adapter _SUBPROCESS_WRAPPER literal respelling

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 902
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** yes

**Rationale.** Measured physics impact exactly zero: the 1-ULP literal differences cancel bit-for-bit in every derived constant that enters the solver (h, dos, x_phot, c, a_Delta, rounded Delta, n_bar[49] all bit-identical); only the recorded diagnostic T_star — which the author's own comment says does not enter the simulation — moves 1 ULP. That cannot justify the most expensive regeneration in the project (fresh unseeded author run + anchor + replay sweep + entire C0..C7 chain rebind + 17-digit quotes in >=6 doc locations).

**Measured.** My checks: the ordering constraint against doc-line-927 is honored in the plan (W2 executes with 902 parked; any later 902 application would re-stale the regenerated C0 chain). Bit-level chain computation inherited; accepted.

**Risk.** Fresh-run nondeterminism, wide artifact churn, zero physics benefit, and a reopened finding unless step (4) ships too.

**Note.** REJECT STANDALONE — PARKED, not permanent: bundle the respell + abscissae contract + quote updates ONLY into a future SUBSTANTIVE author-leg regeneration (new F&C archive or wrapper functional change), and never after W2 without budgeting a second full C0-chain rebind. Add the accepted-approximation ledger entry now (digest-free): recorded anchor abscissa is the wrapper spelling, 1 ULP (1.6e-16) above the authentic program's; solver inputs bit-identical.

### P08 fig6_q0_sweep committable Q0 receipt

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 1068
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** Provenance engineering aimed at the wrong hole: the sweep artifact already embeds per-point ratios/sources/grid/solver; the operative defect is downstream (make_figures.py filters only on converged, guards only figure_curves, and silently savefig()s blank tracked figures on a fresh clone) — a receipt sidecar fixes none of that and cannot even be produced for the existing artifacts from this checkout. Applying now would also break a green, verified ladder binding for a writer that emits nothing until the next multi-hour sweep.

**Measured.** My checks: post-Bundle-R the Q0 diagnostic numbers become stale anyway (101 moves fig6 physics), which strengthens the case for doing the real fail-closed work at the next full sweep rather than patching the receipt now. Ladder-hash recomputation inherited; accepted.

**Risk.** Breaks the ladder binding in exchange for a no-op writer while leaving the silent-figure-clobber defect intact — worst possible trade.

**Note.** REJECT AS FILED, permanently in this form. The real work item (papers/qpsim/fig6-numerics owner, digest-free, schedule with the next Q0 sweep — which Bundle R makes necessary anyway): (1) fail-closed make_figures.py (require artifact sha + ratio match vs committed receipts, exit nonzero on missing/mismatch, never write empty figures); (2) commit per-point receipts; (3) fold THIS receipt-emission patch in at that same moment with one ladder source_canonical rebind. Do not apply piecemeal.

### P11 per-row paper_observable gates

- **Verdict:** REJECT (high confidence)
- **Held-back doc line:** 1180
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** no

**Rationale.** The per-row atol is a 1.2e7x/9.3e3x/2.5e2x loosening of gates that currently PASS and function as de facto producer-identity pins, while audit N37 is open — precisely what ledger 3.29 forbids (never weaken identity gates because a physical tolerance would be looser; split identity from portable acceptance at the next provenance-breaking revision instead). The strictly superior restatement needs no tolerance: both derived ratio columns rebuild bit-exactly 66/66 from persisted anchors, so gate the physical columns physically and assert the ratios as identities. The patch is also internally sloppy (row-max drive scaling under-gates weak-drive elements; pure-rtol vacuous across the 0.10 K sign crossing).

**Measured.** My checks: ledger 3.29 confirmed verbatim as the governing precedent; the 'next provenance-breaking revision' it requires is now scheduled (W1), closing the loop. Identity-rebuild 66/66 measurements inherited; accepted.

**Risk.** A cold-row ratio drift up to +-12 absolute would pass silently during the imminent recertification — accepting ~1e-6 ueV of invisible gap drift in the one column built to resolve 8e-8 ueV.

**Note.** REJECT AS FILED, permanently in this form. At W1's re-pin of test_matches_pinned_baseline, replace both flat ratio gates with the producer-anchor identity form (physical tolerances on gap/analytic columns + roundoff-level identity assertions of both ratio columns rebuilt from the baseline's own anchors, mirroring fig6_paper.py:1156-1214). If cross-environment tolerance is ever needed, put it on the GAP columns, never the amplified ratios.

---

## Apply — no recertification

### P10 Benchmark-2 oracle: conserved cell-average measure

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 762
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** The benchmark is wrong, the backend is right: the operator conserves the exact represented cell-average BCS capacity, and reading the same evolved state in that measure collapses the A1 'drift' by 1.9e5 while the p=0 controls (bitwise measure-independent) match the cell-average analytic 15-20x better. Repairs the same stale-oracle defect class the July-2026 correction fixed in the sibling benchmarks, makes the committed figure match the paper's own methodology sentence, and closes a +6.2e-3 um/ns positive-leak blind spot in the relative gate.

**Measured.** My checks: order-independence from Bundle R verified via item-57's evidence (gap_gradient_drift support-fraction columns bitwise gap-independent, s_L==s_R at every spatial face, so the face-rule changes are a no-op here) — this commit can land before or after the bundle. Measure-collapse and control numbers inherited; accepted, including the critical post-fix refinement-test failure the doc missed.

**Risk.** No digest advance, no recert (validation/ + papers/ only). Regenerates the outputs CSV and the committed gap_gradient_drift.pdf (shifts invisible at figure scale); folds into the already-open qp-diffusion paper adjudication. Code half without P13's test amendments reds the suite.

**Note.** IMMEDIATE, digest-free: land as ONE commit merged with doc-line-812 (one defect, one fix). Composition: P10's code half (single N1 array feeding COM weights AND analytic velocity, mirroring _n1_columns) + P13's test half (absolute A1 gate 1e-7, DELETE test_a1_drift_collapses_under_energy_refinement) + P10's analytic rel-gate tightening 0.15->0.05 at NE=12 + docstring fixes + regeneration. No paper-text numeric edits expected (verified by the adjudicator).

### P13 Benchmark-2 COM weighting + absolute A1 gate

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 812
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** yes

**Rationale.** Same defect as P10 — merged, not applied twice. P13's three contributions all adjudicate correct and are adopted in the merged commit: analytic velocity rebuilt from the same cell-average N1 (mixed-measure gate degradation 0.025->0.104 proves necessity), absolute A1 gate at 1e-7 (7.4x margin over the measured deterministic NE-independent roundoff floor 1.34e-8; 4.5 orders tighter than today; documented fallback 1e-6 on cross-host flakiness per the fig7-floor precedent), and REQUIRED deletion of the refinement test (its assertion is measurably false after the fix — the artifact it tracked no longer exists).

**Measured.** My checks: both adjudicators' merge prescriptions are identical (verified line-by-line — no conflict to resolve); nothing else consumes gap_gradient_drift (their grep). Residual-floor measurements inherited; accepted.

**Risk.** Identical to P10 (same commit). Small cross-platform flakiness risk on the 1e-7 gate with a stated no-loss fallback.

**Note.** Merged into the doc-line-762 commit — see that entry for the six-part composition. Do not file or apply separately.

### P16 CRLF-portability of the Fig.6/Fig.8 P0 provenance chain

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 851
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** No physics; matched-measure provenance consistency, fail-closed either way. The reconciled resolution: keep bfff95f/24dcb73 (verifier must verify in the pin's own canonical measure — the same producer/verifier matched-measure principle as the physics certificates; the newline equivalence class is exactly the parser's invariance class and byte-exactness is still enforced where it belongs), the reviewer's characterization of :745 as 'the re-extraction gate' is factually wrong (that gate lives in verify_external_source and passed live), and add the two .gitattributes rules as a separate hygiene commit (renormalization-safe: every covered blob is already i/lf).

**Measured.** My checks: repo-state memory independently confirms the root cause (producer/verifier helper split, 26->6 red gates at bfff95f) and that the earlier '713 CR bytes in the committed blob' diagnosis was retracted — the adjudicator's account matches the corrected history exactly. Live re-extraction/digest measurements inherited; accepted.

**Risk.** Near zero: rules cannot change committed blobs, cost no recert. Existing CRLF clones do NOT self-heal (need renormalize/re-clone — state this in the commit message); the two slightly-stale cleanroom comments sit inside score closures — leave for the W2 rebuild.

**Note.** IMMEDIATE, digest-free, own commit: exactly 'validation/paper_data/**/*.csv text eol=lf' and 'validation/fischer_2023/*.py text eol=lf'. Do NOT revert :745/:823 to raw (would re-break 20 gates on every unrenormalized Windows clone) and do not expand to a blanket *.py rule or touch the -text -eol baseline markers. Defer the verify_external_source CRLF hint to the next oracle.json-touching regeneration (script_sha256 rebind cascades into the A0 ladder binding).

### P07 C0 A1 gate: enforce qp and phonon sector metrics

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 927
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** yes

**Rationale.** Certificate strengthening with zero physics motion: the concatenated state metric IS the phonon metric (norm ratio 2.4e5), so the 1e-11 state gate alone tolerates f-only error up to ~2.4e-6 — five orders looser than advertised — on the quasiparticle sector the whole C0/C7 chain is about. Committed values sit 150x/25000x inside the new limits so acceptance cannot flip; the fix is two limits + two dict entries plus a regeneration that is executable on this machine today.

**Measured.** My checks: the five dependent scores' rebind requirement and the atomic code+score constraint (load_c0_summary re-hashes the live module) are consistent with the repo-state memory's evidence-chain mechanics (manifests embed live-authenticated source closures; rebind structurally impossible). Norm-ratio and pin values inherited; accepted.

**Risk.** Artifact churn only: c0 score regeneration + summary_sources rebinds in c2/c3/c5/c6/c7 + reproduction-ladder, red until all land together; numpy runtime re-pin (once, in W2).

**Note.** WINDOW W2, atomically with the C-ladder regeneration Bundle R forces anyway (c2..c7 are being rebuilt in-window regardless — this rides at near-zero marginal cost despite 'earns yes' standalone). Same session as doc-line-958 and the pending rebuilds; tighten tests/validation/test_fig6_author_c0_evidence.py:45 to assert the qp metric. Hard ordering: doc-line-902 must never land after this without a second C0-chain rebind budget (it is parked anyway).

### P05 fig6 author-output extractor dashed-sliver fix

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 958
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** Oracle-integrity defect confirmed against the authenticated raster: 7/11/9 columns per curve assign dashed ink to the solid numerical trace, the corruption dilates its own error envelope (so the strongest independent oracle currently cannot reject a swapped trace — the 170-sample probe moved the extracted value by the size of the entire published discrepancy and still 'accepted'), and legend swatches survive only by ordering luck. The three added guards fail loud on a future different raster. The prepared window constant is wrong as written (0.08 excludes the true 0.20K solid minimum 0.0796) — use 0.07.

**Measured.** My checks: the required-anyway score regeneration is confirmed independently (repo-state: test_checked_author_output_score_is_current red since bfff95f because paper_parity.py is in producer sources; rebuild now safe, cost <=9.6e-12, pending only the numpy runtime decision) — so this fix rides a regeneration that must happen regardless. Raster measurements inherited; accepted.

**Risk.** author-output-score churn (9 effU rows tighten; 0.20K analytic maxnorm 0.2193->0.2782, quoted nowhere; numpy re-pin 2.4.2->2.5.1 — the same open runtime decision as the other rebuilds). Constants are raster-specific but guarded.

**Note.** WINDOW W2 (score/evidence session, after Bundle R): land code fix + author-output score regeneration ATOMICALLY (the test asserts producer-source sha equality every run). Use ACCEPTED_Y_VALUE_WINDOW low edge 0.07 (verified end-to-end), GROUP_GAP=60, MINIMUM_TRACE_SEPARATION=150, MAXIMUM_SAMPLE_STEP=0.005 (do not tighten below ~0.003). Also fix the false comment at fig6_author_output_parity.py:342-343. W2 = one session on the pinned ladder venv: C1-C7 regeneration (incl. 101's C6 re-pins and 468's c3-score mirror edit), this item, doc-line-927, and the three already-pending score rebuilds — the numpy re-pin happens exactly once.

### P15 fig6_paper Eq.53 x_qp convention docstring + tau_0^PB wording

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 1030
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** The code is right (factor-2 conversion verified against the paper-convention thermal closed form to the expected finite-T corrections) and the prose contradicts it plus three sibling statements; deleting the factor instead would re-introduce the known 3ad904e damage and break the passing dashed controls. Hunk 2 matches the ledger's settled tau_0^PB round-trip adjudication (section 3.25) verbatim.

**Measured.** My checks: ledger 3.25 confirmed verbatim, INCLUDING its explicit instruction that the stale wording be corrected 'in a coordinated provenance-breaking regeneration' — which window W1 is; and Bundle R now also regenerates fig3/fig5 baselines, so the same 3.25 wording in fig3/fig5 solve modules can ride those regenerations at zero extra cost (optional scope extension, owner's call). Factor-2 arithmetic inherited; accepted.

**Risk.** Zero numerical risk; applied ALONE it reds the fig6 fingerprint-currentness gates — hence W1 bundling.

**Note.** WINDOW W1 rider (fig6 re-promotion, with doc-line-1130). Hunk 1 applies clean; hunk 2 by hand (live line 30 differs trivially). Fix fig6_solve.py:58-63's stale ~255 ps wording in the same W1 edit; optionally fix the fig3/fig5 solve-module twins inside Bundle R's forced fig3/fig5 regenerations instead of waiting.

### P14 grid-consistent thermal reference for the promoted Fig.6 ordinate

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 1130
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** The mixed-quadrature ordinate carries a constant additive bias (up to 22.8% of the weakest promoted point) where the n_bar->0 physics requires exactly 0; the discrete zero-drive fixed point IS the center-sampled thermal state (detailed-balance residuals at machine floor), so the grid solve_gap reference restores an exact 0 and matches both the author pipeline and qpsim's own direct mode. The SPLIT form is decisively right: dragging the Eq.53 overlay onto the grid reference breaks the transcription control outright (normalized errors 0.388->2.49 > 1.0) — exactly the control-vs-result separation ledger 3.33 mandates. Moving AWAY from the paper trace is honest: the removed bias was qpsim-side discretization error masquerading as agreement.

**Measured.** My checks: ledger 3.33 (overlay is a control, score numerics separately) confirmed — the split form is its direct application. CRITICAL AMENDMENT: the adjudicator's cheap no-re-solve regeneration path is VOID once Bundle R lands — I verified fig6_solve runs the dynamic phonon-side path, so item 101 changes the 66 solver states themselves; W1 requires a fresh 66-point sweep on the new engine. Offset/pin/control measurements inherited; accepted.

**Risk.** Every promoted fig6 row's delta_eq and paper_observable_num move (-22.8% to -0.87%; eq53 column unchanged under the split); the known 33-39% paper mismatch widens honestly. Partial application fails loud (validator's continuum _require_close rejects a grid producer) — coordination is mandatory, which W1 provides.

**Note.** WINDOW W1 anchor (first sequel after Bundle R lands, same campaign): producer split (delta_T_grid feeds obs_num/stored delta_eq; delta_T_continuum feeds obs_eq53 at fig6_solve.py:950) + fig6_paper.py re-pin mechanics per the adjudicator + FRESH 66-point sweep on the post-bundle engine (supersedes the persisted-state shortcut) + score.json regeneration (inputs pin csv_sha256) + bundle PDF + promotion metadata. doc-line-1030's docstrings and doc-line-1180's identity-form gate restatement land here. The adjudicator's correction stands: test_fig6_paper.py:834-856 calibrate_gap pins do NOT need re-pinning; the fig6_solve OPEN docstring section and test_P14 framing do.

### P08 F24 Fig8 NEWTON_BACKWARD_ERROR_TOL 1e-6 -> 1e-7 + re-promotion

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 1241
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** Producer stop tolerance equal to the acceptance gate means the certificate adds no independent evidence; tightening to 1e-7 improves the binding points' backward error by ~6 decades with sub-1e-6 payload shifts and removes a live cross-platform determinism hazard (the identical fig5 stopping-branch incident already happened). Same equations, tighter root; no conservation/normalization touched; the _artifact.py <=0.1x invariant names fig8 as the outlier pending exactly this.

**Measured.** My checks: F24 fig8 is thermal-bath (fig8_paper.py:232 tau_l=0, use_thermal_phonons=True — verified by grep), so it is IMMUNE to item 101; only 468-class trailing-bit movement is possible from Bundle R. F24 artifacts are tier-1 (the 17-file rebind list includes F24x4), so post-bundle they must be re-verified anyway — scheduling this re-promotion in-window folds the mandatory rebind into the one regeneration. Bitwise branch reproductions inherited; accepted.

**Risk.** Artifact-level only (payload sha, promotion, PDF binding, pinned baseline — all moving anyway); must run on the recorded producer environment (available, probe reproduced promoted numbers bitwise).

**Note.** WINDOW W3 (after Bundle R lands, parallel to W2/W4): one 36-point rerun (<1 min solve time) with tol 1e-7 on the recorded env. The 32 non-binding points double as a free bundle regression check (expect bitwise or 468-trailing-bit only; anything more = regression signal). Apply the adjudicator's doc corrections when landing: FOUR binding points at three temperatures (not 'three highest-T'); new artifact maxima ~5.04e-8/~7.26e-10 (not the binding point's ~2.7e-13); cite fig5_paper.py:167-173. Do NOT parameterize the TARGET_* limits; fig8_xqp_pb.py and figs_5_7_fe_pb.py stay at 1e-6.

### P09 local _H_OVER_KB literal in fig3_paper.py:103

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 1270
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** partial

**Rationale.** Correct 7-sig-fig truncation, 1.5e-8 low — hygiene, not physics, and in THIS file it feeds only plot decorations (the doc's impact framing is wrong; the solver-input literal lives in fig3_chemical_potentials.py:79, untouched by this patch). Worth doing only as a rider on the M25 regeneration doc-line-1303 already forces. The import swap works with the CURRENT derived constant (captures 99.5% of the correction).

**Measured.** My checks: SCOPE DOWNGRADE enforced — the qpsim/constants.py SI-exact h/k_B redefinition mentioned in the adjudication is NOT approved by this ledger: its whole-tree blast radius was never measured (a +7.7e-11 relative shift in a pervasive constant would move every consumer at the 1e-10 class and silently convert tier-1 rebinds into regenerations, the same hazard class as doc-line-468 but unquantified). Park it under 468-style conditions: only inside a future full re-baseline window, after someone measures which baselines move. Literal/derived/SI arithmetic inherited; accepted.

**Risk.** None beyond the M25 regeneration W4 already performs, IF the constants.py half stays out. Ordering hazard eliminated by parking it.

**Note.** WINDOW W4 rider only (with doc-line-1303's regeneration) — rejected as a standalone edit. Recommended same-wave extensions per the adjudicator: sibling literal swaps in fig3_chemical_potentials.py:79 (moves every M25 fig3 CSV row ~1.5e-8 — a real re-baseline, fine inside the wave) and fig4_paper.py:107, the 7-8 test-site literals, and a guard test with an explicit allowlist — one coordinated M25 re-certification wave, regenerated once on the recorded environment.

### P09 Fig.3 inset gray guide line

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 1303
- **Touches `qpsim/`:** no — no recertification
- **Prepared patch correct as written:** no

**Rationale.** The coded expression is a self-cancelling unit conversion leaving y=T (Kelvin on a dimensionless axis); the M25 paper's own caption settles the intended reference — 'the grey solid lines show the thermal energy' — so the correct replacement is the DIAGONAL Delta_mu=k_B*T, i.e. y=T/Delta_L_K (slopes 0.4209/0.3859 per K), NOT the doc's horizontal axhline at omega_LR/Delta_L (the doc's own conditioned fallback fired). The corrected line shows the quasiequilibrium crossover ~30 mK the paper text discusses; the promoted CSV maxima independently confirm the omega_LR statement, and the caption's parameter list bit-matches the repo constants (right figure consulted). Rendering-only: zero CSV/certificate movement.

**Measured.** My checks: this is the item that justifies the W4 regeneration (the published PDF currently contradicts the paper's own caption) — doc-line-1270 and the 505-forced M25 manifest republication ride it, one regeneration instead of three. Caption fetch and slope arithmetic inherited; accepted.

**Risk.** M25 fig3 bundle regeneration + re-render + manifest re-pin on the recorded generator env (numpy 2.5.1 win32 — the b3cd161 numpy-2.4.6 tail-drift trap applies).

**Note.** WINDOW W4 anchor (after Bundle R, parallel to W2/W3): replacement inset_ax.plot([T_lo,T_hi],[T_lo/Delta_L_K,T_hi/Delta_L_K], gray) with Delta_L_K=Delta_L_GHz*1e9*H_OVER_KB (shared constant via doc-line-1270's swap in the same edit); rewrite the comment at fig3_paper.py:553-555 to quote the caption; KEEP the ymax headroom term and the right-hand Delta_mu/omega_LR axis. Close out with a visual side-by-side against the paper inset (crossing near ~30 mK in panel a). W4 also absorbs the 505-forced M25 Fig3/4 manifest republication — expected numerically identical except the inset and ~1.5e-8 decorations (plus ~1.5e-8 CSV rows if the sibling literal swaps are included).

---

## Apply — inside `qpsim/`, one shared recertification

These advance the whole-tree source digest. Landing them separately would pay
the recertification cost once per commit, so they are one changeset.

### P05 KL series half-cell (t3_spatial_1d.py:600)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 20
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** Adjudicator's series-composition argument is exact for piecewise-constant W (the two half-cells ARE the missing dx/2 resistances); the current bare override lets an interface ADD conductance beyond the bulk face, which is unphysical, and the measured deficit dx/D0 matches all three error magnitudes. Stiffness strictly falls, conservation and the uniform-f nullspace are face-weight-invariant, no certified artifact reaches the interface path. No contradiction with any other item or the false-positives ledger.

**Measured.** My reconciliation checks: doc header at line 20 confirmed; interface_trap.py named as mandatory companion in the doc itself (line 54); gap_gradient_drift benchmark is untouched by this face (item-57 adjudicator measured s_L==s_R bitwise at every spatial face), so the P10/P13 commit is order-independent of this one. Adjudicator's numbers (machine-precision series reference, substeps 18->9, interface_trap RuntimeError 5.6e-2 vs 1e-8 without the companion) accepted; internally consistent.

**Risk.** Shares Bundle R's single digest advance (see item 588 note for the bundle definition). MANDATORY same-change companions or the tree goes red: re-pin tests/backends/test_t3_spatial_1d.py:978; fix validation/diffusion_operators/interface_trap.py:135-156 (fires RuntimeError otherwise, loud not silent); docs/Diffusion_Operators.md L112-118 + docstring/limitation-comment reverts.

**Note.** BUNDLE R member. One commit with doc-line-57 (same function; patches compose in either order; together the series bulk term at a cut interface bin becomes min-based). CAUTION: t3_spatial_1d.py is also edited by doc-line-468's mirror hunk (lines 122-125) — coordinate the three edits to this file in one merge.

### P05 q==0 stepped face min(s_L,s_R) (t3_spatial_1d.py:594)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 57
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** min(s_L,s_R) is the exact cell average of the per-sub-energy series indicator product — the same face-measure convention _kl_interface_cell_average already commits to — while harm-of-averages lets blocked sub-channels borrow conductance (up to 2x overweight). min<=harm so stiffness cannot rise; bit-identical wherever s_L==s_R. The adjudicator's honesty caveat stands: this makes the FACE coefficient exact, not cut-bin transport (irreducible ~+21% single-f-per-bin lumping bias remains at NE=31 under any scheme).

**Measured.** My checks: doc header/patch at line 57 confirmed; all four benchmark-oracle no-op claims (interface_trap zero cut cells, gap_gradient_drift bitwise s-columns, uniform trivial, self_consistent_feedback gates on an s=1 bin) are the load-bearing safety facts and were measured by the adjudicator with named mechanisms — accepted. Only inherited-unverified doc claim (2000-ns webui deltas) is self-flagged and ratio-consistent (1.2297 vs 1.2273).

**Risk.** Bundle R digest cost only; zero pinned numbers or benchmark gates move (measured). Stepped/ramped A1 webui runs lose spurious cross-step leakage by design (~1e-4 integrated).

**Note.** BUNDLE R member; same commit as doc-line-20. Companions: replace the six-line limitation comment at :587-593 and note the q!=0 Jensen gap in docs/Diffusion_Operators.md ~L97-99. Do not advertise as making stepped-gap transport exact.

### P01 Kaplan S_+ correction two-sided (phonon.py:449)

- **Verdict:** ~~APPLY (high confidence)~~ → **REJECT (2026-08-19)**
- **Held-back doc line:** 101
- **Touches `qpsim/`:** yes — recert: yes
- **Prepared patch correct as written:** no — the number is right, the
  operation is wrong

> **OVERTURNED 2026-08-19. Do not apply this item.** The verdict below answered
> the bookkeeping symptom correctly and prescribed an operation that is
> mathematically wrong. `corr(w)`'s threshold value **π/4 IS the gap-corner
> cell's exact two-bin overlap fraction** — the right number applied as a
> **RESCALE** where it should be a **SPLIT**.
>
> Established by four disjoint routes: the corner cell's exact overlap; the
> shipped production path itself (`_pair_breaking_quadrature_correction`
> returns `corr[bin0] = 0.7857613 = 1/1.27264` at NE=1620); a Dirichlet
> integral, `∫_{s+t<1} (st)^{-1/2} = Γ(½)²/Γ(2) = π` (numerically
> 3.141592653662, so `S₀₀ → π/4`); and an analytic decomposition of 4/π into
> separately convergent limits — numerator `(W₀²+N₀²)/(Δh) → 4`, denominator
> `I₀(2Δ+h)/Δ → π`.
>
> Applying the factor to BOTH equations restores agreement between two
> identically altered ledgers, **corrupts the already-converging quasiparticle
> marginal, and drops it from order 1.48 to 1.00**. Fixed points survive only
> because gain and loss receive the same erroneous factor. That is weaker than
> either compatibility or correctness, so this option has no defensible use.
>
> **Postponement is not the disposition — rejection is.** The correct target is
> the exact two-bin cut-cell split with a mesh face at ω = 2Δ; see the
> `qpsim-pair-marginal-threshold` note for the derivation, the entry criteria,
> and the collocation question (whether stored `f_i` is a ρ-weighted cell
> average or a midpoint sample) that must be settled with it — that ambiguity
> alone moves the threshold-bin pair source by −9.18% at T = 0.02Δ, the same
> size as everything the fix repairs.
>
> **Do not quote "27%" as the size of the physics error.** It is a per-bin
> kernel error in a purely local boundary layer; its mass-weighted share of the
> ω-integrated pair source is 0.009% (flat f) / 1.32% (thermal) / 9.47%
> (steep). The transfer to observables is state-dependent over three orders of
> magnitude.
>
> The original reasoning is kept below unedited, because it is a correct
> account of the symptom and of why the one-sided form is also wrong.

**Rationale.** The decisive structural facts are measured: the cross-equation pair ledger is EXACT (3.2e-14) without the correction and broken per-bin by (1-corr) with it one-sided; two-sided application closes the ledger identically, preserves elementwise detailed balance (thermal gain/loss shift by identical -3.337%, fixed points unchanged), keeps the tau_PB(2Delta)=tau_0^PB pin, and improves the QP threshold channels (currently 4/pi-1=27% high). Gating the correction OFF is correctly rejected (regresses the phonon quadrature O(1) and moves 929 C6 bins up to 21%). This is the only deliberate physics change in the qpsim bundle and, with item 588, is what the recert is FOR. Consistent with the ledger's C6 'recorded, not gated' entry and the settled omega-labeling adjudication (corr=1 below threshold untouched).

**Measured.** My checks: fischer fig3/fig5/fig7 producers ALL run the dynamic phonon-side path (fig3_solve.py:290/314-324 use_phonon_side_kernel=True + coupled_newton analytic_cross=True; fig5_solve.py:90 + tau_l=1.0*tau_0_pb; fig7_solve.py:74 finite TAU_L) — so this item moves fig3/5/7 as well as fig6, i.e. the ~2.5h fig3 republish becomes a RE-BASELINE, not a bitwise check. F24 fig8 and figs_9_13 are tau_l=0/use_thermal_phonons (fig8_paper.py:232, figs_9_13:196) — immune, per the item's own thermal-bath scope. Adjudicator's ledger/convergence/fixed-point numbers accepted.

**Risk.** Full recertification is the point; every dynamic-Ph0 figure and C6/C7 move up to ~2-3% at rate level (fixed-point shifts smaller, thermal anchors unchanged). If the companion Jacobian edit (phonon_collision_jacobian_nph -> coupled_newton.py:642 consumer) or the precomputed K_r product sites (newton_steady_state.py:205, t3_diffusion fast paths) are missed, FD-vs-analytic Jacobian tests fail and Newton degrades — those are same-commit requirements, not options.

**Note.** BUNDLE R co-anchor (with 588). MANDATORY pre-merge gates before the recert starts: run tests/collisions/test_phonon.py:244-254 (tau_0^PB pin) and :883-918 (Kaplan Eq.8 continuum check) and the FD-vs-analytic Jacobian tests against the full patch — the adjudicator reasoned these survive but did not report running them post-patch. Item 168 lands in the same bundle in its reshaped form; C6 'recorded' entries re-pin in window W2. Document (do not 'fix') the residual convention wart: the thermal-bath path stays uncorrected, so dynamic tau_l->0 differs from thermal-bath by ~2-3% in threshold channels.

### P01 commensurability guard, phonon.py:306 (merged implementation with doc-line 364)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 135
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** no

**Rationale.** Defect fully verified by both adjudicators independently (0 shared omega bins on shipped defaults; 2537-2672x dynamical error on incommensurate grids; 2.8-7.4% silent x_qp bias end-to-end). Fail-loud on dynamic-phonon entry points only is the right posture and mirrors the repo's own photon-lattice policy; every certified artifact is commensurate so nothing published moves. Both prepared patches are wrong as written (unconditional placement bricks harmless thermal-bath paths; the 364 variant's n_valid suggestion formula returns the same invalid bin count it rejects).

**Measured.** My checks: confirmed the two doc items are the same guard filed twice (headers at 135 and 364, phonon.py:306 vs ph0_local.py:397); confirmed live exposure — scripts/run_prelim_readout_heating_overnight.py NE=101 at lines 109/126 with its own comment at 720 acknowledging the off-grid mode; confirmed ledger 3.11 requires keying the coupled_newton guard on canonically-built maps, which the merged design honors. Structural/dynamical numbers inherited from the two adjudicators, mutually consistent.

**Risk.** Bundle R digest cost. Intended availability break: incommensurate DYNAMIC configs (webui 400-bin default in dynamic modes, 64-bin spatial preset, NE=101 campaign) start raising — so shipped defaults must move 400->405 and 64->63 or 66 IN THE SAME COMMIT, and the error message must give CORRECT alternative bin counts (fix the broken n_valid formula).

**Note.** BUNDLE R member — ONE shared validator, implemented once for both this item and doc-line 364: structural check on the built maps (every diff-lattice bin above 2Delta must be shared; tolerance-free, auto-exempts narrow windows and E_min=0) per the 364 adjudicator's stronger evidence, called from ph0_local.phonon_steady_state, coupled_newton_solve setup (keyed on canonically-built maps per ledger 3.11), t3_diffusion dynamic branch, and webui validate_setup for dynamic modes; scripts runner calls it digest-free. Update tests/review_2026_08_03/test_P16.py::TestPhononOmegaLatticeCommensurability same commit. IMMEDIATE digest-free side action (do now, before the bundle): move the overnight campaign to NE=100 in scripts/ — its current numbers carry a 7-15%-class bias, though the campaign is already flagged stale for independent blind-spot reasons. The rate-preserving omega remap (D3 reversal) remains a recorded open physicist design item (ledger section 2) — this guard neither blocks nor prejudges it; do not fold it in.

### P01 QP/phonon pair ledger residual diagnostic (phonon.py:449)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 168
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** partial

**Rationale.** Correct conservation-side tripwire for the item-101 defect class, but the doc's analytic (1-corr) shortcut is derived FOR the one-sided scheme and becomes a counterfactual after the two-sided fix. Apply reshaped: directly measure assembled QP-side vs phonon-side pair flux under the verified Debye omega^2 conversion, record in steady-state diagnostics, assert ~roundoff post-fix — it graduates from imbalance recorder to a cheap permanent certificate that the matched-measure ledger stays closed. A bare helper with no caller is dead code; the solver/certificate plumbing is inside the 101 bundle anyway.

**Measured.** My checks: dependency direction (168 formula validity depends on 101's outcome) confirmed from both entries; no other item touches this surface. Adjudicator's numbers (0.767%/0.394% one-sided imbalance at NE=400; 3.2e-14 corr-off control) accepted.

**Risk.** None on solve outputs (diagnostic only); free inside Bundle R. Only real hazard is landing the doc's formula unmodified after 101 — it would report the pre-fix counterfactual as a live residual and mislead the next audit.

**Note.** BUNDLE R member, same change set as 101 (or immediately after within the bundle branch). Record-first; promote to a gated certificate only after the recert establishes the roundoff envelope (~1e-12-class pending BLAS-order variance). State the normalization explicitly (pair-channel-relative, not total-turnover).

### P14 TiN.yaml D_0 10.0 -> 0.082 um^2/ns

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 196
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** partial

**Rationale.** Internal consistency settles the film-class judgement the packet deferred: the file's own explicitly-sourced rho_F fixes the Gao 2012 KID film class (rho ~30-200 uOhm-cm), and the Einstein relation then bounds D to 0.041-0.274 um^2/ns; the stored 10.0 back-implies rho=0.82 uOhm-cm and a 30 nm mfp inside a 40 nm film — indefensible for ANY TiN. Mid-class 0.082 is the right canonical value. No conservation law, certificate, or measure involves material.D_0; consumer audit shows webui-only exposure. Consistent with the ledger's settled TiN rho_F entry (value defensible — and now the D it implies is used).

**Measured.** My checks: doc header confirmed; no cross-item conflict (materials yaml is in the source manifest, hence Bundle R). Einstein-relation arithmetic and consumer grep inherited from the adjudicator; accepted.

**Risk.** Bundle R digest cost only; certified outputs bit-identical (nothing certified reads D_0). WebUI TiN spatial runs change on purpose (diffusion length 11x shorter). Residual band uncertainty factor ~7 across the film class — strictly better than 122x wrong, and now provenanced.

**Note.** BUNDLE R member (pure rebind rider). Patch INCOMPLETE as written: must rewrite TiN.yaml lines 8-17 ('flagged, NOT corrected') in the same edit and add the source note to docs/Material_Database.md:31. Do NOT touch the filing's Al/Nb values (wrong direction per verifier_corrections); Nb v_F/xi_0 inconsistency stays a separate item.

### P07 sigma_2 masked partner reconstruction (ac_conductivity.py:188)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 268
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** Genuine defect: a published observable currently depends on inert placeholder values in zero-capacity sub-gap cells (measured 1.58e-1 spread between identical physical states), on an input class the WebUI self-consistent-gap route forces. The masked reconstruction restores placeholder-independence, is bound-preserving, and matches the repo's established masked gap-edge convention; sigma_1 is structurally immune (confirmed bitwise). Bit-identical on every min_factor>=1 grid, so zero certified numbers move.

**Measured.** My checks: no conflict with the 222 REJECT (different lines; sigma_1 untouched); the fixture-vacuity trap the adjudicator flags matches ledger section-3 trap #3 exactly. Placeholder-spread and no-op measurements inherited; accepted.

**Risk.** Bundle R digest cost only; sub-gap-grid WebUI sigma_2/frac_freq_shift move BY DESIGN toward the placeholder-independent value.

**Note.** BUNDLE R member. Same-commit companions: (1) update the stale KNOWN CONVENTION MISMATCH comment at :176-187 and state the convention in the module docstring; (2) add the min_factor<1 regression fixture as a CUT-GEOMETRY case with the first active center ABOVE Delta (e.g. [0.9D,4D] NE=64) or the test is vacuous (NE=400-class cuts are bitwise unaffected — ledger trap #3); (3) optional >=2-active-cells guard mirroring gap_integral_from_distribution_direct.

### P16 compute_gap_suppression grid-consistent reference (gap_suppression.py:97)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 297
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** partial

**Rationale.** A difference observable must evaluate both terms in the same quadrature measure; routing the reference through solve_gap on the caller's grid makes zero drive exactly 0.0 and cancels the grid bias to leading order (weak-drive error -100x at 400 bins), matching the certified _direct family's convention. Rerouting the WebUI to _direct was correctly rejected (semantic swap). This APPLY also carries the 403 REJECT's exposure argument: it removes the sign-flipped rel_suppression class that item worried about.

**Measured.** My checks: dependency direction 403->297 confirmed (403's REJECT leans on this APPLY — consistent); gap_suppression.py is C1's whole evidence closure (repo-state memory), so this edit is covered by window W2's C1-C7 regeneration. Zero-drive bias and weak-drive numbers inherited; accepted.

**Risk.** Bundle R digest cost; every WebUI delta_suppression/rel_gap_suppression moves (intended); GapBelowGridSupportError surfaces on a rare enhancement corner; ~2x solve cost trivial.

**Note.** BUNDLE R member; ONE commit with doc-line-329 (same file). Same-change companions: rewrite the 'Quadrature caveat' docstring; update tests/observables/test_gap_suppression.py::test_thermal_roundtrip (delta_eq pin fails at 3.8e-6) and strengthen test_P16's quadrature-bias test to assert exact 0.0; qpsim/webui/execute.py:215 must publish gs.delta_eq as delta_eq_ueV on success to preserve the delta_suppression identity.

### P16 clip reconstructed gap-edge samples to [0,1] (gap_suppression.py:132)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 329
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** partial

**Rationale.** Fail-open is right because the [0,1] contract is already enforced on INPUT data; the overshoot is internal truncation error of the edge extrapolation, the clip is measurably closer to truth than any alternative (incl. the raise, which makes the headline observable undefined for the whole smooth saturated-edge class with an unsatisfiable remediation), and it restores f -> 1-f symmetry the shipped code already breaks silently on the low side. Bit-identical for every shipped distribution.

**Measured.** My checks: no cross-item conflict; shares C1-closure coverage with 297. Step-function and saturated-Gaussian error numbers inherited; accepted.

**Risk.** Pathological sharply-structured inputs now return a bounded conservative bias (worst -5.7%) instead of raising — acceptable and documented. Bundle R digest cost.

**Note.** BUNDLE R member; same commit as 297. Companions: rewrite the two test_P16 raise-pinning tests as value pins (e.g. clip=0.212047 for the f0=1/f1=0.3 fixture); fix the edge_samples_from_centers docstring and centers-branch comment; keep the strict samples='edges' DATA check; recommended RuntimeWarning when the clip bites beyond ~64 eps.

### P16 Ph0 omega-lattice commensurability guard, ph0_local.py:397 (twin of doc-line 135)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 364
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** no

**Rationale.** Same defect and same verdict as doc-line 135 — this entry is MERGED into that item's single implementation; do not implement two guards. This adjudicator's contributions carry into the merged design: the structural check on built maps (tolerance-free, exempts narrow windows where the ratio test would false-positive), the ledger-3.11 constraint (key on canonically-built maps at solve setup, never blanket-reject custom maps), the measured wrongness of the prepared n_valid suggestion formula, and the live NE=101 campaign exposure with its digest-free interim fix.

**Measured.** My checks: twin-status confirmed from both doc headers; NE=101 at scripts/run_prelim_readout_heating_overnight.py:109/126 confirmed by grep; ledger 3.11 text confirmed. End-to-end x_qp bias (2.8/7.4/5.7% at three tau_l) inherited; accepted.

**Risk.** Identical to item 135 (one implementation, one risk).

**Note.** MERGED INTO doc-line-135's Bundle R change — one validator, one commit, implemented once. Immediate digest-free action now: campaign NE 101->100 in scripts/. The D3-reversal remap stays a recorded separate physicist design item.

### P14 spectral.py:127/142 factored BCS radicand

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 436
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** Pure floating-point reformulation, strictly better at the gap edge (Sterbenz), completing the convention bcs_quadrature.py and gap_equation.py already use. No physics content; on-face production grids bit-identical structurally; the only shipped movement is figs_9_13 sigma_2/Q_i at ~3e-14 against a 1e-4 gate.

**Measured.** My checks: fig7 publishes sigma_1-only Q (fig7_solve/fig7_paper structure consistent with the adjudicator's line cite), and figs_9_13 is thermal-bath (tau_l=0 verified) so its ~3e-14 movement is from this item alone, not 101. Decimal-reference accuracy and bitwise-grid numbers inherited; accepted.

**Risk.** Bundle R digest cost; figs_9_13 baseline needs REGENERATION not rebind (moves bitwise ~3e-14, passes its gate untouched) — scheduled in-window. fig3/5/6/7 bit-identical under THIS item alone, but they re-baseline anyway under 101/468.

**Note.** BUNDLE R member; one commit with doc-line-468 (same file, windows merge). Honor the patch's 'do NOT change line 452' instruction (separately-skipped design decision).

### P14 spectral.py:435 arccosh -> factored arcsinh (3-site bundle)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 468
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** Mathematically exact identity that removes a real reachable gap-edge cancellation (2.2e-3-class at hi-gap=1e-12) and converges spectral.py, t3_spatial_1d.py, and the C3 mirror script on one convention. The adjudicator's enlarged blast radius (not bit-neutral even on on-face grids; every BCS-context root moves O(1e-12)) made it conditional on a planned full-Fischer regeneration window. RECONCILIATION: that condition is now STRUCTURALLY SATISFIED — I verified fig3/fig5/fig7 are all dynamic-phonon producers, so item 101 already forces genuine re-baselining of fig3/5/6/7, and 436 forces figs_9_13; 468 adds no regeneration this bundle does not already perform and no bitwise anchor survives 101 for it to destroy. Confidence raised from the adjudicator's medium accordingly (the medium reflected only the scheduling contingency, which is resolved; the measurements were solid).

**Measured.** My checks: fig3_solve.py/fig5_solve.py/fig7_solve.py dynamic-phonon path confirmed by grep (the load-bearing fact for resolving the condition). Bitwise blast-radius counts (509/1620 fig3, 590/1701 fig7, 137/405 figs_9_13, 55% of K_plus at the fig6 anchor) inherited; accepted.

**Risk.** All root shifts O(1e-12), pass every scientific gate; risk is purely certification churn, which Bundle R already pays. Applied incompletely it breaks the C3 mirror property and the fig6 frozen-state test — hence atomicity below.

**Note.** BUNDLE R member, ONE atomic three-site edit: spectral.py:435-438 + t3_spatial_1d.py:122-125 + validation/fischer_2023/fig6_author_c3_score.py:626-628, with tests/validation/test_fig6_author_frozen_state.py re-pinned in the same change. Same commit as 436. t3_spatial_1d.py is also edited by items 20/57 — one coordinated file edit. The fig6_author_c3_score edit is covered by window W2's C-ladder regeneration.

### P04 remove 1e-14 Hz absolute floor (rate_equation.py:584)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 505
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** The un-floored gate is a true relative certificate plus a rigorous per-row summation backward-error bound; the floor silently converts weak-drive density rows to an absolute gate that certifies non-roots (measured up to 28x-wrong states accepted, 1000x bound), while being measurably inactive (236x/988x headroom) at every shipped M25 point — so removal changes zero shipped numbers and completes the design the false-positives ledger records as intentional (the 'bare 1e-14 reading is the OLD miscalibration' entry). True roots and exact-zero vacuum still pass; the anti-surrogate warning is correct.

**Measured.** My checks: ledger section-1 entry confirmed verbatim (row-wise source-scaled + backward-error gate intentional; fixed absolute = old miscalibration) — this APPLY closes that design rather than contradicting it. Sweep-minimum tolerances and pseudo-root flips inherited; accepted.

**Risk.** Bundle R digest cost; un-shipped weak-drive regimes flip from silent acceptance to RuntimeError (intended fail-loud). rate_equation.py is inside the M25 Fig3/4 manifests AND the C3-C7 score closures — the forced republications are absorbed by windows W2 (C-ladder) and W4 (M25 wave); never land alone.

**Note.** BUNDLE R member. One prepared-patch omission: docs/Part_III_Numerics.md:132 and :157 still state the floored gate — matching one-line digest-free doc edits required or the thesis numerics doc contradicts the code. No test re-pins needed (verified: only upper-bound assertions).

### P15 transient snapshot clamp (transient.py:214)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 560
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** no

**Rationale.** Real defect (interior snapshots are first-order linear dense output; O(1) wrong when dt > interval) and the clamp eliminates the error class at no order/stability cost (ETD1/2 one-step self-starting). But the doc's patch AS WRITTEN silently truncates the integration horizon whenever interval < dt (measured: 4.0 of 80 ns returned) — the loop bound must count clamped steps (max_steps = ceil(total/dt) + ceil(total/interval) + O(1), keeping _SNAPSHOT_HARD_CAP). With that correction the fix is exact (worst interior error 4.297 -> 2.4e-15) and shipped callers move only at 4e-15 against a 1e-6 pin.

**Measured.** My checks: no conflict with the item-19 (327) adjudication — that item concerns the early-stop veto, this one snapshot placement; both in transient.py, compose trivially. Truncation and exactness measurements inherited; accepted as the patch refutation and replacement.

**Risk.** Bundle R digest cost; integer n_steps metadata pins re-pin (tests/services/test_transient.py lines 133/162/187/217/235/461/471); photon_kick_response.csv re-verify (drift 4e-15, trivially passes; tier-1 rebind covers the stamp).

**Note.** BUNDLE R member. MUST NOT ship the doc's patch uncorrected — use the adjudicator's cap-corrected form. Companions: drop the interval<dt warning and its docstring sentence (the clamp makes that regime exact); keep _SNAPSHOT_HARD_CAP; interpolation path may stay (identity at landed boundaries).

### P11 coupled-Newton projected-vacuum guard (coupled_newton.py)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 588
- **Touches `qpsim/`:** yes — recert: yes
- **Prepared patch correct as written:** yes

**Rationale.** Identical physics to ledger trap #23, already adjudicated and fixed in newton_solve_f: a smaller residual norm does not make f==0 physical while finite pair absorption drives QPs out of it; the guard backtracks the same direction and preserves exact loss-only-vacuum semantics. The doc's stated blocker is void (canonical fig6 CSV manifest mode flags prove the Picard path never calls coupled_newton_solve), and converged certified rows are bit-identical under the guard by structure (a vacuum-accepted solve can never converge afterward). Co-anchors the recert with item 101.

**Measured.** My checks: ledger trap #23 confirmed verbatim (the newton_steady_state fix this ports); repo-state memory confirms the test_P11 xfail was made STRICT at 1711bdd, so the marker turns red the moment the guard lands — the same-commit deletion requirement is exactly right. Seed-sweep and bitwise-identity measurements inherited; accepted. AMENDMENT REQUIRED to this adjudicator's recert-expectation note: I verified fischer fig3 runs coupled_newton on the dynamic phonon path, so with item 101 in the same bundle the fig3 republish is NOT expected bit-identical — the 'treat non-bit-identical fig3 as regression' rule is void for the combined bundle.

**Risk.** Bundle R digest cost (this item and 101 jointly justify it). Replace the bit-identity regression signal with a drift-budget doctrine: fig3/5/6/7 dynamic outputs move within 101's measured envelope (up to ~2-3% rate-level, thermal-anchored rows ~unchanged) plus 468's O(1e-12) trailing bits; figs_9_13 ~3e-14 (436); F24/photon-kick/M25 expected bit-identical up to 468-class trailing bits — any drift OUTSIDE these envelopes, or any drift in a channel no bundle item can reach, is a regression: stop and bisect. A future Q0 re-run may newly converge some 0.10 K points (capability gain; re-word the 'not certifiable' narrative then).

**Note.** BUNDLE R co-anchor and the bundle definition lives here. BUNDLE R = one branch, one digest advance, one recertification, members: doc-lines 20+57, 101, 135(+364 merged), 168, 196, 268, 297+329, 436+468, 505, 560(corrected), 588, 677, 730, plus two riders: the queued mypy fix at phonon.py:487 (repo-state item B6 — un-bricks the deliberately-red CI mypy gate so the pytest + 180-min slow gates actually run on the branch) and optionally P03's componentwise-error diagnostic recording. NOT in the bundle: qpsim/constants.py SI-exact h/k_B redefinition (see doc-line-1270 note). Recert on the pinned env only (Quasiparticle-Physics-Simulation/.venv numpy 2.5.1, all thread vars=1, PYTHONIOENCODING=utf-8 — never the in-repo .venv, the b3cd161 trap). This item's own same-commit musts: delete the strict xfail on test_P11.py::test_flat_hot_seed_reaches_the_unforced_thermal_root; fix the patch comment citation (mirrored guard now ~:1307-1350).

### P06 Picard convergence test -> explicit atol/rtol + normwise guard

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 677
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** yes

**Rationale.** The shipped scale+tol floor is precisely the additive-floor-in-a-certificate pattern the project's hard-won rule bans (measured: accepts a 100%-wrong root at c=1e-16 that it correctly rejects at c=1.0); the patch restores scale invariance, mirrors the production _picard_convergence_ratio faithfully, and has zero production callers so nothing pinned can move.

**Measured.** My checks: no production caller claim consistent with the solvers-package structure; no cross-item interaction. Scale-invariance A/B and the two breaking inline-formula tests inherited; accepted, including the honest exact-zero-fixed-point semantics change (geometric decay to exact zero no longer 'converges' — deliberate production semantics, must be documented).

**Risk.** Bundle R digest cost only; two inline-formula tests plus PicardInfo/tol docstrings must move in the same commit or the branch goes red.

**Note.** BUNDLE R member. Land atomically: patch + both test updates (tests/solvers/test_picard.py:104-111, :146-153) + the two new regressions (c=1e-16 rejects; identical verdict across c scales) + docstrings including the exact-zero/normwise-guard behavior. Line refs at 24dcb73: replace block picard.py:189-193.

### P07 WebUI coupled-Newton analytic_cross mapping (webui/builders.py + schemas.py)

- **Verdict:** APPLY (high confidence)
- **Held-back doc line:** 730
- **Touches `qpsim/`:** yes — recert: ride_along
- **Prepared patch correct as written:** partial

**Rationale.** Jacobian-construction choice only — residual acceptance contract identical, roots agree to <=1.6e-8, 86x per-iteration speedup at the shipped default, and the analytic cross blocks are FD-verified in-tree. Every certified driver already opts in (I verified fig3_solve.py:324 analytic_cross=True); the webui is the last production caller on FD. Turning errors into results at strong drive, never the reverse.

**Measured.** My checks: fig3_solve.py:324 coupled_newton_analytic_cross=True confirmed by grep (corroborates 'every certified driver opts in'). Timing/root-agreement A/B inherited; accepted.

**Risk.** Bundle R digest cost (webui/*.py is inside the source rglob); UI observables shift <=1.6e-8; saved setups silently switch Jacobian (intended).

**Note.** BUNDLE R member. Patch is correct ONLY with the out-of-packet schemas.py field coupled_newton_analytic_cross: bool = True in the same commit; do NOT flip the backend default at t3_diffusion.py:568; do NOT adopt the silent min(newton_max_iter,50) cap. Add a mapping assert to test_schemas_builders.py. The queued mypy phonon.py:487 annotation rides this same bundle (repo-state B6).

