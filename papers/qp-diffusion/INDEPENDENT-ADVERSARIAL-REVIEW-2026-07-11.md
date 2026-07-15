# Independent adversarial physics review — qp-diffusion — 2026-07-11

Reviewer: independent automated adversarial pass (44 dedicated per-subsection
referees → adversarial refutation → independent derivation of surviving
blocker/major findings), plus reviewer-conducted independent derivations of the
two central results. This review was performed **blind** of `CLAUDE.md` and
`ADVERSARIAL-REVIEW-2026-07-10.md`; those were read only afterward, for the fix
audit (Section 6).

This document is the **only** repository write made by this review. No
manuscript, supplement, bibliography, figure, verifier, `CLAUDE.md`, or prior
review record was modified. Nothing was staged or committed.

> **Post-review reconciliation - 2026-07-11 (primary agent, after the
> independent review).** The independent verdict and scorecard below remain the
> record of the blind review at `95c2b85`. Both surviving notes were subsequently
> resolved in `d9e64b2` without touching a human gate. Three record-only claims
> were also corrected in this report: `verify_proximity.py` contains exact
> symbolic complex-angle identities in addition to numeric guards; `latexmk`
> works when Git Perl is selected explicitly; and an unsanitized live-xr import
> can continue to a PDF only without halt mode (with exit 1), while
> `-halt-on-error` stops at the first imported undefined macro. These corrections
> do not change any physics finding or PDF diagnostic.

> **Approved-gate reconciliation - 2026-07-11.** This report's human-gate
> recommendations remain the historical review verdict at `95c2b85`; they are
> no longer open. Three independent advisory panels produced a conservative
> bundle, this review's author endorsed it, and the user explicitly approved it.
> Commit `229956b` implements the replacement abstract, qualitative B1/C1
> caveat without a quantitative trap correction, scoped C2 diagnostic, and M5
> passive-probe wording. See
> `PRESUBMISSION-GATE-DISPOSITION-2026-07-11.md` for the exact D1-D7 decisions,
> exclusions, and final validation. PyPI publication and author contact remain
> deliberately unexecuted.

---

## 1. Executive verdict

**In the manuscript's explicitly delimited sector** — homogeneous material,
local BCS spectrum, isotropic real *s*-wave gap, nonmagnetic Born impurities,
no charge imbalance, dirty limit — **the physics is sound, and the four repair
commits genuinely resolved the prior audited findings.** A 44-unit blind
adversarial pass surfaced 9 candidate findings; after an independent refutation
pass, **only 2 survived, both NOTE-level (cosmetic/presentational), with zero
surviving blocker, major, or minor physics error.** I independently reproduced
the paper's two most-contested results (Section 8):

- the ideal-BCS Usadel spectral coefficients `D_L = 1`, `D_T = N_1^2`, together
  with the fact that the advanced convention `g^A = -τ3 g^{R†} τ3` is
  **physically forced** (it alone satisfies `[L0, g^A] = 0`; the opposite
  anomalous sign gives a nonzero commutator and would swap the coefficients to
  the "legacy" `D_L = N_1^2`);
- the scalar-route coherence-factor transport closure
  `τ_tr = N_1 τ_N ⇒ N_1 D(E) = D_N`, and the agreement of the two routes on the
  conservative operator `N_1 ∂_t f = ∇·(D_N ∇f)`.

**The paper is nonetheless NOT submission-ready**, and this verdict does not
follow merely from "the scripts pass." It is gated by human-only items that
remain correctly open:

- **B1 / C1 / C2** — the prepared quantitative Riwar–Catelani trap correction
  (the `7 → 32 µm` claim) is **correctly not inserted**. It is algebraically
  correct only conditional on an underived sharp-interface transfer law; it is
  not a theorem. Verified absent at HEAD.
- **M5** — the "self-focusing" wording that reaches the abstract, a caption, and
  the body **overstates** what the benchmark demonstrates. I **independently
  re-assessed M5 and do NOT concur with my own workflow's automated refutation
  of it** (that refutation used a drift-only compression estimate and omitted
  the competing diffusion broadening; see Sections 4 and 6). The load-bearing
  claim — that the microscopic operator A1 predicts *no* self-focusing because
  its DOS-gradient drift is identically zero (`q = 0`) — is airtight. But the
  characterization of the *legacy* operators as producing "self-focusing" (net
  concentration) is stronger than a passive-tracer inward drift / reduced
  broadening warrants. M5 should remain a human wording gate.
- The guarded-abstract formal review pass, decisions D1–D7, PyPI upload, and
  contacting Riwar/Catelani are human-only and were left untouched.

Bottom line: **no new physics defect; the delimited-sector results are correct
and independently reproduced; the repairs are real; submission remains blocked
on the standing human gates (B1/C1/C2 hold, M5 wording, abstract pass, D1–D7,
external contact), not on any newly discovered error.**

---

## 2. Exact repository and baseline state

- Repository: `B:\AEinstein\Einstein\Documents\Soren\qpsim-mainctl`
- Branch: `fix/gpt-review-2026-07-05`
- Local HEAD: `95c2b85fa7359e0bb7090333920d08a89f25b4bc`
- Working tree: **clean** (`git status --porcelain` empty) at review start.
- Ahead/behind origin: **6 ahead, 0 behind** `origin/fix/gpt-review-2026-07-05`.
  This review targets **local HEAD**, as instructed.
- Repair commits present and audited (chronological):
  - `01ecd61` paper: correct audited scope and reference issues
  - `46c71f2` paper: repair audited convention and scaling defects
  - `76d822e` paper: fix interface current normalization
  - `3423000` paper: complete nonadiabatic spectral correction
  - `95c2b85` docs: record adversarial repair status (documentation only)
- Engine track, engine branches, Fischer baselines, and engine fig5/6/7 were
  not touched. Paper figures under `papers/qp-diffusion/` were treated as
  current.

---

## 3. Verification and PDF-build results

### 3.1 Symbolic/numeric verifiers — 7/7 PASS

All seven scripts were run **sequentially** (not concurrently) with
`A:\Einstein\Documents\qp-diffusion-paper\.venv\Scripts\python.exe` (Python
3.14.3) and `PYTHONUTF8=1`. Every script reports `ALL PASS` / no `FAIL` token:

| Script | Result | What it establishes (and false-pass assessment) |
|---|---|---|
| `verify_traces.py` | PASS | Trace/DOS/current calibration (unequal-DOS guard, voltage calibration, Wiedemann–Franz). Tests actual coefficients, not mere channel-absence. |
| `verify_gA_convention.py` | PASS | Selects the corrected advanced column `g^A=-τ3 g^{R†}τ3` over the superseded `g^A=-g^{R†}`. By design a *convention-selection* regression (immutable baseline); the physical selection arguments are separate and present in the text. |
| `verify_supercurrent.py` | PASS | Fixed-energy outer limit + edge sum rule for the supercurrent L–T kernel. |
| `verify_proximity.py` | PASS | Exact symbolic complex-angle coefficient, current, and `D_dress · D_undress = N_1^2` identities within the real-order-parameter/zero-phase parametrization, plus numeric guards. |
| `verify_nonadiabatic.py` | PASS | O(ℏ²) shell-slaved and constant-`f_L` zero-source identities; explicit-`d` structure. |
| `verify_fT.py` | PASS | τ3-correction trace `= −2 R_grad f_T`; no linear-η correction to `D_L`, `D_T`. |
| `verify_tdep_inhomogeneous.py` | PASS | Time-dependent cross-order vanishing and the local-Δ(x,t) advective identity `∂_t(N_1 f_L)+∂_E(D_d N_2 f_L)=N_1[∂_t+(Δ/E)D_d ∂_E]f_L`. |

The verifiers are symbolic identity checks. They establish the **algebra** of the
encoded starting equations; they do not by themselves establish the physical
**selection** of conventions (e.g. the advanced sign) — those arguments live in
the text and were checked independently (Section 8). Per-unit reviewers audited
the scripts relevant to their sections for false-pass risk (S10→`tdep`,
S13→`fT`, S17→`supercurrent`, S18→`proximity`, S19→`nonadiabatic`) and found
they test the claimed coefficients rather than merely the absence of a
cross-channel dependence.

### 3.2 PDF build

Under the inherited/default `PATH`, `latexmk` selects a stripped LyX-bundled
Perl lacking `Digest::MD5` and aborts. It is usable after prepending
`C:\Program Files\Git\usr\bin`, whose Perl supplies that module. The independent
review used the documented direct `pdflatex`/`bibtex` xr-hyper bootstrap rather
than changing its inherited environment: supplement → paper → supplement →
`bibtex`×2 → four alternating passes to converge cross-references.

- `paper.pdf`: **55 pages**. 0 undefined references, 0 undefined citations,
  0 overfull boxes. Cosmetic only: 26 underfull `\hbox`, and 12
  `A float is stuck (cannot be placed)` placement warnings (page count stable;
  floats are placed).
- `supplement.pdf`: **52 pages**. 0 undefined references, 0 undefined citations,
  0 overfull boxes, 0 underfull boxes.

Both page counts match the post-fix state recorded in the prior review.

**Build note (documented workflow limitation).** An unsanitized *live-xr*
supplement build emits 28 `Undefined control sequence` messages at
`supplement.tex:24`, the `\externaldocument[M-]{paper}` import. Cause:
`paper.aux` stores paper-defined macros (e.g. `\hatg`) inside nameref/caption
text, and `\externaldocument` runs before `hyperref`/`cleveref` are loaded. In
non-halt mode TeX continues to a 52-page PDF with resolved M- references but
exits 1; with `-halt-on-error` it stops at the first imported macro and produces
no PDF. The repository's `sanitize_aux.py` procedure or the pre-submission
`make arxiv-labels` snapshot removes those nameref macros and permits the clean
halt-on-error build reported above. This is a build-workflow limitation, not a
manuscript physics defect.

---

## 4. Findings ordered by severity

**Surviving findings: 0 blocker, 0 major, 0 minor, 2 note.** Plus one standing
human-gated scope item (M5) that I independently judge should remain open.

### NOTE-1 — `P05-boltz-b-1` — `f_T` mislabeled "the branch-odd mode"
- **Location:** `paper.tex:898` (discussion of `I_n[f_L,n]`, near `eq:intro_phonon`).
- **Printed claim:** "…the branch-odd mode `f_T` drops out of the phonon
  collision integral to linear order in the charge imbalance; the surviving
  `f_T²` corrections are neglected."
- **Why it is a defect:** The subsection's own parity dictionary
  (`eq:intro_parity_dictionary`, lines 782–797) establishes that `φ_T` is the
  branch-odd mode, while `f_T = λ_k φ_T` has **odd harmonics built from the
  branch-even occupation** — the current-carrying channel that explicitly does
  *not* vanish (reinforced at 862–863, 909–913). Calling `f_T` wholesale "the
  branch-odd mode" under a branch-parity argument is internally inconsistent
  terminology. The physics is correct (the charge-imbalance content does drop
  out); only the label is imprecise.
- **Classification / severity:** stylistic / **note**. No equation changes.
- **Narrowest correction:** write "`φ_T` (the branch-odd, charge-imbalance
  mode)" at line 898, or otherwise restrict the label to the charge-imbalance
  content.
- **Rebuttal outcome:** SURVIVES as note. The refuter confirmed the terminology
  tension is real and cannot be refuted on physics/convention grounds, but that
  adjacent lines strongly guard against the misreading, so it stays a note.
- **Remaining uncertainty:** authors' intended symbol; whether to keep the
  "`f_T²`" notation (ties to Kaplan/Chang–Scalapino) or write `φ_T²`.
- **Post-review disposition:** **FIXED** in `d9e64b2`. The text now names
  branch-odd `φ_T`, calls the omitted terms quadratic charge-imbalance
  corrections, and explicitly preserves the odd current-carrying harmonic of
  `f_T` built from the branch-even occupation.

### NOTE-2 — `S19-nonadiab-1` — non-uniform trace normalization between two adjacent displays
- **Location:** `supplement.tex` `eq:nonad_coherence` (line ~3152) vs
  `eq:nonad_coherence_source` (lines ~3157–3163).
- **Printed claim:** `eq:nonad_coherence` displays `(1/4)Tr[τ1 i[L0,gK]]|_{O(ℏ)} = 0`;
  the adjacent `eq:nonad_coherence_source` displays a **bare** `Tr[τ1 i[L0,gK]]|_{O(ℏ²)} = −(2iℏ²/W³)[…]`.
- **Why it is a defect:** The two adjacent coherence-channel projections use
  different normalizations (`(1/4)Tr` vs bare `Tr`); the source RHS is thus 4×
  the `(1/4)Tr` channel amplitude used ~10× elsewhere. **Both equations are
  literally true** (`(1/4)·0 = 0`; bare `Tr` = printed RHS), each LHS displays
  its own normalization, and the surrounding prose is consistently in the same
  bare-trace normalization as the source. So no equation is false and nothing is
  mis-scaled — but the display is non-uniform.
- **Classification / severity:** the reviewer proposed *demonstrated-error /
  minor*; the refuter down-graded to **note** (cosmetic presentational
  non-uniformity, no physical consequence). I concur with **note**.
- **Origin:** introduced by the B2 repair commit `3423000`, which added the
  O(ℏ²) source equation.
- **Narrowest correction:** put both displays in the same normalization — write
  the source as `(1/4)Tr[…]` with RHS divided by 4 (and relabel the two
  downstream quoted amplitudes at ~3168, ~3173 consistently), or drop the `1/4`
  in `eq:nonad_coherence`.
- **Rebuttal outcome:** SURVIVES as note. The refuter could not deny the
  non-uniformity but established it has no physical consequence.
- **Post-review disposition:** **FIXED** in `d9e64b2` by dropping the irrelevant
  `1/4` from the zero first-order display. Both adjacent equations now use the
  verifier's bare-trace normalization; no RHS or downstream amplitude changed.

### HUMAN-GATE (standing) — M5 self-focusing wording — I judge this should remain OPEN
This is a pre-existing human hold, not a new finding, but my workflow's
automated refuter tried to close it and I **do not concur**. Recorded here
because the assignment requires M5 to be assessed independently, not assumed.

- **Location:** `paper.tex` abstract (121–123), body (~2859–2975), conclusion
  (~3076–3078); `self_consistent_feedback` benchmark.
- **The claim at issue:** legacy `q<0` placements cause quasiparticles to
  "self-focus" into the self-generated gap well; the microscopic operator
  predicts "no such self-focusing."
- **Independent adjudication:** The **load-bearing half is correct and
  airtight** — A1 is `(p,q)=(1,0)`, so its DOS-gradient drift
  `v = D_N q N_1^{q−p−1} ∂_x N_1` is identically zero (`q=0`), parameter-free,
  by both routes; A1 genuinely predicts no self-focusing. **The other half — that
  the legacy operators produce "self-focusing" (net concentration) — is
  overstated.** My workflow's refuter argued the compression rate `−∂_x v` is
  positive at the population center for `q<0` (`C=+4.54`, `B=+3.43`) and declared
  M5 refuted; but that estimate is **drift-only** — it omits the competing
  diffusion broadening. A direct one-population width test (recorded in the prior
  review) finds A1, C, and B **all broaden**, the negative-`q` drift only
  *reducing* broadening for B. So what is demonstrated for the legacy operators
  is a passive down-gradient inward drift / reduced broadening, not net
  concentration; "self-focusing" connotes the latter. The refuter's own
  remaining-uncertainty note concedes the net peak-density growth of a fully
  coupled single population was never shown.
- **Disposition:** unchanged from the prior review — **human gate**. Either
  soften "self-focusing" to passive inward-drift language where it reaches the
  abstract/caption/body, or define a focusing observable (e.g. peak density or
  `∫f²`) and demonstrate it in a fully coupled one-population run. Not a physics
  error in the microscopic result; a scope/wording decision.

---

## 5. Rebuttal results — all 9 candidate findings (including rejected)

Nine candidates were produced by the blind pass; each was sent to a separate
agent tasked to refute it. Seven were refuted; two survived (Section 4).

| # | ID / unit | Reviewer severity | Outcome | Basis |
|---|---|---|---|---|
| 1 | `P04-boltz-a-1` (Lorentz force uses bare `v_F` not `v_g`) | note | **REFUTED** | Bare `v_F` is correct: in the quasiparticle Boltzmann picture `k` is the electron Bloch momentum, bent at the electron cyclotron rate `ħk̇=(e/c)v_F k̂×H`, while the quasiparticle streams in real space at `v_g`; this two-velocity structure is exactly the gauge coupling in the starting equation. Non-load-bearing (H→0 in-sector). |
| 2 | `P05-boltz-b-1` (`f_T` labeled "branch-odd mode") | note | **SURVIVES (note)** | Terminology inconsistency vs the parity dictionary; no physics error. → NOTE-1. |
| 3 | `P21-benchmarks-1` ("self-focusing" is passive-tracer drift, not reciprocal) | major | **REFUTED by workflow — I PARTIALLY RESTORE as M5 human gate** | Refuter's compression-sign calc confirms A1's zero drift (correct, load-bearing) but is drift-only; net focusing of a coupled population not shown. See Section 4 HUMAN-GATE. |
| 4 | `P21-benchmarks-2` (A1 dynamic residual, direction unstated) | minor | **REFUTED** | Reviewer's worry (A1 weakly self-focuses under feedback) disproved by numerical decomposition of the benchmark: ~86% of A1's toward-well residual is per-step DOS reweighting bookkeeping; the genuine `f`-transport component is oppositely signed (away from the well); A1's analytic `q=0` drift is exactly zero at every well depth. |
| 5 | `P21-benchmarks-3` (benchmark numbers unverifiable) | note | **REFUTED** | Factually wrong premise: the generating modules exist and are cited (README 21–23, Makefile `benchfigs`) in the parent package `validation/diffusion_operators/*.py`; refuter reproduced conservation (1e-15), markers-on-spectra (5e-14), interface splitting (2e-3). |
| 6 | `P22-conclusion-1` (opening omits "dirty limit") | note | **REFUTED** | "Diffusion operator" is by definition the dirty-limit object; the qualifier appears two lines later (3003) and in the regime-boundary paragraph (3085); the three listed items are the perturbable within-framework assumptions, categorically distinct from the dirty limit. |
| 7 | `S14-step3-scalarred-1` (exact relabelling drops a singular gap-edge source) | note | **REFUTED** | sympy-confirmed: both `∂_t θ` and `∂_E`-of-flux boundary terms produce deltas whose net coefficient `∝ Δ̇(Δ−E)` vanishes on the delta support; the reviewer tracked only one. Genuine turning-point (Landau–Zener) conversion is nonadiabatic and separately scoped to the nonadiabatic appendix. |
| 8 | `S15-branchboltz-1` (`uv−vu=0` ambiguous notation) | note | **REFUTED** | The shorthand is `u_k v_k − v_k u_k = 0` with subscripts suppressed under the local `u=u_k, v=v_k` convention; algebraically identical to the reviewer's proposed rewrite. No-op. |
| 9 | `S19-nonadiab-1` (non-uniform trace normalization) | minor | **SURVIVES (note)** | Real display non-uniformity, no physical consequence. → NOTE-2. |

The three candidates I most wanted probed (M5 over-scope, the A1 dynamic
residual, and the moving-turning-point source) all received genuine independent
computation in the refutation pass. Two are fully refuted (4, 7); the third (3)
I partially restore as the M5 human gate on the wording only — the microscopic
result stands.

---

## 6. Audit of the 2026-07-11 fixes

The prior review (`ADVERSARIAL-REVIEW-2026-07-10.md`, against commit `8598056`)
raised blockers/majors B1–B5 and additional findings M1–M9. The repair commits
claim the dispositions in its *Implementation resolution* table. My blind pass
independently re-reviewed **every affected area at HEAD** and found it clean
(apart from the two cosmetic notes above), which is itself the strongest audit:
the current text, reviewed without knowledge of the prior findings, contains no
surviving blocker/major/minor error. Specifically:

| Prior finding | Claimed fix (commit) | Independent verdict at HEAD |
|---|---|---|
| **B1** (C1 quantitative trap correction, `7→32 µm`) | HOLD-HUMAN; not inserted | **Correctly held.** `grep` finds no C1 subsection, no `32 µm`/`7→32` claim in either `.tex`. The C1 argument (`G_N→∞` KL Robin limit giving `f` continuity across a coherence-scale proximity layer) is conditional on an underived transfer law (`R_ξ(E)` not shown negligible). Correctly absent. |
| **B2** (nonadiabatic Keldysh correction) | FIXED `3423000` | Resolved. `S19` found the O(ℏ²) completion sound; its cosmetic display-normalization NOTE-2 was subsequently fixed in `d9e64b2`. `verify_nonadiabatic.py` PASS. |
| **B3** (supercurrent O(Q³) nonuniform edge scaling) | FIXED `46c71f2` | Resolved. `S17` reviewer clean; `verify_supercurrent.py` PASS (fixed-E outer + edge sum rule). |
| **B4a–e** (equivalence over-statements) | FIXED/SCOPED `01ecd61`, B4c in `46c71f2` | Resolved. `P16` (agreement), `P18` (projection-vs-average + `antipodal_average_identity`), `S11`/`S12` (moving-ξ / branch-projector) all clean. B4c's false commutator is replaced by the antipodal-average identity, which `P18` verified. |
| **B5** (KL boundary current normalization) | FIXED `76d822e` | Resolved. `S16-KL` (and `P19`) clean: factors of `e`, spin, side-dependent material normalization, phase scope, units, and observable-current conversion check out. |
| **M1, M2** (convention/Moyal bridge; missing factor of `i`) | FIXED `46c71f2` | Resolved. `S02` independently verified the antipodal `i`/sign bookkeeping (a naive `×i` would flip the commutator sign; the `p̂→−p̂` flip is essential and correctly supplied), the Moyal "no ½", and the impurity self-energy `−(iℏ/2τ)⟨g⟩`. |
| **M3, M4, M7, M8, M9** (transverse loop, Dynes, proximity wording, bib, scope) | FIXED/SCOPED `01ecd61` | Resolved. `S18` (proximity), `S19`/`S21` (Dynes/fixed-E), `P02` (bib provenance) clean. |
| **M5** (benchmark self-focusing scope) | HOLD-HUMAN | **Correctly held; I concur it must stay open** (Section 4). |
| **M6** (repository completeness) | REFUTED, no change | Concur: the four generating modules + regression tests exist in `validation/diffusion_operators/`. |

**On the prior "core claims that survived."** The prior review's 12-item list
(its lines 939–971) — `N_1=E/√(E²−Δ²)`, `v_g=v_F/N_1`, `τ_tr=N_1 τ_N`,
`ℓ_tr=ℓ_N`, `N_1 D=D_N`, `L_D=(1/N_1)∇·(D_N∇f)`, `D_L=1`/`D_T=N_1²`, the
fixed-energy first-Moyal projection, the advective conversion, A1's zero bulk
drift, distinct legacy solutions, and `g^A=-τ3 g^{R†}τ3` — **coincides exactly
with what I independently reproduced** (Section 8). Independent triangulation,
not deference.

**On the `CLAUDE.md` guards.** `CLAUDE.md` §"Physics already checked — please do
not re-flag" and its "ALL FIXES APPLIED" framing were treated as claims to be
tested, not evidence. Every guarded area was re-checked blind; reviewers were
explicitly told that "settled/deliberate/SymPy-verified" is not proof of
physical correctness. The self-declared "Known incomplete (not defects)" items
(abstract not yet formally reviewed; the riwar2019 journal (A-)equation pinpoint
still to be read off the PDF) are properly scoped and match human gates; `P02`
independently confirmed the riwar2019 provenance against arXiv:1907.04781
(their Eq. 46: `D_qp=D_0/ν_BCS=D_N/N(E)` at a varying gap — exactly the "legacy
placement C" the abstract attributes to it).

---

## 7. Per-subsection scorecard (all 44 units)

Legend: PASS = reviewed, no surviving finding. 38 PASS; 6 units produced
candidates (Section 5).

### Main paper (23 units)
| Unit | Scope | Result |
|---|---|---|
| P01-abstract | Abstract, all quantitative claims + attributions | PASS |
| P02-intro | Introduction; legacy-placement provenance (riwar2019 Eq.46 fetched & confirmed) | PASS |
| P03-kinetic | QP kinetic eqs; `h`-matrix, `f_L/f_T` traces, two-mode/spectral | PASS |
| P04-boltz-a | Boltzmann reduction A (change of variables, Liouville, `p·̇`) | note (P04-boltz-a-1, **refuted**) |
| P05-boltz-b | Boltzmann reduction B (impurity collision, parity dictionary) | **note survived review** (NOTE-1; fixed post-review in `d9e64b2`) |
| P06-usadel | Usadel equation + scalar reduction | PASS |
| P07-bcs-kin | BCS kinematics, `v_g`, `ρ v_g` identity | PASS |
| P08-angavg-a | Angular averaging A (hidden harmonic, impurity kernel, transport rate) | PASS |
| P09-angavg-b | Angular averaging B (Fick closure, `L_D` forms) | PASS |
| P10-tautr | Transport-time closure `τ_tr=N_1 τ_N`, `N_1 D=D_N` | PASS |
| P11-cons | Local density conservation `N_1 f`, `−D_N∇f` | PASS |
| P12-longop-a | Matrix dirty limit + advanced matrix + `D_L`/`D_T` BCS bulk | PASS |
| P13-longop-b | Matrix-current decomposition, longitudinal Usadel current | PASS |
| P14-flux2scalar | Flux→scalar; p=1 vs p=2 placement, spectral flow | PASS |
| P15-tdep-spectral | Moving energy-shell transform, `(Δ/E)Δ̇ ∂_E` term | PASS |
| P16-agreement | Agreement of the two routes | PASS |
| P17-taxonomy | `L_{p,q}` family; selected (1,0); DOS-gradient drift term | PASS |
| P18-projvsang | Projection vs angular averaging; antipodal-average identity | PASS |
| P19-KLcurrent | Conserved currents; KL scalar interface current | PASS |
| P20-gapfeedback | Self-consistent gap feedback; DOS-gradient response | PASS |
| P21-benchmarks | Benchmark problems | 3 candidates, **all refuted** (M5 wording → human gate) |
| P22-conclusion | Conclusion; over-scoping | note (P22-conclusion-1, **refuted**) |
| P23-appendix-cov | Appendix change of variables `k→(E,k̂)` | PASS |

### Supplement (21 units)
| Unit | Scope | Result |
|---|---|---|
| S01-conventions | Statement of result + Conventions | **REVIEWER MISFIRED** (see Section 9); covered by S02 + reviewer's own derivation + cross-checks |
| S02-starteq | Starting Keldysh–Eilenberger; antipodal argument; Moyal "no ½"; self-energy | PASS (thorough) |
| S03-dirtyparam | Dirty-limit parameter, angular harmonic ansatz | PASS |
| S04-step1-a | Step 1A angular moments + dirty-limit slaving | PASS |
| S05-step1-b | Step 1B matrix Keldysh–Usadel equation | PASS |
| S06-step2-a | Step 2A Keldysh component + BCS spectral | PASS |
| S07-step2-b | Step 2B spatial flux decomposition; `D_L`, `D_T` | PASS |
| S08-step3-moyal | Step 3 matrix Moyal commutator with `L0` | PASS |
| S09-step3-plaintrace | Step 3 plain-trace projection; conserved spectral density | PASS |
| S10-step3-checks | Step 3 spectral-flow flux checks (+ `verify_tdep` false-pass audit) | PASS |
| S11-step3-moving | Step 3 moving energy shell + weak form | PASS |
| S12-step3-branchproj | Step 3 adiabatic branch-projector form | PASS |
| S13-step3-LT | Step 3 conservative longitudinal + transverse eqs (+ `verify_fT` audit) | PASS |
| S14-step3-scalarred | Step 3 scalar reduction `f_T→0` | note (S14-...-1, **refuted**) |
| S15-branchboltz | Branch-Boltzmann route; BRT cancellation; coincidence | note (S15-...-1, **refuted**) |
| S16-KL | Kupriyanov–Lukichev scalar boundary conditions | PASS |
| S17-supercurrent | Supercurrent-induced L–T coupling (+ `verify_supercurrent` audit) | PASS |
| S18-proximity | Non-BCS/proximity spectra; `D_dress·D_undress=N_1²` (+ `verify_proximity` audit) | PASS |
| S19-nonadiab | Nonadiabatic O(ℏ²) + Dynes (+ `verify_nonadiabatic` audit) | **note survived review** (NOTE-2; fixed post-review in `d9e64b2`) |
| S20-SII-setup-proof | Sec SII setup, target identity, conservative flow, invariance | PASS |
| S21-SII-fixedE | Sec SII fixed-E spatial-current check | PASS |

---

## 8. Independent derivations reproduced by the reviewer

To avoid the failure mode of "conflating algebra with physics," I reproduced the
two central results myself (hand + sympy), independent of the manuscript's
algebra and the automated agents.

**(a) Advanced convention and the ideal-BCS spectral coefficients (sympy).**
With `g^R = g^R τ3 + i f^R τ2`, the fold `g^A = −τ3 g^{R†} τ3` gives
`g^A = −g^A τ3 − i f^A τ2` (both components sign-flipped), reproducing
`eq:bcs_advanced_matrix`. Above the gap (`g^R=g^A=N_1`, `f^R=f^A=N_2`):
`g^R g^A = −(N_1²−N_2²)𝟙 = −𝟙` and
`g^R τ3 g^A τ3 = −(N_1²+N_2²)𝟙 + 2N_1N_2 τ1`, reproducing
`eq:gRgA_BCS_bulk`/`eq:gRtau3gAtau3_BCS_bulk`. Hence
`D_L = ¼Tr[𝟙 − g^R g^A] = 1` and
`D_T = ¼Tr[𝟙 − g^R τ3 g^A τ3] = N_1²`. The **opposite** anomalous sign gives
`D_L = N_1²`, `D_T = 1` (the "legacy" swap). Decisively, with
`L0 = E τ3 + i Δ τ2`: `[L0, chosen-form] = 0` but
`[L0, opposite-form] = −4ΔE τ1 ≠ 0`. So the advanced sign underpinning `D_L=1`
is **physically forced** by the static spectral equation, not a free choice.
This is exactly the pivot of the historical B1/C1/C2 dispute, and it favors the
manuscript's current convention.

**(b) Coherence-factor transport closure (hand derivation).** The intra-branch
coherence factor `(u_k u_{k'} − v_k v_{k'})² = (ξ/E)² = 1/N_1²`; the fixed-energy
shell supplies a final-state DOS factor `N_1`; net rate
`w = N_1·(1/N_1²)/τ_N = 1/(N_1 τ_N)`, so `τ_tr = N_1 τ_N`. With `v_g = v_F/N_1`:
`ℓ_tr = v_g τ_tr = v_F τ_N = ℓ_N` (energy-independent) and
`D(E) = v_g² τ_tr / 3 = D_N/N_1`, i.e. `N_1 D = D_N`. Both routes then give the
conservative operator `N_1 ∂_t f = ∇·(D_N ∇f)`; the constant `D_N` sits inside
the divergence, so no DOS-gradient drift — whereas the legacy
`∇·[(D_N/N_1)∇f]` carries a spurious `∇(1/N_1)` drift once `Δ(x)` is
inhomogeneous. Matches the abstract and body.

These two independent anchors, plus the blind pass finding no surviving physics
error, are the basis for the "physics is sound in-sector" half of the verdict.

---

## 9. Human-only items left untouched

Per instructions, none of the following was altered or executed:

- **B1 / C1 / C2** — no quantitative trap correction inserted (verified absent).
- **M5** — the guarded self-focusing wording in abstract/caption/body was **not**
  changed; this review only records that it should remain open.
- The guarded abstract and its formal review pass; decisions **D1–D7**; the
  abstract review; **PyPI** upload; contacting **Riwar/Catelani**.

No file other than this report was written; nothing was staged or committed;
`git add -A` was never used.

---

## 10. Limitations and calculations not independently reproduced

1. **S01-conventions reviewer misfired.** That agent returned a placeholder
   (`checked = "Test"`, zero findings) and performed no review — a real gap in
   the automated fan-out on the single most foundational unit. **Compensating
   coverage:** the adjacent `S02-starteq` reviewer independently verified the
   antipodal argument, the Moyal "no ½", the impurity self-energy, the gap
   matrix, and explicitly the `g^A=−τ3 g^{R†}τ3` fold with `N_1²−N_2²=1`; the
   reviewer independently reproduced the conventions (Section 8a); and ~6 other
   units (`P03`, `P12`, `S06`, `S07`, …) grounded in and cross-checked the same
   Conventions block. The conventions are therefore covered, but not by a single
   dedicated agent as intended.
2. **Benchmark numerics partially re-run.** The refutation pass reproduced
   n_qp conservation (~1e-15), markers-on-analytic-spectra (~5e-14), and the
   interface equilibrium splitting (~2e-3). The full 20-ns dynamic
   self-consistent feedback run — the exact "factor of 3 below legacy" figure and
   the one-population width behavior underlying M5 — was **not** end-to-end
   re-executed here (it lives in `validation/diffusion_operators/`, outside the
   review-only manuscript scope). My M5 adjudication relies on the analytic
   drift/compression structure plus the prior review's recorded one-population
   width test, not a fresh dynamic run.
3. **Proximity-verifier scope.** Contrary to the initial review note,
   `verify_proximity.py` proves the complex-angle coefficient/current identities
   and `D_dress·D_undress = N_1²` symbolically for arbitrary real `a,b` in
   `θ=a+ib`; its sampled spectra are supplementary numeric guards. Its scope is
   the real-order-parameter, zero-phase complex-angle parametrization, not an
   unrestricted finite-phase Nambu theorem.
4. **Literature provenance is spot-checked, not exhaustive.** `P02` fetched and
   confirmed riwar2019 (arXiv:1907.04781, Eq. 46). The Belzig-1999 `D_L`/`D_T`
   definition attribution and the Kupriyanov–Lukichev interface law were checked
   for internal consistency and against the reviewers' reading; the primary PRB
   page/equation pinpoints (and the riwar2019 journal (A-)number) were not
   independently read off the published PDFs.
5. **Nonadiabatic O(ℏ²), supercurrent, and branch-Boltzmann derivations** were
   reviewed and their verifiers audited for false-pass, and the surviving
   candidate in that region (NOTE-2) is cosmetic — but I did **not** re-derive
   the full O(ℏ²) Keldysh expansion or the BRT cancellation from scratch myself
   (they were covered by the dedicated units and refuters, not by a reviewer-run
   independent derivation as in Section 8).
6. **This is a review of local HEAD `95c2b85`.** Origin is 6 commits behind and
   was not reviewed.

---

## Method (for reproducibility)

- Baseline: recorded branch/HEAD/status; ran 7 verifiers sequentially; rebuilt
  both PDFs via the direct pdflatex/bibtex xr-hyper bootstrap because the
  inherited PATH selected the incomplete LyX Perl (latexmk works with Git Perl).
- Blind pass: 44 dedicated per-subsection referees over `paper.tex` (23 units)
  and `supplement.tex` (21 units), each grounded in the Conventions block and
  told that "settled/previously reviewed/SymPy-verified" is not evidence of
  physical correctness, and not to read `CLAUDE.md` or the prior review.
- Refutation: each of the 9 candidate findings sent to a separate agent tasked
  to refute it; a finding survived only if it withstood refutation.
- Blocker/major gate: 0 blocker/major survived, so the independent-derivation
  phase produced no items; the reviewer nonetheless independently derived the
  two central results (Section 8) and re-adjudicated M5 by hand.
- Fix audit: performed after the blind pass, reading `CLAUDE.md`,
  `ADVERSARIAL-REVIEW-2026-07-10.md`, and the four repair-commit maps, then
  cross-referencing against the blind-pass results at HEAD.
- Totals: 53 subagents, 0 errors, ~2.84M subagent tokens.
