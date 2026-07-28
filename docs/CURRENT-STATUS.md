# qpsim current status — AI-agent handoff

> **2026-07-28 CI-portability/provenance repair:** the Fig. 6 canonical
> states were fully recertified under current code without rerunning the
> expensive solver. All 66 CSV data rows, payload SHA-256
> `5b237ad6…39ab`, worker semantic hashes, and PDF bytes are unchanged.
> Generation evidence now carries its captured generation-time fingerprint
> and explicitly authenticates the original v1 run identity
> `cbfb0006…f9a43`; it no longer depends on the reader host's last-bit libm
> rounding or falsely requires the historical runner to equal today's
> source. Current CSV/PDF/promotion SHA-256 values are
> `384f8234…1c11`, `52f7372c…0837`, and `bfefdb0f…00f7`.
> The six previously failing CI nodes pass on both Windows and native Linux;
> the full default suite passes `2533` tests with `2` intentional skips and
> `20` slow deselections, and the independent full Fig. 6 replay passes all
> 66 persisted states.
>
> **2026-07-28 Round-9 paper-oracle update:** Fischer-2023 Fig. 6 now has the
> repository's first quantitative, provenance-bound paper-data comparison.
> `validation/paper_data/fischer_2023/fig6/` binds the exact arXiv-v2 source
> archive and original `gap_test.png`, two-axis calibration with held-out tick
> residuals, 42 extracted points with raster uncertainty, an independent
> qpsim-column mapping, and a checked deterministic score. The score binds
> exact promoted CSV/promotion-record bytes but does not replay all 66 stored
> states; authoritative state replay remains the separate slow gate under the
> recorded single-thread environment. After the provenance-only input rebind,
> its SHA-256 is `caaad6ff…11e8`; curve scores and the
> `diagnostic_mismatch` result are unchanged.
> Re-extraction from the
> exact source archive reproduces `points.csv` byte-for-byte. The three dashed
> Eq. 53 controls pass (`0.388`, `0.200`, `0.253` maximum
> raster-uncertainty-normalized mismatch); the three solid numerical curves
> fail (`8.05`, `9.13`, `7.59`), with maximum relative discrepancies of about
> `33%`, `39%`, and `37%` over seven sampled points on the visible rising
> branch (`T*/Delta ≈ 0.250–0.410`). The honest status is
> `diagnostic_mismatch`, not a paper-parity pass. It is deliberately
> `gate_eligible=false` until parameter and discretization uncertainties are
> bounded. The focused oracle/scorer suite passes 19 tests with the external
> source replay opt-in skipped, and 20/20 with the exact source archive
> supplied. No expensive numerical producer was changed or rerun.
>
> **2026-07-28 Round-8 regeneration update:** current-source Fischer-2023
> Fig. 3 is complete and promoted after all 14 continuation steps
> (`12022.121 s` aggregate step time). A follow-up artifact-layer repair now
> rejects persisted occupations outside inclusive `[0,1]` and rejects complex
> values before float coercion; republishing used the unchanged authenticated
> raw payload rather than rerunning the solve. Focused
> readback/currentness tests pass 58/58 (3 slow/manual cases deselected), and
> the one-page PDF passed
> structural and visual inspection. Final CSV/PDF/validation-record SHA-256
> values are `2264f6c0…8616`, `1f3c762a…e43`, and
> `6776776f…9b89b`. The record explicitly labels the final publication as an
> authenticated cache hit and preserves the original solver-invoked payload
> identity `78c2e181…aa7a45`.
>
> Current-source Fig. 7, Fischer-2024, and Marchegiani-2025 bundles are also
> complete. Fischer-2023 Fig. 6's three single-thread temperature-row workers
> completed all 66 points in `15223.850 s` wall time (`43470.213 s` aggregate
> worker time), and the canonical bundle is promoted. CSV/PDF/promotion-record
> SHA-256 are `384f8234…1c11`, `52f7372c…0837`, and `bfefdb0f…00f7`.
> Fischer-2023 Fig. 5 is now complete too: six independent continuation rows
> produced all 81 points in `9380.161 s` wall time (`35471.716 s` aggregate
> worker time) under campaign identity `01e22c38…cccb`. Its CSV/two
> PDFs/promotion/campaign SHA-256 are `4e187e32…80df`,
> `d12bb185…895c`/`e6307690…6bd5`, `a48cbc15…391a`, and
> `f43d2938…b3fa`. The terminal current-tree default gate passed **2509 tests**
> with 1 intentional opt-in skip, 20 slow/manual deselections, and 12 warnings
> in `836.01 s`. The bounded non-manual slow gate passed **15 tests** with the
> one expected Figs. 9–13 quarantine xfail, 2514 deselections, and 1 warning in
> `4497.19 s`. The opt-in six-row Fig. 5 archive reauthentication passed 1/1 in
> `96.14 s`. Hosted CI remains separate post-push evidence.
>
> At Round-8 closeout no checked-in helper quantitatively compared these
> curves with digitized paper data. The Round-9 layer above adds that separate
> comparison without changing what the regenerated qpsim baselines themselves
> establish.

> **Historical Round-8 note:** the 2026-07-27 live banners described the tree
> before the final Fig. 3/5/6/7 and transient promotions and recorded seven
> expected artifact-currentness failures. Those failures and “producers
> active” statements are superseded by the current banner above; their detailed
> evidence remains in the dated audit record.

> **2026-07-25 live Round-7 update:** work is on branch
> `codex/audit-round7-fixes`, based on `cd4fb81`. The replacement `NE=1620`
> Fig. 3 producer completed all 14 continuation steps in `10671.777 s`
> (2 h 57 m 51.8 s), versus 49.29 h on the superseded pathological route.
> Its authenticated raw state was subsequently reassembled and recertified
> under the final current equations, preserving producer digest `522539fc…`
> and recording validated digest `34fd48de…`. Current
> CSV/PDF/validation-record SHA-256 values are
> `1f92507f04cd06de826342a97da8a3694b7d2819bc07cd0172a1763ef66a60c8`,
> `7e38be09b9b7eaafb02b83015da7cc21c8e5db172954757ac0cdd94256635812`,
> and
> `680454ae17835717a2f52874448fdefa380d367a843a273cf02dab28001a9371`.
>
> All four Fischer-2024 families were freshly regenerated again after the
> final source-digest change and promoted as strict-v3 CSV/PDF pairs. Their
> focused collection passes `69` tests with `4` slow deselections. Exact
> current-source Fig. 7 completed all 48 fresh targets under hardened identity
> `82ef6da816fedbe89d6920b51cdcbd3d1dabe40d8b265f38bbaf997d0639f320`
> and solve digest
> `5d66e4de331acaa73c1d190e71b40cb05c503789efbd94f84ee4d9ec37d86502`.
> The six-worker campaign took `3642.094 s` wall time (`13292.818 s`
> aggregate worker time) and transactionally promoted CSV, PDF, and
> attestation hashes `3298d00b…8628`, `3ad21536…b9b`, and
> `a5d31ac4…357`. Maximum QP, QP-number, and representability-aware phonon
> backward errors were `3.701e-10`, `8.006e-9`, and `9.687e-9`, all below
> the `2e-8` gate. Its focused suite passed `74` tests with `2` slow
> deselections; independent attestation and visual inspection passed.
> The raw direct-form phonon diagnostic reached `0.429269` where the exact
> bath correction is below one binary64 ULP; it is retained as a
> representability diagnostic, not used as the acceptance metric.
>
> The final consolidated non-slow aggregate passes **2188 tests with
> 0 failures**, `18` intentional deselections, and `12` warnings in
> `716.22 s`. Hosted CI remains separate post-push evidence.
>
> Fig. 3 now persists an amplitude-sensitive pair-number certificate and
> distinct producer/validated solve-contract records. Finite-escape validation
> reconstructs the unique affine Ph0 root implied by stored `f`; this proves
> current-equation root membership for that reconstructed pair, not the
> producer's original omitted finite-ratio `n_ph`. The promoted validation
> record states that qualification explicitly.

> **Historical 2026-07-22 working-tree update (superseded by the banner
> above):** this file intentionally remains the
> historical handoff for pre-fix tree `71c5f02`; it is not the status of the
> current `codex/audit-fixes-2026-07-19` working tree. The original
> 123-agent audit, four external-review rounds, and the current Round-6
> repair are recorded in
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md), and overturned
> adjudications are tracked in
> [`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md). Round 6
> is still uncommitted, but all five affected numerical artifact families
> (CSV/PDF pairs) have been regenerated through their real solve paths and
> pass their strict currentness readers. The authoritative post-regeneration
> default collection passed: **1866 passed, 17 deselected, 13 warnings in
> 657.40 s**. The current Fig. 7 regeneration covered all 48 independent
> targets (six workers, 3421.3 s wall time; 13582.0 aggregate worker-seconds)
> under solve-contract digest
> `71d2730e43b41fba106e3066de156eb6d4f69a23ea4aac91c64de6bd552d0503`;
> maximum QP/phonon backward errors were `3.70072e-10` and `9.82942e-9`
> against the `2e-8` limit. The historical tables below still describe
> `71c5f02`; do not quote their old green counts as evidence for this working
> tree. The exact refinement slow gate passed in 4906.90 s, and the transient
> slow battery passed 4/4 in 850.77 s. The Fig. 5 high-drive and reduced Fig. 6
> slow checks pass; the two legacy baselines remain documented xfails. A
> durable commit and completion of the full 1620-bin Fig. 3 pinned-baseline
> slow node were still pending. That node was restarted on 2026-07-22 in the
> original checkout and remained active at that historical update. Its prior
> interrupted run's durable log preserves a certified full-grid `0.1` target
> (QP backward error
> `1.218e-16`, scaled phonon backward error `1.321e-6`) and successful
> continuation returns at `0.3` and `0.5`; those partial checkpoints are not
> represented here as a completed baseline test.

> **Historical 2026-07-22 Round-7 isolated-working-tree update (superseded by
> the banner above):** a further audit found
> and repaired issues in gap/threshold physics, WebUI/backend contract parity,
> public complex-input validation, overflow-safe spatial diffusion, Picard
> controls, and campaign artifact promotion/resume binding. Those edits live
> on uncommitted branch `codex/audit-round7-fixes`, based on `cd4fb81` plus the
> copied Round-6 tree; see the Round-7 section in
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md). The active Fig. 3
> process was deliberately left in the original checkout, so it is not
> Round-7 verification. Focused regressions are green, and the final
> un-rebound default collection produced **1922 passed, 5 failed, 17
> deselected, and 13 warnings in 545.55 s**. The five failures are exactly the
> strict Fig. 7/F24 source-provenance checks; bypass-only validation retained
> all payload/certificate checks and passed, and an exact reachability audit
> found the Kaplan endpoint numerically inert on all five artifact paths.
> Final static gates are clean. Exact regeneration of Fig. 7 and the
> summary-only F24 artifacts, source-honest handling of full-state F24
> re-certification, hosted CI, and a durable commit remain pending.

> **2026-07-19 follow-up (historical and subsequently superseded):** an
> independent 123-agent audit initially reported that it had confirmed the
> core engine and this document's claims, but found four
> distinct high-severity defect classes in the `scripts/` campaign
> drivers and the paper-anchor/M25 layer (an earlier tally said five;
> see the errata in the fixes doc). All are fixed on branch
> `codex/audit-fixes-2026-07-19`. Four later external-review rounds and Round
> 6 overturned parts of that initial conclusion — see
> [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md).
> The tables below describe the pre-fix tree `71c5f02`.

Snapshot date: **2026-07-19**
Audited code head: **`71c5f02310db0d65e7a9aa0bc5e09a4034d97bf3`**

Read the working-tree banner above first, then use
[`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md) for the live branch.
The remainder of this file is the compact historical `71c5f02` handoff, not a
replacement for the evidence in
[`AUDIT-2026-07-15-numerical-software.md`](AUDIT-2026-07-15-numerical-software.md).

## Scope of the completed work

The completed review was a **code-only numerical-software audit**. It was not a
paper audit. Papers were consulted only where necessary to check that an
implemented code path used the physics model, normalization, or operating mode
that it claimed to use.

Do not infer paper agreement from a passing figure-baseline test. Most figure
CSVs are self-pinned qpsim regression artifacts, not independently digitized
paper data.

## Repository state

| Item | Historical snapshot value |
|---|---|
| Remote | `Soren-O/Quasiparticle-Physics-Simulation` |
| Base | `origin/main` at `b92571a635aee9e8efd3fc228ac4ac7a7e69c150` |
| Audit branch | `codex/qpsim-deep-audit-fixes` |
| Audited code head | `71c5f02310db0d65e7a9aa0bc5e09a4034d97bf3` |
| Pull request | [Draft PR #5 — Fix qpsim numerical correctness and validation](https://github.com/Soren-O/Quasiparticle-Physics-Simulation/pull/5) |
| PR state at snapshot | Open, draft, merge state `CLEAN` |
| Exact-head CI | [Run 29667989929](https://github.com/Soren-O/Quasiparticle-Physics-Simulation/actions/runs/29667989929), green on Python 3.13 and 3.14 |
| Branch size | 8 commits ahead and 0 behind; 150 files changed; 26,623 insertions and 6,425 deletions relative to `origin/main` |
| Python | 3.13+; CI covers 3.13 and 3.14 |

The branch has **not been merged to `main`**. Before this handoff document was
created, the branch and remote were synchronized and the worktree was clean.
Treat `71c5f02` as the exact code/test/validation tree for the results below; a
later commit containing only this document does not alter that numerical tree.

## What the audit branch changes

This is a broad numerical repair branch, not a one-bug patch. The main repaired
contracts are recorded as findings N1–N43 and include:

- one matched finite-volume measure for BCS singular capacity, support,
  coherence factors, collision terms, photon partners, spatial transport,
  remapping, and observables;
- deterministic, branch-anchored nonequilibrium gap solving and fail-closed
  support coverage for self-consistent and moving-gap states;
- a stage-constrained moving-gap ETD2 update whose public occupation and gap
  satisfy the same algebraic constraint;
- raw residual and gain/loss backward-error certificates for Newton, coupled
  Newton, and Picard-facing steady-state paths;
- structured handling of coupled-Newton line-search failure and
  superconducting gap collapse, without converting unrelated failures into
  fallback states or silent `NaN` output;
- balance-preserving stiff collision stepping and fail-loud retry behavior;
- stricter grid/domain validation for collision operators, external sources,
  direct gap observables, and spatial transport;
- cache, provenance, schema, and atomic-write protections for validation
  artifacts;
- cross-platform deterministic validation policy: CI pins OpenBLAS, OMP, and
  MKL to one thread and uses measured OS-family envelopes only where exact
  Windows/Linux agreement is not justified;
- adversarial tests for unsupported grids, nonlinear pseudo-roots, non-finite
  derived measurements, singular-edge omissions, malformed inputs, and stale
  artifact metadata.

The final follow-up commit additionally fixes:

- the Fischer 2023 Fig. 5 high-drive thermal pseudo-root by tightening the
  inner Newton and final Picard/certificate contracts;
- Fischer 2023 Fig. 6 direct-mode grid coverage, crossing reconstruction,
  structured failure propagation, signed plotting, and canonical-file
  no-clobber behavior;
- the self-consistent diffusion benchmark's discrete fixed-point calibration,
  edge reconstruction, iteration cap, invalid-depth checks, and zero-capacity
  drift handling;
- roundoff-scale alignment of direct BCS gap-edge support;
- exact-source Fig. 7 regeneration and source-honest Fischer 2024
  regeneration/re-certification (producer and validator identities kept
  distinct);
- Windows CP1252-safe executable messages in the four Fischer 2024 generators,
  guarded by normalized-AST tests so numerical expressions remain unchanged.

## Historical `71c5f02` verified state

| Gate | Exact evidence |
|---|---|
| Local default aggregate | **1549 passed, 17 deselected, 4 warnings in 525.03 s** |
| Collection inventory | 1566 tests: 16 non-manual `slow` tests and one `manual_slow` test |
| Hosted Python 3.13 default | **1549 passed, 17 deselected, 4 warnings in 302.96 s** |
| Hosted Python 3.13 slow | **14 passed, 2 expected xfailed, 1550 deselected, 1 warning in 2235.62 s**; job wall 47m19s |
| Hosted Python 3.14 default | **1549 passed, 17 deselected, 4 warnings in 246.92 s** |
| Hosted Python 3.14 slow | **14 passed, 2 expected xfailed, 1550 deselected, 1 warning in 2140.25 s**; job wall 44m08s |
| Static/tooling gates | `ruff check .`, `mypy qpsim` over 75 source files, `compileall`, all seven bundled symbolic-verification scripts, and diff checks passed |
| Fischer 2023 Fig. 7 | Exact 48-target recertification completed in 1123.7 s; all axes, observables, and certificate arrays were bitwise identical to the active rows |
| Fischer 2024 artifacts | Exact current-solver recertification covered 84 states in 95.07 s; all certificates passed; maximum pinned/live row drift `2.58e-14` |
| Final focused slice | 82 passed, 2 deselected; Fig. 6 fast file 45 passed, 2 deselected; diffusion feedback 10 passed; gap suppression 18 passed |

The Fischer 2023 Fig. 7 solve-contract digest active at historical head
`71c5f02` is:

```text
ebe1382d509f6c52f11bca95b8d0161a211c4002a59f38de942cb2aefd193165
```

The only annotation on the exact-head hosted run is GitHub's non-failing
Node.js-20 deprecation warning for `actions/checkout@v4` and
`actions/setup-python@v5`; it is not a qpsim test failure.

On that historical hosted tree, the four default-suite warnings were one
Starlette dependency warning plus three explicit high-energy-tail diagnostics,
and the two slow xfails were the then-pre-v2 Fig. 5 artifact and the Figs. 9–13
legacy artifact. The current working tree has replaced the Fig. 5 quarantine
with a promoted state-bound canonical and an active full-recertification gate;
Figs. 9–13 remain quarantined.

## Figure and numerical-validation status

| Surface | Historical snapshot status plus superseding notes |
|---|---|
| Fischer 2023 Fig. 3 | **Corrected strict-v3 replacement promoted and current-equation recertified.** The source-frozen `NE=1620` producer completed all 14 continuation steps in `12022.121 s`. A final publication-layer repair rejects real occupations outside inclusive `[0,1]` and rejects complex values before coercion; republishing used the unchanged authenticated raw payload (`78c2e181fab5d3a25d5936e2bb5b76cbfb84fc3fcee7ba1066af65a3a2aa7a45`). Current CSV/PDF/validation-record SHA-256 are `2264f6c09f2917d5863d274a5edbaf0e8484e9ec86e51018720cf868a4378616`, `1f3c762a461a83d999dc9013ecd1f167c2c2f6963b2208ef518e211833e20e43`, and `6776776f643e73667fe4f836c8352ded670f4e0c239addbf5f85516e53b3f89b`; 58 focused checks passed with 3 slow/manual deselections and visual inspection passed. The finite-ratio `n_ph` omission qualification remains explicit. This is a fixed-grid qpsim regression, not paper parity. |
| Fischer 2023 Fig. 5 | **Current state-bound v3 canonical complete and promoted; grid refinement remains separate.** Six independent single-thread continuation rows completed all 81 points under campaign identity `01e22c384d6473ec12df22bf3f557af544cd66a6d8f2fc12b80b1d610dedcccb` in `9380.161 s` wall (`35471.716 s` aggregate worker time). CSV/two-PDF/promotion/campaign SHA-256 are `4e187e3286e45cb45e0a0b580789559fb8989044e7c80e9b5c10090b80af80df`, `d12bb18591102a7a833edde026cb0581b50bd225f0846004a6d6534b4400895c`/`e630769039c2853c4209dd33bbcac74493bc9a5048b9e95788df341191ce6bd5`, `a48cbc150aec9952b5175e25416c5e7a656318442d27dd64950ab8b044ef391a`, and `f43d2938e31a2011f6e9a816fd1e2489a50e0a9dee704f250693b4abe9efb3fa`. Maximum QP residual/backward/number-backward errors were `1.388e-17`/`1.207e-16`/`9.770e-11`; phonon residual/raw-backward/certified-backward maxima were `1.201e-11`/`4.391e-7`/`9.855e-10`. Exact pre-coercion dtypes and all six raw-row hashes were independently checked; the focused closeout passed 56 tests, the one-pass 81-state slow recertification passed in `84.08 s`, and both PDFs passed visual inspection. This closes tight-contract canonical production at fixed `NE=1620`; it does not establish commensurate-grid refinement or digitized-paper parity. |
| Fischer 2023 Fig. 6 | **Current canonical complete and promoted, with scope qualification.** The three-row campaign completed all 66 points in `15223.850 s` wall (`43470.213 s` aggregate worker time). CSV/PDF/promotion-record SHA-256 are `384f82344fd2e194e733f184f608587670afc1c23f2e4585cc313e2136d01c11`, `52f7372c53ac87b3039f6461aa94188d714f86524b65579a2ca68dc7694a0837`, and `bfefdb0f728d1486dde35a2f9fcdd3ae08df51ee8d14c8f91ad6047d368f00f7`. The provenance-only migration retained all 66 numerical rows, payload hash, PDF bytes, and historical `cbfb0006…f9a43` producer identity while making reader-host float drift non-authoritative. Maximum QP residual/backward/number-backward errors were `9.272e-15`/`6.066e-8`/`9.877e-7`; maximum phonon residual/raw-backward/certified-backward errors were `3.691e-12`/`1.099e-5`/`9.757e-6`; maximum gap-map absolute error was `9.911e-11 µeV`. The canonical PDF intentionally retains the paper window; the separately authenticated noncanonical signed diagnostic exposes all 66 finite numerical and Eq. 53 samples, including 27/26 samples hidden by that window. Neither artifact establishes digitized-paper parity or continuum refinement. |
| Fischer 2023 Fig. 7 | **Exact current-source regeneration complete and promoted.** All 48 targets completed under hardened identity `ea166442…` and solve digest `d674ca…` in `4387.907 s` wall (`15458.707 s` aggregate worker time). CSV/PDF/promotion-record SHA-256 are `2bb97283…5634`, `d0c3029f…7586`, and `32fc656b…f37d`; 67 focused checks passed with 2 slow deselections, and strict readback plus visual inspection passed. Maximum gated QP/QP-number/representability-aware phonon errors remain below their declared limits; the raw direct-form phonon diagnostic is retained as representability evidence rather than used as the gate. This remains a scoped fixed-grid regression, not bitwise portability or paper parity. |
| Fischer 2023 Figs. 9–13 | **Quarantined.** Low-power `Q_i` is still nonconverged: the aligned `NE=3240 -> 6480` rung changes by 4.44368%. Existing evidence does not justify rewriting the photon operator. |
| Fischer 2024, four families | Freshly regenerated through their real solve paths and promoted as strict-v3 CSV/PDF pairs; the focused collection passes 69 tests with 4 slow deselections. These are independently certified **qpsim-native** regressions at paper topology. Analytic paper-target overlays remain incomplete. |
| Diffusion feedback benchmark | The default `NE=24`, `NX=201`, 10% well now has a discrete fixed point and certifies its raw map. The bounded guard plateau and fixed edge-node reconstruction are intentional parts of this benchmark contract. |
| Moving-gap integration | Verified second-order only within its documented ideal-BCS, uniform-work-grid, spatially homogeneous DAE and support domain. |

## Open work and residual risk

The historical `71c5f02` snapshot had no known failing tests or merge
conflicts. The final Round-7 non-slow aggregate passed 2188 tests with
0 failures, 18 intentional deselections, and 12 warnings. Round 8 now has
current promoted Fig. 3/5/6/7 and F24 bundles. Its terminal current-tree
default gate passed 2509 tests with 1 intentional opt-in skip, 20
slow/manual deselections, and 12 warnings in 836.01 s; the bounded non-manual
slow selection passed 15 tests with the one expected Figs. 9–13 quarantine
xfail, 2514 deselections, and 1 warning in 4497.19 s. The separately opted-in
Fig. 5 six-row archive reauthentication passed in 96.14 s. Future hosted CI
remains separate post-push evidence. The important remaining items are
numerical-qualification work, not hidden green checkmarks:

1. **Fig. 5:** tight-contract canonical production is complete. A future
   commensurate-grid refinement campaign is a separate qualification required
   before making continuum-accuracy claims beyond the fixed `NE=1620`
   regression.
2. **Fig. 6:** canonical default/self-consistent production and promotion are
   complete. Any future `_direct`-mode or grid-refinement study is a separate
   qualification and must not be described as pending canonical production or
   digitized-paper parity.
3. **Figs. 9–13:** continue commensurate refinement. The audit proposes—without
   claiming a derived error budget—`<=1%` maximum `Q_i` change on two
   consecutive rungs plus `<=0.25%` exact-cell/FV observable discrepancy before
   promotion.
4. **Fig. 3:** keep quantitative claims in scalar moments and weak measures
   unless the ideal-BCS threshold model is explicitly regularized; strong-norm
   threshold layers remain slowly convergent.
5. **Fischer 2024:** paper-target analytic overlays are still incomplete even
   though the qpsim-native fixed-grid regressions are certified.
6. **Roadmap:** T2/T1 electronic backends and Ph1/Ph2 phonon transport are not
   implemented. These are roadmap gaps, not regressions introduced by this
   branch.
7. **Orthogonal manuscript question:** `papers/qp-diffusion` still has contested
   physics items documented elsewhere. Passing its symbolic scripts proves the
   code/manuscript identities agree; it does not prove the manuscript physics.
   Do not turn the next code audit into a paper audit unless explicitly asked.

## Behaviors that are intentional

Do not “fix” these without new evidence:

- Direct gap integrals raise when the energy grid does not cover the
  superconducting edge. Only roundoff-sized positive face offsets are aligned.
- The canonical Fig. 6 paper-style PDF deliberately keeps the paper window;
  its authenticated CSV retains every signed finite value. The noncanonical
  signed diagnostic plots those canonical samples on a full signed scale. The
  separate `_direct` mode also retains finite signed suppression values. Only
  explicit superconducting collapse maps to `NaN`; other numerical failures
  propagate.
- `solve_gap` warns near `T_c` when the grid cannot represent below-gap
  occupation. This is a declared domain limitation.
- The phonon pair-breaking kernel uses the current gap, not the zero-temperature
  gap, under its documented approximation.
- Spatial Crank–Nicolson transport now subcycles under-resolved diffusion
  steps automatically and fails loudly only when the required substep count
  exceeds its explicit safety cap.
- The default pytest configuration excludes `slow`; CI runs
  `slow and not manual_slow` separately.
- M25 steady-state acceptance uses a row-wise source-scaled residual plus a
  backward-error gate. Replacing it with a bare absolute threshold would
  reinstate an already-fixed conditioning error.
- `validation/sweep_cache.py` intentionally excludes downstream observables
  from the Fig. 7 **solve-source** digest. Artifact/dependency tests cover the
  downstream contracts separately.
- Canonical Fig. 6 output, the canonical-data signed diagnostic, and
  direct-mode output are distinct. Programmatic or CLI direct generation must
  use the `_direct` CSV/PDF paths.
- A Windows/Linux regression envelope is an OS-family calibration, not a claim
  of hardware-independent bitwise identity.

## Documentation hierarchy and known doc debt

Use the documents in this order:

1. [`AUDIT-2026-07-19-fixes.md`](AUDIT-2026-07-19-fixes.md) — live branch repair and verification record.
2. [`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md) — adjudication ledger, including overturned verdicts.
3. [`CURRENT-STATUS.md`](CURRENT-STATUS.md) — this historical `71c5f02` branch/CI handoff; use its opening banner for current evidence.
4. [`AUDIT-2026-07-15-numerical-software.md`](AUDIT-2026-07-15-numerical-software.md) — full original findings, derivations, timings, hashes, and unresolved-risk evidence.
5. [`NEXT-AUDIT-BRIEF.md`](NEXT-AUDIT-BRIEF.md) — concise rabbit-hole and false-positive guide.
6. [`STATUS.md`](STATUS.md) — broad project/gate history, including work outside this audit.
7. [`../ENGINE-REVIEW-NOTES.md`](../ENGINE-REVIEW-NOTES.md) — integration history and older timing context.

Known documentation debt:

- The current working tree has corrected `README.md`'s former unqualified
  Fischer paper-parity checkmark. Older status/audit text that predates that
  edit remains historical evidence, not an independent paper-data claim.
- The first Fig. 6 post-promotion closeout run took `336.06 s` because its
  advertised fast preflight replayed all 66 states four times. The locked
  scalar preflight now checks exact bytes/config/axes/stored certificates; the
  two corrected checks take `5.2 s`. Full state-derived authentication remains
  the separate `slow` gate
  `test_canonical_bundle_authenticates_and_recertifies`; the live solve
  comparison remains `manual_slow`.
- Fig. 5 had the same test-composition problem: its former fast preflight
  replayed all 81 stored states twice (`160.88 s` in the test body). The
  promotion-locked scalar preflight now takes `1.71 s`, while one explicit
  slow full recertification takes `82.58 s`. Separately, the source-frozen
  publisher still validates all 81 states five times; this added
  `504.201 s` after the solve phase and is deferred to the next
  provenance-breaking publisher revision.
- Older hosted counts of 1513 tests belong to earlier exact trees. The current
  `71c5f02` hosted count is 1549.
- `ENGINE-REVIEW-NOTES.md` retains an explicitly historical 802-pass command;
  its opening disclaimer is important.
- Focused test counts overlap. Do not add them together to manufacture an
  aggregate larger than the collected suite.

## Recommended first pass for the next agent

1. Confirm identity before drawing conclusions:

   ```bash
   git status -sb
   git rev-parse HEAD
   git diff --stat origin/main...HEAD
   gh pr checks 6
   ```

2. Read the audit outcome, fixed-findings table, verification table, and
   explicit unresolved-risk section. Do not start with the historical audits.
3. Independently review the highest-risk invariants in:

   - `qpsim/physics/spectral.py`
   - `qpsim/physics/gap_equation.py`
   - `qpsim/observables/gap_suppression.py`
   - `qpsim/solvers/coupled_newton.py`
   - `qpsim/solvers/etd.py`
   - `qpsim/backends/t3_diffusion.py`
   - `qpsim/backends/t3_spatial_1d.py`
   - `validation/fischer_2023/`
   - `validation/fischer_2024/`

4. Prefer adversarial invariant checks over reproducing plots: represented
   support, matched measures, raw residuals, gain/loss backward error,
   conservation, branch continuity, cache keys, and fail-loud boundaries.
5. If a change touches solver-source provenance, do not casually rewrite a
   canonical CSV. Recompute into a temporary path, independently certify it,
   read it back through the strict artifact reader, compare rows/certificates,
   and promote only source-honest evidence. Fig. 7 and summary-only F24
   artifacts require exact regeneration. A full-state re-certification may
   record a later validator identity, but it must preserve the producer
   identity and must not be described as current-algorithm execution.

## Reproduction commands

Install with Python 3.13 or 3.14:

```bash
pip install -e ".[dev,ui]"
```

Use one BLAS thread for comparable numerical evidence:

```bash
export BLIS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

PowerShell uses the same names, for example
`$env:OPENBLAS_NUM_THREADS="1"` and `$env:OMP_DYNAMIC="FALSE"`.

Core gates:

```bash
ruff check .
mypy qpsim
python -m compileall -q qpsim scripts tests validation
make -C papers/qp-diffusion verify PY=python
pytest -q
pytest -q -m "slow and not manual_slow"
```

Useful focused gates:

```bash
pytest -q tests/observables/test_gap_suppression.py
pytest -q validation/diffusion_operators/test_self_consistent_feedback.py
pytest -q validation/fischer_2023/test_fig5_paper.py
pytest -q validation/fischer_2023/test_fig6_paper.py
pytest -q validation/fischer_2024
```

Figs. 5 and 6 use promotion-locked scalar preflights in the ordinary fast
gate. CI's non-manual slow slice performs one full canonical state
recertification per figure. Their multi-hour `manual_slow` wrappers remain
opt-in because they re-run the complete live production comparisons rather
than merely authenticating the promoted bundles.
