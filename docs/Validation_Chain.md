# Validation Chain

Inventory of test and validation tiers under `tests/` and
`validation/`. The canonical reference is `New Framework Plan.md` §6;
this document maps the tiers onto the live test directories.

## Tier 1 — Analytic fixed points (`validation/analytic/`)

Identities that must hold at thermal equilibrium or at known
parameter limits. Fast and included in the default test gate (not slow-marked).

- `test_detailed_balance.py` — e-ph, sub-gap photon, and pair-breaking
  photon channels each vanish at `(f = f_FD(T), n_ph = n_BE(T))` to
  roundoff on the active window.
- `test_mattis_bardeen_thermal.py` — `σ_1 → 0` as `T → 0`;
  `σ_2 → π Δ/ω` (kinetic-inductance limit).
- `test_gap_equation_equilibrium.py` — `solve_gap(f_FD(T_B))` recovers
  `Δ_eq` from `calibrate_gap(T_c, T_B)`.

## Tier 2 — Tier reductions (`validation/tier_reductions/`)

Structural placeholder for T1 → T2 → T3 reductions when those
backends ship. Empty in v1.

## Tier 3 — Paper-topology numerical and artifact regressions

Pinned against qpsim-generated CSV baselines and PDF plots under
`validation/baselines/{ph0_constant, ph0_kaplan, transient}/`. The paired
tests cover some combination of solver
residual/certificate checks, exact producer provenance, artifact
currentness, formula transcription, and regression against an earlier
qpsim solve. Full-size producers are generally slow or manual-slow; the
default test gate authenticates their promoted artifacts rather than
rerunning every solve.

No canonical CSV in these directories is digitized paper data. A separate
paper-data layer now exists under `validation/paper_data/`: its Fischer-2023
Fig. 6 and Fig. 8 datasets authenticate and digitize original source assets.
The Fig. 6 layer scores promoted qpsim scalar arrays downstream without
modifying or replaying the expensive producer; the Fig. 8 layer currently
scores only a qpsim-independent Eq. E2 transcription against the dashed
trace. Optional raster side-by-side helpers remain manual visual aids only
and are not part of either score.
The Fig. 6 scorer binds and validates the exact promoted CSV/promotion-record
bytes; it does not decode and recertify all 66 stored states. That
authoritative state replay remains the separate slow gate under the recorded
single-thread environment.

The evidence layers must therefore be read separately:

| Layer | What it establishes | What it does not establish |
|---|---|---|
| Analytic identity/formula tests | A transcribed helper or limiting identity evaluates as expected | Agreement of a full numerical curve with the paper |
| Solver certificates | The discrete equations/residual contracts are satisfied on the represented grid | Continuum accuracy or agreement with experiment |
| Provenance/currentness | CSV/PDF/records came from the declared source, configuration, runtime, and raw payload | Physical correctness of that source |
| Pinned qpsim regression | A current solve stays within the declared tolerance of a prior authenticated qpsim solve | An independent reference truth |
| Digitized paper-data parity | A declared figure's source/archive/raster/calibration/points, comparison mapping, and scorer are independently versioned and hashed | A full scientific pass while parameter or discretization uncertainty remains unbounded, or validation of an unscored trace in the same raster |
| Recorded author-supplied source | The ZIP the repository owner records as received and its declared members match checked content hashes | Independent sender-chain authentication, historical-publication identity, modern-runtime compatibility, or correctness |
| Clean-room equation reference | A qpsim-independent transcription reproduces a declared analytic trace | Author numerical-source identity or validation of qpsim numerics |
| Component-replacement ladder | Declared parent/child topology changes one named component, and completed-stage evidence files match checked hashes | That the implementations truly differ only in that component, or that a measured output change has already been localized |

The Fig. 6 result is intentionally split: all three dashed analytic controls
agree with the digitized source within the predeclared raster-uncertainty
limit (maximum normalized errors `0.388`, `0.200`, and `0.253`), while all
three solid numerical curves fail that same diagnostic (`8.05`, `9.13`, and
`7.59`; maximum relative discrepancies about `33%`, `39%`, and `37%`) over
seven sampled points on the visible rising branch
(`T*/Delta ≈ 0.250–0.410`). This does not characterize the unsampled parts of
the curves.
Accordingly, the durable status is `diagnostic_mismatch` and
`gate_eligible=false`; the analytic control is evidence that the calibration
and curve identity are sensible, not permission to waive the numerical
discrepancy.

The recovered author-supplied Fig. 6 archive adds a new distinction. Its
authenticated entry point uses fixed-gap kinetics and a direct edge-grid gap
observable, whereas the promoted qpsim comparison uses the self-consistent
model. The old solid-curve mismatch remains a valid output comparison, but
is not attributable until the fixed-gap/direct-observable author-equivalent
ladder reaches its qpsim endpoint. See `docs/PAPER-REPRODUCTION-LADDER.md`.

One exact author point in that ladder is complete, not the full 300-point
replay. At `T_B = 0.20 K`, original sweep index 49, and `N = 1620`, the
author source returned `T*/Delta = 0.33990789737294363` and ordinate
`0.12090908988993258`; the digitized paper trace is approximately
`0.12093496` at the nearby paper-oracle coordinate `0.34006849`, while the
promoted qpsim comparison is approximately `0.08967258`. The formal qpsim-free
C0 port verifies all four author-policy Newton transitions, reproduces the A1
full state to `3.951e-16` relative L2 (`6.581e-14` for the QP subvector), and
returns the identical ordinate. C1 applies qpsim's direct-gap observable to
both the exact C0 occupation and the independently reconstructed author
thermal occupation; with exact inherited binary64 parameters, both gaps and
the final ordinate match bit-for-bit. Formal C2 then changes only explicit
Figure 6 parameters on the immutable C0 state: author-value plumbing,
`180.0 µeV` energy literals, modern `k_B`, literal `1 Hz` photon coupling,
`tau_0^PB = 0.255 ns`, and finite-cutoff-derived
`T_c = 1.184309192877208 K`. The generic Al YAML is excluded. Its checked
score independently recomputes all 124 retained channel and residual arrays
bit-for-bit. With authenticated `n_bar` held fixed, the Eq. 35 coordinate
moves from `0.33990789737294363` to `0.3399503360830364`; because C2 does
not re-solve, it claims no changed root or ordinate.

Formal C3 then adopts the live 1640-cell qpsim grid as a frozen differential
from C2b5. The 1620 parent cells map ordinally to child indices `20:1640`;
there is no interpolation, the active occupations copy bit-for-bit, and the
20 zero-capacity guard cells carry canonical positive zero. The full
`0..3599 micro-eV` qpsim phonon lattice is recorded, while the author
`1..1619 micro-eV` support remains the only evaluated support so C3 does not
silently replace the C6 phonon balance.

The checked [`c3-grid-score.json`](../validation/paper_data/fischer_2023/fig6/c3-grid-score.json)
independently derives the full BCS finite-volume weights/coherence and
reassembles every gain/loss/net channel for C3p (projection control), C3a
(finite-volume coherence), C3b (center pair labels), and C3c (native
cell-density arithmetic). C3p reproduces every active C2b5 channel array
bit-for-bit. The 449 non-bit-identical mapped **left faces** are explicitly
reported, with roundoff bounded by `2.2737367544323206e-13 micro-eV`;
separately, all 1620 author left-edge samples move to qpsim center carriers
approximately `+0.5 micro-eV` away. The score verifies both the nearly
invariant author-semantics re-embedding control and the actual center-carrier
diagnostic (`+11.1321%` driven integral, `+2.9560%` thermal integral, frozen
ratio `0.1209090899 -> 0.0510985119`), so the latter is not hidden in the
former. This completes only the one-point
frozen C3 contract: no root, Newton history, stopping result, ordinate, curve,
or paper parity is claimed.

Formal C4 then holds that accepted C3c state and spectral grid fixed and
substitutes only qpsim's public sub-gap photon operator. Its checked
[`c4-photon-score.json`](../validation/paper_data/fischer_2023/fig6/c4-photon-score.json)
independently replays the selected C3 and C2 raw parents and transcribes the
public loop without importing the producer or public operator. The comparison
correctly converts the returned loss-rate coefficient into physical loss
(`loss_rate * f`) before comparing it with C3c.

The public and author-form photon balances agree to approximately `2.02e-15`
symmetric relative L1. The only semantic difference is the public operator's
representable terminal transition `1619 <-> 1639`, omitted by the author
residual; its two frozen net contributions are only about
`-2.88825e-35` and `+2.88858e-35 s^-1`. The public weighted number drift is
about `4.27e-17` of photon turnover. C4 reconstructs the changed QP residual
while binding all non-photon channels and the phonon residual unchanged.
This is still one frozen point: no root, Newton history, stopping result,
observable, ordinate, curve, or paper parity is claimed.

Formal C5 then holds that accepted C4 state, grid, projected phonon
occupation, public photon channel, and every C3c phonon-equation channel
fixed. Its checked
[`c5-qp-phonon-score.json`](../validation/paper_data/fischer_2023/fig6/c5-qp-phonon-score.json)
replaces only the QP-side scattering and pair/recombination operators.
C5s, C5p, and C5sp isolate the scattering, pair, and combined QP residual
updates. The inherited phonon residual remains bit-exact.

The combined physical net agrees with C4 to `5.677749589118144e-16`
symmetric relative L1. The author's source-order scattering buckets include
the same Pauli cross-term in both gain and loss, while public qpsim removes
it from both. Raw gain/loss differences are therefore not like-for-like; the
rebucketed gain/loss L1 differences are only
`9.50321362825228e-14`/`1.9342844642219452e-13 s^-1`, and the physical
scattering net agrees to `5.682685376326191e-16` symmetric relative L1.
Pair gain, loss, and net each agree within `4.48e-16` symmetric relative L1.
Scattering passes its weighted-number conservation gate; pair
generation/recombination intentionally changes QP number, so its nonzero
number moment is retained as a diagnostic rather than tested against zero.

C5 is still one frozen point. It makes no root, Newton-history, stopping,
observable, ordinate, 300-point-curve, coupled QP-phonon-conservation, or
paper-parity claim. The inherited author phonon equation is not evaluated by
qpsim until C6.

Formal C6 then holds that accepted C5 state, grid, projected phonon
occupation, and every public QP channel fixed. Its checked
[`c6-phonon-balance-score.json`](../validation/paper_data/fischer_2023/fig6/c6-phonon-balance-score.json)
replaces only the phonon-side balance with qpsim's public phonon-side
kernels, frequency map, `compute_phonon_source_sink` contraction, and
`ph0_local` bath-escape form on the full 3600-bin native lattice. The
scattering net matches the author channel to `1.279582321835755e-15`
symmetric relative L1, the same-kernel correction-off pair control to
`2.351064591980878e-16`, and escape gain/loss below `1e-12` with an
elementwise-bounded near-thermal net. The one material endpoint difference
is qpsim's Kaplan `S_+` pair-breaking quadrature correction: it moves the
public pair net by `9.203114358766813e-3` symmetric relative L1 and the
formal C6spe phonon residual by `2.0546278031187717e-6`, versus
`2.1765109703719185e-10` for the correction-off control. Detailed balance at
a native thermal control holds per channel below `6.3e-16` of turnover; the
C5 hybrid QP residual is inherited bit-exact. C6 is still one frozen point
and makes no root, Newton-history, stopping, observable, ordinate,
300-point-curve, coupled-conservation, or paper-parity claim; the nonlinear
solver remains the author policy until C7.

Formal C7 then performs the first re-solve. Its checked
[`c7-nonlinear-solver-score.json`](../validation/paper_data/fischer_2023/fig6/c7-nonlinear-solver-score.json)
replaces only the author Newton driver with qpsim's public
`coupled_newton_solve` under the accepted C4/C5/C6 operator configuration,
seeded from the identical C0-bound author continuation state. The solver's
residual assembly at the frozen C6 state matches the accepted hybrid
residuals to `1.32e-13`/`1.69e-16` symmetric relative L1; the solve
converges at the minimal public cap of 4 iterations (the author control's
count), with root balance ratios `3.7e-14`/`5.8e-12` against `1e-7`. The
re-solved ordinate is `0.14542377851441587` under author sampling versus
the exactly reproduced author control `0.12090908988993258`, and
`0.07767766591390296` under the center-carrier interpretation — adjacent to
the historical promoted qpsim `0.08967258`. The one-point attribution
therefore identifies observable sampling semantics as the dominant driver
of the historical qpsim-versus-author discrepancy. C7 claims no 300-point
curve, paper parity, or author-iteration-equivalence.

At the captured A1 frozen state, the author-equivalent phonon residual is
about `7e-11` of turnover. Frozen
substitutions identify the first substantive
implementation differences as left-edge versus finite-volume coherence and
the pair-frequency shift from `2 Delta + (i+j)h` to
`2 Delta + (i+j+1)h`. A third grid-level substep isolates the much smaller
binary64 difference from evaluating the analytically identical cell-density
formula in qpsim's native micro-eV units.

A same-seed single-point pilot has now re-solved those changes plus the
isolated native-DOS arithmetic substitution, with every independent variant
converging in four Newton iterations. The
ordinates are `0.12090908988993258` for the author control,
`0.12070758916263027` for coherence only (`-0.00020150072730`),
`0.14590106106941977` for the `+h` pair label only
(`+0.02499197117949`), and `0.14570562776829468` for both changes
(`+0.02479653787836`). C3c changes only the DOS arithmetic and returns
`0.14570561703489288`, a `-1.07334e-8` shift from C3b. The control reproduces the captured author state to
`3.951e-16` full-state relative L2 and the exact ordinate; independent and
continuation combined solves agree to `4.240e-20` relative L2 and the exact
ordinate. Independent and continued C3c states agree to `8.73e-19` relative
L2. QP residual/turnover ratios are approximately
`1.3e-15`–`2.1e-15`, and phonon ratios are approximately
`6.2e-12`–`1.60e-11`; the re-solved author control is `1.596e-11`, distinct
from the frozen captured-A1 diagnostic above. The checked
`staged-resolve-pilot.json` independently recomputes stage identities,
same-seed/path constraints, every Newton transition and linear backward error,
stopping tests, final bounds, channels, certificates, and observables from
retained external-array bytes. The solve preserves all three authenticated
author residual/Jacobian inconsistencies rather than silently
differentiating a corrected equation.

Before the public qpsim photon operator is substituted, C3c now separately
adopts qpsim's native `cell_density` arithmetic. Its first-cell binary64
difference from the retained author DOS is below the checked `2e-7` relative
bound, but its
one-point ordinate effect is only `-1.07334e-8`. This prevents C4 from
changing grid arithmetic and photon endpoint semantics at the same time.

The coherence effect is negligible at this point. The pair-label effect is
material but moves upward, away from the promoted qpsim ordinate
`0.08967258`, so it does not explain qpsim's lower value. This remains a
single-point pilot under author parameters, not the formal C2-parent result.
Formal C0–C7 evidence is complete at the declared one-point scopes,
while the Q0/Q1 model endpoints and the full 300-point replay remain
incomplete.

The Fig. 8 paper-data oracle independently binds both the blue solid numerical
trace and black dashed analytic trace. Its checked clean-room Eq. E2 score
accepts the dashed trace within raster uncertainty (maximum normalized error
`0.2971`). The blue numerical trace is retained as a provenance-bound,
digitized reference but remains unscored until an author implementation or
qpsim replacement is attached to a later reproduction ladder. The dashed
acceptance therefore makes no claim about the blue curve or qpsim.

The Fischer-2023 source manifests are deliberately conservative: their solve
identities hash the broad qpsim numerical tree, and artifact identities hash
the complete figure modules. Fig. 6 also has a real Eq. 47 dependency on a
Fig. 5 analytical helper, but currently hashes both full Fig. 5 modules; a
plotting, comment, or unrelated solver edit can therefore invalidate a
multi-hour Fig. 6 result even when Eq. 47 is unchanged. A future refactor
should isolate Eqs. 35/47/53 in a small shared analytical module and derive
overlays at publication time. The older figure-solve manifests do not
enumerate validation package `__init__.py` files; those initializers must
remain non-executable unless that older closure is extended. Formal C5 uses a
separate fail-closed source manifest that deliberately over-invalidates: it
binds the complete qpsim Python/material tree plus its producer, the executed
C2/C3/C4 replay verifiers, package initializers, and provenance helpers, and
rechecks the manifest after array production.

### Digitized paper-data contract

The implemented Fig. 6 and Fig. 8 layers remain separate from
qpsim-generated baselines and fail closed unless they record all applicable
items below:

1. paper/version/page/panel provenance, retrieval date, source/crop hash, DPI,
   and licensing note;
2. per-panel axis calibration (linear/log scales, units, tick control points)
   plus held-out calibration residuals;
3. tidy extracted points with curve identity, pixel coordinates, data
   coordinates, digitizer/operator/version, and digitization uncertainty;
4. declared unit/normalization conversion and interpolation of qpsim at the
   paper coordinates;
5. predeclared uncertainty-aware metrics and an error budget separating
   digitization, parameter, discretization, and solver effects; and
6. separate manifests hashing the paper data/calibration, comparison mapping,
   exact promoted qpsim artifact, scorer source, and scored output.

Adjacent raster images are useful manual review aids, but they satisfy none of
these quantitative requirements.

### Fischer 2023 (`validation/fischer_2023/`)

| Figure | Module | Tolerance |
|---|---|---|
| Fig 3, paper legend ratios 0 / 0.1 / 1 / 10 | `fig3_paper.py` | manual-slow producer; 1620-bin paper grid + phonon-side Eq. 12 kernels; authenticated qpsim regression, not digitized-data parity |
| Fig 5, paper-topology x_qp two-panel | `fig5_paper.py` | state-bound v3 qpsim regression; reader reassembles persisted `f/n_ph`; promotion + campaign companions bind the exact six raw continuation rows; Eq. 47 + Appendix-E analytic overlay is pinned separately |
| Fig 6, paper-topology gap suppression | `fig6_paper.py` + downstream `fig6_paper_parity.py` | state-bound v2 qpsim regression plus an independent arXiv-v2 raster oracle; dashed Eq. 53 controls pass, solid numerical curves show a diagnostic mismatch; one exact author point agrees with the paper; formal one-point C0/C1 plus frozen-state C2/C3/C4/C5/C6 evidence are complete, with every retained raw array independently recomputed; formal C4 shows public photon balance agrees with C3c to roundoff apart from a numerically tiny terminal-pair policy difference, formal C5 shows the QP-side scattering/pair physical net agrees with C4 at about `5.68e-16` symmetric relative L1 after like-for-like Pauli rebucketing, and formal C6 shows the phonon-side balance agrees to roundoff except for qpsim's Kaplan `S_+` pair-breaking quadrature correction (public pair net `9.2e-3` symmetric relative L1, formal residual `2.05e-6` versus `2.18e-10` for the correction-off control); the supplemental author-parameter C3 pilot shows a material upward pair-label effect that cannot explain qpsim's lower ordinate; C7, full author replay, and full author-equivalent qpsim parity remain incomplete |
| Fig 7, paper-topology Q_i,tot(T_B) | `fig7_paper.py` | summary-v2 qpsim regression; solved state is omitted, so certificate scalars are authenticated producer assertions requiring explicit opt-in; Tables II/III parameters + Eq. 65 helper |
| Fig 8, recombination-coefficient ratio | `extract_fig8_paper_data.py` + `fig8_cleanroom_parity.py` | provenance-bound blue solid and black dashed traces; the qpsim-independent Eq. E2 transcription passes the dashed-trace raster score, while the solid numerical trace remains an unscored oracle for a later replacement ladder |
| Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | **Development-only; no active accepted pin.** The historical canonical CSV is deliberately quarantined pending an independent energy-grid refinement study. A future promoted artifact would use the nominal 1e-4 `nbar_loop`/quadrature gate; this is not a literal paper figure. |

The resumable Fig. 5/6 campaign archives are intermediate evidence, not
canonical artifacts. Their frozen Round-8 readers validate physical values
after converting raw NPZ members to `float`; a synthetic complex, boolean, or
integer member can therefore lose its original dtype before the domain and
certificate checks. Until pre-coercion dtype rejection lands in a future
provenance-breaking regeneration, closeout must independently require every
raw member of every promoted row archive to match its exact dtype schema:
real `float64` state/axis/certificate arrays, with only explicitly declared
integer metadata such as Fig. 5 `num_bins` permitted to be `int64`.
That archive check says nothing about paper-data agreement; it only closes the
type-integrity gap for the accepted campaign.

Fig. 5 continuation is serial within each of six independent rows. Its durable
checkpoint is the authenticated whole-row NPZ: a restart reuses completed rows
but recomputes every in-flight row from its first point. The committed campaign
companion binds those six row hashes to the promoted CSV/two-PDF bundle and
certificate maxima. It does not claim durable per-point checkpoints or
digitized-paper parity. Closeout can set `QPSIM_FIG5_RUN_ROOT` to reauthenticate
the retained external rows, including their exact pre-coercion dtype schema.

Campaign row recertification must also use the exact recorded single-thread
environment (`BLIS_NUM_THREADS=1`, `MKL_DYNAMIC=FALSE`,
`MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`). A different BLAS reduction order can move a
recomputed normalized certificate by a few parts in \(10^{11}\), which is
physically negligible but deliberately outside the near-bitwise artifact
identity tolerance.

That near-bitwise policy is appropriate for resume-row identity, but the
frozen Fig. 5/6 public canonical readers currently reuse it under the caller's
ambient environment. Consequently, a valid canonical artifact can be rejected
with a generic certificate-mismatch error when the BLAS thread controls differ.
Current closeout and CI therefore use all eight controls above. In the next
provenance-breaking regeneration, split strict producer-environment row
authentication from portable canonical semantic recertification, using the
scientific gates or independently measured field-specific portability
envelopes rather than one blanket relaxed equality.

The canonical Fig. 6 PDF intentionally retains the paper's plotting window.
Consequently, finite signed samples outside that window remain authenticated in
the CSV/state bundle but are not visible in the paper-style panel. Run
`python -m scripts.render_fischer_fig6_signed_diagnostic` after a canonical
promotion to produce a separately authenticated, full-range, marker-based
diagnostic. That diagnostic exposes every finite stored sample and any
clipping, and records counts for nonfinite samples that cannot be plotted; it
is still a qpsim self-regression view. The separate
`validation/paper_data/fischer_2023/fig6/` oracle and score own the
digitized-paper comparison.

The Fig. 5 configuration preflight likewise authenticates the promotion
record, current fingerprint, scalar metadata, axes, table hash, and stored
certificate gates without decoding all 81 states. Full state-derived replay is
the separate non-manual `slow` test
`test_canonical_bundle_authenticates_and_recertifies`; the complete live solve
comparison remains `manual_slow`. Before this split, the advertised fast test
composed `read_baseline_metadata()` and `read_baseline()` and replayed all
states twice (`160.88 s` in the test body). The corrected preflight takes
`1.71 s`; the one-pass slow recertification takes `82.58 s` on the same host
and exact single-thread environment.

The source-frozen Fig. 5 campaign publisher still performs five full
81-state validation passes across assembly, publication, staged readback, and
final readback. After the sixth row became durable, those passes added
`504.201 s` before final status publication. Deduplicating them without
weakening rollback/currentness checks is deferred to the next
provenance-breaking publisher revision.

The Fig. 6 configuration preflight is deliberately scalar and fast. It holds
the publication lock, binds exact CSV/PDF bytes to the promotion record,
validates the live fingerprint and generation evidence, then checks the axes
and stored certificate columns without decoding all 66 states. Those scalar
checks reject negative/nonfinite metrics and gate all three certified backward
errors, including the amplitude-sensitive QP-number error. Full
state-derived replay is the separate non-manual `slow` test
`test_canonical_bundle_authenticates_and_recertifies`; the complete live solve
comparison remains `manual_slow`. This split was added after the former
preflight composed the source-frozen public readers and replayed the same
states four times (`336.06 s`); the two corrected fast checks take `5.2 s`.
The signed-diagnostic publisher reuses one authenticated canonical snapshot
through its internal PDF/JSON commit checks. It locks both output resources,
rebinds the cached result to the still-current canonical identities under the
canonical publication lock, and holds those locks through commit-marker
promotion; canonical or renderer-source drift fails closed without a second
state replay. An external diagnostic read still performs one fresh full
recertification.
The frozen public Fig. 6 reader retains a duplicate internal replay. Each
Fig. 5 public reader performs one replay, but composing its result and metadata
readers duplicates that work; the fast preflight above therefore avoids that
composition. Both families retain the ambient-environment portability
limitation until the next provenance-breaking reader revision.

### Fischer 2024 (`validation/fischer_2024/`)

| Figure | Module | Tolerance |
|---|---|---|
| Fig 5, paper-topology distributions | `fig5_paper.py` | state-backed v5 qpsim regression with reader-side certificate reassembly; drive normalization remains qualified |
| Figs 5–7, qpsim-native PB f(E) sweep | `figs_5_7_fe_pb.py` | state/curve-backed v4 qpsim regression with reader-side certificate reassembly; paper-topology framing only |
| Fig 8, qpsim-native PB x_qp(T_B) sweep | `fig8_xqp_pb.py` | summary-only v4 qpsim regression; certificate scalars are producer assertions requiring explicit opt-in |
| Fig 8, paper-topology density sweep | `fig8_paper.py` | summary-only v5 qpsim regression; certificate scalars are producer assertions requiring explicit opt-in |

### Retired paper reproduction: Marchegiani 2025

The Marchegiani 2025 paper-reproduction modules, tests, and canonical bundles
were retired on 2026-08-28. They remain recoverable from Git history. The
generic M25 rate-equation, device, and UI capabilities remain active and retain
their independent unit and integration coverage.

### Transient (`validation/transient/`)

`photon_kick_response.py` drives the ETD2 transient stepper from an exact
thermal initial state under a step photon kick. The current source contract is
strict-v4. If a strict-v3 canonical file is still present while current-source
regeneration is in progress, it is historical/stale evidence rather than an
accepted v4 artifact.

The v4 CSV stores sparse `f(E)` snapshots and a separately stored steady state,
not every internal ETD2 substep or an independently replayable trajectory. Its
reader can authenticate the source/configuration/runtime and paired
structurally parsed one-page PDF, then validate the stored state domain,
thermal seed, monotone snapshot times/`x_qp`, `x_qp` reassembly, endpoint
proximity to the steady state, and steady-state
residual/backward-error/pair-number certificates. Those checks establish
snapshot structure and endpoint semantics only. The slow-marked live
regression executes `run()` and compares the recomputed snapshots; that live
execution, not artifact readback alone, owns the intervening-dynamics claim.
The historical three-`dt` campaign established driver-partition
insensitivity rather than formal order and was not rerun for the current
artifact contract.

## Tier 4 — Unit tests (`tests/`)

Per-module tests mirroring the library layout. Run with `pytest -q`. Test
counts are intentionally reported in the dated status/audit record rather
than frozen here.

## Slow tier (`pytest -m slow`)

Selected Fischer numerical regressions at larger grids and the transient live
regression are slow-marked.
`manual_slow` producers are separate regeneration work and are not
implied by a green default or hosted slow gate.

## See also

- `STATUS.md` — running gate tracker, current test count.
- `Part_II_Physics.md`, `Part_III_Numerics.md` — what's being
  validated.
