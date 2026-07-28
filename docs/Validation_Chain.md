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
`validation/baselines/{ph0_constant, ph0_kaplan, transient,
marchegiani_2025}/`. The paired tests cover some combination of solver
residual/certificate checks, exact producer provenance, artifact
currentness, formula transcription, and regression against an earlier
qpsim solve. Full-size producers are generally slow or manual-slow; the
default test gate authenticates their promoted artifacts rather than
rerunning every solve.

No canonical CSV in these directories is digitized paper data. A separate
paper-data layer now exists under `validation/paper_data/`: its first dataset
authenticates and digitizes the original author raster for Fischer-2023
Fig. 6, then scores the promoted qpsim scalar arrays downstream without
modifying or replaying the expensive producer. Optional raster side-by-side
helpers remain manual visual aids only and are not part of that score.
The scorer binds and validates the exact promoted CSV/promotion-record bytes;
it does not decode and recertify all 66 stored states. That authoritative
state replay remains the separate slow gate under the recorded single-thread
environment.

The evidence layers must therefore be read separately:

| Layer | What it establishes | What it does not establish |
|---|---|---|
| Analytic identity/formula tests | A transcribed helper or limiting identity evaluates as expected | Agreement of a full numerical curve with the paper |
| Solver certificates | The discrete equations/residual contracts are satisfied on the represented grid | Continuum accuracy or agreement with experiment |
| Provenance/currentness | CSV/PDF/records came from the declared source, configuration, runtime, and raw payload | Physical correctness of that source |
| Pinned qpsim regression | A current solve stays within the declared tolerance of a prior authenticated qpsim solve | An independent reference truth |
| Digitized paper-data parity | Fig. 6 source/archive/raster/calibration/points, comparison mapping, and scorer are independently versioned and hashed | A full scientific pass while parameter or discretization uncertainty remains unbounded |

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

The Fischer-2023 source manifests are deliberately conservative: their solve
identities hash the broad qpsim numerical tree, and artifact identities hash
the complete figure modules. Fig. 6 also has a real Eq. 47 dependency on a
Fig. 5 analytical helper, but currently hashes both full Fig. 5 modules; a
plotting, comment, or unrelated solver edit can therefore invalidate a
multi-hour Fig. 6 result even when Eq. 47 is unchanged. A future refactor
should isolate Eqs. 35/47/53 in a small shared analytical module and derive
overlays at publication time. Conversely, validation package `__init__.py`
files are not currently enumerated. They contain docstrings only; executable
initializer behavior must not be added without first extending the manifest
closure.

### Digitized paper-data contract

The implemented Fig. 6 layer remains separate from qpsim-generated baselines
and fails closed unless it records all of the following:

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
| Fig 6, paper-topology gap suppression | `fig6_paper.py` + downstream `fig6_paper_parity.py` | state-bound v2 qpsim regression plus an independent arXiv-v2 raster oracle; dashed Eq. 53 controls pass, solid numerical curves show a diagnostic mismatch; parity is not gate-eligible pending parameter/discretization bounds |
| Fig 7, paper-topology Q_i,tot(T_B) | `fig7_paper.py` | summary-v2 qpsim regression; solved state is omitted, so certificate scalars are authenticated producer assertions requiring explicit opt-in; Tables II/III parameters + Eq. 65 helper |
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

### Marchegiani 2025 (`validation/marchegiani_2025/`)

All sweeps run through the branch-continuation driver
(`qpsim.services.rate_equation.solve_rate_equation_branch`) on the
Γ̄-normalized density equations (2026-07-04); the historical
multi-stability noise was a conditioning artifact of the missing
`Γ̄ = Γ̃/N_CP(R)` normalization and is gone. All fast — the whole
directory runs in the default gate (~15 s).

| Figure | Module | Status |
|---|---|---|
| Eq. 8 Lambert-W T̄ | `fig3_crossover_temperature.py` | closed-form, machine precision |
| Fig 3, μ_α vs T (small + large gap asymmetry) | `fig3_chemical_potentials.py` | authenticated qpsim regression with reader-reassembled full-state residual certificates and transcribed paper-formula μ inversions (SI Eqs. S2–S5); broad topology is checked manually, not against digitized points |
| Fig 3, paper-styled panels + insets | `fig3_paper.py` | paper-topology qpsim regression; CSVs and one-page PDF are one manifest-authenticated bundle; manual broad paper anchors only |
| Fig 4, Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T | `fig4_parity_rates.py` | authenticated summary-observable qpsim regression; the artifact records a producer assertion because raw branch states are not persisted; manual broad paper anchors only |
| Fig 4, paper-styled two-stack with comparison models | `fig4_paper.py` | paper-topology qpsim regression: full model + global-QE and renormalized reductions; summary-only CSV and one-page PDF are authenticated together, without claiming reader-side residual reconstruction or digitized-data parity |

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

Selected Fischer numerical regressions at larger grids. The
Marchegiani sweeps are fast; the transient live regression is slow-marked.
`manual_slow` producers are separate regeneration work and are not
implied by a green default or hosted slow gate.

## See also

- `STATUS.md` — running gate tracker, current test count.
- `Part_II_Physics.md`, `Part_III_Numerics.md` — what's being
  validated.
