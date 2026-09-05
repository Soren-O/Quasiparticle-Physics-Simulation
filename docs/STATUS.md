# qpsim status

> **2026-08-28 update:** the Marchegiani-2025 paper reproduction, its tests,
> and its canonical bundles are retired for now. Generic M25 engine/device/UI
> support remains. Fischer paper reproduction and certificate tests moved to
> the explicit `paper_validation` lane and no longer run in the default gate.
> The authoritative post-change default gate is **2,540 passed, 1 skipped,
> 2 expected xfails, and 851 deselected in 986.08 s**; 847 paper-validation
> tests remain explicitly collectible. Older entries below
> are retained as historical records.
>
> **Corrective-head M25 status (2026-07-28):** hosted run `30423446933` on
> `f319bf5` exposed 13 default-suite failures on both Python 3.13 and 3.14.
> The cascade had two roots: cancellation-floor producer/reader residuals were
> compared as values rather than independently certified, and the Eq. 8
> crossover artifact fingerprinted and exactly replayed host-generated
> `np.logspace` samples. The readers now use independent semantic residual
> gates and a semantic log-grid recipe with a few-ULP axis check; Eq. 8 is
> reassembled from the persisted axis. Five affected bundles were republished
> transactionally, with all eight CSV numerical payload hashes unchanged.
> M25 passes 37/37 on Windows and native Linux; the full Windows default suite
> passes 2559 tests with two intentional skips and 20 slow deselections.
> Hosted CI on the new corrective head is still pending.
>
> **Corrective-head CI status (2026-07-28):** hosted run `30409496531`
> passed Ruff, mypy, and the default suite on Python 3.13/3.14, then exposed
> three slow-gate portability defects in Fischer-2023 Figs. 5/6 and
> Fischer-2024 Fig. 5. The corrected artifacts and readers pass the exact
> failed nodes on Windows and native Linux; the combined changed-area suite
> passes 188 tests with two intentional external-input skips and seven slow
> deselections. The full local default suite passes 2,557 tests with two
> intentional skips and 20 slow deselections. A hosted rerun has not yet
> produced a verdict. The older green
> counts below are historical evidence and must not be read as a result for
> this corrected head.
>
> **Live CI-portability/provenance status (2026-07-28):** Fig. 6's 66
> persisted states were fully recertified under current code without a new
> solve. The numerical rows, payload SHA `5b237ad6…39ab`, worker semantic
> hashes, and PDF are unchanged. The migrated generation-v2 envelope preserves
> the original v1 run identity `cbfb0006…f9a43` and generation-time
> fingerprint while current artifact matching permits only eight ULP at float
> leaves and keeps source hashes, types, shapes, and axes exact. Current
> CSV/PDF/promotion hashes are `5a5e3c6e…bc9c` / `52f7372c…0837` /
> `a9a8f09d…a823`.
> Cross-platform reproduction is green for all six former CI failures; the
> default suite passes `2557` tests with `2` intentional skips and `20` slow
> deselections, and the full Fig. 6 state replay passes.
>
> **Live Round-9 validation status (2026-07-28):** Fischer-2023 Fig. 6 now
> has a separate, deterministic paper-data oracle derived from the original
> author raster in the exact arXiv-v2 source archive. All three dashed Eq. 53
> controls pass the predeclared raster-uncertainty metric, but all three solid
> numerical curves disagree with the published traces by roughly 33–39% at
> worst over seven sampled points on the visible rising branch
> (`T*/Delta ≈ 0.250–0.410`). The score therefore records
> `diagnostic_mismatch`, `numerical_parity_accepted=false`, and
> `gate_eligible=false`; parameter and discretization uncertainty remain
> explicitly unbounded. This cheap downstream score did not rerun or modify
> the promoted numerical producer. It byte-attests the promoted CSV/record;
> full stored-state recertification remains the separate slow gate under the
> recorded single-thread environment. The provenance-rebound checked score is
> `d92243ff…46b1`; its numerical scores and verdict are unchanged;
> its focused suite passes 19 tests plus one external-source opt-in skip, or
> 20/20 when the exact arXiv-v2 source archive is supplied.

> **Live Round-8 status (2026-07-28):** current-source Fischer-2023
> Figs. 3/5/6/7 plus the hardened Fischer-2024 and Marchegiani-2025 bundles are
> complete. Fig. 3's final CSV/PDF/validation record are
> `2264f6c0…8616` / `1f3c762a…e43` / `6776776f…9b89b`; its repaired
> artifact readers and writers enforce real occupations in inclusive `[0,1]`
> before any float coercion.
> Fischer-2023 Fig. 6's three single-thread temperature rows completed all
> 66 points in `15223.850 s` wall (`43470.213 s` aggregate worker time), and
> its CSV/PDF/promotion record are promoted at `5a5e3c6e…bc9c` /
> `52f7372c…0837` / `a9a8f09d…a823`. Fig. 5's six single-thread continuation
> rows completed all 81 points in `9380.161 s` wall (`35471.716 s` aggregate
> worker time); its CSV/two PDFs/promotion record are promoted at
> `c114a291…98bf`, `d12bb185…895c`/`e6307690…6bd5`, and
> `b1f86b0d…7f5f`. The separately published campaign companion
> `f43d2938…b3fa` binds the completed row hashes to that promoted bundle.
> The terminal current-tree default gate passed **2509 tests** with 1
> intentional opt-in skip, 20 slow/manual deselections, and 12 warnings in
> `836.01 s`. The bounded non-manual slow gate passed **15 tests** with the one
> expected Figs. 9–13 quarantine xfail, 2514 deselections, and 1 warning in
> `4497.19 s`; opt-in reauthentication of the six durable Fig. 5 rows passed
> 1/1 in `96.14 s`. Hosted CI remains separate post-push evidence. These
> baselines remain qpsim regression artifacts; the independent Round-9
> paper-data oracle is a separate evidence layer.

> **Historical Round-8 snapshot (2026-07-27; superseded above):** artifact
> regeneration was then in progress on `codex/audit-round7-fixes`, based on
> `6d3c512`. The final current-source Fischer-2023 producers had not yet all
> published. No figure baseline in this repository was or is digitized paper
> data.

The detailed block below begins with the **historical Round-7 snapshot**. Its
counts, hashes, and completion claims are not current Round-8 evidence.

Historical narrative revised 2026-07-28 (spanning the Round-7 numerical audit
follow-up and its Round-8 closeout; see
`docs/AUDIT-2026-07-19-fixes.md`). Paper-reproduction status is qualified:
self-pinned CSV parity is regression evidence, not independent paper truth.
The corrected Fischer 2023 Fig. 3 artifact completed its source-frozen
full-grid solve in `12022.121 s`, passed independent certificate/readback and
visual checks, and is promoted with its validation record. The prior
49.29-hour route was pathological solver policy, not intrinsic physics cost.
All four Fischer 2024 families were genuinely regenerated after the final
source-digest change and promoted as strict-v3 pairs. Exact current-source
Fig. 7 completed all 48 targets under the hardened content-addressed driver;
its matched CSV/PDF/attestation passed strict readback, independent
attestation, and visual inspection. Fig. 3 persists the new
amplitude-sensitive certificate, and Fig. 7 now persists it too; legacy
summary artifacts are not themselves proof of the number mode. At the
Round-7 snapshot Fig. 5 and Figs. 9--13 were quarantined; Round 8 has since
promoted the state-bound Fig. 5 replacement, while Figs. 9--13 remain
quarantined. At the Round-7 snapshot the repaired Fig. 6
direct path lacked a full campaign; the current promoted canonical is now
complete in default/self-consistent mode. `_direct` outputs remain distinct,
and the canonical-data signed diagnostic is not a `_direct` artifact. Prior:
2026-07-04 M25 closeout:
single-quasiparticle Γ̄ =
Γ̃/N_CP(R) normalization fix in the density equations
(`M25Coefficients.cooper_pair_number_R`), branch-continuation driver
`solve_rate_equation_branch`, transcribed paper-formula μ inversions, real Fig-4
comparison models (global quasiequilibrium + renormalized), all M25
baselines regenerated and pinned tight — see the Marchegiani junction
path and Device Architecture rows, plus the Marchegiani
validation-figure rows below.
Prior: 2026-07-03 web frontend `qpsim.webui` shipped — see
`docs/Frontend.md`; optional extra `qpsim[ui]`, CI installs it; the
only engine changes are the physics-neutral `progress_hook` on
`services.transient.run_time_dependent` and
`SpatialBackend.run`, plus — from the same-day
frontend code review — float coercion in the materials loader
(YAML 1.1 loads unsigned-exponent notation like `1.74e28` as a
*string*; Al/Nb/TiN `v_F`/`rho_F` were affected), a public
`COMMENSURATE_TOL` alias in `collisions.sub_gap_photon`, and a derived
`H_OVER_KB_K_PER_HZ` in `qpsim.constants`. Prior: 2026-07-02
a1-diffusion-operators merged to main; CI green; code-health review
fixes).

Central snapshot of what's done and what's in progress.

## Capabilities

| Capability | Status |
|---|---|
| Phonon-model decisions (D1–D5) | ✅ resolved in `docs/Phonon_Model_Decisions.md` |
| Repo skeleton | ✅ (`77fd516`) |
| Physics + grid + collisions + solvers ported from legacy qpsim; fixed-gap ETD2 / persistent-coordinate stage-constrained moving-gap ETD2 / coupled Newton upgrades landed | ✅ moving-gap order verified against a refined reference within its documented DAE/support domain |
| Reproduction regression against self-pinned baselines | Round-8 current-source F23 Figs. 3/5/6/7 plus hardened F24 and transient artifacts are complete. The M25 paper reproduction is retired; Figs. 9--13 remain quarantined. None of these self-pins establishes paper parity. |
| Full audit chain | ✅ Current local tree: default **2509 passed, 1 skipped, 20 deselected, 12 warnings in 836.01 s**; bounded non-manual slow **15 passed, 1 expected xfail, 2514 deselected, 1 warning in 4497.19 s**; opt-in Fig. 5 row reauthentication **1 passed in 96.14 s**. Hosted CI remains separate post-push evidence. |
| Kaplan pair-breaking characterization | ✅ sc-gap + acoustic-escape τ_l (Fig 6) + frequency-resolved τ_PB(Ω) per Kaplan 1976 (`qpsim.physics.kaplan_pair_breaking`, closed-form elliptic-integral evaluator + `tau_0_phonon` Al/Nb material field) |
| Marchegiani junction path (strategy A: rate equation) | ✅ Eq. 8 Lambert-W T̄ + coefficients-in steady-state solver + SI Notes III/IV coefficient integrals + SI Note V photon-assisted spectral density (`M25PhotonDrive` dataclass + back-solver) + population-dependent `g_ph_α_per_state` arrays + **(2026-07-04) the single-quasiparticle normalization fix and the branch-continuation driver**. The normalization fix: M25 Eqs. 4–6 run on Γ̄ = Γ̃/N_CP(R) (M25 text below Eq. 6), not the ensemble Γ̃ rates the qubit equation uses; the residual previously used Γ̃ (~1e10 Hz) in the density equations, which (a) mixed 1e10 Hz tunneling currents with 1e-8 Hz generation in one system — the entire "flat-valley multi-stability / cancellation-floor" pathology was this conditioning artifact — and (b) drained x_R< ~8 orders below the paper's own small-asymmetry approximation. New `M25Coefficients.cooper_pair_number_R` field (default 1.0 = legacy for opaque bundles; the Note-V builder sets 2ν₀Δ_R·V ≈ 1.61e10). With the fix the Fig 3/4 system has a **unique physical root** at every T (verified by 1-D reduced-residual scans: exactly one sign change), and hybr converges from the default seed with residuals 1e-12–1e-24 Hz. The qpsim curves satisfy broad manually read topology/scale anchors, including panel-a's merged μ_α and Fig. 4a's low-T nonmonotonic dip; no digitized pointwise paper comparison exists. Sweep entry point: `solve_rate_equation_branch` (bidirectional natural continuation with warm starts + adaptive step bisection + reduced 1-D bracketing fold handler + documented exchange rule); `solve_rate_equation_steady_state_multi_seed` retained for single-point solves (default picker now `min_residual`; the old `max_x_L` default admitted sub-1-Hz pseudo-roots on the recombination slope). **Corrected manual paper anchors** (transcribed paper-formula μ inversion, SI Eqs. S2–S5): μ_L/Δ_L = 0.938 @ 10 mK, 0.872 @ 20 mK, 0.804 @ 30 mK, 0.669 @ 50 mK, linear to ≈0 at T̄ ≈ 146 mK. The previously quoted anchors (0.9534/0.9076/0.8563/…) were pixel misreads that happened to coincide with a max-x_L pseudo-root (residual 5.7e-4 Hz vs the 5e-8 Hz source scale — not a fixed point) and are inconsistent with the paper's own linearity statement and its x_L ≈ √(g^ph_R/r^L) approximation. |
| Device Architecture — Region/Junction/Device/Qubit composition layer | ✅ Phases 1-6 shipped: design doc, ExternalFlux through the solver stack (Phase 2), Region+Junction+Device+solve_device_steady_state (Phase 3), Qubit+parity+JunctionQubitCoupling (Phase 4), `M25GapAsymmetricJJ` Junction subclass (Phase 5 v1), no-double-counting plumbing (Phase 5b: `owns_region_dissipation` flag, `external_dissipation_only=True` backend path, Device-solver routing), moment-solver wiring (Phase 5c: caches the 4-unknown moment-solver fixed point on first evaluate, sidesteps the cross-tunneling bootstrap problem), historical Fig 3/4 validation (Phase 6, retired 2026-08-28), deterministic `method='hybr'` + `solve_rate_equation_steady_state_multi_seed` (Phase 6 hardening), and **(2026-07-04) the Γ̄ = Γ̃/N_CP(R) normalization fix flowing through the junction's density-equation flux assembly** (`evaluate` now divides the tunneling gammas by `cooper_pair_number_R`, matching the corrected residual; qubit channels keep Γ̃). Junction default `branch_picker_mode` switched to `min_residual` — with the normalization fix the moment system's root is unique and `max_x_L` (which could pick sub-1-Hz pseudo-roots on the recombination slope) is retained for legacy comparison only. **M25 Fig 3a comparison, corrected:** device-path moments at T = 20 mK are x_L = 5.58e-8, x_R> = 2.02e-8, x_R< = 4.70e-8, p_1 = 8.3e-4 (the unique root; consistent with the paper's small-asymmetry x_L ≈ x_R> + x_R< ≈ √(g^ph_R/r^L)), giving μ_L/Δ_L = 0.864 (leading-order inversion) / 0.872 (full SI-Eq.-S3 inversion) vs the published figure's ≈0.87 at 20 mK. The eighth-session claim of 0.9076 at 20 mK rested on a pseudo-root (x_L = 1.5e-5, residual 5.7e-4 Hz ≈ 10⁴× the source scale) matched against a misread of the figure; both sides of that comparison were wrong and canceled. Picker/lm-determinism history retained in git; `TestLmDeterminism` still pins lm reproducibility. Ancillary fixes: `SpectralContext.active_mask` epsilon switched from `mean(dE)` to the bin spacing of the first bin above the gap (uniform-grid behavior unchanged; correctly handles piecewise grids); default solver seed's p₁ now includes the SI-Eq.-S73 photon term (reduces to the old ee-balance seed when Γ̃^ph_01 = 0) |

## Validation figures

The active validation artifacts are self-pinned CSV baselines + PDF plots under
`validation/baselines/{constant, kaplan, transient}/` with regression
tests under `validation/{fischer_2023, fischer_2024, transient}/`. A
passing pin proves stability against prior qpsim output, not independent
agreement with a publication. Current artifact readers fail closed on
versioned schema, exact configuration/axes, dependency hashes, physical
domains, and complete data. Certificate strength is family-specific: F23
Fig. 5 v3 and Fig. 6 v2 persist `f` and `n_ph` for reader-side reassembly;
F23 Fig. 7 summary-v2 omits those states and exposes authenticated producer
assertions only after explicit opt-in. F24 Fig. 5 v5 and Figs. 5--7 v4 are
state/curve-backed, while its two Fig. 8 families are explicitly summary-only
producer assertions. Only genuinely pre-schema artifacts take the narrow
`LegacyArtifactError` quarantine path, and multi-file publishers use matched
records and OS locks. The four current F24 canonicals bind ordered solve
certificates and passed live pin recomputation;
their pre-v2 CSV/PDF pairs are retained under
`validation/baselines/legacy/fischer_2024_pre_strict_v2/` as rejected audit
evidence. The transient photon-kick artifact is now strict v3: it binds the
whole numerical source/configuration and producer runtime, authenticates a
structurally parsed one-page PDF, reconstructs `x_qp` from stored state, checks
exact thermal initial data and monotone time/`x_qp`, compares the late
trajectory with an independent steady solve, and reassembles QP residual,
backward-error, and pair-number certificates. The historical
`dt=0.2/0.1/0.05 ns` study established driver-partition insensitivity, not a
formal-order result; it has not been rerun as part of the v3 artifact.
Historical Fig. 7 evidence: the pin was regenerated under the
tightened solver contract at all 48 targets on Windows and Linux. After a
post-publish NumPy-2.5 typing
repair outside the Fig. 7 call path advanced the conservative whole-tree
digest, an exact uncached Windows recertification completed in `975.48 s`. It
records maximum QP/phonon backward errors `9.819e-9`/`8.271e-9` against a `2e-8`
gate. The later direct-gap quadrature and structured-exception changes advanced
the digest again; a `1082.915 s` exact Windows rerun reproduced all 48 axes,
observables, and certificate arrays bitwise. The later structured gap-collapse
exception advanced it once more; a `1123.7 s` exact rerun was again bitwise
identical. The solve-contract SHA-256 for that historical pin was
`ebe1382d509f6c52f11bca95b8d0161a211c4002a59f38de942cb2aefd193165`.
That historical hardened-driver campaign completed all 48 fresh targets under
identity `82ef6da8…f320`. It has since been superseded by the current promoted
campaign under identity `ea166442…`; current evidence is summarized in the
figure table below.
In the Fischer rows below, “Windows/Linux envelope” means the OS-family case,
not an exact hardware/runtime identity; its bounds are calibrated from the
hosted runs recorded in the audit.

The amplitude-sensitive pair-number certificate is now persisted with
different evidence scopes. Fig. 3 persists `f` and reconstructs the affine
phonon state qualified below. Fig. 5 v3 and Fig. 6 v2 persist complete `f/n_ph`
state and readers independently reassemble the certificate. Fig. 7
summary-v2 intentionally omits state; its persisted certificate scalars are
authenticated producer assertions, not reader-reassembled evidence, and
require explicit opt-in.

Fig. 3 also separates two source identities.
`producer_solve_contract_digest` is immutable provenance for the numerical
contract that actually produced the stored `f(E)` curves.
`validated_solve_contract_digest` names the contract under which those curves
were most recently re-certified. For finite-escape points the CSV does not
store the producer's `n_ph`; re-certification reconstructs the affine phonon
fixed point implied by the stored `f` and the validation-time equations.
Passing that check proves current-equation root membership for the
reconstructed `(f, n_ph)`.
It does not recover the producer's original `n_ph`, and it does not prove that
the current solver algorithm executed or converged to the stored state.

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 paper legend ratios {0, 0.1, 1, 10} | `validation/fischer_2023/fig3_paper.py` | `constant/` | ✅ **Corrected strict-v3 regression promoted and final-equation recertified.** The source-frozen `NE=1620` producer completed all 14 continuation steps in `12022.121 s`, versus `177440.15 s` (49.29 h) on the superseded end-only/legacy-routing path. A final publication-layer repair rejects real occupations outside inclusive `[0,1]` and rejects complex values before coercion; the canonical bundle was republished from unchanged authenticated raw payload SHA-256 `78c2e181fab5d3a25d5936e2bb5b76cbfb84fc3fcee7ba1066af65a3a2aa7a45`. Current CSV/PDF/validation-record SHA-256 are `2264f6c09f2917d5863d274a5edbaf0e8484e9ec86e51018720cf868a4378616`, `1f3c762a461a83d999dc9013ecd1f167c2c2f6963b2208ef518e211833e20e43`, and `6776776f643e73667fe4f836c8352ded670f4e0c239addbf5f85516e53b3f89b`; 58 focused checks passed with 3 slow/manual deselections and visual inspection passed. The raw payload omitted finite-ratio `n_ph`; validation reconstructs the unique affine phonon fixed point implied by stored `f`, so it proves current-equation membership for that reconstructed pair, not the producer's original omitted phonon state or current-solver execution. Historical refinement/platform qualifications remain error-characterization evidence, not pointwise/total-variation or paper-parity claims. |
| Fischer 2023 Fig 5 paper-topology x_qp two-panel | `fig5_paper.py` | `constant/` | ✅ **Current state-bound v3 canonical complete and promoted; refinement remains a separate qualification.** Six independent single-thread continuation rows completed all 81 points under identity `01e22c38…cccb` in `9380.161 s` wall (`35471.716 s` aggregate worker time). CSV/two-PDF/promotion/campaign SHA-256 are `c114a291…98bf`, `d12bb185…895c`/`e6307690…6bd5`, `b1f86b0d…7f5f`, and `f43d2938…b3fa`. Maximum QP residual/backward/number-backward errors were `1.388e-17`/`1.207e-16`/`9.770e-11`; phonon residual/raw-backward/certified-backward maxima were `1.201e-11`/`4.391e-7`/`9.855e-10`. All six raw rows passed exact pre-coercion dtype/hash/currentness checks; 56 focused checks, one 81-state slow recertification, and both visual inspections passed. This closes fixed-`NE=1620` production, not commensurate-grid refinement or digitized-paper parity. |
| Fischer 2023 Fig 6 paper-topology gap suppression | `fig6_paper.py`; paper oracle `paper_data/fischer_2023/fig6/` | `kaplan/` | ⚠️ **Current state-bound v2 canonical complete and promoted; independent paper-data diagnostic mismatches.** Three parallel single-thread temperature rows completed all 66 points in `15223.850 s` wall (`43470.213 s` aggregate worker time). CSV/PDF/promotion-record SHA-256 are `5a5e3c6eec534cec121eaff93b5efdfac8fc4225f72f3e006c0936aab683bc9c`, `52f7372c53ac87b3039f6461aa94188d714f86524b65579a2ca68dc7694a0837`, and `a9a8f09dd9408c2f3ed2b6c0bde5c22d76da5a99e889e864550cd05905cea823`. A provenance-only migration retained all numerical rows, payload/PDF bytes, worker hashes, and historical `cbfb0006…f9a43` run identity. Maximum QP residual/backward/number-backward errors were `9.272e-15`/`6.066e-8`/`9.877e-7`; phonon residual/raw-backward/certified-backward maxima were `3.691e-12`/`1.099e-5`/`9.757e-6`; maximum gap-map absolute error was `9.911e-11 µeV`. The original-raster oracle's dashed Eq. 53 controls pass, but its three solid numerical traces differ by roughly 33–39% maximum relative error and record `diagnostic_mismatch`. The comparison is not gate-eligible until parameter/discretization uncertainty is bounded. The signed diagnostic still exposes all 66 finite stored samples and paper-window clipping; `_direct` remains separate. |
| Fischer 2023 Fig 7 paper-facing Q_i,tot(T_B) | `fig7_paper.py` | `constant/` | ✅ **Exact current-source regeneration completed and promoted.** All 48 targets completed under content/runtime identity `ea166442…` and solve digest `d674ca…`; the six-worker campaign took `4387.907 s` wall (`15458.707 s` aggregate worker time). CSV/PDF/promotion-record SHA-256 are `2bb97283…5634`, `d0c3029f…7586`, and `32fc656b…f37d`; 67 focused checks passed with 2 slow deselections, and strict readback plus visual inspection passed. Historical hosted portability evidence remains a qualification, not bitwise identity. This is a fixed-grid qpsim regression, not paper parity. |
| Fischer 2023 Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | `constant/` | ⚠️ all legacy `Q_i` values moved by up to `14.5144%`. The commensurate ladder remains nonconverged: at aligned `NE=6480`, `-100 dBm`, `Q_i=3.2994464e10`, the `3240 -> 6480` change is `4.44368%` with certificate `9.83e-12`. Exact-cell/FV variants are negligible; thermal response is monotone; cancellation condition is `16.6`; an overlap-aware photon prototype moves standard `Q_i` only `0.027%/0.010%` and leaves the rung at `4.567%`, so no photon-operator rewrite is justified. A proposed conservative policy—not a derived error budget—requires `<=1%` maximum `Q_i` change on two consecutive commensurate rungs and `<=0.25%` observable discrepancy. The pre-schema pin remains quarantined. |
| Fischer 2024 Fig. 5 paper-topology distributions | `fischer_2024/fig5_paper.py` | `constant/` | ✅ freshly regenerated after the final source digest as a strict-v3 qpsim-native `NE=810` regression. The hosted-portability follow-up tightened the number-mode solve gate from `1e-6` to `1e-7`; only the `1e-2 Hz` curve changed, while `E`, the thermal seed, and both lower-drive curves remain byte-identical. Certificate maxima are QP backward error `2.7247e-13`, QP-number backward error `1.3999e-10`, and residual `3.4664e-23`. Payload/CSV/PDF SHA-256 are `33f1d29e1511711659bed1780dd0a615ce24de158a1104cfc95f84755760ee4e` / `83d4c9373f0164c0ff1f042544acfa2aa96d093ea3766fe66de618d17c568745` / `81e1c08490321dc5db848078debe7a7bc36c8b5ee17c466b44d6772ceb39503d`. The commensurate `405/810/1620` characterization and its paper-parity qualifications remain as previously documented. ⚠️ Three analytic overlays remain `TODO(paper-parity)`; this is not paper parity. |
| Fischer 2024 Figs 5-7-topology f(E) characterization | `fischer_2024/figs_5_7_fe_pb.py` | `constant/` | ✅ freshly regenerated after the final source digest as a strict-v3 qpsim-native `NE=810` regression. Certificate maxima: QP backward error `2.2260e-14`, residual `2.0817e-17`. CSV/PDF SHA-256 `4e633cf49e3ae7c6d9b54ab4ffa0378f072c6d7a8a700f4301ffeb82c5dd872f` / `0616ec72d83670c7a4aed480315193a91daf1f597e5b7c175d3b4d67259a96bd`. ⚠️ Fixed-grid regression, not a pointwise continuum-shape or paper-parity claim. |
| Fischer 2024 Fig-8-topology x_qp(T_B) characterization | `fischer_2024/fig8_xqp_pb.py` | `constant/` | ✅ freshly regenerated after the final source digest as a strict-v3 qpsim-native `NE=810` regression. Certificate maxima: QP backward error `2.7912e-11`, residual `4.1484e-15`. CSV/PDF SHA-256 `a5f89c5e9300eace746e3530e6a5323efa4b6a2f57f426009f8af3c6e10c9348` / `fbce04e70ac3818e5e101e7b55b992b429abbe806eb06dcab36b0610fcfd5825`. ⚠️ Qpsim-native characterization, not paper parity. |
| Fischer 2024 Fig. 8 paper-topology density | `fischer_2024/fig8_paper.py` | `constant/` | ✅ freshly regenerated after the final source digest as a strict-v3 qpsim-native `NE=810` regression. Certificate maxima: QP backward error `3.3034e-7`, residual `5.9210e-13`. CSV/PDF SHA-256 `dc157773b84c6fcbd75c4add3dae9f215870fe746950c0b5533f936bce81b552` / `0b0430d3d12bad02c3338944f59d585f260aa4e8326eaf38443872589f07616a`. ⚠️ Analytic density overlay remains a placeholder; not paper parity. |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ matched-measure pin regenerated with provenance; 4 slow regressions passed in 752.05 s and the exact module passed again in 726.54 s after the encoding repair; `dt=0.2/0.1/0.05 ns` study establishes driver-partition insensitivity, not formal order. The CSV is now explicitly BOM-free UTF-8/LF after a Windows CP1252 title byte broke Ubuntu readback; the numerical payload is byte-identical and the canonical SHA-256 is `18e2a2424c037e2b6dd64189848765d0a0c75a6b6cc4bed63364c3f2d05c51d1`. |

## Analytic tests

`validation/analytic/` (runs in the default suite — fast, no markers):
- Detailed balance: e-ph, sub-gap photon, pair-breaking photon channels vanish at `(f_FD(T), n_BE(ω, T))`.
- Mattis-Bardeen thermal limits: σ_1 → 0 at T → 0, σ_2 → π Δ / ω kinetic-inductance limit.
- Gap-equation round-trip: `solve_gap(f_FD(T_B))` recovers `Δ_eq`.

## Spatial diffusion operators (A1 / §7.5)

`qpsim.transport.diffusion.base` — `DiffusionModel` operator family
parametrized by `(p, q)`, `L_{p,q}[f] = N_1^{-p} d_x(D_N N_1^q d_x f)`:
**A1 = (1, 0)** dirty-limit Keldysh–Usadel (default; conserves `N_1 f`,
undressed flux `D_L` — 1 above the local gap edge, 0 below — rate
`D_N / N_1`, same uniform-gap rate as C), A1P = (1, 2) transverse-dressing
diagnostic (the pre-errata energy-channel assignment), A2 = (2, 2)
diagnostic, C = (0, -1) legacy `D_E` closure, B = (0, -2) constant-τ.
June 2026 advanced-propagator errata (g^A = -τ³(g^R)†τ³): the `N_1²` flux
dressing belongs to the transverse (charge) channel, so the energy-mode
operator is (1, 0), not (1, 2); see `docs/Diffusion_Operators.md` and the
qp-diffusion paper's `verify_gA_convention.py`.

`SpatialBackend.apply_transport` is an exactly-conservative
finite-volume Crank–Nicolson step on the conserved density `u = N_1^p f`
(harmonic-mean faces, reflective ends; `Σ_x N_1^p f` conserved to ~1e-15);
reproduces the legacy modal C-step to round-off (A1 coincides with it at a
uniform gap). Optional `gap_profile`
(per-cell DOS) + `interface_conductance` add a Kupriyanov–Lukichev two-gap
interface `F = G_N (N_1^L N_1^R - N_2^L N_2^R)(f_L - f_R)` — the
coherence-factor energy-channel weight, regular at matched gaps
(current-continuous, f-discontinuous). Prelim scripts unchanged
(their committed outputs are historical C runs).

§7.5 benchmarks in `validation/diffusion_operators/` (fast co-located tests +
`python -m …` scripts → CSV/figure under `outputs/`): `uniform_gap_packet`
(measured `D_eff(E)/D_N` traces `N_1^{q-p}`; A1 ≡ C at uniform gap),
`gap_gradient_drift` (COM drift
`D_N q N_1^{q-p-1} d_x N_1`; A1 no drift, A1P/A2 drift up the gap, C/B down),
`interface_trap`
(current-continuity + f-discontinuity with the coherence weight; A1 vs A2
distinct closed equilibria), and `self_consistent_feedback` (a represented
`0.5*Delta_0` closure floor, target-fixed-point seed calibration, and a
fail-loud `1e-12` raw-map convergence gate).

## Services (`qpsim.services.*`)

- ✅ `steady_state` — Newton + Picard + Anderson + coupled-Newton routing; homogeneous Newton requires dimensional residual plus normwise L1 gain/loss backward error, and Picard requires local/normwise mapped-fixed-point tests plus a physical nearest-binary64 phonon certificate. The raw affine residual is retained separately, and projection to a certified phonon map is followed by one final Newton solve so the returned pair is matched. Finite-escape Picard accepts a validated same-grid `initial_phonon_guess`; the diffusion backend forwards both `f` and `n_ph` during continuation and rejects a state whose public gap/model disagrees with the spectral/phonon operators.
- ✅ `nbar_loop` — F23 Eqs. 59-60 fixed-point on (n̄, Q_i).
- ✅ `transient` — `run_time_dependent` with fixed-gap ETD2 collision substeps, stage-constrained second-order moving-gap backend steps, snapshot cadence, autonomous stop-tol, and observables dict.
- ✅ `rate_equation` — Eq. 8 Lambert-W T̄ closed-form observable + `solve_rate_equation_steady_state` Newton solver on the 4-unknown M25 boxed system (density equations on Γ̄ = Γ̃/`cooper_pair_number_R`; auto-scaled residual tolerance) + `solve_rate_equation_branch` bidirectional continuation driver for temperature sweeps (warm-started natural continuation, adaptive step bisection, reduced 1-D bracketing fold handler, documented photon/thermal exchange rule) + `thermal_equilibrium_seed` / `analytic_low_T_seed` seeds.
- partial `rate_equation_coefficients` — `M25PhysicalParameters` + `coefficients_from_physical_parameters` build the `M25Coefficients` bundle from primitive physical inputs (gaps, ω_10, transmon E_J/E_C, T, junction tunneling scale `R_T = g_T Δ̄/e²`, caption-level `r^L`/`r^{R<}`). Implements SI Notes III (12 tunneling rates) and IV (recombination, generation-by-thermal-phonons, intraband relaxation, branching fraction). Note V photon-assisted tunneling shipped via `M25PhotonDrive` + `coefficients_from_physical_parameters_with_photon_drive`: elliptic-integral spectral densities `_S_ph_total` (S57), `_S_ph_Rgt` (S59), `_S_ph_Rlt = total − Rgt`; all four `Γ^{ph}_{ij}` derived from single back-solvable `Gamma_nu_scale_Hz`; per-state `g^{ph}_α` arrays feed the population-dependent residual.

## Observables (`qpsim.observables.*`)

- ✅ `ac_conductivity` — Mattis-Bardeen σ_1, σ_2.
- ✅ `quality_factor` — Q_i = σ_2 / (α σ_1).
- ✅ `frequency_shift` — δω/ω fractional.
- ✅ `density` — `qp_number_density`; historical qpsim `qp_fraction = n_qp/(4ρ_FΔ_0)`; explicit Fischer/Catelani `qp_fraction_paper = n_qp/(2ρ_FΔ_0)`.
- ✅ `gap_suppression` — δΔ, δΔ/Δ_eq; `compute_gap_suppression(f, E, T_c, T_bath)` + `gap_suppression_from_deltas(Δ_eq, Δ_final)`. Occupation-backed solves fail closed unless the first cell face covers the selected gap; web summaries retain independently calibrated `delta_eq_ueV` when derived suppression is unsupported.
- ✅ `effective_temperature` — `effective_phonon_temperature(n_ph, ω_bins, gap, T_bath)` via weighted BE fit per F23 Eq. 36.

## Test suite

The current Round-8 terminal default aggregate is **2509 passed, 1 skipped,
20 deselected, and 12 warnings in 836.01 s**. The skip is the deliberately
opt-in external Fig. 5 archive reauthentication; with
`QPSIM_FIG5_RUN_ROOT=C:\tmp\qpsim-fig5-runs-final-v1`, that exact node passed
in **96.14 s** after rechecking all six durable row archives. The bounded
non-manual slow selection is **15 passed, 1 expected xfail, 2514 deselected,
and 1 warning in 4497.19 s**. The xfail is the explicitly quarantined
Figs. 9–13 legacy baseline. Full paper producers marked `manual_slow` are
release/regeneration gates and are not implied by either aggregate.

The historical `71c5f02` default aggregate is **1549 passed, 17 deselected, 4
warnings in 525.03 s** on Windows (`pytest -q`). The deselections are the opt-in
slow/manual validation selections. That exact tree also passes `ruff check
.`, `mypy qpsim` across 75 source files, `python -m compileall -q qpsim tests
validation`, and `git diff --check` (apart from Git's Windows line-ending
notices). The earlier unit/API-only checkpoint was **1272/1272 passed, 13
warnings in 258.88 s**. These aggregates do not by themselves establish the
separately recorded Fig. 3/Fig. 6/Fig. 7 production outcomes, refinement, or
paper parity.

Hosted CI run `29629653466` passed on both supported interpreters. Python 3.13
ran the default suite in 289.82 s and the slow suite in 2224.94 s; Python 3.14
ran them in 272.11 s and 2035.36 s. Each default matrix reported 1513 passed,
17 deselected, and 4 warnings; each slow matrix reported 14 passed, 2 expected
xfails, 1514 deselected, and 1 warning. Run wall time was 46m59s.

The 17 default deselections were exhaustively re-inventoried on 2026-07-18
against 1566 collected nodes: 16 non-manual `slow` tests and one `manual_slow`.
All 16 non-manual nodeids have direct CI evidence on the recorded hosted tree:
14 numerical passes and 2 narrow expected xfails for F23 pre-schema canonicals.
Later changed solve contracts have the targeted exact recertification evidence
recorded in the audit.
The Fig. 7 full-pin wrapper passed under hosted Linux on Python 3.13 and 3.14;
the four F24 full live pytest pins passed together in 76.86 s after refinement
and strict-v2 promotion. A later single-threaded current-solver recertification
covered all 84 F24 states in 95.07 s; every certificate passed and maximum
pinned/live row drift was `2.58e-14`. A subsequent ASCII-only source change
was proven string-only by normalized-AST identity. That is compatibility
evidence, not permission to replace producer provenance or declare a stale
summary artifact current; summary-only F24 artifacts require exact
regeneration after a contract change. That requirement has since been
satisfied by fresh strict-v3 regeneration of all four families; the
`76.86 s`/`95.07 s` numbers above remain historical strict-v2 evidence. Other
focused reruns
include the tight-contract Fig. 7 plateau (4.85 s), reduced Fig. 6 continuation
(3.22 s), and post-schema F23 Fig. 5 high-drive branch (299.94 s). The serial
Fig. 6 `manual_slow` wrapper was not separately invoked;
the measured row times sum to `21726.6965 s` (`6.04 h`), superseding the stale
14-hour estimate. Its 66/66 production sweep certifies the historical loose
contract, but does not close the later observable mismatch.

The initial default non-slow validation run reported **5 failed, 122 passed,
17 deselected, 3 expected xfailed, 1 warning in 556.56 s**. All five failures
were stale finite-volume validation oracles: two interface tests multiplied
separately averaged KL factors instead of averaging their complete product; two
self-consistent-feedback tests used point DOS instead of represented cell DOS;
and the uniform-packet test combined point DOS with a one-step CN inversion even
when production subcycled. The corrected references use `cell_weights / dE`,
the exact `q = 0` support fraction, an `m`-substep CN inversion, and independent
direct quadrature of the full KL product. The affected folder then passed
**14/14 in 209.10 s**.

The complete post-repair non-slow validation rerun then passed **127 tests,
with 17 deselected, 3 expected xfails, and 1 warning in 217.75 s**. The expected
xfails are the two quarantined Fig. 3 baseline schema/nonvacuity checks and the
stale Fig. 6 configuration-metadata check. The sole warning is the explicit
`8.28066788472e-08 µeV` support-edge diagnostic in the analytic gap-equilibrium
check. This is a historical complete selection from before the N29 artifact
regressions landed. It closes that recorded non-slow gate, not the 17
slow/manual selections or any paper-parity claim.

Additional settled focused evidence on historical audited snapshots includes the refined-reference
second-order moving-gap regression and its invariant tests; 8 focused
self-consistent-gap tests without the former extrapolation warnings; 37 passing
tests in the broader affected diffusion backend group; 132 passing homogeneous
Newton/Picard/phonon certificate tests; 4 passing slow transient tests in 752.05 s
and an exact post-encoding-repair module rerun in 726.54 s;
exact tight-contract 48-point Fig. 7 production runs on Windows in 982.54 s and
Linux in 946.16 s, plus exact-source Windows recertifications in 901.13 s,
975.48 s, 1082.915 s, and 1123.7 s; the latest two were bitwise identical. A
now-superseded NE=1620 Fig. 3 production run took 5768.40 s, followed by an
uncached full-pin pass in 5224.19 s; those runs predated the
amplitude-sensitive pair-number certificate and are not current baseline
evidence. All 66 historical Fig. 6 targets completed in 7599.29 s concurrent
wall time. The cache-off Fig. 5 audit took 8088.34 s:
its physical branch guard passed, while its stale pin failed 20/42 upper-panel
values. Figs. 9--13 artifact hardening passed 12 fast tests. Fischer 2024's
current focused suite passed **59 tests with 4 slow deselected in 17.70 s**,
including four reduced live solve/write/read paths; its initial four full pytest
pins passed in **76.86 s**, followed by the all-84-state **95.07 s**
current-solver recertification. F23 Fig. 5 artifact hardening passed 10 fast
tests; the current combined F23/F24 selection passed 81 tests with 7 slow tests
deselected in 94.79 s. These focused results are scoped, not a repository-wide aggregate,
and do not establish paper parity or close the stated refinement
qualifications.

The historical Fig. 3/Fig. 7 non-slow policy slice passed **44 tests with 4 slow
deselected in 4.70 s**. It includes the exact hosted Fig. 7 drift values and a
rounded Fig. 3 policy fixture; both fail strict gates and pass the calibrated
Windows/Linux OS-family envelopes, while macOS or unknown pairs remain strict.

Recent hardening pass (Claude+GPT cross-review, seventh session):
- Photon collision kernels (`sub_gap_photon`, `pair_breaking_photon`) and the analytic Newton Jacobian now hard-reject nonuniform energy grids via `qpsim.collisions._uniform_grid.uniform_grid_spacing`; previously they silently used `dE[0]` as a uniform stride.
- `PhononState.__post_init__` now validates finite/nonneg `n_ph`, finite/nonneg/strict-monotone `omega_bins`, and finite/nonneg `tau_l` (was shape-only).
- `phonon_steady_state` raises on singular/runaway phonon balances instead of clipping negative occupations to zero — turns "no phonon fixed point" from a fake solution into a loud failure.
- Homogeneous Newton/Picard convergence now has scale-independent physical balance gates. Affine phonon certification distinguishes the raw direct-form residual from whether the stored value is the nearest nonnegative binary64 root; negative, signed-underflow, singular, overflowed, and nonfinite roots fail closed.
- `accept_lm_convergence` bypass narrowed to its documented scope: only the `is_no_progress_stall` case is exempted from the residual check; success-with-high-residual now raises.
- `transient.run_time_dependent` truncates the final substep so `total_time` is honored exactly.
- Four M25 Fig 3/4 multi-stable branch points shifted under the stricter residual acceptance; CSV+PDF baselines regenerated; visual paper agreement preserved.

Slow tests (opt in with `-m slow`):
- Fischer validation reproductions at Fischer-scale grids.
- The transient photon-kick regression is slow-marked and its four-test subset
  passed on 2026-07-16. Current Fig. 7 evidence is the completed exact
  48-target hardened-driver campaign under identity `ea166442…`; its
  promoted CSV/PDF/attestation passed strict and independent readback.
  Historical Windows/Linux production remains portability evidence only.
  The four historical strict-v2 F24 pytest pins passed in 76.86 s and their
  later current-solver 84-state recertification passed in 95.07 s. Current F24
  strict-v3 regeneration is recorded in the figure-status rows above.
  All 16 historical non-manual slow nodeids have direct CI evidence on the
  recorded hosted tree (14 passes and 2 F23 legacy xfails), including the
  Fig. 7 full-pin wrapper under Python 3.13 and 3.14. Current Fig. 6 adds a
  promotion-locked fast scalar preflight and a separate non-manual `slow`
  full-state recertification gate; the complete live comparison remains
  `manual_slow`. The first closeout run exposed four redundant full replays
  (`336.06 s`); the two corrected fast checks now take `5.2 s`. M25 Eq. 8
  sweeps remain fast.
- Current Fig. 5 likewise uses a promotion-locked scalar fast preflight plus a
  separate non-manual slow full-state recertification. The old composed
  preflight replayed 81 states twice (`160.88 s` body); the replacement takes
  `1.71 s`, and one full replay takes `82.58 s`. Its live 81-point producer
  remains `manual_slow`. The frozen publisher still repeats five validation
  passes, accounting for `504.201 s` of post-solve closeout overhead.

## Build/dev notes

- **Local venv:** `.venv/` at repo root, Python ≥ 3.13.
  - macOS/Linux: `.venv/bin/pip install -e ".[dev]"`, `.venv/bin/pytest -q`.
  - Windows: `.venv/Scripts/python.exe -m pip install -e ".[dev]"`,
    `.venv/Scripts/python.exe -m pytest -q`. (`PYTHONUTF8=1` is no longer
    required since the material-YAML loader reads UTF-8 explicitly, but
    remains harmless belt-and-braces.)
  - `-m slow` opts into the Fischer/M25 full reproductions.
- **Mac-only paths** (do not exist on the Windows box):
  - Legacy repo: `~/Documents/Quasiparticle Simulation/Active Code/qpsim/` (read-only port source).
  - Prelim deck: `~/Documents/Graduate/Preliminary Exam/build_presentation.py` (uses `/opt/homebrew/bin/python3` for python-pptx).
