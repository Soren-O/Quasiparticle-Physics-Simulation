# qpsim status (gate tracker)

Last updated: 2026-07-18 (code-only numerical audit follow-up; see
`docs/AUDIT-2026-07-15-numerical-software.md`). Paper-reproduction status is
qualified: self-pinned CSV parity is regression evidence, not independent
paper truth. Fischer 2023 Fig. 3 now has a promoted nonzero, certified NE=1620
artifact. Its prior 30--42% strong shape warning is localized to ideal-BCS
photon-threshold layers: scalar mass and weak Wasserstein shape converge much
faster, while pointwise/total-variation convergence remains unclaimed. An
uncached full-pin repeat passed. Fig. 6 completed and certified all 66
historical loose-contract production targets, but its default suppression
observable is now known to be mis-scaled relative to its accepted gap-map
tolerance; the repaired fixed-gap/direct path has a strictly certified negative
low-drive point but no full canonical. Fig. 7 now has exact tight-contract 48-point Windows and
Linux production evidence. Its historical loose-contract stopping defect is
fixed; hosted tight-contract runs subsequently established a smaller,
balance-certified Windows/Linux envelope handled by platform-scoped regression
bounds. The old F23 Fig. 5 and Figs. 9--13 pins remain quarantined. Fig. 5's
tested `T*/Delta=0.60` continuation split is resolved as tolerance-induced
pseudo-roots, but the full sweep still needs tight-contract regeneration and
refinement; Figs. 9--13 has a
nonconverged `Q_i` observable through an aligned `NE=6480` point. All four Fischer 2024 families
now have strict-v2, independently certified qpsim-native pins after
commensurate-grid refinement; the pre-v2 files remain archived and rejected,
and the Fischer 2024 paper-target analytic overlays remain incomplete. Prior:
2026-07-04 M25 closeout:
single-quasiparticle Γ̄ =
Γ̃/N_CP(R) normalization fix in the density equations
(`M25Coefficients.cooper_pair_number_R`), branch-continuation driver
`solve_rate_equation_branch`, paper-exact μ inversions, real Fig-4
comparison models (global quasiequilibrium + renormalized), all M25
baselines regenerated and pinned tight — see Gate 8 / Device
Architecture rows and the Marchegiani validation-figure rows below.
Prior: 2026-07-03 web frontend `qpsim.webui` shipped — see
`docs/Frontend.md`; optional extra `qpsim[ui]`, CI installs it; the
only engine changes are the physics-neutral `progress_hook` on
`services.transient.run_time_dependent` and
`T3Spatial1DBackend.run_until_steady_state`, plus — from the same-day
frontend code review — float coercion in the materials loader
(YAML 1.1 loads unsigned-exponent notation like `1.74e28` as a
*string*; Al/Nb/TiN `v_F`/`rho_F` were affected), a public
`COMMENSURATE_TOL` alias in `collisions.sub_gap_photon`, and a derived
`H_OVER_KB_K_PER_HZ` in `qpsim.constants`. Prior: 2026-07-02
a1-diffusion-operators merged to main; CI green; code-health review
fixes).

Central snapshot of what's done, what's in progress, and what's deferred. The New Framework Plan (`~/Documents/Quasiparticle Simulation/Documentation/Current/New Framework Plan.md`) is the authoritative spec; this is the running status against it.

## Gates

| Gate | Scope | Status |
|---|---|---|
| 0 | Phonon-model decisions (D1–D5) | ✅ resolved in `docs/Phonon_Model_Decisions.md` |
| 1 | Repo skeleton | ✅ (tag `gate-1-skeleton` at `77fd516`) |
| 2 | Physics + grid + collisions + solvers ported from legacy qpsim; fixed-gap ETD2 / persistent-coordinate stage-constrained moving-gap ETD2 / coupled Newton upgrades landed | ✅ moving-gap order verified against a refined reference within its documented DAE/support domain |
| 3 | T3 reproduction regression against self-pinned baselines | ⚠️ qualified, not complete paper parity: Fischer 2023 Fig. 3 is promoted as a nonzero NE=1620 certified local regression and its uncached repeat passed. Its strong curve error is an ideal-BCS threshold-layer/norm limitation; mass and weak shape converge, but pointwise/total-variation continuum convergence is not claimed. Fig. 6's 66-point production run certifies its historical loose contract, not the observable to its pinned tolerance; the repaired fixed-gap/direct path still needs a full production/refinement campaign. Fig. 7 is certified at all 48 targets under exact tight-contract Windows and Linux production runs. The legacy F23 Fig. 5 and Figs. 9--13 pins remain quarantined; Fig. 5's tested `T*/Delta=0.60` pseudo-root diagnosis is resolved without settling the full sweep, while Figs. 9--13 remains grid-nonconverged through `NE=6480`. Four F24 families are promoted as certified qpsim-native regressions at paper topology after nested-grid refinement, while their analytic paper-target overlays remain incomplete. |
| 4 | Full Layer-4 audit chain | ⚠️ the exact final repository aggregate is green at **1549 passed, 17 slow/manual deselected, 4 warnings**; five stale finite-volume validation oracles were repaired and the expanded diffusion folder passes 19/19. All 16 non-manual slow nodeids had direct CI evidence on the recorded hosted tree (**14 passed, 2 intended F23 quarantine xfails**) under Python 3.13 and 3.14, with later changed solve contracts covered by targeted exact recertifications. The Fig. 6 `manual_slow` wrapper remains uninvoked; its measured serial row cost is `6.04 h`, but running the old loose-contract wrapper would not close the newly diagnosed observable mismatch. Figure-specific qualifications remain; see `AUDIT-2026-07-15-numerical-software.md`. |
| 4.5 | Characterization tier (Ph0-Kaplan) | ✅ sc-gap + acoustic-escape τ_l (Fig 6) + frequency-resolved τ_PB(Ω) per Kaplan 1976 (`qpsim.physics.kaplan_pair_breaking`, closed-form elliptic-integral evaluator + `tau_0_phonon` Al/Nb material field) |
| 5 | Ph1 phonon spatial transport | ❌ not started |
| 6 | T2 kinetic scalar backend | ❌ not started |
| 7 | T1 two-component backend | ❌ not started (requires new derivation) |
| 8 | Marchegiani junction path (strategy A: rate equation) | ✅ Eq. 8 Lambert-W T̄ + coefficients-in steady-state solver + SI Notes III/IV coefficient integrals + SI Note V photon-assisted spectral density (`M25PhotonDrive` dataclass + back-solver) + population-dependent `g_ph_α_per_state` arrays + **(2026-07-04) the single-quasiparticle normalization fix and the branch-continuation driver**. The normalization fix: M25 Eqs. 4–6 run on Γ̄ = Γ̃/N_CP(R) (M25 text below Eq. 6), not the ensemble Γ̃ rates the qubit equation uses; the residual previously used Γ̃ (~1e10 Hz) in the density equations, which (a) mixed 1e10 Hz tunneling currents with 1e-8 Hz generation in one system — the entire "flat-valley multi-stability / cancellation-floor" pathology was this conditioning artifact — and (b) drained x_R< ~8 orders below the paper's own small-asymmetry approximation. New `M25Coefficients.cooper_pair_number_R` field (default 1.0 = legacy for opaque bundles; the Note-V builder sets 2ν₀Δ_R·V ≈ 1.61e10). With the fix the Fig 3/4 system has a **unique physical root** at every T (verified by 1-D reduced-residual scans: exactly one sign change), hybr converges from the default seed with residuals 1e-12–1e-24 Hz, and figures match the published curves including panel-a's merged μ_α and Fig 4a's low-T nonmonotonic dip. Sweep entry point: `solve_rate_equation_branch` (bidirectional natural continuation with warm starts + adaptive step bisection + reduced 1-D bracketing fold handler + documented exchange rule); `solve_rate_equation_steady_state_multi_seed` retained for single-point solves (default picker now `min_residual`; the old `max_x_L` default admitted sub-1-Hz pseudo-roots on the recombination slope). **Corrected paper anchors** (paper-exact μ inversion, SI Eqs. S2–S5): μ_L/Δ_L = 0.938 @ 10 mK, 0.872 @ 20 mK, 0.804 @ 30 mK, 0.669 @ 50 mK, linear to ≈0 at T̄ ≈ 146 mK — matching the published Fig 3a curve (linear from ~0.94 to zero at ~0.146 K). The previously quoted anchors (0.9534/0.9076/0.8563/…) were pixel misreads that happened to coincide with a max-x_L pseudo-root (residual 5.7e-4 Hz vs the 5e-8 Hz source scale — not a fixed point) and are inconsistent with the paper's own linearity statement and its x_L ≈ √(g^ph_R/r^L) approximation |
| Device Architecture | Region/Junction/Device/Qubit composition layer | ✅ Phases 1-6 shipped: design doc, ExternalFlux through T3 stack (Phase 2), Region+Junction+Device+solve_device_steady_state (Phase 3), Qubit+parity+JunctionQubitCoupling (Phase 4), `M25GapAsymmetricJJ` Junction subclass (Phase 5 v1), no-double-counting plumbing (Phase 5b: `owns_region_dissipation` flag, `external_dissipation_only=True` backend path, Device-solver routing), moment-solver wiring (Phase 5c: caches the 4-unknown moment-solver fixed point on first evaluate, sidesteps the cross-tunneling bootstrap problem), Fig 3 + Fig 4 paper-figure baselines under `validation/marchegiani_2025/` (Phase 6), deterministic `method='hybr'` + `solve_rate_equation_steady_state_multi_seed` (Phase 6 hardening), and **(2026-07-04) the Γ̄ = Γ̃/N_CP(R) normalization fix flowing through the junction's density-equation flux assembly** (`evaluate` now divides the tunneling gammas by `cooper_pair_number_R`, matching the corrected residual; qubit channels keep Γ̃). Junction default `branch_picker_mode` switched to `min_residual` — with the normalization fix the moment system's root is unique and `max_x_L` (which could pick sub-1-Hz pseudo-roots on the recombination slope) is retained for legacy comparison only. **M25 Fig 3a comparison, corrected:** device-path moments at T = 20 mK are x_L = 5.58e-8, x_R> = 2.02e-8, x_R< = 4.70e-8, p_1 = 8.3e-4 (the unique root; consistent with the paper's small-asymmetry x_L ≈ x_R> + x_R< ≈ √(g^ph_R/r^L)), giving μ_L/Δ_L = 0.864 (leading-order inversion) / 0.872 (full SI-Eq.-S3 inversion) vs the published figure's ≈0.87 at 20 mK. The eighth-session claim of 0.9076 at 20 mK rested on a pseudo-root (x_L = 1.5e-5, residual 5.7e-4 Hz ≈ 10⁴× the source scale) matched against a misread of the figure; both sides of that comparison were wrong and canceled. Picker/lm-determinism history retained in git; `TestLmDeterminism` still pins lm reproducibility. Ancillary fixes: `SpectralContext.active_mask` epsilon switched from `mean(dE)` to the bin spacing of the first bin above the gap (uniform-grid behavior unchanged; correctly handles piecewise grids); default solver seed's p₁ now includes the SI-Eq.-S73 photon term (reduces to the old ee-balance seed when Γ̃^ph_01 = 0) |

## Validation figures

The validation artifacts are self-pinned CSV baselines + PDF plots under
`validation/baselines/{ph0_constant, ph0_kaplan, transient, marchegiani_2025}/`
with regression tests under
`validation/{fischer_2023, fischer_2024, marchegiani_2025, transient}/`. A
passing pin proves stability against prior qpsim output, not independent
agreement with a publication. Current Fischer artifact readers fail closed on
versioned schema, exact configuration/axes, dependency hashes, physical
domains, complete data, and independently reassembled balance certificates;
only genuinely pre-schema artifacts take the narrow `LegacyArtifactError`
quarantine path, and writers replace files atomically. The four F24 strict-v2
canonicals bind 84 ordered solve certificates and passed live pin recomputation;
their pre-v2 CSV/PDF pairs are retained under
`validation/baselines/legacy/fischer_2024_pre_strict_v2/` as rejected audit
evidence. The transient photon-kick pin was regenerated 2026-07-16 under the
matched finite-volume measure and records its provenance in the CSV/PDF. Its
four slow regressions pass; full 810-bin runs at driver steps `0.2/0.1/0.05 ns`
bound the canonical `0.1`-versus-`0.05` differences by `2.56e-11` in `f` and
`5.89e-12` in `x_qp`. That is driver-partition step-insensitivity, not a
formal-order result, because adaptive ETD subcycling dominated the accepted
partitions. The Fig. 7 pin was regenerated under the tightened solver contract
at all 48 targets on Windows and Linux. After a post-publish NumPy-2.5 typing
repair outside the Fig. 7 call path advanced the conservative whole-tree
digest, an exact uncached Windows recertification completed in `975.48 s`. It
records maximum QP/Ph0 backward errors `9.819e-9`/`8.271e-9` against a `2e-8`
gate. The later direct-gap quadrature and structured-exception changes advanced
the digest again; a `1082.915 s` exact Windows rerun reproduced all 48 axes,
observables, and certificate arrays bitwise. The later structured gap-collapse
exception advanced it once more; a `1123.7 s` exact rerun was again bitwise
identical. The active solve-contract SHA-256 is
`ebe1382d509f6c52f11bca95b8d0161a211c4002a59f38de942cb2aefd193165`.
M25 baselines carry a `# pinned_on:` platform stamp; their strict pin tests run
only on the generating platform. (Historical reason: pre-normalization-fix
fixed-point selection was platform-sensitive. With the Γ̄ fix the tracked root
is unique and residuals sit at ~1e-12 Hz, so cross-platform scatter should now
be rounding-level — the stamp is kept as cheap insurance until that's verified
on a second machine.)

In the Fischer rows below, “Windows/Linux envelope” means the OS-family case,
not an exact hardware/runtime identity; its bounds are calibrated from the
hosted runs recorded in the audit.

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 paper legend ratios {0, 0.1, 1, 10} | `validation/fischer_2023/fig3_paper.py` | `ph0_constant/` | ⚠️ promoted nonzero NE=1620 certified local regression: peak occupations `[4.738e-10, 1.464e-9, 1.798e-8, 1.889e-7]`, max QP/Ph0 backward errors `9.60e-11`/`2.37e-6`. Its uncached full-pin repeat passed in 5224.19 s with roundoff-level curve differences and a pixel-identical PDF. Exact-capacity scalar refinement estimates `0.77--1.41%` remaining error. The prior `30--42%` Richardson strong-L1 warning is threshold-localized: NE=648→1620 direct strong L1 is `18.93--19.04%`, of which `87.5--88.0%` lies within `2.5 ueV` of `Delta+n*20 ueV`; ratio-zero NE=2592→5184 gives `0.613%` mass change and `0.178%` normalized W1 shape distance despite `12.34%` strong L1. CI pins BLAS to one thread and gives only the strong-bottleneck Newton polish a `1e-6` step/balance gate. Hosted Python 3.13/3.14 ratio-10 runs agreed with each other to about `1e-8` relative but differed from the stamped Windows pin by `1.27723%`; ratios through one and same-platform ratio 10 retain `1e-4`, while ratio 10 on the measured Windows/Linux pair uses `1.5e-2`. The `# pinned_on: win32` insertion changed only CSV metadata; numerical rows and the PDF are unchanged. No pointwise/total-variation or paper-parity claim. |
| Fischer 2023 Fig 5 paper-topology x_qp two-panel | `fig5_paper.py` | `ph0_constant/` | ⚠️ the cache-off full run failed 20/42 upper-panel legacy values. The former forward/reverse split was tolerance-induced at the tested `NE=1620`, `T*/Delta=0.60` target: tightening inner Newton backward error to `1e-10` moves direct/forward/reverse results to `2.45437e-6`/`2.45443e-6`/`2.45550e-6` (`<=0.046%` spread), so that target shows no physical branch ambiguity. Production now requires `1e-9` Picard balance/final certificates. The legacy `2.43414e-6` CSV/PDF remain quarantined pending full tight-contract regeneration/refinement of the broader sweep; no paper parity claim. |
| Fischer 2023 Fig 6 paper-topology gap suppression | `fig6_paper.py` | `ph0_kaplan/` | ⚠️ all 66 historical production targets certified under the declared loose contract, but a full-grid `T_B=0.1 K`, `nbar=1e4` tight solve shifts the default observable from `0.0371890241403` to `0.0367549216498` (`-4.341e-4` versus pin `atol=1e-6`) while `x_qp` moves only `0.0400%`. The fixed-gap/direct repair makes guard storage invariant, requires roundoff-scale gap-edge coverage, narrows strict Picard fallback to roundoff-level line-search stalls, pairs the mode flags, retains signed finite observables, requires a `1e-9` certificate, and keeps Windows console diagnostics ASCII-safe. Only explicit superconducting collapse becomes NaN; generic solver and non-finite derived-measurement failures propagate. Its exact point is `-0.0168056837` with QP backward error `1.619e-11`, exposed in arrays/console and by expanded signed direct-mode plotting. Programmatic generation uses distinct `_direct` paths and cannot clobber the default pin. No full direct canonical yet; the default baseline remains a tolerance-limited regression, not paper parity. |
| Fischer 2023 Fig 7 paper-facing Q_i,tot(T_B) | `fig7_paper.py` | `ph0_constant/` | ✅ exact tight-contract 48/48 production runs passed on Windows in `982.54 s` and Linux in `946.16 s`; later whole-tree provenance changes were followed by exact uncached Windows runs in `901.13 s`, `975.48 s`, `1082.915 s`, and `1123.7 s`. The latest two runs reproduced every axis, observable, and certificate array bitwise. The active pin has max QP/Ph0 backward errors `9.819e-9`/`8.271e-9` against a `2e-8` gate; solve-contract SHA-256 `ebe1382d509f6c52f11bca95b8d0161a211c4002a59f38de942cb2aefd193165`; CSV/PDF `b824b42cc3875a3a19d98134642745495bf4c746fb2be5e3ff43f80294f44890` / `93fc6db803dd8fc0226a3fd137a9052d5a96409c166684b2563f5d2bae524d05`. The latest provenance-only rebind preserved all 48 numerical rows and the PDF. Hosted single-thread CI measured loss drift `0.463267%` and `Q_tot` drift `1.634166e-4` while every state remained below the independent gate. Same-OS loss/`Q_tot` comparisons remain at `0.4%`/`1e-4`; the measured Windows/Linux pair uses `0.8%`/`2e-4`. The `2e-19` loss floor is unchanged. Further solver tightening selected a different low-occupation branch and was rejected. BLAS variables remain runtime provenance rather than serialized CSV-header fields. ⚠️ Regression/balance evidence, not bitwise identity or paper parity. |
| Fischer 2023 Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | `ph0_constant/` | ⚠️ all legacy `Q_i` values moved by up to `14.5144%`. The commensurate ladder remains nonconverged: at aligned `NE=6480`, `-100 dBm`, `Q_i=3.2994464e10`, the `3240 -> 6480` change is `4.44368%` with certificate `9.83e-12`. Exact-cell/FV variants are negligible; thermal response is monotone; cancellation condition is `16.6`; an overlap-aware photon prototype moves standard `Q_i` only `0.027%/0.010%` and leaves the rung at `4.567%`, so no photon-operator rewrite is justified. A proposed conservative policy—not a derived error budget—requires `<=1%` maximum `Q_i` change on two consecutive commensurate rungs and `<=0.25%` observable discrepancy. The pre-schema pin remains quarantined. |
| Fischer 2024 Fig. 5 paper-topology distributions | `fischer_2024/fig5_paper.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression after the commensurate `405/810/1620` ladder: `810 -> 1620` changes `x_qp` by `0.0794--0.0862%`, versus `13.37--14.84%` legacy/current offsets; BCS-capacity shape TV is `2.258--2.834%`. Max QP backward error `1.13e-7`; CSV/PDF SHA-256 `c58941e68e14a0080cbd48680cba082a5ae6906544755a47f6dfd132d8abca68` / `8a11a07739a770dd784cbaececbc8f8911cff8ca249428ce507cab8e64e0fd46`. Thermal-seed readback is cross-libm portable at an `8*eps`, zero-absolute-floor gate; runtime diagnostics and failures are ASCII-safe. ⚠️ Three analytic overlays remain `TODO(paper-parity)`; this is not paper parity. |
| Fischer 2024 Figs 5-7-topology f(E) characterization | `fischer_2024/figs_5_7_fe_pb.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. Across all five powers, `810 -> 1620` changes `x_qp` by `0.0062--0.1903%` and shape TV by `0.495--1.156%`, versus `0.858--4.283%` legacy/current scalar offsets. Certificate maxima `3.39e-13`/`7.45e-17`; CSV/PDF SHA-256 `de2094308cebeaa0d07799a2b9d7c51eee83e97702c4976da169d8fb5e9786dd` / `069aa4f61e9d36ba6ac0ea0eb35d26c115ad79869948a6d7261d501e370d59f9`. Thermal-seed readback is cross-libm portable at an `8*eps`, zero-absolute-floor gate; runtime diagnostics and failures are ASCII-safe. ⚠️ Fixed-grid regression, not a pointwise continuum-shape or paper-parity claim. |
| Fischer 2024 Fig-8-topology x_qp(T_B) characterization | `fischer_2024/fig8_xqp_pb.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. Low/fixed/high-temperature probes reproduce the five-drive `0.0062--0.1903%` production-grid scalar range; all 40 points recertify. CSV/PDF SHA-256 `f1155dd44879661d8c1cff7d105e0a8d1012c3a0008f17bc37d866498b888be5` / `8636d8d9dd0d4900e1481de02187d177c6bb42d68af62e29769d309a7b0354c1`. Runtime diagnostics and failures are ASCII-safe. ⚠️ Qpsim-native characterization, not paper parity. |
| Fischer 2024 Fig. 8 paper-topology density | `fischer_2024/fig8_paper.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. The old weak-drive 0.05 K values were false-converged thermal seeds; direct seeds now fail the backward-error gate, while strong-to-weak continuation gives `4.045e-7`/`4.048e-8` and converges on all ladder grids. `810 -> 1620` scalar changes are `0.0830--0.1089%` at 0.05 K and `0.3102%` at 0.30 K. All 36 points certify with max backward error `9.10e-7`; CSV/PDF SHA-256 `69a577e633ef12afee5008fa54e7a593cdf06211bcbc7c2b3481854255cd120d` / `3e89910af8777541f88bdd71c5308c170dc3e17764052b2eda013690ca3dfd12`. Runtime diagnostics and failures are ASCII-safe. ⚠️ Analytic density overlay remains a placeholder; not paper parity. |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ matched-measure pin regenerated with provenance; 4 slow regressions passed in 752.05 s and the exact module passed again in 726.54 s after the encoding repair; `dt=0.2/0.1/0.05 ns` study establishes driver-partition insensitivity, not formal order. The CSV is now explicitly BOM-free UTF-8/LF after a Windows CP1252 title byte broke Ubuntu readback; the numerical payload is byte-identical and the canonical SHA-256 is `18e2a2424c037e2b6dd64189848765d0a0c75a6b6cc4bed63364c3f2d05c51d1`. |
| Marchegiani 2025 Eq. 8 T̄ | `validation/marchegiani_2025/fig3_crossover_temperature.py` | `marchegiani_2025/` | ✅ closed-form Lambert-W |
| Marchegiani 2025 Fig 3 (μ_α vs T) | `validation/marchegiani_2025/fig3_chemical_potentials.py` | `marchegiani_2025/` | ✅ branch-continuation driver + Γ̄-normalized density eqs + paper-exact μ inversion (SI Eqs. S2–S5); both panels match the published curves (panel a: three μ_α merged, linear 0.94 → 0 at T̄ ≈ 146 mK; panel b: μ_L ≳ μ_R> > μ_R< at low T with the R-band merge at ~50 mK); smooth through the crossover, μ → 0 at T̄ |
| Marchegiani 2025 Fig 3 paper-styled panels | `validation/marchegiani_2025/fig3_paper.py` | `marchegiani_2025/` | ✅ paper-faithful (`m25_fig3{a,b}_paper.csv`, insets + regime shading); same solver path as the row above; strict pin rtol=1e-6, fast (runs in the default gate) |
| Marchegiani 2025 Fig 4 (Γ_P, Γ̃^eo_01/Γ̃^eo_10 vs T) | `validation/marchegiani_2025/fig4_parity_rates.py` | `marchegiani_2025/` | ✅ multi-stability scatter gone (was a conditioning artifact of the missing Γ̄ normalization); Γ_P smooth with the paper's low-T nonmonotonic dip on panel a; smoothness enforced in the test (max adjacent |Δlog10 Γ_P| < 0.2) |
| Marchegiani 2025 Fig 4 paper-styled two-stack | `validation/marchegiani_2025/fig4_paper.py` | `marchegiani_2025/` | ✅ paper-faithful (`m25_fig4_paper.csv`): full model (both ω_LR) + global-quasiequilibrium reduction (SI Note 1 density ratios + total-density closure, M25 Eq. 7) + renormalized global-QE curve (Fig. 4 caption: Γ^ph_00 = 600 Hz, ω_LR/2π = 6 GHz; 5 GHz family only) + exp(−ω_10/T) detailed-balance dotted reference; reproduces the paper's diagnostic (renorm mimics full Γ_P but deviates on the excitation/relaxation ratio at low T) |

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

`T3Spatial1DBackend.apply_transport` is an exactly-conservative
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

- ✅ `steady_state` — Newton + Picard + Anderson + coupled-Newton routing; homogeneous Newton requires dimensional residual plus normwise L1 gain/loss backward error, and Picard requires local/normwise mapped-fixed-point tests plus a physical nearest-binary64 Ph0 certificate. The raw affine residual is retained separately, and projection to a certified phonon map is followed by one final Newton solve so the returned pair is matched. Finite-escape Picard accepts a validated same-grid `initial_phonon_guess`; the T3 backend forwards both `f` and `n_ph` during continuation and rejects a state whose public gap/model disagrees with the spectral/Ph0 operators.
- ✅ `nbar_loop` — F23 Eqs. 59-60 fixed-point on (n̄, Q_i).
- ✅ `transient` — `run_time_dependent` with fixed-gap ETD2 collision substeps, stage-constrained second-order moving-gap backend steps, snapshot cadence, autonomous stop-tol, and observables dict.
- ✅ `rate_equation` — Eq. 8 Lambert-W T̄ closed-form observable + `solve_rate_equation_steady_state` Newton solver on the 4-unknown M25 boxed system (density equations on Γ̄ = Γ̃/`cooper_pair_number_R`; auto-scaled residual tolerance) + `solve_rate_equation_branch` bidirectional continuation driver for temperature sweeps (warm-started natural continuation, adaptive step bisection, reduced 1-D bracketing fold handler, documented photon/thermal exchange rule) + `thermal_equilibrium_seed` / `analytic_low_T_seed` seeds.
- partial `rate_equation_coefficients` — `M25PhysicalParameters` + `coefficients_from_physical_parameters` build the `M25Coefficients` bundle from primitive physical inputs (gaps, ω_10, transmon E_J/E_C, T, junction tunneling scale `R_T = g_T Δ̄/e²`, caption-level `r^L`/`r^{R<}`). Implements SI Notes III (12 tunneling rates) and IV (recombination, generation-by-thermal-phonons, intraband relaxation, branching fraction). Note V photon-assisted tunneling shipped via `M25PhotonDrive` + `coefficients_from_physical_parameters_with_photon_drive`: elliptic-integral spectral densities `_S_ph_total` (S57), `_S_ph_Rgt` (S59), `_S_ph_Rlt = total − Rgt`; all four `Γ^{ph}_{ij}` derived from single back-solvable `Gamma_nu_scale_Hz`; per-state `g^{ph}_α` arrays feed the population-dependent residual.
- planned `parametric_sweep` — factor the nested-loop pattern duplicated across validation modules.

## Observables (`qpsim.observables.*`)

- ✅ `ac_conductivity` — Mattis-Bardeen σ_1, σ_2.
- ✅ `quality_factor` — Q_i = σ_2 / (α σ_1).
- ✅ `frequency_shift` — δω/ω fractional.
- ✅ `density` — `qp_number_density`; historical qpsim `qp_fraction = n_qp/(4ρ_FΔ_0)`; explicit Fischer/Catelani `qp_fraction_paper = n_qp/(2ρ_FΔ_0)`.
- ✅ `gap_suppression` — δΔ, δΔ/Δ_eq; `compute_gap_suppression(f, E, T_c, T_bath)` + `gap_suppression_from_deltas(Δ_eq, Δ_final)`. Occupation-backed solves fail closed unless the first cell face covers the selected gap; web summaries retain independently calibrated `delta_eq_ueV` when derived suppression is unsupported.
- ✅ `effective_temperature` — `effective_phonon_temperature(n_ph, ω_bins, gap, T_bath)` via weighted BE fit per F23 Eq. 36.
- planned `charge_imbalance` — δN from f_T (NFP §5 Table row 443).

## Test suite

The final default repository aggregate is **1549 passed, 17 deselected, 4
warnings in 525.03 s** on Windows (`pytest -q`). The deselections are the opt-in
slow/manual validation selections. The final exact tree also passes `ruff check
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
pinned/live row drift was `2.58e-14`. The subsequent ASCII-only provenance
rebind was proven string-only by normalized-AST identity. Other focused reruns
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

Additional settled focused evidence on the current code includes the refined-reference
second-order moving-gap regression and its invariant tests; 8 focused
self-consistent-gap tests without the former extrapolation warnings; 37 passing
tests in the broader affected T3 backend group; 132 passing homogeneous
Newton/Picard/Ph0 certificate tests; 4 passing slow transient tests in 752.05 s
and an exact post-encoding-repair module rerun in 726.54 s;
exact tight-contract 48-point Fig. 7 production runs on Windows in 982.54 s and
Linux in 946.16 s, plus exact-source Windows recertifications in 901.13 s,
975.48 s, 1082.915 s, and 1123.7 s; the latest two were bitwise identical. A certified
NE=1620 Fig. 3 production run in 5768.40 s plus an uncached full-pin pass in
5224.19 s; and all 66 certified Fig. 6 targets in 7599.29 s concurrent wall
time. The cache-off Fig. 5 audit took 8088.34 s:
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

The final Fig. 3/Fig. 7 non-slow policy slice passed **44 tests with 4 slow
deselected in 4.70 s**. It includes the exact hosted Fig. 7 drift values and a
rounded Fig. 3 policy fixture; both fail strict gates and pass the calibrated
Windows/Linux OS-family envelopes, while macOS or unknown pairs remain strict.

Recent hardening pass (Claude+GPT cross-review, seventh session):
- Photon collision kernels (`sub_gap_photon`, `pair_breaking_photon`) and the analytic Newton Jacobian now hard-reject nonuniform energy grids via `qpsim.collisions._uniform_grid.uniform_grid_spacing`; previously they silently used `dE[0]` as a uniform stride.
- `PhononState.__post_init__` now validates finite/nonneg `n_ph`, finite/nonneg/strict-monotone `omega_bins`, and finite/nonneg `tau_l` (was shape-only).
- `phonon_steady_state` raises on singular/runaway phonon balances instead of clipping negative occupations to zero — turns "no Ph0 fixed point" from a fake solution into a loud failure.
- Homogeneous Newton/Picard convergence now has scale-independent physical balance gates. Affine Ph0 certification distinguishes the raw direct-form residual from whether the stored value is the nearest nonnegative binary64 root; negative, signed-underflow, singular, overflowed, and nonfinite roots fail closed.
- `accept_lm_convergence` bypass narrowed to its documented scope: only the `is_no_progress_stall` case is exempted from the residual check; success-with-high-residual now raises.
- `transient.run_time_dependent` truncates the final substep so `total_time` is honored exactly.
- Four M25 Fig 3/4 multi-stable branch points shifted under the stricter residual acceptance; CSV+PDF baselines regenerated; visual paper agreement preserved.

Slow tests (opt in with `-m slow`):
- Fischer validation reproductions at Fischer-scale grids.
- The transient photon-kick regression is slow-marked and its four-test subset
  passed on 2026-07-16. Fig. 7 has exact tight-contract 48-point Windows/Linux
  production evidence plus the final-source Windows recertification. The four
  promoted F24 pytest pins passed in 76.86 s and their later current-solver
  84-state recertification passed in 95.07 s.
  All 16 non-manual slow nodeids have direct CI evidence on the recorded hosted
  tree (14 passes and 2 F23 legacy xfails), including the Fig. 7 full-pin wrapper
  under Python 3.13 and 3.14. Fig. 6 has historical 66-point loose-contract production
  evidence, but its unrun serial wrapper is not closure of the direct-observable
  qualification; M25 Eq. 8 sweeps remain fast.

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
  - Specs: `~/Documents/Quasiparticle Simulation/Documentation/Current/` — `New Framework Plan.md` is authoritative.
  - Prelim deck: `~/Documents/Graduate/Preliminary Exam/build_presentation.py` (uses `/opt/homebrew/bin/python3` for python-pptx).
