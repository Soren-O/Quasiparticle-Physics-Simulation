# qpsim status (gate tracker)

Last updated: 2026-07-17 (code-only numerical audit follow-up; see
`docs/AUDIT-2026-07-15-numerical-software.md`). Paper-reproduction status is
qualified: self-pinned CSV parity is regression evidence, not independent
paper truth. Fischer 2023 Fig. 3 now has a promoted nonzero, certified NE=1620
artifact. Its prior 30--42% strong shape warning is localized to ideal-BCS
photon-threshold layers: scalar mass and weak Wasserstein shape converge much
faster, while pointwise/total-variation convergence remains unclaimed. An
uncached full-pin repeat passed. Fig. 6 completed and certified all 66
production targets. Fig. 7 now has exact tight-contract 48-point Windows and
Linux production evidence; its historical scatter was traced to the loose
inner-Newton stopping rule and closed with a strict, provenance-bound pin. The
old F23 Fig. 5 and Figs. 9--13 pins remain quarantined after targeted refinement
found, respectively, certified continuation hysteresis and a nonconverged
`Q_i` observable. All four Fischer 2024 families
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
| 3 | T3 reproduction regression against self-pinned baselines | ⚠️ qualified, not complete paper parity: Fischer 2023 Fig. 3 is promoted as a nonzero NE=1620 certified local regression and its uncached repeat passed. Its strong curve error is an ideal-BCS threshold-layer/norm limitation; mass and weak shape converge, but pointwise/total-variation continuum convergence is not claimed. Fig. 6 is certified at all 66 production targets. Fig. 7 is certified at all 48 targets under exact tight-contract Windows and Linux production runs; its loose-inner-Newton cross-platform defect is resolved, though the pin remains regression rather than paper-parity evidence. The legacy F23 Fig. 5 and Figs. 9--13 pins remain quarantined. Four F24 families are promoted as certified qpsim-native regressions at paper topology after nested-grid refinement, while their analytic paper-target overlays remain incomplete. |
| 4 | Full Layer-4 audit chain | ⚠️ the final default repository aggregate is green at **1499 passed, 17 slow/manual deselected, 4 warnings**; five stale finite-volume validation oracles were repaired and their diffusion folder passes 14/14. Fifteen of 16 non-manual slow nodeids have direct execution evidence (13 passes, 2 intended F23 legacy quarantines); the Fig. 7 full-pin wrapper instead has equivalent exact-source 48-target production evidence. The serial Fig. 6 `manual_slow` wrapper likewise has equivalent certified 66-target production evidence. Figure-specific qualifications remain; see `AUDIT-2026-07-15-numerical-software.md`. |
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
at all 48 targets on Windows and Linux. A final frozen-source Windows rerun
records maximum QP/Ph0 backward errors `9.819e-9`/`9.422e-9` against a `2e-8`
gate and solve-contract SHA-256
`b89aa5232c87ec8dc4e2e0f9037bd7c635afa65602d85ea6da77a07a24aeeaff`.
M25 baselines carry a `# pinned_on:` platform stamp; their strict pin tests run
only on the generating platform. (Historical reason: pre-normalization-fix
fixed-point selection was platform-sensitive. With the Γ̄ fix the tracked root
is unique and residuals sit at ~1e-12 Hz, so cross-platform scatter should now
be rounding-level — the stamp is kept as cheap insurance until that's verified
on a second machine.)

| Figure | Module | Baseline dir | Status |
|---|---|---|---|
| Fischer 2023 Fig 3 paper legend ratios {0, 0.1, 1, 10} | `validation/fischer_2023/fig3_paper.py` | `ph0_constant/` | ⚠️ promoted nonzero NE=1620 certified local regression: peak occupations `[4.738e-10, 1.464e-9, 1.798e-8, 1.889e-7]`, max QP/Ph0 backward errors `9.60e-11`/`2.37e-6`. Its uncached full-pin repeat passed in 5224.19 s with roundoff-level curve differences and a pixel-identical PDF. Exact-capacity scalar refinement estimates `0.77--1.41%` remaining error. The prior `30--42%` Richardson strong-L1 warning is threshold-localized: NE=648→1620 direct strong L1 is `18.93--19.04%`, of which `87.5--88.0%` lies within `2.5 ueV` of `Delta+n*20 ueV`; ratio-zero NE=2592→5184 gives `0.613%` mass change and `0.178%` normalized W1 shape distance despite `12.34%` strong L1. No pointwise/total-variation or paper-parity claim. |
| Fischer 2023 Fig 5 paper-topology x_qp two-panel | `fig5_paper.py` | `ph0_constant/` | ⚠️ high-drive branch guard passed, but the cache-off full run failed 20/42 upper-panel pin values after 8088.34 s total. A representative `NE=162/324/648/1296/1620` ladder reduced the maximum selected scalar rung change from `23.89%` to `0.443%`, but left up to `9.24%` strong shape change and about `2.21%` Richardson scalar remainder near the transition. More decisively, certified forward/reverse paper-grid roots at `T*/Delta=0.60` give `x_qp=4.57835e-8`/`4.82952e-6` (`99.052%` separation; max QP/phonon backward errors `4.56e-7`/`5.70e-7`). Branch selection is an unresolved physical contract, not a residual failure. The pre-schema CSV/PDF remain untouched and quarantined; no paper parity claim. |
| Fischer 2023 Fig 6 paper-topology gap suppression | `fig6_paper.py` | `ph0_kaplan/` | ⚠️ all 66 production targets completed in the exact 3x22 schema with max QP/certified-Ph0/gap-map errors `9.27e-7`/`3.31e-6`/`9.92e-11 µeV`, inside `1e-5/1e-5/1e-10` gates. This is certified local regression evidence, not paper parity. |
| Fischer 2023 Fig 7 paper-facing Q_i,tot(T_B) | `fig7_paper.py` | `ph0_constant/` | ✅ exact tight-contract 48/48 production runs passed on Windows in `982.54 s` and Linux in `946.16 s`; after final unrelated whole-tree provenance changes, a frozen-source Windows rerun passed in `901.13 s`. The active pin has max QP/Ph0 backward errors `9.819e-9`/`9.422e-9` against a `2e-8` gate; solve-contract SHA-256 `b89aa5232c87ec8dc4e2e0f9037bd7c635afa65602d85ea6da77a07a24aeeaff`; CSV/PDF `4a0dec419764f2e7bf7eb0cdec0823a3e8303456f4e49c9898bdabd4e4c8bef2` / `17bda31fdbae541769a3d347a77a940c49adf75d7348b77fab0fbfcb85872262`. Meaningful cross-platform loss drift is at most `0.244554%`; the larger `0.06 K` relative tail is only `1.0941e-19` absolute. The loose pair and tight Linux predecessor are archived. ⚠️ Regression/balance evidence, not bitwise identity or paper parity. |
| Fischer 2023 Sec. V Q_i(P_read) characterization | `figs_9_13_qi_vs_pread.py` | `ph0_constant/` | ⚠️ fresh 21-point run certified, but all legacy `Q_i` values moved by up to `14.5144%`. The full commensurate `NE=405/810/1620/3240` ladder has successive maximum `Q_i` changes `2.83%/3.99%/4.38%`; at `-100 dBm` the error grows rather than settles. Mass changes improve `4.12%→1.83%→0.938%` and normalized W1 remains `≤0.242%`, while strong L1 decreases `24.43%→19.84%→17.30%` with `84–89%` threshold localization. NE=1620 forward/reverse `Q_i` agrees to `6.67e-6`, and certificate maxima are QP `1.37e-11`/outer `1.56e-11`: observable discretization, not branch or residual failure. The `NE=405` pre-schema pin remains untouched and quarantined. Not a literal paper figure. |
| Fischer 2024 Fig. 5 paper-topology distributions | `fischer_2024/fig5_paper.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression after the commensurate `405/810/1620` ladder: `810 -> 1620` changes `x_qp` by `0.0794--0.0862%`, versus `13.37--14.84%` legacy/current offsets; BCS-capacity shape TV is `2.258--2.834%`. Max QP backward error `1.13e-7`; CSV/PDF SHA-256 `28bf2ec12d30936276c6a57f89d84132f5758fce03d569c30b06f3aa95af8e4f` / `8a11a07739a770dd784cbaececbcf8911cff8ca249428ce507cab8e64e0fd46`. ⚠️ Three analytic overlays remain `TODO(paper-parity)`; this is not paper parity. |
| Fischer 2024 Figs 5-7-topology f(E) characterization | `fischer_2024/figs_5_7_fe_pb.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. Across all five powers, `810 -> 1620` changes `x_qp` by `0.0062--0.1903%` and shape TV by `0.495--1.156%`, versus `0.858--4.283%` legacy/current scalar offsets. Certificate maxima `3.39e-13`/`7.45e-17`; CSV/PDF SHA-256 `a7442881e91ca4b26bf6a2a5364c3c0281034a4bd6c91677fa0086b0d9961add` / `069aa4f61e9d36ba6ac0ea0eb35d26c115ad79869948a6d7261d501e370d59f9`. ⚠️ Fixed-grid regression, not a pointwise continuum-shape or paper-parity claim. |
| Fischer 2024 Fig-8-topology x_qp(T_B) characterization | `fischer_2024/fig8_xqp_pb.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. Low/fixed/high-temperature probes reproduce the five-drive `0.0062--0.1903%` production-grid scalar range; all 40 points recertify. CSV/PDF SHA-256 `31fe7a4f46b57c88c9f53389959f4e8a7680d97fa4e65c5e12535d0087a5aa00` / `8636d8d9dd0d4900e1481de02187d177c6bb42d68af62e29769d309a7b0354c1`. ⚠️ Qpsim-native characterization, not paper parity. |
| Fischer 2024 Fig. 8 paper-topology density | `fischer_2024/fig8_paper.py` | `ph0_constant/` | ✅ certified qpsim-native `NE=810` regression. The old weak-drive 0.05 K values were false-converged thermal seeds; direct seeds now fail the backward-error gate, while strong-to-weak continuation gives `4.045e-7`/`4.048e-8` and converges on all ladder grids. `810 -> 1620` scalar changes are `0.0830--0.1089%` at 0.05 K and `0.3102%` at 0.30 K. All 36 points certify with max backward error `9.10e-7`; CSV/PDF SHA-256 `b0c884f670fbcee2e193c849b8ca87902f2595e1115bdd3e2e2fb7115d9342eb` / `3e89910af8777541f88bdd71c5308c170dc3e17764052b2eda013690ca3dfd12`. ⚠️ Analytic density overlay remains a placeholder; not paper parity. |
| Transient photon-kick f(E, t) | `validation/transient/photon_kick_response.py` | `transient/` | ✅ matched-measure pin regenerated with provenance; 4 slow regressions passed in 752.05 s; `dt=0.2/0.1/0.05 ns` study establishes driver-partition insensitivity, not formal order |
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
distinct closed equilibria).

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

The final default repository aggregate is **1499 passed, 17 deselected, 4
warnings in 533.61 s** on Windows (`pytest -q`). The deselections are the opt-in
slow/manual validation selections. The final exact tree also passes `ruff check
.`, `mypy qpsim` across 75 source files, `python -m compileall -q qpsim tests
validation`, and `git diff --check` (apart from Git's Windows line-ending
notices). The earlier unit/API-only checkpoint was **1272/1272 passed, 13
warnings in 258.88 s**. These aggregates do not by themselves establish the
separately recorded Fig. 3/Fig. 6/Fig. 7 production outcomes, refinement, or
paper parity.

The 17 default deselections were exhaustively inventoried on 2026-07-17.
Fifteen of 16 non-manual `slow` nodeids have direct execution evidence: 13
numerical passes and 2 narrow expected xfails for F23 pre-schema canonicals; the
latter reran together in 1.84 s. The Fig. 7 full-pin wrapper instead has
equivalent exact-source 48-target production evidence, including the final
frozen-source Windows run in 901.13 s. The four F24
full live pins passed together in 76.86 s after refinement and strict-v2
promotion. Other focused reruns include the tight-contract Fig. 7 plateau (4.20 s), reduced
Fig. 6 continuation (3.71 s), and post-schema F23 Fig. 5 high-drive branch
(287.42 s). The exact Fig. 7 full-pin and serial Fig. 6 `manual_slow` wrappers
were not separately invoked on the final contracts; their underlying 48/48 and
66/66 production sweeps are independently certified, which is equivalent
numerical evidence but not exact node-pass evidence.

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
Newton/Picard/Ph0 certificate tests; 4 passing slow transient tests in 752.05 s;
exact tight-contract 48-point Fig. 7 production runs on Windows in 982.54 s and
Linux in 946.16 s, plus the frozen-final-source Windows run in 901.13 s; a certified
NE=1620 Fig. 3 production run in 5768.40 s plus an uncached full-pin pass in
5224.19 s; and all 66 certified Fig. 6 targets in 7599.29 s concurrent wall
time. The cache-off Fig. 5 audit took 8088.34 s:
its physical branch guard passed, while its stale pin failed 20/42 upper-panel
values. Figs. 9--13 artifact hardening passed 12 fast tests. Fischer 2024
artifact hardening passed 55 fast tests plus four reduced live solve/write/read
paths; after promotion its current focused suite passed **55 tests with 4 slow
deselected in 17.58 s**, and those four full pins passed in **76.86 s**. F23
Fig. 5 artifact hardening passed 9 fast tests; the
combined F23/F24 selection passed 75 tests with 7 slow tests deselected in
92.01 s. These focused results are scoped, not a repository-wide aggregate,
and do not establish paper parity or close the stated refinement
qualifications.

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
  production evidence plus the final-source Windows rerun, and the four
  promoted F24 pins passed in 76.86 s.
  Fifteen of 16 non-manual slow nodeids ran directly (13 passes and 2 F23 legacy
  xfails); the Fig. 7 full-pin wrapper has equivalent exact-source production
  evidence. The serial Fig. 6 full-pin wrapper likewise has equivalent 66-point
  production evidence; M25 Eq. 8 sweeps remain fast.

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
