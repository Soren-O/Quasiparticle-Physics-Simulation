# qpsim software paper — architecture and project plan

Status: planning document, created 2026-07-03. No manuscript exists yet; this file is the
blueprint. The charter comes from the round-3 progress deck (slide 10, "Publication plan"):

> **Paper 2 — qpsim.** Introduce the simulation package to the detector/qubit community.
> Full pipeline: validated collision kernels → spatial transport (A1 operator family + gap
> feedback) → KID observables. Validation spine: Fischer 2023 Figs. 3/5/6/7, Fischer 2024
> pair-breaking photons, Marchegiani 2025 junction rates. Drafting begins after Paper 1
> submission and the merge to main.

The merge to main is done. Paper 1 (`papers/qp-diffusion/`) is in final polishing
(items 2–10 of the 2026-07-02 open-items list). This document architects Paper 2 so drafting
can start the moment Paper 1 is submitted.

---

## 1. Working title and one-paragraph pitch

Working title (variants to choose from):

- *qpsim: a validated Python framework for nonequilibrium quasiparticle and phonon kinetics
  in superconducting devices*
- *qpsim: simulating nonequilibrium superconducting quasiparticles from collision kernels to
  device observables*

Pitch: Nonequilibrium quasiparticles limit superconducting qubits (parity switching, T1) and
set the response of kinetic-inductance detectors. Every group models them with hand-rolled,
unpublished kinetic solvers; results are hard to reproduce and the literature disagrees even
on the form of the diffusion operator. qpsim is an open-source (MIT), pure NumPy/SciPy
framework that solves the energy-resolved kinetic equations for f(E,x,t) coupled to phonons
n_ph(ω,t) and a self-consistent gap Δ(x,t), composes multi-region devices with junctions and
qubits, and maps solutions to experimental observables (x_qp, σ₁/σ₂, Q_i, δf_r, parity rates).
Physics channels are pinned by analytic identities (exact discrete detailed balance,
Mattis–Bardeen limits) and paper-topology numerical regressions derived from three
papers (Fischer & Catelani 2023, 2024; Marchegiani & Catelani 2025). These are
qpsim-generated regression artifacts, not an independent digitized-data comparison.
The diffusion-operator
family it implements is the subject of the companion theory paper [qp-diffusion], which
adjudicates the correct operator; qpsim is the reference implementation of that result.

## 2. Venue

Recommendation: **Computer Physics Communications** (primary). Full-length software papers
with real physics + numerics + validation are exactly CPC's format; the detector/qubit
modeling community publishes solver papers there; CPC's Program Library gives the release a
citable archival home.

Alternatives, in order:
- **SciPost Physics Codebases** — open access, community-visible, allows full physics
  exposition, versioned releases. Strong second choice; arguably better read by the
  superconducting-qubit community.
- **SoftwareX** — shorter format; would force cutting the physics/numerics exposition that
  is half the point. Fallback only.
- **JOSS** — too short for this scope; not recommended (and dual publication is discouraged).

Decision needed from Soren (see §8).

## 3. Positioning

1. **Relative to Paper 1 (`papers/qp-diffusion/`)**: Paper 1 answers *which* diffusion
   equation is correct (A1, (p,q)=(1,0)) and why. Paper 2 presents the *framework*: the full
   collision/phonon/gap/transport/device/observable stack, the numerical methods, the
   software architecture, and the validation program. Paper 2 cites Paper 1 for the operator
   adjudication and implements the whole L_{p,q} family so users can reproduce the
   comparison. Do not re-derive; re-show at most one operator-benchmark figure (§6, Fig. 7)
   with fresh rendering.
2. **Relative to other codes**: there is no community-standard open kinetic solver for
   nonequilibrium superconducting QPs. Related-work paragraph should survey: ad-hoc
   kinetic-equation solvers in the cited papers (Fischer/Catelani, Marchegiani/Catelani),
   QP-diffusion device modeling in the trap-engineering literature (Riwar et al.),
   Rothwarf–Taylor lumped models, KID-response tools, and circuit-level packages (scqubits,
   pyEPR etc.) which do *not* treat QP kinetics — that's the gap qpsim fills. (Verify
   novelty claim with a literature sweep before submission.)
3. **Audience**: superconducting-qubit groups (parity/T1 budgeting), KID/MKID detector
   groups (responsivity, Q_i forecasting), materials/trap-engineering groups.

## 4. Section-by-section outline

Source material for prose is deliberately reused from in-repo docs (paths given).

### §1 Introduction
- QP problem statement for qubits + detectors; reproducibility gap; hand-rolled solvers.
- What qpsim is: an energy- and space-resolved solver for f(E,x,t) in the isotropic dirty
  limit, coupled to a local phonon bath with acoustic escape to the substrate; design goals
  (validated, minimal deps, typed/strict, guarded conventions).
- Relationship to Paper 1. Contributions list.
- Sources: `README.md`, `docs/STATUS.md`, round-3 deck slides 1–3, 10.

### §2 Physics models
- State variables f(E), n_ph(ω), Δ; unit system (µeV/ns) — `qpsim/constants.py`.
- Spectral layer: BCS + Dynes DOS, coherence factors K±, SpectralContext caching —
  `qpsim/physics/spectral.py`.
- Collision channels (each with its kernel and detailed-balance property):
  e-ph scattering + recombination (Kaplan kernels, `physics/kernels.py`,
  `collisions/phonon.py`); sub-gap photon (`collisions/sub_gap_photon.py`); pair-breaking
  photon (`collisions/pair_breaking_photon.py`, Fischer 2024 Eqs. 2–5).
- Phonon sector: local balance with escape time τ_l(ω) (`phonon_models/local.py`,
  `physics/phonon_escape.py` — constant and acoustic-mismatch builders), pair-breaking
  lifetime τ_PB(Ω) (`physics/kaplan_pair_breaking.py`, closed elliptic form). Why the
  Rothwarf–Taylor ζ factor is *not* double-counted in PDE backends
  (`docs/Phonon_Model_Decisions.md`).
- Self-consistent gap: calibrate/solve, cosh substitution — `physics/gap_equation.py`.
- Spatial transport: the L_{p,q} operator family, A1 default, sub-gap flux blocking (D_L
  indicator), KL interface condition — `transport/diffusion/base.py`,
  `docs/Diffusion_Operators.md`; cite Paper 1.
- Device composition: Region/Junction/Device/Qubit, M25 gap-asymmetric junction rate
  equations, parity-tracked qubit master equation — `qpsim/devices/`,
  `docs/Device_Architecture.md`, `services/rate_equation*.py`,
  `docs/M25_coefficient_integrals.md`.
- Observables: Mattis–Bardeen σ₁/σ₂, Q_i, δω/ω, x_qp, gap suppression, effective phonon
  temperature; self-consistent readout photon-number loop (Eqs. 59–60 of F23) —
  `qpsim/observables/`, `services/nbar_loop.py`.
- Materials database: YAML schema, Al/Nb/TiN shipped — `qpsim/materials/`,
  `docs/Material_Database.md`.
- Main prose skeleton: `docs/Part_II_Physics.md`.

### §3 Numerical methods
- Grids: cell-centered uniform energy grid; photon-commensurability guards; phonon frequency
  map as union {|E_i−E_j|, E_i+E_j} ⇒ *exact* discrete detailed balance (a headline claim —
  most codes only get this approximately).
- Steady state: Newton with analytic Jacobian; coupled (f, n_ph) Newton for the strong
  phonon-bottleneck regime; Picard + Anderson (type-II); continuation strategies in n̄, T_B,
  τ_l; M25 multi-seed hybr root-finding.
- Time dependence: Strang split (collision/transport/gap), ETD2 exponential collision
  substep with cancellation-safe Taylor fallback, Crank–Nicolson conservative FV transport
  on u = N₁^p f (conservation to ~1e-15), TVD spectral-flow advection + SSPRK(2,2) for Δ̇.
- Conservation and guard philosophy: raise-don't-clip, sentinel conventions, tolerance tiers
  (1e-12 / 1e-6 / 1e-4) — `docs/Validation_Chain.md`.
- Convergence verification: prelim convergence-check discipline (≤3% envelope;
  `scripts/run_prelim_convergence_checks.py`) generalized into a stated method.
- Main prose skeleton: `docs/Part_III_Numerics.md`.

### §4 Software architecture
- Package layout diagram (physics / collisions / solvers / grid / transport / backends /
  services / devices / observables / phonon_models / materials / experiments / webui).
- Backend–service–device layering; "UI adds no physics" principle.
- Web UI: FastAPI local app, four run modes, JSON setups + NPZ results, server-side plots —
  `docs/Frontend.md`. One figure (§6, Fig. 10).
- QA: ~700-test default gate, mypy --strict, ruff, validation chain, the two physics audits
  (2026-06-10, 2026-07-02) as a *methodology* worth describing (physics-faithfulness
  auditing as part of software QA is novel-ish and reviewers will like it).
- Availability: GitHub (public repo), MIT, `pip install qpsim[ui]`, Python ≥3.13.

### §5 Verification and validation
Three layers, in increasing integration:
1. **Analytic identities** (run in the default gate): detailed balance vanishing at
   (f_FD, n_BE) for all three collision channels; Mattis–Bardeen thermal limits; gap-equation
   round-trip — `validation/analytic/`.
2. **Literature-derived numerical regressions** (the validation spine; `validation/{fischer_2023,
   fischer_2024, marchegiani_2025}/` + pinned CSV baselines):
   - F23 Fig. 3 — f(E) under sub-gap drive, finite τ_l family.
   - F23 Figs. 5/6 — x_qp and gap enhancement/suppression observable.
   - F23 Fig. 7 (+9–13) — MKID chain: Q_i(T_B), Q_i(P_read) with self-consistent n̄ loop.
   - F24 Figs. 5–7 — pair-breaking-photon f(E); placeholder Neumann-series
     estimates are explicitly non-canonical until derived.
   - F24 Fig. 8 — x_qp(T_B) under fixed PB drive.
   - M25 Eq. 8 — Lambert-W crossover T̄ (machine precision).
   - M25 Fig. 3 — three chemical potentials; authenticated qpsim
     paper-topology regression with broad manual anchors, not a digitized
     pointwise agreement measurement.
   - M25 Fig. 4 — parity-switching rate + eo ratio.
3. **Operator benchmarks** (shared with Paper 1; cite, show at most one).

Presentation rule: figures show **qpsim output only**. A future validation
campaign may digitize and overlay published curves where licensing permits;
that quantitative paper-data layer is not currently implemented in this tree.
Never reprint published figure panels (avoids permissions entirely).

### §6 Demonstration: an end-to-end device study
The showcase that no other section provides: the preliminary-exam experiment forecast,
rebuilt as a worked example. 100 µm × 0.1 µm Al strip at the current antinode of six
quarter-wave resonators (~5–6 GHz, α_KI = 0.08), JJ injection source (Gaussian at E = 2Δ,
σ = 0.08Δ), reflective 1D diffusion, local phonon bottleneck.
- Sweep 1: δf_r vs τ_l at three injection rates (phonon bottleneck is the dominant knob).
- Sweep 2: δf_r vs D₀ (0.6–60 µm²/ns) — transport-limited → saturated regimes.
- Sweep 3: Q_i,total vs τ_l for lowest/highest-frequency mode.
- Readout-heating 108-run sweep (D₀ × τ_l × injection × n̄) as a capability demonstration.
- Convergence-check panel (≤3% envelope) as methodological hygiene.
- Sources: `qpsim/experiments/prelim_resonators.py`, `scripts/run_prelim_*.py`,
  `docs/prelim_experiment_simulation_notes.md`, prelim deck slides 20–28.
- Plus: transient pair-breaking photon kick (`validation/transient/photon_kick_response.py`)
  as the time-domain demo, late-time limit checked against the independent Newton steady
  state.
- Plus: minimal listing — "ten lines from `load_material("Al")` to Q_i" — the paper's code
  example. Write this against the services API and make it a doctest/CI-run script so it can
  never rot.

### §7 Performance
- Runtime table for representative solves (0-D steady state, 1D strip transient, M25 T-sweep,
  a full sweep campaign) on stated hardware; memory footprint; scaling vs N_E, N_x.
- Honest statement: pure NumPy/SciPy, single-node; no GPU/MPI. Frame as adequacy-for-purpose
  (all validation figures reproducible on a laptop/desktop in stated wall-clock).

### §8 Limitations and roadmap
Straight from `docs/STATUS.md` + round-3 caveats slide: phonons are a local bath with
acoustic escape to the substrate rather than a spatially transported population;
S21/telegrapher readout layer absent; f_T charge mode not commissioned; Dynes-consistent
transport guarded but unimplemented; strong-drive fold bifurcation awaits arclength
continuation; TVD spectral-flow solver not yet wired into the spatial path (verify current
status at drafting time).

### §9 Conclusion + availability statement

## 5. Figure plan

| # | Content | Generator / source | Status |
|---|---------|--------------------|--------|
| 1 | Pipeline + architecture schematic (JJ source → f(E,x,t) ⇄ n_ph → σ₁/σ₂ → δf_r, Q_i; package layers) | new (TikZ/vector); prelim deck slide 2 as basis | to make |
| 2 | F23 Fig. 3 reproduction: f(E) family over τ_l/τ_PB | `validation/fischer_2023/fig3_*.py` | authenticated qpsim regression; source-honest Round-8 regeneration in progress; no digitized parity |
| 3 | F23 Figs. 5+6 panel: x_qp sweeps + gap observable | `fig5_paper.py`, `fig6_paper.py` | Eq. 47/Eq. 53 helpers and overlays implemented; state-bound current-source regeneration in progress; no digitized parity |
| 4 | MKID chain: Q_i(T_B) and Q_i(P_read) | `fig7_with_drive.py`, `figs_9_13_qi_vs_pread.py` | model-only framing OK for this paper |
| 5 | F24 PB-photon f(E) + x_qp(T_B) | `validation/fischer_2024/*.py` | hardened current qpsim artifacts complete; analytic paper overlays remain absent |
| 6 | M25 junction: T̄(g^ph) Lambert-W + μ_α(T) + Γ_P(T) | `validation/marchegiani_2025/*.py` | authenticated bundles complete after the Γ̄ normalization fix; broad paper anchors checked manually, with no digitized pointwise parity |
| 7 | One operator benchmark (self-consistent feedback well) | `validation/diffusion_operators/self_consistent_feedback.py` | done (Paper 1 Fig. 5); re-render, don't reuse the exact Paper 1 figure |
| 8 | Al-strip device study: δf_r / Q_i vs (τ_l, D₀) | `scripts/run_prelim_*.py` | rerun on current main; prelim numbers exist |
| 9 | Transient photon kick f(E,t), x_qp(t) → steady state | `validation/transient/photon_kick_response.py` | strict-v3 paired artifact/regression implemented; current-source regeneration in progress |
| 10 | Web UI screenshot (1D strip mode) | `qpsim-ui` | trivial once figures frozen |

Figure style: one shared matplotlib style file for the whole paper; colorblind-safe palette
(webui palette already validated); every figure regenerable by one script each, wired into a
`papers/qpsim/Makefile` like Paper 1's `make verify` pattern.

## 6. Gap-closure work list (before/during drafting)

Blocking for their figures (from `qpsim_validation_plan` status tags + 2026-07-02 review):
1. F23 Fig. 5: Eq. 47 overlay is implemented; finish current-source
   regeneration and numerical qualification.
2. F23 Fig. 6: paper ordinate and Eq. 53 overlay are implemented; finish the
   state-bound parallel campaign and numerical qualification.
3. F23 Fig. 3: threshold-discretization residual (asymptotic-fit extraction ticket); check
   `_paper_envelope.py:64` Airy-argument precedence (review item 4) — it touches this
   figure's dashed overlay.
4. F24 Figs. 5/8: source/runtime artifact hardening is complete; the paper
   analytic overlays remain an explicit separate gap.
5. ~~M25 Figs. 3/4~~ done 2026-07-04, and bigger than expected: the "multi-stability" was a
   Γ̄ = Γ̃/N_CP(R) normalization bug in the density equations (M25 text below Eq. 6); with
   the fix the root is unique. `solve_rate_equation_branch` (continuation driver) shipped,
   real global-QE + renormalized comparison models implemented, all baselines regenerated,
    corrected broad manual anchors documented in STATUS.md. Paper §5 must
    label them as manual anchors rather than a digitized agreement metric.
6. Transient demo: paired strict-v3 regression implemented; preserve its
   provenance/currentness and trajectory/certificate checks.
7. Rerun the prelim sweep campaigns (§6 figures) on current main — prelim numbers predate
   the ×2 recombination fix and the 2026-07 fixes, so all demo numbers must be regenerated.
8. Performance table: benchmark on one stated reference machine.

Non-blocking but wanted for release credibility (from `REVIEW-2026-07-02-code-health.md`):
steady_state branch-collapse guard, M25 cache staleness (frozen=True), nbar_loop NaN
handling, version-string de-dup. A "v0.1.0" tagged release should accompany submission.

## 7. Sequencing

1. **Now → Paper 1 submission**: finish qp-diffusion items 2–10 (separate track, A: desktop).
   In parallel, low-risk Paper 2 prep: this plan, figure-gap tickets, prelim-campaign rerun
   scripts checked against current main.
2. **Phase A (post-submission)**: close gap-closure items 1–6; freeze figure set.
3. **Phase B**: scaffold `papers/qpsim/paper.tex` (match Paper 1's REVTeX/Makefile
   conventions if SciPost/CPC template allows; else venue template); write §2–§4 from the
   docs skeletons; minimal code example as CI-run script.
4. **Phase C**: demonstrations (§6) + performance (§7); write intro/related-work last;
   full-paper review pass (use the nine-review pattern that worked for Paper 1).
5. **Release engineering**: tag v0.1.0, Zenodo DOI (or CPC Program Library deposit), README
   badges, CITATION.cff, docs pass.

## 8. Open decisions (Soren)

1. **Venue**: CPC vs SciPost Physics Codebases (recommendation: CPC; SciPost if community
   visibility outweighs archival-library convention).
2. **Authorship**: solo vs. advisor/collaborators — affects acknowledgments and the
   experimental-overlay question (F23 Fig. 7 experimental points belong to Fischer's data,
   so model-only framing is recommended regardless).
3. **Scope of the operator-benchmark content**: one figure + citation (recommended) vs.
   none (pure citation) — depends on how much Paper 1 and Paper 2 referees overlap.
4. **Web UI prominence**: one figure + subsection (recommended) vs. appendix.
5. **Package name check**: confirm `qpsim` is claimable on PyPI before the paper names it in
   print (there may be a squatter; check early — renaming after submission is painful).
