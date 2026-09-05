# Handoff: qpsim physics-faithfulness audit (2026-06-10)

**Mission:** the paper's physics is now settled, machine-checked, and externally
reviewed (four GPT rounds + one full independent review, all adjudicated). The
next phase is the simulator: audit `qpsim` line-by-line against the frozen
physics spec, fix what disagrees, and add the analytic regression fixtures the
paper now makes available. Read this whole file before touching code.

**Context:** Both 2026-06-09 handoffs are FULLY EXECUTED and live in
`archive/` — do not re-run anything in them. The settled-conventions list is
`~/Documents/qp-diffusion-paper/archive/HANDOFF-2026-06-09-post-errata.md` §non-negotiables (with its
2026-06-10 correction about Belzig Eq. (49)). The paper's external-review
record is in the qp-diffusion-paper project's Claude memory
(`qp-paper-review-2026-06-10.md`).

## Worksites, builds, gates

- **Spec (read-only for this track):** `~/Documents/qp-diffusion-paper/paper.tex`
  at `f0cfdd2` — 74 pp, zero errors, zero undefined refs, no oversized floats.
  Build: `make`. Symbolic ground truth: `make verify` (all six `verify_*.py`
  must come back clean; `verify_gA_convention.py` is the immutable regression
  baseline — never edit it). **Table II (`tab:channel_dictionary`) is the
  one-page version of the spec** — every kinetic coefficient, its value, and
  which scalar mode it dresses.
- **Code (the worksite):** `~/Developer/qpsim`, branch `a1-diffusion-operators`,
  HEAD `85716bd` (docs-only commits atop code HEAD `8315c5f`). Tests: `.venv/bin/python -m pytest tests/
  validation/diffusion_operators/ -q`. Last recorded status: 594 pass, 2 known
  failures (`test_m25_junction::test_fig3a_quantitative_match`,
  `test_rate_equation::test_accept_lm_convergence...`) — the foundation's
  in-flight rate_equation WIP, **not this track's problem**; re-establish the
  baseline at session start before changing anything.
- **Gate after ANY qpsim physics change:** full pytest suite; the three
  benchmark scripts in `validation/diffusion_operators/` still match their
  analytic spectra; if a benchmark's output changes, regenerate the figures,
  re-commit them in the paper repo (`figures/`), and rebuild the paper green.

## The spec (settled — audit against this, do not re-derive)

1. **Operator:** dirty-limit default is A1 = (p,q) = (1,0):
   `∂_t(N₁f) = ∇·(D_N 𝒟_L ∇f)` with 𝒟_L = 1 above the local gap edge, 0
   below; conserved density N₁f; uniform-gap rate D_N/N₁(E)
   (paper eq:LD_p1q0, eq:usadel_LD_p1q0). Family: L_{p,q} = N₁^{-p}∇·(D₀N₁^q∇f);
   labels A1(1,0), A1P(1,2), A2(2,2), B(0,−2), C(0,−1) per paper Table III.
   A1 ≡ C exactly at uniform gap.
2. **Time-dependent gap:** conservative spectral flow
   `∂_t(N₁f) + ∂_E[(Δ/E)Δ̇ N₁f] = …` (eq:full_kinetic_conservative), with the
   DOS-continuity identity ∂_tN₁ + ∂_E[(Δ/E)Δ̇N₁] = 0 (eq:dos_continuity)
   underwriting exact discrete fixtures (below).
3. **Gap edge (new in the paper — §V "The local gap edge"):** at E = Δ(x,t)
   the operator is read in weak/flux form: zero-flux face for the energy mode
   (diffusive Andreev retroreflection), 1/N₁ rate → 0 at the edge, the charge
   channel continues via 𝒟_T ≠ 0. Implement as an explicit zero-flux face or
   Dynes-regularized 𝒟_L, N₁ — check what qpsim actually does at cells where
   N₁ → 0/∞ and at the moving edge in energy space.
4. **KL interface:** energy weight 𝒲_L = N₁N₁′ − N₂N₂′ (eq:scalar_BC_energy;
   = 1 at matched gaps; → N₁ at a normal contact), mobility G_N = (R_bA)⁻¹,
   **current** continuous (Robin), f discontinuous. The DOS product N₁N₁′ is
   the CHARGE-channel weight and must not appear in the energy mode (it carries
   the SIS matched-gap singularity).
5. **Collisions:** occupation-form kernels (Kaplan: partner DOS only) relate to
   the trace-form integral by J₁ = −2N₁ I_occ (eq:J1_occ_bridge) — the external
   N₁ weight is the classic normalization trap. Detailed balance to
   f₀ = 1/(e^{E/T}+1); recombination threshold 2Δ(t) instantaneous.
6. **Observables:** n_qp = 4N₀∫_Δ^∞ N₁ f dE with N₀ the SINGLE-SPIN normal
   DOS (4 = 2 spin × 2 branches).
7. **Benchmarks (already implemented; keep matching):** uniform-gap rates trace
   N₁^{q−p} with A1/C coincident; ramp drift v = D₀ q N₁^{q−p−1} ∂ₓN₁ measured
   on the center of mass of the CONSERVED density N₁^p f (the f-center of A1
   would drift via the rate prefactor — that is not a bug); closed-box
   interface relaxation distinguishes conserved densities N₁f vs N₁²f.
8. **Standing limit checks:** 𝒟_L = 0 sub-gap (energy blocked); 𝒟_T ≠ 0
   sub-gap (Andreev); dirty κ_s reduces to the gapped-normal-integrand form
   (undressed flux, lower limit Δ).

## Primary task: the audit

Module-by-module comparison against the spec, in this order (highest physics
risk first):

1. `qpsim/physics/spectral.py` — N₁, N₂ definitions, branch/regularization,
   N₁² − N₂² = 1 above gap, sub-gap behavior.
2. `qpsim/transport/diffusion/base.py` — the (p,q) family (its docstring
   already mirrors the paper; verify the code does too), edge/blocked-cell
   handling vs spec item 3, default = A1.
3. `qpsim/backends/spatial.py` + `diffusion.py` — flux discretization
   (which average of the flux coefficient at faces — must conserve ∫N₁f and
   reduce to the zero-flux face at the edge), KL face vs spec item 4.
4. `qpsim/solvers/spectral_flow_tvd.py` (+ ssprk/crank_nicolson as used) —
   conservative spectral-flow step vs spec item 2.
5. `qpsim/collisions/` + `qpsim/physics/kernels.py` — normalization vs spec
   item 5, thresholds, detailed balance.
6. `qpsim/observables/` — densities and prefactors vs spec item 6.

**Protocol:** read code → compare to the labeled paper equation → when in
doubt, machine-check (sympy in the paper repo's `.venv`, or a small numeric
experiment in qpsim) → only then edit → full gate. Cite the paper equation
label in any fix's commit message. Do not trust pre-errata comments anywhere:
anything claiming A1 = (1,2), D_L = N₁², or the −(g^R)† conjugation is stale;
the corrected column of `verify_gA_convention.py` and paper Table II are the
only source of truth for expected values.

**New analytic fixtures worth adding as tests** (cheap, exact, currently
unexploited):
- Frozen-shell exactness: with collisions off and uniform Δ(t), f = G(ξ),
  ξ = √(E²−Δ²(t)), is an exact solution — evolve a smooth G under the
  spectral-flow step and assert per-shell invariance and exact conservation of
  ∫N₁f dE (including the moving lower limit).
- Discrete DOS continuity: the spectral-flow step applied to f ≡ 1 must
  reproduce ∂_tN₁ + ∂_E(u_EN₁) = 0 to scheme order.
- KL matched-gap limit: 𝒲_L → 1 exactly; normal-contact limit 𝒲_L → N₁.
- Edge fixture: a packet pushed against a gap ramp must conserve ∫N₁f to
  round-off with zero leakage past the local edge.

## Pre-adjudicated leads (external GPT spot-audit, 2026-06-10, re-verified)

A second external model spot-audited qpsim against the paper + thesis Ch. 4
the day this handoff was written. Every claim below was re-verified at the
cited line before being recorded here. Its clean corroborations — diffusion
taxonomy + A1 default (`transport/diffusion/base.py:77,102`), conservative
N₁^p f update (`backends/spatial.py:170`), KL energy weight (`:256`),
e-ph/photon coherence factors, gap equation, spectral-flow conservation, plus
a green 126-test targeted subset — raise confidence but replace nothing; the
audit order above stands.

1. **Recombination factor 2 — fold into audit step 5 (in scope).**
   `collisions/phonon.py:537` applies `loss_rate += 2.0·(K_r0·N_emit)@(ρ f dE)`
   with the gain mirrored at `:538` ("factor-2 pair convention", `:570`).
   Adjudicate against Kaplan and eq:J1_occ_bridge. Trap: detailed balance is
   BLIND to this factor (it multiplies gain and loss symmetrically) — check
   the absolute decay rate against Kaplan's analytic τ_r(E) (or a
   Rothwarf–Taylor R fixture), and reconcile the kernel-builder prefactors
   while there.
2. **Spatial runner silently drops nondefault physics — quick fix.**
   `FinitePhononSpatialRunner` (`scripts/run_prelim_spatial_finite_phonon_one.py:196`)
   rebuilds `SpatialState` without `diffusion_model` / `gap_profile` /
   `interface_conductance` (defaulted fields, `backends/spatial.py:125`),
   resetting them mid-step. Harmless for today's uniform default-A1 runs,
   wrong for any nondefault run. Fix: `dataclasses.replace(state, f=f_mid)`.
   Full gate applies.
3. **τ_l = 0 means opposite things in two modules — reconcile during step 5.**
   `backends/diffusion.py:159` (`use_thermal_phonons` doc): τ_l = 0 is
   Fischer's instantaneous-thermalization limit, n_ph pinned at the bath.
   `phonon_models/local.py:12`: τ_l = 0 is a sentinel for NO substrate
   coupling, n_ph = −a_ph/b_ph — the opposite (τ_l → ∞) limit of the same
   escape term. Each module is internally consistent; the API is a trap.
   Reconcile naming/docstrings (e.g. `tau_l=None` for the no-bath sentinel);
   do not change numerics silently.
4. **Phonon-side kernel default — decision item, OUTSIDE the paper spec.**
   Phonon-equation rates default to the legacy QP-side kernels; the F&C 2023
   Eq. 12 phonon-side form exists but is opt-in (`use_phonon_side_kernel=False`,
   `backends/diffusion.py:101,174`; `collisions/phonon.py:261`), and the
   prelim finite-phonon script never opts in (its `_phonon_escape_step` passes
   no phonon-side kernels). The paper does not govern phonon dynamics (single
   passing mention, paper.tex:370) — this is thesis-Ch.4/F&C faithfulness, and
   the False default is documented as deliberate ("legacy bit-for-bit").
   Do NOT flip it mid-audit; flipping is a separately commissioned decision
   with baseline regeneration.

## Secondary tasks

1. **Merge decision** for `a1-diffusion-operators` → main (carry-over; do after
   the audit is clean). Note the untracked baseline artifacts: when this
   handoff was written there were two
   (`validation/baselines/constant/fischer_fig3_qpsim_native.{csv,pdf}`);
   by end of 2026-06-10 there are FOURTEEN — fig3+fig5 in `constant/` and
   ten `fischer_fig6_*` variants in `kaplan/` (the
   `paper_direct`/`paper_fast`/`partial_postfix` names look like
   comparison-run output of the external spot-audit session). Decide
   commit-or-clean for the lot, don't leave them floating.
2. **M25 rate_equation fix-or-baseline-regen** (the 2 known failures; see the
   qpsim project memory `project_qpsim_a1_diffusion.md`). Separate track —
   touch only if asked.

## Landmines

- The 2 failing tests are the foundation's WIP — verify they are STILL the
  only failures at session start; if new ones appear, that's this track.
- The benchmark figures are wired into the paper (§VII.5): any change to
  `validation/diffusion_operators/` outputs obligates regenerating
  `figures/{uniform_gap_packet,gap_gradient_drift,interface_trap}.pdf` in the
  paper repo, committing, and rebuilding the paper green.
- Per-repo `.venv`s: qpsim's for pytest, the paper repo's for `make verify`
  (system python has no sympy).
- `paperNotes.bib` and `paper*.pdf` are build artifacts/ignored in the paper
  repo; don't re-add them.
- The thesis directory is not git and iCloud-adjacent: archive, never delete;
  files can be evicted (hang at 0% CPU → `brctl download`).
- The old "numerics LAST" directive is obsolete — numerics IS this track.
