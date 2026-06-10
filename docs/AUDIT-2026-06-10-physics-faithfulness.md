# Audit record: qpsim physics faithfulness vs paper @ f0cfdd2 (2026-06-10)

Executes `docs/HANDOFF-2026-06-10-qpsim-audit.md`. Spec: the frozen paper
(`~/Documents/qp-diffusion-paper/paper.tex` @ f0cfdd2; only a 7-line
citation tweak since, b6f5b1b). Session baseline at start: 611 pass /
2 known failures (the foundation's rate_equation WIP — unchanged, still
the only failures at end). Gate at end: **625 pass / same 2 known
failures**; the three `validation/diffusion_operators/` benchmark tests
(inside the gate) unchanged — no paper-figure regeneration triggered.

## Per-module verdicts (audit order)

1. **`physics/spectral.py` — CLEAN.** N₁ = E/√(E²−Δ²), N₂ = Δ/√(E²−Δ²)
   (so N₁²−N₂² = 1 above the gap exactly), both zero sub-gap. Dynes
   branch Re[(E−iΓ)/√((E−iΓ)²−Δ²)] is positive-real for all E > 0
   (verified analytically; the max(0,·) clamp is inert there). K± =
   1 ± Δ²/(EE′) match the Kaplan occupation-form coherence factors.
   Legacy D(E) = D₀√(1−(Δ/E)²) = D₀/N₁ = the A1 uniform-gap rate.
2. **`transport/diffusion/base.py` — CLEAN.** (p,q) family table matches
   paper Table III exactly: A1(1,0) default, A1P(1,2), A2(2,2), C(0,−1),
   B(0,−2). `flux_weight` implements 𝒟_L = 1 above / 0 below the local
   edge at every q via the sub-gap zero guard.
3. **`backends/t3_spatial_1d.py` — CLEAN.** Conservative CN update on
   u = N₁^p f; harmonic face weights vanish against sub-gap neighbours
   (the explicit zero-flux edge face of paper §V); KL faces carry exactly
   𝒲_L = N₁N₁′ − N₂N₂′ (eq:scalar_BC_energy) as G_N·𝒲_L/dx with current
   continuity and an f jump. The post-solve f∈[0,1] clip never bites in
   the audited regimes (edge-packet fixture: conservation 1e-11 over 200
   steps, zero sub-edge leakage).
4. **`solvers/spectral_flow_tvd.py` — core CLEAN, one wrapper fix.** The
   TVD+SSPRK(2,2) advection of ∂_t u + ∂_E[(Δ/E)Δ̇ u] = 0 conserves
   Σu·dE to machine precision and reproduces DOS continuity
   (eq:dos_continuity) to scheme order (bulk |f−1| ≈ 2e-4 at NE=800,
   first-order converging). **FIXED:** `apply_gap_update` passed the
   *pre-step* `active_mask`, which zeroed the legitimate spectral inflow
   into newly opened bins whenever Δ̇ < 0 (eq:full_kinetic_conservative
   carries density below the old edge); measured as a strict worsening of
   the inflow band with no benefit in either ramp direction. Two
   documented (not fixed) finite-domain caveats now in the docstring:
   zero-flux at E_max distorts the top ~|ΔΔ|(Δ/E_max)/dE cells, and
   grids with `energy_min_factor ≥ 1` cannot represent a falling gap's
   inflow band.
5. **`collisions/` + `physics/kernels.py` — one real physics error,
   FIXED (lead 1).** The QP-equation recombination loss and
   pair-breaking gain carried a legacy 2× "pair convention"
   (`phonon.py:537–538` and mirrors). Adjudication, all agreeing:
   - Machine check: qpsim loss = **2.000000 ×** a same-grid raw-formula
     quadrature of Kaplan Eq. (8) at T ∈ {0.10, 0.15, 0.20} K.
   - Kaplan's canonical analytic τ_r(Δ,T) derives from the *un*-doubled
     integral (closed-form check).
   - F&C 2023's printed envelope (Eqs. 47/48/E2 as transcribed 1:1 in
     `validation/fischer_2023/_paper_envelope.py`) satisfies
     R̄·n_th = 1/τ_r^Kaplan at τ̄ = τ₀ — Kaplan-normalized, no doubling.
   - The paper's bridge (eq:J1_occ_bridge): ∂_t f = I_occ with
     Kaplan-form kernels; the −2N₁ is exhausted by f_L = 1−2f.
   Why it survived: detailed balance and *thermal-dominated* steady
   states are exactly blind to a symmetric doubling (√(2G_T/2R) =
   √(G_T/R)); the drive-dominated regime where it shows is precisely
   where the Picard/Newton solvers currently fail (the fig6 sweep's
   nan/collapse tail). Removed at: `phonon.py` rates + `J_fn` Jacobian,
   `t3_spatial_1d.py` inlined collisions, `newton_steady_state.py`
   analytic Jacobian. `coupled_newton` routes through the shared
   helpers and needed no change (FD-vs-analytic Jacobian tests stayed
   green).
   - Kernel-builder prefactors reconciled while there: QP-side
     K₀ʳ/K₀ˢ = Kaplan ω²K±/(τ₀(k_BT_c)³) (no factor); phonon-side
     Eq. 12 kernels (2K⁻ and K⁺ over πΔτ₀^PB) are a *separate, correct*
     normalization (full-range integral defined into τ₀^PB) — untouched,
     still opt-in (lead 4 stands: default NOT flipped).
   - **Lead 3 (τ_l = 0) reconciled, docs only:** `phonon_steady_state`'s
     `tau_l = 0.0` = no-substrate sentinel (τ_l → ∞ physics limit);
     bath-pinned (Fischer τ_l → 0) is spelled `phonon_escape_time=None`
     / `use_thermal_phonons=True`. All three docstrings now cross-warn.
6. **`observables/density.py` — CLEAN.** n_qp = 4ρ_F∫N₁f dE with ρ_F
   documented single-spin (spec item 6); the half-Fischer x_qp
   convention is internally consistent and explicitly flagged.

Lead 2 (spatial runner dropping nondefault fields) fixed with
`dataclasses.replace(state, f=f_mid)`.

## Measured impact of the normalization fix

- Sub-gap-photon-driven observables (fischer_2023 figs 3/5/6/7 regime):
  x_qp shift ratio **1.0000** (thermally anchored; sub-gap drive creates
  no QPs) → tracked fischer_2023 baselines remain valid.
- Pair-breaking-driven observables (fischer_2024 figs): x_qp rises by
  **×1.41 → ×1.18** from recombination-limited to Pauli-compressed
  drive (the √2 quadratic-recombination signature). The four
  fischer_2024 baselines were regenerated post-fix and their slow pin
  tests pass; entries shifted by exactly the predicted factors, thermal
  entries identical.
- M25 track untouched: its rates come from analytic
  `rate_equation_coefficients` formulas; the 2 known failures carry
  byte-identical values before/after.

## New analytic fixtures (15 tests)

- `tests/solvers/test_spectral_flow_fixtures.py` — frozen-shell
  exactness f = G(ξ) under falling/rising ramps (+ resolution
  convergence, edge-cell-scaled mass bound) and discrete DOS continuity
  (f ≡ 1 invariance both directions + single-step N₁→N₁′ check, exact
  full-grid conservation).
- `tests/backends/test_t3_spatial_1d.py` — KL weight fixtures
  (matched-gap 𝒲_L = 1, normal-contact 𝒲_L = N₁, sub-gap closure,
  operator-level extraction of the face conductance) and the gap-edge
  packet fixture (conservation + zero leakage past the local edge).
- `tests/collisions/test_phonon.py` — Kaplan Eq. (8) normalization
  guards: same-grid raw-formula identity for loss and gain (rtol 1e-12;
  a factor-2 regression reads as ratio 2) and a continuum ξ-substituted
  quadrature check at the gap edge.

## Other repairs

- `validation/fischer_2024/test_fig8_paper.py` had a latent
  always-fail (lossy %g header serialization vs exact float equality on
  the derived drives) present since its introduction and masked by the
  slow marker; now an allclose at rtol 1e-12.
- The 14 untracked `validation/baselines/` artifacts (spot-audit session
  output: `*_fast`, `*_partial_postfix`, `*_paper_direct` A/B scratch
  and unwired `fischer_fig{3,5,6}_qpsim_native` exports) were deleted:
  all pre-fix physics, no in-repo generator/consumer, canonical pins
  tracked separately.

## Merge decision (secondary task 1): NOT YET — blocked by the other track

`a1-diffusion-operators` is 44 commits ahead of `main`, zero behind
(fast-forward, no conflicts). Everything this audit track owns is
green. But the branch's 2 known test failures
(`test_m25_junction::test_fig3a_quantitative_match`,
`test_rate_equation::test_accept_lm_convergence_bypasses_residual_check`)
were verified to **pass on main** (throwaway worktree, 2026-06-10):
they are regressions introduced by the branch's in-flight 412-line
`rate_equation.py` WIP — the separate rate_equation track of handoff
secondary task 2, not pre-existing landscape. Merging now would redden
main on tests it currently passes. Merge once that track lands its
fix-or-baseline-regen; no other blocker remains.

## Standing caveats / out of scope

- Spectral-flow top-boundary and gap-anchored-grid caveats (documented
  in `advect_spectral_flow`): production tails there are thermally
  empty; gap-dynamics studies should build grids with sub-gap room.
- Strong-drive solver fragility (fig6 sweep nan/collapse above
  n̄ ~ 4e6 at 0.1 K) predates this audit and now *matters more*, since
  that is the regime where absolute recombination normalization is
  observable. Separate track.
- Phonon-side default (lead 4) deliberately not flipped; flipping
  remains a separately commissioned decision with baseline regen.
- f_T (charge) module: not commissioned; spec recorded in the handoff.
