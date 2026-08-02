# Audit 2026-08-01 — fig7 determinism root cause, fig3 live collapse, M25 + remap fan-out

Executed against main `b92571a` (probes ran on `perf/hot-path-2026-08-01` @ `4c57298`,
whose four touched files are bit-identical in results through the non-chord fixes;
noted per finding where relevant). Follows the queue in `NEXT-AUDIT-BRIEF.md`.
Method note per Soren's direction: findings below rest on direct experiments and
independent re-derivation, not on the (frequently vacuous — see F2) test suite.

## F1 — fig7 "non-determinism" ROOT-CAUSED: BLAS thread-count-dependent rounding,
amplified ~1e11× by the low-occupancy fixed-point chain

Controlled experiment (`fig7` point P=−100 dBm, T_B=0.06 K, NUM_BINS=1701, fresh
process per run, win32/OpenBLAS/numpy 2.4.6):

| config | converged `f` SHA256 (16h) | x_qp |
|---|---|---|
| OPENBLAS_NUM_THREADS=1, run A | `d73d961178b72306` | 4.42474564313547041e-15 |
| OPENBLAS_NUM_THREADS=1, run B | `d73d961178b72306` | (identical) |
| OPENBLAS_NUM_THREADS=4, run A | `02e8dbf8e7bcfc7a` | 4.42452173258215793e-15 |
| OPENBLAS_NUM_THREADS=4, run B | `02e8dbf8e7bcfc7a` | (identical) |
| threads=1, PYTHONHASHSEED=12345 | `d73d961178b72306` | (identical to threads=1) |

- **Fixed thread count ⇒ bitwise deterministic.** There is no true run-to-run
  nondeterminism on a given (platform, BLAS build, thread count).
- **Thread count changes the answer**: Δx_qp/x_qp ≈ 5.1e-5 — eleven orders of
  magnitude above ulp. The dense LU solves are the thread-sensitive step; the
  near-cancelling gain/loss balance at f ~ 1e-16 amplifies reduction-order noise.
- **Hash randomization ruled out.** Anderson ruled out *at this point* (depth-0
  runs are bit-identical to depth-3 — it never engages here), though its
  `lstsq(rcond=1e-10)` truncation on near-collinear history remains a plausible
  *additional* discrete amplifier at points where it does engage.
- CI implication: ubuntu runners vary in core count AND OpenBLAS per-arch kernels,
  so "identical code, different results across CI runs" is exactly this mechanism.
  The 3.13/3.14 leg disagreement is the same class (different wheel builds).
- Remedy options: pin `OPENBLAS_NUM_THREADS=1` in the slow-CI step for cross-run
  reproducibility on a given runner arch (cheap; does NOT fix cross-arch variance);
  accept + keep the widened rtol=1e-3/atol=1e-10 gates (already landed); treat the
  low-occupancy amplification itself as inherent conditioning of the observable.
  Probe harness: session scratchpad `fig7_probe.py` (trivially re-creatable: solve
  one point, hash `f.tobytes()`).

## F2 — fig3 τ_l/τ_0^PB = 10 curve is a LIVE solver collapse, and the whole fig3
regression gate is vacuous (worse than the brief's "one dead column")

Ran the real `fig3_paper.run()` (no cache) and diffed against the pinned baseline
(`validation/baselines/ph0_constant/fischer_fig3_paper.csv`, pinned once 2026-05-26,
never regenerated):

| ratio | max f (current) | max f (baseline) | max rel diff | passes atol=1e-6 gate |
|---|---|---|---|---|
| 0   | 3.63e-10 | 3.65e-10 | 4.5e-3 | yes |
| 0.1 | 1.35e-09 | 1.55e-09 | **6.1e-1** | yes |
| 1   | 1.52e-08 | 1.53e-08 | **4.3e-1** | yes |
| 10  | **0.0 (all 1620 bins)** | 0.0 (all 1620 bins) | — | yes |

- **F2a (solver bug, live today):** the r=10 target solve
  (`fig3_solve._solve_coupled_newton`) returns f ≡ exactly 0.0 — unphysical
  (strongest phonon trapping must exceed the r=1 curve; gap-edge sequence
  3.6e-10 → 1.4e-09 → 1.5e-08 → **0.0**). Mechanism: f ≡ 0 is a near-root
  (at T_B=0.1 K, thermal n_ph(2Δ) ~ e^−42, so every rate ≈ 0 and an absolute
  residual gate at `coupled_newton_tol=1e-10` certifies the collapsed state), and
  the finite-difference Jacobian uses `coupled_newton_fd_step=1e-8` — the same
  order as the f-values being perturbed (r=5 seed max f ~ 1e-8), i.e. ~100%
  relative perturbation → garbage Newton directions → collapse to the trivial
  branch. The paper-target curve has NEVER been reproduced; the shipped baseline
  pinned the collapse on day one. Candidate fixes (untested): scale fd_step
  relative to the seed (`fd_step * max(|f|)`), add a relative/source-scaled
  residual criterion (cf. the M25 gate), or finish the ladder with damped Picard
  instead of coupled Newton and reserve coupled Newton for polish.
- **F2b (vacuous gate):** every ratio column is compared at `rtol=0, atol=1e-6`
  while the physical signal is ≤ 1.5e-8 — 66× headroom. 43–61% relative drift
  (the 1c5af1a 2×-recomb correction era → now) passed silently, as would
  all-zeros in every column. The `f_FD` column's atol=1e-14 gate is the only live
  assertion in the test. Fix: after F2a, regenerate the baseline on corrected
  physics and gate at a relative tolerance with a floor well below signal
  (e.g. rtol=1e-3, atol=1e-12), per the platform-stamp convention.

## F3 — M25 rate-equation acceptance layer (fresh-eyes agent review)

No silently-wrong-accepted-state path found. Strongest clean result: the fully
assembled residual at exact thermal equilibrium is machine-zero (row-relative
~1e-16) in BOTH kinematic cases ω_10 ≷ ω_LR, with per-channel detailed-balance
ratios = 1 to 1e-15 — jointly confirming the S30–S36 closed forms, ξ partition,
τ_R/τ_E balance, g_pn erf/erfc split, δ conversions, and the Γ̄ chokepoint; the
elliptic S56 spectral densities match direct numerical integration to ≥8 digits.
Findings (all loud-failure or out-of-envelope; file:line in the agent report,
summarized):
1. Sub-mK T crashes `rate_equation_coefficients` via exp·K₁ overflow/underflow and
   erfc underflow (reproduced; T ≲ 1.6 mK at ω_LR=1.15 K). Loud, and fixable with
   `scipy.special.k1e`/log-erfc if the range is ever needed. Low.
2. `_relocate_root_1d` misses an exact-zero residual at the last/only sample
   (probability ~0). Negligible.
3. `thermal_seed=None` + corrector failure at `T_grid[-1]` + no exchange hint
   hard-fails a sweep whose photon pass is complete (availability, loud; in-repo
   caller always passes both). Low-moderate.
4. Bisection ladder's per-sub-step continuity floor (0.25·jump_tol × up to 2⁸
   sub-steps) can track a branch exchange as continuous — cannot bite on the
   unique-root M25 family; design note for future multi-branch families.
5. `min_residual` ranking uses a raw ∞-norm across rows with ~12-orders-different
   scales (noise-ranks the qubit row when several candidates pass). Design wart.
6. Doc nits: Fig-3 photon input is Γ^ph_00 not Γ_01^ph (rate_equation.py:68);
   `omega_nu_kelvin` docstring claims a >Δ_L+Δ_R check that isn't enforced
   (below-threshold silently zeroes photon rates); `analytic_low_T_seed`
   small-asymmetry branch omits a p_0 factor (seed-only).

## F4 — Moving-gap remap + BCS quadrature + solve_gap (fresh-eyes agent review)

The PR #3 frozen-ξ remap is solid: bitwise-identical to an independent O(N²)
overlap in both directions, exact conservation (round-trip 1.6e-16), correct
characteristic map (√(ξ²+Δ'²) verified), exact handling of cells crossing the
active window, loud failure on saturation/boundary escape. Exact BCS weights
correct including band splitting (splits reconstruct to 0.0 difference; total =
√(E_max²−Δ²) to machine precision). Findings:
1. **No dead-band between the remap trigger (1e-14 μeV) and `solve_gap`'s own
   root resolution (brentq xtol ≈ 2e-4 μeV)** — during slow-relaxation phases,
   xtol-scale root noise fires real remaps and each conservative remap smears the
   gap edge; measured 8.5e-4 absolute occupation movement per 1000 events
   (NE=300) with conservation audits blind to it. Percent-level edge distortion
   reachable in 1e5–1e6-step transients. THE one to act on: dead-band at the
   xtol scale (or reuse `rebuild_tolerance`). Exact steady states are safe
   (deterministic brentq short-circuits).
2. `bcs_dos_cell_weights` truncation asymmetry: missing lower band support raises,
   missing upper support silently clamps (`upper_bound=1e9` returns bit-identical
   weights to default). Only the M25 band-moment path passes bounds through.
3. σ₁ gap-edge quadrature is correct but first-order at the edge: 11% off at
   NE=200, 2.0% at NE=800, 0.06% at NE=3200 (vs scipy-quad reference; Δ=200,
   ω₀=40). Resolution requirement worth documenting, not a bug. σ₂ excellent.
4. Stale doc/dead export: `spectral_flow_tvd.py:90` claims `apply_gap_update`
   uses the TVD path; nothing calls it. Misleading to auditors.

## Not examined this round

Spatial/webui robustness (brief's fourth churn area); fig6 turnover reproduction
(unchanged: needs a fresh approach, not the merged branch); the G1 midpoint-vs-
exact-weights measure inconsistency (known, deferred, baseline-moving).
