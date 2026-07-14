# Audit 2026-07-13 — 12-agent fan-out + adversarial verification

> **⚠️ CORRECTION (2026-07-13, same day): this audit's "essentially clean" verdict was
> overturned.** An external GPT audit filed ~20 findings; independent reproduction
> confirmed **19 as real** (4 medium + 15 low; still no blocker/high, no corrupted
> published result, no default-path crash). This fan-out found only 2 of them because it
> was structurally blind to cross-operator inconsistency (G1), robustness/DoS, and
> validation-honesty issues, and never ran the slow gate. **Read
> `AUDIT-2026-07-13-gpt-reconciliation.md` for the corrected picture** — it supersedes the
> "strong health / clean audit" framing below. The physics-core verifications recorded here
> (detailed balance, Jacobians, conservation) remain valid; the *overall verdict* does not.

**Method.** Twelve independent domain auditors (Fable 5), one per subsystem,
each read its assigned files in full and verified hypotheses with the repo venv
Python. Every filed finding was then handed to a separate *adversarial* verifier
told to refute it and required to reproduce a concrete failing input (running
code) before confirming. A final synthesis pass deduplicated and ranked. The
documented false-positive/known-gap list from `NEXT-AUDIT-BRIEF.md` was supplied
to every agent so settled non-bugs were not re-chased.

- Branch: `codex/reaudit-fixes-2026-07-13` (PR #2), not merged to main.
- Baseline at audit time: fast `pytest` 1060 pass, `ruff` clean, `mypy` strict clean.
- Scope: all of `qpsim/` (17.1k LOC non-test), `tests/`, `validation/`. Fast
  tests only; the slow validation gate and the paper physics were **not** run.
- Result: **4 findings filed → 2 confirmed (both Low, both latent) → 2 refuted.**
  Ten of twelve subsystems returned zero findings.

## Executive summary

The codebase is in strong health. No Blocker, High, or Medium correctness defect
survived verification. Both survivors are latent quality gaps — one numerical
branch no shipped grid triggers, one regression assertion too loose to test the
regime it nominally covers. Neither corrupts a validated/published result,
crashes a supported path, or is reachable from any shipped in-repo caller. The
pinned Fischer/Marchegiani baselines and all fast tests remain trustworthy for
what they actually assert.

## Confirmed findings

### L1 — First-cell gap clamp drops the below-gap linear offset (Low, latent)

`qpsim/observables/gap_suppression.py:143`. `gap_integral_from_distribution_direct`
reconstructs `f` piecewise-linearly between left-edge nodes and integrates the
singular BCS measure `2/sqrt(E²−Δ²)` analytically per cell. When the first left
edge lies below the gap, `x_lo[0]` is clamped to 0, but the constant term
(line 148) still uses `vals[0]` (the node value defined at the below-gap
position) and the linear term (lines 152–160) anchors its slope at the clamped
`x=0`. The reconstruction therefore omits the `slope·(gap−edges[0])` constant
offset over the singular, highest-weight first cell — an error that does **not**
vanish under grid refinement.

- **Reproduced (venv):** gap=180, `E=180+4·arange(500)` (so edges[0]=178<gap),
  `f=0.1·exp(−(E−gap)/50)`. First-cell contribution code=0.030550 vs
  scipy.quad reference on the true reconstruction 0.029405 (~3.9% cell error),
  propagating to +0.89% on the total integral; steeper near-gap `f` → ~2–3%.
  `Δ[f]=Δ₀·exp(−I)` and the suppression ratio inherit the bias.
- **Why Low / latent:** every in-repo caller of the `*_from_distribution_direct`
  family uses `build_energy_grid` with `energy_min_factor ≥ 1.0`
  (`fig6_solve.py` = 1.0 → edges[0]=gap exactly, clamp is a no-op;
  `self_consistent_feedback.py` = 1.02 → edges[0]>gap). No shipped grid puts the
  first left edge below the gap, so the branch is inert and the pinned Fischer
  Fig.6 baseline is provably unaffected (verified ratio=1.000000). It is a real
  latent input-validation gap in a public API, not an active defect.
- **Fix:** on clamping, shift the constant to the reconstructed lower-limit value
  `vals_eff[i] = vals[i] + slope_i·(gap − edges[i])` (keep the linear term
  anchored at x=0); or follow the `bcs_dos_cell_weights` contract and **fail loud**
  when `edges[0] < gap − tol`, documenting that the first left edge must
  coincide with the gap.

### L2 — Vacuous ratio-10 regression assertion over an all-zero baseline (Low, already known)

`validation/baselines/ph0_constant/fischer_fig3_paper.csv` (asserted in
`validation/fischer_2023/test_fig3_paper.py:93-100`). The `f_ratio_10` column is
exactly 0.0 across all 1620 bins; the per-ratio comparison uses `atol=1e-6`
against a physical signal of ~1e-8, so any live output passes — the hardest
strong-bottleneck regime has zero effective regression coverage. Root cause: the
generator `fig3_solve.py:282-300` pins `_solve_coupled_newton` output with no
convergence/positivity guard (it only checks the ratio *keys* are present).

- **Reproduced (venv):** per-column max|·| = f_FD 8.00e-10, f_ratio_0 3.64e-10,
  f_ratio_0.1 1.54e-9, f_ratio_1 1.53e-8, **f_ratio_10 = 0.0 exactly**. This is
  the only all-zero column across all 18 baseline CSVs.
- **Status:** already listed in `NEXT-AUDIT-BRIEF.md` KNOWN GAPS — **confirmed and
  root-caused, not new.** Whether the *live* ratio-10 solver actually returns
  all-zeros (a broken solver) vs a pinning artifact remains **unverified**: it
  requires the several-minute slow solve, deliberately not run here.
- **Fix:** regenerate with a converged non-zero ratio-10 `f` (add a
  convergence+positivity guard in `fig3_solve.solve` before pinning); switch the
  per-ratio compare to `rtol` (e.g. `rtol=1e-6, atol=1e-30`) so it scales with the
  ~1e-8 signal; add a fast tripwire asserting `max(f_ratio_10) > 0`.

## Refuted findings (recorded so the next audit does not re-chase them)

### R1 — `build_variable_diffusion_laplacian` "silently drops a boundary condition" — REFUTED

`qpsim/grid/spatial_grid.py:293`. Claim: the variable-diffusion Laplacian builder
lacks the `missing_edges` completeness check its constant-D sibling
(`build_laplacian_with_boundaries`, lines 211-216) enforces, so an unassigned
edge yields an operator that misrepresents the requested physics.

**Verdict: the API asymmetry is real but the correctness claim is false.** The
verifier built the operator twice — with the ghost edge unassigned, and assigned
a Dirichlet BC=5.0 — and got byte-identical operators (`nnz==0` difference). A BC
placed on a face between two *interior* cells is degenerate and is ignored by
**both** builders (the interior branch at line 307 is taken); the constant-D
builder merely fails louder and earlier. No reachable input yields a wrong
Laplacian. Reachability is also weak: neither builder is called by production
code (the 1D backend reimplements its own harmonic-mean weighting). Adding the
symmetric guard is a harmless fail-loud hardening, not a bug fix.

### R2 — `occupation_heatmap` 500s on an all-underflow strip (`vmin>vmax`) — REFUTED

`qpsim/webui/plots.py:182`. Claim: a low-`T_bath`, injection-disabled `spatial_1d`
run leaves `f_final` at an all-underflow thermal seed, making `vmin=1e-12 >
vmax≈1e-300` in the `LogNorm`, which matplotlib rejects — and `runs_plot`
(`server.py:218`) only catches `KeyError`, so it escapes as HTTP 500.

**Verdict: the crash mechanism exists but its only claimed reachable path does
not reach it.** The plotted array is the T3-backend-*evolved* `final.f`, not the
seed. The verifier ran `run_spatial_1d` with injection disabled across
`T_bath ∈ [1e-2 … 1e-6] K` and Al/Nb/extreme-gap materials: the seed does
underflow to ~9.86e-305, but the backend regenerates a stable non-thermal QP
floor of ~1e-217 (T- and material-independent below ~1 mK), orders of magnitude
above the 1e-290 trigger. `np.any(f>1e-290)` is always True, so the safe branch
is taken and the heatmap rendered in every case. The bug fires only for a
hand-fabricated all-underflow array no run produces. (Broadening the `except` in
`runs_plot` to map rendering `ValueError`s to a 4xx is still reasonable defensive
hygiene.)

## Per-subsystem clean verifications (the 10 zero-finding auditors)

These are what each auditor *verified by running code*, not merely read — useful
provenance for the "clean" verdict:

- **M25 rate-equation core** (`services/rate_equation.py`): particle conservation
  in tunneling assembly to −1.9e-16; L→R channel bookkeeping exact; scattering-in
  vs -out signs correct across all three density rows; detailed-balance qubit seed
  reduces to `ee_01/(ee_01+ee_10)`; residual gates reject NaN/inf; all coefficient
  arrays taken as read-only copies (no shared-state mutation).
- **Rate-eq coefficients + drivers** (`rate_equation_coefficients.py`,
  `steady_state.py`, `transient.py`, `nbar_loop.py`): `_S_ph_*` closed forms match
  their double integrals to ~1e-12; `_K_incomplete` matches scipy Bessel to 1e-?;
  detailed balance `γ_ee[0,1]=γ_ee[1,0]·exp(−ω10/T)`; nbar convergence residual on
  the raw pre-relaxation map (under-relaxation can't fake it); transient reaches
  `total_time` exactly; snapshots never extrapolate.
- **T3 diffusion backend + moving-gap remap** (`backends/t3_diffusion.py`):
  frozen-ξ characteristic is the correct collisionless gap-motion law; conserved
  invariant is QP *number* (energy exchanged with condensate) with relative drift
  ~1e-16 for rising/falling gap; DOS-weight consistency assert cannot false-raise;
  escaped-mass reinjection capacity-bounded (f≤1); fail-loud paths intentional.
- **BCS physics** (`physics/*`, `grid/energy_grid.py`): Kaplan `S₊=x·ellipe(1−4/x²)`
  matches direct quad to ~1e-11; Dynes DOS sign convention correct;
  `bcs_dos_cell_weights` singular measure exact to double precision; gap-equation
  cosh substitution algebraically exact, `solve_gap` round-trips to 4e-7;
  coherence factors K± correct including the sub-gap `max(0,·)`.
- **Collision integrals** (`collisions/*`, `phonon_models/*`): thermal detailed
  balance `df/dt` residuals ~1e-33 (phonon), ~1e-14 (source/sink), ~1e-17
  (sub-gap photon), ~1e-25 (grid-aligned pair-breaking); uniform-grid index-shift
  channels match the O(N²) path; ω=0 bin correctly decoupled.
- **Observables** (besides L1): AC conductivity, density, T_eff, Q, frequency
  shift checked; only the `gap_suppression` first-cell clamp filed.
- **Device layer** (`devices/*`): M25 Eqs. 4-6 moment coupling algebraically
  identical to the solver residual `_rate_equation_terms`; qubit rate matrix is a
  proper generator (column sums ≤1.7e-18); parity/level indexing consistent
  end-to-end; device p₁ marginal reproduces the moment solver to 1.6e-16.
- **Solvers** (`solvers/*`): every analytic Newton Jacobian term matched against
  the differentiated collision residual (scattering transpose, recombination
  "no factor of 2" per-QP normalization, sub-gap and pair-breaking photon
  blocks); Anderson Type-II index layout consistent; TVD minmod monotonicity and
  SSP-CFL subcycling correct; ETD `expm1` φ-function accurate at μdt~1e-15;
  SSPRK(2,2) Shu-Osher coefficients correct.
- **Materials / constants** (`materials/*`, `constants.py`, `experiments/*`):
  all three physical constants match CODATA to full precision; `Δ₀/(k_B T_c)` =
  1.770 Al / 1.882 Nb / 1.805 TiN all defensible; resonator frequency↔length not
  transposed (`L·f≈28507 µm·GHz` constant across six resonators).
- **Test-suite / baselines** (besides L2): all 18 baseline CSVs scanned for
  all-zero/constant/NaN columns — only `f_ratio_10` is all-zero; fig8 constant
  columns are an honestly-labeled placeholder (not a masked defect); analytic
  tests each pair a positive assertion with a counter-test.

## Coverage caveats — what this audit could still miss

- **Slow self-pinned validation baselines were not run.** The multi-minute
  Fischer (fig3/fig6) and Marchegiani solves were not executed. In particular,
  whether the live ratio-10 coupled-Newton solver actually returns all-zeros (a
  broken solver) vs a pure pinning artifact (L2) remains unverified.
- **Cross-subsystem integration bugs.** A file-partitioned review sees each
  module in isolation; defects that manifest only across boundaries
  (grid construction → observable → self-consistent feedback loop) can slip through.
- **Paper physics B1–B5 / M1 are orthogonal and out of scope.** `verify_*.py`
  passing means the code matches the manuscript, not that the manuscript is
  correct; that needs a human physicist.

## Suggested additions to `NEXT-AUDIT-BRIEF.md`

- Add L1 to KNOWN GAPS: the `gap_suppression` first-cell clamp looks like a
  medium numerical bug but is unreachable from every shipped grid — do not re-file
  as new.
- Add R1/R2 to FALSE POSITIVES: the `build_variable_diffusion_laplacian`
  completeness-check asymmetry and the `occupation_heatmap` `vmin>vmax` guard both
  look like bugs but were refuted by execution (no reachable wrong-output path).
