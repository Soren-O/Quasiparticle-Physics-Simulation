# Audit 2026-08-02 — fig7 "non-determinism" root-caused (BLAS thread count)

Controlled experiment executed 2026-08-01 on the Einstein2 win32 box
(numpy 2.4.6/OpenBLAS), one cold fig7 point (P = −100 dBm, T_B = 0.06 K,
NUM_BINS = 1701), fresh process per run — two runs per thread count, one
for the hash-seed leg, one host, one BLAS build:

| config | converged `f` SHA-256 (16 hex) | x_qp |
|---|---|---|
| OPENBLAS_NUM_THREADS=1, run A | `d73d961178b72306` | 4.42474564313547041e-15 |
| OPENBLAS_NUM_THREADS=1, run B | `d73d961178b72306` | (bit-identical) |
| OPENBLAS_NUM_THREADS=4, run A | `02e8dbf8e7bcfc7a` | 4.42452173258215793e-15 |
| OPENBLAS_NUM_THREADS=4, run B | `02e8dbf8e7bcfc7a` | (bit-identical) |
| threads=1, PYTHONHASHSEED=12345 | `d73d961178b72306` | (bit-identical to threads=1) |

Environment: the recorded `numpy 2.4.6` is the in-repo `.venv`
(`B:\AEinstein\Einstein\Documents\Soren\qpsim\.venv`: numpy 2.4.6 /
SciPy 1.17.1 / CPython 3.14.3), verified 2026-08-03. Note this is *not*
the pinned ladder environment used for evidence regeneration, which is
numpy 2.5.1 — see the regen-env rule in `CURRENT-STATUS.md`; mixing the
two is what produced the recorded 2.7% drift incident.

Scope of the digests: no commit or tree hash was captured with the run,
so the two converged-`f` SHA-256 values bind to no checkable revision —
they are within-session identifiers demonstrating *equality across thread
counts*, not portable pins, and the hot path has since been rewritten.
Re-establish the result with the recipe below on a revision you hold
rather than treating the digests as pins.

Conclusions:

1. **There is no true run-to-run nondeterminism.** At a fixed (platform,
   BLAS build, thread count) the chain is bitwise reproducible.
2. **The BLAS thread count changes the converged answer** by ~5.1e-5
   relative on x_qp — eleven orders of magnitude above ulp. The dense
   linear-algebra reductions are the thread-sensitive step; the
   near-cancelling gain/loss balance at occupations ~1e-16 supplies the
   amplification. CI's "identical code, different results across runs" is
   runner heterogeneity (core count and per-arch OpenBLAS kernels); the
   3.13/3.14 leg disagreement is the same class via differing wheel builds.
3. Hash randomization is ruled out (table row 5). Anderson acceleration
   (`anderson_depth=3`, validation/fischer_2023/fig7_solve.py:85) was
   reported ruled out *at the probed point* — depth-0 and depth-3 runs
   were said to be bit-identical there because it never engages — but
   those two runs are not in the table above and were not recorded
   anywhere else, so that half is unverified and must be re-run before it
   is relied on. Its `lstsq(rcond=1e-10)` truncation on near-collinear
   history remains a plausible additional discrete amplifier at points
   where it does engage.

Status of remedies: the CI-level `OPENBLAS_NUM_THREADS=1` pin (already
present at the job level on this branch, `.github/workflows/ci.yml:21`)
makes each runner arch bitwise reproducible; cross-arch kernel variance
remains and stays absorbed by the widened fig7 gates. The "rtol=1e-3 /
atol=1e-10" quoted in the 2026-08-01 record were `main`'s values and do
not exist in this tree; the live constants are
`validation/fischer_2023/fig7_paper.py:204-208` —
`QP_LOSS_REGRESSION_RTOL = 4e-3`, `QP_LOSS_REGRESSION_ATOL = 1e-18`,
`Q_TOTAL_REGRESSION_RTOL = 1e-4`, with cross-platform legs
`QP_LOSS_CROSS_PLATFORM_RTOL = 8e-3` and
`Q_TOTAL_CROSS_PLATFORM_RTOL = 2e-4`. The absolute floor is eight orders
tighter than the retired 1e-10, which masked every loss below Q = 1e10
(see `validation/fischer_2023/test_fig7_paper.py:266-268`), so the remedy
claim must not be read against the old number; the 5.1e-5 relative shift
measured above still sits ~80x inside the 4e-3 relative leg. The
low-occupancy amplification itself is conditioning of the observable, not
a code defect.

Probe recipe (re-creatable in a few lines): build the fig7 point via
`validation.fischer_2023.fig7_solve` helpers, run
`backend.steady_state(state, photon_params=..., **SOLVER_KWARGS)` in a
fresh process per config, hash `f.tobytes()`.

Provenance note: this record was produced during the 2026-08-01 audit
round that ran against `main` while the `codex/*` stack was unmerged; its
other findings (fig3 r=10 trivial-branch collapse + vacuous fig3 gates,
coupled-Newton absolute-floor acceptance hole, sc-gap fig6 grid-support
failure at E_min = Δ_0, transient gap-remap noise dead-band) were later
found to have been independently discovered and fixed on this branch's
history — they are recorded here only as corroboration, with the fig7
result above the sole surviving novel finding.
