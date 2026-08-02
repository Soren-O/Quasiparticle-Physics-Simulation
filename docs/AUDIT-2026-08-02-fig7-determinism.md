# Audit 2026-08-02 — fig7 "non-determinism" root-caused (BLAS thread count)

Controlled experiment executed 2026-08-01 on the Einstein2 win32 box
(numpy 2.4.6/OpenBLAS), one cold fig7 point (P = −100 dBm, T_B = 0.06 K,
NUM_BINS = 1701), fresh process per run:

| config | converged `f` SHA-256 (16 hex) | x_qp |
|---|---|---|
| OPENBLAS_NUM_THREADS=1, run A | `d73d961178b72306` | 4.42474564313547041e-15 |
| OPENBLAS_NUM_THREADS=1, run B | `d73d961178b72306` | (bit-identical) |
| OPENBLAS_NUM_THREADS=4, run A | `02e8dbf8e7bcfc7a` | 4.42452173258215793e-15 |
| OPENBLAS_NUM_THREADS=4, run B | `02e8dbf8e7bcfc7a` | (bit-identical) |
| threads=1, PYTHONHASHSEED=12345 | `d73d961178b72306` | (bit-identical to threads=1) |

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
3. Hash randomization is ruled out. Anderson acceleration is ruled out *at
   the probed point* (depth-0 and depth-3 runs are bit-identical there —
   it never engages), though its `lstsq(rcond=1e-10)` truncation on
   near-collinear history remains a plausible additional discrete
   amplifier at points where it does engage.

Status of remedies: the CI-level `OPENBLAS_NUM_THREADS=1` pin (already
present at the job level on this branch) makes each runner arch bitwise
reproducible; cross-arch kernel variance remains and stays absorbed by the
widened fig7 gates (rtol=1e-3 / atol=1e-10). The low-occupancy
amplification itself is conditioning of the observable, not a code defect.

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
