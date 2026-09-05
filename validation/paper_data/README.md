# Independent paper-data oracles

This directory contains coordinates extracted from publication assets. These
files are independent references; they are not qpsim-generated baseline CSVs.

Each dataset keeps three contracts separate:

1. `oracle.json` authenticates the paper/version, exact source archive and
   raster member, panel calibration, extraction method, uncertainty model,
   declared curves, and `points.csv`.
2. `comparison-spec.json` maps paper curves to named qpsim observables and
   predeclares interpolation, metrics, and the scientific error budget.
3. `score.json` binds those inputs to exact promoted qpsim CSV/promotion
   bytes and the scorer source that produced the diagnostic result. It does
   not replay all persisted solver states; that remains a separate slow gate
   under the recorded single-thread environment.

Independent reference and author-source evidence are additional, separately
typed layers:

- `author-source.json` authenticates an externally held author attachment
  without assuming redistribution permission or silently treating it as a
  proven historical publication checkout.
- `author-output-score.json` verifies that the attachment's bundled
  high-resolution output agrees with the published raster; it does not claim
  that the source was replayed on the current runtime.
- `cleanroom-analytic-score.json` scores a qpsim-independent paper-equation
  implementation. It cannot validate the numerical solid curves.
- `reproduction-ladder.json` declares the author, clean-room, hybrid, and
  qpsim stages, checks one changed **declared** component per adjacent stage,
  and authenticates listed implementation/result artifacts. It does not by
  itself prove that the underlying code changed only that component; frozen
  state/operator comparisons supply that evidence as stages are completed.

The source raster is not redistributed. A fresh checkout can score the
checked coordinates immediately. Reproducing the coordinates from pixels
requires separately obtaining the exact source archive identified by URL and
SHA-256 in the oracle.

## Fischer–Catelani 2023, Figure 6

The first oracle is under `fischer_2023/fig6/`. Reproduce its point CSV from
the full single panel on PDF page 12 of the exact arXiv-v2 source archive
with:

```bash
python -m validation.fischer_2023.extract_fig6_paper_data \
  path/to/fischer-catelani-2023-arxiv-v2-source.tar \
  --output reproduced-fig6-points.csv
```

Verify the checked downstream score with:

```bash
python scripts/score_fischer_2023_fig6_paper_parity.py --verify
```

The current canonical score is `score.json` SHA-256
`ff0e12b5037f08ab7fc48cec42e5d81b42acaaa2c2e66e3e7f03583ec6c450bc`.
This digest changed when the shared paper-parity validator was generalized
for the Figure 8 trace-mask schema; every Figure 6 curve score and the
`diagnostic_mismatch` result stayed unchanged.

The dashed analytic curves are calibration/identity controls and agree with
qpsim's independently transcribed Eq. 53 values. The solid published
numerical curves do not agree with the currently promoted qpsim numerical
curves at the seven sampled points on the visible rising branch
(`T*/Delta ≈ 0.250–0.410`). The comparison spec explicitly binds both axes
as dimensionless identity mappings and fixes the gap-suppression sign and
denominator convention. This is reported as a diagnostic
mismatch, not a release-gate failure: paper-parameter and qpsim
discretization uncertainties remain unbounded in the comparison-specific
error budget.

The exact author-supplied Figure 6 attachment has since been recovered and
content-bound by `author-source.json`. Its algorithm uses fixed-gap kinetics
and a direct edge-grid gap observable, so the existing self-consistent qpsim
score is not an author-equivalent endpoint. The systematic replacement plan
and its qualifications are documented in
`docs/PAPER-REPRODUCTION-LADDER.md`.

One exact full-resolution author point is now replayed and bound by
`author-point-T020-sweep049-exact-anchor.json`: at `T_B = 0.20 K`, original
sweep index 49, and `N = 1620`, the author source converged to
`T*/Delta = 0.33990789737294363` and ordinate
`0.12090908988993258`. The digitized paper trace is approximately
`0.12093496` near `T*/Delta = 0.34006849`; the promoted qpsim comparison is
approximately `0.08967258`. This completes one exact point, **not** the
author attachment's full 300-point replay.

The formal qpsim-free C0 port verifies all four author-policy Newton
transitions, reproduces the A1 full state to `3.951e-16` relative L2
(`6.581e-14` for the QP subvector), and returns the identical ordinate. At C1,
qpsim's direct-gap observable reproduces the driven gap, independently
reconstructed thermal gap, and final ordinate bit-for-bit when every binary64
parent parameter is inherited exactly. Formal C2 then changes only the
explicit Figure 6 parameter choices on that immutable state. Its raw bundle
retains 124 arrays, and the checked score independently recomputes every one
bit-for-bit in
[`c2-parameter-score.json`](fischer_2023/fig6/c2-parameter-score.json). The
separate committed raw-manifest receipt keeps those external arrays and the
complete checked-score bytes fail-closed in a clean checkout. The
authenticated `n_bar` remains fixed, shifting the Eq. 35 coordinate from
`0.33990789737294363` to `0.3399503360830364`; no nonlinear root or C2
ordinate is claimed.

Formal C3 is now checked separately in
[`c3-grid-score.json`](fischer_2023/fig6/c3-grid-score.json). It starts from
the accepted C2b5 endpoint and adopts the actual 1640-cell qpsim Figure 6
grid, including its 20 zero-capacity sub-gap guard cells. Parent cell `i`
maps to child cell `i+20` with no interpolation; the driven/thermal
occupations copy bit-for-bit and the guard prefix is canonical positive zero.
The full qpsim `0..3599 micro-eV` phonon lattice is recorded, but only the
author `1..1619 micro-eV` support is evaluated so the later phonon-balance
stage remains isolated.

The external raw bundle retains 105 grid, projection, spectral, state, and
per-channel arrays across C3p/C3a/C3b/C3c. Its checked score independently
rederives the finite-volume BCS weights/coherence and all six gain/loss/net
balances; the committed raw-manifest receipt binds that external manifest and
the complete checked-score bytes. C3p reproduces all active C2b5 channel
arrays bit-for-bit. The coordinate relabel is not falsely called exact:
449 mapped left cell faces differ by binary64 construction order, with
maximum magnitude `2.2737367544323206e-13 micro-eV`, while all 1620 retained
author samples are carried at qpsim centers approximately `+0.5 micro-eV`
from their parent left edges. A separate native-center observable diagnostic
records the resulting `+11.1321%` driven-integral and `+2.9560%`
thermal-integral shifts, preventing the nearly invariant author-semantics
projection control from hiding that half-bin reinterpretation.

This completes C3 only as a frozen one-point differential. It claims no C3
root, Newton history, stopping result, ordinate, curve, or paper parity.

Formal C4 is checked separately in
[`c4-photon-score.json`](fischer_2023/fig6/c4-photon-score.json). It holds the
accepted C3c state, grid, `K_plus`, and native partner `cell_density` fixed
and substitutes only qpsim's public `sub_gap_photon_collision_rates`.
The raw bundle retains the public per-nanosecond gain and loss-rate
coefficient, the correctly converted physical per-second loss
(`loss_rate * f`), gain/net arrays, an author-terminal-policy arithmetic
control, the separated endpoint and roundoff deltas, and the reconstructed QP
residual. Non-photon channels and the phonon residual remain bound unchanged
to C3c.

The public photon net agrees with C3c to approximately `2.02e-15` symmetric
relative L1. The one semantic difference is the representable terminal pair
between child cells `1619` and `1639`, omitted by the author residual; its
frozen net contributions are only approximately `-2.88825e-35` and
`+2.88858e-35 s^-1`. The public weighted number drift is approximately
`4.27e-17` of photon turnover. Receipt creation independently rebuilds C3
from selected C3/C2 raw evidence and then rebuilds C4, so a self-consistent
forged parent score/receipt is not accepted. C4 remains one frozen operator
comparison and claims no root, Newton history, stopping result, observable,
ordinate, curve, or paper parity.

Formal C5 is checked separately in
[`c5-qp-phonon-score.json`](fischer_2023/fig6/c5-qp-phonon-score.json). It
holds the accepted C4 state, 1640-cell grid, projected phonon occupation,
public photon channel, and every inherited phonon-equation channel fixed.
It replaces only the QP-side scattering and pair/recombination kernels and
their public gain/loss-rate contractions. Physical loss is formed as
`loss_rate * frozen_f`; the loss-rate coefficient is retained separately.
C5s, C5p, and C5sp isolate the scattering-only, pair-only, and combined
QP-residual updates.

The combined physical net differs from C4 by
`1.8534719101728832e-13 s^-1` in L1 and
`5.677749589118144e-16` symmetric relative L1. The author source-order
scattering gain and loss buckets both contain the same Pauli cross-term,
whereas public qpsim removes it from both. Raw gain/loss differences are
therefore bookkeeping differentials, not physical disagreement. After
like-for-like rebucketing, gain/loss L1 differences are
`9.50321362825228e-14`/`1.9342844642219452e-13 s^-1`, and the physical
scattering net agrees to `5.682685376326191e-16` symmetric relative L1.
Pair gain/loss/net each agree within `4.48e-16` symmetric relative L1.
Scattering passes its weighted-number conservation gate. Pair processes
change QP number by construction, so their nonzero weighted number moment is
retained as a diagnostic rather than tested against zero.

The external raw bundle contains 58 arrays and a fail-closed 100-file source
manifest. The committed
[`c5-raw-manifest-receipt.json`](fischer_2023/fig6/c5-raw-manifest-receipt.json)
binds that raw manifest and the complete checked score. The source closure
includes the complete qpsim Python/material tree, producer, executed
C2/C3/C4 replay verifiers, package initializers, and provenance helpers. The
independent C5 verifier uses fixed-order reductions and floating-point error
bounds, so the producer's retained public arrays remain runtime-authenticated
without requiring cross-platform BLAS last-bit identity.

C5 remains one frozen operator comparison. It claims no nonlinear root,
Newton history, stopping result, observable change, ordinate, 300-point
curve, coupled QP-phonon conservation, or paper parity. The phonon residual
is inherited bit-exact from C3c; qpsim's phonon-side balance is the C6
substitution.

Formal C6 is checked separately in
[`c6-phonon-balance-score.json`](fischer_2023/fig6/c6-phonon-balance-score.json).
It holds the accepted C5 state, grid, projected phonon occupation, and every
public QP channel fixed and replaces only the phonon-side balance: qpsim's
public phonon-side kernels `2K-/(pi Delta tau_0^PB)` and
`K+/(pi Delta tau_0^PB)`, the public frequency map, the
`compute_phonon_source_sink` contraction, and the `local` bath-escape
form on the full 3600-bin native omega lattice. The scattering net matches
the inherited author channel to `1.279582321835755e-15` symmetric relative
L1, the same-kernel correction-off pair control to
`2.351064591980878e-16`, and escape gain/loss below `1e-12` with an
elementwise-bounded near-thermal net. The single material endpoint
difference is qpsim's Kaplan `S_+` pair-breaking quadrature correction:
public pair net `9.203114358766813e-3` symmetric relative L1, formal C6spe
phonon residual `2.0546278031187717e-6` versus `2.1765109703719185e-10` for
the correction-off control. Detailed balance at a native thermal control
holds per channel below `6.3e-16` of turnover; scattering is structurally
confined to the author support, out-of-support totals stay at or below
`8.77e-26 s^-1`, and the C5 hybrid QP residual is inherited bit-exact. The
external raw bundle contains 86 arrays; the committed
[`c6-raw-manifest-receipt.json`](fischer_2023/fig6/c6-raw-manifest-receipt.json)
binds that raw manifest, the complete checked score, and the independently
replayed C5/C4/C3/C2 chain. C6 remains one frozen operator comparison and
claims no nonlinear root, Newton history, stopping result, observable
change, ordinate, 300-point curve, coupled QP-phonon conservation, or paper
parity; the nonlinear solver is the C7 substitution.

Formal C7 is checked separately in
[`c7-nonlinear-solver-score.json`](fischer_2023/fig6/c7-nonlinear-solver-score.json).
It is the first re-solve stage: only the author Newton driver is replaced
by qpsim's public `coupled_newton_solve` under the accepted C4/C5/C6
operator configuration, seeded from the identical C0-bound author
continuation state. The solver's residual assembly at the frozen C6 state
matches the accepted hybrid residuals to `1.32e-13`/`1.69e-16` symmetric
relative L1; the minimal converging public iteration cap is 4 (the author
control's count) and the root balance ratios are `3.7e-14`/`5.8e-12`
against the `1e-7` certificate. The re-solved author-sampled ordinate is
`0.14542377851441587` versus the exactly reproduced author control
`0.12090908988993258`; the same root reads `0.07767766591390296` under the
C3-documented center-carrier interpretation, adjacent to the historical
promoted qpsim `0.08967258`. The committed
[`c7-raw-manifest-receipt.json`](fischer_2023/fig6/c7-raw-manifest-receipt.json)
binds the 44-array raw bundle, the checked score, and the replayed
C6/C5/C4/C3/C2 chain. C7 claims no 300-point curve, paper parity, or
author-iteration-equivalence; Q0/Q1 remain.

At the captured A1 frozen state, the author-equivalent phonon accumulator's
residual is approximately `7e-11` of turnover. Controlled
operator substitutions localize the first substantive numerical differences
to two discretization conventions: author left-edge coherence factors versus
qpsim's exact finite-volume coherence, and author pair-frequency labels
`2 Delta + (i+j)h` versus qpsim center labels
`2 Delta + (i+j+1)h`. A third grid-level substep separately isolates the
small binary64 difference from evaluating the analytically identical
cell-density formula in qpsim's native micro-eV units.

A same-seed, single-point staged-resolve pilot has now tested those two
changes plus the isolated native-DOS arithmetic substitution. All independent
variants converged in four Newton
iterations. The author control returned `0.12090908988993258`; coherence only
returned `0.12070758916263027` (`-0.00020150072730`); the `+h` pair label only
returned `0.14590106106941977` (`+0.02499197117949`); and both changes returned
`0.14570562776829468` (`+0.02479653787836`). C3c changes only the DOS
arithmetic and returns `0.14570561703489288`, a `-1.07334e-8` shift from C3b.
The control matches the captured
author final state to `3.951e-16` full-state relative L2 with the identical
ordinate. Independent and continuation solves of the combined variant agree
to `4.240e-20` relative L2 and give the identical ordinate. Independent and
continued C3c states agree to `8.73e-19` relative L2. QP
residual/turnover ratios are approximately `1.3e-15`–`2.1e-15`; phonon ratios
are approximately `6.2e-12`–`1.60e-11`. The independently re-solved author
control is the `1.596e-11` endpoint, distinct from the frozen captured-A1
diagnostic above.

The checked
[`staged-resolve-pilot.json`](fischer_2023/fig6/staged-resolve-pilot.json)
does not trust the raw staged manifest's scientific claims: it reloads one
retained snapshot of every external NPY and independently recomputes stage
specifications, shared-seed and solve-path constraints, every Newton
transition and linear backward error, stopping tests, final bounds, channel
residuals, cancellation certificates, observables, and path comparison. The
producer also preserves the supplied author Jacobian's
terminal-photon, partner-occupation, and shifted-`D_n N` index behavior; this
is an author-numerics comparison, not a silently corrected Newton method.

C3c now adopts qpsim's native `cell_density` arithmetic before C4 invokes
the public photon operator. The formulas match analytically, but the binary64
evaluation difference in the singular first cell is below the checked `2e-7`
relative bound; the
one-point ordinate effect is only `-1.07334e-8`. Keeping it separate preserves
the one-change-per-pilot-substep discipline.

At this point, coherence is negligible and the pair-label shift is material,
but the latter raises the ordinate and therefore moves away from the promoted
qpsim value `0.08967258`. None of the three isolated substitutions explains qpsim's lower
ordinate. Formal one-point C0/C1 and frozen-state C2 evidence are complete;
the re-solve remains a supplemental author-parameter pilot rather than a
formal C3 ordinate or causal attribution over the curve. The separate
C2-parent frozen evidence above completes formal C3, the C3c-parent photon
evidence completes formal C4, the C4-parent QP-phonon evidence completes
formal C5, the C5-parent phonon-balance evidence completes formal C6, and
the C6-parent re-solve completes formal C7. The Q0/Q1 model endpoints and
the full 300-point author replay remain open.

## Fischer–Catelani 2023, Figure 8

The `fischer_2023/fig8/` oracle independently authenticates and digitizes
both traces in the author raster: the blue solid numerical curve and the
black dashed analytic curve. Reproduce its point CSV from the exact
arXiv-v2 source archive with:

```bash
python -m validation.fischer_2023.extract_fig8_paper_data \
  path/to/fischer-catelani-2023-arxiv-v2-source.tar \
  --output reproduced-fig8-points.csv
```

The checked `cleanroom-analytic-score.json` accepts a qpsim-independent
transcription of Eq. E2 against the dashed trace (maximum
uncertainty-normalized error `0.2971`). The provenance-bound blue solid
numerical trace is deliberately retained as `digitized_reference_only`: no
author numerical implementation or qpsim replacement is yet bound to that
comparison, so it is unscored and reserved for a later component-replacement
ladder. The accepted dashed score does not validate the blue curve or qpsim.
