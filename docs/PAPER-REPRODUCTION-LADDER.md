# Paper reproduction ladder

This is the validation workflow for Fischer–Catelani figures. It separates
three questions that the older figure scripts mixed together:

1. What did the authors' supplied program actually compute?
2. Can a small independent implementation reproduce that program and the
   published figure?
3. At which first component does replacing that implementation with qpsim
   change the result?

A green qpsim regression answers none of those questions by itself.

## Evidence classes

Every result must identify one of these classes:

| Class | Meaning |
|---|---|
| `author_supplied_reproduction_source` | Bytes the repository owner records as received from the authors; exact attachment/member bytes are content-bound here. Sender metadata is not independently preserved, and historical-publication identity is not established. |
| `clean_room_paper_equations` | An independent, small transcription of equations and stated parameters. It is not author code. |
| `clean_room_author_equivalent_numerics` | A small numerical port that has matched the author source's native intermediate arrays and output at declared points. |
| `hybrid_component_substitution` | A child of the minimal numerical port in which exactly one component is supplied by qpsim. |
| `qpsim_author_semantics` | All relevant components use qpsim, while retaining the authors' physical model and iteration semantics. |
| `qpsim_model_extension` | A deliberate change beyond the author model, such as qpsim's self-consistent moving-gap path. |

The machine-readable transition contract is
[`validation/reproduction_ladder.py`](../validation/reproduction_ladder.py).
It rejects a child stage that changes zero or more than one declared
scientific-component label and authenticates listed stage evidence. This is a
topology/intent check, not proof that arbitrary implementations differ only
in that component; frozen-state operator comparisons must establish that
before a stage is marked complete. Captured solver arrays are hashed in
`.npy` form without coercing dtype, shape, byte order, or complexness;
adapter-derived coordinate arrays are labelled as derived rather than native.

## Recovered Fischer 2023 Figure 6 source

The ZIP the repository owner records as the emailed author attachment has
been recovered outside the repository:

```text
PhysApplPaper_Figure_6.zip
size       3,827,356 bytes
SHA-256    31d76c92ef12c8056b583ef0d770f9f5c1ef48c561db971cc0dab1b733481bc1
```

Its checked manifest is
[`author-source.json`](../validation/paper_data/fischer_2023/fig6/author-source.json).
The archive is not committed because redistribution permission has not been
established. Set `QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE` to replay its
content authentication. The manifest binds the entry point, both solver
modules, material table, array helper, and bundled output PNG independently.

The classification is intentionally conservative:

- the external project record says the repository owner received the authors'
  Figure 6 code by email on 2026-05-20;
- source headers identify `paulb`;
- bundled CPython 3.8 bytecode identifies the original interpreter family;
- exact package versions and email message metadata were not preserved here;
- the attachment is therefore **author-supplied reproduction source**, not a
  sender-authenticated record or proven byte-for-byte historical publication
  checkout.

The unmodified `examples/Figure_6.py` declares:

- 100 logarithmic photon occupations for each of three bath temperatures
  (`0.10`, `0.15`, and `0.20 K`);
- 1620 energy intervals, `E_max = 10 Δ`, `ω0 = 20 μeV`,
  `Δ = 180 μeV`, and `τ_l = 255 ps`;
- ten dense coupled Newton steps with a `1e-7` successive-state threshold;
- a fixed kinetic gap followed by the direct, linearly interpolated
  edge-grid `delta_not_consistent(f)` gap integral.

The supplied 6400×4800 output PNG is also authenticated. Its independent
high-resolution extraction is scored against the arXiv-v2 raster in
[`author-output-score.json`](../validation/paper_data/fischer_2023/fig6/author-output-score.json).
All six solid/dashed curves agree within the combined raster bounds; maximum
uncertainty-normalized errors range from `0.037` to `0.283` (limit `1.0`).
This establishes that the bundled artifact is the high-resolution counterpart
of the published figure. The full 300-point calculation has not yet been
replayed; the bundled output is an authenticated author artifact, not a claim
that the current machine regenerated it.

## Figure 6 pilot

The checked ladder is
[`reproduction-ladder.json`](../validation/paper_data/fischer_2023/fig6/reproduction-ladder.json).
Its important endpoints are:

| Stage | Purpose | Current state |
|---|---|---|
| P0 | Independent arXiv-v2 raster oracle | Complete |
| A0 | Authenticate and eventually run the exact author attachment unchanged | Source and bundled output authenticated; full replay pending |
| A1 | Select one point and capture native arrays without changing author equations | Complete at `T_B=0.20 K`, original sweep index 49 |
| D0 | Independent Eq. 47/53 implementation for the dashed analytic curves | Complete |
| C0 | Minimal author-equivalent numerical port | Complete at the authenticated one-point state |
| C1 | Replace only the frozen-state direct observable | Complete at the accepted C0 state |
| C2 | Replace only scalar parameters and numerical constants on the frozen C1 state | Complete at the authenticated one-point state; no C2 root or ordinate is claimed |
| C3–C7 | Replace grid, photon operator, QP–phonon operator, phonon balance, then nonlinear solver | Formal C3–C7 are complete on the one authenticated point: frozen-state C3–C6 plus the C7 re-solve. The older same-seed C3a/C3b/C3c re-solve remains supplemental because it used the author-parameter endpoint, not C2 |
| Q0 | Full qpsim under author fixed-gap/direct-observable semantics | Planned |
| Q1 | qpsim self-consistent moving-gap extension | Existing model, but deliberately not an author-equivalence endpoint |

Before running a curve, the first numerical comparison point is fixed at
`T_B = 0.20 K` and author sweep index 49, whose actual coordinate is
`T*/Δ = 0.33990789737294363`. It is the nearest retained author point to the
paper-oracle coordinate `0.340068493151`, not the identical coordinate. Every
stage must bind its applicable evidence. A nonlinear re-solve stage must
emit:

- native quasiparticle and phonon grids, sample locations, dtype, and hashes;
- `f` and `n_ph`, seed identity, iteration count, and stopping reason;
- each photon, scattering, recombination, pair-breaking, and escape
  gain/loss contribution separately;
- full QP and phonon residuals and conserved-number diagnostics;
- the direct driven and thermal gap integrals;
- the final Figure 6 ordinate.

Frozen-state stages instead bind the immutable parent inputs and every
per-channel differential array, and explicitly state that they claim no new
root, stopping history, or ordinate.

Collision operators are compared on the same frozen state before any
nonlinear solve is replaced. This makes the first differing equation visible
instead of blaming a final-curve mismatch on “the solver.”

### Exact-point and frozen-state result

The exact full-resolution A1 replay is bound by
[`author-point-T020-sweep049-exact-anchor.json`](../validation/paper_data/fischer_2023/fig6/author-point-T020-sweep049-exact-anchor.json).
It used the authenticated source, `N=1620`, ten-step author policy, and
original sweep index 49. It converged in four Newton iterations and returned:

```text
actual T*/Delta       0.33990789737294363
author ordinate       0.12090908988993258
digitized paper       0.12093496 near T*/Delta = 0.34006849
promoted qpsim        0.08967258
```

The exact path and the separately labelled modern-runtime compatibility path
returned the identical ordinate. The compatibility path is not called
author-exact: it initializes an undefined masked `np.divide` output below the
pair threshold, while the exact path preserves the source bytes and behavior.

The formal qpsim-free C0 port starts from the exact A1 initial state, verifies
all four explicit-inverse Newton transitions and every final gain/loss
channel, reproduces the A1 full state to `3.951e-16` relative L2
(`6.581e-14` for the QP subvector), and returns the identical author
ordinate. Its checked result is
[`c0-author-equivalent-score.json`](../validation/paper_data/fischer_2023/fig6/c0-author-equivalent-score.json).
This is a one-point result; it does not validate the author initializer,
continuation path, or full 300-point curve.

At C1, on the exact C0 author grid, qpsim's direct gap observable reproduces the
driven gap, independently reconstructed thermal gap, and final ordinate
bit-for-bit when it inherits the attachment's exact `180*10**-6` binary64
`Delta_0`. The earlier one-ULP wording came from spelling that parameter as
`180.0e-6`, which rounds one ULP higher, and from holding the author thermal
gap fixed. That validation-script leak is corrected; the plotted gap
postprocessor is therefore not the source of the discrepancy. The checked C1
result is
[`c1-observable-score.json`](../validation/paper_data/fischer_2023/fig6/c1-observable-score.json).

Formal C2 is bound by
[`c2-parameter-score.json`](../validation/paper_data/fischer_2023/fig6/c2-parameter-score.json).
C2a first proves that explicit author-parameter plumbing is bit-preserving.
Five cumulative C2b steps then adopt only the current Figure 6 parameter
choices: explicit `180.0 µeV` energy literals, modern qpsim `k_B`, literal
`1 Hz` photon coupling, declared `tau_0^PB = 0.255 ns`, and the
finite-cutoff-derived `T_c = 1.184309192877208 K`. The generic Al material
YAML is deliberately not loaded.

Every step evaluates the same immutable C0 energy grid, driven occupation,
phonon occupation, and thermal occupation with the same authenticated
`n_bar`; there is no projection and no nonlinear solve. The immutable raw
bundle retains 124 frozen arrays, and the checked score independently
recomputes all 124 bit-for-bit, including the C0/C1 parent calculations and
per-channel locality checks. Because the raw arrays remain outside the
repository, a committed receipt binds both their manifest digest and the
complete checked-score bytes; the machine ladder binds that receipt as a
separate evidence item. Holding `n_bar` fixed shifts the Eq. 35
coordinate from `0.33990789737294363` to `0.3399503360830364`
(`+0.012485%`). The resulting nonzero residuals are expected operator
differentials at the old root. C2 therefore does **not** claim a changed
root or plotted ordinate; that same-seed nonlinear comparison belongs after
the frozen operator ladder.

### Formal C3 frozen grid differential

Formal C3 is bound by
[`c3-grid-score.json`](../validation/paper_data/fischer_2023/fig6/c3-grid-score.json).
It is a child of the accepted C2b5 endpoint, not a relabeling of the earlier
author-parameter staged-resolve pilot. It adopts the live Figure 6
`SpectralContext` domain: 1640 one-micro-eV cells on faces
`[160, 1800] micro-eV`, with 20 zero-capacity guard cells followed by the
1620 represented author cells.

The projection is an ordinal cell embedding, `parent i -> child i+20`.
There is no interpolation: the driven and thermal occupations copy
bit-for-bit into the active suffix, and the guard prefix is canonical positive
zero. The physical cell correspondence is exact by ordinal interval, while
the coordinates are deliberately not conflated: 449 of the 1620 mapped
**left cell faces** differ from the author left edges because of binary64
eV-to-micro-eV/grid-construction order, over
`[-1.1368683772161603e-13, +2.2737367544323206e-13] micro-eV`, while every
stored author sample is re-carried at a qpsim cell center approximately
`+0.5 micro-eV` from its parent left edge. Both arrays are retained.
The full qpsim phonon-frequency lattice `0..3599 micro-eV` is recorded, but
C3 retains only the author support `1..1619 micro-eV`; extending the phonon
equation into the remaining bins would be a C6 phonon-balance change.

The frozen path contains four cumulative substeps:

1. C3p pads the full domain while retaining the exact C2b5 active operator;
   every active gain/loss/net array reproduces C2b5 bit-for-bit.
2. C3a replaces only the author left-edge coherence by qpsim's
   finite-volume `K_plus/K_minus`.
3. C3b additionally moves pair labels from
   `2 Delta + (i+j)h` to the center-carrier
   `2 Delta + (i+j+1)h`.
4. C3c additionally adopts the same full `SpectralContext`'s native
   `cell_density`.

The raw bundle contains the full grid, faces, widths, mask, projection,
native spectral arrays, projected state, and all six gain/loss/net channel
decompositions for all four substeps. The checked score independently derives
the finite-volume BCS weights, anomalous weights, coherence matrices, maps,
and channel equations rather than trusting the producer. Under the true live
grid, the C3c native-density difference is much smaller than in the old
1620-cell pilot: the maximum symmetric relative density difference is about
`4.72e-13`, and the frozen DOS-weighted occupation measure shifts by about
`-8.29e-16` relative.

The observable control likewise keeps two meanings separate. Re-reading the
embedded vector with retained author left-edge semantics checks ordinal
projection identity. Interpreting that same vector at its declared qpsim
cell-center carriers reports the real half-bin sampling effect: at this
frozen point the driven and thermal integrals shift by `+11.1321%` and
`+2.9560%`, respectively, and the diagnostic suppression ratio changes from
`0.1209090899` to `0.0510985119`. This is deliberately reported rather than
hidden behind the nearly invariant left-edge control; neither value is a C3
root or plotted ordinate.

This completes C3 only as a one-point frozen grid/operator differential.
The inherited C2b5 state has a nonzero residual by design, so residual ratios
are diagnostics, not convergence certificates. C3 claims no nonlinear root,
Newton history, stopping result, plotted ordinate, curve, or paper parity.
The re-solve remains assigned to the later nonlinear-solver stage.

### Formal C4 frozen photon-operator differential

Formal C4 is bound by
[`c4-photon-score.json`](../validation/paper_data/fischer_2023/fig6/c4-photon-score.json).
It holds the accepted C3c state, 1640-cell grid, active mask, finite-volume
`K_plus`, and native partner `cell_density` fixed. The only component
replacement is the clean-room author photon residual with qpsim's public
`sub_gap_photon_collision_rates`.

The public return contract matters: its second array is a loss-rate
coefficient, whereas C3c stores the physical loss term. The formal comparison
therefore uses
`loss_s_inv = loss_rate_ns_inv * frozen_f / 1e-9`; comparing the raw
coefficient directly would manufacture a many-orders-of-magnitude mismatch.
The exact inherited inputs are `omega_0 = 20 micro-eV`, `dE = 1 micro-eV`,
`m = 20`, zero snap error, `c_photon = 1 s^-1 = 1e-9 ns^-1`, and the
authenticated fixed `n_bar`.

The score separates two effects:

1. qpsim source-order/per-nanosecond arithmetic with the author's
   terminal-cell omission restored differs from C3c only by binary64
   operation ordering;
2. the public qpsim endpoint policy additionally includes the representable
   terminal transition between child cells `1619` and `1639`, which the
   authenticated author residual omitted in both directions.

At this frozen tail, that terminal pair changes the two net rows by only
approximately `-2.88825e-35` and `+2.88858e-35 s^-1`. Across the full photon
net, public qpsim and C3c differ by approximately `2.02e-15` symmetric
relative L1, while the public cell-weighted number drift is approximately
`4.27e-17` of photon turnover. Thus C4 confirms the earlier qualitative
roundoff diagnosis while retaining the real endpoint semantic difference
instead of declaring the arrays identical.

The raw evidence retains public native-unit gain/loss-rate arrays, normalized
physical gain/loss/net arrays, the endpoint-control arrays, both separated
deltas, and the reconstructed hybrid QP residual. Every non-photon channel
and the phonon residual remain bound to C3c. An independent verifier
replays the selected C3 and C2 raw parents, transcribes the photon loop
without importing the producer or public operator, and rebuilds the checked
score before a receipt can be issued.

This completes C4 only as a one-point frozen photon-operator differential.
It changes no observable and claims no nonlinear root, Newton history,
stopping result, plotted ordinate, curve, or paper parity. Formal C5 now
performs the next QP-phonon operator substitution.

### Formal C5 frozen QP–phonon-operator differential

Formal C5 is bound by
[`c5-qp-phonon-score.json`](../validation/paper_data/fischer_2023/fig6/c5-qp-phonon-score.json).
It holds the accepted C4 state, 1640-cell grid, active mask, projected
phonon occupation, public photon channel, and every C3c phonon-equation
channel fixed. It replaces only the QP-side scattering and
pair/recombination kernels and their gain, physical-loss, and net
contractions through qpsim's public `phonon_collision_rates` path.

The frozen bundle separates three locality controls: C5s changes only the QP
scattering net, C5p changes only the QP pair net, and C5sp changes both to
form the formal C5 QP residual. The public frequency map is accepted only
after exact identity to the frozen C3 center labels. The inherited phonon
residual remains bit-for-bit identical; evaluating qpsim's phonon-side
balance is deliberately reserved for C6.

The combined QP physical net differs from the C4 parent by
`1.8534719101728832e-13 s^-1` in L1 and
`5.677749589118144e-16` symmetric relative L1. Pair gain, loss, and net each
agree within `4.48e-16` symmetric relative L1. The raw scattering gain and
loss arrays are not like-for-like buckets: the author source-order form
places the same Pauli cross-term in both, while public qpsim removes it from
both. That term cancels from the physical net. After rebucketing, scattering
gain and loss differ by only `9.50321362825228e-14` and
`1.9342844642219452e-13 s^-1` in L1; the physical scattering net differs by
`5.682685376326191e-16` symmetric relative L1.

The independent fixed-order verifier measures cell-weighted QP-number
conservation at `9.33258086197451e-17` relative drift. Pair
generation/recombination changes
QP number by construction: its nonzero weighted number moment
(`-0.1738186684181618 s^-1 micro-eV`, relative diagnostic about `0.0321`) is
recorded but is intentionally not a zero-drift conservation gate.

The 58-array raw bundle binds a 100-file source closure: the complete qpsim
Python/material tree plus the C5 producer, the executed C2/C3/C4 replay
verifiers, package initializers, and provenance helpers. An independent C5
verifier does not import the producer or changed public collision APIs; it
uses fixed-order `math.fsum` reductions and elementwise floating-point bounds
so producer-array authentication is kept separate from host-independent
scientific acceptance.

This completes C5 only as a one-point frozen QP-phonon-operator differential.
It claims no nonlinear root, Newton history, stopping result, observable
change, plotted ordinate, 300-point curve, coupled QP-phonon conservation, or
paper parity. Formal C6, described next, replaces the inherited author
phonon balance with qpsim's phonon-side equation.

### Formal C6 frozen phonon-balance differential

Formal C6 is bound by
[`c6-phonon-balance-score.json`](../validation/paper_data/fischer_2023/fig6/c6-phonon-balance-score.json).
It holds the accepted C5 frozen occupation, projected phonon occupation,
1640-cell grid, every public QP channel, and all parameters fixed. The only
component replacement is the inherited author phonon balance: qpsim's public
phonon-side kernels `2K⁻/(π Δ τ_0^PB)` and `K⁺/(π Δ τ_0^PB)`, the public
frequency map, the `compute_phonon_source_sink` contraction, and the
`ph0_local` bath-escape form `(n_th − n_ph)/τ_l` with the public thermal
occupation, evaluated on the full 3600-bin native ω lattice. Each channel is
retained as public affine coefficients `(a, b)` with
`dn_ph/dt = a + b·n_ph`; gain/loss/net are declared derived identities of
that decomposition. The C5 hybrid QP residual is inherited bit-exact — the
mirror of C5's bit-exact phonon inheritance.

Three channels match the inherited author equation to roundoff:

- the scattering net agrees to `1.279582321835755e-15` symmetric relative
  L1;
- the same-kernel correction-off pair control agrees to
  `2.351064591980878e-16`;
- escape gain and loss agree below `1e-12`, with the near-thermal escape
  net difference bounded elementwise by 16-eps rounding of the two
  thermal-occupation unit paths (maximum observed fraction `0.61`).

The one material endpoint difference is qpsim's Kaplan `S_+` pair-breaking
quadrature correction. The public pair path rescales each
complete-pair-interval ω bin from midpoint quadrature to the analytic
Kaplan total `S_+(ω/Δ)/(π τ_0^PB)`; 929 support bins change by more than
`1e-6`, with factors down to `0.7857612777062984` at the `2Δ` threshold.
This moves the public pair net by `9.203114358766813e-3` symmetric relative
L1 against the author channel and the formal C6spe phonon residual by
`2.0546278031187717e-6`, versus `2.1765109703719185e-10` for the
correction-off control — every non-Kaplan substitution is
roundoff-equivalent. Like C4's terminal-pair policy, this difference is
recorded, not gated; it is the first material frozen-state physics
difference in the qpsim endpoint semantics found by the ladder.

Detailed balance is scored per channel at a native center-grid thermal
control (public Fermi occupation and public thermal phonon occupation at
`T_B`): both e-ph channels balance to below `6.3e-16` of their turnover
against a `1e-12` gate. The scattering channel is structurally confined to
the author support `[1, 1620) µeV`; pair and escape out-of-support totals
are at most `8.763820847555044e-26 s⁻¹` and the ω=0 bookkeeping bin is
exactly zero. qpsim's public `phonon_balance_diagnostics` certifies the
full three-term balance at the frozen state as a diagnostic (the frozen
state is an author-model root, not a qpsim balance root). C6s, C6p, C6p0,
C6e, C6spe, and C6spe0 isolate every locality combination. An independent
C6 verifier does not import the producer or the changed public
phonon-balance APIs; it transcribes the kernels, contractions, Kaplan
correction (closed elliptic form), and occupations with fixed-order
`math.fsum` reductions and predeclared gamma bounds.

This completes C6 only as a one-point frozen phonon-balance differential.
It claims no nonlinear root, Newton history, stopping result, observable
change, plotted ordinate, 300-point curve, coupled QP-phonon conservation,
or paper parity. Formal C7, described next, replaces the nonlinear solver.

### Formal C7 nonlinear-solver re-solve

Formal C7 is bound by
[`c7-nonlinear-solver-score.json`](../validation/paper_data/fischer_2023/fig6/c7-nonlinear-solver-score.json).
It is the first re-solve stage: every operator stays at its accepted public
configuration — the C4 photon channel, the C5 QP-side kernels, and the C6
phonon-side balance with the Kaplan `S_+` correction — and only the
author's explicit-inverse Newton driver is replaced by qpsim's public
`coupled_newton_solve` (fixed gap, analytic cross-Jacobian, author-intent
`max_iter=10` and `step_rtol=1e-7`). The solve starts from the identical
captured author continuation seed bound by the accepted C0 evidence,
ordinally projected exactly as C3 projected the root.

The one-component claim is bound in both directions. Before the solve, the
solver's residual assembly evaluated at the frozen C6 state matches the
accepted C6 hybrid residuals to `1.3227587648e-13` (QP) and
`1.6878314118e-16` (phonon) symmetric relative L1. The kernels, frequency
map, and thermal occupation reproduce the accepted C5/C6 arrays
bit-for-bit. The public solver returns only its final state, so the
accepted iteration count is measured through the public API alone: the
smallest converging `max_iter` cap is **4 — the same count as the author
control** — and the declared-cap solve reproduces that root bit-for-bit.
At the root, the L1 balance ratios are `3.7094353580e-14` (QP) and
`5.7565249895e-12` (phonon) against the `1e-7` certificate.

The one-point attribution result:

| Observable convention | Value |
|---|---:|
| Author control ordinate (frozen author root, author sampling) | `0.12090908988993258` |
| C7 re-solved root, author sampling | `0.14542377851441587` |
| C7 re-solved root, center-carrier interpretation | `0.07767766591390296` |
| Historical promoted qpsim ordinate (Q1 model) | `0.08967258` |

The frozen author root reproduces the author control **exactly** under the
accepted observable path, so the observable chain is bit-consistent.
Re-solving with the full qpsim operator set moves the author-sampled
ordinate up (dominated by the center pair-frequency label, consistent with
the earlier pilot's `0.1457`, with the Kaplan correction pulling slightly
down), while interpreting the same root at qpsim's center carriers lands
adjacent to the historical promoted qpsim value. At this point the
long-standing 33–39% rising-branch discrepancy is attributable primarily to
the **observable sampling semantics** (the C3-documented half-bin carrier
shift), not to defective collision kernels, which match the author
equations to roundoff throughout C4–C6.

This completes C7 at one authenticated seed/root pair. It claims no
300-point curve, no paper parity, and no author-equivalence of the changed
iteration policy; the qpsim-internal Newton path is bound by the
hash-closed solver source and certified through host-independent residual
bounds. The remaining endpoints are Q0 (the fair author-semantics qpsim
endpoint over the full curve) and Q1 (the moving-gap extension).

The frozen collision comparison now contains an independent, qpsim-free
transcription of the author discretization plus controlled substitutions.
Its QP-channel difference from the captured author evaluation is
`3.0e-14` of channel turnover. The first substantive replacement is the grid
contract. The pilot scores two scientific discretization conventions plus one
separately isolated implementation-arithmetic substep:

1. the author evaluates `U±` coherence at left edges, whereas qpsim uses the
   exact finite-volume coherence average;
2. the author labels a pair at `2Delta+(i+j)h`, whereas qpsim center carriers
   label it one bin higher at `2Delta+(i+j+1)h`;
3. the analytically identical author and qpsim cell-density formulas acquire
   a small binary64 difference when qpsim evaluates them natively in micro-eV.

An exploratory frozen-state diagnostic suggested that, once the spectral
coefficients were held fixed, qpsim's photon operator matched the independently
transcribed author algebra to roundoff. Formal C4 subsequently bound and
verified that gain/loss comparison, including the different public loss-rate
return semantics and the numerically tiny terminal-pair policy extension,
before any re-solve claim. The one-bin pair-label change carries essentially
all of the frozen-state DOS-weighted QP-number imbalance; the coherence
convention dominates the pointwise residual.

The phonon residual must not be interpreted by comparing two nearly zero
totals. The author root cancels about `8.19e5 s^-1` of channel turnover.
One-at-a-time substitutions, normalized by that turnover, give:

| Substitution at captured A1 frozen author state | Residual / channel turnover |
|---|---:|
| Author-equivalent accumulator | `7.0e-11` |
| Pair-frequency label only | `1.11e-5` |
| Finite-volume coherence only | `1.03e-2` |
| Fully native qpsim grid with author constants | `1.03e-2` |
| Current numerical constants only | `1.24e-1` |
| Fully native qpsim evaluation | `1.23e-1` |

The constants row is stiffness evidence, not a recommendation to replace
qpsim's modern SI constant: a `3.39e-7` relative change in `k_B/e` is
amplified by the `0.255 ns` escape time when evaluated at a state converged
with the older author literal. Each variant must be re-solved before its
effect on the plotted ordinate is judged.

### Same-seed staged-resolve pilot

A single-point counterfactual pilot has now re-solved three localized grid
changes separately and cumulatively. Every primary variant started from the same captured
author seed; each independent variant converged in four Newton iterations.
The compact checked result is
[`staged-resolve-pilot.json`](../validation/paper_data/fischer_2023/fig6/staged-resolve-pilot.json).
Its generator reloads each external NPY from one retained byte snapshot and
independently recomputes stage identity, seed/path constraints, stopping
tests, final bounds, channel residuals, cancellation certificates, and
observables rather than trusting the raw manifest's scientific metadata.

| Variant | Figure 6 ordinate | Change from author control |
|---|---:|---:|
| Author control | `0.12090908988993258` | — |
| Finite-volume coherence only (C3a pilot) | `0.12070758916263027` | `-0.00020150072730` |
| `+h` pair-frequency label only (auxiliary counterfactual) | `0.14590106106941977` | `+0.02499197117949` |
| Both finite-volume coherence and center labels (C3b pilot) | `0.14570562776829468` | `+0.02479653787836` |
| `+` native-micro-eV qpsim cell density (C3c pilot) | `0.14570561703489288` | `+0.02479652714496` |

The control state matches the captured author final state to
`3.951e-16` relative L2 over the full state and gives the identical
ordinate. Solving the combined variant independently or by continuation gives
the identical ordinate and states agreeing to `4.240e-20` relative L2.
For C3c, independent and C3b-continuation solves also give the identical
ordinate and states agreeing to `8.73e-19` relative L2. Across the converged
variants, QP residual/channel-turnover ratios are
approximately `1.3e-15`–`2.1e-15`, and phonon ratios are approximately
`6.2e-12`–`1.60e-11`; the independently re-solved author control is the
`1.596e-11` endpoint. This is distinct from the `7.0e-11` frozen captured-A1
accumulator diagnostic above.

The nonlinear path preserves the authenticated source's three historical
residual/Jacobian inconsistencies: terminal-bin photon transitions, photon
off-diagonals that use the partner occupation, and two shifted indices in the
phonon-pair `D_n N` diagonal. Those are part of the author-control numerical
semantics here, not silent corrections. The v3 raw bundle now retains every
exact Newton delta; the checked v4 pilot rebuilds each source-bound update
matrix, authenticates every intermediate transition, and independently
recomputes every linear-solve backward error.

The last grid-only substep before replacing the photon operator is now
measured. C3a uses
qpsim's finite-volume coherence matrix in native micro-eV units but
deliberately retains the author DOS array so that it changes coherence only.
Although the DOS formulas are analytically identical, qpsim's native
`cell_density` evaluation differs by less than the checked `2e-7` relative
bound in the singular first cell through binary64 gap-edge arithmetic. C3c adopts that native DOS
separately and changes the ordinate by only `-1.07334e-8` relative to C3b.
C4 can therefore call the public qpsim photon operator without silently
mixing this grid-arithmetic change with photon terminal-bin semantics.

This is strong one-point evidence that the coherence convention has a
negligible effect on this ordinate, while the `+h` pair label has a material
effect. The label change moves the result upward, however—away from the
promoted qpsim ordinate `0.08967258`—so none of the localized C3 substitutions
explains qpsim's lower value. The photon, QP-phonon, and phonon-balance
substitutions are now completed by formal C4, C5, and C6; the frozen C6
comparison isolates the Kaplan `S_+` pair-breaking quadrature correction as
the first material endpoint difference. The downstream nonlinear-solver
substitution, followed by the author-semantics qpsim endpoint and only then
the moving-gap extension, remains to be tested.

These remain supplemental pilot counterfactuals because they use the
author-parameter endpoint and a 1620-cell coefficient carrier rather than the
formal C2-to-1640-cell projection. The separate frozen-state C3 artifact above
now completes the machine-declared C3 contract; it does not promote these
pilot ordinates into formal C3 results. Formal frozen C4, C5, and C6 are now
complete as described above; C7 and the 300-point author replay remain
pending.

### Independent analytic control

[`fig6_analytic.py`](../validation/reference_models/fischer_2023/fig6_analytic.py)
contains no qpsim import. It implements Eq. 35, the corrected Eq. 47 balance,
Eq. 53, and a continuum thermal BCS gap integral after
`E = Δ cosh(u)`. Its provenance-bound score is
[`cleanroom-analytic-score.json`](../validation/paper_data/fischer_2023/fig6/cleanroom-analytic-score.json).

All three dashed curves pass the pre-existing raster uncertainty metric:

| Bath temperature | Maximum uncertainty-normalized error |
|---:|---:|
| 0.10 K | 0.341 |
| 0.15 K | 0.204 |
| 0.20 K | 0.251 |

The acceptance limit is `1.0`. This validates the narrow analytic
transcription and the raster calibration. It does **not** validate the solid
curves, the author numerical solver, or qpsim.

## What is now attributable—and what is not

The existing paper-parity job compares the solid paper trace with qpsim's
self-consistent model. The attachment instead uses a fixed kinetic gap and a
direct postprocessed gap observable. Those are different scientific models.
The current 33–39% rising-branch discrepancy is real as a comparison of those
two plotted outputs. At the anchored point it is no longer attributable to
the author replay, the runtime compatibility patch, or the direct gap
observable. The first frozen-state divergence is the discretization contract
described above.

The one-point staged-resolve pilot shows that finite-volume coherence is
negligible there and that pair-frequency labelling is material but moves in
the wrong direction to explain qpsim's lower ordinate. It does **not**
establish behavior over the curve or replace a C2-parent re-solve. Formal C2
through C6 now have their one-point frozen-state evidence contracts;
the nonlinear-solver replacement C7, Q0, and Q1 remain.

Q0, not Q1, is the fair endpoint for judging author equivalence. Q1 should be
compared with Q0 only after the component ladder is complete.

## Extending the method to the other figures

The exact arXiv-v2 source archive contains the original rasters for all
Fischer–Catelani 2023 numerical figures:

```text
T_Bstar.png
Distributions.png
Distributions_n.png
Density_1_new.png
Density_2_new.png
gap_test.png
Q.png
freq.png
R_bar.png
```

`gap_test.png` (Figure 6) has a complete six-curve raster-oracle/score chain,
not a completed numerical replacement ladder. `R_bar.png`
(Figure 8) now has a provenance-bound solid/dashed raster oracle and an
accepted clean-room score for the dashed Eq. E2 trace; its blue numerical
trace is intentionally retained as an unscored oracle for a later numerical
replacement ladder. The remaining assets must receive separate
`oracle.json`, `points.csv`,
`comparison-spec.json`, and score records before their qpsim baselines can be
called paper validation. Existing qpsim CSV pins remain useful regression
evidence, but they are not independent paper data.

For each remaining figure:

1. bind the exact arXiv version, archive hash, raster member hash, caption,
   panel, axes, units, curve identity, and line/marker style;
2. predeclare calibration controls, held-out tick residuals, sampling
   coordinates, and uncertainty;
3. extract points without reading qpsim output;
4. write the smallest equation-level reference that covers analytic traces;
5. recover author source if available; otherwise label the numerical
   reference clean-room;
6. establish the minimal numerical result before substituting qpsim
   components one at a time;
7. report the first divergent component and preserve all downstream
   differences as consequences, not independent root causes.

No visual “looks right” verdict, self-pinned CSV, or regenerated qpsim plot
can substitute for those steps.
