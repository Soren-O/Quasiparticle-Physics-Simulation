# qp-diffusion adversarial review - 2026-07-10

## Status

This is a read-only review record. No finding in this document has been
applied to paper.tex, supplement.tex, refs.bib, a verification script, a
figure, or qpsim.

- Repository: Soren-O/Quasiparticle-Physics-Simulation
- Branch reviewed: fix/gpt-review-2026-07-05
- Commit reviewed: 85980561003b50fd95184dd90400b8ec67d61c44
- Working tree before review: clean
- Working tree after the first review: clean except for this new report
- Review date: 2026-07-10
- Review mode: one fresh reviewer for each of 44 manuscript units, followed
  by independent rebuttal of every substantive candidate finding and a
  counter-audit of CLAUDE.md
- Second-round mode: full read of the 98 KB external physics audit, followed
  by fresh counter-review of its derivations, numerical scripts, severity
  assignments, and public-repository claims
- Current working tree: Claude's status correction in CLAUDE.md is modified;
  this report is untracked; no manuscript, bibliography, verifier, figure, or
  engine file has been changed
- Engine branches, Fischer baselines, and qpsim engine fixes were excluded.

The required baseline was reproduced before review:

- verify_fT.py: pass
- verify_gA_convention.py: pass
- verify_nonadiabatic.py: pass
- verify_proximity.py: pass
- verify_supercurrent.py: exit zero; all displayed checks completed
- verify_tdep_inhomogeneous.py: pass
- verify_traces.py: pass
- Aggregate: 7/7 scripts exited zero

The baseline being green does not dispose of the findings below. Several
findings identify missing assertions or a same-order contribution that the
current scripts do not test.

## Second-round counter-audit

The external audit in
`B:/AEinstein/Einstein/Documents/Soren/GPT-REVIEW-PHYSICS-AUDIT-2026-07-10.md`
substantially corroborated the first review, but its aggregate tally is not a
reliable severity tally. Several of its own judge and verifier reports
contradict one another, and a fresh counter-audit found additional numerical
and order-counting errors in the audit itself. The final dispositions used by
this document are:

| Finding | Second-round disposition |
|---|---|
| B1 | CONFIRMED blocker to C1 insertion; the external audit's proposed quantitative rescue is not established |
| B2 | CONFIRMED major appendix/verifier error; static A1 unaffected |
| B3 | CONFIRMED nonuniform edge scaling; exact kernel and fixed-energy outer expansion survive |
| B4a | PARTLY CONFIRMED, minor notation only; the paper already states the even-harmonic guard and retains the hidden odd harmonic |
| B4b | NARROWED to wording/evidentiary scope; the paper discloses that the printed Kopnin pair is linearized and the nonlinear law is a separate completion |
| B4c | CONFIRMED false displayed commutator; no downstream A1 result uses it |
| B4d | PARTLY CONFIRMED formal overclaim; positive-shell kinematics can be used, but the negative-energy density-matrix argument is incomplete |
| B4e | PARTLY CONFIRMED formal overclaim; fixed-E A1 survives, while the claimed moving-frame closure was not derived |
| B5 | PARTLY CONFIRMED major implementation/presentation defect, not a failed KL trace derivation |
| M1 | CONFIRMED incomplete convention bridge and local factor-two typo; no downstream equation changes |
| M2 | CONFIRMED localized missing factor of i |
| M3 | CONFIRMED false transverse-response loop in the stated sector |
| M4 | PARTLY CONFIRMED, minor; Dynes leakage is real, but the general coefficient identities are already in the paper |
| M5 | CONFIRMED benchmark-scope/claim error; the external audit's superposition defense is false |
| M6 | REFUTED: the public repository had all seven scripts before this review |
| M7 | NARROWED to wording: the A1 ordering skeleton survives proximity, not the unchanged BCS coefficient |
| M8 | CONFIRMED malformed bibliography entry; minor metadata repair |
| M9 | PARTLY CONFIRMED minor scope qualifiers |

The decisive second-round checks are recorded here so that another agent does
not repeat the audit's mistakes.

### B1 audit failure: the claimed sub-percent rescue used a clamped edge

The correct retarded Usadel equation in the manuscript's convention is
\[
  \hbar D\,\theta''
  =-2i\,[E\sinh\theta-\Delta(y)\cosh\theta],
\]
so a varying real gap does not imply a real above-gap \(\theta\). The external
audit correctly refuted its judge's real-\(\theta\) theorem, but its recommended
action then proposed inserting that same false theorem and porting the judge's
wrong no-\(i\) solver.

The verifier's own `window_average.py` has a second, independent defect. It
computes \(R_{\rm exc}\) only down to \(3\times10^{-6}\) but integrates to
\(10^{-8}\) with `np.interp`, which silently holds the resistance constant
below the first computed point. Direct solutions of the correct complex BVP
give, for its \(w=3\xi\) model,

| \(\epsilon=(E-\Delta)/\Delta\) | \(R_{\rm exc}\) |
|---:|---:|
| \(10^{-7}\) | \(43.0~\mu\mathrm m\) |
| \(10^{-8}\) | \(144.8~\mu\mathrm m\) |
| \(10^{-9}\) | \(474.1~\mu\mathrm m\) |

The local behavior approaches \(R_{\rm exc}\propto\epsilon^{-1/2}\). Together
with \(N_1\propto\epsilon^{-1/2}\), the proposed box-window average is at
least logarithmically cutoff-sensitive if this asymptote persists. Repeating
the interpolation with actual BVP points changes the reported averages from
approximately \(0.10\)--\(0.37~\mu\mathrm m\) to

| window | cutoff \(10^{-6}\) | cutoff \(10^{-8}\) | cutoff \(10^{-9}\) |
|---|---:|---:|---:|
| \(kT=0.1\Delta\) | 0.106 | 0.201 | 0.254 \(\mu\mathrm m\) |
| \(kT=0.01\Delta\) | 0.331 | 0.636 | 0.805 \(\mu\mathrm m\) |

These values can still be small compared with a \(50~\mu\mathrm m\) outer
length for a chosen physical cutoff, but they do not prove a cutoff-insensitive
sub-percent theorem. Nor does a window average establish the factorized
energy/spatial mode used in C1. A broadening mechanism, energy-relaxation
closure, or energy-resolved eigenmode calculation remains necessary. Equal
outer gradients require only conserved current and equal outer conductivities;
they do not establish continuity of \(f\) across the spectral layer.

### B4e audit failure: the transformed diffusion operator was incomplete

For \(F(\xi,x)=f(E(\xi,x),x)\), define
\[
  \mathscr D_x=\partial_x\big|_\xi
  -A_x\partial_\xi,
  \qquad
  A_x=\frac{\Delta\,\partial_x\Delta}{\xi}.
\]
The exact coordinate transform of the fixed-energy operator is
\[
  D_N\,\mathscr D_x\mathscr D_x F,
\]
not merely
\(D_N\partial_x[\partial_xF-A_x\partial_\xi F]\). The external audit's
recommended equation omits the outer \(-A_x\partial_\xi\) action. Its claim
that connection effects are automatically next-gradient-order is also not a
proof: the connection is first order and its divergence contributes at the
same second spatial-gradient order as diffusion. This does not alter the
complete fixed-E A1 equation; it shows only that the manuscript and the audit
have not supplied the claimed intrinsic moving-frame reduction.

### M4 audit correction: the Dynes edge value is not exactly one half

For \(z=E+i\Gamma\), the exact ideal Dynes coefficient at \(E=\Delta\) is
\[
  \mathcal D_L(\Delta)
  =\frac12\left(1+
  \frac{\Gamma}{\sqrt{\Gamma^2+4\Delta^2}}\right),
\]
not exactly \(1/2\) at finite \(\Gamma\), as the external verifier states.
It tends to \(1/2\) as \(\Gamma/\Delta\to0\); for
\(\Gamma=0.01\Delta\) it is \(0.50249997\). The deep-subgap result
\(\mathcal D_L(0)=\Gamma^2/(\Delta^2+\Gamma^2)\) and the physical conclusion
that finite Dynes broadening creates a leaky rather than hard zero-flux face
remain correct.

### M5 audit failure: passive response does not prove net self-focusing

The gap closure makes the combined map \(L[\Delta[f]]f\) nonlinear. Linearity
of transport at a fixed prescribed \(\Delta\) therefore does not license the
external audit's superposition claim for a population digging its own well:
linearizing the coupled problem also produces
\((\delta L/\delta\Delta)\,\delta\Delta[f]\,f\), which the passive-probe test
omits.

A direct one-population counter-test used the paper's own heavy initial
population, the same \(80\times0.25\) ns dynamic loop and the same disclosed
omission of spectral-flow advection, and tracked the width of each model's
conserved density. The scratch calculation used the benchmark defaults
\(N_E=24\), \(N_x=201\), \(L=100~\mu\mathrm m\), target well depth 0.05,
heavy center \(40~\mu\mathrm m\), and heavy width \(10~\mu\mathrm m\). For
each model it advanced that same population, recomputed the gap from that
population after every step, and measured
\(\sigma_x=[\langle(x-\langle x\rangle)^2\rangle_{N_1^p f}]^{1/2}\); no
separate probe entered either the transport or closure.

At \(E=1.082\Delta_0\), every packet broadened:

| model | initial width | width at 20 ns |
|---|---:|---:|
| A1 | 7.306 | 12.322 \(\mu\mathrm m\) |
| C (\(q=-1\)) | 7.071 | 12.149 \(\mu\mathrm m\) |
| B (\(q=-2\)) | 7.071 | 9.520 \(\mu\mathrm m\) |

The energy-integrated widths likewise increased (A1 7.237 to 13.218,
C 7.071 to 13.362, B 7.071 to 11.178 \(\mu\mathrm m\)), and even the first
0.25 ns width derivative was positive for all three. Thus the legacy drift
opposes diffusion and can reduce broadening, but this benchmark does not show
net compression. The passive inward-drift result remains valid; the abstract
and conclusion should not call it demonstrated reciprocal self-focusing
without a clearly defined focusing observable and a coupled calculation.

### M6 correction: the public repository is complete

Direct Git transport, a fresh clone, and commit-history inspection give public
`main`/`HEAD` = `7116d6eb9b329a8056b6944a43e15dfe296c0229` with three commits:

- `7a4a913` (2026-06-10): initial six scripts
- `8ffcdf2` (2026-07-07): adds `verify_tdep_inhomogeneous.py` and updates the README
- `7116d6e` (2026-07-07): strengthens `verify_nonadiabatic.py`

The current tree and README contain all seven scripts, and their normalized
Git blobs match the seven files at manuscript commit 8598056. The earlier
one-commit/six-script browser rendering was stale. M6 is not a manuscript
finding; recording or tagging the public hash remains optional good practice.

## Executive verdict

The central static result survives:

> In the stated homogeneous-material, real-gap, charge-balanced,
> ideal-local-BCS dirty sector above the gap, the longitudinal fixed-energy
> equation conserves \(N_1 f\) and carries the undressed spectral current
> \(-D_{\mathrm N}\nabla f\). The Born/BRT closure
> \(N_1D(E)=D_{\mathrm N}\), the ideal coefficients
> \(\mathcal D_L=1\) and \(\mathcal D_T=N_1^2\), and the absence of the
> legacy DOS-gradient drift remain supported.

For a spatially varying local spectrum, this statement additionally
requires control of the local-spectral expansion, for example
\[
\rho_\Delta\sim
\frac{\hbar D_{\mathrm N}|\nabla^2\vartheta|}{W}\ll1.
\]
That control is nonuniform near the ideal gap edge.

The paper is not ready for the planned C1/C2 insertion or submission pass.
The review found no reason to reverse the core A1 result, but it found
multiple errors or overstatements in the claimed exact equivalence,
time-dependent extensions, supercurrent scaling, general interface law,
proximity extrapolation, and benchmark interpretation.

Most importantly, the ready-to-insert C1 result is not yet a microscopic
correction to Riwar-Catelani. Its \(7\to32~\mu{\rm m}\) result is conditional
on a sharp-interface transfer law that has not been derived for the actual
coherence-scale proximity layer.

B1 is the only surviving blocker in the narrow sense of preventing the next
named insertion. B2, B3, B4c, M1, M3, and M5 are mandatory pre-submission
corrections. B5 is a serious normalization/presentation hazard, but its KL
trace derivation and the supplement's primary side-labeled law are correct.

## Severity convention

- BLOCKER: invalidates the premise of a named next action. In this review B1
  blocks the prepared C1 insertion.
- MAJOR: a scientific, mathematical, evidentiary, or reproducibility claim
  must be corrected before submission, but the core static A1 result can
  survive.
- MINOR: localized scope, notation, provenance, or presentation correction.

## Highest-priority findings

### B1. BLOCKER: The prepared C1 quantitative correction is conditional, not derived

Locations:

- PRESUBMISSION-PACKAGE-2026-07-07.md, section 2
- paper.tex:1025-1048
- paper.tex:1082-1090
- supplement.tex:2584-2697
- supplement.tex:2776-2867

The C1 draft treats the step in Riwar-Catelani as an interface between two
bulk BCS regions and obtains continuity of \(f\) and the undressed
\(D_{\mathrm N}\partial_y f\) flux by taking \(G_N\to\infty\) in the
paper's Kupriyanov-Lukichev Robin condition.

That argument is not controlled for the cited device:

1. Riwar's lateral step is a coarse-grained, coherence-scale proximity
   region in one continuous S film above an S' layer. It is not the
   low-transparency barrier for which the displayed KL relation was
   derived.
2. The manuscript's local-BCS sector requires slow variation on the
   coherence length and explicitly excludes proximity-modified spectra.
3. Sending \(G_N\to\infty\) is only a formal Robin limit of a
   low-transparency boundary law. It is not a derivation of a transparent
   proximity interface.
4. The proximity extension itself gives the correct starting current,
   \[
   j_L(E,y)=-\sigma_N(y)\mathcal D_L(E,y)\partial_y f_L,
   \qquad
   \mathcal D_L=\cos^2[\operatorname{Im}\theta(E,y)].
   \]
   Across a finite spectral layer this gives
   \[
   f_L^+-f_L^-=-j_L R_\xi(E),\qquad
   R_\xi(E)=\int_{\rm layer}
   \frac{dy}{\sigma_N(y)\mathcal D_L(E,y)}.
   \]
   Endpoint continuity requires \(R_\xi(E)\) to be negligible. That has
   not been shown uniformly in the near-edge cold window used to obtain
   the large factor \(\beta\).

What remains established:

- Riwar Appendix A starts from the legacy placement.
- A1 and the legacy placement have the same bulk rate inside each
  constant-gap outer region.
- The generalized proximity energy current is conserved through a
  source-free stationary layer.
- Equal outer gradients follow from current conservation when the two outer
  conductivities are equal, irrespective of the layer resistance. That fact
  does not imply continuity of (f); the layer resistance instead fixes the
  endpoint jump.
- The C1 changes to the eigenvalue equation, relaxation-limited rate,
  and \(d_S\approx32~\mu{\rm m}\) are algebraically correct conditional on
  that unproved limit.

Required before insertion:

- Solve the retarded Usadel problem through the laterally terminated
  S/S' bilayer, or prove that the energy-weighted \(R_\xi(E)\) is
  negligible over the active cold window.
- Derive the resulting energy-resolved transfer law.
- Reintegrate it into the \(x_{\rm qp}\) matching and rerun the
  eigenvalue/minimal-length calculation.

Disposition: do not insert the C1 subsection or its \(7\to32~\mu{\rm m}\)
claim as presently drafted. A conditional sharp-interface illustration
could be retained only if it is labeled conditional and not presented as a
correction to the published device numbers.

### B2. MAJOR: The nonadiabatic verifier omits a same-order Keldysh correction

Locations:

- supplement.tex:2868-2945
- paper.tex:2355-2362
- verify_nonadiabatic.py:184-229

Write the solved spectral correction as
\[
g^R=g_0^R+\hbar^2 d^R,\qquad
g^A=g_0^A+\hbar^2 d^A.
\]
The script solves \(d^{R/A}\) but computes the kinetic residual using a
Keldysh function built only from \(g_0^{R/A}\). Consistency requires
\[
\delta g^K=\hbar^2(d^R h-hd^A).
\]

Writing
\[
[L_0,g_0^R]_\star=\hbar^2 r\tau_1+O(\hbar^3),\qquad
r=\frac{3E\Delta^2\ddot\Delta}{4W^5},
\]
the solved coefficient satisfies
\[
[L_0,d^R]=-r\tau_1,\qquad
d^A=-d^R.
\]
For \(h=f_L\openone\),
\[
\delta g^K=2\hbar^2 f_Ld^R
\]
and its missing contribution to the verifier's unnormalized
\(\tau_1\) trace is
\[
-\frac{3iE\Delta^2\ddot\Delta}{W^5}f_L.
\]
This cancels the final positive term in the current M1a pin exactly.

Consequences:

- For a gap-slaved distribution \(f_L=G(W)\), the consistently completed
  \(O(\hbar^2)\) branch-coherence source vanishes identically.
- A generic independently driven source survives.
- The static-gap source proportional to \(\partial_t^2f_L\) survives.
- The plain and \(\tau_3\) projections and the no-L/T-mixing result are
  unaffected at this order.

Disposition: repair verify_nonadiabatic.py before relying on its
"corrected-claims" block; revise the SM and main-text echoes.

### B3. MAJOR: The claimed \(O(Q^3)\) supercurrent mixing is nonuniform

Locations:

- supplement.tex:2698-2775
- supplement.tex:3259-3272
- verify_supercurrent.py

The displayed outer expansion
\[
S(E;Q)=-\frac{4Q\Gamma E^2\Delta^2}
{(E^2-\Delta^2)^{5/2}},\qquad
\Gamma=2\hbar D_{\mathrm N}Q^2,
\]
is correct at fixed \(E>\Delta\) with
\(W^3\gg\Gamma E\Delta\). It is not uniform at the depairing-rounded
edge.

In the edge layer:

- energy width is \(O(Q^{4/3})\);
- peak height is \(O(Q^{-1/3})\);
- integrated weight is \(O(Q)\), not \(O(Q^3)\).

Distributionally on the positive-energy axis,
\[
\frac{S(E;Q)}{Q}\longrightarrow
-\pi\Delta\,\delta(E-\Delta).
\]
Therefore a smooth energy integral is generally linear in \(Q\), unless
its weight vanishes at the edge.

Disposition: label the current formula as the fixed-energy outer result.
Replace the global "mixing begins at \(Q^3\)" claim with separate
pointwise and energy-integrated statements. Extend the verifier to the
physical root in the edge layer if the integrated claim is retained.

### B4. MAJOR: The claimed exact equivalence is overstated in several places

Locations:

- paper.tex:746-790
- paper.tex:799-855
- paper.tex:894-910
- paper.tex:2611-2639
- supplement.tex:1496-1886
- supplement.tex:2967-3258

#### Full-angle parity conflict

At fixed momentum direction,
\[
f_L(\hat k)=1-F^{\rm even}(\hat k)+\phi_T^{\rm odd}(\hat k),
\]
\[
\frac{f_T'(\hat k)}{\lambda}
=\phi_T^{\rm even}(\hat k)-F^{\rm odd}(\hat k).
\]
Thus \(f_T'=\lambda\phi_T\) is an even-harmonic identity only. The paper
actually states this guard at paper.tex:269-277 and 773-791 and explicitly
retains the hidden odd harmonic at 894-906 and 1150-1192. The surviving
defect is narrower: shorthand at 746-755 and 799-855 drops the qualifier,
and the bracket at 907-910 invokes the momentum-space inverse at
\(f_T'=0\) even though the odd momentum-space harmonic survives.

At charge balance, \(\phi_T=0\), but a common trajectory occupation
\(f=f_0+\hat p\cdot{\bf a}\) has
\[
f_L=1-2f_0,\qquad
f_T'=-2\lambda\,\hat k\cdot{\bf a}.
\]
The odd transverse harmonic therefore survives. This is a notation and
representation-reference defect, not a lost current channel or a failure of
the scalar dirty reduction.

#### Nonlinear recombination overclaim

The displayed Kopnin pair is linearized: the longitudinal drive contains
\(\partial_E f_L^{(0)}\), and the displayed transverse row omits
\(\partial_t f_T'\). The adopted nonlinear branch Boltzmann equation is an
independent adiabatic completion, not a literal algebraic recombination of
those two printed rows. The manuscript already discloses this at
paper.tex:358-362 and 1733-1740, so the remaining problem is the stronger
*recombine* and *exact change of variables* wording, not a hidden nonlinear
derivation error.

#### Projection-average commutator

For antipodal relabeling \(R_sX(\hat p)=X(s\hat p)\),
\[
\langle R_sX\rangle_{\hat p}=\langle X\rangle_{\hat k}.
\]
The full Fermi-surface average is measure-preserving, so the displayed
\[
[\mathcal P_{\rm qp},\langle\cdot\rangle]\check g\ne0
\]
is false as written. Order sensitivity belongs to retaining, solving, and
eliminating the first harmonic, not to the bare full-sphere average.

#### Branch-projector construction

The longitudinal Keldysh quantity \(f_L\) is an amplitude, not an
occupation. At the two BdG poles particle-hole symmetry gives opposite
longitudinal amplitudes:
\[
q_+=f_L(E_\xi),\qquad q_-=-f_L(E_\xi).
\]
The physical positive-energy occupation is
\[
n_\xi=\frac{1-f_L(E_\xi)}2.
\]
The current branch-projector discussion assigns the positive-energy
amplitude too broadly and incorrectly argues that charge balance removes
the diagonal branch difference. The positive-shell extraction can be
rescued, but the negative-energy extension and the "exact" wording need
repair.

#### Moving-\(\xi\) spatial representation

For \(F(\xi,{\bf r})=f(E(\xi,{\bf r}),{\bf r})\),
\[
\nabla_E f=
\left[\nabla_\xi-\frac{\Delta}{\xi}\nabla\Delta\,\partial_\xi\right]F.
\]
The moving-\(\xi\) representation therefore contains spatial connection
and mixed space-energy terms. Appendix A explains the clean-force
cancellation that removes a new physical drift, but the SM does not
derive the intrinsic dirty moving-frame current it claims to close.

Disposition: keep the fixed-energy A1 result. Remove the false bare
projection/average commutator, repair the representation shorthand, and
narrow exactness claims to the proved sector. The route end points still
agree on static ideal-BCS A1; this group is major because a displayed equation
and several formal-proof claims are wrong, not because the core operator
failed.

### B5. MAJOR: The general KL boundary law mixes current normalizations

Locations:

- paper.tex:2645-2706
- supplement.tex:2584-2697

The consistent conductivity-weighted longitudinal current is
\[
\mathcal J_L
=-\sigma_i\mathcal D_L^{(i)}\partial_n f_{L,i}
=G_N\mathcal W_L(f_{L,1}-f_{L,2}),
\]
\[
\sigma_i=2e^2N_0^{(i)}D_{\mathrm N}^{(i)}.
\]

The diffusion-normalized current on side \(i\) is
\[
j_L^{(i)}
=-D_{\mathrm N}^{(i)}\mathcal D_L^{(i)}\partial_n f_{L,i}
=\frac{\mathcal J_L}{2e^2N_0^{(i)}}.
\]

The manuscript uses the same \(j_L\) symbol for both objects. For unequal
materials only \(\mathcal J_L\) is continuous; the diffusion-normalized
currents differ if the \(N_0^{(i)}\) differ.

Additional scope corrections:

- The displayed weight is the zero-phase result. At phase difference
  \(\delta\chi\),
  \[
  \mathcal W_L=N_1^{(1)}N_1^{(2)}
  -N_2^{(1)}N_2^{(2)}\cos\delta\chi.
  \]
- Converting \(f_L=1-2f\) requires redefining the occupation current by
  \(-1/2\).
- Finite \(G_N\) permits a jump; it does not require one at zero current.
- \(G_N\to\infty\) is a formal Robin limit, not a controlled
  high-transparency KL derivation.

The KL matrix relation, the zero-phase trace identities, and the side-labeled
conductivity-weighted implementation in supplement.tex:2597-2605 and
2654-2663 are correct. The defect is the subsequent symbol reuse, the dropped
side label on \(g_N\), and the compressed main-text presentation in a section
that advertises dissimilar-material traps. The zero-phase ansatz is explicit
in the supplement but should also qualify the main-text law.

Disposition: repair units, symbols, side-dependent normalizations, and phase
scope before using the law as a general interface theorem. This is a serious
implementation hazard, but it is not a failed KL derivation and does not
invalidate the same-material interface benchmark.

## Additional findings and dispositions

### M1. MAJOR: The starting-equation convention bridge is incomplete

Locations:

- supplement.tex:228-311
- supplement.tex:1080-1105

Changing only the Wigner/Moyal kernel cannot flip the zeroth-order
commutator sign. The supplement's statement that its starting equation is
related to the main equation by kernel reversal is therefore incomplete.

More precisely, changing only the sign in the Moyal exponential while
holding the symbols fixed leaves the ordinary commutator unchanged. Reversing
the full Fourier-transform kernel also maps the energy symbol \(E\to-E\).
The external audit's judge incorrectly conflated these operations when it
said every kernel reversal acts only at \(O(\hbar)\); either interpretation
still needs the additional gap, self-energy, or conjugation bookkeeping that
the manuscript omits.

Two full maps are available:

- antipodal trajectory relabeling \(\hat p\to-\hat p\); or
- full complex conjugation with the induced propagator, self-energy, and
  retarded/advanced relabeling.

The text mentions an "equivalently conjugated equation" but does not give
the map and elsewhere says the products are the same.

The scalar-symbol Moyal commutator also has a local factor-two typo:
\[
[A,B]_\star=[A,B]+i\hbar\{A,B\}_{\rm PB}+\cdots,
\]
not \(i\hbar\{A,B\}_{\rm PB}/2\). The later explicit four-product matrix
formula is correct, so the final Usadel equation is not changed by this
schematic typo.

Disposition: print one complete map, preferably the antipodal relabeling,
and reconcile the three product-identification statements. This is a major
conventions/documentation repair in a derivation paper, not evidence that the
downstream Usadel or A1 equations are wrong.

### M2. MINOR: The channel-dictionary relaxation trace is missing a factor of i

Locations:

- paper.tex:2005-2029
- supplement.tex:2133-2157

With
\[
\hat\Delta=-i\Delta\tau_2,\qquad
g^A=-\tau_3g^{R\dagger}\tau_3,
\]
direct algebra gives
\[
\frac14\operatorname{Tr}
[\hat\Delta(g^R+g^A)]
=i\Delta\,\operatorname{Im}\sinh\theta.
\]
The raw trace is imaginary. In the manuscript's kinetic convention the
real relaxation coefficient is
\[
\mathcal R=
\frac{i}{4}\operatorname{Tr}
[\hat\Delta(g^R+g^A)]
=-\Delta\,\operatorname{Im}\sinh\theta,
\]
up to the explicitly chosen side of the kinetic equation.

On a uniform or zeroth-order local-BCS background, the ideal above-gap
zero and subgap positive relaxation are recovered after the factor is
included. For a spatially varying spectrum, the same-order retarded
correction can produce a gradient-induced nonzero \(\mathcal R\); that is
the \(\tau_3\)-trace rewriting of the transverse curvature term, not an
additional kernel term. The missing factor remains a localized convention
defect in the main table and SM.

### M3. MAJOR: The claimed transverse \(O(\dot\Delta)\) response loop is absent

Locations:

- supplement.tex:2023-2131
- supplement.tex:3118-3142
- verify_fT.py
- verify_nonadiabatic.py

The derived transverse equation is homogeneous in \(f_T\):
\[
\left[
\partial_t+\frac{\Delta\dot\Delta}{E}\partial_E
+\frac{\Delta\dot\Delta}{E^2}
\right]f_T
=\text{diffusion}+\text{relaxation proportional to }f_T.
\]
In the stated real-gap, no-superflow, particle-hole-symmetric sector,
\[
J_T[f_L,0,n]=0.
\]
Zero initial and boundary imbalance therefore imply \(f_T=0\) through
the computed order. A moving gap advects or dilutes existing imbalance;
it does not create it.

Disposition: replace the claimed generated
\(f_T=O(\dot\Delta)\to O(\dot\Delta^2)\) loop with the stronger supported
statement that charge imbalance gives no feedback in this sector. State
separately which beyond-sector ingredient could create a nonzero loop.

### M4. MINOR: The Dynes remark is internally inconsistent and understates leakage

Locations:

- paper.tex:1989-1994
- paper.tex:2099-2114
- supplement.tex:2946-2966

Use the normalized continuation
\[
z=E+i\Gamma,\quad
g_\Gamma^R=\frac{z}{\sqrt{z^2-\Delta^2}},\quad
f_\Gamma^R=\frac{\Delta}{\sqrt{z^2-\Delta^2}},
\]
with the causal advanced partner.

The manuscript defines \(N_1=\operatorname{Re}g^R\) and
\(N_2=\operatorname{Re}f^R\), so calling
\(N_1(E+i\Gamma)\) and \(N_2(E+i\Gamma)\) complex spectral functions is
a type error.

For a complex spectral angle,
\[
\mathcal D_L=\frac{1+|g|^2-|f|^2}{2},\qquad
\mathcal D_T=\frac{1+|g|^2+|f|^2}{2},
\]
\[
\mathcal D_L\mathcal D_T=N_1^2.
\]
The ideal individual equalities
\(\mathcal D_L=1\) and \(\mathcal D_T=N_1^2\) fail, but the general
product identity survives.

Finite broadening gives \(\mathcal D_L>0\) below the ideal gap. It
therefore removes the hard zero-flux face and creates a leaky subgap
problem; it is not merely numerical smoothing.

The manuscript already prints the general complex-angle coefficient and
product identities. The required correction is to cross-reference them,
repair the spectral-function terminology, and state that a finite-\(\Gamma\)
edge is physically leaky. No result in the paper was computed with this
Dynes replacement, so this is minor rather than a failed derivation.

### M5. MAJOR: The benchmark demonstrates passive-tracer drift, not reciprocal
nonlinear self-focusing

Locations:

- paper.tex:121-123
- paper.tex:2774-2930
- paper.tex:3002-3005
- validation/diffusion_operators/self_consistent_feedback.py

The code:

- creates separate heavy and probe states;
- computes and updates the gap from the heavy state only;
- never includes the probe in the gap closure;
- measures probe center-of-mass drift;
- omits energy-space spectral-flow advection in dynamic mode.

The probe is not uniformly negligible: its peak is one half of the heavy
peak and exceeds the local heavy background at the probe center.

What is established:

- static A1 has zero DOS-gradient tracer drift to numerical roundoff;
- legacy placements attract a passive tracer toward a gap well generated
  by a maintained population;
- the static analytic drift comparison is quantitatively successful.

What is not established:

- net focusing or compression of a population in the well generated by that
  same population;
- a fully coupled one-population calculation including energy-space
  spectral-flow advection.

Dynamic mode does update the heavy population and its gap every step, and the
A1 probe residual is consistent with the explicitly omitted spectral-density
reweighting rather than a DOS-gradient drift. That does not turn the probe into
a reciprocal self-response. In the direct one-population width test recorded
above, A1, C, and B all broaden; the negative-\(q\) drift only reduces the
broadening for B at this benchmark point.

Disposition: replace demonstrated self-focusing with passive inward-drift
language, or define a focusing observable and show it in a fully coupled
one-population calculation. Making the probe genuinely weak would also make
the stated benchmark design accurate.

### M6. REFUTED: The public verification repository is complete

Locations:

- paper.tex:3025-3031
- CLAUDE.md review record
- https://github.com/Soren-O/qp-diffusion-verification

The initial browser rendering used by the first review was stale. Direct Git
transport and a fresh clone show public `main` at `7116d6e`, with three
commits. Commit `8ffcdf2` added `verify_tdep_inhomogeneous.py` and updated the
README on 2026-07-07; `7116d6e` then strengthened
`verify_nonadiabatic.py`. The repository currently contains all seven scripts,
and their normalized Git blobs match the local files at manuscript commit
8598056.

Disposition: no manuscript correction. Optionally record the public commit
hash and create an archival tag or DOI before submission.

### M7. MINOR: Proximity does not leave the full BCS A1 operator "intact"

Location: supplement.tex:2776-2867

The A1 continuity/placement skeleton survives: there is no inserted DOS
power and no L/T mixing on a stationary zero-current spectral background.
The full BCS operator does not remain unchanged:
\[
j_L=-D_{\mathrm N}\mathcal D_L\nabla f_L,\qquad
\mathcal D_L=\cos^2(\operatorname{Im}\theta).
\]
Spatial \(\mathcal D_L\) variation generates a real spectral-mobility
drift. This is not the legacy DOS-placement artifact, but it is a changed
operator.

The coefficient identities hold for any normalized complex angle. A
physical kinetic equation additionally requires a stationary spectral
angle that solves the retarded Usadel problem with its boundary
conditions and self-energies.

The same paragraph that says *intact* immediately prints the promoted
coefficient and calls the resulting mobility drift a new ingredient. The
physics is present; the opening sentence contradicts its own explanation.

Disposition: say the A1 ordering skeleton generalizes, not that the full BCS
operator survives intact.

### M8. MINOR: catelani2019 is a malformed composite bibliography entry

Locations:

- refs.bib:253-260
- paper.tex:139
- paper.tex:913

The local entry combines the intended title with unrelated authors,
journal, locator, and DOI. The intended paper is:

G. Catelani and D. M. Basko, "Non-equilibrium quasiparticles in
superconducting circuits: photons vs. phonons," SciPost Physics 6, 013
(2019), DOI 10.21468/SciPostPhys.6.1.013.

The local DOI 10.1103/PhysRevB.99.174512 belongs to "Circuit quantization
in the presence of time-dependent external flux."

Disposition: replace the entry metadata while retaining the key and its
two uses.

### M9. MINOR: Macroscopic and diagnostic conservation claims need static/scope
qualifiers

Locations:

- paper.tex:2202-2229
- paper.tex:2981-2987

For a shape-preserving profile \(f=A({\bf r})\phi(E)\),
\[
D_{\rm eff}=D_{\mathrm N}
\frac{\int f\,dE}{\int N_1 f\,dE}
\]
is correct. A generic one-parameter local-equilibrium closure instead
uses susceptibility weights:
\[
D_{\rm eff}=D_{\mathrm N}
\frac{\int \partial_\alpha f\,dE}
{\int N_1\partial_\alpha f\,dE}.
\]
The conclusion should not present the shape-preserving formula as generic
local equilibrium.

Likewise \(N_1^2 f\) is a static-spectrum diagnostic density for A2. It
is not conserved by the physical moving-shell spectral velocity when
\(\dot\Delta\ne0\). Only the \(p=1\) weight has the derived physical
time-dependent completion.

## Refuted or narrowed candidate findings

These were found in the first 44-unit pass but did not survive independent
rebuttal in their original form.

### R1. No missing moving-edge term in A1

The conservative law, Reynolds lower-limit cancellation, and spatial
zero-flux or matched-region condition are already present across
paper.tex:1512-1522, 1693-1726, 2099-2114, and the SM moving-bin
derivation. No additional A1 source is missing.

### R2. The local-BCS spatial spectral correction does not change the
displayed transverse kernel at the retained order

The same-order retarded correction induces a Keldysh image-channel term
that cancels the local-BCS image residual. It leaves the displayed kernel
equation unchanged; kernel corrections enter at higher spatial-gradient
order. A scope parameter near the edge would still improve the text.

### R3. The Dynes product identity does not fail

Finite \(\Gamma\) invalidates the ideal individual coefficient
equalities, not the exact general identity
\(\mathcal D_L\mathcal D_T=N_1^2\).

### R4. A conjugation route between starting conventions exists

The initial objection that no conjugation was mentioned was too strong.
The supplement does mention it. The remaining finding is that the
actual map is not displayed and kernel reversal alone is insufficient.

### R5. Dynamic A1 residual is bookkeeping within the truncated benchmark

For (q=0), the transport contribution to the conserved-density center of
mass telescopes for an arbitrary prescribed (N_1(x,t)). In the implemented
equation, the remaining A1 probe motion therefore comes from reweighting by
the updated (N_1), exactly the spectral-flow bookkeeping the benchmark says
it omits. This validates the narrow attribution inside the truncated model;
it does not validate reciprocal self-focusing or the omitted physical
spectral-flow dynamics.

## Core claims that survived

The following claims were rederived independently and survived rebuttal.
For statements involving spatially varying local spectra, survival includes
the control condition \(\rho_\Delta\ll1\) stated in the executive verdict.

1. \(N_1=E/\sqrt{E^2-\Delta^2}\) is the positive-energy BCS shell
   Jacobian per radial branch.
2. \(v_g=v_F/N_1\) and \(N_1v_g=v_F\).
3. The Born coherence factor and final-state DOS force
   \(\tau_{\rm tr}=N_1\tau_{\mathrm N}\).
4. \(\ell_{\rm tr}=v_g\tau_{\rm tr}=\ell_{\mathrm N}\).
5. \(D(E)=D_{\mathrm N}/N_1\) and \(N_1D=D_{\mathrm N}\).
6. The scalar route therefore gives
   \[
   L_D[f]=\frac1{N_1}\nabla\cdot(D_{\mathrm N}\nabla f).
   \]
7. The ideal local-BCS Usadel traces give
   \(\mathcal D_L=1\) above the gap, zero below, and
   \(\mathcal D_T=N_1^2\) above the gap.
8. The fixed-energy first-Moyal projection gives
   \[
   \partial_t(N_1f)+
   \partial_E(N_2\dot\Delta f)
   =\nabla\cdot(D_{\mathrm N}\nabla f)+N_1I_{\rm coll}.
   \]
9. DOS continuity converts this exactly to the advective form inside the
   ideal BCS domain.
10. A1 has no smooth bulk DOS-gradient drift in this sector.
11. The legacy placements genuinely produce different solutions on
    prescribed inhomogeneous profiles.
12. The advanced convention
    \(g^A=-\tau_3g^{R\dagger}\tau_3\) remains supported.

## Per-unit scorecard

Only findings promoted in the ranked sections above should be treated as
surviving substantive findings. Lower-level notes in this table are
review suggestions unless separately promoted.

### Main paper

| Unit | Scope | Review result |
|---|---|---|
| M01 | Front matter and abstract | MAJOR: abstract scope and passive-probe/self-focusing overclaim |
| M02 | Introduction before I.A | MAJOR: Riwar proximity-scope mismatch; MINOR: malformed catelani2019 |
| M03 | I.A Quasiparticle kinetic equations | MINOR: full-energy DOS convention and \(J_T\) sign definition clarity |
| M04 | I.B Quasiparticle Boltzmann-like equations | MINOR: representation shorthand and nonlinear-completion wording |
| M05 | I.C Usadel equation | MINOR: dirty hierarchy stated too narrowly |
| M06 | II opener and BCS kinematics | PASS |
| M07 | Clean scalar Boltzmann form | MINOR: inelastic collision-symbol scope |
| M08 | Angular averaging after branch projection | MINOR: closure valid; qualify arbitrary-kernel harmonic gap |
| M09 | Transport-time closures | PASS |
| M10 | Local density conservation | PASS after edge rebuttal; normalization wording remains optional |
| M11 | Matrix dirty limit | MINOR: selective Moyal truncation needs clearer power counting |
| M12 | Longitudinal distribution operator | MINOR: missing \(i\) in \(\mathcal R\); Dynes edge scope |
| M13 | Flux to scalar equation | MAJOR: proximity examples exceed sector; A2 conservation is static-only |
| M14 | Matrix spectral flow | MINOR: nonnegative shell-coordinate and mixed-gradient scope |
| M15 | Agreement with scalar route | MINOR: above-gap and assumption-list qualifiers |
| M16 | Candidate-operator taxonomy | MINOR: terminology only |
| M17 | Projection versus angular averaging | MAJOR: displayed nonzero commutator is false |
| M18 | Currents and boundary conditions | MAJOR: current notation/units, unequal materials, phase scope |
| M19 | Gap feedback | MINOR: edge exponent classification and \(x_{\rm qp}\) provenance |
| M20 | Benchmarks | MAJOR: passive tracer presented as reciprocal focusing; scalar-only interface test |
| M21 | Conclusion and availability | MAJOR: \(D_{\rm eff}\) scope and self-focusing recap; public script claim is current |
| M22 | Change-of-variables appendix | MINOR: body says general \(k\), appendix later sets \(k\to k_F\) |

### Supplemental Material

| Unit | Scope | Review result |
|---|---|---|
| S01 | Preamble, result, conventions | MAJOR: result over-scoped; Moyal-order language |
| S02 | Starting Eilenberger equation | MAJOR: incomplete convention bridge; Poisson factor two |
| S03 | Dirty hierarchy and ansatz | MINOR: independent small parameters and Moyal notation |
| S04 | Derivation outline | MINOR: conservative-form reference wording |
| S05 | Step 1 matrix Usadel | MINOR: mixed ordinary/Moyal product truncation not stated |
| S06 | Step 2 spatial flux | MINOR: no bare gradient source does not mean no transverse coefficient drift |
| S07 | Step 3 Moyal/plain trace | PASS at first order |
| S08 | Moving shell and branch projector | MINOR: negative-energy construction and adiabatic-exactness scope |
| S09 | Final longitudinal/transverse modes | MINOR: spectral-order objection refuted; relaxation-trace convention remains |
| S10 | Alternative branch-Boltzmann route | PASS |
| S11 | Scalar KL boundary conditions | MAJOR: normalization notation, material dependence, and phase scope |
| S12 | Supercurrent coupling | MAJOR: nonuniform \(Q^3\) claim |
| S13 | Proximity spectra | MINOR: A1 "intact" wording and local spectral-closure scope |
| S14 | Nonadiabatic dynamics | MAJOR: missing \(\delta g^K\) cancels pinned gap-slaved source |
| S15 | Dynes remark | MINOR: notation, cross-reference, and leaky-edge correction |
| S16 | Branch setup | MINOR: omitted transverse first-Moyal correction is trace-inert |
| S17 | Exact target identity | PASS |
| S18 | Direct plain trace | MINOR: collision-sector scope |
| S19 | Spectral identities | PASS |
| S20 | Transverse \(O(\dot\Delta^2)\) | MAJOR: claimed generated response is absent |
| S21 | Interpretation | MAJOR: scope and false transverse-order rationale |
| S22 | Spatial representation independence | MINOR: moving-\(\xi\) closure is asserted, not derived |

## CLAUDE.md records that are now stale

CLAUDE.md remains useful for the core notation and static A1 result, but
the following settled-record claims should no longer be used to reject
review findings:

1. The blanket statement that all 2026-07-07 fixes introduced no physics
   change.
2. The claim that the nonadiabatic corrected-claims block machine-verifies
   the gap-slaved \(O(\hbar^2)\) source.
3. The settled treatment of moving-shell/projector constructions as exact
   re-expressions without the negative-energy and spatial-connection
   qualifications.
4. The global \(O(Q^3)\) supercurrent-mixing statement without its edge
   nonuniformity.
5. The 2026-07-07 record that only the abstract review and a small set of
   mechanical items remained after that review round.

The even-harmonic parity guard in CLAUDE.md is not stale. The live
manuscript contains a few ambiguous shorthand references, but it also states
the guard and retains the hidden odd harmonic. Likewise, the decision to keep
the full Dynes algebra in the SM remains valid; the current remark needs a
terminology, cross-reference, and leaky-edge correction.

The following guards survived and should remain:

- \(N_0\) versus \(N(0)\) is a deliberate distinction.
- The advanced convention is correct.
- The gap-gauge \(i\) placement is correct.
- The displayed parity dictionary itself is correct.
- The hidden odd transverse harmonic is real and necessary.
- The BRT closure \(N_1D=D_{\mathrm N}\) is correct.
- The fixed-energy A1 conservation law is correct.
- The ideal moving-edge zero-current or matched-region handling is
  correct.
- The local xr-hyper limitation is not a manuscript defect.
- Engine Fischer figures and their solver history are unrelated to this
  paper review.

## Recommended action order

1. Put the C1/C2 package on technical hold. Derive or bound the proximity
   transfer law, including its edge cutoff, before presenting a quantitative
   correction to Riwar.
2. Repair verify_nonadiabatic.py and revise every dependent claim.
3. Repair the convention bridge, Moyal factor, representation shorthand,
   and false projection/averaging display from one consistent basis.
4. Disambiguate the KL current symbols, side-dependent material
   normalization, phase scope, and occupation-current conversion.
5. Rework the supercurrent-edge, transverse-response, Dynes, branch-
   projector, and moving-\(\xi\) extension claims.
6. Reframe the feedback benchmark as a passive-tracer inward-drift test;
   define and demonstrate a focusing observable before restoring stronger
   language.
7. Correct catelani2019. Optionally record/tag public verification commit
   7116d6e; no script push is required.
8. Apply minor scope/provenance fixes only after the physics-blocking
   items are settled.
9. Re-run all seven local scripts, rebuild both PDFs, and run a new
   independent rebuttal pass before submission.

## Human-only gates not attempted

The review did not decide or execute any of the following:

- D1: guarded abstract-sentence wording
- D2: abstract candidate selection
- D3: C1 subsection versus appendix placement
- D4: courtesy contact with Riwar/Catelani
- D5: caption-disclaimer scope
- D6: "wrong" versus "unsupported"
- D7: shipping the SM verification sentence
- the author's abstract review pass
- PyPI qpsim 0.0.1 publication
- contacting any author
- authorship, venue, or access-route decisions

## Final repository state at review completion

- Branch: fix/gpt-review-2026-07-05
- Commit: 85980561003b50fd95184dd90400b8ec67d61c44
- Manuscript/engine changes: none
- Review artifact: this untracked file, updated through the second-round audit
- Status record: CLAUDE.md has Claude's uncommitted status correction plus the
  refined second-round note described below
- Staging/commit: not performed
