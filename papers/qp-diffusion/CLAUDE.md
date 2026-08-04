# Merged Paper — Conventions and Review Notes

**Purpose.** This file records notation choices and physics points that have
already been derived, checked, and *deliberately* chosen, plus the current
state of the two decisions that remain open. It exists so a reviewer — human
or AI — can spend effort on genuinely new issues rather than re-flagging
settled ones. Everything below (outside the "Open gates" section) was vetted
across several adversarial review rounds; please do not report it as an error
without a substantively new argument. Finding *new* problems is welcome;
re-deriving these is not.

*(This file was consolidated on 2026-07-17 from five separate files —
`CLAUDE.md`, `README.md`, and three dated adversarial-review records — into
this one. The chronological blow-by-blow of who found what on which date has
been compressed; the physics conclusions and the still-open reasoning have
not. If you need the original review transcripts, they're in git history at
this path from before 2026-07-17.)*

---

## Scope and provenance

This is the single merged manuscript (assembled 2026-07-01). Base = paper3
wording/template (intro + scalar route + agreement); imported from paper1:
the Usadel-route derivation (longitudinal operator, D_L/D_T traces,
flux-to-scalar, time-dependent spectral flow), taxonomy, consistency
checks/benchmarks, conclusion, the entire Supplemental Material, figures, and
verify scripts; imported from paper2: the explicit change-of-variables
appendix. (The sibling `paper1/`, `paper2/`, `paper3/` directories this was
merged from are no longer present in this location — they were archived
sources only; nothing here depends on them existing.)

**Files:**
- `paper.tex` — the manuscript (REVTeX, ~56 pp preprint).
- `supplement.tex` — Supplemental Material (Sec. SI = detailed dirty-limit
  derivation, Sec. SII = branch-covariant verification, plus supercurrent /
  proximity / nonadiabatic appendices, plus the taxonomy and
  consistency-check/benchmark material — see the 2026-07-15 restructure
  note below). Cross-references between the two documents go through
  `xr-hyper` (`SM-` prefix in paper.tex, `M-` prefix in supplement.tex), so
  each document needs the other's `.aux`.
- `refs.bib` — shared bibliography.
- `figures/` — benchmark figures (regenerated from `~/Developer/qpsim`,
  `validation/diffusion_operators/`) and the reduction-routes roadmap
  (`routes_roadmap.tex`; rebuild with `make roadmap`).
- `verify_*.py` — symbolic/numeric computer-algebra checks (sympy).
  `verify_gA_convention.py` is the immutable regression baseline — never
  edit. Its "PAPER" column tests the superseded June-2026 audit convention
  (`gA = -(gR)^dagger`), kept as a historical anchor; the current manuscript
  uses its "CORRECTED" column (`gA = -tau3 (gR)^dagger tau3`).

## References

Kopnin materials live at `B:\AEinstein\Einstein\Documents\Soren\kopnin-numbered-equations\`:
the full book PDF ("Kopnin, Theory of Nonequilibrium Superconductivity
(2001).pdf"), numbered-equation transcriptions of Chapters 10 and 15 ("Local
Chapter Copies\Kopnin Chapter 10.tex" / "... 15.tex"), and the
transcription-audit tooling. A second copy of the Ch. 10/15 transcriptions
is at `G:\My Drive\qp-diffusion-handoff\kopnin\`. Chapters 10 and 15 are the
most relevant; Ch. 1 (the \(\nu(0)\) single-spin DOS definition) is in the
book PDF.

Cited-paper PDFs live in `G:\My Drive\Academic Texts\`
(Articles/Textbooks/Theses subfolders, "Author et al., Title (Year).pdf"
naming). As of 2026-07-17 this includes martinis2009 (plus its P(E)
supplement), segall2004, bardeen1959, catelani2019, fischer2023,
fischer2024, marchegiani2025, and freshly added copies of dong2022
(CC-BY, from MDPI), riwar2019 (arXiv v1), and kozorezov2002 (journal
PDF, obtained via institutional access). Still absent: riwar2016,
hosseinkhani2018, and goldie2013.

---

## Notation conventions (deliberate — not typos)

- **Two distinct direction symbols, and they are not interchangeable.**
  - \(\hat{\mathbf k}\) = momentum (wave-vector) direction.
  - \(\hat{\mathbf p}\) = trajectory (group-velocity) direction.
  - They differ by the branch sign: \(\hat{\mathbf p}=s\,\hat{\mathbf k}\),
    \(s=\operatorname{sgn}\xi_{\mathbf k}=\pm\). For hole-like quasiparticles
    (\(\xi<0\)) they are antiparallel.
  - The single-particle / canonical-Boltzmann derivation
    (`eq:intro_canonical_boltzmann` … `eq:intro_boltzmann_clean`) is written
    in the **momentum** direction \(\hat{\mathbf k}\). From the
    branch→trajectory relabelling onward — and throughout the **entire
    body** — the **trajectory** direction \(\hat{\mathbf p}\) is used. The
    Eilenberger equation is trajectory-resolved (\(\hat{\mathbf p}\)). This
    switch is intentional and is stated in the text.

- **\(\mathbf k\) is a wave vector and \(\hbar\) is kept explicit.**
  Consequences (all dimensionally checked; every kinetic term is
  \(1/\text{time}\)):
  - \(\mathbf v_g=\hbar^{-1}\partial E_{\mathbf k}/\partial\mathbf k
    =(\xi_{\mathbf k}/E_{\mathbf k})v_F\hat{\mathbf k}\).
  - Boltzmann drift is \((\mathbf F/\hbar)\cdot\partial_{\mathbf k}f\).
  - Momentum-space radial part carries \(\hbar\):
    \(\hat{\mathbf k}\,\partial_k f=\hbar\,\mathbf v_g\,\partial_E f\).
  - Angular / refraction term is \(\dfrac{\Delta}{\hbar E k_F}[\dots]\).
  - \((k_F\xi_0)^{-1}\) is dimensionless (the quasiclassical small
    parameter).
  - The radial-force cancellation is **unchanged** by the \(\hbar\)
    accounting: the \(\hbar\) from \(\mathbf F/\hbar\) cancels the \(\hbar\)
    in \(\hbar\mathbf v_g\). If you see explicit \(\hbar\)'s here and \(k_F\)
    (not \(p_F\)), that is the wave-vector convention, not an
    inconsistency.

- **Gap-gauge \(\hat\Delta=-\mathrm i\Delta\tau_2\): the \(\mathrm i\) is
  OUTSIDE the commutator by design — don't "distribute" it.**
  \(\hbar(\cdots)+\mathrm i[E\tau_3-\hat\Delta-\check\Sigma,\check g]=0\) is
  correct (real streaming/diffusion). Putting the \(\mathrm i\) on the
  energy term, \([\mathrm iE\tau_3-\hat\Delta-\dots]\), is **wrong**: it
  co-phases \(E\) and \(\Delta\), giving a gapless \(\sqrt{E^2+\Delta^2}\)
  spectrum instead of BCS \(\sqrt{E^2-\Delta^2}\).

- **Distribution symbols.** \(f\) is a quasiparticle *occupation
  probability*; \(\fL,\fT\) are distribution *amplitudes* (\(\fL=1-2f\),
  \(\fL^{(0)}=\tanh(E/2T)\), \(\fT^{(0)}=0\)). "\(f\)-type symbols" are not
  all occupations.

- **Intro \(\fL/\fT\) definition is deliberate — don't "fix" it.**
  \(\fL=f^{(1)}\), \(\fT=f^{(2)}\) (Kopnin's \(\openone/\tau_3\)
  coefficients) is correct as written. Do **not** rewrite it as a
  sum/difference: \(f^{(1)}\pm f^{(2)}=\fL\pm\fT\) are the diagonal/per-branch
  amplitudes, a *different* object (call them \(h_\pm\) if you need them).
  \(\fL=1-2f\) is correct (\(\fL\) is an amplitude, not an occupation) — not
  a missing \(\tfrac12\). Don't re-add a "valid only in a fixed Bogoliubov
  basis" hedge.

- **Isotropic, inversion-symmetric band with a spherical Fermi surface** is
  assumed (\(\mathbf v_F=v_F\hat{\mathbf k}\)), stated before the
  derivation. The radial/angular split and the single scalar \(k_F\) rely on
  it.

---

## Physics already checked — please do not re-flag

- **Diagonal \(\fL/\fT\) streaming is correct (NOT cross-coupled) at the
  ballistic-drift level.** For a real order parameter with no superflow,
  \(\fL\) and the *unweighted* branch-odd combination
  \(\phi_T\equiv\widetilde f_--\widetilde f_+\) share the *same* diagonal
  drift
  \(D=\partial_t+\frac{\Delta}{E}\dot\Delta\,\partial_E+v_g\hat{\mathbf p}\cdot
  \nabla_{\mathbf r}\) and couple only through the collision integrals and
  the self-consistent gap. **Kopnin's transverse mode is the
  \(\lambda\)-weighted object, not the bare difference:**
  \(\fT=f^{(2)}=\lambda_{\mathbf k}\phi_T\) with
  \(\lambda_{\mathbf k}=|\xi|/E\), and it inherits the drift only up to the
  spectral-weight derivative,
  \(D\fT-\fT\,D\ln\lambda_{\mathbf k}=\lambda_{\mathbf k}I_T\)
  (`eq:intro_fT_kopnin`). That extra term drops only when \(\fT\to0\) or
  \(\lambda_{\mathbf k}\) is constant, so do **not** rewrite
  \(\fT=\widetilde f_--\widetilde f_+\) without the \(\lambda_{\mathbf k}\)
  weight — the bare difference is \(\phi_T\), not \(\fT=f^{(2)}\). The
  branch-explicit derivation shows why the *drift* is shared: the antipodal
  relabelling \(\hat{\mathbf p}=s\hat{\mathbf k}\) turns the branch-odd
  \(s\,v_g\hat{\mathbf k}\cdot\nabla\) into the branch-even
  \(v_g\hat{\mathbf p}\cdot\nabla\) (`eq:intro_branch` →
  `eq:intro_branch_traj`), so \(\fL\) and \(\phi_T\) inherit it.
  - Cross-coupling arises **only** if one (a) keeps the refraction force
    when forming the modes, or (b) works at fixed *momentum* direction.
    Both are deliberately avoided.
  - Consistency cross-checks: matches paper 1's statement that
    \(\fL\)–\(\fT\) mixing enters only via the supercurrent and vanishes for
    real \(\Delta\); reduces exactly to the literature scalar equation
    `eq:intro_three_f` when \(\fT\to0\), \(\fL=1-2f\).
  - **Parity dictionary (`eq:intro_parity_dictionary`): \(\fT=\lambda_{\bk}\phi_T\)
    is an even-harmonic identity ONLY.** Odd harmonics swap sectors:
    \(\fL^{\rm odd}=\phi_T^{\rm odd}\),
    \(\fT^{\rm odd}=-\lambda_{\bk}(\widetilde f_++\widetilde f_-)^{\rm odd}\).
    Don't extend \(\fT=\lambda\phi_T\) to odd harmonics or delete the
    dictionary as "redundant".
  - **Hidden-harmonic map (`eq:sc_hidden_harmonic`):**
    \(f=f_0+\hat{\bp}\cdot\mathbf a\) ⇒ \(\fL=1-2f_0\),
    \(\fT'=-2\lambda_{\bk}\hat{\bk}\cdot\mathbf a\). The scalar route's
    \(P_1\) harmonic IS Kopnin's anisotropic transverse amplitude;
    \(\phi_T\equiv0\) ≡ Kopnin §10.5 charge-balanced sector
    \(\{\langle f_2\rangle=0,\ \mathbf f_1=0,\ \mathbf f_2\neq0\}\), a
    bijection.
  - \(\mathbf a\) (not \(\mathbf f_1\)) for the \(P_1\) harmonic is
    deliberate: it maps to Kopnin's \(\mathbf f_2\), while Kopnin's
    \(\mathbf f_1\) is the longitudinal vector harmonic (zero in this
    sector). Renaming back would collide with Kopnin §10.5.

- **Refraction / pair-potential-gradient (PPG) force is dropped** as next
  order in \((k_F\xi_0)^{-1}\) (augmented-Eilenberger), beyond standard
  Eilenberger order. It is deliberately *retained* through the
  single-particle `eq:intro_boltzmann_clean` ("keep, then drop") and dropped
  at the two-mode promotion. The non-uniformity near the gap edge (where
  \(v_g\to0\)) is acknowledged and handled as a boundary condition, not as a
  bulk term.

- **Chain-rule cancellation** (4th and 5th terms of
  `eq:intro_boltzmann_substituted`): the gap-gradient piece of streaming
  cancels the radial gap force in the local-energy variable. Correct
  local-energy bookkeeping; survives the \(\hbar\) accounting.

- **Adiabatic projection** onto the instantaneous positive-energy Bogoliubov
  band; validity \(\hbar|\xi_{\mathbf k}\dot\Delta|/E^3\ll1\) (plus a smooth
  spatial analogue). Deliberate scope statement; numerical \(O(1)\) factors
  are immaterial in a \(\ll1\) condition.

- **Electrostatics / charge imbalance.** The four-equation "kinetic core"
  deliberately omits the electrostatic closure because the applications
  below use \(\fT\to0\). For nonzero \(\fT\) a gauge-invariant
  scalar-potential equation (local neutrality) is needed; the text says so
  and distinguishes "neglect backreaction" from "impose neutrality" (the
  latter fixes a generally *nonzero* potential, it does not set it to
  zero). The displayed set is therefore the kinetic core, not claimed
  closed for arbitrary \(\fT\).

- **Phonon collision integral \(I_n[\fL,n]\)** (no \(\fT\) dependence):
  justified — electron–phonon kernels are branch-even, so \(\fT\) drops to
  linear order in the imbalance; \(O(\fT^2)\) corrections are neglected.
  The charge-imbalance content that drops is properly \(\phi_T\) (the
  branch-odd mode); \(\fT=\lambda_{\bk}\phi_T\) itself retains an odd,
  current-carrying harmonic built from the branch-even occupation, which
  does not vanish here — don't call \(\fT\) itself "the branch-odd mode."

- **Gap equation uses \(\langle\fL\rangle_{\hat{\mathbf p}}\)** (Fermi-surface
  average): only the isotropic part sources the \(s\)-wave gap. Deliberate.

- **"Clean limit" = ballistic streaming**, *not* collisionless. The
  collision integral is retained on the RHS; "clean" refers to the absence
  of impurity-driven diffusion (contrast the dirty/Usadel route).
  Terminology is intentional.

- **Comparison with the classical Boltzmann equation.** The text
  deliberately does *not* claim the distinguishing feature is molecular
  chaos vs. microscopic rates (both descriptions use microscopic rates with
  Pauli factors). The stated distinction is the BCS dispersion plus
  coherence factors in the kernels.

- **Normalization wording.** \(\check g\otimes\check g=\openone\) "closes
  the transport problem and selects the physical solution manifold"; it
  does **not** "restore" the off-shell / particle–hole-asymmetric
  information discarded in the \(\xi\) integration. The wording is
  intentional and correct.

- **Phonon source \(S\).** Volumetric/coarse-grained source–sink. Boundary
  reflection/transmission/substrate escape belong in *boundary conditions*;
  they are represented by \(S\) only after spatial/angular coarse-graining.
  Intentional.

- **\(k\to k_F\) in the angular term**: relative error
  \(O(|\xi_{\mathbf k}|/E_F)\) (\(O(\Delta/E_F)\) near-gap, at most
  \(O(\omega_D/E_F)\) over the shell), quasiclassically negligible. Stated.
  (Note: the change-of-variables appendix states the more general \(k\)
  dependence first and specializes to \(k\to k_F\) later — that's the
  appendix walking through the same approximation, not an inconsistency.)

- **The projector in the momentum-derivative split** is harmless: the
  surface gradient \(\nabla_{\hat{\mathbf k}}f\) is already tangent, and the
  projector \((\openone-\hat{\mathbf k}\hat{\mathbf k})\) makes the
  transversality explicit (it equals the Jacobian
  \(\partial\hat{\mathbf k}/\partial\mathbf k\) up to \(1/k\)). Not a
  redundancy error.

- **These intro clauses are deliberate, not over-hedging — don't strip
  them:** the singlet-BCS-model qualifier on the gap-absorption sentence;
  photons as an *effective self-energy* in \(\check\Sigma_{\mathrm{coll}}\)
  (coherent EM field stays in \(\widetilde\nabla\)); the "fixed-spectrum
  approximation" framing; \(k_{\mathrm B}T\) not bare \(T\) (\(\hbar,k_B\)
  kept explicit).

### Transport-time closure, conservation form, and Usadel traces

- **Transport-time closure (`eq:sc_tau_closure`) is forced, not chosen:**
  \(\tautr(E)=N_1(E)\tau_{\mathrm N}\) (intra-branch
  \((uu'-vv')^2=N_1^{-2}\), final-state DOS \(N_1\); Kopnin's own
  single-mode rate \(-f^{(a)}/(g\tau)\), §15.2). Hence
  \(\elltr=\ell_{\mathrm N}\) (E-independent), \(D(E)=D_{\mathrm N}/N_1\),
  \(N_1D=D_{\mathrm N}\),
  \(L_D=(1/N_1)\nabla\!\cdot\!(D_{\mathrm N}\nabla f)\): the gap-gradient
  drift cancels for a homogeneous material. \(\tautr=\tau_{\mathrm N}\)
  (→ \(D_{\mathrm N}/N_1^2\)) is the documented trap, not an alternative
  closure.
- **Conservation form (`eq:cons_form`):**
  \(\partial_t(N_1f)+\partial_E(N_2\dot\Delta f)=\nabla\!\cdot\!(D_{\mathrm N}\nabla f)+N_1I_{\rm coll}\),
  via \(\partial_tN_1+\partial_E(N_1\dot E_\Delta)=0\),
  \(N_1\dot E_\Delta=N_2\dot\Delta\) (sympy-verified). Conserved density
  \(N_1f\); current \(-D_{\mathrm N}\nabla f\) (occupation gradient, NOT
  density gradient).
- **Usadel trace coefficients (`sec:usadel_longitudinal`):** longitudinal
  \(D_{\mathrm N}\tfrac12(1+|\cosh\theta|^2-|\sinh\theta|^2)\to D_{\mathrm N}\)
  above the gap (\(N_1^2-N_2^2=1\)), 0 below; transverse \(\to N_1^2\)
  (charge-current dressing). For a complex spectral angle (finite Dynes
  broadening), \(\mathcal D_L\mathcal D_T=N_1^2\) survives generally but the
  ideal individual equalities \(\mathcal D_L=1\)/\(\mathcal D_T=N_1^2\) do
  not — finite \(\Gamma\) gives \(\mathcal D_L>0\) below the ideal gap,
  which is a physically leaky (not merely numerically smoothed) subgap face.
  The exact ideal Dynes edge value at \(E=\Delta\) is
  \(\mathcal D_L(\Delta)=\frac12\big(1+\Gamma/\sqrt{\Gamma^2+4\Delta^2}\big)\)
  — only \(\to1/2\) as \(\Gamma/\Delta\to0\), not exactly \(1/2\) at finite
  \(\Gamma\).
- **Gap-edge caution in the closures subsection is deliberate:**
  \(\tautr=N_1\tau_{\mathrm N}\) diverges at the edge, so
  \(\omega\tautr\ll1\)/inelastic conditions fail there in TIME while
  \(\elltr=\ell_{\mathrm N}\) stays short. A scope statement, not an
  inconsistency.
- **Kopnin printing quirk:** the §10.5 reprint (10.98) omits the
  \(g_-\partial_tf_1\) term that IS present in (10.55), so (10.107) is
  quasistatic. Don't flag the paper's \(\partial_t\) term as disagreeing
  with Kopnin.
- **Explicit conjugation map (as of 2026-07-20 an unnumbered inline
  equation in a footnote attached to `eq:beta_f_from_modes`; the former
  display `eq:beta_conjugation_map` was folded into that footnote and
  the label no longer exists):**
  \(h=\sigma(1-2n^{\mathrm K})=f^{(1)}+\sigma\lambda^{-1}f_2'\) — Kopnin's
  \(n^{\mathrm K}\) is the Fermi function only on the electron-like branch;
  the \(\sigma\) placement differs from Kopnin's Eq. (15.4) by exactly this
  conjugation, by design.

### Settled items inherited from paper1/paper2, and from later review rounds

- **Advanced-propagator convention** \(\hat g^A=-\tau_3\hat g^{R\dagger}\tau_3\)
  (\(=-\hat g^R\) above the gap) is correct and, more than a mere
  convention choice, is **physically forced**: with
  \(L_0=E\tau_3+i\Delta\tau_2\), \([L_0,\hat g^A]=0\) only for this sign;
  the opposite anomalous sign gives \([L_0,\hat g^A]=-4\Delta E\tau_1\neq0\).
  Correspondingly \(D_L=\tfrac14\mathrm{Tr}[\openone-g^Rg^A]=1\),
  \(D_T=\tfrac14\mathrm{Tr}[\openone-g^R\tau_3g^A\tau_3]=N_1^2\) for the
  chosen sign, and swap (\(D_L=N_1^2\), \(D_T=1\)) for the wrong one. The
  text documents why Belzig et al. Eq. (49) (opposite anomalous sign) is
  NOT used. Don't re-flag either direction.
- **Channel dictionary** (tab:channel_dictionary): \(\mathcal D_L=1/0\)
  above/below gap, \(\mathcal D_T=N_1^2\) (charge),
  \(\mathcal D_L\mathcal D_T=N_1^2\) generally; \(N_1^2\) dressing belongs
  to the TRANSVERSE channel. Sympy-verified.
- **A1 is selected by BOTH routes**; taxonomy rows B \((0,-2)\)/C
  \((0,-1)\) are legacy placements (Fick ansätze for \(f\)), kept as labeled
  diagnostics only. For the DOS-gradient drift velocity
  \(v=D_N q N_1^{q-p-1}\partial_xN_1\) of the general \(L_{p,q}\) family,
  A1 is \((p,q)=(1,0)\) so \(v\equiv0\) identically and parameter-free by
  both routes. The benchmark establishes legacy-placement passive-tracer
  drift into a population-generated gap well and zero static
  DOS-gradient tracer drift for A1. It does **not** establish reciprocal
  net self-focusing of one coupled population — see "Open gates" below for
  why that wording remains restricted.
- **Kupriyanov–Lukichev scalar BC** (eq:scalar_BC_energy): the consistent
  conductivity-weighted longitudinal current is
  \(\mathcal J_L=-\sigma_i\mathcal D_L^{(i)}\partial_nf_{L,i}=G_N\mathcal W_L(f_{L,1}-f_{L,2})\),
  \(\sigma_i=2e^2N_0^{(i)}D_{\mathrm N}^{(i)}\); the diffusion-normalized
  current on side \(i\) is
  \(j_L^{(i)}=-D_{\mathrm N}^{(i)}\mathcal D_L^{(i)}\partial_nf_{L,i}=\mathcal J_L/2e^2N_0^{(i)}\).
  For unequal materials only \(\mathcal J_L\) is continuous — the
  diffusion-normalized currents differ if the \(N_0^{(i)}\) differ; don't
  equate them. Energy weight \(N_1N_1'-N_2N_2'\) (regular at matched gaps),
  charge weight \(N_1N_1'\) (carries the SIS edge singularity); at nonzero
  phase difference \(\delta\chi\) the weight is
  \(\mathcal W_L=N_1^{(1)}N_1^{(2)}-N_2^{(1)}N_2^{(2)}\cos\delta\chi\) (the
  displayed zero-phase result is the \(\delta\chi=0\) case and should be
  scoped as such). \(G_N\to\infty\) is a formal Robin limit, not by itself
  a controlled high-transparency derivation. \(f\) jumps (Robin condition);
  finite \(G_N\) permits but does not require a jump at zero current.
- **Time-dependent spectral flow** is produced directly by the fixed-\(E\)
  projection (eq:fixedE_conservative_flow); the moving-\(\xi\)/branch-projector
  scalar construction is an exact coordinate relabelling away from the gap
  edge, not an intrinsic moving-frame matrix derivation, and is nonuniform
  at \(\xi=0\).
- **Gap-feedback closure** (eq:gap_feedback_closure) is the exponentiated
  T-free form; \(n_{qp}=4N_0\int N_1 f\,dE\) with single-spin \(N_0\)
  (factor 4 = spin × two \(\xi\) branches). Deliberate.
- **Appendix A** holds the explicit \(\bk\to(E,\hat\bk)\) change of
  variables; the chain-rule cancellation guard lives THERE (kept there
  because it's "rarely done explicitly in the literature").
- **DOS symbols, two not one:** \(N_0\) (single-spin,
  \(mp_F/2\pi^2\hbar^3\), Kopnin's \(\nu(0)\)) is the constant used
  everywhere except the materials-extension paragraph, which deliberately
  keeps \(N(0)\) as the *local*, position-dependent normal-state DOS of a
  phenomenological multi-material extension — its spin normalization
  cancels between the two factors of eq:sc_LD_inhomogeneous. Don't merge
  \(N(0)\) into \(N_0\).
- **S-symbol overload (phonon source \(S\) vs. supercurrent \(S(E;Q)\))
  is deliberately NOT renamed** — the channel-dictionary parenthetical
  disambiguates; renaming would touch a boxed, sympy-verified SM
  supercurrent equation for cosmetic gain only.
- **Nonuniform \(O(Q^3)\) supercurrent-mixing onset:** the displayed outer
  expansion
  \(S(E;Q)=-4Q\Gamma E^2\Delta^2/(E^2-\Delta^2)^{5/2}\), \(\Gamma=2\hbar D_NQ^2\),
  is correct at fixed \(E>\Delta\) with \(W^3\gg\Gamma E\Delta\), but is not
  uniform at the depairing-rounded edge: there the energy width is
  \(O(Q^{4/3})\), peak height \(O(Q^{-1/3})\), and integrated weight
  \(O(Q)\) (not \(O(Q^3)\)) —
  \(S(E;Q)/Q\to-\pi\Delta\,\delta(E-\Delta)\) distributionally. A smooth
  energy integral is therefore generally linear in \(Q\) unless its weight
  vanishes at the edge. The current formula is labeled as the fixed-energy
  outer result, not a global mixing-onset claim.
- **Nonadiabatic \(O(\hbar^2)\) source, completed:** writing
  \(g^{R/A}=g_0^{R/A}+\hbar^2d^{R/A}\), the consistent kinetic residual
  needs \(\delta g^K=\hbar^2(d^Rh-hd^A)\), not just the
  \(g_0^{R/A}\)-built Keldysh function. With
  \([L_0,g_0^R]_\star=\hbar^2r\tau_1+O(\hbar^3)\),
  \(r=3E\Delta^2\ddot\Delta/4W^5\), the completed source for
  \(h=f_L\openone\) is
  \(\delta g^K=2\hbar^2f_Ld^R\), contributing
  \(-3iE\Delta^2\ddot\Delta f_L/W^5\) to the \(\tau_1\) trace. For a
  gap-slaved distribution \(f_L=G(W)\) this consistently completed source
  vanishes identically; a generic independently-driven source and the
  static-gap source \(\propto\partial_t^2f_L\) survive. `verify_nonadiabatic.py`
  checks both spectral equations, R/A/K star normalization, this Keldysh
  completion, the shell-slaved and equilibrium cancellations, and zero
  scalar/transverse second-order projections.
- **Transverse \(O(\dot\Delta)\) response:** in the stated real-gap,
  no-superflow, particle-hole-symmetric sector the derived transverse
  equation is homogeneous in \(f_T\) (\(J_T[f_L,0,n]=0\)); zero initial and
  boundary imbalance imply \(f_T=0\) through the computed order. A moving
  gap advects/dilutes existing imbalance; it does not create it in this
  sector.
- **Projection-average commutator:** for antipodal relabeling
  \(R_sX(\hat p)=X(s\hat p)\), \(\langle R_sX\rangle_{\hat p}=\langle X\rangle_{\hat k}\)
  — the full Fermi-surface average is measure-preserving. Order
  sensitivity belongs to retaining/solving/eliminating the first harmonic,
  not to the bare full-sphere average (this is the antipodal-average
  identity used in `sec:projection_vs_averaging`).
- **Moyal/plain product bridge:** ∘ = main-text ⊗; ⋆ = its (E,t)
  restriction, stated at SM Conventions, eq:moyal_def, and app:branch
  setup. The Moyal commutator is
  \([A,B]_\star=[A,B]+i\hbar\{A,B\}_{\rm PB}+\cdots\) (no factor of
  \(1/2\)); the antipodal trajectory relabeling \(\hat p\to-\hat p\) is the
  displayed map between the starting-equation conventions.
- \(\mathcal D_L,\mathcal D_T\) calligraphic everywhere; taxonomy
  conserved-density weight is \(\mathcal W\) (kernel rate \(w\) unchanged);
  branch-Boltzmann route uses \(\tau\) (= main-text \(\tau_N\)) and bold
  \(\mathbf a\) for the \(P_1\) harmonic, \(u^2(E)-v^2(E)\) on-shell
  coherence factors.
- **"Clean" advanced-scan note:** \(N_1=\operatorname{Re}g^R\),
  \(N_2=\operatorname{Re}f^R\) — calling \(N_1(E+i\Gamma)\),
  \(N_2(E+i\Gamma)\) "complex spectral functions" is a type error; they're
  the real parts of the analytically-continued propagators, evaluated at
  complex argument.

---

## Open gates (not settled — do not treat as resolved)

Two items were deliberately left **human-gated** after multiple independent
adversarial review rounds (2026-07-10/11) found no other surviving
blocker/major/minor physics defect in the paper's delimited sector (real
gap, no superflow, charge-balanced, dirty local-BCS, isotropic Born
impurities). Everything else those reviews raised was fixed and is folded
into the settled content above. These two remain open on their merits, not
because they were never checked:

### 1. The quantitative Riwar–Catelani trap-length correction (was "C1/C2") is NOT in the paper, and shouldn't be inserted without new derivation

The idea: treat the Riwar-Catelani lateral gap step as a sharp interface,
take \(G_N\to\infty\) in the Kupriyanov-Lukichev Robin condition, get
continuity of \(f\) and the undressed \(D_N\partial_yf\) flux across it,
and use that to correct their trap length (their number would move from
\(7\to32\,\mu\mathrm m\)).

This is **not established** for the actual device: Riwar's step is a
coarse-grained, coherence-scale proximity region in one continuous S film
above an S' layer, not the low-transparency tunnel barrier the KL relation
was derived for; the manuscript's local-BCS sector explicitly excludes
proximity-modified spectra; and \(G_N\to\infty\) is a formal Robin limit,
not a proof that a transparent proximity interface behaves that way. The
generalized proximity current is
\(j_L(E,y)=-\sigma_N(y)\mathcal D_L(E,y)\partial_yf_L\),
\(\mathcal D_L=\cos^2[\operatorname{Im}\theta(E,y)]\); across a finite
spectral layer this gives an endpoint jump
\[
f_L^+-f_L^- = -J_LR_\xi(E),\qquad
R_\xi(E)=\int_{\rm layer}\frac{dy}{\sigma_N(y)\mathcal D_L(E,y)},
\]
and continuity of \(f\) requires \(R_\xi(E)\) to be negligible — which has
not been shown uniformly over the near-edge cold window that produces the
large trap-length factor. (One specific proposed rescue — averaging over a
boundary-layer window to argue \(R_\xi\) is cutoff-insensitive — was
checked numerically and found to depend on the cutoff: the exact
near-edge resistance behaves like \(R_{\rm exc}\propto\epsilon^{-1/2}\)
down to at least \(\epsilon=10^{-9}\), so a claimed sub-percent theorem
needs an actual broadening mechanism, energy-relaxation closure, or
energy-resolved eigenmode calculation — not a fixed-cutoff window average.
This is a reusable warning for any future attempt at the same argument, not
just a historical footnote.)

What **is** established and safe to use: Riwar Appendix A starts from the
legacy placement; A1 and the legacy placement have the same bulk rate
inside each constant-gap outer region; the generalized proximity energy
current is conserved through a source-free stationary layer; equal outer
gradients follow from current conservation when the two outer
conductivities are equal (this does *not* imply continuity of \(f\) — the
layer resistance instead fixes the endpoint jump).

The paper's approved main-text caveat states the endpoint-drop formula
above (for a stationary, 1-D, constant-area layer with no other kinetic
sources/sinks) and explicitly declines to revise Riwar-Catelani's number,
because \(R_\xi(E)\)'s negligibility hasn't been shown. It also carries one
local, strictly fixed-energy Péclet diagnostic for legacy placement C
(\(\mathrm{Pe}_E=\Delta|\delta\Delta|/(E^2-\Delta^2)\)), explicitly labeled
as not a device observable. **Do not** reinsert a quantitative trap-length
correction, an assumed endpoint continuity, or a claim that the proximity
transfer law has been verified, unless someone actually derives the
energy-resolved transfer law through the laterally-terminated S/S' bilayer
(or proves \(R_\xi(E)\) negligible over the relevant window) and
reintegrates it into the eigenvalue/matching calculation.

### 2. "Self-focusing" wording for the legacy operators remains restricted

The benchmark (`validation/diffusion_operators/self_consistent_feedback.py`)
creates a heavy population and a separate passive probe, updates the gap
from the heavy population only, and measures probe drift. It correctly
shows: A1 has zero static DOS-gradient tracer drift (parameter-free,
\(q=0\)); the legacy placements (\(q<0\)) pull a passive tracer toward a
gap well generated by a *separate* population.

It does **not** show net compression/focusing of a population in the well
*it itself* generates — the coupled map \(L[\Delta[f]]f\) is nonlinear, so
linear-transport-at-fixed-\(\Delta\) superposition arguments don't carry
over, and a direct one-population counter-test (same benchmark defaults,
tracking \(\sigma_x\) of the conserved density) found that **all three
models broaden** over 20 ns (A1: 7.306→12.322 µm; C: 7.071→12.149 µm; B:
7.071→9.520 µm) — the negative-\(q\) drift only *reduces* broadening for B
at this point, it doesn't reverse it. So "self-focusing" (net
concentration) overstates what's demonstrated; "passive inward drift" or
"reduced broadening" does not.

The approved manuscript wording (abstract/body/captions) already reflects
this: every reciprocal "self-focusing" claim was replaced with passive-probe
language, and the benchmark caption states the 5% well is exaggerated, the
probe is passive, and inward probe drift is not net compression. **Don't
reintroduce "self-focusing" language** for the legacy operators unless
someone defines an actual focusing observable (e.g. peak density or
\(\int f^2\)) and demonstrates it in a fully coupled one-population
calculation that also includes energy-space spectral-flow advection (the
existing benchmark omits that).

---

## Known incomplete (work in progress — not defects)

- **Abstract**: current version is Soren's own text (simplified directly by
  the author after the gated review rounds above concluded). Not an open
  item.
- **riwar2019 Appendix-A evidence** (verified 2026-07-17 against both the
  ar5iv HTML and the compiled arXiv v1 PDF of arXiv:1907.04781): their
  Eq. (A1) is
  \(\dot f_{\rm qp}(\epsilon,y)=\partial_y[D_{\rm qp}(\epsilon,y)\,
  \partial_y f_{\rm qp}]\) with \(D_{\rm qp}=D_0/\nu_{\rm BCS}(\epsilon)\),
  \(\nu_{\rm BCS}=\epsilon/\sqrt{\epsilon^2-\Delta^2(y)}\), attributed by
  them to Belzig et al. 1999 — exactly placement C at a varying gap.
  The arXiv v1 PDF (the sole arXiv version, REVTeX/PRB layout) prints
  per-appendix numbers (A1)–(A7); the "(46)" recorded earlier exists only
  in ar5iv's LaTeXML continuous renumbering, so do not cite it as "arXiv
  numbering." The (A1) pinpoint now appears at the intro citation and in
  the SM taxonomy discussion (referee-risk pre-emption satisfied).
- Intro ¶1's evidence base spans three distinct forms of the same BRT
  coefficient, all verified against their own papers/arXiv/PDFs and none
  claimed to be in error for the uniform-gap case their own analyses use:
  constant-coefficient class (riwar2016, hosseinkhani2018, kozorezov2002
  PRB 66, 094510 Eq. 7, segall2004 PRB 70, 214520 Eq. 1); the
  energy-resolved occupation-Fick form (riwar2019 Appendix A, above); and
  the energy-resolved *density*-Fick form
  \(\partial_t\chi_{qp}=(D_N/N_1)\nabla^2\chi_{qp}\) (dong2022, Appl. Sci.
  12, 8461 (2022), their Eq. 3, with \(\chi_{qp}\propto N_1f\) at fixed
  energy). martinis2009 (PRL 103, 097002) is cited as the source quoting
  \(D=60\,v_{qp}\)cm²/s for aluminum with no accompanying transport
  operator (confirmed 2026-07-17 from the local PDF: no spatial operator
  anywhere in the letter or its P(E) supplement; the D quote is uncited
  prose on p. 097002-3, and dong2022 cite martinis2009 for the
  coefficient). A 2026-07-17 re-verification sweep (local PDFs, ar5iv,
  MDPI, plus the kozorezov2002 journal PDF obtained the same day)
  confirmed every entry above from primary sources. The kozorezov2002
  pinpoint: their Eq. (7) is the generalized Rothwarf–Taylor pair with
  constant-coefficient area-density diffusion terms \(D_i\triangle n_i\),
  and the freezing assumption is stated verbatim at the end of
  Sec. II A, directly below Eqs. (7)–(8) (p. 094510-3): "In Eq. (7) we
  implicitly assume that once diffusion has started, the diffusion
  constants \(D_i\) remain constant. Since the diffusion constant
  depends on the quasiparticle energy distribution, we therefore assume
  that the latter does not change significantly after the generation of
  quasiparticles has completed." New data
  point from the sweep: riwar2016 Appendix B (arXiv Eqs. (54)–(56))
  contains an energy-resolved density-Fick equation
  \(\dot p_S=D_S(\epsilon)\nabla^2 p_S\) with
  \(D_S(\epsilon)=D_S/\nu_S(\epsilon)\), \(p_S=\nu_S f_{qp}\) — a
  further published legacy-family placement citable alongside dong2022.
- All former paper3 gaps are closed in the merge: `sec:sc_scalar_equations`
  references were remapped to `eq:beta_f_from_modes`, the supplement is now
  local (SM- refs resolve), and Data/code availability carries the paper1
  URLs.

---

## Structure notes (2026-07-15 restructure — current, not historical)

The paper's section structure was reorganized this session (advisor- and
Soren-directed); this describes the *current* layout, not a change log:

- The introduction is now motivation + a short applications overview
  (KIDs/STJ sensors, qubit traps) + outline only. The four machinery
  subsections (kinetic equations, Boltzmann reductions, Usadel equation,
  BCS kinematics/closures) were promoted out of the introduction into
  `\section{Quasiclassical kinetic framework}` (`sec:quasiclassical`),
  which now closes with the "Sector of this paper" paragraph (the
  compact assumption list: homogeneous s-wave BCS, real gap/no
  superflow/no drive, adiabatic local-BCS spectra, Born impurities,
  charge-balanced \(\fT\to0\), PPG force dropped, \(E>\Delta\) domain).
- The taxonomy section and the consistency-check/benchmark material
  (`sec:taxonomy`, `sec:program`, `sec:projection_vs_averaging`,
  `sec:self_consistent_feedback`, the four bench figures, the roadmap
  figure, `tab:operator_taxonomy`) moved wholesale to the end of the
  Supplemental Material, keeping their labels. This is a length-only
  move and is reversible if the length constraint changes.
- The Kupriyanov–Lukichev subsection stayed in the main text, promoted to
  `\section{Conserved currents and boundary conditions}`
  (`sec:conserved_currents`), because the abstract still headlines that
  derivation.
- The spurious-drift exhibit moved to the SM along with the taxonomy
  (this bullet corrected 2026-07-17; an earlier version placed it in
  `sec:quasiclassical`): `eq:Lpq_expanded`, `eq:Lpq_first_moment` (the
  exact center-of-mass drift law
  \(d\langle r\rangle/dt=\langle qD_NN_1^{q-p-1}\nabla N_1\rangle\) for
  the whole \(L_{p,q}\) family, sympy-verified total-derivative
  identity), `eq:legacy_drift_velocity` (\(v_C\)), and the fixed-energy
  Péclet diagnostic `eq:legacy_local_peclet` with its numeric triples
  all live in supplement.tex. The main text reaches the drift term via
  SM~\cref{SM-eq:Lpq_expanded} (Sec. IV), and the intro's closing
  paragraph points to SM~\cref{SM-sec:taxonomy,SM-sec:program}.
- The \(L_{p,q}\) family definition appears inline in the introduction's
  closing paragraph so the body's (p,q) labels stay defined even though
  the taxonomy table moved to the SM.
- Acknowledgments thank Thomas Stevenson (advisor feedback).

---

## Build

- `make bootstrap` on a clean tree (paper.tex and supplement.tex
  cross-reference via xr-hyper, each needs the other's .aux), plain `make`
  afterwards; `make roadmap` rebuilds the routes figure; `make verify` runs
  the sympy proof-check scripts (needs `../.venv` or `make setup`).
- Current clean-build state (verified 2026-08-04): paper.pdf 47pp,
  supplement.pdf 65pp, zero undefined references/citations, zero overfull
  hboxes or vboxes (the former ~3.3pt overfull vbox p.9 cleared when the
  harmonic-exposition trims shifted pagination). `make
  verify` is 7/7 PASS (`verify_fT.py`, `verify_gA_convention.py`,
  `verify_nonadiabatic.py`, `verify_proximity.py`, `verify_supercurrent.py`,
  `verify_tdep_inhomogeneous.py`, `verify_traces.py`).
- **xr-hyper cross-document links**: on this Windows box the installed
  xr-hyper is v6.00beta4 (2000) and ignores the
  `\externaldocument[..]{..}[url]` URL argument, so local cross-document
  links (SM-/M- prefixed) can silently land on the wrong local object where
  anchor names collide (`figure.N`, `table.N`) across the two PDFs.
  Reference *numbers* are all correct regardless. The URL arguments are
  kept because modern toolchains honor them. Don't claim working local
  cross-document links on this box.
- On this Windows box the verify venv is the A: archive clone's
  (`A:\Einstein\Documents\qp-diffusion-paper\.venv`); the Makefile's `PY`
  default matches `make setup` (paper-local `.venv`).
- **latexmk PATH note**: the inherited/default `PATH` may select a
  stripped LyX-bundled Perl lacking `Digest::MD5`, which aborts `latexmk`.
  Prepend `C:\Program Files\Git\usr\bin` (Git's Perl supplies that module),
  or use the direct `pdflatex`/`bibtex` xr-hyper bootstrap: supplement →
  paper → supplement → `bibtex`×2 → alternate passes until cross-references
  converge. Run `sanitize_aux.py` (tracked) on the other document's `.aux`
  before each latexmk pass if alternating supplement/paper.
- An unsanitized *live-xr* supplement build (before `sanitize_aux.py` runs)
  emits `Undefined control sequence` warnings at the
  `\externaldocument[M-]{paper}` import, because `paper.aux` stores
  paper-defined macros inside nameref/caption text and
  `\externaldocument` runs before `hyperref`/`cleveref` load. Non-halt mode
  still produces a full PDF with resolved M- references (exit 1); with
  `-halt-on-error` it stops at the first imported macro. This is a
  build-workflow quirk, not a manuscript defect; `sanitize_aux.py` or
  `make arxiv-labels` clears it.
