# Merged Paper — Conventions and Review Notes

**Purpose.** This file records notation choices and physics points that have
already been derived, checked, and *deliberately* chosen. It exists so a
reviewer — human or AI — can spend effort on genuinely new issues rather than
re-flagging settled ones. Everything below was vetted (several adversarial
review passes); please do not report these as errors without substantively new
argument. Finding *new* problems is welcome; re-deriving these is not.

Scope: THIS is the single merged manuscript (assembled 2026-07-01). Base =
paper3 wording/template (intro + scalar route + agreement); imported from
paper1: the Usadel-route derivation (longitudinal operator, D_L/D_T traces,
flux-to-scalar, time-dependent spectral flow), taxonomy, consistency
checks/benchmarks, conclusion, the entire Supplemental Material, figures, and
verify scripts; imported from paper2: the explicit change-of-variables
appendix. The sibling directories paper1/ paper2/ paper3/ are ARCHIVED
sources — editing them does NOT change this manuscript.

## References

You can find kopnin chapters in Documents/kopnin-numbered-equations/

In particular, chapters 10 and 15 are quite relevant.
---

## Notation conventions (deliberate — not typos)

- **Two distinct direction symbols, and they are not interchangeable.**
  - \(\hat{\mathbf k}\) = momentum (wave-vector) direction.
  - \(\hat{\mathbf p}\) = trajectory (group-velocity) direction.
  - They differ by the branch sign: \(\hat{\mathbf p}=s\,\hat{\mathbf k}\),
    \(s=\operatorname{sgn}\xi_{\mathbf k}=\pm\). For hole-like quasiparticles
    (\(\xi<0\)) they are antiparallel.
  - The single-particle / canonical-Boltzmann derivation
    (`eq:intro_canonical_boltzmann` … `eq:intro_boltzmann_clean`) is written in
    the **momentum** direction \(\hat{\mathbf k}\). From the branch→trajectory
    relabelling onward — and throughout the **entire body** — the **trajectory**
    direction \(\hat{\mathbf p}\) is used. The Eilenberger equation is
    trajectory-resolved (\(\hat{\mathbf p}\)). This switch is intentional and is
    stated in the text.

- **\(\mathbf k\) is a wave vector and \(\hbar\) is kept explicit.** Consequences
  (all dimensionally checked; every kinetic term is \(1/\text{time}\)):
  - \(\mathbf v_g=\hbar^{-1}\partial E_{\mathbf k}/\partial\mathbf k
    =(\xi_{\mathbf k}/E_{\mathbf k})v_F\hat{\mathbf k}\).
  - Boltzmann drift is \((\mathbf F/\hbar)\cdot\partial_{\mathbf k}f\).
  - Momentum-space radial part carries \(\hbar\):
    \(\hat{\mathbf k}\,\partial_k f=\hbar\,\mathbf v_g\,\partial_E f\).
  - Angular / refraction term is \(\dfrac{\Delta}{\hbar E k_F}[\dots]\).
  - \((k_F\xi_0)^{-1}\) is dimensionless (the quasiclassical small parameter).
  - The radial-force cancellation is **unchanged** by the \(\hbar\) accounting:
    the \(\hbar\) from \(\mathbf F/\hbar\) cancels the \(\hbar\) in
    \(\hbar\mathbf v_g\). If you see explicit \(\hbar\)'s here and \(k_F\)
    (not \(p_F\)), that is the wave-vector convention, not an inconsistency.

- **Gap-gauge \(\hat\Delta=-\mathrm i\Delta\tau_2\): the \(\mathrm i\) is OUTSIDE the commutator by
  design — don't "distribute" it.** \(\hbar(\cdots)+\mathrm i[E\tau_3-\hat\Delta-\check\Sigma,\check g]=0\)
  is correct (real streaming/diffusion). Putting the \(\mathrm i\) on the energy term,
  \([\mathrm iE\tau_3-\hat\Delta-\dots]\), is **wrong**: it co-phases \(E\) and \(\Delta\), giving a
  gapless \(\sqrt{E^2+\Delta^2}\) spectrum instead of BCS \(\sqrt{E^2-\Delta^2}\).

- **Distribution symbols.** \(f\) is a quasiparticle *occupation probability*;
  \(\fL,\fT\) are distribution *amplitudes* (\(\fL=1-2f\),
  \(\fL^{(0)}=\tanh(E/2T)\), \(\fT^{(0)}=0\)). "\(f\)-type symbols" are not all
  occupations.

- **Intro \(\fL/\fT\) definition is deliberate — don't "fix" it.** \(\fL=f^{(1)}\), \(\fT=f^{(2)}\)
  (Kopnin's \(\openone/\tau_3\) coefficients) is correct as written. Do **not** rewrite it as a
  sum/difference: \(f^{(1)}\pm f^{(2)}=\fL\pm\fT\) are the diagonal/per-branch amplitudes, a
  *different* object (call them \(h_\pm\) if you need them). \(\fL=1-2f\) is correct (\(\fL\) is an
  amplitude, not an occupation) — not a missing \(\tfrac12\). Don't re-add a "valid only in a fixed
  Bogoliubov basis" hedge.

- **Isotropic, inversion-symmetric band with a spherical Fermi surface** is
  assumed (\(\mathbf v_F=v_F\hat{\mathbf k}\)), stated before the derivation.
  The radial/angular split and the single scalar \(k_F\) rely on it.

---

## Physics already checked — please do not re-flag

- **Diagonal \(\fL/\fT\) streaming is correct (NOT cross-coupled) at the ballistic-drift level.** This was the
  central question across multiple review rounds; the drift result is settled. The formerly-OPEN companion
  question — whether the *diffusive reduction* reintroduces an effective cross-channel — was RESOLVED
  2026-07-01 (see the parity-dictionary / hidden-harmonic / closure bullets below): in the paper's sector it
  does not; the scalar route is an exact change of variables of the dirty two-mode reduction. For a real order
  parameter with no superflow, \(\fL\) and the *unweighted* branch-odd
  combination \(\phi_T\equiv\widetilde f_--\widetilde f_+\) share the *same*
  diagonal drift
  \(D=\partial_t+\frac{\Delta}{E}\dot\Delta\,\partial_E+v_g\hat{\mathbf p}\cdot
  \nabla_{\mathbf r}\) and couple only through the collision integrals and the
  self-consistent gap. **Kopnin's transverse mode is the \(\lambda\)-weighted
  object, not the bare difference:** \(\fT=f^{(2)}=\lambda_{\mathbf k}\phi_T\)
  with \(\lambda_{\mathbf k}=|\xi|/E\), and it inherits the drift only up to the
  spectral-weight derivative, \(D\fT-\fT\,D\ln\lambda_{\mathbf k}=\lambda_{\mathbf k}I_T\)
  (`eq:intro_fT_kopnin`). That extra term drops only when \(\fT\to0\) or
  \(\lambda_{\mathbf k}\) is constant, so do **not** rewrite
  \(\fT=\widetilde f_--\widetilde f_+\) without the \(\lambda_{\mathbf k}\) weight —
  the bare difference is \(\phi_T\), not \(\fT=f^{(2)}\). (This corrected a real
  inconsistency: the \(\beta\)-section recombination `eq:beta_fT_inverse` carries
  the \(\lambda_{\mathbf k}\), while the real-\(\Delta\) `eq:intro_branch_traj`
  combination had dropped it.) The branch-explicit derivation shows why the
  *drift* is shared: the antipodal
  relabelling \(\hat{\mathbf p}=s\hat{\mathbf k}\) turns the branch-odd
  \(s\,v_g\hat{\mathbf k}\cdot\nabla\) into the branch-even
  \(v_g\hat{\mathbf p}\cdot\nabla\) (`eq:intro_branch` → `eq:intro_branch_traj`),
  so \(\fL\) and \(\phi_T\) inherit it.
  - Cross-coupling arises **only** if one (a) keeps the refraction force when
    forming the modes, or (b) works at fixed *momentum* direction. Both are
    deliberately avoided. Dropping the refraction force *before* the relabelling
    is essential and intentional (kept diagonal as a result).
  - Consistency cross-checks: matches paper 1's statement that \(\fL\)–\(\fT\)
    mixing enters only via the supercurrent and vanishes for real \(\Delta\);
    reduces exactly to the literature scalar equation `eq:intro_three_f` when
    \(\fT\to0\), \(\fL=1-2f\).

- **Refraction / pair-potential-gradient (PPG) force is dropped** as next order
  in \((k_F\xi_0)^{-1}\) (augmented-Eilenberger), beyond standard Eilenberger
  order. It is deliberately *retained* through the single-particle
  `eq:intro_boltzmann_clean` ("keep, then drop") and dropped at the two-mode
  promotion. The non-uniformity near the gap edge (where \(v_g\to0\)) is
  acknowledged and handled as a boundary condition, not as a bulk term.

- **Chain-rule cancellation** (4th and 5th terms of
  `eq:intro_boltzmann_substituted`): the gap-gradient piece of streaming cancels
  the radial gap force in the local-energy variable. Correct local-energy
  bookkeeping; survives the \(\hbar\) accounting.

- **Adiabatic projection** onto the instantaneous positive-energy Bogoliubov
  band; validity \(\hbar|\xi_{\mathbf k}\dot\Delta|/E^3\ll1\) (plus a smooth
  spatial analogue). Deliberate scope statement; numerical \(O(1)\) factors are
  immaterial in a \(\ll1\) condition.

- **Electrostatics / charge imbalance.** The four-equation "kinetic core"
  deliberately omits the electrostatic closure because the applications below
  use \(\fT\to0\). For nonzero \(\fT\) a gauge-invariant scalar-potential
  equation (local neutrality) is needed; the text says so and distinguishes
  "neglect backreaction" from "impose neutrality" (the latter fixes a generally
  *nonzero* potential, it does not set it to zero). The displayed set is
  therefore the kinetic core, not claimed closed for arbitrary \(\fT\).

- **Phonon collision integral \(I_n[\fL,n]\)** (no \(\fT\) dependence): justified
  — electron–phonon kernels are branch-even, so \(\fT\) drops to linear order in
  the imbalance; \(O(\fT^2)\) corrections are neglected. Intentional.

- **Gap equation uses \(\langle\fL\rangle_{\hat{\mathbf p}}\)** (Fermi-surface
  average): only the isotropic part sources the \(s\)-wave gap. Deliberate.

- **"Clean limit" = ballistic streaming**, *not* collisionless. The collision
  integral is retained on the RHS; "clean" refers to the absence of
  impurity-driven diffusion (contrast the dirty/Usadel route). Terminology is
  intentional.

- **Comparison with the classical Boltzmann equation.** The text deliberately
  does *not* claim the distinguishing feature is molecular chaos vs. microscopic
  rates (both descriptions use microscopic rates with Pauli factors). The stated
  distinction is the BCS dispersion plus coherence factors in the kernels.

- **Normalization wording.** \(\check g\otimes\check g=\openone\) "closes the
  transport problem and selects the physical solution manifold"; it does **not**
  "restore" the off-shell / particle–hole-asymmetric information discarded in the
  \(\xi\) integration. The wording is intentional and correct.

- **Phonon source \(S\).** Volumetric/coarse-grained source–sink. Boundary
  reflection/transmission/substrate escape belong in *boundary conditions*; they
  are represented by \(S\) only after spatial/angular coarse-graining. Intentional.

- **\(k\to k_F\) in the angular term**: relative error \(O(|\xi_{\mathbf k}|/E_F)\)
  (\(O(\Delta/E_F)\) near-gap, at most \(O(\omega_D/E_F)\) over the shell),
  quasiclassically negligible. Stated.

- **The projector in the momentum-derivative split** is harmless: the surface
  gradient \(\nabla_{\hat{\mathbf k}}f\) is already tangent, and the projector
  \((\openone-\hat{\mathbf k}\hat{\mathbf k})\) makes the transversality explicit
  (it equals the Jacobian \(\partial\hat{\mathbf k}/\partial\mathbf k\) up to
  \(1/k\)). Not a redundancy error.

- **These intro clauses are deliberate, not over-hedging — don't strip them:** the singlet-BCS-model
  qualifier on the gap-absorption sentence; photons as an *effective self-energy* in
  \(\check\Sigma_{\mathrm{coll}}\) (coherent EM field stays in \(\widetilde\nabla\)); the
  "fixed-spectrum approximation" framing; \(k_{\mathrm B}T\) not bare \(T\) (\(\hbar,k_B\) kept explicit).

### Settled 2026-07-01 (verified against the Kopnin transcriptions + sympy)

- **Parity dictionary (`eq:intro_parity_dictionary`): \(\fT=\lambda_{\bk}\phi_T\) is an
  even-harmonic identity ONLY.** Odd harmonics swap sectors:
  \(\fL^{\rm odd}=\phi_T^{\rm odd}\), \(\fT^{\rm odd}=-\lambda_{\bk}(\widetilde f_++\widetilde f_-)^{\rm odd}\).
  Follows from `eq:beta_fL_inverse`/`eq:beta_fT_inverse` + the antipodal relabelling. Don't
  extend \(\fT=\lambda\phi_T\) to odd harmonics or delete the dictionary as "redundant".
- **Hidden-harmonic map (`eq:sc_hidden_harmonic`):** \(f=f_0+\hat{\bp}\cdot\mathbf a\) ⇒
  \(\fL=1-2f_0\), \(\fT'=-2\lambda_{\bk}\hat{\bk}\cdot\mathbf a\). The scalar route's \(P_1\)
  harmonic IS Kopnin's anisotropic transverse amplitude; \(\phi_T\equiv0\) ≡ Kopnin §10.5
  charge-balanced sector \(\{\langle f_2\rangle=0,\ \mathbf f_1=0,\ \mathbf f_2\neq0\}\), a bijection.
  The pre-2026-07 sentence "the scalar starting point has already discarded the cross-gradient
  channel" was WRONG and was removed — do not reinstate it.
- **Transport-time closure (`eq:sc_tau_closure`) is forced, not chosen:**
  \(\tautr(E)=N_1(E)\tau_{\mathrm N}\) (intra-branch \((uu'-vv')^2=N_1^{-2}\), final-state DOS
  \(N_1\); Kopnin's own single-mode rate \(-f^{(a)}/(g\tau)\), §15.2). Hence
  \(\elltr=\ell_{\mathrm N}\) (E-independent), \(D(E)=D_{\mathrm N}/N_1\), \(N_1D=D_{\mathrm N}\),
  \(L_D=(1/N_1)\nabla\!\cdot\!(D_{\mathrm N}\nabla f)\): the gap-gradient drift cancels for a
  homogeneous material. \(\tautr=\tau_{\mathrm N}\) (→ \(D_{\mathrm N}/N_1^2\)) is the documented
  trap, not an alternative closure.
- **Conservation form (`eq:cons_form`):**
  \(\partial_t(N_1f)+\partial_E(N_2\dot\Delta f)=\nabla\!\cdot\!(D_{\mathrm N}\nabla f)+N_1I_{\rm coll}\),
  via \(\partial_tN_1+\partial_E(N_1\dot E_\Delta)=0\), \(N_1\dot E_\Delta=N_2\dot\Delta\)
  (sympy-verified). Conserved density \(N_1f\); current \(-D_{\mathrm N}\nabla f\) (occupation
  gradient, NOT density gradient). Matches thesis Ch. 4 / paper1 A1=(1,0) structure.
- **Usadel trace coefficients (`sec:usadel_longitudinal`):** longitudinal
  \(D_{\mathrm N}\tfrac12(1+|\cosh\theta|^2-|\sinh\theta|^2)\to D_{\mathrm N}\) above the gap
  (\(N_1^2-N_2^2=1\)), 0 below; transverse \(\to N_1^2\) (charge-current dressing). Consistent
  with paper1's dressings — don't re-derive.
- **\(\mathbf a\) (not \(\mathbf f_1\)) for the \(P_1\) harmonic is deliberate:** it maps to
  Kopnin's \(\mathbf f_2\), while Kopnin's \(\mathbf f_1\) is the longitudinal vector harmonic
  (zero in this sector). Renaming back would collide with Kopnin §10.5.
- **Gap-edge caution in the closures subsection is deliberate:** \(\tautr=N_1\tau_{\mathrm N}\)
  diverges at the edge, so \(\omega\tautr\ll1\)/inelastic conditions fail there in TIME while
  \(\elltr=\ell_{\mathrm N}\) stays short. A scope statement, not an inconsistency.
- **Kopnin printing quirk (stated in the conservation subsection):** the §10.5 reprint (10.98)
  omits the \(g_-\partial_tf_1\) term that IS present in (10.55), so (10.107) is quasistatic.
  Don't flag the paper's \(\partial_t\) term as disagreeing with Kopnin.
- **Explicit conjugation map (`eq:beta_conjugation_map`):**
  \(h=\sigma(1-2n^{\mathrm K})=f^{(1)}+\sigma\lambda^{-1}f_2'\) — Kopnin's \(n^{\mathrm K}\) is the
  Fermi function only on the electron-like branch; the \(\sigma\) placement differs from Kopnin's
  Eq. (15.4) by exactly this conjugation, by design.

### Settled 2026-07-02 (Soren-directed, evening session)

- **Intro architecture (former open item 1): the introduction now OPENS with the
  device-modeling ambiguity** — three paragraphs before the Eilenberger machinery:
  (i) published operator choices differ in form (constant-D integrated-density
  Fick in riwar2016/hosseinkhani2018; the energy-resolved \(\DN/\DOS\)-inside-the-
  divergence Fick ansatz in riwar2019, Appendix A — all three VERIFIED against the
  sources 2026-07-02), (ii) uniform-gap invisibility vs. inhomogeneous-gap drift,
  (iii) the paper's two-route resolution and program. The former end-of-intro
  motivation paragraph was folded into the opening (its citations preserved).
  Do not move the motivation back to the end of the intro or re-flag the opening
  as unmotivated; the machinery-first alternative was deliberately retired.
- The abstract names the legacy placement as in common use in device modeling
  (Soren-approved wording, same session). Wording and evidence base are settled;
  don't soften or extend without a new argument.

### Settled 2026-07-04 (Soren-directed: open items 2, 3, 5; item 4 dispositioned)

- **"Sector of this paper" paragraph** now closes the introduction (just before
  \section{Diffusion from the scalar...}): one compact collection of the sector
  assumptions (homogeneous s-wave BCS, real gap / no superflow / no drive,
  adiabatic local-BCS spectra with the \(\hbar|\xi\dot\Delta|/E^3\ll1\) condition,
  Born impurities, charge-balanced \(\fT\to0\), PPG force dropped, \(E>\Delta\)
  domain), ending "claimed within this sector and no further." Don't scatter these
  again or flag the paragraph as redundant with the inline sector list in intro ¶3
  — the paragraph is the consolidation the intro review asked for (open item 2).
- **Early notation flag** (open item 3): the f/\(\fL\)/\(\fT\) reservation
  sentence at the top of the Quasiparticle Kinetic Equations subsection now also
  warns that Kopnin's transverse amplitude is \(\fT=\lambda_{\bk}\phi_T\), not the
  bare branch difference \(\phi_T\), with cross-refs. Deliberately forward-looking
  (\(\phi_T\), \(\lambda_{\bk}\) defined later) — don't "fix" the forward refs.
- **Dynes footnote relocated** (open item 5): the long broadening footnote moved
  out of BCS kinematics (§II opening) into the Usadel-route trace discussion; a
  one-line pointer remains at the original site. SUPERSEDED SAME DAY by the
  "Dynes footnote final form" bullet below (second 2026-07-04 block): the
  footnote's current anchor is the channel-dictionary sentence, with the full
  algebra in SM app:dynes_remark — that bullet, not this one, describes the
  shipped state.
- **Electrostatic-closure explicitness** (open item 4, dispositioned): the settled
  kinetic-core passage was NOT edited; the explicit negative ("a kinetic core, not
  a closed charge-imbalance theory") now lives in the sector paragraph instead.

### Settled 2026-07-04, second session (Soren-directed: remaining open items 6–10 + machine-B 6/11)

- **§II opener thesis** (item 6): the scalar-route section now opens with a short
  paragraph announcing the operator-ordering point before BCS kinematics. Adapted
  from the GPT patch; don't fold it back into the subsection.
- **DOS symbols unified to TWO, not one** (item 7 — resolved after the Kopnin
  check the item demanded): Kopnin defines \(\nu(0)=mp_F/2\pi^2\hbar^3\)
  (Ch. 1) — SINGLE-spin, same object as this paper's \(N_0\). All \(\nu(0)\)
  instances renamed to \(N_0\), now defined at first use (intro δN formula) with
  the \(mp_F/2\pi^2\hbar^3\) value and a "Kopnin's ν(0)" bridge. \(N(0)\) in the
  materials-extension paragraph is DELIBERATELY kept: it is the *local*,
  position-dependent normal-state DOS of a phenomenological multi-material
  extension, not the constant \(N_0\); its spin normalization cancels between
  the two factors of eq:sc_LD_inhomogeneous (now said in the text). Don't
  merge \(N(0)\) into \(N_0\).
- **Conclusion recap trimmed** (item 9): the branch-route paragraph now states
  the irreducible chain (BRT cancellation ⇒ \(\DOS D_B=\DN\) ⇒ A1, legacy
  placement unsupported) without re-displaying the ordering identity or the
  \(D_B\) definition. Don't re-inflate.
- **§IV forward reference** (item 10): the taxonomy text and roadmap caption now
  point to \cref{sec:projection_vs_averaging} (new label) instead of
  forward-referencing eq:projection_average_commutator.
- **Machine-B finding 6**: §III's "Matrix dirty limit" now cites
  \cref{eq:intro_eilenberger} at the Eilenberger reintroduction.
- **Machine-B finding 11**: \label{sec:usadel_longitudinal} MOVED from
  "Agreement with the scalar route" to "Longitudinal distribution operator"
  (where the D_L/D_T traces actually live, matching this file's own usage) and
  is now referenced by the §II Dynes pointer.
- **Dynes footnote final form** (refines the item-5 execution): the full
  algebraic discussion now lives in SM app:dynes_remark (end of the derivation
  appendix); the main-text footnote at the channel-dictionary sentence is a
  compact pointer. Rationale: the 17-line footnote overfilled whichever §III
  page anchored it (23.9–28.7 pt vbox under three different anchors). Don't
  re-inflate the main-text footnote.
- **S-symbol overload (item 8): deliberately NOT renamed.** The
  channel-dictionary parenthetical disambiguates; a true rename touches the
  boxed, sympy-verified SM supercurrent equation for cosmetic gain. Closed.

### Settled items inherited from paper1/paper2 (verified in those review rounds)

- **Advanced-propagator convention** \(\hat g^A=-\tau_3\hat g^{R\dagger}\tau_3\)
  (\(=-\hat g^R\) above the gap) is the 2026-06-09 errata result; the text
  documents why Belzig et al. Eq. (49) (opposite anomalous sign) is NOT used.
  Don't re-flag either direction.
- **Channel dictionary** (tab:channel_dictionary): \(\mathcal D_L=1/0\) above/below
  gap, \(\mathcal D_T=N_1^2\) (charge), \(\mathcal D_L\mathcal D_T=N_1^2\) generally;
  \(N_1^2\) dressing belongs to the TRANSVERSE channel. Sympy-verified (I7/I8 etc.).
- **A1 is selected by BOTH routes**; taxonomy rows B \((0,-2)\)/C \((0,-1)\) are
  legacy placements (Fick ansätze for \(f\)), kept as labeled diagnostics only.
  The benchmarks quantify the legacy-placement artifact (spurious drift,
  self-focusing); the microscopic operator predicts NO self-focusing.
- **Kupriyanov–Lukichev scalar BC** (eq:scalar_BC_energy): energy weight
  \(N_1N_1'-N_2N_2'\) (regular at matched gaps), charge weight \(N_1N_1'\)
  (carries the SIS edge singularity); the CURRENT is continuous, \(f\) jumps
  (Robin condition). Don't swap the weights.
- **Time-dependent spectral flow** is produced directly by the fixed-\(E\)
  projection (eq:fixedE_conservative_flow); the moving-\(\xi\)/branch-projector
  constructions are re-expressions, not missing physics. Settled June 2026 —
  don't re-open the "reformulation" question.
- **Gap-feedback closure** (eq:gap_feedback_closure) is the exponentiated
  T-free form; \(n_{qp}=4N_0\int N_1 f\,dE\) with single-spin \(N_0\)
  (factor 4 = spin × two \(\xi\) branches). Deliberate.
- **Appendix A (from paper2)** holds the explicit \(\bk\to(E,\hat\bk)\)
  change of variables; the chain-rule cancellation guard above
  (`eq:intro_boltzmann_substituted`) now lives THERE, kept by Soren's
  directive ("rarely done explicitly in the literature").

---

## Known incomplete (work in progress — not defects)

- Abstract: merged draft written 2026-07-01; sharpened 2026-07-02 with
  Soren-approved wording (the legacy placement named as in common use in
  device modeling, including published gap-engineered-trap analyses —
  backed by the verified riwar2019 Appendix-A instance).
- All former paper3 gaps are closed in the merge: `sec:sc_scalar_equations`
  references were remapped to `eq:beta_f_from_modes`, the supplement is now
  local (SM- refs resolve), and Data/code availability carries the paper1 URLs.

---

## Build

- `make bootstrap` on a clean tree (paper.tex and supplement.tex cross-reference
  via xr-hyper, each needs the other's .aux), plain `make` afterwards;
  `make roadmap` rebuilds the routes figure; `make verify` runs the sympy
  proof-check scripts (needs `../.venv` or `make setup`).
- Clean state (as of the 2026-07-04 second session, items 6–10 + MB-6/11):
  paper.pdf 52pp, supplement.pdf 48pp, ZERO undefined references, zero
  overfull hboxes, ONE structural overfull \vbox (~29pt, p. 29 — the §III
  trace-derivation/table stretch; returned when the item-6/7 content
  additions reflowed §III, persists at 24–29pt under any Dynes-footnote
  anchor, so it is display+table density, not footnote mass — note only per
  the standing rule). `make verify` 7/7 no-FAIL. On this Windows box the
  verify venv is the A: archive clone's
  (`A:\Einstein\Documents\qp-diffusion-paper\.venv`) — the qpsim repo venv
  has no sympy and the Makefile's `../.venv/bin/python` default is the Mac
  layout.
