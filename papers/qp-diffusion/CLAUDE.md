# Merged Paper — Conventions and Review Notes

**Purpose.** This file records notation choices and physics points that have
already been derived, checked, and *deliberately* chosen. It exists so a
reviewer — human or AI — can spend effort on genuinely new issues rather than
re-flagging settled ones. Everything below was vetted (several adversarial
review passes); please do not report these as errors without substantively new
argument. Finding *new* problems is welcome; re-deriving these is not.

> **FIRST-PASS STATUS CORRECTION — 2026-07-10 (read with the second-round
> refinement immediately below; supersedes the confident framing in this
> Purpose note and in the 2026-07-04/07 blocks below).** An external
> adversarial review (GPT, `ADVERSARIAL-REVIEW-2026-07-10.md` in this dir) plus an
> independent per-issue physics audit
> (`GPT-REVIEW-PHYSICS-AUDIT-2026-07-10.md` in the Soren folder) found that ~18 of
> GPT's 19 findings hold up under independent scrutiny — several **blocker-class
> physics** issues: the nonadiabatic verifier omits a same-order Keldysh term (B2);
> a displayed projection-average commutator is false (B4c); the O(Q³)
> supercurrent-mixing onset is nonuniform (B3); the KL boundary law mixes current
> normalizations (B5); the starting-equation convention bridge is incomplete (M1).
> **The paper is NOT submission-ready.** Two rules this whole file must now be read
> under: (1) "sympy-verified" proves only *algebraic consistency given the starting
> equations*; "settled" means an *editorial choice* — NEITHER is evidence of
> *physical* correctness (a step can be algebraically exact yet physically wrong —
> the audit's own B1 agent "confirmed" a result by solving the wrong ODE). (2) Do
> NOT apply the queued C1/C2 pre-submission package as-is: its C1 derivation as
> drafted would insert a *false theorem* (B1). The blocker findings need a human
> physicist to adjudicate before any fixes or submission. This was the
> pre-counter-audit status; the next block supersedes its tally and severity.

> **SECOND-ROUND REFINEMENT — 2026-07-10 (supersedes the numerical tally and
> blocker severity above).** The counter-audit is complete; see
> `ADVERSARIAL-REVIEW-2026-07-10.md`, section *Second-round counter-audit*.
> It confirmed B2, B3, the false B4c display, M1's incomplete convention map,
> M3's absent transverse-response loop, and the M5 benchmark overclaim. It
> also found new errors in the external audit: its B1 window average clamps
> the uncomputed edge and does not establish a cutoff-insensitive sub-percent
> rescue; its moving-coordinate recommendation omits the outer covariant
> derivative; its finite-Dynes edge value is not exactly one half; and its M5
> superposition defense is invalid for the nonlinear gap closure. M6 is
> **refuted**: public `main` is `7116d6e`, contains all seven scripts, and its
> normalized Git blobs match manuscript commit 8598056. B5 is a mandatory
> current-normalization/notation repair but the KL trace derivation and the
> supplement's side-labeled law are correct; M1 is a convention-bridge repair,
> not a failed downstream derivation. B1 is the only narrow **BLOCKER** because
> it prevents the named C1 insertion; multiple **MAJOR** corrections still
> prevent submission. Do not apply C1/C2 or alter the physics until a human
> physicist has adjudicated the surviving derivation-level findings.

> **IMPLEMENTATION STATUS - 2026-07-11 (current; supersedes the repair status
> above, but not its audit history).** The user authorized the non-human
> manuscript repairs. The audited scope/reference fixes are in `01ecd61`; the
> convention, projection, spatial-curvature, and supercurrent-edge repairs are
> in `46c71f2`; the KL current normalization and phase scope are repaired in
> `76d822e`; and the complete star-normalized nonadiabatic spectral/Keldysh
> expansion is in `3423000`. Two independent post-patch reviews found no
> remaining coefficient or sign defect. The final gate is 7/7 symbolic scripts
> PASS (266.8 s); `paper.pdf`/`supplement.pdf` build cleanly at 55/52 pages with
> zero undefined references/citations and zero overfull boxes, and the changed
> pages passed visual inspection. **The paper is still NOT submission-ready:**
> B1 keeps the queued C1/C2 package on technical hold, and M5's demonstrated
> passive-tracer inward drift still needs human-approved replacement of the
> stronger self-focusing language in the protected abstract/caption/body echo.
> The author's abstract pass and every D1-D7 decision remain human-only.

> **INDEPENDENT VALIDATION - 2026-07-11.** A separate blind 44-unit review,
> followed by independent refutation of all nine candidates, found zero
> surviving blocker, major, or minor physics errors in the delimited sector.
> Its two surviving NOTE-level presentation issues were fixed in `d9e64b2`
> (the branch-odd charge-imbalance symbol and adjacent nonadiabatic trace
> normalization). The review is recorded in
> `INDEPENDENT-ADVERSARIAL-REVIEW-2026-07-11.md`. It independently retained B1
> and M5 as human gates; all other human-only items below remain untouched.

> **APPROVED GATE DISPOSITION - 2026-07-11 (current; supersedes the open-gate
> conclusions in the two blocks immediately above).** After three independent
> advisory panels and the independent reviewer's concurrence, the user explicitly
> approved the bundle. Commit `229956b` replaces the guarded abstract, removes the
> reciprocal self-focusing overclaim in favor of passive-probe language, adds a
> qualitative eliminated-proximity-layer resistance caveat without revising any
> Riwar--Catelani number, adds a strictly local fixed-energy Peclet diagnostic,
> and tailors the three inhomogeneous benchmark captions. The prepared quantitative
> C1 correction, `7 -> 32 um` result, device-error table, and old SM verification
> sentence remain rejected. D4 is "do not contact now" and PyPI remains deferred;
> neither is a manuscript gate. Three focused rechecks plus a final whole-diff
> review pass. The seven-script suite is 7/7 PASS (266.5 s), the final PDFs are
> 56/52 pages with zero undefined references/citations and zero overfull boxes,
> and all affected pages passed visual inspection. The complete decision record is
> `PRESUBMISSION-GATE-DISPOSITION-2026-07-11.md`.

Scope: THIS is the single merged manuscript (assembled 2026-07-01). Base =
paper3 wording/template (intro + scalar route + agreement); imported from
paper1: the Usadel-route derivation (longitudinal operator, D_L/D_T traces,
flux-to-scalar, time-dependent spectral flow), taxonomy, consistency
checks/benchmarks, conclusion, the entire Supplemental Material, figures, and
verify scripts; imported from paper2: the explicit change-of-variables
appendix. The sibling directories paper1/ paper2/ paper3/ are ARCHIVED
sources — editing them does NOT change this manuscript.

## References

Kopnin materials live at `B:\AEinstein\Einstein\Documents\Soren\kopnin-numbered-equations\`
(on this Windows box): the full book PDF ("Kopnin, Theory of Nonequilibrium
Superconductivity (2001).pdf"), numbered-equation transcriptions of
Chapters 10 and 15 ("Local Chapter Copies\Kopnin Chapter 10.tex" / "... 15.tex"),
and the transcription-audit tooling. A second copy of the Ch. 10/15
transcriptions is at `G:\My Drive\qp-diffusion-handoff\kopnin\`.
Chapters 10 and 15 are the most relevant; Ch. 1 (the \(\nu(0)\)
single-spin DOS definition) is in the book PDF.
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
  2026-07-04 deep-review follow-up (Soren-approved): the flag now carries the
  even-harmonic qualifier inline ("an identity for the even angular harmonics
  only; the odd harmonics swap sectors, eq:intro_parity_dictionary"), so the
  early preview can no longer be read as extending \(\fT=\lambda\phi_T\) to odd
  harmonics — consistent with the parity-dictionary guard above.
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
- **Dynes pointer final form** (refines the item-5 execution): the full
  algebraic discussion lives in SM app:dynes_remark (end of the derivation
  appendix); the channel-dictionary paragraph carries one compact pointer
  sentence. The pointer was moved out of a footnote after the approved abstract
  repagination put that insertion beside the large dictionary float and created
  a small overfull page; no scientific content changed. Don't re-inflate it.
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
  The current benchmark establishes legacy-placement passive-tracer drift into
  a population-generated gap well and zero static DOS-gradient tracer drift for
  A1. It does not establish reciprocal net self-focusing of one coupled
  population; commit `229956b` now states that limitation consistently in the
  abstract, benchmark setup/body/caption, and conclusion.
- **Kupriyanov–Lukichev scalar BC** (eq:scalar_BC_energy): energy weight
  \(N_1N_1'-N_2N_2'\) (regular at matched gaps), charge weight \(N_1N_1'\)
  (carries the SIS edge singularity); the physical spectral-current density
  \(\mathcal J\) is continuous while the diffusion-normalized side fluxes
  differ with \(N_{0i}\), and \(f\) jumps (Robin condition). Don't swap the
  weights or equate the side fluxes for dissimilar materials.
- **Time-dependent spectral flow** is produced directly by the fixed-\(E\)
  projection (eq:fixedE_conservative_flow); the moving-\(\xi\)/branch-projector
  scalar construction is an exact coordinate relabelling away from the gap
  edge. It is not an intrinsic moving-frame matrix derivation and is nonuniform
  at \(\xi=0\).
- **Gap-feedback closure** (eq:gap_feedback_closure) is the exponentiated
  T-free form; \(n_{qp}=4N_0\int N_1 f\,dE\) with single-spin \(N_0\)
  (factor 4 = spin × two \(\xi\) branches). Deliberate.
- **Appendix A (from paper2)** holds the explicit \(\bk\to(E,\hat\bk)\)
  change of variables; the chain-rule cancellation guard above
  (`eq:intro_boltzmann_substituted`) now lives THERE, kept by Soren's
  directive ("rarely done explicitly in the literature").

---

## 2026-07-07 — per-subsection review round (44 referees + CLAUDE.md skeptic), ALL FIXES APPLIED

- **What ran**: one dedicated Fable/xhigh referee per subsection of both
  documents (44 units) instructed to trust this file, plus one adversarial
  skeptic auditing this file itself. Full report (verdicts, findings,
  derivations, dispositions):
  `B:\AEinstein\Einstein\Documents\Soren\qp-diffusion-SUBSECTION-REVIEW-2026-07-07.md`.
- **Outcome**: 0 critical / 9 major (8 distinct) / 74 minor / 77 nit; no
  finding changed a physics result; everything actionable applied
  2026-07-07 on `fix/gpt-review-2026-07-05` (this round also executed all
  gpt_review.txt PAPER items).
- **Major fixes** (details in the report): nonadiabatic source
  characterization scoped to gap-slaved distributions + O(ℏ²)
  sector/robustness sentence corrected (SM app:nonadiabatic; main-text echo
  at sec:tdep_spectral_flow) — both now MACHINE-VERIFIED by the new
  block (d) of verify_nonadiabatic.py; stale "negative result" opener of
  SM sec:coordinate_lift rewritten to the settled positive framing; N₂
  sign in the SM Conventions display fixed (Re, parallel to N₁); KL Robin
  matching now carries σ_i=2e²N₀D_N (SM + main text, with the
  diffusion-units g_N=G_N/2e²N₀ bridge); outline preview of the matrix
  Keldysh–Usadel equation carries its factor of i; Data-availability claim
  made true by pushing verify_tdep_inhomogeneous.py to the public repo;
  benchmark figure PDFs committed with Makefile provenance rules.
- **New settled decisions from this round** (don't re-litigate):
  \(\mathcal D_L,\mathcal D_T\) calligraphic everywhere incl.
  app:supercurrent/app:proximity; product-symbol bridges (∘ = main text ⊗;
  ⋆ = its (E,t) restriction) stated at SM Conventions, eq:moyal_def, and
  app:branch setup; taxonomy conserved-density weight is \(\mathcal W\)
  (kernel rate w unchanged); §V.B retitled "Conserved currents and
  boundary conditions"; branch-Boltzmann route uses τ (SM convention, =
  main-text τ_N) and bold **a** for the P1 harmonic and u²(E)−v²(E)
  on-shell coherence factors; intro opening reworded so abstract and intro
  no longer share their first ten words. COMPLETE record of abstract
  edits this round (the abstract is the guarded pre-submission gate, so
  this list is exhaustive): symbol glosses (\(N_1,\tau_N,\DN,\DE\), "for
  the quasiparticle occupation \(f\)"), one dash-for-comma swap in the
  guarded sentence's tail, and the \(\bnabla_{\br}\to\bnabla\) notation
  unification at the spectral-current line. Nothing else.
- **verify_nonadiabatic.py strengthened**: structural no-coupling checks
  are now `.has()`-based (see derivative-carried dependence, which
  `.diff()` missed) and a "(d) 2026-07-07 corrected-claims" block pins the
  generic τ₁ O(ℏ²) source, its static-gap survival, its gap-slaved
  vanishing, and the τ₂/τ₃-sector dg solve. Full suite re-run 2026-07-07
  with the A: venv: 7/7 clean, ALL PASS.
- **gpt_review.txt disposition**: paper items all fixed (P1 figure
  packaging; P2 τΔ/ℏ; P3 benchmark-1 wording, Dynes pointer word,
  log warnings addressed by rebuild). ENGINE items deliberately deferred
  to a qpsim code session: solve_gap near-T_c bracketing, spatial-backend
  geometry/conductance validation, webui path containment, CI ruff red,
  CI slow-coverage gap, picard mixing=0 false convergence, sympy in dev
  deps + Makefile PY default (the last one IS fixed here). On gpt P3 "log
  warnings": stuck-float and stale warnings cleared by the rebuild; the
  SM-/M- dest-warning class persists on this box (xr-hyper v6.00beta4
  limitation, see the corrected Build note) until the channel-dictionary
  short-caption fix + snapshots (20569b4) are exercised on a modern
  toolchain.
- **Skeptic audit disposition**: C1 (record gpt_review) = this block;
  C2 Dynes rationale corrected above; C3 Kopnin pointer made precise —
  NOTE the skeptic's "directory missing" claim was itself wrong (the
  directory exists in the Soren folder, including the full book PDF);
  C4 riwar2019 evidence recorded under Known-incomplete; C5 abstract
  guard scope clarified.
- **REMAINING after this round** (SUPERSEDED by the approved-gate block at the
  top of this file): B1/C1/C2, M5, D1-D7, and the abstract pass have received
  their recorded dispositions. The optional riwar2019 published-equation
  pinpoint, deferred package release/contact, and engine work are not paper
  correctness gates.

---

## 2026-07-11 - adversarial repair round

- **Applied:** B2, B3, B4a-e, B5, M1-M4, and M7-M9, plus the newly found
  local-BCS spatial-curvature completion. The exact commit map is recorded in
  `ADVERSARIAL-REVIEW-2026-07-10.md` under *Implementation resolution*.
- **Verification:** all seven `verify_*.py` scripts pass in the pinned A: SymPy
  environment. The strengthened nonadiabatic verifier checks both spectral
  equations, R/A/K star normalization, the missing Keldysh anticommutator,
  the completed generic coherence trace, exact shell-slaved and equilibrium
  cancellations, and zero scalar/transverse second-order projections.
- **Subsequent disposition:** the user approved the advisory bundle and the
  manuscript gates were resolved in `229956b`; see
  `PRESUBMISSION-GATE-DISPOSITION-2026-07-11.md`. PyPI publication and external
  contact remain deliberately unexecuted and are not paper correctness gates.

---

## Known incomplete (work in progress — not defects)

- Abstract: the previous protected draft and its single-sentence guard are
  superseded. The user approved the complete replacement on 2026-07-11 after
  independent abstract/M5 review. The current version is scoped to a homogeneous
  normal-state material in the dirty, slowly varying local-BCS sector, names one
  published trap analysis, uses "not selected by either reduction," and expressly
  disclaims net nonlinear focusing. This abstract pass is complete.
- riwar2019 Appendix-A evidence (recorded 2026-07-07, re-verified against
  arXiv:1907.04781 via ar5iv): their Eq. (46) [arXiv numbering; PRB
  appendix (A-)number still to be read off the journal PDF] is
  \(\dot f_{\rm qp}(\epsilon,y)=\partial_y[D_{\rm qp}(\epsilon,y)\,
  \partial_y f_{\rm qp}]\) with
  \(D_{\rm qp}=D_0/\nu_{\rm BCS}(\epsilon)\),
  \(\nu_{\rm BCS}=\epsilon/\sqrt{\epsilon^2-\Delta^2(y)}\), attributed by
  them to Belzig et al. 1999 — exactly placement C at a varying gap.
  Referee-risk pre-emption: those authors are plausible referees; if
  desired, add the (A-)equation pinpoint at the citation once checked.
- All former paper3 gaps are closed in the merge: `sec:sc_scalar_equations`
  references were remapped to `eq:beta_f_from_modes`, the supplement is now
  local (SM- refs resolve), and Data/code availability carries the paper1 URLs.

---

## Build

- `make bootstrap` on a clean tree (paper.tex and supplement.tex cross-reference
  via xr-hyper, each needs the other's .aux), plain `make` afterwards;
  `make roadmap` rebuilds the routes figure; `make verify` runs the sympy
  proof-check scripts (needs `../.venv` or `make setup`).
- Clean state (as of the 2026-07-07 review-fix rebuild): paper.pdf 54pp,
  supplement.pdf 49pp, ZERO undefined references, ZERO overfull hboxes,
  and ZERO overfull vboxes — the long-standing ~29 pt §III vbox vanished
  when the review round compressed the duplicated "Time-dependent spectral
  flow" paragraph in §III.C (pagination luck regained; if future edits
  reflow §III it may return — note-only per the standing rule). The only
  remaining pdfTeX dest warnings ARE the cross-document SM-/M- anchors
  (CORRECTED 2026-07-07, second pass — the first record here wrongly
  called them cosmetic/fixed): this box's xr-hyper is v6.00beta4 (2000)
  and IGNORES the `\externaldocument[..]{..}[url]` URL argument, so on
  THIS machine the cross-document links are dead, and where anchor names
  collide across the two PDFs (`figure.N`, `table.N`) a few links land
  silently on the wrong local object. Reference NUMBERS are all correct.
  The URL arguments are KEPT because modern toolchains honor them (being
  confirmed by the 2026-07-07 arXiv/TeX-Live dry run). Do not claim
  working local cross-document links in any record. `make verify` 7/7 no-FAIL (now
  including the block-(d) corrected-claims checks). On this Windows box
  the verify venv is the A: archive clone's
  (`A:\Einstein\Documents\qp-diffusion-paper\.venv`); the Makefile's `PY`
  default now matches `make setup` (paper-local `.venv`). Rebuild recipe
  on this box: `sanitize_aux.py` (tracked) on the other document's .aux
  before each latexmk pass, alternating supplement/paper twice — see
  rebuild-2026-07-07.log for a clean transcript.
