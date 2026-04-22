---
title: Phonon Escape Time (Acoustic-Mismatch Derivation)
description: Derivation of the thin-film phonon escape time τ_l ≈ 4d/(ηs), reference chain, numerical examples, and the double-counting argument.
sourced_from: Phonon Analysis.md (Gate 0 working draft, 2026-04-12), §6
---

# Phonon Escape Time

## Scope

This document fixes the acoustic-escape model for $\tau_l$ used in the
Ph0 phonon tier of the greenfield framework (see `Phonon_Model_Decisions.md`
for the committed decisions D1–D5). The escape time is a **thin-film
acoustics** problem — it is set by the film geometry and the acoustic
impedance contrast between film and substrate. It is **not** a
superconducting-kinetics quantity; the electron–phonon coupling enters
nowhere in the derivation.

---

## 1. Physical setup

Consider a superconducting film of thickness $d$ deposited on a
semi-infinite substrate. A phonon in the film has sound velocity $s$
(Debye average for a single-branch Ph0 model) and propagates
isotropically. The film has two surfaces:

- **Substrate-side surface.** On impact, a phonon transmits into the
  substrate with angle- and mode-dependent probability; after angle
  averaging we call this $\eta \in (0,1]$. Reflection probability is
  $1-\eta$.
- **Free-side surface.** A superconductor–vacuum interface. Acoustic
  impedance mismatch is essentially infinite, so reflection is total
  (probability $1$).

The escape time $\tau_{\mathrm{esc}}$ is the mean lifetime of a phonon
in the film against transmission through the substrate-side surface.
Once transmitted, the phonon is absorbed into the substrate bath.

In the Fischer–Catelani 2023 convention used here, $\tau_l$ is
identified with $\tau_{\mathrm{esc}}$:
$$
\tau_l \equiv \tau_{\mathrm{esc}}.
$$

---

## 2. Derivation of $\tau_l \approx 4d/(\eta s)$

The derivation assembles four pieces:

1. **One-way transit time.** A phonon moving normal to the film crosses
   it in time $d/s$. For oblique propagation at angle $\theta$ from the
   film normal, the projected transit time is $d/(s\cos\theta)$.

2. **Bounces before escape.** With total reflection on the free side
   and transmission probability $\eta$ on the substrate side, a phonon
   makes on average $1/\eta$ **substrate-side** encounters before
   escaping. Between consecutive substrate-side encounters it bounces
   once off the free side, so the total number of one-way transits is
   roughly $2/\eta$.

3. **Angular average.** For an isotropic distribution of phonon
   directions in the film, the flux onto a surface element scales as
   $\cos\theta$. Averaging $1/\cos\theta$ over this flux-weighted
   distribution gives
   $$
   \langle 1/\cos\theta \rangle_{\text{flux}}
   = \frac{\int_0^{\pi/2} (1/\cos\theta)\cos\theta\sin\theta\,d\theta}
          {\int_0^{\pi/2} \cos\theta\sin\theta\,d\theta}
   = \frac{1}{1/2}
   = 2.
   $$
   The mean actual path length for one transit is therefore $2d$ and
   the mean transit time is $2d/s$.

4. **Assembly.** Combining bounces and angular averaging:
   $$
   \tau_{\mathrm{esc}} \approx \underbrace{\frac{2}{\eta}}_{\text{bounces}}
   \times \underbrace{\frac{2d}{s}}_{\text{flux-averaged transit}}
   = \frac{4d}{\eta\, s}.
   $$

The factor-of-4 assembly is the canonical form quoted in Kaplan 1979 and
Eisenmenger 1976. More careful treatments that integrate angle-dependent
transmission $T(\theta)$ against the acoustic-mismatch model yield the
same scaling with an order-unity geometric prefactor set by the angular
profile of $T(\theta)$.

---

## 3. Acoustic-mismatch model for $\eta$

At normal incidence across a film/substrate interface (Little 1959):
$$
\eta_\perp = \frac{4 Z_1 Z_2}{(Z_1 + Z_2)^2},
\qquad Z_i = \rho_i s_i,
$$
where $\rho_i$ is mass density and $Z_i$ is acoustic impedance. The
angle-averaged transmission, accounting for mode conversion and
total-internal-reflection cutoff at the critical angle
$\theta_c = \arcsin(s_{\mathrm{film}}/s_{\mathrm{sub}})$, is:
$$
\eta = \frac{\int_0^{\pi/2} T(\theta)\cos\theta\sin\theta\,d\theta}
            {\int_0^{\pi/2} \cos\theta\sin\theta\,d\theta}.
$$
For most MKID-relevant film/substrate combinations, the critical-angle
cutoff and mode-conversion losses drive $\eta$ well below $\eta_\perp$.
Canonical values: Al on sapphire, $\eta \approx 0.2$ (Fischer 2023 Ref.
26); Al on silicon, $\eta \approx 0.3\text{–}0.5$; Nb on silicon,
$\eta \approx 0.2\text{–}0.3$.

The AMM treats the interface as two elastic half-spaces. A competing
model is the diffuse-mismatch model (DMM), in which interface roughness
scrambles the angle of the transmitted phonon. The two models agree
at low frequency (wavelength much larger than roughness scale) and
diverge near the Debye frequency. For pair-breaking-relevant phonons
in Al ($\omega \sim 2\Delta \sim 0.36$ meV, $\lambda \sim 30$ nm) the
AMM is adequate.

---

## 4. Reference chain

The $\tau_l \approx 4d/(\eta s)$ result traces to three primary papers:

- **Little 1959** (`docs/references/little_1959.pdf`). The
  acoustic-mismatch model for the Kapitza boundary resistance between
  two elastic media. Source of the $\eta_\perp = 4Z_1 Z_2 / (Z_1+Z_2)^2$
  formula and the angle-averaging machinery.
- **Eisenmenger 1976** (`docs/references/eisenmenger_1976.pdf`). Phonon
  trapping in superconducting tunnel junctions. Works out the
  film-geometry multi-bounce physics and establishes the $4d/(\eta s)$
  scaling.
- **Kaplan 1979** (`docs/references/kaplan_1979.pdf`). Applies the
  acoustic-escape model to thin-film superconductors relevant to
  detector physics; this is the reference that Fischer–Catelani 2023
  Appendix A Eq. (A5) cites for $\tau_l$.

**Essential distinction: Kaplan 1976 ≠ Kaplan 1979.**

| Paper | Quantity | Physics | Formula |
|---|---|---|---|
| Kaplan **1976** (PRB 14, 4854) | $\tau_{\mathrm{PB}}(\omega)$ | Bulk BCS pair-breaking lifetime | $1/\tau_{PB} = (\Delta/\pi\Delta_0\tau_0^{PB})\,S_+(\omega/\Delta)$ |
| Kaplan **1979** (JLTP 37, 343) | $\tau_l(\omega)$ | Thin-film acoustic escape | $\tau_l \approx 4d/(\eta s)$ |

Kaplan 1976 gives $\tau_{\mathrm{PB}}$ via elliptic integrals and has
nothing to do with film geometry or the substrate. Kaplan 1979 gives the
escape time and has nothing to do with the superconducting gap or BCS
coherence factors. Citing Kaplan 1976 as the source of $\tau_l$ is a
common mistake.

---

## 5. Numerical example: Al on sapphire (Fischer 2023 parameters)

Film parameters (Fischer & Catelani 2023 Table II):

- $d = 90$ nm (the v1 Gate 4 benchmark uses a film from that
  parameter family; Fischer's own baseline is $d = 63$ nm and gives
  $\tau_l \approx 170$ ps).
- $\eta \approx 0.2$ (Al/Al$_2$O$_3$, angle-averaged AMM, Fischer 2023
  Ref. 26).

Sound velocities in Al: $s_L = 6420$ m/s (longitudinal),
$s_T = 3040$ m/s (transverse). The Debye average is
$s_D \approx 3200$ m/s.

Evaluating $\tau_l = 4d/(\eta s)$ for $d = 90$ nm and $\eta = 0.2$:

- **Transverse** ($s_T = 3040$ m/s):
  $\tau_l = 4 \times 90\,\text{nm} / (0.2 \times 3040\,\text{m/s})
       = 3.6\times 10^{-7} / 608 \approx 592$ ps.
- **Longitudinal** ($s_L = 6420$ m/s):
  $\tau_l = 4 \times 90\,\text{nm} / (0.2 \times 6420\,\text{m/s})
       \approx 280$ ps.
- **Debye-averaged** ($s_D = 3200$ m/s):
  $\tau_l \approx 563$ ps.

For the Fischer baseline $d = 63$ nm with an effective
$s \approx 5000$ m/s (between $s_D$ and $s_L$) and $\eta = 0.2$:
$\tau_l \approx 252$ ps; Fischer reports $\tau_l = 170$ ps
(consistent within the geometric prefactor ambiguity and the
$s$ choice).

The v1 validation suite should reproduce Fischer's $\tau_l = 170$ ps at
$d = 63$ nm under the documented $\eta$, $s$, and angular-averaging
convention (see §6 below for the reabsorption consistency check).

---

## 6. Double-counting and the reabsorption question

A frequent concern: if we evolve $n_{ph}(\omega,\mathbf{r},t)$
explicitly as a PDE variable **and** apply a phonon-trapping factor
$\tau_l$ in the bath-relaxation term, do we double-count the
reabsorption physics that the Rothwarf–Taylor trapping factor
$\zeta = 1 + \tau_l/\tau_{\mathrm{PB}}$ encodes?

**Answer: no, under the committed decisions.** Two independent reasons:

1. **We do not apply $\zeta$ in the PDE.** The Rothwarf–Taylor
   composition renormalizes the recombination time,
   $\tilde\tau_0 = \tau_0\,\zeta$, as a rate-equation closure when
   $n_{ph}$ is eliminated adiabatically. In the PDE backends this
   elimination has not been performed — $n_{ph}$ is a live dynamical
   variable. Committing to evolve $n_{ph}$ explicitly **and** to
   renormalize $\tau_0 \to \tau_0\,\zeta$ on the QP side would indeed
   double-count. The Gate 0 decision forbids this mixed configuration;
   the construction-time validator rejects it.

2. **$\tau_l$ and $\tau_{\mathrm{PB}}$ act on different physics.**
   These two timescales appear in different terms of different
   equations:

   - $\tau_l$ lives in the phonon bath-relaxation term
     $-(n_{ph}-n_{BE}(T_B))/\tau_l$, modeling **acoustic escape** of
     the local phonon population into the substrate. It removes phonons
     from the film; the rate has no dependence on the QP distribution
     $f(E)$ or the gap $\Delta$.
   - $\tau_{\mathrm{PB}}$ lives inside $I_{qp\to ph}$, specifically in
     the $K^+$ pair-breaking (recombination) term, modeling
     **electron–phonon coupling**: a phonon of energy $\omega \ge 2\Delta$
     breaks a Cooper pair, creating two QPs. The rate depends on the
     gap and the QP occupation via $(1-f(E))(1-f(\omega-E))$.

   Reabsorption — the "trap" in phonon trapping — is the absorption
   leg of $I_{qp\to ph}$: the $(1-f(E))(1-f(\omega-E))\,n(\omega)$
   term. This is already in the collision integral. It does not need
   to be modeled separately via a $\zeta$ factor when $n_{ph}$ is
   dynamical.

In the strong-trapping limit $\tau_l/\tau_{\mathrm{PB}}\gg 1$, the
dynamic-phonon solve produces a large nonequilibrium $n_{ph}(\omega)$
concentrated near $\omega \approx 2\Delta$; this enhances the absorption
leg of $I_{qp\to ph}$ and slows effective recombination by exactly the
factor $\zeta$ that the rate-equation closure predicts. Eliminating
$n_{ph}$ adiabatically from the full coupled system should reproduce the
$\zeta$-renormalized QP rate equation; a constructive algebraic proof
of this equivalence is an open derivation gap (Phonon Analysis §13).

For v1 code, the practical rule is simple:

| Backend | Use $\zeta$? | Use dynamic $n_{ph}$? |
|---|---|---|
| PDE backends (T2, T3) with Ph0 | **No** (forbidden) | **Yes** |
| Rate-equation service (M25-style) | **Yes** | **No** |

Mixed configurations are construction-time errors.

---

## 7. Functional shape of $\tau_l(\omega)$

Unlike $\tau_{\mathrm{PB}}(\omega)$, $\tau_l(\omega)$ has **no singular
behavior at $\omega = 2\Delta$**. The acoustic-escape physics makes no
reference to the superconducting gap. $\eta(\omega)$ varies smoothly
with frequency through the frequency dependence of the AMM transmission,
and for wavelengths large compared to the interface-roughness scale
(all $\omega$ of interest to pair-breaking kinetics in Al) it is
essentially constant.

Smooth limits:

- **Low-$\omega$** ($\omega \to 0$): $\eta$ approaches the AMM
  angle-averaged value at the long-wavelength limit. $\tau_l$ is
  finite and well-defined; there is no singular behavior near
  $\omega = 0$ beyond the density-of-states prefactor $\omega^2$ in
  phonon phase-space measures (which does not enter $\tau_l$ itself).
- **Near the gap** ($\omega \to 2\Delta$): **completely smooth** — no
  step, no square-root onset, no kink. Contrast $\tau_{\mathrm{PB}}$,
  which jumps from infinity (no pair-breaking for $\omega < 2\Delta$)
  to a finite value at the threshold (step-like onset from
  $S_+(2) = \pi$).
- **High-$\omega$** (toward Debye): $\eta(\omega)$ may acquire weak
  frequency dependence as the DMM regime sets in, but $\tau_l(\omega)$
  remains a smooth, bounded function with no characteristic features
  at any BCS energy scale.

This cleanness of $\tau_l(\omega)$ is a numerical convenience: the
bath-relaxation operator in the Ph0 equation is well-conditioned at
all frequencies, and no special grid refinement is needed at
$\omega = 2\Delta$ on the phonon-transport side. (Refinement **is**
needed there for the collision-integral side, driven by
$\tau_{\mathrm{PB}}(\omega)$ and the BCS DOS.)

---

## 8. Validation

The following checks are required before $\tau_l$ can be used in
quantitative simulations:

1. **Escape-only relaxation.** With $I_{qp\to ph} = 0$ as a forcing,
   $n_{ph}$ relaxes exponentially to $n_{BE}(\omega,T_B)$ with time
   constant $\tau_l(\omega)$. This is a pure transport-term test.
2. **Fischer baseline.** With the Al/sapphire parameter set
   ($d = 63$ nm, $\eta = 0.2$, $s \approx 5000$ m/s), $\tau_l$ from the
   acoustic-escape formula agrees with Fischer's $\tau_l = 170$ ps
   within the geometric-prefactor tolerance documented here.
3. **Trapping-factor consistency.** A rate-equation reduction using
   $\zeta = 1 + \tau_l/\tau_{\mathrm{PB}}$ should match the PDE solve
   in the quasi-steady phonon limit (small deviation from BE, $f$
   close to a Boltzmann tail). Outside this regime the PDE solve is
   allowed (and expected) to depart from the $\zeta$-closure result.

---

## References

- Little 1959 — AMM Kapitza resistance (`docs/references/little_1959.pdf`).
- Eisenmenger 1976 — phonon trapping in tunnel junctions
  (`docs/references/eisenmenger_1976.pdf`).
- Kaplan 1979 — $\tau_l \approx 4d/(\eta s)$ for thin superconducting
  films (`docs/references/kaplan_1979.pdf`).
- Kaplan 1976 — **distinct**; gives $\tau_{\mathrm{PB}}(\omega)$, not
  $\tau_l$ (`docs/references/kaplan_1976.pdf`).
- Fischer & Catelani 2023 — Appendix A Eq. (A5) for $\tau_l$ and Eq.
  (A2) for $\tau_{\mathrm{PB}}$
  (`docs/references/fischer_catelani_2023.pdf`).
- `Phonon_Model_Decisions.md` — committed decisions D1–D5 and the
  three-timescale glossary.
