# M25 coefficient integrals (SI Notes III + IV)

Reference transcription of the Marchegiani & Catelani 2025 Supplementary
Information equations needed to build `M25Coefficients` from a
`M25PhysicalParameters` input bundle. Source PDF:
`~/Documents/Academic Texts/Articles/Marchegiani et al., Nonequilibrium Regimes for QPs in SC Qubits - Supplementary (2025).pdf`.

Equation labels here (`S10`, `S30`, …) match the SI. Main-paper
equations are labeled `M#`.

---

## 1. Units and notation

* Energies (temperature T, gaps Δ_L/Δ_R, transition frequencies ω_10,
  gap asymmetry ω_LR = Δ_L − Δ_R, pair-breaking photon ω_ν, transmon
  parameters E_J/E_C) are all in the same energy unit. **Natural units:
  ℏ = k_B = 1.** In code we use **Kelvin** for energies (matching the
  `Delta_R_kelvin` API). The conversion to Hz is implicit: the
  tunneling-rate prefactor `R_T = g_T Δ̄/e²` is supplied directly in
  Hz on `M25PhysicalParameters.R_T_Hz`, so the dimensionless ratios of
  Kelvin energies (`y`, `y±`, `δ`, etc.) combine with `R_T_Hz` to
  give Hz output without any explicit `k_B/h ≈ 20.8366 GHz/K`
  conversion factor in the code.
* Rates are in **Hz** (not rad/s, so no extra 2π).
* Densities x_α are dimensionless, normalized to the local-gap Cooper-
  pair number per Eq. (S2).
* Gap asymmetry `ω_LR ≡ Δ_L − Δ_R > 0` (small-gap right electrode).
* `Δ̄ = (Δ_L + Δ_R)/2`.
* `y = ω_LR/(2T)`; `y_± = (ω_10 ± ω_LR)/(2T)`.
* `δ = Δ_R / Δ_L ∈ (0, 1]`.

Shorthand (S10):
* `s_{ii'} ≡ |⟨i' j̄| sin(φ̂/2) |i j⟩|`
* `c_{ii'} ≡ |⟨i' j̄| cos(φ̂/2) |i j⟩|`

Transmon matrix elements (S25–S28, E_J/E_C ≫ 1 limit):
* `s_10 ≃ (E_C / (8 E_J))^{1/4}`               (S25)
* `s_{ii} ≃ 0`                                  (S26; exp-suppressed via charge dispersion)
* `c_{ii} ≃ 1 − (i + 1/2)√(E_C/(8 E_J)) − (3/2)(i + 1/4)(E_C/(8 E_J))`    (S27; at leading order, i-independent: `c_{00} = c_{11} = 1`)
* `c_10 ≃ 0`                                    (S28; exp-suppressed)

---

## 2. M25 Fig 3 parameter set (validation target)

From the Fig 3 caption (main paper, page 4):
* Δ_R/h = 49 GHz
* ω_10/(2π) = 5.5 GHz
* E_J/h = 14.5 GHz, E_C/h = 290 MHz
* ω_ν/(2π) = 119 GHz (pair-breaking photon)
* Γ^{ph}_{00} = 300 Hz (bulk photon-assisted qubit rate, logical-state-conserving)
* ν_0 = 0.73 × 10⁴⁷ J⁻¹ m⁻³
* V_L = V_R = V = 506 × 240 × 0.028 μm³ = 3400 μm³ = 3.4 × 10⁻¹⁵ m³
* Γ^{ee}_{10} = 100 kHz (parity-preserving relaxation)
* r^L = r^{R<} = 6.25 MHz

Two panels:
* Fig 3a (small gap asymmetry): ω_LR/(2π) = 0.5 GHz → Δ_L/h = 49.5 GHz
* Fig 3b (large gap asymmetry): ω_LR/(2π) = 5 GHz → Δ_L/h = 54 GHz

The caption-listed coefficients (r^L, r^{R<}) are
**outputs** that the SI integrals must reproduce from the primitive
physical inputs (b_L, b_R, volumes, etc.). Equivalently, one can use
the r values to back out b_L, b_R if the material-dependent phonon
coupling is not independently available — see §6 below.

---

## 3. Primitive physical inputs (→ `M25PhysicalParameters`)

Required to evaluate every coefficient:

| Field              | Meaning                                                | Fig 3 value            |
|--------------------|--------------------------------------------------------|------------------------|
| `Delta_L_kelvin`   | Left-electrode gap                                     | 2.376 K (49.5 GHz)     |
| `Delta_R_kelvin`   | Right-electrode gap                                    | 2.352 K (49 GHz)       |
| `omega_10_kelvin`  | Qubit transition frequency                             | 0.264 K (5.5 GHz)      |
| `omega_nu_kelvin`  | Pair-breaking photon frequency                         | 5.711 K (119 GHz)      |
| `E_J_kelvin`       | Transmon Josephson energy                              | 0.696 K (14.5 GHz)     |
| `E_C_kelvin`       | Transmon charging energy                               | 0.01392 K (290 MHz)    |
| `T_kelvin`         | Bath temperature                                       | sweep, 0.005–0.150 K   |
| `nu_0_per_J_per_m3`| Normal-state DoS at Fermi level (both electrodes)      | 0.73 × 10⁴⁷ J⁻¹ m⁻³   |
| `volume_m3`        | Electrode volume (V_L = V_R = V)                       | 3.4 × 10⁻¹⁵ m³         |
| `g_T_siemens`      | Junction normal-state conductance (SI units)           | (derived from I_c; see §4) |
| `b_L_per_J3_s`     | Electron-phonon spectral density prefactor, left elec. | from r^L (§6)          |
| `b_R_per_J3_s`     | Electron-phonon spectral density prefactor, right el.  | from r^{R<} (§6)       |
| `Gamma_nu_scale_Hz`| Photon-drive amplitude `Γ_ν` (on `M25PhotonDrive`)      | back-solved from Γ^{ph}_{00} (§5) |
| `Gamma_ee_10_Hz`   | Parity-preserving relaxation rate (detailed balance sets `Γ^{ee}_{01}`) | 100 kHz |

---

## 4. Tunneling rates `Γ̃^α_{ij}` (SI Note III)

Core prefactor: `g_T Δ̄ / e²` (units of rate). In practice we combine with
the DoS structure. The 12 entries split by the initial sub-band α ∈
{L, R>, R<} and the (ij) logical transition:

**Two normalizations (load-bearing).** The tilde rates below are the
*ensemble* rates that enter the qubit master equation (M25 Eq. 3) via
`Γ^{eo}_{ij} = Γ^{ph}_{ij} + Σ_α Γ̃^α_{ij} x_α`. The *density*
equations (M25 Eqs. 4–6) instead use the single-quasiparticle rates

```
Γ̄^α_{ij} = Γ̃^α_{ij} / N_CP(R),    N_CP(R) = 2 ν₀ Δ_R V
```

(M25 main text below Eq. 6: "These rates correspond to the tilde
rates … divided by the Cooper pair number in the low-gap electrode").
At the Fig 3 parameter set `N_CP(R) ≈ 1.61 × 10¹⁰`. In code the
division happens inside `_rate_equation_residual` using
`M25Coefficients.cooper_pair_number_R`, which
`coefficients_from_physical_parameters_with_photon_drive` sets from
the drive's `ν₀` and `V` (the plain builder lacks those inputs and
leaves the legacy default 1.0 — do not use it for absolute-density
predictions). Running the density equations on Γ̃ instead of Γ̄ was
the root cause of the historical "flat-valley multi-stability": it
mixed ~10¹⁰ Hz tunneling currents with ~10⁻⁸ Hz generation in one
residual, degrading the Jacobian conditioning by ~10 orders and
draining `x_{R<}` ~8 orders below the paper's own small-asymmetry
approximation (S66).

### 4.1 Logical-state-conserving (ii ∈ {00, 11}) — Eq. (S30)–(S31)

```
Γ̃^L_{ii}   ≃ c²_{ii} · (g_T Δ_L / e²) · √((Δ_L − Δ_R)/(2Δ_L)) · √(2y/π) · eʸ · K_1(y)          (S30)

Γ̃^{R>}_{ii} ≃ c²_{ii} · (g_T Δ_R / e²) · (Δ_L − Δ_R)/√(2TΔ_R/π) · e⁻ʸ K_1(y) / (π · erfc(√(2y)))   (S31)

Γ̃^{R<}_{ii} = 0     (kinematic: R< QP cannot tunnel with logical-conserving transition)
```

At leading order `c_{00} = c_{11} = 1`, so `Γ̃^α_{00} ≃ Γ̃^α_{11}` in this approximation.

### 4.2 Relaxation (ij = 10) — Eqs. (S32)–(S34)

```
Γ̃^L_{10}    ≃ s²_{10} · (g_T Δ̄/e²) · √(2Δ_L/(ω_LR + ω_10)) · √(2y_+/π) · e^{y_+} · K_0(y_+)                           (S32)

Γ̃^{R>}_{10} ≃ s²_{10} · (g_T Δ̄/e²) · √(2Δ_R/|ω_10 − ω_LR|) · √(2|y_-|/π) · e^{y_-} · K_0(|y_-|, w) / erfc(√(ω_LR/T))   (S33)

Γ̃^{R<}_{10} ≃ s²_{10} · (g_T Δ̄/e²) · √(2Δ_R/|ω_LR − ω_10|) · √(2|y_-|/π) · e^{y_-} · (K_0(|y_-|) − K_0(|y_-|, w)) / erf(√(ω_LR/T))   (S34)
```

where `w = cosh⁻¹[(ω_10 + ω_LR)/|ω_10 − ω_LR|]` and `K_n(z, w) = ∫_w^∞ e^{-z cosh t} cosh(n t) dt` is the lower **incomplete** modified Bessel function of the 2nd kind (Eq. S19).

Regime notes:
* For ω_10 > ω_LR (case I): `Γ̃^{R<}_{10}` is finite.
* For ω_10 < ω_LR (case II): `Γ̃^{R<}_{10} ∝ e^{-(ω_LR - ω_10)/T}` exp-suppressed.

### 4.3 Excitation (ij = 01) — Eqs. (S35)–(S36)

```
Γ̃^L_{01}    ≃ s²_{10} · (g_T Δ̄/e²) · √(2Δ_L/|ω_LR − ω_10|) · √(2|y_-|/π) · e^{-y_-} · K_0(|y_-|)                  (S35)

Γ̃^{R>}_{01} ≃ s²_{10} · (g_T Δ̄/e²) · √(2Δ_R/(ω_10 + ω_LR)) · √(2y_+/π) · e^{-y_+} · K_0(y_+) / erfc(√(ω_LR/T))     (S36)

Γ̃^{R<}_{01} = 0       (kinematic)
```

Regime notes:
* `Γ̃^L_{01}` is finite for ω_10 < ω_LR (case II) and ∝ e^{-(ω_10 - ω_LR)/T} for ω_10 > ω_LR (case I).
* `Γ̃^{R>}_{01} ∝ e^{-ω_10/T}` always exponentially suppressed (at low T).

### 4.4 Branching fraction ξ (SI Note III D, Eq. S37)

For the `Γ̃^L_{01}` channel (L → R tunneling with qubit 0→1), ξ ∈ [0, 1] is the fraction that lands in R>. For single-junction transmon (c_10 ≃ 0), the general expression reduces to:

```
ξ ≃ K_0(z, w) / K_0(z)
```

with `z = |ω_10 − ω_LR|/(2T)`, `w = cosh⁻¹[(ω_10 + ω_LR)/|ω_10 − ω_LR|]`.

Low-T asymptotic (S38, S39): `ξ ≃ √(T|ω_10 − ω_LR|/(π ω_10 ω_LR)) · e^{-min(ω_10, ω_LR)/T}`.

---

## 5. Photon-assisted tunneling (SI Note V)

**Bulk contribution `Γ̃^{ph}_{ij}`** (enters via `Γ̃^{eo}_{ij}` in M25 Eq. M4):

```
S^±_qp(ω_{if}) → Γ_ν · (g_T Δ_L / (8 g_K)) · S^±_ph((ω_ν + ω_{if})/Δ_L; Δ_R/Δ_L)       (S55)
```

where:
* `g_K = e²/(2π)` is the Gaussian-unit conductance quantum
* `Γ_ν ∝ (coupling strength)² × n̄_ν` is proportional to the photon number at frequency ω_ν
* `S^±_ph(x; z)` is the dimensionless photon spectral density (S56, S57)

The total pair-breaking rate from photons is:

```
Γ^ph = p_0 (Γ^ph_{00} + Γ^ph_{01}) + p_1 (Γ^ph_{11} + Γ^ph_{10})                       (S58)
```

with individual `Γ^ph_{ij}` obtained by plugging (S55, S57) into (S10).

**Generation from pair-breaking photons** (explicit in thesis Appendix A.2,
Eq. 122). The absorbed photon breaks a pair *across* the junction — one
quasiparticle in each electrode — so the partner DoS and the upper limit
carry the *opposite* electrode index ᾱ (ᾱ = R for α = L; ᾱ = L for both R
sub-bands):

```
g^ph_α = Γ_PB · n̄_PB · ∫_{Δ_α}^{ω_ν - Δ_ᾱ} χ_α(E) ρ_α(E) ρ_ᾱ(ω_ν - E) K⁻(E, ω_ν - E) dE
```

where `χ_α(E)` is the indicator function for the α sub-band:
* χ_L(E) = θ(E − Δ_L)
* χ_{R<}(E) = θ(Δ_L − E) · θ(E − Δ_R)
* χ_{R>}(E) = θ(E − Δ_L)

**Transcription fix (2026-08-03):** an earlier revision of this file carried
the same index α on both DoS factors and on the upper limit
(`ρ_α(ω_ν − E)`, `∫^{ω_ν − Δ_α}`), i.e. a pair broken *within* one electrode
with threshold ω_ν > 2Δ_α. That contradicts the χ's listed directly above,
which reproduce the shipped thresholds only under the cross-electrode
reading: α = L and α = R< both open at ω_ν > Δ_L + Δ_R, and α = R> at
ω_ν > 2Δ_L.

**What the code evaluates.** `coefficients_from_physical_parameters_with_photon_drive`
does not quadrature this integral. It evaluates the main-text identity

```
g^ph_R = Γ^ph / N_CP(R) = Γ^ph / (2 ν₀ Δ_R V),      g^ph_L = δ · g^ph_R
```

per qubit state, resolving R< from R> with the *per-channel* spectral-density
fraction `S^{<,±}_ph(x_{ij}; z) / S^±_ph(x_{ij}; z)`. Its support is therefore
the support of `S^±_ph`: identically zero for ω_ν ≤ Δ_L + Δ_R on every
channel, and additionally zero on the R> branch for ω_ν ≤ 2Δ_L. On a strongly
asymmetric junction the two thresholds separate visibly — Δ_L/h = 60 GHz,
Δ_R/h = 49 GHz, ω_ν/h = 105 GHz clears 2Δ_R = 98 GHz but not
Δ_L + Δ_R = 109 GHz, so the shipped `g^ph` is exactly zero and
`calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00` raises. That is a model-scope
boundary, not a numerical bug. At the Fig 3 set the two coincide to within
0.5 GHz, so the distinction is invisible there.

`Γ_PB`, `n̄_PB` and `K⁻` are thesis symbols only; they are not qpsim inputs
and appear nowhere in the code — the drive enters through `Γ_ν`
(`Gamma_nu_scale_Hz`) and the `Γ^ph_{ij}`.

In the Fig 3 parameter set, `Γ^ph_{00} = 300 Hz` is the caption input;
this fixes `Γ_ν` (and hence `Γ_PB n̄_PB`) for the configured ω_ν, E_J, E_C.

---

## 6. Recombination `r^α, r^{<>}` (SI Note IV C)

At leading order in T/Δ_α ≪ 1 (approximating the DoS and coherence factors at the gap edge):

```
r^L  = 8 π b_L Δ_L³                                      (SI Note IV C)
r^{R>} ≃ r^{<>} ≃ r^{R<} ≃ 8 π b_R Δ̄³                    (Δ̄ = (Δ_L+Δ_R)/2; to leading order δ → 1)
```

**Transcription fix (2026-07-20):** an earlier revision of this file wrote
the R-side coefficients with `Δ_R³`; the paper's App. D.3 equation reads
`8 π b_R Δ̄³` (verified against the arXiv v2 math source). Consequently the
S48/S50 prefactor conversion carries a factor:

```
2 π b_R Δ_R³ = (r^{R<}/4) · (Δ_R/Δ̄)³      (≈ 0.985 for Fig 3a, ≈ 0.861 for Fig 3b)
```

**Consistency check** with Fig 3 caption: `r^L = r^{R<} = 6.25 MHz` ⇒ can back-solve
* `b_L = (6.25 × 10⁶ Hz) / (8 π Δ_L³_in_Hz³)`
* `b_R = (6.25 × 10⁶ Hz) / (8 π Δ_R³_in_Hz³)`

In the Fig 3 parameter set with Δ_L = 49.5 GHz and Δ_R = 49 GHz, the two
b's are nearly equal (ratio ≈ 1.03) — consistent with treating the
junction as Al/Al with similar material parameters on both sides.

---

## 7. Generation by thermal phonons `g^{pn}_α` (SI Note IV B / D)

For the **left electrode** (single sub-band), detailed balance at `µ_L = 0`
gives `g^{pn}_L = r^L · (x_L^{eq})²`:

```
g^{pn}_L   ≃ r^L · (2π T/Δ_L) · e^{-2Δ_L/T}                      ← single sub-band
```

For the **right electrode**, each pair-breaking event creates two
quasiparticles that partition independently between R< and R> with
probabilities `erf(√(ω_LR/T))` and `erfc(√(ω_LR/T))`. Counting both
QPs per event and summing both-in-<, one-in-each, and both-in->
contributions yields branching that is **linear** (not quadratic) in
erf/erfc:

```
g^{pn}_{R<} = r · x_{R<}^{eq} · x_R^{eq}   =  r · (2π T/Δ_R) · e^{-2Δ_R/T} · erf(√(ω_LR/T))
g^{pn}_{R>} = r · x_{R>}^{eq} · x_R^{eq}   =  r · (2π T/Δ_R) · e^{-2Δ_R/T} · erfc(√(ω_LR/T))
```

with `x_R^{eq} = x_{R<}^{eq} + x_{R>}^{eq} = sqrt(2πT/Δ_R) e^{-Δ_R/T}`
the un-partitioned full-R density and `r ≃ r^L ≃ r^{R<} ≃ r^{<>} ≃
r^{R>}` at leading order in `δ → 1`.

**Invariant check** (the quantity entering the Lambert-W Eq. 8):

```
g^{pn}_{R<} + g^{pn}_{R>} = r · (2π T/Δ_R) · e^{-2Δ_R/T}
```

independent of the erf/erfc branching (``erf + erfc = 1``). Squared
erf/erfc (what a naive ``g_α = r_α · (x_α^{eq})²`` per-sub-band
application would give) **violates** this invariant and was a bug in
the first draft.

---

## 8. Intraband relaxation `τ_R⁻¹, τ_E⁻¹` (SI Note IV A–B)

**Full form** (Eqs. S48, S53):

```
τ_R⁻¹ = 2π b_R Δ_R³ · √(Δ_R/(πT)) / erfc(√(ω_LR/T)) · I(Δ_R/T, ω_LR/Δ_R)       (S48, R> → R<)
τ_E⁻¹ = 2π b_R Δ_R³ · √(Δ_R/(πT)) / erf(√(ω_LR/T))  · I(Δ_R/T, ω_LR/Δ_R)       (S53, R< → R>; but integral scale is different: see below)
```

with (S49):

```
I(a, b) = ∫_b^∞ dx (e^{-ax}/√x) · ∫_0^b dy (y-x)² · (xy + x + y) / (√(y(y+2))) · 1/(1 - e^{-a(x-y)})
```

**Low-T leading order** (S50, T ≪ ω_LR ≪ Δ_R):

```
τ_R⁻¹ ≃ 2π b_R Δ_R³ · (64√2 / 105) · (ω_LR/Δ_R)^{7/2} · [1 + (7/2)(T/ω_LR) + 7(T/ω_LR)²]    (S50)
```

The physical detailed-balance relation
`τ_E⁻¹ · x_{R<}^{eq} = τ_R⁻¹ · x_{R>}^{eq}` at `µ_α = 0` yields:

```
τ_E⁻¹ / τ_R⁻¹  =  x_{R>}^{eq} / x_{R<}^{eq}  =  erfc(√(ω_LR/T)) / erf(√(ω_LR/T))
```

The `e^{-ω_LR/T}` suppression the thesis appendix quotes in the low-T
limit is **already inside** `erfc(√(ω_LR/T))` asymptotically
(`erfc(z) ≃ e^{-z²}/(z√π)` for `z ≫ 1`) — multiplying by an explicit
`e^{-ω_LR/T}` in addition would double-count the suppression and was
a bug in the first draft.

---

## 9. Implementation notes

1. **Bessel functions**: `scipy.special.k0`, `k1` for `K_n(z)`; implement
   `K_n(z, w)` via numerical quadrature on `∫_w^∞ e^{-z cosh t} cosh(nt) dt`
   (finite interval if upper cutoff is chosen at cosh t = z_cutoff/z; use
   `scipy.integrate.quad` with `limit=100`).
2. **erf/erfc**: `scipy.special.erf`, `erfc` — robust for arguments up
   to ~6 then underflow is fine because `Γ̃^α_{...}` quantities are
   multiplied by `e^{-ω_LR/T}` anyway (numerator/denominator tracking).
3. **`τ_R⁻¹` implementation (updated 2026-07-19/20):** the full exact
   `I(a, b)` quadrature (S48/S49) IS the shipped evaluation path — the
   Fig 3a/4a sweeps reach `T/ω_LR ≈ 0.4–6`, far outside S50's
   `T ≪ ω_LR` domain, where the series is low by up to ~5x. The S50
   series is retained only as the in-domain regression reference
   (`_tau_R_inverse_series_s50`). An earlier revision of this note
   recommended S50 as primary; that recommendation predates the
   domain-violation discovery and is superseded.
   Normalization (corrected 2026-07-20, second review): M25 v2 App. D.3
   defines the R-side coefficients as `r^{R<} ≃ 8π b_R Δ̄³` (average
   gap; verified against the paper's equation source), so the S48/S50
   prefactor is `2π b_R Δ_R³ = (r^{R<}/4)·(Δ_R/Δ̄)³`. A first-pass
   adjudication refuted the factor based on a truncated extract of D.3
   (which hid the `Δ̄³` tail) and this file's own earlier `Δ_R³`
   transcription; both are corrected and an absolute-normalization pin
   test guards the conversion.
4. **Detailed-balance tests** (self-consistent checks):
   * `Γ̃^{ee}_{01} / Γ̃^{ee}_{10} = e^{-ω_10/T}` (paper imposes)
   * `g^{pn}_α / (r^α (x_α^{eq})²) → 1` (at full thermal equilibrium)
5. **Splitting `Γ̃^{R<}_{10}` case I vs case II**: use the case-II
   exp-suppressed closed form when `ω_10 < ω_LR` to avoid catastrophic
   cancellation in `K_0(|y_-|) - K_0(|y_-|, w)`.
