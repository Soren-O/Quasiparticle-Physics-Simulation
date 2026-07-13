# Fischer baseline regeneration & paper-fidelity comparison — corrected BCS gap-edge quadrature

Follows the integrated audit (`docs/AUDIT-2026-07-12-integrated.md`). The corrected
observable quadrature (`qpsim.physics.bcs_quadrature.bcs_dos_cell_weights`, exact per-cell
measure `√(E_hi²−Δ²)−√(E_lo²−Δ²)` for the integrable BCS DOS edge) changes the derived
observables `x_qp`, `σ₁`, and passive `Q_i`. The self-pinned Fischer *figure* CSV baselines
were generated from qpsim's own prior (uncorrected) output, so this document establishes two
things: (1) that the corrected quadrature improves fidelity to the paper's analytic forms, not
just changes numbers; and (2) the magnitude of the baseline shift per figure.

## 1. Paper-fidelity check (decisive) — `x_qp` vs the exact integral and the paper's analytic form

The paper's Eq. 47 reduces at `n̄ = 0` to the thermal closed form
`x_qp = √(π T_B / 2Δ) · e^{−Δ/T_B}` (a *leading-order low-T* approximation). The unambiguous
"truth" is the exact `x_qp = (1/Δ)∫_Δ^{E_max} N₁(E) f_FD(E) dE` by adaptive quadrature on the
`E = Δ cosh u` substitution (no singularity). On the **actual Fischer Fig. 5 grid**
(Δ₀ = 180 µeV, NE = 1620, E ∈ [1, 10]Δ):

| T_B (K) | x_qp exact | **corrected err** | old-midpoint err | leading-order analytic err |
|---|---|---|---|---|
| 0.10 | 2.366e-10 | **−0.83 %** | −11.54 % | −1.74 % |
| 0.15 | 3.088e-07 | **−0.46 %** | −9.31 % | −2.57 % |
| 0.20 | 1.169e-05 | **−0.30 %** | −7.98 % | −3.37 % |
| 0.25 | 1.064e-04 | **−0.21 %** | −7.07 % | −4.14 % |
| 0.30 | 4.725e-04 | **−0.16 %** | −6.40 % | −4.85 % |

**Findings.**
- The corrected quadrature is accurate to **≤0.83 % against the exact integral at every T_B**,
  vs **6.4–11.5 % low** for the old midpoint rule (worst exactly where it matters — the low-T,
  low-`x_qp` QP-poisoning regime).
- Against the paper's leading-order analytic thermal form, the corrected numeric agrees to
  **<1 % at 0.1 K** (vs ~10 % old). The *analytic form itself* carries a 1.7–4.9 % error vs the
  exact integral (it is a low-T asymptote), so an earlier apparent "overshoot" of corrected vs
  analytic at higher T is the analytic form under-approximating — corrected is closer to truth
  than the analytic form at all T.
- Conclusion: **regenerating the figure baselines under the corrected quadrature moves them
  toward paper fidelity.** The prior self-pinned CSVs encoded a systematic 6–12 % low-`x_qp`
  bias.

## 2. Baseline shift per figure

Running the slow figure suite against the corrected code confirms **all four
quadrature-dependent baselines are stale** (they encode the old midpoint rule):

| Figure | Observable | Slow test | Measured shift (corrected vs old baseline) |
|---|---|---|---|
| Fischer 2023 Fig. 5 | `x_qp` (driven, two-panel) | FAILED (stale) → regenerated | `x_qp` up (same mechanism) |
| Fischer 2023 Fig. 7 | `Q_i(T_B)` (σ₁-driven) | FAILED (stale) → regenerated | `Q_i` down (σ₁ up) |
| Fischer 2024 Fig. 8 (paper) | `x_qp` | FAILED (stale) → regenerated | `x_qp` up |
| Fischer 2024 Fig. 8 (`xqp_pb`) | `x_qp(T_B)` | FAILED (stale) → regenerated | **+11.6 % … +17.6 %** (larger at low T) |

The `fig8_xqp_pb` mismatch is fully diagnostic — corrected `x_qp` is uniformly **higher**,
by +11.6 % at the warm end up to **+17.6 %** at the coldest point, exactly the direction and
low-T-weighting predicted by §1 (the old midpoint rule was most-low where the gap-edge DOS
dominates). σ₁ carries the same DOS-edge singularity, so `Q_i = σ₂/(α σ₁)` shifts *down* by the
same mechanism. The fast analytic Eq. 47 / Eq. 53 overlay tests already pass under the corrected
quadrature (paper-anchored, not self-pinned).

## 3. Regeneration status

All four stale baselines were regenerated from the corrected quadrature
(`python -m validation.<module>`, re-running the full 1620/1701-bin solves; fig5's two-panel
driven sweep was the ~3.4 h pole, the others minutes). The regenerated CSVs contain the
corrected values verbatim — e.g. `f24_fig8_xqp_pb.csv` now holds `x_qp_thermal =
5.719117e-11 … 1.911167e-05` (the corrected-quadrature output) and none of the old biased
values remain. **Per §1 these regenerated baselines are more paper-faithful, not merely
re-stabilized** — they remove the systematic 6–18 % low-`x_qp` bias.

Regenerated files (CSV + PDF): `fischer_fig5_paper`, `fischer_fig7_paper`,
`fischer2024_fig8_qpsim_native`, `f24_fig8_xqp_pb`. The three cheaper figure tests (fig7, both
fig8) are re-run to confirm `test_matches_pinned_baseline` passes; fig5 is trusted by
construction (identical mechanism, and re-running its test is another 3.4 h solve).

## 4. Fig. 6 (gap suppression) — deferred full sweep

The full Fig. 6 `(δΔ_T − δΔ)/δΔ_T` sweep is a many-hour target and is left as a manual
regeneration. The fast analytic Eq. 53 overlay (`test_fig6_paper_eq53.py`) passes under the
corrected quadrature.
