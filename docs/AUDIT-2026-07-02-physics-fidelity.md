# Audit record: qpsim physics fidelity vs merged paper @ 1d1efaa (2026-07-02)

Executes the HANDOFF at the top of the desktop file
`qp-diffusion-OPEN-ITEMS-2026-07-02.md`. This is a **delta re-audit** on
top of `docs/AUDIT-2026-06-10-physics-faithfulness.md`: that audit's
verdicts are inherited wherever neither the implementing module nor the
governing paper text moved; everything that moved on either side since
2026-06-10 was re-checked, and the merged paper's content the June audit
never saw (assembled 2026-07-01, nine-review fixes @ 1d1efaa) was audited
fresh. Spec: `papers/qp-diffusion/{paper.tex,supplement.tex}` in this
repo @ 1d1efaa.

Code delta since the June audit commit (7c0175a): `t3_spatial_1d.py`
(74f2573 cache keying), `gap_equation.py` + `ac_conductivity.py`
(0a76f9d loud domain guards), `pair_breaking_photon.py` +
`sub_gap_photon.py` (e0f80a8 partner alignment / comment fix), and the
**new benchmark 4** `validation/diffusion_operators/
self_consistent_feedback.py` (6499a8a) — never audited before.
Unchanged since June (verdicts stand, spot-confirmed against the merged
paper's labels): `physics/spectral.py`, `transport/diffusion/base.py`,
`solvers/spectral_flow_tvd.py`, `collisions/phonon.py`,
`physics/kernels.py`, `physics/kaplan_pair_breaking.py`,
`observables/density.py`.

Gate at session start: **633 pass / 0 fail** (the June session's 2 known
rate_equation failures are gone — that track landed before the merge to
main). Gate at end: **634 pass / 0 fail** (one new regression test, below).

## Per-module verdicts

1. **`physics/spectral.py` — unchanged, CLEAN (June verdict stands).**
   Re-anchored against the merged paper: N₁ = E/√(E²−Δ²)
   (eq:bcs_dos), N₂ = Δ/√(E²−Δ²), both zero sub-gap; K± = 1 ± Δ²/(EE′);
   Dynes branch kept as a separate modeling choice exactly as the paper
   prescribes ("added as a separate modeling choice rather than folded
   into the diffusion operator by notation").

2. **`transport/diffusion/base.py` — unchanged, CLEAN.** The (p,q)
   family matches tab:operator_taxonomy row for row: A1 (1,0)
   **default** (`DEFAULT_DIFFUSION_MODEL`), A1P (1,2), A2 (2,2),
   C (0,−1), B (0,−2); uniform-gap rate D_N·N₁^(q−p); `flux_weight`
   implements 𝒟_L = 1 above / 0 below the local edge at every q via the
   sub-gap zero guard. The transport closure is A1's rate D_N/N₁
   (eq:sc_tau_closure); the documented trap τ_tr = τ_N → D_N/N₁² is
   present only as the *labeled* diagnostic B, exactly as the taxonomy
   table's "legacy placements" rows demand. The paper settling on ONE
   correct operator while the code ships five is by design: B/C/A1P/A2
   are the taxonomy's labeled diagnostics, and the benchmarks exist to
   quantify their artifacts.

3. **`backends/t3_spatial_1d.py` — re-audited (changed), core CLEAN,
   one guard added.** The 74f2573 diff is cache-keying only (spectral
   value fingerprint replacing shape/identity keys — correct and
   physics-safe; the fingerprint covers E, dE, Δ, Γ). Re-verified
   against the merged paper's anchors:
   - conservative CN on u = N₁^p f with current −D₀N₁^q ∇f — the
     **occupation** gradient, never a density gradient (eq:cons_form's
     current −D_N∇f at A1);
   - zero-flux faces at the local gap edge via harmonic face weights
     vanishing against sub-gap neighbours — the explicit weak-form
     reading the paper's "local gap edge" paragraph requires;
   - KL faces carry exactly j = G_N(N₁N₁′ − N₂N₂′)(f_L − f_R)
     (eq:scalar_BC_energy): current-continuous, f-jump (Robin); the
     SIS-singular N₁N₁′ product appears nowhere in the energy channel.
     The paper writes the BC in f_L = 1−2f amplitudes; the −2 cancels
     between the interface current and the bulk flux when both are
     rewritten in occupations, so the code's f-form is exact, not a
     factor-2 approximation;
   - **no elastic double-count** (new explicit check from
     eq:sc_scalar_diffusion): the collision RHS in `apply_collisions`
     is inelastic-only (e-ph scattering + recombination + external
     flux); elastic impurity scattering lives entirely inside D₀ = D_N.
   - **FIXED (the one real finding): Dynes trap in transport.** The
     transport dressings are clean-BCS traces — 𝒟_L as the indicator of
     N₁ > 0, and the KL weight from real N₁/N₂ — which rest on the
     above-gap identity N₁²−N₂² = 1. The paper's Dynes footnote (below
     eq:bcs_dos, new in the merged manuscript) states that with Γ ≠ 0
     this real-coefficient identity fails and the traces must be
     re-evaluated from the complex spectral functions. The backend
     accepted a Dynes-broadened `SpectralContext` silently: the
     uniform-gap path would broadcast the *broadened* N₁ under the
     *clean* 𝒟_L (full-strength sub-gap transport wherever the
     broadened DOS is nonzero), while the `gap_profile` path would
     silently switch to pure-BCS spectral functions (inconsistent
     provenance between the two paths). No in-repo caller hits this
     (the only Γ > 0 usage is the non-spatial coupled-Newton test), so
     it was latent, not active. `apply_transport` now raises with the
     physics reason; collisions with a Dynes DOS remain allowed (a
     legitimate modeling choice independent of the transport traces).
     Regression test:
     `test_dynes_context_rejected_by_transport_not_collisions`.

4. **`physics/gap_equation.py` — re-audited (changed), CLEAN.** The
   0a76f9d diff only adds a warning on the bracket-widening fallback
   (behavior unchanged). Anchor check against eq:gap_feedback_closure:
   the code solves the exact reference-subtracted BCS gap equation
   ∫_Δ^{ω_D}(1−2f)/√(E²−Δ²)dE = 1/λ, of which the paper's exponentiated
   closure Δ = Δ₀·exp[−2∫(dE/E)N₁f] is the ω_c → ∞ limit. Numeric
   cross-check (thermal + near-edge bump population): solver vs closure
   fixed point agree to 3.9e-4 relative — the expected O((Δ/ω_D)²)
   ≈ 3e-4 cutoff correction at ω_D = 100 k_B T_c. The code is the more
   exact form; no discrepancy. n_qp = 4N₀∫N₁f dE (single-spin N₀) and
   x_qp = n_qp/(2N₀Δ₀) unchanged in `observables/density.py` (June
   verdict stands).

5. **Collisions delta — CLEAN.** `pair_breaking_photon.py` (e0f80a8)
   adds a partner-alignment warning; the lattice condition
   (ω_snapped − 2E₀)/dE ∈ ℤ is exactly the requirement for every K⁻
   reflection partner ω − E_i to land on-grid on a uniform grid
   (verified algebraically), and the residual is i-independent as the
   comment claims. `sub_gap_photon.py`: the corrected comment now
   matches the code, and the code matches physics — gain at i from the
   i+m partner is emission (n̄+1), loss at i is absorption (n̄), with
   the mirrored factors on the i−m partner; consistent with the
   detailed-balance fixtures. Kaplan kernels themselves unchanged since
   the June normalization fix (K± per eq:J1_occ_bridge; verdicts stand).

6. **`observables/ac_conductivity.py` — new guards verified.** Outside
   the paper's scope (MB is an observable, not part of the kinetic
   spec), but the 0a76f9d guards are faithful to their own docstring:
   the implementation keeps only sub-gap MB terms, so raising for
   ω₀ ≥ Δ (and for Dynes contexts) converts a silently-wrong domain
   into a loud failure. No change needed.

7. **`solvers/spectral_flow_tvd.py` — unchanged, June verdict stands.**
   Conservative form ∂_t(N₁f) + ∂_E[(Δ/E)Δ̇N₁f] with the gap edge as a
   zero-flux boundary matches eq:cons_form / eq:cons_dos_continuity in
   the merged paper (same equations, new labels); the June fixtures
   (frozen-shell exactness, discrete DOS continuity) still gate it.

8. **`validation/diffusion_operators/self_consistent_feedback.py`
   (benchmark 4, NEW) — audited fresh, CLEAN.** Realizes the paper's
   fourth benchmark (§V "Benchmark problems" item 4, fig:bench_feedback):
   - the well is dug with the *direct* closure
     Δ(x) = Δ₀·exp(−∫2ρf/E dE) — precisely eq:gap_feedback_closure,
     Picard-iterated so ρ is evaluated at the self-consistent local gap
     (fixed point verified to 1e-10·Δ₀ in the co-located tests);
   - drift readout against v = D_N·q·N₁^(q−p−1)·∂ₓN₁ matches
     eq:dos_gradient_response's drift form; sign logic verified: the
     well lowers Δ hence N₁, so C/B (q<0) fall *into* the well
     (self-focusing), A1P/A2 (q=2) are expelled, A1 does not move
     (its measured COM shift is < 1e-8, "zero to round-off");
   - the marginal-exponent statement (drift finite exactly at
     q−p = −2, placement B) checks out analytically: ∂ₓN₁ ~ N₁³ near
     the edge, so v ~ N₁^(q−p+2);
   - dynamic mode is the paper's 20 ns run (80 × 0.25 ns) and the
     "bookkeeping drift measured rather than hidden" claim is
     implemented literally (`conservation_drift`; zero for p = 0,
     finite and reported for p ≥ 1 — the neglected energy-space
     spectral-flow advection under per-step gap updates);
   - the caption's probe energy E = 1.08Δ₀ is the realized grid bin
     (1.0821Δ₀) for the requested 1.1Δ₀ — caption faithful to the
     script.
   `uniform_gap_packet.py`'s post-June diff is style-only; the
   2026-07-02 §V caption verification therefore stands for all four
   figures.

## Fix record

- `qpsim/backends/t3_spatial_1d.py`: `_build_transport_operators` now
  rejects Dynes-broadened spectral contexts (ValueError citing the
  paper's Dynes footnote). Test added in
  `tests/backends/test_t3_spatial_1d.py`. This implements the handoff's
  explicit check "the code never relies on [N₁²−N₂² = 1] there".

## Open items (judgment calls, recorded not decided)

- **Dynes-consistent transport is unimplemented, not just guarded.** If
  spatial transport with lifetime broadening is ever wanted (the paper's
  gap-edge paragraph explicitly allows Dynes regularization as an
  *alternative* to the explicit zero-flux face), the dressings need the
  numerically evaluated complex traces: 𝒟_L = ¼Tr[1−g^Rg^A] with
  N₁(E+iΓ), N₂(E+iΓ), and the KL weight likewise. That is new physics
  code with its own fixtures — a commissioning decision, not a bug fix.
- **`SpectralContext.D_E` keeps the clean-BCS legacy-C closure even at
  Γ > 0** (docstring-labeled LEGACY; only used by the modal reference
  path, not by the operator-family transport). Harmless today; worth a
  guard only if the legacy path ever meets a Dynes context in earnest.

## Standing caveats carried forward (unchanged)

- f_T / charge module: not commissioned (scope fence intact —
  `𝒟_T = N₁²` sub-gap Andreev transport applies only once it exists).
- Strong-drive solver fragility (fig6 nan/collapse tail): separate track.
- Phonon-side kernel default (June lead 4): deliberately not flipped.
- Spectral-flow finite-domain caveats (top boundary, gap-anchored
  grids): documented in `advect_spectral_flow`, unchanged.
- The open items in `docs/REVIEW-2026-07-02-code-health.md` are
  code-health, not physics-fidelity; none intersect the anchors above.
