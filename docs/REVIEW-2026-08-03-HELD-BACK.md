# Held back from the 2026-08-03 review fixes — needs a physics decision

The 2026-08-03 review applied 92 confirmed findings as behaviour-neutral repairs
(proven bit-identical across 35 engine surfaces, including four full Newton solves).

The **34** items below were deliberately **NOT applied**. Each would change a number
this project has published or pinned. They are held back for two reasons:

1. A physics change in this repo requires a physicist's decision, not a code review's.
2. Mixing them into the same commit would make the recertification uninterpretable —
   intended drift could not be distinguished from a regression.

Each entry records the exact patch that was prepared, so applying one is a deliberate,
reviewable act rather than a rediscovery.

---

## `qpsim/backends/t3_spatial_1d.py`

### P05 — Kupriyanov-Lukichev face omits the series half-cell bulk resistance (medium, numerics)

Location: `qpsim/backends/t3_spatial_1d.py:600`

**Why held back.** BEHAVIOUR-CHANGING. It changes the assembled transport operator for every stepped-gap run that supplies a finite interface_conductance, changes the CN substep count, and directly contradicts tests/backends/test_t3_spatial_1d.py:978 (`test_backend_face_carries_exact_weight`, which pins the current no-series behaviour and is NOT in my packet) plus the flux-continuity certificate in validation/diffusion_operators/interface_trap.py. It is a physics discretization change and needs physicist sign-off before the recertification. I did apply the neutral half: the `_flux_laplacian_from_conductances` docstring now states plainly that the override is the bare interface conductance, is first-order in dx, and does not cap at the bulk face.

**Expected numerical impact.** Uniform-gap and no-interface runs: bit-exact no-op (the branch never fires). Stepped-gap runs with a finite g_N, at Al D_0=60 um^2/ns, L=100 um, 180->200 ueV step, NX=31: the interface face conductance drops by the factor g_series/g_code = 1/1.0056, 1/1.0556, 1/4.335 for g_N = 0.1, 1, 60 um/ns. End-to-end center-to-center transport resistance error vs the continuum reference goes from -0.48% / -2.08% / -3.30% to machine precision (verifier measured 1e-13 relative after substituting only the series value at that one face), and the scheme recovers second order at the face (errors were shrinking only as O(dx): -0.048% / -0.208% / -0.330% at NX=301). Secondary: CN substeps fall (23 -> 11 bulk-limited at g_N=60), so the operator gets strictly less stiff — the change cannot destabilise it. In the tunnel regime the benchmark actually uses (g_N=0.1) the shift is <0.5%. No certified or published number in this tree depends on the interface path.

**Prepared patch.**

```
In `_build_transport_operators`, replace the unconditional overwrite

                        weight = interface_weights[int(idx[m])][i]
                        g_face[m] = g_interface * weight / dx

with the series composition

                        weight = interface_weights[int(idx[m])][i]
                        # Series with the two adjoining half-cells of bulk:
                        # 1/g = 1/g_int + 1/g_bulk. Exact for piecewise-
                        # constant W with a singular interface resistance;
                        # reduces to the pure-KL value in the tunnel limit and
                        # caps at the bulk face as g_N -> infinity.
                        g_int = g_interface * weight / dx
                        g_bulk = g_face[m]
                        g_face[m] = (
                            0.0
                            if (g_int <= 0.0 or g_bulk <= 0.0)
                            else 1.0 / (1.0 / g_int + 1.0 / g_bulk)
                        )

The zero guard is a no-op today: the KL weight and the support fraction vanish on the same sub-gap energy cells, so g_int and g_bulk close together.

Companion edits required, none of which are in my packet: re-pin tests/backends/test_t3_spatial_1d.py:978 to the series value; update docs/Diffusion_Operators.md L112-118 ("dx-independent ... instead of a bulk diffusive flux"); strengthen validation/diffusion_operators/interface_trap.py:135-152 from a flux-continuity check (a tautology of the conservative FV assembly) to a mesh-refinement order check, which is the only kind of gate that can detect this class of error. Also revert the two limitation sentences I added to the `_flux_laplacian_from_conductances` docstring (t3_spatial_1d.py ~L1348-1353), which describe the pre-fix behaviour.
```

### P05 — q==0 gap-stepped face uses harmonic-of-averages instead of the exact min(s_L, s_R) (low, numerics)

Location: `qpsim/backends/t3_spatial_1d.py:594`

**Why held back.** BEHAVIOUR-CHANGING. It changes the flux across every face whose two cells have different gaps — stepped profiles without an interface conductance, and every face of a smooth/ramped profile (validation/diffusion_operators/gap_gradient_drift.py, self_consistent_feedback.py). It is a physics discretization change requiring sign-off. I applied the neutral half: an accurate comment at the face-weight line stating that the harmonic mean of the cell averages is exact only where the two neighbours share a gap, that the exact energy-cell average is min(s_L, s_R), and that the bias is a first-order truncation error, not wrong physics.

**Expected numerical impact.** Bit-exact no-op wherever s_L == s_R, i.e. all uniform-gap runs and every face away from a gap change (harm(a,a) == min(a,a) == a). KL-interface faces are unaffected (overridden downstream). At the webui step defaults (Al 180 ueV, EnergyGrid min_factor 1.0 / max_factor 4.0 / 64 bins, step 180->200 ueV) exactly one energy bin per stepped face changes: bin 2, face weight g_face/D0 0.772727 -> 0.629630 (-18.5%; the +22.7% overestimate quoted in the finding). Whole-channel conductance bias at NX=31 is +0.75%, and a real 2000 ns webui run shifts x_qp by 2.2e-5 relative and the across-step x_qp ratio by 3.8e-5 relative; the f-drop across the step face itself changes by the full factor (3.997e-7 -> 4.915e-7). Worst reachable alignment inside the builder validator is gap_right just under a cell top edge (e.g. 205.0, s_R = 0.037, ratio 1.93). The bias grows on coarse meshes (+3.2% at NX=8) and converges away under energy refinement.

**Prepared patch.**

```
In the per-energy loop, replace

            if q == 0:
                # Dirty-limit D_L is an above-gap indicator.  Its exact
                # finite-volume average is the supported fraction of a cut
                # cell, not one merely because the cell has some capacity.
                w_cell = D0 * support_fraction[i, idx]
            g_face = _harmonic_face_weights(w_cell) * inv_dx2

with

            if q == 0:
                # Dirty-limit D_L is an above-gap indicator.  Its exact
                # finite-volume average is the supported fraction of a cut
                # cell, not one merely because the cell has some capacity.
                w_cell = D0 * support_fraction[i, idx]
                # Sub-energies inside one bin are parallel channels; the two
                # half-cells at a face are in series.  Series-per-sub-energy
                # then parallel-sum gives min(s_L, s_R) -- the fraction above
                # the LARGER of the two local gaps -- not the harmonic mean of
                # the two cell averages.  Same convention the KL branch
                # already uses via `_kl_interface_cell_average`.
                g_face = np.minimum(w_cell[:-1], w_cell[1:]) * inv_dx2
            else:
                g_face = _harmonic_face_weights(w_cell) * inv_dx2

`np.minimum` preserves the zero-closure semantics (min is 0 whenever either side is 0). Companion doc edit, not in my packet: docs/Diffusion_Operators.md near L97-99 should note that for q != 0 the face weight remains harm(<N1_L>^q, <N1_R>^q) and carries the same, uncorrected, Jensen gap on gap-cut cells. Also revert the six-line limitation comment I added above `g_face` (t3_spatial_1d.py ~L587-593).
```

---

## `qpsim/collisions/phonon.py`

### P01 — Kaplan S_+ quadrature correction is applied to the phonon equation only, so the QP and phonon equations rate the same pair events differently

Location: `qpsim/collisions/phonon.py:449`

**Why held back.** Explicitly designated behaviour-changing by the task brief, and independently so: this is the live default path (use_phonon_side_kernel=True in every Fischer driver and the webui default), it moves published Fig-3/5/6/7-class numbers, and CURRENT-STATUS.md's C6 freeze already records the correction as a material frozen-state physics difference (929 bins rescaled down to 0.7857612777, pair net moved 9.203114358766813e-3, 'recorded, not gated'). The verifier also showed the proposed two-sided fix is not free: it trades the QP side's Kaplan Eq. 8 / tau_0 normalization (pinned by tests/collisions/test_phonon.py:883-918) for the Kaplan S_+/tau_0^PB normalization, and the alternative of gating the phonon-side correction off breaks the tau_PB(2*Delta)=tau_0^PB calibration pinned at tests/collisions/test_phonon.py:244-254 by 4/pi. Which side is authoritative is a physicist decision, to be adjudicated jointly with the recorded C6 item. Note also that neither Jacobian is buggy: phonon_source_sink_jacobian_f:671-675 and phonon_collision_jacobian_nph each correctly differentiate their own residual, so nothing should be 'fixed' in isolation there.

**Expected numerical impact.** Restores the exact discrete QP<->phonon pair-event/energy ledger (verifier-measured residual 4.7e-15 with the correction bypassed vs 3.5e-2 with it on). Per-omega the QP-side recombination/pair-breaking rates move by the correction factor: 0.834 in the threshold bin at NE=1601 on a linspace(0.98D, 8D) grid, 1.315 at NE=801, pi/4 ~ 0.786 on the energy_min_factor=1.0 production family; ~2.9% / ~0.8% on the integrated QP pair-breaking gain at NE=400 for tau_l = 0.1 / 1.0 ns, and ~1.4-2.7% of the total pair-event rate at thermal f. Every dynamic-Ph0 Fischer figure and the recorded C6/C7 attribution shift; needs a physicist sign-off and a recertification pass.

**Prepared patch.**

```
Two-sided variant, expressed inside this file (caller wiring is outside the packet):

@@ def phonon_collision_rates(..., K_r_abs_override: np.ndarray | None = None,
+    pair_quadrature_correction: np.ndarray | None = None,   # (NE, NE) = correction[omega_idx_sum]
 )
@@     if enable_recombination and K_r0 is not None:
             K_r_emit = K_r0 * N_emit
             K_r_abs = K_r0 * N_abs
+        if pair_quadrature_correction is not None:
+            # Same per-omega Kaplan S_+ endpoint rescale the phonon equation
+            # applies at compute_phonon_source_sink:449-450, so one pair event
+            # carries one rate in both coupled kinetic equations.
+            K_r_emit = K_r_emit * pair_quadrature_correction
+            K_r_abs = K_r_abs * pair_quadrature_correction

plus, at each QP-side call site that already owns the phonon-side kernel (qpsim/backends/t3_diffusion.py:1241 and :1646, qpsim/solvers/coupled_newton.py, qpsim/solvers/newton_steady_state.py):

+    pair_corr = _pair_breaking_quadrature_correction(ctx, K_r0_phonon_side, omega_idx_sum, n_omega)[omega_idx_sum]
     gain, loss_rate = phonon_collision_rates(..., pair_quadrature_correction=pair_corr)

It must be applied at the call site, never inside build_recombination_kernel_base: K_r0 is shared with the thermal-bath path, where no phonon-side kernel exists. The correction is also deliberately suppressed on truncated-pair-interval bins (phonon.py:589-596 pre-edit numbering), so the folded matrix inherits that suppression and must be checked against the QP-side row integral.
```

### P01 — Commensurability guard (and the rate-preserving omega remap) for the split phonon omega lattice

Location: `qpsim/collisions/phonon.py:306`

**Why held back.** The guard rejects inputs the code accepts and runs today, so it is an availability/behaviour change, not a neutral guard: the shipped web-UI default EnergyGrid(1.0, 10.0, 400) gives 2*E_face/dE = 88.889, the Spatial1DSetup default (1.0, 4.0, 64) gives 42.667, and in-repo test grids such as (0.75, 6.0, 18) -> 5.143 would all start raising. build_phonon_frequency_map is also reached from paths where n_ph is not dynamic (services/steady_state.py, coupled_newton.py, webui/builders.py), so a raise there would break currently-working configurations, and the guard cannot be placed only at the dynamic entry points without editing files outside this packet (t3_diffusion.py, webui/builders.py). The proper fix - conservative redistribution of each pair event onto one common omega lattice - changes every dynamic-phonon number and reverses committed decision D3 in docs/Phonon_Model_Decisions.md ('no off-grid interpolation', exact discrete detailed balance), i.e. a design amendment plus consistent updates to both analytic Jacobians and to _pair_breaking_quadrature_correction. I documented the property in the docstring instead (see applied).

**Expected numerical impact.** Zero on every commensurate grid, i.e. on every published artifact (1620 over [D,10D] -> 360, 810 -> 180, prelim 28 over [D,5D] -> 14). On incommensurate grids the guard converts a silent wrong answer into a hard failure. The real fix (common-lattice redistribution) changes the dynamic-phonon answer on incommensurate grids by up to a factor ~180 (QP pair-breaking gain 8.7e-6 -> ~1.2e-3 /ns on the injected Al state at NE=101) and by O(1-3%) even on commensurate grids, where 2.7% of the above-2*Delta scattering source still lands in bins with no pair sink at NE=200.

**Prepared patch.**

```
@@ qpsim/collisions/phonon.py, build_phonon_frequency_map, after the strictly-increasing check
     if E.size > 1 and np.any(np.diff(E) <= 0.0):
         raise ValueError("E_bins must be strictly increasing.")
+    if E.size > 1:
+        # The Eq. 12 scattering integral lives on the difference lattice k*dE
+        # and the pair integral on the sum lattice 2*E_face + m*dE; they share
+        # a bin only when 2*E_face/dE is an integer.  Mirrors the photon
+        # channels' _COMMENSURATE_TOL policy (pair_breaking_photon.py:99-123).
+        h = float((E[-1] - E[0]) / (E.size - 1))
+        ratio = 2.0 * float(E[0] - 0.5 * h) / h
+        if abs(ratio - round(ratio)) > _COMMENSURATE_TOL:
+            raise ValueError(
+                f"Phonon omega lattice is incommensurate: 2*E_min/dE = {ratio:.6f} "
+                "is not an integer, so the F&C Eq. 12 scattering and pair "
+                "integrals occupy disjoint omega sublattices and phonons emitted "
+                "by scattering can never break pairs. Choose num_energy_bins so "
+                "2*E_min/dE is an integer (e.g. 405, not 400, on [Delta, 10*Delta])."
+            )

If adopted, it belongs at the dynamic-phonon entry points plus webui validate_setup rather than unconditionally here, and the shipped defaults must move (405 not 400 on [D,10D]; 63 or 66 not 64 on [D,4D]), as must the (0.75, 6.0, 18) test grids.
```

### P01 — Run diagnostic for the QP/phonon pair-and-energy ledger residual

Location: `qpsim/collisions/phonon.py:449`

**Why held back.** The verifier's recommended low-risk half of the low-severity ledger finding (option b: assert/record that the residual equals exactly sum_omega D_ph*omega*(1-corr)*(rec*(1+n) - pb*n)) has to be plumbed through the solver residual/certificate layer (qpsim/solvers/newton_steady_state.py, qpsim/solvers/coupled_newton.py, validation/fischer_2023/steady_state_certificate.py) to be useful. All of those files are outside this packet, and a bare helper in phonon.py with no caller would be dead code. It is also the same physics item as the held-back one-sidedness finding and should land with that adjudication.

**Expected numerical impact.** None on any solve output (diagnostic only). Reported value would be ~1.2e-5 relative at NE=400 and ~7.6e-6 at NE=1000 on the verifier's driven state, i.e. the recorded C6 9.2e-3 pair-net move seen from the conservation side.

**Prepared patch.**

```
Add, alongside compute_phonon_source_sink, a diagnostic that returns the pair-ledger imbalance implied by the one-sided correction:

+def pair_quadrature_ledger_residual(ctx, K_r0_phonon_side, omega_idx_sum, n_omega, rec, pb, n_ph, *, tau_0, T_c, tau_0_pb_ns):
+    """Sum_omega D_ph(omega)*omega*(1 - corr)*(rec*(1 + n_ph) - pb*n_ph).
+
+    Exactly the QP<->phonon pair/energy ledger broken by the one-sided Kaplan
+    rescale at compute_phonon_source_sink:449-450; zero when the correction is
+    off or applied to both equations.
+    """

and record it in the steady-state run diagnostics so no solve silently carries it.
```

---

## `qpsim/materials/data/TiN.yaml`

### P14 — Replace D_0 = 10.0 with a resistivity-derived value (~0.08 μm²/ns) (finding 4)

Location: `qpsim/materials/data/TiN.yaml:18`

**Why held back.** BEHAVIOUR-CHANGING physics parameter requiring a physicist sign-off: a 122x change to a material constant that sets the spatial diffusion length sqrt(D*tau) for every T3 spatial run using TiN. Nothing in the library, scripts, tests or any certified artifact reads material.D_0 today (exposure is a Web-UI user who selects TiN and leaves D_0 at the served default), so no pinned number moves — but this is a physics-model decision, not a defect repair, and the verifier's own replacement band (0.04-0.27 μm²/ns for rho = 30-200 μΩ·cm) spans a factor of 7, i.e. the right value is a judgement about which film class TiN.yaml is meant to represent. I applied the neutral half instead (see `applied`): the false inline comment is corrected and the implied rho/mfp/xi recorded, so the number cannot be mistaken for a sourced one. A peer packet made the same call on Al.yaml (comment-only note, value untouched). Out-of-scope but worth a separate look, per verifier_corrections: Nb.yaml:15 stores v_F = 1.37e6 m/s, which yields a clean-limit xi_0 = 191 nm against the accepted Nb value of 38 nm; and the filer's proposed loader guard `3*D_0/v_F <= film_thickness` is the diffuse Fuchs-Sondheimer limit, not a theorem, so it must not be an error.

**Expected numerical impact.** No shipped artifact moves (nothing certified reads material.D_0). For a Web-UI TiN spatial run the diffusion length sqrt(D*tau) at tau = 1 us shortens from ~100 um to ~9 um — an ~11x change in every position-resolved observable (trap-loading profile, spatially resolved x_qp, AC response vs position). All runs stay stable: a smaller D only loosens the CFL/diffusion-number check at qpsim/webui/builders.py:283.

**Prepared patch.**

```
--- a/qpsim/materials/data/TiN.yaml
+++ b/qpsim/materials/data/TiN.yaml
-D_0: 10.0            # μm²/ns (normal-state; see the provenance note above)
+D_0: 0.082           # μm²/ns (rho = 100 μΩ·cm via sigma = e^2*(2*rho_F)*D;
+                     # gives mfp = 0.25 nm, xi = sqrt(hbar*D/Delta) = 8.8 nm,
+                     # consistent with the Gao APL 101, 142602 (2012) film
+                     # class this file's rho_F is taken from)

Do NOT copy the filing's Nb/Al replacement values (0.1-0.5 and 6 μm²/ns): per verifier_corrections they are wrong by 3-10x in the other direction; the Einstein-relation values are ~2-5 μm²/ns for Nb at rho = 1-3 μΩ·cm and ~10-20 μm²/ns for Al. Any Al.yaml change also moves the Web-UI default (qpsim/webui/schemas.py:72) and should be paired with provenance in docs/Material_Database.md:31, which currently documents the unit and no source for any of the three materials.
```

---

## `qpsim/observables/ac_conductivity.py`

### P07 — sigma_1 super-gap quadrature: cell-center sampling of the regular factor against an exact gap-edge measure

Location: `qpsim/observables/ac_conductivity.py:148`

**Why held back.** Explicitly named in my instructions as behaviour-changing, and independently so: it changes the quadrature rule of a published observable. Q_i values in docs/prelim_experiment_simulation_notes.md (NE=28 sweep) and the Figs. 9-13 chain move by tens of percent. Physics sign-off required; also note the verifier's scope correction — qpsim/observables/density.py:73 uses the identical cell-constant pairing (n_qp low by -4.9%/-17.4%/-39.2% at NE=40), so this is a repo-wide finite-volume convention and should be decided for the convention, not patched in one observable. I documented the true behaviour in the docstring instead (applied item 4).

**Expected numerical impact.** sigma_1 rises; the centroid/cell-average half alone recovers roughly -6% (0.5 K) to -11% (0.1 K) of the deficit at NE=40. The full fix removes the whole one-signed deficit I measured: -12.4%/-26.7% at NE=40 on [D,5D] (T=0.5/0.2 K), -11.1% at the WebUI default NE=64 on [D,4D] at 0.2 K, -63% at NE=28 for a 10 ueV gap-edge-peaked f. Q_i = sigma_2/(alpha sigma_1) falls correspondingly (the published prelim Q_i range 4.31e4-2.05e5 is overstated by somewhere between ~1.1x and ~2.7x depending on the real 7 mK f). sigma_2 and delta_f/f unchanged.

**Prepared patch.**

```
Minimal, information-free half (removes only the analytic-factor sampling error; the verifier measures this as -6.0%/-8.6%/-11.1% of the total at NE=40 for T=0.5/0.2/0.1 K):

  from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights, cell_edges_from_widths
  ...
  # replace lines 143-152 (E_partner/rho_partner/K_plus_partner/U_plus/dos_weights)
  dos_weights = bcs_dos_cell_weights(E, dE, gap)
  edges = cell_edges_from_widths(E, dE)
  lo = np.maximum(edges[:-1], gap)
  hi = np.maximum(edges[1:], gap)
  xi_lo = np.sqrt(np.maximum(lo * lo - gap * gap, 0.0))
  xi_hi = np.sqrt(np.maximum(hi * hi - gap * gap, 0.0))
  d_xi = xi_hi - xi_lo
  safe = d_xi > 0.0
  denom = np.where(safe, d_xi, 1.0)
  # <E> under the exact measure: int E dxi = 0.5*[xi*sqrt(xi^2+D^2) + D^2 asinh(xi/D)]
  moment = 0.5 * (
      xi_hi * hi + gap ** 2 * np.arcsinh(xi_hi / gap)
      - xi_lo * lo - gap ** 2 * np.arcsinh(xi_lo / gap)
  )
  E_eff = np.where(safe, moment / denom, E)          # rho-measure centroid, not the geometric center
  E_partner = E_eff + omega_0
  f_partner = np.interp(E_partner, E, f, right=0.0)
  rho_partner = bcs_density_of_states(E_partner, gap)
  # cell-averaged coherence ratio, as spectral.py:435-444 already does for the kinetic kernel
  coherence = np.where(
      safe,
      gap * (np.arccosh(np.maximum(hi / gap, 1.0)) - np.arccosh(np.maximum(lo / gap, 1.0))) / denom,
      gap / E,
  )
  K_plus_partner = 1.0 + coherence * gap / np.maximum(E_partner, 1e-30)
  U_plus = rho_partner * K_plus_partner

The remaining (larger, at low T dominant) half is the sampling of f itself and cannot be fixed without extra information: it needs a sub-cell Gauss-Legendre rule in xi with f reconstructed at the sub-cell nodes by gap-edge-aware extrapolation (linear extrapolation from (E[0], E[1]) down to Delta, clipped to [0,1] — a naive np.interp makes it WORSE because it holds f constant below E[0]). Both should land together with a resolution gate on dE/kT and a convergence table in Q_i (not delta_f1, which depends only on sigma_2 and is blind to this).
```

### P07 — sigma_2 partner reconstruction interpolates f through frozen zero-capacity sub-gap cells

Location: `qpsim/observables/ac_conductivity.py:187`

**Why held back.** Behaviour-changing for a currently-valid input class: every grid with energy_min_factor < 1 (which the WebUI self-consistent-gap route effectively forces — min_factor >= 1 with a suppressed gap raises from bcs_quadrature.py:126). Verifier measures sigma_2 shifts up to +1.8e-2 relative and frac_freq_shift bias +0.9% to +3.9%, essentially drive-independent. It is a choice of physical reconstruction convention at the gap edge (point-interpolation vs active-mask vs cell-constant over the first active cell), i.e. exactly the class needing physicist sign-off. Documented in place instead (applied item 6).

**Expected numerical impact.** No change for any grid with energy_min_factor >= 1 (all cells active, so the masked interpolation is identical). On sub-gap-extended grids sigma_2 moves by up to ~2e-2 relative and the reported frac_freq_shift by ~1-4%; sigma_1 and Q_i unchanged.

**Prepared patch.**

```
In compute_ac_conductivity, sigma_2 branch only (sigma_1's line is a no-op for this and must NOT be changed — it is bit-identical under every placeholder variant):

-        f_sub_partner = np.interp(E_sub_partner, E, f, right=0.0)
+        # Zero-capacity sub-gap cells hold caller placeholders that no solver
+        # updates; reconstruct f only from cells that carry spectral weight,
+        # the same convention as gap_suppression.edge_samples_from_centers.
+        active = ctx.active_mask
+        f_sub_partner = np.interp(
+            E_sub_partner, E[active], f[active], right=0.0
+        )

and state the convention in the module docstring. np.interp left-clamps to f[first active], which the verifier measured at +8.9e-05 against an exact reference where the shipped code gives +4.3e-03 (~48x better). Add a test fixture with energy_min_factor < 1 — every current fixture in tests/observables/test_ac_conductivity.py uses 1.0, so this path is untested.
```

---

## `qpsim/observables/gap_suppression.py`

### P16 — Take the compute_gap_suppression reference through solve_gap on the caller's own grid

Location: `qpsim/observables/gap_suppression.py:97`

**Why held back.** BEHAVIOUR-CHANGING: it changes every delta_suppression_ueV / rel_gap_suppression the WebUI publishes (qpsim/webui/execute.py:217). Two further caveats: (i) execute.py:215 separately publishes summary['delta_eq_ueV'] = calibration.delta_eq, so swapping the reference silently breaks the identity delta_suppression = delta_eq_ueV - delta_final unless execute.py (not my file) also publishes the grid-consistent reference; (ii) per the verifier it does NOT make the discretization cancel - it only removes the zero-drive offset and roughly halves the weak-drive error (-4.24e-5 -> -1.89e-5 ueV at 400 bins; +2.21e-3 -> +9.72e-4 ueV at 64 bins). The genuinely accurate route already exists in the same module (the expm1-based _direct family used by Fig. 6); routing the WebUI diagnostic there is the better call and is likewise a published-number change. Do NOT plumb xtol as the fix - tightening both brentq tolerances to 1e-12 moves the answer by <5%.

**Expected numerical impact.** Zero-drive output becomes exactly 0.0 instead of the measured bias (as a fraction of the thermal suppression Delta_0,BCS - Delta_eq at 400 bins / min_factor 0.9: +0.735%, +0.460%, +0.111%, -0.516%, +0.269% at T_bath = 0.15/0.2/0.3/0.4/0.5 K; +50.7% ... +5.4% at 64 bins). Weak-drive accuracy improves from ~1.3% to ~0.58% of signal. No certified/pinned artifact is affected - the Fig. 6 path does not use this wrapper - but every WebUI-published suppression number moves.

**Prepared patch.**

```
calibration = calibrate_gap(T_c=T_c, T_bath=T_bath)
+    # Reference and driven gap must come through the SAME finite-volume rule,
+    # otherwise the cell-constant error at the gap-edge cell does not cancel
+    # and an exactly thermal f returns a sign-indeterminate suppression.
+    delta_reference = solve_gap(
+        calibration,
+        fermi_dirac_occupation(E_arr, T_bath),
+        E_arr,
+    )
     delta_final = solve_gap(
         calibration,
         f_arr,
         E_arr,
     )
-    return gap_suppression_from_deltas(calibration.delta_eq, delta_final)
+    return gap_suppression_from_deltas(delta_reference, delta_final)

# Requires a matching change in qpsim/webui/execute.py:215 so that the
# published delta_eq_ueV is the same grid-consistent reference.
```

### P16 — Clip the reconstructed edge samples to [0, 1] instead of rejecting (or bound-preserving limited reconstruction)

Location: `qpsim/observables/gap_suppression.py:132`

**Why held back.** BEHAVIOUR-CHANGING for the overshoot case: it returns a different number where the shipped code returns one (and where my applied guard now raises). The verifier's exact-reference check makes the clip the better numerics (one-cell step: exact I = 0.163254, current 0.208333 = +27.6%, clipped 0.153909 = -5.7%, scheme's intrinsic error -1.1%) and it is bit-identical whenever the extrapolate is already in [0,1] - but the strictly correct remedy is a bound-preserving limited reconstruction that redistributes the clipped excess inside the cell rather than discarding the first cell's occupation mean, and choosing between them is a discretization decision for the physicist, on the repo's headline observable, on the eve of recertification. I applied the fail-loud guard instead, which is the conservative half of the fix and leaves this choice open. Note the filer's 'add a [0,1] input check to edge_samples_from_centers' is aimed at the wrong end - it is that function's OUTPUT (out[0]) that violates the bound.

**Expected numerical impact.** Bit-identical (verified to 2e-14 by the verifier, and by my own c269af2 comparison on a representative smooth driven f) for every input whose gap-edge extrapolate lies in [0,1] - which is every shipped Fischer/WebUI distribution, since the overshoot needs f > 2/3 at the gap edge. Changes only sharply structured, strongly occupied inputs, where it removes a one-sided +27.6%-class over-report of I[f] (i.e. of gap suppression).

**Prepared patch.**

```
@@ edge_samples_from_centers
-    return np.maximum(out, 0.0)
+    return np.clip(out, 0.0, 1.0)

@@ gap_integral_from_distribution_direct, centers branch
-        vals = np.maximum(vals, 0.0)
-        first_active = int(np.flatnonzero(active)[0])
-        if vals[first_active] > 1.0 + 64.0 * np.finfo(float).eps:
-            raise ValueError(...)                      # the guard I applied
+        vals = np.maximum(vals, 0.0)
+        # Only the first active node is an extrapolation that carries weight;
+        # bound it to the occupation domain the input is held to.  This is a
+        # strict no-op when the extrapolate already lies in [0, 1].
+        vals = np.minimum(vals, 1.0)

# Dead-code note (verifier, not acted on): the `vals = np.maximum(f_arr, 0.0)`
# in the edges branch can never bite - f_arr already passed the strict
# (f_arr < 0) | (f_arr > 1) rejection above.
```

---

## `qpsim/phonon_models/ph0_local.py`

### P16 — Commensurability guard for the Ph0 omega lattice (2*E_min/dE must be an integer)

Location: `qpsim/phonon_models/ph0_local.py:397`

**Why held back.** BEHAVIOUR-CHANGING and out of packet. (a) The guard rejects configurations that run today and produce numbers: the WebUI default EnergyGrid(min_factor=1.0, max_factor=10.0, 400 bins) with a dynamic phonon mode, the SteadyState0D 64-bin default, and the whole scripts/run_prelim_readout_heating_overnight.py campaign (NE=101, max_factor 5.0). (b) The correct placement is qpsim/collisions/phonon.py build_phonon_frequency_map's uniform branch (phonon.py:301-312), which I do not own - T3DiffusionBackend._validate_phonon_on_physics_grid (t3_diffusion.py:2154) and webui/builders.py:385 call it directly and would otherwise keep constructing split grids. (c) The real remedy (emit both channels onto one common uniform lattice with rate-preserving two-bin deposition, updating phonon_occupation_matrices_from_state, phonon_source_sink_jacobian_f, phonon_collision_jacobian_nph and the _pair_breaking_quadrature_correction bin map in lockstep) is a physics change needing physicist sign-off. Note the guard must NOT fire on the thermal-bath path (n_ph pinned, split harmless), and must pass trivially for E_min = 0 (ratio 0).

**Expected numerical impact.** No promoted canonical changes (all satisfy the condition). On split grids the guard converts silently-wrong results into an error; the verifier's end-to-end solves put the current error at 7.2% low in x_qp at the shipped tau_l = 0.170 ns, 12.8% at 1 ns, 14.7% at 10 ns, and measured x_qp = 3.488e-3 / 2.207e-3 / 3.007e-3 for grid-origin shifts of 0 / 2 / 4 ueV at fixed dE (a 32% non-monotone swing in a physically meaningless parameter). Structurally verified here: diff/sum index sets have exactly zero overlap on every incommensurate grid, with 311 orphaned above-2*Delta bins on the WebUI default.

**Prepared patch.**

```
# qpsim/collisions/phonon.py, uniform fast path of build_phonon_frequency_map
# (immediately after `spacing` is established, before diff_values/sum_values):
+    # The pair-difference lattice {k*dE} and the pair-sum lattice
+    # {2*E[0] + m*dE} are merged only within a roundoff-sized tolerance, so
+    # they coincide only when 2*E_min/dE is an integer.  Otherwise every omega
+    # bin receives exactly one of the two channels: difference-only bins above
+    # 2*Delta lose the Kaplan pair-breaking sink entirely and the coupled
+    # solution stops converging under refinement.  Fail loudly, mirroring the
+    # photon-lattice policy in pair_breaking_photon.validate_pair_breaking_photon_grid.
+    lattice_ratio = 2.0 * float(E[0]) / spacing - 1.0        # == 2*E_min/dE
+    if abs(lattice_ratio - round(lattice_ratio)) > 1e-9:
+        n_valid = int(round(len(E) * round(lattice_ratio) / lattice_ratio))
+        raise ValueError(
+            "Phonon frequency map requires 2*E_min/dE to be an integer so the "
+            "pair-difference and pair-sum lattices coincide; got "
+            f"{lattice_ratio:.6g}. The pair-breaking channel would be absent "
+            "from every difference-only bin above 2*Delta. Use "
+            f"num_energy_bins={n_valid} on this energy range, or E_min = 0."
+        )

# If it must live in my file instead, the same three lines go at the top of
# qpsim/phonon_models/ph0_local.phonon_steady_state, derived from ctx.E.
```

---

## `qpsim/physics/gap_equation.py`

### P12 — Tightening / re-basing the solve_gap gap-edge resolution threshold (edge_width > 0.25 * gap_scale)

Location: `qpsim/physics/gap_equation.py:511`

**Why held back.** BEHAVIOUR-CHANGING for the warning surface, and the filed fix is a regression. Changing which configurations emit a RuntimeWarning changes what a recertification run reports, and any correct version needs a new physics estimator (the temperature amplification), which is a physics change requiring sign-off. The verifier measured that the filer's proposed `0.03 * gap_scale` threshold (a) still admits +0.45% at T/T_c=0.95 and ~+83% at T/T_c=0.99, so it does not bound the error, and (b) fires spuriously on the shipped webui spatial preset EnergyGrid(min_factor=1.0, max_factor=4.0, num_bins=64) at qpsim/webui/schemas.py:246 (dE = 0.047*Delta, true error ~2e-5 at its default T_bath=0.1 K), sitting only 25% above the 0-D webui default. Error scales as (dE/Delta)^~1.5 times an amplification spanning ~1e-16 to O(1), so NO fixed width ratio is a correct gate. I applied only the neutral half: the comment and message no longer assert the false O(dE) law or present the width ratio as an error estimate.

**Expected numerical impact.** The guard change alters no solved number - only which runs warn. With a naive 0.03 threshold: new spurious warnings on the shipped webui spatial preset; with the temperature-aware form: silent there, warning near T_c where the error is real. The gap_suppression alternative DOES change published numbers: on a thermal state at T/T_c=0.83 with dE=0.225*Delta it removes a spurious rel_suppression of -2.4e-2 (a sign-flipped 2.4% apparent gap enhancement). Shipped production grids (dE ~ 0.006*Delta_ref for Fig. 6) are unaffected either way.

**Prepared patch.**

```
In solve_gap, replace the bare ratio test with a temperature-aware estimate (needs a new helper, hence the sign-off):

     edge_idx = min(int(np.searchsorted(E, gap_scale)), widths.size - 1)
     edge_width = float(widths[edge_idx])
-    if edge_width > 0.25 * gap_scale:
+    # (dE/Delta_ref)^{3/2} times the amplification A from the flattening
+    # residual slope d(residual)/d(Delta) at Delta_ref, which is what turns
+    # a sub-percent quadrature error into a percent-level gap error near T_c.
+    amplification = _residual_slope_amplification(calibration, gap_scale)
+    est_rel_error = (edge_width / gap_scale) ** 1.5 * amplification
+    if est_rel_error > 1e-3:
         warnings.warn(
-            ... "is {edge_width / gap_scale:.0%} of the reference gap scale" ...
+            ... "implies an estimated relative gap error of {est_rel_error:.1e}" ...

Cheaper alternative that removes the real user-facing exposure instead (also held back, and in a file I do not own): have compute_gap_suppression (qpsim/observables/gap_suppression.py:73) solve its THERMAL REFERENCE on the same grid as the driven state, as compute_gap_suppression_direct already does, so the discretization bias cancels to leading order.
```

---

## `qpsim/physics/spectral.py`

### P14 — Factored BCS radicand in bcs_density_of_states / bcs_anomalous_weight (finding 1, lines 127 and 142)

Location: `qpsim/physics/spectral.py:127`

**Why held back.** BEHAVIOUR-CHANGING on a CERTIFIED baseline — I ran the pinned-digest check the verifier asked for and it FAILED. The verifier said "apply as proposed" but also "should be landed with a pinned-digest check rather than blind"; the filer's claim that it is "bit-identical to the factored form on all four production grids" is only true when the gap sits exactly on a cell face. Measured (numpy 2.5.1, review checkout): on the Fig. 6 grid with an OFF-lattice gap (e.g. the promoted anchor 179.996345344668612) rho and N2 move by up to 4.4e-15 / 4.5e-15 relative; on the Al NE=101/200/400 grids by 2e-16..1.3e-15; on the ac_conductivity sub-gap partner grid by 3.2e-11. Propagating that through `compute_ac_conductivity`: sigma_2 moves by 1.7e-13 relative in 12/12 trials on the EXACT certified fig7 grid (Delta=189, 1701 bins, omega_0=22), and sigma_2 feeds `qpsim/observables/quality_factor.py:55` -> `Q_qp = s2/(alpha*s1)`, which produces the certified baseline validation/baselines/ph0_constant/fischer_figs_9_13_qi_vs_pread.csv. (sigma_1, which fig7_paper.py:539 uses for Q_i,qp, WAS bit-identical in 12/12 on that grid, so fischer_fig7_paper.csv itself would survive; figs_9_13 would not.) Under an imminent recertification this is exactly the intended-drift-vs-regression ambiguity the packet rules forbid. Note the finding is filed LOW and explicitly "a latent public-API defect, not a live wrong number in any shipped figure", so the accuracy gain buys nothing today. Recommend landing it WITH the next figs_9_13 regeneration.

**Expected numerical impact.** Public-API accuracy at the gap edge improves from rel err ~eps/(2*delta) to ~1 ulp (measured: 5.3e-4 -> 0 at E = Delta*(1+1e-14), 1.05e-5 -> 0 at 1e-12, 7.3e-9 -> 0 at 1e-9). Cost on shipped artifacts: fischer_figs_9_13_qi_vs_pread.csv Q_i moves at ~1.7e-13 relative (via sigma_2); fischer_fig7_paper.csv sigma_1/Q_i bit-identical (12/12 probes); fig6/fig3/fig5 unaffected because the BCS path takes cell_weights from the already-factored `bcs_dos_cell_weights` and only stores `spectral.rho` (consumed as a >0 mask, by the Dynes-only m25_junction.py:154, and by the runtime signature at device.py:225).

**Prepared patch.**

```
--- a/qpsim/physics/spectral.py
+++ b/qpsim/physics/spectral.py
@@ -125,7 +125,9 @@ def bcs_density_of_states(E: np.ndarray, gap: float) -> np.ndarray:
     rho = np.zeros_like(E, dtype=float)
     valid = gap < E
-    rho[valid] = E[valid] / np.sqrt(E[valid] ** 2 - gap ** 2)
+    # Factored form avoids cancellation in E^2-Delta^2 at the gap edge
+    # (Sterbenz makes E - Delta exact for Delta/2 <= E <= 2 Delta), matching
+    # the convention of qpsim/physics/bcs_quadrature.py:143-144.
+    rho[valid] = E[valid] / np.sqrt((E[valid] - gap) * (E[valid] + gap))
     return rho
@@ -140,7 +142,7 @@ def bcs_anomalous_weight(E: np.ndarray, gap: float) -> np.ndarray:
     n2 = np.zeros_like(E, dtype=float)
     valid = gap < E
-    n2[valid] = gap / np.sqrt(E[valid] ** 2 - gap ** 2)
+    n2[valid] = gap / np.sqrt((E[valid] - gap) * (E[valid] + gap))
     return n2

(The `valid = gap < E` mask keeps the radicand strictly positive, and the form degenerates correctly to E at gap = 0. Do NOT also change line 452 — see `skipped`.)
```

### P14 — arccosh -> factored arcsinh for the BCS anomalous cell weight in SpectralContext._rebuild (finding 2)

Location: `qpsim/physics/spectral.py:435`

**Why held back.** BEHAVIOUR-CHANGING on the promoted Fig. 6 baseline, and it cannot be landed from this packet alone. (a) The change is not bit-neutral whenever the gap sits off a cell face, which is the normal state of the self-consistent moving-gap solve: with the cut-cell face-to-gap distance hi-gap ~ 3.65e-3 ueV at the promoted anchor (delta_driven = 179.996345344668612, faces on the integers), the anomalous weight moves at ~5e-12 relative, which propagates into K_minus/K_plus and hence into the Picard fixed point and every promoted ordinate. The repo pins Fig. 6 state bit-exactly (tests/validation/test_fig6_author_frozen_state.py), so this is provenance-breaking. (b) Per verifier_corrections (b), the fix list is incomplete and two of the three sites are OUTSIDE my packet: qpsim/backends/t3_spatial_1d.py:122-125 repeats the identical arccosh formula, and validation/fischer_2023/fig6_author_c3_score.py:620-628 mirrors the qpsim kernel (factored DOS weights + arccosh anomalous weight in the same block). Changing spectral.py alone would silently stop that certified-evidence script from mirroring the kernel it scores. All three must move together, in one packet, with a Fig. 6 regeneration.

**Expected numerical impact.** Removes an eps/(2*(hi-gap)/gap) relative error confined to the single gap-cut cell (verifier-measured: 2.2e-3 relative on the weight at hi-gap = 1e-12 ueV, 5.3e-6 at 1e-9, 1.1e-10 at 1e-6). Elementwise K_minus[cut,cut] error up to O(1) relative; measure-weighted error in any K+/K--integrated quantity <= ~1e-10 relative. On the shipped Fig. 6 solves the driven suppression keeps hi-gap >= ~1e-4 ueV so today's reachable error is <= ~1e-10 relative — but the promoted CSV and the frozen-state test would still move in their trailing digits.

**Prepared patch.**

```
--- a/qpsim/physics/spectral.py
+++ b/qpsim/physics/spectral.py
@@ -432,10 +432,14 @@
             edges = cell_edges_from_widths(E, self._dE)
             lo = np.maximum(edges[:-1], gap)
             hi = np.maximum(edges[1:], lo)
-            anomalous_weight = gap * (
-                np.arccosh(np.maximum(hi / gap, 1.0))
-                - np.arccosh(np.maximum(lo / gap, 1.0))
-            )
+            # asinh(sqrt(E^2-Delta^2)/Delta) == acosh(E/Delta), while the
+            # factored radicand remains accurate for a cell face arbitrarily
+            # close to Delta (cf. qpsim/physics/gap_equation.py:196-200).
+            xi_hi = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0))
+            xi_lo = np.sqrt(np.maximum((lo - gap) * (lo + gap), 0.0))
+            anomalous_weight = gap * (
+                np.arcsinh(xi_hi / gap) - np.arcsinh(xi_lo / gap)
+            )

Mirror the identical replacement at qpsim/backends/t3_spatial_1d.py:122-125 and at validation/fischer_2023/fig6_author_c3_score.py:620-628 in the SAME change (both outside packet P14).
```

---

## `qpsim/services/rate_equation.py`

### P04 — Remove the 1e-14 Hz absolute floor from _source_scaled_residual_tolerances (covers BOTH floor findings - the 'test-gate' one and the 'numerics' one, same line)

Location: `qpsim/services/rate_equation.py:584`

**Why held back.** BEHAVIOUR-CHANGING on two counts. (1) It is a tolerance that currently gates accepted results: the verifier measured that removing it converts currently-silent returns into RuntimeErrors for weak-drive bundles (four reproduced pseudo-roots at g = 1e-20 Hz flip from accepted to rejected). (2) rate_equation.py is inside the hashed source closure of the Fig 6 C3-C7 ladder scores AND the M25 Fig 3/4 canonical artifact manifests (validation/marchegiani_2025/_artifact.py binds the source fingerprint), so any edit here forces a provenance-breaking republication - which is precisely why the verifier said to bundle it into the next planned regeneration rather than ship it alone. The physics-facing half (an acceptance criterion for M25 Eqs. 4-6) also wants a physicist sign-off. I applied the neutral half instead: the docstring now records the limit and the measured headroom.

**Expected numerical impact.** Zero at every shipped M25 Fig 3/4 operating point - I measured the floor to be inactive there by 240x (Fig 3a) to 1000x (Fig 3b), and tests/services/test_rate_equation.py::test_fig3a_death_valley_uses_row_wise_source_gate asserts only an upper bound (tolerances[1:] < 3e-11), which the removal cannot break. Away from the shipped points it is a behaviour change: for any bundle whose per-row aggregate generation is below ~1e-11 Hz the density rows tighten by up to ~1e9x (1e-14 -> ~1e-23 in the verifier's probe), turning currently-accepted single-seed states (measured up to 28x wrong, theoretical bound ~1000x) into RuntimeErrors. It also advances the source fingerprint of every M25 and Fig-6-ladder artifact manifest.

**Prepared patch.**

```
In qpsim/services/rate_equation.py, replace

    backward_error = 64.0 * np.finfo(float).eps * term_sums
    return np.maximum(
        1e-14,
        residual_tol_relative * row_sources + backward_error,
    )

with

    backward_error = 64.0 * np.finfo(float).eps * term_sums
    return residual_tol_relative * row_sources + backward_error

and drop the ``max(1e-14, ...)`` from the two docstring statements of the gate (line 538 in this function and line 649 in _passes_source_scaled_residual_gate's caller docstring), plus the floor paragraph I added at line 549. Do NOT substitute a 'tiny * term_sum' surrogate: 64*eps*sum_j|term_ij| is already a rigorous per-row summation backward-error bound, and the all-zero row passes with tolerance 0 because its residual is then exactly 0.
```

### P04 — Add large-asymmetry rows to _default_seed_grid so it brackets both M25 Fig 3 orderings

Location: `qpsim/services/rate_equation.py:1970`

**Why held back.** BEHAVIOUR-CHANGING: adding seeds changes the multi-seed candidate pool and therefore which fixed point solve_rate_equation_steady_state_multi_seed can select, and it changes the pinned certified Fig 3/4 rows' provenance fingerprint. The verifier explicitly rejected this half of the filer's fix ('Do not add a second hand-tuned magic ratio (e.g. 7)') and pointed at wiring analytic_low_T_seed into the one call site that passes neither preferred_seed nor extra_seeds instead - which is qpsim/devices/m25_junction.py:340-344, outside my packet. I applied only the docstring half and pinned the existing ratios so they cannot drift silently.

**Expected numerical impact.** None expected at the shipped Fig 3a/3b parameters (single-seed and multi-seed already agree to ~1e-15 relative there, and the picker gates on residual), but it doubles the candidate pool, so any bundle where the picker currently ties on residual could switch branch. It also advances the M25 artifact source fingerprint, forcing a certified-row republication.

**Prepared patch.**

```
Two options, both outside what I may land. (a) Minimal, inside this file:

    seeds: list[np.ndarray] = []
    for p_1 in (1e-4, 3e-4, 1e-3):
        for x in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
            seeds.append(np.array([p_1, x, 0.4 * x, 0.02 * x]))
            # SI Eq. S69 ordering x_R< >> x_L >> x_R> (M25 Fig 3b).
            seeds.append(np.array([p_1, x, 0.005 * x, 7.0 * x]))
    return seeds

(b) Preferred per the verifier: leave _default_seed_grid alone and pass an analytic_low_T_seed-derived preferred_seed from M25GapAsymmetricJJ._ensure_moment_solution_cached (qpsim/devices/m25_junction.py:340-344).
```

---

## `qpsim/services/transient.py`

### P15 — Full fix (a) for the snapshot dense-output defect: clamp the driver step so every snapshot lands on an integration endpoint

Location: `qpsim/services/transient.py:214`

**Why held back.** BEHAVIOUR-CHANGING. It changes step sizes, hence bits, and n_steps for any cadence not commensurate with dt. Several tests in tests/services/test_transient.py assert exact n_steps / n_etd_substeps (e.g. test_final_step_is_truncated_to_total_time asserts n_steps == 4) and would need re-pinning, and the pinned photon-kick baseline (validation/baselines/transient/photon_kick_response.csv, compared at rtol=1e-6) would need re-checking for the sub-picosecond residual steps at DT=0.1 / SNAPSHOT_INTERVAL=6.0. The verifier explicitly flagged that the filer's 'this is harmless' framing is wrong on exactly this point. Order and stability are safe (etd1_step/etd2_step are one-step self-starting, no multistep history is threaded through apply_collisions), so this is the right real fix for a regeneration window — just not for a recertification pass. Fix (c) was landed instead (see applied).

**Expected numerical impact.** Removes the interior-snapshot error entirely (verifier measured up to 4.36x pointwise in f(E) and 3.03x in x_qp on the real T3DiffusionBackend at dt=1000 ns / interval=100 ns). For in-repo callers (dt/interval ratios 1/20-1/60) it perturbs step boundaries at the ~1e-12 level and changes integer step counts; no physics changes.

**Prepared patch.**

```
In run_time_dependent's time loop:

-        step_dt = min(dt, remaining)
+        # Land every cadence boundary on an integration endpoint: the linear
+        # dense output below is only first-order consistent with exponential
+        # relaxation, and ETD2 here is one-step self-starting with internal
+        # rate subcycling, so shortening a driver step costs neither order
+        # nor stability.
+        step_dt = min(dt, remaining, next_snap - t)

and then drop the `snapshot_interval < dt` warning added by the applied fix, plus the docstring sentence that points at it. Re-pin the exact n_steps / n_etd_substeps assertions in tests/services/test_transient.py and re-verify validation/baselines/transient/photon_kick_response.csv.
```

---

## `qpsim/solvers/coupled_newton.py`

### P11 — Port the newton_solve_f projected-vacuum guard into the coupled-Newton line search

Location: `qpsim/solvers/coupled_newton.py:729`

**Why held back.** BEHAVIOUR-CHANGING on a certified path. The line search is the solver's acceptance rule; changing it changes which iterate sequence a solve follows. Concretely: validation/fischer_2023/fig6_solve.py:534-545 catches CoupledNewtonLineSearchError and silently retries with strict Picard whenever residual_norm <= PICARD_TOL = 1e-12. A vacuum collapse at Fig.-6 cold amplitudes lands at exactly such a tiny norm, so if ANY of the 66 pinned Fig.-6 points currently reaches its baseline value through that Picard fallback, installing the guard makes coupled Newton converge on its own and the pinned number is then produced by a different solver. docs/CURRENT-STATUS.md separately records the 0.10 K Fig. 6 curve as 'not certifiable in double precision -- line search at the float64 noise floor'; whether any of those points are this collapse is untested and cannot be settled without re-running the 12.075 aggregate-worker-hour sweep. That adjudication plus a physicist sign-off is a precondition, and mixing this into a neutral batch would make intended physics drift indistinguishable from a regression during recertification. NOTE: I did NOT hold back a fix I am unsure of -- I verified the patch below works (see expected_numerical_impact). Also per verifier_corrections (a), I explicitly REJECT the filer's alternative 'stronger and cheaper' merit `norm_t < norm AND balance_ratio_t <= balance_ratio': the balance ratio is a return certificate, it is exactly 1.0 at any non-absorbing vacuum and is not monotone along a good Newton path, and adopting it makes the solver strictly worse.

**Expected numerical impact.** No currently-converging solve in my sweep changes by a single printed digit; three currently-FAILING flat seeds start converging. Measured with the patch exec'd into a scratch module (repo untouched), undriven Ph0 case NE=40, Delta = 1.764*k_B*1.2 K = 182.4117 ueV, dE = Delta/8, tau_0 = 1 ns, T_bath = 0.2 K, tau_l = 2 ns, tol = 1e-14, step_rtol = 1e-8, analytic_cross = True, max_iter = 120. Current code vs guarded, flat seed -> f_max: 1e-1 1.3065791673e-05 / 1.3065791673e-05 (identical); 1e-2 RuntimeError('did not converge in 120 iterations, max|residual| = 1.78e-10') / 1.3065791795e-05; 3e-3 RuntimeError / 1.3065791674e-05; 1e-3 RuntimeError / 1.3065791689e-05; 3e-4, 1e-4, 1e-5 all identical between the two. The true root is the unforced thermal fixed point, max f_FD(E, 0.2 K) = 1.3065791673027851e-05, reproduced exactly from a thermal seed. Confirmed the verifier's correction that at max_iter=120 the raised exception is RuntimeError (iteration exhaustion), not CoupledNewtonLineSearchError -- the line search keeps succeeding on the phonon block after the f-block collapses. Also confirmed the separate cold-undershoot stall at flat 1e-6 is out of scope and NOT fixed by this guard, so it must not be folded into the same regression. A regression already exists in the tree as an xfail: tests/review_2026_08_03/test_P11.py::test_flat_hot_seed_reaches_the_unforced_thermal_root -- delete its @pytest.mark.xfail when the guard lands. tests/review_2026_08_03/test_P11.py::test_shaped_seeds_still_reach_the_unforced_thermal_root is the guardrail that the currently-passing seeds stay passing.

**Prepared patch.**

```
--- a/qpsim/solvers/coupled_newton.py
+++ b/qpsim/solvers/coupled_newton.py
@@ (backtracking line search, immediately after the projection)
         for _ in range(20):
             f_trial = np.clip(f + alpha * delta_f, 0.0, 1.0)
             n_trial = np.maximum(n_ph + alpha * delta_n, 0.0)
+            if not np.any(f_trial[active_f] != 0.0) and np.any(f[active_f] != 0.0):
+                # Projection can turn an overshooting cold Newton direction
+                # into the exact vacuum.  A smaller dimensional residual does
+                # not make that boundary point physical when finite pair
+                # absorption still drives QPs out of it, and the f-block
+                # cannot leave f == 0 afterwards.  Mirrors the newton_solve_f
+                # guard (newton_steady_state.py:1213-1255): reject only a
+                # *proved non-absorbing* vacuum and backtrack the same Newton
+                # direction.
+                _, N_emit_t, N_abs_t = phonon_occupation_matrices_from_state(
+                    n_trial, omega_idx_diff, omega_idx_sum, diff_sign,
+                )
+                gain_trial_number, _, _ = number_changing_gain_loss(
+                    f_trial, ctx, K_r0, T_bath,
+                    pb_photon_params=pb_photon_params,
+                    N_emit=N_emit_t, N_abs=N_abs_t,
+                    external_flux=external_flux,
+                )
+                if not _is_exact_absorbing_vacuum(
+                    f_trial, gain_trial_number, active_f,
+                    T_bath=T_bath, K_r0=K_r0, N_abs=N_abs_t,
+                    N_abs_override=None,
+                    pb_photon_params=pb_photon_params, ctx=ctx,
+                ):
+                    alpha *= 0.5
+                    continue
             R_f_t, R_ph_t, balance_ratio_t = residual(f_trial, n_trial)

(All four helpers -- phonon_occupation_matrices_from_state, number_changing_gain_loss,
_is_exact_absorbing_vacuum -- are already imported and already used elsewhere in this
file, e.g. slow_balance_ratio at lines 520-564; no new imports are needed.)

And in tests/review_2026_08_03/test_P11.py, drop the @pytest.mark.xfail(...) decorator
on test_flat_hot_seed_reaches_the_unforced_thermal_root.
```

---

## `qpsim/solvers/newton_steady_state.py`

### P03 — Componentwise (per-bin relative) return gate alongside the existing normwise certificates

Location: `qpsim/solvers/newton_steady_state.py:1060`

**Why held back.** BEHAVIOUR-CHANGING, and both verifiers independently said it must not ship without a driven-regime study. Adding `max_i |R_i| / max(|gain_i|, |loss_i f_i|) <= rel_tol` to the return path converts states the solver accepts today into either extra Newton iterations or loud RuntimeErrors, at exactly the cold/driven operating points the project has pinned. Three concrete blockers the verifiers measured: (1) at T=0.015 K, 39 active rows have f underflowed to exactly 0 while gain_i is still representable, giving a per-bin relative residual of exactly 1.000 in 9 rows -- a naive gate makes the solver unable to ever return there; (2) docs/CURRENT-STATUS.md already records that the 0.10 K Fig-6 Q0 point 'is not certifiable in double precision -- certificate mode fails loud at any threshold (line search at the float64 noise floor)', so a hard gate would very likely convert currently-passing cold driven points into failures; (3) a return-only gate is insufficient anyway, because the line-search merit inside `try_direction` (lines ~1265-1315) is also purely normwise, so the gate must be paired with a componentwise merit or cold solves just trade silent-accept for RuntimeError. The turnover-fraction cut also cannot be chosen naively: one verifier's ExternalFlux probe showed the affected bin at 9.0e-19 of peak turnover, so any 'exceeds a fraction of peak turnover' cutoff (the filer's own proposal) reinstates the blind zone it is meant to close. This needs a physicist/numerics sign-off and a Fig-7 cold/high-drive calibration, not a review patch. I applied the documentation half instead.

**Expected numerical impact.** Zero on any state that is already a true root: measured attainable per-bin relative residual at the converged root is 8.45e-15 at 0.15 K, 2.07e-14 at 0.06 K, 4.45e-15 at 0.30 K, 6.97e-14 at 0.02 K -- ~8 decades of headroom under a 1e-6 gate. Nonzero only where the solver currently exits with zero Newton steps: the demonstrated case returns a tail that is a factor 2 (128/200 bins above 0.1 relative imbalance) or 1e3 too large above ~2.8 Delta. x_qp moves 3.1e-12 relative and energy density 8.5e-12, so no published observable is impeached today; the change would be visible in log-axis f(E), effective-tail-temperature extraction, and any tail-sensitive observable added later. A driven ExternalFlux source below ~1e-14/ns would start being solved for rather than discarded.

**Prepared patch.**

```
In `newton_solve_f`, alongside `converged_abs`/`converged_balance` at lines 1059-1060:

    # Componentwise companion to the normwise certificate. The per-row
    # physical scale is the same one `_row_scaled_newton_system` uses;
    # exactly-zero-turnover rows carry no local scale and are exempt by
    # construction, as they already are there (line ~713).
    row_scale = np.maximum(np.abs(gain_int[active]), np.abs(loss_int[active] * f_cur[active]))
    scalable = row_scale > 0.0
    componentwise_error = (
        float(np.max(np.abs(R[active][scalable]) / row_scale[scalable]))
        if np.any(scalable) else 0.0
    )
    converged_componentwise = componentwise_error <= backward_error_tol

then require `converged_componentwise` on every `return f_cur.copy()` path (lines 1064, 1138, 1152) and add it to the two RuntimeError diagnostic strings (lines ~1489 and ~1508). Pair it with a componentwise term in the `try_direction` merit (lines ~1265-1315) so a rejected state still has a descent direction. Do NOT restrict the gate to rows above a fraction of peak turnover -- that reinstates the blind zone. Rows whose f has underflowed to exactly 0 while gain_i is representable need an explicit, documented exemption or the 15 mK path cannot return.
```

---

## `qpsim/solvers/picard.py`

### P06 — Replace the floored-denominator convergence test with an explicit two-tolerance (atol/rtol) test plus normwise guard

Location: `qpsim/solvers/picard.py:177`

**Why held back.** BEHAVIOUR-CHANGING on two counts. (1) It changes both `converged` and the returned `final_residual` for every call - most visibly it flips converged True->False for components whose magnitude sits far below tol (the verifier's reproduction: G(z)=z+1e-7*(c-z), z0=0, tol=1e-10 gives converged=True/rel_err=1.0 at c=1e-16 today and would correctly reject after the fix). (2) tests/solvers/test_picard.py:104-111 and :146-153 recompute the current `change / (np.maximum(|x|,|gx|) + tol)` formula inline inside their assertions and would have to be updated in lockstep - that file is NOT in packet P06, so I cannot make the change atomic. Applying it here would leave the branch red and would mix a solver-semantics change into a recertification run. Mitigating context (verifier-confirmed): picard_iterate has zero production callers (only the qpsim.solvers re-export and its own tests), the sole production fixed-point site carries its own correct two-tolerance + normwise test at qpsim/services/steady_state.py:93-110, and the `+ tol` denominator predates commit 50cbc03, so nothing shipped or pinned is affected. Recommend routing this through the packet that owns tests/solvers/test_picard.py, or a follow-up commit of its own.

**Expected numerical impact.** No published, pinned, or figure-facing number moves (no reachable caller). Within picard_iterate: `final_residual` changes definition for all inputs; the verdict changes only for components with |x| <~ tol*1e-3 (1e-13 at defaults) - e.g. the f ~ 1e-52 weak-drive thermal roots, where today rtol_eff ~ 1e32 and everything passes trivially. The xqp ~ 1.22e-10 case cited in the original filing does NOT change (loosening there is a factor 1.8, benign) - the verifier struck that half of the justification.

**Prepared patch.**

```
In the signature, after `tol: float = 1e-10,` add:
    atol: float = 0.0,
and validate it beside the tol check:
    if not (np.isfinite(atol) and atol >= 0.0):
        raise ValueError(f"atol must be finite and non-negative; got {atol!r}.")

Replace picard.py:176-180
        change = np.abs(gx - x)
        scale = np.maximum(np.abs(x), np.abs(gx)) + tol
        final_residual = float(np.max(change / scale))

        if final_residual < tol:
with
        change = np.abs(gx - x)
        # Explicit two-tolerance test (mirrors services/steady_state.py:93-101):
        # a floored denominator makes the verdict depend on the units of x.
        allowed = atol + tol * np.maximum(np.abs(x), np.abs(gx))
        ratio = np.divide(
            change,
            allowed,
            out=np.full_like(change, np.inf),
            where=allowed > 0.0,
        )
        ratio[(allowed == 0.0) & (change == 0.0)] = 0.0
        final_residual = float(np.max(ratio)) * tol
        # Amplitude-independent normwise guard: the per-bin test alone is
        # unsafe when *every* component sits below the absolute floor
        # (steady_state.py:66-70).
        norm_scale = float(np.sum(np.abs(x) + np.abs(gx)))
        normwise_ok = (
            float(np.sum(change)) <= tol * norm_scale if norm_scale > 0.0 else True
        )

        if bool(np.all(change <= allowed)) and normwise_ok:

Update PicardInfo.final_residual (lines 36-39) and the tol/atol parameter docs to the new quantity, and update the two inline recomputations in tests/solvers/test_picard.py (lines ~104-111 and ~146-153). Add regressions: (a) G(z)=z+1e-7*(c-z), z0=0, tol=1e-10 asserts converged=False at c=1e-16; (b) scale invariance - identical verdict at c=1e-16 and c=1.0.
```

---

## `qpsim/webui/builders.py`

### P07 — WebUI coupled-Newton route never maps coupled_newton_analytic_cross, so it runs the O(NE^3) finite-difference cross-Jacobian

Location: `qpsim/webui/builders.py:456`

**Why held back.** Selects a different Jacobian for a user-reachable solve. Both paths converge to the same root only to ~1e-19, and the FD path can terminate in CoupledNewtonLineSearchError where the analytic one converges — so it is not bit-neutral for a currently-valid input, and a solver-mode flip mixed into an otherwise comment-only diff is exactly what makes recertification drift unattributable. Zero certified numbers flow through steady_state_solver_kwargs (only qpsim/webui/execute.py:161 calls it), so this is safe to land immediately after recertification. I followed the verifier's preference over the filer's alternative: do NOT flip the backend default at t3_diffusion.py:539 — tests/backends/test_t3_coupled_newton_path.py and tests/devices/test_external_flux.py reach steady_state without naming the flag and are part of the blast radius.

**Expected numerical impact.** Identical root to ~1e-19 where the FD path converges today; ~88x faster per Newton iteration at the shipped 400-bin default (0.38 s vs 34 s, verifier-measured). Some strong-drive cold-bath cases that currently fail or hit max_iter would instead converge, i.e. a result where there is now an error.

**Prepared patch.**

```
In qpsim/webui/builders.py, steady_state_solver_kwargs:

     if method == "coupled_newton":
         kwargs["coupled_newton_tol"] = s.newton_tol
         kwargs["coupled_newton_max_iter"] = s.newton_max_iter
+        # The analytic cross blocks are FD-verified exact and every in-tree
+        # driver opts in; the UI has no reason to pay (NE + N_omega) residual
+        # evaluations per iteration.
+        kwargs["coupled_newton_analytic_cross"] = s.coupled_newton_analytic_cross

with a matching field in qpsim/webui/schemas.py SolverOptions (NOT in this packet — it needs the schemas owner):

     coupled_newton_analytic_cross: bool = True

Second, independent half the verifier rates as equally important and which is also held: builders.py:455 maps the UI's newton_max_iter default of 300 onto coupled_newton_max_iter (backend default 50), and qpsim/webui/execute.py checks cancellation only outside backend.steady_state (lines 164, 248), so a non-converging default-grid FD run is a multi-hour uninterruptible job pinning the worker. Cap it (e.g. min(s.newton_max_iter, 50)) or add a dedicated UI control.
```

---

## `validation/diffusion_operators/gap_gradient_drift.py`

### P10 — Benchmark 2 oracle uses point-sampled BCS DOS instead of the backend's conserved cell-average measure (packet findings #1 high and #2 medium - one defect, one patch)

Location: `validation/diffusion_operators/gap_gradient_drift.py:76`

**Why held back.** BEHAVIOUR-CHANGING: it moves numbers this project has published. The A1/A1P/A2 measured columns and all four q != 0 analytic columns of outputs/diffusion_operators/gap_gradient_drift.csv change, as do the markers in papers/qp-diffusion/figures/gap_gradient_drift (paper.tex:2963), and the paper's methodology sentence + fig caption (paper.tex:2929-2935, 2971-2974) would have to be re-checked against the regenerated data. It is also a validation-oracle/physics-convention change of the same family as the July-2026 correction, so it wants the recertification pass and a physicist sign-off rather than being mixed into a neutral batch. I applied only the doc half (the false 'conserved density' docstring + an explicit open-item note). Note that the sibling packet owning validation/diffusion_operators/test_gap_gradient_drift.py has already rewritten that test's comments to name the artifact and added a mesh-refinement gate, which is consistent with holding the numeric change.

**Expected numerical impact.** Measured myself on this branch (scratchpad/oracle_probe.py, same evolved state read under both measures). NE=12/NX=31 (test config): A1 max|v| 2.591018e-03 -> 1.344498e-08 um/ns (factor 1.9e5); A1P bin0 +7.267826e-02 -> +7.830744e-02 (+7.7%); A2 bin0 +4.770578e-02 -> +5.537239e-02 (+16%); C -1.952544e-02 and B -2.778021e-02 bit-identical (p = 0 => N_1^0 = 1). NE=40/NX=41 (published config, the CSV/figure): A1 max|v| 4.864780e-04 -> 7.915096e-09; A1P bin0 0.1096871 -> 0.1110223 (+1.2%); A2 bin0 0.07082564 -> 0.07243580 (+2.3%); C/B unchanged. Analytic columns change for all q != 0 models (e.g. C bin0 -1.84405e-02 -> -1.94577e-02, i.e. from 5.2% off the measured -1.952544e-02 to 0.35% off). Point/cell weight ratio at bin 0 runs 0.996887 -> 0.933390 across the ramp at NE=12 and 0.999574 -> 0.977958 at NE=40, so the artifact is mesh-convergent, not a fixed offset. Sign of the artifact is negative while A1P is positive, so the current relative gate admits a genuine positive A1 leak up to +6.2e-3 um/ns (8.6% of the A1P drift) on a quantity whose true value is 1e-8.

**Prepared patch.**

```
--- a/validation/diffusion_operators/gap_gradient_drift.py
+++ b/validation/diffusion_operators/gap_gradient_drift.py
@@ imports
-from qpsim.physics.spectral import SpectralContext, bcs_density_of_states
+from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights, cell_edges_from_widths
+from qpsim.physics.spectral import SpectralContext
@@ run(), line 76
-    ramp = np.linspace(gap_lo_factor * base_gap, gap_max, NX)
-    N1 = np.column_stack([bcs_density_of_states(E, float(g)) for g in ramp])  # (NE, NX)
+    ramp = np.linspace(gap_lo_factor * base_gap, gap_max, NX)
+    # Represented cell-average BCS DOS -- the measure T3Spatial1DBackend
+    # actually conserves (identical to self_consistent_feedback._n1_columns).
+    first_edge = float(cell_edges_from_widths(E, dE)[0])
+    N1 = np.column_stack(
+        [
+            bcs_dos_cell_weights(E, dE, float(g), lower_bound=max(float(g), first_edge))
+            / dE
+            for g in ramp
+        ]
+    )  # (NE, NX)

(dN1_dx / N1_center at lines 84-85 and _center_of_mass at 103/107 then
automatically use the same array; the module-docstring oracle note I added
should be deleted in the same commit.)

Test-side follow-up, in validation/diffusion_operators/test_gap_gradient_drift.py
(NOT my file - coordinate with its owner):
-    a1 = float(np.max(np.abs(result.drift_measured["A1"])))
-    a1p = float(np.max(np.abs(result.drift_measured["A1P"])))
-    assert a1 < 0.05 * a1p, (a1, a1p)
+    # A1 (q = 0) telescopes exactly on this grid: support_fraction == 1
+    # everywhere, so the true residual is CN round-off.
+    assert float(np.max(np.abs(result.drift_measured["A1"]))) < 1e-6
and tighten the analytic gate at line 57 from 0.15 to 0.05 for the NE=12
configuration only (measured max rel there becomes 0.0254/0.0157/0.0035/0.0002
for A1P/A2/C/B). Do NOT use the sibling's absolute 1e-8: I measured 1.34e-8
here, so 1e-8 fails immediately; 1e-6 still tightens the gate by ~3 orders.
```

### P13 — Weight the benchmark's center of mass with the backend's cell-average capacity and gate A1 absolutely at 1e-7

Location: `validation/diffusion_operators/gap_gradient_drift.py:76`

**Why held back.** BEHAVIOUR-CHANGING and out-of-packet. gap_gradient_drift.py is not in P13's file list, and the change would move published §7.5 benchmark-2 numbers: the A1 measured drift goes 2.591e-3 -> 1.34e-8 um/ns and the whole gap_gradient_drift.csv / fig:bench_drift figure would be regenerated. The verifier also measured a second-order cost the filer missed: re-weighting the COM while leaving drift_analytic on the point DOS raises the max relative error at the existing rel<0.15 gate from 0.025 -> 0.104 (A1P) and 0.067 -> 0.083 (A2), so the analytic velocity must be rebuilt from the same cell-average N1 (as self_consistent_feedback.py:347-359 does) in the same change. That is a physics-oracle change to a paper benchmark and needs the benchmark's owner plus a figure/CSV regeneration, not a review patch. Note also that copying benchmark 4's `< 1e-8` literal would FAIL at this benchmark's own configuration (true residual 1.34e-8); quote 1e-7.

**Expected numerical impact.** A1 measured drift 2.591e-3 -> 1.34e-8 um/ns at NE=12/NX=31 (1.3e-8 at 24/31, 7.9e-9 at the 40/41 default); A1P/A2 measured drifts shift by ~7-8%; C/B bit-identical (p = 0, weight cancels). Regenerates outputs/diffusion_operators/gap_gradient_drift.csv and the paper's fig:bench_drift. Gate ceiling tightens from 3.63e-3 to 1e-7 (~4.5 orders).

**Prepared patch.**

```
In validation/diffusion_operators/gap_gradient_drift.py, replace the point-DOS oracle

    N1 = np.column_stack([bcs_density_of_states(E, float(g)) for g in ramp])

with the represented cell-average capacity the backend conserves (the same construction as validation/diffusion_operators/self_consistent_feedback.py::_n1_columns):

    N1 = _n1_columns(E, dE, ramp)   # bcs_dos_cell_weights(...)/dE, lower_bound = max(gap, first edge)

and rebuild the analytic velocity from that same N1:

    dN1_dx = np.gradient(N1, x, axis=1)[:, center]
    N1_center = N1[:, center]

Then in validation/diffusion_operators/test_gap_gradient_drift.py replace the relative gate

    assert a1 < 0.05 * a1p, (a1, a1p)

with the absolute one

    assert np.max(np.abs(result.drift_measured['A1'])) < 1e-7

and drop the now-redundant test_a1_drift_collapses_under_energy_refinement.
```

---

## `validation/fischer_2023/extract_fig6_paper_data.py`

### P16 — CRLF-portability of the Fig. 6 / Fig. 8 P0 provenance chain

Location: `validation/fischer_2023/extract_fig6_paper_data.py:452`

**Why held back.** I wrote the fix (a diagnostic distinguishing a CRLF checkout from a genuine content difference at the byte-for-byte compare), then REVERTED it: manifest.extraction.script_sha256 in validation/paper_data/fischer_2023/fig6/oracle.json is bound to this very file's content, so ANY edit to it invalidates the binding and turns every Fig. 6 paper-parity test red on a correct (LF) checkout and in CI. Rebinding requires oracle.json, which is not in my packet. The real anchors are outside my packet too: validation/paper_parity.py:488 / :742 (raw-byte file_sha256) and .gitattributes. I confirmed the defect is live in this working tree - extract_fig6_paper_data.py and validation/paper_data/fischer_2023/fig6/points.csv are CRLF on disk, core.autocrlf=true, `git check-attr` covers only the .json - and that 9 of 20 tests in tests/validation/test_fischer_fig6_paper_parity.py already fail here with 'Digitizer source does not match manifest.extraction.script_sha256.' (pre-existing, present at c269af2). validation/paper_parity.py is being modified by another packet right now, which is where this belongs.

**Expected numerical impact.** None - fail-closed over-invalidation, no false accept is possible and no scientific value changes. Restores loadability of the strongest independent oracle on Windows checkouts. I did add the cheap portable guard the verifier suggested, in my own test file: source_sha256(digitizer) == manifest.extraction.script_sha256 and canonical-LF sha256(points.csv) == manifest.data.sha256, which fail loudly if anyone 'fixes' this by rebinding the digests to the CRLF values.

**Prepared patch.**

```
# 1. .gitattributes (not my file) - complete the existing pattern:
+validation/paper_data/**/*.csv text eol=lf
+validation/fischer_2023/*.py text eol=lf

# 2. validation/paper_parity.py (not my file), ~line 488 - content-defined
#    identity for the SCRIPT binding only:
-    if file_sha256(script_path) != expected_script_sha:
+    from validation.source_provenance import source_sha256
+    if source_sha256(script_path) != expected_script_sha:
#    (no digest re-derivation needed: source_sha256 of the CRLF working-tree
#     file already equals the recorded 488646f1... / fig8 9ccf8d13...)
#    Keep manifest.data.sha256 (points.csv) a RAW-byte binding - that is the
#    P0 re-extraction gate and the anti-CRLF regression guard at
#    tests/validation/test_fischer_fig6_paper_parity.py:434.

# 3. Only after 1+2, and only together with an oracle.json rebind, the
#    diagnostic I reverted from verify_external_source:
-    if generated != checked.read_bytes():
+    checked_bytes = checked.read_bytes()
+    if generated != checked_bytes:
+        if generated == checked_bytes.replace(b"\r\n", b"\n"):
+            raise PaperParityError(
+                f"{checked} was checked out with CRLF line endings; the "
+                "manifest binds the LF bytes. Re-checkout with core.autocrlf "
+                "disabled, or add a 'text eol=lf' rule for "
+                "validation/paper_data/**/*.csv, and do not rebind the digest."
+            )
         raise PaperParityError(
             "Re-extracted points differ byte-for-byte despite their declared hash."
         )

# Apply identically to fig8. Existing Windows clones need a renormalization
# (git rm --cached -r . && git reset --hard) or a re-clone; they will not
# self-heal from .gitattributes alone.
```

---

## `validation/fischer_2023/fig6_author_adapter.py`

### P06 — Respell the _SUBPROCESS_WRAPPER gap / photon-energy literals to match the authenticated entry point bit-for-bit

Location: `validation/fischer_2023/fig6_author_adapter.py:326`

**Why held back.** BEHAVIOUR-CHANGING and explicitly certificate-breaking. Editing _SUBPROCESS_WRAPPER changes SUBPROCESS_WRAPPER_SHA256 and the adapter digest, which are pinned in the anchor bundle and compared for exact equality at validation/fischer_2023/fig6_author_frozen_state.py:363 and :928, so the edit cannot land without a fresh authenticated author run - which also re-rolls the author's unseeded inverse-iteration start vector. It moves the repository's certified anchor coordinate `actual_t_star_over_delta` from 0.33990789737294363 to ...358, a 17-digit value quoted in validation/paper_data/fischer_2023/fig6/author-point-T020-sweep049-exact-anchor.json:31, docs/PAPER-REPRODUCTION-LADDER.md:107/:137/:184, docs/Validation_Chain.md:84/:100 and docs/CODE-REVIEW-FALSE-POSITIVES.md:279 - all outside this packet. Additionally, the verifier established the literal fix ALONE does not deliver the provenance claim it is meant to support: qp.T_star/qp.Delta disagrees with the author's own plotted expression T_star_list/qp.Delta at 1 ULP on 29 of 100 sweep indices even with the author's exact literals, so a genuine 17-digit abscissa-provenance fix must also evaluate the author's T_star_list expression (already pinned as a contract fragment at fig6_author_adapter.py:105). This is a paper-facing provenance change and needs the physicist/author-replay owner, not a review-fix agent.

**Expected numerical impact.** Provenance-only, one ULP. qp.T_star is a diagnostic the author documents as not entering the simulation; h, x_phot, qp.c, a_Delta, the rounded qp.Delta and n_bar[49] are all bit-identical between the two spellings, so no array, neither gap and not the ordinate moves. The reported abscissa scalar moves 0.33990789737294363 -> 0.33990789737294358 (0x1.009f32d95b560p-14 -> 0x1.009f32d95b55fp-14 before dividing by qp.Delta). Every pinned digest that covers the wrapper changes.

**Prepared patch.**

```
In `_SUBPROCESS_WRAPPER`:
-param["photon energy"] = 20.0e-6
+param["photon energy"] = 20 * 10**-6
-param["gap"] = 180.0e-6
+param["gap"] = 180 * 1e-6
(matching _ENTRYPOINT_CONTRACT_FRAGMENTS lines 95 and 99 exactly; the wrapper's other two rewrites, 1000.0e-18*1.544 and 255.0e-12, are already bit-identical to the fragments and must not be touched.)

Then, in the same change set and NOT separable from it: (1) re-run the authenticated author point and regenerate validation/paper_data/fischer_2023/fig6/author-point-T020-sweep049-exact-anchor.json (SUBPROCESS_WRAPPER_SHA256, adapter digest, actual_t_star_over_delta) and author-replay-sweep.json; (2) update the 17-digit quotations in docs/PAPER-REPRODUCTION-LADDER.md:107/:137/:184, docs/Validation_Chain.md:84/:100, docs/CODE-REVIEW-FALSE-POSITIVES.md:279; (3) scope docs/PAPER-REPRODUCTION-LADDER.md:162-163 ('That validation-script leak is corrected') to the C1 observable script, since it currently reads repo-wide; (4) to actually close the abscissa-provenance claim, evaluate the author's pinned T_star_list expression instead of qp.T_star/qp.Delta, which otherwise leaves a second independent 1-ULP source on ~29% of the curve.
```

---

## `validation/fischer_2023/fig6_author_c0_summary.py`

### P07 — C0's A1 agreement is gated only on the concatenated [f, n_phonon] state; a1_qp_final_relative_l2 is computed and reported but never enforced

Location: `validation/fischer_2023/fig6_author_c0_summary.py:684`

**Why held back.** TWO independent artifact-breaking effects, so I did not touch this file at all — not even a comment. (1) The file's own sha256 is pinned as a summary source in six committed certificates (c0/c2/c3/c5/c6/c7 score JSONs) and validation/paper_data/fischer_2023/fig6/reproduction-ladder.json, and load_c0_summary (:797-808) re-hashes the live file on every load and raises "Checked C0 summary source binding is stale" on any byte change — so even a comment turns tests/validation/test_fig6_author_c0_evidence.py and the C0-C7 evidence chain red. (2) Adding gates changes the acceptance.checks/acceptance.limits blocks inside c0-author-equivalent-score.json, so test_c0_checked_summary_regenerates_byte_exactly fails until the score is regenerated from the external raw bundles (which are not in this checkout). This is a gate-tolerance change on an accepted certified result — squarely the class my instructions reserve for recertification.

**Expected numerical impact.** No physics number changes. Only the certificate's acceptance.checks/limits blocks grow (and every dependent score's summary_sources digest rebinds). The blind direction on f shrinks from ~2.4e-6 relative L2 (the verifier's corrected figure: ||state||/||f|| = 2.40e5, not the filer's 166.6, so the gap is ~1400x worse than filed and ~3.7e7x above the 6.58e-14 the certificate advertises) to 1e-11.

**Prepared patch.**

```
_ACCEPTANCE_LIMITS (line ~124), add:
     "a1_qp_final_relative_l2_max": 1.0e-11,
     "a1_phonon_final_relative_l2_max": 1.0e-11,

acceptance_checks (line ~683), add alongside a1_final_state_relative_l2:
     "a1_qp_final_relative_l2": (
         qp_relative <= _ACCEPTANCE_LIMITS["a1_qp_final_relative_l2_max"]
     ),
     "a1_phonon_final_relative_l2": (
         phonon_relative
         <= _ACCEPTANCE_LIMITS["a1_phonon_final_relative_l2_max"]
     ),

Then regenerate c0-author-equivalent-score.json with the external C0/A1 raw bundles, rebind the summary-source digests in the six dependent scores and the ladder, and tighten tests/validation/test_fig6_author_c0_evidence.py:45 to assert the QP metric as well (it currently asserts only the concatenated one, < 1e-12). Observed values stay far inside the new limits (qp 6.58e-14, phonon 3.95e-16), so acceptance does not flip.
```

---

## `validation/fischer_2023/fig6_author_output_parity.py`

### P05 — Author-output extractor assigns a dashed-line sliver to the solid numerical trace (high, oracle-integrity)

Location: `validation/fischer_2023/fig6_author_output_parity.py:337`

**Why held back.** BEHAVIOUR-CHANGING, and uniquely so: this file's OWN source hash is pinned in the committed artifact. I verified that validation/paper_data/fischer_2023/fig6/author-output-score.json records `producer.sources["validation/fischer_2023/fig6_author_output_parity.py"].sha256` and `extraction.script_sha256` both = dae9f9e4078c535a8a05f46e54249413edc7cf8020c45f5233ef92c61ccbe2b3, which equals `source_sha256` of the file on disk today, and that tests/validation/test_fig6_author_output_parity.py:107-116 asserts that equality on EVERY run (it is not archive-gated). So ANY edit to this file — including a comment-only correction of the false invariant asserted at lines 341-342 — breaks an ungated test unless author-output-score.json is regenerated. Regeneration requires QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE and changes published numbers (9 of the 21 paper_numerical effective_uncertainty values, per the verifier). docs/PAPER-REPRODUCTION-LADDER.md:78 cites this score affirmatively. This is exactly the coupled physics-artifact change the recertification must not have mixed into a neutral batch, so I left the file byte-identical to c269af2.

**Expected numerical impact.** Extracted `paper_numerical` becomes smooth: max sample-to-sample jump 0.03503 / 0.03927 / 0.04666 -> 0.00040 / 0.00044 / 0.00051 for the 0.10 / 0.15 / 0.20 K curves. The 42 committed `author_output_y` values move by at most 4.9e-5, far inside the ~1.4e-3 recorded `author_raster_uncertainty`, so the accepted claim survives. But 9 of the 21 `paper_numerical` rows have their `effective_uncertainty` shrink 3.7x-9.3x (e.g. 0.10K x=0.2497: 0.024590 -> 0.005210; 0.20K x=0.4099: 0.051741 -> 0.005564), and their `uncertainty_normalized_error` rises correspondingly (0.10K x=0.2497: 0.0023 -> 0.0110). Every compared value and every per-curve maximum is unchanged. Currently 7 green / 11 blue / 9 red sample columns carry a swapped trace; today's committed `author_output_y` values happen to interpolate between uncorrupted samples, but changing only `sample_count_per_curve` (recorded as free metadata) to 170 already lands a corrupted sample inside a paper-point bracket, and the score still reports accepted_author_output_identity.

**Prepared patch.**

```
Module constants (replace GROUP_GAP_PIXELS = 20):

GROUP_GAP_PIXELS = 60
MINIMUM_GROUP_ROWS = 5
MINIMUM_TRACE_SEPARATION_PIXELS = 150.0
ACCEPTED_Y_VALUE_WINDOW = (0.08, 0.25)
MAXIMUM_SAMPLE_STEP = 0.005

In `extract_author_output`, replace

            groups = _groups_at_x(mask, x_pixel)
            if len(groups) < 2:
                raise ValueError(
                    f"Could not isolate two traces for {curve_id} at x={x_value:.6g}."
                )

with

            groups = [
                group
                for group in _groups_at_x(mask, x_pixel)
                if ACCEPTED_Y_VALUE_WINDOW[0]
                <= y_slope * float(np.median(group)) + y_intercept
                <= ACCEPTED_Y_VALUE_WINDOW[1]
            ]
            if len(groups) != 2:
                raise ValueError(
                    f"Could not isolate exactly two traces for {curve_id} at "
                    f"x={x_value:.6g}; found {len(groups)} inside the plot region."
                )
            separation = abs(
                float(np.median(groups[1])) - float(np.median(groups[0]))
            )
            if separation < MINIMUM_TRACE_SEPARATION_PIXELS:
                raise ValueError(
                    f"Traces for {curve_id} at x={x_value:.6g} are only "
                    f"{separation:.1f} px apart; below "
                    f"{MINIMUM_TRACE_SEPARATION_PIXELS:g} px the window cannot "
                    "distinguish a dash gap from a second trace."
                )

and add, after the `for x_value in SAMPLE_X:` loop closes (still inside the per-curve loop, before the `for curve_kind in medians:` packaging loop):

        for curve_kind, values in medians.items():
            steps = np.abs(np.diff(np.asarray(values, dtype=float)))
            if steps.size and float(steps.max()) > MAXIMUM_SAMPLE_STEP:
                raise ValueError(
                    f"Extracted {curve_kind} for {curve_id} jumps "
                    f"{float(steps.max()):.5g} between adjacent samples; the "
                    "sampling window is picking different traces."
                )

Rationale for each piece, from the verifier's measurements: the y-value window (mirroring `accepted_y_value_window` in validation/fischer_2023/extract_fig6_paper_data.py's oracle.json) excludes the legend colour swatches, which legitimately produce a third group at ~49 of the 171 columns — without it, the filer's `len(groups) == 2` would raise on a currently-correct extraction. GROUP_GAP_PIXELS 20 -> 60 merges dash pieces (dash gap ~25 px) while staying far below the 300-600 px true dashed-to-solid separation. The separation floor and the smoothness diagnostic are the positive checks the current design lacks; note the filer's proposed "separation > the larger group's full height" does NOT catch the defect (at green x=0.247 the two dashed slivers are 33.5 px apart with heights 11 and 6, so it passes).

Release procedure this fix REQUIRES, and the reason it cannot be batched with neutral edits: regenerate author-output-score.json via `build_score(Path(os.environ["QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE"]))` on the archive whose sha256 is 31d76c92...81bc1, re-pin `producer.sources` and `extraction.script_sha256`, and re-run tests/validation/test_fig6_author_output_parity.py with the archive present so the exact-replay test executes. Also worth adding: the verifier notes `score_curve`'s uncertainty envelope (validation/paper_parity.py:1185-1194) is built from the same corrupted curve, so normalized error saturates at 0.77-0.90 no matter how large the corruption — the 1.0 acceptance threshold is structurally blind to trace confusion and should not be relied on as the gate for this class of error.
```

---

## `validation/fischer_2023/fig6_paper.py`

### P15 — fig6_paper module docstring states Eq. 53 with qpsim's x_qp convention instead of the paper's (the code applies _XQP_QPSIM_TO_PAPER = 2.0)

Location: `validation/fischer_2023/fig6_paper.py:47`

**Why held back.** Two independent reasons. (1) docs/CODE-REVIEW-FALSE-POSITIVES.md item 25 records an explicit project decision covering this exact header: 'correcting that source wording is deliberately deferred because it participates in the newly promoted source fingerprints. Correct it in a coordinated provenance-breaking regeneration rather than immediately making the accepted bundles stale.' The verifier reached the same handling conclusion independently. (2) This file's own SHA-256 is inside artifact_fingerprint() -> source_manifest(Path(__file__), ...) (fig6_paper.py:775-799) and the manifest hashes whole-file canonical bytes (validation/source_provenance.py:26-28), so a docstring edit advances the promoted Fig-6 bundle digest — and I cannot rebind the stored digests in validation/baselines/ph0_kaplan/fischer_fig6_paper.promotion.json because they are outside my packet. Note for the orchestrator: the qpsim edits in this campaign already stale every artifact fingerprint (source_manifest globs qpsim/**/*.py and qpsim/**/*.yaml), so if a digest rebinding step is planned anyway, this patch can be folded into it at zero extra cost. Numerically the edit is inert — it changes no computed value.

**Expected numerical impact.** None on any computed value. It advances fig6_paper.py's source_sha256 inside artifact_fingerprint(), which stales the promoted Fig-6 bundle and any fingerprint equality check until the stored manifest is rebound. Acting on the WRONG reading of the current header (i.e. deleting _XQP_QPSIM_TO_PAPER) would instead map the stored ordinate obs -> (1+obs)/2, pushing the paper-window values from [0, 0.25] to 0.5-0.625 — the exact damage recorded in commit 3ad904e — and break the three dashed raster-oracle controls (0.388/0.200/0.253).

**Prepared patch.**

```
In validation/fischer_2023/fig6_paper.py, replace the two lines at 47-48:

-with qpsim's Fischer-convention $x_\\mathrm{qp}
-= n_\\mathrm{qp}/(4\\rho_F\\Delta_0)$. The bracketed factor is verified
+with the *paper's* prefactor convention $x_\\mathrm{qp}^\\mathrm{paper}
+= n_\\mathrm{qp}/(2\\rho_F\\Delta_0)$ — twice qpsim's Fischer-convention
+$x_\\mathrm{qp} = n_\\mathrm{qp}/(4\\rho_F\\Delta_0)$, so the Eq. 53 helpers
+apply that factor-2 conversion internally
+(``fig6_solve._XQP_QPSIM_TO_PAPER``). The bracketed factor is verified

Worth batching into the same edit (verifier's closing note; clears ledger item 25 in this header too), lines 28-33:

-The
-extracted $\\tau_0^{PB}$ diagnostic is pinned to the phonon-side
-F&C/Kaplan pair-breaking rate and reproduces the paper-quoted ~255 ps
-for the Table I parameters.
+The
+$\\tau_0^{PB}$ diagnostic inverts a sink assembled from the same
+phonon-side kernel it was handed, so it is a normalization round-trip on
+the input lifetime (constant ratio 0.9986, a discretization effect), not
+an independent extraction of the paper's ~255 ps.
```

---

## `validation/fischer_2023/fig6_q0_sweep.py`

### P08 — Emit a committable Q0 receipt (artifact digest + step_rtol + per-temperature max measured balance ratio) instead of printing the digest and discarding it

Location: `validation/fischer_2023/fig6_q0_sweep.py:393`

**Why held back.** Two reasons. (1) This file's canonicalized source SHA-256 is bound as Q0 evidence in validation/paper_data/fischer_2023/fig6/reproduction-ladder.json:536 (source_canonical b96d913d5ff21dd482033c900563328a702c03c9476722f3d3e5fb0b836c27b2 — I verified it still matches the live bytes exactly), and validation/reproduction_ladder.py:_verify_repository_binding re-hashes it under tests/validation/test_reproduction_ladder.py::test_checked_fig6_ladder_obeys_one_component_rule. ANY byte change here — including a comment — invalidates that binding, and reproduction-ladder.json is not in my packet. (2) More importantly, this patch does not actually close the finding. The verifier's operative defect is downstream: papers/qpsim/fig6-numerics/make_figures.py reads gitignored tmp/author-runs/*.json with no digest and no quality gate (its only filter is p.get('converged') at lines 87 and 177), and figure_refinement at line 224 is unguarded, so a fresh clone silently overwrites the tracked fig_refinement.pdf/.png with an empty figure and exits 0. Fixing that needs make_figures.py plus committed per-point records under validation/paper_data/fischer_2023/fig6/ — neither in my packet — and the source artifacts do not exist in this checkout (tmp/ is absent), so I cannot produce the data. Recommend routing the whole item to whoever owns papers/qpsim/fig6-numerics/ together with the ladder rebind.

**Expected numerical impact.** None. Writes an additional sidecar file and one extra stdout line; the sweep artifact bytes and every point are unchanged.

**Prepared patch.**

```
--- a/validation/fischer_2023/fig6_q0_sweep.py
+++ b/validation/fischer_2023/fig6_q0_sweep.py
@@ (after the existing output.write_text(...) block, replacing lines 393-395)
     digest = hashlib.sha256(output.read_bytes()).hexdigest()
+    # Repository trust anchor for an artifact that is retained outside the
+    # checkout (tmp/ is gitignored): commit this receipt alongside the C2-C7
+    # raw-manifest receipts so a reader can bind the bytes and the certificate
+    # quality the figures were built from.
+    worst = {}
+    for point in records:
+        if not point.get("converged"):
+            continue
+        key = repr(point["T_bath_K"])
+        worst[key] = max(
+            worst.get(key, 0.0),
+            float(point["measured_qp_balance_l1_ratio"]),
+            float(point["measured_ph_balance_l1_ratio"]),
+        )
+    receipt = output.with_name(output.name + ".receipt.json")
+    receipt.write_text(
+        json.dumps(
+            {
+                "artifact_path": output.name,
+                "artifact_sha256": digest,
+                "grid": artifact["grid"],
+                "indices": list(indices),
+                "max_measured_balance_ratio_by_T": worst,
+                "n_points": len(records),
+                "schema": SCHEMA,
+                "solver": artifact["solver"],
+                "sources": dict(_SOURCE_HASHES),
+                "temperatures_K": list(temperatures),
+            },
+            indent=2,
+            sort_keys=True,
+            allow_nan=False,
+        )
+        + "\n",
+        encoding="utf-8",
+    )
     print(f"{output} sha256={digest}")
+    print(f"{receipt} written")
     return output

Then (outside this packet): commit the receipt as validation/paper_data/fischer_2023/fig6/q0-raw-manifest-receipt.json, rebind reproduction-ladder.json's Q0 source_canonical hash, and make papers/qpsim/fig6-numerics/make_figures.py verify the digests and each point's recorded ratio against the caption's certificate, exiting nonzero on mismatch or on a missing artifact rather than writing an empty figure.
```

---

## `validation/fischer_2023/fig6_solve.py`

### P14 — Grid-consistent thermal reference for the promoted Fig. 6 ordinate (finding 5)

Location: `validation/fischer_2023/fig6_solve.py:881`

**Why held back.** BEHAVIOUR-CHANGING and explicitly provenance-breaking: it moves EVERY row of validation/baselines/ph0_kaplan/fischer_fig6_paper.csv, and the current validator would reject the corrected producer (fig6_paper.py:1150 `_require_close`s the stored delta_eq against a fresh CONTINUUM `calibrate_gap` at rtol 256*eps, while the grid-consistent value differs by 8.8e-6 ueV at 0.20 K). It also requires coordinated edits to two files outside my packet (validation/fischer_2023/fig6_paper.py:1138-1155 and validation/fischer_2023/test_fig6_paper.py:834-856) and a physicist decision the verifier flagged: variant (i) numerator-only vs variant (ii) both numerator and denominator — they differ by the 0.86%/0.45%/0.22% denominator change, and (ii) also shifts `obs_eq53`, which currently agrees with the digitized dashed Eq. 53 trace within raster uncertainty. Applied instead: the NEUTRAL half of the verifier's recommended handling — the convention and its zero-drive offsets are now documented in the module docstring and pinned by a regression test, so the choice is explicit rather than incidental. Two verifier notes to carry forward: do NOT include the filer's "Delta_0 from solve_gap at f=0" step (a 2.8e-14 ueV no-op), and confirm before regenerating that the discrete zero-drive kinetic fixed point really is the center-sampled f_FD.

**Expected numerical impact.** Restores an exact 0 at zero drive; removes a constant additive bias of +8.6192e-3 / +4.5428e-3 / +2.1883e-3 of deltaDelta_T at T_B = 0.10 / 0.15 / 0.20 K (values I reproduced independently on the real 1640-cell grid). Because the offset is additive it dominates at weak drive: verifier-recomputed shifts of -22.8% (T_B=0.10, weakest drive) to -0.87% (top of sweep), -6.70% at the promoted anchor (0.114844187603317 -> 0.107148517007253 at T_B=0.10), -2.20% at the T_B=0.20 anchor (0.090787795351478 -> 0.088793782922556). Direction moves qpsim AWAY from the digitized paper trace, so it slightly widens the known 33-39% mismatch; its real significance is that -2.0e-3 at the anchor is ~10x the ladder's formally staged "coherence only" component and is absent from the Q1 decomposition.

**Prepared patch.**

```
--- a/validation/fischer_2023/fig6_solve.py
+++ b/validation/fischer_2023/fig6_solve.py
@@ (in _solve_sweep, the non-direct-gap branch)
             calibration = calibrate_gap(
                 T_c=T_C,
                 T_bath=T_bath,
                 Delta_0=DELTA_0,
                 xtol=GAP_SOLVE_XTOL_UEV,
             )
-            delta_eq_per_T[i] = calibration.delta_eq
-            # deltaDelta_T = Delta_0 - Delta_eq(T_B), the thermal-equilibrium
-            # suppression at this T_B (independent of drive). Used as the
-            # denominator of the paper observable.
-            delta_T = DELTA_0 - calibration.delta_eq
+            # Take the thermal reference through the SAME finite-volume
+            # operator that produces Delta_driven, so the cell-constant
+            # representation error cancels and the zero-drive ordinate is
+            # exactly 0 (it is +8.6e-3 at T_B=0.10 K under the continuum
+            # reference).
+            delta_eq_grid = float(
+                solve_gap(
+                    calibration,
+                    fermi_dirac_occupation(spectral.E, T_bath),
+                    spectral.E,
+                    dE_bins=spectral.dE,
+                    reference_gap=calibration.delta_eq,
+                    xtol=GAP_SOLVE_XTOL_UEV,
+                )
+            )
+            delta_eq_per_T[i] = delta_eq_grid
+            delta_T = DELTA_0 - delta_eq_grid

Then re-pin validation/fischer_2023/fig6_paper.py:1138-1155 (compare the stored delta_eq against the grid-consistent value, not `calibrate_gap`) and validation/fischer_2023/test_fig6_paper.py:834-856 (the three continuum deltaDelta_T values 8.321535460709129e-8 / 1.0737232068436242e-4 / 4.019584551002708e-3), and regenerate the promoted CSV.
```

---

## `validation/fischer_2023/test_fig6_paper.py`

### P11 — Restate the paper_observable_num / paper_observable_eq53 gates against each row's own suppression scale

Location: `validation/fischer_2023/test_fig6_paper.py:2023`

**Why held back.** BEHAVIOUR-CHANGING: this rewrites a tolerance that currently gates acceptance of the pinned Fig.-6 baseline, and the change is a LOOSENING of up to 7 orders on the coldest row. Weakening a certification gate is exactly the kind of edit that hides drift during recertification. The verifier also ties it to open audit item N37 (docs/AUDIT-2026-07-15-numerical-software.md:157, high, still open), whose larger claim -- that the 66-point production sweep certifies a loose solver contract and not this observable -- is the reason the gate cannot simply be restated without adjudication. I applied only the comment corrections (see applied) so the false 'inherits via composition' claim is gone and the real amplification is on record. Note the filer's first option (a pure rtol gate) is unusable: paper_observable_num changes sign inside a single temperature row (0.10 K spans -2.304305e6 to +1.4126e-01), so rtol is vacuous near the zero crossing -- only the per-row-atol form below, or the producer-anchor identity, is sound.

**Expected numerical impact.** No shipped number changes -- this is a manual_slow test that is deselected by the default `-m 'not slow'` addopts and is currently unreachable in CI. Effect on the gate: paper_observable_num atol goes from a flat 1e-6 to 1e-6/deltaDelta_T per row, i.e. 1.2017e-06/8.3215e-08 = 1.2017e+01 at T_B = 0.10 K, 9.3134e-03 at 0.15 K, 2.4878e-04 at 0.20 K. That is a loosening of 1.2e7x / 9.3e3x / 2.5e2x respectively, and it makes the gate exactly as strict as the 1e-6 ueV gap gate it is paired with -- no stricter, no looser. Equivalent for eq53 requires scaling by dDelta_drive/deltaDelta_T^2 (3.9838e+13 per ueV at 0.10 K, 2.3929e+07 at 0.15 K, 1.7076e+04 at 0.20 K).

**Prepared patch.**

```
--- a/validation/fischer_2023/test_fig6_paper.py
+++ b/validation/fischer_2023/test_fig6_paper.py
@@ in test_matches_pinned_baseline
-    np.testing.assert_allclose(
-        result.paper_observable_num, baseline.paper_observable_num,
-        rtol=0.0, atol=1e-6,
-        err_msg="(δΔ_T - δΔ)/δΔ_T numerical drift",
-    )
+    # The observable is (Δ_driven - Δ_eq)/(Δ_0 - Δ_eq): a 1/δΔ_T amplification of the
+    # gap columns gated 1e-6 abs above.  Carry that same physical tolerance
+    # through per temperature row instead of re-asserting it absolutely on the
+    # amplified ratio.  A pure rtol is unusable here -- the row changes sign.
+    for i in range(len(baseline.delta_eq)):
+        delta_T = fig6_solve.DELTA_0 - float(baseline.delta_eq[i])
+        np.testing.assert_allclose(
+            result.paper_observable_num[i], baseline.paper_observable_num[i],
+            rtol=0.0, atol=1e-6 / delta_T,
+            err_msg=f"(δΔ_T - δΔ)/δΔ_T numerical drift, T_B row {i}",
+        )
@@
-    np.testing.assert_allclose(
-        result.paper_observable_eq53, baseline.paper_observable_eq53,
-        rtol=0.0, atol=1e-6,
-        err_msg="Eq. 53 dashed-overlay drift",
-    )
+    # obs_eq53 = 1 - ΔΔ_drive/δΔ_T, so its Δ_eq sensitivity is ΔΔ_drive/δΔ_T².
+    for i in range(len(baseline.delta_eq)):
+        delta_T = fig6_solve.DELTA_0 - float(baseline.delta_eq[i])
+        drive = delta_T * float(
+            np.max(np.abs(1.0 - baseline.paper_observable_eq53[i]))
+        )
+        np.testing.assert_allclose(
+            result.paper_observable_eq53[i], baseline.paper_observable_eq53[i],
+            rtol=0.0, atol=1e-6 * drive / delta_T**2,
+            err_msg=f"Eq. 53 dashed-overlay drift, T_B row {i}",
+        )

Stronger alternative worth considering during the N37 adjudication (the verifier's
preference, and I confirmed the premise: recomputing (delta_driven - delta_eq)/(180 -
delta_eq) from the two pinned gap columns reproduces the stored paper_observable_num
column bit-exactly over all 66 rows) -- gate delta_eq/delta_driven at their physical
tolerance and assert the derived ratio identity to roundoff against the producer
anchor, mirroring the reader-side rebuild already used at
validation/fischer_2023/fig6_paper.py:1166-1213.
```

---

## `validation/fischer_2024/fig8_paper.py`

### P08 — F24 Fig 8 solve gate NEWTON_BACKWARD_ERROR_TOL 1e-6 -> 1e-7 (plus artifact re-promotion)

Location: `validation/fischer_2024/fig8_paper.py:148`

**Why held back.** BEHAVIOUR-CHANGING and outside my packet. The verifier reproduced that at T_bath=0.2545454545454546 K / drive=1e-2 Hz the 1e-6 gate stops one Newton step short: certificate 7.331251e-07 (bit-matching the promoted maximum) and x_qp=8.01727591989760425e-05, versus certificate 2.689157e-13 and x_qp=8.01727005353934245e-05 at 1e-7. Tightening therefore moves the published x_qp payload by 7.32e-07 relative — 73% of the rtol=1e-6 pinned-baseline budget at test_fig8_paper.py:347/354 — so certified_payload_sha256, the promotion record and the companion PDF binding all move. It is an artifact re-promotion requiring a full 36-point run() (12 temperatures x 3 drives under strong-to-weak continuation) to confirm 1e-7 converges everywhere, and a physicist/owner sign-off. I also did NOT parameterize TARGET_QP_BACKWARD_ERROR_LIMIT / TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT (see skipped) and did NOT touch fig8_xqp_pb.py or figs_5_7_fe_pb.py, whose gates never bind (maxima 1.2e-11 and 4.5e-13). The invariant is now recorded in a comment at validation/fischer_2024/_artifact.py:53.

**Expected numerical impact.** x_qp payload shifts by ~7.3e-07 relative at the three highest-T points (8.01727591989760425e-05 -> 8.01727005353934245e-05 at T=0.2545 K, drive=1e-2 Hz); certificate_max_qp_number_backward_error 7.331250545662921e-07 -> ~2.7e-13; certificate_max_qp_backward_error 3.30338642886604e-07 -> ~3e-14. Solve cost +~0.2 s/point. No physical curve changes beyond the 7th significant figure.

**Prepared patch.**

```
--- a/validation/fischer_2024/fig8_paper.py
+++ b/validation/fischer_2024/fig8_paper.py
@@ -148 +148 @@
-NEWTON_BACKWARD_ERROR_TOL = 1.0e-6
+# One decade below TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT so the solve does
+# not stop on the first iterate under the acceptance gate: at 1e-6 the three
+# highest-T points halt one Newton step short of the tight root (certificate
+# 7.33e-7 vs 2.69e-13), which is the same discrete stopping branch that made
+# hosted Linux and Windows disagree for Fig. 5 (see fig5_paper.py:165-170).
+NEWTON_BACKWARD_ERROR_TOL = 1.0e-7

Then: regenerate validation/baselines/ph0_constant/fischer2024_fig8_qpsim_native.csv via the driver's run(), re-promote (payload digest + promotion record + PDF binding), and re-pin test_fig8_paper.py::test_matches_pinned_baseline.
```

---

## `validation/marchegiani_2025/fig3_paper.py`

### P09 — Local `_H_OVER_KB = 4.799243e-11` literal instead of qpsim.constants.H_OVER_KB_K_PER_HZ

Location: `validation/marchegiani_2025/fig3_paper.py:103`

**Why held back.** BEHAVIOUR-CHANGING twice over, and the verifier explicitly says to defer this file to a scheduled re-certification. (1) Numerical: the literal is 1.521e-8 relative below the derived constant, and it feeds Delta_R_K and omega_10_K at fig3_paper.py:488, 491, 609 -- swapping it shifts every published M25 Fig. 3 Kelvin input, and the M25 rate coefficients contain a bare exp(-2*Delta_R/T) (rate_equation_coefficients.py:777, 791) that amplifies it. (2) Provenance: fig3_paper.py hashes its own source bytes into the published bundle (source_fingerprint, imported at line 75, used at line 244), so even a comment-only edit invalidates the manifest-authenticated M25 artifacts and forces a full regeneration -- which b3cd161 has just shown is environment-sensitive (a regen on numpy 2.4.6 drifted tail bins by 2.7e-2 and had to be discarded). The genuinely free part of this finding lives in files I do not own: the 7 test sites carrying the same literal (tests/devices/test_m25_junction.py:58, tests/services/test_rate_equation_branch.py:47, test_rate_equation_coefficients.py:31, test_rate_equation_note_v.py:24, plus validation/marchegiani_2025/fig4_paper.py:107, fig3_chemical_potentials.py:79 and the inline literal at validation/marchegiani_2025/test_fig4_paper.py:147 the filer missed), the two Fischer KB_UEV_PER_K re-declarations, and qpsim/constants.py itself.

**Expected numerical impact.** All Kelvin-built M25 inputs shift by +1.521e-8 relative. Divergence in the shared exp(-2*Delta_R/T) factor between the webui path and the validation driver: 7.15e-6 at T_MIN_K = 0.010, 1.43e-6 at T = 0.050, 4.77e-7 at T = 0.150 (the filer's headline 1.4e-6 understates it ~5x by quoting mid-sweep). That is ~140x inside the 1e-3 residual gate, so no gate flips -- but every certified M25 row and bundle manifest moves and must be regenerated and re-pinned.

**Prepared patch.**

```
In validation/marchegiani_2025/fig3_paper.py, delete line 103 and import the shared constant instead:

-_H_OVER_KB = 4.799243e-11   # K / Hz
+from qpsim.constants import H_OVER_KB_K_PER_HZ as _H_OVER_KB  # K / Hz

(placed with the other absolute imports; the three use sites at 488, 491, 609 are unchanged).

Paired change in qpsim/constants.py (NOT in this packet -- do NOT extend HBAR_UEV_NS, which would perturb Fischer Figs 9/13 and the prelim campaign for no M25 benefit; define h/k_B directly from the SI-exact values instead):

-# h/k_B in K/Hz -- the GHz<->Kelvin conversion the M25 layer runs on.
-# Derived from the two constants above ... the literal several M25
-# validation modules and tests carry locally.
-H_OVER_KB_K_PER_HZ = 2.0 * math.pi * HBAR_UEV_NS / KB_UEV_PER_K * 1e-9
+# h/k_B in K/Hz -- the GHz<->Kelvin conversion the M25 layer runs on.
+# Built from the SI-exact defining values (h = 6.62607015e-34 J*s,
+# k_B = 1.380649e-23 J/K) so it is exact to binary64 rather than
+# inheriting the truncation in HBAR_UEV_NS.
+H_OVER_KB_K_PER_HZ = 6.62607015e-34 / 1.380649e-23

Sequence this as a deliberate M25 re-certification: land the test-side and constants-side changes first, then regenerate and re-pin the M25 tables on the recorded generator environment. A guard test asserting no k_B or h/k_B literal outside qpsim/constants.py should carry the three certified drivers on an explicit, comment-justified allowlist until then.
```

### P09 — Fig. 3 inset gray guide line: unit conversion multiplied then divided out, so the plotted reference is 99x (panel a) / 10.8x (panel b) too steep

Location: `validation/marchegiani_2025/fig3_paper.py:559`

**Why held back.** Two independent reasons. (1) The correct replacement is NOT established -- the verifier's correction is explicit ("Do not merge the filed one-liner as-is"). The source comment is self-inconsistent: "traces omega_LR/Delta_L on the right axis" describes a CONSTANT (a horizontal Delta_mu = omega_LR marker, which is also what the ymax headroom of 1.2*omega_LR/Delta_L at lines 569-572 sizes for), while "a unit-slope reference in (Delta_mu/omega_LR) units" describes a diagonal; and both the coded expression and the filed replacement plot a Kelvin number on a dimensionless Delta_mu/Delta_L axis. Choosing requires consulting M25 Fig. 3 itself. (2) Any edit here re-renders the manifest-authenticated m25_fig3_paper.pdf and changes the driver's source_fingerprint (line 75/244), so it carries the same forced M25 regeneration cost as the constants item -- for a decoration inside a stage whose recorded contract (docs/Validation_Chain.md:424) is "manual broad paper anchors only".

**Expected numerical impact.** Zero on every CSV column, certificate, residual and gate -- rendering only. Measured from the published PDF's content stream: the stroke's data-space slope is 1.000 in both panels versus the intended 0.010101 (panel a, 99.0x too steep, exits the inset at 5.8% of its width -- reads as a near-vertical stroke) and 0.092593 (panel b, 10.8x, exits at 44.5%). Re-rendering changes the artifact's bytes and payload hash, so the M25 bundle must be regenerated and re-pinned.

**Prepared patch.**

```
Preferred reading (horizontal Delta_mu = omega_LR level; matches the comment's first clause, the reserved headroom, and the physics the inset exists to show -- the curve crosses the level in panel a, 0.01210 vs 0.01010, and stays below it in panel b, 0.08547 vs 0.09259):

     # Diagonal gray line in the paper-figure inset traces omega_LR/Delta_L on
     # the right axis -- i.e. a unit-slope reference in
     # (Delta_mu/omega_LR) units. We render it as a faint guide.
-    inset_ax.plot(
-        [T_inset_lo, T_inset_hi],
-        [
-            T_inset_lo * (panel.omega_LR_GHz / Delta_L_GHz)
-            * (1.0 / max(panel.omega_LR_GHz / Delta_L_GHz, 1e-30)),
-            T_inset_hi * (panel.omega_LR_GHz / Delta_L_GHz)
-            * (1.0 / max(panel.omega_LR_GHz / Delta_L_GHz, 1e-30)),
-        ],
-        color="gray", lw=2.0, alpha=0.6, zorder=2,
-    )
+    # Gray guide: the Delta_mu = omega_LR level, i.e. 1 on the right-hand
+    # (Delta_mu/omega_LR) axis. The former expression multiplied by
+    # omega_LR/Delta_L and immediately divided it out again, leaving
+    # y = T -- a Kelvin value on the dimensionless Delta_mu/Delta_L axis.
+    inset_ax.axhline(
+        panel.omega_LR_GHz / Delta_L_GHz,
+        color="gray", lw=2.0, alpha=0.6, zorder=2,
+    )

If M25 Fig. 3 instead shows a diagonal, use a dimensionless one -- e.g. Delta_mu = k_B*T, y = T / Delta_L_K (slope 0.4209 per K for panel a) -- not the filed y = T*(omega_LR/Delta_L), which is still dimensionally incoherent. Whichever is chosen, correct the comment at lines 553-555 at the same time and regenerate the M25 bundle.
```

---

## Findings judged wrong, stale, or already fixed (39)

Recorded so a future review does not re-file them without new evidence.

- **`.gitattributes:1`** — The actual fix for the paper-oracle CRLF breakage (pin *.py and paper_data CSVs to LF)  
  *P13* — Not in my packet - I may not edit .gitattributes. What I could do inside validation/paper_parity.py (my file) is make the failure self-explanatory, which I did. The real remedy, per the verifier, is two lines in the root .gitattributes: `*.py text eol=lf` (fixes the digitizer digest at paper_parity.py:517) and `validation/paper_data/**/*.csv text eol=lf` (mandatory for the SECOND, independent brea
- **`docs/Phonon_Escape_Time.md:260`** — Reword the 'shall be rejected at PhononState construction time' specification  
  *P13* — Not in my packet (docs/ files are not listed in P13). The verifier's preferred fix is doc-side: docs/Phonon_Escape_Time.md:221 and :260 and docs/Phonon_Model_Decisions.md:160-161 should be marked as an unimplemented roadmap item and should drop the 'at PhononState construction time' siting, because that constructor structurally cannot see the QP-side tau_0. I recorded the same fact from the code s
- **`docs/STATUS.md:301`** — docs/STATUS.md:301 repeats the same fabricated 'via weighted BE fit per F23 Eq. 36' provenance  
  *P11* — Not in my packet. The effective_temperature.py finding's fix explicitly calls for correcting docs/STATUS.md:301 in the same change, but that file belongs to another agent (or nobody), and the ABSOLUTE RULES forbid touching it. The line still reads '... via weighted BE fit per F23 Eq. 36'; with the module docstring now corrected, STATUS.md is the last surviving copy of the bad citation. Needs a one
- **`qpsim/backends/t3_spatial_1d.py:1275`** — Add a plausibility band to the interface_conductance validator  
  *P05* — The verifier explicitly withdrew this part of the proposed fix: it is largely redundant with the `_MAX_CN_SUBSTEPS` guard (line 67 / 601-608), which already rejects the whole SI-magnitude band loudly at operator-build time, and a band keyed to D0/dx would have to reject legitimately near-transparent interfaces used to approach the Robin limit. The doc/symbol change is the whole fix.
- **`qpsim/backends/t3_spatial_1d.py:353`** — Bound `_transport_cn_cache` like `_collision_cache`  
  *P05* — Adjacent observation in the verifier's notes, not a filed finding. `_transport_cn_cache` is an unbounded plain dict while `_collision_cache` is a size-bounded OrderedDict LRU. Harmless for a fixed-dt fixed-gap run (one entry) and my hoist does not change how many entries are created. Converting it to an LRU could evict an entry a long sweep still wants, which is a behaviour change in wall time and
- **`qpsim/backends/t3_spatial_1d.py:1174`** — The sibling spatial driver carries the identical linear-interpolation defect, plus a mislabeled residual at in  
  *P15* — OUT OF PACKET. Same `prev + fraction*(cur - prev)` pattern at t3_spatial_1d.py:1174-1181, with the same docstring claim at :1089-1091 that 'dt > snapshot_interval does not drop cadence points or mislabel endpoint states'. The verifier additionally found (correction 3(ii)) that the residual recorded with every interior interpolated snapshot there is the max_rate computed from the END-of-step state 
- **`qpsim/collisions/_uniform_grid.py:73`** — Identical non-atomic read-then-pop eviction in the _VALIDATED_GRIDS cache  
  *P01* — Not in packet P01. This is the reachable half of the eviction-race finding (the webui's sync handlers run on the anyio thread pool and reach uniform_grid_spacing through validate_setup, with a guaranteed miss per request), so whoever owns qpsim/collisions/_uniform_grid.py should apply the same `pop(next(iter(cache)), ...)` tolerance I applied in phonon.py at lines 73-75.
- **`qpsim/collisions/_uniform_grid.py:7`** — Module comment claimed "a recycled id with different content misses the guards", which was false for interior-  
  *P07* — Not in my packet, and already fixed by its owner in this working tree: the memo is now keyed on (E.tobytes(), dE.tobytes()) and the comment has been rewritten to say a stale hit is impossible. My two new tests in tests/collisions/test_sub_gap_photon.py lock that behaviour in (the in-place-mutation one fails against the old id-keyed implementation and passes against the new one).
- **`qpsim/collisions/phonon.py:875`** — 'phonon_collision_jacobian_nph uses the uncorrected kernel' treated as a separate Jacobian defect  
  *P01* — Not a defect. Both verifiers confirmed phonon_collision_jacobian_nph correctly differentiates the QP residual as written, and phonon_source_sink_jacobian_f correctly mirrors the corrected phonon residual; each Jacobian is self-consistent with its own residual, so Newton convergence is unaffected. Fixing this line in isolation would introduce a genuine Jacobian error. It moves only if the held-back
- **`qpsim/collisions/phonon.py:485`** — 'The 3-element kernel fingerprint is degenerate on real kernels' (entries 0 and 2 bit-identical)  
  *P01* — Refuted by the verifier: the claim is grid-specific (the standard grid gives three distinct samples) and immaterial, because the correction is exactly invariant to the kernel prefactor (4.4e-16 across a 68x tau_0^PB change), so two canonical kernels SHOULD alias. I did not act on that framing; the underlying key-completeness problem is fixed by the weakref identity guards plus the full-array conte
- **`qpsim/collisions/phonon.py:485`** — Content-hash the kernel and omega map (blake2b of .tobytes()) instead of a fingerprint  
  *P01* — Measured and rejected on cost: on the NE=1620 production grid blake2b over both arrays is 131 ms per call against a 446 ms impl, i.e. it would return only ~70% of the memo's benefit on a path the Picard loop hits every iteration. The chosen fingerprint (three sampled entries + float(K.sum()) + omega_idx_sum shape/sum) touches every entry of both arrays for 11.7 ms per call (97.4% of the benefit re
- **`qpsim/collisions/phonon.py:169`** — "Distinct from the Kaplan tau_0^{ph} (Eq. 30)" is the one outright false physics statement in the Al.yaml find  
  *P15* — OUT OF PACKET. The verifier relocated this finding: the genuinely wrong statements are qpsim/collisions/phonon.py:169-170 (asserts tau_0^PB is 'Distinct from the Kaplan tau_0^{ph}' when the derivation printed ten lines above it in the same docstring shows the two rate expressions are identical under u = E/Delta) and qpsim/materials/database.py:78-79 ('For Al, F&C 2023 quotes 0.255 ns from the para
- **`qpsim/devices/qubit.py:64`** — Delete omega_kelvin / E_J_kelvin / E_C_kelvin from Qubit, or add a junction cross-check (the finding's fixes (  
  *P05* — Both structural options are out of scope for a neutral batch and are wrong as filed. (a) Deleting three public dataclass fields is an API break: `Qubit(omega_kelvin=...)` is constructed that way in tests/devices/test_m25_junction.py:750-755 and elsewhere outside my packet, and the fields are also documented in docs/Device_Architecture.md:466-473 (not mine). (b) The verifier showed the cross-check 
- **`qpsim/materials/database.py:67`** — Delete tau_s/tau_r from the dataclass, __post_init__, required_positive and _MATERIAL_FLOAT_FIELDS (the verifi  
  *P08* — Not implementable inside this packet: deletion requires editing tests/materials/test_database.py (lines 16-18, 49-52, 115 assert the fields and their defaulting), docs/Material_Database.md:28 and docs/REVIEW-2026-07-02-code-health.md:31, none of which I own. Applied the verifier's stated acceptable interim instead (accurate field comments plus a warning rather than the filer's raise, which would b
- **`qpsim/observables/gap_suppression.py:73`** — compute_gap_suppression compares a continuum delta_eq against a grid-solved delta_final  
  *P12* — File not in my packet, and behaviour-changing (it moves published rel_suppression values). Recorded in held_back as the cheaper alternative to the resolution-guard change; it is the actual user-facing exposure of the near-T_c grid bias (sign-flipped 2.4% spurious enhancement on a purely thermal state at T/T_c=0.83, dE=0.225*Delta).
- **`qpsim/physics/spectral.py:262`** — Alternative fix: memoize the validated spacing on SpectralContext (verifier-preferred option (b) in several co  
  *P02* — spectral.py is not in packet P02. It is also strictly weaker on its own: it does not cover the raw-array public entry point `validate_pair_breaking_photon_grid` (pair_breaking_photon.py:89), which is the 100%-deterministic bypass the verifiers reproduced. The content key covers both paths inside the one file I own. The weakref variant was also rejected: `ctx.E` returns a fresh view per access, so 
- **`qpsim/physics/spectral.py:452`** — Rewrite the LEGACY D(E) closure as D0*sqrt((E-gap)*(E+gap))/E (part of finding 1)  
  *P14* — The verifier explicitly instructed against this sub-item and I confirmed the reasoning by reading the code. (1) The proposed replacement is actively defective: it divides by E, and at E = 0 with gap > 0 the clamped numerator gives 0/0 = NaN, whereas the current `d_ratio = np.minimum(gap / E, 1.0)` maps inf -> 1 and correctly returns 0. That NaN would propagate into `self._D_E` (committed at spectr
- **`qpsim/services/rate_equation.py:730`** — Half B of the Part III finding: inline comment says the "no further improvement possible" status is accepted w  
  *P07* — Not in my packet — rate_equation.py is owned by another agent and is already modified in the working tree. I fixed the doc half (which the verifier identified as the half carrying the severity and the leverage) and left the source comment alone. For that owner: NO_PROGRESS_MARKER = "iteration is not making good progress" matches hybrd info 4/5 only; info 3's message is "xtol=... is too small, no f
- **`qpsim/services/rate_equation_coefficients.py:116`** — Give tests/services/test_rate_equation_coefficients.py::_fig3a_params the Note-V 01/10 values, or annotate the  
  *P08* — The test file is not in my packet. Also, per the verifier, changing the fixture values would move p_1 by 463x at 20 mK and could disturb the x_Rlt > x_L > x_Rgt ordering assertions those tests exist to protect. The docstring now carries the warning the fixture's reader needs, and the new packet test pins the true 01/10 magnitudes.
- **`qpsim/services/rate_equation_coefficients.py:1102`** — Fix the four vacuous thermal-generation gates in tests/services/test_rate_equation_coefficients.py (lines 363,  
  *P08* — That test file is not in my packet. Those assertions recover g_pn by the same subtraction and pass only on pytest.approx's default abs=1e-12, so the _g_pn_L/_g_pn_Rlt/_g_pn_Rgt closed forms have no effective coverage. Covered instead by tests/review_2026_08_03/test_P08.py::TestThermalGenerationFields, which asserts against _g_pn_*(params) with abs=0 both with and without loud scalar g_ph_* inputs;
- **`qpsim/services/rate_equation_coefficients.py:1102`** — Same lossy g_ph subtraction in PRODUCTION (coefficients_from_physical_parameters_with_photon_drive)  
  *P12* — File not in my packet - I may not edit it. This is the verifier's item 2 on the medium finding and is more important than the test defect I did fix: lines 1102-1104 rebuild the thermal-only scalars as `g_L = base.g_L - params.g_ph_L_Hz` (and _Rgt/_Rlt), so any caller that passes scalar g_ph_*_Hz into coefficients_from_physical_parameters and then builds a photon-drive bundle loses the entire therm
- **`qpsim/solvers/newton_steady_state.py:654`** — Make `_is_exact_absorbing_vacuum` key off the resolved `N_abs` instead of the raw `N_abs_override` presence fl  
  *P03* — Sub-recommendation attached to the N_abs_override finding; unnecessary after the fix and behaviour-changing if applied literally. The override/resolved divergence it targets (branching on `N_abs_override is not None` at line 654 while reading the substituted thermal `N_abs` at line 659) is only reachable when a lone `N_abs_override` is silently replaced, which the pairing guard now rejects at the 
- **`qpsim/solvers/newton_steady_state.py:718`** — Leave overflowing/subnormal-turnover rows unscaled inside `_row_scaled_newton_system` rather than raising  
  *P03* — Rejected in favour of the handler fix after weighing the two verifiers against each other. It is a third, untested linear system: it reintroduces exactly the cross-row scale disparity the function's docstring at lines 697-699 says is deliberately avoided ('the Fischer Fig. 7 cold point has legitimate row scales spanning ~1e142'), and neither verifier's control experiment exercised it -- both monke
- **`qpsim/solvers/newton_steady_state.py:0`** — Suggested regression test (3): pin _jacobian_analytical by hashing it over a sweep of (NE, support pattern, m,  
  *P07* — Out of packet and actively hazardous this round: newton_steady_state.py is being edited by another agent in this same working tree, and a digest pin is a behaviour lock, not a correctness test — it would either fight their diff or freeze an unreviewed Jacobian. Should be filed as follow-up work once the round is committed.
- **`qpsim/solvers/spectral_flow_tvd.py:26`** — De-export advect_spectral_flow/ssprk22_step and annotate docs/Part_III_Numerics.md:79-84  
  *P12* — Both remaining halves of that finding land outside my packet (qpsim/solvers/__init__.py:35,41 and docs/Part_III_Numerics.md). The verifier also explicitly recommended AGAINST dropping the symbols from __all__ ('a public-API break traded for no safety gain' that would orphan the tests). The doc note at Part_III_Numerics.md:79 (and :82-84, which advertises ssprk.py for a use that exists nowhere in t
- **`qpsim/webui/builders.py:454`** — webui maps newton_tol onto coupled_newton_tol without setting coupled_newton_step_rtol, so the displayed toler  
  *P11* — Not in my packet (builders.py is owned by another agent and is already modified in the working tree). Flagged by the verifier as the strongest instance of the `tol`-is-inert finding and as code rather than docs: builders.py sets kwargs['coupled_newton_tol'] = s.newton_tol but never sets coupled_newton_step_rtol, so the backend default 1e-8 (t3_diffusion.py:536) applies and the user-facing toleranc
- **`tests/backends/test_t3_diffusion.py:242`** — Tighten the n_ph vs Bose-Einstein gate (atol=1e-10, 63 of 89 omega bins unconstrained)  
  *P16* — Followed the verifier's explicit counsel to leave line 242 alone. I measured the achieved deviation here as exactly 0.0 (max abs and max rel), so an rtol gate would pass today, but it buys nothing physically - the assertion at line 237 already rejects an all-zero n_ph, the thermal n_ph is produced by the same expm1 path being compared against, and a bitwise-0 gate is brittle if that path is ever r
- **`tests/collisions/test_pair_breaking_photon.py:197`** — Suggested regression test (2): assert _pair_breaking_quadrature_correction returns different arrays for two di  
  *P07* — Out of packet: neither qpsim/collisions/pair_breaking_photon.py nor its test file is mine (both are modified by other agents right now). Asserting a specific cache behaviour there could contradict the fix its owner is landing.
- **`tests/collisions/test_phonon.py:259`** — Comment at :259 repeats the '2/pi underestimate' generalization; suggested relaxation of the assert at :282  
  *P01* — File not in packet P01, so I did not touch it. The verifier is also right that the assert `correction[threshold_idx] > 1.0` must NOT be relaxed - it is a valid, non-vacuous pin for the centre-aligned grid (measured pi/2). Only the comment's over-generalization needs scoping. I added the complementary face-aligned case (correction < 1, -> pi/4) in tests/review_2026_08_03/test_P01.py instead.
- **`tests/conftest.py:1`** — Autouse conftest fixture draining _VALIDATED_GRIDS between tests (sub-recommendation of finding #1)  
  *P02* — tests/conftest.py is not in packet P02, so I must not touch it. It is also no longer needed for correctness: with a content-addressed key a surviving entry can only be hit by byte-identical E/dE, whose validation outcome is identical, so cross-test/cross-request contamination is impossible. My own test module installs a local autouse `_hermetic_cache` fixture calling the new `_clear_validated_grid
- **`tests/solvers/test_newton_steady_state.py:161`** — Byte-level pinning of the Jacobian (`J.tobytes() == ref.tobytes()` against a checked-in reference)  
  *P10* — Deliberately not done, following the verifier's correction that the priority is 'the first EXECUTING test of the PB Jacobian block'. A checked-in byte reference pins BLAS/numpy-version-specific low bits (this repo already has provenance trouble from regenerating certified rows on a different numpy), would go red on any legitimate reassociation, and cannot distinguish a correct rewrite from a regre
- **`tests/solvers/test_newton_steady_state.py:161`** — First unit tests for `uniform_grid_spacing` and the `_VALIDATED_GRIDS` / `_PB_QUAD_CORR_CACHE` module caches (  
  *P10* — Out of packet scope. `uniform_grid_spacing` and `_VALIDATED_GRIDS` live in qpsim/collisions/_uniform_grid.py and `_PB_QUAD_CORR_CACHE` in qpsim/collisions/phonon.py - both owned by other packets and both currently modified in the working tree. A cache-identity/invalidation test belongs in tests/collisions/ next to the implementation (also another packet's file), and writing it into my P10 file wou
- **`validation/diffusion_operators/interface_trap.py:101`** — Add a second closed-relaxation measurement at a low energy bin (ed = 0)  
  *P12* — Requires editing the benchmark module, which is not in my packet: run() populates and stores only row `ed`, so a low-energy relaxation cannot be reached from the test file without reimplementing the benchmark. Also note the verifier's warning that `ed` must NOT simply be moved to 0 - it is shared with the driven KL benchmark, whose 1e-8 flux-continuity certificate and scipy-quad weight sit at the 
- **`validation/fischer_2023/extract_fig6_paper_data.py:452`** — Raw-byte compare of re-extracted points is not CRLF-portable  
  *P16* — Applied, then reverted - see held_back. Editing this file at all breaks manifest.extraction.script_sha256 (verified: the file's canonical-LF digest is exactly the recorded 488646f1ffe2a7cdcabfdf8e5483759c5a466a8ad619de869380d1eff16be6e5, and I confirmed it is restored byte-for-byte after the revert; the file no longer appears in git diff). The fix belongs with .gitattributes + validation/paper_par
- **`validation/fischer_2023/fig6_q0_sweep.py:393`** — Correct main.tex's 'Every number in this draft is traceable to a committed, hash-bound evidence artifact' head  
  *P08* — papers/qpsim/fig6-numerics/main.tex and docs/PAPER-REPRODUCTION-LADDER.md are not in my packet. Flagging for the orchestrator: the verifier established the header claim is false for four of the five refinement columns, the entire centers row, and both Q0 figures (only 0.14542377851441587 is committed, in c7-nonlinear-solver-score.json), and that the ladder JSON says Q0 is 'runnable' while the doc 
- **`validation/fischer_2023/figs_9_13_qi_vs_pread.py:1`** — Rename the module and the baseline artifacts away from the fabricated 'figs_9_13' identifier  
  *P06* — Out of packet scope and not a code-review fix. The verifier established the paper has no Figs. 9-13 (Fischer & Catelani 2023 ends at Fig. 8), but the corrective rename would touch validation/baselines/ph0_constant/fischer_figs_9_13_qi_vs_pread.{csv,pdf}, validation/fischer_2023/rasterize_baselines.py:31, validation/test_canonical_baselines_exist.py:80-81, validation/fischer_2023/test_figs_9_13_qi_
- **`validation/fischer_2023/figs_9_13_qi_vs_pread.py:388`** — Drop the warm start (n_bar_warm) now that the docstring rationale for it is known to be false  
  *P06* — BEHAVIOUR-CHANGING and not asked for. The verifier showed a cold start would converge in one iteration at every point, whereas the warm start costs a second pass at 20 of 21 points - but the `iterations` column (1,2,2,...,2) is persisted data in the artifact and re-checked by _validated_result and read_baseline, and assert_array_equal(result.iterations, baseline.iterations) is part of the slow com
- **`validation/fischer_2024/_artifact.py:775`** — Make TARGET_QP_BACKWARD_ERROR_LIMIT / TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT caller-parameterizable like residu  
  *P08* — Rejected on the verifier's reasoning, which I confirmed by reading the code: both constants are written into the artifact metadata (_artifact.py:924-927 and 1062-1065) and re-checked on read (1061-1071) precisely so a promoted artifact advertises one repo-wide acceptance limit a later reader cannot silently loosen. Parameterizing them would let a driver widen its own acceptance gate — the opposite
- **`validation/test_canonical_baselines_exist.py:96`** — Strengthen test_quarantined_archive_is_preserved from existence-only to a content assertion (pytest.raises(Leg  
  *P06* — File is not in packet P06 - another agent owns it. The verifier recommended this as the second half of the figs_9_13 hardening (it converts an out-of-band clobber by ANY route, not just write_baseline, into an immediate fast-suite failure, and closes the prior audit's un-filed near-miss at audit-2026-07-19-findings.json:1702). I covered the same invariant from inside my own packet instead: tests/r

---

## Addendum — newly added fail-closed guards (contract changes, applied)

Recorded after an independent whole-diff review of `a47bad3` observed that six
packets independently added or tightened guards while the commit described
itself as "behaviour-neutral", and the ledger recorded none of them. The
numerical claim in that commit stands — 35 engine surfaces bit-identical,
including four full Newton solves — but "behaviour-neutral" is precise only
about *computed values*. These **13 guards change the error contract**: input
that previously computed a number now raises.

| file | guard |
|---|---|
| `qpsim/solvers/newton_steady_state.py` | 4 — effective-kernel/occupation override shape, dtype, finiteness, pairing |
| `qpsim/collisions/phonon.py` | 2 — `K_*_eff_override` shape and emit/abs pairing |
| `qpsim/backends/t3_diffusion.py` | 1 — stranded gap-edge mass on remap |
| `qpsim/devices/device.py` | 1 — regions-key validation |
| `qpsim/devices/m25_junction.py` | 1 — gap/ω_LR consistency |
| `qpsim/materials/database.py` | 1 — `rho_F` magnitude window |
| `qpsim/observables/density.py` | 1 — Dynes sub-gap grid coverage |
| `qpsim/observables/gap_suppression.py` | 1 — super-unity extrapolated gap-edge sample (`samples="centers"`) |
| `qpsim/services/rate_equation_coefficients.py` | 1 — `nu_0` J⁻¹m⁻³ window |

Each was checked against every shipped caller, material YAML, validation
driver, script and webui path before being applied, and the full suite shows no
new failure outside the authentication class. Two are worth knowing about:

- **`gap_suppression.py` `samples="centers"`** now rejects an input that
  satisfies the function's *documented* contract (`f ∈ [0, 1]`) when the
  first-cell extrapolation `1.5·f[k] − 0.5·f[k+1]` leaves `[0, 1]`, i.e. once
  `f[k] > (2 + f[k+1])/3 ≈ 0.67`. It is unreachable in the Fischer/M25 regimes
  (gap-edge occupations there are `≪ 1`), and it makes the `centers` branch
  consistent with `edges`, which already rejected such an array. Retained
  deliberately, but it is a contract change, not a neutral repair.
- **`materials/database.py` and `rate_equation_coefficients.py`** impose
  numeric windows on physical constants. Every value in the repo was
  enumerated against them, but a *user-supplied* material outside the window
  is now refused rather than silently mis-scaled. Widen the window rather than
  delete the guard if a legitimate material is ever rejected.

If any of these proves too tight in use, the guard — not the caller — is what
should change.
