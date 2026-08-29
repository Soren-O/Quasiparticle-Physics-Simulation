# Next-audit brief — qpsim repo state and rabbit-hole flags

Deliberately-lean starting point for a fresh code audit of `qpsim`. **Read this
first, then audit with independent eyes.** The dated `docs/AUDIT-2026-07-12-
integrated.md` and `docs/AUDIT-2026-07-13-reaudit.md` are historical findings
records: useful for context, but treat their conclusions as *done* and verify
independently rather than assuming the areas they touched are now flawless. This
brief exists so a fresh pass need not re-derive settled false alarms — not to
tell you what is or isn't a bug.

## Current state (2026-08-29)

- Active branch `fix/review-2026-08-03`; **not merged to main**. The current
  closeout retires the Marchegiani-2025 reproduction modules, tests, and
  canonical bundles while retaining the generic M25 engine/device/UI surface.
- Fischer reproduction, ladder, and certification coverage is now explicitly
  marked `paper_validation`. Collection contains 3,394 tests: the everyday
  lane selects 2,543 and `pytest -m paper_validation` selects 847.
- The authoritative everyday gate passed **2,540 tests, 1 skipped, 2 expected
  xfails, and 851 deselected in 986.08 s**. A focused retained-capability gate
  covering phonon setup, canonical manifests, M25 services/device behavior,
  transient artifact I/O, and web UI passed **526 tests with 1 skip**.
- The three largest everyday costs consume 443.09 s (44.9% of the full gate):
  interface-trap setup (197.63 s), pair-split corner-cell convergence
  (123.39 s), and spatial closing-barrier monotonicity (122.07 s). Audit their
  unique coverage before deciding whether to optimize or move them to `slow`.
- Highest-churn areas remain the phonon lattice/collision contracts, moving-gap
  and spatial backends, the M25 rate-equation acceptance layer, BCS gap-edge
  quadrature/observables, and source-bound artifact provenance.

## Gate gotchas (a green run can still be hiding a break)

- The default `pytest` **deselects `slow` and `paper_validation`**. A numerical
  change can leave the everyday gate green while breaking a source-bound paper
  workflow. CI runs `pytest -m "slow and not manual_slow"` as a separate step;
  source-bound non-slow workflows require an explicit `pytest -m
  paper_validation` run during planned recertification.
- Fischer **Fig. 6** is `manual_slow` and excluded from CI. Measured production
  rows total `6.04 h` serial (`2.11 h` concurrent), not the stale 14-hour
  estimate. Running the old wrapper does not close the newly diagnosed default-
  observable mismatch; a full repaired fixed-gap/direct campaign, including
  refinement of its corrected negative low-drive point, is the useful expensive run.
  Direct-mode generation must keep using the explicit `_direct` path contract;
  never route a parameterized direct result through the canonical default pin.
- Fischer figure CSVs are **self-pinned regression baselines**, not
  digitized paper data. Passing them proves code stability, *not* paper fidelity.
- Direct gap integrals now fail if the grid omits the superconducting edge.
  Sub-gap guard storage is valid only when the grid face covers every gap the
  caller permits; only roundoff-sized positive face offsets are aligned to the
  gap, and inactive values must not be extrapolated into active support.

## False positives — look like bugs, are intentional (don't re-file)

> **Canonical ledger:** [`CODE-REVIEW-FALSE-POSITIVES.md`](CODE-REVIEW-FALSE-POSITIVES.md)
> now holds the complete, maintained list (refuted findings, documented
> approximations, and the traps section — claims that looked like false
> positives but were real). Add every newly adjudicated false positive
> there at the end of each audit round. The entries below predate it and
> are kept for context.

- `solve_gap` is biased high near `T_c` because the energy grid can't sample
  below-gap occupation. This is a documented domain-contract limitation and it
  now **warns** — not a quadrature bug.
- The phonon-side pair-breaking kernel normalizes by the *current* Δ (not Δ₀) —
  a documented approximation (Δ ≈ Δ₀ at Fischer temperatures).
- `_tau_R_inverse` uses `Δ_R` — correct per arXiv:2408.17218 **v2** Eq. D11. (A
  "should be mean-gap Δ̄" claim came from a different version/label; false alarm.)
- TiN material `rho_F ≈ 3.8e28 eV⁻¹ m⁻³` — film/disorder-dependent, defensible;
  not a unit typo.
- M25 steady-state acceptance uses a **row-wise source-scaled + backward-error**
  residual gate, not a fixed tolerance. A bare `1e-14` or `1.0 Hz` reading is the
  *old* miscalibration (already fixed); the current gate is intentional.
- Spatial Crank–Nicolson transport **raises** on an under-resolved step
  (diffusion number `D₀·dt/dx²` too large) by design — fail-loud, not a crash to
  "fix". Keep that number ≲ 5 (validate_setup warns above ~8).

## Current qualified numerical items

- **F23 Fig. 5:** the apparent continuation hysteresis was loose-Newton
  pseudo-root acceptance, now resolved with a `1e-10` inner gate and `1e-9`
  final certificates. The tested `T*/Delta=0.60` split is not evidence of a
  physical branch ambiguity; the legacy pin still needs a full tight-contract
  regeneration/refinement campaign before broader branch behavior is settled.
  Known + accepted meanwhile (2026-07-20, don't re-file): the slow-test x_qp
  gates (`atol=1e-6`) are vacuous for the low-drive branch (signal
  2.1e-10..1e-6); signal-scaled tolerances are a required part of the
  regeneration campaign's re-pin.
- **F23 Fig. 6:** the 66-point run certified a loose solver contract, but the
  default suppression pin is much tighter than its accepted gap-map error.
  The repaired fixed-gap/direct path has a strictly certified full-grid point,
  but its corrected low-drive observable is `-0.01680568`; it still needs a
  full canonical sweep and observable refinement.
- **F23 Figs. 9--13:** `Q_i` is still nonconverged at aligned `NE=6480`.
  Exact-cell/FV variants, modest cancellation, and an overlap-aware photon
  prototype do not support rewriting the photon operator. A proposed
  conservative promotion policy—not a derived error budget—requires `<=1%`
  maximum `Q_i` change on two consecutive commensurate rungs and `<=0.25%`
  exact-cell/FV observable discrepancy so quadrature error stays subdominant.
- **Diffusion feedback benchmark:** its grid now covers the direct closure down
  to `0.5*Delta_0`, seed amplitude is calibrated at the requested fixed point,
  and the raw map has a fail-loud `1e-12`/64-iteration convergence contract.
  Its deliberately bounded guard plateau is reconstructed on fixed edge nodes;
  changing the center stencil with gap support can jump over the fixed point.

## Orthogonal: paper physics

The bundled `papers/qp-diffusion` manuscript still has contested blocker physics
(B1–B5, M1). `verify_*.py` passing means the code **matches the manuscript**, not
that the manuscript is correct — that needs a human physicist. The engine work is
independent and could land without it.
