# Deep review — 2026-07-04 session commits (0623b6d..8141202)

Max-effort multi-agent review of the session's six commits (Airy pins +
transient regression test, paper-2 architecture doc, demo scripts + low-D0
sweep, the M25 Γ̄-normalization fix + branch driver, and the two qp-diffusion
manuscript rounds). Method: 10 independent finder angles (5 correctness, 3
cleanup, altitude, CLAUDE.md conventions) → dedup → 1-vote adversarial
verification per candidate (6 verifiers) → fresh gap sweep. 25 deduped
candidates; 15 reported (all CONFIRMED); every finding below was fixed the
same day unless marked OPEN.

## What was independently verified clean

- The Γ̄ = Γ̃/N_CP(R) normalization: four finders traced it term-by-term
  through `_rate_equation_residual`, `m25_junction.evaluate`, and the seeds —
  applied exactly once at every density-equation site, qubit channels
  correctly on raw Γ̃; the global-QE tunneling-cancellation algebra is
  correct.
- The paper edits: all labels resolve, Dynes footnote content fully preserved
  in SM app:dynes_remark, zero violations of the settled CLAUDE.md guards,
  N₀ unification dimensionally sound at every site.
- No mutable-default/closure/aliasing pitfalls in the new Python; deleted
  baselines unreferenced; photon-kick read/write round-trip exact.

## Findings and resolutions (FIXED same day, commits of 2026-07-04 evening)

1. **demo_kid_pulse_response readout energy ×1000 too small** (µeV/GHz
   coefficient 4.1357e-3 instead of 4.1357) — δf/f and Q_i were computed at
   an effective 5.5 MHz. Fixed; demo dataset regenerated (Q_i at pulse peak
   is now ~4.3e3, physically sensible for x_qp ≈ 2.5e-2, vs the bogus 3.5e6).
2. **webui M25 chemical-potential ordering inverted** — naive μ = Δ + T·ln x
   omitted the √(Δ/2πT)·erf/erfc partition (≈ +5.4 GHz on μ_R> at 20 mK,
   5 GHz asymmetry). The paper-exact inversion is now a public service
   (`rate_equation.chemical_potentials_kelvin`); webui consumes it; numeric
   smoke test pins the ordering and μ_L/Δ_L ≈ 0.872 at 20 mK.
3. **fig4 global-QE fixed-point loop** silently returned unconverged iterates
   and stamped `residual_inf_norm=0.0`. Now raises on cap; reports NaN (the
   reduced closure has no full-model residual).
4. **Branch-driver exchange fallback** `min(merged_Ts)` could anchor a cold
   spurious merge; rule was untested. Replaced by contiguity-validated
   selection (raises with merged-pattern diagnostic); unit-tested both ways.
5. **webui M25 numeric shift unguarded + stale guidance** — smoke test added;
   "multi-stable — try a different branch picker" reworded in execute.py and
   app.js (root is unique for M25-like parameters post-fix).
6. **accept_lm_convergence accept path + 1.0 Hz ceiling untested** (deleted
   test was the only coverage; path live via webui). Restored: real
   cancellation-floor stall accept test (legacy bundle still stalls at
   30 mK), ceiling raise test, `match=` message pins.
7. **Platform skip rationale stale** — `skip_unless_pinned_here` skipped all
   strict M25 pins off-platform citing pre-fix branch scatter. Replaced by
   `assert_pinned_match`: rtol=1e-6 on the stamped platform, rtol=1e-3
   elsewhere (runs everywhere, never skips).
8. **Robust 70%-fraction comparisons too loose for deterministic data** —
   `assert_robust_match` removed; all four M25 pin tests use the new policy.
9. **Error-message pins widened** — 'negative quasiparticle' and
   'cancellation floor' asserted nowhere. Restored, including a direct
   positivity-guard test; the legacy bundle's failure mode is
   temperature-dependent (maxfev at 20 mK, stall at 30 mK) — both pinned.
10. **max_x_L / expected_ordering zero coverage** — now covered by
    characterization tests. ⚠ OPEN DECISION: the tests PROVE max_x_L is
    still dangerous post-fix (lm/lsq candidate pools admit sub-1-Hz slope
    pseudo-roots; max_x_L picks x_L=3.16e-6 / residual 6.5e-5 Hz over the
    true root 5.28e-8 / 2.5e-22 Hz). It remains a webui-selectable mode —
    consider deprecating it (Soren's call).
11. **Duplicated pinned baselines with no cross-check** (fig3a/fig3b
    paper ↔ chemical_potentials; fig4_paper full-model ↔ fig4a/b parity) —
    new test_baseline_cross_consistency.py pins exact equality.
12. **Γ̄ division open-coded ×3 behind silent default 1.0** — new
    `M25Coefficients.density_gammas()` chokepoint; all three sites
    refactored; consume-rule documented on the class.
13. **gamma_ph_00_Hz structurally unreachable** through
    solve_panel_branch_sweep — parameter added and forwarded.
14. **Driver/sweep economics** — M25BranchSweep now carries `coefficients`
    (fig4 modules consume instead of rebuilding); solve_panel_branch_sweep
    memoized (4 modules share 2 sweeps per process);
    max_function_evaluations forwarded. The unconditional second (thermal)
    pass is KEPT as a documented determinism cross-check.
15. **lowD0 wrapper** — import-time monkeypatch moved under __main__ with
    base.__doc__ forwarding; the already-generated metadata.json description
    corrected in place.

## Below-cap notes (minor; mostly OPEN as acceptable debt)

- Demo scripts duplicate `fermi_dirac_distribution` (clip 500 vs library 700)
  and the 0-D state builder (`webui.builders.build_state_0d` exists);
  demo_materials rebuilds T-independent state per T point. Cleanup when the
  demos graduate into paper-2 figure scripts. (except-narrowing and the
  snapshot-dedup tolerance were fixed same day.)
- photon_kick_response stores snapshot times as %g header labels; the test's
  atol=1e-9 works only for %g-exact times. Store times at full precision if
  the cadence ever changes.
- Fig-3a parameter bundle dedup: branch tests now import from
  test_rate_equation_note_v (FIXED); the validation module keeps its own copy
  by design (figure scripts are self-contained).
- Editorial (Soren): whether the intro's early λ_k notation flag should add
  "for the even harmonics" to fully mirror the parity-dictionary guard —
  verifier judged it genuinely arguable.
- Off-platform μ comparisons near μ→0: rtol=1e-3 density scatter maps to
  ~3e-3 GHz in μ at 150 mK, marginally above the 1e-3 GHz atol some callers
  pass — only relevant if a genuinely different BLAS ever pushes points to
  the tolerance ceiling.

Gate after all fixes: 792 passed / 0 failed (779 + 13 new tests), ~5.5 min.
