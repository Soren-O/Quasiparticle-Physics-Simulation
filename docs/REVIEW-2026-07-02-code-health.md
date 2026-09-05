# Code-health review — 2026-07-02

Five parallel review passes over the full repo at fe4ec54 (~37k lines:
qpsim/, validation/, tests/, scripts/, config), hunting bugs, latent
traps, dead code, and doc drift. Physics decisions documented in the
other docs/ files were treated as settled. Everything below was
verified against callers/tests before being acted on; probe scripts
reproduced the confirmed bugs before fixing.

## Fixed in the 74f2573..365097e series

| Commit | What |
|---|---|
| 74f2573 | **Bug (reproduced):** spatial-backend transport-operator cache keyed only on (NX, dx, dt, model, D0, gap) — reusing a backend across states with a different energy grid crashed (smaller NE) or silently never diffused the extra rows (larger NE); same-shape grids with different E/dynes_gamma shared operators. Collision-kernel cache keyed on `id(state.spectral)` (stale after id reuse). Both now key on the spectral value fingerprint; regression test added. |
| 9bb7874 | **Bug (reproduced):** `load_material` opened YAMLs with platform default encoding — the root cause of the machine-wide `PYTHONUTF8=1` requirement on Windows. Now reads UTF-8 explicitly. |
| e0f80a8 | **Latent (probe: 0.9% detailed-balance violation):** pair-breaking photon K⁻ reflection partner silently lands up to dE/2 off on grids whose origin is not commensurate with ω − 2·E[0]; now warns like the existing ω check. Current validation grids are aligned. Also fixed the swapped absorption/emission comment in sub_gap_photon. |
| 0a76f9d | **Latent:** Mattis–Bardeen observable silently wrong for ω₀ ≥ Δ (σ₂ clamp, σ₁ missing pair-breaking term) — now raises; `solve_gap`'s positive-residual bracket fallback silently returned Δ_eq (an underestimate) — now warns. |
| e7d0287 | **API gap:** `M25GapAsymmetricJJ` hardwired the `max_x_L` branch picker, documented wrong for the large-asymmetry regime (~600× parity-rate inflation). `branch_picker_mode`/`expected_ordering` now pass through; default unchanged (baselines pin it). |
| 6597dd4 | Slow-marked M25 native pin tests (fig3_paper/fig4_paper) lacked the `# pinned_on:` platform gate that fe4ec54 gave the fast pins — `pytest -m slow` off-platform would redden on platform-dependent fixed-point selection. |
| 19982bb | Dead code: fig6_solve `_FAST_SUFFIX`/`_MODE_SUFFIX` leftovers, fig5_paper `plot_path()` + its orphaned tracked PDF, empty `tests/integration/`; added missing `tests/{experiments,scripts}/__init__.py` (top-level module-name collision trap). |
| 365097e | Doc drift: rate_equation docstrings (actual default seed, TestLmDeterminism, Fig 3b/4b), solvers/__init__ Anderson Type-II, STATUS.md (test counts, CI, analytic-not-opt-in, transient demo status, cross-platform build notes), baselines README rewrite, fig7-vs-nbar-loop claims, validation/__init__ figure list. |

## Known-and-accepted (flagged, deliberately not changed)

- **Unused API surface (test-covered, zero production callers):**
  the 2D machinery in `qpsim/grid/spatial_grid.py` (~330 lines),
  `qpsim/solvers/picard.py`'s generic `picard_iterate`,
  `transport/diffusion/base.py::from_name` + legacy aliases,
  `JunctionQubitCoupling` (unwired),
  `Material.tau_s/tau_r` + `Substrate.density/sound_velocity` (AMM
  inputs), `physics/kernels.py` with-phonon kernel variants
  (production path uses the collisions/phonon equivalents),
  `effective_phonon_temperature` (shipped observable, unused in-repo).
- **`validation/transient/photon_kick_response.py`** is demo-only: its
  committed baselines are verified by nothing.
- **scripts/**: four of eight `run_prelim_*` scripts are one-off
  analysis artifacts; `launch_prelim_readout_heating_overnight.sh` is
  zsh/macOS-only.
- **Unbounded operator caches** in the spatial backend for varying dt
  (fine for fixed-dt callers; an adaptive-step caller would grow them).

## Open items (worth a future decision, not fixed here)

1. **steady_state branch-collapse guard livelock** (services/steady_state.py
   ~255–278): `x_qp_ref` is the initial-guess x_qp, so a legitimately
   converged solution with x_qp < 0.1× a hot warm-start guess is
   discarded and the deterministic reset livelocks to max_picard_iter.
   Only reachable with anderson_depth>0 + warm starts. The "retry
   without Anderson" comment also doesn't match the code.
2. **M25 junction cache staleness**: `M25GapAsymmetricJJ` caches
   coefficients/moments on first evaluate but the dataclass is mutable —
   reassigning `m25_params`/`m25_drive` afterwards silently reuses stale
   physics (gap-consistency check catches gap edits only). Consider
   `frozen=True`.
3. **nbar_loop NaN handling**: a NaN/inf from `compute_Q_i` is treated
   as Q_i→∞ (Q_tot=Q_c), masking a failed observable.
4. **`_paper_envelope.py:64` Airy-argument precedence**: parses as
   x²/4^(1/3); if Fischer 2023 Eq. 31 means (x²/4)^(1/3) the fig3 dashed
   overlay is subtly off. The overlay visually matched at pin time —
   settle against the paper.
5. **Phonon source/sink dE convention**: `compute_phonon_source_sink`
   weights (i,j) pairs with dE[j] — unambiguous only on uniform grids
   and unguarded (unlike the photon channels). Add
   `uniform_grid_spacing` if dynamic phonons ever meet piecewise grids.
6. **Config**: pytest-cov installed but never invoked; CI lacks a pip
   cache and a concurrency group; version string duplicated between
   pyproject and `qpsim/__init__`.
