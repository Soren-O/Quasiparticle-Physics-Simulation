# Delivery plan — closing the legacy-app capability gap in `qpsim`

Source: the 138-entry audit plus the 3 refutations. Repo: `B:/AEinstein/Einstein/Documents/Soren/qpsim`. All probes below must be run as
`PYTHONPATH=B:/AEinstein/Einstein/Documents/Soren/qpsim C:/Users/Einstein2/Quasiparticle-Physics-Simulation/.venv/Scripts/python.exe` (the repo's own `.venv` has no pydantic).

---

## 1. Headline numbers

**138 audit entries** across 8 areas. Status as filed, then corrected by the challenge:

| | entries |
|---|---|
| `already_closed` (as filed) | 40 |
| — refuted on challenge (now open) | −3 |
| **`already_closed` (after challenge)** | **37** |
| `false_gap` — nothing was ever lost | 24 |
| `partial` | 39 |
| `absent` | 29 |
| `blocked_with_cause` | 6 |
| refuted `already_closed` | 3 |
| **genuinely open** | **77** |

The 77 open entries contain heavy duplication (Dynes appears 3×, "preview before run" 4×, "gap expression in the UI" 4×, "phonon numbers out" 2×, …). **De-duplicated, the open work is 53 distinct deliverables:**

| effort | deliverables | covers entries |
|---|---|---|
| install/config only | **1** (`pip install gdstk`) | 1 |
| `ui_only` | **14** | 22 |
| `small` | **23** | 30 |
| `medium` | **8** | 13 |
| `large` | **3** | 5 |
| `research` / physics decision | **4** | 7 |

**The shape of the remaining work is not physics.** 38 of 53 deliverables are `ui_only` or `small`, and almost all of them are the same defect: the engine and the pydantic schema support the capability, it is measured working from Python, and `qpsim/webui/static/app.js` has no control for it. Nothing in Waves 0–2 below requires touching a kernel.

Recommended to **not build**: 24 false gaps + 4 more (scalar-diffusion mode, numba backend, `T_bath=0`, an old-format setup translator) — see §4.

---

## 2. What the mode collapse already bought

"Transient dynamics on a geometry" existing as `kinetics` + `strategy=time_march` is why the remaining Output/Analysis work is *plumbing over arrays that already exist* rather than missing computation. Concretely, these stopped being gaps because a spatial time march now emits `snap_f (frames, NE, cells)`, `snap_n_ph (frames, omega, cells)`, `snap_xqp_profile`, `snap_gap`, `snap_t_ns`, `snap_omega_bins`, `obs_x_qp_mean/max`:

- **QP field animation** (Output #1) — `field_over_time` scrubber exists *and* fixes one colour norm across frames, which the old viewer did not.
- **Energy-resolved map with an E slider** (Output #2) — `snap_f` supplies (frame, energy); only the *quantity* (`f` vs `ρ·f`) is still wrong.
- **Phonon field with an ω slider** (Output #3) — `snap_n_ph` supplies it; only the axis *label* is missing.
- **Pulse rise-then-decay, measured** (Drives #2): 9.06e-06 → 2.83e-02 at t=60 → 2.18e-02 at t=120 on a 1×4 strip. Not expressible before.
- **Non-separable moving hot spot** (Drives #3) — verified translating across cells between t=0/20/50.
- **Per-edge boundary conditions producing three distinct column-mean profiles** (BC #1) — needs a spatial run to demonstrate at all.
- **Self-consistent Δ per cell per frame** (`snap_gap`) — a capability the old app never had, recorded on every time march.
- **Every spatial initial-condition probe** (IC #1–#9) — gaussian/point/uniform/expression all verified against a real per-cell gradient.
- **Phonon sector distinctions** (`thermal_bath` vs `dynamic_escape` vs `dynamic_closed`, `use_phonon_side_kernel`) — all measured on spatial runs.
- **Saved setups still load** via `RETIRED_MODE_UPGRADES`, so no migration work is owed.

**The cost of the collapse, which this audit surfaced:** making `strategy` a setting rather than a mode created a route (`steady_state` → `run_steady_state_0d`, 1-cell only) that reads a *strict subset* of the setup and silently ignores the rest — `setup.initial`, `setup.drives`, `setup.injection` are all dropped with no error, no warning, no note, while `terms.py` still reports `src = on`. That is Wave 0, item 1, and it is the single most important fix in this plan.

---

## 3. Ordered plan

### Wave 0 — "the app stops saying things that are not true" (≈1–2 days, no physics)
Nine one-to-ten-line fixes. Highest value per line because each one closes a *false statement*, and every one of them is this repo's signature defect.

| # | fix | file |
|---|---|---|
| 0.1 | `validate_setup` **errors** when `strategy='steady_state'` and (`injection.enabled` or `drives` non-empty or `initial.kind != 'thermal'`); `terms._injection` reports ABSENT under `steady_state` | `qpsim/webui/builders.py`, `qpsim/webui/terms.py` |
| 0.2 | `PhononSector` model_validator refuses non-`bath` `initial` under `mode='thermal_bath'` | `qpsim/webui/schemas.py` |
| 0.3 | `save_setup`/`load_setup` persist and return `benchmark` | `qpsim/webui/store.py` |
| 0.4 | Load-setup and material-select call `showWizardStep(wizard.index)`, not bare `renderForm()` | `static/app.js:341, 548` |
| 0.5 | `showRunDetail` pulls `summary.benchmark` out and calls `renderBenchmark` (today it renders `[object Object]`) | `static/app.js:488` |
| 0.6 | `_plot_phonon_frame` reads `snap_omega_bins` → title "ω = 671 μeV" not "ω bin 60" | `qpsim/webui/plots.py:440` |
| 0.7 | `_plot_xqp_profile_2d` plots against the stored `x_um` (cell **centres**), not `arange` | `qpsim/webui/plots.py` |
| 0.8 | `JobRunner.overlay` emits `elapsed_s` for active jobs; kinetics progress message becomes `t = x / y ns` like the 0-D path | `qpsim/webui/runner.py:328`, `execute.py:482` |
| 0.9 | Remove/gate the Dynes Γ field for `kinetics` (it can never validate) | `static/app.js:55` |

**Done =** a pytest that, for each of 0.1/0.2, asserts a `ValidationError`/warning where the audit measured silence; a round-trip test asserting `GET /api/setups/{slug}` returns `benchmark`; a plots test asserting `"μeV"` in the phonon title and that the 1-D profile abscissa equals `x_um`; a runner test asserting `elapsed_s` is present on a `running` job.

### Wave 1 — make the built physics reachable from the browser (≈1 week, `ui_only`)
Five form sections in `static/app.js` `FORMS` + `index.html`. Every one is engine-complete and measured.

- `gap_regions.kind` gains `expression` + expression/params inputs. *(closes 4 audit entries incl. one refutation)*
- `boundary.kind` gains `robin` + `aux_value`, plus a `per_edge` override table.
- A repeatable `drives` list: kind selects for energy/space/time, amplitude, `channel` gain/loss. *(the old app's `pulse` and `custom` modes are unreachable today)*
- A **Phonon sector** section: `mode`, `tau_l_ns`, `use_phonon_side_kernel`, `phonons.initial`. Today the shipped default `thermal_bath` makes three of ten term buttons permanently inert.
- An **Initial condition** section: QP `kind`/`amplitude`/energy profile/space profile/expression/params, and the phonon seed.

**Done =** for each section, a browser-authored setup reproduces a number already recorded in the audit: gap `[187.5 … 262.5]` from `gap*(1.0+0.5*x)`; robin β sweep monotone 2.59e-2 → 2.6e-41; pulse drive 9.06e-06 → 2.83e-02 → 2.18e-02; `dynamic_escape` emits `snap_n_ph`; `pesc`/`psc`/`prc` term buttons become live. **A test that only asserts the field posts is not acceptable** — see §6.

### Wave 2 — get the numbers out (≈1 week, `small`, all in `plots.py`/`server.py`)
Everything here reads arrays that already exist in `result.npz`.

- `GET /api/runs/{id}/result.npz` (FileResponse + Content-Disposition). One route, unblocks every array nobody has written a figure for.
- Phonon CSV keyed on `snap_n_ph` (`omega_ueV, cell, t_ns, n_ph`) + an `obs_n_ph_mean` series. Today the phonon sector on a geometry is **view-only**.
- `x_qp` vs `t` figure from `obs_x_qp_mean/max` (+ `snap_max_rate` on a twin log axis), honouring `summary.x_qp_convention`.
- `frame` query parameter on the CSV route + `x_um`/`y_um` columns (today: final frame only, cell indices).
- `snap_gap` figure (`requires='snap_gap'`).
- Integrated **occupation** phonon map `∫n_ph dω` — label it occupation, **not** energy (see §5.2).
- `energy_resolved_map` draws `n = ρ_norm·f` using the existing `bcs_quadrature.bcs_dos_cell_weights`, or is relabelled "occupation".
- `<a download>` on figures.

**Done =** `available_csvs` on a spatial `dynamic_escape` run includes `phonons`; `GET .../result.npz` returns 200 with a filename; a test that the CSV `frame` parameter yields different numbers for frame 0 vs frame N; a test that the time-series figure's plotted y-values equal `obs_x_qp_mean` element-wise.

### Wave 3 — pre-run preview (≈3–5 days, `medium`, one endpoint closes 4 entries + unblocks 2 more)
`POST /api/preview` taking a setup envelope and returning: mask PNG, seeded QP field PNG, seeded phonon field PNG, the clipping notes from `seed_occupation`, and the geometry's edge ids/segments. `build_geometry_2d` + `build_initial_state_2d` already run in well under a second.

**Done =** posting `initial.amplitude=5.0` returns the clip note *without running*; posting a GDS setup at too coarse a mesh shows a 3-cell-wide device before any solve. Feeds the catalogue mask cards and the edge picker.

### Wave 4 — geometry reach (≈1 week)
1. `pip install gdstk` into the designated venv (`pyproject.toml` already declares the `gds` extra). Effort *none*, unblocks 3 entries.
2. `GET /api/geometry` returning edge ids, normals, endpoints, face counts. Prerequisite for everything else here — on an annulus the direction aliases resolve `right` to **both** rims (`edge_0006` inner and `edge_0008` outer), silently.
3. A third `GeometrySource.kind` carrying explicit polygons (`rasterize_polygons` needs no optional dependency) — a gdstk-free route to a non-rectangular device.
4. GDS layer discovery endpoint + select.
5. Edge-picker overlay on the mask figure (colour by BC kind, red = unassigned, as the old app did).
6. Bind `Geometry.bounds` to `x_um`/`y_um`, or delete the field. It is currently **dead metadata with zero consumers**.

**Done =** an annulus with the inner rim absorbing and the outer reflective, authored entirely from the browser, addressing `edge_0006` and `edge_0008` separately.

### Wave 5 — batch and verdicts (≈1 week, `medium`)
- Promote `scratchpad/benchmark_verdicts.py` (already walks the catalogue, ~70 s for ten cases) into a run-all action + aggregate report. Today exercising the catalogue is 38 manual clicks; 23 of 38 cases produce no automated verdict at all.
- Put `benchmark` into `envelope()` + a selector from `/api/benchmarks`, so a user-authored setup can be scored. The benchmarks already refuse setups they do not describe, loudly.
- **CI gap:** 9 of 10 `bench/*._build` functions have no test that calls them. The ten recorded pass numbers come from a session-local scratchpad script, not CI. Fix this in the same wave.

**Done =** one action produces a verdict table matching the recorded rel-errors (diffusion 2.73e-03, scattering 3.70e-04, recombination 6.06e-06, subgap 4.21e-08, pb 6.53e-06, injection 2.85e-15, psc 3.90e-05, prc 5.70e-08, pesc 3.44e-12, gapeq 3.15e-04), and each `_build` is imported by a CI test.

### Wave 6 — the analytic BC/geometry suite (`large`, the biggest real gap)
Port the old strip cases onto the current operator — they have **already been run once** and pass: Neumann q=±0.02 rel 9.46e-08; absorbing sine-1 2.01e-04; Dirichlet sine-1 2.01e-04; Robin even 8.14e-03; Robin odd 2.41e-03. Add rectangle Dirichlet and the annulus Bessel cross-product. Today four of five BC kinds are reachable, honoured, and completely unvalidated against a closed form, and the inhomogeneous-forcing branch (Dirichlet g≠0, Neumann q≠0, Robin γ≠0) is the least-exercised code in the transport layer.

**Write the Robin case with a first-order convergence statement, not a fixed tolerance** — refining 32→512 cells gives 1.63e-2, 8.14e-3, 4.07e-3, 2.03e-3, 1.02e-3. That is order 1, because `grid/spatial_grid.py:332` applies β to the cell-centre value rather than an extrapolated face value. **Done =** benchmarks registered, green in CI, each carrying its measured convergence order.

### Wave 7 — deferred (see §5)
Energy budget, phonon temperature field, live frame streaming, 2-D analytic heatmap comparison, `steady_state` with a static source folded into `external_flux`.

---

## 4. Not worth doing

**24 false gaps.** Do not bill these against the rewrite. The old app could not do them either, or the new mechanism is better: import IC from a file; compare two runs; parameter sweeps; per-region materials; coherence-factor convention; collision-solver selection; Pauli warn/error thresholds; D(E) closure selection; self-consistent gap; loss-channel drives; the Kupriyanov–Lukichev interface; a trap model; monoenergetic/gap-edge seed shapes; the phonon-side kernel split; excess-vs-absolute seeding; solver/strategy selection.

Three of those are genuinely desirable **new** features (run comparison, parameter sweeps, fitting τ from a transient — `bench/diffusion.py::_fit_rates` already exists and should be lifted into `qpsim/observables/`). Build them if you want them; do not call them gap-closing.

**Ported bugs — refuse explicitly:**
- The old energy ledger summed `ω·n_ph·dω` with **no mode density** and `E·n·dE` with **no BCS DOS**, so `energy_exchange_residual` was the difference of two wrong numbers and read ≈0 for the wrong reason. Rebuild, never port.
- The old per-frame renormalised animation colour scale (the classic way an animation lies about a decaying field). Already fixed; keep it fixed.
- The old drive used left-endpoint sampling with `state += dt*g_ext` — first order, lags a rising edge. The current midpoint sampling is correct; do not "restore parity".
- The old `setup_id`-preserved-across-edits hazard and the `.precompute.npz` staleness machinery. The new app has no precompute cache; the whole hazard class is gone.
- The old Pauli *warning* threshold. The new hard `RuntimeError` on a clip that moves conserved density is a defensible tightening — record that a marginal old run can now fail, do not loosen it.
- The old broadened-ρ-against-pure-BCS-coherence-factors Dynes implementation. See §5.1.

**Superseded / not worth the effort:**
- **Scalar diffusion mode** (`energy_gap=0`, `large`): a scalar field is not a quasiparticle model, and its only role — being the substrate for the analytic suite — is now served by energy-resolved benchmarks. Deliberate narrowing, do not reopen.
- **`T_bath = 0` exactly** (`ui_only`): 1e-6 K *is* the zero-temperature answer for Al (kT = 8.6e-5 μeV vs a 180 μeV gap), and the `gt=0` bound removes a divide-by-kT NaN class. Cosmetic only.
- **Numba collision backend**: performance only, was never user-visible (`simulation_service` never passed it). Keep it stripped; the mitigation is the existing O(NE²) warning.
- **Old-app setup-format translator**: build only if real archived setups must be replayed. **Do** fix the two cheap parts: `DriveSpec.amplitude` carries no note that the old `rate` was a *density* rate in μeV⁻¹ μm⁻² ns⁻¹ and does not transfer (the `InitialCondition` docstring already admits this for seeds), and `qpsim/collisions/__init__.py:9` still lists a planned `external_generation.py` that shipped as `fields/drive.py`.
- **A named "uniform energy weight" kind**: the old shape diverges in occupation at the gap edge, which is *why* the reframed model does not offer it. Put `rho` in the energy-expression scope or document the workaround; do not add the kind.

---

## 5. The hard ones — research, not plumbing

**5.1 Dynes broadening (3 entries, blocked at 10+ sites).** Refused with a stated reason at `collisions/spatial.py:100`, `t3_spatial.py:494`, `t3_diffusion.py:710/1255/2087/2190`, `collisions/phonon.py:60`, both photon modules, and `observables/ac_conductivity.py:110`. The refusal text names the old implementation as the problem: the old app multiplied a *broadened* ρ into *pure-BCS* coherence kernels. **Unresolved:** whether a consistently broadened normal/anomalous coherence kernel is tractable in this quadrature scheme at all. **Contradiction to flag to a physicist:** `devices/m25_junction.py:163` *requires* Γ>0 and rejects Γ below the local cell width, so the codebase currently holds two opposite positions on Γ. Report as blocked-with-cause; do not schedule.

**5.2 Energy budget / conservation residual.** Rebuild, not port. The QP side needs the BCS DOS; the phonon side needs a mode density. **Unresolved and dangerous:** per this repo's own pair-marginal work, adding a per-bin Debye ω² on top of a lattice whose kernels may already carry it is a double count that was caught once before, by inches. Do not ship any ω-weighted phonon quantity until someone establishes whether the ω lattice carries mode density. The plain `∫n_ph dω` occupation map in Wave 2 needs no such judgement and is the honest interim.

**5.3 Phonon temperature field.** A non-thermal `n_ph` has no single temperature. A per-cell fit of `effective_phonon_temperature` over `snap_n_ph` is a physics decision, not plumbing. Ship the occupation map; only promise a temperature where the fit is defensible.

**5.4 τ_s ≠ τ_r (the one real capability regression).** In the old app these were two editable boxes that moved numbers; today `Material.tau_s/tau_r` exist, warn, and are read by nothing. The refusal is defensible — Kaplan 1976 normalises both channels by a single τ₀ and `tau_0_pb_ns` is matched to the QP side, so scaling one channel alone breaks QP↔phonon energy consistency in the dynamic-Ph₀ backends. **Unresolved:** whether a per-channel τ with a consistent phonon-side rescale is physically meaningful. **Decide one way or the other** — if it stays refused, delete the fields from `Material` rather than keep warning about them; a field that admits it is inert is exactly this repo's characteristic defect.

**5.5 Material data completeness.** Only Al is fully populated. NbN/Sn/Ta have `rho_F=0`, no sound velocities, no film thickness, neither phonon time; Nb lacks `tau_0_pb_ns`; TiN lacks both. These are hard requirements: `observables/density.py:149` refuses `rho_F=0` for absolute n_qp, `phonon_escape.py:83` requires `film_thickness>0`, the default dynamic-Ph₀ kernel needs `tau_0_pb_ns`. Four of six materials cannot run half the engine, and the UI renders the missing values as "—", which reads as *not applicable* rather than *this material cannot run*. Literature work (Kaplan Table II-style τ₀^ph, DOS, sound velocities). **Cheap interim: mark in the materials view which capabilities each entry can support.**

**5.6 Robin is first order.** The assembly is byte-identical to the old app's, so this is inherited, not a regression — but fixing it (extrapolate to the face) moves every Robin number. Decide whether Wave 6 documents order 1 or fixes it. Documenting is the safe default.

**5.7 Sub-`dt` pulses.** Both apps are wrong, in opposite directions: the old over-delivers an on-boundary pulse by a full `dt`; the new delivers a 0.4 ns pulse as **exactly zero** and a 0.5 ns pulse as a full 1.0 ns (2× over-delivery), silently. The correct fix is an exact time-average of the gate over each step (`medium`). Ship the `(t_off - t_on) < 5·dt` warning in Wave 0/1 and schedule the real fix.

---

## 6. Risks specific to this codebase

**(a) Settings that are read and ignored.** This repo's signature defect, and this audit found open instances in every area:

- `steady_state` drops `initial`, `drives`, `injection` — measured **bit-identical to 11 digits** with injection at 2e-4 and at 2e-1, while `terms.py` reports `src = on`.
- `phonons.initial` dropped whenever `phonons.mode='thermal_bath'` — validates cleanly, seed becomes `None`, no note.
- `MaterialParams.tau_s/tau_r` — warn, do nothing.
- `save_setup` drops `benchmark`.
- Written and read by **nothing**: `snap_omega_bins`, `x_um`, `snap_gap`, `obs_x_qp_mean_paper`, `obs_x_qp_max_paper`, `obs_Q_i`, `Geometry.bounds`, and `run_time_dependent` (imported at `execute.py:52`, never called — it reads as though a transient mode exists).
- The `pesc` term button is inert under the shipped default.

**Mitigation, mandatory for Wave 1:** every new UI control ships with a probe that *measures a number moving*, not one that asserts the field posts. And the run must be **driven** — the audit records that on an undriven setup all solver methods and all phonon sectors return bit-identical `x_qp`, because the fixed point is the thermal state. A binding test on an undriven setup is vacuous.

**(b) Second implementations of the same quantity.**
- `renderForm()` and `showWizardStep()` render the same `FORMS` into two different panels (`#setup-form` vs `#form-geometry`). That duplication *is* the load-setup bug, and it has two callers.
- `x_qp` vs `x_qp_paper` are two conventions; the CSVs and figures do not consistently honour `summary.x_qp_convention`. Any new figure must.
- The gap map is validated only in `validate_setup`, so `execute_setup` called directly dies with the raw quadrature error `first cell edge 180 > 129.121` instead of the named one. Two paths, one check.
- The BCS DOS weight exists in `bcs_quadrature.bcs_dos_cell_weights` and `execute.py::_xqp_profile_2d` already uses it — **use it**, do not re-derive it in `plots.py`.
- `_CSV_REQUIRES` keys the phonon CSV on `n_ph` (0-D route) while the spatial route writes `snap_n_ph`. Follow the either-shape dispatch `_csv_kinetics_occupation_either_shape` already demonstrates.
- **Before believing any fix landed, grep for a second copy.** This repo has already shipped an adjudicated fix onto one of two copies of the same reconstruction, leaving it inert on the path that mattered, with nothing going red.

**(c) Changes that move numbers.**
- Making `steady_state` honour drives/injection **changes every driven steady-state answer**. It is a correction, not a regression — the current answer is the *undriven* fixed point reported for a driven setup — but every stored driven-steady-state result becomes wrong-by-construction and should be re-run or invalidated.
- Fixing the Robin face extrapolation (order 1 → 2) moves every Robin result.
- Changing `energy_resolved_map` from `f` to `ρ_norm·f` changes every figure of that name. If you would rather not, relabel the figure instead.
- **Do not touch Al's `D_0 = 60`.** The out-of-band warning is the intended state per `docs/HELD-BACK-ADJUDICATION-2026-08-11.md` item P14; correcting it to ~6 shortens every Al diffusion length √(Dτ) by ~2.4×. Same for Nb.
- Installing gdstk moves no number but enables geometries that have never been validated — Wave 6 should land before anyone trusts a GDS result.
- **Nothing in Waves 0–2 should move a number. If one does, that is the finding** — stop and investigate rather than updating the expected value.

Per the standing repo directive (engine before figures, nothing published yet), a forced digest/figure regeneration is **not** a cost to weigh against a correctness fix — but each number-moving change must be declared in its commit, not discovered later in a stale digest.