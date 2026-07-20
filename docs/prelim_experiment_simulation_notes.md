# Preliminary Experiment Simulation Notes

> **STALE NUMBERS — regeneration required (2026-07-19 audit, findings H1/H2).**
> Every finite-phonon quantitative result below (7 mK sweeps, temperature
> sweep, convergence checks, readout-heating percentages) was produced with
> the legacy QP-side kernels in the PHONON equation, which under-weight
> phonon emission/pair-breaking by 4–17× across 2–6Δ. At the nominal case the
> corrected kernels move δf_r by ~65%, Q_i by −40%, x_qp by +54%, and
> n_ph,max by ×307 — far above the 1–2% convergence budget quoted below.
> The readout-heating rows additionally used a silently snapped photon
> energy. `FinitePhononSpatialRunner` now uses the phonon-side kernels and
> snaps the readout mode explicitly, so rerunning the scripts regenerates
> honest numbers; until then, treat every number below as qualitative only.

This is the compact source-of-truth file for the 100 um Al-strip simulations
supporting the preliminary-exam experiment.

## Experimental Goal

Measure the full complex resonator response `S21(f)` while voltage-biasing an
Al/AlOx/Al quasiparticle-injection junction near the SIS threshold
`e V ~= 2 Delta_Al`.

Primary observables:

- Resonance-frequency shift, `delta f_r`.
- Change in resonance depth / internal loss, modeled as `Q_i`.
- Voltage dependence once the Gaussian source is replaced by a real SIS
  injection spectrum.

Current working assumption: the biased junction drives the Al strip into a
nonthermal steady state; the readout tone is weak enough that readout heating is
ignored in the first simulation pass.

## Geometry

Coordinate convention: `x = 0` is the shorted end of the quarter-wave resonator.
The Al strip extends from the short toward the open/coupled end.

Al strip:

- Length: `100 um`
- Width: `0.1 um`
- Thickness: `~100 nm`
- Junction: `Al/AlOx/Al`, located at the top/shorted end in the 1D model

Nb resonator:

- CPW cross section: `6-10-6 um`
- Nb thickness: nominally `60-100 nm`; TeX presentation table uses `80 nm`
- Ground plane is Nb
- Al section is shorted, then in series with Nb; the other end is open and
  capacitively coupled to the feedline

## TeX Presentation Resonators

Source: `Graduate/Presentations/Simulation for Experiment Presentation/slides_revised.tex`.

Target frequencies are `f_n = 5 + n/7 GHz`, `n = 1..6`.

Length construction:

`l_R = l_fixed + L_coup + 5 L_var`

- `l_fixed = 3016.194 um`
- `L_coup = 240 um`

| index | location | f0 (GHz) | L_var (um) | l_R (um) |
|---:|---|---:|---:|---:|
| 1 | Top Left | 5.142857 | 457.444 | 5543.414 |
| 2 | Top Middle | 5.285714 | 427.401 | 5393.199 |
| 3 | Top Right | 5.428571 | 398.941 | 5250.899 |
| 4 | Bottom Left | 5.571429 | 372.125 | 5116.819 |
| 5 | Bottom Middle | 5.714286 | 346.468 | 4988.534 |
| 6 | Bottom Right | 5.857143 | 322.063 | 4866.509 |

TeX/Sonnet values quoted for the 7 mK Nb CPW:

- `Z0 = 51.343 Ohm`
- `epsilon_eff = 6.8875`
- `L' = 4.4946e-7 H/m`
- `C' = 1.7050e-10 F/m`

These constants are encoded in `qpsim/experiments/prelim_resonators.py`.

## Current Weighting

For a shorted quarter-wave resonator, using strip coordinate `s` measured away
from the short:

`I^2(s) = cos^2(pi s / (2 l_R))`

The strip participation used in the response calculation is:

`p_I = integral_strip I^2(s) ds / integral_0^lR I^2(s) ds`

with:

`integral_0^lR I^2(s) ds = l_R / 2`

For the 100 um strip this gives roughly `p_I = 0.036-0.041` across the six
resonators. Higher-frequency/shorter resonators have slightly larger strip
participation, so their absolute shifts in Hz are slightly larger.

## Simulation Model

Spatial QP backend:

- State: `f(E, x)`
- Spatial transport: 1D diffusion with reflective boundaries
- QP collisions: electron-phonon scattering and recombination kernels
- Source: currently a normalized Gaussian injection spectrum near `2 Delta_Al`
- Main material: Al from the qpsim material database

Finite-phonon bottleneck extension:

- Adds local dynamic phonon occupation `n_ph(omega, x, t)`
- Phonon escape to bath: `(n_th - n_ph) / tau_l`
- No lateral phonon transport yet
- Operator splitting: QP diffusion half-step, local QP collision step, phonon
  escape/source step, QP diffusion half-step

Readout-heating extension:

- Optional fixed-`nbar` sub-gap photon scattering channel using
  `qpsim.collisions.sub_gap_photon.sub_gap_photon_collision_rates`
- Local drive strength is weighted by the normalized quarter-wave `I^2(x)`
  profile, so `nbar` is currently interpreted as the peak occupancy at the
  shorted end
- This is not direct pair breaking for the 5-6 GHz modes because
  `h f_readout < 2 Delta_Al`
- The self-consistent Fischer-style `nbar(P_read, Q_i, Q_c)` loop has not yet
  been wrapped around the spatial finite-phonon runner

## Response Model

The upgraded response path computes local Mattis-Bardeen conductivities first:

- `sigma_1(E,x; f_probe)` for loss
- `sigma_2(E,x; f_probe)` for kinetic inductance / frequency shift

Then it integrates the local response over the resonator current profile. This
is better than applying Mattis-Bardeen to a pre-averaged `f(E)`.

Frequency shift:

`delta f / f = (alpha / 2) * integral_strip I^2(s) [sigma2(s)-sigma2_ref(s)]/sigma2_ref(s) ds / (l_R/2)`

Loss estimate:

`1 / Q_i = alpha * integral_strip I^2(s) [sigma1(s)/sigma2(s)] ds / (l_R/2)`

Current working value:

- `alpha = 0.08`

The response helper is `qpsim/observables/spatial_ac_response.py`.

## Current Numerical Checks

Full 7 mK finite-phonon sweep:

- Output: `outputs/prelim_finite_phonon_sweep_7mk/summary.csv`
- Response table: `outputs/prelim_finite_phonon_sweep_7mk/resonator_shifts.csv`
- Grid: `NX=21`, `NE=28`, `dt=1 ns`
- Sweep: `D0 = 6, 20, 60 um^2/ns`
- Source rates: `6.264e10`, `3.132e11`, `6.264e11 QP/s`
- Phonon escape times: `tau_l = 0.1, 0.3, 1, 3, 10 ns`
- Completed: `45/45` converged

Full-sweep ranges with the spatial Mattis-Bardeen response:

- `delta f_r`: `-66.6 kHz` to `-407.7 kHz`
- `Q_i`: `4.31e4` to `2.05e5`

Focused convergence check:

- Output: `outputs/prelim_convergence_checks/summary.csv`
- Bath: `7 mK`
- `D0 = 20 um^2/ns`
- `tau_l = 1 ns`
- Fixed total source: `3.132e11 QP/s`
- Baseline grid: `NX=21`, `NE=28`, `dt=1 ns`

Representative result for resonator 1:

| case | NX | NE | dt (ns) | delta f1 (kHz) | relative shift change |
|---|---:|---:|---:|---:|---:|
| coarse | 15 | 20 | 2.0 | -126.08 | 2.86% |
| baseline | 21 | 28 | 1.0 | -129.79 | 0.00% |
| dt_half | 21 | 28 | 0.5 | -129.78 | 0.01% |
| energy_40 | 21 | 40 | 1.0 | -130.46 | 0.51% |
| space_41 | 41 | 28 | 1.0 | -130.91 | 0.86% |
| space_41_energy_40 | 41 | 40 | 1.0 | -131.59 | 1.39% |

Interpretation: `dt=1 ns` is already adequate for this case; the remaining
baseline-grid bias is around `1-2%` for the current finite-phonon setup.

Fixed-`nbar` readout-heating probe:

- Output: `outputs/prelim_readout_heating_probe/summary.csv`
- Response table: `outputs/prelim_readout_heating_probe/resonator_shifts.csv`
- Bath: `7 mK`
- `D0 = 20 um^2/ns`, `tau_l = 1 ns`
- Fixed total source: `6.264e11 QP/s` on the `NX=11` probe grid
- Grid: `NX=11`, `NE=101`, `dt=1 ns`, `t_max=1500 ns`
- Readout mode: resonator 1 at `5.143 GHz`; `c_phot = 1e-9 ns^-1`
- Probe cases reached `t_max` rather than the strict steady-state tolerance, so
  this is a sign/magnitude comparison rather than a final calibrated sweep

Relative to `nbar=0`, the fixed-`nbar` readout channel made the predicted
frequency shifts less negative and increased the quasiparticle-only `Q_i`:

| peak nbar | delta f change across modes | Q_i change across modes |
|---:|---:|---:|
| `1e5` | `+1.0%` to `+1.3%` | `+4.6%` to `+5.8%` |
| `1e6` | `+6.4%` to `+7.2%` | `+27%` to `+33%` |

Queued deeper fixed-`nbar` overnight sweep:

- Script: `scripts/run_prelim_readout_heating_overnight.py`
- Planned output: `outputs/prelim_readout_heating_overnight/`
- Current launched copy: `/tmp/qpsim_prelim_overnight/outputs/prelim_readout_heating_overnight/`
  because macOS `launchd` does not have permission to read this repo under
  `Documents`; copy CSV outputs back after the run completes.
- Bath: `7 mK`
- Grid: `NX=21`, `NE=101`, `dt=0.5 ns`, `t_max=30000 ns`
- Readout mode for heating channel: resonator 1 at `5.143 GHz`
- Fixed peak `nbar` values: `0`, `1e5`, `1e6`, `1e7`
- Diffusion constants: `D0 = 20, 6, 60 um^2/ns`
- Source rates: `3.132e11`, `6.264e10`, `6.264e11 QP/s` on the
  `NX=21` source-cell calibration
- Phonon escape times: `tau_l = 1, 0.3, 3 ns`
- Runner is resume-safe and wall-time-limited; it prioritizes the nominal
  `D0=20`, source `5e-4/ns`, `tau_l=1 ns` block before broader sweeps.

## Inputs Still Needed

These are the main pieces of experimental or design information still needed to
turn the present simulations into a quantitatively calibrated model.

Device layout and placement:

- The final `.gds` or an exported centerline/region description for the 2D
  layout.
- Exact Al strip start/end coordinates relative to the resonator short.
- Exact junction location on the strip and whether the 1D source should be a
  point source, finite-area source, or distributed over an overlap region.
- Nb/Al overlap geometry, including whether Al sits on top of Nb anywhere and
  the length of any proximity/intermediate region.
- Confirmation that the simulated `100 um` strip is the exact strip being
  measured, not a nominal design length.

Film and material parameters:

- Measured Al thickness for the specific device/wafer.
- Measured or assumed Al gap `Delta_Al`; ideally from tunneling, `T_c`, or a
  resonator fit.
- Al normal-state resistivity or sheet resistance for the film, not just the
  junction.
- Best estimate of Al diffusion constant `D`; current simulations sweep it.
- Best estimate of Kaplan/material parameter `tau_0` for this film.
- Nb thickness, gap, resistivity/sheet resistance, and penetration depth for
  the measured wafer.
- Any evidence for gap inhomogeneity or proximity-induced `Delta_eff` near the
  Nb/Al interface.

Junction and injection calibration:

- Clarify whether `1.5 kOhm` is the total junction normal resistance `R_N` or a
  resistance-area product `R_N A`.
- Measured junction area after fabrication. Nominal area used so far:
  `245000 nm^2 = 0.245 um^2`.
- Full measured junction `I(V)` at the experimental temperature, including
  subgap leakage.
- Bias range and bias stability near `V ~= 2 Delta_Al/e`.
- Any series resistance or filtering that changes the actual junction voltage.
- Dynes broadening or another subgap-leakage parameter for the SIS model.
- Fraction of injected power that creates quasiparticles in the Al strip versus
  escaping promptly as phonons.

Phonon and bath parameters:

- A literature or device-specific prior for phonon escape time `tau_l` for
  `~100 nm` Al on the actual substrate.
- Substrate material, thickness, backside treatment, and mounting details that
  affect phonon escape/reabsorption.
- Whether phonons should be treated as local, laterally diffusing in the film,
  or propagating into the substrate.
- Actual electron/phonon base temperature under bias; the fridge mixing chamber
  temperature is `7 mK`, but the film may be warmer.

Readout and `S21` calibration:

- Measured dark resonator frequencies, `Q_i`, `Q_c`, and loaded `Q`.
- Feedline/coupler parameters or a measured baseline `S21(f)` for each mode.
- Readout powers at the device and uncertainty in line attenuation.
- Resonator photon number calibration.
- Planned voltage sweep values and dwell times.
- Readout-power sweep data, once available, to separate injection effects from
  microwave heating.
- The fitting model to be used for extracting `f_r`, `Q_i`, `Q_c`, and complex
  depth from measured `S21(f)`.

Boundary/interface physics:

- Whether QPs are perfectly reflected/trapped at the Nb interface or whether a
  finite leakage/Andreev/proximity boundary condition is needed.
- Whether any high-energy injection tail can exceed the Nb gap.
- Whether the open end of the Al/Nb series section needs a more detailed
  electromagnetic boundary condition than the ideal quarter-wave current shape.

## Simulation Upgrade Roadmap

Highest-payoff next upgrade:

- Replace the Gaussian source with a voltage-dependent SIS tunneling source.
  Use measured `R_N`, `V`, `Delta_Al`, temperature, and Dynes broadening to
  compute an energy-resolved injection spectrum and total QP/s.

Geometry/current-profile upgrade:

- Parse the final `.gds` and build the Al/Nb section coordinates directly from
  layout.
- Replace the ideal `cos^2(pi s / 2l_R)` profile with a piecewise or EM-derived
  current-density profile.
- Compute an Al-specific kinetic-inductance participation instead of using the
  working scalar `alpha = 0.08`.
- Keep the TeX-derived resonator lengths as the design prior, but compare them
  to measured dark resonator frequencies.

Response/S21 upgrade:

- Convert local nonequilibrium `sigma_1`, `sigma_2` into a perturbed surface
  impedance and resonator pole, not only `delta f_r` and `Q_i`.
- Generate synthetic complex `S21(f,V)` traces using measured or designed `Q_c`
  and feedline background.
- Report the predicted change in resonance depth directly, not only the
  inferred quasiparticle `Q_i`.
- Include extrinsic loss channels so the simulation predicts total measured
  `Q_i`/`Q_l`.

Numerics and convergence upgrade:

- Extend convergence checks over several source rates and `tau_l` values, not
  just the representative `3.132e11 QP/s`, `tau_l=1 ns` case.
- Add a finer nonuniform energy grid near `Delta` and near the injection energy.
- Run a few high-resolution references, e.g. `NX=81`, `NE>=60`, smaller `dt`,
  to bound the remaining discretization error.
- Add automated convergence summaries to every sweep output directory.

Phonon-model upgrade:

- Make phonon escape frequency dependent if literature/device data justify it.
- Add lateral phonon transport or substrate reabsorption when the local
  bottleneck approximation becomes limiting.
- Track energy conservation between injected QP power, QP energy, phonon
  energy, and escaped phonon power.

Interface/proximity upgrade:

- Add a finite Nb/Al boundary condition instead of perfect trapping.
- Include an effective gap profile or proximity section if the layout has a
  substantial Nb/Al overlap.
- Test sensitivity to a small QP leakage probability into Nb.

Readout-heating upgrade:

- Upgrade the fixed-`nbar` readout-heating probe to a self-consistent
  `nbar(P_read, Q_i, Q_c)` loop using measured or designed coupling `Q_c`.
- Decide whether `nbar` should represent peak current at the short or an
  energy-normalized mode occupancy after integrating over the full resonator.
- Use readout-power sweeps to fit or bound `c_phot` independently of the
  junction-injection source.

Fitting workflow upgrade:

- Fit `D`, `tau_l`, `tau_0`, Dynes broadening, injection efficiency, and
  possible `Delta_Al` corrections against measured `S21(f,V,T,P_readout)`.
- Compare voltage sweeps, readout-power sweeps, and bath-temperature sweeps
  jointly so parameters are not overfit to one curve.

## Important Caveats

- Injection is not yet a voltage-dependent SIS tunneling spectrum.
- `tau_l` is swept phenomenologically; it should be tied to literature or a
  device-specific phonon escape estimate.
- Phonons are local in `x`; no substrate/film phonon transport is included.
- Nb/Al proximity and interface leakage are not yet modeled.
- Readout heating is currently only included as a fixed-`nbar` sub-gap photon
  scattering channel; it is not yet a self-consistent readout-power model.
- The `alpha = 0.08` kinetic-inductance fraction is a working value and should
  eventually be replaced with a geometry/material participation estimate.
