# qpsim deep re-audit — 2026-07-13

## Scope and method

This review re-audited the repository after the integrated 2026-07-12 fixes.
It covered the M25 rate-equation path, BCS observables and moving-gap
transport, collision and phonon kernels, transient and spatial solvers,
device composition, grid construction, Web UI execution, persistence, and
validation structure.

The review combined ten read-only subsystem agents with independent numerical
reproduction of the highest-impact claims, followed by an integration review
of the combined patch.  The checks included source-to-paper comparison,
manufactured finite-volume solutions, equilibrium and detailed-balance
probes, adversarial NaN/infinity inputs, branch-solver probes over the paper's
temperature window, targeted regression suites, Ruff, strict mypy, and the
seven symbolic verifiers in `papers/qp-diffusion`.

Three standards were kept separate throughout:

1. whether the continuous equation matches the paper;
2. whether the discrete implementation converges to that equation; and
3. whether a stored figure baseline is anchored to external paper data rather
   than qpsim's own earlier output.

## Bottom line

The central physics conclusion is unchanged.  The A1 `(p, q) = (1, 0)`
diffusion operator, the structure of the M25 moment equations, the collision
integrals, the ideal-BCS spectral identities, and the paper's corrected
nonadiabatic derivation remain sound.  The fixed-gap equilibrium and
steady-state pipelines are still the most thoroughly validated paths.

This re-audit nevertheless found a second-generation cluster of defects around
the preceding fixes.  The most important were a miscalibrated M25 residual
acceptance layer, incomplete propagation of exact BCS cell measures, and
convergence tests that measured an under-relaxed update rather than the raw
fixed-point residual.  Integration review also found persistence races,
snapshot-cadence errors, and local-gap observable inconsistencies that were
not visible in subsystem tests.  These findings have been fixed and covered
by regressions on the re-audit branch.

## Findings and fixes

### Theme A — M25 residual acceptance after rate normalization

The earlier `Gamma = Gamma_tilde / N_CP(R)` normalization correction changed
the physical residual scale without re-tuning the solver's absolute gates.
The old scalar `1e-14 Hz` default was simultaneously too strict for the
probability row's floating-point cancellation floor and far too loose when
used as a bypass ceiling for the density rows.

The defect was directly reproduced in the Fig. 3a temperature window:

| Temperature | Rejected computed residual | Previous tolerance | Independently recovered true-root residual |
| --- | ---: | ---: | ---: |
| `0.080 K` | `4.55e-13 Hz` | `1e-14 Hz` | `1.06e-22 Hz` |
| `0.100 K` | `9.09e-13 Hz` | `1e-14 Hz` | `8.27e-25 Hz` |
| `0.110 K` | `1.82e-12 Hz` | `1e-14 Hz` | `7.44e-24 Hz` |

Thus the documented strict solver could reject a real root across roughly
`0.075–0.135 K`.  Conversely, the old `accept_lm_convergence` ceiling of
`1 Hz` was about 8–14 orders of magnitude above the relevant residual scale;
probes found accepted pseudo-roots with `x_L` wrong by approximately 7%, 17%,
and 24%.

The acceptance layer now uses a row-wise, source-scaled and
backward-error-aware gate:

```text
|R_i| <= max(
    1e-14,
    residual_tol_relative * source_i
      + 64 * eps * sum_j(abs(term_ij)),
)
```

The source is defined per physical row.  The final term admits only the
floating-point cancellation granularity of that row, so a large probability
flux cannot loosen a density equation.  Every candidate path—strict solve,
status-stall compatibility path, LM/least-squares helpers, root relocation,
and multi-seed branch selection—must pass this same physics gate.  Solver
status can no longer bypass it, and the residual is recomputed from the exact
state returned to the caller.  The hintless branch driver's failure modes are
also distinguished instead of reporting every missing cross-check state as a
branch disagreement.

The formerly failing paper-window points now solve, while deliberately
injected slope pseudo-roots are rejected.  M25 coefficient arrays are copied
and made read-only, region moment bands use exclusive boundaries and exact
BCS cell measures, and region temperatures/gaps are checked against the M25
parameter bundle before evaluation.

### Theme B — convergence must test the raw fixed-point map

Three loops declared convergence from the under-relaxed step,
`alpha * (G(x) - x)`, rather than the true fixed-point residual
`G(x) - x`.  Lowering the relaxation factor therefore loosened the answer by
`1 / alpha`, the opposite of the parameter's intended effect.  A direct
Picard probe was 19.8 times too loose at `mixing = 0.05`; the gap loop could
report convergence with a roughly 1% raw gap error, and the photon-number
loop was measured 2.02% from its raw fixed point.

The Picard, self-consistent-gap, and `nbar` loops now apply relaxation only to
the update and test convergence on the unrelaxed map residual.  Regression
tests vary the relaxation factor and assert the same physical tolerance.
Steady-state phonon iteration already used the correct pattern and served as
the reference implementation.

### Theme C — exact BCS edge measure throughout the pipeline

For ideal BCS spectra, midpoint `rho(E) * dE` is inaccurate near the
integrable gap-edge singularity.  The exact measure of a cell is

```text
sqrt(E_hi^2 - Delta^2) - sqrt(E_lo^2 - Delta^2),
```

with the bounds clipped to represented above-gap support.  The previous fix
used this measure in `x_qp`, `n_qp`, and `sigma_1`, but it silently omitted
the interval `[Delta, E_min_edge]` when a grid started above the gap.  At
`energy_min_factor = 1.01`, the measured `x_qp` was only 34.0% of the
gap-covering result at `0.2 K` and 47.1% at `0.1 K`; even the former unit-test
grid at `1.001` was low by 11.0% and 15.5%, respectively.  Corresponding
`sigma_1` errors were 39–51%, making passive `Q_i` 62–102% too high.

Pure-BCS observables now fail loudly unless the reconstructed first cell edge
covers `Delta`; no unrepresented occupation is invented.  Web and experiment
grid builders use sub-gap room where self-consistent or edge-sensitive
physics requires it.  Cell geometry validation also rejects gaps, overlaps,
and inconsistent center/width pairs.

The same exact measure is now propagated to the remaining consumers:

- `sigma_2` uses an endpoint-removing sine-squared change of variables.  This
  removes the prior 1.72% error at 500 bins and 2.72% error at the 200-bin
  preliminary-script setting.
- M25 junction moments use exact ideal-BCS cell measures rather than
  `sum(rho(center) * dE)`, removing a roughly 0.9% resolution-dependent
  normalization mismatch when device output is read through the corrected
  observables.
- Moving-gap transport maps finite-volume mass exactly along frozen normal-
  state-energy `xi` characteristics.  Since `E = sqrt(xi^2 + Delta^2)` and
  `rho(E) dE = dxi`, overlap of old and new `xi` cells is the characteristic
  solution of the pure-BCS conservative spectral-flow equation.  It replaces
  the diffusive/non-monotone TVD remap that remained 1.8–9.7% away from the
  exact characteristic solution over tested resolutions despite conserving
  total mass.

The exact moving-gap remap diagnoses the finite upper boundary explicitly.
If more than `0.1%` of quasiparticle mass would leave through `E_max`, the
update raises and asks for a larger energy domain.  A smaller but resolvable
tail is conservatively retained in the highest represented active cell and
warns above a `1e-9` relative fraction; it may not overflow `f <= 1`.
Manufactured frozen-`xi` profiles now test both accuracy and conservation.

### Theme D — fail-closed physical and numerical contracts

Top-level NaN gates had been hardened, but several primitives could still
turn invalid inputs into apparently benign output.  The review added finite,
shape, range, and physical-domain checks at the library boundaries, including:

- thermal phonon occupation and spectral-context construction;
- gap calibration/solution, materials, substrate, and phonon escape inputs;
- photon channel strengths and time-dependent drive values;
- steady-state relative-change helpers and `nbar` controls;
- transient, spatial, and moving-gap states, time steps, tolerances, and
  cadence controls; and
- grid factors and integer bin counts (including rejection of booleans).

NaN can no longer compare false and masquerade as zero change or a converged
state.  Invalid states now fail at the boundary that owns the contract rather
than later through accidental behavior in SciPy, SuperLU, clipping, or JSON
serialization.

### Phonon frequency mapping and Kaplan correction domain

Two phonon issues preserved equilibrium detailed balance and were therefore
invisible to equilibrium-only tests:

- absolute `1e-12` frequency rounding split a single physical transition into
  twin bins on non-binary-exact grids (601 of 2401 bins in one ordinary
  probe); and
- the analytic Kaplan `S_+` correction was applied where the pair
  anti-diagonal lay outside the represented quasiparticle domain.  In those
  truncation-starved high-frequency bins it could amplify the discrete result
  by as much as roughly 1800 while multiplying forward and reverse terms
  equally.

Frequency grouping now uses a grid-aware canonical map, and the analytic
Kaplan correction is restricted to the represented pure-BCS pair-breaking
domain for which it was derived.  Detailed-balance validation now exercises
all represented pair-breaking bins instead of masking the generation and
recombination region.

### Transient, spatial, Web UI, and runner integration

The integration pass found defects that no individual subsystem owned:

- time-varying flux was sampled at the substep's left edge, reducing the
  nominal ETD2 drive treatment to first order; it is now sampled at the
  midpoint;
- transient and spatial snapshot logic could skip requested cadence points or
  extrapolate beyond the terminal state when `dt` exceeded the snapshot
  interval; all crossed cadence points now use bounded dense interpolation;
- spatial `x_qp` used one global gap even when a local gap profile was active;
  its numerator now uses each cell's exact local BCS measure while retaining
  the material reference gap in the conventional denominator;
- Crank–Nicolson clipping could silently change conserved density on an
  unresolved front; small changes warn and a change above `0.1%` raises;
- smooth local-gap profiles combined with a single sharp-interface
  conductance are rejected unless the profile contains exactly one step,
  avoiding an ambiguous interface model;
- a failed NPZ write could escape the runner's failure transition;
- terminal jobs accumulated indefinitely in the in-memory registry;
- Windows file sharing could turn deletion during download into a server
  error or leave a zombie run directory; deletion is now staged safely and
  makes the manifest disappear first; and
- concurrent runner paths could overwrite a terminal manifest with stale
  `running` state.  Per-job locking and a terminal-state guard now make the
  state transition monotone.  Malformed/non-object manifests are treated as
  unreadable rather than trusted.

## Validation improvements

The green suite had structural blind spots, so tests were added or strengthened
to pin the risky behavior rather than merely the current output:

- M25 tests cover the former `0.075–0.135 K` death valley, source-scaled row
  gates, status stalls, rejected pseudo-roots, final-state residuals, branch
  cross-checks, immutable coefficient inputs, and state/parameter consistency;
- observable tests cover exact adaptive references, BCS edge coverage, and
  `sigma_2` endpoint convergence;
- moving-gap tests use manufactured frozen-`xi` solutions, finite-`E_max`
  tails, conservation, bounds, and grid refinement;
- relaxation tests check the raw map independently of the mixing factor;
- primitive-contract tests inject NaN, infinity, invalid dimensions, and
  out-of-domain physical values;
- pair-breaking detailed balance is exercised over the full represented
  domain rather than a mask that removed the terms under test; and
- runner tests cover write failures, malformed manifests, concurrent stale
  status, Windows-style deletion conflicts, registry retirement, and exact
  snapshot cadence.

Passing the Fischer CSV tests still establishes regression stability, not
independent agreement with the paper.  That distinction remains explicit.

## Paper and material claims adjudicated

### Recombination-gap convention

The claim that `_tau_R_inverse` should use a mean gap instead of `Delta_R` was
checked against the current manuscript, not only the repository transcription.
Appendix D, Eq. D11 of
[arXiv:2408.17218v2](https://arxiv.org/pdf/2408.17218v2) uses `Delta_R`.
The implementation's `Delta_R` convention is therefore correct for the cited
paper and was not changed.

### TiN density of states

The database value near `3.8e28 eV^-1 m^-3` is not an obvious unit typo.
Gao et al.'s TiN measurement reports approximately
`3.9e10 eV^-1 micrometre^-3`, equivalent to `3.9e28 eV^-1 m^-3`; see the
[NIST-hosted primary paper](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=911819).
The YAML comment now cites the scale while noting that it is film- and
disorder-dependent, so it should not be treated as a universal TiN constant.

### One-mode effective phonon temperature

A single mode supplies one occupation value.  If both an effective amplitude
and a temperature are free, fitting that one scalar is underdetermined; there
is no unique effective temperature without another constraint.  The API
therefore continues to reject the one-mode fit rather than returning an
arbitrary inverse-Bose value that silently assumes a fixed amplitude.

## Remaining limitations

### Unrepresented support in self-consistent gap solves

`solve_gap` cannot reconstruct an occupation that was never represented on
the incoming energy grid.  If a candidate gap lies below the reconstructed
lower support, the missing interval contains unknown nonequilibrium `f`, not a
value that a general solver may safely invent.  The solver now warns, and
high-temperature or strongly gap-suppressed callers must provide sub-gap grid
room, an adaptive/remapped distribution, or an explicit extrapolation model.
This is especially important near `T_c`; it is a domain-contract limitation,
not a remaining quadrature singularity.

### External anchoring of figure validation

Several Fischer figure CSVs remain self-pinned qpsim regression baselines
rather than digitized paper data.  They are useful for detecting code drift,
but regeneration after a numerical fix cannot by itself prove paper fidelity.
An independent digitized-data comparison and at least one paper-parameter
kinetic solve should remain a separate validation deliverable.

### Bounded discretization/model approximations

- The Robin spatial boundary uses a half-cell, first-order boundary
  approximation.  It is not used by the published figure pipelines and
  should be convergence-tested before precision boundary studies.
- Widths inferred only from centers are necessarily ambiguous at joins
  between piecewise grid blocks.  Callers with such grids should supply the
  intended cell widths explicitly.
- The spatial backend can carry a local gap profile in transport and
  observables, while some collision updates still use a shared spectral
  context.  Strongly varying local-gap collision physics therefore remains a
  bounded model approximation rather than a fully local Usadel treatment.
- Dynes-broadened moving-gap flow remains intentionally unsupported until the
  corresponding complex spectral characteristic is derived.

## Verification status

- Focused combined regression gate: **929 passed, 15 warnings**.
- Independent targeted reviewer gate: **288 passed**.
- Symbolic paper verification: **all seven scripts passed**.
- Repository-wide default gate: **1060 passed, 15 slow tests deselected,
  15 expected warnings**.

The focused counts are not assumed to be disjoint.  The repository-wide count
is from the final combined working tree; the earlier 888-pass baseline is
reported only as historical context and was not copied forward.

## Reproduction commands

```text
pytest -q -m "not slow"
pytest tests/services/test_rate_equation.py tests/devices/test_m25_junction.py -q
pytest tests/observables tests/physics tests/collisions -q
pytest tests/backends/test_t3_transient.py tests/solvers/test_spectral_flow_tvd.py -q
pytest tests/backends/test_t3_spatial_1d.py tests/webui tests/grid -q
make -C papers/qp-diffusion verify PY=python
ruff check qpsim tests validation scripts
mypy qpsim
git diff --check
```

Slow/manual Fischer sweeps remain separate because some are multi-hour
validation targets; when run, their status should be recorded independently
from the default gate.
