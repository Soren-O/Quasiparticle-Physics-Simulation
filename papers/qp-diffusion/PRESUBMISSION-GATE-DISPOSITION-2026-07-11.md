# qp-diffusion presubmission gate disposition - 2026-07-11

## Authorization and scope

The author explicitly approved the advisory bundle on 2026-07-11 after three
independent AI panels adjudicated B1/C1/C2, M5 plus the abstract, and D1-D7 plus
release/contact questions. The independent adversarial reviewer separately
endorsed the bundle, with the precision caveats incorporated below. This record
describes the resulting manuscript decision; it does not retroactively alter the
historical review records.

- Branch: `fix/gpt-review-2026-07-05`
- Pre-edit HEAD: `703e270`
- Manuscript implementation: `229956b` (`paper: resolve approved presubmission gates`)
- Paper-only scope: no engine file, engine branch, package release, or external
  contact was touched.

## Approved decisions

### Abstract and register (D1, D2, D6)

The protected abstract was replaced in full. The approved version binds the
dirty limit to both reductions, attributes the coherence-factor closure to the
scalar route without assigning that mechanism to the Usadel trace, identifies
one published gap-engineered-trap use of the legacy placement, and says that the
placement is "not selected by either reduction" in the stated sector. It does
not say "microscopically wrong," "in common use," "measurably," or "as it
always is." It distinguishes coupled two-mode from full-matrix extensions.

### B1/C1 and placement (D3)

The prepared quantitative C1 correction remains rejected. The paper does not
print the conditional `7 -> 32 um` result, revise a trap length, assume endpoint
amplitude continuity, or claim that a proximity transfer law has been verified.

The approved main-text caveat instead derives the endpoint drop for a
stationary, one-dimensional, constant-area eliminated layer with the collision
integral and all other kinetic sources or sinks neglected:

```text
f_L^+ - f_L^- = -J_L R_xi(E),
R_xi(E) = integral_layer dy / [sigma_N(y) D_L(E,y)].
```

It defines `R_xi` as a specific spectral resistance, fixes endpoint orientation,
states the common-phase/charge-balanced/no-supercurrent scope, and explicitly
declines to revise the integrated matching or trap-length estimates of
Riwar--Catelani because this resistance has not been shown negligible throughout
the relevant near-edge window.

### Local artifact diagnostic (C2)

The large proposed device-error table and universal three-regime conclusions
remain rejected. The paper includes only the local fixed-energy diagnostic for
legacy placement C:

```text
|v_C| = D_N Delta |partial_x Delta| / [E sqrt(E^2 - Delta^2)],
Pe_E = |v_C| L_Delta / D(E)
     = Delta |delta Delta| / (E^2 - Delta^2).
```

It reports the three checked energy/amplitude triples and immediately states
that `Pe_E` is not a device observable or predicted device error, is nonuniform
at the ideal BCS edge, and needs a broadened spectrum, energy distribution,
collision kernel, and full boundary-value problem before it can inform an
observable. `verify_traces.py` now checks the exact formula, rational energy
coefficients, and the printed three-significant-figure rounding.

### Benchmark claim and captions (M5, D5)

Every reciprocal "self-focusing" claim was replaced. The feedback benchmark is
now described consistently as transport of a separate passive probe excluded
from the gap closure. It demonstrates inward probe drift for the legacy
placements and zero fixed-energy DOS-gradient probe drift for A1; it does not
demonstrate net nonlinear compression of the well-generating population.

Tailored limitations were added only where relevant:

- prescribed drift ramp: 60% is deliberately large for discrimination;
- interface: the idealized K-L face is not a transfer law for the
  coherence-scale Riwar--Catelani proximity crossover;
- feedback: the 5% well is exaggerated, the probe is passive, and inward probe
  drift is not net compression.

The uniform-gap caption was left unchanged.

### Verification sentence, contact, and release (D4, D7)

- The old SM sentence claiming a verified quantitative trap-matching transfer
  law was not inserted. No `verify_trap_matching.py` claim ships.
- Riwar/Catelani were not contacted. The panel recommendation is no correction
  notice now; an optional neutral courtesy note may accompany a near-final PDF
  later, without a quantitative device correction.
- No PyPI upload was attempted. Package publication remains deferred until the
  engine integration is landed and an actual release is built and tested; the
  present `0.1.0.dev0` metadata is not a `0.0.1` release candidate.

## Independent checks and validation

Three focused post-edit reviews separately returned PASS for the abstract/M5
wording, the eliminated-proximity-layer caveat, and the C2 diagnostic/captions.
A final whole-diff review found two minor precision issues; both were fixed
before commit by making the no-source/no-sink hypothesis explicit and replacing
a loose percentage guard with exact half-unit-in-the-last-place rounding checks.

- All seven `verify_*.py` scripts: **7/7 PASS** in 266.5 s with
  `A:/Einstein/Documents/qp-diffusion-paper/.venv/Scripts/python.exe` and
  `PYTHONUTF8=1`.
- Final focused `verify_traces.py`: **ALL PASS** after the last review fixes.
- PDFs: `paper.pdf` 56 pages; `supplement.pdf` 52 pages.
- Logs: zero undefined references/citations, zero undefined controls, and zero
  overfull boxes. REVTeX's known stuck-float warnings remain cosmetic; every
  float is present.
- Visual inspection passed for the abstract, channel dictionary repagination,
  C1/C2 equations and numeric display, benchmark setup/body/conclusion, and all
  three revised figure-caption pages.

## Remaining items

The human gates covered by this bundle are resolved. The following are not
manuscript physics findings and were deliberately not executed:

- PyPI/package release work;
- external contact with Riwar/Catelani;
- the optional journal-PDF check that converts the verified arXiv Eq. (46)
  pinpoint for `riwar2019` into its published appendix number;
- any quantitative trap-length revision, unless a proximity-aware transfer law
  and kinetic eigenmode calculation are derived later.

No known blocker, major, or minor physics finding remains in the paper's
explicitly delimited sector as a result of the completed adversarial reviews.
