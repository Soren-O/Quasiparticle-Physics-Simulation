# Material Database

The `Material` dataclass and the YAML-backed loader live in
`qpsim.materials.database`. Substrate descriptors live in
`qpsim.materials.substrate`. YAML files live in
`qpsim/materials/data/`.

## Loading

```python
from qpsim.materials import load_material, list_materials

list_materials()           # ['Al', 'Nb', 'TiN']
mat = load_material("Al")  # Material(...)
```

`load_material` reads `qpsim/materials/data/{name}.yaml` by default;
pass `database_dir=` to point at a user-curated directory.

## `Material` fields

| Field | Unit | Required | Notes |
|---|---|---|---|
| `name` | — | ✓ | Display string. |
| `Delta_0` | μeV | ✓ | T=0 superconducting gap. |
| `T_c` | K | ✓ | Critical temperature. |
| `tau_0` | ns | ✓ | Kaplan e-ph characteristic time. |
| `tau_s`, `tau_r` | ns |  | Default to `tau_0` if omitted. |
| `tau_0_phonon` | ns |  | Kaplan 1976 Table II `τ₀^ph`. Required by `kaplan_pair_breaking.tau_PB_inverse_Hz`. |
| `tau_0_pb_ns` | ns |  | F&C Eq. 12 phonon-side characteristic time. Required by the default dynamic-phonon kernel; omit only when using thermal phonons or the explicit legacy-kernel opt-out. |
| `D_0` | μm²/ns |  | Normal-state diffusion. |
| `v_F` | m/s |  | Fermi velocity. |
| `rho_F` | eV⁻¹ m⁻³ |  | Conventional single-spin DOS at the Fermi level; density observables convert qpsim's µeV integration measure to eV explicitly. |
| `sound_velocity_longitudinal` | m/s |  | `s_L`. |
| `sound_velocity_transverse` | m/s |  | `s_T`. |
| `sound_velocity_debye` | m/s |  | `s_D`. Derived from `s_L` and `s_T` if both supplied via `s_D⁻³ = ⅓(s_L⁻³ + 2 s_T⁻³)`; otherwise must be provided explicitly. |
| `film_thickness` | nm |  | Film thickness. |
| `substrate` | `Substrate` |  | Nested mapping in YAML. |
| `substrate_transmission_eta` | — |  | `η` for `acoustic_escape_tau_l`. |

`rho_F` is stored per eV. Custom material files and direct
density-observable callers that carry a per-µeV value (for example Al
`1.74e22`) must migrate it by multiplying by `1e6` (Al `1.74e28`). Values
in the legacy material-scale band fail loudly instead of producing a density
that is silently `1e6` low; version-1 Web UI setup files are migrated
automatically when loaded.

## BCS calibration caveat

`qpsim.physics.gap_equation.calibrate_gap` treats `T_c` as the authoritative
pairing-scale input. It derives `1/λ` from the finite-cutoff gap equation
linearized at `T_c`, then solves the nonlinear equation below `T_c`. This makes
the modeled gap close continuously as `T -> T_c`. With the default cutoff
`ω_D/(k_B T_c) = 100`, the implied zero-temperature ratio is
`Δ₀^BCS/(k_B T_c) = 1.76374` (close to the infinite-cutoff value 1.76388).

Measured materials can depart from that weak-coupling ratio:

| Material | `Δ₀/(k_B T_c)` from YAML | Δ from `calibrate_gap(T_c)` | YAML `Delta_0` | Δ error |
|---|---|---|---|---|
| Al  | 1.770 | 179 μeV | 180 μeV | 0.4% |
| TiN | 1.805 | 684 μeV | 700 μeV | 2% |
| Nb  | 1.882 | 1406 μeV | 1500 μeV | 7% |

Passing the material value records it for provenance and comparison:

```python
cal = calibrate_gap(T_c=mat.T_c, T_bath=T_bath, Delta_0=mat.Delta_0)
print(cal.delta_0_reference)  # measured material value
print(cal.delta_0_bcs)        # prediction of this weak-coupling kernel
```

`Delta_0` is diagnostic-only in `calibrate_gap`: it does not change `1/λ` or
`Δ_eq`. The measured `Delta_0` remains the material scale used elsewhere for
grid construction and observable normalization. A single-coupling
weak-coupling BCS kernel cannot reproduce an arbitrary measured `Delta_0` and
an independently measured `T_c` simultaneously. Anchoring the coupling at the
measured gap would imply a different critical temperature and, if the declared
`T_c` were then imposed as a hard cutoff, leave a finite gap immediately below
it. Reproducing both measurements requires an explicit strong-coupling or
phenomenological gap model rather than a second anchor in this kernel.

## Substrate

```python
@dataclass
class Substrate:
    name: str
    density: float          # kg/m³
    sound_velocity: float   # m/s (substrate-side, for AMM η)
```

Used by the acoustic-mismatch model to derive the substrate-transmission
coefficient `η` when not supplied directly. `qpsim/materials/data/`
ships starter values for Al/Al₂O₃, Nb/Si, and TiN/Si.

## Adding a new material

1. Drop a YAML file at `qpsim/materials/data/{name}.yaml` with at
   minimum `name`, `Delta_0`, `T_c`, `tau_0`. Add the optional fields
   you need for the physics you're solving.
2. Sanity-check the BCS ratio (above). A material whose measured ratio differs
   materially from `1.76374` needs a strong-coupling or phenomenological gap
   model for quantitative temperature-dependent-gap work; passing `Delta_0`
   to `calibrate_gap` records the discrepancy but does not change the BCS
   coupling.
3. If you compute `τ_l` from acoustic escape, set `film_thickness`,
   `substrate_transmission_eta`, and one of the sound velocities (or
   both `s_L` and `s_T` so the Debye average is derived).

## See also

- `Phonon_Escape_Time.md` — derivation of `τ_l ≈ 4d/(η s)`,
  acoustic-mismatch model for `η`, and the Kaplan 1976 vs 1979
  glossary.
- `Phonon_Model_Decisions.md` D5 — sound-velocity convention.
