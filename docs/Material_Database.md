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
| `D_0` | μm²/ns |  | Normal-state diffusion. |
| `v_F` | m/s |  | Fermi velocity. |
| `rho_F` | J⁻¹ m⁻³ |  | Single-spin DOS at Fermi level. |
| `sound_velocity_longitudinal` | m/s |  | `s_L`. |
| `sound_velocity_transverse` | m/s |  | `s_T`. |
| `sound_velocity_debye` | m/s |  | `s_D`. Derived from `s_L` and `s_T` if both supplied via `s_D⁻³ = ⅓(s_L⁻³ + 2 s_T⁻³)`; otherwise must be provided explicitly. |
| `film_thickness` | nm |  | Film thickness. |
| `substrate` | `Substrate` |  | Nested mapping in YAML. |
| `substrate_transmission_eta` | — |  | `η` for `acoustic_escape_tau_l`. |

## BCS calibration caveat

`qpsim.physics.gap_equation.calibrate_gap` derives `Δ_eq` from `T_c`
using the BCS weak-coupling ratio `Δ₀/(k_B T_c) = 1.764`. Materials
that depart from weak coupling won't satisfy this ratio:

| Material | `Δ₀/(k_B T_c)` from YAML | Δ from `calibrate_gap(T_c)` | YAML `Delta_0` | Δ error |
|---|---|---|---|---|
| Al  | 1.770 | 179 μeV | 180 μeV | 0.4% |
| TiN | 1.805 | 685 μeV | 700 μeV | 2% |
| Nb  | 1.882 | 1407 μeV | 1500 μeV | 7% |

For Nb (and any other strong-coupling material), pass `Delta_0`
downstream rather than recomputing it from `T_c` through the
calibrator.

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
2. Sanity-check the BCS ratio (above) so users know whether to trust
   `calibrate_gap` or to pass `Delta_0` explicitly.
3. If you compute `τ_l` from acoustic escape, set `film_thickness`,
   `substrate_transmission_eta`, and one of the sound velocities (or
   both `s_L` and `s_T` so the Debye average is derived).

## See also

- `Phonon_Escape_Time.md` — derivation of `τ_l ≈ 4d/(η s)`,
  acoustic-mismatch model for `η`, and the Kaplan 1976 vs 1979
  glossary.
- `Phonon_Model_Decisions.md` D5 — sound-velocity convention.
