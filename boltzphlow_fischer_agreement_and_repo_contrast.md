# BoltzPhlow vs Fischer-Catelani (2023) and Current Repo Dynamics

## Scope
This note checks whether:
- `References/BoltzPhlow_Documentation.pdf`
- `References/Fischer et al., Nonequilibrium QP Distribution in SC Resonators (2023).pdf`

agree on scattering/recombination handling, and then contrasts that with this repo's implementation.

## 1. Agreement check between BoltzPhlow documentation and Fischer-Catelani 2023

### Summary verdict
They agree on the core scattering/recombination structure.

### What matches
1. Master kinetic equation structure.
- Fischer-Catelani writes a quasiparticle kinetic equation in the form
  `df(E)/dt = St_phon[f,n] + St_phot[f,n_gamma]` (their Eq. (1) form), with `St_phon` containing phonon-mediated quasiparticle processes.
- BoltzPhlow documentation presents the same local collision picture for quasiparticles with a phonon collision integral, and also separates photon-mediated terms.

2. Same decomposition of phonon collision physics.
- Both split phonon-mediated kinetics into:
  - quasiparticle-number-conserving scattering (phonon emission/absorption), and
  - quasiparticle-number-changing recombination and pair-breaking.
- In Fischer this appears as `St_phon = St_phon,sp + St_phon,st + St_phon,r + St_phon,PB`.
- In BoltzPhlow this appears as `S_phon,coll = S_phon,sc + S_phon,r + S_phon,pb` (same physical partitioning, different notation granularity).

3. Same kernel construction ingredients.
- Both use a kernel prefactor proportional to `1/(tau_0 * T_c^3)` (equivalently `1/(tau_0*(k_B T_c)^3)` depending on unit convention).
- Both use BCS density of states/coherence factors and phonon occupation factors for emission/absorption.

4. Same recombination and pair-breaking coupling logic.
- Recombination and pair-breaking are treated as linked inverse processes through phonon occupation.
- Both formulations explicitly include a pair-breaking channel (not only recombination).

## 2. Contrast with current repo implementation

The current repo captures the same core scattering/recombination kernel physics, but implements a reduced model compared with the full coupled quasiparticle+phonon kinetic system in Fischer/BoltzPhlow.

### What the repo matches well
1. Scattering kernel form (`qpsim/solver.py`, `scattering_kernel`).
- Uses
  `K_s(E_i,E_j) ~ (1/tau_s) * (E_i-E_j)^2 / (k_B T_c)^3 * coherence * N_p(E_i-E_j)`
- Includes emission vs absorption asymmetry through `N_p` and sets diagonal `K_s(E,E)=0`.

2. Recombination kernel form (`qpsim/solver.py`, `recombination_kernel`).
- Uses
  `K_r(E_i,E_j) ~ (1/tau_r) * (E_i+E_j)^2 / (k_B T_c)^3 * coherence * N_p(E_i+E_j)`
- Includes spontaneous emission limit at `T_bath = 0`.

3. Pauli blocking and scattering in/out structure.
- Collision update uses `f = n/rho` and `max(1-f,0)` factors in scattering-in/out terms, matching the same physical blocking structure.

4. Thermal balance closure for recombination.
- Uses `G_therm = 2 * n_eq * dE * (K_r @ n_eq)` to enforce equilibrium stationarity for recombination channel.

### Main differences from full Fischer/BoltzPhlow treatment
1. No dynamic phonon kinetic equation.
- Fischer/BoltzPhlow treat phonon distribution dynamics explicitly (with thermalization and nonequilibrium phonon effects).
- Repo currently uses a fixed bath-temperature phonon occupation in kernels and a fixed-temperature phonon scaffold for output.

2. No explicit pair-breaking collision term in runtime QP update.
- Fischer/BoltzPhlow include explicit `pair-breaking` terms (`St_phon,PB` / `S_phon,pb`).
- Repo runtime collision step currently includes scattering + recombination (+ thermal generation), but not an explicit pair-breaking integral term.

3. No photon collision integral term (`St_phot`) in the collision solver.
- Fischer includes direct photon-driven collision operator in kinetic equation.
- Repo uses external generation `g_ext(E,x,t)` as a source term instead of a full photon collision kernel.

4. Numerical strategy differs.
- Fischer/BoltzPhlow discuss stiff coupled kinetic equations with phonons.
- Repo uses a positivity-preserving time-relaxation collision step (`boltzphlow_relaxation`) and operator-splitting with diffusion.

### Implementation-specific additions in this repo
1. Spatially nonuniform-gap precompute path.
- Per-pixel `K_s`, `K_r`, `rho`, and `G_therm` arrays are precomputed and cached by unique gap value.

2. Strong parity/validation checks.
- Tests include legacy MKID parity for kernels and collision evolution, detailed-balance checks, and equilibrium-stability checks.

## 3. Bottom line
BoltzPhlow documentation and Fischer-Catelani (2023) are consistent on the scattering/recombination framework. The current repo implements that same core kernel-level physics for `K_s` and `K_r`, but with a reduced collision model: fixed phonon bath, no explicit pair-breaking collision operator, and no explicit photon collision operator.
