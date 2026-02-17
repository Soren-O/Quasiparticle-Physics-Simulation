# Quasiparticle Physics Simulator

Desktop Python app for building, saving, running, and reviewing quasiparticle-inspired diffusion simulations from GDS geometry.

## Features

- Retro-style desktop launcher and menu UI.
- Create setup from GDS layer geometry.
- Boundary assignment by clicking highlighted geometry edges.
- Boundary types: `Reflective`, `Neumann`, `Dirichlet`, `Absorbing`, `Robin`.
- Crank-Nicolson heat/diffusion solver on 2D geometry mask.
- Initial condition presets plus constrained custom Python expression.
- Save/load setup JSON files and save simulation JSON output.
- Live simulation viewer with:
  - 2D heatmap (fixed color scale),
  - total mass-over-time plot.
- Test case generator with 10 intrinsic 1D validation cases and replay viewer with analytic overlay and formula panel.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Or run:

```powershell
.\Install Requirements.bat
```

## Run

```powershell
python app.py
```

Or run:

```powershell
.\Run Quasiparticle Physics Simulator.bat
```

## Data storage

- Setups: `data/setups/*.json`
- Simulations: `data/simulations/*.json`
- Test suites: `data/test_cases/*.json`
