# Quasiparticle Physics Simulator

Desktop Python application for creating, saving, running, and reviewing quasiparticle-inspired diffusion simulations on 2D geometries. The current physics backend solves a 2D heat/diffusion equation with a Crank-Nicolson integrator.

## Current Scope

- Interactive desktop workflow (Tkinter + Matplotlib).
- Geometry from GDSII import or intrinsic built-in geometry.
- Edge-based boundary condition assignment.
- Numerical simulation and replay.
- Analytic-vs-numerical validation suites across multiple geometries.

## Main UI Flow

### Start screen
- `Run Quasiparticle Physics Simulator`
- `Install Requirements.bat`

### Launch menu
- `Create New Setup`
- `Load Saved Setup`
- `View Saved Simulations`
- `Generate Test Simulations`
- `View Test Simulations`

### Setup editor
- Import `.gds` geometry (layer picker shown when multiple layers exist).
- Rasterize geometry to mesh and validate connectivity (exactly one connected region).
- Hover and click edges to assign boundary conditions.
- Configure solver parameters (`D`, `dx`, `dt`, `total_time`, `store_every`).
- Configure initial conditions.
- Save setup JSON.
- Run simulation.

## Boundary Conditions

Supported boundary kinds:
- `Reflective` (insulated / zero flux)
- `Neumann` (non-zero outward gradient, value required)
- `Dirichlet` (fixed value, value required)
- `Absorbing`
- `Robin` (mixed; value required, aux value optional)

Simulation is blocked until all edges are assigned.

## Initial Conditions

Built-in initial condition modes:
- `gaussian` (`amplitude`, `x0`, `y0`, `sigma`)
- `uniform` (`value`)
- `point` (`value`, `x0`, `y0`)
- `custom`

Custom mode uses a constrained scaffold:
- Fixed function shape: `user_expression(x, y, params)`
- Editable body text
- JSON dictionary for custom parameters

## Simulation Viewer

- 2D heatmap replay over geometry.
- Mass-over-time trace.
- Fixed color scaling across frames.
- Play/pause and frame navigation.

In the setup editor, `View Live Simulation On Run` is a checkbox:
- Checked: running simulation auto-opens the viewer when the run completes.
- Unchecked: run completes with standard completion feedback only.

## Test Simulation System

The test suite compares solver output (`simulated`) with analytic references (`analytic`) for benchmark cases.

### Generation and loading UX

- `Generate Test Simulations` runs in a background thread with a busy/progress dialog.
- `View Test Simulations` first loads a lightweight manifest, then opens a geometry landing page.
- Geometry cases are lazy-loaded when the user opens a geometry card.

### Geometry groups

| Geometry ID | Title | Cases | Viewer Mode |
| --- | --- | ---: | --- |
| `strip_1d_effective` | Effective 1D Strip | 10 | 1D line overlay (simulated vs analytic) |
| `rectangle_2d` | 2D Rectangle | 6 | Side-by-side heatmaps |
| `polygon_donut` | Polygon Donut | 4 | Side-by-side heatmaps |

Total generated cases: **20**.

Current default generator parameters:
- `dx = 1.0`
- `D = 25.0`
- `dt = 0.05`
- `total_time = 8.0`
- `store_every = 2`

Heatmap test viewers include a persistent density legend/colorbar.

## Data Storage

- Setups: `data/setups/*.json`
- Simulations: `data/simulations/*.json`
- Test suite manifest: `data/test_cases/test_suite_<id>.json`
- Test suite geometry payloads: `data/test_cases/test_suite_<id>/<geometry_id>.json`

The test suite format is split-manifest to keep initial load fast and support lazy loading.

## Install

Manual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Script:

```powershell
.\Install Requirements.bat
```

## Run

Manual:

```powershell
python app.py
```

Script:

```powershell
.\Run Quasiparticle Physics Simulator.bat
```

## Key Project Files

- `app.py` - app entrypoint.
- `qpsim/ui/main_app.py` - main UI, setup editor, viewers, loading dialogs.
- `qpsim/solver.py` - 2D Crank-Nicolson solver and boundary operator assembly.
- `qpsim/geometry.py` - GDS import/rasterization and edge extraction.
- `qpsim/test_cases.py` - analytic benchmark case generation.
- `qpsim/storage.py` - JSON persistence and split test-suite format.
- `App_Description.txt` - detailed product/application specification.

## References

Local files:
- `References/2DCrank.pdf`
- `References/ChapterSimulation.txt`
- `References/ChapterAppendix.txt`
- `References/Quasiparticle_Master_Equation.pdf`

Analytic PDE references used for benchmark case construction:
- https://math.stackexchange.com/questions/4720247/1d-heat-equation-with-insulated-boundary-conditions-greens-function
- https://math.stackexchange.com/questions/4723649/1d-heat-equation-with-insulated-boundaries-with-dirac-delta
- https://www.math.toronto.edu/ivrii/PDE-textbook/Chapter4/S4.1.html
- https://www.math.toronto.edu/ivrii/PDE-textbook/Chapter6/S6.1.html
- https://www.math.toronto.edu/ivrii/PDE-textbook/Chapter3/S3.2.html
- https://www.math.toronto.edu/ivrii/PDE-textbook/Chapter4/S4.2.html
