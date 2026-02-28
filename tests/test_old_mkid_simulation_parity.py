from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from qpsim.geometry import extract_edge_segments
from qpsim.models import (
    BoundaryCondition,
    ExternalGenerationSpec,
    InitialConditionSpec,
    SimulationParameters,
)
from qpsim.precompute import precompute_arrays
from qpsim.solver import (
    build_energy_grid,
    recombination_kernel,
    run_2d_crank_nicolson,
    scattering_kernel,
)


def _import_old_qp_simulator():
    """Import the legacy MKID_Simulation qp_simulator package.

    The legacy repo lives outside this repo root; skip if it's not present.
    """
    # Avoid writing `__pycache__` into the read-only Old/ tree in this workspace.
    sys.dont_write_bytecode = True

    repo_root = Path(__file__).resolve().parents[1]
    old_root = repo_root.parent / "Old" / "MKID_Simulation"
    if not old_root.exists():
        pytest.skip(f"Legacy repo not found at {old_root}")

    sys.path.insert(0, str(old_root))
    import qp_simulator  # type: ignore[import-not-found]

    # Silence legacy module INFO logs during tests.
    logging.getLogger("qp_simulator.qp_simulator").setLevel(logging.WARNING)

    # Patch a small API mismatch in the legacy copy (property vs method call).
    def _create_diffusion_array_fixed(self, energies):  # noqa: ANN001
        delta_vals = self.geometry.delta_at_inner_boundaries  # property array (nx-1,)
        E = energies[None, :]
        delta = delta_vals[:, None]
        arg = 1.0 - (delta / E) ** 2
        D_array = np.where(E > delta, self.D_0 * np.sqrt(np.maximum(0.0, arg)), 0.0)
        self._diff_array = D_array
        return D_array

    qp_simulator.DiffusionSolver.create_diffusion_array = _create_diffusion_array_fixed
    return qp_simulator


def _reflective_line_geometry(nx: int) -> tuple[np.ndarray, list, dict[str, BoundaryCondition]]:
    mask = np.ones((1, nx), dtype=bool)
    edges = extract_edge_segments(mask)
    bcs = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
    return mask, edges, bcs


def _frozen_thermal_phonon_ic_spec(bath_temperature: float) -> InitialConditionSpec:
    return InitialConditionSpec(
        spatial_kind="uniform",
        spatial_params={"value": 1.0},
        energy_kind="dos",
        energy_params={},
        phonon_spatial_kind="uniform",
        phonon_spatial_params={"value": 1.0},
        phonon_energy_kind="bose_einstein",
        phonon_energy_params={"temperature": float(bath_temperature)},
    )


def test_kernels_match_legacy_mkid_simulation_constant_gap() -> None:
    old = _import_old_qp_simulator()

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    ne = 24
    nx = 16
    D0 = 6.0

    tau_s = 400.0
    tau_r = 500.0
    T_c = 1.2
    bath_temperature = 0.1

    E_bins, _ = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    Ks_new = scattering_kernel(E_bins, gap, tau_s, T_c, bath_temperature)
    Kr_new = recombination_kernel(E_bins, gap, tau_r, T_c, bath_temperature)

    # Legacy uses eV internally.
    gap_eV = gap * 1e-6
    material = old.MaterialParameters(
        tau_s=tau_s,
        tau_r=tau_r,
        D_0=D0,
        T_c=T_c,
        gamma=0.0,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=1000,              # dt = 0.001 ns (well below CFL for dx=1, D0=6)
        L=float(nx),          # dx = 1 μm
        T=1.0,
        E_min=energy_min_factor * gap_eV,
        E_max=energy_max_factor * gap_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=lambda x: gap_eV,
        T_bath=bath_temperature,
    )
    sim.initialize("thermal")

    # Legacy kernels are [1/ns/eV]; convert to [1/ns/μeV] for comparison.
    Ks_old = sim.scattering._Ks_array[0] * 1e-6
    Kr_old = sim.scattering._Kr_array[0] * 1e-6

    def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
        denom = max(1e-30, float(np.max(np.abs(a))))
        return float(np.max(np.abs(a - b)) / denom)

    assert _max_rel(Ks_old, Ks_new) < 1e-11
    assert _max_rel(Kr_old, Kr_new) < 1e-11


def test_collision_only_step_close_to_legacy_for_small_dt() -> None:
    old = _import_old_qp_simulator()

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    ne = 18
    nx = 32
    D0 = 6.0

    tau_s = 400.0
    tau_r = 500.0
    T_c = 1.2
    bath_temperature = 0.0

    # Legacy setup
    gap_eV = gap * 1e-6
    material = old.MaterialParameters(
        tau_s=tau_s,
        tau_r=tau_r,
        D_0=D0,
        T_c=T_c,
        gamma=0.0,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=1000,              # dt = 0.001 ns
        L=float(nx),          # dx = 1 μm
        T=1.0,
        E_min=energy_min_factor * gap_eV,
        E_max=energy_max_factor * gap_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=lambda x: gap_eV,
        T_bath=bath_temperature,
    )
    sim.initialize("thermal")

    # Delta-like initial spectrum: only one energy bin occupied.
    k = ne // 2
    n0 = 1e-6
    sim.n_density[:] = 0.0
    sim.n_density[:, k] = n0
    n_new_old = sim.scattering.scattering_step(sim.n_density, sim.n_thermal, sim.sim_params.dt, sim.sim_params.dE)

    # Current setup
    mask, edges, bcs = _reflective_line_geometry(nx)
    _, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    weights = np.zeros(ne, dtype=float)
    weights[k] = 1.0
    initial_field = np.full((1, nx), n0 * dE, dtype=float)

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=D0,
        dt=sim.sim_params.dt,
        total_time=sim.sim_params.dt,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        energy_weights=weights,
        enable_diffusion=False,
        enable_recombination=True,
        enable_scattering=True,
        dynes_gamma=0.0,
        collision_solver="fischer_catelani_local",
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=bath_temperature,
        initial_condition_spec=_frozen_thermal_phonon_ic_spec(bath_temperature),
        freeze_phonon_dynamics=True,
    )

    assert energy_frames is not None
    state_new = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)  # (NE, nx)
    state_old = n_new_old.T  # (NE, nx)

    max_abs = float(np.max(np.abs(state_new - state_old)))
    max_ref = float(np.max(np.abs(state_old)))
    rel = max_abs / max(1e-30, max_ref)
    assert rel < 2e-4


@pytest.mark.parametrize(
    ("bath_temperature", "k", "steps", "profile_kind", "tol"),
    [
        (0.00, 3, 1, "flat", 3e-4),
        (0.10, 7, 6, "gaussian", 6e-4),
        (0.20, 12, 8, "skewed", 9e-4),
    ],
)
def test_fischer_frozen_phonons_collision_only_matches_legacy_fixed_bath(
    bath_temperature: float,
    k: int,
    steps: int,
    profile_kind: str,
    tol: float,
) -> None:
    """Frozen thermal phonons in coupled solver should reproduce legacy fixed-bath collisions."""
    old = _import_old_qp_simulator()

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    ne = 18
    nx = 32
    D0 = 6.0

    tau_s = 400.0
    tau_r = 500.0
    T_c = 1.2
    dt = 0.001

    x = (np.arange(nx, dtype=float) + 0.5) / nx
    if profile_kind == "flat":
        profile = np.full(nx, 1.1e-6, dtype=float)
    elif profile_kind == "gaussian":
        profile = 8e-7 + 2.1e-6 * np.exp(-((x - 0.37) ** 2) / (2.0 * 0.09**2))
    elif profile_kind == "skewed":
        profile = 5e-7 + 1.9e-6 * (0.2 + 0.8 * x**1.5)
    else:
        raise ValueError(f"Unsupported profile kind '{profile_kind}'.")

    # Legacy setup (eV)
    gap_eV = gap * 1e-6
    material = old.MaterialParameters(
        tau_s=tau_s,
        tau_r=tau_r,
        D_0=D0,
        T_c=T_c,
        gamma=0.0,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=1000,
        L=float(nx),
        T=1.0,
        E_min=energy_min_factor * gap_eV,
        E_max=energy_max_factor * gap_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=lambda x_: gap_eV,
        T_bath=bath_temperature,
    )
    sim.initialize("zero")
    sim.n_density[:] = 0.0
    sim.n_density[:, k] = profile

    for _ in range(steps):
        sim.n_density = sim.scattering.scattering_step(
            sim.n_density,
            sim.n_thermal,
            dt,
            sim.sim_params.dE,
        )
    state_old = sim.n_density.T.copy()  # (NE, nx)

    # Current setup (μeV)
    mask, edges, bcs = _reflective_line_geometry(nx)
    _, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    weights = np.zeros(ne, dtype=float)
    weights[k] = 1.0
    initial_field = (profile * dE)[None, :]

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=D0,
        dt=dt,
        total_time=steps * dt,
        dx=1.0,
        store_every=max(1, steps),
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        energy_weights=weights,
        enable_diffusion=False,
        enable_recombination=True,
        enable_scattering=True,
        dynes_gamma=0.0,
        collision_solver="fischer_catelani_local",
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=bath_temperature,
        initial_condition_spec=_frozen_thermal_phonon_ic_spec(bath_temperature),
        freeze_phonon_dynamics=True,
    )
    assert energy_frames is not None
    state_new = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)

    max_abs = float(np.max(np.abs(state_new - state_old)))
    max_ref = float(np.max(np.abs(state_old)))
    rel = max_abs / max(1e-30, max_ref)
    assert rel < tol


def test_diffusion_only_step_matches_legacy_crank_nicolson() -> None:
    old = _import_old_qp_simulator()

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    ne = 12
    nx = 48
    D0 = 6.0

    tau_s = 400.0
    tau_r = 500.0
    T_c = 1.2
    bath_temperature = 0.0

    # Legacy setup
    gap_eV = gap * 1e-6
    material = old.MaterialParameters(
        tau_s=tau_s,
        tau_r=tau_r,
        D_0=D0,
        T_c=T_c,
        gamma=0.0,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=100,               # dt = 0.01 ns
        L=float(nx),          # dx = 1 μm
        T=1.0,
        E_min=energy_min_factor * gap_eV,
        E_max=energy_max_factor * gap_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=lambda x: gap_eV,
        T_bath=bath_temperature,
    )
    sim.initialize("thermal")

    # Single-bin initial spectrum, with a spatial bump.
    k = 2
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    profile = 1e-6 + 2e-6 * np.exp(-((x - 0.33) ** 2) / (2.0 * 0.08**2))
    sim.n_density[:] = 0.0
    sim.n_density[:, k] = profile

    # Legacy CN diffusion step
    sim.diffusion.diffusion_step(sim.n_density, sim.sim_params.dt)
    state_old = sim.n_density[:, k].copy()

    # Current diffusion-only step
    mask, edges, bcs = _reflective_line_geometry(nx)
    _, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    weights = np.zeros(ne, dtype=float)
    weights[k] = 1.0
    initial_field = (profile * dE)[None, :]

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=D0,
        dt=sim.sim_params.dt,
        total_time=sim.sim_params.dt,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        energy_weights=weights,
        enable_diffusion=True,
        enable_recombination=False,
        enable_scattering=False,
        dynes_gamma=0.0,
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=bath_temperature,
    )

    assert energy_frames is not None
    state_new = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)[k]

    max_abs = float(np.max(np.abs(state_new - state_old)))
    max_ref = float(np.max(np.abs(state_old)))
    rel = max_abs / max(1e-30, max_ref)
    assert rel < 1e-12


def test_pulse_injection_matches_legacy_totals_and_final_state_diffusion_only() -> None:
    """End-to-end parity: localized pulse injection + diffusion (collisions effectively off)."""
    old = _import_old_qp_simulator()
    import qp_simulator.qp_simulator as old_mod  # type: ignore[import-not-found]

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    ne = 14
    nx = 60
    D0 = 6.0

    dt = 0.01
    steps = 40
    total_time = steps * dt
    pulse_duration = 0.10

    # Inject a small, dimensionless rate into one (x, E) bin.
    inject_rate = 1e-5  # dn/dt (dimensionless per ns)
    ix = nx // 3
    k = 4

    # Disable collisions by making timescales astronomically long.
    tau_big = 1e30
    T_c = 1.2
    bath_temperature = 0.0

    # Legacy setup (eV)
    gap_eV = gap * 1e-6
    material = old.MaterialParameters(
        tau_s=tau_big,
        tau_r=tau_big,
        D_0=D0,
        T_c=T_c,
        gamma=0.0,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=steps,             # dt = total_time / steps
        L=float(nx),          # dx = 1 μm
        T=total_time,
        E_min=energy_min_factor * gap_eV,
        E_max=energy_max_factor * gap_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=lambda x: gap_eV,
        T_bath=bath_temperature,
    )
    sim.initialize("zero")

    injection = old.InjectionParameters(
        location=float(sim.geometry.x_centers[ix]),
        energy=float(sim.energies[k]),
        rate=4.0 * material.N_0 * inject_rate,
        type="pulse",
        pulse_duration=pulse_duration,
    )

    totals_old = [float(np.sum(sim.n_density) * sim.sim_params.dx * sim.sim_params.dE)]
    for it in range(steps):
        t = it * sim.sim_params.dt
        # Use the legacy "safe" injector directly to avoid double-injecting:
        # QuasiparticleSimulator.inject_quasiparticles() calls inject_quasiparticles_safe()
        # (which *already* injects) and then injects again.
        old_mod.inject_quasiparticles_safe(sim, injection, float(t))
        sim.step()
        totals_old.append(float(np.sum(sim.n_density) * sim.sim_params.dx * sim.sim_params.dE))

    state_old_final = sim.n_density.T.copy()  # (NE, nx)

    # Current setup (μeV)
    mask, edges, bcs = _reflective_line_geometry(nx)
    E_bins, _ = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    E_inj = float(E_bins[k])

    generation = ExternalGenerationSpec(
        mode="custom",
        custom_body=(
            "((t <= params['t_end']) and (abs(E - params['E_inj']) < params.get('E_tol', 1e-12))) "
            "* np.where(np.arange(params['nx']) == params['px'], params['rate'], 0.0)"
        ),
        custom_params={
            "px": ix,
            "nx": nx,
            "rate": inject_rate,
            "t_end": pulse_duration,
            "E_inj": E_inj,
            "E_tol": 1e-9,
        },
    )

    times, _, mass, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=np.zeros((1, nx), dtype=float),
        diffusion_coefficient=D0,
        dt=dt,
        total_time=total_time,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        enable_diffusion=True,
        enable_recombination=False,
        enable_scattering=False,
        dynes_gamma=0.0,
        external_generation=generation,
    )

    assert energy_frames is not None
    state_new_final = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)

    # Convert current mass (μeV) to eV for comparison with legacy totals.
    totals_new = np.array(mass, dtype=float) * 1e-6

    assert len(times) == steps + 1
    assert np.allclose(times, np.linspace(0.0, total_time, steps + 1), atol=1e-15)
    assert np.allclose(totals_new, totals_old, rtol=1e-11, atol=1e-18)

    # Full spectral state parity at final time.
    max_abs = float(np.max(np.abs(state_new_final - state_old_final)))
    max_ref = float(np.max(np.abs(state_old_final)))
    rel = max_abs / max(1e-30, max_ref)
    assert rel < 1e-11


def test_nonuniform_gap_precompute_matches_legacy_per_pixel_kernels() -> None:
    """Compare legacy Δ(x) collision kernels to current precompute per-pixel arrays."""
    old = _import_old_qp_simulator()

    nx = 18
    ne = 16
    D0 = 6.0
    gap0 = 180.0
    slope = 0.30
    energy_min_factor = 1.0
    energy_max_factor = 3.0

    tau_s = 400.0
    tau_r = 500.0
    T_c = 1.2
    bath_temperature = 0.1
    dynes_gamma = 0.0

    # Current precompute (μeV) uses normalized x in [0,1] at pixel centers.
    gap_expression = f"return {gap0} * (1.0 + {slope} * (x - 0.5))"

    mask, edges, bcs = _reflective_line_geometry(nx)
    params = SimulationParameters(
        diffusion_coefficient=D0,
        dt=0.1,
        total_time=0.1,
        mesh_size=1.0,
        store_every=1,
        energy_gap=gap0,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        dynes_gamma=dynes_gamma,
        gap_expression=gap_expression,
        collision_solver="fischer_catelani_local",
        enable_diffusion=True,
        enable_recombination=True,
        enable_scattering=True,
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=bath_temperature,
    )
    pre = precompute_arrays(mask, edges, bcs, params)

    assert bool(pre["is_uniform"]) is False
    E_bins = np.asarray(pre["E_bins"], dtype=float)
    gap_values = np.asarray(pre["gap_values"], dtype=float)
    D_pre = np.asarray(pre["D_array"], dtype=float)  # (NE, N_spatial)
    Ks_pre = np.asarray(pre["K_s_all"], dtype=float)  # (N_spatial, NE, NE)
    Kr_pre = np.asarray(pre["K_r_all"], dtype=float)
    rho_pre = np.asarray(pre["rho_all"], dtype=float)  # (N_spatial, NE)
    G_pre = np.asarray(pre["G_therm_all"], dtype=float)  # (N_spatial, NE)

    # Legacy simulation (eV) with the matching gap profile.
    gap0_eV = gap0 * 1e-6
    L = float(nx)

    def gap_fn(x: float) -> float:
        return gap0_eV * (1.0 + slope * (x / L - 0.5))

    material = old.MaterialParameters(
        tau_s=tau_s,
        tau_r=tau_r,
        D_0=D0,
        T_c=T_c,
        gamma=dynes_gamma * 1e-6,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=10,
        L=L,
        T=1.0,
        E_min=energy_min_factor * gap0_eV,
        E_max=energy_max_factor * gap0_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=gap_fn,
        T_bath=bath_temperature,
    )
    sim.initialize("thermal")

    gap_old = np.asarray(sim.geometry.gap_at_center, dtype=float) * 1e6  # μeV
    rho_old = np.asarray(sim.scattering._rho_array, dtype=float)  # (nx, ne), dimensionless
    Ks_old = np.asarray(sim.scattering._Ks_array, dtype=float)  # (nx, ne, ne) in 1/ns/eV
    Kr_old = np.asarray(sim.scattering._Kr_array, dtype=float)

    # Compare gaps and diffusion coefficients at cell centers.
    assert np.allclose(gap_values, gap_old, rtol=0.0, atol=1e-12)

    # D_pre is computed at pixel centers; match that against legacy center formula.
    E_old = np.asarray(sim.energies, dtype=float) * 1e6  # μeV
    D_old_centers = D0 * np.sqrt(np.maximum(0.0, 1.0 - (gap_old[:, None] / E_old[None, :]) ** 2))
    assert np.allclose(D_pre.T, D_old_centers, rtol=1e-12, atol=1e-15)

    # Spot-check per-pixel collision arrays (convert legacy kernels to μeV units).
    sample_px = [0, nx // 2, nx - 1]
    for px in sample_px:
        assert np.allclose(rho_pre[px], rho_old[px], rtol=1e-12, atol=1e-15)
        assert np.allclose(Ks_pre[px], Ks_old[px] * 1e-6, rtol=1e-11, atol=1e-15)
        assert np.allclose(Kr_pre[px], Kr_old[px] * 1e-6, rtol=1e-11, atol=1e-15)

        n_th = np.asarray(sim.n_thermal[px], dtype=float)
        G_old = 2.0 * n_th * sim.sim_params.dE * (Kr_old[px] @ n_th)
        assert np.allclose(G_pre[px], G_old, rtol=1e-11, atol=1e-15)


def test_nonuniform_gap_variable_diffusion_close_to_legacy() -> None:
    """Non-uniform gap: diffusion evolution should be close (discretization differs at interfaces)."""
    old = _import_old_qp_simulator()

    nx = 80
    ne = 10
    D0 = 6.0

    gap0 = 180.0
    slope = 0.4
    energy_min_factor = 1.4
    energy_max_factor = 2.6

    dt = 0.05
    steps = 20
    total_time = steps * dt

    k = ne - 2

    T_c = 1.2
    bath_temperature = 0.0
    dynes_gamma = 0.0

    # Spatial profile (dimensionless density in one energy bin).
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    profile = 1e-6 + 4e-6 * np.exp(-((x - 0.33) ** 2) / (2.0 * 0.08**2))

    # --- Legacy diffusion (eV; uses gap evaluated at boundaries) ---
    gap0_eV = gap0 * 1e-6
    L = float(nx)

    def gap_fn(xpos_um: float) -> float:
        return gap0_eV * (1.0 + slope * (xpos_um / L - 0.5))

    material = old.MaterialParameters(
        tau_s=1e30,
        tau_r=1e30,
        D_0=D0,
        T_c=T_c,
        gamma=dynes_gamma * 1e-6,
        N_0=1.0,
    )
    sim_params = old.SimulationParameters(
        nx=nx,
        ne=ne,
        nt=steps,
        L=L,  # dx = 1 μm
        T=total_time,
        E_min=energy_min_factor * gap0_eV,
        E_max=energy_max_factor * gap0_eV,
        verbose=False,
    )
    sim = old.QuasiparticleSimulator(
        material,
        sim_params,
        gap_function=gap_fn,
        T_bath=bath_temperature,
    )
    sim.initialize("zero")
    sim.n_density[:] = 0.0
    sim.n_density[:, k] = profile
    for _ in range(steps):
        sim.diffusion.diffusion_step(sim.n_density, sim.sim_params.dt)
    state_old = sim.n_density[:, k].copy()

    # --- Current diffusion (μeV; uses harmonic mean of center D at interfaces) ---
    mask, edges, bcs = _reflective_line_geometry(nx)
    params = SimulationParameters(
        diffusion_coefficient=D0,
        dt=dt,
        total_time=total_time,
        mesh_size=1.0,
        store_every=steps,  # only store final
        energy_gap=gap0,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        dynes_gamma=dynes_gamma,
        gap_expression=f"return {gap0} * (1.0 + {slope} * (x - 0.5))",
        collision_solver="fischer_catelani_local",
        enable_diffusion=True,
        enable_recombination=False,
        enable_scattering=False,
        tau_s=1e30,
        tau_r=1e30,
        T_c=T_c,
        bath_temperature=bath_temperature,
    )
    pre = precompute_arrays(mask, edges, bcs, params)
    _, dE = build_energy_grid(gap0, energy_min_factor, energy_max_factor, ne)
    weights = np.zeros(ne, dtype=float)
    weights[k] = 1.0
    initial_field = (profile * dE)[None, :]

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=D0,
        dt=dt,
        total_time=total_time,
        dx=1.0,
        store_every=steps,
        energy_gap=gap0,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        energy_weights=weights,
        enable_diffusion=True,
        enable_recombination=False,
        enable_scattering=False,
        dynes_gamma=dynes_gamma,
        tau_s=1e30,
        tau_r=1e30,
        T_c=T_c,
        bath_temperature=bath_temperature,
        precomputed=pre,
    )

    assert energy_frames is not None
    state_new = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)[k]

    max_abs = float(np.max(np.abs(state_new - state_old)))
    max_ref = float(np.max(np.abs(state_old)))
    rel = max_abs / max(1e-30, max_ref)
    assert rel < 1e-5
