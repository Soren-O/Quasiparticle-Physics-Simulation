"""Benchmark 4: self-consistent gap feedback (the well you dig yourself).

The local quasiparticle occupation suppresses the local gap through the
BCS gap equation. Per cell this benchmark applies the package's direct
(Fischer-path) closure,

    Delta(x) = Delta_0 * exp(-I[f(., x)]),
    I[f]     = integral 2 rho(E) f(E) / E dE

(:func:`qpsim.observables.gap_suppression.gap_from_distribution_direct`),
whose linearization is the Owen--Scalapino form
``delta_Delta/Delta_0 = -2 int dE/E N_1 f``. A localized quasiparticle
population therefore digs a gap well under itself, and the gap acquires
exactly the spatial gradient that separates the ``L_{p,q}`` operators:
``v_com = D_N q N_1^{q-p-1} d_x N_1`` (benchmark 2), now evaluated on a
*self-generated* profile rather than an imposed ramp. (The package's
``self_consistent_gap`` support in the homogeneous ``t3_diffusion``
backend solves the same closure with no spatial resolution; this
benchmark supplies the spatial coupling.)

Setup: a "heavy" population digs the well (Picard-iterated to the
self-consistent fixed point); a weak probe packet on the well's flank is
evolved as a passive tracer and its conserved-density center of mass is
tracked. In the measurement configuration the heavy population -- think a
maintained hot spot or trapped population, quasi-static on probe-transport
timescales -- holds the well fixed, so the benchmark-2 drift analytics
apply exactly and the expected signature is parameter-free:

* A1 ``(q = 0)``  -- no DOS-gradient drift: the probe COM stays put
  (to round-off, since its undressed face flux telescopes);
* C/B ``(q < 0)`` -- drift *toward* the well the population dug
  (self-focusing: quasiparticles are attracted to their own gap
  suppression);
* A1P/A2 ``(q = 2)`` -- drift *away* from the well (expelled from the
  depression they create).

``run(dynamic=True)`` instead lets the heavy population diffuse with the
gap tracking it every step (fully dynamic feedback). The well then
flattens during the measurement and the per-step gap updates re-weight the
conserved density (the neglected energy-space spectral-flow advection);
the resulting bookkeeping drift is measured and reported
(``conservation_drift``) rather than hidden.

Run ``python -m validation.diffusion_operators.self_consistent_feedback``
to write the CSV + figure under ``outputs/diffusion_operators/``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.gap_suppression import (
    edge_samples_from_centers,
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
)
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel

from validation.diffusion_operators import (
    BENCHMARK_MODELS,
    D0_DEFAULT,
    results_dir,
    write_csv,
)

GAP_FLOOR_FACTOR = 0.5
"""Lowest gap represented by the benchmark's energy grid and closure."""

GAP_FIXED_POINT_RTOL = 1e-12
"""Maximum raw-map residual relative to the bulk gap."""

GAP_FIXED_POINT_MAX_ITERATIONS = 64
"""Hard iteration cap for the undamped self-consistent gap map."""


@dataclass(frozen=True)
class FeedbackResult:
    """Outcome of the self-consistent gap-feedback benchmark."""

    E: np.ndarray
    x: np.ndarray
    gap0: float
    e_index: int
    well_depth_target: float
    dynamic: bool
    times: np.ndarray
    gap_initial: dict[str, np.ndarray]
    gap_final: dict[str, np.ndarray]
    probe_com: dict[str, np.ndarray]
    drift_measured: dict[str, np.ndarray]
    drift_analytic: dict[str, np.ndarray]
    conservation_drift: dict[str, float]
    fit_steps: int


def _n1_columns(
    E: np.ndarray,
    dE: np.ndarray,
    gap_profile: np.ndarray,
) -> np.ndarray:
    """Represented cell-average BCS DOS, shape ``(NE, NX)``.

    The transport state stores a cell-constant occupation against the exact
    finite-volume BCS measure. Using point DOS values here would define a
    different conserved density and manufacture center-of-mass motion even
    for the undressed ``q = 0`` flux.
    """
    first_edge = float(cell_edges_from_widths(E, dE)[0])
    return np.column_stack(
        [
            bcs_dos_cell_weights(
                E,
                dE,
                float(g),
                lower_bound=max(float(g), first_edge),
            )
            / dE
            for g in gap_profile
        ]
    )


def _suppressed_gap(
    f_heavy: np.ndarray,
    E: np.ndarray,
    gap_profile: np.ndarray,
    gap0: float,
    floor_factor: float = GAP_FLOOR_FACTOR,
) -> np.ndarray:
    """One per-cell direct gap update ``Delta_j = Delta_0 exp(-I[f(., x_j)])``.

    The spectral weight inside ``I`` is evaluated at the supplied local gap
    (Picard form). Unlike a general kinetic state, this benchmark deliberately
    extends a bounded occupation plateau through its inactive guard cells so
    those samples become meaningful as the local gap falls. Reconstruct that
    controlled center profile onto fixed edge nodes before iterating; changing
    the reconstruction stencil whenever a cell gains support would make the
    discrete gap map jump at cell faces and can eliminate its fixed point.
    The suppression is floored at ``floor_factor * gap0`` as a safety net (the
    benchmark operates far above it).
    """
    out = np.empty_like(gap_profile)
    for j, g in enumerate(gap_profile):
        edge_occupation = edge_samples_from_centers(f_heavy[:, j], E)
        out[j] = gap_from_distribution_direct(
            edge_occupation,
            E,
            gap=float(g),
            delta0=gap0,
            samples="edges",
        )
    return np.maximum(out, floor_factor * gap0)


def dig_well(
    f_heavy: np.ndarray,
    E: np.ndarray,
    gap0: float,
    *,
    initial_gap: np.ndarray | None = None,
    max_iterations: int = GAP_FIXED_POINT_MAX_ITERATIONS,
) -> np.ndarray:
    """Iterate the raw gap map to a certified fixed point or fail loudly."""
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    NX = f_heavy.shape[1]
    if initial_gap is None:
        gap_profile = np.full(NX, gap0)
    else:
        gap_profile = np.asarray(initial_gap, dtype=float).copy()
        if gap_profile.shape != (NX,):
            raise ValueError(
                f"initial_gap must have shape {(NX,)}, got {gap_profile.shape}."
            )
        if (
            np.any(~np.isfinite(gap_profile))
            or np.any(gap_profile < GAP_FLOOR_FACTOR * gap0)
            or np.any(gap_profile > gap0)
        ):
            raise ValueError(
                "initial_gap must be finite and lie within the represented "
                f"interval [{GAP_FLOOR_FACTOR}*gap0, gap0]."
            )
    residual = float("inf")
    for _ in range(max_iterations):
        updated = _suppressed_gap(f_heavy, E, gap_profile, gap0)
        residual = float(np.max(np.abs(updated - gap_profile)))
        if residual <= GAP_FIXED_POINT_RTOL * gap0:
            return updated
        gap_profile = updated
    raise RuntimeError(
        "Self-consistent diffusion-benchmark gap map did not converge after "
        f"{max_iterations} iterations: relative raw-map residual="
        f"{residual / gap0:.3e}, limit={GAP_FIXED_POINT_RTOL:.3e}."
    )


def _calibrate_heavy_amplitude(
    seed_weight: np.ndarray,
    E: np.ndarray,
    gap0: float,
    well_depth: float,
) -> float:
    """Choose the seed amplitude from the realized self-consistent well.

    The direct closure is linear in the distribution at a fixed trial gap.
    Evaluating the unit-amplitude integral at the requested target gap therefore
    gives the amplitude whose raw gap-map image is exactly that target. This
    avoids searching through discrete-grid amplitudes where the active-cell map
    can be discontinuous.
    """
    if not np.isfinite(well_depth) or not 0.0 < well_depth < 0.5:
        raise ValueError("well_depth must be finite and lie in (0, 0.5).")

    target_gap = gap0 * (1.0 - well_depth)
    edge_seed_weight = edge_samples_from_centers(seed_weight, E)
    target_integral = gap_integral_from_distribution_direct(
        edge_seed_weight,
        E,
        gap=target_gap,
        samples="edges",
    )
    if not np.isfinite(target_integral) or target_integral <= 0.0:
        raise RuntimeError(
            "The diffusion-benchmark seed has no finite positive direct-gap "
            f"weight at target gap {target_gap:.6g}."
        )
    amplitude = float(-np.log1p(-well_depth) / target_integral)
    if not np.isfinite(amplitude) or amplitude > 0.5:
        raise ValueError(
            f"well_depth={well_depth} needs seed amplitude {amplitude:.6g}; "
            "the benchmark requires a finite amplitude no greater than 0.5."
        )
    return amplitude


def _com_per_energy(
    N1: np.ndarray, p: int, f: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Per-energy first moment of the conserved density ``N_1**p f``."""
    u = np.power(N1, p) * f
    numerator = np.sum(x[None, :] * u, axis=1)
    denominator = np.sum(u, axis=1)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def run(
    *,
    well_depth: float = 0.05,
    dynamic: bool = False,
    NE: int = 24,
    NX: int = 201,
    length_um: float = 100.0,
    heavy_center_um: float = 40.0,
    heavy_sigma_um: float = 10.0,
    probe_center_um: float = 50.0,
    probe_sigma_um: float = 4.0,
    seed_E_scale: float = 0.2,
    measure_E_factor: float = 1.1,
    D0: float = D0_DEFAULT,
    dt: float = 0.25,
    n_steps: int = 80,
    fit_steps: int = 4,
) -> FeedbackResult:
    """Evolve a probe packet in a self-consistently dug gap well, all models.

    ``well_depth`` is the target initial fractional gap suppression at the
    heavy-population center, calibrated directly at that fixed point and then
    independently reached by the raw Picard map. Static mode (default) holds
    the well fixed;
    ``dynamic=True`` lets the heavy population spread with the gap
    tracking it. Drift velocities are fitted over the first ``fit_steps``
    steps; the analytic prediction ``D_N q N_1^{q-p-1} d_x N_1`` is
    averaged over the initial probe's conserved density.
    """
    if not np.isfinite(well_depth) or not 0.0 < well_depth < 0.5:
        raise ValueError("well_depth must be finite and lie in (0, 0.5).")
    if not 1 <= fit_steps <= n_steps:
        raise ValueError(
            f"fit_steps={fit_steps} must satisfy 1 <= fit_steps <= n_steps={n_steps}."
        )
    gap0 = float(load_material("Al").Delta_0)
    # The direct closure may lower the gap as far as its configured safety
    # floor. Keep represented guard cells over that entire interval; omitting
    # the singular support between a lowered gap and the first cell face used
    # to make the benchmark depend on an implicit truncated integral.
    E, _ = build_energy_grid(
        gap=gap0,
        energy_min_factor=GAP_FLOOR_FACTOR,
        energy_max_factor=4.0,
        num_energy_bins=NE,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(
        E_bins=E, dE_bins=dE, gap=gap0, diffusion_coefficient=D0
    )
    material = load_material("Al")
    x = np.linspace(0.0, length_um, NX)
    e_index = int(np.argmin(np.abs(E - measure_E_factor * gap0)))

    heavy_shape = np.exp(-(((x - heavy_center_um) / heavy_sigma_um) ** 2))
    probe_shape = np.exp(-(((x - probe_center_um) / probe_sigma_um) ** 2))
    # Continue the gap-edge occupation as a bounded plateau into inactive
    # guard storage. Those values acquire meaning only if a suppressed local
    # gap gives the cells positive spectral capacity.
    seed_weight = np.exp(
        -np.maximum(E - gap0, 0.0) / (seed_E_scale * gap0)
    )

    # Calibrate against the realized self-consistent closure, rather than only
    # its first fixed-gap update.
    amp_heavy = _calibrate_heavy_amplitude(
        seed_weight,
        E,
        gap0,
        well_depth,
    )
    f_heavy0 = amp_heavy * seed_weight[:, None] * heavy_shape[None, :]
    f_probe0 = 0.5 * amp_heavy * seed_weight[:, None] * probe_shape[None, :]

    times = dt * np.arange(n_steps + 1)
    gap_initial: dict[str, np.ndarray] = {}
    gap_final: dict[str, np.ndarray] = {}
    probe_com: dict[str, np.ndarray] = {}
    drift_measured: dict[str, np.ndarray] = {}
    drift_analytic: dict[str, np.ndarray] = {}
    conservation_drift: dict[str, float] = {}

    well0 = dig_well(f_heavy0, E, gap0)

    for model in BENCHMARK_MODELS:
        p, q = model.p, model.q
        gap_profile = well0.copy()
        gap_initial[model.name] = gap_profile.copy()

        N1_x = _n1_columns(E, dE, gap_profile)
        # Analytic drift on the realized initial profile, averaged over the
        # probe's conserved density (finite-packet prediction).
        dN1_dx = np.gradient(N1_x, x, axis=1)
        v_field = np.zeros_like(N1_x)
        if q != 0:
            positive_capacity = N1_x > 0.0
            v_field[positive_capacity] = (
                D0
                * q
                * np.power(N1_x[positive_capacity], q - p - 1)
                * dN1_dx[positive_capacity]
            )
        u0 = np.power(N1_x, p) * f_probe0
        drift_numerator = np.sum(v_field * u0, axis=1)
        drift_denominator = np.sum(u0, axis=1)
        drift_analytic[model.name] = np.zeros_like(drift_numerator)
        np.divide(
            drift_numerator,
            drift_denominator,
            out=drift_analytic[model.name],
            where=drift_denominator > 0.0,
        )

        def make_state(
            f0: np.ndarray,
            model: DiffusionModel = model,
            gap_profile: np.ndarray = gap_profile,
        ) -> T3Spatial1DState:
            return T3Spatial1DState(
                f=f0.copy(),
                x=x,
                gap=gap0,
                spectral=spectral,
                material=material,
                T_bath=0.1,
                diffusion_model=model,
                gap_profile=gap_profile.copy(),
            )

        heavy = make_state(f_heavy0)
        probe = make_state(f_probe0)

        def conserved_total(state: T3Spatial1DState, p: int = p) -> float:
            N1 = _n1_columns(E, dE, state.gap_profile)
            return float(np.sum(dE[:, None] * np.power(N1, p) * state.f))

        total0 = conserved_total(heavy)

        com_t = np.empty((n_steps + 1, NE))
        N1_now = N1_x
        com_t[0] = _com_per_energy(N1_now, p, probe.f, x)

        if not dynamic:
            backend = T3Spatial1DBackend()
        for step in range(n_steps):
            if dynamic:
                backend = T3Spatial1DBackend()  # profile changes every step
                heavy = backend.apply_transport(heavy, dt)
            probe = backend.apply_transport(probe, dt)
            if dynamic:
                gap_profile = dig_well(
                    heavy.f,
                    E,
                    gap0,
                    initial_gap=heavy.gap_profile,
                )
                heavy = replace(heavy, gap_profile=gap_profile)
                probe = replace(probe, gap_profile=gap_profile)
                N1_now = _n1_columns(E, dE, gap_profile)
            com_t[step + 1] = _com_per_energy(N1_now, p, probe.f, x)

        gap_final[model.name] = gap_profile.copy()
        probe_com[model.name] = com_t[:, e_index]
        # Mean drift over the early window, energy by energy.
        drift_measured[model.name] = (com_t[fit_steps] - com_t[0]) / (
            fit_steps * dt
        )
        conservation_drift[model.name] = abs(
            conserved_total(heavy) / total0 - 1.0
        )

    return FeedbackResult(
        E=E,
        x=x,
        gap0=gap0,
        e_index=e_index,
        well_depth_target=well_depth,
        dynamic=dynamic,
        times=times,
        gap_initial=gap_initial,
        gap_final=gap_final,
        probe_com=probe_com,
        drift_measured=drift_measured,
        drift_analytic=drift_analytic,
        conservation_drift=conservation_drift,
        fit_steps=fit_steps,
    )


def main() -> None:
    """Write the CSV + figure for the self-consistent feedback benchmark."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_depth = 0.05
    depths = (0.01, 0.02, reference_depth, 0.10)
    results = {d: run(well_depth=d) for d in depths}
    ref = results[reference_depth]
    dyn = run(well_depth=reference_depth, dynamic=True)
    out = results_dir()
    ei = ref.e_index

    header = ["well_depth", "mode", "model", "v_measured_um_ns",
              "v_analytic_um_ns", "conservation_drift"]
    rows = []
    for d in depths:
        r = results[d]
        for name in r.drift_measured:
            rows.append(
                [d, "static", name, r.drift_measured[name][ei],
                 r.drift_analytic[name][ei], r.conservation_drift[name]]
            )
    for name in dyn.drift_measured:
        rows.append(
            [reference_depth, "dynamic", name, dyn.drift_measured[name][ei],
             dyn.drift_analytic[name][ei], dyn.conservation_drift[name]]
        )
    write_csv(out / "self_consistent_feedback.csv", header, rows)

    colors = {"A1": "C0", "A1P": "C4", "A2": "C1", "C": "C2", "B": "C3"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4))

    ax = axes[0]
    ax.plot(ref.x, ref.gap_initial["A1"] / ref.gap0, "-", color="C0",
            label="self-consistent well")
    ax.plot(dyn.x, dyn.gap_final["A1"] / dyn.gap0, "--", color="C0",
            label=rf"dynamic, $t={dyn.times[-1]:.0f}$ ns (A1)")
    ax.axvline(50.0, color="0.6", lw=0.8, ls=":")
    ax.annotate("probe", (50.0, 1.0), textcoords="offset points",
                xytext=(4, -12), fontsize=8, color="0.4")
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$\Delta(x) / \Delta_0$")
    ax.set_title("(a) the well the population digs", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    for name, com in ref.probe_com.items():
        ax.plot(ref.times, com - com[0], color=colors.get(name), label=name)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"$t$ (ns)")
    ax.set_ylabel(r"probe COM shift ($\mu$m)")
    ax.set_title(
        rf"(b) probe drift at $E = {ref.E[ei] / ref.gap0:.2f}\,\Delta_0$",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    ax = axes[2]
    for name in colors:
        v_meas = [results[d].drift_measured[name][ei] for d in depths]
        v_pred = [results[d].drift_analytic[name][ei] for d in depths]
        ax.plot(depths, v_pred, "-", color=colors[name], lw=1.0)
        ax.plot(depths, v_meas, "o", color=colors[name], markersize=4,
                label=name)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel(r"initial well depth $\delta\Delta/\Delta_0$")
    ax.set_ylabel(r"$v_{\rm com}$ ($\mu$m/ns)")
    ax.set_title("(c) drift vs feedback strength", fontsize=9)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "self_consistent_feedback.pdf")
    plt.close(fig)
    print(f"Wrote {out / 'self_consistent_feedback.csv'} and .pdf")
    print(f"measurement bin: E = {ref.E[ei] / ref.gap0:.3f} Delta_0")
    print("static well:")
    for name in colors:
        print(
            f"  {name:3s}: v_meas = {ref.drift_measured[name][ei]:+.4f}, "
            f"v_pred = {ref.drift_analytic[name][ei]:+.4f} um/ns"
        )
    print("dynamic feedback (same depth):")
    for name in colors:
        print(
            f"  {name:3s}: v_meas = {dyn.drift_measured[name][ei]:+.4f}, "
            f"|dN|/N = {dyn.conservation_drift[name]:.2e}"
        )


if __name__ == "__main__":
    main()
