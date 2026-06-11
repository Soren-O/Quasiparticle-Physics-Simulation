"""One finite-escape phonon-bottleneck timing run for the prelim strip.

This script is intentionally a single-case probe, not a sweep.  It adds a
local dynamic phonon occupation ``n_ph(omega, x, t)`` to the 1D spatial T3
strip run:

    dn_ph/dt = a_ph[f] + b_ph[f] n_ph + (n_th - n_ph) / tau_l.

The QP distribution still diffuses spatially; the phonons are local Ph0
phonons coupled to a substrate bath by the finite escape time ``tau_l``.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.experiments.prelim_resonators import (
    PrelimResonator,
    current_squared_profile,
)
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.solvers.etd import etd2_step
from scripts.run_prelim_spatial_overnight import (
    SweepConfig,
    _build_state,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_spatial_finite_phonon_one"

CONFIG = SweepConfig(
    name="finite_phonon_one",
    NX=21,
    NE=28,
    dt_ns=1.0,
    max_time_ns=10_000.0,
    stop_tol=2e-9,
    snapshot_interval_ns=500.0,
    D0_values=(20.0,),
    source_rates_per_ns=(5e-4,),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)

TAU_L_NS = 1.0


@dataclass(frozen=True)
class ReadoutPhotonDrive:
    """Fixed-occupancy sub-gap readout drive on the spatial strip.

    The photon collision channel is local in ``x``.  ``spatial_profile`` scales
    the resonator photon number at each spatial cell; for the prelim strip we
    use the normalized quarter-wave ``I^2`` profile, so ``n_bar`` is the peak
    occupancy at the shorted end.
    """

    omega_0: float
    n_bar: float
    c_phot: float
    spatial_profile: np.ndarray
    label: str = "readout"

    def __post_init__(self) -> None:
        profile = np.asarray(self.spatial_profile, dtype=float)
        if profile.ndim != 1:
            raise ValueError("spatial_profile must be one-dimensional.")
        if self.omega_0 <= 0.0:
            raise ValueError("omega_0 must be positive.")
        if self.n_bar < 0.0:
            raise ValueError("n_bar must be non-negative.")
        if self.c_phot < 0.0:
            raise ValueError("c_phot must be non-negative.")
        if not np.all(np.isfinite(profile)):
            raise ValueError("spatial_profile must contain only finite values.")
        if np.any(profile < 0.0):
            raise ValueError("spatial_profile must be non-negative.")
        object.__setattr__(self, "spatial_profile", profile.copy())

    def validate_for_state(self, state: T3Spatial1DState) -> None:
        if self.spatial_profile.shape != (state.x.size,):
            raise ValueError(
                "readout spatial_profile length does not match state.x: "
                f"{self.spatial_profile.shape} vs {(state.x.size,)}"
            )

    def local_n_bar(self, ix: int) -> float:
        return float(self.n_bar * self.spatial_profile[ix])


def readout_drive_from_resonator(
    state: T3Spatial1DState,
    resonator: PrelimResonator,
    *,
    n_bar: float,
    c_phot: float = 1e-9,
) -> ReadoutPhotonDrive:
    """Build an ``I^2``-weighted sub-gap readout drive for one prelim mode."""
    weights = current_squared_profile(state.x, resonator.total_length_um)
    peak = float(np.max(weights))
    if peak <= 0.0:
        raise RuntimeError("Current profile vanished on the simulated strip.")
    return ReadoutPhotonDrive(
        omega_0=resonator.probe_energy_uev,
        n_bar=n_bar,
        c_phot=c_phot,
        spatial_profile=weights / peak,
        label=f"mode_{resonator.index}_{resonator.frequency_ghz:.3f}GHz",
    )


@dataclass(frozen=True)
class FinitePhononSnapshot:
    t_ns: float
    max_dfdt_per_ns: float
    max_dnphdt_per_ns: float
    xqp_mean: float
    xqp_source: float
    xqp_open_end: float
    nph_mean: float
    nph_max: float


class FinitePhononSpatialRunner:
    """Operator-split finite-escape phonon dynamics for one spatial strip."""

    def __init__(self, state: T3Spatial1DState, tau_l_ns: float) -> None:
        if tau_l_ns <= 0.0:
            raise ValueError("tau_l_ns must be positive.")
        self.tau_l_ns = float(tau_l_ns)
        self.transport = T3Spatial1DBackend()
        self.omega, self.idx_diff, self.idx_sum, self.diff_sign = (
            build_phonon_frequency_map(state.spectral.E)
        )
        self.n_th = thermal_phonon_occupation(self.omega, state.T_bath)
        self.n_ph = np.repeat(self.n_th[:, None], state.x.size, axis=1)
        self.K_s0 = build_scattering_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        self.K_r0 = build_recombination_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )

    def step(
        self,
        state: T3Spatial1DState,
        dt_ns: float,
        source: object,
        *,
        readout_drive: ReadoutPhotonDrive | None = None,
    ) -> tuple[
        T3Spatial1DState,
        float,
        float,
    ]:
        before_f = state.f
        before_nph = self.n_ph.copy()

        state = self.transport.apply_transport(state, 0.5 * dt_ns)
        f_mid = self._qp_collision_step(
            state,
            dt_ns,
            source,
            readout_drive=readout_drive,
        )
        # replace() keeps every other field — a positional rebuild here
        # used to silently reset defaulted physics (diffusion_model,
        # gap_profile, interface_conductance) mid-step.
        state = replace(state, f=f_mid)
        self._phonon_escape_step(state, dt_ns)
        state = self.transport.apply_transport(state, 0.5 * dt_ns)

        max_dfdt = float(np.max(np.abs(state.f - before_f)) / dt_ns)
        max_dnphdt = float(np.max(np.abs(self.n_ph - before_nph)) / dt_ns)
        return state, max_dfdt, max_dnphdt

    def _qp_collision_step(
        self,
        state: T3Spatial1DState,
        dt_ns: float,
        source: object,
        *,
        readout_drive: ReadoutPhotonDrive | None = None,
    ) -> np.ndarray:
        if readout_drive is not None:
            readout_drive.validate_for_state(state)
        f_new = np.empty_like(state.f)
        for ix in range(state.x.size):
            N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
                self.n_ph[:, ix],
                self.idx_diff,
                self.idx_sum,
                self.diff_sign,
            )
            gain_external = source.gain[:, ix]
            loss_external = source.loss_rate[:, ix]

            def rhs(
                f_col: np.ndarray,
                N_p: np.ndarray = N_p,
                N_emit: np.ndarray = N_emit,
                N_abs: np.ndarray = N_abs,
                gain_external: np.ndarray = gain_external,
                loss_external: np.ndarray = loss_external,
                ix: int = ix,
            ) -> tuple[np.ndarray, np.ndarray]:
                gain, loss = phonon_collision_rates(
                    f_col,
                    state.spectral,
                    self.K_s0,
                    self.K_r0,
                    state.T_bath,
                    N_p_override=N_p,
                    N_emit_override=N_emit,
                    N_abs_override=N_abs,
                )
                if readout_drive is not None:
                    photon_gain, photon_loss = sub_gap_photon_collision_rates(
                        f_col,
                        state.spectral,
                        readout_drive.omega_0,
                        readout_drive.local_n_bar(ix),
                        readout_drive.c_phot,
                    )
                    gain = gain + photon_gain
                    loss = loss + photon_loss
                return gain + gain_external, loss + loss_external

            f_new[:, ix] = etd2_step(state.f[:, ix], rhs, dt_ns)
        return f_new

    def _phonon_escape_step(self, state: T3Spatial1DState, dt_ns: float) -> None:
        inv_tau = 1.0 / self.tau_l_ns
        next_n = np.empty_like(self.n_ph)
        for ix in range(state.x.size):
            a_ph, b_ph = compute_phonon_source_sink(
                state.f[:, ix],
                state.spectral,
                self.K_s0,
                self.K_r0,
                self.idx_diff,
                self.idx_sum,
                self.diff_sign,
                self.omega.size,
            )
            A = a_ph + inv_tau * self.n_th
            B = b_ph - inv_tau
            z = np.clip(B * dt_ns, -700.0, 50.0)
            exp_z = np.exp(z)
            small = np.abs(B) < 1e-14
            updated = np.empty_like(self.n_ph[:, ix])
            updated[~small] = (
                self.n_ph[~small, ix] * exp_z[~small]
                + A[~small] * (exp_z[~small] - 1.0) / B[~small]
            )
            updated[small] = self.n_ph[small, ix] + A[small] * dt_ns
            next_n[:, ix] = np.clip(updated, 0.0, np.inf)
        self.n_ph = next_n


def _snapshot(
    t_ns: float,
    state: T3Spatial1DState,
    runner: FinitePhononSpatialRunner,
    max_dfdt: float,
    max_dnphdt: float,
) -> FinitePhononSnapshot:
    xqp = _xqp_profile(state)
    return FinitePhononSnapshot(
        t_ns=float(t_ns),
        max_dfdt_per_ns=float(max_dfdt),
        max_dnphdt_per_ns=float(max_dnphdt),
        xqp_mean=float(np.mean(xqp)),
        xqp_source=float(xqp[0]),
        xqp_open_end=float(xqp[-1]),
        nph_mean=float(np.mean(runner.n_ph)),
        nph_max=float(np.max(runner.n_ph)),
    )


def _write_trace(path: Path, snapshots: list[FinitePhononSnapshot]) -> None:
    fieldnames = list(FinitePhononSnapshot.__dataclass_fields__.keys())
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for snap in snapshots:
            writer.writerow(snap.__dict__)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    D0 = CONFIG.D0_values[0]
    source_rate = CONFIG.source_rates_per_ns[0]
    center_delta = CONFIG.source_centers_delta[0]
    sigma_delta = CONFIG.source_sigmas_delta[0]

    state = _build_state(CONFIG, D0)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=center_delta,
        sigma_delta=sigma_delta,
    )
    runner = FinitePhononSpatialRunner(state, tau_l_ns=TAU_L_NS)

    snapshots = [_snapshot(0.0, state, runner, float("inf"), float("inf"))]
    t_ns = 0.0
    next_snapshot = CONFIG.snapshot_interval_ns
    n_steps = 0
    converged = False
    wall_start = time.monotonic()
    last_dfdt = float("inf")
    last_dnphdt = float("inf")

    max_steps = int(np.ceil(CONFIG.max_time_ns / CONFIG.dt_ns))
    for _ in range(max_steps):
        state, last_dfdt, last_dnphdt = runner.step(state, CONFIG.dt_ns, source)
        t_ns += CONFIG.dt_ns
        n_steps += 1

        if t_ns >= next_snapshot - 1e-12:
            snapshots.append(_snapshot(t_ns, state, runner, last_dfdt, last_dnphdt))
            next_snapshot += CONFIG.snapshot_interval_ns

        if last_dfdt < CONFIG.stop_tol:
            converged = True
            break

    if snapshots[-1].t_ns < t_ns:
        snapshots.append(_snapshot(t_ns, state, runner, last_dfdt, last_dnphdt))

    wall_s = time.monotonic() - wall_start
    shift_rows = _resonator_shifts(state, f_ref)
    delta_values = [row["delta_fr_hz_current_weighted"] for row in shift_rows]
    source_calibration = _source_calibration(CONFIG, source_rate)
    metadata = {
        "D0_um2_per_ns": D0,
        "source_rate_per_ns": source_rate,
        **source_calibration,
        "source_center_delta": center_delta,
        "source_sigma_delta": sigma_delta,
        "tau_l_ns": TAU_L_NS,
        "dt_ns": CONFIG.dt_ns,
        "total_time_ns": t_ns,
        "n_steps": n_steps,
        "wall_seconds": wall_s,
        "ns_per_wall_second": t_ns / wall_s if wall_s > 0 else None,
        "converged": converged,
        "final_max_dfdt_per_ns": last_dfdt,
        "final_max_dnphdt_per_ns": last_dnphdt,
        "xqp_mean": snapshots[-1].xqp_mean,
        "xqp_source": snapshots[-1].xqp_source,
        "xqp_open_end": snapshots[-1].xqp_open_end,
        "nph_mean": snapshots[-1].nph_mean,
        "nph_max": snapshots[-1].nph_max,
        "delta_fr_hz_min": min(delta_values),
        "delta_fr_hz_max": max(delta_values),
        "model_note": (
            "Dynamic local Ph0 phonons with finite escape to thermal bath; "
            "no lateral phonon transport."
        ),
    }

    with (OUT_DIR / "metadata.json").open("w") as fp:
        json.dump(metadata, fp, indent=2)
    _write_trace(OUT_DIR / "trace.csv", snapshots)
    with (OUT_DIR / "resonator_shifts.csv").open("w", newline="") as fp:
        fieldnames = list(shift_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shift_rows)

    print(
        f"Finite tau_l={TAU_L_NS:g} ns run: t={t_ns:g} ns, "
        f"steps={n_steps}, wall={wall_s:.2f} s, converged={converged}"
    )
    print(
        f"xqp_mean={metadata['xqp_mean']:.3e}, "
        f"nph_max={metadata['nph_max']:.3e}, "
        f"delta_fr=[{min(delta_values):.3e}, {max(delta_values):.3e}] Hz"
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
