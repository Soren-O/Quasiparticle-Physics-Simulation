from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023.fig6_author_adapter import (
    AuthorPointConfig,
    AuthorPointResult,
    _runtime_compatibility_patch,
    run_author_point,
    write_author_point_bundle,
)


def test_author_equivalent_mode_rejects_silent_numerical_changes() -> None:
    with pytest.raises(ValueError, match="requires N=1620"):
        AuthorPointConfig(
            temperature_K=0.2,
            target_t_star_over_delta=0.340068493151,
            num_energy_cells=81,
        ).validate()
    with pytest.raises(ValueError, match="unseeded RNG"):
        AuthorPointConfig(
            temperature_K=0.2,
            target_t_star_over_delta=0.340068493151,
            rng_seed=0,
        ).validate()


def test_runtime_compatibility_patch_is_narrow_and_explicit() -> None:
    original = (
        b"before\n"
        b"        tau_PB=np.divide(1.,self.const_phon*np.sum(c_plus,axis=1),"
        b"where=(np.arange(1,self.qp.N_dis)>=2*self.qp.a_Delta))\n"
        b"after\n"
    )
    patched, record = _runtime_compatibility_patch(
        original,
        member_path="author/coupled.py",
    )
    assert b"tau_PB=np.full(self.qp.N_dis-1,np.inf)" in patched
    assert b"out=tau_PB" in patched
    assert patched.startswith(b"before\n")
    assert patched.endswith(b"after\n")
    assert record["transform_id"] == "initialize-subthreshold-tau-pb-to-infinity-v1"
    assert record["member_path"] == "author/coupled.py"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature_K", True),
        ("target_t_star_over_delta", True),
        ("relative_step_threshold", True),
    ],
)
def test_real_controls_reject_booleans(field: str, value: object) -> None:
    kwargs = {
        "temperature_K": 0.2,
        "target_t_star_over_delta": 0.34,
        "num_energy_cells": 81,
        "rng_seed": 0,
        "mode": "compatibility_smoke",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite and positive"):
        AuthorPointConfig(**kwargs).validate()  # type: ignore[arg-type]

    config = AuthorPointConfig(
        temperature_K=0.2,
        target_t_star_over_delta=0.34,
        num_energy_cells=81,
        rng_seed=0,
        mode="compatibility_smoke",
    )
    with pytest.raises(ValueError, match="timeout_seconds"):
        run_author_point(config, timeout_seconds=True)


def test_bundle_writer_rejects_array_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe Python identifiers"):
        write_author_point_bundle(
            AuthorPointResult(
                metadata={},
                arrays={"../escape": np.array([1.0])},
                stdout="",
                stderr="",
            ),
            tmp_path / "bundle",
        )
    assert not (tmp_path / "escape.npy").exists()


def test_unchanged_author_source_compatibility_smoke_when_supplied() -> None:
    configured = os.environ.get("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE")
    if not configured:
        pytest.skip("Set QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE for exact-source smoke.")
    result = run_author_point(
        AuthorPointConfig(
            temperature_K=0.2,
            target_t_star_over_delta=0.340068493151,
            num_energy_cells=81,
            max_newton_steps=10,
            relative_step_threshold=1e-7,
            rng_seed=0,
            mode="compatibility_smoke",
        ),
        archive_path=Path(configured),
        timeout_seconds=120.0,
    )
    assert result.metadata["schema"] == "qpsim.fischer2023.fig6-author-point.v1"
    assert result.metadata["all_finite"] is True
    assert result.metadata["h_eV"] == pytest.approx(20e-6, rel=1e-12)
    assert result.metadata["rounded_gap_eV"] == pytest.approx(180e-6, rel=1e-12)
    assert result.metadata["solved_photon_energy_eV"] == pytest.approx(20e-6, rel=1e-12)
    assert result.metadata["actual_t_star_over_delta"] == pytest.approx(
        0.340068493151,
        rel=1e-12,
    )
    constants = result.metadata["author_numerical_constants"]
    assert constants["boltzmann_constant_J_per_K"] == 1.38064852e-23
    assert constants["electron_charge_C"] == 1.60217662e-19
    assert constants["reduced_planck_constant_J_s"] == pytest.approx(
        6.626070e-34 / (2.0 * np.pi),
    )
    assert 1 <= result.metadata["newton_iterations"] <= 10
    assert result.arrays["f_final"].shape == (81,)
    assert result.arrays["n_phonon_final"].shape == (80,)
    assert result.arrays["initial_state"].shape == (161,)
    assert result.arrays["residual"].shape == (161,)
    assert all(np.all(np.isfinite(value)) for value in result.arrays.values())
    provenance = result.provenance_record()
    assert provenance["arrays"]["f_final"]["dtype"] == "<f8"
    assert result.metadata["executed_source"]["transformations"] == []
    assert result.metadata["subprocess_isolation"] == {
        "flags": ["-I", "-B"],
        "isolated_python": True,
        "os_sandbox": False,
        "user_site_enabled": False,
    }
    assert "isolated mode (-I)" in result.metadata["execution_security"]
    assert result.metadata["executed_source"]["loading"] == (
        "immutable_child_byte_snapshots_v1"
    )
    assert result.metadata["executed_source"]["module_sha256s"] == {
        Path(path).stem: digest
        for path, digest in result.metadata["executed_source"][
            "member_sha256s"
        ].items()
    }
    for path, digest in result.metadata["executed_source"]["member_sha256s"].items():
        assert result.metadata["author_source"]["member_sha256s"][path] == digest
    assert result.metadata["author_driver_contract"]["fragment_count"] >= 10
    # Preserve current-runtime warnings as evidence; the adapter does not patch
    # the author's masked square-root/divide behavior.
    assert "RuntimeWarning" in result.stderr
