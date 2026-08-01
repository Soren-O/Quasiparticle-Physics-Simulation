"""Adversarial checks for the authenticated M25 artifact bundles."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qpsim.services.rate_equation import crossover_temperature_kelvin

from validation.marchegiani_2025 import (
    fig3_chemical_potentials,
    fig3_crossover_temperature,
    fig3_paper,
    fig4_paper,
    fig4_parity_rates,
)
from validation.marchegiani_2025._artifact import (
    ArtifactValidationError,
    _file_record,
    artifact_bundle_lock,
    producer_runtime_provenance,
    verified_bundle,
    write_table,
)


def _write_one_page_matplotlib_pdf(
    path: Path,
    *,
    label: str | None,
) -> Path:
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2.0, 1.0))
    if label is not None:
        figure.text(0.5, 0.5, label, ha="center", va="center")
    FigureCanvasPdf(figure).print_pdf(path)
    return path


@pytest.mark.parametrize(
    ("reader", "kwargs"),
    [
        (fig3_chemical_potentials.read_baseline, {}),
        (fig3_crossover_temperature.read_baseline, {}),
        (fig3_paper.read_baseline, {}),
        (
            fig4_parity_rates.read_baseline,
            {"accept_producer_certificate_claims": True},
        ),
        (
            fig4_paper.read_baseline,
            {"accept_producer_certificate_claims": True},
        ),
    ],
)
def test_canonical_bundle_authenticates(
    reader: Callable[..., object],
    kwargs: dict[str, bool],
) -> None:
    reader(**kwargs)


def test_fig4_summary_readers_require_explicit_claim_acceptance() -> None:
    with pytest.raises(ArtifactValidationError, match="producer assertion"):
        fig4_paper.read_baseline()
    with pytest.raises(ArtifactValidationError, match="producer assertion"):
        fig4_parity_rates.read_baseline()


def test_fig4_summary_readers_return_certificate_scope() -> None:
    paper = fig4_paper.read_baseline(
        accept_producer_certificate_claims=True,
    )
    parity = fig4_parity_rates.read_baseline(
        accept_producer_certificate_claims=True,
    )
    assert paper.certificate_scope == fig4_paper.SUMMARY_CERTIFICATE_SCOPE
    assert (
        parity.certificate_scope
        == fig4_parity_rates.SUMMARY_CERTIFICATE_SCOPE
    )


def test_direct_canonical_member_write_is_forbidden() -> None:
    with pytest.raises(ArtifactValidationError, match="Direct writes"):
        fig3_crossover_temperature.write_baseline(
            fig3_crossover_temperature.run()
        )


def test_second_bundle_lock_fails_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.artifact.json"
    with (
        artifact_bundle_lock(manifest, operation="outer test"),
        pytest.raises(ArtifactValidationError, match="locked"),
        artifact_bundle_lock(manifest, operation="nested test"),
    ):
        pytest.fail("nested lock unexpectedly acquired")


def test_explicit_resolved_canonical_paths_use_bundle_lock() -> None:
    crossover = fig3_crossover_temperature
    with (
        artifact_bundle_lock(crossover.manifest_path(), operation="outer test"),
        pytest.raises(ArtifactValidationError, match="locked"),
    ):
        crossover.read_baseline(crossover.baseline_path().resolve())

    paper = fig4_paper
    with (
        artifact_bundle_lock(paper.manifest_path(), operation="outer test"),
        pytest.raises(ArtifactValidationError, match="locked"),
    ):
        paper.read_baseline(
            paper.baseline_path().resolve(),
            accept_producer_certificate_claims=True,
        )


def test_manifest_rejects_tampered_companion_pdf(tmp_path: Path) -> None:
    module = fig4_paper
    originals = module._member_paths()
    copied = {
        name: tmp_path / name
        for name in originals
    }
    for name, source in originals.items():
        shutil.copyfile(source, copied[name])
    manifest = tmp_path / module.manifest_path().name
    shutil.copyfile(module.manifest_path(), manifest)

    with verified_bundle(
        manifest_path=manifest,
        bundle=module._BUNDLE,
        fingerprint=module.artifact_fingerprint(),
        expected_members=module._expected_members(),
        member_paths=copied,
    ):
        pass

    pdf = copied[module.plot_path().name]
    pdf.write_bytes(pdf.read_bytes() + b"\n% adversarial mutation\n")
    with (
        pytest.raises(
            ArtifactValidationError,
            match=r"complete nonempty one-page PDF|does not match manifest",
        ),
        verified_bundle(
            manifest_path=manifest,
            bundle=module._BUNDLE,
            fingerprint=module.artifact_fingerprint(),
            expected_members=module._expected_members(),
            member_paths=copied,
        ),
    ):
        pytest.fail("tampered PDF unexpectedly authenticated")


def test_pdf_parser_rejects_visually_blank_matplotlib_page(
    tmp_path: Path,
) -> None:
    blank = _write_one_page_matplotlib_pdf(tmp_path / "blank.pdf", label=None)
    with pytest.raises(
        ArtifactValidationError,
        match="complete nonempty one-page PDF",
    ):
        _file_record(blank, kind="pdf")

    text_only = _write_one_page_matplotlib_pdf(
        tmp_path / "text-only.pdf",
        label="semantic content",
    )
    assert _file_record(text_only, kind="pdf")["page_count"] == 1


def test_reader_rejects_false_table_config(tmp_path: Path) -> None:
    module = fig3_crossover_temperature
    result = module.read_baseline()
    wrong_config = dict(module._artifact_config())
    wrong_config["r_Rlt_rate_Hz"] = module.R_RLT_RATE_HZ * 2.0
    forged = tmp_path / "false-config.csv"
    write_table(
        forged,
        bundle=module._BUNDLE,
        role="closed_form_sweep",
        config=wrong_config,
        columns=module._COLUMNS,
        rows=list(
            zip(
                result.g_photon_R_Hz,
                result.T_bar_kelvin,
                strict=True,
            )
        ),
        certificate=module._CERTIFICATE,
    )
    with pytest.raises(ArtifactValidationError, match="false config"):
        module.read_baseline(forged)


def test_fig3_reader_reassembles_full_state_certificate(tmp_path: Path) -> None:
    module = fig3_chemical_potentials
    panel = module.read_baseline().panel_a
    wrong_x_L = panel.x_L * 2.0
    wrong_mu = module._chemical_potentials_GHz(
        panel.omega_LR_GHz,
        panel.T_kelvin,
        wrong_x_L,
        panel.x_Rgt,
        panel.x_Rlt,
    )
    wrong = replace(
        panel,
        x_L=wrong_x_L,
        mu_L_GHz=wrong_mu[0],
        mu_Rgt_GHz=wrong_mu[1],
        mu_Rlt_GHz=wrong_mu[2],
    )
    forged = tmp_path / "wrong-state.csv"
    module._write_panel_csv(wrong, forged)
    with pytest.raises(ArtifactValidationError, match="residual certificate"):
        module._read_panel_csv(forged, panel.omega_LR_GHz)


def test_fig3_certificate_families_are_independently_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Producer and reader evidence must pass, not match near-zero bits."""
    module = fig3_chemical_potentials
    panel = module.read_baseline().panel_a
    shape = panel.T_kelvin.shape

    # Exact-zero producer stamps need not reproduce the reader's
    # host-sensitive cancellation floor; both independently pass.
    module._validate_reassembled_certificate(
        panel,
        np.zeros(shape),
        np.zeros(shape),
        context="test panel",
    )

    with pytest.raises(
        ArtifactValidationError,
        match="producer-stamped residual certificate",
    ):
        module._validate_reassembled_certificate(
            panel,
            np.zeros(shape),
            np.full(shape, module._RESIDUAL_RATIO_LIMIT + 1e-6),
            context="test panel",
        )
    with pytest.raises(
        ArtifactValidationError,
        match="producer-stamped residual certificate",
    ):
        module._validate_reassembled_certificate(
            panel,
            np.full(shape, np.nan),
            np.zeros(shape),
            context="test panel",
        )
    monkeypatch.setattr(
        module,
        "_certificate_metrics",
        lambda _panel: (
            np.zeros(shape),
            np.full(shape, module._RESIDUAL_RATIO_LIMIT + 1e-6),
        ),
    )
    with pytest.raises(
        ArtifactValidationError,
        match="freshly reassembled residual certificate",
    ):
        module._validate_reassembled_certificate(
            panel,
            np.zeros(shape),
            np.zeros(shape),
            context="test panel",
        )


def test_crossover_fingerprint_uses_semantic_log_grid_inputs() -> None:
    module = fig3_crossover_temperature
    config = module._artifact_config()
    assert "g_photon_R_grid_Hz" not in config
    assert config["g_photon_R_grid"] == {
        "minimum_Hz": module.G_PHOTON_MIN_HZ,
        "maximum_Hz": module.G_PHOTON_MAX_HZ,
        "num_points": module.NUM_POINTS,
        "spacing": "log10",
    }


def test_crossover_reader_accepts_ulp_grid_drift_but_not_material_drift(
    tmp_path: Path,
) -> None:
    module = fig3_crossover_temperature
    result = module.run()

    ulp_grid = np.nextafter(result.g_photon_R_Hz, np.inf)
    ulp_result = replace(
        result,
        g_photon_R_Hz=ulp_grid,
        T_bar_kelvin=np.array(
            [
                crossover_temperature_kelvin(
                    Delta_R_kelvin=module.DELTA_R_KELVIN,
                    r_Rlt_rate_Hz=module.R_RLT_RATE_HZ,
                    g_photon_R_rate_Hz=float(g_photon),
                )
                for g_photon in ulp_grid
            ]
        ),
    )
    ulp_path = tmp_path / "ulp-grid.csv"
    module.write_baseline(ulp_result, ulp_path)
    accepted = module.read_baseline(ulp_path)
    np.testing.assert_array_equal(accepted.g_photon_R_Hz, ulp_grid)

    wrong_grid = result.g_photon_R_Hz.copy()
    wrong_grid[wrong_grid.size // 2] *= 1.0 + 1e-10
    wrong_result = replace(
        result,
        g_photon_R_Hz=wrong_grid,
        T_bar_kelvin=np.array(
            [
                crossover_temperature_kelvin(
                    Delta_R_kelvin=module.DELTA_R_KELVIN,
                    r_Rlt_rate_Hz=module.R_RLT_RATE_HZ,
                    g_photon_R_rate_Hz=float(g_photon),
                )
                for g_photon in wrong_grid
            ]
        ),
    )
    wrong_path = tmp_path / "wrong-grid.csv"
    module.write_baseline(wrong_result, wrong_path)
    with pytest.raises(ArtifactValidationError, match="generation grid"):
        module.read_baseline(wrong_path)


def test_manifest_records_exact_runtime_shape() -> None:
    runtime = producer_runtime_provenance()
    assert set(runtime) == {
        "matplotlib",
        "numpy",
        "platform",
        "python",
        "runtime_cpu_features",
        "scipy",
        "thread_environment",
    }
    assert set(runtime["numpy"]) == {"build", "version"}
    assert set(runtime["scipy"]) == {"build", "version"}
    for library in ("numpy", "scipy"):
        assert set(runtime[library]["build"]) == {
            "build_dependencies",
            "compilers",
            "simd_extensions",
        }
    assert all(
        isinstance(enabled, bool)
        for enabled in runtime["runtime_cpu_features"].values()
    )
    fingerprint = fig3_chemical_potentials.artifact_fingerprint()
    config = fingerprint["config"]
    assert isinstance(config, dict)
    assert np.isfinite(
        float(config["residual_tol_relative"])
    )
