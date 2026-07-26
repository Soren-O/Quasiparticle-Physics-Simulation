"""Unit tests for the shared Fischer 2024 artifact contract."""

import ast
import csv
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from validation.fischer_2024 import _artifact as artifact_module
from validation.fischer_2024 import fig5_paper, fig8_paper, fig8_xqp_pb, figs_5_7_fe_pb
from validation.fischer_2024._artifact import (
    THERMAL_OCCUPATION_RTOL,
    ArtifactValidationError,
    QPCertificate,
    capture_producer_identity,
    publish_artifact_pair,
    read_artifact,
    source_hashes,
    thermal_occupations_match,
    validated_numeric_array,
    write_artifact,
)


def test_thermal_occupation_match_is_ulp_scale_and_shape_strict() -> None:
    np.testing.assert_equal(
        THERMAL_OCCUPATION_RTOL,
        8.0 * np.finfo(np.float64).eps,
    )
    expected = np.geomspace(1.0e-96, 1.0e-10, 810)
    assert thermal_occupations_match(expected, expected)
    rounded_up = np.nextafter(np.nextafter(expected, np.inf), np.inf)
    assert thermal_occupations_match(rounded_up, expected)
    assert thermal_occupations_match(np.nextafter(expected, 0.0), expected)

    drifted = expected.copy()
    drifted[0] *= 1.0 + 32.0 * THERMAL_OCCUPATION_RTOL
    assert not thermal_occupations_match(drifted, expected)
    assert not thermal_occupations_match(expected[:-1], expected)
    assert not thermal_occupations_match(np.array([np.inf]), np.array([np.inf]))
    assert not thermal_occupations_match(
        np.array([1.0 + 2.0j]),
        np.array([1.0]),
    )

    smallest_subnormal = np.nextafter(0.0, 1.0)
    assert not thermal_occupations_match(
        np.array([smallest_subnormal]),
        np.array([0.0]),
    )
    assert not thermal_occupations_match(
        np.array([0.0]),
        np.array([smallest_subnormal]),
    )


def test_complex_numeric_payloads_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError, match="real-valued"):
        validated_numeric_array(
            np.array([1.0 + 2.0j]),
            context="test payload",
            expected_shape=(1,),
        )

    with pytest.raises(ValueError, match="real-valued"):
        write_artifact(
            tmp_path / "complex.csv",
            schema="test-v1",
            fingerprint={"source": "test"},
            columns=("value",),
            rows=((1.0 + 2.0j,),),
            certificates={"point": QPCertificate(0.0, 0.0)},
        )


def test_console_and_exception_text_is_ascii_safe() -> None:
    module_paths = (
        Path(__file__).with_name("fig5_paper.py"),
        Path(__file__).with_name("fig8_paper.py"),
        Path(__file__).with_name("fig8_xqp_pb.py"),
        Path(__file__).with_name("figs_5_7_fe_pb.py"),
    )
    offenders: list[str] = []

    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded_nodes: list[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    guarded_nodes.extend(node.args)
                    guarded_nodes.extend(
                        keyword.value for keyword in node.keywords
                    )
                guarded_nodes.extend(
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"help", "description"}
                )
            elif isinstance(node, ast.Raise) and node.exc is not None:
                guarded_nodes.append(node.exc)
            elif isinstance(node, ast.Assert) and node.msg is not None:
                guarded_nodes.append(node.msg)

        for guarded in guarded_nodes:
            for child in ast.walk(guarded):
                if (
                    not isinstance(child, ast.Constant)
                    or not isinstance(child.value, str)
                ):
                    continue
                try:
                    child.value.encode("ascii")
                except UnicodeEncodeError:
                    offenders.append(
                        f"{path.name}:{child.lineno}: {child.value!r}"
                    )

    assert offenders == [], (
        "non-ASCII console/exception text:\n" + "\n".join(offenders)
    )


def test_all_generators_freeze_identity_before_solve_and_publish_as_pair() -> None:
    module_paths = (
        Path(__file__).with_name("fig5_paper.py"),
        Path(__file__).with_name("fig8_paper.py"),
        Path(__file__).with_name("fig8_xqp_pb.py"),
        Path(__file__).with_name("figs_5_7_fe_pb.py"),
    )
    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        generator = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "generate_baseline"
        )
        calls = [
            node
            for node in ast.walk(generator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ]
        capture_line = min(
            node.lineno
            for node in calls
            if node.func.id == "capture_producer_identity"
        )
        solve_line = min(node.lineno for node in calls if node.func.id == "run")
        publish_line = min(
            node.lineno
            for node in calls
            if node.func.id == "publish_artifact_pair"
        )
        assert capture_line < solve_line < publish_line, path.name


@pytest.mark.parametrize(
    "module",
    (fig5_paper, fig8_paper, fig8_xqp_pb, figs_5_7_fe_pb),
)
def test_single_file_writers_cannot_overwrite_canonical_pair(module: Any) -> None:
    write_signature = inspect.signature(module.write_baseline)
    assert write_signature.parameters["path"].default is inspect.Parameter.empty
    assert write_signature.parameters["producer"].default is inspect.Parameter.empty

    producer = capture_producer_identity(module.solver_fingerprint())
    with pytest.raises(ArtifactValidationError, match="generate_baseline"):
        module.write_baseline(
            object(),
            module.baseline_path(),
            producer=producer,
        )
    with pytest.raises(ArtifactValidationError, match="generate_baseline"):
        module.write_plot(object(), module.plot_path())


def test_source_hashes_cover_the_entire_qpsim_source_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    qpsim_root = root / "qpsim"
    expected = {
        path.relative_to(root).as_posix()
        for pattern in ("*.py", "*.yaml", "*.yml")
        for path in qpsim_root.rglob(pattern)
    }

    actual = source_hashes(Path(fig5_paper.__file__))
    actual_qpsim = {name for name in actual if name.startswith("qpsim/")}

    assert actual_qpsim == expected
    assert all(len(actual[name]) == 64 for name in expected)


def test_runtime_provenance_captures_numeric_build_and_cpu_dispatch() -> None:
    runtime = capture_producer_identity({"source": "test"}).runtime

    assert set(runtime) == {
        "matplotlib",
        "numpy",
        "platform",
        "python",
        "runtime_cpu_features",
        "scipy",
        "thread_environment",
    }
    assert runtime["matplotlib"]["backend"].lower() == "agg"
    assert runtime["numpy"]["version"] == np.__version__
    for library in ("numpy", "scipy"):
        build = runtime[library]["build"]
        assert set(build) == {
            "build_dependencies",
            "compilers",
            "simd_extensions",
        }
        assert {"blas", "lapack"} <= set(build["build_dependencies"])
    assert isinstance(runtime["runtime_cpu_features"], dict)
    assert all(
        isinstance(enabled, bool)
        for enabled in runtime["runtime_cpu_features"].values()
    )
    serialized = json.dumps(runtime, sort_keys=True)
    assert "include directory" not in serialized
    assert "lib directory" not in serialized


def test_pair_publication_binds_pdf_and_promotes_csv_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "artifact.csv"
    pdf_path = tmp_path / "artifact.pdf"
    fingerprint = {"source": "stable"}
    producer = capture_producer_identity(fingerprint)
    promoted: list[Path] = []
    original_replace = os.replace

    def tracking_replace(source: str | Path, destination: str | Path) -> None:
        destination_path = Path(destination)
        if destination_path in {csv_path, pdf_path}:
            promoted.append(destination_path)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracking_replace)

    def render_pdf(path: Path) -> Path:
        path.write_bytes(b"%PDF-1.4\nunit-test\n%%EOF\n")
        return path

    def write_csv(path, identity, pdf_record):
        return write_artifact(
            path,
            schema="test-pair-v1",
            fingerprint=fingerprint,
            columns=("value",),
            rows=((1.0,),),
            certificates={"point": QPCertificate(0.0, 0.0)},
            producer=identity,
            companion_pdf=pdf_record,
        )

    assert publish_artifact_pair(
        csv_path=csv_path,
        pdf_path=pdf_path,
        producer=producer,
        current_fingerprint=lambda: fingerprint,
        render_pdf=render_pdf,
        write_csv=write_csv,
    ) == (csv_path, pdf_path)
    assert promoted == [pdf_path, csv_path]

    # Runtime is provenance, not a portability gate: only the portable
    # source/configuration fingerprint is compared with the current reader.
    monkeypatch.setattr(
        artifact_module,
        "producer_runtime_provenance",
        lambda: {"different_reader_host": True},
    )
    artifact = read_artifact(
        csv_path,
        schema="test-pair-v1",
        fingerprint=fingerprint,
        columns=("value",),
        expected_row_count=1,
        expected_certificate_ids=("point",),
        companion_pdf_path=pdf_path,
        require_companion_pdf=True,
    )
    assert artifact.data[0, 0] == 1.0
    with csv_path.open(encoding="utf-8", newline="") as fp:
        metadata = json.loads(list(csv.reader(fp))[1][0].split("=", 1)[1])
    assert metadata["companion_pdf"]["size_bytes"] == pdf_path.stat().st_size
    assert metadata["producer_runtime"] == producer.runtime

    pdf_path.write_bytes(b"%PDF-1.4\ntampered\n%%EOF\n")
    with pytest.raises(ArtifactValidationError, match="does not authenticate"):
        read_artifact(
            csv_path,
            schema="test-pair-v1",
            fingerprint=fingerprint,
            columns=("value",),
            expected_row_count=1,
            expected_certificate_ids=("point",),
            companion_pdf_path=pdf_path,
            require_companion_pdf=True,
        )


def test_pair_publication_refuses_mid_solve_identity_drift(tmp_path: Path) -> None:
    csv_path = tmp_path / "artifact.csv"
    pdf_path = tmp_path / "artifact.pdf"
    csv_path.write_text("old csv", encoding="utf-8")
    pdf_path.write_bytes(b"%PDF-1.4\nold\n%%EOF\n")
    current = {"source": "before"}
    producer = capture_producer_identity(current)

    def render_pdf(path: Path) -> Path:
        path.write_bytes(b"%PDF-1.4\nnew\n%%EOF\n")
        current["source"] = "after"
        return path

    def write_csv(path, identity, pdf_record):
        return write_artifact(
            path,
            schema="test-pair-v1",
            fingerprint=producer.fingerprint,
            columns=("value",),
            rows=((1.0,),),
            certificates={"point": QPCertificate(0.0, 0.0)},
            producer=identity,
            companion_pdf=pdf_record,
        )

    with pytest.raises(ArtifactValidationError, match="changed during the solve"):
        publish_artifact_pair(
            csv_path=csv_path,
            pdf_path=pdf_path,
            producer=producer,
            current_fingerprint=lambda: current,
            render_pdf=render_pdf,
            write_csv=write_csv,
        )
    assert csv_path.read_text(encoding="utf-8") == "old csv"
    assert pdf_path.read_bytes() == b"%PDF-1.4\nold\n%%EOF\n"
