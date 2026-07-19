"""Unit tests for shared Fischer 2024 artifact semantics."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from validation.fischer_2024._artifact import (
    THERMAL_OCCUPATION_RTOL,
    thermal_occupations_match,
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

    smallest_subnormal = np.nextafter(0.0, 1.0)
    assert not thermal_occupations_match(
        np.array([smallest_subnormal]),
        np.array([0.0]),
    )
    assert not thermal_occupations_match(
        np.array([0.0]),
        np.array([smallest_subnormal]),
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
                    guarded_nodes.extend(keyword.value for keyword in node.keywords)
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
                if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                    continue
                try:
                    child.value.encode("ascii")
                except UnicodeEncodeError:
                    offenders.append(f"{path.name}:{child.lineno}: {child.value!r}")

    assert offenders == [], "non-ASCII console/exception text:\n" + "\n".join(offenders)
