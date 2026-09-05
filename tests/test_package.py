"""Smoke test: qpsim imports and exposes a version string.

Exists so that CI's pytest step always collects at least one item.
"""

import qpsim


def test_package_imports_and_has_version() -> None:
    assert isinstance(qpsim.__version__, str)
    assert qpsim.__version__
