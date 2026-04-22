"""Smoke test: qpsim imports and exposes a version string.

Exists so that CI's pytest step collects at least one item until Gate 2
adds real unit tests.
"""

import qpsim


def test_package_imports_and_has_version() -> None:
    assert isinstance(qpsim.__version__, str)
    assert qpsim.__version__
