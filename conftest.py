"""Repository-wide pytest collection policy.

Paper reproduction and certification suites are valuable forensic tools, but
they are source-bound historical workflows rather than the everyday engine
gate.  Keep them runnable explicitly without allowing stale certificates to
turn one source edit into hundreds of default-suite failures.
"""

from __future__ import annotations

import pytest

_PAPER_VALIDATION_PREFIXES = (
    "validation/fischer_2023/",
    "validation/fischer_2024/",
    "validation/reference_models/fischer_2023/",
    "tests/scripts/test_regenerate_fischer_",
    "tests/scripts/test_render_fischer_",
    "tests/validation/test_author_source.py",
    "tests/validation/test_fig6_author_",
    "tests/validation/test_fig6_cleanroom_",
    "tests/validation/test_fig7_promotion.py",
    "tests/validation/test_fischer_",
    "tests/validation/test_reproduction_ladder.py",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Move paper-specific reproduction/certification tests to an opt-in lane."""
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if nodeid.startswith(_PAPER_VALIDATION_PREFIXES):
            item.add_marker(pytest.mark.paper_validation)
