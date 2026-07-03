"""Webui tests need the optional ``ui`` extra; skip cleanly without it."""

import pytest

pytest.importorskip("pydantic", reason="webui tests need the qpsim[ui] extra")
