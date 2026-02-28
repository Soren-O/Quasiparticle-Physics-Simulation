from __future__ import annotations

import sys
from pathlib import Path


# Ensure the repo root (which contains `qpsim/`) is importable even when pytest
# is invoked with a specific test file path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

