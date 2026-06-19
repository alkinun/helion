"""Make ``tests/`` importable so test modules can ``from _utils import ...``.

The repo root is also placed on ``sys.path``: when ``pytest`` is launched as a
console script the working directory is not on ``sys.path``, and without the
repo root the ``tests/tritium`` and ``tests/helion`` directories (which lack
``__init__.py``) would be picked up as namespace packages shadowing the real
``tritium`` / ``helion`` packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_TESTS_DIR))
sys.path.insert(0, str(_TESTS_DIR.parent))
