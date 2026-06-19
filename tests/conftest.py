"""Make ``tests/`` importable so test modules can ``from _utils import ...``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
