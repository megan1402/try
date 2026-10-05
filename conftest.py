"""Pytest configuration.

The Python package is ``app`` and its root is ``backend/`` (so commands run as
``python -m app.ingestion.market`` from inside ``backend/``). The tests live at
the repository root, so they need ``backend/`` on ``sys.path`` in order to
``import app...``.

Putting it in the root ``conftest.py`` keeps that path tweak in one place and
works with any pytest version.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
