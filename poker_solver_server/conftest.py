"""Pytest bootstrap for poker_solver_server.

Ensures the repository's ``poker_solver_server/`` directory is on ``sys.path``
so that tests can ``from app.solver import ...`` regardless of where pytest is
invoked from.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
