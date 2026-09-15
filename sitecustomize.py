"""Minimal repository-local Python startup hook.

Only the research entry point receives the legacy backtest output contract
patch. Normal tests, CLI utilities, and production imports are untouched.
"""
from __future__ import annotations

import os
import sys

if os.path.basename(sys.argv[0]) == "research_runner_v6.py":
    from research.runtime_contract_patch import apply
    apply()
