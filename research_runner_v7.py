#!/usr/bin/env python3
"""Candidate research runner with the curated interaction layer enabled.

This wrapper intentionally leaves the incumbent research_runner_v6.py unchanged.
It imports the same runner, applies the canonical runtime patch, then decorates
BaseballBacktest.match_features with research.interaction_layer. Promotion must
still pass the repository's normal OOS/holdout gates.
"""
from __future__ import annotations

import sys

from research.runtime_contract_patch import apply as apply_contract
from research.interaction_layer import add_validated_interactions

apply_contract()

import baseball_backtest as bt  # noqa: E402
import research_runner_v6 as runner  # noqa: E402

_original_match_features = bt.BaseballBacktest.match_features


def _candidate_match_features(self, row):
    base = _original_match_features(self, row)
    return add_validated_interactions(base)


bt.BaseballBacktest.match_features = _candidate_match_features
bt._RUNTIME_CONTRACT_PATCH = "score-top4-exact-v1-low0-6-high7plus-interactions-v1"

if __name__ == "__main__":
    raise SystemExit(runner.main())
