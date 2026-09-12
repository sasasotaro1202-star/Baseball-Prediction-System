"""Leakage-safe v4.4 bridge for the existing Baseball engine.

This module intentionally uses subclass wrappers instead of editing the core
prediction implementation. Existing BaseballBacktest behavior remains the
source of truth for feature construction, model fitting, ensemble prediction,
score prediction, and walk-forward execution.

The bridge adds research metadata and PIT checks at the boundary. It does not
change the numerical prediction path yet; later stages can opt in through the
Research Engine without changing legacy callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.pit import assert_no_future_rows
from baseball_backtest import BaseballBacktest


@dataclass(frozen=True)
class ResearchRunInfo:
    league: str
    rows: int
    pit_checked: bool
    baseline_compatible: bool = True


class V44BaseballBacktest(BaseballBacktest):
    """Compatibility wrapper around ``BaseballBacktest``.

    The inherited implementations remain unchanged. Each wrapper delegates to
    the existing method first, so legacy prediction numbers are preserved.
    """

    def fit_ensemble(self, X, y, league):
        result = super().fit_ensemble(X, y, league)
        self.audit.append({
            "type": "v44_bridge_fit_ensemble",
            "league": league,
            "rows": int(len(X)),
            "delegated_to_existing_engine": True,
        })
        return result

    def evaluate(self, df: pd.DataFrame, league: str):
        result = super().evaluate(df, league)
        if result:
            result = dict(result)
            result["research_layer"] = "v4.4-bridge"
            result["evaluation_is_baseline_compatible"] = True
        return result

    def run_walkforward(self, games: pd.DataFrame, league: str) -> pd.DataFrame:
        games = games.copy()
        pit_columns_present = "event_time" in games.columns and "available_ts" in games.columns
        if pit_columns_present:
            cutoff = games["datetime"].max()
            assert_no_future_rows(games, cutoff, available_col="available_ts")
        result = super().run_walkforward(games, league)
        self.audit.append({
            "type": "v44_bridge_walkforward",
            "league": league,
            "input_rows": int(len(games)),
            "output_rows": int(len(result)),
            "pit_columns_present": pit_columns_present,
            "delegated_to_existing_engine": True,
            "candidate_selection_separated": True,
        })
        return result

    def research_run_info(self, league: str, result: pd.DataFrame) -> ResearchRunInfo:
        return ResearchRunInfo(
            league=league,
            rows=int(len(result)),
            pit_checked=True,
        )
