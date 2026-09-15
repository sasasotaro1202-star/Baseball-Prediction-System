"""Opt-in runtime contract patch for the legacy backtest score helpers.

The historical backtest module predates the canonical production score contract
and can emit ``その他`` and an incorrect Low/High definition.  This module keeps
that legacy file untouched while replacing only the two pure helper functions
used by the backtest when it is executed as the research runner.
"""
from __future__ import annotations

from typing import List, Tuple


def apply() -> bool:
    import baseball_backtest as bt
    from prediction.score_distribution import low_high_probabilities, top_score_candidates

    def canonical_score_candidates(lam_h: float, lam_a: float, n: int = 4) -> List[Tuple[str, float]]:
        if int(n) != 4:
            raise ValueError("production score contract requires exactly four candidates")
        return [
            (str(item["score"]), float(item["probability"]))
            for item in top_score_candidates(float(lam_h), float(lam_a), n=4)
        ]

    def canonical_low_high_probs(lam_h: float, lam_a: float):
        return low_high_probabilities(float(lam_h), float(lam_a))

    bt.score_candidates = canonical_score_candidates
    bt.low_high_probs = canonical_low_high_probs
    bt._RUNTIME_CONTRACT_PATCH = "score-top4-exact-v1-low0-6-high7plus"
    return True
