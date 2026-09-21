"""Leakage-safe regime router for model selection.

The router does not learn regimes from targets. Regimes are deterministic
quantile buckets derived from training features only. Model performance is
estimated on chronological validation windows and shrunk toward the global
model score, preventing tiny regimes from overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class RegimeRouter:
    min_regime_rows: int = 35
    shrinkage: float = 80.0
    min_relative_edge: float = 0.03

    def _signal(self, X: pd.DataFrame, kind: str) -> pd.Series:
        cols = list(X.columns)
        if kind == "strength":
            preferred = [c for c in cols if any(k in c.lower() for k in (
                "elo", "starter_x_quality_proxy", "starter_kbb_gap",
                "offense_power_gap", "bullpen_quality_era_diff",
            )) and ("diff" in c.lower() or c.lower().startswith("d_") or "proxy" in c.lower())]
            if not preferred:
                preferred = [c for c in cols if "diff" in c.lower()]
        else:
            preferred = [c for c in cols if any(k in c.lower() for k in (
                "expected_env", "gf_10", "ga_10", "run_volatility", "weather_run_signal"
            )) and "diff" not in c.lower()]
        if not preferred:
            return pd.Series(np.zeros(len(X)), index=X.index, dtype=float)
        vals = X[preferred].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Median is deliberately robust to individual noisy features.
        return vals.median(axis=1).fillna(0.0)

    def fit(self, X: pd.DataFrame) -> "RegimeRouter":
        strength = self._signal(X, "strength")
        env = self._signal(X, "environment")
        self.strength_q = tuple(float(x) for x in strength.quantile([0.33, 0.67]).values)
        self.env_q = tuple(float(x) for x in env.quantile([0.33, 0.67]).values)
        return self

    def labels(self, X: pd.DataFrame) -> np.ndarray:
        strength = self._signal(X, "strength").to_numpy()
        env = self._signal(X, "environment").to_numpy()
        sq = getattr(self, "strength_q", (-np.inf, np.inf))
        eq = getattr(self, "env_q", (-np.inf, np.inf))
        s = np.digitize(strength, sq, right=False)
        e = np.digitize(env, eq, right=False)
        return np.asarray([f"s{s[i]}_e{e[i]}" for i in range(len(X))], dtype=object)

    def weights(
        self,
        global_losses: Mapping[str, float],
        regime_losses: Mapping[str, Mapping[str, float]],
        regime_counts: Mapping[str, int],
    ) -> dict[str, dict[str, float]]:
        """Return conservative per-regime inverse-loss weights.

        Small regimes are shrunk toward global losses. No regime with fewer
        min_regime_rows gets a distinct model allocation.
        """
        names = list(global_losses)
        out: dict[str, dict[str, float]] = {}
        for regime, losses in regime_losses.items():
            n = int(regime_counts.get(regime, 0))
            alpha = n / (n + self.shrinkage)
            adjusted = {}
            for name in names:
                g = float(global_losses[name])
                r = float(losses.get(name, g))
                adjusted[name] = alpha * r + (1.0 - alpha) * g
            if n < self.min_regime_rows:
                adjusted = {name: float(global_losses[name]) for name in names}
            # Do not specialize a regime merely because one model wins by noise.
            # A regime-specific allocation is enabled only when the best adjusted
            # loss beats the global loss for that model by the configured margin.
            best_name = min(names, key=lambda name: adjusted[name])
            global_best = min(global_losses[name] for name in names)
            best_adjusted = adjusted[best_name]
            relative_edge = (global_best - best_adjusted) / max(global_best, 1e-9)
            if n < self.min_regime_rows or relative_edge < self.min_relative_edge:
                adjusted = {name: float(global_losses[name]) for name in names}
            inv = np.array([1.0 / max(v, 1e-6) for v in adjusted.values()], dtype=float)
            inv /= max(inv.sum(), 1e-12)
            out[regime] = {name: float(w) for name, w in zip(names, inv)}
        return out
