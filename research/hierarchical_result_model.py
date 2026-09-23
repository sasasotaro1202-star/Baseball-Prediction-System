"""Leakage-safe hierarchical NPB result model.

Stage 1 estimates draw vs non-draw.
Stage 2 estimates home vs away conditional on non-draw.
The output is a coherent 3-class probability vector and can be used as an
ordinary sklearn-style classifier inside the existing chronological ensemble.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


class HierarchicalNPBClassifier:
    def __init__(self, *, draw_c=0.35, hgb_max_iter=180, random_state=42):
        self.draw_c = float(draw_c)
        self.hgb_max_iter = int(hgb_max_iter)
        self.random_state = int(random_state)
        self.classes_ = np.asarray([0, 1, 2], dtype=int)
        self.draw_model = LogisticRegression(
            C=self.draw_c,
            class_weight="balanced",
            max_iter=2500,
            random_state=self.random_state,
        )
        self.non_draw_model = HistGradientBoostingClassifier(
            max_iter=self.hgb_max_iter,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=self.random_state,
        )

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=int)
        draw_target = (y == 1).astype(int)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        if len(np.unique(draw_target)) < 2:
            raise ValueError("hierarchical NPB draw gate needs both draw and non-draw rows")

        self.draw_model.fit(X, draw_target, sample_weight=w)
        mask = y != 1
        if len(np.unique(y[mask])) < 2:
            raise ValueError("hierarchical NPB home/away stage needs both non-draw classes")
        self.non_draw_model.fit(
            X.iloc[mask] if hasattr(X, "iloc") else X[mask],
            y[mask],
            sample_weight=None if w is None else w[mask],
        )
        return self

    def predict_proba(self, X):
        draw_prob = self.draw_model.predict_proba(X)[:, 1]
        non_draw = self.non_draw_model.predict_proba(X)
        # HistGradientBoosting classes are [0, 2] after fitting on non-draw rows.
        home_idx = int(np.flatnonzero(self.non_draw_model.classes_ == 0)[0])
        away_idx = int(np.flatnonzero(self.non_draw_model.classes_ == 2)[0])
        p = np.column_stack(
            (
                (1.0 - draw_prob) * non_draw[:, home_idx],
                draw_prob,
                (1.0 - draw_prob) * non_draw[:, away_idx],
            )
        )
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return p

    def fit_predict_proba(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight=sample_weight)
        return self.predict_proba(X)
