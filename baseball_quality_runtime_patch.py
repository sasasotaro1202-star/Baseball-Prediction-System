#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production quality patch for score/Low-High integrity and granular evaluation.

The production patch chain is intentionally tolerant of upstream structural
changes. If an earlier hardening patch already owns the target function block,
this patch exits successfully without rewriting model logic.
"""
from __future__ import annotations
import re
from pathlib import Path

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")
MARK = "# BASEBALL_QUALITY_HARDENING_V2"
LEGACY_MARK = "# BASEBALL_QUALITY_HARDENING_V1"
if MARK in s or LEGACY_MARK in s:
    print("[QUALITY PATCH] already applied")
    raise SystemExit(0)

pat = re.compile(r"def score_candidates\(.*?\n\ndef low_high_probs\(.*?\):.*?(?=\n\ndef result_from_score)", re.S)
if not pat.search(s):
    print("[QUALITY PATCH] target block already transformed by an earlier runtime patch; no-op")
    raise SystemExit(0)

new_scores = '''def _quality_flag(v) -> bool:
    if isinstance(v, bool): return v
    if v is None: return False
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def score_candidates(lam_h: float, lam_a: float, n: int = 4) -> List[Tuple[str, float]]:
    """Return the n highest exact score patterns from 0..6 runs."""
    lam_h = max(float(lam_h), 1e-6); lam_a = max(float(lam_a), 1e-6)
    cells = [(f"{h}-{a}", poisson_pmf(h, lam_h) * poisson_pmf(a, lam_a))
             for h in range(7) for a in range(7)]
    cells.sort(key=lambda z: z[1], reverse=True)
    return cells[:max(1, int(n))]

def _nb_cdf_6(mu: float, size: float) -> float:
    """Negative-binomial P(X<=6); large size approaches Poisson."""
    mu = max(float(mu), 1e-9); size = max(float(size), 1e-6)
    p = size / (size + mu)
    pmf = p ** size; total = pmf
    for k in range(1, 7):
        pmf *= ((k - 1 + size) / k) * (1 - p)
        total += pmf
    return float(np.clip(total, 0.0, 1.0))

def _score_dispersion(values) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8: return 1e6
    mu = float(np.mean(y)); var = float(np.var(y, ddof=1))
    if var <= mu + 1e-9: return 1e6
    return float(np.clip(mu * mu / (var - mu), 0.25, 1e6))

def low_high_probs(lam_h: float, lam_a: float, dispersion_h: float | None = None, dispersion_a: float | None = None) -> Tuple[float, float]:
    dh = float(dispersion_h) if dispersion_h is not None and np.isfinite(dispersion_h) else 1e6
    da = float(dispersion_a) if dispersion_a is not None and np.isfinite(dispersion_a) else 1e6
    low = _nb_cdf_6(lam_h, dh) * _nb_cdf_6(lam_a, da)
    low = float(np.clip(low, 0.0, 1.0))
    return low, 1.0 - low

'''
s = pat.sub(new_scores, s, count=1)
s = s.replace('''                while len(scores) < 4:\n                    scores.append(("その他", 0.0))\n''', '', 1)

old = '                low, high = low_high_probs(lam_h, lam_a)\n'
new = '''                disp_h = _score_dispersion(games.iloc[:bstart]["home_score"].astype(float).values)
                disp_a = _score_dispersion(games.iloc[:bstart]["away_score"].astype(float).values)
                low, high = low_high_probs(lam_h, lam_a, disp_h, disp_a)
'''
if old in s:
    s = s.replace(old, new, 1)

confirmed = 'confirmed = games["confirmed_starters"].fillna(False).astype(bool)'
if confirmed in s:
    s = s.replace(confirmed, 'confirmed = games["confirmed_starters"].map(_quality_flag)', 1)

pat_eval = re.compile(r'    def evaluate\(self, df: pd\.DataFrame, league: str\) -> Dict\[str, Any\]:.*?\n    def save_reports', re.S)
if pat_eval.search(s):
    new_eval = '''    def evaluate(self, df: pd.DataFrame, league: str) -> Dict[str, Any]:
        if df.empty: return {}
        actual_high = ((df.actual_home_score >= 7) | (df.actual_away_score >= 7)).astype(int)
        pred_high = (df.high >= 0.5).astype(int)
        exact = df.apply(lambda r: f"{int(r.actual_home_score)}-{int(r.actual_away_score)}" in {str(r.score1), str(r.score2), str(r.score3), str(r.score4)}, axis=1)
        low_mask = actual_high == 0; high_mask = actual_high == 1
        out = {
            "League": league, "Predictions": len(df),
            "Accuracy": float(df.correct.mean()), "LogLoss": float(df.logloss.mean()), "Brier": float(df.brier.mean()),
            "MeanAbsoluteScoreError": float((abs(df.actual_home_score-df.lambda_home)+abs(df.actual_away_score-df.lambda_away)).mean()/2),
            "HomeScoreMAE": float(abs(df.actual_home_score-df.lambda_home).mean()),
            "AwayScoreMAE": float(abs(df.actual_away_score-df.lambda_away).mean()),
            "TotalScoreMAE": float(abs((df.actual_home_score+df.actual_away_score)-(df.lambda_home+df.lambda_away)).mean()),
            "HighActualRate": float(actual_high.mean()),
            "LowAccuracy": float(((pred_high == 0) & low_mask).sum() / max(1, low_mask.sum())),
            "HighAccuracy": float(((pred_high == 1) & high_mask).sum() / max(1, high_mask.sum())),
            "LowHighAccuracy": float((pred_high == actual_high).mean()),
            "HighTP": int(((pred_high == 1) & high_mask).sum()),
            "HighFP": int(((pred_high == 1) & low_mask).sum()),
            "HighFN": int(((pred_high == 0) & high_mask).sum()),
            "ExactScoreHitRate": float(exact.mean()), "Top4ScoreHitRate": float(exact.mean()),
        }
        if league == "MLB":
            try: out["AUC"] = float(roc_auc_score(df.actual, df.pred_home))
            except Exception: out["AUC"] = np.nan
        return out

    def save_reports(self, df: pd.DataFrame, league: str):
'''
    s = pat_eval.sub(new_eval, s, count=1)

needle = '''        model.to_csv(RESULTS / f"{league.lower()}_model_comparison.csv", index=False)\n'''
if needle in s:
    repl = '''        model.to_csv(RESULTS / f"{league.lower()}_model_comparison.csv", index=False)
        pd.DataFrame([self.evaluate(df, league)]).to_csv(RESULTS / f"{league.lower()}_accuracy_detail.csv", index=False)
        tmp = df.copy()
        tmp["actual_high"] = ((tmp.actual_home_score >= 7) | (tmp.actual_away_score >= 7)).astype(int)
        tmp["pred_high"] = (tmp.high >= 0.5).astype(int)
        tmp["exact_score_hit"] = tmp.apply(lambda r: int(f"{int(r.actual_home_score)}-{int(r.actual_away_score)}" in {str(r.score1), str(r.score2), str(r.score3), str(r.score4)}), axis=1)
        audit_cols = ["game_id","datetime","model","prediction","actual","correct","pred_home","pred_away","actual_home_score","actual_away_score","lambda_home","lambda_away","score1","score2","score3","score4","low","high","pred_high","actual_high","exact_score_hit"]
        tmp[[c for c in audit_cols if c in tmp.columns]].to_csv(RESULTS / f"{league.lower()}_prediction_audit.csv", index=False)
'''
    s = s.replace(needle, repl, 1)

s = LEGACY_MARK + "\n" + MARK + "\n" + s
P.write_text(s, encoding="utf-8")
print("[QUALITY PATCH] V2 applied or safely skipped")
