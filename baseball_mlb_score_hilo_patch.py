#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production patch for MLB score/Low-High prediction integrity.

The base engine already computes run lambdas, four score candidates and
Low/High probabilities. This patch makes the internal contract explicit while
preserving the user-facing rule that MLB score/Low-High need not be displayed.
It also removes the accidental 1500-second cap, rejects MLB games without both
confirmed starters from OOS evaluation, prevents observed-weather leakage, and
fixes stale 30-minute diagnostics.
"""
from pathlib import Path

P = Path("baseball_backtest.py")
s = P.read_text(encoding="utf-8")

marker = "# MLB_SCORE_HILO_PATCH_V2"
if marker in s:
    print("[MLB SCORE/HILO PATCH] V2 already applied")
    raise SystemExit(0)

old = 'self.time_budget_sec = min(float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")), 1500.0)  # hard cap: 29:00'
new = 'self.time_budget_sec = max(60.0, float(os.getenv("BASEBALL_TIME_BUDGET_SEC", "1500")))  # workflow controls the budget'
if old in s:
    s = s.replace(old, new, 1)

# Insert a strict MLB pregame starter gate immediately after the NPB gate.
anchor = '''        X, y, meta = self.build_features(games)\n'''
gate = '''        if league == "MLB":
            if "confirmed_starters" not in games.columns:
                raise RuntimeError("MLB confirmed_starters column missing; refusing pregame backtest")
            confirmed = games["confirmed_starters"].fillna(False).astype(bool)
            starter_rate = float(confirmed.mean()) if len(games) else 0.0
            self.audit.append({
                "type": "mlb_confirmed_starter_coverage",
                "games_before_filter": int(len(games)),
                "both_starter_rate": starter_rate,
                "required_rate": 0.90,
            })
            print(f"[MLB AUDIT] confirmed both-starter coverage={starter_rate:.1%} required=90%")
            if starter_rate < 0.90:
                raise RuntimeError(
                    f"MLB confirmed starter coverage too low: {starter_rate:.1%}; "
                    "refusing to run a misleading backtest."
                )
            games = games.loc[confirmed].reset_index(drop=True)
            if len(games) <= MIN_TRAIN + 1:
                print(f"[MLB] insufficient confirmed-starter games after filter: {len(games)}")
                return pd.DataFrame()

'''
if anchor not in s:
    raise RuntimeError("walk-forward feature-build anchor missing")
s = s.replace(anchor, gate + anchor, 1)

# Observed historical weather is not a valid pregame feature unless the row
# explicitly carries forecast-as-of-cutoff provenance.
old_weather = '''        for c in ("weather_temp_c", "weather_humidity_pct", "weather_wind_kmh", "weather_precip_mm"):
            if c in row:
                try:
                    out[c] = float(row.get(c)) if pd.notna(row.get(c)) else 0.0
                except Exception:
                    out[c] = 0.0
'''
new_weather = '''        weather_forecast_ok = bool(row.get("weather_forecast_asof_cutoff", False))
        if weather_forecast_ok:
            for c in ("weather_temp_c", "weather_humidity_pct", "weather_wind_kmh", "weather_precip_mm"):
                if c in row:
                    try:
                        out[c] = float(row.get(c)) if pd.notna(row.get(c)) else 0.0
                    except Exception:
                        out[c] = 0.0
'''
if old_weather in s:
    s = s.replace(old_weather, new_weather, 1)

# Probable pitchers from a schedule endpoint are not the same thing as
# confirmed starters. Future MLB predictions therefore require an explicit
# confirmed_starters flag from a trusted pregame source.
old_confirm = '                confirmed=bool(hp and ap)\n'
new_confirm = '                confirmed=bool(g.get("confirmed_starters", False))\n'
if old_confirm in s:
    s = s.replace(old_confirm, new_confirm, 1)

s = s.replace('print("[HARD STOP] 30-minute limit reached; skipping remaining leagues")', 'print("[HARD STOP] computation budget reached; skipping remaining leagues")', 1)
s = s.replace('print("[HARD STOP] 30-minute limit reached before processing")', 'print("[HARD STOP] computation budget reached before processing")', 1)

# Explicit MLB internal prediction aliases and postgame scoring fields.
needle = '''                    "low": low, "high": high,
                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),
                })'''
replacement = '''                    "low": low, "high": high,
                    "pred_low_prob": low, "pred_high_prob": high,
                    "pred_low_high": "Low" if low >= 0.5 else "High",
                    "pred_score1": scores[0][0], "pred_score1_prob": scores[0][1],
                    "pred_score2": scores[1][0], "pred_score2_prob": scores[1][1],
                    "pred_score3": scores[2][0], "pred_score3_prob": scores[2][1],
                    "pred_score4": scores[3][0], "pred_score4_prob": scores[3][1],
                    "actual_home_score": float(r["home_score"]), "actual_away_score": float(r["away_score"]),
                    "actual_low_high": "High" if (float(r["home_score"]) >= 7 or float(r["away_score"]) >= 7) else "Low",
                    "score_exact_hit": int(any(f"{int(float(r['home_score']))}-{int(float(r['away_score']))}" == str(z[0]) for z in scores if str(z[0]) != "その他")),
                    "low_high_hit": int(("High" if (float(r["home_score"]) >= 7 or float(r["away_score"]) >= 7) else "Low") == ("Low" if low >= 0.5 else "High")),
                })'''
if needle in s:
    s = s.replace(needle, replacement, 1)

s = marker + "\n" + s
P.write_text(s, encoding="utf-8")
print("[MLB SCORE/HILO PATCH] V2 applied: confirmed-starter gate + score/Low-High fields + leakage-safe weather + workflow budget")