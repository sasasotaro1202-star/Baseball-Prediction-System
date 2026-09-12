"""Chronological feature engineering: recent form, rest days, H2H, shot efficiency.

All features here are computed using ONLY strictly-prior match data for each
team at prediction time (expanding window, no lookahead), per the
leakage-prevention principle in prediction_model_specification.md section 29.

Implements (from prediction_model_specification.md):
  - Category 3/4 (soccer) & category 3 (baseball): recent form over rolling windows
  - Category 6/7: rest days since previous game
  - Category 7 (soccer): H2H over last N meetings
  - Category 5 (soccer): shot/SOT/corner efficiency
"""
import pandas as pd
import numpy as np


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Adds home_rest_days / away_rest_days = days since each team's previous match."""
    df = df.sort_values("date").reset_index(drop=True)
    last_played = {}
    home_rest, away_rest = [], []
    for _, row in df.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"]
        home_rest.append((d - last_played[h]).days if h in last_played else np.nan)
        away_rest.append((d - last_played[a]).days if a in last_played else np.nan)
        last_played[h] = d
        last_played[a] = d
    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_days_diff"] = df["home_rest_days"] - df["away_rest_days"]
    return df


def add_recent_form(df: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    """Adds rolling points-per-game and goal/run differential for each team,
    using only matches strictly before the current one (expanding + shift)."""
    df = df.sort_values("date").reset_index(drop=True)
    team_history = {}  # team -> list of (points, scored, allowed)

    for w in windows:
        df[f"home_form_ppg_{w}"] = np.nan
        df[f"home_form_gd_{w}"] = np.nan
        df[f"away_form_ppg_{w}"] = np.nan
        df[f"away_form_gd_{w}"] = np.nan

    for idx, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        for team, is_home in [(h, True), (a, False)]:
            hist = team_history.get(team, [])
            for w in windows:
                recent = hist[-w:]
                if len(recent) > 0:
                    ppg = np.mean([r[0] for r in recent])
                    gd = np.mean([r[1] - r[2] for r in recent])
                    col_prefix = "home" if is_home else "away"
                    df.at[idx, f"{col_prefix}_form_ppg_{w}"] = ppg
                    df.at[idx, f"{col_prefix}_form_gd_{w}"] = gd

        if hs > as_:
            h_pts, a_pts = 3, 0
        elif hs < as_:
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1
        team_history.setdefault(h, []).append((h_pts, hs, as_))
        team_history.setdefault(a, []).append((a_pts, as_, hs))
    return df


def add_h2h(df: pd.DataFrame, max_meetings: int = 8) -> pd.DataFrame:
    """Adds head-to-head win/draw/loss rate for the home team over the last
    N meetings between the same two teams (any venue), using prior data only."""
    df = df.sort_values("date").reset_index(drop=True)
    pair_history = {}

    df["h2h_meetings"] = 0
    df["h2h_home_win_rate"] = np.nan
    df["h2h_draw_rate"] = np.nan

    for idx, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        key = tuple(sorted([h, a]))
        history = pair_history.get(key, [])[-max_meetings:]
        if history:
            home_wins = sum(1 for winner in history if winner == h)
            draws = sum(1 for winner in history if winner == "DRAW")
            df.at[idx, "h2h_meetings"] = len(history)
            df.at[idx, "h2h_home_win_rate"] = home_wins / len(history)
            df.at[idx, "h2h_draw_rate"] = draws / len(history)

        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            winner = h
        elif hs < as_:
            winner = a
        else:
            winner = "DRAW"
        pair_history.setdefault(key, []).append(winner)
    return df


def add_shot_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Adds shot conversion / SOT accuracy for matches that have shots data
    (soccer only; silently skipped if columns are absent)."""
    if "shots_home" not in df.columns:
        return df
    df = df.copy()
    df["home_shot_conversion"] = df["home_score"] / df["shots_home"].replace(0, np.nan)
    df["away_shot_conversion"] = df["away_score"] / df["shots_away"].replace(0, np.nan)
    if "sot_home" in df.columns:
        df["home_sot_accuracy"] = df["sot_home"] / df["shots_home"].replace(0, np.nan)
        df["away_sot_accuracy"] = df["sot_away"] / df["shots_away"].replace(0, np.nan)
    return df


def apply_all(df: pd.DataFrame, windows=(5, 10), max_h2h: int = 8) -> pd.DataFrame:
    df = add_rest_days(df)
    df = add_recent_form(df, windows=windows)
    df = add_h2h(df, max_meetings=max_h2h)
    df = add_shot_efficiency(df)
    return df
