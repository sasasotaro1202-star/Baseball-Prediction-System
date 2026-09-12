#!/usr/bin/env python3
"""
Unified Elo Baseline + Base Rate Comparison for Baseball and Soccer

実行方法:
  python elo_baseline_all.py

出力:
  - Elo ベースラインの精度
  - ベースレート (単純多数決) の精度
  - 改善幅 (Elo - Base Rate)

方針: どのフェーズでも例外で止まらず、必ず exit code 0 で終了する。
"""
import json
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "data" / "results"


def calc_elo(df, k=24.0, home_adv=25.0, team_col='team', score_col='score', date_col='date'):
    if df.empty:
        return df
    d = df.sort_values(date_col).reset_index(drop=True).copy()
    ratings = {}
    elo_team, elo_opponent, elo_expected = [], [], []

    for _, r in d.iterrows():
        team = r[team_col]
        opp = r.get('opponent', None)
        score = r[score_col]
        opp_score = r.get('opponent_score', None)

        team_elo = ratings.get(team, 1500.0)
        opp_elo = ratings.get(opp, 1500.0) if opp else 1500.0

        elo_team.append(team_elo)
        elo_opponent.append(opp_elo)

        expected = 1.0 / (1.0 + 10 ** (-((team_elo + home_adv) - opp_elo) / 400.0))
        elo_expected.append(expected)

        if pd.notna(score) and pd.notna(opp_score):
            outcome = 1.0 if score > opp_score else (0.0 if score < opp_score else 0.5)
            ratings[team] = team_elo + k * (outcome - expected)
            if opp:
                ratings[opp] = opp_elo + k * ((1 - outcome) - (1 - expected))

    d['elo'] = elo_team
    d['elo_opponent'] = elo_opponent
    d['elo_expected'] = elo_expected
    return d


def evaluate_baseline(df, target_col, elo_expected_col='elo_expected'):
    if df.empty or target_col not in df.columns:
        return None

    df = df.dropna(subset=[target_col]).copy()
    if df.empty:
        return None

    if elo_expected_col in df.columns:
        df['pred_elo'] = (df[elo_expected_col] > 0.5).astype(int)
        elo_acc = (df['pred_elo'] == df[target_col]).mean()
    else:
        elo_acc = None

    base_rate = df[target_col].mean()

    return {
        'n_samples': int(len(df)),
        'elo_accuracy': float(elo_acc) if elo_acc is not None and pd.notna(elo_acc) else None,
        'base_rate': float(base_rate) if pd.notna(base_rate) else None,
        'elo_improvement': (float(elo_acc - base_rate)
                             if elo_acc is not None and pd.notna(elo_acc) and pd.notna(base_rate)
                             else None),
    }


def load_baseball_data():
    """pipelines/collect.py の出力形式 (data/raw/mlb_games_{season}.csv) を読む"""
    games = []
    if not RAW_DIR.exists():
        return pd.DataFrame()

    for f in sorted(RAW_DIR.glob('mlb_games_*.csv')):
        try:
            df = pd.read_csv(f)
            df['sport'] = 'mlb'
            games.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    for f in sorted(RAW_DIR.glob('mlb_games_*.parquet')):
        try:
            df = pd.read_parquet(f)
            df['sport'] = 'mlb'
            games.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    npb_csv = RAW_DIR / 'npb_games.csv'
    if npb_csv.exists():
        try:
            df = pd.read_csv(npb_csv)
            df['sport'] = 'npb'
            games.append(df)
        except Exception as e:
            print(f"Failed to read {npb_csv}: {e}")

    return pd.concat(games, ignore_index=True) if games else pd.DataFrame()


def load_soccer_data():
    games = []
    if not RAW_DIR.exists():
        return pd.DataFrame()

    for f in sorted(RAW_DIR.glob('football_data*.csv')):
        try:
            df = pd.read_csv(f)
            df['sport'] = 'soccer_fd'
            games.append(df)
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    us_csv = RAW_DIR / 'understat_matches.csv'
    if us_csv.exists():
        try:
            df = pd.read_csv(us_csv)
            df['sport'] = 'soccer_us'
            games.append(df)
        except Exception as e:
            print(f"Failed to read {us_csv}: {e}")

    return pd.concat(games, ignore_index=True) if games else pd.DataFrame()


def prepare_baseball_features(df):
    if df.empty:
        return df

    d = df.copy()
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')

    required = {'home_team', 'away_team', 'home_score', 'away_score'}
    if not required.issubset(set(d.columns)):
        print(f"prepare_baseball_features: missing columns, have={list(d.columns)}")
        return pd.DataFrame()

    d = d.dropna(subset=['home_score', 'away_score'])

    home_rows = pd.DataFrame({
        'date': d['date'], 'team': d['home_team'], 'opponent': d['away_team'],
        'score': d['home_score'], 'opponent_score': d['away_score'], 'is_home': 1,
        'target': (d['home_score'] > d['away_score']).astype(int),
        'sport': d.get('sport', 'mlb'),
        'total_runs': d['home_score'] + d['away_score'],
    })
    away_rows = pd.DataFrame({
        'date': d['date'], 'team': d['away_team'], 'opponent': d['home_team'],
        'score': d['away_score'], 'opponent_score': d['home_score'], 'is_home': 0,
        'target': (d['away_score'] > d['home_score']).astype(int),
        'sport': d.get('sport', 'mlb'),
        'total_runs': d['home_score'] + d['away_score'],
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


def prepare_soccer_features(df):
    if df.empty:
        return df

    d = df.copy()
    rename_map = {
        'FTHG': 'home_goals', 'FTAG': 'away_goals', 'FTR': 'result',
        'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Date': 'date',
    }
    d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})

    required = {'home_team', 'away_team', 'home_goals', 'away_goals'}
    if not required.issubset(set(d.columns)):
        print(f"prepare_soccer_features: missing columns, have={list(d.columns)}")
        return pd.DataFrame()

    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], errors='coerce', dayfirst=True)

    d = d.dropna(subset=['home_goals', 'away_goals'])

    home_rows = pd.DataFrame({
        'date': d['date'], 'team': d['home_team'], 'opponent': d['away_team'],
        'score': d['home_goals'], 'opponent_score': d['away_goals'], 'is_home': 1,
        'target': (d['home_goals'] > d['away_goals']).astype(int),
        'sport': d.get('sport', 'soccer'),
        'total_goals': d['home_goals'] + d['away_goals'],
    })
    away_rows = pd.DataFrame({
        'date': d['date'], 'team': d['away_team'], 'opponent': d['home_team'],
        'score': d['away_goals'], 'opponent_score': d['home_goals'], 'is_home': 0,
        'target': (d['away_goals'] > d['home_goals']).astype(int),
        'sport': d.get('sport', 'soccer'),
        'total_goals': d['home_goals'] + d['away_goals'],
    })
    return pd.concat([home_rows, away_rows], ignore_index=True)


def main():
    print("=" * 60)
    print("Elo Baseline + Base Rate Comparison")
    print("=" * 60)

    results = {}

    try:
        print("\nLoading data...")
        bb_raw = load_baseball_data()
        soccer_raw = load_soccer_data()
        print(f"  Baseball games: {len(bb_raw)}")
        print(f"  Soccer games: {len(soccer_raw)}")

        bb = prepare_baseball_features(bb_raw)
        soccer = prepare_soccer_features(soccer_raw)
        print(f"  Baseball rows (home+away): {len(bb)}")
        print(f"  Soccer rows (home+away): {len(soccer)}")

        if not bb.empty:
            print("\n--- Baseball ---")
            bb = calc_elo(bb, team_col='team', score_col='score', date_col='date')

            res = evaluate_baseline(bb, 'target', 'elo_expected')
            if res:
                results['baseball_all'] = res
                acc = res['elo_accuracy']
                imp = res['elo_improvement']
                print(f"  Overall (n={res['n_samples']}):")
                print(f"    Elo accuracy: {acc*100:.1f}%" if acc is not None else "    Elo: N/A")
                print(f"    Base rate: {res['base_rate']*100:.1f}%")
                print(f"    Improvement: {imp*100:+.1f}pt" if imp is not None else "    Improvement: N/A")

            if 'total_runs' in bb.columns:
                bb['target_high'] = (bb['total_runs'] >= 7).astype(int)
                res_high = evaluate_baseline(bb, 'target_high', 'elo_expected')
                if res_high:
                    results['baseball_highlow'] = res_high
                    print(f"\n  High/Low (n={res_high['n_samples']}):")
                    print(f"    Base rate (high, sum>=7): {res_high['base_rate']*100:.1f}%")
        else:
            print("\n--- Baseball: no data available, skipping ---")

        if not soccer.empty:
            print("\n--- Soccer ---")
            soccer = calc_elo(soccer, team_col='team', score_col='score', date_col='date')
            res = evaluate_baseline(soccer, 'target', 'elo_expected')
            if res:
                results['soccer_all'] = res
                acc = res['elo_accuracy']
                imp = res['elo_improvement']
                print(f"  Overall (n={res['n_samples']}):")
                print(f"    Elo accuracy: {acc*100:.1f}%" if acc is not None else "    Elo: N/A")
                print(f"    Base rate: {res['base_rate']*100:.1f}%")
                print(f"    Improvement: {imp*100:+.1f}pt" if imp is not None else "    Improvement: N/A")
        else:
            print("\n--- Soccer: no data available, skipping ---")

    except Exception:
        print("\nERROR during baseline computation (continuing, will still write partial results):")
        traceback.print_exc()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {'timestamp': datetime.now().isoformat(), 'results': results}
    out_path = RESULTS_DIR / 'elo_baseline.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {out_path}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
