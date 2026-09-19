#!/usr/bin/env python3
"""Production NPB prediction entrypoint.

Contract:
official NPB schedule/announced starters -> PIT gate -> chronological historical
features -> full-history ensemble -> coherent score distribution -> validated JSON.

The target game itself is never appended to historical training data.
"""
from __future__ import annotations
import argparse, json, re, html as html_lib
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

from baseball_backtest import BaseballBacktest, norm_team, score_candidates, low_high_probs
from research.correlated_score import npb_final_outcomes

ROOT = Path(__file__).resolve().parent
TIMEOUT = 30
NPB_STARTER_URL = "https://npb.jp/announcement/starter/"
NPB_DAY_URL = "https://npb.jp/bis/eng/2026/games/gm{date}.html"

TEAM_MAP = {
    "Yomiuri":"読売ジャイアンツ","Yakult":"東京ヤクルトスワローズ",
    "Chunichi":"中日ドラゴンズ","Hiroshima":"広島東洋カープ",
    "Hanshin":"阪神タイガース","DeNA":"横浜DeNAベイスターズ",
    "Nippon-Ham":"北海道日本ハムファイターズ","ORIX":"オリックス・バファローズ",
    "Rakuten":"東北楽天ゴールデンイーグルス","SoftBank":"福岡ソフトバンクホークス",
    "Lotte":"千葉ロッテマリーンズ","Seibu":"埼玉西武ライオンズ",
}

def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent":"Baseball-Prediction-System/production"})
    r.raise_for_status()
    enc = (r.apparent_encoding or r.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
        return r.content.decode("cp932", errors="strict")
    return r.content.decode(r.apparent_encoding or r.encoding or "utf-8", errors="strict")

def _clean_name(value: str) -> str:
    return re.sub(r"\s+", " ", html_lib.unescape(value)).replace("　", " ").strip()

def parse_official_starters_html(page_html: str, target_date: str) -> list[dict]:
    """Parse the current NPB official announced-starter page without guessing."""
    month_day = f"{int(target_date[5:7])}月{int(target_date[8:10])}日"
    heading = re.search(rf"<h4[^>]*>\s*{re.escape(month_day)}の予告先発投手\s*</h4>", page_html, re.I)
    if not heading:
        raise RuntimeError(f"Official NPB starter page does not contain {month_day}; refusing prediction.")
    tail = page_html[heading.end():]
    next_heading = re.search(r"<h4\b", tail, re.I)
    section = tail[:next_heading.start()] if next_heading else tail

    teams = [
        "読売ジャイアンツ","東京ヤクルトスワローズ","中日ドラゴンズ","広島東洋カープ",
        "阪神タイガース","横浜DeNAベイスターズ","北海道日本ハムファイターズ",
        "オリックス・バファローズ","東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス",
        "千葉ロッテマリーンズ","埼玉西武ライオンズ",
    ]
    occurrences=[]
    for team in teams:
        pat = rf'alt=["\']{re.escape(team)}["\'][^>]*>.*?<a[^>]*>([^<]+)</a>'
        m = re.search(pat, section, re.I | re.S)
        if not m:
            continue
        pos = m.start()
        occurrences.append((pos, team, _clean_name(m.group(1))))
    occurrences.sort()
    times=[m.group(1) for m in re.finditer(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", section)]
    if len(occurrences) != 12 or len(times) < 6:
        raise RuntimeError(f"PIT starter gate failed: expected 12 team/starter pairs and 6 times, got {len(occurrences)} and {len(times)}.")
    out=[]
    for i in range(0,12,2):
        out.append({
            "home":occurrences[i][1],"away":occurrences[i+1][1],
            "home_starter":occurrences[i][2],"away_starter":occurrences[i+1][2],
            "confirmed_starters":True,"starter_evidence_status":"official_announced",
            "starter_source":NPB_STARTER_URL,"official_start_time":times[i//2],
        })
    return out

def official_starters(target_date: str) -> list[dict]:
    return parse_official_starters_html(fetch_text(NPB_STARTER_URL), target_date)

def build_target_rows(target_date: str) -> pd.DataFrame:
    rows=official_starters(target_date)
    for i,r in enumerate(rows):
        r["league"]="NPB"; r["game_id"]=f"NPB-{target_date}-{i+1}"
        start_time=r.get("official_start_time")
        if not start_time:
            raise RuntimeError("Official NPB schedule time missing; refusing prediction.")
        r["datetime"]=pd.Timestamp(f"{target_date} {start_time}").tz_localize("Asia/Tokyo").tz_convert("UTC")
        r["home_score"]=float("nan"); r["away_score"]=float("nan")
        r["starter_evidence_status"]="official_announced"
    return pd.DataFrame(rows)

def predict(target_date: str, data_dir: str) -> dict:
    try:
        games=build_target_rows(target_date)
    except RuntimeError as exc:
        # A missing official date section means NPB has not published the
        # target day's announced starters yet. Fail closed, but persist a
        # machine-readable blocked state so scheduled retries can continue.
        if "Official NPB starter page does not contain" not in str(exc):
            raise
        result={
            "schema_version":"npb-production-v1", "target_date":target_date,
            "execution_status":"BLOCKED_STARTERS", "pit_status":"NOT_RUN",
            "starter_gate":"BLOCKED", "model_status":"NOT_RUN",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),
            "predictions":[], "block_reason":str(exc),
            "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        }
        out=ROOT/"results"/f"npb_production_{target_date}.json"; out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
    if games.empty or not bool(games["confirmed_starters"].all()):
        raise RuntimeError("PIT gate failed: every target game must have confirmed official starters.")

    bt=BaseballBacktest(Path(data_dir))
    raw=bt.load_npb_pbp()
    hist=bt.aggregate_npb_games(raw)
    hist=hist[hist["datetime"] < games["datetime"].min()].copy()
    if hist["datetime"].max() >= games["datetime"].min():
        raise RuntimeError("PIT history contamination: historical data reaches target cutoff.")
    if hist["game_id"].isin(games["game_id"]).any():
        raise RuntimeError("PIT history contamination: target game appears in training history.")
    if len(hist) < 100:
        raise RuntimeError(f"Insufficient PIT-safe NPB history: {len(hist)} games.")

    # Build chronological state from historical games only.
    # Production training must replay chronology; never build target rows into history.
    # The existing feature builder is the canonical chronological state constructor.
    X,y,meta=bt.build_features(hist)
    if len(X) != len(hist) or len(y) != len(hist):
        raise RuntimeError("Chronological feature contract failed: feature/label row count mismatch.")
    fitted, validation_scores, _=bt.fit_ensemble(X,y,"NPB")
    if not fitted:
        raise RuntimeError("Production ensemble fitting failed.")

    score_fit=bt.fit_score_ensemble(X,hist["home_score"].astype(float).values,hist["away_score"].astype(float).values,"NPB")
    outputs=[]
    for _,r in games.iterrows():
        xrow=pd.DataFrame([bt.match_features(r)]).replace([float("inf"),float("-inf")],float("nan")).fillna(0.0).astype(float)
        p=bt.ensemble_proba(fitted,xrow,"NPB")[0]
        lh,la,shared=bt.predict_scores(score_fit,xrow,"NPB")
        split=float(max(-0.35,min(0.35,float(p[0]-p[2]))))
        lh*=1.0+0.08*split; la*=1.0-0.08*split
        scores=score_candidates(lh,la,shared,4)
        low,high=low_high_probs(lh,la,shared)
        # NPB final-result probabilities must distinguish a 9-inning tie from
        # a final draw. A tied regulation score can still be decided in innings
        # 10-12, so replace the raw 3-class classifier draw probability with the
        # coherent score/extra-inning result model while retaining the ensemble
        # probability difference as a small run-expectation adjustment above.
        home_final, draw_final, away_final = npb_final_outcomes(lh,la,shared)
        outputs.append({
          "game_id":r.game_id,"datetime_jst":pd.Timestamp(r.datetime).tz_convert("Asia/Tokyo").isoformat(),
          "home":r.home,"away":r.away,"home_starter":r.home_starter,"away_starter":r.away_starter,
          "starter_evidence_status":r.starter_evidence_status,
          "home_win_pct":round(float(home_final)*100,4),"draw_pct":round(float(draw_final)*100,4),"away_win_pct":round(float(away_final)*100,4),
          "low_pct":round(float(low)*100,4),"high_pct":round(float(high)*100,4),
          "top4_exact_scores":[{"score":s,"prob_pct":round(float(v)*100,4)} for s,v in scores],
          "lambda_home":float(lh),"lambda_away":float(la),"shared_lambda":float(shared),
          "model":"BaseballBacktest.fit_ensemble + fit_score_ensemble + NPB extra-inning result calibration",
          "validation_scores":validation_scores,
          "historical_games_used":int(len(hist)),
          "pit_status":"PASS",
          "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        })
    result={"schema_version":"npb-production-v1","target_date":target_date,"execution_status":"EXECUTED",
            "pit_status":"PASS","starter_gate":"PASS","model_status":"FITTED_ON_PIT_SAFE_HISTORY",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),"predictions":outputs}
    # Output validation: probabilities are finite, win probabilities sum to 100,
    # Low/High sum to 100, and exactly four score candidates exist.
    for o in outputs:
        assert abs(o["home_win_pct"]+o["draw_pct"]+o["away_win_pct"]-100) < 0.05
        assert abs(o["low_pct"]+o["high_pct"]-100) < 0.05
        assert len(o["top4_exact_scores"]) == 4
    out=ROOT/"results"/f"npb_production_{target_date}.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--date",required=True,help="YYYY-MM-DD, JST")
    ap.add_argument("--data-dir",default="data")
    args=ap.parse_args()
    print(json.dumps(predict(args.date,args.data_dir),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
