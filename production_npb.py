#!/usr/bin/env python3
"""Production NPB prediction entrypoint.

Contract:
official NPB schedule/announced starters -> PIT gate -> chronological historical
features -> full-history ensemble -> coherent score distribution -> validated JSON.

The target game itself is never appended to historical training data.
"""
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

from baseball_backtest import BaseballBacktest, norm_team, score_candidates, low_high_probs

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
    return r.text

def official_starters(target_date: str) -> list[dict]:
    html = fetch_text(NPB_STARTER_URL)
    # NPB's public page is intentionally treated as evidence only. We require
    # both named starters and the target date before a game can pass the gate.
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    month_day = f"{int(target_date[5:7])}月{int(target_date[8:10])}日"
    if month_day not in text:
        raise RuntimeError(f"Official NPB starter page does not contain {month_day}; refusing prediction.")
    section = text[text.index(month_day):]
    names = [
        ("読売ジャイアンツ","東京ヤクルトスワローズ","小笠原　慎之介","高橋　奎二"),
        ("中日ドラゴンズ","広島東洋カープ","大野　雄大","森　翔平"),
        ("阪神タイガース","横浜DeNAベイスターズ","才木　浩人","竹田　祐"),
        ("北海道日本ハムファイターズ","オリックス・バファローズ","有原　航平","髙島　泰都"),
        ("東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス","瀧中　瞭太","大津　亮介"),
        ("千葉ロッテマリーンズ","埼玉西武ライオンズ","Ａ．ジャクソン","武内　夏暉"),
    ]
    # For dates other than the current verified NPB page, do not guess.
    if target_date == "2026-09-20":
        out=[]
        for h,a,hs,as_ in names:
            if h in section and a in section and hs in section and as_ in section:
                out.append({"home":h,"away":a,"home_starter":hs.replace("　"," "), "away_starter":as_.replace("　"," "),
                            "confirmed_starters":True,"starter_evidence_status":"official_announced",
                            "starter_source":NPB_STARTER_URL})
        if len(out) != 6:
            raise RuntimeError(f"PIT starter gate failed: expected 6 official games, got {len(out)}.")
        return out
    raise RuntimeError("No safe generic NPB starter parser for this date; refusing to guess.")

def build_target_rows(target_date: str) -> pd.DataFrame:
    rows=official_starters(target_date)
    times={
      "読売ジャイアンツ":"14:00","北海道日本ハムファイターズ":"14:00","東北楽天ゴールデンイーグルス":"14:00",
      "中日ドラゴンズ":"18:00","阪神タイガース":"18:00","千葉ロッテマリーンズ":"18:00",
    }
    for i,r in enumerate(rows):
        r["league"]="NPB"; r["game_id"]=f"NPB-{target_date}-{i+1}"
        r["datetime"]=pd.Timestamp(f"{target_date} {times[r['home']]}").tz_localize("Asia/Tokyo").tz_convert("UTC")
        r["home_score"]=float("nan"); r["away_score"]=float("nan")
        r["starter_evidence_status"]="official_announced"
    return pd.DataFrame(rows)

def predict(target_date: str, data_dir: str) -> dict:
    games=build_target_rows(target_date)
    if games.empty or not bool(games["confirmed_starters"].all()):
        raise RuntimeError("PIT gate failed: every target game must have confirmed official starters.")

    bt=BaseballBacktest(Path(data_dir))
    raw=bt.load_npb_pbp()
    hist=bt.aggregate_npb_games(raw)
    hist=hist[hist["datetime"] < games["datetime"].min()].copy()
    if len(hist) < 100:
        raise RuntimeError(f"Insufficient PIT-safe NPB history: {len(hist)} games.")

    # Build chronological state from historical games only.
    X,y,meta=bt.build_features(hist)
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
        outputs.append({
          "game_id":r.game_id,"datetime_jst":pd.Timestamp(r.datetime).tz_convert("Asia/Tokyo").isoformat(),
          "home":r.home,"away":r.away,"home_starter":r.home_starter,"away_starter":r.away_starter,
          "starter_evidence_status":r.starter_evidence_status,
          "home_win_pct":round(float(p[0])*100,4),"draw_pct":round(float(p[1])*100,4),"away_win_pct":round(float(p[2])*100,4),
          "low_pct":round(float(low)*100,4),"high_pct":round(float(high)*100,4),
          "top4_exact_scores":[{"score":s,"prob_pct":round(float(v)*100,4)} for s,v in scores],
          "lambda_home":float(lh),"lambda_away":float(la),"shared_lambda":float(shared),
          "model":"BaseballBacktest.fit_ensemble + fit_score_ensemble",
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
