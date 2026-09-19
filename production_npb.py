#!/usr/bin/env python3
"""Production NPB prediction entrypoint.

Contract:
official NPB schedule/announced starters -> PIT gate -> chronological historical
features -> full-history ensemble -> coherent score distribution -> validated JSON.

The target game itself is never appended to historical training data.
"""
from __future__ import annotations
import time
import argparse, json, re, html as html_lib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
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
    """Parse NPB's announced-starter section with semantic fail-closed validation."""
    month_day = f"{int(target_date[5:7])}月{int(target_date[8:10])}日"
    heading = re.search(
        rf"<h4[^>]*>\s*{re.escape(month_day)}の予告先発投手\s*</h4>",
        page_html, re.I,
    )
    if not heading:
        raise RuntimeError(
            f"Official NPB starter page does not contain {month_day}; refusing prediction."
        )
    tail = page_html[heading.end():]
    next_heading = re.search(r"<h4\b", tail, re.I)
    section = tail[:next_heading.start()] if next_heading else tail

    teams = [
        "読売ジャイアンツ","東京ヤクルトスワローズ","中日ドラゴンズ","広島東洋カープ",
        "阪神タイガース","横浜DeNAベイスターズ","北海道日本ハムファイターズ",
        "オリックス・バファローズ","東北楽天ゴールデンイーグルス","福岡ソフトバンクホークス",
        "千葉ロッテマリーンズ","埼玉西武ライオンズ",
    ]
    occurrences = []
    for team in teams:
        team_pat = re.escape(team)
        # On the live NPB page each game is a .unit whose team logo is
        # immediately associated with a player <span>. Bound the search so
        # footer/navigation links can never become a pitcher.
        m = re.search(
            rf'<div[^>]+class=["\'][^"\']*unit[^"\']*["\'][^>]*>.*?'
            rf'<img[^>]+alt=["\']{team_pat}["\'][^>]*>.*?'
            rf'<(?:div|p)[^>]+class=["\'][^"\']*team_left[^"\']*["\'][^>]*>.*?'
            rf'<span[^>]*>\s*([^<]+?)\s*</span>',
            section, re.I | re.S,
        )
        if not m:
            # Deterministic fixture/backward-compatible fallback: still bound
            # extraction to a short window after the team's logo.
            tm = re.search(rf'alt=["\']{team_pat}["\']', section, re.I)
            if tm:
                window = section[tm.end():tm.end()+1200]
                sm = re.search(r'<span[^>]*>\s*([^<]+?)\\s*</span>', window, re.I | re.S)
                if not sm:
                    sm = re.search(r'<a[^>]*>\s*([^<]+?)\s*</a>', window, re.I | re.S)
                if sm:
                    m = sm
                    name = _clean_name(sm.group(1))
                    occurrences.append((tm.start(), team, name))
                    continue
            continue
        name = _clean_name(m.group(1))
        occurrences.append((m.start(), team, name))

    occurrences.sort()
    times = [m.group(1) for m in re.finditer(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)", section)]
    if len(occurrences) != 12 or len(times) < 6:
        raise RuntimeError(
            f"PIT starter gate failed: expected 12 team/starter pairs and 6 times, "
            f"got {len(occurrences)} and {len(times)}."
        )

    invalid = {"一般社団法人日本野球機構について", "採用情報", "プライバシーポリシー"}
    for _, team, name in occurrences:
        if not name or name in invalid or name in teams or "日本野球機構" in name:
            raise RuntimeError(f"PIT starter gate failed: invalid starter extracted for {team}: {name!r}")

    out = []
    for i in range(0, 12, 2):
        home_team, home_starter = occurrences[i][1], occurrences[i][2]
        away_team, away_starter = occurrences[i+1][1], occurrences[i+1][2]
        out.append({
            "home": home_team, "away": away_team,
            "home_starter": home_starter, "away_starter": away_starter,
            "confirmed_starters": True, "starter_evidence_status": "official_announced",
            "starter_source": NPB_STARTER_URL, "official_start_time": times[i//2],
        })
    return out

def _load_official_starter_snapshot(target_date: str) -> list[dict] | None:
    path = ROOT / "data" / "official_starters" / f"{target_date}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "npb-official-starter-snapshot-v1":
        raise RuntimeError("Official starter snapshot schema mismatch; refusing prediction.")
    if payload.get("target_date") != target_date or payload.get("source_type") != "NPB_OFFICIAL":
        raise RuntimeError("Official starter snapshot provenance mismatch; refusing prediction.")
    retrieved = pd.Timestamp(payload.get("retrieved_at_utc"))
    if retrieved.tzinfo is None:
        retrieved = retrieved.tz_localize("UTC")
    games = payload.get("games", [])
    if len(games) != 6:
        raise RuntimeError("Official starter snapshot must contain exactly 6 games.")
    for g in games:
        g["confirmed_starters"] = True
        g["starter_evidence_status"] = "official_announced_snapshot"
        g["starter_source"] = payload["source_url"]
    return games

def official_starters(target_date: str) -> list[dict]:
    try:
        return parse_official_starters_html(
            fetch_text(NPB_STARTER_URL + "?_ts=" + str(int(time.time()))), target_date
        )
    except RuntimeError as exc:
        if "does not contain" not in str(exc):
            raise
        snapshot = _load_official_starter_snapshot(target_date)
        if snapshot is not None:
            return snapshot
        raise

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

def direct_pit_safe_lambdas(hist: pd.DataFrame, row: pd.Series, bt: BaseballBacktest) -> tuple[float,float,float]:
    """Fast independent PIT-safe run-rate model from historical games only.

    It deliberately avoids target-game-derived state and uses exponentially weighted
    team offense/defense plus announced-starter historical quality. This is the
    production fallback when the ML score ensemble is numerically degenerate.
    """
    league="NPB"; cutoff=pd.Timestamp(row["datetime"])
    h=norm_team(row["home"],league); a=norm_team(row["away"],league)
    hh=hist[hist["home"].map(lambda x:norm_team(x,league))==h]
    ha=hist[hist["away"].map(lambda x:norm_team(x,league))==h]
    ah=hist[hist["home"].map(lambda x:norm_team(x,league))==a]
    aa=hist[hist["away"].map(lambda x:norm_team(x,league))==a]
    hist=hist.sort_values(["datetime","game_id"]).copy()
    def ew_team(team, side, scored, default):
        rows=[]
        for _,g in hist.iterrows():
            if norm_team(g["home"],league)==team:
                val=g["home_score"] if scored else g["away_score"]
                rows.append((g["datetime"],float(val)))
            elif norm_team(g["away"],league)==team:
                val=g["away_score"] if scored else g["home_score"]
                rows.append((g["datetime"],float(val)))
        if not rows:return default
        vals=np.asarray([v for _,v in rows[-30:]],dtype=float)
        w=np.exp(np.linspace(-2.2,0,len(vals)))
        return float(np.average(vals,weights=w))
    league_h=float(pd.to_numeric(hist["home_score"],errors="coerce").mean())
    league_a=float(pd.to_numeric(hist["away_score"],errors="coerce").mean())
    league_mean=max(1.0,0.5*(league_h+league_a))
    h_for=ew_team(h,"any",True,league_mean); h_against=ew_team(h,"any",False,league_mean)
    a_for=ew_team(a,"any",True,league_mean); a_against=ew_team(a,"any",False,league_mean)
    # Matchup run rates are team-specific and chronology-preserving. A small
    # home-field multiplier is applied after the PIT-safe team interaction.
    lh=max(.65,min(7.0,0.55*h_for+0.45*a_against))
    la=max(.65,min(7.0,0.55*a_for+0.45*h_against))
    lh*=1.035
    hs=bt.starter_features(league,str(row.get("home_starter","") or ""),cutoff,"hs_")
    aas=bt.starter_features(league,str(row.get("away_starter","") or ""),cutoff,"as_")
    # Shrunk starter adjustment: stronger historical FIP suppresses opponent scoring.
    lh*=float(np.exp(np.clip((float(aas.get("as_fip",4.0))-4.0)*0.055,-0.20,0.20)))
    la*=float(np.exp(np.clip((float(hs.get("hs_fip",4.0))-4.0)*0.055,-0.20,0.20)))
    return float(max(.65,min(7.0,lh))),float(max(.65,min(7.0,la))),0.0

def blend_classifier_run_share(lh: float, la: float, p: np.ndarray, weight: float = 0.25) -> tuple[float,float]:
    """Use the chronological classifier only for home/away run-share direction.

    The coherent score model controls total expected runs; the classifier adds a
    bounded matchup signal without allowing its NPB draw class to inflate final
    draws. All inputs are target-time PIT-safe.
    """
    total=max(float(lh)+float(la),1e-6)
    score_share=float(lh)/total
    hp=float(p[0]); ap=float(p[2])
    denom=max(hp+ap,1e-9)
    clf_share=hp/denom
    share=(1.0-weight)*score_share+weight*clf_share
    share=max(0.15,min(0.85,share))
    return total*share,total*(1.0-share)


def robust_target_lambdas(bt: BaseballBacktest, hist: pd.DataFrame, row: pd.Series) -> tuple[float,float,float]:
    """Fail-safe target-specific run model used only when the fitted score ensemble degenerates."""
    league="NPB"; dt=pd.Timestamp(row["datetime"])
    hteam,a_team=norm_team(row["home"],league),norm_team(row["away"],league)
    hf=bt._team_features(league,hteam,"home",dt); af=bt._team_features(league,a_team,"away",dt)
    if hf.get("matches",0.0)<5 or af.get("matches",0.0)<5:
        raise RuntimeError("Target-specific state coverage too low; refusing fallback prediction.")
    gh=float(hist["home_score"].mean()); ga=float(hist["away_score"].mean())
    h_attack=0.65*hf.get("gf_10",gh)+0.35*gh
    a_attack=0.65*af.get("gf_10",ga)+0.35*ga
    h_def=0.65*af.get("ga_10",gh)+0.35*gh
    a_def=0.65*hf.get("ga_10",ga)+0.35*ga
    lh=0.52*h_attack+0.48*h_def
    la=0.52*a_attack+0.48*a_def
    hs=bt.starter_features(league,str(row.get("home_starter","") or ""),dt,"hs_")
    aas=bt.starter_features(league,str(row.get("away_starter","") or ""),dt,"as_")
    lh*=float(__import__("math").exp(0.07*(float(aas.get("as_fip",4.0))-4.0)))
    la*=float(__import__("math").exp(0.07*(float(hs.get("hs_fip",4.0))-4.0)))
    lh*=1.035
    lh=max(0.8,min(6.0,lh)); la=max(0.8,min(6.0,la))
    return lh,la,0.0

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
    hist_total = pd.to_numeric(hist["home_score"], errors="coerce") + pd.to_numeric(hist["away_score"], errors="coerce")
    hist_total = hist_total.replace([np.inf, -np.inf], np.nan).dropna()
    if len(hist_total) < 100:
        raise RuntimeError("Historical score labels are insufficient for production.")
    hist_score_mean = float(hist_total.mean())
    hist_score_zero_rate = float((hist_total <= 0).mean())
    if hist_score_mean < 3.0 or hist_score_zero_rate > 0.08 or hist_total.nunique() < 5:
        raise RuntimeError(
            f"Historical score data quality failed: mean_total={hist_score_mean:.3f}, "
            f"zero_rate={hist_score_zero_rate:.3f}, unique_totals={hist_total.nunique()}."
        )

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
        fallback_used = score_fit is None or (abs(lh-la)<1e-12 and abs(lh-2.35)<1e-12)
        if fallback_used:
            lh,la,shared=direct_pit_safe_lambdas(hist,r,bt)
            model_label="Production ML ensemble + PIT-safe direct run-rate fallback (degeneracy recovery)"
        else:
            model_label="BaseballBacktest.fit_ensemble + fit_score_ensemble + NPB extra-inning result calibration"
        lh,la=blend_classifier_run_share(lh,la,p,weight=0.25)
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
          "model":model_label,
          "validation_scores":validation_scores,
          "historical_games_used":int(len(hist)),
          "pit_status":"PASS",
          "prediction_generated_at":datetime.now(timezone.utc).isoformat(),
        })
    result={"schema_version":"npb-production-v1","target_date":target_date,"execution_status":"EXECUTED",
            "pit_status":"PASS","starter_gate":"PASS","model_status":"FITTED_ON_PIT_SAFE_HISTORY",
            "git_commit":__import__("os").environ.get("GITHUB_SHA","unknown"),
            "data_quality_status":"PASS",
            "historical_score_mean_total":round(hist_score_mean,6),
            "historical_score_zero_rate":round(hist_score_zero_rate,6),
            "predictions":outputs}
    # Recover from silent cross-target model collapse instead of emitting a
    # misleadingly uniform forecast. The recovery remains PIT-safe because it
    # recomputes every target from historical games strictly before the target set.
    if len(outputs) == 6:
        sig={(round(o["lambda_home"],6),round(o["lambda_away"],6),round(o["home_win_pct"],4),round(o["away_win_pct"],4)) for o in outputs}
        if len(sig) < 3:
            recovered=[]
            for idx, (_, r) in enumerate(games.iterrows()):
                xrow=pd.DataFrame([bt.match_features(r)]).replace([float("inf"),float("-inf")],float("nan")).fillna(0.0).astype(float)
                p=bt.ensemble_proba(fitted,xrow,"NPB")[0]
                # Prefer the raw historical run-rate recovery here. It is
                # independent of target state and therefore remains PIT-safe, while
                # also avoiding the failure mode where a sparse team-state feature
                # vector collapses every target to the same lower clamp.
                try:
                    lh,la,shared=direct_pit_safe_lambdas(hist,r,bt)
                    recovery_model="Production ML ensemble + PIT-safe chronological direct run-rate recovery"
                    if abs(lh-la) < 1e-6 and abs(lh-0.65) < 1e-6:
                        raise RuntimeError("direct run-rate recovery collapsed to floor")
                except RuntimeError:
                    lh,la,shared=robust_target_lambdas(bt,hist,r)
                    recovery_model="Production ML ensemble + PIT-safe chronological team-state recovery"
                lh,la=blend_classifier_run_share(lh,la,p,weight=0.25)
                scores=score_candidates(lh,la,shared,4)
                low,high=low_high_probs(lh,la,shared)
                home_final,draw_final,away_final=npb_final_outcomes(lh,la,shared)
                o=outputs[idx].copy()
                o.update({
                    "home_win_pct":round(float(home_final)*100,4),
                    "draw_pct":round(float(draw_final)*100,4),
                    "away_win_pct":round(float(away_final)*100,4),
                    "low_pct":round(float(low)*100,4),
                    "high_pct":round(float(high)*100,4),
                    "top4_exact_scores":[{"score":s,"prob_pct":round(float(v)*100,4)} for s,v in scores],
                    "lambda_home":float(lh),"lambda_away":float(la),"shared_lambda":float(shared),
                    "model":recovery_model,
                })
                recovered.append(o)
            outputs=recovered
            result["predictions"]=outputs
            sig={(round(o["lambda_home"],6),round(o["lambda_away"],6),round(o["home_win_pct"],4),round(o["away_win_pct"],4)) for o in outputs}
            if len(sig) < 2:
                debug = [
                    {
                        "home": o["home"], "away": o["away"],
                        "lambda_home": o["lambda_home"], "lambda_away": o["lambda_away"],
                        "home_win_pct": o["home_win_pct"], "away_win_pct": o["away_win_pct"],
                    }
                    for o in outputs
                ]
                teams = sorted(set(hist["home"].map(lambda x: norm_team(x, "NPB"))) | set(hist["away"].map(lambda x: norm_team(x, "NPB"))))
                print("PRODUCTION_DEGENERACY_DEBUG", json.dumps({"predictions": debug, "historical_team_count": len(teams), "historical_teams": teams}, ensure_ascii=False))
                raise RuntimeError("Production degeneracy guard: PIT-safe recovery remained insufficiently differentiated.")
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

# Production execution trigger: use current JST date for manual verification.
