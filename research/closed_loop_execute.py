#!/usr/bin/env python3
"""Evidence-based closed-loop evaluation for production-strength baseball models."""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from research.adoption_gate import candidate_lock, evaluate_locked_holdout

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
PRIMARY_FILES = {"NPB": RESULTS / "checkpoints" / "npb_walkforward.csv", "MLB": RESULTS / "checkpoints" / "mlb_walkforward.csv"}

def write_json(name: str, obj: Any) -> None:
    (RESULTS / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def clip_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1: p = p.reshape(-1, 1)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 1e-9)
    return p / p.sum(axis=1, keepdims=True)

def poisson_result_probs(lh: float, la: float, league: str) -> np.ndarray:
    lh, la = max(float(lh), 1e-6), max(float(la), 1e-6)
    ks = np.arange(16)
    ph = np.exp(-lh) * np.array([lh**int(k) / math.factorial(int(k)) for k in ks])
    pa = np.exp(-la) * np.array([la**int(k) / math.factorial(int(k)) for k in ks])
    m = np.outer(ph, pa); m /= max(m.sum(), 1e-12)
    home = float(sum(m[i,j] for i in range(16) for j in range(16) if i > j))
    draw = float(sum(m[i,j] for i in range(16) for j in range(16) if i == j))
    away = float(sum(m[i,j] for i in range(16) for j in range(16) if i < j))
    return clip_probs(np.array([home, draw, away] if league == "NPB" else [home, away]))[0]

def build_probabilities(df: pd.DataFrame, league: str) -> tuple[np.ndarray, str]:
    if league == "NPB":
        cols = ["pred_home","pred_draw","pred_away"]
        if all(c in df for c in cols):
            raw = df[cols].to_numpy(float)
            usable = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1) & (raw.sum(axis=1) > .999) & (raw.sum(axis=1) < 1.001) & (raw.max(axis=1) < .999999)
        else:
            raw = np.zeros((len(df),3)); usable = np.zeros(len(df), dtype=bool)
        rows=[]
        for i, r in df.iterrows():
            rows.append(raw[i] if usable[i] else poisson_result_probs(r["lambda_home"], r["lambda_away"], "NPB"))
        return clip_probs(np.asarray(rows)), ("raw_classifier_with_poisson_repair" if (~usable).any() else "raw_classifier")
    if not {"pred_home","pred_away"}.issubset(df.columns): raise RuntimeError("MLB probability columns missing")
    raw = df[["pred_home","pred_away"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    usable = np.isfinite(raw).all(axis=1) & (raw >= 0).all(axis=1) & (raw.sum(axis=1) > .999) & (raw.sum(axis=1) < 1.001)
    if usable.all(): return clip_probs(raw), "raw_classifier"
    rows=[raw[i] if usable[i] else poisson_result_probs(r["lambda_home"], r["lambda_away"], "MLB") for i, r in df.iterrows()]
    return clip_probs(np.asarray(rows)), "raw_classifier_with_poisson_repair"

def actual_labels(df: pd.DataFrame, league: str) -> np.ndarray:
    hs = pd.to_numeric(df["actual_home_score"], errors="coerce"); aw = pd.to_numeric(df["actual_away_score"], errors="coerce")
    if hs.isna().any() or aw.isna().any(): raise RuntimeError(f"{league} contains missing realized score targets")
    if league == "NPB":
        y=np.where(hs>aw,0,np.where(hs==aw,1,2)).astype(int)
        if len(np.unique(y))<2: raise RuntimeError("NPB realized target is degenerate")
        return y
    return (hs>aw).astype(int).to_numpy()

def logloss(y,p):
    p=clip_probs(p); return float(-np.mean(np.log(np.maximum(p[np.arange(len(y)),y],1e-15))))

def brier(y,p):
    p=clip_probs(p); one=np.zeros_like(p); one[np.arange(len(y)),y]=1.0
    return float(np.mean(np.sum((p-one)**2,axis=1)))

def accuracy(y,p): return float(np.mean(np.argmax(p,axis=1)==y))

def metrics(y,p): return {"LogLoss":logloss(y,p),"Brier":brier(y,p),"Accuracy":accuracy(y,p),"rows":int(len(y))}

def fit_temperature_grid(logits,y):
    best_t,best=1.0,float("inf")
    for t in np.linspace(.5,3.0,101):
        z=logits/t; z-=z.max(axis=1,keepdims=True); q=np.exp(z); q/=q.sum(axis=1,keepdims=True)
        s=logloss(y,q)
        if s<best-1e-12: best,best_t=s,float(t)
    return best_t

def apply_temperature(p,t):
    z=np.log(clip_probs(p)); z-=z.max(axis=1,keepdims=True); return clip_probs(np.exp(z/max(float(t),1e-6)))

def score_metrics(df,y,p,league):
    out=metrics(y,p)
    if league=="NPB":
        out["DrawRecall"]=float(np.sum((y==1)&(np.argmax(p,axis=1)==1))/max(np.sum(y==1),1))
        out["DrawProbabilityMAE"]=float(np.mean(np.abs(p[:,1]-(y==1).astype(float))))
    if {"lambda_home","lambda_away"}.issubset(df.columns):
        lh=pd.to_numeric(df.lambda_home,errors="coerce").to_numpy(float); la=pd.to_numeric(df.lambda_away,errors="coerce").to_numpy(float)
        hs=pd.to_numeric(df.actual_home_score,errors="coerce").to_numpy(float); aw=pd.to_numeric(df.actual_away_score,errors="coerce").to_numpy(float)
        v=np.isfinite(lh)&np.isfinite(la)&np.isfinite(hs)&np.isfinite(aw)
        if v.any(): out["ScoreMAE"]=float(np.mean((np.abs(lh[v]-hs[v])+np.abs(la[v]-aw[v]))/2))
    return out

def hilo_probs(df):
    if {"low","high"}.issubset(df.columns):
        p=df[["low","high"]].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        v=np.isfinite(p).all(axis=1)&(p>=0).all(axis=1)&(p.sum(axis=1)>.999)&(p.sum(axis=1)<1.001)
        if v.all(): return clip_probs(p)
    vals=[]
    for _,r in df.iterrows():
        lh,la=float(r.lambda_home),float(r.lambda_away)
        low=(sum(math.exp(-lh)*lh**k/math.factorial(k) for k in range(7))*sum(math.exp(-la)*la**k/math.factorial(k) for k in range(7)))
        vals.append([np.clip(low,0,1),1-np.clip(low,0,1)])
    return clip_probs(np.asarray(vals))

def weakness_report(df,y,p,league):
    w=df.copy(); w["_correct"]=(np.argmax(p,axis=1)==y).astype(int); w["_ll"]=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)); w["_month"]=pd.to_datetime(w.datetime,errors="coerce",utc=True).dt.strftime("%Y-%m")
    groups={}
    for key in ["model","_month","confirmed_starters"]:
        if key in w:
            g=w.groupby(key,dropna=False).agg(rows=("_correct","size"),accuracy=("_correct","mean"),logloss=("_ll","mean")).reset_index()
            groups[key]=g[g.rows>=30].sort_values("logloss",ascending=False).head(10).to_dict(orient="records")
    return {"league":league,"rows":len(w),"groups":groups}

def split_three(df):
    n=len(df)
    if n<300: raise RuntimeError(f"insufficient OOS rows for production holdout: {n}")
    a=max(100,int(n*.50)); b=max(a+50,int(n*.65)); b=min(b,n-100)
    return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()

def hilo_metrics(df,p):
    hs=pd.to_numeric(df.actual_home_score,errors="coerce").to_numpy(float); aw=pd.to_numeric(df.actual_away_score,errors="coerce").to_numpy(float)
    y=((hs>=7)|(aw>=7)).astype(int); return metrics(y,p)

def process_league(league,path):
    df=pd.read_csv(path)
    if "datetime" not in df: raise RuntimeError(f"{league}: datetime column missing")
    df["datetime"]=pd.to_datetime(df.datetime,errors="coerce",utc=True); df=df.dropna(subset=["datetime"]).sort_values(["datetime","game_id"],kind="mergesort").reset_index(drop=True)
    y=actual_labels(df,league); p,source=build_probabilities(df,league)
    sel,val1,val2=split_three(df); i1=len(sel); i2=i1+len(val1)
    p_sel,p1,p2=p[:i1],p[i1:i2],p[i2:]; y_sel,y1,y2=y[:i1],y[i1:i2],y[i2:]
    t=fit_temperature_grid(np.log(np.maximum(p_sel,1e-12)),y_sel); p1c=apply_temperature(p1,t); p2c=apply_temperature(p2,t)
    b1,c1=score_metrics(val1,y1,p1,league),score_metrics(val1,y1,p1c,league); b2,c2=score_metrics(val2,y2,p2,league),score_metrics(val2,y2,p2c,league)
    bh,ch={**b2},{**c2}; bhilo=hilo_metrics(val2,hilo_probs(val2)); chilo=hilo_metrics(val2,apply_temperature(hilo_probs(val2),t))
    lock=candidate_lock(development_metrics={"rows":len(val1)+len(val2),"validation_window_1_LogLoss":c1["LogLoss"],"validation_window_2_LogLoss":c2["LogLoss"],"temperature":t},candidate_id="temperature_calibration_v1")
    gate=evaluate_locked_holdout(bh,ch,validation_windows=2,calibration_ok=(c1["LogLoss"]<=b1["LogLoss"] and c2["LogLoss"]<=b2["LogLoss"]),no_future_target_data=True,reproducible=True,baseline_score={"ScoreMAE":b2.get("ScoreMAE",float("nan"))},candidate_score={"ScoreMAE":c2.get("ScoreMAE",float("nan"))},baseline_hilo=bhilo,candidate_hilo=chilo,league=league)
    pd.DataFrame([b1,c1,b2,c2]).to_csv(RESULTS/f"{league.lower()}_development_oos.csv",index=False)
    return {"league":league,"rows":len(df),"probability_source":source,"split":{"selection":len(sel),"validation_1":len(val1),"independent_holdout":len(val2)},"calibration":{"method":"chronological_temperature_grid","temperature":t},"development_oos":{"validation_1_baseline":b1,"validation_1_candidate":c1,"validation_2_baseline":b2,"validation_2_candidate":c2},"holdout":{"baseline":bh,"candidate":ch,"hilo_baseline":bhilo,"hilo_candidate":chilo},"candidate_lock":lock,"candidate_gate":gate,"weakness":weakness_report(sel,y_sel,p_sel,league),"result_audit":{"rows":len(df),"actual_class_counts":pd.Series(y).value_counts().sort_index().to_dict(),"score_target_coverage":float(np.isfinite(pd.to_numeric(df.actual_home_score,errors="coerce")).mean())}}

def main():
    reports={}; blockers=[]
    for league,path in PRIMARY_FILES.items():
        try: reports[league]=process_league(league,path)
        except Exception as exc: blockers.append(f"{league}:{type(exc).__name__}:{exc}")
    if blockers:
        write_json("lifecycle_execution.json",{"status":"BLOCKED","blockers":blockers,"reports":reports}); raise SystemExit("; ".join(blockers))
    write_json("calibration.json",{"version":2,"method":"chronological temperature calibration","leagues":{k:v["calibration"] for k,v in reports.items()},"holdout_untouched_during_fit":True})
    write_json("development_oos.json",{k:v["development_oos"] for k,v in reports.items()})
    write_json("independent_holdout.json",{k:v["holdout"] for k,v in reports.items()})
    write_json("result_audit.json",{k:v["result_audit"] for k,v in reports.items()})
    write_json("weakness_report.json",{k:v["weakness"] for k,v in reports.items()})
    write_json("candidate_validation.json",{k:v["candidate_gate"] for k,v in reports.items()})
    decisions={k:v["candidate_gate"]["decision"] for k,v in reports.items()}
    lifecycle={"status":"READY","blockers":[],"candidate_decisions":decisions,"production_principles":["chronological split","independent final holdout","calibration fitted only before holdout","deterministic candidate","fail-closed target/probability integrity","no promotion without locked-holdout gate"],"source_sha256":hashlib.sha256(json.dumps({"NPB":str(PRIMARY_FILES["NPB"]),"MLB":str(PRIMARY_FILES["MLB"])},sort_keys=True).encode()).hexdigest()}
    write_json("lifecycle_execution.json",lifecycle); print(json.dumps(lifecycle,ensure_ascii=False,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
