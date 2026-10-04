"""Chronological WFO evaluator and checkpoint/resume layer for game-script v4."""
from __future__ import annotations

import hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .game_script_common import (
    PIT_STATUS, SCHEMA_VERSION, TeamStrength, TransitionKernel, _base, _count,
    _team_key, aggregate, boundary, canonicalize_pbp_frame, simulate_game, summary,
)


def record(game: pd.DataFrame) -> dict[str, Any]:
    g=game.copy(); g["_h"]=g.half.map({"T":0,"B":1}); x=g.sort_values(["inning","_h","play_order","pitch_number"],kind="mergesort").iloc[-1]
    return {"game_id":str(x.game_id),"game_date":pd.Timestamp(x.game_date),"home":_team_key(x.home_team_id,x.home),"away":_team_key(x.away_team_id,x.away),"home_score":int(x.state_home_score),"away_score":int(x.state_away_score),"complete_status":str(x.complete_status)}


def iter_games(paths: Iterable[Path]):
    for p in sorted(paths):
        f=canonicalize_pbp_frame(pd.read_csv(p,low_memory=False))
        for _,g in f.groupby("game_id",sort=False):
            g=g.copy(); g["_h"]=g.half.map({"T":0,"B":1}); yield g.sort_values(["inning","_h","play_order","pitch_number"],kind="mergesort").drop(columns=["_h"])


def _cfg(files: Sequence[Path], **kwargs): return {**kwargs,"schema_version":SCHEMA_VERSION,"input_files":[{"path":str(p),"size":p.stat().st_size} for p in files]}


def load_checkpoint(path: Path|None,cfg):
    if path is None or not path.exists(): return set(),[],[]
    try: obj=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return set(),[],[]
    if obj.get("config")!=dict(cfg): return set(),[],[]
    rows=[r for r in obj.get("validation_rows",[]) if isinstance(r,dict)]
    ab=[r for r in obj.get("ablation_rows",[]) if isinstance(r,dict)]
    return {str(x) for x in obj.get("processed_game_ids",[])},rows,ab


def save_checkpoint(path:Path,cfg,ids,rows,ablation):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps({"checkpoint_schema":1,"config":dict(cfg),"processed_game_ids":sorted(ids),"validation_rows":list(rows),"ablation_rows":list(ablation)},ensure_ascii=False,indent=2,default=str)+"
",encoding="utf-8")
    tmp.replace(path)


def evaluate(paths: Sequence[str|Path], *, development_end="2024-12-31", validation_start="2025-01-01", validation_end="2025-12-31", shadow_start=None, shadow_end=None, max_validation_games=160, max_shadow_games=60, simulations=800, seed=42, ablation=True, checkpoint_path=None, checkpoint_every=1):
    files=sorted(Path(p) for p in paths); dev_end=boundary(development_end,end_of_day=True); val_start=boundary(validation_start); val_end=boundary(validation_end,end_of_day=True); sh_start=boundary(shadow_start); sh_end=boundary(shadow_end,end_of_day=True) if shadow_end else None
    if not files: raise ValueError("no input files matched")
    ck=Path(checkpoint_path) if checkpoint_path else None; cfg=_cfg(files,development_end=development_end,validation_start=validation_start,validation_end=validation_end,max_validation_games=int(max_validation_games),simulations=int(simulations),seed=int(seed))
    kernel=TransitionKernel(); strength=TeamStrength(); dev_games=games_seen=0; cov=Counter()
    for game in iter_games(files):
        rec=record(game); games_seen+=1; cov["rows"]+=len(game); rows=game.to_dict("records")
        cov["count"]+=sum(_count(r) is not None for r in rows); cov["base"]+=sum(_base(r) is not None for r in rows)
        cov["batter"]+=int(game.batter_id.ne("").sum()); cov["pitcher"]+=int(game.pitcher_id.ne("").sum()); cov["pitch"]+=int(game.pitch_id.notna().sum()); cov["speed"]+=int(game.release_speed_kmh.notna().sum()); cov["xy"]+=int(game[["plate_x","plate_y"]].notna().all(axis=1).sum())
        if rec["game_date"]<=dev_end and rec["complete_status"]=="PASS" and kernel.add_game(game):
            strength.update(rec["home"],rec["away"],rec["home_score"],rec["away_score"],rec["game_date"],dev_end); dev_games+=1
    if dev_games==0 or kernel.transitions<100: raise ValueError(f"insufficient development support: games={dev_games} transitions={kernel.transitions}")
    processed,val_rows,ablation_rows=load_checkpoint(ck,cfg); processed=set(processed); val_rows=list(val_rows); ablation_rows=list(ablation_rows); done=len(processed)
    if done: print(f"[CHECKPOINT] restoring {done} validation games")
    for game in iter_games(files):
        rec=record(game)
        if rec["game_date"]<val_start: continue
        if rec["game_date"]>val_end: break
        if rec["complete_status"]!="PASS": continue
        if done>=max_validation_games: break
        if rec["game_id"] in processed:
            kernel.add_game(game); strength.update(rec["home"],rec["away"],rec["home_score"],rec["away_score"],rec["game_date"],rec["game_date"]); continue
        hf,af=strength.factors(rec["home"],rec["away"]); gid=int.from_bytes(hashlib.sha256(rec["game_id"].encode()).digest()[:4],"big")
        sim=simulate_game(kernel,home_factor=hf,away_factor=af,simulations=simulations,seed=seed^gid,profile="rich"); s=summary(np.asarray(sim["score_distribution"],float),rec["home_score"],rec["away_score"])
        val_rows.append({"game_id":rec["game_id"],"game_date":str(rec["game_date"]),"period":str(rec["game_date"])[:7],**s,"predictability":sim["uncertainty"]["outcome_predictability"],"fallback_rate":sim["uncertainty"]["fallback_rate"],"mc_se_home_win":sim["uncertainty"]["mc_se_home_win"]})
        if ablation:
            c=simulate_game(kernel,home_factor=hf,away_factor=af,simulations=max(250,simulations//2),seed=seed^gid^0xA5A5,profile="coarse"); cs=summary(np.asarray(c["score_distribution"],float),rec["home_score"],rec["away_score"])
            ablation_rows.append({"rich_logloss":s["logloss"],"coarse_logloss":cs["logloss"],"rich_brier":s["brier"],"coarse_brier":cs["brier"]})
        kernel.add_game(game); strength.update(rec["home"],rec["away"],rec["home_score"],rec["away_score"],rec["game_date"],rec["game_date"]); processed.add(rec["game_id"]); done+=1
        if ck and done%max(1,int(checkpoint_every))==0: save_checkpoint(ck,cfg,processed,val_rows,ablation_rows)
    if ck: save_checkpoint(ck,cfg,processed,val_rows,ablation_rows)
    if not val_rows: raise ValueError("validation set is empty")
    v=pd.DataFrame(val_rows);\n    if v.empty:\n        raise ValueError("validation set is empty")\n    warmup={"status":"NOT_REQUESTED","games":0,"transitions":0}
    if sh_start is not None:
        warmup_latest=sh_start-pd.Timedelta(nanoseconds=1)
        warmup_games=0
        warmup_transitions=0
        for game in iter_games(files):
            rec=record(game)
            if rec["game_date"]<=val_end:
                continue
            if rec["game_date"]>=sh_start:
                break
            if rec["complete_status"]!="PASS":
                continue
            added=kernel.add_game(game)
            if added:
                strength.update(rec["home"],rec["away"],rec["home_score"],rec["away_score"],rec["game_date"],warmup_latest)
                warmup_games+=1
                warmup_transitions+=added
        warmup={
            "status":"RECENT_CONTEXT_WARMUP",
            "games":int(warmup_games),
            "transitions":int(warmup_transitions),
            "start_exclusive":str(val_end),
            "end_exclusive":str(sh_start),
            "updates_used_for_shadow_only":True,
        }
    shadow={"status":"NOT_REQUESTED","rows":0}
    if sh_start is not None and max_shadow_games>0:
        sr=[]
        for game in iter_games(files):
            rec=record(game)
            if rec["game_date"]<sh_start: continue
            if sh_end is not None and rec["game_date"]>sh_end: break
            if rec["complete_status"]!="PASS": continue
            if len(sr)>=max_shadow_games: break
            hf,af=strength.factors(rec["home"],rec["away"]); gid=int.from_bytes(hashlib.sha256(rec["game_id"].encode()).digest()[:4],"big"); sim=simulate_game(kernel,home_factor=hf,away_factor=af,simulations=simulations,seed=seed^gid,profile="rich")
            sr.append({"game_id":rec["game_id"],**summary(np.asarray(sim["score_distribution"],float),rec["home_score"],rec["away_score"])})
        sdf=pd.DataFrame(sr); shadow={"status":"FROZEN_SHADOW_ONLY","rows":len(sdf),"aggregate":aggregate(sdf) if not sdf.empty else {},"updates":"FORBIDDEN","cases":sr}
    adf=pd.DataFrame(ablation_rows); ab={"rows":len(adf)}
    if not adf.empty: ab.update({"rich_logloss":float(adf.rich_logloss.mean()),"coarse_logloss":float(adf.coarse_logloss.mean()),"rich_vs_coarse_logloss_relative_improvement":float((adf.coarse_logloss.mean()-adf.rich_logloss.mean())/max(1e-12,adf.coarse_logloss.mean()))})
    total=max(1,cov["rows"]); coverage={"sample_rows":int(cov["rows"]),"count_available":float(cov["count"]/total),"base_state_complete":float(cov["base"]/total),"batter_id_available":float(cov["batter"]/total),"pitcher_id_available":float(cov["pitcher"]/total),"pitch_id_available":float(cov["pitch"]/total),"release_speed_available":float(cov["speed"]/total),"plate_xy_available":float(cov["xy"]/total)}
    latest=v.tail(min(30,len(v)))
    return {"schema_version":SCHEMA_VERSION,"validation_cases":val_rows,"shadow_cases":shadow.get("cases",[]),"warmup":warmup,"status":"RESEARCH_SCREENING_ONLY","candidate_role":"CHALLENGER_RESEARCH","pit_status":PIT_STATUS,"production_eligible":False,"decision":"HOLD_RESEARCH_ONLY","reason":"Historical PBP publication/availability timestamps are not proven; no production OOS or promotion evidence is claimed.","development_end":development_end,"validation_start":validation_start,"validation_end":validation_end,"shadow_start":shadow_start or "","shadow_end":shadow_end or "","games_seen":games_seen,"development_games":dev_games,"validation_games":len(v),"kernel":kernel.snapshot(),"aggregate":{"validation":aggregate(v),"latest_validation_30":aggregate(latest.reset_index(drop=True))},"period_aggregates":{str(k):aggregate(g.reset_index(drop=True)) for k,g in v.groupby("period",sort=True)},"ablation":ab,"recent_shadow":shadow,"data_coverage":coverage,"checkpoint":{"enabled":ck is not None,"processed_validation_games":len(processed),"every_games":int(checkpoint_every)},"reproducibility":{"seed":int(seed),"simulations":int(simulations),"chronological_order":"file_month_then_game_date_then_game_id","validation_updates":"POST_TARGET_ONLY","shadow_updates":"FORBIDDEN","holdout_tuning":"FORBIDDEN","production_promotion":"FORBIDDEN"}}
