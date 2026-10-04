"""Core PIT-conscious game-state, transition, and simulation primitives."""
from __future__ import annotations

import hashlib, json, math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "game-script-v4"
PIT_STATUS = "UNVERIFIABLE_HISTORICAL_PBP_AVAILABILITY"
ALPHA, MIN_SUPPORT, MAX_SCORE_DIFF, MAX_STEPS, SCORE_GRID = 0.5, 24, 8, 420, 31


def _first(df: pd.DataFrame, names: Sequence[str], default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series([default] * len(df), index=df.index)


def _id(v: Any) -> str:
    if v is None or pd.isna(v): return ""
    s = str(v).strip(); return "" if s.lower() in {"", "nan", "none", "nat", "0.0"} else s


def _half(v: Any) -> str:
    s = str(v).strip().upper(); return "T" if s in {"T","TOP","表"} else "B" if s in {"B","BOT","BOTTOM","裏"} else ""


def _team_key(tid: Any, name: Any) -> str:
    t = _id(tid); return f"id:{t}" if t else f"name:{_id(name)}"


def _base(row: Mapping[str, Any]) -> int | None:
    bits=[]
    for names in (("on_1b","base1"),("on_2b","base2"),("on_3b","base3")):
        v=next((row.get(n) for n in names if n in row),None)
        if v is None or pd.isna(v): return None
        s=str(v).strip().lower(); empty=s in {"","nan","none","nat","0","0.0","false"}
        if not empty:
            try: empty=float(s)==0.0
            except (TypeError,ValueError): pass
        bits.append(not empty)
    return int(bits[0]) | int(bits[1])<<1 | int(bits[2])<<2


def _count(row: Mapping[str, Any]) -> tuple[int,int] | None:
    b,s=row.get("balls"),row.get("strikes")
    if b is None or s is None or pd.isna(b) or pd.isna(s): return None
    try: b,s=int(b),int(s)
    except (TypeError,ValueError): return None
    return (b,s) if b in range(4) and s in range(3) else None


def _band(inning:int)->int: return 1 if inning<=3 else 2 if inning<=6 else 3 if inning<=9 else 4

def _diff(v:float)->int: return max(-MAX_SCORE_DIFF,min(MAX_SCORE_DIFF,int(round(v))))


def _state(row: Mapping[str,Any], *, count:bool=True, diff:bool=True)->tuple:
    base=_base(row)
    if base is None: raise ValueError("base occupancy unavailable")
    key=[_band(int(row["inning"])),str(row["half"]),int(row["outs"]),int(base)]
    if diff: key.append(_diff(float(row["state_home_score"])-float(row["state_away_score"])))
    if count:
        c=_count(row)
        if c is None: raise ValueError("count unavailable")
        key.extend(c)
    return tuple(key)


def boundary(v: str|None, *, end_of_day: bool=False)->pd.Timestamp|None:
    if not v: return None
    ts=pd.Timestamp(v)
    ts=ts.tz_localize("Asia/Tokyo") if ts.tzinfo is None else ts.tz_convert("Asia/Tokyo")
    if end_of_day: ts=ts.normalize()+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
    return ts.tz_convert("UTC")


def canonicalize_pbp_frame(raw: pd.DataFrame)->pd.DataFrame:
    if raw.empty: return pd.DataFrame()
    o=pd.DataFrame(index=raw.index)
    o["game_id"]=_first(raw,["game_id","GameID"],"").map(_id); o["inning"]=pd.to_numeric(_first(raw,["inning","Inning"]),errors="coerce")
    o["half"]=_first(raw,["TB","half","Half"],"").map(_half); o["play_order"]=pd.to_numeric(_first(raw,["PlayInfo_SeqNo","play_id","ID","page"]),errors="coerce")
    o["pitch_number"]=pd.to_numeric(_first(raw,["pitch_number","atBatBallCount"]),errors="coerce"); o["page"]=_first(raw,["page","fiveDigitSerialNumber"],"").map(_id)
    o["game_date"]=pd.to_datetime(_first(raw,["game_date","GameDate"]),errors="coerce",utc=True); o["home"]=_first(raw,["home_team_name","H_NameS"],"").map(_id); o["away"]=_first(raw,["away_team_name","V_NameS"],"").map(_id)
    o["home_team_id"]=_first(raw,["home_team_id","H_ID"],"").map(_id); o["away_team_id"]=_first(raw,["away_team_id","V_ID"],"").map(_id)
    o["home_score"]=pd.to_numeric(_first(raw,["home_total_runs","H_R"]),errors="coerce"); o["away_score"]=pd.to_numeric(_first(raw,["away_total_runs","V_R"]),errors="coerce")
    o["added_runs"]=pd.to_numeric(_first(raw,["addedRuns","added_runs"]),errors="coerce"); o["outs"]=pd.to_numeric(_first(raw,["outs_when_up","out"]),errors="coerce")
    o["balls"]=pd.to_numeric(_first(raw,["balls","ball"]),errors="coerce"); o["strikes"]=pd.to_numeric(_first(raw,["strikes","strike"]),errors="coerce")
    o["batter_id"]=_first(raw,["batter","BatID","batter_id"],"").map(_id); o["pitcher_id"]=_first(raw,["pitcher","PitID","pitcher_id"],"").map(_id)
    o["pitch_id"]=pd.to_numeric(_first(raw,["pitch_id","ballKind"]),errors="coerce"); o["release_speed_kmh"]=pd.to_numeric(_first(raw,["release_speed_kmh","ballSpeed"]),errors="coerce")
    o["plate_x"]=pd.to_numeric(_first(raw,["plate_x","x"]),errors="coerce"); o["plate_y"]=pd.to_numeric(_first(raw,["plate_y","y"]),errors="coerce")
    o["result_id"]=_first(raw,["result_id","ResultID"],"").map(_id); o["end_time"]=_first(raw,["end_time","EndTime"],"").map(_id)
    for src,dst in (("on_1b","base1"),("on_2b","base2"),("on_3b","base3")): o[dst]=raw[src] if src in raw else _first(raw,[dst])
    m=o.game_id.ne("")&o.home.ne("")&o.away.ne("")&o.half.isin({"T","B"})&o.inning.notna()&o.game_date.notna()&o.home_score.notna()&o.away_score.notna()&o.added_runs.notna()&o.outs.notna()
    o=o.loc[m].copy()
    if o.empty: return o
    o["inning"]=o.inning.round().astype(int); o["outs"]=o.outs.round().astype(int); o=o[o.inning.between(1,12)&o.outs.between(0,2)]
    o["_h"]=o.half.map({"T":0,"B":1}); o=o.sort_values(["game_id","inning","_h","play_order","pitch_number"],kind="mergesort")
    o["_identity"]=np.where(o.page.ne(""),o.game_id+"|"+o.page,o.game_id+"|"+o.inning.astype(str)+"|"+o.half+"|"+o.play_order.astype(str))
    return _reconstruct_scores(o.drop_duplicates("_identity",keep="last").drop(columns=["_identity","_h"])).reset_index(drop=True)


def _reconstruct_scores(frame:pd.DataFrame)->pd.DataFrame:
    kept=[]
    for _,g in frame.groupby("game_id",sort=False):
        g=g.copy(); v=pd.to_numeric(g.added_runs,errors="coerce").to_numpy(float)
        if not np.isfinite(v).all() or np.any(v<0)|np.any(v>4): continue
        v=np.rint(v).astype(int); hs=aw=0; hp=[]; ap=[]
        for half,add in zip(g.half.astype(str),v):
            if half=="T": aw+=int(add)
            elif half=="B": hs+=int(add)
            else: break
            hp.append(hs); ap.append(aw)
        if len(hp)!=len(g) or not hp or hp[0]!=0 or ap[0]!=0: continue
        fh,fa=float(g.home_score.iloc[-1]),float(g.away_score.iloc[-1])
        if not np.isfinite(fh) or not np.isfinite(fa) or hp[-1]!=round(fh) or ap[-1]!=round(fa): continue
        g["state_home_score"],g["state_away_score"]=hp,ap
        final_result_id=_id(g.result_id.iloc[-1]) if "result_id" in g else ""
        final_end_time=_id(g.end_time.iloc[-1]) if "end_time" in g else ""
        g["complete_status"]="PASS" if (final_result_id or final_end_time) else "UNKNOWN"
        kept.append(g)
    return pd.concat(kept,ignore_index=True) if kept else frame.iloc[0:0].copy()


def _transition(a:Mapping[str,Any],b:Mapping[str,Any])->tuple|None:
    ai,bi=int(a["inning"]),int(b["inning"]); ah,bh=str(a["half"]),str(b["half"])
    if not ((ai==bi and ah==bh) or (ai==bi and ah=="T" and bh=="B") or (ah=="B" and bh=="T" and bi==ai+1)): return None
    if ai==bi and ah==bh and int(b["outs"])<int(a["outs"]): return None
    if None in (_base(a),_base(b),_count(a),_count(b)): return None
    dh=float(b["state_home_score"])-float(a["state_home_score"]); da=float(b["state_away_score"])-float(a["state_away_score"])
    if dh<0 or da<0 or dh>4 or da>4 or (dh>0 and da>0): return None
    cb=_count(b); return (bh,max(0,min(2,int(b["outs"]))),_base(b),cb[0],cb[1],int(round(dh+da)),"H" if dh>0 else "A" if da>0 else "N")


class TransitionKernel:
    def __init__(self,alpha=ALPHA,min_support=MIN_SUPPORT):
        self.alpha=float(alpha); self.min_support=int(min_support); self.views={k:defaultdict(Counter) for k in ("rich","no_count","no_diff","coarse","half","global")}; self.transitions=self.games=0; self.fingerprint=""
    def add_game(self,game:pd.DataFrame)->int:
        added=0; rows=game.to_dict("records")
        for a,b in zip(rows,rows[1:]):
            t=_transition(a,b)
            if t is None: continue
            try: keys=(_state(a,count=True,diff=True),_state(a,count=False,diff=True),_state(a,count=True,diff=False),_state(a,count=False,diff=False))
            except ValueError: continue
            for n,k in zip(("rich","no_count","no_diff","coarse"),keys): self.views[n][k][t]+=1
            self.views["half"][str(a["half"])][t]+=1; self.views["global"]["ALL"][t]+=1; added+=1
        if added: self.transitions+=added; self.games+=1; self.fingerprint=self._fp()
        return added
    def _fp(self):
        raw=[[str(k),sorted((str(o),int(n)) for o,n in v.items())] for k,v in sorted(self.views["rich"].items(),key=lambda x:str(x[0]))]; return hashlib.sha256(json.dumps(raw,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
    def snapshot(self): return {"transitions":self.transitions,"games":self.games,"fingerprint":self.fingerprint,"alpha":self.alpha,"min_support":self.min_support}
    def _select(self,key,profile):
        ib,h,o,b,d,ball,strike=key; c=[("rich",self.views["rich"].get(key)),("no_count",self.views["no_count"].get((ib,h,o,b,d))),("no_diff",self.views["no_diff"].get((ib,h,o,b,ball,strike))),("coarse",self.views["coarse"].get((ib,h,o,b))),("half",self.views["half"].get(h)),("global",self.views["global"].get("ALL"))]
        if profile=="coarse": c=c[3:]
        for lvl,x in c:
            if x and sum(x.values())>=self.min_support: return x,lvl
        for lvl,x in c:
            if x: return x,lvl
        raise LookupError("no transition support")
    def sample(self,key,rng,factors,profile="rich"):
        counts,level=self._select(key,profile); outs=list(counts); w=np.array([float(counts[o]) for o in outs],float)
        for i,o in enumerate(outs):
            s,r=o[-1],int(o[-2]); f=float(np.clip(factors.get(s,1.0),.55,1.50)); w[i]*=f**(.85*r) if r else 1.0
        p=(w+self.alpha)/(w.sum()+self.alpha*len(w)); i=int(rng.choice(len(outs),p=p)); return outs[i],level,float(-(p*np.log(np.clip(p,1e-12,1))).sum())


class TeamStrength:
    def __init__(self,base_run=4.0,shrink_games=24.0,half_life_days=90.0):
        self.base_run=float(base_run); self.shrink_games=float(shrink_games); self.half_life_days=float(half_life_days); self.gf=defaultdict(float); self.ga=defaultdict(float); self.games=defaultdict(float); self.total_runs=self.total_weight=0.0
    def update(self,h,a,hs,aw,date,latest):
        w=math.exp(-math.log(2)*max(0.0,(latest-date).total_seconds()/86400)/self.half_life_days); self.gf[h]+=hs*w; self.ga[h]+=aw*w; self.games[h]+=w; self.gf[a]+=aw*w; self.ga[a]+=hs*w; self.games[a]+=w; self.total_runs+=(hs+aw)*w; self.total_weight+=w
    def factors(self,h,a):
        league=max(.1,self.total_runs/max(1e-9,2*self.total_weight)) if self.total_weight else self.base_run
        def one(t):
            n=self.games[t]; return float(np.clip(((self.gf[t]+self.shrink_games*league)/(n+self.shrink_games))/league,.60,1.55)), float(np.clip(((self.ga[t]+self.shrink_games*league)/(n+self.shrink_games))/league,.60,1.55))
        ho,hd=one(h); ao,ad=one(a); return float(np.clip(math.sqrt(max(.01,ho/max(.01,ad))),.60,1.55)),float(np.clip(math.sqrt(max(.01,ao/max(.01,hd))),.60,1.55))


def simulate_game(kernel,home_factor,away_factor,simulations=800,seed=42,max_innings=12,profile="rich"):
    if simulations<=0: raise ValueError("simulations must be positive")
    rng=np.random.default_rng(int(seed)); scores=[]; levels=Counter(); ents=[]; extras=changes=comebacks=0; first=Counter(); aborted=0
    for _ in range(int(simulations)):
        inning,half,outs,bases,balls,strikes,hs,aw=1,"T",0,0,0,0,0,0
        trailing_h=trailing_a=changed=False; first_in=None; ok=False
        for _ in range(MAX_STEPS):
            if hs<aw: trailing_h=True
            if aw<hs: trailing_a=True
            if half=="B" and inning>=9 and hs>aw: ok=True; break
            if inning>=max_innings and half=="B" and outs>=2: ok=True; break
            try:
                out,lvl,e=kernel.sample(
                    (_band(inning),half,outs,bases,_diff(hs-aw),balls,strikes),
                    rng,{"H":home_factor,"A":away_factor},profile
                )
            except LookupError:
                break
            levels[lvl]+=1; ents.append(e)
            nh,no,nb,nbal,nstr,runs,scorer=out
            ni=inning+1 if half=="B" and nh=="T" else inning
            prev=hs-aw
            if scorer=="H": hs+=runs
            elif scorer=="A": aw+=runs
            if prev*(hs-aw)<0:
                changed=True; first_in=first_in if first_in is not None else ni
            outs,bases,balls,strikes=int(no),int(nb),int(nbal),int(nstr)
            inning,half=ni,nh
            if inning>max_innings: ok=True; break
        if not ok:
            aborted+=1; continue
        scores.append((hs,aw)); extras+=int(inning>9); changes+=int(changed)
        comebacks+=int((hs>aw and trailing_h) or (aw>hs and trailing_a))
        if changed and first_in is not None: first[first_in]+=1
    valid=len(scores)
    if valid<math.ceil(simulations*.995):
        raise RuntimeError(f"simulation coverage below fail-closed threshold: {valid}/{simulations}")
    m=np.zeros((SCORE_GRID,SCORE_GRID),float)
    for h,a in scores: m[min(SCORE_GRID-1,h),min(SCORE_GRID-1,a)]+=1
    m/=m.sum()
    hw=float(np.tril(m,-1).sum()); awp=float(np.triu(m,1).sum()); dr=float(np.trace(m))
    low=float(sum(m[i,j] for i in range(SCORE_GRID) for j in range(SCORE_GRID) if i+j<=6))
    flat=m.ravel(); top=np.argsort(-flat,kind="mergesort")[:4]
    ent=float(-(np.array([hw,dr,awp])*np.log(np.clip([hw,dr,awp],1e-12,1))).sum())
    predictability=float(1-ent/math.log(3))
    return {
        "schema_version":SCHEMA_VERSION,"simulations":simulations,"valid_simulations":valid,
        "aborted_simulations":aborted,"coverage":valid/simulations,"seed":int(seed),"profile":profile,
        "score_distribution":m.tolist(),
        "probabilities":{"home_win":hw,"draw":dr,"away_win":awp,"low_le_6":low,"high_ge_7":1-low},
        "top4_exact_score":[{"score":f"{i//SCORE_GRID}-{i%SCORE_GRID}","probability":float(flat[i])} for i in top],
        "extra_inning_probability":extras/valid,"lead_change_probability":changes/valid,
        "comeback_probability":comebacks/valid,
        "first_lead_change_inning_distribution":{str(k):v/valid for k,v in sorted(first.items())},
        "uncertainty":{
            "outcome_entropy":ent,"outcome_predictability":predictability,
            "mc_se_home_win":math.sqrt(max(1e-12,hw*(1-hw))/valid),
            "state_transition_entropy_mean":float(np.mean(ents)) if ents else float("nan"),
            "fallback_level_distribution":{k:v/max(1,sum(levels.values())) for k,v in sorted(levels.items())},
            "fallback_rate":sum(v for k,v in levels.items() if k!="rich")/max(1,sum(levels.values()))
        }
    }

def low_probability(m): return float(sum(m[i,j] for i in range(m.shape[0]) for j in range(m.shape[1]) if i+j<=6))


def summary(m,hs,aw):
    hw=float(np.tril(m,-1).sum()); awp=float(np.triu(m,1).sum()); dr=float(np.trace(m))
    actual="H" if hs>aw else "A" if aw>hs else "D"
    p={"H":hw,"A":awp,"D":dr}[actual]
    pred="H" if hw>=max(awp,dr) else "A" if awp>=dr else "D"
    flat=m.ravel(); top=np.argsort(-flat,kind="mergesort")[:4]; i=int(top[0]); th,ta=divmod(i,m.shape[1])
    low=float(low_probability(m)); low_actual=float(hs+aw<=6)
    eps=1e-12
    low_ll=-(math.log(max(eps,low)) if low_actual else math.log(max(eps,1-low)))
    low_brier=(low-low_actual)**2
    return {
        "accuracy":float(pred==actual),"logloss":float(-math.log(max(eps,p))),
        "brier":float((hw-(actual=="H"))**2+(dr-(actual=="D"))**2+(awp-(actual=="A"))**2),
        "score_mae":float((abs(th-hs)+abs(ta-aw))/2),
        "top1_exact":float(th==hs and ta==aw),
        "top4_exact":float(any(divmod(int(j),m.shape[1])==(hs,aw) for j in top)),
        "confidence":float(max(hw,dr,awp)),"correct":float(pred==actual),
        "low_le_6_probability":low,"low_le_6_accuracy":float((low>=0.5)==bool(low_actual)),
        "low_le_6_logloss":float(low_ll),"low_le_6_brier":float(low_brier),
    }


def aggregate(df):
    if df.empty: raise ValueError("cannot aggregate empty evaluation set")
    ece=0.0
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(df.confidence>=lo)&((df.confidence<hi) if hi<1 else (df.confidence<=hi))
        if mask.any(): ece+=float(mask.mean())*abs(float(df.loc[mask,"correct"].mean())-float(df.loc[mask,"confidence"].mean()))
    return {
        "rows":int(len(df)),
        "accuracy":float(df.accuracy.mean()),"logloss":float(df.logloss.mean()),
        "brier":float(df.brier.mean()),"ece":float(ece),"score_mae":float(df.score_mae.mean()),
        "top1_exact":float(df.top1_exact.mean()),"top4_exact":float(df.top4_exact.mean()),
        "low_le_6_accuracy":float(df.low_le_6_accuracy.mean()),
        "low_le_6_logloss":float(df.low_le_6_logloss.mean()),
        "low_le_6_brier":float(df.low_le_6_brier.mean()),
    }

