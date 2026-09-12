#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable multi-source NPB dataset builder.

Design goals:
- Never lose completed API work when GitHub Actions approaches its 30-minute limit.
- Resume from committed checkpoints on the next run.
- Fetch only missing/failed games on subsequent runs.
- Keep granular starter/game data separate from current-season official snapshots.
- Add data-quality diagnostics and refuse to claim a completed dataset until
  the requested coverage gates are satisfied.

The collector is deliberately incremental: adding older seasons or extending the
end date does not require re-downloading games already present in the checkpoint.
"""
from __future__ import annotations
import concurrent.futures as cf
import os, re, time, json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

SPAIA = "https://spaia.jp/baseball/npb/api"
NPB = "https://npb.jp"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
START_YEAR = int(os.getenv("NPB_START_YEAR", os.getenv("NPB_YEAR", "1990")))
_end_env = os.getenv("NPB_END_YEAR", "")
END_YEAR = int(_end_env) if _end_env.strip() else pd.Timestamp.utcnow().year
if START_YEAR > END_YEAR:
    START_YEAR, END_YEAR = END_YEAR, START_YEAR
WORKERS = int(os.getenv("NPB_DOWNLOAD_WORKERS", "6"))
BUDGET_SEC = float(os.getenv("NPB_COLLECTION_BUDGET_SEC", "1560"))
SAFETY_SEC = float(os.getenv("NPB_COLLECTION_SAFETY_SEC", "90"))
DEADLINE = time.monotonic() + max(60.0, BUDGET_SEC - SAFETY_SEC)
TIMEOUT = 30
DATA = Path("data")
CP = DATA / "checkpoints"
SEASON_DIR = DATA / "npb_games"
WEATHER_DIR = DATA / "weather"
ALL_OUT = DATA / "npb_multi_source_games_all.csv"
ALL_STATUS = CP / "npb_collection_status.json"
COVERAGE = DATA / "source_coverage.csv"

def season_paths(year: int):
    return {"out": SEASON_DIR / f"{year}_multi_source_pbp.csv", "cp": CP / f"{year}_game_enrichment.csv", "status": CP / f"{year}_status.json", "failures": CP / f"{year}_failures.csv", "weather": WEATHER_DIR / f"{year}_weather_cache.csv"}

ALIASES = {
"巨人":"読売ジャイアンツ","読売":"読売ジャイアンツ","読売ジャイアンツ":"読売ジャイアンツ",
"阪神":"阪神タイガース","阪神タイガース":"阪神タイガース","中日":"中日ドラゴンズ","中日ドラゴンズ":"中日ドラゴンズ",
"広島":"広島東洋カープ","広島東洋":"広島東洋カープ","広島東洋カープ":"広島東洋カープ",
"ヤクルト":"東京ヤクルトスワローズ","東京ヤクルト":"東京ヤクルトスワローズ","東京ヤクルトスワローズ":"東京ヤクルトスワローズ",
"DeNA":"横浜DeNAベイスターズ","ＤｅＮＡ":"横浜DeNAベイスターズ","横浜DeNA":"横浜DeNAベイスターズ","横浜DeNAベイスターズ":"横浜DeNAベイスターズ",
"ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンク":"福岡ソフトバンクホークス","福岡ソフトバンクホークス":"福岡ソフトバンクホークス",
"西武":"埼玉西武ライオンズ","埼玉西武":"埼玉西武ライオンズ","埼玉西武ライオンズ":"埼玉西武ライオンズ",
"日本ハム":"北海道日本ハムファイターズ","日ハム":"北海道日本ハムファイターズ","北海道日本ハム":"北海道日本ハムファイターズ","北海道日本ハムファイターズ":"北海道日本ハムファイターズ",
"ロッテ":"千葉ロッテマリーンズ","千葉ロッテ":"千葉ロッテマリーンズ","千葉ロッテマリーンズ":"千葉ロッテマリーンズ",
"楽天":"東北楽天ゴールデンイーグルス","東北楽天":"東北楽天ゴールデンイーグルス","東北楽天ゴールデンイーグルス":"東北楽天ゴールデンイーグルス",
"オリックス":"オリックス・バファローズ","オリックス・バファローズ":"オリックス・バファローズ"}
PARKS = {"神　宮":(35.6827,139.6841),"神宮":(35.6827,139.6841),"東京ドーム":(35.7056,139.7519),"横　浜":(35.4431,139.6400),"横浜":(35.4431,139.6400),"バンテリンドーム":(35.1859,136.9470),"マツダスタジアム":(34.3916,132.4848),"甲子園":(34.7214,135.3616),"エスコンＦ":(43.0151,141.4094),"ベルーナドーム":(35.7684,139.4745),"ZOZOマリン":(35.6456,140.0307),"楽天モバイル":(38.2560,140.9014),"京セラD大阪":(34.6694,135.4761),"ほっと神戸":(34.6795,135.0980),"みずほPayPay":(33.5950,130.3620)}
def near_deadline(): return time.monotonic() >= DEADLINE
def get_json(url, params=None, retries=4):
    last=None
    for i in range(retries):
        if near_deadline(): raise TimeoutError("collection deadline reached")
        try:
            r=requests.get(url,params=params,timeout=TIMEOUT,headers={"User-Agent":"Mozilla/5.0 baseball-backtest"}); r.raise_for_status(); return r.json()
        except Exception as e: last=e; time.sleep(min(1.5*(i+1),5))
    raise RuntimeError(f"request failed: {url}: {last}")
def official_name(x): return ALIASES.get(str(x or '').strip(),str(x or '').strip())
def _norm_key(k): return re.sub(r'[^a-z0-9]','',str(k).lower())
def _first(g,*names,default=None):
    if not isinstance(g,dict): return default
    for name in names:
        if name in g and g[name] not in (None,'','-'): return g[name]
    wanted={_norm_key(n) for n in names}
    for k,v in g.items():
        if _norm_key(k) in wanted and v not in (None,'','-'): return v
    return default
def parse_dt(g):
    direct=_first(g,'datetime','gameDateTime','game_datetime','dateTime','DateTime')
    if direct is not None:
        dt=pd.to_datetime(direct,errors='coerce')
        if pd.notna(dt): return dt
    d=_first(g,'DateJPN','date_jpn','GameDate','gameDate','game_date','MatchDate','matchDate','Date','date'); t=_first(g,'TimeJPN','time_jpn','GameTime','gameTime','gametime','game_time','MatchTime','matchTime','Time','time',default='1800')
    if d is None:return pd.NaT
    ds=re.sub(r'[^0-9]','',str(d).strip()); ts=(re.sub(r'[^0-9]','',str(t).strip()) or '1800')[:4]
    if len(ds)==8:
        try:return pd.to_datetime(ds+ts,format='%Y%m%d%H%M')
        except Exception:pass
    return pd.to_datetime(str(d),errors='coerce')
def game_kind(g): return str(_first(g,'GameKindName','game_kind_name','GameTypeName','game_type_name','gameTypeName','LeagueName','league_name',default='') or '')
def official_game(g):
    s=game_kind(g)
    if any(x in s for x in ('オープン戦','オールスター','ファーム','二軍','教育','練習試合')): return False
    return ('公式戦' in s) or ('交流戦' in s) or ('セ・リーグ' in s) or ('パ・リーグ' in s) or s==''
def num(v):
    try:
        s=str(v).replace(',','').strip()
        if s in ('','-','nan','None'):return np.nan
        return float(s)
    except Exception:return np.nan

def first_pitchers(game_id):
    raw=get_json(f'{SPAIA}/flash_atbat_history',{'gameId':game_id})
    if not isinstance(raw,list): return '',''
    arr=[]
    for x in raw:
        if not isinstance(x,dict):continue
        pid=x.get('pitId',x.get('pitcher',x.get('PitcherCD',''))); ser=str(x.get('fiveDigitSerialNumber',''))
        if pid in (None,'',0) or not ser:continue
        arr.append((ser,str(pid)))
    arr.sort(key=lambda z:z[0]); away=home=''; seen=set()
    for ser,pid in arr:
        if len(ser)>=3:
            half=ser[2]; inning=ser[:2]; key=(inning,half)
            if key in seen:continue
            seen.add(key)
            if half in ('T','t','1') and not away:away=pid
            elif half in ('B','b','2') and not home:home=pid
        if away and home:break
    # STRICT STARTER RULE: never substitute an arbitrary pitcher when the
    # inning-side evidence is missing. A blank starter is safer than using a
    # reliever/position player and contaminating the historical pregame model.
    # Never substitute an arbitrary pitcher.
    return away,home

def _walk_dicts(obj,seen=None):
    if seen is None:seen=set()
    if isinstance(obj,dict):
        oid=id(obj)
        if oid in seen:return
        seen.add(oid);yield obj
        for v in obj.values():yield from _walk_dicts(v,seen)
    elif isinstance(obj,list):
        for v in obj:yield from _walk_dicts(v,seen)
def _get_any(d,names):
    wanted={_norm_key(x) for x in names}
    for k,v in d.items():
        if _norm_key(k) in wanted and v not in (None,'','-'):return v
    return None
def _looks_like_pitcher_record(d):return bool({_norm_key(k) for k in d}&{'pitchercd','playercd','personinfoid','pitcherid','pitcheridno','playerid','playercdid'})
def _record_pid(d):return _get_any(d,['PitcherCD','PlayerCD','PersonInfoID','pitcherId','PitcherId','PlayerId','playerId','PitcherCDID'])
def _metric_from_record(d):
    ip=num(_get_any(d,['InningsPitched','Innings','IP','投球回','投球回数','投球イニング','PitchingInnings'])); ip3=num(_get_any(d,['InningsPitched3rd','IP3','inningsPitched3rd','投球回3分の1','投球回数3分の1']))
    if np.isfinite(ip3):ip=ip+ip3/3.0 if np.isfinite(ip) else ip3/3.0
    er=num(_get_any(d,['EarnedRun','EarnedRuns','ER','自責点','自責点数','EarnedRunCount'])); h=num(_get_any(d,['HitsAllowed','Hit','Hits','H','被安打','被安打数','HitAllowed'])); hr=num(_get_any(d,['HomeRun','HomeRunsAllowed','HR','被本塁打','被本塁打数','HomeRunAllowed'])); bb=num(_get_any(d,['Walk','Walks','BB','BaseOnBalls','与四球','四球','与四球数'])); so=num(_get_any(d,['Strikeout','Strikeouts','SO','奪三振','奪三振数','StrikeoutCount'])); pitches=num(_get_any(d,['PitchCount','Pitches','投球数','投球数合計']))
    era=num(_get_any(d,['ERA','EarnedRunAverage','防御率'])); whip=num(_get_any(d,['WHIP','Whip'])); k9=num(_get_any(d,['K9','StrikeoutPer9'])); bb9=num(_get_any(d,['BB9','WalkPer9'])); hr9=num(_get_any(d,['HR9','HomeRunPer9'])); fip=num(_get_any(d,['FIP','Fip']))
    if not np.isfinite(ip) or ip<=0:return None
    if not np.isfinite(era) and np.isfinite(er):era=9*er/ip
    if not np.isfinite(whip) and np.isfinite(h) and np.isfinite(bb):whip=(h+bb)/ip
    if not np.isfinite(k9) and np.isfinite(so):k9=9*so/ip
    if not np.isfinite(bb9) and np.isfinite(bb):bb9=9*bb/ip
    if not np.isfinite(hr9) and np.isfinite(hr):hr9=9*hr/ip
    if not np.isfinite(fip) and all(np.isfinite(x) for x in (hr,bb,so)):fip=(13*hr+3*bb-2*so)/ip+3.20
    return {'era':era,'whip':whip,'k9':k9,'bb9':bb9,'hr9':hr9,'fip':fip,'ip':ip,'er':er,'h':h,'hr':hr,'bb':bb,'so':so,'pitches':pitches}

def _side_for_record(d,home,away):
    vals=[str(_get_any(d,[k])) for k in ('TeamName','teamName','TeamNameS','team_name','Team','team','ClubName','clubName','球団名','チーム名') if _get_any(d,[k]) not in (None,'','-')]
    text=' '.join(vals); h=official_name(home); a=official_name(away)
    if h and (h in text or text in h):return 'home'
    if a and (a in text or text in a):return 'away'
    return None
def _bat_metric_from_record(d):
    ab=num(_get_any(d,['AtBat','AB','打数','打席数'])); pa=num(_get_any(d,['PlateAppearance','PA','打席','打席数'])); h=num(_get_any(d,['Hit','Hits','H','安打','安打数'])); hr=num(_get_any(d,['HomeRun','HomeRuns','HR','本塁打','本塁打数'])); bb=num(_get_any(d,['Walk','Walks','BB','BaseOnBalls','四球','与四球'])); so=num(_get_any(d,['Strikeout','Strikeouts','SO','三振','三振数'])); d2=num(_get_any(d,['Double','Doubles','2B','二塁打','二塁打数'])); d3=num(_get_any(d,['Triple','Triples','3B','三塁打','三塁打数'])); sb=num(_get_any(d,['StolenBase','StolenBases','SB','盗塁'])); cs=num(_get_any(d,['CaughtStealing','CS','盗塁死']))
    if not np.isfinite(ab):ab=0.0
    if not np.isfinite(pa):pa=ab+max(0.0,bb)
    if not any(np.isfinite(x) and x>0 for x in (ab,pa,h,hr,bb,so,d2,d3,sb,cs)):return None
    return {'pa':pa,'ab':ab,'h':0.0 if not np.isfinite(h) else h,'hr':0.0 if not np.isfinite(hr) else hr,'bb':0.0 if not np.isfinite(bb) else bb,'so':0.0 if not np.isfinite(so) else so,'2b':0.0 if not np.isfinite(d2) else d2,'3b':0.0 if not np.isfinite(d3) else d3,'sb':0.0 if not np.isfinite(sb) else sb,'cs':0.0 if not np.isfinite(cs) else cs}
def _extract_batter_records(raw):return [(d,m) for d in _walk_dicts(raw) if (m:=_bat_metric_from_record(d)) is not None]
def _fetch_game_batter_records(game_id,match_date):
    attempts=[{'gameId':game_id,'matchDate':match_date},{'GameID':game_id,'matchDate':match_date},{'GameID':game_id,'MatchDate':match_date},{'gameId':game_id,'MatchDate':match_date},{'game_id':game_id,'match_date':match_date}]; last=None
    for params in attempts:
        try:
            raw=get_json(f'{SPAIA}/both_batter_stats',params); recs=_extract_batter_records(raw)
            if recs:return recs,None
            last='empty_response'
        except Exception as e:last=repr(e)
    return [],last or 'no_batter_records'
def game_batting_metrics(game_id,match_date,home,away):
    recs,err=_fetch_game_batter_records(game_id,match_date); agg={'home':{'pa':0,'ab':0,'h':0,'hr':0,'bb':0,'so':0,'2b':0,'3b':0,'sb':0,'cs':0},'away':{'pa':0,'ab':0,'h':0,'hr':0,'bb':0,'so':0,'2b':0,'3b':0,'sb':0,'cs':0}}; identities=set(); seen=0
    for d,m in recs:
        side=_side_for_record(d,home,away); pid=_get_any(d,['BatterCD','BatterId','PlayerCD','PlayerId','PersonInfoID','playerId','playerCD'])
        if not side or pid is None:continue
        ident=(side,str(pid))
        if ident in identities:continue
        identities.add(ident)
        for k,v in m.items():agg[side][k]+=float(v)
        seen+=1
    return agg if seen else None

def _extract_pitcher_records(raw):return [d for d in _walk_dicts(raw) if _looks_like_pitcher_record(d)]
def _fetch_game_pitcher_records(game_id,match_date):
    attempts=[{'gameId':game_id,'matchDate':match_date},{'GameID':game_id,'matchDate':match_date},{'GameID':game_id,'MatchDate':match_date},{'gameId':game_id,'MatchDate':match_date},{'game_id':game_id,'match_date':match_date}]; last=None
    for params in attempts:
        try:
            raw=get_json(f'{SPAIA}/both_pitcher_game_stats',params); recs=_extract_pitcher_records(raw)
            if recs:return recs,None
            last='empty_response'
        except Exception as e:last=repr(e)
    return [],last or 'no_pitcher_records'
def pitcher_line(game_id,match_date,starter_id):
    recs,err=_fetch_game_pitcher_records(game_id,match_date); sid=str(starter_id)
    for d in recs:
        pid=_record_pid(d)
        if pid is not None and str(pid)==sid:
            m=_metric_from_record(d)
            if m:return m
    try:
        sn=int(float(sid))
        for d in recs:
            pid=_record_pid(d)
            try:
                if int(float(str(pid)))==sn:
                    m=_metric_from_record(d)
                    if m:return m
            except Exception:pass
    except Exception:pass
    return None

PLAYER_KEYS=['PlayerCD','PlayerId','playerId','playerCD','BatterCD','PitcherCD','PersonInfoID','personId','player_id']; NAME_KEYS=['PlayerName','playerName','BatterName','PitcherName','Name','name','選手名']; ORDER_KEYS=['BattingOrder','battingOrder','Order','order','打順']; POS_KEYS=['Position','position','守備位置','守備']; HAND_KEYS=['Bats','bats','BatSide','batSide','ThrowingHand','throwingHand','投打','利き腕']
def _fetch_variants(endpoint,game_id,match_date):
    attempts=[{'gameId':game_id,'matchDate':match_date},{'GameID':game_id,'matchDate':match_date},{'GameID':game_id,'MatchDate':match_date},{'gameId':game_id,'MatchDate':match_date},{'game_id':game_id,'match_date':match_date},{'gameId':game_id},{'GameID':game_id}]
    for params in attempts:
        try:
            raw=get_json(f'{SPAIA}/{endpoint}',params)
            if raw not in (None,{},[]):return raw
        except Exception:continue
    return None
def _player_id(d):
    v=_get_any(d,PLAYER_KEYS);return str(v).strip() if v not in (None,'','-') else ''
def _player_name(d):
    v=_get_any(d,NAME_KEYS);return str(v).strip() if v not in (None,'','-') else ''
def extract_starting_lineup(game_id,match_date,home,away):
    raw=_fetch_variants('starting_members_for_flash',game_id,match_date);out={'home':[],'away':[]}
    if raw is None:return out
    seen=set()
    for d in _walk_dicts(raw):
        pid=_player_id(d);side=_side_for_record(d,home,away)
        if not pid or not side or (side,pid) in seen:continue
        order=_get_any(d,ORDER_KEYS);pos=_get_any(d,POS_KEYS);name=_player_name(d)
        if order is None and pos is None and not name:continue
        out[side].append({'player_id':pid,'player_name':name,'batting_order':num(order),'position':str(pos or ''),'hand':str(_get_any(d,HAND_KEYS) or '')});seen.add((side,pid))
    for side in out:out[side]=sorted(out[side],key=lambda z:(999 if not np.isfinite(num(z.get('batting_order'))) else num(z.get('batting_order')),z['player_id']))[:12]
    return out

def _event_text(d):return ' '.join(str(_get_any(d,[k])) for k in ['Result','result','Event','event','AtBatResult','atBatResult','PlayResult','playResult','Description','description','Text','text','ResultName','resultName','打席結果','結果'] if _get_any(d,[k]) not in (None,'','-'))
def _event_batter_id(d):return str(_get_any(d,['BatterCD','BatterId','batterId','BatterID','HitterID','HitterId','PlayerCD','PlayerId','playerId']) or '').strip()
def _blank_player_stats():return {'pa':0,'ab':0,'h':0,'2b':0,'3b':0,'hr':0,'bb':0,'hbp':0,'so':0,'rbi':0,'r':0,'sb':0,'cs':0,'sh':0,'sf':0,'gidp':0,'errors':0,'bunt':0,'groundout':0,'flyout':0,'lineout':0,'power_events':0,'contact_events':0,'event_count':0}
def _apply_event(st,text):
    t=str(text).lower();terminal=any(x in t for x in ('本塁打','ホームラン','二塁打','三塁打','安打','四球','死球','三振','犠打','犠飛','併殺','失策','ゴロ','フライ','ライナー','バント','盗塁','盗塁死','単打'))
    if not terminal:return False
    st['event_count']+=1
    if any(x in t for x in ('本塁打','ホームラン')):st['hr']+=1;st['h']+=1;st['ab']+=1;st['pa']+=1;st['power_events']+=1;st['contact_events']+=1
    elif '二塁打' in t:st['2b']+=1;st['h']+=1;st['ab']+=1;st['pa']+=1;st['power_events']+=1;st['contact_events']+=1
    elif '三塁打' in t:st['3b']+=1;st['h']+=1;st['ab']+=1;st['pa']+=1;st['power_events']+=1;st['contact_events']+=1
    elif '四球' in t or '死球' in t:st['bb']+=int('四球' in t);st['hbp']+=int('死球' in t);st['pa']+=1
    elif '三振' in t:st['so']+=1;st['ab']+=1;st['pa']+=1
    elif '犠打' in t or 'バント' in t:st['sh']+=1;st['bunt']+=1;st['pa']+=1
    elif '犠飛' in t:st['sf']+=1;st['pa']+=1
    elif '併殺' in t:st['gidp']+=1;st['ab']+=1;st['pa']+=1
    elif '失策' in t:st['errors']+=1;st['ab']+=1;st['pa']+=1
    elif '安打' in t or '単打' in t:st['h']+=1;st['ab']+=1;st['pa']+=1;st['contact_events']+=1
    elif 'ゴロ' in t:st['groundout']+=1;st['ab']+=1;st['pa']+=1
    elif 'フライ' in t:st['flyout']+=1;st['ab']+=1;st['pa']+=1
    elif 'ライナー' in t:st['lineout']+=1;st['ab']+=1;st['pa']+=1
    else:return False
    if '打点' in t or 'rbi' in t:st['rbi']+=1
    if '盗塁死' in t:st['cs']+=1
    elif '盗塁' in t:st['sb']+=1
    return True

def player_pitcher_game_metrics(game_id,match_date,home,away):
    recs,_=_fetch_game_pitcher_records(game_id,match_date);out=[]
    for d in recs:
        pid=_record_pid(d);side=_side_for_record(d,home,away);m=_metric_from_record(d)
        if pid in (None,'') or not side or not m:continue
        m.update({'game_id':str(game_id),'side':side,'player_id':str(pid),'player_name':str(_get_any(d,NAME_KEYS) or ''),'role':'pitcher','batting_order':np.nan,'position':'P','hand':str(_get_any(d,HAND_KEYS) or '')})
        ip=max(float(m.get('ip',0.0) or 0.0),1/3);m['k_rate']=float(m.get('so',0.0) or 0.0)/max(float(m.get('ip',0.0) or 0.0)*3.0,1.0);m['bb_rate']=float(m.get('bb',0.0) or 0.0)/max(float(m.get('ip',0.0) or 0.0)*3.0,1.0);m['k_minus_bb']=m['k_rate']-m['bb_rate'];m['hr_rate']=float(m.get('hr',0.0) or 0.0)/max(float(m.get('ip',0.0) or 0.0)*3.0,1.0);out.append(m)
    return out

def player_game_metrics(game_id,match_date,home,away,lineup):
    raw=_fetch_variants('game_text_pbp',game_id,match_date);agg={}
    if raw is not None:
        for d in _walk_dicts(raw):
            pid=_event_batter_id(d);txt=_event_text(d);side=_side_for_record(d,home,away)
            if not pid or not txt or not side:continue
            agg.setdefault((side,pid),_blank_player_stats());_apply_event(agg[(side,pid)],txt)
    if len(agg)<2:
        recs,_=_fetch_game_batter_records(game_id,match_date)
        for d,m in recs:
            pid=_player_id(d);side=_side_for_record(d,home,away)
            if not pid or not side:continue
            st=agg.setdefault((side,pid),_blank_player_stats())
            for src in ('pa','ab','h','2b','3b','hr','bb','so','sb','cs'):
                v=num(m.get(src));
                if np.isfinite(v):st[src]=max(st[src],int(v))
    out=[]
    for (side,pid),st in agg.items():
        if st['event_count']==0 and sum(st[k] for k in ('pa','ab','h','hr','bb','so','sb','cs'))==0:continue
        meta=next((x for x in lineup.get(side,[]) if x.get('player_id')==pid),{});st.update({'game_id':str(game_id),'side':side,'player_id':pid,'player_name':meta.get('player_name',''),'batting_order':meta.get('batting_order',np.nan),'position':meta.get('position',''),'hand':meta.get('hand','')});st['pa']=max(st['pa'],st['ab'],st['bb'],st['hbp'],st['sf'],st['sh']);st['avg']=st['h']/max(st['ab'],1);st['obp_proxy']=(st['h']+st['bb']+st['hbp'])/max(st['ab']+st['bb']+st['hbp']+st['sf'],1);st['slg_proxy']=(st['h']+st['2b']+2*st['3b']+3*st['hr'])/max(st['ab'],1);st['iso_proxy']=max(0,st['slg_proxy']-st['avg']);st['bb_rate']=st['bb']/max(st['pa'],1);st['so_rate']=st['so']/max(st['pa'],1);st['sb_rate']=st['sb']/max(st['pa'],1);st['bunt_rate']=st['bunt']/max(st['pa'],1);st['power_rate']=st['power_events']/max(st['pa'],1);st['contact_rate']=st['contact_events']/max(st['pa'],1);out.append(st)
    return out

def fetch_games(year):
    raw=get_json(f'{SPAIA}/schedules',{'Year':year});start=pd.Timestamp(f'{year}-03-01').normalize();end=pd.Timestamp(f'{year}-12-31').normalize()+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1);rows=[]
    for g in raw if isinstance(raw,list) else []:
        dt=parse_dt(g)
        if pd.isna(dt) or dt>end or dt<start or not official_game(g):continue
        gid=str(_first(g,'GameID','game_id','gameId','gameID','gamePk','GamePk',default='') or '');hs=num(_first(g,'HScore','home_score','homeScore','HomeScore','HomeTeamScore'));aas=num(_first(g,'VScore','away_score','awayScore','AwayScore','VisitorScore','VTeamScore'))
        if not gid or not np.isfinite(hs) or not np.isfinite(aas):continue
        park=str(_first(g,'StadiumName','stadiumName','BallparkName','ballpark','venue','VenueName',default='') or '');home=official_name(_first(g,'HTeamNameS','home_team_short_name','homeTeamShort','homeTeam','home_team','HomeTeamName',default=''));away=official_name(_first(g,'VTeamNameS','away_team_short_name','visitorTeamShort','visitorTeam','away_team','AwayTeamName',default=''))
        if not home or not away:continue
        rows.append({'game_id':gid,'datetime':dt,'home':home,'away':away,'home_score':hs,'away_score':aas,'game_type':game_kind(g) or '公式戦','venue':park})
    return pd.DataFrame(rows,columns=['game_id','datetime','home','away','home_score','away_score','game_type','venue']).drop_duplicates('game_id').sort_values(['datetime','game_id']).reset_index(drop=True) if rows else pd.DataFrame(columns=['game_id','datetime','home','away','home_score','away_score','game_type','venue'])

def load_checkpoint(year):
    p=season_paths(year)
    if p['cp'].exists():
        try:return pd.read_csv(p['cp'],low_memory=False)
        except Exception:pass
    return pd.DataFrame()
def atomic_csv(df,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');df.to_csv(tmp,index=False);tmp.replace(path)
def atomic_json(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(path)

def enrich_one(r):
    away,home=first_pitchers(r.game_id);date=pd.Timestamp(r.datetime).strftime('%Y%m%d');hm=pitcher_line(r.game_id,date,home) if home else None;am=pitcher_line(r.game_id,date,away) if away else None;bat=game_batting_metrics(r.game_id,date,r.home,r.away);lineup=extract_starting_lineup(r.game_id,date,r.home,r.away);players=player_game_metrics(r.game_id,date,r.home,r.away,lineup)+player_pitcher_game_metrics(r.game_id,date,r.home,r.away);z=r._asdict();z.update({'home_starter':home,'away_starter':away,'home_lineup_json':json.dumps(lineup.get('home',[]),ensure_ascii=False,separators=(',',':')),'away_lineup_json':json.dumps(lineup.get('away',[]),ensure_ascii=False,separators=(',',':')),'player_rows_count':len(players)})
    for side,m in [('home',hm),('away',am)]:
        for k in ('era','whip','k9','bb9','hr9','fip','ip','er','h','hr','bb','so','pitches','k_rate','bb_rate'):z[f'{side}_starter_{k}']=m.get(k) if m else np.nan
    if bat:
        for side in ('home','away'):
            for k in ('pa','ab','h','hr','bb','so','2b','3b','sb','cs'):z[f'{side}_bat_{k}']=bat[side][k]
    z['home_starter_line_ok']=bool(hm);z['away_starter_line_ok']=bool(am);z['_player_rows']=players;return z

def save_status(year,games,cp,done,failures,complete=False):
    p=season_paths(year);both_starters=int(((cp.get('home_starter','').fillna('').astype(str)!='')&(cp.get('away_starter','').fillna('').astype(str)!='')).sum()) if not cp.empty else 0;both_lines=int((cp.get('home_starter_line_ok',pd.Series(dtype=bool)).fillna(False)&cp.get('away_starter_line_ok',pd.Series(dtype=bool)).fillna(False)).sum()) if not cp.empty else 0;atomic_json({'year':year,'schedule_games':len(games),'checkpoint_games':len(cp),'done':int(done),'failures':len(failures),'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':round(100*both_lines/max(1,both_starters),2),'complete':bool(complete),'updated_at':pd.Timestamp.utcnow().isoformat()},p['status'])
def official_audit(year):
    out=[];DATA.mkdir(exist_ok=True);urls=[f'{NPB}/bis/{year}/stats/std_c.html',f'{NPB}/bis/{year}/stats/std_p.html',f'{NPB}/bis/{year}/stats/tmb_c.html',f'{NPB}/bis/{year}/stats/tmb_p.html',f'{NPB}/bis/{year}/stats/tmp_c.html',f'{NPB}/bis/{year}/stats/tmp_p.html']
    for url in urls:
        if near_deadline():break
        try:r=requests.get(url,timeout=TIMEOUT,headers={'User-Agent':'Mozilla/5.0 baseball-backtest'});r.raise_for_status();(DATA/f'official_{year}_'+url.rsplit('/',1)[-1]).write_text(r.text,encoding='utf-8',errors='ignore');out.append({'url':url,'status':'ok','bytes':len(r.content)})
        except Exception as e:out.append({'url':url,'status':'skip','error':str(e)[:300]})
    atomic_csv(pd.DataFrame(out),DATA/'source_official_npb_audit.csv')
def add_weather(d,year):
    if d.empty or near_deadline():return d
    p=season_paths(year);cache=pd.read_csv(p['weather']) if p['weather'].exists() else pd.DataFrame()
    for v in sorted(set(d.venue.astype(str))):
        if near_deadline():break
        if v not in PARKS:continue
        lat,lon=PARKS[v];existing=cache[cache.get('venue',pd.Series(dtype=str)).astype(str)==v] if not cache.empty and 'venue' in cache else pd.DataFrame()
        try:
            raw=get_json(OPEN_METEO,{'latitude':lat,'longitude':lon,'start_date':f'{year}-03-01','end_date':f'{year}-12-31','hourly':'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m','timezone':'Asia/Tokyo'});h=raw.get('hourly',{});w=pd.DataFrame({'datetime':pd.to_datetime(h.get('time',[])),'weather_temp_c':h.get('temperature_2m',[]),'weather_humidity_pct':h.get('relative_humidity_2m',[]),'weather_precip_mm':h.get('precipitation',[]),'weather_wind_kmh':h.get('wind_speed_10m',[])});w['datetime']=w.datetime.dt.floor('h');w['venue']=v;existing=existing[~existing.datetime.isin(w.datetime)] if 'datetime' in existing else existing;cache=pd.concat([cache,existing,w],ignore_index=True).drop_duplicates(['venue','datetime'],keep='last')
        except Exception as e:print('[WEATHER SKIP]',v,str(e)[:120])
    if not cache.empty:atomic_csv(cache,p['weather'])
    if cache.empty:return d
    c=cache.copy();c['datetime']=pd.to_datetime(c.datetime).dt.floor('h');sub=d[['game_id','venue','datetime']].copy();sub.datetime=pd.to_datetime(sub.datetime).dt.floor('h');sub=sub.merge(c,on=['venue','datetime'],how='left');return d.drop(columns=[x for x in ('weather_temp_c','weather_humidity_pct','weather_precip_mm','weather_wind_kmh') if x in d],errors='ignore').merge(sub[['game_id','weather_temp_c','weather_humidity_pct','weather_precip_mm','weather_wind_kmh']],on='game_id',how='left')

def main():
    DATA.mkdir(exist_ok=True);CP.mkdir(parents=True,exist_ok=True);SEASON_DIR.mkdir(parents=True,exist_ok=True);WEATHER_DIR.mkdir(parents=True,exist_ok=True);coverage_rows=[];all_parts=[];season_summaries=[]
    for year in range(START_YEAR,END_YEAR+1):
        if near_deadline():print(f'[STOP] deadline before season {year}; next run resumes automatically');break
        paths=season_paths(year);games=fetch_games(year);print(f'[SP AIA] year={year} schedule games={len(games)}');cp=load_checkpoint(year);failures=[];player_path=paths['cp'].with_name(f'{year}_player_game_features.csv');player_rows=[]
        if player_path.exists():
            try:player_rows=pd.read_csv(player_path,low_memory=False).to_dict('records')
            except Exception:player_rows=[]
        if games.empty:
            atomic_csv(pd.DataFrame(columns=['game_id','datetime','home','away','home_score','away_score','game_type','venue']),paths['cp']);atomic_json({'year':year,'schedule_games':0,'checkpoint_games':0,'done':0,'failures':0,'both_starters':0,'both_starter_lines':0,'coverage_pct':0.0,'complete':True,'unavailable':True,'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status']);coverage_rows.append({'year':year,'games':0,'both_starters':0,'home_starter_lines':0,'away_starter_lines':0,'both_starter_lines':0,'starter_line_coverage_pct':0.0,'checkpoint_complete':True,'remaining_games':0});continue
        if not cp.empty and 'game_id' in cp:cp=cp.drop_duplicates('game_id',keep='last')
        player_done_ids=set(pd.DataFrame(player_rows).get('game_id',pd.Series(dtype=str)).astype(str)) if player_rows else set();missing_ids=set(games.game_id.astype(str))-player_done_ids;missing=games[games.game_id.astype(str).isin(missing_ids)].copy();print(f'[CHECKPOINT] year={year} existing={len(cp)} player_enriched={len(player_done_ids)} missing_for_full_enrichment={len(missing)}')
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures={ex.submit(enrich_one,r):r.game_id for r in missing.itertuples(index=False)};completed=0
            for f in cf.as_completed(futures):
                gid=futures[f]
                if near_deadline():
                    for ff in futures:
                        if not ff.done():ff.cancel()
                    break
                try:
                    row=f.result();prs=row.pop('_player_rows',[]) if isinstance(row,dict) else []
                    if prs:player_rows.extend(prs);atomic_csv(pd.DataFrame(player_rows).drop_duplicates(['game_id','player_id','side','role'],keep='last'),player_path)
                    cp=pd.concat([cp,pd.DataFrame([row])],ignore_index=True).drop_duplicates('game_id',keep='last')
                except Exception as e:failures.append({'game_id':gid,'year':year,'error':repr(e),'ts':pd.Timestamp.utcnow().isoformat()})
                completed+=1;atomic_csv(cp,paths['cp']);print(f'[CHECKPOINT] year={year} saved {len(cp)}/{len(games)}')
        atomic_csv(cp,paths['cp'])
        if player_rows:atomic_csv(pd.DataFrame(player_rows).drop_duplicates(['game_id','player_id','side','role'],keep='last'),player_path)
        if failures:
            old=pd.read_csv(paths['failures']) if paths['failures'].exists() else pd.DataFrame();atomic_csv(pd.concat([old,pd.DataFrame(failures)],ignore_index=True).drop_duplicates(['game_id','error'],keep='last'),paths['failures'])
        if len(cp)==0:continue
        cp=cp.sort_values(['datetime','game_id']).reset_index(drop=True);complete=bool(set(games.game_id.astype(str)).issubset(set(cp.game_id.astype(str))));d=add_weather(cp,year);d['league']='NPB';d['date']=pd.to_datetime(d.datetime);d['inning']=1;d['half']='';d['event']='';d['addedRuns']=0;d['pitcher']='';atomic_csv(d,paths['out']);both_starters=int(((d.home_starter.fillna('').astype(str)!='')&(d.away_starter.fillna('').astype(str)!='')).sum());both_lines=int((d.home_starter_line_ok.fillna(False)&d.away_starter_line_ok.fillna(False)).sum());home_lines=int(d.home_starter_line_ok.fillna(False).sum());away_lines=int(d.away_starter_line_ok.fillna(False).sum());coverage=round(100*both_lines/max(1,both_starters),2);coverage_rows.append({'year':year,'games':len(d),'both_starters':both_starters,'home_starter_lines':home_lines,'away_starter_lines':away_lines,'both_starter_lines':both_lines,'starter_line_coverage_pct':coverage,'checkpoint_complete':complete,'remaining_games':max(0,len(games)-len(cp))});save_status(year,games,cp,completed,failures,complete=complete);season_summaries.append({'year':year,'games':len(d),'complete':complete,'remaining':max(0,len(games)-len(cp))});print(f'[OUTPUT] year={year} {paths["out"]} games={len(d)} starters={both_starters} both_starter_lines={both_lines}');print(f'[COVERAGE] year={year} home_lines={home_lines} away_lines={away_lines} both={both_lines}/{both_starters} ({coverage:.1f}%) remaining={max(0,len(games)-len(cp))}');
        if not near_deadline():official_audit(year)
        all_parts.append(d)
    player_parts=[]
    for year in range(START_YEAR,END_YEAR+1):
        pf=season_paths(year)['cp'].with_name(f'{year}_player_game_features.csv')
        if pf.exists():
            try:player_parts.append(pd.read_csv(pf,low_memory=False))
            except Exception as e:print('[PLAYER AGGREGATE SKIP]',pf,e)
    if player_parts:atomic_csv(pd.concat(player_parts,ignore_index=True,sort=False).drop_duplicates(['game_id','player_id','side','role'],keep='last'),DATA/'npb_player_game_features_all.csv')
    disk_parts=[]
    for year in range(START_YEAR,END_YEAR+1):
        f=season_paths(year)['out']
        if f.exists():
            try:disk_parts.append(pd.read_csv(f,low_memory=False))
            except Exception as e:print('[AGGREGATE SKIP]',f,e)
    if disk_parts:atomic_csv(pd.concat(disk_parts,ignore_index=True,sort=False).drop_duplicates('game_id',keep='last').sort_values(['date','game_id']).reset_index(drop=True),ALL_OUT)
    if coverage_rows:
        old=pd.read_csv(COVERAGE) if COVERAGE.exists() else pd.DataFrame();new=pd.DataFrame(coverage_rows)
        if not old.empty and 'year' in old:new=pd.concat([old[~old.year.isin(new.year)],new],ignore_index=True)
        atomic_csv(new.sort_values('year'),COVERAGE)
    statuses=[]
    for year in range(START_YEAR,END_YEAR+1):
        st=season_paths(year)['status']
        if st.exists():
            try:statuses.append(json.loads(st.read_text(encoding='utf-8')))
            except Exception:pass
    complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR<=int(x.get('year',-1))<=END_YEAR) and len(statuses)>=(END_YEAR-START_YEAR+1);atomic_json({'start_year':START_YEAR,'end_year':END_YEAR,'seasons_requested':END_YEAR-START_YEAR+1,'seasons_status':statuses,'aggregate_games':int(len(pd.read_csv(ALL_OUT,low_memory=False))) if ALL_OUT.exists() else 0,'complete':complete_all,'updated_at':pd.Timestamp.utcnow().isoformat()},ALL_STATUS)
    if not complete_all:print('[PARTIAL] massive historical collection is resumable; next run continues from season/game checkpoints');return 0
    cov=pd.read_csv(COVERAGE) if COVERAGE.exists() else pd.DataFrame()
    if not cov.empty:
        eligible=cov[cov.year.between(START_YEAR,END_YEAR)];bad=eligible[eligible.starter_line_coverage_pct<70]
        if not bad.empty:raise RuntimeError('completed multi-season schedule but starter-line coverage below 70% in: '+','.join(map(str,bad.year.tolist())))
    print(f'[COMPLETE] all requested seasons collected: {START_YEAR}-{END_YEAR}');return 0
if __name__=='__main__':raise SystemExit(main())
