#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPB production quality patch: never declare an empty/partial season complete
and repair stale checkpoints lacking prediction-critical starter lines."""
from __future__ import annotations
from pathlib import Path
import re
P=Path("npb_multi_source.py"); s=P.read_text(encoding="utf-8"); MARK="# NPB_QUALITY_HARDENING_V1"
if MARK in s:
    print("[NPB QUALITY PATCH] V1 already applied"); raise SystemExit(0)

def _flag(v):
    if isinstance(v,bool): return v
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

pat=re.compile(r"        if games\.empty:\n.*?        if not cp\.empty and 'game_id' in cp:",re.S)
new='''        if games.empty:\n            if not cp.empty and 'game_id' in cp:\n                prev_games=len(cp)\n                both_starters=int(((cp.get('home_starter','').fillna('').astype(str).str.strip()!='')&(cp.get('away_starter','').fillna('').astype(str).str.strip()!='')).sum()) if 'home_starter' in cp and 'away_starter' in cp else 0\n                both_lines=int((cp.get('home_starter_line_ok',pd.Series(dtype=bool)).map(_flag)&cp.get('away_starter_line_ok',pd.Series(dtype=bool)).map(_flag)).sum()) if 'home_starter_line_ok' in cp and 'away_starter_line_ok' in cp else 0\n                coverage=round(100*both_lines/max(1,both_starters),2)\n                atomic_json({'year':year,'schedule_games':prev_games,'checkpoint_games':prev_games,'done':0,'failures':0,'both_starters':both_starters,'both_starter_lines':both_lines,'coverage_pct':coverage,'complete':False,'unavailable':False,'schedule_complete':False,'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])\n                print(f'[SP AIA] year={year} EMPTY SCHEDULE -> preserved checkpoint rows={prev_games}; NOT complete')\n                continue\n            atomic_json({'year':year,'schedule_games':0,'checkpoint_games':0,'done':0,'failures':0,'both_starters':0,'both_starter_lines':0,'coverage_pct':0.0,'complete':False,'unavailable':True,'schedule_complete':False,'updated_at':pd.Timestamp.utcnow().isoformat()},paths['status'])\n            print(f'[SP AIA] year={year} no schedule rows; marked unavailable/INCOMPLETE')\n            continue\n        if not cp.empty and 'game_id' in cp:'''
if not pat.search(s): raise SystemExit("empty schedule block not found")
s=pat.sub(new,s,count=1)
pat=re.compile(r"        player_done_ids=set\(pd\.DataFrame\(player_rows\).*?print\(f'\[CHECKPOINT\].*?\)\n",re.S)
new='''        player_done_ids=set(pd.DataFrame(player_rows).get('game_id',pd.Series(dtype=str)).astype(str)) if player_rows else set()\n        existing_ids=set(cp.game_id.astype(str)) if not cp.empty and 'game_id' in cp else set()\n        needs_ids=set()\n        for gid in games.game_id.astype(str):\n            if gid not in existing_ids:\n                needs_ids.add(gid); continue\n            rr=cp[cp.game_id.astype(str)==gid].iloc[-1]\n            hs=str(rr.get('home_starter','') or '').strip(); aw=str(rr.get('away_starter','') or '').strip()\n            hl=_flag(rr.get('home_starter_line_ok',False)); al=_flag(rr.get('away_starter_line_ok',False))\n            if not (hs and aw and hl and al) or gid not in player_done_ids: needs_ids.add(gid)\n        missing=games[games.game_id.astype(str).isin(needs_ids)].copy()\n        print(f'[CHECKPOINT] year={year} existing={len(existing_ids)} player_enriched={len(player_done_ids)} repair_or_missing={len(missing)}')\n'''
if not pat.search(s): raise SystemExit("checkpoint block not found")
s=pat.sub(new,s,count=1)
s=s.replace('save_status(year,games,cp,completed,failures,complete=complete)', 'save_status(year,games,cp,completed,failures,complete=(complete and coverage >= MIN_STARTER_LINE_COVERAGE))',1)
s=s.replace("bad=eligible[eligible.starter_line_coverage_pct<70]", "bad=eligible[eligible.starter_line_coverage_pct<MIN_STARTER_LINE_COVERAGE]",1)
s=s.replace("complete_all=bool(statuses) and all(x.get('complete',False) for x in statuses if START_YEAR<=int(x.get('year',-1))<=END_YEAR) and len(statuses)>=(END_YEAR-START_YEAR+1)", "complete_all=(len(statuses)>=(END_YEAR-START_YEAR+1) and all((START_YEAR<=int(x.get('year',-1))<=END_YEAR) and bool(x.get('complete',False)) and not bool(x.get('unavailable',False)) and float(x.get('coverage_pct',0.0))>=MIN_STARTER_LINE_COVERAGE for x in statuses))",1)
s=MARK+'\n'+s; P.write_text(s,encoding='utf-8'); print('[NPB QUALITY PATCH] V1 applied')
