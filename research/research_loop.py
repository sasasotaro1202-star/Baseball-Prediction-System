#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Autonomous baseball research controller with fixed promotion policy."""
from __future__ import annotations
import json, math, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RESULTS=ROOT/'results'; STATE=ROOT/'research_state.json'; HISTORY=ROOT/'research_history.json'; PLAN=ROOT/'research_plan.json'
OBJECTIVES=[
 ('win','勝敗確率の校正とモデル重み',['Accuracy','LogLoss','Brier']),
 ('score','得点分布・スコア候補',['MeanAbsoluteScoreError','Top4ScoreHitRate']),
 ('low_high','Low/High境界と確率校正',['LowHighAccuracy']),
 ('starter','先発投手情報の品質と寄与',['starter_coverage','starter_accuracy']),
 ('bullpen','ブルペン疲労・救援力',['HighAccuracy','MeanAbsoluteScoreError']),
 ('batting','打線・対左右・直近フォーム',['Accuracy','MeanAbsoluteScoreError']),
 ('environment','球場・天候・日程環境',['MeanAbsoluteScoreError','LowHighAccuracy']),
]
PROMOTION_GATE={
 'require_oos':True,
 'min_oos_rows':200,
 'min_relative_improvement':0.01,
 'max_logloss_regression':0.005,
 'max_brier_regression':0.005,
 'max_accuracy_regression':0.005,
 'require_two_validation_windows':True,
 'require_calibration_check':True,
 'require_no_future_target_data':True,
 'require_reproducible_candidate':True,
 'require_npb_three_way_check':True,
 'max_draw_recall_regression':0.0,
 'max_draw_probability_mae_regression':0.005,
 'require_score_check':True,
 'require_hilo_check':True,
 'rollback_on_post_promotion_regression':True,
}

def read_csv(name):
 p=RESULTS/name
 if not p.exists(): return pd.DataFrame()
 try: return pd.read_csv(p)
 except Exception: return pd.DataFrame()

def finite(x,default=float('nan')):
 try:
  v=float(x); return v if math.isfinite(v) else default
 except Exception: return default

def latest_summary(league):
 df=read_csv(f'{league.lower()}_backtest_summary.csv'); return {} if df.empty else df.iloc[-1].to_dict()

def model_leaderboard(league):
 df=read_csv(f'{league.lower()}_model_comparison.csv')
 if df.empty or 'model' not in df.columns: return []
 def key(r): return (finite(r.get('LogLoss'),99.0),finite(r.get('Brier'),99.0),-finite(r.get('Accuracy'),0.0))
 return sorted(df.to_dict('records'),key=key)

def detailed_weakness(league):
 df=read_csv(f'{league.lower()}_accuracy_detail.csv')
 if df.empty: return {}
 out={}
 for col in ('home_team','away_team','model','condition','segment'):
  if col in df.columns: out[col]=int(df[col].nunique(dropna=True))
 for metric in ('Accuracy','LogLoss','Brier','MeanAbsoluteScoreError','LowHighAccuracy','HighAccuracy','LowAccuracy'):
  if metric in df.columns: out[metric]=finite(df[metric].mean())
 return out

def build_state():
 now=datetime.now(timezone.utc).isoformat(); leagues={}; weaknesses=[]
 for league in ('NPB','MLB'):
  summary=latest_summary(league); leaderboard=model_leaderboard(league); detail=detailed_weakness(league)
  leagues[league]={'summary':summary,'model_leaderboard':leaderboard,'detail':detail}
  if summary:
   acc=finite(summary.get('Accuracy'),0.5); ll=finite(summary.get('LogLoss'),1.0); brier=finite(summary.get('Brier'),0.25); mae=finite(summary.get('MeanAbsoluteScoreError'),3.0); lh=finite(summary.get('LowHighAccuracy'),0.5)
   weaknesses += [
    {'league':league,'objective':'win','priority':max(0,0.65-acc)+max(0,ll-0.67),'reason':'勝敗精度/確率損失'},
    {'league':league,'objective':'score','priority':max(0,mae-2.0)*0.35,'reason':'スコア誤差'},
    {'league':league,'objective':'low_high','priority':max(0,0.68-lh),'reason':'Low/High精度'},
    {'league':league,'objective':'calibration','priority':max(0,brier-0.25),'reason':'確率校正'},
   ]
 weaknesses.sort(key=lambda x:x['priority'],reverse=True); focus=weaknesses[0] if weaknesses else {'league':'NPB','objective':'win','priority':1.0,'reason':'評価データ待ち'}
 return {
  'schema_version':3,'updated_at':now,'goal':'maximize validated out-of-sample accuracy for every prediction item',
  'processing_budget_seconds':int(os.getenv('BASEBALL_TIME_BUDGET_SEC','3600')),
  'focus':focus,'weaknesses':weaknesses[:20],'leagues':leagues,
  'promotion_gate':PROMOTION_GATE,
  'next_research':{'objective':focus['objective'],'league':focus['league'],'reason':focus['reason'],'rule':'candidate must pass the fixed promotion gate and improve OOS without unacceptable regression; NPB must preserve the Draw class'},
 }

def append_history(state):
 history=[]
 if HISTORY.exists():
  try:
   raw=json.loads(HISTORY.read_text(encoding='utf-8')); history=raw if isinstance(raw,list) else []
  except Exception: pass
 history.append({'timestamp':state['updated_at'],'focus':state['focus'],'next_research':state['next_research'],'promotion_gate':state['promotion_gate'],'model_winners':{l:(d['model_leaderboard'][0] if d['model_leaderboard'] else None) for l,d in state['leagues'].items()}})
 HISTORY.write_text(json.dumps(history[-500:],ensure_ascii=False,indent=2),encoding='utf-8')

def main():
 state=build_state(); STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8'); append_history(state)
 PLAN.write_text(json.dumps({'generated_at':state['updated_at'],'priority':state['focus'],'next':state['next_research'],'promotion_gate':PROMOTION_GATE,'automatic_adoption_policy':{'require_oos':True,'require_no_material_regression':True,'keep_failed_candidates_for_learning':True,'never_use_future_target_data':True,'fixed_gate':True,'npb_three_way_required':True}},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'focus':state['focus'],'next':state['next_research'],'promotion_gate':PROMOTION_GATE},ensure_ascii=False))
if __name__=='__main__': main()
