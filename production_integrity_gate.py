#!/usr/bin/env python3
"""Fail-closed production output validation and provenance manifest."""
from __future__ import annotations
import csv, hashlib, json, math, os, platform, socket, subprocess
from datetime import datetime, timezone
from pathlib import Path
RESULTS=Path('results'); MANIFEST=Path('production_run_manifest.json')
def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def git_value(args):
 try: return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
 except Exception: return 'unknown'
def finite(value):
 try: return math.isfinite(float(value))
 except Exception: return False
def main():
 if not RESULTS.exists(): raise SystemExit('INTEGRITY_FAIL: results directory missing')
 files=[p for p in RESULTS.rglob('*') if p.is_file() and p.stat().st_size>0]; csvs=[p for p in files if p.suffix.lower()=='.csv']
 if not csvs: raise SystemExit('INTEGRITY_FAIL: no non-empty CSV results')
 checks={'csv_nonempty':True,'probabilities_valid':True,'duplicate_rows':True}; total_rows=0; probability_columns=[]
 for path in csvs:
  with path.open('r',encoding='utf-8',errors='replace',newline='') as f:
   reader=csv.DictReader(f); fields=reader.fieldnames or []; rows=list(reader)
  if not fields or not rows: raise SystemExit(f'INTEGRITY_FAIL: empty/headerless CSV: {path}')
  total_rows+=len(rows)
  for c in [c for c in fields if c.lower().endswith('prob') or 'probability' in c.lower()]:
   probability_columns.append(f'{path}:{c}')
   for i,row in enumerate(rows,2):
    if row.get(c,'')=='' or not finite(row[c]): raise SystemExit(f'INTEGRITY_FAIL: invalid probability {path}:{c} row={i}')
  seen=set()
  for row in rows:
   key=tuple(row.get(c,'') for c in fields)
   if key in seen: raise SystemExit(f'INTEGRITY_FAIL: duplicate result row in {path}')
   seen.add(key)
 if total_rows<2: raise SystemExit('INTEGRITY_FAIL: fewer than 2 total result rows')
 now=datetime.now(timezone.utc).isoformat()
 manifest={'schema':'production-integrity-v1','status':'PASS','validation_timestamp_utc':now,'prediction_timestamp_utc':os.environ.get('PREDICTION_TIMESTAMP_UTC','unknown'),'repository':os.environ.get('GITHUB_REPOSITORY','unknown'),'run_id':os.environ.get('GITHUB_RUN_ID','unknown'),'run_attempt':os.environ.get('GITHUB_RUN_ATTEMPT','unknown'),'commit_sha':os.environ.get('GITHUB_SHA',git_value(['rev-parse','HEAD'])),'branch':os.environ.get('GITHUB_REF_NAME',git_value(['branch','--show-current'])),'runner_name':os.environ.get('RUNNER_NAME',socket.gethostname()),'runner_os':platform.platform(),'python_version':platform.python_version(),'requirements_sha256':sha256(Path('requirements.txt')) if Path('requirements.txt').exists() else 'missing','checks':checks,'total_result_rows':total_rows,'probability_columns':probability_columns,'result_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(files)]}
 MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
