#!/usr/bin/env python3
"""Generic fail-closed data-quality gate for production result files."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path

ROOT=Path('results')
REPORT=ROOT/'production_data_quality.json'

def main():
    if not ROOT.exists(): raise SystemExit('DATA_QUALITY_FAIL: results directory missing')
    files=sorted(p for p in ROOT.rglob('*.csv') if p.is_file())
    if not files: raise SystemExit('DATA_QUALITY_FAIL: no CSV results')
    report={'schema':'production-data-quality-v1','status':'PASS','files':[]}
    for p in files:
        try:
            with p.open('r',encoding='utf-8',errors='strict',newline='') as f:
                reader=csv.DictReader(f); fields=reader.fieldnames or []; rows=list(reader)
        except Exception as e:
            raise SystemExit(f'DATA_QUALITY_FAIL: malformed CSV {p}: {e}')
        if not fields: raise SystemExit(f'DATA_QUALITY_FAIL: missing header: {p}')
        if not rows: raise SystemExit(f'DATA_QUALITY_FAIL: empty CSV: {p}')
        seen=set()
        for row in rows:
            key=tuple(row.get(c,'') for c in fields)
            if key in seen: raise SystemExit(f'DATA_QUALITY_FAIL: exact duplicate row: {p}')
            seen.add(key)
        missing={c:sum(1 for r in rows if r.get(c,'') in ('',None)) for c in fields}
        high_missing={c:n for c,n in missing.items() if n/len(rows)>0.20}
        numeric_outliers={}
        for c in fields:
            vals=[]
            for r in rows:
                try:
                    v=float(r.get(c,''));
                    if math.isfinite(v): vals.append(v)
                except Exception: pass
            if len(vals)>=20:
                vals.sort(); q1=vals[len(vals)//4]; q3=vals[(3*len(vals))//4]; iqr=q3-q1
                if iqr>0:
                    lo=q1-5*iqr; hi=q3+5*iqr
                    n=sum(v<lo or v>hi for v in vals)
                    if n/len(vals)>0.10: numeric_outliers[c]=n
        report['files'].append({'path':str(p),'rows':len(rows),'columns':len(fields),'missing_over_20pct':high_missing,'numeric_extreme_outliers_over_10pct':numeric_outliers,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
