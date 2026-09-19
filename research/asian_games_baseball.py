"""Fail-closed Aichi-Nagoya 2026 Asian Games baseball schedule adapter."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

OFFICIAL_NEWS="https://www.aichi-nagoya2026.org/ja/news-2050/"
OFFICIAL_BASEBALL="https://www.aichi-nagoya2026.org/ja/sport/baseball/"
OUT=Path("results/asian_games_baseball_schedule.json")

def fetch_schedule(url: str=OFFICIAL_NEWS, timeout: int=20)->dict:
    retrieved_at=datetime.now(timezone.utc).isoformat()
    r=requests.get(url,timeout=timeout,headers={"User-Agent":"Baseball-Prediction-System/1.0"})
    r.raise_for_status()
    body=r.text
    pub=None
    m=re.search(r'(?:article:published_time|datePublished)[^>]*content=["\']([^"\']+)',body,re.I)
    if m: pub=m.group(1)
    rows=[]
    for table in pd.read_html(body):
        text=table.to_string(index=False)
        if "BBL" not in text: continue
        for _,row in table.iterrows():
            vals=[str(x).strip() for x in row.tolist()]
            if "BBL" in " | ".join(vals): rows.append({"raw":vals})
    payload={"competition_id":"asian_games_baseball","source_url":url,
             "official_schedule_url":OFFICIAL_BASEBALL,"retrieved_at":retrieved_at,
             "publication_time":pub,
             "source_sha256":hashlib.sha256(body.encode()).hexdigest(),
             "schedule_rows":rows,"prediction_eligible":False,
             "eligibility_reason":"Starter PIT evidence and competition-specific historical OOS evidence are not assumed."}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload

if __name__=="__main__":
    print(json.dumps(fetch_schedule(),ensure_ascii=False,indent=2))
