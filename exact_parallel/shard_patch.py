#!/usr/bin/env python3
"""Runtime-only patch that shards existing walk-forward blocks.

The feature matrix and every training prefix remain canonical. Only which
already-defined chronological retraining blocks a worker evaluates changes.
No model parameters, OOS split, feature logic, or random seed are changed.
"""
from pathlib import Path
import re
P=Path('baseball_backtest.py'); s=P.read_text(encoding='utf-8')
marker='# EXACT_PARALLEL_BLOCK_SHARD_V1'
if marker in s:
    print('[SHARD PATCH] already applied'); raise SystemExit(0)
old='''        for bstart in range(start, len(X), RETRAIN_EVERY):\n            bend = min(len(X), bstart + RETRAIN_EVERY)'''
new='''        exact_worker = int(os.getenv("BASEBALL_EXACT_SHARD_ID", "-1"))\n        exact_count = int(os.getenv("BASEBALL_EXACT_SHARD_COUNT", "4"))\n        exact_blocks = 0\n        for bstart in range(start, len(X), RETRAIN_EVERY):\n            block_no = exact_blocks\n            exact_blocks += 1\n            if exact_worker >= 0 and exact_count > 0 and (block_no % exact_count) != exact_worker:\n                continue\n            bend = min(len(X), bstart + RETRAIN_EVERY)'''
if old not in s: raise SystemExit('[SHARD PATCH] loop anchor not found')
s=s.replace(old,new,1)
s=marker+'\n'+s
P.write_text(s,encoding='utf-8')
print('[SHARD PATCH] exact chronological block sharding enabled')
