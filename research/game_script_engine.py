"""Research-only entrypoint for NPB Game-Script v4.

Production integration is disabled until PIT availability is proven and the
challenger passes incumbent-comparable chronological OOS, calibration,
robustness, and frozen-holdout gates.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from .game_script_common import *  # noqa: F401,F403
from .game_script_common import _base, _count, _state, _transition
from .game_script_wfo import evaluate


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description="NPB game-script v4 research screening")
    p.add_argument("--data-glob",default="data/pbp/*.csv")
    p.add_argument("--development-end",default="2024-12-31")
    p.add_argument("--validation-start",default="2025-01-01")
    p.add_argument("--validation-end",default="2025-12-31")
    p.add_argument("--shadow-start",default="")
    p.add_argument("--shadow-end",default="")
    p.add_argument("--max-validation-games",type=int,default=160)
    p.add_argument("--max-shadow-games",type=int,default=60)
    p.add_argument("--simulations",type=int,default=600)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-ablation",action="store_true")
    p.add_argument("--shadow-only",action="store_true",help="refresh only the frozen current-month shadow using a complete compatible validation checkpoint")
    p.add_argument("--checkpoint-path",default="results/game_script_checkpoint.json")
    p.add_argument("--checkpoint-every",type=int,default=1)
    p.add_argument("--output",default="results/game_script_screening.json")
    a=p.parse_args(argv)
    files=sorted(Path().glob(a.data_glob))
    if not files: raise SystemExit(f"no PBP files matched: {a.data_glob}")
    result=evaluate(files,development_end=a.development_end,validation_start=a.validation_start,validation_end=a.validation_end,shadow_start=a.shadow_start or None,shadow_end=a.shadow_end or None,max_validation_games=max(1,a.max_validation_games),max_shadow_games=max(0,a.max_shadow_games),simulations=max(1,a.simulations),seed=a.seed,ablation=not a.no_ablation,checkpoint_path=a.checkpoint_path or None,checkpoint_every=max(1,a.checkpoint_every),run_validation=not a.shadow_only)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"decision":result["decision"],"pit_status":result["pit_status"],"validation":result["aggregate"]["validation"],"latest_validation_30":result["aggregate"]["latest_validation_30"],"ablation":result["ablation"],"recent_shadow":result["recent_shadow"],"checkpoint":result["checkpoint"]},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
