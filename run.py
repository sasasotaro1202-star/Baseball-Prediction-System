#!/usr/bin/env python3
"""Main entry point for sports prediction pipeline"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_collect_baseball(args):
    """Collect MLB data from StatsAPI"""
    sys.path.insert(0, str(ROOT))
    from pipelines.collect import collect_mlb
    collect_mlb()
    return 0


def cmd_features(args):
    """Build Elo baseline + base rate comparison (writes data/results/elo_baseline.json)"""
    result = subprocess.run([sys.executable, str(ROOT / "elo_baseline_all.py")], cwd=str(ROOT))
    return result.returncode


def cmd_research(args):
    """Alias for features - the current pipeline treats Elo baseline as the research step"""
    return cmd_features(args)


def cmd_predict(args):
    """Placeholder - staged model training not yet implemented, does not fail the pipeline"""
    print("predict: no trained model yet, nothing to do (exit 0)")
    return 0


def cmd_verify(args):
    """Placeholder - nothing to verify yet, does not fail the pipeline"""
    print("verify: no predictions yet, nothing to verify (exit 0)")
    return 0


def cmd_inventory(args):
    """Show data inventory by scanning data/ directly (no core.storage dependency)"""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        print("No data/ directory yet.")
        return 0

    for layer_dir in sorted(data_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        files = list(layer_dir.rglob("*.csv")) + list(layer_dir.rglob("*.parquet")) + list(layer_dir.rglob("*.json"))
        if files:
            print(f"{layer_dir.name}: {len(files)} files")
            for f in files[:10]:
                print(f"  - {f.relative_to(data_dir)} ({f.stat().st_size} bytes)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="run.py")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("collect-baseball")
    sp.set_defaults(func=cmd_collect_baseball)

    sp = sub.add_parser("features")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("research")
    sp.set_defaults(func=cmd_research)

    sp = sub.add_parser("predict")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("verify")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("inventory")
    sp.set_defaults(func=cmd_inventory)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except Exception as e:
        print(f"Command '{args.command}' failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
