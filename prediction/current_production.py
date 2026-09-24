"""Stable dispatcher for the current checked-in baseball production runtime.

This layer deliberately separates:
1) the production runtime actually callable right now;
2) research candidates, which can never become a prediction fallback;
3) formal promotion/adoption evidence, which remains governed by the existing
fail-closed gates.

A model update changes the manifest/configuration, not the user-facing command.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "current_production_runtime.json"


def _git_commit() -> str:
    value = os.getenv("GITHUB_SHA", "").strip()
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_registry() -> dict[str, Any]:
    if not CONFIG.exists():
        raise RuntimeError(f"current production runtime registry missing: {CONFIG}")
    try:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid current production runtime registry: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("current production runtime registry must be an object")
    if payload.get("schema_version") != "current-production-runtime-v1":
        raise RuntimeError("unsupported current production runtime registry schema")
    runtimes = payload.get("runtimes")
    if not isinstance(runtimes, dict):
        raise RuntimeError("current production runtime registry has no runtimes object")
    return payload


def current_runtime(league: str) -> dict[str, Any]:
    league = str(league).strip().upper()
    payload = _load_registry()
    runtime = payload["runtimes"].get(league)
    if not isinstance(runtime, dict):
        return {
            "available": False,
            "league": league,
            "status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
            "reason": "no current production runtime is registered for this league",
            "git_commit": _git_commit(),
        }

    status = str(runtime.get("formal_adoption_status", "")).strip().upper()
    required = ("entrypoint", "model_version", "contract")
    missing = [key for key in required if not str(runtime.get(key, "")).strip()]
    if status.startswith("BLOCKED_"):
        # GOVERNED_* means the runtime is callable, while formal adoption is
        # still controlled by the independent promotion gate.
        return {
            "available": False,
            "league": league,
            "status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
            "reason": f"runtime registration is not currently callable: {status or 'UNSPECIFIED'}",
            **runtime,
            "git_commit": _git_commit(),
        }
    if missing:
        raise RuntimeError(
            "current production runtime registry is incomplete: " + ", ".join(missing)
        )

    return {
        "available": True,
        "league": league,
        "status": "AVAILABLE",
        **runtime,
        "git_commit": _git_commit(),
    }


def predict_current(
    *,
    league: str,
    target_date: str | None = None,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Run the current production runtime only.

    No research candidate, baseline, or prior model is used as a fallback.
    """
    info = current_runtime(league)
    if not info["available"]:
        return {
            **info,
            "execution_status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
            "predictions": [],
            "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    if info["entrypoint"] == "production_npb":
        from production_npb import predict

        date_value = (
            target_date
            or datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
        )
        result = predict(date_value, data_dir)
        if not isinstance(result, Mapping):
            raise RuntimeError("current production runtime returned a non-object result")

        # Add an immutable runtime identity to every current-production result.
        out = dict(result)
        out["current_production_runtime"] = {
            "league": league.upper(),
            "model_version": info["model_version"],
            "contract": info["contract"],
            "entrypoint": info["entrypoint"],
            "git_commit": _git_commit(),
        }
        return out

    raise RuntimeError(
        f"unsupported current production entrypoint for {league}: {info['entrypoint']!r}"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="current-production",
        description="Run the current checked-in production runtime only.",
    )
    parser.add_argument("--league", required=True, choices=("NPB", "MLB"))
    parser.add_argument("--date", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)

    if args.describe:
        print(json.dumps(current_runtime(args.league), ensure_ascii=False, indent=2))
        return 0

    result = predict_current(
        league=args.league,
        target_date=args.date,
        data_dir=args.data_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("execution_status") == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME":
        return 1
    # BLOCKED_STARTERS and NO_FUTURE_GAMES are explicit, fail-closed
    # operational states that the scheduler is expected to revisit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
