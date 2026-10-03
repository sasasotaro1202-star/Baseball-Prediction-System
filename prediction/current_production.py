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
    # Manual workflows may checkout the live main branch after dispatch.
    # Prefer the commit actually checked out by the runner.
    for env_name in ("BASEBALL_CHECKED_OUT_SHA", "GITHUB_SHA"):
        value = os.getenv(env_name, "").strip()
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
            "freshness_policy": "LIVE_CALL_TIME;DO_NOT_REUSE_PRIOR_PREDICTION_OUTPUT",
            "prediction_requested_at_utc": datetime.now(timezone.utc).isoformat(),
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


def latest_target_date_jst() -> str:
    """Resolve the target date at call time; stale prior prediction dates are never reused."""
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")


def predict_current(
    *,
    league: str,
    target_date: str | None = None,
    data_dir: str = "data",
    pregame_only: bool = False,
) -> dict[str, Any]:
    """Run the current production runtime at the live call-time target date.

    A stale explicit date is fail-closed rather than silently reusing an older
    prediction horizon. Historical dates belong to research/audit entry points,
    not the user-facing current-production path.
    """
    requested_at = datetime.now(timezone.utc).isoformat()
    resolved_target_date = latest_target_date_jst()
    info = current_runtime(league)
    if not info["available"]:
        return {
            **info,
            "execution_status": "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
            "prediction_request_mode": "on_demand",
            "prediction_schedule": "on_demand",
            "prediction_source": "MANUAL_LIVE",
            "prediction_requested_at_utc": requested_at,
            "resolved_target_date": resolved_target_date,
            "predictions": [],
            "prediction_generated_at": requested_at,
        }
    if target_date is not None and str(target_date) != resolved_target_date:
        return {
            **info,
            "execution_status": "BLOCKED_STALE_TARGET_DATE",
            "requested_target_date": str(target_date),
            "resolved_target_date": resolved_target_date,
            "predictions": [],
            "prediction_generated_at": datetime.now(timezone.utc).isoformat(),
            "reason": "current-production prediction requests must use the call-time JST target date",
        }

    if info["entrypoint"] == "production_npb":
        from production_npb import predict

        date_value = resolved_target_date
        result = predict(date_value, data_dir, pregame_only=bool(pregame_only))
        if not isinstance(result, Mapping):
            raise RuntimeError("current production runtime returned a non-object result")

        # Add an immutable runtime identity to every current-production result.
        out = dict(result)
        out["prediction_request_mode"] = "on_demand"
        out["prediction_requested_at_utc"] = requested_at
        for pred in out.get("predictions", []) if isinstance(out.get("predictions"), list) else []:
            if not isinstance(pred, dict):
                continue
            pred.setdefault("prediction_source", "MANUAL_LIVE")
            pred.setdefault("prediction_schedule", "on_demand")
            pred.setdefault("prediction_request_mode", "on_demand")
            try:
                generated = datetime.fromisoformat(
                    str(pred["prediction_generated_at"]).replace("Z", "+00:00")
                )
                game_time = datetime.fromisoformat(
                    str(pred["datetime_jst"]).replace("Z", "+00:00")
                )
                if generated.tzinfo is not None and game_time.tzinfo is not None:
                    pred.setdefault(
                        "prediction_target_lead_minutes",
                        round((game_time - generated).total_seconds() / 60.0, 3),
                    )
            except Exception:
                # Prediction-time contract validation remains the authoritative
                # fail-closed guard; metadata annotation must never invent a lead.
                pass
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
    parser.add_argument(
        "--pregame-only",
        action="store_true",
        help="Only emit games inside the hard 30-minute pregame automation window.",
    )
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)

    if args.describe:
        print(json.dumps(current_runtime(args.league), ensure_ascii=False, indent=2))
        return 0

    result = predict_current(
        league=args.league,
        target_date=args.date,
        data_dir=args.data_dir,
        pregame_only=args.pregame_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("execution_status") == "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME":
        return 1
    # BLOCKED_STARTERS and NO_FUTURE_GAMES are explicit, fail-closed
    # operational states that the scheduler is expected to revisit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
