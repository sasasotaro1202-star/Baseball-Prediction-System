#!/usr/bin/env python3
"""Route a user-requested baseball prediction to the strongest eligible GitHub lane.

This is an execution router, not a model. It never promotes research output,
never bypasses PIT gates, and never substitutes an external forecast for a
missing GitHub result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = ROOT / "prediction_requests" / "inbox" / "active.json"
POLICY_PATH = ROOT / "config" / "prediction_request_policy.json"
RUNTIME_PATH = ROOT / "config" / "current_production_runtime.json"
JST = ZoneInfo("Asia/Tokyo")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today_jst() -> str:
    return _utc_now().astimezone(JST).strftime("%Y-%m-%d")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def _fingerprint(request: dict[str, Any]) -> str:
    stable = {
        k: v for k, v in request.items()
        if k not in {"request_id", "requested_at_utc"}
    }
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_target_date(request: dict[str, Any]) -> str:
    value = str(request.get("target_date", "")).strip()
    if not value or value.lower() == "today":
        return _today_jst()
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("target_date must be YYYY-MM-DD or 'today'") from exc
    today = datetime.strptime(_today_jst(), "%Y-%m-%d").date()
    if parsed < today:
        raise ValueError("past target dates are not valid for user-facing prediction requests")
    return value


def _competition_id(request: dict[str, Any]) -> str:
    value = (
        request.get("competition_id")
        or request.get("league")
        or request.get("competition")
    )
    value = str(value or "").strip().upper()
    if not value:
        raise ValueError("competition_id is required for the GitHub request router")
    return value


def _format_command(template: list[str], target_date: str) -> list[str]:
    return [str(part).replace("{target_date}", target_date) for part in template]


def _production_available(competition_id: str, runtime: dict[str, Any]) -> bool:
    item = runtime.get("runtimes", {}).get(competition_id)
    if not isinstance(item, dict):
        return False
    return (
        str(item.get("formal_adoption_status", "")).strip().upper() == "CURRENT_PRODUCTION"
        and bool(str(item.get("entrypoint", "")).strip())
        and bool(str(item.get("model_version", "")).strip())
        and bool(str(item.get("contract", "")).strip())
    )


def _validate_generated_output(payload: dict[str, Any], request: dict[str, Any], lane: str) -> None:
    status = str(payload.get("execution_status", "")).strip()
    allowed = {
        "EXECUTED",
        "RESEARCH_SHADOW_EXECUTED",
        "NO_FUTURE_GAMES",
        "NO_DUE_PREGAME_GAMES",
        "BLOCKED_STARTERS",
        "BLOCKED_PRODUCTION_GATE",
        "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
    }
    if status not in allowed:
        raise ValueError(f"unrecognized prediction execution status: {status}")
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("prediction output predictions must be a list")

    usable = {"EXECUTED", "RESEARCH_SHADOW_EXECUTED"}
    if status in usable and not predictions:
        raise ValueError("prediction output claims execution but contains no predictions")
    if status in usable:
        if str(payload.get("pit_status", "")) != "PASS":
            raise ValueError("usable GitHub prediction must have PIT PASS")
        for pred in predictions:
            if not isinstance(pred, dict):
                raise ValueError("prediction row must be an object")
            for key in ("game_id", "home", "away"):
                if not str(pred.get(key, "")).strip():
                    raise ValueError(f"prediction row missing {key}")
        expected_research = lane == "VALIDATED_RESEARCH_SHADOW"
        if expected_research and status != "RESEARCH_SHADOW_EXECUTED":
            raise ValueError("research lane output must retain RESEARCH_SHADOW_EXECUTED status")


def _find_cached(result_dir: Path, fingerprint: str) -> Path | None:
    if not result_dir.exists():
        return None
    for path in sorted(result_dir.glob("*.json"), reverse=True):
        try:
            obj = _load_json(path)
        except Exception:
            continue
        if str(obj.get("request_fingerprint", "")) == fingerprint:
            return path
    return None


def _run(command: list[str], timeout_seconds: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _load_generated_output(path: Path) -> dict[str, Any]:
    return _load_json(path)


def _write_result(
    *,
    result_dir: Path,
    request: dict[str, Any],
    request_id: str,
    fingerprint: str,
    target_date: str,
    competition_id: str,
    lane: str,
    source_commit: str,
    status: str,
    output: dict[str, Any],
    reused_from: str | None = None,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "baseball-user-prediction-result-v1",
        "request_id": request_id,
        "request_fingerprint": fingerprint,
        "request": request,
        "target_date": target_date,
        "competition_id": competition_id,
        "source_repository": "sasasotaro1202-star/Baseball-Prediction-System",
        "source_commit": source_commit,
        "generation_lane": lane,
        "generation_status": status,
        "generated_at_utc": _utc_now().isoformat(),
        "reused_from_request_id": reused_from,
        "prediction_output": output,
    }
    out = result_dir / f"{request_id}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", default=str(DEFAULT_REQUEST))
    args = parser.parse_args(argv)

    request_path = Path(args.request_file)
    if not request_path.is_absolute():
        request_path = ROOT / request_path

    request = _load_json(request_path)
    if request.get("schema_version") != "baseball-prediction-request-v1":
        raise ValueError("unsupported request schema_version")
    request_id = str(request.get("request_id", "")).strip()
    if not request_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for ch in request_id):
        raise ValueError("request_id must be a non-empty filesystem-safe identifier")

    target_date = _resolve_target_date(request)
    competition_id = _competition_id(request)
    fingerprint = _fingerprint(request)
    policy = _load_json(POLICY_PATH)
    runtime = _load_json(RUNTIME_PATH)
    result_dir = ROOT / str(policy.get("result_directory", "prediction_requests/results"))
    source_commit = os.environ.get("GITHUB_SHA") or "unknown"

    cached = _find_cached(result_dir, fingerprint)
    if cached is not None:
        cached_obj = _load_json(cached)
        cached_output = cached_obj.get("prediction_output")
        if not isinstance(cached_output, dict):
            raise ValueError("cached result does not contain a valid prediction_output object")
        _validate_generated_output(cached_output, request, str(cached_obj.get("generation_lane", "")))
        out = _write_result(
            result_dir=result_dir,
            request=request,
            request_id=request_id,
            fingerprint=fingerprint,
            target_date=target_date,
            competition_id=competition_id,
            lane=str(cached_obj.get("generation_lane", "CACHED")),
            source_commit=source_commit,
            status="CACHED_VERIFIED_RESULT",
            output=cached_output,
            reused_from=str(cached_obj.get("request_id", "")) or None,
        )
        print(json.dumps({"result_path": str(out), "generation_status": "CACHED_VERIFIED_RESULT"}, ensure_ascii=False))
        return 0

    production_template = policy.get("production_commands", {}).get(competition_id)
    research_template = policy.get("validated_research_shadow_commands", {}).get(competition_id)

    command: list[str] | None = None
    lane = ""
    if target_date == _today_jst() and _production_available(competition_id, runtime) and isinstance(production_template, list):
        command = _format_command(production_template, target_date)
        lane = "CURRENT_PRODUCTION_RUNTIME"
    elif isinstance(research_template, list):
        command = _format_command(research_template, target_date)
        lane = "VALIDATED_RESEARCH_SHADOW"

    if command is None:
        output = {
            "schema_version": "baseball-user-prediction-output-v1",
            "target_date": target_date,
            "execution_status": "BLOCKED_NO_VALIDATED_PREDICTION_RUNTIME",
            "pit_status": "NOT_RUN",
            "predictions": [],
            "block_reason": (
                "No current production runtime or explicitly registered validated "
                f"research runtime exists for competition {competition_id}."
            ),
        }
        out = _write_result(
            result_dir=result_dir,
            request=request,
            request_id=request_id,
            fingerprint=fingerprint,
            target_date=target_date,
            competition_id=competition_id,
            lane="BLOCKED",
            source_commit=source_commit,
            status="BLOCKED_NO_VALIDATED_PREDICTION_RUNTIME",
            output=output,
        )
        print(json.dumps({"result_path": str(out), "generation_status": "BLOCKED_NO_VALIDATED_PREDICTION_RUNTIME"}, ensure_ascii=False))
        return 0

    timeout_seconds = int(policy.get("max_runtime_seconds", 5400))
    try:
        rc, stdout, stderr = _run(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        output = {
            "execution_status": "GENERATION_TIMEOUT",
            "pit_status": "UNKNOWN",
            "predictions": [],
            "error": "prediction generation command exceeded bounded runtime",
            "stderr": str(exc),
        }
        out = _write_result(
            result_dir=result_dir,
            request=request,
            request_id=request_id,
            fingerprint=fingerprint,
            target_date=target_date,
            competition_id=competition_id,
            lane=lane,
            source_commit=source_commit,
            status="GENERATION_TIMEOUT",
            output=output,
        )
        print(json.dumps({"result_path": str(out), "generation_status": "GENERATION_TIMEOUT"}, ensure_ascii=False))
        return 1

    generated_path = ROOT / "results" / f"npb_{'shadow' if lane == 'VALIDATED_RESEARCH_SHADOW' else 'production'}_{target_date}.json"
    if competition_id != "NPB" or not generated_path.exists():
        # Production/other routes are expected to return their JSON on stdout
        # when they don't use the shared results path.
        try:
            output = json.loads(stdout)
        except json.JSONDecodeError:
            output = {
                "execution_status": "GENERATION_OUTPUT_UNVERIFIABLE",
                "pit_status": "UNKNOWN",
                "predictions": [],
                "stderr": stderr[-4000:],
                "stdout_tail": stdout[-4000:],
            }
    else:
        output = _load_generated_output(generated_path)

    if rc != 0 and str(output.get("execution_status", "")).strip() not in {
        "BLOCKED_STARTERS",
        "BLOCKED_PRODUCTION_GATE",
        "NO_FUTURE_GAMES",
        "NO_DUE_PREGAME_GAMES",
    }:
        output = {
            "execution_status": "GENERATION_FAILED",
            "pit_status": "UNKNOWN",
            "predictions": [],
            "returncode": rc,
            "stderr": stderr[-4000:],
            "stdout_tail": stdout[-4000:],
        }

    _validate_generated_output(output, request, lane)
    final_status = str(output.get("execution_status", "UNKNOWN"))
    out = _write_result(
        result_dir=result_dir,
        request=request,
        request_id=request_id,
        fingerprint=fingerprint,
        target_date=target_date,
        competition_id=competition_id,
        lane=lane,
        source_commit=source_commit,
        status=final_status,
        output=output,
    )
    print(json.dumps({
        "result_path": str(out),
        "generation_status": final_status,
        "generation_lane": lane,
        "prediction_count": len(output.get("predictions", [])),
    }, ensure_ascii=False))
    return 0 if final_status in {
        "EXECUTED",
        "RESEARCH_SHADOW_EXECUTED",
        "NO_FUTURE_GAMES",
        "NO_DUE_PREGAME_GAMES",
        "BLOCKED_STARTERS",
        "BLOCKED_PRODUCTION_GATE",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
