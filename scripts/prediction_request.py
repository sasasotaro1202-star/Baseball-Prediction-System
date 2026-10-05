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
        "BLOCKED_NO_VALIDATED_PREDICTION_RUNTIME",
        "GENERATION_OUTPUT_UNVERIFIABLE",
        "GENERATION_FAILED",
        "GENERATION_TIMEOUT",
    }
    if status not in allowed:
        raise ValueError(f"unrecognized prediction execution status: {status}")
    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        raise ValueError("prediction output predictions must be a list")

    non_usable = {"GENERATION_OUTPUT_UNVERIFIABLE", "GENERATION_FAILED", "GENERATION_TIMEOUT"}
    if status in non_usable:
        if predictions:
            raise ValueError(f"{status} must not contain predictions")
        pit = str(payload.get("pit_status", "")).strip().upper()
        if pit not in {"UNKNOWN", "NOT_RUN"}:
            raise ValueError(f"{status} must preserve UNKNOWN/NOT_RUN PIT status")
        return

    usable = {"EXECUTED", "RESEARCH_SHADOW_EXECUTED"}
    if status in usable and not predictions:
        raise ValueError("prediction output claims execution but contains no predictions")
    if status in usable:
        pit_status = str(payload.get("pit_status", "")).strip().upper()
        expected_research = lane in {
            "VALIDATED_RESEARCH_SHADOW",
            "COMPETITION_SPECIFIC_RESEARCH_RUNTIME",
        }
        if expected_research:
            if status != "RESEARCH_SHADOW_EXECUTED":
                raise ValueError("research lane output must retain RESEARCH_SHADOW_EXECUTED status")
            if str(payload.get("scope", "")).strip().upper() != "RESEARCH_SHADOW":
                raise ValueError("research prediction must declare RESEARCH_SHADOW scope")
            if payload.get("production_eligibility") is not False:
                raise ValueError("research prediction must explicitly remain production-ineligible")
            if pit_status not in {"PASS", "UNVERIFIABLE"}:
                raise ValueError("research prediction must declare PIT PASS or UNVERIFIABLE")
        elif pit_status != "PASS":
            raise ValueError("current production prediction must have PIT PASS")
        for pred in predictions:
            if not isinstance(pred, dict):
                raise ValueError("prediction row must be an object")
            for key in ("game_id", "home", "away"):
                if not str(pred.get(key, "")).strip():
                    raise ValueError(f"prediction row missing {key}")

            # Competition / phase identity is part of the prediction contract.
            # Never infer a missing or UNKNOWN classification from team names
            # or calendar position, even in the research lane.
            classification_status = str(
                pred.get("competition_classification_status", "")
            ).strip().lower()
            competition = str(pred.get("competition", "")).strip().lower()
            stage = str(pred.get("competition_stage", "")).strip().lower()
            competition_key = str(pred.get("competition_key", "")).strip()
            if (
                classification_status != "classified"
                or not competition_key
                or not competition
                or competition.endswith("_unknown")
                or stage in {"", "unknown"}
            ):
                raise ValueError(
                    "competition classification must be explicit and non-UNKNOWN"
                )


def _find_cached(result_dir: Path, fingerprint: str, lane: str, max_age_seconds: int) -> Path | None:
    if not result_dir.exists():
        return None
    for path in sorted(result_dir.glob("*.json"), reverse=True):
        try:
            obj = _load_json(path)
        except Exception:
            continue
        if str(obj.get("request_fingerprint", "")) != fingerprint:
            continue
        if str(obj.get("generation_lane", "")) != lane:
            continue
        cached_output = obj.get("prediction_output")
        if not isinstance(cached_output, dict):
            continue
        status = str(cached_output.get("execution_status", "")).strip()
        if status not in {"EXECUTED", "RESEARCH_SHADOW_EXECUTED"}:
            continue
        try:
            generated_at = datetime.fromisoformat(
                str(obj.get("generated_at_utc")).replace("Z", "+00:00")
            )
            age = (_utc_now() - generated_at).total_seconds()
        except Exception:
            continue
        if age < 0 or age > max_age_seconds:
            continue
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


def _default_output_template(competition_id: str, lane: str) -> str | None:
    """Return the repository's conventional prediction output path when policy omits one."""
    prefixes = {
        ("NPB", "CURRENT_PRODUCTION_RUNTIME"): "results/npb_production_{target_date}.json",
        ("NPB", "VALIDATED_RESEARCH_SHADOW"): "results/npb_shadow_{target_date}.json",
        ("MLB", "CURRENT_PRODUCTION_RUNTIME"): "results/mlb_production_{target_date}.json",
        ("MLB", "VALIDATED_RESEARCH_SHADOW"): "results/mlb_shadow_{target_date}.json",
    }
    return prefixes.get((str(competition_id).upper(), str(lane)))


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

    # Repository execution uses ROOT, while isolated tests and explicitly supplied
    # request files may live outside the repository. Resolve relative result paths
    # from the request's containing execution root so artifact discovery remains
    # deterministic without hard-coding the test filesystem.
    resolved_request = request_path.resolve()
    try:
        request_root = ROOT if resolved_request.is_relative_to(ROOT) else resolved_request.parent
    except AttributeError:
        request_root = ROOT if str(resolved_request).startswith(str(ROOT)) else resolved_request.parent
    result_dir_value = Path(str(policy.get("result_directory", "prediction_requests/results")))
    result_dir = result_dir_value if result_dir_value.is_absolute() else request_root / result_dir_value
    source_commit = os.environ.get("GITHUB_SHA") or "unknown"

    production_template = policy.get("production_commands", {}).get(competition_id)
    research_template = policy.get("validated_research_shadow_commands", {}).get(competition_id)
    competition_research_template = policy.get("competition_specific_research_commands", {}).get(competition_id)

    command: list[str] | None = None
    lane = ""
    if target_date == _today_jst() and _production_available(competition_id, runtime) and isinstance(production_template, list):
        command = _format_command(production_template, target_date)
        lane = "CURRENT_PRODUCTION_RUNTIME"
    elif isinstance(research_template, list):
        command = _format_command(research_template, target_date)
        lane = "VALIDATED_RESEARCH_SHADOW"
    elif isinstance(competition_research_template, list):
        command = _format_command(competition_research_template, target_date)
        lane = "COMPETITION_SPECIFIC_RESEARCH_RUNTIME"

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

    cache_ttl = int(policy.get("max_verified_cache_age_seconds", 900))
    force_refresh = bool(request.get("force_refresh", False))
    if cache_ttl < 0:
        raise ValueError("max_verified_cache_age_seconds must be non-negative")
    if not force_refresh:
        cached = _find_cached(result_dir, fingerprint, lane, cache_ttl)
        if cached is not None:
            cached_obj = _load_json(cached)
            cached_output = cached_obj.get("prediction_output")
            if not isinstance(cached_output, dict):
                raise ValueError("cached result does not contain a valid prediction_output object")
            _validate_generated_output(cached_output, request, lane)
            out = _write_result(
                result_dir=result_dir,
                request=request,
                request_id=request_id,
                fingerprint=fingerprint,
                target_date=target_date,
                competition_id=competition_id,
                lane=lane,
                source_commit=str(cached_obj.get("source_commit") or source_commit),
                status="CACHED_VERIFIED_RESULT",
                output=cached_output,
                reused_from=str(cached_obj.get("request_id", "")) or None,
            )
            print(json.dumps({"result_path": str(out), "generation_status": "CACHED_VERIFIED_RESULT"}, ensure_ascii=False))
            return 0

    timeout_seconds = int(policy.get("max_runtime_seconds", 5400))

    def execute_lane(run_command: list[str], run_lane: str) -> tuple[int, dict[str, Any], str]:
        try:
            run_rc, run_stdout, run_stderr = _run(run_command, timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            return 1, {
                "execution_status": "GENERATION_TIMEOUT",
                "pit_status": "UNKNOWN",
                "predictions": [],
                "error": "prediction generation command exceeded bounded runtime",
                "stderr": str(exc),
            }, str(exc)

        run_output_paths = policy.get("output_paths", {}).get(competition_id, {})
        run_path_template = (
            run_output_paths.get(run_lane)
            if isinstance(run_output_paths, dict)
            else None
        )
        if not run_path_template:
            run_path_template = _default_output_template(competition_id, run_lane)
        if run_path_template:
            run_generated_value = Path(
                str(run_path_template).replace("{target_date}", target_date)
            )
            run_generated_path = (
                run_generated_value
                if run_generated_value.is_absolute()
                else request_root / run_generated_value
            )
        else:
            run_generated_path = None

        if run_generated_path is None or not run_generated_path.exists():
            try:
                run_output = json.loads(run_stdout)
            except json.JSONDecodeError:
                run_output = {
                    "execution_status": "GENERATION_OUTPUT_UNVERIFIABLE",
                    "pit_status": "UNKNOWN",
                    "predictions": [],
                    "stderr": run_stderr[-4000:],
                    "stdout_tail": run_stdout[-4000:],
                }
        else:
            run_output = _load_generated_output(run_generated_path)

        if run_rc != 0 and str(run_output.get("execution_status", "")).strip() not in {
            "BLOCKED_STARTERS",
            "BLOCKED_PRODUCTION_GATE",
            "NO_FUTURE_GAMES",
            "NO_DUE_PREGAME_GAMES",
        }:
            run_output = {
                "execution_status": "GENERATION_FAILED",
                "pit_status": "UNKNOWN",
                "predictions": [],
                "returncode": run_rc,
                "stderr": run_stderr[-4000:],
                "stdout_tail": run_stdout[-4000:],
            }

        return run_rc, run_output, run_stderr

    rc, output, stderr = execute_lane(command, lane)

    # A user request must remain answerable even when the current production
    # runtime is temporarily blocked (most commonly by starter publication).
    # This is an explicit research fallback, never a production fallback:
    # production output is preserved as provenance and the research result stays
    # RESEARCH_SHADOW / production-ineligible.
    primary_lane = lane
    primary_status = str(output.get("execution_status", "")).strip()
    research_fallback_statuses = {
        "BLOCKED_STARTERS",
        "BLOCKED_PRODUCTION_GATE",
        "BLOCKED_NO_CURRENT_PRODUCTION_RUNTIME",
        "GENERATION_FAILED",
        "GENERATION_TIMEOUT",
    }
    if (
        lane == "CURRENT_PRODUCTION_RUNTIME"
        and isinstance(research_template, list)
        and not output.get("predictions")
        and primary_status in research_fallback_statuses
    ):
        research_command = _format_command(research_template, target_date)
        fallback_rc, fallback_output, fallback_stderr = execute_lane(
            research_command,
            "VALIDATED_RESEARCH_SHADOW",
        )
        fallback_status = str(fallback_output.get("execution_status", "")).strip()
        if fallback_status in {"RESEARCH_SHADOW_EXECUTED", "NO_FUTURE_GAMES", "NO_DUE_PREGAME_GAMES"}:
            fallback_output["user_fallback_from_lane"] = primary_lane
            fallback_output["user_fallback_reason"] = primary_status
            fallback_output["primary_production_status"] = primary_status
            if stderr:
                fallback_output["primary_production_stderr_tail"] = stderr[-4000:]
            lane = "VALIDATED_RESEARCH_SHADOW"
            rc, output, stderr = fallback_rc, fallback_output, fallback_stderr
        else:
            output = {
                "execution_status": "GENERATION_FAILED",
                "pit_status": "UNKNOWN",
                "predictions": [],
                "error": "production lane was blocked and its explicit research fallback also failed",
                "primary_production_status": primary_status,
                "primary_production_stderr_tail": stderr[-4000:],
                "research_fallback_status": fallback_status,
                "research_fallback_stderr_tail": fallback_stderr[-4000:],
            }
            rc = fallback_rc if fallback_rc != 0 else 1

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
