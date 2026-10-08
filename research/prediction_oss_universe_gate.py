"""Fail-closed policy checks for the user-supplied prediction OSS universe.

The universe is intentionally broader than the external-tool registry. It is
an intake/triage specification, not a list of trusted dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "research" / "prediction_oss_universe_policy.json"

REQUIRED_SCOPE = {
    "data_acquisition",
    "feature_engineering",
    "machine_learning",
    "time_series",
    "anomaly_detection",
    "probabilistic_modeling",
    "simulation",
    "agents",
    "mlops",
    "evaluation",
    "monitoring",
    "visualization",
    "serving",
    "orchestration",
    "storage",
    "retrieval",
    "security",
    "research_automation",
}

def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported prediction OSS universe policy schema")
    if payload.get("source_type") != "USER_SUPPLIED_RANKING":
        raise ValueError("prediction OSS universe must remain explicitly source-attributed")
    return payload

def validate_policy(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    scopes = set(payload.get("scope") or [])
    missing_scopes = sorted(REQUIRED_SCOPE - scopes)
    if missing_scopes:
        errors.append("missing_scope:" + ",".join(missing_scopes))

    truth = payload.get("source_truth_policy") or {}
    for key in (
        "github_repository_identity_must_be_verified",
        "current_star_count_requires_retrieval_timestamp",
        "ranking_is_not_performance_evidence",
        "ranking_is_not_license_evidence",
        "ranking_is_not_pit_evidence",
        "ranking_is_not_production_eligibility",
    ):
        if truth.get(key) is not True:
            errors.append("source_truth_policy_must_be_true:" + key)

    normalization = payload.get("normalization") or {}
    for key in (
        "deduplicate_exact_repository",
        "deduplicate_case_insensitive_repository",
        "resolve_aliases_only_with_explicit_evidence",
        "do_not_guess_repository_from_display_name",
    ):
        if normalization.get(key) is not True:
            errors.append("normalization_policy_must_be_true:" + key)

    fail_closed = payload.get("fail_closed_rules") or {}
    expected = {
        "unknown_repository_identity": "HOLD",
        "unknown_license": "HOLD",
        "unknown_cost": "HOLD",
        "unknown_security_behavior": "HOLD",
        "unknown_pit": "FAIL_CLOSED_FOR_HISTORICAL_USE",
        "unreproduced_method": "RESEARCH_ONLY",
        "ranking_without_local_evidence": "RESEARCH_ONLY",
        "external_performance_claim_without_local_oos": "REJECT_AS_EVIDENCE",
    }
    for key, value in expected.items():
        if fail_closed.get(key) != value:
            errors.append(f"fail_closed_policy_mismatch:{key}")

    ladder = payload.get("promotion_ladder") or []
    required_ladder = [
        "RAW_UNVERIFIED",
        "REPOSITORY_VERIFIED",
        "SOURCE_VERIFIED",
        "LICENSE_VERIFIED",
        "COST_VERIFIED",
        "SECURITY_REVIEWED",
        "PIT_RELEVANCE_REVIEWED",
        "LOCAL_REPRODUCED",
        "OOS",
        "WFO",
        "CALIBRATION",
        "ROBUSTNESS",
        "FROZEN_HOLDOUT",
        "ADOPT_CANDIDATE",
        "ADOPTED",
        "PRODUCTION",
    ]
    position = {name: i for i, name in enumerate(ladder)}
    for left, right in zip(required_ladder, required_ladder[1:]):
        if position.get(left, -1) >= position.get(right, -1):
            errors.append(f"promotion_ladder_invalid:{left}->{right}")
    return errors

def main() -> int:
    payload = load_policy()
    errors = validate_policy(payload)
    if errors:
        for error in errors:
            print("ERROR", error)
        return 1
    print(json.dumps({
        "status": "PASS",
        "universe_id": payload["universe_id"],
        "scope_count": len(payload["scope"]),
        "promotion_states": len(payload["promotion_ladder"]),
        "source_attribution": payload["source_type"],
        "production_eligibility": "NOT_GRANTED_BY_RANKING",
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
