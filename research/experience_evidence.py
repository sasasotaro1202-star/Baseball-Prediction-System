"""Evidence metadata for post-game experience artifacts.

Experience artifacts are historical reconciliation evidence.  They are useful for
future research only when their prediction-time provenance is preserved and
must never be presented as current-model performance without a separate
chronological OOS/holdout evaluation.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def source_head() -> str:
    """Return the commit that generated the artifact when provenance is available."""
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


def _unique_values(frame: Any, column: str) -> list[str]:
    if frame is None or not hasattr(frame, "columns") or column not in frame.columns:
        return []
    values: set[str] = set()
    for raw in frame[column].tolist():
        if raw is None:
            continue
        value = str(raw).strip()
        if value and value not in {"nan", "NaN", "None", "null"}:
            values.add(value)
    return sorted(values)


def build_experience_evidence(frame: Any) -> dict[str, Any]:
    """Describe the evidence class without implying current-model verification."""
    return {
        "scope": "HISTORICAL_POSTGAME_EXPERIENCE",
        "performance_status": "HISTORICAL_ONLY_NOT_CURRENT_MODEL_VERIFICATION",
        "generated_from_commit": source_head(),
        "prediction_source_commits": _unique_values(frame, "git_commit"),
        "prediction_models": _unique_values(frame, "model"),
        "prediction_model_versions": _unique_values(frame, "model_version"),
        "prediction_feature_versions": _unique_values(frame, "feature_version"),
        "prediction_calibration_versions": _unique_values(frame, "calibration_version"),
        "prediction_targets": _unique_values(frame, "target"),
        "comparison_to_current_production": {
            "status": "UNVERIFIED",
            "current_runtime_equivalence": "NOT_ESTABLISHED",
            "reason": (
                "Post-game experience is historical reconciliation evidence, not "
                "a chronological OOS/frozen-holdout evaluation of the currently "
                "registered production runtime."
            ),
        },
    }
