"""Leakage-safe experience learning policy for the baseball system.

This module converts matured prediction experience into a bounded research-only
policy. It is deliberately separate from the Production Champion: it learns
empirical calibration corrections from prior outcomes, evaluates them with a
chronological replay, and emits an auditable policy that a later OOS experiment
may consume.

Core contract:
- only experiences with experience_available_at_utc <= prediction_time are usable;
- current/future outcomes are never visible while generating a prediction;
- sparse strata fall back to broader strata rather than overfitting;
- learned corrections are bounded and renormalized;
- this module never edits production model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE_PATH = ROOT / "data" / "experience" / "snapshot_experience_ledger.csv"
RESULTS = ROOT / "results"


OUTCOME_ORDER = ("HOME_WIN", "DRAW", "AWAY_WIN")
OUTCOME_INDEX = {name: idx for idx, name in enumerate(OUTCOME_ORDER)}

# Most specific -> broadest. Specific strata are used only when enough
# matured cases exist, so this remains conservative on sparse experience.
SCOPE_NAMES = (
    "target_regime_score",
    "target_regime",
    "target_score_regime",
    "target_model",
    "target_tag",
    "target",
    "global",
)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _as_utc_series(frame: pd.DataFrame, field: str) -> pd.Series:
    values = pd.to_datetime(frame[field], errors="coerce", utc=True)
    if values.isna().any():
        raise ValueError(f"{field} contains invalid timestamps")
    return values


def load_experience(path: str | Path = EXPERIENCE_PATH) -> pd.DataFrame:
    """Load and validate the immutable experience snapshot ledger."""
    src = Path(path)
    if not src.exists() or src.stat().st_size == 0:
        return pd.DataFrame()

    frame = pd.read_csv(src)
    required = {
        "prediction_id",
        "target",
        "prediction_cutoff_utc",
        "experience_available_at_utc",
        "actual_outcome",
        "home_win_pct",
        "draw_pct",
        "away_win_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"experience ledger missing required fields: {missing}")

    if frame["prediction_id"].astype(str).duplicated().any():
        raise ValueError("experience ledger contains duplicate prediction_id values")

    frame["prediction_cutoff_utc"] = _as_utc_series(frame, "prediction_cutoff_utc")
    frame["experience_available_at_utc"] = _as_utc_series(
        frame, "experience_available_at_utc"
    )

    # An outcome-derived experience row must not be usable before its own
    # prediction was made. This catches impossible/corrupted timestamps.
    if (
        frame["experience_available_at_utc"]
        < frame["prediction_cutoff_utc"]
    ).any():
        raise ValueError(
            "experience_available_at_utc precedes prediction_cutoff_utc"
        )

    probs = frame[["home_win_pct", "draw_pct", "away_win_pct"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(probs).all()
        or (probs < 0).any()
        or (probs > 100).any()
    ):
        raise ValueError("experience probabilities are invalid")

    sums = probs.sum(axis=1)
    if np.any(np.abs(sums - 100.0) > 0.0003):
        raise ValueError("experience probability rows are not normalized")

    frame["actual_outcome"] = frame["actual_outcome"].astype(str).str.strip()
    unknown = sorted(set(frame["actual_outcome"]) - set(OUTCOME_ORDER))
    if unknown:
        raise ValueError(f"experience contains unknown outcomes: {unknown}")

    frame = frame.sort_values(
        ["prediction_cutoff_utc", "prediction_id"], kind="mergesort"
    ).reset_index(drop=True)
    return frame


def _tags(row: pd.Series) -> list[str]:
    raw = row.get("situation_tags")
    if raw in (None, "", "nan"):
        return []
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = json.loads(str(raw))
        except Exception:
            return []
    if not isinstance(values, list):
        return []
    return sorted({str(x).strip() for x in values if str(x).strip()})


def _scope_keys(row: pd.Series) -> list[tuple[str, str]]:
    target = str(row.get("target") or "").strip()
    regime = str(row.get("regime") or "").strip()
    score_regime = str(row.get("score_regime") or "").strip()
    model = str(row.get("model") or "").strip()

    keys: list[tuple[str, str]] = [("global", "GLOBAL")]
    if target:
        keys.append(("target", target))
    if target and regime:
        keys.append(("target_regime", f"{target}|{regime}"))
    if target and score_regime:
        keys.append(("target_score_regime", f"{target}|{score_regime}"))
    if target and regime and score_regime:
        keys.append(
            ("target_regime_score", f"{target}|{regime}|{score_regime}")
        )
    if target and model:
        keys.append(("target_model", f"{target}|{model}"))
    if target:
        for tag in _tags(row):
            keys.append(("target_tag", f"{target}|{tag}"))
    return keys


def _normalize(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("probability vector must contain three finite values")
    if (p < 0).any():
        raise ValueError("probabilities cannot be negative")
    total = float(p.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("probability vector has invalid sum")
    return p / total


def _bounded_correct(
    probabilities: np.ndarray,
    delta: np.ndarray,
    *,
    strength: float,
    max_log_ratio: float,
) -> np.ndarray:
    p = _normalize(probabilities)
    d = np.asarray(delta, dtype=float)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError("invalid correction vector")
    # Convert additive historical calibration error into a bounded
    # multiplicative correction so a tiny class probability cannot explode.
    target = np.clip(p + d, 1e-6, 1.0)
    ratio = np.clip(target / np.clip(p, 1e-6, 1.0), 0.5, 2.0)
    log_ratio = np.clip(np.log(ratio), -max_log_ratio, max_log_ratio)
    adjusted = p * np.exp(float(strength) * log_ratio)
    return _normalize(adjusted)


class _Accumulator:
    def __init__(self) -> None:
        self.rows = 0
        self.pred_sum = np.zeros(3, dtype=float)
        self.actual_sum = np.zeros(3, dtype=float)
        self.logloss_sum = 0.0
        self.brier_sum = 0.0

    def add(self, row: pd.Series) -> None:
        p = np.asarray(
            [
                float(row["home_win_pct"]),
                float(row["draw_pct"]),
                float(row["away_win_pct"]),
            ],
            dtype=float,
        ) / 100.0
        p = _normalize(p)
        idx = OUTCOME_INDEX[str(row["actual_outcome"])]
        self.rows += 1
        self.pred_sum += p
        self.actual_sum[idx] += 1.0
        self.logloss_sum += float(-np.log(np.clip(p[idx], 1e-12, 1.0)))
        one = np.zeros(3, dtype=float)
        one[idx] = 1.0
        self.brier_sum += float(np.sum((p - one) ** 2))

    def snapshot(self, *, global_actual: np.ndarray, prior_strength: float) -> dict[str, Any]:
        if self.rows <= 0:
            raise ValueError("cannot snapshot empty accumulator")
        mean_pred = self.pred_sum / self.rows
        empirical = self.actual_sum / self.rows
        prior = _normalize(global_actual)
        s = max(float(prior_strength), 0.0)
        shrunk = (self.rows * empirical + s * prior) / (self.rows + s)
        delta = shrunk - mean_pred
        return {
            "rows": int(self.rows),
            "mean_pred": [float(x) for x in mean_pred],
            "empirical_outcome_rate": [float(x) for x in empirical],
            "shrunk_outcome_rate": [float(x) for x in shrunk],
            "delta": [float(x) for x in delta],
            "mean_logloss": float(self.logloss_sum / self.rows),
            "mean_brier": float(self.brier_sum / self.rows),
        }


def build_policy(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp | str | None = None,
    min_group_rows: int = 30,
    prior_strength: float = 50.0,
    correction_strength: float = 0.50,
    max_log_ratio: float = 0.35,
) -> dict[str, Any]:
    """Learn a bounded calibration policy from matured experience only."""
    if min_group_rows < 1:
        raise ValueError("min_group_rows must be >= 1")
    if prior_strength < 0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be finite and non-negative")
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("correction_strength must be in [0,1]")
    if max_log_ratio <= 0 or not np.isfinite(max_log_ratio):
        raise ValueError("max_log_ratio must be positive and finite")

    if frame.empty:
        return {
            "schema_version": 1,
            "status": "NO_EXPERIENCE",
            "cutoff_utc": str(cutoff or _utc_now()),
            "matured_rows": 0,
            "policy": {},
        }

    cutoff_ts = (
        pd.Timestamp(cutoff).tz_convert("UTC")
        if pd.Timestamp(cutoff).tzinfo is not None
        else pd.Timestamp(cutoff).tz_localize("UTC")
    ) if cutoff is not None else _utc_now()

    matured = frame.loc[
        frame["experience_available_at_utc"] <= cutoff_ts
    ].copy()
    matured = matured.loc[
        matured["prediction_cutoff_utc"] < cutoff_ts
    ].copy()

    if matured.empty:
        return {
            "schema_version": 1,
            "status": "NO_MATURED_EXPERIENCE",
            "cutoff_utc": cutoff_ts.isoformat(),
            "matured_rows": 0,
            "policy": {},
        }

    global_actual = np.zeros(3, dtype=float)
    accs: dict[tuple[str, str], _Accumulator] = defaultdict(_Accumulator)

    for _, row in matured.iterrows():
        idx = OUTCOME_INDEX[str(row["actual_outcome"])]
        global_actual[idx] += 1.0
        for key in _scope_keys(row):
            accs[key].add(row)

    policy_rows: dict[str, dict[str, Any]] = {}
    for scope, key in sorted(accs.keys(), key=lambda x: (x[0], x[1])):
        snap = accs[(scope, key)]
        if snap.rows < min_group_rows:
            continue
        item = snap.snapshot(
            global_actual=global_actual,
            prior_strength=prior_strength,
        )
        item.update(
            {
                "scope": scope,
                "key": key,
                "min_group_rows": int(min_group_rows),
                "correction_strength": float(correction_strength),
                "max_log_ratio": float(max_log_ratio),
            }
        )
        policy_rows[f"{scope}:{key}"] = item

    global_key = "global:GLOBAL"
    status = "READY" if global_key in policy_rows else "INSUFFICIENT_EXPERIENCE"
    return {
        "schema_version": 1,
        "status": status,
        "cutoff_utc": cutoff_ts.isoformat(),
        "matured_rows": int(len(matured)),
        "min_group_rows": int(min_group_rows),
        "prior_strength": float(prior_strength),
        "correction_strength": float(correction_strength),
        "max_log_ratio": float(max_log_ratio),
        "policy": policy_rows,
    }


def _candidate_keys(row: pd.Series) -> list[tuple[int, int, str, str]]:
    """Return possible policy strata ranked by specificity, then sample size."""
    target = str(row.get("target") or "").strip()
    regime = str(row.get("regime") or "").strip()
    score_regime = str(row.get("score_regime") or "").strip()
    model = str(row.get("model") or "").strip()

    raw: list[tuple[int, int, str, str]] = []
    for scope, key in _scope_keys(row):
        raw.append((scope_specificity(scope), 0, scope, key))

    # Tag keys are already present in _scope_keys.
    return raw


def scope_specificity(scope: str) -> int:
    return {
        "target_regime_score": 4,
        "target_regime": 3,
        "target_score_regime": 3,
        "target_model": 3,
        "target_tag": 3,
        "target": 2,
        "global": 1,
    }.get(scope, 0)


def select_policy_entry(
    row: pd.Series,
    policy: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    candidates = []
    entries = policy.get("policy", {})
    for specificity, _unused, scope, key in _candidate_keys(row):
        item = entries.get(f"{scope}:{key}")
        if item is None:
            continue
        candidates.append(
            (int(specificity), int(item.get("rows", 0)), scope, key, item)
        )
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    chosen = candidates[0]
    return f"{chosen[2]}:{chosen[3]}", chosen[4]


def apply_policy(
    probabilities: Iterable[float],
    row: pd.Series,
    policy: dict[str, Any],
) -> tuple[np.ndarray, str | None]:
    p = _normalize(np.asarray(list(probabilities), dtype=float))
    key, item = select_policy_entry(row, policy)
    if item is None:
        return p, None
    delta = np.asarray(item["delta"], dtype=float)
    adjusted = _bounded_correct(
        p,
        delta,
        strength=float(item["correction_strength"]),
        max_log_ratio=float(item["max_log_ratio"]),
    )
    return adjusted, key


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for i in range(bins):
        hi = conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1]
        mask = (conf >= edges[i]) & hi
        if not np.any(mask):
            continue
        result += float(mask.sum()) / len(y) * abs(
            float(conf[mask].mean()) - float(correct[mask].mean())
        )
    return float(result)


def _score_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = p.argmax(axis=1)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))
    one = np.zeros_like(p)
    one[np.arange(len(y)), y] = 1.0
    return {
        "rows": int(len(y)),
        "accuracy": float(np.mean(pred == y)) if len(y) else None,
        "logloss": float(np.mean(ll)) if len(y) else None,
        "brier": float(np.mean(np.sum((p - one) ** 2, axis=1))) if len(y) else None,
        "ece": _ece(y, p) if len(y) else None,
    }


def replay(
    frame: pd.DataFrame,
    *,
    min_group_rows: int = 30,
    prior_strength: float = 50.0,
    correction_strength: float = 0.50,
    max_log_ratio: float = 0.35,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Chronological replay: each row sees only already-matured experiences."""
    if frame.empty:
        return {"status": "NO_EXPERIENCE", "rows": 0}, pd.DataFrame()

    work = frame.sort_values(
        ["prediction_cutoff_utc", "prediction_id"], kind="mergesort"
    ).reset_index(drop=True)

    matured: list[int] = []
    used_cursor = 0
    accs: dict[tuple[str, str], _Accumulator] = defaultdict(_Accumulator)
    global_actual = np.zeros(3, dtype=float)
    results: list[dict[str, Any]] = []

    for i, row in work.iterrows():
        cutoff = pd.Timestamp(row["prediction_cutoff_utc"])
        while used_cursor < i:
            prior = work.iloc[used_cursor]
            if pd.Timestamp(prior["experience_available_at_utc"]) > cutoff:
                break
            # A prior prediction whose cutoff is not strictly before the
            # current prediction is never used, even if its experience file
            # happened to be reconciled earlier due to scheduling anomalies.
            if pd.Timestamp(prior["prediction_cutoff_utc"]) < cutoff:
                global_actual[OUTCOME_INDEX[str(prior["actual_outcome"])]] += 1.0
                for key in _scope_keys(prior):
                    accs[key].add(prior)
                matured.append(used_cursor)
            used_cursor += 1

        p = np.asarray(
            [
                float(row["home_win_pct"]),
                float(row["draw_pct"]),
                float(row["away_win_pct"]),
            ],
            dtype=float,
        ) / 100.0
        p = _normalize(p)

        # Build a minimal in-memory policy from accumulators, without touching
        # the current row. This is the critical anti-leakage boundary.
        entries: dict[str, dict[str, Any]] = {}
        for scope, key in sorted(accs.keys(), key=lambda x: (x[0], x[1])):
            if accs[(scope, key)].rows < min_group_rows:
                continue
            item = accs[(scope, key)].snapshot(
                global_actual=global_actual,
                prior_strength=prior_strength,
            )
            item.update(
                {
                    "scope": scope,
                    "key": key,
                    "correction_strength": correction_strength,
                    "max_log_ratio": max_log_ratio,
                }
            )
            entries[f"{scope}:{key}"] = item

        pseudo_policy = {"policy": entries}
        adjusted, source = apply_policy(p, row, pseudo_policy)

        y = OUTCOME_INDEX[str(row["actual_outcome"])]
        results.append(
            {
                "prediction_id": str(row["prediction_id"]),
                "prediction_cutoff_utc": pd.Timestamp(
                    row["prediction_cutoff_utc"]
                ).isoformat(),
                "actual_outcome": str(row["actual_outcome"]),
                "baseline_home_probability": float(p[0]),
                "baseline_draw_probability": float(p[1]),
                "baseline_away_probability": float(p[2]),
                "learned_home_probability": float(adjusted[0]),
                "learned_draw_probability": float(adjusted[1]),
                "learned_away_probability": float(adjusted[2]),
                "policy_source": source,
                "matured_experience_rows": int(len(matured)),
            }
        )

    out = pd.DataFrame(results)
    y = out["actual_outcome"].map(OUTCOME_INDEX).to_numpy(int)
    base = out[
        [
            "baseline_home_probability",
            "baseline_draw_probability",
            "baseline_away_probability",
        ]
    ].to_numpy(float)
    learned = out[
        [
            "learned_home_probability",
            "learned_draw_probability",
            "learned_away_probability",
        ]
    ].to_numpy(float)

    return {
        "status": "REPLAYED",
        "rows": int(len(out)),
        "policy_applied_rows": int(out["policy_source"].notna().sum()),
        "policy_coverage": float(out["policy_source"].notna().mean()),
        "baseline": _score_metrics(y, base),
        "learned": _score_metrics(y, learned),
    }, out


def deterministic_hash(obj: dict[str, Any]) -> str:
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run(
    *,
    input_path: str | Path = EXPERIENCE_PATH,
    cutoff: str | None = None,
    min_group_rows: int = 30,
    prior_strength: float = 50.0,
    correction_strength: float = 0.50,
    max_log_ratio: float = 0.35,
) -> dict[str, Any]:
    frame = load_experience(input_path)
    cutoff_ts = (
        pd.Timestamp(cutoff).tz_convert("UTC")
        if cutoff is not None and pd.Timestamp(cutoff).tzinfo is not None
        else pd.Timestamp(cutoff).tz_localize("UTC")
        if cutoff is not None
        else _utc_now()
    )

    policy = build_policy(
        frame,
        cutoff=cutoff_ts,
        min_group_rows=min_group_rows,
        prior_strength=prior_strength,
        correction_strength=correction_strength,
        max_log_ratio=max_log_ratio,
    )
    replay_summary, ledger = replay(
        frame,
        min_group_rows=min_group_rows,
        prior_strength=prior_strength,
        correction_strength=correction_strength,
        max_log_ratio=max_log_ratio,
    )

    stable = {
        "schema_version": 1,
        "status": policy["status"],
        "cutoff_utc": policy["cutoff_utc"],
        "matured_rows": policy["matured_rows"],
        "min_group_rows": policy.get("min_group_rows", min_group_rows),
        "prior_strength": policy.get("prior_strength", prior_strength),
        "correction_strength": policy.get(
            "correction_strength", correction_strength
        ),
        "max_log_ratio": policy.get("max_log_ratio", max_log_ratio),
        "policy": policy.get("policy", {}),
    }
    policy_hash = deterministic_hash(stable)

    result = {
        **stable,
        "policy_hash": policy_hash,
        "replay": replay_summary,
        "replay_contract": {
            "chronological_prediction_cutoff": True,
            "experience_available_at_le_prediction_time": True,
            "current_row_excluded_from_learning": True,
            "future_outcomes_excluded": True,
            "production_model_modified": False,
            "promotion_enabled": False,
        },
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "experience_learning_policy.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not ledger.empty:
        ledger.to_csv(
            RESULTS / "experience_learning_replay.csv",
            index=False,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(EXPERIENCE_PATH))
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--min-group-rows", type=int, default=30)
    parser.add_argument("--prior-strength", type=float, default=50.0)
    parser.add_argument("--correction-strength", type=float, default=0.50)
    parser.add_argument("--max-log-ratio", type=float, default=0.35)
    args = parser.parse_args()
    result = run(
        input_path=args.input,
        cutoff=args.cutoff,
        min_group_rows=args.min_group_rows,
        prior_strength=args.prior_strength,
        correction_strength=args.correction_strength,
        max_log_ratio=args.max_log_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
