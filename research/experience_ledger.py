"""Accumulate production prediction experience and post-game performance.

This module keeps prediction snapshots immutable-ish at the event/cutoff level,
then reconciles them with official completed NPB results. It produces a durable
experience ledger that can be used by future forward research only after the
result became available.

No model is modified by this module.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.npb_official_results import _fetch_month
from research.experience_dimensions import add_dimensions

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "data" / "experience"
PRED_DIR = EXPERIENCE / "predictions"
RESULT_DIR = EXPERIENCE / "official_results"
SUMMARY_PATH = EXPERIENCE / "experience_summary.json"
LEDGER_PATH = EXPERIENCE / "experience_ledger.csv"
LEDGER_JSONL = EXPERIENCE / "experience_ledger.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(v: Any) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("non-finite value")
    return x


def _normalize_percentage_rows(
    values: pd.DataFrame,
    *,
    label: str,
    row_ids: pd.Series | None = None,
) -> np.ndarray:
    """Validate percentage probabilities and normalize only serialization rounding."""
    arr = values.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"{label} probability matrix must have 2 or 3 columns")
    if not np.isfinite(arr).all() or (arr < 0.0).any() or (arr > 100.0).any():
        raise ValueError(f"invalid {label} probabilities")
    sums = arr.sum(axis=1)
    rounding_tolerance_pct = 0.0003
    bad = np.abs(sums - 100.0) > rounding_tolerance_pct
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        ident = row_ids.iloc[idx] if row_ids is not None and len(row_ids) > idx else idx
        raise ValueError(
            f"invalid {label} probabilities: row={ident!r} sum_pct={sums[idx]:.8f}"
        )
    return arr / sums[:, None]


def _revision_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize forecast revisions without treating stability as accuracy."""
    required = {
        "revision_status",
        "revision_l1_pct_points",
        "revision_max_abs_pct_points",
        "revision_outcome_changed",
    }
    if not required.issubset(frame.columns) or frame.empty:
        return {"status": "UNAVAILABLE", "eligible_rows": 0}
    work = frame.loc[frame["revision_status"].astype(str) == "REVISED"].copy()
    if work.empty:
        return {
            "status": "MEASURED",
            "eligible_rows": int(len(frame)),
            "revised_rows": 0,
            "revision_rate": 0.0,
            "outcome_reversal_rows": 0,
            "outcome_reversal_rate": 0.0,
            "mean_l1_pct_points": 0.0,
            "median_l1_pct_points": 0.0,
            "maximum_l1_pct_points": 0.0,
            "maximum_single_class_change_pct_points": 0.0,
        }
    l1 = pd.to_numeric(work["revision_l1_pct_points"], errors="coerce")
    max_abs = pd.to_numeric(work["revision_max_abs_pct_points"], errors="coerce")
    valid = l1.notna() & max_abs.notna()
    if not bool(valid.any()):
        return {
            "status": "UNAVAILABLE",
            "eligible_rows": int(len(frame)),
            "revised_rows": int(len(work)),
        }
    l1 = l1.loc[valid]
    max_abs = max_abs.loc[valid]
    reversals = work.loc[valid, "revision_outcome_changed"].astype(bool)
    return {
        "status": "MEASURED",
        "eligible_rows": int(len(frame)),
        "revised_rows": int(len(work)),
        "revision_rate": float(len(work) / max(1, len(frame))),
        "outcome_reversal_rows": int(reversals.sum()),
        "outcome_reversal_rate": float(reversals.mean()),
        "mean_l1_pct_points": float(l1.mean()),
        "median_l1_pct_points": float(l1.median()),
        "maximum_l1_pct_points": float(l1.max()),
        "maximum_single_class_change_pct_points": float(max_abs.max()),
    }


def _timing_30m_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Measure strict 30-minute pregame timing without changing headline skill metrics."""
    required = {"datetime_jst", "prediction_generated_at", "prediction_cutoff_utc"}
    if not required.issubset(frame.columns) or frame.empty:
        return {"status": "UNAVAILABLE", "eligible_rows": 0}
    game_time = pd.to_datetime(frame["datetime_jst"], utc=True, errors="coerce")
    generated = pd.to_datetime(frame["prediction_generated_at"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(frame["prediction_cutoff_utc"], utc=True, errors="coerce")
    valid = game_time.notna() & generated.notna() & cutoff.notna()
    if not bool(valid.any()):
        return {"status": "UNAVAILABLE", "eligible_rows": 0}
    game_time = game_time.loc[valid]
    generated = generated.loc[valid]
    cutoff = cutoff.loc[valid]
    required_cutoff = game_time - pd.Timedelta(minutes=30)
    on_time = generated <= required_cutoff
    scheduled_lead = (game_time - cutoff).dt.total_seconds() / 60.0
    actual_lead = (game_time - generated).dt.total_seconds() / 60.0
    return {
        "status": "MEASURED",
        "eligible_rows": int(len(game_time)),
        "on_time_rows": int(on_time.sum()),
        "late_rows": int((~on_time).sum()),
        "compliance_rate": float(on_time.mean()),
        "minimum_actual_lead_minutes": float(actual_lead.min()),
        "median_actual_lead_minutes": float(actual_lead.median()),
        "maximum_actual_lead_minutes": float(actual_lead.max()),
        "scheduled_cutoff_minimum_lead_minutes": float(scheduled_lead.min()),
        "scheduled_cutoff_below_30m_rows": int((scheduled_lead < 30.0).sum()),
    }


def _validate_prediction_probability_contract(row: Mapping[str, Any]) -> None:
    """Fail closed on malformed serialized probability distributions."""
    try:
        values = np.asarray(
            [
                float(row["home_win_pct"]),
                float(row["draw_pct"]),
                float(row["away_win_pct"]),
            ],
            dtype=float,
        )
        low_high = np.asarray(
            [float(row["low_pct"]), float(row["high_pct"])],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("prediction snapshot probability fields are invalid") from exc

    for label, arr in (("win", values), ("Low/High", low_high)):
        if (
            not np.isfinite(arr).all()
            or (arr < 0.0).any()
            or (arr > 100.0).any()
            or abs(float(arr.sum()) - 100.0) > 0.0003
        ):
            raise ValueError(
                f"prediction snapshot {label} probabilities are not a valid normalized distribution"
            )


def _validate_prediction_time_contract(row: dict[str, Any]) -> None:
    """Fail closed on prediction snapshots with ambiguous or impossible timing."""
    required = (
        "game_id",
        "datetime_jst",
        "prediction_cutoff_utc",
        "prediction_generated_at",
        "starter_evidence_observed_at_utc",
        "pit_status",
    )
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise ValueError(
            "prediction snapshot missing PIT timing/provenance fields: "
            + ", ".join(missing)
        )

    try:
        game_time = pd.Timestamp(row["datetime_jst"])
        cutoff = pd.Timestamp(row["prediction_cutoff_utc"])
        generated = pd.Timestamp(row["prediction_generated_at"])
        observed = pd.Timestamp(row["starter_evidence_observed_at_utc"])
    except Exception as exc:
        raise ValueError("prediction snapshot contains invalid PIT timestamps") from exc

    if any(ts.tzinfo is None for ts in (game_time, cutoff, generated, observed)):
        raise ValueError("prediction snapshot PIT timestamps must be timezone-aware")

    game_time = game_time.tz_convert("UTC")
    cutoff = cutoff.tz_convert("UTC")
    generated = generated.tz_convert("UTC")
    observed = observed.tz_convert("UTC")

    if not cutoff < game_time:
        raise ValueError("prediction snapshot information cutoff is not pregame")
    if not cutoff <= generated < game_time:
        raise ValueError(
            "prediction snapshot generation time is inconsistent with information cutoff"
        )
    if observed > cutoff:
        raise ValueError(
            "prediction snapshot starter evidence was observed after information cutoff"
        )
    if str(row["pit_status"]).upper() != "PASS":
        raise ValueError("prediction snapshot is not PIT PASS")


def _revision_metadata(
    record: dict[str, Any],
    existing: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare a new snapshot only with an earlier snapshot of the same game."""
    current_cutoff = pd.Timestamp(record["prediction_cutoff_utc"])
    if current_cutoff.tzinfo is None:
        raise ValueError("prediction cutoff must be timezone-aware")

    prior: tuple[pd.Timestamp, dict[str, Any]] | None = None
    for candidate in existing.values():
        if str(candidate.get("game_id", "")) != str(record.get("game_id", "")):
            continue
        # Legacy scheduled-cutoff rows remain preserved for audit/history, but
        # are not valid ancestors for revision analysis because their cutoff
        # was a planned deadline rather than an observed information time.
        if _is_legacy_scheduled_cutoff(candidate):
            continue
        raw_cutoff = candidate.get("prediction_cutoff_utc")
        if raw_cutoff in (None, ""):
            continue
        try:
            cutoff = pd.Timestamp(raw_cutoff)
        except Exception:
            continue
        if cutoff.tzinfo is None or cutoff >= current_cutoff:
            continue
        if prior is None or cutoff > prior[0]:
            prior = (cutoff, candidate)

    if prior is None:
        return {
            "revision_status": "INITIAL",
            "revision_previous_prediction_id": None,
            "revision_l1_pct_points": None,
            "revision_max_abs_pct_points": None,
            "revision_outcome_changed": False,
        }

    previous = prior[1]
    fields = ("home_win_pct", "draw_pct", "away_win_pct")
    try:
        current = np.asarray([float(record[field]) for field in fields], dtype=float)
        old = np.asarray([float(previous[field]) for field in fields], dtype=float)
    except (KeyError, TypeError, ValueError):
        return {
            "revision_status": "NO_COMPARABLE_PREVIOUS",
            "revision_previous_prediction_id": str(previous.get("prediction_id") or "") or None,
            "revision_l1_pct_points": None,
            "revision_max_abs_pct_points": None,
            "revision_outcome_changed": False,
        }

    if (
        not np.isfinite(current).all()
        or not np.isfinite(old).all()
        or (current < 0).any()
        or (old < 0).any()
    ):
        raise ValueError("invalid probabilities for revision comparison")

    delta = current - old
    previous_winner = int(np.argmax(old))
    current_winner = int(np.argmax(current))
    return {
        "revision_status": "REVISED",
        "revision_previous_prediction_id": str(previous.get("prediction_id") or "") or None,
        "revision_l1_pct_points": float(np.abs(delta).sum()),
        "revision_max_abs_pct_points": float(np.abs(delta).max()),
        "revision_outcome_changed": bool(previous_winner != current_winner),
    }


def _is_legacy_scheduled_cutoff(row: Mapping[str, Any]) -> bool:
    """Identify pre-v18 snapshots whose cutoff was a planned 30m deadline."""
    raw_generated = row.get("prediction_generated_at")
    raw_cutoff = row.get("prediction_cutoff_utc")
    if raw_generated in (None, "") or raw_cutoff in (None, ""):
        return False
    try:
        generated = pd.Timestamp(raw_generated)
        cutoff = pd.Timestamp(raw_cutoff)
    except Exception:
        return False
    if generated.tzinfo is None or cutoff.tzinfo is None:
        return False
    # Current contract records actual observation time plus explicit preferred
    # deadline metadata. A pre-v18 row lacking those fields and generated before
    # its stored cutoff is historical scheduled-cutoff metadata, not a verified
    # actual observation cutoff. Preserve it but never reuse it as PIT evidence.
    modern_markers = (
        "lead_minutes_at_generation",
        "prediction_deadline_utc",
        "preferred_prediction_cutoff_utc",
    )
    # DataFrame/dict normalization can materialize absent columns as NaN.
    # Treat a marker as present only when it carries an actual non-null value;
    # a mere key presence must not reclassify a legacy row as modern PIT data.
    for key in modern_markers:
        value = row.get(key)
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        if str(value).strip():
            return False
    return generated < cutoff


def _prediction_id(row: dict[str, Any]) -> str:
    raw = f"{row['game_id']}|{row['prediction_cutoff_utc']}|{row.get('git_commit','unknown')}"
    import hashlib
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _iter_prediction_files() -> list[Path]:
    return sorted(PRED_DIR.glob("*.jsonl"))


def archive_production_output(input_json: str | Path, *, run_id: str | None = None) -> dict[str, int]:
    """Archive every production snapshot, preserving every prediction cutoff."""
    src = Path(input_json)
    obj = json.loads(src.read_text(encoding="utf-8"))
    if obj.get("execution_status") != "EXECUTED":
        return {"archived": 0, "skipped": len(obj.get("predictions", [])), "updated": 0}
    target_date = str(obj.get("target_date"))
    predictions = obj.get("predictions", [])
    if not target_date or not isinstance(predictions, list):
        raise ValueError("invalid production output contract")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{target_date}.jsonl"

    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("prediction_id") or "")
            if key:
                existing[key] = row

    archived = updated = 0
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError("prediction row must be an object")
        if not pred.get("game_id") or not pred.get("prediction_cutoff_utc"):
            raise ValueError("prediction row missing game_id or cutoff")
        record = dict(pred)
        legacy_scheduled_cutoff = _is_legacy_scheduled_cutoff(record)
        if not legacy_scheduled_cutoff:
            _validate_prediction_time_contract(record)
        else:
            # Preserve pre-v18 scheduled-cutoff rows for audit/history. They are
            # quarantined from Experience reuse later by _load_predictions().
            print(json.dumps({
                "event": "EXPERIENCE_LEGACY_TIMING_QUARANTINE",
                "game_id": str(record.get("game_id") or ""),
                "reason": "pre-v18 scheduled cutoff is not an actual observed prediction cutoff",
            }, ensure_ascii=False))
        _validate_prediction_probability_contract(record)
        # Canonical target identity is preserved for per-target metrics. Prefer
        # an explicit target/competition/league field; the NPB production
        # archive supplies NPB when these fields are absent.
        target = (
            record.get("target")
            or record.get("competition_id")
            or record.get("league")
            or obj.get("target")
            or obj.get("competition_id")
            or obj.get("league")
            or "NPB"
        )
        record["target"] = str(target)
        record["competition_id"] = str(record.get("competition_id") or target)
        record["prediction_id"] = _prediction_id(record)
        record["source_run_id"] = str(run_id) if run_id is not None else None
        revision = _revision_metadata(record, existing)
        record.update(revision)
        record["archived_at_utc"] = _utc_now()
        key = record["prediction_id"]
        if key not in existing:
            existing[key] = record
            archived += 1
        else:
            updated += 1

    rows = sorted(
        existing.values(),
        key=lambda x: (str(x.get("datetime_jst", "")), str(x.get("prediction_cutoff_utc", "")), str(x.get("game_id", "")))
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    tmp.replace(path)
    return {"archived": archived, "skipped": 0, "updated": updated}


def _load_predictions() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _iter_prediction_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "prediction_id" not in df:
        df["prediction_id"] = df.apply(lambda r: _prediction_id(r.to_dict()), axis=1)

    # Re-validate the immutable archive before any result matching or training reuse.
    # Pre-v18 scheduled-cutoff snapshots are quarantined as unverified legacy
    # history instead of being silently accepted or rewriting the raw archive.
    records = df.to_dict("records")
    legacy_rows = {
        idx for idx, row in enumerate(records) if _is_legacy_scheduled_cutoff(row)
    }
    for idx, row in enumerate(records):
        if idx in legacy_rows:
            continue
        _validate_prediction_time_contract(row)
        _validate_prediction_probability_contract(row)
    if legacy_rows:
        print(json.dumps({
            "event": "EXPERIENCE_LEGACY_TIMING_QUARANTINE",
            "rows": len(legacy_rows),
            "reason": "pre-v18 scheduled cutoff is not an actual observed prediction cutoff",
        }, ensure_ascii=False))

    df["prediction_cutoff_utc"] = pd.to_datetime(df["prediction_cutoff_utc"], utc=True, errors="coerce")
    df["datetime_jst"] = pd.to_datetime(df["datetime_jst"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_cutoff_utc", "datetime_jst", "game_id"])
    # Only pregame predictions are valid experience. Anything made at or after
    # first pitch is excluded so late re-runs cannot masquerade as pregame skill.
    df = df.loc[
        (~df.index.isin(legacy_rows))
        & (
            df["prediction_cutoff_utc"]
            < df["datetime_jst"].dt.tz_convert("UTC")
        )
    ].copy()
    df = df.sort_values(["game_id", "prediction_cutoff_utc", "prediction_id"])
    df = df.drop_duplicates("game_id", keep="last").reset_index(drop=True)
    return df


def _should_refresh_result_cache(
    year: int,
    month: int,
    *,
    now: pd.Timestamp | None = None,
) -> bool:
    """Refresh the currently active UTC month; reuse closed-month caches."""
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    return int(year) == int(current.year) and int(month) == int(current.month)


def _result_cache_path(year: int, month: int) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{year:04d}-{month:02d}.csv"


def _load_cached_results(dates: list[pd.Timestamp]) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame()
    needed = sorted({(int(d.year), int(d.month)) for d in dates})
    chunks: list[pd.DataFrame] = []
    columns = ["date", "home", "away", "home_score", "away_score", "source_url"]
    for year, month in needed:
        path = _result_cache_path(year, month)
        got: pd.DataFrame | None = None

        # The active month is mutable: new games can finish after a
        # previously non-empty cache was written. Refresh it every run.
        # Closed months remain cached for efficient, reproducible reuse.
        refresh = _should_refresh_result_cache(year, month)
        if not refresh and path.exists() and path.stat().st_size > 0:
            try:
                got = pd.read_csv(path)
            except Exception:
                path.unlink(missing_ok=True)
                got = None

        if refresh or got is None or got.empty:
            rows = _fetch_month(year, month)
            refreshed = pd.DataFrame(rows, columns=columns)
            refreshed.to_csv(path, index=False)
            got = refreshed

        chunks.append(got)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def _expand_weights(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("classification_regime_model_weights", "score_regime_model_weights"):
        raw = row.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict):
            prefix = "classification_weight_" if key.startswith("classification") else "score_weight_"
            for name, value in raw.items():
                try:
                    out[f"{prefix}{name}"] = float(value)
                except (TypeError, ValueError):
                    continue
    return out


def _multiclass_ece(probabilities: np.ndarray, y_true: np.ndarray, *, bins: int = 10) -> float:
    """Compute max-probability multiclass ECE without refitting or reordering."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 2 or y.ndim != 1 or len(p) != len(y) or len(y) == 0:
        raise ValueError("invalid ECE inputs")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid ECE probabilities")
    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        upper_ok = confidence < hi if i < bins - 1 else confidence <= hi
        mask = (confidence >= lo) & upper_ok
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / total) * abs(
            float(confidence[mask].mean()) - float(correct[mask].mean())
        )
    return float(ece)


def _binary_ece(probabilities: np.ndarray, y_true: np.ndarray, *, bins: int = 10) -> float:
    """Compute calibration error for one binary target without refitting."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or len(y) == 0:
        raise ValueError("invalid binary ECE inputs")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid binary ECE probabilities")
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        upper_ok = p < hi if i < bins - 1 else p <= hi
        mask = (p >= lo) & upper_ok
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / total) * abs(
            float(p[mask].mean()) - float(y[mask].mean())
        )
    return float(ece)


def _prediction_target_metrics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Evaluate win, Low/High, and exact-score targets independently."""
    out: dict[str, dict[str, Any]] = {}

    def _as_unit_interval(values: Any, *, label: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            scale = float(np.nanmax(arr)) if arr.size else 0.0
        else:
            scale = float(np.nanmax(arr)) if arr.size else 0.0
        if not np.isfinite(arr).all() or (arr < 0.0).any() or scale > 100.0:
            raise ValueError(f"invalid {label} probabilities")
        # Experience sources may store percentage points (0..100) or
        # normalized probabilities (0..1). Normalize once, never twice.
        if scale > 1.0:
            arr = arr / 100.0
        if not np.isfinite(arr).all() or (arr > 1.0).any():
            raise ValueError(f"invalid {label} probabilities after normalization")
        return arr

    required = {"home_win_pct", "draw_pct", "away_win_pct", "actual_outcome"}
    if required.issubset(frame.columns) and len(frame):
        probs = _as_unit_interval(
            frame[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
            label="win",
        )
        y = frame["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int)
        pred = probs.argmax(axis=1)
        selective: dict[str, dict[str, Any]] = {}
        confidence = probs.max(axis=1)
        for threshold in (0.60, 0.70, 0.80, 0.90):
            mask = confidence >= threshold
            key = f"selective_{threshold:.2f}"
            selective[key] = {
                "coverage": float(mask.mean()),
                "n": int(mask.sum()),
                "accuracy": float((pred[mask] == y[mask]).mean()) if mask.any() else None,
            }
        out["win_3way"] = {
            "rows": int(len(frame)),
            "accuracy": float((pred == y).mean()),
            "logloss": float(-np.log(np.clip(probs[np.arange(len(frame)), y], 1e-12, 1.0)).mean()),
            "brier": float(np.mean(np.sum((probs - np.eye(3)[y]) ** 2, axis=1))),
            "ece": _multiclass_ece(probs, y),
            **selective,
        }
    else:
        out["win_3way"] = {"rows": 0, "status": "UNAVAILABLE"}

    required = {"high_pct", "low_high_actual"}
    if required.issubset(frame.columns) and len(frame):
        p_high = _as_unit_interval(frame["high_pct"].to_numpy(float), label="Low/High")
        y_high = frame["low_high_actual"].to_numpy(int)
        pred_high = (p_high >= 0.5).astype(int)
        out["low_high"] = {
            "rows": int(len(frame)),
            "accuracy": float((pred_high == y_high).mean()),
            "logloss": float(-np.mean(y_high * np.log(np.clip(p_high, 1e-12, 1.0)) + (1 - y_high) * np.log(np.clip(1 - p_high, 1e-12, 1.0)))),
            "brier": float(np.mean((p_high - y_high) ** 2)),
            "ece": _binary_ece(p_high, y_high),
        }
    else:
        out["low_high"] = {"rows": 0, "status": "UNAVAILABLE"}

    required = {"top1_exact_hit", "top4_hit", "score_mae"}
    if required.issubset(frame.columns) and len(frame):
        out["exact_score"] = {
            "rows": int(len(frame)),
            "top1_exact_hit_rate": float(frame["top1_exact_hit"].mean()),
            "top4_exact_hit_rate": float(frame["top4_hit"].mean()),
            "score_mae": float(frame["score_mae"].mean()),
        }
    else:
        out["exact_score"] = {"rows": 0, "status": "UNAVAILABLE"}
    return out


def _parse_top4(row: Any) -> list[tuple[str, float]]:
    if isinstance(row, list):
        return [(str(x.get("score")), float(x.get("prob_pct", 0.0))) for x in row if isinstance(x, dict)]
    if isinstance(row, str):
        try:
            return _parse_top4(json.loads(row))
        except Exception:
            return []
    return []


def reconcile() -> dict[str, Any]:
    pred = _load_predictions()
    if pred.empty:
        payload = {
            "generated_at_utc": _utc_now(),
            "status": "NO_PREDICTIONS",
            "matched_rows": 0,
            "new_experiences": 0,
        }
        EXPERIENCE.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    dates = [d for d in pred["datetime_jst"] if pd.notna(d)]
    results = _load_cached_results(dates)
    if results.empty:
        payload = {
            "generated_at_utc": _utc_now(),
            "status": "NO_COMPLETED_RESULTS",
            "matched_rows": 0,
            "new_experiences": 0,
            "prediction_rows": int(len(pred)),
        }
        EXPERIENCE.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    pred = pred.copy()
    pred["date_key"] = pred["datetime_jst"].dt.tz_convert("Asia/Tokyo").dt.strftime("%Y-%m-%d")
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    if results["date"].isna().any():
        raise RuntimeError("official experience results contain an invalid date")
    results["date_key"] = results["date"].dt.strftime("%Y-%m-%d")
    for col in ("home", "away"):
        pred[f"{col}_key"] = pred[col].astype(str)
        results[f"{col}_key"] = results[col].astype(str)

    merged = pred.merge(
        results[["date_key", "home_key", "away_key", "home_score", "away_score", "source_url"]],
        on=["date_key", "home_key", "away_key"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_result"),
    )
    matched = merged.dropna(subset=["home_score", "away_score"]).copy()
    if matched.empty:
        return {
            "generated_at_utc": _utc_now(),
            "status": "NO_COMPLETED_RESULTS",
            "matched_rows": 0,
            "new_experiences": 0,
            "prediction_rows": int(len(pred)),
        }

    matched["actual_home_score"] = matched["home_score"].astype(int)
    matched["actual_away_score"] = matched["away_score"].astype(int)
    matched["actual_total_runs"] = matched["actual_home_score"] + matched["actual_away_score"]
    matched["actual_outcome"] = np.where(
        matched["actual_home_score"] > matched["actual_away_score"], "HOME_WIN",
        np.where(matched["actual_home_score"] == matched["actual_away_score"], "DRAW", "AWAY_WIN")
    )
    prob_cols = ["home_win_pct", "draw_pct", "away_win_pct"]
    matched[prob_cols] = _normalize_percentage_rows(
        matched[prob_cols],
        label="win",
        row_ids=matched["game_id"],
    )
    matched["predicted_outcome"] = np.array([
        ["HOME_WIN", "DRAW", "AWAY_WIN"][int(np.argmax(r))]
        for r in matched[prob_cols].to_numpy(float)
    ])
    matched["outcome_correct"] = (matched["predicted_outcome"] == matched["actual_outcome"]).astype(int)

    y_idx = matched["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int)
    pp = matched[prob_cols].to_numpy(float)
    matched["logloss"] = -np.log(np.clip(pp[np.arange(len(pp)), y_idx], 1e-12, 1.0))
    matched["brier"] = np.sum((pp - np.eye(3)[y_idx]) ** 2, axis=1)
    matched["home_probability_error"] = pp[:, 0] - (y_idx == 0)
    matched["draw_probability_error"] = pp[:, 1] - (y_idx == 1)
    matched["away_probability_error"] = pp[:, 2] - (y_idx == 2)

    low_high_cols = ["low_pct", "high_pct"]
    low_high_probs = _normalize_percentage_rows(
        matched[low_high_cols],
        label="Low/High",
        row_ids=matched["game_id"],
    )
    matched["low_probability"] = low_high_probs[:, 0]
    matched["high_probability"] = low_high_probs[:, 1]
    matched["low_high_actual"] = (matched["actual_total_runs"] >= 7).astype(int)
    matched["low_high_predicted"] = (matched["high_probability"] >= 0.5).astype(int)
    matched["low_high_correct"] = (matched["low_high_predicted"] == matched["low_high_actual"]).astype(int)

    def score_eval(row: pd.Series) -> tuple[float, int, int]:
        picks = _parse_top4(row.get("top4_exact_scores"))
        actual = f"{int(row['actual_home_score'])}-{int(row['actual_away_score'])}"
        top1 = picks[0][0] if picks else None
        hit = int(actual in {x[0] for x in picks})
        top1_hit = int(actual == top1) if top1 else 0
        # MAE of the 4-candidate mean is intentionally not claimed as exact-score accuracy.
        pred_lambda_h = float(row.get("lambda_home", 0.0))
        pred_lambda_a = float(row.get("lambda_away", 0.0))
        mae = (abs(pred_lambda_h - row["actual_home_score"]) + abs(pred_lambda_a - row["actual_away_score"])) / 2.0
        return float(mae), hit, top1_hit

    score_vals = matched.apply(score_eval, axis=1, result_type="expand")
    score_vals.columns = ["score_mae", "top4_hit", "top1_exact_hit"]
    matched = pd.concat([matched.reset_index(drop=True), score_vals.reset_index(drop=True)], axis=1)

    expanded_weight_rows = matched.apply(lambda r: _expand_weights(r.to_dict()), axis=1).tolist()
    for row in expanded_weight_rows:
        for k, v in row.items():
            # Add lazily; later cast below.
            pass
    weight_keys = sorted({k for row in expanded_weight_rows for k in row})
    for k in weight_keys:
        matched[k] = [row.get(k, np.nan) for row in expanded_weight_rows]
    matched["dominant_classification_expert"] = matched.apply(
        lambda r: max(
            ((k, r[k]) for k in weight_keys if k.startswith("classification_weight_") and pd.notna(r[k])),
            key=lambda x: x[1],
            default=(None, np.nan),
        )[0],
        axis=1,
    )

    # Capture the reconciliation observation time once. Existing ledger rows
    # retain their first-seen availability timestamp below so reruns do not
    # move the PIT boundary forward.
    reconciled_at_utc = _utc_now()
    matched["experience_available_at_utc"] = reconciled_at_utc

    matched["prediction_actual_lead_minutes"] = (
        pd.to_datetime(matched["datetime_jst"], utc=True, errors="coerce")
        - pd.to_datetime(matched["prediction_generated_at"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    matched["prediction_30m_on_time"] = matched["prediction_actual_lead_minutes"] >= 30.0
    matched["scheduled_cutoff_lead_minutes"] = (
        pd.to_datetime(matched["datetime_jst"], utc=True, errors="coerce")
        - pd.to_datetime(matched["prediction_cutoff_utc"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0

    keep = [
        "prediction_id", "game_id", "target", "competition_id", "date_key", "datetime_jst", "prediction_cutoff_utc",
        "prediction_generated_at", "prediction_source", "prediction_schedule", "prediction_target_lead_minutes",
        "home", "away", "home_starter", "away_starter",
        "regime", "score_regime", "model", "situation_tags",
        "home_win_pct", "draw_pct", "away_win_pct", "predicted_outcome",
        "actual_outcome", "outcome_correct", "logloss", "brier",
        "home_probability_error", "draw_probability_error", "away_probability_error",
        "prediction_actual_lead_minutes", "prediction_30m_on_time", "scheduled_cutoff_lead_minutes",
        "revision_status", "revision_previous_prediction_id", "revision_l1_pct_points",
        "revision_max_abs_pct_points", "revision_outcome_changed",
        "low_pct", "high_pct", "low_high_actual", "low_high_correct",
        "score_mae", "top1_exact_hit", "top4_hit",
        "lambda_home", "lambda_away", "shared_lambda",
        "dominant_classification_expert", "experience_available_at_utc", "source_url",
    ] + weight_keys
    keep = [c for c in keep if c in matched.columns]
    experience = matched[keep].copy()

    # Upsert by prediction_id so reruns never duplicate an experience case.
    existing = pd.DataFrame()
    if LEDGER_PATH.exists() and LEDGER_PATH.stat().st_size > 0:
        existing = pd.read_csv(LEDGER_PATH)
    existing_ids = set(existing["prediction_id"].astype(str)) if "prediction_id" in existing.columns else set()

    # Experience availability is the first reconciliation observation for a
    # prediction, not the wall-clock time of the latest rerun. Restore the
    # persisted value at the final ledger-row boundary so repeated runs cannot
    # move the chronological replay boundary forward.
    if not existing.empty and "experience_available_at_utc" in existing.columns:
        prior_available = (
            existing[["prediction_id", "experience_available_at_utc"]]
            .dropna(subset=["prediction_id"])
            .assign(prediction_id=lambda frame: frame["prediction_id"].astype(str))
            .drop_duplicates("prediction_id", keep="first")
            .set_index("prediction_id")["experience_available_at_utc"]
            .to_dict()
        )
        for idx, prediction_id in enumerate(experience["prediction_id"].astype(str)):
            previous_available = prior_available.get(prediction_id)
            if previous_available is None or str(previous_available).strip() in {"", "nan", "NaT"}:
                continue
            try:
                parsed_available = pd.Timestamp(previous_available)
            except Exception as exc:
                raise ValueError("existing experience availability timestamp is invalid") from exc
            if parsed_available.tzinfo is None:
                raise ValueError("existing experience availability timestamp must be timezone-aware")
            experience.loc[experience.index[idx], "experience_available_at_utc"] = parsed_available.tz_convert("UTC").isoformat()




    if not existing.empty:
        if "target" not in existing.columns:
            existing["target"] = existing.get("league", "NPB")
        existing["target"] = existing["target"].fillna(existing.get("league", "NPB")).fillna("NPB").astype(str)
        if "competition_id" not in existing.columns:
            existing["competition_id"] = existing["target"]
        existing["competition_id"] = existing["competition_id"].fillna(existing["target"]).astype(str)
        experience = pd.concat([existing, experience], ignore_index=True)
    if not experience.empty:
        experience["target"] = experience["target"].fillna("NPB").astype(str)
        experience["competition_id"] = experience["competition_id"].fillna(experience["target"]).astype(str)
        experience = experience.drop_duplicates("prediction_id", keep="last").sort_values(
            ["prediction_cutoff_utc", "game_id"]
        ).reset_index(drop=True)
    experience = add_dimensions(experience)
    EXPERIENCE.mkdir(parents=True, exist_ok=True)
    experience.to_csv(LEDGER_PATH, index=False)
    LEDGER_JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str, sort_keys=True) + "\n"
                for r in experience.to_dict("records")),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "status": "UPDATED",
        "prediction_rows": int(len(pred)),
        "matched_rows_total": int(len(experience)),
        "newly_matched_rows": int(sum(1 for x in matched["prediction_id"].astype(str) if x not in existing_ids)),
        "outcome_accuracy": float(experience["outcome_correct"].mean()),
        "logloss": float(experience["logloss"].mean()),
        "brier": float(experience["brier"].mean()),
        "ece": _multiclass_ece(
            experience[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
            experience["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
        ),
        "draw_rows": int((experience["actual_outcome"] == "DRAW").sum()),
        "draw_recall": float((
            (experience["predicted_outcome"] == "DRAW")
            & (experience["actual_outcome"] == "DRAW")
        ).sum() / max(1, (experience["actual_outcome"] == "DRAW").sum())),
        "low_high_accuracy": float(experience["low_high_correct"].mean()),
        "score_mae": float(experience["score_mae"].mean()),
        "top1_exact_hit_rate": float(experience["top1_exact_hit"].mean()),
        "top4_exact_hit_rate": float(experience["top4_hit"].mean()),
        "by_regime": {},
        "by_target": {},
        "by_prediction_target": _prediction_target_metrics(experience),
        "by_prediction_source": {},
        "by_league": {},
        "by_competition": {},
        "by_phase": {},
        "by_dominant_expert": {},
        "timing_30m": _timing_30m_metrics(experience),
        "revision_intelligence": _revision_metrics(experience),
        "rolling": {},
    }

        for key, group in experience.groupby("league", dropna=False):
        summary["by_league"][str(key)] = {
            "rows": int(len(group)),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
            "brier": float(group["brier"].mean()),
            "ece": _multiclass_ece(
                group[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
                group["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
            ),
            "low_high_accuracy": float(group["low_high_correct"].mean()),
            "top4_exact_hit_rate": float(group["top4_hit"].mean()),
            "score_mae": float(group["score_mae"].mean()),
            "targets": _prediction_target_metrics(group),
        }

    for key, group in experience.groupby("competition_key", dropna=False):
        summary["by_competition"][str(key)] = {
            "rows": int(len(group)),
            "league": str(group["league"].iloc[0]),
            "competition": str(group["competition"].iloc[0]),
            "competition_stage": str(group["competition_stage"].iloc[0]),
            "season_type": str(group["season_type"].iloc[0]),
            "competition_classification_status": str(group["competition_classification_status"].iloc[0]),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
            "brier": float(group["brier"].mean()),
            "ece": _multiclass_ece(
                group[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
                group["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
            ),
            "low_high_accuracy": float(group["low_high_correct"].mean()),
            "top4_exact_hit_rate": float(group["top4_hit"].mean()),
            "score_mae": float(group["score_mae"].mean()),
            "targets": _prediction_target_metrics(group),
        }

    for key, group in experience.groupby("competition_stage", dropna=False):
        summary["by_phase"][str(key)] = {
            "rows": int(len(group)),
            "season_type": str(group["season_type"].iloc[0]),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
            "brier": float(group["brier"].mean()),
            "ece": _multiclass_ece(
                group[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
                group["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
            ),
            "low_high_accuracy": float(group["low_high_correct"].mean()),
            "top4_exact_hit_rate": float(group["top4_hit"].mean()),
            "score_mae": float(group["score_mae"].mean()),
            "targets": _prediction_target_metrics(group),
        }

# Target/competition is a first-class evaluation axis. Never mix
    # NPB/MLB/etc. performance into one headline when target labels exist.
    if "target" in experience.columns:
        for key, group in experience.groupby("target", dropna=False):
            summary["by_target"][str(key)] = {
                "rows": int(len(group)),
                "accuracy": float(group["outcome_correct"].mean()),
                "logloss": float(group["logloss"].mean()),
                "brier": float(group["brier"].mean()),
                "ece": _multiclass_ece(
                    group[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
                    group["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
                ),
                "low_high_accuracy": float(group["low_high_correct"].mean()),
                "top4_exact_hit_rate": float(group["top4_hit"].mean()),
            }

    for key, group in experience.groupby("regime", dropna=False):
        summary["by_regime"][str(key)] = {
            "rows": int(len(group)),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
            "brier": float(group["brier"].mean()),
        }
    if "prediction_source" in experience.columns:
        for key, group in experience.groupby("prediction_source", dropna=False):
            summary["by_prediction_source"][str(key)] = {
                "rows": int(len(group)),
                "accuracy": float(group["outcome_correct"].mean()),
                "logloss": float(group["logloss"].mean()),
                "brier": float(group["brier"].mean()),
                "top4_exact_hit_rate": float(group["top4_hit"].mean()),
                "mean_actual_lead_minutes": float(pd.to_numeric(
                    group.get("prediction_actual_lead_minutes"), errors="coerce"
                ).dropna().mean()) if "prediction_actual_lead_minutes" in group else None,
            }

    for key, group in experience.groupby("dominant_classification_expert", dropna=False):
        summary["by_dominant_expert"][str(key)] = {
            "rows": int(len(group)),
            "accuracy": float(group["outcome_correct"].mean()),
            "logloss": float(group["logloss"].mean()),
        }
    sorted_exp = experience.sort_values(["prediction_cutoff_utc", "game_id"])
    for n in (30, 100, 300):
        g = sorted_exp.tail(n)
        if len(g):
            summary["rolling"][str(n)] = {
                "rows": int(len(g)),
                "accuracy": float(g["outcome_correct"].mean()),
                "logloss": float(g["logloss"].mean()),
                "brier": float(g["brier"].mean()),
                "ece": _multiclass_ece(
                    g[["home_win_pct", "draw_pct", "away_win_pct"]].to_numpy(float),
                    g["actual_outcome"].map({"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}).to_numpy(int),
                ),
                "low_high_accuracy": float(g["low_high_correct"].mean()),
                "top4_exact_hit_rate": float(g["top4_hit"].mean()),
            }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=str)
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.archive:
        print(json.dumps(archive_production_output(args.archive, run_id=args.run_id), ensure_ascii=False, indent=2))
    if args.reconcile:
        print(json.dumps(reconcile(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
