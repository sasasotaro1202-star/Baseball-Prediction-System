"""Production Baseball research engine.

Lifecycle:
chronological OOS -> weakness discovery -> NPB candidate replay on Development
OOS -> Candidate Lock -> independent Locked Holdout -> ADOPT/REJECT/HOLD.

No holdout result is allowed to influence candidate selection. NPB remains a
three-class Home / Draw / Away target throughout the lifecycle. A separate
Top-Draw prediction is also researched: one game per Japan-local calendar date,
chosen by maximum predicted Draw probability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baseball_backtest import BaseballBacktest
from evaluation.npb_outcome import NPB_OUTCOME_LABELS, validate_npb_probabilities
from research.candidate_registry import record_candidate
from research.npb_candidate_replay import run_npb_candidate_cycle
from research.npb_draw_research import write_top_draw_research

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class ResearchCycle:
    cycle_id: str
    git_commit: str
    started_at: str
    finished_at: str | None
    leagues: tuple[str, ...]
    stages: tuple[str, ...]
    promotion_decision: str


@dataclass(frozen=True)
class CompatibilityResult:
    passed: bool
    details: dict[str, Any]


def _git_commit() -> str:
    import os
    value = os.getenv("GITHUB_SHA", "").strip()
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _cycle_id(started_at: str, commit: str) -> str:
    return hashlib.sha256(f"{started_at}|{commit}".encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _research_state() -> dict[str, Any]:
    from research.research_loop import build_state
    return build_state()


def _verify_npb_outcome_contract() -> dict[str, Any]:
    """Fail closed if an NPB OOS artifact has collapsed Draw into binary form."""
    path = RESULTS / "npb_backtest_results.csv"
    if not path.exists():
        raise RuntimeError("NPB results artifact is missing; cannot verify Draw output")
    rows = 0
    draw_actual = 0
    draw_predicted = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"pred_home", "pred_draw", "pred_away", "actual_home_score", "actual_away_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError("NPB results artifact is missing explicit three-way fields: " + ", ".join(sorted(missing)))
        for row in reader:
            p = validate_npb_probabilities([float(row["pred_home"]), float(row["pred_draw"]), float(row["pred_away"])])
            rows += 1
            h = float(row["actual_home_score"]); a = float(row["actual_away_score"])
            draw_actual += int(h == a)
            draw_predicted += int(int(max(range(3), key=lambda i: p[i])) == 1)
    contract = {
        "labels": list(NPB_OUTCOME_LABELS),
        "rows_verified": rows,
        "actual_draw_rows": draw_actual,
        "predicted_draw_rows": draw_predicted,
        "draw_field_required": True,
        "probability_order": ["pred_home", "pred_draw", "pred_away"],
        "status": "PASS",
    }
    _write_json(RESULTS / "npb_outcome_contract.json", contract)
    return contract


def _npb_research_lifecycle(git_commit: str, data_dir: str | Path) -> dict[str, Any]:
    """Run candidate replay, lock before holdout, then evaluate the locked holdout."""
    feature_version = __import__("os").getenv("BASEBALL_NPB_FEATURE_VERSION", "baseball-features-v1")
    replay = run_npb_candidate_cycle(data_dir=data_dir, git_commit=git_commit, feature_version=feature_version)
    if replay.get("decision") != "HOLDOUT_READY":
        return replay
    holdout = replay["holdout"]
    candidate = replay["candidate"]["candidate"]
    record = record_candidate(
        candidate_id=candidate["candidate_id"], git_commit=git_commit,
        feature_version=candidate["feature_version"], model_version=candidate["model_version"],
        development_metrics=candidate["development_metrics"], holdout_baseline=holdout["baseline"],
        holdout_candidate=holdout["candidate"], validation_windows=int(holdout["validation_windows"]),
        calibration_ok=bool(holdout["calibration_ok"]), no_future_target_data=bool(holdout["no_future_target_data"]),
        reproducible=bool(holdout["reproducible"]), holdout_score_baseline=holdout["baseline_score"],
        holdout_score_candidate=holdout["candidate_score"], holdout_hilo_baseline=holdout["baseline_hilo"],
        holdout_hilo_candidate=holdout["candidate_hilo"], league="NPB",
    )
    return {"stage":"locked_holdout_evaluated","decision":record.decision,"candidate_id":record.candidate_id,
            "candidate_model":record.model_version,"registry":str(RESULTS / "candidate_registry.json"),"record":asdict(record)}


class BaseballResearchEngine:
    ENGINE_VERSION = "baseball-research-engine-v4-npb-three-way-top-draw"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.git_commit = _git_commit()
        self.cycle_id = _cycle_id(self.started_at, self.git_commit)
        self.stages: list[str] = []

    def fit_ensemble_compatibility(self, X, y, league: str) -> CompatibilityResult:
        """Prove the v4.4 bridge delegates to the unchanged Baseball engine.

        Two fresh engine instances are used so no fitted estimator state is
        shared. The comparison is limited to the existing fit_ensemble return
        contract: validation scores, selected best model, and fitted model
        names. No prediction logic is replaced or reimplemented here.
        """
        from research.v44_bridge import V44BaseballBacktest
        legacy = BaseballBacktest(self.data_dir)
        bridge = V44BaseballBacktest(self.data_dir)
        legacy_fitted, legacy_scores, legacy_best = legacy.fit_ensemble(X, y, league)
        bridge_fitted, bridge_scores, bridge_best = bridge.fit_ensemble(X, y, league)

        import numpy as np
        score_equal = legacy_scores == bridge_scores
        if not score_equal and legacy_scores and bridge_scores:
            keys_equal = set(legacy_scores) == set(bridge_scores)
            score_equal = keys_equal and all(np.isclose(legacy_scores[k], bridge_scores[k], rtol=1e-12, atol=1e-12) for k in legacy_scores)
        legacy_names = [item[2] for item in (legacy_fitted or [])]
        bridge_names = [item[2] for item in (bridge_fitted or [])]
        details = {
            "score_equal": bool(score_equal),
            "best_model_equal": legacy_best == bridge_best,
            "model_names_equal": legacy_names == bridge_names,
            "legacy_best_model": legacy_best,
            "bridge_best_model": bridge_best,
            "legacy_model_names": legacy_names,
            "bridge_model_names": bridge_names,
            "bridge_audit_tail": list(bridge.audit[-5:]),
            "delegated_to_existing_engine": True,
        }
        return CompatibilityResult(
            passed=bool(details["score_equal"] and details["best_model_equal"] and details["model_names_equal"] and details["bridge_audit_tail"]),
            details=details,
        )

    def _backtest(self, *, npb: bool, mlb: bool, mlb_start: int, mlb_end: int) -> None:
        self.stages.append("chronological_oos_backtest")
        bt = BaseballBacktest(self.data_dir)
        bt.run(npb=npb, mlb=mlb, mlb_start=mlb_start, mlb_end=mlb_end)
        if npb:
            self.stages.append("npb_home_draw_away_contract_check")
            _verify_npb_outcome_contract()
            self.stages.append("npb_top_draw_research")
            write_top_draw_research()

    def _research_state(self) -> dict[str, Any]:
        self.stages.append("weakness_discovery")
        state = _research_state()
        state["engine_version"] = self.ENGINE_VERSION
        state["cycle_id"] = self.cycle_id
        state["git_commit"] = self.git_commit
        state["promotion_policy"] = {
            "development_oos": "candidate replay/selection only",
            "candidate_lock": "required before holdout",
            "locked_holdout": "independent confirmation only",
            "npb_target": "HOME / DRAW / AWAY plus DrawRecall and DrawProbabilityMAE",
            "npb_separate_prediction": "one game per Japan-local calendar date with maximum pred_draw",
            "npb_separate_prediction_metrics": ["top_draw_hit_rate", "top_draw_probability_mae", "lift_vs_all_game_draw_rate"],
            "decision": "ADOPT only when research.validation_pipeline permits it",
        }
        _write_json(ROOT / "research_state.json", state)
        return state

    def run(self, *, npb: bool = True, mlb: bool = True, mlb_start: int = 2020, mlb_end: int = 2026) -> ResearchCycle:
        RESULTS.mkdir(parents=True, exist_ok=True)
        self._backtest(npb=npb, mlb=mlb, mlb_start=mlb_start, mlb_end=mlb_end)
        self._research_state()
        promotion_decision = "NO_CHANGE"
        if npb:
            self.stages.append("npb_development_candidate_replay")
            try:
                lifecycle = _npb_research_lifecycle(self.git_commit, self.data_dir)
            except Exception as exc:
                lifecycle = {"stage": "npb_research_error", "decision": "HOLD", "error": f"{type(exc).__name__}: {exc}"}
            _write_json(RESULTS / "npb_research_lifecycle.json", lifecycle)
            self.stages.append(lifecycle["stage"])
            promotion_decision = lifecycle["decision"]
        else:
            self.stages.append("npb_lifecycle_skipped")

        finished = datetime.now(timezone.utc).isoformat()
        cycle = ResearchCycle(
            cycle_id=self.cycle_id,
            git_commit=self.git_commit,
            started_at=self.started_at,
            finished_at=finished,
            leagues=tuple(x for x, enabled in (("NPB", npb), ("MLB", mlb)) if enabled),
            stages=tuple(self.stages),
            promotion_decision=promotion_decision,
        )
        _write_json(ROOT / "research_cycle_manifest.json", asdict(cycle))
        return cycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.engine")
    parser.add_argument("--npb-only", action="store_true")
    parser.add_argument("--mlb-only", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mlb-start", type=int, default=2020)
    parser.add_argument("--mlb-end", type=int, default=2026)
    args = parser.parse_args(argv)
    if args.npb_only and args.mlb_only:
        parser.error("--npb-only and --mlb-only are mutually exclusive")
    engine = BaseballResearchEngine(args.data_dir)
    cycle = engine.run(npb=not args.mlb_only, mlb=not args.npb_only, mlb_start=args.mlb_start, mlb_end=args.mlb_end)
    print(json.dumps(asdict(cycle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
