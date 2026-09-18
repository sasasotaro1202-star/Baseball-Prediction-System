"""Evidence-first research planner.

The planner does not optimize for the largest backtest gain. It prioritizes
changes that are most likely to improve live robustness while preserving
point-in-time correctness and production gates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ResearchTask:
    area: str
    priority: int
    required_evidence: tuple[str, ...]
    stop_condition: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TASKS: tuple[ResearchTask, ...] = (
    ResearchTask("starting_pitcher", 100, ("announcement_cutoff_audit", "walk_forward_oos", "calibration"), "Do not promote if starter timing is ambiguous or post-cutoff."),
    ResearchTask("bullpen_fatigue", 90, ("point_in_time_replay", "walk_forward_oos", "ablation"), "Reject if any feature uses post-game relief appearances."),
    ResearchTask("batting_form", 85, ("lagged_feature_audit", "walk_forward_oos", "ablation"), "Reject if rolling windows include the current game."),
    ResearchTask("park_factor", 75, ("historical_snapshot_audit", "walk_forward_oos", "ablation"), "Reject if park statistics are computed with future games."),
    ResearchTask("weather", 65, ("forecast_cutoff_audit", "walk_forward_oos", "ablation"), "Reject if observed post-start weather is substituted for forecast data."),
    ResearchTask("schedule_fatigue", 60, ("point_in_time_replay", "walk_forward_oos", "ablation"), "Reject if future schedule/results leak into the feature."),
    ResearchTask("model_family", 55, ("two_validation_windows", "locked_holdout", "calibration"), "Reject if gains disappear outside the development window."),
    ResearchTask("market_baseline", 50, ("closing_line_timestamp_audit", "walk_forward_oos"), "Never use closing information as a pregame model feature unless it was available at cutoff."),
)


def create_plan(analysis: dict | str, *, completed: set[str] | None = None) -> dict:
    """Return the next evidence-first research queue.

    Completed tasks are removed without changing the priority of remaining
    tasks. The caller still owns the adoption gate; this planner never
    promotes a model.
    """
    completed = completed or set()
    queue = [
        task.to_dict()
        for task in sorted(TASKS, key=lambda x: (-x.priority, x.area))
        if task.area not in completed
    ]
    return {
        "current_issue": analysis,
        "objective": "maximize defensible future-game performance, not backtest score",
        "queue": queue,
        "required_global_gates": [
            "chronological_walk_forward_only",
            "point_in_time_features",
            "temporal_calibration",
            "locked_holdout",
            "reproducible_candidate",
            "fail_closed_on_missing_evidence",
        ],
    }
