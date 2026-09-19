from research.improvement_planner import create_plan


def test_planner_prioritizes_production_critical_evidence():
    plan = create_plan("starter timing weakness")
    assert plan["objective"].startswith("maximize defensible future-game performance")
    assert plan["queue"][0]["area"] == "starting_pitcher"
    assert "chronological_walk_forward_only" in plan["required_global_gates"]
    assert "locked_holdout" in plan["required_global_gates"]
    assert "fail_closed_on_missing_evidence" in plan["required_global_gates"]


def test_planner_can_remove_completed_tasks_without_reordering_remaining():
    plan = create_plan("x", completed={"starting_pitcher"})
    assert plan["queue"][0]["area"] == "target_permutation"
    assert all(item["area"] != "starting_pitcher" for item in plan["queue"])


def test_planner_does_not_promote_candidates():
    plan = create_plan("x")
    assert "adopt" not in plan
    assert "promote" not in plan
