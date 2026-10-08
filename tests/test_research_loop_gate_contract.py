from research.adoption_gate import GatePolicy
from research.research_loop import PROMOTION_GATE


def test_research_loop_uses_formal_adoption_gate():
    gate = GatePolicy()
    assert PROMOTION_GATE["formal_gate_module"] == "research.adoption_gate.GatePolicy"
    assert PROMOTION_GATE["auto_promotion"] is False
    assert PROMOTION_GATE["min_relative_improvement"] == gate.min_relative_improvement
    assert PROMOTION_GATE["min_relative_brier_improvement"] == gate.min_relative_brier_improvement
    assert PROMOTION_GATE["min_oos_rows"] == gate.min_oos_rows
    assert PROMOTION_GATE["min_non_worsening_period_fraction"] == gate.min_non_worsening_period_fraction
    assert PROMOTION_GATE["holdout_selection_forbidden"] is True
    assert PROMOTION_GATE["require_pit_starter_evidence"] is True


def test_research_loop_gate_is_not_weaker_than_reference_policy():
    assert PROMOTION_GATE["min_relative_improvement"] >= 0.03
    assert PROMOTION_GATE["min_relative_brier_improvement"] >= 0.01
    assert PROMOTION_GATE["auto_promotion"] is False
