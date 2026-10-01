from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "production_npb.py"


def _source_tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_production_declares_canonical_npb_target_contract_bundle():
    tree = _source_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "standard_target_strategies"
    ]
    assert calls
    assert any(
        node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "NPB"
        for node in calls
    )


def test_production_persists_target_strategy_contracts_for_success_and_blocked_states():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"target_strategy_contracts":NPB_TARGET_CONTRACTS' in source
    assert source.count('"target_strategy_contracts":NPB_TARGET_CONTRACTS') >= 3
