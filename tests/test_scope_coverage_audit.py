import json
from pathlib import Path

from research import scope_coverage_audit as s


def test_scope_coverage_audit_is_fail_closed_without_discovery_or_pit(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "DISCOVERY_FILE", tmp_path / "results/scope_discovery.json")
    result = s.audit()
    assert result["scope_count"] > 0
    assert result["policy"]["promotion_by_coverage_audit"] is False
    assert all(row["pit_snapshot_rows"] == 0 for row in result["scopes"])


def test_scope_coverage_stage_advances_only_with_explicit_evidence():
    assert s._stage(registry=False, source_count=0, adapter_count=0, discovered=False, pit_count=0, oos_pass=False, production_pass=False) == "CATALOG_ONLY"
    assert s._stage(registry=True, source_count=1, adapter_count=0, discovered=False, pit_count=0, oos_pass=False, production_pass=False) == "SOURCE_REGISTERED"
    assert s._stage(registry=True, source_count=1, adapter_count=1, discovered=True, pit_count=0, oos_pass=False, production_pass=False) == "ADAPTER"
    assert s._stage(registry=True, source_count=1, adapter_count=1, discovered=True, pit_count=2, oos_pass=False, production_pass=False) == "PIT"
    assert s._stage(registry=True, source_count=1, adapter_count=1, discovered=True, pit_count=2, oos_pass=True, production_pass=False) == "OOS"
    assert s._stage(registry=True, source_count=1, adapter_count=1, discovered=True, pit_count=2, oos_pass=True, production_pass=True) == "PRODUCTION"
    json.dumps(s.audit(), ensure_ascii=False)


def test_frontier_game_metrics_counts_discovered_and_deferred(tmp_path, monkeypatch):
    root = tmp_path
    cycle = root / "results/24h/cycle_1"
    cycle.mkdir(parents=True)
    (cycle / "kbo.json").write_text(
        json.dumps({"status": "EXECUTED", "game_count": 5}), encoding="utf-8"
    )
    (cycle / "cpbl.json").write_text(
        json.dumps({
            "status": "EXECUTED",
            "game_token_count": 4,
            "availability_status": "DISCOVERED_NOT_PARSED",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(s, "ROOT", root)
    metrics = s._frontier_game_metrics()
    assert metrics["KBO"]["games_discovered"] == 5
    assert metrics["CPBL"]["token_count"] == 4
    assert metrics["CPBL"]["deferred_or_unparsed"] == 4
