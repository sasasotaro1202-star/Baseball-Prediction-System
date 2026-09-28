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


def test_catalog_exact_aliases_resolve_to_canonical_registry_ids():
    assert s.CATALOG_TO_REGISTRY_ID["NCAA_D1"] == "NCAA_D1_BASEBALL"
    assert s.CATALOG_TO_REGISTRY_ID["Japan_Independent"] == "JAPAN_INDEPENDENT"
    assert s.CATALOG_TO_REGISTRY_ID["Japan_University"] == "JAPAN_UNIVERSITY_BASEBALL"
    assert s.CATALOG_TO_REGISTRY_ID["AsianGames_2026"] == "ASIAN_GAMES_BASEBALL"


def test_scope_coverage_uses_canonical_registry_alias_for_narrow_catalog_id():
    result = s.audit()
    row = next(x for x in result["scopes"] if x["scope_id"] == "NCAA_D1")
    assert row["registry_registered"] is True
    assert row["registry_competition_id"] == "NCAA_D1_BASEBALL"
    assert row["registry_alias_applied"] is True
