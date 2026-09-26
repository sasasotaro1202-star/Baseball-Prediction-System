from research.universal_readiness import (
    STATES,
    all_source_readiness,
    readiness_report,
    scope_readiness,
    source_readiness,
)


def test_every_source_has_readiness_state():
    rows = all_source_readiness()
    assert rows
    assert all(r.registered for r in rows)
    assert all(r.state in STATES for r in rows)


def test_unverified_sources_do_not_appear_production_ready():
    for row in all_source_readiness():
        if row.pit_status != "PASS" or row.oos_status != "PASS":
            assert row.production_status != "PASS"


def test_known_statcast_adapter_is_not_auto_promoted():
    row = source_readiness("statcast")
    assert row.adapter_status == "PASS"
    assert row.pit_status == "UNVERIFIED"
    assert row.oos_status == "UNVERIFIED"
    assert row.state != "PRODUCTION"


def test_scope_readiness_is_explicit():
    row = scope_readiness("Japan_HighSchool")
    assert row["sources"] >= 2
    assert row["production_pass"] == 0


def test_readiness_report_is_auditable():
    report = readiness_report()
    assert report["source_count"] == len(all_source_readiness())
    assert report["scope_count"] >= 1
    assert report["promotion_rule"]
