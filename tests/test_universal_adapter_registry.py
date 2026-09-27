from research.universal_adapter_registry import (
    adapter_metadata,
    adapter_present,
    adapter_spec,
    adapter_specs,
    audit_adapters,
)
from data.source_registry import SOURCES


def test_every_adapter_mapping_points_to_a_registered_source():
    source_ids = {s.source_id for s in SOURCES}
    assert all(spec.source_id in source_ids for spec in adapter_specs())


def test_every_mapped_adapter_file_is_present():
    for spec in adapter_specs():
        assert adapter_present(spec.source_id), spec.source_id


def test_unmapped_source_is_fail_closed():
    meta = adapter_metadata("weather")
    assert meta["mapped"] is False
    assert meta["implemented"] is False
    assert adapter_spec("weather") is None


def test_statcast_has_concrete_collector_contract_without_pit_promotion():
    meta = adapter_metadata("statcast")
    assert meta["implemented"] is True
    assert meta["kind"] == "collector"
    assert meta["collection_ready"] is True


def test_npb_tracking_is_normalizer_only():
    meta = adapter_metadata("npb_hawkeye_npbplus")
    assert meta["implemented"] is True
    assert meta["kind"] == "normalizer"
    assert meta["collection_ready"] is False


def test_audit_has_no_unknown_mappings_and_preserves_unwired_state():
    report = audit_adapters()
    assert report["unknown_adapter_mappings"] == []
    assert report["source_count"] == len(SOURCES)
    assert report["unwired_count"] > 0
    assert report["implemented_count"] >= 1


def test_kbo_and_cpbl_schedule_adapters_are_concrete():
    for source_id in ("kbo_official_stats", "cpbl_rebas"):
        meta = adapter_metadata(source_id)
        assert meta["implemented"] is True
        assert meta["collection_ready"] is True

def test_espn_discovery_adapters_are_concrete():
    for source_id in ("espn_mlb", "espn_college_baseball", "espn_international"):
        meta = adapter_metadata(source_id)
        assert meta["implemented"] is True
        assert meta["collection_ready"] is True
