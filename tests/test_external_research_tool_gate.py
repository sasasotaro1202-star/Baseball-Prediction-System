from research.external_research_tool_gate import (
    load_registry,
    selected_pilot_tools,
    validate_registry,
)


def test_external_tool_registry_is_fail_closed():
    payload = load_registry()
    assert payload["auto_install"] is False
    assert payload["auto_promotion"] is False
    assert payload["default_scope"] == "NON_PRODUCTION"


def test_external_tool_registry_passes_validation():
    payload = load_registry()
    errors = validate_registry(payload)
    assert errors == []


def test_pilot_tools_never_become_production():
    payload = load_registry()
    pilots = selected_pilot_tools(payload)
    assert pilots
    assert all(tool["production_eligible"] is False for tool in pilots)
    assert all(tool["auto_promotion"] is False for tool in pilots)


def test_agent_reach_is_pinned_and_research_only():
    payload = load_registry()
    tool = next(item for item in payload["tools"] if item["tool_id"] == "agent_reach")
    assert tool["status"] == "RESEARCH_CANDIDATE"
    assert tool["production_eligible"] is False
    assert len(tool["pinned_ref"]) == 40


def test_timesfm_requires_license_and_pit_review():
    payload = load_registry()
    tool = next(item for item in payload["tools"] if item["tool_id"] == "timesfm")
    assert "LICENSE_CHECK" in tool["required_evidence"]
    assert "PIT_CHECK" in tool["required_evidence"]
    assert tool["status"] == "RESEARCH_CANDIDATE"
