"""
    errors = workflow_contract_errors(bad, tmp_path / "bad.yml")
    assert any("workflow_failure_masking" in x for x in errors)
    assert any("workflow_failure_masking_or_true" in x for x in errors)
    assert any("workflow_unpinned_action" in x for x in errors)


def test_source_file_is_present():
    source = Path("PROJECT_SOURCE.md")
    assert source.is_file()
    assert source.stat().st_size > 0


def test_source_section_parser_ignores_numbered_lists():
    from research.project_governance import SECTION_RE
    text = "71. COST FIREWALL\n\n1. verified free\n2. free quota\n3. OSS/local\n\n⸻\n\n72. SECURITY / DATA GOVERNANCE\n"
    assert [int(m[0]) for m in SECTION_RE.findall(text)] == [71, 72]

def test_source_provenance_accepts_matching_hash(tmp_path: Path):
    import hashlib
    import json
    source = tmp_path / "PROJECT_SOURCE.md"
    provenance = tmp_path / "PROJECT_SOURCE_PROVENANCE.json"
    payload = "canonical source\n"
    source.write_text(payload, encoding="utf-8")
    provenance.write_text(
        json.dumps({
            "schema_version": 1,
            "source_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "source_bytes": len(payload.encode("utf-8")),
        }),
        encoding="utf-8",
    )
    assert source_provenance_errors(tmp_path) == []


def test_source_provenance_rejects_drift(tmp_path: Path):
    import hashlib
    import json
    source = tmp_path / "PROJECT_SOURCE.md"
    provenance = tmp_path / "PROJECT_SOURCE_PROVENANCE.json"
    source.write_text("changed\n", encoding="utf-8")
    provenance.write_text(
        json.dumps({
            "schema_version": 1,
            "source_sha256": hashlib.sha256(b"canonical\n").hexdigest(),
            "source_bytes": len(b"canonical\n"),
        }),
        encoding="utf-8",
    )
    errors = source_provenance_errors(tmp_path)
    assert errors and errors[0].startswith("project_source_sha256_mismatch:")


def test_critical_automation_files_are_in_governance_contracts():
    from research.project_governance import WORKFLOW_CONTRACTS
    for path in (
        ".github/workflows/baseball_governance_autopilot.yml",
        ".github/workflows/npb_prediction_experience_archive.yml",
        ".github/workflows/npb_experience_reconciliation.yml",
        ".github/workflows/npb_experience_learning.yml",
        ".github/workflows/baseball_game_script_lab.yml",
    ):
        assert path in WORKFLOW_CONTRACTS


def test_governance_workflow_expressions_are_not_backslash_escaped():
    workflow = Path(".github/workflows/baseball_governance_autopilot.yml").read_text(encoding="utf-8")
    expression = "$" + "{{"
    assert "\\" + expression not in workflow
    assert expression + " github.token }}" in workflow
    assert expression + " github.repository }}" in workflow

def test_game_script_lab_is_monitored():
    from research.project_governance import WORKFLOW_CONTRACTS

    path = ".github/workflows/baseball_game_script_lab.yml"
    assert path in WORKFLOW_CONTRACTS
    assert WORKFLOW_CONTRACTS[path]["monitor"] is True
    assert WORKFLOW_CONTRACTS[path]["max_age_hours"] <= 8
    assert "research/game_script_lab.py" in WORKFLOW_CONTRACTS[path]["required"]
