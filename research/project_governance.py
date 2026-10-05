        "required": ("schedule:", "cron: \"13 */6 * * *\"", "actions: read", "--actions"),
    },
    ".github/workflows/npb_prediction_experience_archive.yml": {
        "monitor": True,
        "max_age_hours": 6,
        "allowed_conclusions": ("success", "skipped"),
        "defer_stale_conclusions": ("skipped",),
        "required": ("workflow_run:", "schedule:", "contents: write"),
    },
    ".github/workflows/baseball_24h_research_keeper.yml": {
        "monitor": True,
        "max_age_hours": 1,
        "required": ("schedule:", "cron: '*/5 * * * *'", "actions: write"),
    },
    ".github/workflows/npb_experience_reconciliation.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "workflow_dispatch:", "experience_ledger --reconcile"),
    },
    ".github/workflows/npb_experience_learning.yml": {
        "monitor": True,
        "max_age_hours": 12,
        "required": ("schedule:", "workflow_dispatch:", "research.experience_learning", "research.experience_learning_gate"),
    },
    ".github/workflows/baseball_game_script_lab.yml": {
        "monitor": True,
        "max_age_hours": 8,
        "required": ("schedule:", "workflow_dispatch:", "research/game_script_lab.py"),
    },
    ".github/workflows/project_source_provenance_audit.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "PROJECT_SOURCE.md", "PROJECT_SOURCE_PROVENANCE.json", "sha256"),
    },
    ".github/workflows/baseball_phase1_gate.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "Phase 1 scope", "matrix:", "league: [NPB, MLB]"),
    },
    ".github/workflows/baseball_universal_readiness.yml": {
        "monitor": True,
        "max_age_hours": 30,
        "required": ("schedule:", "research.universal_readiness", "tests/test_universal_readiness.py"),
    },
    ".github/workflows/baseball_actions_recovery.yml": {
        "monitor": False,
        "max_age_hours": 0,
        "required": ("workflow_run:", "actions: write", "Re-run failed jobs with bounded recovery"),
    },
}

REQUIRED_SOURCE_PHRASES = (
    "available_at <= prediction_cutoff",
    "retrieved_at ≠ published_at ≠ available_at",
    "HOME\nDRAW\nAWAY",
    "HOME\nAWAY",
    "LOW = total runs <= 6",
    "HIGH = total runs >= 7",
    "random split禁止",
    "PIT Integrity",
    "PIT violations = 0",
    "NO-FAKE-SUCCESS",
    "Future Generalization",
    "Case-Level Correctness",
    "Calibration",
    "Uncertainty Quality",
    "Safe Degradation > False Prediction",
)

SECTION_RE = re.compile(r"(?m)^(?:⸻\n\n)?\s*(\d+)\.\s+([A-Z0-9][A-Z0-9 /_&/-]+)\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def source_contract_errors(text: str) -> list[str]:
    errors: list[str] = []
    matches = SECTION_RE.findall(text)
    numbers = [int(n) for n, _ in matches]
    expected = list(range(1, 86))
    if numbers != expected:
        missing = [n for n in expected if n not in numbers]
        unexpected = [n for n in numbers if n not in expected]
        errors.append(f"project_source_sections_invalid:missing={missing[:20]}:unexpected={unexpected[:20]}:count={len(numbers)}")
    for phrase in REQUIRED_SOURCE_PHRASES:
        if phrase not in text:
            errors.append(f"project_source_required_text_missing:{phrase!r}")
    return errors


def source_provenance_errors(root: Path = ROOT) -> list[str]:
    source = root / "PROJECT_SOURCE.md"
    provenance = root / "PROJECT_SOURCE_PROVENANCE.json"
    if not source.is_file() or not provenance.is_file():
        return ["project_source_provenance_missing"]
    try:
        payload = json.loads(_read(provenance))
    except Exception as exc:
        return [f"project_source_provenance_invalid_json:{type(exc).__name__}"]
    expected = str(payload.get("source_sha256") or "").strip().lower()