#!/usr/bin/env python3
"""Cheap, deterministic preflight checks for the Baseball Actions pipeline.

Keep this gate dependency-light: it must catch broken production imports and
unsafe workflow wiring before expensive research starts, without masking
failures from the real test suite.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_MODULES = (
    "baseball_backtest",
    "research_runner",
    "research_runner_v6",
    "data.npb_pbp_adapter",
    "data.availability",
    "research.validation_pipeline",
    "research.adoption_gate",
    "research.closed_loop_governance",
    "prediction.runner",
    "prediction.prediction_log",
    "prediction.score_distribution",
    "core.atomic_io",
    "evaluation.calibration",
)

REQUIRED_FILES = (
    Path("requirements.txt"),
    Path("requirements-ci.txt"),
    Path("requirements-pit.txt"),
    Path("research_runner.py"),
    Path("research_runner_v6.py"),
    Path("baseball_backtest.py"),
    Path("data/pit_acquisition.py"),
    Path("data/availability.py"),
    Path("core/pit_snapshot.py"),
    Path("core/pit_replay.py"),
    Path("prediction/runner.py"),
    Path("prediction/prediction_log.py"),
    Path("prediction/score_distribution.py"),
    Path("evaluation/calibration.py"),
    Path("core/atomic_io.py"),
    Path("research/closed_loop_governance.py"),
)

WORKFLOW_DIR = ROOT / ".github" / "workflows"
SHA_ACTION_RE = re.compile(r"^[ \t]*uses:[ \t]*[^@\s]+@([0-9a-fA-F]{40})[ \t]*$", re.MULTILINE)
USES_RE = re.compile(r"^[ \t]*uses:[ \t]*([^@\s]+)@([^\s#]+)", re.MULTILINE)


def _workflow_reliability_errors(text: str, path: Path) -> list[str]:
    """Return deterministic workflow hardening failures.

    These checks are intentionally conservative: a workflow must be reproducible,
    bounded, explicitly permissioned, and unable to hide command failures.
    """
    errors: list[str] = []
    uses = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        match = re.match(r"^[ \t]*-?[ \t]*uses:[ \t]*([^@\s]+)@([^\s#]+)", raw)
        if not match:
            continue
        uses.append((match, lineno))
        ref = match.group(2)
        if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
            errors.append(
                f"Workflow action is not pinned to an immutable full SHA: "
                f"{path}:{lineno}: {match.group(1)}@{ref}"
            )
    if uses and any(
        not SHA_ACTION_RE.match(raw)
        for raw in text.splitlines()
        if re.match(r"^[ \\t]*-?[ \\t]*uses:", raw)
    ):
        # Keep the detector conservative, but report malformed action lines
        # rather than silently accepting an unpinned/misparsed reference.
        if not all(re.fullmatch(r"[0-9a-fA-F]{40}", m.group(0).group(2)) for m in uses):
            pass
    if "runs-on:" not in text:
        errors.append(f"Workflow has no runner declaration: {path}")
    if not re.search(r"^permissions:\s*$", text, re.MULTILINE):
        errors.append(f"Workflow has no explicit top-level permissions: {path}")

    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"^continue-on-error\s*:", stripped):
            errors.append(f"Workflow masks step/job failure with continue-on-error: {path}:{lineno}")
        if "|| true" in stripped:
            errors.append(f"Workflow masks command failure with '|| true': {path}:{lineno}")

    jobs_match = re.search(r"^jobs:\s*$", text, re.MULTILINE)
    if jobs_match:
        jobs_text = text[jobs_match.end():]
        job_blocks = list(re.finditer(
            r"^  ([A-Za-z0-9_-]+):\s*$(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
            jobs_text, re.MULTILINE | re.DOTALL,
        ))
        if not job_blocks:
            errors.append(f"Workflow has no recognizable jobs: {path}")
        for block in job_blocks:
            job_name = block.group(1)
            body = block.group("body")
            if not re.search(r"^    timeout-minutes:\s*\d+\s*$", body, re.MULTILINE):
                errors.append(f"Job '{job_name}' has no bounded timeout-minutes: {path}")
    return errors


def _workflow_reliability_checks() -> None:
    """Reject workflow wiring that would make CI non-reproducible or fragile."""
    if not WORKFLOW_DIR.is_dir():
        raise SystemExit("Missing .github/workflows directory")
    workflows = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    if not workflows:
        raise SystemExit("No GitHub Actions workflow files found")

    checked = 0
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        errors = _workflow_reliability_errors(text, path)
        if errors:
            raise SystemExit("\n".join(errors))
        checked += 1
    print(f"OK: {checked} workflow files pass immutable-action, permission, timeout, and failure-propagation checks")


def main() -> int:
    missing = [str(p) for p in REQUIRED_FILES if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    for name in REQUIRED_MODULES:
        importlib.import_module(name)
        print(f"OK: {name}")
    _workflow_reliability_checks()
    print("Baseball Action preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
