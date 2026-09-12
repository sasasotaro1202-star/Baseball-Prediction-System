#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a non-authoritative research memo with the OpenAI Responses API.

The API key is read only from OPENAI_API_KEY. The model is advisory only: this
script never edits model weights, prediction code, or promotion state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research_state.json"
PLAN = ROOT / "research_plan.json"
OUT = ROOT / "research_ai_advice.json"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def extract_text(data: dict) -> str:
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            value = content.get("text")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()


def main() -> None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is required")

    model = os.environ.get("OPENAI_RESEARCH_MODEL", "gpt-5.6-luna")
    state = load_json(STATE, {})
    plan = load_json(PLAN, {})
    prompt = {
        "role": "You are an independent sports-model research reviewer.",
        "rules": [
            "Use only the supplied out-of-sample research state.",
            "Do not invent missing metrics or data.",
            "Do not recommend using future target-game information.",
            "Do not directly modify production models; provide testable research hypotheses only.",
            "Prioritize calibration, log loss, Brier score, OOS robustness, and regression risk.",
        ],
        "baseball_state": state,
        "next_plan": plan,
        "requested_output": [
            "top_research_hypotheses",
            "evidence_from_current_state",
            "recommended_oos_tests",
            "failure_modes_to_check",
            "promotion_conditions",
        ],
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": json.dumps(prompt, ensure_ascii=False),
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    text = extract_text(data)
    if not text:
        raise RuntimeError("OpenAI response contained no text")

    OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "source_policy": "prior_rows_only",
                "advisory_only": True,
                "advice": text,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OpenAI research advisory generated with model={model}")


if __name__ == "__main__":
    main()
