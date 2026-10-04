from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.game_script_model import (
    SCHEMA_VERSION,
    TransitionStore,
    _hash_inputs,
    _pa_starts,
    _summary_from_score_matrix,
    build_game_index,
    simulate_game,
)


def _game(game_id: str, when: str, home: str, away: str, final_h: int, final_a: int, game_type: str = "公式戦") -> pd.DataFrame:
    rows = [
        {
            "game_id": game_id,
            "game_date": when,
            "home_team_name": home,
            "away_team_name": away,
            "game_type_name": game_type,
            "PlayInfo_SeqNo": 1,
            "page": f"01T01",
            "inning": 1,
            "TB": "T",
            "outs_when_up": 0,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "home_total_runs": 0,
            "away_total_runs": 0,
        },
        {
            "game_id": game_id,
            "game_date": when,
            "home_team_name": home,
            "away_team_name": away,
            "game_type_name": game_type,
            "PlayInfo_SeqNo": 2,
            "page": f"01B01",
            "inning": 1,
            "TB": "B",
            "outs_when_up": 0,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "home_total_runs": final_h,
            "away_total_runs": final_a,
        },
    ]
    return pd.DataFrame(rows)


def test_pa_starts_keep_one_row_per_plate_appearance():
    frame = _game("g1", "2026-01-01T09:00:00Z", "A", "B", 1, 0)
    duplicated = pd.concat([frame.iloc[[0]], frame], ignore_index=True)
    starts = _pa_starts(duplicated)
    assert len(starts) == 2
    assert list(starts["page"]) == ["01T01", "01B01"]


def test_transition_store_records_terminal_run_and_simulation_contract():
    store = TransitionStore()
    frame = _game("g1", "2026-01-01T09:00:00Z", "A", "B", 2, 0)
    added = store.update_game(frame, "A", "B")
    assert added == 2
    keys, probs, observed = store.distribution("B", (1, "T", 0, 0, 0))
    assert keys
    assert observed >= 1
    assert np.isclose(probs.sum(), 1.0)

    result = simulate_game(store, "A", "B", n_sims=200, seed=7, collect_script=True)
    assert np.isclose(sum(result["outcome_probability"]), 1.0)
    assert 0.0 <= result["fallback_transition_rate"] <= 1.0
    assert len(result["top4"]) == 4
    assert "inning_expected_runs" in result


def test_score_summary_preserves_three_way_probability():
    matrix = np.zeros((4, 4), dtype=float)
    matrix[2, 1] = 0.4
    matrix[1, 1] = 0.2
    matrix[0, 2] = 0.4
    summary = _summary_from_score_matrix(matrix)
    assert np.isclose(
        summary["home_probability"] + summary["draw_probability"] + summary["away_probability"],
        1.0,
    )
    assert np.isclose(summary["low_probability"] + summary["high_probability"], 1.0)


def test_game_index_excludes_exhibition_and_retains_chronology():
    regular = _game("g1", "2026-01-01T09:00:00Z", "A", "B", 1, 0, "公式戦")
    exhibition = _game("g2", "2026-01-01T10:00:00Z", "C", "D", 3, 0, "オープン戦")
    frame = pd.concat([regular, exhibition], ignore_index=True)
    frame["__game_id"] = frame["game_id"].astype(str)
    frame["__game_time"] = pd.to_datetime(frame["game_date"], utc=True)
    frame["__order"] = frame["PlayInfo_SeqNo"]
    index = build_game_index(frame)
    assert index["game_id"].tolist() == ["g1"]


def test_input_fingerprint_is_stable_across_runner_mtime(tmp_path: Path):
    path = tmp_path / "2026-01_pbp.csv"
    path.write_text("game_id,value\ng1,1\n", encoding="utf-8")
    first = _hash_inputs([path], config={"schema": SCHEMA_VERSION})
    stamp = path.stat().st_atime_ns, path.stat().st_mtime_ns
    os.utime(path, ns=(stamp[0], stamp[1] + 123456789))
    second = _hash_inputs([path], config={"schema": SCHEMA_VERSION})
    assert first == second
