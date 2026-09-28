from pathlib import Path

from core.pit_snapshot import append_snapshot, make_snapshot
from core.pit_replay import replay


def test_pit_snapshot_accepts_expanded_baseball_competition():
    p = make_snapshot(
        event_id="KBO:1",
        league="KBO",
        entity_type="game",
        entity_id="1",
        source="TEST",
        payload={"teams": ["A", "B"]},
        prediction_cutoff="2026-09-28T10:00:00+00:00",
        available_at="2026-09-28T09:00:00+00:00",
        retrieved_at="2026-09-28T09:05:00+00:00",
    )
    assert p.league == "KBO"


def test_replay_is_competition_agnostic(tmp_path: Path):
    path = tmp_path / "snapshots.jsonl"
    for league in ("KBO", "CPBL", "NCAA_D1"):
        append_snapshot(
            make_snapshot(
                event_id=f"{league}:1",
                league=league,
                entity_type="game",
                entity_id="1",
                source="TEST",
                payload={"league": league},
                prediction_cutoff="2026-09-28T10:00:00+00:00",
                available_at="2026-09-28T09:00:00+00:00",
                retrieved_at="2026-09-28T09:05:00+00:00",
            ),
            path,
        )
    rows = replay(path, cutoff="2026-09-28T10:00:00+00:00", league="KBO")
    assert len(rows) == 1
    assert rows[0]["league"] == "KBO"
