import pytest

from research.statshawk_mlb_adapter import (
    evaluate_pregame_eligibility,
    collapse_independent_games,
    normalize_matchups,
    validate_reconciliation_rows,
)


def test_normalizes_matchups_without_promoting_retrieval_to_availability():
    payload = {
        "date": "2026-09-30",
        "game_count": 1,
        "games": [{
            "contest": "cst_game_1",
            "status": "Final",
            "scheduled_at": "2026-09-30T18:00:00Z",
            "lineups_available": True,
            "home": {
                "team": "Atlanta Braves",
                "team_id": "team_home",
                "probable_pitcher": {
                    "name": "Tyler Mahle",
                    "person_id": "per_pitcher_h",
                    "throws": "R",
                },
                "lineup": [
                    {"name": "Player H", "person_id": "per_h", "confirmed": True}
                ],
                "score": 3,
            },
            "away": {
                "team": "Philadelphia Phillies",
                "team_id": "team_away",
                "probable_pitcher": {
                    "name": "Cristopher Sánchez",
                    "person_id": "per_pitcher_a",
                    "throws": "L",
                },
                "lineup": [
                    {"name": "Player A", "person_id": "per_a", "confirmed": True},
                    {"name": "Player A2", "person_id": "per_a2", "confirmed": True},
                ],
                "score": 4,
            },
        }],
    }

    rows = normalize_matchups(payload, retrieved_at="2026-10-02T04:00:00Z")

    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "cst_game_1"
    assert row["source"] == "statshawk"
    assert row["source_role"] == "secondary_validation"
    assert row["available_at"] is None
    assert row["availability_proof"] == "UNVERIFIED"
    assert row["home_probable_pitcher"] == "Tyler Mahle"
    assert row["home_lineup_confirmed_count"] == 1
    assert row["away_lineup_confirmed_count"] == 2
    assert row["result_reconciliation_eligible"] is True


def test_probable_pitcher_does_not_prove_historical_starter_announcement():
    row = {
        "scheduled_at": "2026-10-05T22:00:00Z",
        "home_probable_pitcher": "Pitcher H",
        "away_probable_pitcher": "Pitcher A",
    }

    eligible, reason = evaluate_pregame_eligibility(row, "2026-10-05T18:00:00Z")

    assert eligible is False
    assert reason == "home_starter_announcement_unproven"


def test_explicit_announcement_can_pass_pregame_gate():
    row = {
        "scheduled_at": "2026-10-05T22:00:00Z",
        "home_starter": "Pitcher H",
        "away_starter": "Pitcher A",
        "home_starter_announced_at": "2026-10-05T16:00:00Z",
        "away_starter_announced_at": "2026-10-05T16:30:00Z",
    }

    assert evaluate_pregame_eligibility(row, "2026-10-05T18:00:00Z") == (
        True,
        "eligible",
    )


def test_announcement_after_cutoff_fails_closed():
    row = {
        "scheduled_at": "2026-10-05T22:00:00Z",
        "home_starter": "Pitcher H",
        "away_starter": "Pitcher A",
        "home_starter_announced_at": "2026-10-05T19:00:00Z",
        "away_starter_announced_at": "2026-10-05T16:30:00Z",
    }

    assert evaluate_pregame_eligibility(row, "2026-10-05T18:00:00Z") == (
        False,
        "home_starter_announced_after_cutoff",
    )


def test_final_rows_are_secondary_reconciliation_only():
    rows = validate_reconciliation_rows([{
        "source": "statshawk",
        "event_id": "cst_game_1",
        "status": "final",
        "home_score": 3,
        "away_score": 4,
    }])

    assert rows[0]["reconciliation_role"] == "secondary_result_check"


def test_final_score_missing_is_hard_failure():
    with pytest.raises(ValueError, match="final_score_missing"):
        validate_reconciliation_rows([{
            "source": "statshawk",
            "event_id": "cst_game_1",
            "status": "final",
            "home_score": None,
            "away_score": 4,
        }])


def test_nonfinal_rows_are_not_result_reconciliation_evidence():
    rows = validate_reconciliation_rows([{
        "source": "statshawk",
        "event_id": "cst_game_1",
        "status": "in progress",
        "home_score": None,
        "away_score": None,
    }])

    assert rows[0]["reconciliation_role"] == "observation_only"


def test_invalid_timestamp_fails_closed():
    payload = {
        "games": [{
            "contest": "cst_bad",
            "status": "Final",
            "scheduled_at": "not-a-time",
            "home": {"team_id": "h"},
            "away": {"team_id": "a"},
        }]
    }

    with pytest.raises(ValueError, match="scheduled_at_invalid"):
        normalize_matchups(payload, retrieved_at="2026-10-02T04:00:00Z")


def test_nonfinite_score_fails_closed():
    payload = {
        "games": [{
            "contest": "cst_bad_score",
            "status": "Final",
            "scheduled_at": "2026-09-30T18:00:00Z",
            "home": {"team_id": "h", "score": "nan"},
            "away": {"team_id": "a", "score": 4},
        }]
    }

    with pytest.raises(ValueError, match="home_score_nonfinite"):
        normalize_matchups(payload, retrieved_at="2026-10-02T04:00:00Z")


def test_malformed_game_row_fails_closed():
    payload = {
        "games": [
            {"contest": "cst_ok", "status": "Final", "scheduled_at": "2026-09-30T18:00:00Z",
             "home": {"team_id": "h"}, "away": {"team_id": "a"}},
            "not-an-object",
        ]
    }

    with pytest.raises(ValueError, match=r"games\[1\]_must_be_object"):
        normalize_matchups(payload, retrieved_at="2026-10-02T04:00:00Z")


def test_feature_catalog_is_valid_json_and_nonproduction():
    from pathlib import Path
    import json

    catalog = json.loads(
        Path("research/statshawk_mlb_feature_catalog.json").read_text(encoding="utf-8")
    )

    assert catalog["production_default"] is False
    assert catalog["promotion_rule"]["automatic_promotion"] is False
    assert catalog["fields"]["probable_pitcher"]["pit_status"].startswith(
        "FAIL_CLOSED"
    )


def test_duplicate_provider_views_collapse_to_one_independent_game():
    base = {
        "status": "final",
        "scheduled_at": "2026-09-30T18:00:00+00:00",
        "home_team_id": "home",
        "away_team_id": "away",
        "home_score": 3,
        "away_score": 4,
        "home_lineup_confirmed_count": 9,
        "away_lineup_confirmed_count": 9,
    }
    rows = [
        {**base, "event_id": "cst_primary", "observation_index": 0},
        {
            **base,
            "event_id": "cst_duplicate",
            "observation_index": 1,
            "home_lineup_confirmed_count": 8,
            "away_lineup_confirmed_count": 8,
        },
    ]

    collapsed = collapse_independent_games(rows)

    assert len(collapsed) == 1
    assert collapsed[0]["event_id"] == "cst_primary"
    assert collapsed[0]["collapsed_observation_count"] == 2
    assert collapsed[0]["independent_game_key"] == (
        "home|away|2026-09-30T18:00:00+00:00"
    )


def test_independent_game_identity_fails_closed_when_required_fields_are_missing():
    with pytest.raises(ValueError, match="independent_game_identity_missing"):
        collapse_independent_games(
            [
                {
                    "event_id": "cst_missing",
                    "home_team_id": "home",
                    "away_team_id": "away",
                    "scheduled_at": None,
                }
            ]
        )
