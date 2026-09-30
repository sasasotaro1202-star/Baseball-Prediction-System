from __future__ import annotations

import pytest

from data.competition_registry import COMPETITIONS, get, production_eligible


def test_registry_contains_required_competitions() -> None:
    ids = {spec.competition_id for spec in COMPETITIONS}
    assert {
        "NPB",
        "MLB",
        "WBC",
        "WBSC_PREMIER12",
        "OLYMPICS_BASEBALL",
        "ASIAN_GAMES_BASEBALL",
        "WBSC_U18",
        "WBSC_U23",
        "KOSHIEN_SENBATSU",
        "KOSHIEN_SUMMER",
        "KOSHIEN_QUALIFIERS",
        "JAPAN_UNIVERSITY_BASEBALL",
    } <= ids


def test_registry_never_promotes_by_listing_alone() -> None:
    # NPB is fail-closed until explicit candidate/holdout adoption evidence.
    assert production_eligible("NPB") is False
    for competition_id in (
        "MLB",
        "WBC",
        "WBSC_PREMIER12",
        "OLYMPICS_BASEBALL",
        "ASIAN_GAMES_BASEBALL",
        "WBSC_U18",
        "WBSC_U23",
        "KOSHIEN_SENBATSU",
        "KOSHIEN_SUMMER",
        "KOSHIEN_QUALIFIERS",
        "JAPAN_UNIVERSITY_BASEBALL",
    ):
        assert production_eligible(competition_id) is False


def test_league_and_non_league_phases_are_explicit() -> None:
    npb = get("NPB")
    assert npb.phase_type == "league"
    assert npb.rules_profile == "npb"
    assert get("ASIAN_GAMES_BASEBALL").phase_type == "tournament"
    assert get("WBC").phase_type == "tournament"
    assert get("KOSHIEN_QUALIFIERS").phase_type == "qualifier"
    assert get("JAPAN_UNIVERSITY_BASEBALL").phase_type == "league_and_tournament"


def test_unknown_competition_fails_closed() -> None:
    with pytest.raises(KeyError):
        get("UNKNOWN_COMPETITION")



def test_competition_strategy_profiles_are_phase_specific():
    from research.competition_strategy import strategy_for, eligible_for_competition_calibration

    regular = strategy_for("NPB:npb_regular:regular_season")
    interleague = strategy_for("NPB:npb_interleague:interleague")
    postseason = strategy_for("NPB:npb_climax_series:climax_series")
    tournament = strategy_for("WBC:wbc:tournament")
    unknown = strategy_for("NPB:unknown:unknown")

    assert regular.strategy_id == "league_adaptive_ensemble"
    assert interleague.strategy_id == "league_adaptive_ensemble"
    assert postseason.strategy_id == "postseason_shrunk_ensemble"
    assert tournament.strategy_id == "tournament_shrunk_ensemble"
    assert unknown.strategy_id == "unknown_fail_closed"
    assert eligible_for_competition_calibration(regular, regular.specialist_min_validation_rows)
    assert not eligible_for_competition_calibration(unknown, 999999)
