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
    assert production_eligible("NPB") is True
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
