from __future__ import annotations

import pytest

from data.competition_registry import get


def test_np_b_contract_is_explicit_three_way():
    assert get("NPB").outcome_contract == "HOME_DRAW_AWAY"


def test_mlb_contract_is_explicit_two_way():
    assert get("MLB").outcome_contract == "HOME_AWAY"


def test_international_and_youth_contracts_are_not_silently_assumed_binary():
    for competition_id in ("WBC", "WBSC_PREMIER12", "OLYMPICS_BASEBALL", "ASIAN_GAMES_BASEBALL", "WBSC_U18", "WBSC_U23"):
        assert get(competition_id).outcome_contract == "COMPETITION_DEFINED"


def test_high_school_and_collegiate_contracts_are_not_silently_assumed_binary():
    for competition_id in ("KOSHIEN_SENBATSU", "KOSHIEN_SUMMER", "KOSHIEN_QUALIFIERS", "JAPAN_UNIVERSITY_BASEBALL"):
        assert get(competition_id).outcome_contract == "COMPETITION_DEFINED"


def test_unknown_competition_fails_closed():
    with pytest.raises(KeyError):
        get("NOT_A_REAL_COMPETITION")
