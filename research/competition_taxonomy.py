"""Competition/stage taxonomy for baseball research.

The classifier is metadata-only: it never uses outcomes or post-game information.
It provides a stable partition key so NPB/MLB and future international/other
competitions can be evaluated separately without forcing different datasets into
one statistical population.

Classification is fail-closed: unknown/ambiguous values become UNKNOWN rather
than being guessed into a regular-season bucket.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class CompetitionLabel:
    sport: str = "baseball"
    league: str = "UNKNOWN"
    competition: str = "unknown"
    stage: str = "unknown"
    season_type: str = "unknown"
    game_class: str = "unknown"
    competition_key: str = "baseball:unknown"
    source_field: str = ""
    source_value: str = ""
    status: str = "unknown"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_NPB_RULES: tuple[tuple[tuple[str, ...], dict[str, str]], ...] = (
    (("日本シリーズ",), {"competition": "npb_japan_series", "stage": "japan_series",
                         "season_type": "postseason", "game_class": "championship"}),
    (("クライマックスシリーズ", "クライマックス", "ＣＳ", "CS"), 
     {"competition": "npb_climax_series", "stage": "climax_series",
      "season_type": "postseason", "game_class": "playoff"}),
    (("フレッシュオールスター",), {"competition": "npb_fresh_all_star", "stage": "all_star",
                                  "season_type": "all_star", "game_class": "exhibition"}),
    (("オールスター",), {"competition": "npb_all_star", "stage": "all_star",
                         "season_type": "all_star", "game_class": "exhibition"}),
    (("ファーム日本選手権",), {"competition": "npb_farm_championship", "stage": "championship",
                              "season_type": "farm_postseason", "game_class": "championship"}),
    (("ファーム公式戦", "ファーム", "二軍"), {"competition": "npb_farm", "stage": "regular_season",
                                            "season_type": "farm", "game_class": "official"}),
    (("春季教育リーグ", "秋季教育リーグ", "教育リーグ", "教育"), 
     {"competition": "npb_education_league", "stage": "regular_season",
      "season_type": "education", "game_class": "development"}),
    (("オープン戦", "オープン"), {"competition": "npb_open", "stage": "spring_training",
                                 "season_type": "spring", "game_class": "exhibition"}),
    (("特別試合",), {"competition": "npb_special", "stage": "special",
                    "season_type": "special", "game_class": "special"}),
    (("交流戦",), {"competition": "npb_interleague", "stage": "interleague",
                  "season_type": "regular_season", "game_class": "official"}),
    (("公式戦",), {"competition": "npb_regular", "stage": "regular_season",
                  "season_type": "regular_season", "game_class": "official"}),
)


_MLB_RULES: Mapping[str, dict[str, str]] = {
    "R": {"competition": "mlb_regular", "stage": "regular_season",
          "season_type": "regular_season", "game_class": "official"},
    "S": {"competition": "mlb_spring", "stage": "spring_training",
          "season_type": "spring", "game_class": "exhibition"},
    "E": {"competition": "mlb_exhibition", "stage": "exhibition",
          "season_type": "exhibition", "game_class": "exhibition"},
    "A": {"competition": "mlb_all_star", "stage": "all_star",
          "season_type": "all_star", "game_class": "exhibition"},
    "F": {"competition": "mlb_postseason", "stage": "wild_card",
          "season_type": "postseason", "game_class": "playoff"},
    "D": {"competition": "mlb_postseason", "stage": "division_series",
          "season_type": "postseason", "game_class": "playoff"},
    "L": {"competition": "mlb_postseason", "stage": "league_championship_series",
          "season_type": "postseason", "game_class": "playoff"},
    "W": {"competition": "mlb_postseason", "stage": "world_series",
          "season_type": "postseason", "game_class": "championship"},
    "P": {"competition": "mlb_postseason", "stage": "postseason_unknown",
          "season_type": "postseason", "game_class": "playoff"},
    "C": {"competition": "mlb_championship", "stage": "championship",
          "season_type": "postseason", "game_class": "championship"},
    "I": {"competition": "mlb_intrasquad", "stage": "intrasquad",
          "season_type": "special", "game_class": "exhibition"},
    "B": {"competition": "mlb_preseason", "stage": "preseason",
          "season_type": "preseason", "game_class": "exhibition"},
}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\u3000", " ")
    return re.sub(r"\s+", " ", text)


def _build(league: str, fields: dict[str, str], source_field: str, source_value: str) -> CompetitionLabel:
    competition = fields["competition"]
    stage = fields["stage"]
    return CompetitionLabel(
        league=league,
        competition=competition,
        stage=stage,
        season_type=fields["season_type"],
        game_class=fields["game_class"],
        competition_key=f"{league}:{competition}:{stage}",
        source_field=source_field,
        source_value=source_value,
        status="classified",
    )


def classify_npb(game_type: Any) -> CompetitionLabel:
    value = _clean(game_type)
    if not value:
        return CompetitionLabel(league="NPB", source_field="game_type", status="unknown")
    for needles, fields in _NPB_RULES:
        if any(needle in value for needle in needles):
            return _build("NPB", fields, "game_type", value)
    return CompetitionLabel(
        league="NPB",
        competition="npb_unknown",
        stage="unknown",
        season_type="unknown",
        game_class="unknown",
        competition_key="NPB:npb_unknown:unknown",
        source_field="game_type",
        source_value=value,
        status="unknown",
    )


def classify_mlb(game_type: Any, series_description: Any = "") -> CompetitionLabel:
    code = _clean(game_type).upper()
    series = _clean(series_description)
    if code in _MLB_RULES:
        return _build("MLB", _MLB_RULES[code], "game_type", code)
    # A missing code plus an explicit series description is still not enough
    # to infer a postseason stage safely. Keep it unknown and preserve evidence.
    source_value = code or series
    return CompetitionLabel(
        league="MLB",
        competition="mlb_unknown",
        stage="unknown",
        season_type="unknown",
        game_class="unknown",
        competition_key="MLB:mlb_unknown:unknown",
        source_field="game_type" if code else "series_description",
        source_value=source_value,
        status="unknown",
    )


def classify_game(
    league: Any,
    *,
    game_type: Any = "",
    series_description: Any = "",
) -> CompetitionLabel:
    value = _clean(league).upper()
    if value == "NPB":
        return classify_npb(game_type)
    if value == "MLB":
        return classify_mlb(game_type, series_description)
    return CompetitionLabel(
        league=value or "UNKNOWN",
        source_field="league",
        source_value=value,
        status="unknown",
    )


def classify_record(record: Mapping[str, Any]) -> dict[str, str]:
    label = classify_game(
        record.get("league", ""),
        game_type=record.get("game_type", ""),
        series_description=record.get("series_description", ""),
    )
    return label.as_dict()
