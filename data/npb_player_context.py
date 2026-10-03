"""Detailed NPB player context collector.

Official NPB player pages are used to enrich pregame snapshots with player
identity/profile and season/career statistics. The collector is intentionally
metadata-first: it does not use realized performance from the target game and
does not alter model probabilities.

Historical OOS consumption remains blocked unless the source's historical
publication/availability boundary is separately proven.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import pandas as pd

from core.http import request as http_request, session as http_session

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://npb.jp"
PLAYER_INDEX_URL = BASE_URL + "/bis/players/active/index_{suffix}.html"
HTTP_SESSION = http_session(user_agent="Baseball-Prediction-System/npb-player-context")

# Japanese active-player index suffixes. The English-name a-z indexes are
# intentionally not used as the only resolver because Japanese names also
# appear in the official Japanese kana indexes.
INDEX_SUFFIXES = (
    "a", "i", "u", "e", "o",
    "ka", "ki", "ku", "ke", "ko",
    "sa", "shi", "su", "se", "so",
    "ta", "chi", "tsu", "te", "to",
    "na", "ni", "nu", "ne", "no",
    "ha", "hi", "fu", "he", "ho",
    "ma", "mi", "mu", "me", "mo",
    "ya", "yu", "yo",
    "ra", "ri", "ru", "re", "ro",
    "wa",
)

CANONICAL_LABELS = {
    "position": ("ポジション", "Position"),
    "bats_throws": ("投打", "Bats/Throws"),
    "height_weight": ("身長／体重", "Height / Weight"),
    "birthdate": ("生年月日", "Date of Birth"),
    "career": ("経歴", "Career"),
    "draft": ("ドラフト", "Draft"),
}

TEAM_NORMALIZE = {
    "阪神": "阪神タイガース",
    "読売": "読売ジャイアンツ",
    "横浜DeNA": "横浜DeNAベイスターズ",
    "中日": "中日ドラゴンズ",
    "広島": "広島東洋カープ",
    "東京ヤクルト": "東京ヤクルトスワローズ",
    "福岡ソフトバンク": "福岡ソフトバンクホークス",
    "北海道日本ハム": "北海道日本ハムファイターズ",
    "オリックス": "オリックス・バファローズ",
    "東北楽天": "東北楽天ゴールデンイーグルス",
    "埼玉西武": "埼玉西武ライオンズ",
    "千葉ロッテ": "千葉ロッテマリーンズ",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("　", " ")).strip()


def _fetch(url: str) -> tuple[str, str]:
    response = http_request(HTTP_SESSION, url, timeout=(8, 45), retries=4)
    enc = (response.apparent_encoding or response.encoding or "utf-8").lower().replace("-", "_")
    if "shift_jis" in enc or "cp932" in enc or "shiftjis" in enc:
        body = response.content.decode("cp932", errors="strict")
    else:
        body = response.content.decode(response.apparent_encoding or response.encoding or "utf-8", errors="strict")
    return body, datetime.now(timezone.utc).isoformat()


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href = ""
        self._text: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        data = {k: v or "" for k, v in attrs}
        self._href = data.get("href", "")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._text is not None:
            value = _clean(data)
            if value:
                self._text.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._text is not None:
            self.links.append((self._href, _clean(" ".join(self._text))))
            self._href = ""
            self._text = None


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
            return
        if self._hidden:
            return
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)
            return
        if self._hidden:
            return
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(_clean(" ".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._hidden or self._cell is None:
            return
        value = _clean(data)
        if value:
            self._cell.append(value)


@dataclass(frozen=True)
class PlayerRef:
    player_id: str
    name: str
    url: str
    team: str | None
    position: str | None
    roster_source: str
    roster_available_at_utc: str


def _numeric(text: str) -> float | None:
    s = _clean(text).replace(",", "")
    if s in {"", "-", "－", "―", "N/A", "nan"}:
        return None
    s = s.replace(".2", ".2")  # preserve NPB innings notation as text below
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group(0)) if match else None


def _stat_number(text: str) -> float | None:
    s = _clean(text).replace(",", "")
    if s in {"", "-", "－", "―"}:
        return None
    # NPB innings can be 133.2 meaning 133 2/3 innings; retain a decimal
    # serialization but never use this parser to invent a derived value.
    return _numeric(s)


def _parse_personal_profile(page_html: str) -> dict[str, Any]:
    text = "\n".join(_clean(x) for x in re.findall(r">([^<>]+)<", page_html) if _clean(x))
    profile: dict[str, Any] = {}
    for label, keys in CANONICAL_LABELS.items():
        pattern = r"(?:%s)\s*[|:]\s*([^\n]+)" % "|".join(re.escape(k) for k in keys)
        m = re.search(pattern, text, re.I)
        if m:
            profile[label] = _clean(m.group(1))

    # The official page uses a compact identity block in addition to the
    # profile table. Keep the raw structured labels as evidence.
    for key, pattern in {
        "player_number": r"(?:\n|>)\s*(\d{1,3})\s*(?:\n|<)",
        "team": r"\n([^\n]+)\n[^\n]+\n[^\n]*\s*(?:投手|捕手|内野手|外野手)",
    }.items():
        if key not in profile:
            m = re.search(pattern, text)
            if m:
                profile[key] = _clean(m.group(1))
    return profile


def _header_kind(row: list[str]) -> str | None:
    joined = " ".join(row)
    if "投球回" in joined and "防御率" in joined:
        return "pitching"
    if "打席" in joined and "打率" in joined:
        return "batting"
    return None


def _parse_year_rows(page_html: str) -> dict[str, list[dict[str, Any]]]:
    parser = _TableParser()
    parser.feed(page_html)
    tables: dict[str, list[dict[str, Any]]] = {"pitching": [], "batting": []}
    current_kind: str | None = None
    headers: list[str] = []
    for row in parser.rows:
        kind = _header_kind(row)
        if kind:
            current_kind = kind
            headers = row
            continue
        if current_kind is None or not headers or not row:
            continue
        if not re.match(r"^(19|20)\d{2}$", row[0]) and row[0] not in {"通　算", "通算"}:
            continue
        values = row + [""] * max(0, len(headers) - len(row))
        record: dict[str, Any] = {}
        for i, header in enumerate(headers):
            key = _clean(header)
            if not key:
                continue
            raw = _clean(values[i])
            record[key] = raw
        # Parse only obvious numeric cells into additional *_num fields.
        for key, value in list(record.items()):
            num = _stat_number(value)
            if num is not None:
                record[f"{key}_num"] = num
        tables[current_kind].append(record)
    return tables


def _derive_advanced(record: dict[str, Any], kind: str) -> dict[str, Any]:
    def n(*names: str) -> float | None:
        for name in names:
            value = record.get(f"{name}_num")
            if value is not None:
                return float(value)
        return None

    out: dict[str, Any] = {}
    if kind == "batting":
        pa = n("打席")
        ab = n("打数")
        h = n("安打")
        hr = n("本塁打")
        bb = n("四球")
        hbp = n("死球")
        so = n("三振")
        sb = n("盗塁")
        cs = n("盗塁刺")
        doubles = n("二塁打")
        triples = n("三塁打")
        if pa and pa > 0:
            out["bb_pct"] = bb / pa if bb is not None else None
            out["k_pct"] = so / pa if so is not None else None
            out["hr_per_pa"] = hr / pa if hr is not None else None
            out["hbp_pct"] = hbp / pa if hbp is not None else None
            out["sb_attempt_rate"] = ((sb or 0.0) + (cs or 0.0)) / pa
        if ab and ab > 0:
            out["iso_from_totals"] = (
                ((doubles or 0.0) + 2.0 * (triples or 0.0) + 3.0 * (hr or 0.0)) / ab
                - ((h or 0.0) / ab)
            )
        if bb is not None and so is not None and so > 0:
            out["bb_k_ratio"] = bb / so
        if doubles is not None or triples is not None or hr is not None:
            out["extra_base_hit_rate"] = ((doubles or 0.0) + (triples or 0.0) + (hr or 0.0)) / max(h or 0.0, 1.0)
    else:
        ip = n("投球回")
        so = n("三振")
        bb = n("四球")
        hr = n("本塁打")
        h = n("安打")
        hbp = n("死球")
        tbf = n("打者")
        if ip and ip > 0:
            out["k9"] = 9.0 * so / ip if so is not None else None
            out["bb9"] = 9.0 * bb / ip if bb is not None else None
            out["hr9"] = 9.0 * hr / ip if hr is not None else None
            if h is not None and bb is not None:
                out["whip"] = (h + bb) / ip
        if tbf and tbf > 0:
            out["k_pct"] = so / tbf if so is not None else None
            out["bb_pct"] = bb / tbf if bb is not None else None
            out["hbp_pct"] = hbp / tbf if hbp is not None else None
        if so is not None and bb is not None:
            out["k_minus_bb"] = so - bb
    return {k: v for k, v in out.items() if v is not None}


def build_index() -> dict[str, list[PlayerRef]]:
    refs: dict[str, list[PlayerRef]] = {}
    for suffix in INDEX_SUFFIXES:
        url = PLAYER_INDEX_URL.format(suffix=suffix)
        try:
            body, observed = _fetch(url)
        except Exception:
            continue
        parser = _LinkParser()
        parser.feed(body)
        for href, text in parser.links:
            m = re.search(r"/bis/players/(\d+)\.html$", href)
            if not m or not text:
                continue
            absolute = urljoin(BASE_URL, href)
            player_name = _clean(text)
            team = None
            position = None
            # Index text often includes uniform number, position and team.
            raw = re.sub(r"^\d{1,3}\s*", "", player_name)
            for marker, pos in (
                ("投手", "投手"), ("捕手", "捕手"), ("内野手", "内野手"), ("外野手", "外野手"),
            ):
                if marker in raw:
                    position = pos
                    break
            ref = PlayerRef(
                player_id=m.group(1),
                name=raw.split(" (")[0].strip(),
                url=absolute,
                team=None,
                position=position,
                roster_source=url,
                roster_available_at_utc=observed,
            )
            refs.setdefault(ref.name, []).append(ref)
    return refs


def resolve_player_refs(names: list[str], refs: dict[str, list[PlayerRef]] | None = None) -> dict[str, PlayerRef]:
    index = refs if refs is not None else build_index()
    out: dict[str, PlayerRef] = {}
    for name in names:
        needle = _clean(name)
        exact = index.get(needle, [])
        if len(exact) == 1:
            out[needle] = exact[0]
            continue
        candidates = []
        for candidate_name, rows in index.items():
            if needle and (needle in candidate_name or candidate_name in needle):
                candidates.extend(rows)
        unique = {row.player_id: row for row in candidates}
        if len(unique) == 1:
            out[needle] = next(iter(unique.values()))
    return out


def collect_player(name: str, ref: PlayerRef) -> dict[str, Any]:
    body, observed = _fetch(ref.url)
    profile = _parse_personal_profile(body)
    tables = _parse_year_rows(body)
    current_year = tables["pitching"] + tables["batting"]
    current = [r for r in current_year if r.get("年度") == "2026"]
    if not current:
        current = [r for r in current_year if str(r.get("年度", "")).startswith("2026")]
    pitch = next((r for r in current if _header_kind(list(r.keys())) == "pitching"), None)
    # _header_kind cannot operate on parsed dict keys reliably, so identify from
    # known column names instead.
    pitching_row = next((r for r in tables["pitching"] if str(r.get("年度")) == "2026"), None)
    batting_row = next((r for r in tables["batting"] if str(r.get("年度")) == "2026"), None)
    result = {
        "player_id": ref.player_id,
        "name_requested": name,
        "official_name": profile.get("official_name") or ref.name,
        "player_page": ref.url,
        "profile": profile,
        "current_season_2026": {
            "pitching": pitching_row,
            "batting": batting_row,
        },
        "career": {
            "pitching": next((r for r in tables["pitching"] if r.get("年度") in {"通　算", "通算"}), None),
            "batting": next((r for r in tables["batting"] if r.get("年度") in {"通　算", "通算"}), None),
        },
        "source": {
            "source_id": "npb_official_player_page",
            "url": ref.url,
            "available_at_utc": observed,
            "retrieved_at_utc": observed,
            "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        },
    }
    if pitching_row:
        result["current_season_2026"]["pitching_advanced"] = _derive_advanced(pitching_row, "pitching")
    if batting_row:
        result["current_season_2026"]["batting_advanced"] = _derive_advanced(batting_row, "batting")
    return result


def collect_players(names: list[str]) -> dict[str, Any]:
    refs = resolve_player_refs(names)
    players = []
    unresolved = []
    for name in names:
        ref = refs.get(_clean(name))
        if ref is None:
            unresolved.append(name)
            continue
        try:
            players.append(collect_player(name, ref))
        except Exception as exc:
            unresolved.append(name)
            players.append({
                "player_id": ref.player_id,
                "name_requested": name,
                "player_page": ref.url,
                "source": {
                    "source_id": "npb_official_player_page",
                    "status": "SOURCE_FAILED",
                    "available_at_utc": None,
                    "retrieved_at_utc": None,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            })
    snapshot = {
        "schema_version": "npb-player-context-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "players_requested": names,
        "players_resolved": len(players),
        "players_unresolved": unresolved,
        "historical_oos_consumption": "BLOCKED_UNLESS_HISTORICAL_AVAILABILITY_PROVEN",
        "players": players,
    }
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_id"] = hashlib.sha256(canonical).hexdigest()
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    snapshot = collect_players(args.players)
    out = Path(args.out) if args.out else ROOT / "results" / "npb_player_context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "EXECUTED",
        "snapshot_id": snapshot["snapshot_id"],
        "requested": len(args.players),
        "resolved": snapshot["players_resolved"],
        "unresolved": snapshot["players_unresolved"],
        "output": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
