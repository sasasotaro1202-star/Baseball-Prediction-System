from research.asian_games_schedule import parse_matchup_rows


def test_parse_official_asian_games_matchup():
    html = """
    <table><tr><th>Code</th><th>Visitor</th><th>vs.</th><th>Home</th></tr>
    <tr><td>9/21</td><td>12:00</td><td>BBL02</td><td>PHI</td><td>PLE</td></tr></table>
    """
    games = parse_matchup_rows(html)
    assert len(games) == 1
    assert games[0].game_id.startswith("asian-games-2026-bbl02-")
    assert games[0].date_jst == "2026-09-21"
    assert games[0].starter_evidence_status == "missing"


def test_parser_does_not_invent_starter_evidence():
    games = parse_matchup_rows(
        "<table><tr><td>9/21</td><td>18:30</td><td>BBL04</td><td>JPN</td><td>CHN</td></tr></table>"
    )
    assert games[0].starter_evidence_status == "missing"
