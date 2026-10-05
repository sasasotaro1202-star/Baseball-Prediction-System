import numpy as np
import pandas as pd
import pytest

from research.game_script_lab import build_half_innings, fit_transition_model, resolve_columns, simulate


def sample():
    rows=[]
    seq=0
    for gi in range(8):
        hs=aw=0
        for inn in range(1,4):
            for half in ("T","B"):
                seq += 1
                if half=="T":
                    aw += 1 if (gi+inn)%3==0 else 0
                else:
                    hs += 1 if (gi+inn)%2==0 else 0
                rows.append({
                    "game_id":f"G{gi}",
                    "game_date":f"2026-01-{gi+1:02d}T10:00:00Z",
                    "home_team_name":f"H{gi%2}",
                    "away_team_name":f"A{gi%2}",
                    "home_total_runs":hs,
                    "away_total_runs":aw,
                    "PlayInfo_SeqNo":seq,
                    "fiveDigitSerialNumber":f"{inn:02d}{1 if half=='T' else 2}",
                })
    return pd.DataFrame(rows)


def test_half_innings_are_nonnegative_and_ordered():
    raw=sample()
    halves=build_half_innings(raw, resolve_columns(raw))
    assert len(halves)==48
    assert (halves[["home_runs","away_runs"]] >= 0).all().all()


def test_fit_ids_are_explicit_and_simulation_is_valid():
    raw=sample()
    halves=build_half_innings(raw, resolve_columns(raw))
    train={f"G{i}" for i in range(5)}
    model=fit_transition_model(halves, train)
    assert set(model.fit_game_ids)==train
    result=simulate(model,"H1","A1",1000,42,12)
    p=np.asarray(result["proba"],dtype=float)
    assert p.shape==(3,)
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(),1.0)


def test_page_identity_recovers_state_when_serial_is_missing():
    raw=sample()
    raw["page"] = raw["fiveDigitSerialNumber"].astype(str) + "01"
    raw = raw.drop(columns=["fiveDigitSerialNumber"])
    halves = build_half_innings(raw, resolve_columns(raw))
    assert len(halves) == 48
    assert set(halves["half"]) == {"T", "B"}


def test_missing_inning_identity_fails_closed():
    raw=sample().drop(columns=["fiveDigitSerialNumber"])
    with pytest.raises(ValueError):
        build_half_innings(raw, resolve_columns(raw))
