from __future__ import annotations

import numpy as np
import pandas as pd

from research.game_script_engine import (
    PIT_STATUS, SCHEMA_VERSION, TransitionKernel, _base, _count, _state,
    _transition, boundary, canonicalize_pbp_frame, simulate_game,
)


def _toy_game(game_id="g1", date="2024-01-01", run_on=(2, "B", 1)):
    rows=[]; home=away=0; order=0
    for inning in range(1,4):
        for half in ("T","B"):
            for outs in range(3):
                order+=1; add=int((inning,half,outs)==run_on)
                if half=="T": away+=add
                else: home+=add
                rows.append({
                    "game_id":game_id,"inning":inning,"TB":half,"play_id":order,"game_date":date,
                    "home_team_id":"1","away_team_id":"2","home_team_name":"H","away_team_name":"A",
                    "home_total_runs":home,"away_total_runs":away,"addedRuns":add,"outs_when_up":outs,
                    "balls":0,"strikes":0,"on_1b":0,"on_2b":0,"on_3b":0,"batter":10,"pitcher":20,
                    "pitch_id":1,"release_speed_kmh":145.0,"plate_x":0.0,"plate_y":0.0,
                    "result_id":"1","end_time":"2024-01-01T12:00:00Z",
                })
    return pd.DataFrame(rows)


def test_missing_state_is_not_imputed():
    g=canonicalize_pbp_frame(_toy_game())
    a=g.iloc[0].to_dict(); b=g.iloc[1].to_dict()
    a["base1"]=np.nan; a["balls"]=np.nan
    assert _base(a) is None
    assert _count(a) is None
    assert _transition(a,b) is None


def test_rich_state_contains_count():
    base=canonicalize_pbp_frame(_toy_game()).iloc[0].to_dict()
    a={**base,"balls":0,"strikes":0}; b={**base,"balls":3,"strikes":2}
    assert _state(a,count=True)!=_state(b,count=True)
    assert _state(a,count=False)==_state(b,count=False)


def test_kernel_counts_only_explicit_updates():
    kernel=TransitionKernel(min_support=1)
    game=canonicalize_pbp_frame(_toy_game())
    added=kernel.add_game(game)
    assert added>0
    assert kernel.transitions==added
    assert kernel.snapshot()["games"]==1


def test_simulation_reproducible_and_normalized():
    kernel=TransitionKernel(min_support=1)
    kernel.add_game(canonicalize_pbp_frame(_toy_game()))
    a=simulate_game(kernel,home_factor=1.05,away_factor=.95,simulations=40,seed=7,max_innings=3)
    b=simulate_game(kernel,home_factor=1.05,away_factor=.95,simulations=40,seed=7,max_innings=3)
    assert a==b
    assert abs(sum(a["probabilities"][k] for k in ("home_win","draw","away_win"))-1)<1e-9
    assert a["coverage"]==1.0


def test_state_score_reconstruction_ignores_corrupted_intermediate_labels():
    raw=_toy_game()
    raw.loc[:-2,"home_total_runs"]=9
    raw.loc[:-2,"away_total_runs"]=8
    normalized=canonicalize_pbp_frame(raw)
    assert not normalized.empty
    assert int(normalized.iloc[0]["state_home_score"])==0
    assert int(normalized.iloc[0]["state_away_score"])==0


def test_jst_date_boundary_is_inclusive():
    end=boundary("2024-12-31",end_of_day=True)
    ts=pd.Timestamp("2024-12-31T18:00:00Z")
    assert ts<=end


def test_research_status_remains_non_production_without_pit_proof():
    assert SCHEMA_VERSION=="game-script-v4"
    assert PIT_STATUS.startswith("UNVERIFIABLE")
