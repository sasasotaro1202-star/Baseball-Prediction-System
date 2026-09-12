"""Man-of-the-Match (MOM) 3-choice prediction. Returns insufficient_data
until player-level data (sections 8-11 of prediction_model_specification.md)
is uploaded -- never fabricates a MOM prediction from team-level data alone."""
from parallel.config import MOM_TOPK


def predict_mom_top3(player_features=None):
    if not player_features:
        return {
            "status": "insufficient_data",
            "reason": "No player-level data (goals/assists/xG/rating) uploaded yet.",
            "candidates": [],
        }
    ranked = sorted(player_features, key=lambda p: p.get("score_proxy", 0), reverse=True)
    top = ranked[:MOM_TOPK]
    return {
        "status": "ok",
        "candidates": [
            {"player_id": p.get("player_id"), "probability_pct": round(p.get("prob", 0) * 100, 2)}
            for p in top
        ],
    }
