"""Entry point executed by each GitHub Actions matrix job (one year or one league).
Output-spec fields are attached under "outputs":
  MLB: top-4 score choices (%) + Low(<=6)/High(>=7) probabilities (%)
  Soccer: top-4 score choices (%), MOM top-3 (stub), H/D/A % (regulation-time draw rule)
"""
import argparse, os, sys

from parallel.utils_io import write_json
from parallel.data_loader import load_mlb_csv, load_soccer_csv
from parallel.elo_model import run_elo_walk_forward
from parallel.soccer_elo_model import run_soccer_elo_walk_forward
from parallel.base_rate import summarize_mlb, summarize_soccer
from parallel.feature_store import apply_stages
from parallel.score_distribution import expected_runs_goals, top_k_scores
from parallel.lowhigh_model import low_high_probabilities
from parallel.soccer_outcome_model import apply_regulation_time_labels, three_way_probabilities_pct
from parallel.mom_model import predict_mom_top3
from parallel.config import MLB_SCORE_TOPK, SOCCER_SCORE_TOPK


def _mlb_outputs(df):
    home_lambda, away_lambda = expected_runs_goals(df, "home_score", "away_score")
    return {
        "score_topk_pct": top_k_scores(home_lambda, away_lambda, MLB_SCORE_TOPK),
        "low_high_pct": low_high_probabilities(home_lambda, away_lambda),
        "note": "Stage-0 Poisson estimate from season-average scoring.",
    }


def _soccer_outputs(df, soccer_result):
    df_labeled = apply_regulation_time_labels(df)
    home_lambda, away_lambda = expected_runs_goals(df_labeled, "home_score", "away_score")
    preds = soccer_result.predictions
    if len(preds):
        last = preds.iloc[-1]
        hda_pct = three_way_probabilities_pct(last["prob_home"], last["prob_draw"], last["prob_away"])
        sample_match = {"home_team": last["home_team"], "away_team": last["away_team"], "date": str(last["date"])}
    else:
        hda_pct, sample_match = None, None
    return {
        "score_topk_pct": top_k_scores(home_lambda, away_lambda, SOCCER_SCORE_TOPK),
        "sample_match_hda_pct": hda_pct,
        "sample_match": sample_match,
        "mom_top3": predict_mom_top3(player_features=None),
        "note": "HDA uses regulation-time draw rule (ET/PK winner ignored). MOM pending player data.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", required=True, choices=["mlb", "soccer"])
    parser.add_argument("--key", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", default="elo_only")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        write_json(args.out, {"sport": args.sport, "key": args.key, "status": "skipped",
                               "reason": f"data file not found: {args.data_path}"})
        print(f"[skip] {args.data_path} not found")
        return 0

    stages = [s for s in args.stages.split(",") if s]

    if args.sport == "mlb":
        df = load_mlb_csv(args.data_path)
        df = apply_stages(df, stages)
        result = run_elo_walk_forward(df)
        summary = summarize_mlb(result.predictions)
        outputs = _mlb_outputs(df)
    else:
        df = load_soccer_csv(args.data_path)
        df = apply_stages(df, stages)
        result = run_soccer_elo_walk_forward(df)
        summary = summarize_soccer(result.predictions)
        outputs = _soccer_outputs(df, result)

    write_json(args.out, {"sport": args.sport, "key": args.key, "status": "ok",
                           "stages": stages, "summary": summary, "outputs": outputs})
    print(f"[ok] {args.sport} {args.key}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
