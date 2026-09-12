"""Central configuration for the parallel matrix pipeline."""

MLB_YEARS = list(range(2015, 2027))

SOCCER_LEAGUES = [
    "premier_league", "bundesliga", "serie_a", "la_liga", "ligue_1",
    "eredivisie", "j1", "j2", "j3", "ucl", "uel", "dfb_pokal", "club_friendly",
]

ELO_K = 20.0
ELO_HOME_ADVANTAGE = 24.0
ELO_START_RATING = 1500.0

SOCCER_DRAW_BASE = 0.28
SOCCER_DRAW_DECAY = 0.00045

TARGET_ACCURACY = 0.75
CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# --- Output spec (prediction_model_output_spec.md) ---
SOCCER_SCORE_TOPK = 4
MLB_SCORE_TOPK = 4
MOM_TOPK = 3
LOWHIGH_THRESHOLD = 6
