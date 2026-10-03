#!/usr/bin/env bash
set -euo pipefail

# PIT-safe pregame automation logic extracted from the workflow wrapper.
# The workflow remains intentionally small so GitHub Actions can parse and
# register its workflow_dispatch/schedule triggers reliably.
export GH_TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

echo "=== Baseball 60m Pregame Auto Prediction ==="

set -euo pipefail
python -m py_compile prediction/pregame_scheduler.py
python prediction/pregame_scheduler.py             --min-lead-minutes 50             --preferred-lead-minutes 60             --scan-ahead-minutes 60             --prediction-source AUTO_60M > pregame_scheduler.json
cat pregame_scheduler.json
python - <<'PY'
import json
from pathlib import Path
obj = json.loads(Path("pregame_scheduler.json").read_text(encoding="utf-8"))
if obj.get("status") not in {"DUE", "NO_DUE_GAMES"}:
    raise SystemExit("unexpected scheduler status: " + str(obj.get("status")))
for row in obj.get("due_games", []):
    lead = float(row["lead_minutes"])
    if not 50.0 < lead <= 60.0:
        raise SystemExit("automatic scheduler selected a game outside the 50-60 minute window")
    if row.get("prediction_source") != "AUTO_60M":
        raise SystemExit("automatic scheduler emitted an unexpected prediction source")
    if lead <= 0.0:
        raise SystemExit("scheduler selected a game at/after first pitch")
due_dates = list(obj.get("due_dates", []))
research_shadow_due_dates = list(obj.get("research_shadow_due_dates", []))
research_due = list(obj.get("research_due_games", []))
if any(row.get("status") == "RESEARCH_DUE" for row in obj.get("due_games", [])):
    raise SystemExit("research candidate leaked into production due_games")
if any(row.get("prediction_eligibility") != "RESEARCH_ONLY_BLOCKED_UNTIL_OFFICIAL_STARTERS" for row in research_due):
    raise SystemExit("research candidate has unsafe production eligibility")
all_due_dates = sorted(set(due_dates) | set(research_shadow_due_dates))
Path("due_dates.txt").write_text(
    "\n".join(str(x) for x in all_due_dates) + ("\n" if all_due_dates else ""),
    encoding="utf-8",
)
Path("research_shadow_due_dates.txt").write_text(
    "\n".join(str(x) for x in research_shadow_due_dates)
    + ("\n" if research_shadow_due_dates else ""),
    encoding="utf-8",
)
print(
    "scope separation: PASS; production_due=%d research_due=%d"
    % (len(obj.get("due_games", [])), len(research_due))
)
PY

set -euo pipefail
if [ ! -s due_dates.txt ]; then
  echo "No due pregame games; skipping heavy prediction environment."
  exit 0
fi
python -m pip install --disable-pip-version-check --retries 10 --timeout 120 --prefer-binary -r requirements.txt

set -euo pipefail
if [ ! -s due_dates.txt ]; then
  exit 0
fi
mkdir -p data
date="$(head -n 1 due_dates.txt)"
year="$(echo "$date" | cut -d- -f1)"
month="$(echo "$date" | cut -d- -f2)"
month_num=$((10#$month))
if [ "$month_num" -le 2 ]; then
  exit 0
fi
for m in $(seq 2 $((month_num - 1))); do
  mm=$(printf "%02d" "$m")
  f="data/$year-$mm"_pbp.csv
  if [ -s "$f" ]; then
    continue
  fi
  gh release download pbp --repo armstjc/Nippon-Baseball-Data-Repository               --pattern "$year-$mm"_pbp.csv --dir data --clobber
  test -s "$f"
done

set -euo pipefail
if [ ! -s due_dates.txt ]; then
  exit 0
fi
while IFS= read -r date; do
  [ -n "$date" ] || continue
  echo "=== pregame production/shadow prediction: $date ==="
  if [ -s research_shadow_due_dates.txt ] && grep -qxF "$date" research_shadow_due_dates.txt; then
    python production_npb.py --date "$date" --data-dir data --pregame-only --research-shadow
    output="results/npb_shadow_$date.json"
  else
    python -m prediction.current_production --league NPB --date "$date" --data-dir data --pregame-only
    output="results/npb_production_$date.json"
  fi
  test -s "$output"
  python - "$output" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = obj.get("execution_status")
if status not in {"EXECUTED", "RESEARCH_SHADOW_EXECUTED", "BLOCKED_STARTERS", "NO_DUE_PREGAME_GAMES"}:
    raise SystemExit("unexpected prediction status: " + str(status))
if status not in {"EXECUTED", "RESEARCH_SHADOW_EXECUTED"}:
    print(json.dumps({"status": status, "block_reason": obj.get("block_reason")}, ensure_ascii=False))
    raise SystemExit(0)
for pred in obj.get("predictions", []):
    cutoff = datetime.fromisoformat(pred["prediction_cutoff_utc"].replace("Z", "+00:00"))
    generated = datetime.fromisoformat(pred["prediction_generated_at"].replace("Z", "+00:00"))
    observed = datetime.fromisoformat(pred["starter_evidence_observed_at_utc"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(pred["datetime_jst"].replace("Z", "+00:00"))
    if generated >= start:
        raise SystemExit("prediction generated at/after first pitch")
    if observed > cutoff:
        raise SystemExit("starter evidence observed after prediction information cutoff")
    if cutoff >= start:
        raise SystemExit("prediction information cutoff is not pregame")
    if pred.get("pit_status") != "PASS":
        raise SystemExit("prediction is not PIT PASS")
print("Pregame PIT validation passed; automatic target is approximately 60m before first pitch.")
PY
done < due_dates.txt

set -euo pipefail
if [ ! -s due_dates.txt ]; then
  exit 0
fi
python - <<'PY'
import json
from pathlib import Path
for path in sorted(list(Path("results").glob("npb_production_*.json")) + list(Path("results").glob("npb_shadow_*.json"))):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("execution_status") != "EXECUTED":
        continue
    for pred in obj.get("predictions", []):
        pred["prediction_source"] = "AUTO_60M"
        pred["prediction_schedule"] = "scheduled"
        pred["prediction_target_lead_minutes"] = 60.0
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

set -euo pipefail
if [ ! -s due_dates.txt ]; then
  exit 0
fi
while IFS= read -r date; do
  [ -n "$date" ] || continue
  output="results/npb_production_$date.json"
  if [ ! -s "$output" ]; then
    output="results/npb_shadow_$date.json"
  fi
  if [ -s "$output" ]; then
    if grep -q '"RESEARCH_SHADOW_EXECUTED"' "$output"; then
      python -m research.shadow_experience --archive "$output" --run-id "$GITHUB_RUN_ID"
    else
      python -m research.experience_ledger --archive "$output" --run-id "$GITHUB_RUN_ID"
    fi
  fi
done < due_dates.txt

set -euo pipefail
if [ ! -d data/experience/predictions ] && [ ! -d data/experience/research_shadow ]; then
  exit 0
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
for experience_path in data/experience/predictions data/experience/research_shadow; do
  if [ -d "$experience_path" ]; then
    git add "$experience_path"
  fi
done
if git diff --cached --quiet; then
  echo "No new experience snapshots to commit."
  exit 0
fi
for attempt in 1 2 3; do
  git fetch origin main
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
    git reset --hard origin/main
  fi

  # Re-archive the immutable prediction output after every push race
  # recovery. We never rebase generated prediction JSONL state.
  while IFS= read -r date; do
    [ -n "$date" ] || continue
    output="results/npb_production_$date.json"
    if [ ! -s "$output" ]; then
      output="results/npb_shadow_$date.json"
    fi
    if [ -s "$output" ]; then
      if grep -q '"RESEARCH_SHADOW_EXECUTED"' "$output"; then
        python -m research.shadow_experience --archive "$output" --run-id "$GITHUB_RUN_ID"
      else
        python -m research.experience_ledger --archive "$output" --run-id "$GITHUB_RUN_ID"
      fi
    fi
  done < due_dates.txt

  for experience_path in data/experience/predictions data/experience/research_shadow; do
  if [ -d "$experience_path" ]; then
    git add "$experience_path"
  fi
done
  if git diff --cached --quiet; then
    echo "No new experience snapshots to commit."
    exit 0
  fi
  git commit -m "Archive 60m pregame prediction experience"
  if git push origin HEAD:main; then
    exit 0
  fi
done
echo "bounded experience push recovery exhausted" >&2
exit 1
