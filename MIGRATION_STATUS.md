# Migration status

Source: `sasasotaro1202-star/Baseball`, branch `v4.4-baseball-integration`.

Target: `sasasotaro1202-star/Baseball-Prediction-System`.

Policy:
- Baseball only; the original migration baseline is NPB + MLB.
- The target system now defines an expanded competition scope: NPB, MLB, other reliable professional leagues, Olympics, Asian Games, WBC, WBSC Premier12, WBSC U18/U23 and other age-group competitions, Japanese high-school baseball including Koshien and qualifiers, university/intercollegiate baseball, and other material domestic/international competitions when reliable PIT data exists.
- Expanded competitions are not automatically production-eligible. Each competition requires source provenance, PIT-safe availability/announcement evidence, rule/outcome-contract isolation, sufficient walk-forward OOS and protected holdout evidence, calibration, robustness, and promotion-gate approval.
- Unsupported or insufficiently historical competitions must remain `RESEARCH_ONLY` or `UNAVAILABLE`; they must never be silently mixed into production training or evaluation.
- Preserve the legacy repository unchanged.
- Do not copy Soccer-only workflows/data.
- Do not run heavy backtests during migration.
- Preserve PIT, prediction, evaluation, research, monitoring, and tests.
- For current/future MLB availability, Yahoo! Sports Navi (`baseball.yahoo.co.jp/mlb`) may be used as a secondary/operational source for schedules, probable starters, results, standings, and game pages. It must not be treated as proof of historical official starter-announcement time unless the captured evidence actually establishes `announcement_at <= prediction_cutoff`.
