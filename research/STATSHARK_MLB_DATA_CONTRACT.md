# StatsHawk MLB research contract

StatsHawk is available in the connected ChatGPT environment as a structured
MLB data source. In this repository it is treated as research acquisition /
secondary validation, not as the canonical MLB source.

Observed usable surfaces include:
- MLB matchup cards with contest IDs, scheduled time, probable pitchers and
  confirmed lineup entries when lineups are available.
- MLB standings.
- Final game box scores with batting and pitching lines.
- MLB play-by-play with plate appearances and pitch-level fields such as pitch
  type, velocity and exit velocity in the standard detail tier.

## PIT boundary

StatsHawk matchup responses expose probable pitchers, but the returned record
does not itself prove the historical public announcement timestamp. Therefore:

- retrieved_at is collector metadata only.
- available_at is not inferred from retrieval time or scheduled time.
- announcement_at is required for strict historical starter eligibility.
- probable-pitcher presence or cross-source agreement is insufficient by itself.
- missing or ambiguous timestamps fail closed.

## Safe research uses

Postgame box scores / PBP can be used as candidate historical observations,
but only after the repository's normal PIT maturity and provenance checks show
that the feature would have been available before the target prediction time.

Final StatsHawk scores can be used as a secondary result reconciliation check
against the canonical MLB event/result record. They do not replace the MLB Stats
API canonical ID/result path.

No production model, feature set, or promotion gate is changed by this adapter.
