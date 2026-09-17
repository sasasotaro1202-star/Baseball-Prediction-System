# X data source policy — research only

## Status

`RESEARCH_ONLY`. This source is **not connected to the production prediction model** and does not change production features, weights, calibration, or promotion decisions.

## Official source

- X API documentation: https://docs.x.com/
- Search Posts documentation: https://docs.x.com/x-api/posts/search/introduction
- Recent Search endpoint: `GET https://api.x.com/2/tweets/search/recent`
- X API rules/policy: https://help.x.com/en/rules-and-policies/x-api

X documents Recent Search as covering the most recent 7 days and making it available to developers; full-archive search is a separate higher-access product. citeturn2search2turn2search1

## What is collected

The collector stores public-post observations including:

- post ID
- author ID and returned public author metadata
- `created_at`
- text
- language
- public engagement metrics
- entities/context annotations when returned
- query used to retrieve the post
- `retrieved_at`
- `observed_available_at`
- payload hash

## PIT rule

`created_at` is **event time**, not historical availability evidence.

For this collector, `observed_available_at = retrieved_at` is the earliest availability boundary that this project can actually prove. Therefore:

- a post collected at 18:00 cannot be used for a 17:00 prediction cutoff;
- a post created at 12:00 but first retrieved at 18:00 is treated as unavailable before 18:00;
- current X pages are never used to infer historical availability;
- no historical X feature is eligible for the production backtest unless independent timestamp evidence is later added.

This is intentionally conservative.

## Feature policy

The first phase stores X observations only. It does **not** convert text, sentiment, engagement or author information into production features.

Any candidate feature must subsequently pass:

1. PIT-safe availability filtering.
2. Chronological/walk-forward OOS comparison against the unchanged production baseline.
3. Multiple-season/window consistency.
4. Independent holdout confirmation.
5. Calibration regression check.
6. Missing-source robustness test.
7. Reproducibility check.

If those checks fail, the X source remains research-only and is not promoted.

## Access limitation

Recent Search is intentionally used instead of assuming historical archive access. Full-archive search requires higher X API access. citeturn2search1turn2search2

## Compliance note

X data must be used according to X's current developer agreement/policy and applicable retention/deletion requirements. This research layer must not be treated as a permanent archival copy of X content without separately validating the applicable terms. citeturn2search0turn2search3
