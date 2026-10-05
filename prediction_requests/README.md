# User Prediction Request Queue

This directory is the controlled GitHub request/response lane for user-requested baseball game predictions.

## Request

Create or replace `prediction_requests/inbox/active.json` with:

```json
{
  "schema_version": "baseball-prediction-request-v1",
  "request_id": "safe-unique-id",
  "competition_id": "NPB",
  "target_date": "today",
  "requested_at_utc": "2026-10-04T00:00:00Z"
}
```

Use the canonical competition id from `data/competition_registry.py`. The assistant must resolve the user's natural-language request into this explicit competition/date/game request before writing it.

Optional request fields may include `game_id`, `home_team`, `away_team`, `force_refresh`, and other deterministic filters defined by the request schema.

## Generation

A push to `inbox/active.json` triggers `.github/workflows/baseball-user-prediction-request.yml`.

The router selects:

`CURRENT_PRODUCTION_RUNTIME -> VALIDATED_RESEARCH_SHADOW -> COMPETITION_SPECIFIC_RESEARCH_RUNTIME -> BLOCKED/UNAVAILABLE`.

The request-generation workflow never promotes research to production and never bypasses PIT/eligibility gates.

## Automatic recovery
A user request must not stop at a blocked preferred runtime. The repository router may escalate from the preferred production lane to the validated research-shadow lane when the preferred lane produces no prediction for a known operational reason. The NPB fast lane additionally escalates to the starter-uncertainty-aware daily research implementation and, when that fails, to a PIT-safe emergency direct-run-rate research fallback.

All fallback outputs are explicitly research-only. Production gates, PIT rules and adoption status are unchanged. If every registered lane fails, the result must be recorded as GENERATION_FAILED/UNAVAILABLE with failure evidence; no independent forecast may be substituted silently.

## Result

The immutable result is written to:

`prediction_requests/results/<request_id>.json`

It contains the request fingerprint, target competition/date, source repository, source commit, generation lane/status, generation time, and the underlying repository prediction output.

A missing result is not interpreted as a successful prediction. The assistant must verify the result artifact and its prediction/PIT status before presenting it.
