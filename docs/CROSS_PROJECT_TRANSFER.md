# Cross-Project Transfer Ledger

This document records reusable engineering patterns copied from the related prediction projects. It is a research/operations record only; it does not authorize production promotion.

## Reference snapshots

- BTC-Prediction-Research: 34a166cafbfdeb2a63a50208772e9760a2495bd9
- Stock-Daily-Prediction-3000: f63858928a738634fb26c791ae78e7135d4f7ae4
- Soccer-Prediction-Research: 6209fd98e1f50eb5dc1de9495efe3ae1f17ee325
- 7-Sport-Prediction-Research: 5918bfe2faf89381444978e46e0d61408248ecf3

## Transfers to Baseball

### BTC
1. 24h research is split into sub-6-hour stages.
2. A stateless supervisor/keeper monitors active runs and dispatches only when stale.
3. Production state is fingerprinted before research and checked after research.
4. Deterministic research failures are not converted into blind retries.
5. PIT/OOS is a first-class audit stream.

Applied to Baseball:
- .github/workflows/baseball_24h_research_autopilot.yml
- .github/workflows/baseball_24h_research_keeper.yml
- core/pit_replay.py
- research/pit_history_expansion.py

### Stock
1. 24h marathon execution is paired with a dedicated watchdog.
2. Reliability and recovery are separate from the heavy computation.

Applied to Baseball:
- separate 5-minute keeper from the 24h research workflow
- evidence-preserving closeout and circuit breaker

### Soccer
1. Opta-like data acquisition is treated as its own 24h research lane.
2. Scope frontier, data-source discovery, and recovery are separate concerns.
3. Immutable 24h checkpoints and final reconciliation are used.

Applied to Baseball:
- research/data_intelligence_candidates.py
- research/scope_discovery.py
- 24h frontier wave
- artifact-based closeout manifest

### 7-Sport
1. PIT history expansion is continuously measured.
2. Replayability and exact provenance are operational metrics, not assumptions.
3. Production watchdogs are lightweight and independent.

Applied to Baseball:
- research/pit_history_expansion.py
- competition-agnostic core.pit_snapshot
- expanded PIT tests
- lightweight 5-minute keeper

## Current principle

Reference projects can supply methods, architectures, tests and failure-handling patterns. No external project's metric result is treated as Baseball performance evidence. Baseball OOS/PIT/holdout evidence remains independent.