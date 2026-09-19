# Global Baseball Competition Inventory

Status: research registry, not production support.

Purpose:
- Enumerate official international baseball competition families and major professional/domestic leagues that are candidates for this prediction system.
- Separate "competition exists" from "implemented in the repository".
- Record the minimum evidence required before promotion to production prediction.
- Prevent silently treating a competition as supported when it is not.

## Status definitions

- IMPLEMENTED: repository has an explicit acquisition/backtest path.
- CANDIDATE: competition is in scope for research but is not implemented.
- RESEARCH_ONLY: data may exist, but sample size/PIT/starter evidence is currently expected to be insufficient for production.
- EXCLUDED: different sport/format or outside the baseball prediction objective.

Data-availability rating:
- A = routinely obtainable game-level schedule/results and plausible pregame metadata.
- B = game-level results are obtainable, but historical depth, probable/official starters, or PIT evidence needs work.
- C = partial/irregular game-level data or difficult historical PIT reconstruction.
- D = insufficient evidence for a reliable production pipeline.

Important:
- A rating is not a claim that the repository currently ingests the competition.
- "Starter data" means evidence that can support the project's strict prediction cutoff rule; a current probable-pitcher field is not automatically historical PIT-safe.
- For international youth/short tournaments, player/roster churn and small samples require competition-specific validation.
- Baseball5 and softball are excluded from the ordinary baseball prediction model.

## 1. Global / international baseball

| Competition family | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| World Baseball Classic | Senior | Men | Global | International | National teams, pools/knockout | ~4-year cycle | High | A | B | B | CANDIDATE |
| WBC Qualifiers | Senior | Men | Global | International | Qualifying tournament | Cycle-linked | Medium | B | C | C | CANDIDATE |
| WBSC Premier12 | Senior | Men | Global | International | National teams, group/super rounds | ~4-year cycle | Medium/High | A | B | B | CANDIDATE |
| Premier12 Qualifiers | Senior | Men | Global | International | Qualifying groups | Cycle-linked | Low/Medium | B | C | C | CANDIDATE |
| Olympic Baseball | Senior | Men | Global | International | Olympic tournament | 4-year cycle when baseball is included | Medium | B | B | B | CANDIDATE |
| Olympic Baseball Qualifiers | Senior | Men | Global | International | Qualifying tournament | Cycle-linked | Medium | B | C | C | CANDIDATE |
| WBSC U-23 Baseball World Cup | U23 | Men | Global | International | Groups + Super Round/placement | ~2-year cycle | Medium | A/B | B/C | C | CANDIDATE |
| WBSC U-18 Baseball World Cup | U18 | Men | Global | International | Groups + Super Round/placement | ~2-year cycle | High | A/B | B/C | C | CANDIDATE |
| WBSC U-15 Baseball World Cup | U15 | Men | Global | International | Groups + finals/placement | ~2-year cycle | Medium | A/B | C | C | CANDIDATE |
| WBSC U-12 Baseball World Cup | U12 | Mixed/men's pathway | Global | International | Groups + Super Round/placement | ~2-year cycle | Medium | A/B | C | C | CANDIDATE |
| WBSC Women's Baseball World Cup | Senior | Women | Global | International | Groups + finals | Multi-year cycle | Medium | B | B/C | C | CANDIDATE |

WBSC's published calendars and event pages confirm the U12/U15/U18/U23 World Cups, Women's Baseball World Cup, Premier12 and other global events.

## 2. Asia / Baseball Federation of Asia

| Competition family | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Asian Baseball Championship | Senior | Men | Asia | International | National teams | Periodic | Medium/High | A/B | B/C | C | CANDIDATE |
| U23 Asian Baseball Championship | U23 | Men | Asia | International | National teams | Periodic | Medium | B | C | C | CANDIDATE |
| U18 Asian Baseball Championship | U18 | Men | Asia | International | National teams | Periodic | High | A/B | C | C | CANDIDATE |
| U15 Asian Baseball Championship | U15 | Men | Asia | International | National teams | Periodic | Medium | A/B | C | C | CANDIDATE |
| U12 Asian Baseball Championship | U12 | Men | Asia | International | National teams | Periodic | Medium | A/B | C | C | CANDIDATE |
| Asian Games baseball | Senior | Men | Asia | International / multi-sport | National teams, tournament | 4-year cycle | Medium/High | B | B/C | C | CANDIDATE |
| Asian Games youth/age-category baseball where sanctioned | Various | Various | Asia | International | Tournament | Event-dependent | Low | C | C | D | RESEARCH_ONLY |

BFA regulations explicitly reference U12, U15, U18, U23 and senior championships, and WBSC/BFA event reports expose game schedules, teams and scores.

## 3. Europe

| Competition family | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| European Baseball Championship | Senior | Men | Europe | International | National teams, pools/knockout | Periodic | High | A/B | B/C | C | CANDIDATE |
| European Championship U23 | U23 | Men | Europe | International | National teams | Periodic | Medium | B | C | C | CANDIDATE |
| European Championship U18 | U18 | Men | Europe | International | National teams | Periodic | High | B | C | C | CANDIDATE |
| European Championship U15 | U15 | Men | Europe | International | National teams | Periodic | Medium | B | C | C | CANDIDATE |
| European Championship U12 | U12 | Men | Europe | International | National teams | Periodic | Medium | B | C | C | CANDIDATE |
| European Baseball Championship B-Pools | Various | Various | Europe | International | Promotion/relegation pools | Periodic | Medium | B/C | C | C | CANDIDATE |
| European Championship Women | Senior | Women | Europe | International | National teams | Periodic | Medium | B/C | C | C | CANDIDATE |
| European club championships / Champions Cup | Senior | Men | Europe | International club | Club tournament | Annual | Medium | B | B/C | C | CANDIDATE |
| European Cup / Federations Cup families | Senior | Men | Europe | International club | Club tournament | Annual | Medium | B/C | C | C | RESEARCH_ONLY |

WBSC Europe competition regulations explicitly cover senior, U23, U18, U15 and U12 championships and B-pools; recent competition planning also lists these categories.

## 4. Americas

| Competition family | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WBSC Americas U12 qualifier / Pan American U12 pathway | U12 | Men | Americas | International | Qualifier | Cycle-linked | Medium | B | C | C | CANDIDATE |
| WBSC Americas U15 qualifier / Pan American U15 pathway | U15 | Men | Americas | International | Qualifier | Cycle-linked | Medium | B | C | C | CANDIDATE |
| WBSC Americas U18 qualifier / Pan American U18 pathway | U18 | Men | Americas | International | Qualifier | Cycle-linked | Medium/High | B | C | C | CANDIDATE |
| WBSC Americas U23 qualifier / Pan American U23 pathway | U23 | Men | Americas | International | Qualifier | Cycle-linked | Medium | B | C | C | CANDIDATE |
| Pan American Games baseball | Senior | Men | Americas | International / multi-sport | National teams | 4-year cycle | Medium/High | B | B/C | C | CANDIDATE |
| Caribbean / Central American / regional baseball events | Various | Various | Americas | International | National teams | Event-dependent | Low/Medium | C | C | D | RESEARCH_ONLY |

WBSC's current competition reporting shows Americas qualifiers feeding U12 and U18 World Cups, while the 2026 U12 Americas qualifier is scheduled as a 12-team event.

## 5. Oceania

| Competition family | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Oceania U12 qualifier | U12 | Men | Oceania | International | Regional qualifier | Cycle-linked | Low | C | C | D | RESEARCH_ONLY |
| Oceania U15 qualifier | U15 | Men | Oceania | International | Regional qualifier | Cycle-linked | Low | C | C | D | RESEARCH_ONLY |
| Oceania U18 qualifier | U18 | Men | Oceania | International | Regional qualifier | Cycle-linked | Low | C | C | D | RESEARCH_ONLY |
| Oceania U23 qualifier | U23 | Men | Oceania | International | Regional qualifier | Cycle-linked | Low | C | C | D | RESEARCH_ONLY |
| Oceania senior/open events | Senior | Various | Oceania | International | Regional | Event-dependent | Low | C | C | D | RESEARCH_ONLY |

WBSC Oceania's 2026-2029 planning explicitly lists U12/U18 qualifiers and later U23/U15 qualifiers.

## 6. Major professional / senior domestic leagues

| League | Age | Gender | Region | International/domestic | Format | Frequency | Historical data | Game-level data | Starter data | PIT potential | System status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLB | Senior | Men | North America | Domestic league | Regular season + postseason | Annual | High | A | A/B | A/B | IMPLEMENTED |
| NPB | Senior | Men | Japan | Domestic league | Regular season + postseason | Annual | High | A | A/B | A/B | IMPLEMENTED |
| KBO League | Senior | Men | Korea | Domestic league | Regular season + postseason | Annual | High | A/B | A/B | B | CANDIDATE |
| CPBL | Senior | Men | Chinese Taipei | Domestic league | Regular season + postseason | Annual | High | A/B | A/B | B | CANDIDATE |
| LMB | Senior | Men | Mexico | Domestic league | Regular season + postseason | Annual | High | B | B/C | C | CANDIDATE |
| Dominican League (LIDOM) | Senior | Men | Dominican Republic | Domestic/winter league | Regular season + postseason | Annual | Medium/High | B | B/C | C | CANDIDATE |
| Puerto Rico (LBPRC) | Senior | Men | Puerto Rico | Domestic/winter league | Regular season + postseason | Annual | Medium | B | B/C | C | CANDIDATE |
| Venezuelan League (LVBP) | Senior | Men | Venezuela | Domestic/winter league | Regular season + postseason | Annual | High | B | B/C | C | CANDIDATE |
| Puerto Rico / Caribbean Series | Senior | Men | Caribbean | International club | Tournament | Annual | High | B | B/C | C | CANDIDATE |
| Cuban Serie Nacional | Senior | Men | Cuba | Domestic league | League + postseason | Annual | High historically | C/B | C | C | RESEARCH_ONLY |
| Australian Baseball League | Senior | Men | Oceania | Domestic league | Regular season + postseason | Annual | Medium | B | B/C | C | CANDIDATE |
| Italian Baseball League / Serie A | Senior | Men | Italy | Domestic league | League | Annual | Medium | B/C | C | C | RESEARCH_ONLY |
| Dutch Honkbal Hoofdklasse | Senior | Men | Netherlands | Domestic league | League + postseason | Annual | Medium | B/C | C | C | RESEARCH_ONLY |
| Czech Extraliga | Senior | Men | Czechia | Domestic league | League + postseason | Annual | Medium | B/C | C | C | RESEARCH_ONLY |
| Spanish División de Honor | Senior | Men | Spain | Domestic league | League | Annual | Medium | C | C | C | RESEARCH_ONLY |
| French Division 1 | Senior | Men | France | Domestic league | League + postseason | Annual | Medium | C | C | C | RESEARCH_ONLY |
| German Bundesliga | Senior | Men | Germany | Domestic league | League + postseason | Annual | Medium | C | C | C | RESEARCH_ONLY |
| British Baseball leagues | Senior | Men | UK | Domestic league | League | Annual | Low/Medium | C | C | D | RESEARCH_ONLY |
| Nicaragua LBPN | Senior | Men | Nicaragua | Domestic/winter league | League + postseason | Annual | Medium | C/B | C | C | RESEARCH_ONLY |
| Colombia LPB | Senior | Men | Colombia | Domestic league | League + postseason | Annual | Medium | C | C | C | RESEARCH_ONLY |
| Panama Probeis | Senior | Men | Panama | Domestic/winter league | League | Annual | Low/Medium | C | C | D | RESEARCH_ONLY |

## 7. Women's baseball

Women's baseball must remain a separate model family from men's senior/youth baseball unless validation proves transferability.

Priority candidates:
1. WBSC Women's Baseball World Cup
2. Women's continental championships
3. Women's European Baseball Championship
4. Women's regional/national leagues where game-level and pregame data are sufficient

The 2026 WBSC calendar confirms the Women's Baseball World Cup as a current official baseball event.

## 8. Competition families intentionally excluded from the ordinary baseball model

- Softball World Cups and softball qualifiers
- Baseball5 World Cup / Youth Baseball5
- Slowpitch competitions
- Esports
- Local youth tournaments without stable game-level historical records
- Exhibition/friendly games unless a reproducible PIT-safe data source exists

WBSC's calendar separates Baseball, Softball and Baseball5 events; they should not be mixed into the same baseball probability model.

## 9. Production promotion gate

A competition is NOT promoted merely because schedule/results can be scraped.

Required checks:
1. Stable competition/team identifiers.
2. Historical game-level results.
3. Historical game date/time and venue.
4. Historical pregame information with publication/availability timestamps where used.
5. Officially announced starting pitcher evidence when the production prediction requires starters.
6. No-future-information leakage.
7. Sufficient chronological sample size.
8. Walk-forward OOS validation.
9. Calibration validation.
10. Score and Low/High validation where those markets are offered.
11. Source outage and missingness stress tests.
12. Reproducible raw snapshots and provenance.
13. Locked holdout untouched by model selection.
14. Explicit promotion decision stored as evidence.

For U12/U15/U18/U23 and short international tournaments, an additional minimum-sample and roster-churn gate is required. A model should be allowed to return "no prediction" rather than manufacture confidence.

## 10. Current repository truth

As of this registry:
- NPB: IMPLEMENTED.
- MLB: IMPLEMENTED.
- International competitions: not yet implemented as production acquisition/backtest pipelines.
- Youth competitions U12/U15/U18/U23: not yet implemented.
- Asian Games: not yet implemented.
- WBC/Premier12/Olympics: not yet implemented.
- KBO/CPBL/LMB and other domestic leagues: not yet implemented.

This document is therefore a **candidate inventory**, not a claim of current system coverage.


## Official source anchors

- WBSC baseball calendar: https://www.wbsc.org/en/calendar/2026/baseball
- WBSC U-12 Baseball World Cup: https://www.wbsc.org/en/events/2025-viii-u-12-baseball-world-cup
- WBSC U-18 Baseball World Cup: https://www.wbsc.org/en/events/2025-u18-baseball-world-cup
- WBSC U-23 Baseball World Cup: https://www.wbsc.org/en/events/2026-vi-wbsc-u-23-baseball-world-cup
- WBSC Europe baseball documents: https://www.wbsceurope.org/en/organisation/baseball/documents


## Phase 1 execution scope

Active engineering and production-readiness work is restricted to exactly three targets: **Asian Games Baseball (2026 Aichi-Nagoya), NPB, and MLB**. All other inventory entries are deferred until all three Phase 1 targets independently pass the full production gate.

The official Aichi-Nagoya organizer currently lists the 2026 baseball competition for September 21–27 and has published matchup changes. Schedule, matchup, starter and publication-time evidence must therefore be captured with strict PIT controls. Sources: https://www.aichi-nagoya2026.org/ja/sport/baseball/ and https://www.aichi-nagoya2026.org/ja/news-2050/
