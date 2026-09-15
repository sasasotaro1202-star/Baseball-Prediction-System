# Baseball data-source hierarchy

The system does not force every feature through one provider. Each feature uses the strongest practical source and records missing data instead of guessing. Multiple independent sources are preferred for discovery and cross-validation, but source agreement never replaces PIT evidence.

## Source hierarchy

| Competition / feature | Primary / strongest source | Secondary / validation | Production rule |
|---|---|---|---|
| NPB schedule, game date/time, teams, final result | NPB.jp official | SPAIA schedule API | NPB.jp is canonical where available |
| NPB starting pitchers | NPB.jp official / official game information | SPAIA game PBP | Never substitute an arbitrary pitcher |
| NPB pitcher game line | SPAIA pitcher-game API | official NPB statistics where available | Missing line keeps coverage below the completion gate |
| NPB player game stats / lineup / play style | SPAIA game APIs | public NPB data repositories | Target-game stats are only committed after prediction |
| NPB PBP | SPAIA game PBP | Nippon-Baseball-Data-Repository bulk PBP | Chronological, game-level data only |
| Historical NPB reference stats | NPB.jp official statistics | public NPB data repository | Respect publication/availability date in backtests |
| Historical weather | Open-Meteo archive | none | Venue/date cache; no synthetic weather |
| MLB schedule/results/game IDs | MLB Stats API / MLB.com | Yahoo! Sports Navi MLB | MLB remains canonical for IDs/results; Yahoo is a secondary operational source |
| MLB probable starters | MLB.com / MLB Stats API | Yahoo! Sports Navi MLB, FanGraphs RosterResource, ESPN | Cross-source agreement is useful, but does not prove historical official announcement time |
| MLB pitch/batted-ball features | Baseball Savant / Statcast | MLB data | Apply explicit PIT availability policy |
| International senior baseball | Official competition/WBSC/Olympic/Asian Games sources | reputable sports sources | Separate competition OOS/holdout and rule contract required |
| U18/U23 and other age-group | WBSC / official competition sources | reputable sports sources | Research-only until PIT/OOS/promotion gates pass |
| Koshien / Japanese high school | Japan High School Baseball Federation / official tournament sources | reputable Japanese sports sources | Competition-specific metadata; do not mix blindly with NPB |
| University/intercollegiate | JABA / official university competition sources | reputable sports sources | Separate competition contract and validation required |

## MLB operational source set

### MLB.com / MLB Stats API
Use for canonical game identity, schedule/results and MLB-native probable-pitcher information. MLB's probable-pitcher pages expose the current matchup and listed pitchers. citeturn0search2turn0search7

### Baseball Savant / Statcast
Use for high-resolution MLB pitching, batted-ball and tracking features. Baseball Savant provides per-pitch, per-game, player, team and season Statcast queries and CSV documentation. It is a feature source, not automatically a PIT announcement source.

### Yahoo! Sports Navi MLB
`https://baseball.yahoo.co.jp/mlb/` and its schedule/game pages are supported as a secondary operational source. Current pages expose schedules, probable starters and game information in Japanese. citeturn0search8turn0search12

### FanGraphs RosterResource
Use the probable-pitcher grid as an independent cross-check and for roster context. It is not treated as proof of official announcement time.

## PIT / announcement-time policy

The following are distinct and must be stored separately:

- `retrieved_at`: when our collector obtained the source observation
- `available_at`: when the source/data was demonstrably available to the collector or public
- `announcement_at`: when the starter was actually announced, if explicitly evidenced
- `prediction_cutoff`: the information boundary for the prediction

A displayed probable pitcher is **not automatically an officially announced starter**. Do not infer `announcement_at` from retrieval time, crawl time, first observation time, page modification time, or source agreement. If historical announcement timing cannot be defended, the game remains ineligible and the pipeline fails closed for that critical field.

## Source conflict policy

1. Prefer an authoritative competition source for official event identity and results.
2. Prefer the strongest field-specific source for technical statistics.
3. Use independent sources to detect conflicts and missingness.
4. Never silently overwrite a stronger source with a weaker source.
5. Preserve all source observations needed for auditability.
6. If a conflict affects a critical prediction field and cannot be resolved PIT-safely, mark the record invalid/research-only rather than guessing.

## Reliability and fallback

Fallback order is:
1. authoritative competition/API source
2. independent authoritative/near-authoritative source
3. reputable secondary source
4. cached historical observation whose PIT timestamp is already verified
5. safe stop

Retries, exponential backoff, cache, checkpoints and source redundancy are reliability mechanisms. They must never be used to bypass a PIT gate.

## Competition expansion

The target scope includes NPB, MLB, other reliable professional leagues, Olympics, Asian Games, WBC, WBSC Premier12, U18/U23 and other age-group competitions, Koshien and prefectural high-school competitions, university/intercollegiate baseball, and other material competitions when reliable PIT data exists.

Expanded competitions are not automatically production-eligible. Each competition requires provenance, PIT-safe availability/announcement evidence, rule/outcome-contract isolation, sufficient chronological OOS and protected holdout evidence, calibration, robustness, and promotion-gate approval. Unsupported or insufficiently historical competitions remain `RESEARCH_ONLY` or `UNAVAILABLE` and are never silently mixed into production training/evaluation.
