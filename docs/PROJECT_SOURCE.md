=== COPY START ===

Baseball-Prediction-System — Project Source

TARGET:
https://github.com/sasasotaro1202-star/Baseball-Prediction-System

ROLE:
本SourceをBaseball Prediction Systemの技術・研究・検証・運用仕様の正本とする。
Project Instructionsは常時優先ルール、本Sourceは詳細な実装・データ・PIT・OOS・モデル・運用・研究・自己改善ルールを保持する。

現行GitHubのHEAD、code、config、tests、workflows、Actions、artifacts、registries、実測値が過去文書・会話と矛盾する場合は現行GitHubを優先する。ただし過去実験結果、失敗、holdout、production履歴を後付け変更して整合させない。

⸻

1. SYSTEM MISSION

目的はhistorical fitの最大化ではなく、未知の将来試合へのFuture Generalizationを最大化すること。

評価軸:

* Case-Level Correctness
* Probabilistic Quality
* Calibration
* Predictability Awareness
* Uncertainty Quality
* Robustness
* PIT Integrity
* Information Value
* Selective Prediction
* Operational Reliability
* Recovery
* Reproducibility

System全体を、

Data
→ Identity
→ Coverage
→ PIT
→ Feature
→ Model
→ Routing
→ Calibration
→ Uncertainty
→ OOS/WFO
→ Robustness
→ Holdout
→ Release
→ Production
→ Reconciliation
→ Failure Analysis
→ Research
→ Adoption/Rollback

の閉ループとして扱う。

⸻

2. CURRENT REPOSITORY INTEGRATION

既存repositoryにはproduction、24h research、supervisor、pregame、recovery、PIT acquisition、candidate OOS、experience learning、regime、scope discovery、temporal conformal、universal adapters、NPB draw、MLB PIT、Statcast等の実装群が存在する。

作業時はまず既存implementationを監査する。

同じ機能が存在する場合は、
REUSE → REPAIR → INTEGRATE → TEST
を優先し、重複実装を避ける。

Workflowの存在はcompletionの証拠ではない。

⸻

3. CANONICAL SPORT SCOPE

主要scope:

* NPB
* MLB

Expansion candidates:

* KBO
* CPBL
* international tournaments
* World Baseball Classic
* NCAA
* other validated competitions

未検証competitionをproductionへ自動投入しない。

⸻

4. COMPETITION POLICY

competition hierarchy:

league
→ season
→ phase
→ event/game

phase候補:

* regular season
* interleague
* postseason
* playoff
* tournament
* qualifier
* exhibition
* unknown

UNKNOWNをregular seasonへsilent mappingしない。

competition-specific rules、sample、calibration、home advantage、roster policyを保持する。

⸻

5. CANONICAL TARGETS

NPB

HOME
DRAW
AWAY

3-class probability。

MLB

HOME
AWAY

2-class probability。

NPB drawをMLB binaryへ変換して本番targetとしない。

⸻

6. SCORE TARGET

Score predictionはwin predictionから独立して扱う。

current specification:

* Top-4 exact-score candidates
* home runs
* away runs
* total runs
* expected score/runs
* low/high score classification

Top-4 probabilityは必要に応じてranking/distributionとして保持し、1X2 probabilityと混同しない。

⸻

7. LOW / HIGH TARGET

Current classification:

LOW = total runs <= 6
HIGH = total runs >= 7

このtargetは1X2、score Top-4と独立したtarget_versionを持つ。

target変更時は別versionとして扱う。

⸻

8. PREDICTION CONTRACT

Current predictionはrequest-time再計算を基本とする。

最低限:

* current_time
* timezone
* game_id
* game_time
* prediction_time
* prediction_cutoff
* source snapshots
* model version
* feature version
* calibration
* probability
* uncertainty
* data status
* generation status

を追跡可能にする。

過去artifactはhistorical/reconciliation用であり、current predictionの代替にしない。

⸻

9. TIME MODEL

timestampを厳密に区別する。

* game_time
* prediction_time
* prediction_cutoff
* available_at
* published_at
* retrieved_at
* revision_time
* outcome_time

特に、

retrieved_at ≠ published_at ≠ available_at

を原則とする。

取得した時間だけでは当時利用可能だったことを証明しない。

⸻

10. PIT CONTRACT

基本条件:

available_at <= prediction_cutoff

または、sourceが同等の明確なhistorical availability evidenceを持つこと。

禁止:

* future outcome
* postgame statistics
* future standings
* future roster state
* later correction
* later article
* future market information
* future starter confirmation

の逆流。

availabilityが未証明なら、

UNKNOWN / UNVERIFIABLE

とし、production-quality OOSに使用しない。

⸻

11. STARTER PIT

starter-dependent featureは特別扱いする。

historical starter使用には原則、

* official starter announcement
* source URL/reference
* published_at/available_at
* cutoffとの関係

が必要。

starter announcement timingが曖昧なhistorical dataはfail-closed。

「後からstarterだったと分かった」だけではpre-game usableとはみなさない。

⸻

12. LINEUP / ROSTER

lineup、bench、player availability、starting pitcher、bullpen availabilityをtime-varying informationとして扱う。

同じ選手でも、

* announced
* probable
* confirmed
* scratched
* injured
* unavailable

をstateとして区別する。

late updateをearlier snapshotへbackfillしない。

⸻

13. STATCAST POLICY

Statcast等のpostgame-derived statisticsは、pre-game predictionへ直接利用してはならない。

使用可能なのはprediction cutoff以前に取得可能だったhistorical player/pitcher statisticsのみ。

同一game内のfinalized Statcast values、postgame pitch/result information、outcome-derived aggregatesは禁止。

⸻

14. LEAKAGE TAXONOMY

最低限:

DIRECT
TEMPORAL
REVISION
ROLLING-WINDOW
AGGREGATION
ENTITY
STARTER
ROSTER
POSTGAME
MARKET
CALIBRATION
MODEL-SELECTION
FEATURE-SELECTION
ROUTING
SCOPE
META-LEAKAGE

を監査する。

Meta-leakageには、OOS結果によるsource selection、feature family selection、threshold selection、research policy selectionを含める。

⸻

15. DATA QUALITY CONTRACT

品質評価dimensions:

* game coverage
* entity coverage
* outcome completeness
* feature completeness
* timestamp completeness
* availability completeness
* revision integrity
* duplicate rate
* identity integrity
* schema stability
* source freshness
* source reliability
* coverage breadth

row countだけではcompleteとしない。

⸻

16. MISSINGNESS

missingとzeroを区別する。

state:

UNAVAILABLE
UNKNOWN
NOT_APPLICABLE
DELAYED
SOURCE_FAILED
MALFORMED
NOT_YET_PUBLIC
STRUCTURALLY_ABSENT

critical missing時は、

FALLBACK
ABSTAIN
DEFERRED
FAIL

のいずれか。

missing→0によるsilent distortionは禁止。

⸻

17. IDENTITY CONTRACT

stable identifiers:

* game_id
* team_id
* player_id
* stadium_id
* competition_id
* season_id

を基本とする。

各sourceについてraw ID/nameも保存する。

Fuzzy matchingはreview candidate生成専用。
silent mergeは禁止。

⸻

18. FEATURE ECOSYSTEM

candidate feature families:

Game Context

* home/away
* season
* date
* schedule
* rest
* travel
* venue

Team Strength

* Elo
* historical record
* opponent-adjusted strength
* recent strength

Recent Form

3/5/10/20/30/45/60 game windows等を研究可能。

Pitching

* starter quality
* handedness
* workload
* recent form
* pitch-level metrics

Bullpen

* availability
* workload
* recent usage
* fatigue
* leverage usage

Batting

* team offense
* player availability
* platoon
* recent performance

Matchup

* pitcher/batter
* handedness
* park
* opponent interaction

Statcast

* pitch quality
* contact quality
* velocity
* movement
* launch/contact-related features

External Context

* weather
* market
* schedule
* travel
* park effects

Meta

* data quality
* source reliability
* uncertainty
* regime
* OOD

feature expansion itself is not success.

⸻

19. FEATURE WINDOW CONTRACT

Rolling features must obey:

window_end <= prediction_cutoff

future rows、future game results、later roster statesをrolling aggregationへ混入させない。

window definitionsはversioned。

⸻

20. FEATURE LINEAGE

各featureについて可能な限り:

feature_id
source_id
raw_field
transformation
window
available_at
cutoff
revision_policy
quality_status
version
code_commit

を保存する。

Feature valueだけで再現可能性を失わない。

⸻

21. SOURCE REGISTRY

各source:

* source_id
* owner
* upstream
* endpoint
* data type
* coverage
* freshness
* historical depth
* available_at support
* published_at support
* revision behavior
* license
* cost
* reliability
* parser
* schema
* last success
* last failure
* incremental value
* production status

をRegistryに保持する。

⸻

22. SOURCE INDEPENDENCE

同一upstreamの、

* mirror
* wrapper
* copied dataset
* republished CSV
* derived archive
* scraper output

は独立sourceとして数えない。

Source graphを保持し、独立性をensemble/research evidenceへ反映する。

⸻

23. SURVIVORSHIP / UNIVERSE

historical researchでは後から追加・削除されたteam/player/universe informationに注意する。

現在存在するteam/player listだけで過去populationを構成しない。

historical membership、team relocation、franchise/name changeを可能な範囲で時点管理する。

⸻

24. DATA SNAPSHOT

research runには可能な限り:

* dataset hash
* source snapshot
* retrieval metadata
* schema version
* feature version
* code commit
* cutoff policy
* time range

を保存する。

同じ時点のdataを将来再現できるようにする。

⸻

25. EXPERIMENT SCHEMA

experiment:

* experiment_id
* fingerprint
* hypothesis
* scope
* target_version
* dataset
* PIT status
* feature set
* model
* routing
* calibration
* seed
* folds
* evaluation periods
* metrics
* cost
* result
* decision
* artifact

を記録する。

⸻

26. EXPERIMENT FINGERPRINT

fingerprint候補:

* git commit
* dataset hash
* source snapshot
* feature version
* target version
* model config
* calibration config
* routing policy
* cutoff policy
* seed
* environment

同一fingerprintはcache reuseを優先。

⸻

27. MODEL ECOLOGY

候補:

* Logistic Regression
* ExtraTrees
* HistGradientBoosting
* LightGBM
* XGBoost
* CatBoost
* hierarchical models
* ensemble
* recent-data expert
* regime expert
* starter expert
* Statcast expert
* calibration expert
* uncertainty expert
* selective/fallback model

高度なmodelが必ずsuperiorとは仮定しない。

⸻

28. MODEL ROUTING

routing dimensions:

* league
* asset/sport scope
* season
* phase
* team strength
* starter state
* regime
* recentness
* data quality
* uncertainty

minimum sample thresholdを設定する。

small sample specialistは無理に使用せず、

specific
→ league
→ global

等のhierarchical fallbackを使用する。

⸻

29. BASELINES

各scopeでbaselineを保持する。

候補:

* class frequency
* naive prior
* simple Elo
* logistic baseline
* recent form baseline

complex modelはbaselineに対するincremental valueで評価する。

⸻

30. CALIBRATION

candidate:

* none
* temperature
* sigmoid
* beta-style
* isotonic
* suitable temporal calibrator

validation期間で選択する。

frozen holdoutでtuningしない。

metrics:

* LogLoss
* Brier
* ECE
* calibration slope
* calibration intercept
* reliability

⸻

31. OOS / WFO

基本:

Train
→ Validation
→ Walk-Forward OOS
→ Robustness
→ Frozen Holdout

random split禁止。

OOS predictionは時間順で生成する。

future sampleをtrainingへ混入させない。

⸻

32. OOS PRODUCTION STANDARD

各OOS rowに:

* game_id
* cutoff
* snapshot
* feature version
* model
* calibration
* probability
* actual
* uncertainty
* regime
* data quality
* PIT
* error type

を保持する。

⸻

33. WIN METRICS

Primary:

LogLoss

Secondary:

Accuracy
Brier
ECE
class-wise LogLoss
class precision/recall
calibration slope/intercept

NPBとMLBを必要に応じて別々に報告する。

⸻

34. SCORE METRICS

最低限候補:

* home-run MAE
* away-run MAE
* total-run MAE
* exact-score Top1
* exact-score Top4
* probabilistic score
* low/high metrics

score distributionのcalibrationも評価する。

⸻

35. STARTER-SPECIFIC EVALUATION

starter-dependent modelでは、

* confirmed starter availability
* starter sample size
* starter missingness
* starter announcement timing
* fallback frequency

を別集計する。

starter informationがavailableなcasesだけで改善しているのか、全production populationでも改善しているのかを分離する。

⸻

36. REGIME INTELLIGENCE

候補regime:

* run environment
* bullpen environment
* starting-pitcher environment
* injury/roster regime
* schedule regime
* market regime
* volatility regime
* competition phase

regime detector自体もPIT検証する。

regime transition時はuncertaintyを上げる、alternative modelへrouteする等を検討する。

⸻

37. PREDICTABILITY

gameごとのpredictabilityを研究する。

候補signal:

* model disagreement
* data completeness
* starter uncertainty
* source disagreement
* OOD
* regime ambiguity
* recent volatility
* calibration instability
* outcome entropy

confidenceとpredictabilityを分離する。

⸻

38. UNCERTAINTY DECOMPOSITION

可能な範囲で:

* aleatoric
* epistemic
* data
* source
* temporal
* starter
* regime
* OOD
* disagreement

を分離する。

confidence marginだけをuncertaintyの唯一指標にしない。

⸻

39. DISAGREEMENT

candidate model distributionsを比較する。

候補:

* probability variance
* entropy gap
* KL divergence
* Jensen-Shannon divergence
* ranking disagreement

disagreement spikeはcase review、additional information、fallback、abstention candidateとして扱う。

⸻

40. INFORMATION ACQUISITION

action:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
FALLBACK
ABSTAIN

判断要素:

* expected information value
* probability of material update
* source reliability
* latency
* computation
* cutoff proximity
* uncertainty
* disagreement

情報取得は「多いほど良い」としない。

⸻

41. FORECAST LIFETIME

predictionは時間経過だけでなく情報変化でstale化する。

state:

FRESH
AGING
STALE
UNKNOWN
SHOCKED
REQUIRES_RECALC
ABSTAIN
FALLBACK
INVALIDATED

starter change、weather update、major source revision、late lineup等で再評価する。

⸻

42. SELECTIVE PREDICTION

productionで100% prediction coverageを必須としない。

candidate action:

predict
fallback
abstain
defer
acquire information

評価:

* coverage
* selective risk
* utility
* calibration
* stability
* false-abstention cost

⸻

43. CONFORMAL / RISK CONTROL

research candidates:

* split conformal
* adaptive conformal
* local conformal
* online conformal
* risk-controlling prediction
* prediction set
* selective risk

評価:

coverage
set size
risk
stability
regime robustness

⸻

44. EXPERIENCE LEARNING

canonical unit = game_id。

experience recordには:

* game_id
* final outcome
* latest valid pre-cutoff prediction according to policy
* cutoff
* model
* calibration
* uncertainty
* predictability
* regime
* data quality