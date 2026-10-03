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
* source state
* error class

を保存する。

同一gameの大量revisionで学習weightを不当に増やさない。

⸻

45. POSTGAME RECONCILIATION

outcome確定後にPredictionをreconcileする。

確認:

* prediction probability
* actual outcome
* target version
* cutoff
* source snapshot
* model version
* calibration
* PIT status
* production state
* fallback/abstain
* error type

past artifactとcurrent predictionを混同しない。

⸻

46. FAILURE TAXONOMY

最低限:

DATA_FAILURE
PIT_FAILURE
TIMESTAMP_FAILURE
REVISION_FAILURE
IDENTITY_FAILURE
SOURCE_FAILURE
FEATURE_FAILURE
STARTER_FAILURE
MODEL_FAILURE
CALIBRATION_FAILURE
ROUTING_FAILURE
REGIME_FAILURE
OOD_FAILURE
UNCERTAINTY_FAILURE
TIMING_FAILURE
SCOPE_FAILURE
AUTOMATION_FAILURE
RECOVERY_FAILURE
REPORTING_FAILURE

⸻

47. FAILURE MEMORY

Failure record:

failure_id
game_id
component
failure_type
severity
input_state
prediction
expected
observed
root_cause
counterfactual
repair
verification
reoccurrence

を保存する。

単なるwrong predictionとsystem failureを区別する。

⸻

48. COUNTERFACTUAL FAILURE ANALYSIS

failureについて、

* earlier information
* later valid information
* additional source
* alternate model
* alternate calibration
* alternate routing
* specialist
* fallback
* abstention

で改善した可能性を評価する。

測定済み事実とhypothesisを分離する。

⸻

49. COVERAGE DIGITAL TWIN

coverage dimensions:

* league
* season
* phase
* game
* team
* player
* starter
* feature
* source
* availability metadata
* horizon
* regime

coverage gapをCoverage Debtとして管理する。

「5000 candidate factors」等のfeature-space目標はcandidate search breadthを示すものであり、無条件採用件数ではない。

⸻

50. SOURCE VALUE

Source valueを、

ΔLogLoss
ΔBrier
ΔAccuracy
calibration improvement
uncertainty reduction
failure avoidance
OOD detection
coverage improvement
latency
reliability
cost

で測る。

source数の増加自体を成果としない。

⸻

51. FEATURE RETIREMENT

featureは永続使用ではない。

retirement候補:

* no incremental OOS value
* unstable across periods
* high leakage risk
* poor freshness
* high missingness
* redundant
* high computation
* source unreliable
* maintenance burden

feature retirementもmemoryへ保存する。

⸻

52. COMPLEXITY BUDGET

追加complexityについて、

performance gain
+
robustness gain
+
information gain

と、

maintenance cost
+
runtime cost
+
failure surface
+
operational risk

を比較する。

小さい改善のために大幅なcomplexity増加を無条件に採用しない。

⸻

53. RESEARCH ROUTER

Research categories:

DIRECT_SEARCH
METHOD_SEARCH
FAILURE_SEARCH
COUNTEREXAMPLE_SEARCH
IMPLEMENTATION_SEARCH
BENCHMARK_SEARCH
NEGATIVE_EVIDENCE
FRONTIER_SEARCH
CROSS_DOMAIN_TRANSFER
UNKNOWN_UNKNOWN_SEARCH

Research portfolio:

* exploit
* adjacent
* frontier
* replication
* ablation
* adversarial
* recovery
* meta-research

⸻

54. EXTERNAL RESEARCH INGESTION

External method:

DISCOVERED
→ SOURCE_VERIFIED
→ METHOD_ABSTRACTED
→ RELEVANCE_CHECKED
→ COST_CHECKED
→ PIT_CHECKED
→ LOCAL_IMPLEMENTATION
→ LOCAL_REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT
→ DECISION

論文/GitHub/AIのperformance claimはproduction evidenceではない。

⸻

55. EVIDENCE LEVEL

E0 = idea
E1 = external claim
E2 = external implementation
E3 = local reproduction
E4 = local OOS
E5 = robustness
E6 = frozen holdout
E7 = production evidence

Evidenceを過大評価しない。

⸻

56. NEGATIVE KNOWLEDGE

以下を保存:

* rejected method
* rejected feature
* rejected source
* rejected routing
* rejected calibration
* rejected competition
* failure condition
* PIT issue
* robustness failure
* computational cost

同一失敗の再発を防ぐ。

⸻

57. CROSS-PROJECT TRANSFER

他project知見は、

DISCOVER
→ ABSTRACT_MECHANISM
→ COMPATIBILITY
→ ADAPT
→ LOCAL_PIT
→ LOCAL_OOS
→ LOCAL_HOLDOUT
→ SHADOW
→ PROMOTE

とする。

他domain成功を直接Baseball productionへ移植しない。

⸻

58. EXPANSION POLICY

新competition、新source、新feature、新model、新horizonは、

Discovery
→ Metadata
→ Data feasibility
→ PIT
→ Shadow
→ OOS
→ Robustness
→ Holdout
→ Limited production
→ Stable production

の順に昇格させる。

⸻

59. ADOPTION GATES

candidate採用には:

* same observations
* chronological OOS
* PIT valid
* leakage audit
* reproducibility
* robustness
* calibration
* newest holdout
* sufficient sample
* operational safety

を要求する。

参考benchmark:

primary relative OOS improvement ≥3%
auxiliary improvement ≥1%
≥70% evaluation periods without worsening
latest holdout no worsening
calibration not materially degraded
PIT violations = 0

thresholdは絶対真理ではなく、sample size、variance、confidence interval、game dependence、rare events、cost、riskを考慮する。

⸻

60. STATISTICAL INTEGRITY

必要に応じて:

* paired comparison
* game-cluster bootstrap
* block bootstrap
* confidence interval
* permutation
* forecast comparison tests
* multiple-comparison correction

を使用する。

単一metric差や単一foldだけでsuperiorityを断定しない。

⸻

61. FROZEN HOLDOUT FIREWALL

Holdoutはconfig freeze後のfinal evidence専用。

禁止:

* repeated holdout tuning
* feature tuning
* model tuning
* calibration tuning
* threshold tuning
* routing tuning
* source selection
* scope selection

holdout contaminationが疑われた場合はholdoutのstatusをinvalidateし、再freeze/rebuildする。

⸻

62. ADVERSARIAL VALIDATION

定期的に:

* future timestamp injection
* postgame field injection
* future starter injection
* future standings
* later revision
* source removal
* feature deletion
* missingness
* time shift
* distribution shift
* regime transition
* unseen team/player
* stale source

を攻撃試験する。

⸻

63. ROBUSTNESS

Candidateは平均OOSだけで判断しない。

subset:

* latest period
* recent season
* NPB
* MLB
* phase
* starter-known
* starter-unknown
* high uncertainty
* high disagreement
* high variance
* missing data
* OOD
* new teams/players
* source outage

を確認する。

⸻

64. PRODUCTION MODEL REGISTRY

Production modelに:

* model_id
* version
* target_version
* feature_version
* calibration_version
* training scope
* routing
* data snapshot
* OOS
* robustness
* holdout
* release timestamp
* rollback pointer

を保存する。

⸻

65. PRODUCTION FAIL-CLOSED

以下はproduction predictionを無理に生成しない条件:

* model missing
* corrupted artifact
* critical source failure
* stale required data
* PIT unknown
* identity mismatch
* target mismatch
* feature contract failure
* invalid calibration

fallback/abstain/deferredへ移行する。

⸻

66. FALLBACK CHAIN

基本:

Champion
→ Validated Specialist
→ Generalist
→ Baseline
→ Abstain

fallback使用をprediction logに保存する。

fallback resultをChampion resultと同等と表現しない。

⸻

67. AUTOMATION

GitHub Actionsは:

* checkpoint
* resume
* idempotency
* retry
* backoff
* watchdog
* heartbeat
* stale-run handling
* deterministic writes
* artifact preservation
* concurrency
* recovery
* rollback

を満たすことを優先する。

長時間処理を一発jobだけに依存しない。

⸻

68. SINGLE-WRITER

critical state:

* model registry
* experiment registry
* source registry
* experience ledger
* promotion state
* rollback state
* scope state

はsingle-writer semanticsを優先。

並列researchは可能だがmergeはdeterministic。

⸻

69. CACHE / EFFICIENCY

最適化順序:

cache
→ exact snapshot reuse
→ incremental update
→ deduplication
→ vectorization
→ parallel I/O
→ selective recomputation
→ retraining optimization
→ algorithm optimization

同一fingerprintを重複計算しない。

⸻

70. AUTOMATION QUALITY

System qualityだけでなくautomation qualityを測定する。

* false success
* false recovery
* repeated failure
* recovery time
* checkpoint recovery
* duplicate execution
* stale artifact
* wasted compute
* blocked queue

green Action countをquality proxyにしない。

⸻

71. COST FIREWALL

優先:

1. verified free
2. free quota
3. OSS/local
4. cached/local snapshot
5. lightweight computation

paid-only、billing-risk、unknown-cost、auto-renew trial、quota-overageは自動利用禁止。

cost不明 = HOLD / UNCONFIRMED。

⸻

72. SECURITY / DATA GOVERNANCE

secret/API key/tokenをcode、logs、artifacts、reports、commitsへ出力しない。

external sourceについて:

* license
* attribution
* redistribution
* rate limit
* commercial restriction
* retention

を確認する。

license/cost不明sourceをproduction dependencyにしない。

⸻

73. POLICY REGRET

後から:

* Model Regret
* Timing Regret
* Information Regret
* Scope Regret
* Policy Regret
* Research Regret

を分析する。

例えば、
starter取得を早める価値、
late informationを待つ価値、
specialist routingの価値、
abstentionの価値
を評価する。

⸻

74. FRONTIER SCANS

定期的に:

Data Frontier
Research Frontier
Failure Frontier
Scope Frontier
Source Frontier
Model Frontier
Unknown Frontier

をscanする。

目的は無限拡張ではなく、現在のsystem limitationを発見すること。

⸻

75. SELF-EVOLUTION

detect:

* recurring failure
* obsolete rule
* contradictory source
* stale source
* ineffective workflow
* inefficient computation
* new validated method
* new coverage opportunity

change flow:

PROPOSE
→ CONSISTENCY CHECK
→ HISTORICAL IMPACT CHECK
→ IMPLEMENT
→ TEST
→ PIT
→ OOS
→ ROBUSTNESS
→ HOLDOUT/RELEASE
→ ADOPT/REJECT

過去の成績・失敗・holdoutを良く見せるために改変しない。

⸻

76. RESEARCH STOPPING

以下で停止/保留可能:

* repeated zero incremental value
* insufficient PIT
* insufficient sample
* unresolved source quality
* robustness failure
* high compute cost
* duplicated mechanism
* frontier saturation

停止理由はNegative Knowledgeへ保存する。

⸻

77. PREDICTION STATE

標準state:

FRESH
AGING
STALE
UNKNOWN
SHOCKED
REQUIRES_RECALC
FALLBACK
ABSTAIN
INVALIDATED

stateをprediction outputへ可能な範囲で付与する。

⸻

78. RESULT PRESENTATION

性能が変化した場合、作業報告に自動表示:

* Current Champion
* Prior Champion
* Candidate
* OOS ΔLogLoss
* OOS ΔBrier
* OOS ΔAccuracy
* OOS ΔECE
* Latest Holdout Δ
* Robustness Δ
* PIT status
* sample size
* evaluation period
* decision
* production status

measured improvementとhypothetical improvementを区別する。

⸻

79. STATUS TAXONOMY

厳密に区別:

IMPLEMENTED
EXECUTED
VERIFIED
PERFORMANCE_VERIFIED
PROMOTION_CANDIDATE
ADOPTED
PRODUCTION
STABLE
HOLD
REJECTED
FAILED
BLOCKED
DEFERRED
ROLLED_BACK
UNKNOWN
UNVERIFIABLE
SUPERSEDED
RETIRED

code exists ≠ adopted。
workflow green ≠ performance verified。

⸻

80. NO-FAKE-SUCCESS

禁止:

* fabricated metrics
* missing→zero
* silent exception
* hidden partial completion
* skipped test→passed
* failed job→success
* unknown PIT→valid
* incomplete data→complete
* failed recovery→recovered

実測、推定、仮説、未検証を明確に表示する。

⸻

81. REPRODUCIBILITY

production/research resultは可能な限り、

source snapshot
→ feature
→ model
→ calibration
→ probability
→ decision

をreplay可能にする。

再現不能artifactはevidence levelを下げる。

⸻

82. ARTIFACT INTEGRITY

artifactには可能な範囲で:

* hash
* git commit
* experiment id
* dataset hash
* source snapshot
* generation time
* schema version
* model version

を保存する。

⸻

83. COMPLETION DEFINITION

completionは「Workflowがgreen」「prediction JSONが存在」ではない。

最低限:

* tests
* PIT audit
* leakage/meta-leakage audit
* chronological WFO/OOS
* calibration
* ablation
* robustness
* frozen holdout
* artifact validation
* reproducibility
* recovery
* release gate
* monitoring
* rollback

のevidenceが必要。

未実施は未実施として表示する。

⸻

84. CONTINUOUS OPERATING LOOP

MONITOR
→ DETECT
→ TRIAGE
→ RESEARCH
→ IMPLEMENT
→ TEST
→ PIT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ HOLDOUT
→ ADOPT/HOLD/REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

を継続する。

⸻

86. RECONCILIATION METADATA PRESERVATION

production prediction rows may already contain prediction-time competition metadata such as competition, competition_stage, season_type, game_class, competition_key, classification status, and the source/field/value used to classify the slate.

Postgame reconciliation must preserve these fields into the experience ledger. The reconciliation layer may enrich missing taxonomy metadata only from the immutable prediction snapshot or separately verified prediction-time evidence. It must not infer a historical phase from postgame facts, outcome pages, later revisions, or current schedule state, and it must not silently convert UNKNOWN into a classified phase.

The retained metadata is part of the experience evidence chain because competition/phase segmentation is an evaluation axis. Dropping it during reconciliation invalidates that segmentation even when the original prediction snapshot was classified correctly.

⸻

85. ULTIMATE PRINCIPLE

Baseball Prediction Systemの最適化対象は単なる勝敗Accuracyではない。

Future Generalization
×
Case-Level Correctness
×
Calibration
×
Predictability Awareness
×
Uncertainty Quality
×
Robustness
×
PIT Integrity
×
Information Efficiency
×
Operational Reliability
×
Recovery
×
Reproducibility

を最大化する。

特に、

PIT Integrity > Apparent Backtest Gain
Evidence > Assumption
Future Generalization > Historical Fit
Case-Level Error Analysis > Aggregate Average
Calibration > Raw Confidence
Robustness > Single-Fold Improvement
Failure Learning > Repeated Failure
Safe Degradation > False Prediction
Reproducibility > Convenient Output

を基本原則とする。

目標は「もっと複雑な野球モデル」ではなく、

いつ予測するか、
その時点で本当に何が分かっていたか、
starter/lineupがどの程度確定していたか、
どのモデルが適切か、
どの程度確信すべきか、
追加情報を取得する価値があるか、
予測を出すべきでないgameはどれか、
なぜ失敗したか、
その失敗を次の研究へどう変換するか

まで制御できるAdaptive Baseball Prediction Intelligenceを構築することである。

=== COPY END ===