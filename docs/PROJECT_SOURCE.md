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


⸻

91. PLAYER ROLE AND EVIDENCE COVERAGE

TEAM-WIDE PLAYER CONTEXTは、各選手について現在観測できている成績だけでなく、選手の役割とデータ証拠範囲を明示する。

各player rowへ:

* player_role
* player_role_source
* player_role_evidence
* player_data_coverage

を保存する。

player_roleはofficial positionとpitching usageなどの観測値から保守的に導出し、例として:

* STARTING_PITCHER
* RELIEF_PITCHER
* PITCHER
* TWO_WAY_CANDIDATE
* CATCHER
* INFIELDER
* OUTFIELDER
* UTILITY_POSITION_PLAYER
* UNKNOWN

等を使用する。証拠不足時はUNKNOWNまたはUNVERIFIED系へ倒し、役割を推測で固定しない。

player_data_coverageではstable player ID、batting、pitching、fielding、official profile、derived metric countを分離して保存する。未観測を0や不在へ変換しない。

team snapshotにはplayer_coverage_summaryを追加し、stable ID coverage、各統計sourceのcoverage、profile availability、observed PA/IP、role distributionを保持する。これはinformation acquisition、uncertainty、case-level analysisのためのevidence layerであり、production probabilityを自動変更しない。

NPB公式の2026年度公式戦成績はチーム打撃・投手・守備および個人成績を公開しており、現行collectorはこれらをplayer_idベースで統合する。 


92. LIVE SOURCE AVAILABILITY GATE

「取得できるデータ」と「stable identityを取得できるデータ」は別の品質軸として評価する。公式統計ページに選手成績が存在し、parserが選手名・主要数値を取得できる場合、ページ上にplayer_idリンクが埋め込まれていなくてもdata_status=AVAILABLEとする。一方、stable player_id coverageはidentity_status=COMPLETE/PARTIALとして別管理する。

Live Source Healthは、Roster、試合予定、予告先発、個人打撃、個人投手、個人守備、個人プロフィールを実際に取得して検証する。データ取得成功とidentity coverage不足を同一のSOURCE_FAILEDへ混同しない。公式統計表にstable player_idリンクが無い場合は、対象日Rosterの選手名とチームを起点に、NPB公式「選手検索」を完全一致で照合してstable player_id / player page URLを解決する。完全一致で一意に決まらない場合はIDENTITY_AMBIGUOUS/IDENTITY_NOT_FOUND/IDENTITY_SEARCH_FAILEDとして残し、fuzzy mergeやsilent mergeは禁止する。

=== COPY END ===

86. PREGAME CONTEXT ACQUISITION

試合前に取得可能なcontextを独立snapshotとして拡張する。

追加source/fields:

* official date-specific game schedule (Japanese NPB schedule endpoint: /bis/{year}/games/gm{date}.html)
* official venue / stadium identity
* stadium coordinates / roof class
* NPB Central/Pacific standings snapshot (G/W/L/T/PCT/GB/Home/Road)
* hourly weather forecast near scheduled first pitch
* temperature
* apparent temperature
* relative humidity
* dew point
* precipitation probability
* precipitation/rain
* wind speed/gust/direction
* pressure
* cloud cover
* weather code

各snapshotは、少なくとも:

* snapshot_id
* generated_at_utc
* prediction_cutoff_utc
* source_id
* source URL
* available_at/retrieved_at相当の観測時刻
* per-game source status
* field-level missingness

を可能な範囲で保持する。

PREGAME CONTEXTは現時点では「取得・保存・観測可能性の証跡」が主目的であり、新featureを本番モデルへ自動投入しない。historical OOSでの利用には、prediction cutoff以前のhistorical availabilityを独立に証明し、LOCAL PIT → chronological OOS → robustness → frozen holdout を通過させる。

source failureは明示的にSOURCE_FAILED/UNAVAILABLEとし、missingを0へ変換しない。weather等の動的sourceはcurrent/future prediction snapshotには利用できるが、retrieved_atだけをhistorical published_atの証明として扱わない。

NPB pregame context collectorは30分間隔のsnapshot workflowから実行可能であり、production prediction JSONにも一致するgame単位contextを紐付ける。context acquisition failureはprediction probability failureとは分離して記録し、既存production modelの安全なfail-closed境界を変更しない。


87. DETAILED PLAYER CONTEXT

NPB prediction snapshots may contain detailed official starter-player context in addition to starter identity.

Player context fields include, when available:

* stable NPB player_id
* official player page URL
* player position
* batting/throwing hand
* height / weight
* birth date
* career / draft metadata
* current-season batting record
* current-season pitching record
* career batting / pitching record
* derived batting rates such as BB%, K%, HR/PA, SB attempt rate, ISO-from-totals, BB/K
* derived pitching rates such as K/9, BB/9, HR/9, WHIP, K-BB volume

The profile source is the official NPB player page. NPB's player pages expose biographical attributes and year-by-year batting/pitching tables, and the official 2026 statistics pages expose team/player batting, pitching and fielding tables. citeturn984270view0turn582756search6

Current/future snapshots can preserve this information for prediction-time analysis, but current-page values must not be retroactively treated as historical PIT evidence. Historical OOS consumption requires an independently proven availability/published boundary.

Starter player context is observational metadata in the current release. It does not modify production probabilities automatically. Any conversion into predictive features requires LOCAL PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → adoption gate.


⸻

88. TEAM-WIDE PLAYER CONTEXT

試合対象チームについて、NPB公式の2026年公式戦個人打撃・個人投手・個人守備テーブルをチーム単位で取得し、player_idを軸に統合する。

保存候補:

* player_id
* player_name
* player_page
* identity_status
* position / fielding position
* current-season batting raw stats
* batting derived rates
* current-season pitching raw stats
* pitching derived rates
* fielding raw stats
* source URL
* source_as_of_date
* retrieved_at_utc
* available_at_utc
* published_at_utc
* revision_time_utc
* historical_oos_consumption

official team stat pages expose individual batting, pitching and fielding tables. citeturn944349search1turn944349search0turn436351search1turn436351search0

このteam-wide player contextはstarterだけでなく、同試合の両チームに属する選手群を対象にする。現在は「取得・保存・case analysis」のためのcontextであり、production probabilityには自動投入しない。

current-page season aggregatesは更新・訂正され得るため、retrieved_atをhistorical availabilityの証明に使わない。historical OOSで使う場合は、対象時点でのpublished/available boundaryを別途証明してから LOCAL PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout を実施する。

identityはofficial player_idを優先する。player_idを取得できない場合はNAME_ONLY_UNVERIFIEDとして保存し、fuzzy matchingやsilent mergeを行わない。source failureやpartial failureは明示し、missingをzeroへ変換しない。

production prediction JSONではteam_player_context_snapshot_id、home_team_player_context、away_team_player_contextを保持し、experience ledgerにも再現可能なJSONとして保存する。

⸻

89. EXPANDED PLAYER PROFILE CONTEXT

TEAM-WIDE PLAYER CONTEXTを、単なるcurrent-season aggregateの保存から「選手の個体特性 + 成績水準 + 役割」を同一snapshotで追跡できる構造へ拡張する。

追加候補:

* official personal profile fields
* handedness parsed into throws / bats
* height_cm
* weight_kg
* birth_date
* career
* draft
* profile_source provenance
* profile_status
* batting AVG / OBP / SLG / OPS / ISO
* HR/AB
* XBH/PA
* runs/PA
* RBI/PA
* PA/game
* SB success rate
* pitching IP/game
* start share
* K/BB
* decision win rate
* fielding error rate
* defensive chances/game
* double plays/game

個人プロフィールは公式NPB player pageへの既存player_idリンクから直接取得し、active-player全indexの再走査を追加しない。取得対象は優先度を付けた主要選手へ上限を設け、NPB_PLAYER_PROFILE_LIMIT_PER_TEAMで制御する。これは情報取得コスト・失敗面を抑えるためであり、未取得選手を欠損から0へ補完しない。

選択された選手についてもprofile_statusを、

AVAILABLE
NOT_SELECTED
NO_PROFILE_URL
SOURCE_FAILED

として区別する。SOURCE_FAILEDをAVAILABLEへ偽装しない。

profile snapshotには、少なくとも:

* source_id
* url
* retrieved_at_utc
* available_at_utc
* published_at_utc
* revision_time_utc
* historical_oos_consumption

を保持する。

これらのcurrent-page profile/season statisticsは、現在・将来試合の観測用contextとして保存できる一方、historical OOSへ直接backfillしてはならない。historical availabilityが証明できない場合はUNKNOWN/UNVERIFIABLEとして扱い、production-quality OOSから除外する。

derived metricsは現在の可視値を増やす目的であって、feature adoptionの証拠ではない。production probabilityへの投入は別実験として、LOCAL PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → adoption gate の順で判定する。

⸻

92. LIVE SOURCE VERIFICATION

実取得性を機能存在だけで判定しない。NPB Live Source Health workflowはGitHub Actionsの実環境から、当日JSTの公式試合日程・セントラル/パシフィック順位・試合前天候、日付別Roster、Roster上の実選手プロフィール、対象チームの個人打撃・個人投手・個人守備を実取得してparseする。critical source failureはSOURCE_FAILEDとして失敗させる。

Live verification artifactにはchecked_at_jst、対象日、source URL、取得件数、stable player ID件数、Roster transaction件数、weather取得件数等を保存する。artifactが存在しない、取得処理が実行されていない、またはparseできない場合はVERIFIEDとしない。

なお、現在の公式日程取得は日本語NPB endpointを使用する。英語endpointが利用できることを仮定してproduction source contractを構成しない。

⸻

90. DATE-SCOPED FIRST-TEAM ROSTER CONTEXT

NPB公式の「出場選手登録および登録抹消」日付別ページから、対象日付の出場選手一覧を取得し、player_idを軸に選手群をsnapshot化する。実取得性は独立した NPB Live Source Health workflow で定期検証し、Roster・個人打撃・個人投手・個人守備・個人プロフィールの公式ページを実際に取得・parseできた場合だけ VERIFIED とする。

保存:

* target_date
* team
* stable player_id
* player_name
* official player page URL
* identity_status
* snapshot_id
* source_id
* source URL
* retrieved_at_utc
* available_at_utc
* published_at_utc
* revision_time_utc
* historical_oos_consumption
* per-team player_count
* transactions
* registered_today
* removed_today
* registered_today_count
* removed_today_count
* transaction_parser_status
* roster_transaction_status（REGISTERED_TODAY / REMOVED_TODAY / NO_TRANSACTION_RECORDED / TRANSACTION_CONFLICT）

登録・抹消transactionはteam + stable player_idを基本identityとして保持し、position / uniform_number / player URL / identity_statusも可能な範囲で保存する。stable player_idが取得できないtransactionはNAME_ONLY_UNVERIFIEDとして残し、stable-id前提のjoinには使用しない。

同sourceは日付別の登録・抹消だけでなく「出場選手一覧」を提供するため、試合時点での一軍登録選手群と当日の登録変動をcurrent/future prediction contextとして保持できる。 citeturn457248view0turn433512search0

ただし、日付ページの現在取得時刻だけからhistorical published_atを逆算しない。historical OOSへの投入はavailability boundaryを独立証明するまでBLOCKED/UNKNOWNとする。

roster contextはproduction probabilityの自動変更には使用せず、current/future case analysis、player availability、lineup候補集合、fallback/uncertainty researchのために保持する。feature adoptionにはLOCAL PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → adoption gateを要求する。

source failureはSOURCE_FAILEDとして保存し、未取得選手をゼロや「不在」とは解釈しない。fuzzy name mergeは禁止し、official player_idを優先する。


⸻

86. CANDIDATE OOS STALE-RUN RECOVERY

Candidate OOS research intentionally uses `cancel-in-progress: false` so an active chronological validation is not interrupted.

The watchdog may recover a Candidate OOS run only when status is `in_progress`, the run head SHA equals the current main SHA, and run age is at least 360 minutes. The threshold is deliberately longer than the 260-minute candidate workflow timeout and is reserved for scheduler/runner state that remains incorrectly in progress.

After cancellation, the watchdog must refresh run state and verify that no active candidate run still blocks the current main before dispatching. A failed cancellation is fail-closed and must not create a duplicate dispatch. A verified stale-run cancellation may trigger an immediate current-main redispatch rather than waiting for the generic cancelled-run cooldown.

This recovery mechanism does not alter PIT eligibility, chronological OOS semantics, calibration, frozen holdout protection, adoption gates, or production Champion state. Operational recovery evidence remains distinct from performance evidence.

⸻

91. RECONCILIATION METADATA PRESERVATION

Production prediction snapshots may contain prediction-time competition taxonomy such as competition, competition_stage, season_type, game_class, competition_key, classification status, and source lineage. These fields are evaluation dimensions and evidence-chain metadata, not postgame-derived labels.

The experience ledger must preserve them through reconciliation. Missing taxonomy may be enriched only from the immutable prediction snapshot or separately verified prediction-time evidence. Outcome pages, later revisions, current schedule state, and postgame knowledge must not be used to reconstruct a historical phase, and UNKNOWN must never be silently remapped.

⸻

=== COPY END ===

⸻

## 93. CANDIDATE OOS EVIDENCE FRESHNESS

Candidate OOS evidence is valid only for the implementation and data snapshot it actually evaluates. Any change that can alter candidate selection, calibration, score modeling, routing, adoption gating, or candidate identity must retrigger the Candidate OOS workflow. The workflow trigger contract is therefore part of evidence integrity, not merely CI convenience.

A green Candidate OOS run from an older commit must not be reused as evidence for a changed implementation. Trigger coverage must include direct research dependencies that are not otherwise covered by the existing path filters, while avoiding unnecessary broad recomputation where a narrower dependency set is sufficient.


### Long-running OOS versus non-runtime continuity commits
An already-running Candidate OOS replay may tolerate a main-branch commit only when the complete diff is limited to non-runtime continuity files: `data/experience/**`, `tests/**`, or `.github/workflows/baseball_candidate_oos_watchdog.yml`. Experience files are append-only historical postgame reconciliation outputs; tests and the candidate-OOS watchdog are verification/control-plane files and are not runtime inputs to candidate selection or scoring. Candidate runtime code, configuration, PIT inputs, evaluation logic, the candidate workflow itself, or other data remain evidence-affecting and must fail closed. The watchdog is control-plane orchestration only; it must not be treated as candidate runtime evidence.

⸻

94. GLOBAL GITHUB-BACKED BASEBALL PREDICTION REQUEST POLICY

ユーザーが「試合予想して」「今日の試合を予想して」「NPB予想」「MLB予想」「大会の試合を予想して」など、野球の試合予想を要求した場合、NPBだけでなく、現行repositoryのcompetition registryに登録された全野球競技を対象として、`sasasotaro1202-star/Baseball-Prediction-System` のGitHub生成物を主要予想源とする。

対象は、NPB、MLB、国際シニア大会、U18/U23等のユース大会、高校野球、大学野球、その他registryに登録されたcompetitionを含む。competition registryに存在すること自体は予想実行可能性やproduction eligibilityを意味しない。

Prediction requestの標準flow:

REQUEST
→ CURRENT GITHUB HEAD/DEFAULT BRANCH RECHECK
→ EXACT COMPETITION/GAME/DATE RESOLUTION
→ EXISTING VERIFIED GITHUB OUTPUT CHECK
→ GENERATION WHEN MISSING
→ OUTPUT VALIDATION
→ PIT/ELIGIBILITY VALIDATION
→ USER RESPONSE

既存の当該requestに対するverified outputが存在しない場合、「artifactが無い」だけで処理を終了してはならない。repositoryの `prediction_requests/inbox/active.json` にrequestを登録し、GitHub Actionsの `baseball-user-prediction-request.yml` を起動するか、direct workflow-dispatch capabilityが利用可能なら同等のrequest generation workflowをdispatchする。

Generation laneは次の優先順位を使う:

CURRENT_PRODUCTION_RUNTIME
→ VALIDATED_RESEARCH_SHADOW
→ COMPETITION_SPECIFIC_RESEARCH_RUNTIME
→ BLOCKED / UNAVAILABLE

CURRENT_PRODUCTION_RUNTIMEはcurrent production registryが明示的にCURRENT_PRODUCTIONを示し、entrypoint/model/contractが揃っている場合だけ使用する。

VALIDATED_RESEARCH_SHADOWは、対象competitionについてrepository側に明示的に登録されたPIT-safe research laneが存在する場合だけ使用する。research outputはuser-requested predictionとして表示可能でも、PRODUCTION/Champion/Adoptedとは表現しない。

COMPETITION_SPECIFIC_RESEARCH_RUNTIMEは、対象competition専用のresearch adapter/runtimeがpolicyへ明示登録され、PIT/eligibility contractを満たす場合だけ使用する。

上記いずれにも該当しないcompetitionはBLOCKED / UNAVAILABLEとする。外部Webから独立に作った予想で穴埋めし、それをGitHub予想として表示してはならない。

同一request fingerprintにverified prediction resultが既に存在する場合は、再計算よりcache reuseを優先する。ただしcurrent/live informationが変化してpredictionがstale、invalidated、shock、requires_recalc等になっている場合は再生成を許可する。同一gameのrevisionを重複してexperience learningへ無制限に投入しない。

Prediction request resultはrequest_id、request_fingerprint、target_date、competition_id、source_repository、source_commit、generation_lane、generation_status、generation time、prediction outputを追跡可能にする。prediction outputには可能な範囲でmodel version、feature version、calibration version、prediction cutoff、data/source snapshot、PIT state、uncertainty、predictability、fallback/research stateを保持する。

生成完了の判定はworkflowのgreenだけでは行わない。少なくとも、JSON schema/contract、competition identity、prediction count、probability validity、PIT status、critical starter/personnel gate、artifact integrityを検証する。PIT/eligibilityがUNKNOWN/UNVERIFIABLEの場合はproduction-quality predictionとして扱わず、該当statusをそのまま保持する。

External web/current informationは、GitHub predictionを補助するverification/context layerとしてのみ使用する。schedule、starter、lineup、weather、source health等は現在値の確認に使えるが、GitHub generated predictionを無言で置換・上書きしない。外部推論を併記する場合はGitHub outputと明確に別レイヤーとして表示する。

Target semanticsはcompetition-specific contractを厳守する。NPBのHOME/DRAW/AWAY、MLBのHOME/AWAY、その他competitionのCOMPETITION_DEFINED contractを混在させない。score Top-4、Low/High、market-line、prediction setは勝敗targetとは独立したcontract/versionとして扱う。

Prediction request generationはrelease/promotion mechanismではない。request generationによってChampion、Production、Adoption、Holdout、historical evidenceを変更してはならない。research outputが生成されたことだけを根拠にproduction promotionしてはならない。

PIT原則:

retrieved_at ≠ published_at ≠ available_at

を維持し、prediction cutoff以前の利用可能性を証明できないsourceはUNKNOWN/UNVERIFIABLEとする。future outcome、postgame statistic、later correction、future starter confirmation、future roster state、future market informationをprediction snapshotへ逆流させない。

request workflowはcost firewallにも従い、verified free/OSS/local/cacheを優先する。paid-only、billing-risk、unknown-cost依存を自動追加・自動実行しない。

⸻

95. GITHUB REQUEST GENERATION ARTIFACT INTEGRITY

`prediction_requests/results/<request_id>.json` はimmutable request resultとして扱う。結果にはrequest metadataとgenerated prediction outputを分離して保存し、source commitとgeneration laneを明示する。

GitHub Actionsはrequest generationについて、

* triggering request commitのsnapshotを処理する
* concurrencyをserialized/queue semanticsで扱う
* bounded runtimeを設定する
* timeout/failureをsuccessへ変換しない
* result artifactを保存する
* valid resultのみcache reuse候補とする
* result commitとrequest commitを混同しない
* generation failure時に外部forecastへsilent fallbackしない

を満たす。

Generation artifactが確認できない場合、assistantはGitHub resultを「生成済み」と断定しない。prediction request statusはUNAVAILABLE、UNVERIFIABLE、BLOCKED、FAILED等、実際の状態を表示する。

96. FEATURE CONTRACT AND RUNTIME FEATURE EVIDENCE

Canonical feature documentation is `docs/FEATURE_MANIFEST.md` and machine-readable feature governance is `config/feature_policy.json`.

Feature lifecycle states:
ACTIVE
CONDITIONAL
OBSERVATION_ONLY
RESEARCH_CANDIDATE

Source-code presence alone never means a feature is active in production. The production runtime must preserve actual feature count, feature-schema hash, feature manifest version and context mode.

Expected base feature counts from the current `match_features` contract are:
- NPB: 482
- MLB: 470
before conditional lineup/weather context.

Lineup and weather features require explicit PIT-safe context configuration and cutoff-valid timestamps. Current production collection of player, roster, standings, weather, identity and source-health context is not itself permission to change production probability.

Feature assembly order must be deterministic. Differential columns must remain exactly home-minus-away. Any feature modification that can alter values is evidence-affecting and requires TEST → PIT → chronological OOS/WFO → calibration → ablation → robustness → frozen holdout → adoption/release before promotion.

A runtime feature schema mismatch between prediction games is a fail-closed condition. Feature metadata is part of artifact integrity and reproducibility.

97. MULTIPLE FEATURE-SET VARIANTS AND EXACT RUNTIME TRACEABILITY

Feature engineering is not a single fixed vector. The system may maintain multiple feature-set variants by league, competition, target contract, runtime lane, PIT-safe context, data availability and research stage.

Canonical variant concepts include:
BASELINE_TEAM_STATE
TEAM_PLUS_STARTER
TEAM_PLUS_BULLPEN
TEAM_PLUS_LINEUP_PIT_SAFE
TEAM_PLUS_WEATHER_PIT_SAFE
FULL_VALIDATED_ENSEMBLE
SCORE_MODEL_FEATURE_SET
RESEARCH_STATCAST_SET

These names do not imply current implementation or production eligibility. The actual runtime output is authoritative.

For every prediction artifact, feature provenance should include:
feature_set_id
feature_manifest_version
feature_count
feature_schema_hash
feature_context_mode
feature/data-quality status
source/data snapshot identifiers

A 482-column NPB or 470-column MLB base matrix is an observed current contract for the inspected match_features path, not a permanent universal requirement. The exact feature columns may differ by validated feature-set variant.

Selection must be based on chronological OOS/WFO, PIT eligibility, coverage/missingness, calibration, robustness, cost and frozen holdout. Larger feature count is not a selection criterion by itself.

⸻

## 93. ULTIMATE PATTERN LAB

`research/ultimate_pattern_lab.py` is the broad pattern-exploration layer for future-generalization research.

### 93.1 Feature-family search
The lab exhaustively enumerates all (2^8=256) subsets of the optional feature families:

* volatility
* starter
* bullpen
* offense
* interaction
* lineup
* weather
* context

Core team-state features remain present in every pattern. Missing required families are recorded as `BLOCKED_UNAVAILABLE_FAMILY`; they are never silently replaced with zero or another family.

### 93.2 Nested chronological stages
Stage A uses only an early pre-holdout chronological OOS band and a lightweight LINEAR_TREE lane for broad feature-ecology screening.

Stage B takes only the Stage-A top eight and evaluates:

* core horizon: ALL / SHORT / LONG
* recency half-life: 600 / 900 / 1800 / 3600 / 7200
* model pool: LINEAR_TREE / BROAD_TREE / DIVERSE

Stage B is evaluated on a later, disjoint chronological OOS band.

Stage C takes only the Stage-B top four and runs the full ensemble path with routing and calibration enabled on a third disjoint chronological OOS band.

This structure is successive-halving by chronology rather than one giant in-sample search.

### 93.3 Frozen holdout
The newest 20% is locked before candidate selection. The Development-selected Stage-C winner is fit on the complete pre-holdout prefix and is then scored on the holdout exactly once.

The holdout is score-only evidence. It cannot select feature families, half-life, model pool, routing, calibration, threshold or production state.

### 93.4 Evidence and release
Expected execution breadth is:

* Stage A: 256
* Stage B: 8 × 3 × 5 × 3 = 360
* Stage C: 4
* locked holdout: 1

All failures remain explicit. Unexpected execution failures fail verification. Successful artifacts remain `RESEARCH_ONLY` with `NO_AUTO_ADOPTION`.

Ultimate-pattern results do not replace the incumbent production champion. Any promotion still requires the normal PIT audit, chronological WFO/OOS, calibration, ablation, robustness, frozen holdout and release gate.
⸻

## 94. EXTREME FEATURE REPRESENTATION LAB

The repository contains a complementary representation-focused laboratory.

### 94.1 Representation search
The lab transforms only already-constructed PIT-safe features. No new source is introduced and no transformation parameter is learned from future rows.

Candidate representations:

* LEVEL
* HOME_AWAY_ONLY
* GAP_ONLY
* GAP_ABS
* GAP_POLY2
* GAP_SIGNED_LOG
* LEVEL_GAP_ABS_LOG
* PAIR_LOG_RATIO

The ratio representation is name-gated to semantically positive/rate-like paired features. No ratio is used merely because a pair exists.

### 94.2 Staged evaluation
Stage A evaluates 16 deterministic feature-family seed patterns × 8 representations = 128 configurations on an early chronological band.

Stage B takes the top 12 and evaluates five recency half-lives × three model pools = 180 configurations on a later disjoint band.

Stage C evaluates the top four using the full ensemble/routing/calibration path on a third chronological band.

The newest 20% remains a locked winner-only holdout.

### 94.3 Safety
Transforms are fixed algebraic operations on PIT-safe columns. Missing/non-finite values fail closed. Representation selection never auto-promotes a model. The exact Git SHA is stored in the artifact and checked by the verifier.

⸻

## 96. HYPERPARAMETER PROFILE EXPLORATION

The model profile frontier supplements feature and representation search. It compares six deterministic estimator configurations—BALANCED, ROBUST, SMOOTH, DEEP, LOCAL and REGULARIZED—using only chronological OOS evidence. Profile choice is never tuned on the frozen holdout and cannot bypass the normal promotion gates.
