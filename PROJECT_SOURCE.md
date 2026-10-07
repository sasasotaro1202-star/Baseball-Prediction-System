Baseball Prediction System — Project Source

ULTIMATE MASTER SPECIFICATION

TARGET:
https://github.com/sasasotaro1202-star/Baseball-Prediction-System

⸻

0. Sourceの役割

本SourceをBaseball Prediction Systemの詳細な技術・研究・データ・PIT・評価・運用・自動改善仕様の正本とする。

Project Instructionsは常時適用される上位行動規則、本Sourceはその実装・研究・検証・運用上の詳細仕様を保持する。

現行GitHubのHEAD、code、config、tests、workflows、Actions、artifacts、registries、実測結果が古い文書や過去会話と矛盾する場合、現行GitHubと検証済みEvidenceを優先する。

ただし、過去のexperiment、failure、holdout、production履歴を後知恵で書き換えて整合させてはならない。

⸻

I. SYSTEM IDENTITY

1. 本当の目的

目的は「勝敗を当てるモデル」ではない。

予測対象は、

P(Y_game | Information_available_before_cutoff)

であり、

「その試合について、prediction cutoff以前に本当に利用可能だった情報から、未来の結果分布を推定すること」

を中核とする。

勝敗は結果分布の一部であり、

* win probability
* draw probability
* score distribution
* total-run distribution
* low/high probability
* run-generation path
* bullpen state
* starter stability
* uncertainty
* predictability
* upset risk
* forecast lifetime

までを統合的に扱う。

⸻

2. Ultimate Objective

最大化するものは単一Accuracyではない。

Future Generalization
× Case-Level Correctness
× Probabilistic Quality
× Calibration
× Predictability Awareness
× Uncertainty Quality
× Robustness
× PIT Integrity
× Information Value
× Selective Prediction
× Operational Reliability
× Recovery
× Reproducibility

を長期的に最大化する。

優先順位:

PIT Integrity

Future Generalization

Calibration

Robustness

Case-Level Error Understanding

Information Value

Operational Reliability

Complexity

⸻

3. Prediction Intelligence System

本システムは単なるPredictorではなく、

Prediction Intelligence System

と定義する。

最終的には、

REAL WORLD
↓
EVENT DISCOVERY
↓
IDENTITY
↓
TIME/PIT
↓
DATA ACQUISITION
↓
DATA QUALITY
↓
FEATURE STATE
↓
TEAM STATE
↓
PLAYER STATE
↓
STARTER STATE
↓
BULLPEN STATE
↓
MATCHUP
↓
REGIME
↓
MODEL ECOLOGY
↓
ENSEMBLE
↓
SIMULATION
↓
CALIBRATION
↓
UNCERTAINTY
↓
PREDICTABILITY
↓
OOD
↓
INFORMATION VALUE
↓
ROUTING
↓
FORECAST
↓
DECISION
↓
PREDICTION LOG
↓
OUTCOME
↓
RECONCILIATION
↓
ERROR ANALYSIS
↓
FAILURE MEMORY
↓
RESEARCH
↓
OOS/WFO
↓
ROBUSTNESS
↓
HOLDOUT
↓
ADOPTION
↓
PRODUCTION
↓
MONITOR
↓
NEXT RESEARCH

という閉ループを形成する。

⸻

II. COMPETITION / TARGET

4. Canonical Scope

主要scope:

* NPB
* MLB

Expansion candidates:

* KBO
* CPBL
* World Baseball Classic
* international tournaments
* NCAA
* high-school baseball
* university baseball
* その他、登録・検証済みの大会

未検証competitionをproductionへ自動投入しない。

⸻

5. Competition Hierarchy

competitionは、

league
→ season
→ competition
→ phase
→ event/game

として管理する。

phase例:

* regular season
* interleague
* postseason
* playoff
* tournament
* qualifier
* exhibition
* unknown

UNKNOWNを別phaseへ推測変換しない。

⸻

6. Competition Policy

competitionごとに可能な限り、

* competition_id
* season_id
* phase
* rule set
* innings rules
* extra innings rules
* home/away semantics
* roster rules
* sample characteristics
* data-source availability
* calibration
* model routing
* production eligibility

を独立管理する。

同一モデルを全competitionへ無条件適用しない。

⸻

7. Canonical Targets

NPB:

HOME
DRAW
AWAY

3-class probability。

MLB:

HOME
AWAY

2-class probability。

NPB drawをMLB binaryへ黙って変換してproduction targetにしない。

⸻

8. Score Targets

勝敗targetとscore targetは独立契約とする。

保持候補:

* home runs
* away runs
* total runs
* expected runs
* exact-score distribution
* Top-4 exact-score candidates
* score interval
* tail probability

Score predictionの評価とwin predictionの評価を混同しない。

⸻

9. Low / High Target

Canonical:

LOW = total runs <= 6
HIGH = total runs >= 7

Low/Highは独立target_versionを持つ。

Low/Highの確率は相互補集合として管理し、1X2やexact-score probabilityと混同しない。

⸻

III. TIME / PIT

10. Time Model

以下のtimestampを分離する。

* event/game time
* prediction time
* prediction cutoff
* published_at
* available_at
* retrieved_at
* observation time
* revision time
* outcome time

特に、

retrieved_at ≠ published_at ≠ available_at

を原則とする。

⸻

11. PIT Rule

基本条件:

available_at <= prediction_cutoff

を満たすこと。

取得時刻だけでは当時利用可能だったことを証明しない。

availabilityが検証できない情報は、

UNKNOWN
または
UNVERIFIABLE

として扱う。

critical PIT unknownはproduction fail-closed。

⸻

12. Forbidden Information

pregame predictionへ、

* future outcome
* postgame statistics
* finalized same-game Statcast
* future standings
* future roster changes
* later correction
* later article
* future starter confirmation
* future market information
* postgame player usage
* final score-derived aggregates

を逆流させない。

⸻

13. PIT Leakage Taxonomy

最低限、

* DIRECT
* TEMPORAL
* REVISION
* ROLLING_WINDOW
* AGGREGATION
* ENTITY
* STARTER
* ROSTER
* POSTGAME
* MARKET
* CALIBRATION
* MODEL_SELECTION
* FEATURE_SELECTION
* ROUTING
* SCOPE
* META_LEAKAGE

を監査する。

OOS結果を見てsource、feature、routing、threshold、研究方針を決める行為もmeta-leakageとして扱う。

⸻

IV. IDENTITY / DATA

14. Identity Contract

基本identifier:

* game_id
* team_id
* player_id
* stadium_id
* competition_id
* season_id

source raw ID/nameも保存する。

Fuzzy matchingは候補発見専用。

silent mergeは禁止。

⸻

15. Data Quality

品質dimension:

* game coverage
* team coverage
* player coverage
* starter coverage
* outcome completeness
* feature completeness
* timestamp completeness
* PIT completeness
* source freshness
* revision integrity
* duplicate rate
* identity integrity
* schema stability
* source reliability

row countだけでcompleteと判断しない。

⸻

16. Missingness

Missingとzeroを完全に分離する。

state例:

UNAVAILABLE
UNKNOWN
NOT_APPLICABLE
DELAYED
SOURCE_FAILED
MALFORMED
NOT_YET_PUBLIC
STRUCTURALLY_ABSENT

critical missing時:

FALLBACK
ABSTAIN
DEFERRED
FAIL

から適切な状態を選択する。

⸻

17. Data Snapshot

各研究・予測snapshotで可能な限り、

* snapshot_id
* dataset_hash
* source_snapshot
* source IDs
* retrieval metadata
* PIT metadata
* schema version
* feature version
* code commit
* time range
* cutoff policy

を保存する。

⸻

V. SOURCE ECOLOGY

18. Source Registry

各sourceに、

* source_id
* owner
* upstream
* endpoint
* data type
* coverage
* historical depth
* freshness
* published_at support
* available_at support
* revision behavior
* parser
* schema
* reliability
* latency
* cost
* license
* last success
* last failure
* production status
* incremental value

を保持する。

⸻

19. Source Independence

以下は独立sourceとして二重計上しない。

* mirror
* wrapper
* copied dataset
* republished CSV
* derived archive
* scraper output of the same upstream

source graphを保持し、evidence independenceへ反映する。

⸻

20. Source Value

sourceの価値はsource数ではなく、

* ΔLogLoss
* ΔBrier
* ΔAccuracy
* calibration improvement
* uncertainty reduction
* failure avoidance
* OOD detection
* coverage improvement
* latency
* reliability
* maintenance cost
* cost

で評価する。

⸻

VI. BASEBALL STATE

21. Team Latent State

Team能力を単純な勝敗列だけで表現しない。

可能な限り、

* long-term strength
* recent form
* offense strength
* pitching strength
* bullpen strength
* defensive strength
* home/away effect
* schedule strength
* rest
* fatigue
* park interaction
* regime

を分離する。

基本思想:

Observed Performance

Latent Ability
+
Current State
+
Context
+
Noise

⸻

22. Offense State

候補:

* batting average
* OBP
* SLG
* OPS
* HR
* BB
* SO
* ISO
* contact
* power
* baserunning
* platoon effects
* recent form
* opponent-adjusted offense

単純なrecent averageだけで能力を定義しない。

⸻

23. Starting Pitcher State

starterについて可能な限り、

* identity
* role
* handedness
* ERA
* FIP
* WHIP
* K/9
* BB/9
* HR/9
* K-BB
* workload
* innings
* pitches
* recent form
* recent variability
* reliability
* rest
* pitch-quality metrics
* contact-quality metrics

を扱う。

starter stateはtime-varyingである。

⸻

24. Starter PIT

starter-dependent featureには、

* announcement status
* probable status
* confirmed status
* publication
* availability
* retrieval
* revision

を分離する。

「後からその投手が先発だったと分かった」だけでは歴史的PIT evidenceとはしない。

⸻

25. Bullpen State

bullpenは単純なERAだけで評価しない。

候補:

* availability
* recent workload
* innings
* appearances
* consecutive-day usage
* leverage usage
* fatigue
* ERA
* WHIP
* K/9
* BB/9
* HR/9
* actual workload coverage
* role availability
* likely late-inning availability

Missing bullpen dataを0へ変換しない。

⸻

26. Lineup State

lineupはtime-varying contextとする。

state例:

ANNOUNCED
PROBABLE
CONFIRMED
SCRATCHED
INJURED
UNAVAILABLE
UNKNOWN

lineup featureはprediction cutoff以前のavailability evidenceが必要。

⸻

27. Player Context

player contextは、

* player_id
* identity status
* position
* bats
* throws
* current-season performance
* career performance
* derived rates
* role
* coverage
* source provenance

を保持可能にする。

未観測選手を0へ補完しない。

⸻

28. Player Role

role例:

* STARTING_PITCHER
* RELIEF_PITCHER
* PITCHER
* CATCHER
* INFIELDER
* OUTFIELDER
* UTILITY_POSITION_PLAYER
* TWO_WAY_CANDIDATE
* UNKNOWN

証拠不足時はUNKNOWNへ倒す。

⸻

VII. PARK / WEATHER / CONTEXT

29. Park Environment

可能な限り、

* stadium
* dimensions
* roof class
* park factor
* handedness-specific effects
* scoring environment

を管理する。

⸻

30. Weather

候補:

* temperature
* apparent temperature
* humidity
* dew point
* precipitation
* precipitation probability
* wind speed
* wind direction
* gust
* pressure
* cloud cover
* weather code

weatherはdynamic sourceとして扱う。

retrieved_atだけでhistorical published availabilityを証明しない。

⸻

31. Schedule Context

候補:

* rest days
* travel
* consecutive games
* doubleheader
* previous game duration
* bullpen workload
* series position
* time-zone transition
* schedule density

future schedule informationの逆流は禁止。

⸻

VIII. FEATURE ECOLOGY

32. Feature States

全featureを、

ACTIVE
CONDITIONAL
OBSERVATION_ONLY
RESEARCH_CANDIDATE

に分類する。

sourceにfeature名が存在するだけでproduction使用とはみなさない。

⸻

33. Current Feature Contract

現行確認されたbase pathでは、

NPB = 482 base features
MLB = 470 base features

が契約レベルの基準として存在する。

ただし、

482/470

を永続的な絶対値とみなさない。

runtime、league、context mode、feature policy、source availabilityにより変化し得る。

⸻

34. Current Feature Families

主なfeature family:

* team recent form
* shrunk form
* venue history
* Elo
* schedule/rest
* team batting
* bullpen
* volatility
* trend
* starter
* matchup
* player context
* lineup context
* weather
* regime
* source quality
* OOD
* uncertainty

⸻

35. Lineup / Weather Status

現行仕様ではlineup/weatherはdefault production featureとして無条件投入しない。

PIT-safe context configurationとcutoff-valid availabilityが必要。

観測・保存とproduction probability投入を分離する。

⸻

36. Feature Lineage

各featureについて可能な限り、

feature_id
source_id
raw_field
transformation
window
available_at
cutoff
revision_policy
quality_status
feature_version
code_commit

を保持する。

⸻

37. Rolling Window

すべてのrolling aggregationについて、

window_end <= prediction_cutoff

を満たす。

future rows、future outcomes、later roster stateを混入させない。

⸻

38. Feature Set as First-Class Object

feature setは、

* feature_set_id
* league
* competition
* target_version
* runtime lane
* context mode
* feature family allow-list
* feature version
* ordered schema
* schema hash
* source snapshot

で識別する。

例:

BASELINE_TEAM_STATE
TEAM_FORM_PLUS_ELO
TEAM_PLUS_STARTER
TEAM_PLUS_BULLPEN
TEAM_PLUS_LINEUP_PIT_SAFE
TEAM_PLUS_WEATHER_PIT_SAFE
FULL_VALIDATED_ENSEMBLE
SCORE_MODEL_FEATURE_SET
RESEARCH_STATCAST_SET

⸻

IX. MODEL ECOLOGY

39. Baselines

必須baseline:

* class frequency
* naive prior
* Elo
* simple logistic
* recent-form baseline

Complex modelはbaselineに対するincremental valueで評価する。

⸻

40. Candidate Models

候補:

* Logistic Regression
* HistGradientBoosting
* RandomForest
* ExtraTrees
* LightGBM
* XGBoost
* CatBoost
* KNN analog
* hierarchical model
* dynamic model
* recent-data expert
* regime expert
* starter expert
* Statcast expert
* specialist model
* ensemble
* calibration model
* fallback model

複雑なmodelほど強いとは仮定しない。

⸻

41. Hierarchical Modeling

global
→ league
→ season
→ team
→ player
→ game

の階層を必要に応じて利用する。

sample不足時にはshrinkageを活用する。

⸻

42. Matchup Modeling

候補:

* batter vs pitcher
* handedness
* platoon
* offense vs starter
* offense vs bullpen
* park vs contact/power
* starter quality vs offense quality
* bullpen fatigue vs opponent offense

correlated featuresを独立votesとして数えない。

⸻

43. Model Routing

routing dimensions:

* league
* competition
* season
* phase
* team strength
* starter state
* bullpen state
* recentness
* regime
* data quality
* uncertainty
* OOD
* sample size

small sample specialistは無理に使用しない。

specific
→ league
→ broader validated
→ baseline
→ abstain

のような階層fallbackを採用可能にする。

⸻

X. GAME GENERATIVE ENGINE

44. Score Generation

理想形では、勝敗だけでなく、

P(HomeRuns = h, AwayRuns = a | PIT-safe state)

を推定する。

candidate engines:

* Poisson
* Negative Binomial
* Tweedie
* tree-based count model
* hierarchical count model
* calibrated score ensemble
* generative simulation

⸻

45. Monte Carlo

可能な限り、

virtual games
→ score paths
→ inning outcomes
→ bullpen transitions
→ extra innings
→ win/draw/loss

を生成する。

勝敗確率はこの分布から導出できる。

⸻

46. Latent Ability Worlds

単一能力値ではなく、

Talent World 1
Talent World 2
Talent World 3
…

を考える。

各worldでgame simulationを行い、

parameter uncertainty
+
game randomness

を分離する。

⸻

47. Inning / State Dynamics

game stateは可能な範囲で、

* inning
* outs
* runners
* score
* pitcher
* batter
* bullpen state
* leverage
* fatigue

を持つstate machineとして研究する。

同一timestampのgame stateをfreezeしてから結果を進める。

⸻

48. Tail Risk

平均scoreだけではなく、

* blowout
* low-scoring game
* unexpected offensive spike
* starter early exit
* bullpen collapse
* extra innings

等のtail eventを評価する。

⸻

XI. CALIBRATION / UNCERTAINTY

49. Probability / Confidence / Predictability

明確に分離する。

Probability ≠ Confidence

Confidence ≠ Predictability

Predictability ≠ Accuracy

⸻

50. Calibration

候補:

* none
* temperature
* sigmoid
* beta-style
* isotonic
* temporal calibration
* suitable online calibration

metrics:

* LogLoss
* Brier
* ECE
* calibration slope
* calibration intercept
* reliability curve

⸻

51. Uncertainty Decomposition

可能な限り、

* aleatoric
* epistemic
* data
* source
* starter
* lineup
* bullpen
* temporal
* regime
* OOD
* disagreement

を分離する。

単なる最大確率との差をuncertaintyの唯一指標にしない。

⸻

52. Model Disagreement

候補:

* probability variance
* entropy gap
* KL divergence
* Jensen-Shannon divergence
* ranking disagreement

disagreement spikeは、

* case review
* more information
* recompute
* fallback
* abstention

のtrigger候補。

⸻

53. Predictability

game-level predictabilityを独立評価する。

candidate inputs:

* model agreement
* data completeness
* starter certainty
* lineup certainty
* source agreement
* OOD
* regime ambiguity
* recent volatility
* calibration stability
* outcome entropy

high confidence low predictabilityを危険状態として認識する。

⸻

54. OOD

未観測状態を検出する。

候補:

* unseen player
* unseen team
* sparse history
* unusual starter
* unusual lineup
* extreme scoring
* source conflict
* regime transition
* feature shift
* data distribution shift

OODはconfidenceとは別指標。

⸻

XII. INFORMATION ACQUISITION

55. Action Layer

候補action:

PREDICT_NOW
ACQUIRE_MORE
WAIT
RECOMPUTE
FALLBACK
ABSTAIN

⸻

56. Information Value

追加情報の価値を、

Expected Information Gain
× Probability of Material Update
× Source Reliability

から推定し、

latency
+
cost
+
failure risk
+
cutoff proximity

も考慮する。

情報取得は多ければよいわけではない。

⸻

57. Forecast Lifetime

prediction state:

FRESH
AGING
STALE
UNKNOWN
SHOCKED
REQUIRES_RECALC
FALLBACK
ABSTAIN
INVALIDATED

starter change、lineup update、major source revision、weather shockなどで再評価する。

⸻

XIII. SELECTIVE PREDICTION

58. Selective Policy

全gameで無理にpredictionを出す必要はない。

predict
fallback
defer
acquire
abstain

を選択できる。

評価:

* coverage
* selective risk
* calibration
* stability
* utility
* false-abstention cost

abstentionは失敗ではなく、適切な条件では成功。

⸻

59. Conformal / Risk Control

research candidate:

* split conformal
* adaptive conformal
* online conformal
* local conformal
* prediction sets
* risk-controlling prediction
* selective risk

coverage、risk、set size、stabilityを評価する。

⸻

XIV. OOS / WFO / HOLDOUT

60. Evaluation Structure

基本:

TRAIN
→ VALIDATION
→ WALK-FORWARD OOS
→ ROBUSTNESS
→ FROZEN HOLDOUT

random splitをproduction evidenceとしない。

⸻

61. Candidate Selection

Candidate selectionとfinal OOS evaluationを分離する。

同一OOSを繰り返し見て選択するとeffective test set contaminationとなる。

⸻

62. Primary Metrics

Primary:

LogLoss

Secondary:

* Accuracy
* Brier
* ECE
* class-wise LogLoss
* calibration slope/intercept

⸻

63. Score Metrics

candidate:

* home-run MAE
* away-run MAE
* total-run MAE
* exact-score Top1
* exact-score Top4
* probabilistic score metric
* Low/High metrics
* distribution calibration

⸻

64. Starter-Specific Evaluation

starter-known casesとstarter-unknown casesを分けて評価する。

確認項目:

* starter availability
* starter sample size
* starter missingness
* announcement timing
* fallback frequency

「先発が分かったgameだけ改善した」のか、
「production population全体で改善した」のかを分離する。

⸻

65. Robustness Matrix

最低限、

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
* new player/team
* source outage
* lineup missingness
* weather missingness

を評価する。

⸻

66. Statistical Integrity

必要に応じて、

* paired comparison
* block bootstrap
* game-cluster bootstrap
* confidence interval
* permutation
* forecast comparison
* multiple-comparison correction

を使う。

single fold・single metric differenceだけでsuperiorityを断定しない。

⸻

67. Reference Adoption Gate

参考基準:

relative primary LogLoss improvement >= 3%

auxiliary improvement >= 1%

evaluation periods without worsening >= 70%

さらに、

* PIT valid
* leakage audit pass
* calibration safety
* robustness
* newest holdout no worsening
* sufficient sample
* reproducibility
* operational safety

を要求する。

閾値は絶対法則ではなく、sample size、variance、confidence interval、event dependence、cost、riskを考慮する。

⸻

68. Frozen Holdout Firewall

Holdoutは、

* feature selection
* model selection
* hyperparameter tuning
* calibration
* routing
* source selection
* threshold tuning
* scope selection
* research prioritization

に使わない。

汚染が疑われた場合、

HOLDOUT = INVALID

として再freezeする。

⸻

XV. EXPERIMENT / RESEARCH

69. Experiment Schema

experimentには、

* experiment_id
* fingerprint
* hypothesis
* scope
* target_version
* dataset
* PIT status
* feature_set
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

を保存する。

⸻

70. Experiment Fingerprint

fingerprint候補:

git commit
+
dataset hash
+
source snapshot
+
feature version
+
target version
+
model config
+
calibration config
+
routing policy
+
cutoff policy
+
seed
+
environment

同一fingerprintはcache reuseを優先する。

⸻

71. Research Router

研究を、

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

に分類する。

研究portfolio:

* exploit
* adjacent
* frontier
* replication
* ablation
* adversarial
* recovery
* meta-research

⸻

72. External Research Ingestion

外部論文、GitHub、研究実装、AI生成知見などは直接productionへ入れない。

必ず、

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

を通す。

⸻

73. Evidence Level

E0 = idea
E1 = external claim
E2 = external implementation
E3 = local reproduction
E4 = local OOS
E5 = robustness
E6 = frozen holdout
E7 = production evidence

「GitHubにコードがある」と「未知未来で性能が証明された」を同一視しない。

⸻

XVI. FAILURE INTELLIGENCE

74. Wrong Prediction vs System Failure

モデルが低確率事象を外しただけの場合と、

* PIT violation
* data corruption
* wrong identity
* stale source
* wrong target
* broken calibration
* runtime corruption

などのsystem failureを区別する。

⸻

75. Failure Taxonomy

最低限:

DATA_FAILURE
PIT_FAILURE
TIMESTAMP_FAILURE
REVISION_FAILURE
IDENTITY_FAILURE
SOURCE_FAILURE
FEATURE_FAILURE
STARTER_FAILURE
LINEUP_FAILURE
BULLPEN_FAILURE
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

76. Failure Record

保存:

* failure_id
* game_id
* component
* type
* severity
* input_state
* prediction
* expected
* observed
* root_cause
* counterfactual
* repair
* verification
* recurrence

⸻

77. Counterfactual Failure Analysis

失敗後、

* earlier information
* additional source
* alternate model
* alternate calibration
* alternate routing
* specialist
* fallback
* abstention

で回避できたかをReplayする。

factとhypothesisを分離する。

⸻

78. High-Confidence Wrong

特に、

P(predicted outcome) が非常に高い
なのにwrong

というcaseを重点調査する。

候補原因:

* calibration failure
* regime shift
* OOD
* source error
* stale data
* model blind spot
* hidden uncertainty

⸻

79. Negative Knowledge

保存:

* rejected model
* rejected feature
* rejected source
* rejected routing
* rejected calibration
* rejected competition
* PIT failure
* robustness failure
* computational failure
* failed research hypothesis

失敗もknowledge baseとして再利用する。

⸻

XVII. EXPERIENCE / RECONCILIATION

80. Canonical Experience Unit

canonical unitはgame_id。

同一gameの複数revisionを独立caseとして水増ししない。

⸻

81. Prediction Lifecycle

CREATED
→ PIT_CHECK
→ DATA_CHECK
→ FEATURE_CHECK
→ MODEL_CHECK
→ CALIBRATION_CHECK
→ PREDICTED
→ MONITORED
→ MATURED
→ RECONCILED

failure時:

FALLBACK
ABSTAIN
DEFERRED
INVALIDATED

へ遷移可能。

⸻

82. Outcome Lifecycle

PENDING
→ IMMATURE
→ MATURE
→ VERIFIED
→ REVISED

MATUREになる前の結果をexperience learningへ無条件投入しない。

⸻

83. Reconciliation

outcome確定後、

* prediction
* actual
* target version
* cutoff
* source snapshots
* feature version
* model version
* calibration
* PIT status
* production state
* fallback
* abstention
* error type
* competition taxonomy

をreconcileする。

postgame情報でprediction-time taxonomyを上書きしない。

⸻

XVIII. PRODUCTION

84. Champion / Challenger

Champion:

Current Production-approved bundle

Challenger:

Controlled candidate under validation

Candidate existsだけではpromotion candidateではない。

⸻

85. Production Bundle

Productionはmodel fileだけではない。

最低限、

* model artifact
* feature schema
* feature manifest
* source registry version
* PIT policy
* target definition
* calibration
* router
* fallback
* output schema
* monitoring
* rollback target
* manifest
* artifact hash

を一体として扱う。

⸻

86. Runtime Verification

起動時に、

registry
↔ metadata
↔ artifact
↔ schema

を検証する。

さらに、

* class count
* feature count
* probability shape
* finite values
* probability sum
* hash
* target version
* calibration version

を確認する。

⸻

87. Runtime Candidate Isolation

Production Runtimeが、

「最新のresearch model」

を勝手に選ばない。

Runtime selectionは明示されたProduction Registryだけを参照する。

⸻

88. Production Fail-Closed

以下では無理にproduction predictionを生成しない。

* model missing
* artifact corruption
* critical source failure
* required data stale
* PIT unknown
* identity mismatch
* target mismatch
* feature contract failure
* invalid calibration
* schema mismatch

⸻

89. Fallback Chain

基本:

Champion
→ Validated Specialist
→ Generalist
→ Baseline
→ Abstain

fallback使用をprediction logへ保存する。

Fallback結果をChampionと同等に表現しない。

⸻

XIX. AUTOMATION

90. GitHub Actions

Automationは単なるschedule実行ではなく、

* checkpoint
* resume
* idempotency
* bounded retry
* backoff
* watchdog
* heartbeat
* stale-run detection
* deterministic writes
* artifact preservation
* concurrency control
* recovery
* rollback

を持つ。

⸻

91. Long-Running Research

長時間OOSは途中状態を失わない。

checkpointへ、

* checkpoint_id
* stage
* scope
* input snapshot
* completed outputs
* pending work
* expected next state
* artifact hashes
* recovery safety

を保存する。

⸻

92. Single Writer

critical registry:

* model registry
* experiment registry
* source registry
* experience ledger
* promotion state
* rollback state
* scope state

はsingle-writer semanticsを優先する。

parallel researchは可能だがmergeはdeterministicにする。

⸻

93. Candidate OOS Continuity

Candidate OOSはlong-running validationとして扱う。

検証対象を変えないcontinuity-only changeと、evidence-affecting changeを区別する。

runtime、configuration、PIT、evaluation、candidate identityに影響する変更はfresh current-main OOSを要求する。

⸻

94. Recovery

stale workflowやfailed runを無条件rerunしない。

まず、

* current SHA
* current state
* active run
* checkpoint
* artifact
* cancellation state
* duplicate-run risk

を確認し、verified recoveryだけを実行する。

⸻

XX. EFFICIENCY / COST / SECURITY

95. Compute Optimization

順序:

cache
→ exact snapshot reuse
→ incremental update
→ deduplication
→ vectorization
→ parallel I/O
→ selective recomputation
→ training optimization
→ algorithm optimization

同一fingerprintを重複計算しない。

⸻

96. Cost Firewall

優先順位:

verified free
→ free quota
→ OSS/local
→ cached/local snapshot
→ lightweight compute

禁止自動依存:

* paid-only
* billing-risk
* unknown-cost
* auto-renew trial
* quota-overage risk

cost不明 = HOLD / UNCONFIRMED

⸻

97. Security

secret、API key、tokenを、

* code
* log
* artifact
* report
* commit

へ出力しない。

external sourceは、

* license
* attribution
* redistribution
* rate limit
* retention
* commercial restriction

を確認する。

⸻

XXI. PLUGIN / CONNECTOR INTELLIGENCE

98. Plugin / Connector Routing

利用可能なplugin・connector・外部調査手段は、目的に応じて使い分ける。

基本routing:

GitHub
→ repository/code/config/Actions/evidence/current state

Web / Search
→ current public information / source discovery / schedule/context

Academic / research connector
→ papers / methodology / prior research

Structured data connector
→ current structured data / metrics / datasets

File / Project knowledge
→ project-local source / prior research / uploaded master specifications

⸻

99. Plugin Evidence Firewall

pluginから得た情報は、

DISCOVERY
→ SOURCE_VERIFICATION
→ TIME_VERIFICATION
→ PIT_CHECK
→ COST_CHECK
→ SECURITY_CHECK
→ LOCAL_REPRODUCTION
→ OOS
→ ROBUSTNESS
→ HOLDOUT

を通さずproduction evidenceにしない。

plugin resultそのものをモデル性能の証拠とはしない。

⸻

100. Plugin Cost / Availability

plugin利用で、

* cost
* quota
* billing
* rate limit
* account dependency

が不明な場合、自動的にcritical dependencyへしない。

無料・安全・再現可能な手段を優先する。

⸻

XXII. CROSS-PROJECT TRANSFER

101. Five-Repository Transfer Firewall

他prediction projectから学ぶ際は、

DISCOVER
→ ABSTRACT MECHANISM
→ COMPATIBILITY CHECK
→ ADAPT
→ LOCAL PIT
→ LOCAL OOS
→ ROBUSTNESS
→ HOLDOUT
→ SHADOW
→ PROMOTE

とする。

他projectのaccuracyやOOS結果をBaseballの実績として流用しない。

transfer対象は原則mechanismである。

⸻

XXIII. RESEARCH LABORATORIES

102. Pattern Research

research-only laboratoryでは、

* feature-family combinations
* mathematical representations
* model pools
* routing variants
* score-model composition
* mean shrinkage
* ensemble combinations

を広く探索できる。

⸻

103. Extreme Representation Research

同一PIT-safe feature matrixについて、

* level
* home/away
* gap
* absolute gap
* squared gap
* signed-log gap
* level + gap
* gated log-ratio

などのrepresentationを候補化できる。

selectionはchronological OOSのみで行う。

⸻

104. Score Distribution Research

score distributionについて、

* scoring models
* shared correlation
* mean shrinkage
* tail behavior
* low/high mapping
* exact-score ranking

を研究する。

Score research resultは1X2 productionへ自動移植しない。

⸻

XXIV. RESEARCH FRONTIER

105. Research Frontier Scan

定期的に、

Data Frontier
Source Frontier
Model Frontier
Feature Frontier
Simulation Frontier
Uncertainty Frontier
Failure Frontier
Scope Frontier
Automation Frontier
Unknown Frontier

をscanする。

目的は複雑化ではなくsystem limitationの発見。

⸻

106. Research Priority

ResearchNextは概念的に、

ResearchNext

argmax[
Expected Future Value

Complexity

Operational Risk
]

とする。

短期backtest gainだけで研究優先度を決めない。

⸻

107. Stopping Rule

以下では研究をHOLD/STOP可能:

* repeated zero incremental value
* insufficient PIT
* insufficient sample
* source quality unresolved
* robustness failure
* excessive computation
* duplication
* maintenance burden
* frontier saturation

停止理由をNegative Knowledgeへ保存する。

⸻

XXV. STATE CONSISTENCY

108. BLOCK Conditions

以下の矛盾はBLOCK対象:

PIT FAIL + PRODUCTION ACTIVE

MODEL REGISTRY PRODUCTION + MODEL MISSING

MODEL HASH MISMATCH

TARGET VERSION MISMATCH

FEATURE SCHEMA MISMATCH

HOLDOUT INVALID + PROMOTION VALID

STALE REQUIRED SOURCE + TRUSTED PRODUCTION

SNAPSHOT MISMATCH

UNKNOWN CRITICAL IDENTITY + VALIDATED PRODUCTION

⸻

109. Online / Offline Parity

同一snapshotについて、

Historical Pipeline
vs
Production-like Pipeline

を比較し、

* features
* missingness
* timestamps
* inputs
* probability
* calibration
* routing
* state

のparityを検証する。

⸻

110. Deterministic Replay

再現に必要な要素:

* git SHA
* environment
* dependencies
* config
* seed
* data snapshot
* source snapshot
* feature schema
* model artifact
* calibration
* routing
* target definition

same inputからequivalent outputを再現可能にする。

⸻

XXVI. ADVERSARIAL / CHAOS VALIDATION

111. Adversarial Tests

意図的に、

* future timestamp injection
* postgame field injection
* future starter injection
* future standings
* later revision
* future roster state
* stale source
* missingness
* source removal
* feature deletion
* time shift
* distribution shift
* regime transition
* unseen player
* unseen team

を攻撃する。

failしたcandidateはproduction不可。

⸻

112. Chaos Tests

例:

* source timeout
* API outage
* schema change
* duplicate
* corrupt timestamp
* stale data
* bad artifact
* cancelled workflow
* dependency failure
* resource exhaustion

systemが、

FULL
→ REDUCED
→ FALLBACK
→ SELECTIVE
→ ABSTAIN
→ RECOVERY

へ安全に遷移できることを確認する。

⸻

XXVII. OUTPUT CONTRACT

113. Final Prediction Object

可能な限り、

{
“game_id”: “…”,
“competition_id”: “…”,
“season_id”: “…”,
“phase”: “…”,
“game_time”: “…”,
“prediction_time”: “…”,
“prediction_cutoff”: “…”,
“data_as_of”: “…”,

“home_probability”: 0.638,
“draw_probability”: 0.104,
“away_probability”: 0.258,

“score_top4”: [
{“score”:“5-4”,“probability”:0.061},
{“score”:“4-3”,“probability”:0.057},
{“score”:“4-4”,“probability”:0.051},
{“score”:“5-3”,“probability”:0.047}
],

“low_probability”: 0.42,
“high_probability”: 0.58,

“uncertainty”: {
“aleatoric”: “…”,
“epistemic”: “…”,
“data”: “…”,
“source”: “…”,
“starter”: “…”,
“lineup”: “…”,
“bullpen”: “…”,
“ood”: “…”
},

“predictability”: “…”,
“model_disagreement”: “…”,
“regime”: “…”,
“data_quality”: “…”,
“pit_status”: “…”,

“model_version”: “…”,
“feature_set_id”: “…”,
“feature_manifest_version”: “…”,
“feature_count”: 0,
“feature_schema_hash”: “…”,

“calibration_version”: “…”,
“routing_state”: “…”,
“fallback_state”: “…”,

“generation_status”: “…”,
“production_eligibility”: false,

“git_sha”: “…”,
“dataset_hash”: “…”,
“source_snapshot_id”: “…”,
“request_id”: “…”
}

NPBとMLBでclass semanticsはcompetition contractに従う。

⸻

XXVIII. DECISION OBJECT

114. Predictionの裏側

単に、

A = 64%

と出すだけでは不十分。

理想:

A Win Probability = 64%
Probability interval = …
Predictability = medium-high
Model disagreement = low
Data quality = high
PIT = PASS
Starter certainty = high
Bullpen certainty = medium
Main win paths = …
Main loss paths = …
Largest uncertainty = …
Tail risk = …
Forecast lifetime = …
Counterfactual = …
Action = PREDICT_NOW

まで追跡する。

⸻

XXIX. MODEL BLIND SPOTS

115. Information Diversity

複数modelが同じ方向を出していても、

同一source
同一feature
同一representation

なら本当のdiversityとは限らない。

評価対象:

model diversity
+
feature diversity
+
source diversity
+
assumption diversity

⸻

116. Source Conflict

Source AとSource Bが矛盾する場合、

単純平均しない。

考慮:

* availability
* reliability
* independence
* freshness
* coverage
* historical error
* identity confidence

⸻

XXX. COMPLETION

117. Completion Definition

以下だけではcompletionではない。

* code exists
* workflow exists
* Action green
* prediction JSON exists
* model runs

completion requires evidence connecting:

SPEC
+
CODE
+
DATA
+
TIME
+
PIT
+
LEAKAGE
+
TARGET
+
OOS/WFO
+
CALIBRATION
+
ROBUSTNESS
+
HOLDOUT
+
SHADOW
+
REPRODUCIBILITY
+
RECOVERY
+
MONITORING
+
ROLLBACK
+
STATE CONSISTENCY
+
KNOWLEDGE LINEAGE

⸻

118. Status Taxonomy

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

code exists ≠ adopted

Action green ≠ performance verified

artifact exists ≠ production validated

⸻

119. No Fake Success

禁止:

* fabricated metrics
* silent exception
* missing→zero
* failed job→success
* skipped test→passed
* unknown PIT→valid
* incomplete data→complete
* failed recovery→recovered
* old evidence→current evidence
* research result→production result

⸻

XXXI. CONTINUOUS SELF-IMPROVEMENT

120. Permanent Loop

MONITOR
→ DETECT
→ TRIAGE
→ UNDERSTAND
→ RESEARCH
→ HYPOTHESIS
→ IMPLEMENT
→ TEST
→ PIT
→ OOS/WFO
→ CALIBRATION
→ ROBUSTNESS
→ FROZEN HOLDOUT
→ SHADOW
→ ADOPT/HOLD/REJECT
→ RELEASE
→ PRODUCTION
→ RECONCILE
→ FAILURE ANALYSIS
→ MEMORY
→ NEXT RESEARCH

を継続する。

⸻

121. Ultimate Principles

1. Future Generalization > Historical Fit
2. PIT > Apparent Backtest Gain
3. Evidence > Assumption
4. Calibration > Raw Confidence
5. Robustness > Single-Fold Gain
6. Case-Level Understanding > Aggregate Comfort
7. Missing ≠ Zero
8. Retrieval ≠ Availability
9. Probability ≠ Confidence
10. Confidence ≠ Predictability
11. Disagreement is Information
12. Abstention can be Success
13. Failure is Knowledge
14. Production is a Bundle
15. Runtime never invents Production candidates
16. More complexity requires more evidence
17. Source count is not evidence independence
18. OOS ≠ Holdout
19. Operational success ≠ Research success
20. Workflow completion ≠ Model verification
21. Historical truth is immutable
22. Unknown must remain explicitly unknown
23. Safe Degradation > False Prediction
24. Reproducibility > convenient output
25. The system must continuously search for reasons it is wrong

⸻

XXXII. FINAL SYSTEM DEFINITION

Baseball Prediction Systemとは、

「野球の現在世界をprediction cutoff時点の正しい情報だけから再構成し、team/player/starter/lineup/bullpen/park/weather/schedule/regimeを状態として推定し、複数のmodelとgenerative simulationから未来の結果分布を生成し、その確率をcalibrateし、uncertainty・predictability・OOD・model disagreementを評価し、必要なら追加情報取得・再計算・fallback・abstentionを選択し、predictionとそのprovenanceをimmutableに記録し、結果成熟後にreconcileし、失敗をfailure memoryへ変換し、その知識から次のresearchを自律的に決定し、PIT-safeなOOS/WFO・robustness・frozen holdoutを通過した改善だけを安全にproductionへ昇格させ、失敗時にはrollback・recoveryできるPrediction Intelligence System」

である。

最終的な研究対象は、

P(
Future Baseball Outcome
|
Correct Point-in-Time Information
)

である。

さらに、

Prediction Intelligence

Prediction
+
Calibration
+
Uncertainty
+
Predictability
+
Information Acquisition
+
Decision
+
Failure Learning
+
Self Improvement
+
Operational Reliability

と定義する。

⸻

FINAL LAWS

NO EVIDENCE, NO CLAIM.

NO PIT PROOF, NO HISTORICAL TRUST.

NO ROBUSTNESS, NO PROMOTION.

NO CALIBRATION, NO RELIABLE PROBABILITY.

NO REPRODUCIBILITY, NO DURABLE KNOWLEDGE.

NO SAFE FALLBACK, NO AUTONOMOUS OPERATION.

NO IMMUTABLE HISTORY, NO TRUSTWORTHY EXPERIENCE.

NO LOCAL VALIDATION, NO CROSS-PROJECT ADOPTION.

NO UNKNOWN HAND-WAVING, NO FAIL-OPEN.

NO CONTINUOUS MONITORING, NO CONTINUOUS IMPROVEMENT.