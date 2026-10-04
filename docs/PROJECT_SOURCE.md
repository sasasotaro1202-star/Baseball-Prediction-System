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
## Candidate OOS continuity contract
Candidate OOS snapshot freshness may survive only non-runtime continuity updates: `data/experience/**`, `tests/**`, or `.github/workflows/baseball_candidate_oos_watchdog.yml`. Candidate runtime code, configuration, PIT inputs, evaluation logic, the candidate workflow itself, and other data remain evidence-affecting and must fail closed. The watchdog is control-plane orchestration only; it must not be treated as candidate runtime evidence.
