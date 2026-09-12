# 出力仕様の実装ノート (prediction_model_output_spec.md への対応)

## 実装済み(Stage 0)

- **MLBスコア予測(4択, %)**: `parallel/score_distribution.py` の Poisson近似 (season-average lambda)。100%への正規化はしない。
- **MLB Low/High**: `parallel/lowhigh_model.py`。Low=合計6点以下, High=合計7点以上。
- **サキカーHDA(regulation-time draw rule)**: `parallel/soccer_outcome_model.py`。延長戦/PK戦の勝者情報はラベリングに使用せず、レギュレーショコチいガド専の得昆すべとレヺズ】となやす。
- **サキカースコア予測(4択, %)**: `score_distribution.py` を共用。

## 未実装(データ不足によりスタブ)

- **MOM予測(3択, %)**: `parallel/mom_model.py` は `status: insufficient_data` を返す。選手レベルデータ(SofaScore, goals/assists/xG/rating)が未アフロードのため。データ投入後、`predict_mom_top3(player_features=...)` にリストを渡せば動作する。

## 既知の制約

現在のスコア分布はシーズン平均のみに基づくStage-0近似。先発投手・ブルペン・球場factor・天候・xGを組み込むには、`prediction_model_specification.md` の該当セキションのデータをアップロードし、`parallel/feature_store.py` にステージを追加する必要がある。
