# スポーツ予測モデル 仕様書(サッカー・野球)

目標: 各予測市場で最低75%の正答率を目指す(100%は野心的目標として扱う)。関連データはあらゆるソースから自由に取得可。

## 処理パイプライン(共通)

```
RAW DATA → NORMALIZATION → CHRONOLOGICAL STATE → FEATURE ENGINEERING
→ MODEL-SPECIFIC FEATURES → MULTIPLE MODELS → ENSEMBLE → CALIBRATION → FINAL PREDICTION
```

「最大5000候補要因」とは、5000個を無条件投入する意味ではなく、複数期間・複数条件・チーム差分・対戦差分・トレンド・交互作用・モデル別特徴量・市場との差分・環境補正・先発/ブルペン・選手/ラインアップ等を含めて候補空間を広く探索し、Chronological OOSで実際に有効性を検証したものを採用する。未来情報でバックテスト精度を水増しする特徴量は採用しない。

## サッカー予測モデル

### 対象大会
Premier League, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, J1, J2, J3, UEFA Champions League, UEFA Europa League, DFB-Pokal, Club Friendly Games

### 使用データ一覧
1. 試合基本データ: 試合日/ホーム/アウェイ/得点/結果(H/D/A)/シュート数/枠内シュート/コーナー数
2. 市場・オッズ: Bet365, Betway, Interwetten, Pinnacle, William Hill, VC, 平均/最大/Closing Odds, 市場Home/Draw/Away/Favorite確率, 市場Entropy, オッズ分散
3. Understat: Home/Away xG・xGA, 過去xG/xGA, xG for/against
4. チーム成績: Elo, Global Elo, 勝敗数, 勝ち点, 得失点(/試合), 最近の勝ち点・得点・失点, Home/Away別成績, EMA得点/失点
5. 試合パフォーマンス: Shots, SOT, Corners, xG, xGA, 得点/シュート効率
6. 休養・日程: Rest Days(Home/Away/差), 試合間隔
7. H2H: 過去対戦数, 勝率/引分率, 直近H2H(最大8試合)
8. SofaScore選手データ: 攻撃(Goals, Assists, xG, xA, Shots, Key Passes, Big Chances等), パス, ドリブル, 守備(Tackles, Interceptions等), デュエル, ボール保持, 規律(Fouls, Cards), GK(Saves, Goals Prevented)
9. 選手基本情報: Position, Foot, Height, Age, Matches, Minutes, Rating
10. 選手生成特徴量: Finishing, Shot Volume, Chance Creation, Progression, Passing, Defending, Aerial, Duel, Discipline, Ball Security, Goalkeeping等
11. チーム選手層: Attack, Creation, Progression, Defending, Squad Depth, Top Attack/Creation/Defending等
12. Global Team Strength: Global Elo, Global Elo差, リーグ横断強度
13. Club Friendlyデータ: 参加情報, Friendlyフラグ, Low Informationフラグ
14. MLモデル: Logistic Regression, ExtraTrees, RandomForest, HistGradientBoosting
15. ML内部処理: Chronological/Walk-Forward Validation, Recency Weighting, LogLoss, Temperature Calibration
16. スコアモデル: Home/Away Elo, 攻撃力/守備力, 得点期待値, Poisson分布, Score Probability
17. 最終1X2予測: ML確率 + Closing Market確率 + Poisson確率 → 最終Home/Draw/Away確率, Confidence
18. 最終ブレンド: ML=56%, Market=28%, Score=16%(初期値、過去OOS結果に基づき動的最適化)
19. MOM予測用: 過去Rating/Minutes/Goals/Assists/xG/Key Passes, チーム勝率, 選手Elo相当強度, 対戦相手強度
20. データ品質・リーク防止: 試合日時順処理、過去データのみ使用、未来データ禁止、Chronological/Walk-Forward Validation, Checkpoint Resume

### 最終予測構造
過去試合データ + チーム成績 + Elo/Global Elo + xG/xGA + Shots/SOT/Corners + Home/Away成績 + 休養日 + H2H + 過去選手データ + 選手層 + Closing Market
→ 特徴量生成 → (Logistic Regression + ExtraTrees + RandomForest + HistGradientBoosting) → ML確率 → Temperature Calibration → ML Ensemble → + Market Probability → + Poisson Score Probability → 最終Home/Draw/Away確率 → 勝敗予測

## 野球(MLB/NPB)予測モデル

### 使用データ・特徴量一覧(34カテゴリ)
1. 試合基本情報(league, game_id, date, venue, season等)
2. チーム基本成績(勝敗、勝率、得失点、RPG)
3. 直近チームフォーム(3/5/10/20/30/45/60試合)
4. ホーム成績
5. アウェイ成績
6. Elo(team/opponent/home/away/expected result/movement)
7. 休養(rest days, congestion)
8. ブルペン(直近3/7日投球回, ERA系, 疲労度)
9. チーム打撃(3-30試合区間のAB/PA/H/HR/BB/SO/打率/BB率/SO率等)
10. 打撃変動性(標準偏差・傾き, 直近20試合)
11. 選手プロファイル(PA/AB/H/HR/BB/SO/OBP/SLG/ISO等)
12. ラインアップ(打順、出場可否、ラインアップ強度)
13. 先発投手(履歴、直近成績、ハンド、対戦相手マッチアップ)
14. MLB投手Statcast系(velocity, spin rate, xBA, xSLG, xwOBA, run value等)
15. MLB打者Statcast系(exit velocity, launch angle, barrel rate, xwOBA等)
16. 対戦相性(チーム対戦相手、先発対対戦相手、ハンドマッチアップ)
17. 球場(park factor, run environment, HR environment)
18. 天候(温度、湿度、風速風向、屋根状態)
19. 市場・オッズ(moneyline, implied probability, closing odds, market edge) ※未来市場情報の混入禁止
20. スケジュール(前後試合日程、ダブルヘッダー、シリーズ文脈)
21. シーズン・時間軸(month, day of season, early/mid/late season)
22. 得点モデル(期待得点、Poissonパラメータ、スコア分布)
23. Low/High(Low=両チーム0-6、High=いずれか7+の確率)
24. 勝敗モデル: NPB(Home/Draw/Away), MLB(Home/Away)。候補: Logistic Regression, HistGradientBoosting, RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost, Elo-based, market-based
25. アンサンブル(モデル不一致度、重み付け、キャリブレーション済み確率)
26. 確率校正(Brier score, LogLoss, calibration error)
27. 条件別モデル(home/away, favorite/underdog, close game, 先発確定/未確定, 早期/後期シーズン)
28. データ品質(欠損、重複、チーム名不一致、リーク検知)
29. 時系列リーク防止(prediction cutoff, 過去データのみ、拡張ウィンドウ学習)
30. 学習・検証(expanding walk-forward, OOS prediction)
31. 精度評価(accuracy, AUC, LogLoss, Brier, スコアMAE, Low/High精度)
32. 先発評価(先発カバー率、先発別精度)
33. エラー分析(予測誤差の要因分解)
34. 研究・自動改善用(弱い特徴量検出、feature drift、改善候補管理)

### 重要な原則
未来情報を使ってバックテスト精度を水増しする特徴量は採用しない。全データはChronological/Walk-Forward Validationで検証し、実際に有効性が確認されたものだけを採用する。
