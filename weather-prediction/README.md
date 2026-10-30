# weather-prediction(天気予測)

気象庁の観測データ(青森・弘前)から、翌日の天気(晴/曇/雨/雪の4分類)を予測する機械学習モデルの実験コードです。

## 構成

```
code/
  download_jma_data.py               気象庁の「過去の気象データ」ページから月別CSVを取得
  preprocess.py                      前処理・特徴量作成・ラベル付け(data/processed/dataset.csv を出力)
  train_model.py                     ベースライン(ロジスティック回帰)
  compare_models.py                  5手法の比較(LR / 決定木 / ランダムフォレスト / 勾配ブースティング / kNN)
  tune_gradient_boosting.py          勾配ブースティングのGridSearchCVチューニング
  tune_gradient_boosting_optuna.py   同、Optuna(TPE)によるチューニング
  staged_search.py                   n_estimatorsの水準を1回の学習から評価するGrid探索(GridSearchCVと同じCVスコアで、約3.4倍速い)
  seed_robustness.py                 seedを変えた再実行とブートストラップによる差の検証
  tune_same_space.py                 Optunaと同じ探索空間でのGridSearchCV / RandomizedSearchCV
  gb_balanced_weights.py             勾配ブースティングにもクラス重みを付けて再評価(ロジスティック回帰との公平な比較)
  untested_items.py                  LightGBM・XGBoost、ロジスティック回帰の調整、クラス重みの強さを試して、ロジスティック回帰と比較
  ablation_evening_features.py       夜に作れる特徴量(今日の天気概況・降雪を除く)での再評価
  train_final.py                     全データで再学習し、モデルを保存
  fetch_realtime.py                  気象庁の10分ごとの観測を取得し、日別の特徴量に集計
  jma_daily.py                       直近の日別データを取得・数値化(前処理と同じ規則)
  predict_tonight.py                 夜に翌日の天気(4クラスの確率)を予測してログに記録
  train_hirosaki.py                  弘前の翌日の最高・最低気温とルール天気のモデルを学習して保存
  hirosaki_tonight.py                夜に弘前の予測(最高・最低気温、ルール天気)を行い、別のログに記録
  score_hirosaki.py                  弘前の予測を、後日の実測と照合
  fetch_nwp.py                       数値予報(気象庁GSM)の過去の予報値を、青森の格子点で取得して保存
  fetch_nwp_hirosaki.py              同、弘前の格子点
  nwp_features.py                    GSMの予報値を日別の特徴量(11個)に集計する共通部品(学習・評価・夜の予測で共通)
  nwp_features_eval.py               GSMの予報値を足すと、青森の翌日の天気の予測が良くなるかの評価
  hirosaki_weather_temp.py           弘前の翌日のルール天気と最高・最低気温の評価
  check_nwp_archive.py               夜に取得した予報値と、後日のアーカイブの突き合わせ
  score_predictions.py               記録した予測を、後日の実際の天気と照合
  check_realtime_features.py         10分集計の特徴量と気象庁の日別値のずれを確認
  sensitivity_realtime_bias.py       特徴量のずれが予測クラスに与える影響を測定
data/
  raw/                               取得した生データ(月別CSV、結合済みの *_daily.csv)
  processed/dataset.csv              前処理済みデータセット
```

## 使い方

Python 3.13 で動作を確認しています。パッケージのバージョンは `requirements.txt` に固定しています。

```bash
# 仮想環境と requirements.txt は、ひとつ上のフォルダにあります
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r ../requirements.txt

python code/download_jma_data.py      # データ取得。保存済みの月は再取得せず、直近2か月だけ取り直します(初回は待機を入れるので数分かかります。全期間を取り直すには --refresh)
python code/preprocess.py             # 前処理
python code/train_model.py            # ベースライン
PYTHONPATH=code python code/compare_models.py
```

チューニング系のスクリプトは `compare_models.py` の関数を読み込むため、`PYTHONPATH=code` を付けてこのフォルダ(weather-prediction)から実行します。
`untested_items.py` が使う LightGBM・XGBoost は、macOSでは OpenMP のライブラリ(libomp)が必要です。Homebrew で `libomp` を入れるのが一般的な方法です。
一部のスクリプトは `research/` 以下に結果CSVを書き出します(`research/` は、ひとつ上のフォルダの `research/` へのリンクです。このリポジトリには `research/gb_optuna_trials.csv` のみ含めています)。

## 数値予報(GSM)の利用

夜の予測には、気象庁の全球モデル(GSM)の「約48時間前に出た翌日の予報値」を特徴量として使っています。取得先は Open-Meteo の Previous Runs API(無料版は非商用が条件、データは CC BY 4.0)です。

- 出典: Open-Meteo(気象庁GSMの予報値を加工して作成)
- 弘前の「天気」は観測された天気ではなく、観測値(降水量・降雪・日照)から決めたルールの天気です。弘前のアメダスは天気概況を観測していません。
- 予測は公開せず、自分専用で使っています。

## データ

- 期間: 2021-08-01 〜 2026-08-31(1852行、欠損日なし)
- 地点: 弘前(アメダス)と青森(気象台)。弘前は天気概況を観測していないため、目的変数(天気)は青森の天気概況から作っています。
- 出典: 気象庁ホームページ「過去の気象データ・ダウンロード」
- 特徴量: 弘前の当日の観測値、青森の気圧、当日の天気(one-hot)、季節性(日付のsin/cos)など計24個
- 目的変数: 翌日の天気(晴 / 曇 / 雨 / 雪)

## 評価方法

- 時系列のため、シャッフルせず、直近365日を評価用に分けています。
- チューニングは学習データ内で `TimeSeriesSplit(5)` を使い、評価用データは最終評価にだけ使っています。
- 指標は、クラスの偏りを考えて Macro F1 を中心にしています。

## 結果の要約

| モデル | 評価Accuracy | 評価Macro F1 | 過学習ギャップ(Acc) |
|---|---|---|---|
| 前日と同じ天気(参考) | 0.396 | 0.405 | - |
| ロジスティック回帰 | 0.486 | 0.494 | 0.010 |
| 勾配ブースティング(Optuna, seed=42) | 0.503 | 0.514 | 0.209 |

- 勾配ブースティングは、探索の設定や乱数seedを変えると結果が揺れます。
- ブートストラップでの Macro F1 の差(+0.020)は、95%信頼区間が0をまたいでいました。
- 探索空間をOptunaと揃えると、GridSearchCVの全探索(2160通り)もCV Macro F1はほぼ同じ水準(0.497と0.498)でした。
- 評価データは約1年分なので、差が小さい部分は誤差の範囲と考えています。過学習ギャップと雨クラスのF1も踏まえ、ロジスティック回帰を採用しています。

## 注意

- 気象庁のサイトへのアクセスは、待機時間(1.5秒)を入れて負荷をかけないようにしています。取得したデータの利用は、気象庁の利用規約に従ってください。
