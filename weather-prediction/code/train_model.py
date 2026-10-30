"""
翌日の天気（晴/曇/雨/雪）を予測するベースラインモデルの学習・評価スクリプト。

- 入力: data/processed/dataset.csv (code/preprocess.py で作成済み)
- 分割: 時系列順。シャッフルしない。直近1年程度を評価用に残す。
- 比較対象: persistence baseline（今日の天気がそのまま明日も続くという予測）
- モデル: 多項ロジスティック回帰（Softmax回帰）
- 出力: 標準出力に評価指標を表示（research/配下のMarkdownに転記して使う）

このスクリプトはモデルの学習・評価のみを行う。データの探索・可視化は行わない。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "data/processed/dataset.csv"

FEATURE_COLS = [
    "hiro_precip_total", "hiro_precip_max_1h", "hiro_precip_max_10m",
    "hiro_temp_mean", "hiro_temp_max", "hiro_temp_min",
    "hiro_humidity_mean", "hiro_humidity_min",
    "hiro_wind_mean", "hiro_wind_max", "hiro_gust_max",
    "hiro_sunshine", "hiro_snowfall", "hiro_snow_depth",
    "aomori_pressure_sea", "aomori_pressure_diff",
    "hiro_temp_diff", "hiro_humidity_diff",
    "today_晴", "today_曇", "today_雨", "today_雪",
    "season_sin", "season_cos",
]

TARGET_COL = "target"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def time_series_split(df, eval_days=365):
    """
    時系列順を保ったまま学習用/評価用に分割する。
    直近 eval_days 日分を評価用にする（シャッフルはしない）。
    """
    cutoff_date = df["date"].max() - pd.Timedelta(days=eval_days)
    train_df = df[df["date"] <= cutoff_date].copy()
    eval_df = df[df["date"] > cutoff_date].copy()
    return train_df, eval_df


def persistence_baseline(eval_df):
    """
    「今日と同じ天気が明日も続く」という単純な予測。
    today_晴/today_曇/today_雨/today_雪のワンホットから「今日の天気」を復元して
    そのまま翌日の予測値として使う。
    """
    today_cols = {"today_晴": "晴", "today_曇": "曇", "today_雨": "雨", "today_雪": "雪"}
    onehot = eval_df[list(today_cols.keys())].fillna(0)
    pred = onehot.idxmax(axis=1).map(today_cols)
    # today_* が全て0（=欠測）の行はpersistence予測を作れないため、
    # 最頻クラス（このデータでは「曇」）で埋める。
    valid_mask = onehot.sum(axis=1) > 0
    fallback_class = eval_df[TARGET_COL].mode()[0]
    pred = pred.where(valid_mask, fallback_class)
    return pred


def build_model():
    """
    多項ロジスティック回帰（Softmax回帰）をベースラインモデルとして採用。

    選定理由（note記事向けメモ）:
    - 学習データが約1,850行・特徴量24個と少なめなので、決定木の集団学習
      （ランダムフォレストや勾配ブースティング）のような複雑なモデルは
      過学習しやすく、初心者向けの検証用ベースラインとしては不向き。
    - ロジスティック回帰は「各特徴量が天気カテゴリの確率をどちらに
      押し上げ／押し下げるか」を係数として説明しやすく、
      初心者向けnote記事で仕組みを噛み砕いて紹介しやすい。
    - 決定木1本、k近傍法（kNN）も候補にしたが、
      決定木1本は分割の閾値に依存し季節性のような滑らかな効果を表現しづらく、
      kNNは特徴量のスケールに敏感で欠損値の扱いも煩雑になるため見送った。
    - このモデルはあくまで「まず作る一番シンプルなベースライン」であり、
      これを上回るかどうかで次の改善（決定木系モデルなど）を検討する基準にする。
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    return pipeline


def evaluate(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n===== {label} =====")
    print(f"正解率 (Accuracy): {acc:.3f}")
    print(f"マクロF1 (Macro F1): {f1_macro:.3f}")
    print("\nクラスごとの precision / recall / F1:")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("混同行列 (行=正解, 列=予測, 順序は below labels):")
    labels = sorted(y_true.unique())
    print("labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))
    return {"accuracy": acc, "macro_f1": f1_macro}


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)

    print(f"データ全体: {len(df)}行 ({df['date'].min().date()} 〜 {df['date'].max().date()})")
    print(f"学習期間: {len(train_df)}行 ({train_df['date'].min().date()} 〜 {train_df['date'].max().date()})")
    print(f"評価期間: {len(eval_df)}行 ({eval_df['date'].min().date()} 〜 {eval_df['date'].max().date()})")

    # --- persistence baseline ---
    pers_pred = persistence_baseline(eval_df)
    pers_metrics = evaluate(eval_df[TARGET_COL], pers_pred, "Persistence baseline (今日=明日と予測)")

    # --- ロジスティック回帰モデル ---
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_eval = eval_df[FEATURE_COLS]
    y_eval = eval_df[TARGET_COL]

    model = build_model()
    model.fit(X_train, y_train)
    model_pred = model.predict(X_eval)
    model_metrics = evaluate(y_eval, model_pred, "多項ロジスティック回帰モデル")

    print("\n===== まとめ =====")
    print(f"Persistence baseline : Accuracy={pers_metrics['accuracy']:.3f}, Macro F1={pers_metrics['macro_f1']:.3f}")
    print(f"ロジスティック回帰     : Accuracy={model_metrics['accuracy']:.3f}, Macro F1={model_metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
