"""
翌日の天気（晴/曇/雨/雪）予測モデルの手法比較スクリプト。

同じ時系列分割（学習1,488行: 2021-08-01〜2025-08-30 / 評価364行: 2025-08-31〜2026-08-30、
シャッフルなし）で、複数の分類手法を学習・評価し、以下を比較する。

- 多項ロジスティック回帰（LogisticRegression） … 既存ベースライン（train_model.py と同一設定）
- 決定木1本（DecisionTreeClassifier）
- ランダムフォレスト（RandomForestClassifier）
- 勾配ブースティング（GradientBoostingClassifier, sklearn標準実装）
  ※ LightGBM/XGBoostは実行環境に未インストールのため、sklearn標準の
    GradientBoostingClassifierで代用した（指示に基づく代替）。
- k近傍法（KNeighborsClassifier） … 参考候補として追加

各モデルについて、学習データ・評価データ双方のAccuracy/Macro F1、
クラス別F1、および学習-評価スコアの差（過学習の指標）を算出する。

このスクリプトはモデルの学習・評価のみを行う。データの探索・可視化は行わない。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
CLASSES = ["晴", "曇", "雨", "雪"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def time_series_split(df, eval_days=365):
    cutoff_date = df["date"].max() - pd.Timedelta(days=eval_days)
    train_df = df[df["date"] <= cutoff_date].copy()
    eval_df = df[df["date"] > cutoff_date].copy()
    return train_df, eval_df


def build_models():
    """比較対象の各モデルをPipeline（欠損値補完+標準化+分類器）として定義する。

    木系モデル（決定木・ランダムフォレスト・勾配ブースティング）にとって
    標準化は本来不要だが、パイプラインを揃えて前処理条件を統一するために
    全モデルに同じ標準化を適用する（結果には影響しない）。
    """
    models = {}

    models["ロジスティック回帰"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        )),
    ])

    models["決定木1本"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20,
            class_weight="balanced", random_state=42,
        )),
    ])

    models["ランダムフォレスト"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])

    models["勾配ブースティング(sklearn)"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])

    models["kNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    ])

    return models


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, labels=CLASSES, digits=3, zero_division=0, output_dict=True
    )
    class_f1 = {c: report[c]["f1-score"] for c in CLASSES}
    return acc, f1_macro, class_f1


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)

    print(f"学習期間: {len(train_df)}行 ({train_df['date'].min().date()} 〜 {train_df['date'].max().date()})")
    print(f"評価期間: {len(eval_df)}行 ({eval_df['date'].min().date()} 〜 {eval_df['date'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_eval, y_eval = eval_df[FEATURE_COLS], eval_df[TARGET_COL]

    models = build_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        eval_pred = model.predict(X_eval)

        train_acc, train_f1, _ = evaluate(y_train, train_pred)
        eval_acc, eval_f1, eval_class_f1 = evaluate(y_eval, eval_pred)

        gap_acc = train_acc - eval_acc
        gap_f1 = train_f1 - eval_f1

        print(f"\n===== {name} =====")
        print(f"学習 Accuracy={train_acc:.3f}, Macro F1={train_f1:.3f}")
        print(f"評価 Accuracy={eval_acc:.3f}, Macro F1={eval_f1:.3f}")
        print(f"過学習ギャップ（学習-評価）: Accuracy差={gap_acc:.3f}, MacroF1差={gap_f1:.3f}")
        print("評価データ クラス別F1:", {c: round(v, 3) for c, v in eval_class_f1.items()})

        results.append({
            "model": name,
            "train_acc": train_acc, "train_f1": train_f1,
            "eval_acc": eval_acc, "eval_f1": eval_f1,
            "gap_acc": gap_acc, "gap_f1": gap_f1,
            **{f"f1_{c}": v for c, v in eval_class_f1.items()},
        })

    result_df = pd.DataFrame(results)
    print("\n===== 比較サマリ（評価データ） =====")
    print(result_df[["model", "eval_acc", "eval_f1", "gap_acc", "gap_f1"]].to_string(index=False))

    result_df.to_csv("research/model_comparison_results.csv", index=False)
    print("\n詳細結果を research/model_comparison_results.csv に保存しました。")


if __name__ == "__main__":
    main()
