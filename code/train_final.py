"""
本番用モデルの学習: 全期間(学習+評価の全1852行)を使ってロジスティック回帰(class_weight=balanced)を学習し、保存する。

- 既定(GSMあり): 夜に作れる18特徴量 + 気象庁GSMの翌日予報(previous_day2、06〜18時の集計)11特徴量 = 29特徴量。
  評価時(nwp_features_eval.py の base18+gsm2、評価364行で Macro F1 0.715)と同じ手法・前処理・集計。
  GSMの集計は code/nwp_features.py の daily_features(夜の予測 predict_tonight.py と共通)。
  保存先: models/model.joblib。上書きする前に、既存の model.joblib が18特徴量モデルで、
  models/model_18feat.joblib がまだ無ければ、そこへコピーして残す(GSM取得失敗時の予備)。
- --no-nwp: 18特徴量のみ(評価364行で Macro F1 0.484)。保存先: models/model_18feat.joblib。
- 全期間で学習し直すので、このモデル自体の評価値は存在しない(評価は上記の値と、運用後の score_predictions.py で見る)。
- 保存内容: pipeline, 特徴量名の順序, クラス順(predict_proba の列順), 学習期間, nwp_version など。
- 実行: プロジェクト直下で `.venv/bin/python code/train_final.py`(18特徴量版は `--no-nwp`)
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from compare_models import load_data, build_models, TARGET_COL
from fetch_realtime import FEATURES
from nwp_features import PROD_PREFIX, PROD_LEAD, NWP_VERSION, daily_features, prod_feature_names

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "model.joblib"
MODEL_18 = ROOT / "models" / "model_18feat.joblib"


def build_train_df(with_nwp: bool):
    df = load_data()
    if not with_nwp:
        return df, list(FEATURES)
    from nwp_features_eval import load_hourly  # 保存済みアーカイブ(data/raw/nwp/)
    dly = daily_features(load_hourly(), PROD_PREFIX, PROD_LEAD)
    dly.index = dly.index - pd.Timedelta(days=1)  # 対象日 D+1 の予報を、観測日 D の行に置く
    dly.index.name = "date"
    df = df.merge(dly.reset_index(), on="date", how="left")
    gsm = prod_feature_names()
    assert list(dly.columns) == gsm
    return df, list(FEATURES) + gsm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-nwp", action="store_true", help="18特徴量のみで学習し models/model_18feat.joblib に保存")
    a = ap.parse_args()
    with_nwp = not a.no_nwp
    df, feats = build_train_df(with_nwp)
    lr = build_models()["ロジスティック回帰"]  # SimpleImputer(中央値)+StandardScaler+LogReg(balanced)
    lr.fit(df[feats], df[TARGET_COL])
    bundle = {
        "pipeline": lr,
        "features": feats,
        "classes": list(lr.classes_),
        "train_period": (str(df.date.min().date()), str(df.date.max().date())),
        "n_rows": len(df),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "nwp_version": NWP_VERSION if with_nwp else None,
        "model_name": "lr_18feat+gsm2" if with_nwp else "lr_18feat",
    }
    if with_nwp:
        bundle["nwp_source"] = "open-meteo previous-runs-api jma_gsm (40.82,140.77), 06-18 JST of D+1"
        bundle["n_rows_nwp_missing"] = int(df[feats[len(FEATURES):]].isna().any(axis=1).sum())
    out = MODEL if with_nwp else MODEL_18
    out.parent.mkdir(exist_ok=True)
    if with_nwp and MODEL.exists() and not MODEL_18.exists():
        old = joblib.load(MODEL)
        if old.get("nwp_version") is None and len(old["features"]) == len(FEATURES):
            shutil.copy2(MODEL, MODEL_18)
            print(f"旧モデル(18特徴量)を退避: {MODEL_18}")
    joblib.dump(bundle, out)
    print(f"学習 {len(df)}行 {bundle['train_period'][0]}〜{bundle['train_period'][1]} / 特徴量 {len(feats)}個"
          f" / nwp_version={bundle['nwp_version']}")
    if with_nwp:
        print(f"GSMに欠損を含む行: {bundle['n_rows_nwp_missing']}(学習データの中央値で補完)")
    print(f"クラス順: {bundle['classes']}")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
