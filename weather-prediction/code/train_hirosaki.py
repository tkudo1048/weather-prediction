"""
弘前の翌日の「最高気温」「最低気温」「ルール天気(晴/曇/雨/雪)」の本番用モデルを学習し、models/ に保存する。

手法・特徴量・前処理は、検証(code/hirosaki_weather_temp.py、research/model-comparison.md の弘前の節)と同じ。
- 特徴量: 夜に作れる18個(fetch_realtime.FEATURES)+ GSM day2(弘前格子 (40.5, 140.5))の11個(hgsm2_*)。
  GSMの集計は code/nwp_features.py の daily_features(夜の予測と共通)。予報の対象日 D+1 を観測日 D の行に置く。
- 最高・最低気温: Ridge回帰(係数が大きくなりすぎないよう抑える線形回帰)。中央値補完+標準化+Ridge。
  alpha は 13通り(0.001〜1000)から、学習データ内の TimeSeriesSplit(5) のMAEで選ぶ(検証と同じ手順)。
- ルール天気: ロジスティック回帰(class_weight=balanced。クラスの件数の偏りを重みで補正)。Cはデフォルト。
  ラベル: 降雪量>=2cm → 雪、降水量>=5mm → 雨、日照時間 ÷ その月の日照95%点 >= 0.6 → 晴、それ以外 → 曇。
  月別の日照95%点は、学習に使う全期間(弘前の観測日 = 最初の観測日D 〜 最後の目的変数の日 D+1)で計算し、
  bundle に保存する(答え合わせ code/score_hirosaki.py でも同じ値を使う)。
- 学習行: 既存の load_data の全1852行のうち、目的変数(弘前のD+1の値)がある行。
- 全期間で学習するので、このモデル自体の評価値はない。目安は検証の評価値(1年分・1地点)。
  --check: 検証と同じ分割(学習1,488行・評価364行)で同じコードを学習・評価し、検証の数字と一致するかを確かめる
  (保存はしない)。
- 保存: models/hirosaki_temp_max.joblib、hirosaki_temp_min.joblib、hirosaki_weather.joblib
  (既存の model.joblib / model_18feat.joblib には触らない)。
- 実行: プロジェクト直下で `.venv/bin/python code/train_hirosaki.py`(確認だけなら `--check`)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from compare_models import load_data, time_series_split, build_models, evaluate, CLASSES
from fetch_realtime import FEATURES
from fetch_nwp_hirosaki import out_path as hirosaki_nwp_path
from hirosaki_weather_temp import (ALPHAS, PRECIP_MM, SNOW_CM, SUN_RATIO, ridge, rule_label,
                                   sun_q95_by_month)
from nwp_features import (HIRO_GRID, HIRO_PREFIX, NWP_VERSION, PROD_LEAD, daily_features, hourly_frame,
                          prod_feature_names)
from preprocess import HIROSAKI_COLS, load

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = {"max": MODELS / "hirosaki_temp_max.joblib", "min": MODELS / "hirosaki_temp_min.joblib",
       "weather": MODELS / "hirosaki_weather.joblib"}
NWP_SOURCE = (f"open-meteo previous-runs-api jma_gsm, request (40.60,140.46) -> grid {HIRO_GRID}, "
              "previous_day2, 06-18 JST of D+1")


def gsm_hirosaki():
    d = json.loads(hirosaki_nwp_path().read_text(encoding="utf-8"))
    grid = (float(d["latitude"]), float(d["longitude"]))
    assert grid == HIRO_GRID, f"保存済みデータの格子点 {grid} が想定 {HIRO_GRID} と違う"
    dly = daily_features(hourly_frame(d), HIRO_PREFIX, PROD_LEAD)
    assert list(dly.columns) == prod_feature_names(HIRO_PREFIX)
    return dly


def build_df(q95_last_day=None):
    """観測日Dの行に、18特徴量・GSM(D+1、弘前格子)・弘前のD+1の目的変数を付ける。
    q95_last_day: 日照95%点を計算する最終日(None なら学習に使う全期間 = 最後の目的変数の日)。"""
    df = load_data()
    dly = gsm_hirosaki()
    dly.index = dly.index - pd.Timedelta(days=1)  # 対象日 D+1 の予報を、観測日 D の行に置く
    dly.index.name = "date"
    df = df.merge(dly.reset_index(), on="date", how="left")

    hiro = load("hirosaki", HIROSAKI_COLS)
    first, last = df.date.min(), df.date.max() + pd.Timedelta(days=1)
    hiro_period = hiro[(hiro.index >= first) & (hiro.index <= last)]
    q95 = sun_q95_by_month(hiro_period, q95_last_day if q95_last_day is not None else last)
    lab = rule_label(hiro, q95)
    nxt = pd.DataFrame({"target_h": lab, "target_tmax": hiro["temp_max"], "target_tmin": hiro["temp_min"]})
    nxt.index = nxt.index - pd.Timedelta(days=1)  # D+1 の値を D の行へ
    nxt.index.name = "date"
    df = df.merge(nxt.reset_index(), on="date", how="left")
    return df, FEATURES + prod_feature_names(HIRO_PREFIX), q95


def fit_temp(tr, feats, tcol):
    tr = tr.dropna(subset=[tcol])
    gs = ridge().fit(tr[feats], tr[tcol])
    return gs, tr


def fit_weather(tr, feats):
    tr = tr.dropna(subset=["target_h"])
    lr = build_models()["ロジスティック回帰"]  # SimpleImputer(中央値)+StandardScaler+LogReg(balanced)
    lr.fit(tr[feats], tr["target_h"])
    return lr, tr


def check():
    """検証と同じ分割で学習・評価し、検証の数字(research/model-comparison.md)と比べる。保存しない。"""
    base = load_data()
    tr0, _ = time_series_split(base, eval_days=365)
    df, feats, _ = build_df(q95_last_day=tr0.date.max())  # 検証と同じく、95%点は学習期間の日だけ
    tr, ev = time_series_split(df, eval_days=365)
    for kind, tcol in (("max", "target_tmax"), ("min", "target_tmin")):
        gs, trk = fit_temp(tr, feats, tcol)
        evk = ev.dropna(subset=[tcol])
        mae = np.abs(gs.predict(evk[feats]) - evk[tcol]).mean()
        print(f"[check] 最{'高' if kind == 'max' else '低'}気温: 学習 {len(trk)}行 / 評価 {len(evk)}行 / "
              f"alpha {gs.best_params_['reg__alpha']:g} / 評価MAE {mae:.3f}")
    lr, trw = fit_weather(tr, feats)
    evw = ev.dropna(subset=["target_h"])
    acc, f1, _ = evaluate(evw["target_h"], lr.predict(evw[feats]))
    print(f"[check] ルール天気: 学習 {len(trw)}行 / 評価 {len(evw)}行 / Acc {acc:.3f} / Macro F1 {f1:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="検証と同じ分割で再現確認だけ行う(保存しない)")
    a = ap.parse_args()
    if a.check:
        check()
        return

    df, feats, q95 = build_df()
    now = datetime.now().isoformat(timespec="seconds")
    common = {"features": feats, "nwp_version": NWP_VERSION, "nwp_prefix": HIRO_PREFIX, "grid": HIRO_GRID,
              "nwp_source": NWP_SOURCE, "trained_at": now}
    MODELS.mkdir(exist_ok=True)

    for kind, tcol in (("max", "target_tmax"), ("min", "target_tmin")):
        gs, tr = fit_temp(df, feats, tcol)
        bundle = {**common, "pipeline": gs.best_estimator_, "target": f"hirosaki_temp_{kind}",
                  "model_name": f"ridge_18feat+hgsm2_temp_{kind}",
                  "alpha": float(gs.best_params_["reg__alpha"]), "alphas": [float(x) for x in ALPHAS],
                  "cv_mae": float(-gs.best_score_), "cv": "TimeSeriesSplit(5)",
                  "train_period": (str(tr.date.min().date()), str(tr.date.max().date())),
                  "n_rows": len(tr),
                  "n_rows_nwp_missing": int(tr[feats[len(FEATURES):]].isna().any(axis=1).sum())}
        joblib.dump(bundle, OUT[kind])
        print(f"最{'高' if kind == 'max' else '低'}気温: 学習 {len(tr)}行 {bundle['train_period'][0]}〜"
              f"{bundle['train_period'][1]}(観測日) / alpha {bundle['alpha']:g} / 学習内CV MAE {bundle['cv_mae']:.3f}"
              f" / GSM欠損を含む行 {bundle['n_rows_nwp_missing']} / 保存 {OUT[kind]}")

    lr, tr = fit_weather(df, feats)
    q95_period = (str(df.date.min().date()), str((df.date.max() + pd.Timedelta(days=1)).date()))
    bundle = {**common, "pipeline": lr, "target": "hirosaki_rule_weather", "model_name": "lr_18feat+hgsm2_rule",
              "classes": list(lr.classes_), "alpha": None,
              "rule": {"snow_cm": SNOW_CM, "precip_mm": PRECIP_MM, "sun_ratio": SUN_RATIO,
                       "order": "雪 > 雨 > 晴/曇(日照割合)", "missing": "降雪・降水量の欠測は0、日照の欠測はラベルなし"},
              "sun_q95_by_month": {int(m): float(v) for m, v in q95.items()},
              "sun_q95_period": q95_period,
              "train_period": (str(tr.date.min().date()), str(tr.date.max().date())),
              "n_rows": len(tr),
              "n_rows_nwp_missing": int(tr[feats[len(FEATURES):]].isna().any(axis=1).sum()),
              "label_counts": tr["target_h"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()}
    joblib.dump(bundle, OUT["weather"])
    print(f"ルール天気: 学習 {len(tr)}行 {bundle['train_period'][0]}〜{bundle['train_period'][1]}(観測日) / "
          f"クラス順 {bundle['classes']} / ラベル件数 {bundle['label_counts']} / 保存 {OUT['weather']}")
    print(f"月別の日照95%点(時間、{q95_period[0]}〜{q95_period[1]}): "
          + ", ".join(f"{m}月 {v:.2f}" for m, v in bundle["sun_q95_by_month"].items()))
    # 学習データでの当てはまり(参考。評価値ではない)
    p = lr.predict(tr[feats])
    print(f"(参考)学習データでのルール天気 Macro F1 {f1_score(tr['target_h'], p, average='macro'):.3f}")


if __name__ == "__main__":
    main()
