"""
本番条件での検証: 同じ日について
  (1) 気象庁の日別値から作った特徴量（学習時と同じ作り方）
  (2) 10分データから集計した特徴量（本番と同じ作り方）
を並べ、特徴量ごとのずれ（平均絶対誤差 MAE・相関・平均の差）と、予測クラスの一致日数を出す。

10分データは「23:30 打ち切り（夜の実行を再現）」と「24:00 まで（1日全部）」の2通りで集計する。
10分JSONの保存期間（約9〜10日）のため、比較できる日数は限られる。

出力: research/realtime_feature_check.csv（特徴量ごとの集計）
      research/realtime_feature_check_daily.csv（日ごとの値と予測）
実行: プロジェクト直下で `PYTHONPATH=code python code/check_realtime_features.py --start 2026-09-25 --end 2026-10-02`
"""
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fetch_realtime import FEATURES, build_features, latest_time, setup_logging
from jma_daily import load_recent

ROOT = Path(__file__).resolve().parent.parent


def daily_features(start: date, end: date) -> pd.DataFrame:
    """気象庁の日別値から preprocess.py と同じ定義で18特徴量を作る。"""
    hiro = load_recent("hirosaki", start - timedelta(days=1), end)
    aomo = load_recent("aomori", start - timedelta(days=1), end)
    f = hiro.drop(columns=["wind_max_dir", "gust_max_dir", "wind_mode_dir"]).add_prefix("hiro_")
    f["aomori_pressure_sea"] = aomo["pressure_sea"]
    idx = pd.date_range(start - timedelta(days=1), end)
    f = f.reindex(idx)
    f["aomori_pressure_diff"] = f["aomori_pressure_sea"].diff()
    f["hiro_temp_diff"] = f["hiro_temp_mean"].diff()
    f["hiro_humidity_diff"] = f["hiro_humidity_mean"].diff()
    doy = f.index.dayofyear
    f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return f.loc[pd.Timestamp(start):, FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--cutoff", default="23:30", help="夜の実行を再現する打ち切り時刻")
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    hh, mm = map(int, a.cutoff.split(":"))
    setup_logging()
    latest = latest_time()

    jma = daily_features(start, end)
    rt_cut, rt_full, last = {}, {}, {}
    for d in pd.date_range(start, end).date:
        cut = datetime(d.year, d.month, d.day, hh, mm)
        rt_cut[d], last[d], _ = build_features(d, cutoff=cut, latest=latest)
        rt_full[d], _, _ = build_features(d, cutoff=None, latest=latest)
    rt_cut = pd.DataFrame(rt_cut).T[FEATURES]
    rt_full = pd.DataFrame(rt_full).T[FEATURES]
    rt_cut.index = rt_full.index = pd.to_datetime(rt_cut.index)

    rows = []
    for name, rt in [("cutoff_" + a.cutoff, rt_cut), ("full_day", rt_full)]:
        for c in FEATURES:
            x, y = jma[c], rt[c]
            ok = x.notna() & y.notna()
            corr = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
            rows.append(dict(variant=name, feature=c, n_days=int(ok.sum()),
                             mae=(y[ok] - x[ok]).abs().mean(), bias_rt_minus_jma=(y[ok] - x[ok]).mean(),
                             max_abs_err=(y[ok] - x[ok]).abs().max(), corr=corr,
                             jma_std=x[ok].std(), n_missing_rt=int(y.isna().sum()), n_missing_jma=int(x.isna().sum())))
    res = pd.DataFrame(rows)

    # 予測の一致: 同じモデルに気象庁日別値の特徴量と10分集計の特徴量を入れて比べる
    b = joblib.load(ROOT / "models" / "model.joblib")
    pipe, feats, classes = b["pipeline"], b["features"], b["classes"]
    daily = pd.DataFrame(index=jma.index)
    for name, X in [("jma", jma), ("rt_cut", rt_cut), ("rt_full", rt_full)]:
        proba = pipe.predict_proba(X[feats])
        daily[f"pred_{name}"] = np.array(classes)[proba.argmax(1)]
        for i, k in enumerate(classes):
            daily[f"p_{k}_{name}"] = proba[:, i]
    daily["last_obs_cut"] = pd.Series(last).values
    for c in FEATURES:
        daily[f"{c}__jma"], daily[f"{c}__rt_cut"] = jma[c], rt_cut[c]
    n = len(daily)
    agree_cut = int((daily.pred_jma == daily.pred_rt_cut).sum())
    agree_full = int((daily.pred_jma == daily.pred_rt_full).sum())
    l1 = np.mean([np.abs(daily[[f"p_{k}_jma" for k in classes]].values
                         - daily[[f"p_{k}_rt_cut" for k in classes]].values).sum(1)])
    res = pd.concat([res, pd.DataFrame([
        dict(variant="cutoff_" + a.cutoff, feature="__pred_agree_days", n_days=n, mae=agree_cut),
        dict(variant="full_day", feature="__pred_agree_days", n_days=n, mae=agree_full),
        dict(variant="cutoff_" + a.cutoff, feature="__proba_L1_mean", n_days=n, mae=l1),
    ])])
    out = ROOT / "research"
    res.to_csv(out / "realtime_feature_check.csv", index=False)
    daily.to_csv(out / "realtime_feature_check_daily.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(daily[["pred_jma", "pred_rt_cut", "pred_rt_full"] + [f"p_{k}_jma" for k in classes]
                + [f"p_{k}_rt_cut" for k in classes]].round(2).to_string())
    print(f"予測一致: 打ち切り版 {agree_cut}/{n}日, 1日全部版 {agree_full}/{n}日, 確率のL1差平均 {l1:.3f}")


if __name__ == "__main__":
    main()
