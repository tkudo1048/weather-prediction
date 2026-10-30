"""
数値予報(気象庁GSM)の「前日・前々日に出た予報値」を特徴量に足すと、
翌日の天気(晴/曇/雨/雪)の予測が良くなるかを検証する。

問い: 夜に作れる18特徴量(fetch_realtime.FEATURES)のロジスティック回帰(class_weight=balanced)に、
      GSMの翌日予報値を足すと、評価Macro F1・雨/雪/曇F1は良くなるか。

- 予報データ: data/raw/nwp/ の保存済みJSON(code/fetch_nwp.py で取得。Open-Meteo Previous Runs API、
  CC BY 4.0)。地点は青森(40.82, 140.77)、モデルは jma_gsm。
- previous_day1: 対象時刻の約24時間前に出た予報(夜23時に公開済みかは未確認 → 楽観側の版)
  previous_day2: 約48時間前に出た予報(夜の時点で確実に使える → 保守的な版)
- 日別集計: 予報の対象日(=翌日)の 06〜18時 JST(code/nwp_features.py の daily_features。本番と共通)。
    瞬間値(気温・雲量・湿度・気圧・風速)は 06:00〜18:00 の13時刻。
    積算値(降水量・降雪量。Open-Meteo の毎時値は「直前1時間の合計」)は 07:00〜18:00 の12時刻。
    有効な時刻が足りない日(瞬間値10未満 / 積算値9未満)は NaN にし、学習データの中央値で補完する。
- 結合: 行の日付 = 観測日 D、予報の対象 = D+1(目的変数と同じ日)。
- 分割: 既存の load_data / time_series_split(学習1488行・評価364行)。シャッフルなし。
  学習内 TimeSeriesSplit(5) のCVスコアも参考に出す(LRのCはデフォルトのまま。探索はしない)。
- 実行: プロジェクト直下で `PYTHONPATH=code .venv/bin/python code/nwp_features_eval.py`
- 日付合わせの確認: 予報の対象日を当日D / D+2 にずらした版も評価する(LR_shift_check)。
- 出力: research/nwp_features_eval.csv(各モデルの指標とブートストラップ)、
        research/nwp_features_join_check.csv(結合確認用の数行)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from compare_models import load_data, time_series_split, build_models, evaluate, TARGET_COL, CLASSES
from fetch_realtime import FEATURES
from seed_robustness import boot_diff
from train_model import persistence_baseline
from fetch_nwp import out_path
from nwp_features import daily_features, hourly_frame  # 集計は本番(train_final / predict_tonight)と共通

LEADS = {"gsm1": "previous_day1", "gsm2": "previous_day2"}


def load_hourly():
    d = json.loads(out_path().read_text(encoding="utf-8"))
    return hourly_frame(d)


def build_dataset(day_offset=1, h=None):
    """day_offset=1 が本番の結合(観測日Dの行に、対象日D+1の予報)。
    0(当日Dの予報)と2(D+2の予報)は、日付合わせの確認用。"""
    df = load_data()
    h = load_hourly() if h is None else h
    feats = {}
    for prefix, lead in LEADS.items():
        dly = daily_features(h, prefix, lead)
        dly.index = dly.index - pd.Timedelta(days=day_offset)  # 対象日 D+1 の予報を、観測日 D の行に置く
        dly.index.name = "date"
        feats[prefix] = list(dly.columns)
        df = df.merge(dly.reset_index(), on="date", how="left")
    return df, feats


def row_from(model, fs, y_tr, p_tr, y_ev, p_ev, **kw):
    tra, trf, _ = evaluate(y_tr, p_tr)
    eva, evf, cf = evaluate(y_ev, p_ev)
    r = dict(model=model, feature_set=fs, eval_acc=eva, eval_f1=evf, train_acc=tra, train_f1=trf,
             gap_acc=tra - eva, gap_f1=trf - evf)
    r.update({f"f1_{c}": cf.get(c, 0.0) for c in CLASSES})
    r.update(kw)
    return r


def main():
    df, gsm = build_dataset()
    train_df, eval_df = time_series_split(df, eval_days=365)
    y_tr, y_ev = train_df[TARGET_COL], eval_df[TARGET_COL]
    print(f"学習 {len(train_df)}行 {train_df.date.min().date()}〜{train_df.date.max().date()} / "
          f"評価 {len(eval_df)}行 {eval_df.date.min().date()}〜{eval_df.date.max().date()}")
    for p, cols in gsm.items():
        na_tr = train_df[cols].isna().any(axis=1).sum()
        na_ev = eval_df[cols].isna().any(axis=1).sum()
        print(f"{p}: {len(cols)}特徴量 / 欠損を含む行 学習{na_tr} 評価{na_ev}")

    # 結合の確認用(観測日D、翌日の天気、翌日の予報の集計値)
    chk_cols = ["date", "target", "target_text", "hiro_precip_total", "hiro_temp_mean",
                "gsm1_precip_sum", "gsm2_precip_sum", "gsm1_cloud_mean", "gsm2_cloud_mean",
                "gsm1_temp_mean", "gsm1_snow_sum"]
    pick = pd.to_datetime(["2021-08-01", "2022-01-15", "2024-12-31", "2025-07-15", "2026-02-01", "2026-08-30"])
    chk = df[df.date.isin(pick)][chk_cols]
    chk.to_csv("research/nwp_features_join_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(chk.round(2).to_string(index=False))

    sets = {
        "base18": FEATURES,
        "base18+gsm1": FEATURES + gsm["gsm1"],
        "base18+gsm2": FEATURES + gsm["gsm2"],
        "gsm1_only": gsm["gsm1"],
        "gsm2_only": gsm["gsm2"],
    }

    rows, preds = [], {}
    maj = y_tr.mode()[0]
    rows.append(row_from("majority", f"const={maj}", y_tr, np.full(len(y_tr), maj), y_ev, np.full(len(y_ev), maj)))
    mm = train_df.groupby(train_df.date.dt.month)[TARGET_COL].agg(lambda s: s.mode()[0])
    rows.append(row_from("monthly_mode", "month", y_tr, train_df.date.dt.month.map(mm).to_numpy(),
                         y_ev, eval_df.date.dt.month.map(mm).to_numpy()))
    rows.append(row_from("persistence", "today_*", y_tr, persistence_baseline(train_df).to_numpy(),
                         y_ev, persistence_baseline(eval_df).to_numpy()))

    cv = TimeSeriesSplit(n_splits=5)
    for name, cols in sets.items():
        lr = build_models()["ロジスティック回帰"]
        cvs = cross_val_score(lr, train_df[cols], y_tr, cv=cv, scoring="f1_macro")
        lr.fit(train_df[cols], y_tr)
        p = lr.predict(eval_df[cols])
        preds[name] = p
        rows.append(row_from("LR", name, y_tr, lr.predict(train_df[cols]), y_ev, p,
                             n_features=len(cols), cv_f1_mean=cvs.mean(), cv_f1_std=cvs.std()))

    y = y_ev.to_numpy()
    rng = np.random.default_rng(42)
    pairs = [(n, "base18") for n in sets if n != "base18"] + [("base18+gsm1", "base18+gsm2")]
    for b, a in pairs:
        point = f1_score(y, preds[b], average="macro") - f1_score(y, preds[a], average="macro")
        for meth, blk in [("iid_day", 1), ("block_7d", 7)]:
            d = boot_diff(y, preds[a], preds[b], rng, blk)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(model="LR_bootstrap", feature_set=f"{b}_minus_{a}", method=meth,
                             diff_point=point, diff_mean=d.mean(), ci_low=lo, ci_high=hi, p_ge0=(d >= 0).mean()))

    # 日付合わせの確認: 予報の対象日を1日ずらすと、精度は下がるはず(下がらなければ結合かデータを疑う)
    h = load_hourly()
    for off in [0, 2]:
        dfo, _ = build_dataset(off, h)
        tr_o, ev_o = time_series_split(dfo, eval_days=365)
        for name in ["gsm1_only", "gsm2_only"]:
            cols = sets[name]
            lr = build_models()["ロジスティック回帰"]
            lr.fit(tr_o[cols], tr_o[TARGET_COL])
            rows.append(row_from("LR_shift_check", f"{name}_offset{off}", tr_o[TARGET_COL], lr.predict(tr_o[cols]),
                                 ev_o[TARGET_COL], lr.predict(ev_o[cols]), n_features=len(cols)))

    res = pd.DataFrame(rows)
    res.to_csv("research/nwp_features_eval.csv", index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
