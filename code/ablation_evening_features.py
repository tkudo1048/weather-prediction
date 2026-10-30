"""
「前日夜（23時台）に作れる特徴量だけ」で翌日の天気を予測した場合の精度低下を確認するアブレーション。

想定: 夜の時点では青森の天気概況（today_晴/曇/雨/雪）と降雪・積雪が取れない可能性が高い。
特徴量セット:
  (a) 全24特徴量（現行ベースライン）
  (b) today_* を除く
  (c) (b) から hiro_snowfall, hiro_snow_depth も除く
  (d) (c) から season_sin, season_cos も除く（季節性だけで稼いでいないかの確認）
基準: 多数派クラスを常に予測 / 学習データの月別最頻クラスを予測
      （today_* を使わない前提なので「前日と同じ」基準は使わない）
主モデル: ロジスティック回帰（class_weight=balanced）。参考で勾配ブースティング（Optuna最良パラメータ）。
(a) と (b)(c)(d) の Macro F1 差を日単位・7日ブロックのブートストラップ（各1000回）で区間推定。

- データ: data/processed/dataset.csv のみ。分割は直近365日を評価（シャッフルなし）。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/ablation_evening_features.py`
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from compare_models import load_data, time_series_split, build_models, evaluate, FEATURE_COLS, TARGET_COL, CLASSES
from seed_robustness import make_pipe, load_best_params_42, boot_diff

TODAY = ["today_晴", "today_曇", "today_雨", "today_雪"]
SNOW = ["hiro_snowfall", "hiro_snow_depth"]
SEASON = ["season_sin", "season_cos"]
SETS = {
    "a_all24": FEATURE_COLS,
    "b_no_today": [c for c in FEATURE_COLS if c not in TODAY],
    "c_no_today_snow": [c for c in FEATURE_COLS if c not in TODAY + SNOW],
    "d_no_today_snow_season": [c for c in FEATURE_COLS if c not in TODAY + SNOW + SEASON],
}


def row_from(model, features, y_tr, pred_tr, y_ev, pred_ev):
    tra, trf, _ = evaluate(y_tr, pred_tr)
    eva, evf, cf = evaluate(y_ev, pred_ev)
    r = dict(model=model, feature_set=features, eval_acc=eva, eval_f1=evf,
             train_acc=tra, train_f1=trf, gap_acc=tra - eva, gap_f1=trf - evf)
    r.update({f"f1_{c}": cf.get(c, 0.0) for c in CLASSES})
    return r


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    y_tr, y_ev = train_df[TARGET_COL], eval_df[TARGET_COL]
    print(f"学習 {len(train_df)}行 {train_df.date.min().date()}〜{train_df.date.max().date()} / "
          f"評価 {len(eval_df)}行 {eval_df.date.min().date()}〜{eval_df.date.max().date()}")
    rows, preds = [], {}

    # 基準1: 多数派クラス
    maj = y_tr.mode()[0]
    rows.append(row_from("majority", f"const={maj}", y_tr, np.full(len(y_tr), maj), y_ev, np.full(len(y_ev), maj)))
    # 基準2: 学習データの月別最頻クラス
    mm = train_df.groupby(train_df.date.dt.month)[TARGET_COL].agg(lambda s: s.mode()[0])
    print("月別最頻:", mm.to_dict())
    rows.append(row_from("monthly_mode", "month", y_tr, train_df.date.dt.month.map(mm).to_numpy(),
                         y_ev, eval_df.date.dt.month.map(mm).to_numpy()))

    bp = load_best_params_42()
    for name, cols in SETS.items():
        lr = build_models()["ロジスティック回帰"]
        lr.fit(train_df[cols], y_tr)
        p = lr.predict(eval_df[cols])
        preds[name] = p
        rows.append(dict(row_from("LR", name, y_tr, lr.predict(train_df[cols]), y_ev, p), n_features=len(cols)))
        if name != "d_no_today_snow_season":
            gb = make_pipe(42, **bp)
            gb.fit(train_df[cols], y_tr)
            rows.append(dict(row_from("GB_optuna_ref", name, y_tr, gb.predict(train_df[cols]), y_ev,
                                      gb.predict(eval_df[cols])), n_features=len(cols)))

    # ブートストラップ: (a) を基準に各セットとの差（セット - a）
    y = y_ev.to_numpy()
    rng = np.random.default_rng(42)
    for name in ["b_no_today", "c_no_today_snow", "d_no_today_snow_season"]:
        point = f1_score(y, preds[name], average="macro") - f1_score(y, preds["a_all24"], average="macro")
        for meth, blk in [("iid_day", 1), ("block_7d", 7)]:
            d = boot_diff(y, preds["a_all24"], preds[name], rng, blk)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(model="LR_bootstrap", feature_set=f"{name}_minus_a", method=meth,
                             diff_point=point, diff_mean=d.mean(), ci_low=lo, ci_high=hi, p_ge0=(d >= 0).mean()))

    res = pd.DataFrame(rows)
    res.to_csv("research/evening_features_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
