"""
10分データ集計で見つかった系統的なずれ（日照 -0.63時間、最小湿度 +2.25%）が、
予測クラスをどれだけ入れ替えるかの感度を、手元データ（data/processed/dataset.csv）だけで測る。

- モデル: 特徴量セット(c)のロジスティック回帰（ablation_evening_features.py と同じ学習データ・分割）。
- 評価364行（直近365日分割）の特徴量にずれを足して予測し、元の予測とクラスが違う行の割合を数える。
  日照は0未満にならないよう0で切る。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/sensitivity_realtime_bias.py`
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ablation_evening_features import SETS
from compare_models import load_data, time_series_split, build_models, TARGET_COL

SHIFTS = {
    "sunshine_-0.63h": {"hiro_sunshine": -0.63},
    "humidity_min_+2.25": {"hiro_humidity_min": 2.25},
    "both": {"hiro_sunshine": -0.63, "hiro_humidity_min": 2.25},
}


def main():
    cols = SETS["c_no_today_snow"]
    train_df, eval_df = time_series_split(load_data(), eval_days=365)
    lr = build_models()["ロジスティック回帰"].fit(train_df[cols], train_df[TARGET_COL])
    X, y = eval_df[cols], eval_df[TARGET_COL]
    base = lr.predict(X)
    p = lr.predict_proba(X)
    top2 = np.sort(p, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    rows = [dict(shift="none", n=len(X), changed=0, changed_rate=0.0,
                 macro_f1=f1_score(y, base, average="macro"), acc=(base == y).mean())]
    for name, sh in SHIFTS.items():
        Xs = X.copy()
        for c, v in sh.items():
            Xs[c] = Xs[c] + v
        if "hiro_sunshine" in sh:
            Xs["hiro_sunshine"] = Xs["hiro_sunshine"].clip(lower=0)
        ps = lr.predict(Xs)
        ch = ps != base
        trans = pd.Series([f"{a}->{b}" for a, b in zip(base[ch], ps[ch])]).value_counts().to_dict()
        rows.append(dict(shift=name, n=len(X), changed=int(ch.sum()), changed_rate=ch.mean(),
                         macro_f1=f1_score(y, ps, average="macro"), acc=(ps == y).mean(),
                         changed_margin_max=margin[ch].max() if ch.any() else np.nan,
                         transitions=str(trans)))
    res = pd.DataFrame(rows)
    res.to_csv("research/realtime_bias_sensitivity.csv", index=False)
    print(f"評価 {len(X)}行 {eval_df.date.min().date()}〜{eval_df.date.max().date()}")
    print(res.round(4).to_string())


if __name__ == "__main__":
    main()
