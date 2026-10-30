"""勾配ブースティングにクラス重み(balanced)を入れて、ロジスティック回帰と条件を揃えた再評価。

- パラメータは既存探索(重みなし条件)の最良値を固定して流用する。評価364行では探索しない。
- 学習1,488行で学習し、評価364行で最終評価のみ行う。
"""
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from compare_models import load_data, time_series_split, build_models, evaluate, FEATURE_COLS, TARGET_COL, CLASSES
from seed_robustness import make_pipe, load_best_params_42, boot_diff

MODEL_SEEDS = [0, 1, 2, 3, 4, 42]
BLOCKS = [1, 7]


def load_grid_best():
    g = pd.read_csv("research/gb_tuning_cv_results.csv")
    r = g.loc[g["mean_test_score"].idxmax()]
    return dict(n_estimators=int(r.param_clf__n_estimators), max_depth=int(r.param_clf__max_depth),
                learning_rate=float(r.param_clf__learning_rate),
                min_samples_leaf=int(r.param_clf__min_samples_leaf), subsample=float(r.param_clf__subsample))


def fit_eval(model, X_tr, y_tr, X_ev, y_ev, sw=None):
    model.fit(X_tr, y_tr, **({} if sw is None else {"clf__sample_weight": sw}))
    tra, trf, _ = evaluate(y_tr, model.predict(X_tr))
    pred = model.predict(X_ev)
    eva, evf, cf = evaluate(y_ev, pred)
    row = dict(eval_acc=eva, eval_f1=evf, train_acc=tra, train_f1=trf, gap_acc=tra - eva, gap_f1=trf - evf)
    row.update({f"f1_{c}": v for c, v in cf.items()})
    return row, pred


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]
    sw = compute_sample_weight("balanced", y_tr)
    rows = []
    lr_row, lr_pred = fit_eval(build_models()["ロジスティック回帰"], X_tr, y_tr, X_ev, y_ev)
    rows.append(dict(model="LR", weight="balanced", seed=np.nan, **lr_row))

    preds = {}
    for pname, params in [("Optuna", load_best_params_42()), ("Grid", load_grid_best())]:
        for wname, w in [("none", None), ("balanced", sw)]:
            for rs in MODEL_SEEDS:
                r, pred = fit_eval(make_pipe(rs, **params), X_tr, y_tr, X_ev, y_ev, w)
                rows.append(dict(model=f"GB_{pname}", weight=wname, seed=rs, **r))
                if rs == 42:
                    preds[(pname, wname)] = pred
    res = pd.DataFrame(rows)

    y = y_ev.to_numpy()
    for (pname, wname), pred in preds.items():
        for b in BLOCKS:
            d = boot_diff(y, lr_pred, pred, np.random.default_rng(42), block=b)
            obs = res[(res.model == f"GB_{pname}") & (res.weight == wname) & (res.seed == 42)].eval_f1.iloc[0] - lr_row["eval_f1"]
            rows_b = dict(model=f"GB_{pname}", weight=wname, seed="boot_block%d" % b, eval_f1=obs,
                          ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p_le0=(d <= 0).mean())
            res = pd.concat([res, pd.DataFrame([rows_b])], ignore_index=True)
    res.to_csv("research/gb_balanced_weights_results.csv", index=False)

    cols = ["eval_acc", "eval_f1", "gap_f1"] + [f"f1_{c}" for c in CLASSES]
    print(res[res.seed == 42][["model", "weight"] + cols].round(3).to_string())
    print(res[res.model == "LR"][cols].round(3).to_string())
    s = res[res.seed.isin(MODEL_SEEDS)]
    print(s.groupby(["model", "weight"])[cols].agg(["mean", "std"]).round(3).T.to_string())
    print(res[res.seed.astype(str).str.startswith("boot")][["model", "weight", "seed", "eval_f1", "ci_lo", "ci_hi", "p_le0"]].round(3).to_string())


if __name__ == "__main__":
    main()
