"""
Optuna版 勾配ブースティングの seed 依存性と、ロジスティック回帰との差の統計的確認。

実験A: 最良パラメータ固定で GradientBoostingClassifier の random_state だけを10通り変える。
実験B: Optuna の sampler seed とモデル random_state を5通り変えて探索(150試行)をやり直す。
実験C: 評価364行でロジスティック回帰 vs Optuna版(random_state=42, 元の最良パラメータ)の
       Macro F1 差をブートストラップ(日単位 / 7日ブロック, 各1000回)で区間推定。

- 探索は学習1,488行内の TimeSeriesSplit(5) のみ。評価364行はチューニングに使わない。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/seed_robustness.py`
"""
import time
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from compare_models import load_data, time_series_split, build_models, evaluate, FEATURE_COLS, TARGET_COL, CLASSES

N_TRIALS = 150
MODEL_SEEDS = [0, 1, 2, 3, 7, 13, 21, 42, 100, 2024]
SEARCH_SEEDS = [42, 0, 1, 2, 3]
N_BOOT = 1000
BLOCK = 7
BEST_PARAMS_42 = dict(n_estimators=310, max_depth=4, learning_rate=None, min_samples_leaf=71,
                      subsample=None, max_features="log2")  # 実数値は元探索の trials CSV から読む


def make_pipe(rs, **p):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", GradientBoostingClassifier(random_state=rs, **p))])


def load_best_params_42():
    t = pd.read_csv("research/gb_optuna_trials.csv")
    r = t.loc[t["value"].idxmax()]
    return dict(n_estimators=int(r.params_n_estimators), max_depth=int(r.params_max_depth),
                learning_rate=float(r.params_learning_rate), min_samples_leaf=int(r.params_min_samples_leaf),
                subsample=float(r.params_subsample),
                max_features=None if pd.isna(r.params_max_features) else r.params_max_features)


def fit_eval(model, X_tr, y_tr, X_ev, y_ev):
    model.fit(X_tr, y_tr)
    tra, trf, _ = evaluate(y_tr, model.predict(X_tr))
    pred = model.predict(X_ev)
    eva, evf, cf = evaluate(y_ev, pred)
    row = dict(eval_acc=eva, eval_f1=evf, train_acc=tra, train_f1=trf, gap_acc=tra - eva, gap_f1=trf - evf)
    row.update({f"f1_{c}": v for c, v in cf.items()})
    return row, pred


def search(seed, X_tr, y_tr, cv):
    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 600, step=10),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        )
        s = cross_val_score(make_pipe(seed, **p), X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        trial.set_user_attr("std", float(s.std()))
        return s.mean()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_trial


def boot_diff(y, p_a, p_b, rng, block=1):
    n = len(y)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        if block == 1:
            idx = rng.integers(0, n, n)
        else:  # 移動ブロックブートストラップ
            starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[i] = (f1_score(y[idx], p_b[idx], average="macro", labels=CLASSES, zero_division=0)
                    - f1_score(y[idx], p_a[idx], average="macro", labels=CLASSES, zero_division=0))
    return diffs


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]
    cv = TimeSeriesSplit(n_splits=5)
    rows = []

    lr_row, lr_pred = fit_eval(build_models()["ロジスティック回帰"], X_tr, y_tr, X_ev, y_ev)
    rows.append(dict(experiment="LR", seed=42, **lr_row))

    # 実験A
    bp = load_best_params_42()
    print("元の最良パラメータ:", bp)
    gb42_pred = None
    for rs in MODEL_SEEDS:
        r, pred = fit_eval(make_pipe(rs, **bp), X_tr, y_tr, X_ev, y_ev)
        rows.append(dict(experiment="A_model_seed", seed=rs, **r))
        if rs == 42:
            gb42_pred = pred
    # 実験B
    for s in SEARCH_SEEDS:
        t0 = time.time()
        b = search(s, X_tr, y_tr, cv)
        r, _ = fit_eval(make_pipe(s, **b.params), X_tr, y_tr, X_ev, y_ev)
        rows.append(dict(experiment="B_search_seed", seed=s, cv_f1=b.value, cv_std=b.user_attrs["std"],
                         params=str(b.params), minutes=(time.time() - t0) / 60, **r))
        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("research/seed_robustness_results.csv", index=False)
    cols = ["eval_acc", "eval_f1", "gap_f1", "gap_acc"] + [f"f1_{c}" for c in CLASSES]
    for e in ["A_model_seed", "B_search_seed"]:
        d = res[res.experiment == e]
        print(e, "\n", d[cols].agg(["mean", "std", "min", "max"]).round(3).to_string())
        print(" LR超え(Macro F1)件数:", int((d.eval_f1 > lr_row["eval_f1"]).sum()), "/", len(d))
    print("LR", {k: round(v, 3) for k, v in lr_row.items()})

    # 実験C
    y = y_ev.to_numpy()
    rng = np.random.default_rng(42)
    out = []
    for name, blk in [("iid_day", 1), (f"block_{BLOCK}d", BLOCK)]:
        d = boot_diff(y, lr_pred, gb42_pred, rng, blk)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append(dict(method=name, point=f1_score(y, gb42_pred, average="macro") - f1_score(y, lr_pred, average="macro"),
                        mean=d.mean(), ci_low=lo, ci_high=hi, p_le0=(d <= 0).mean()))
    bs = pd.DataFrame(out)
    bs.to_csv("research/seed_robustness_bootstrap.csv", index=False)
    print(bs.round(4).to_string())


if __name__ == "__main__":
    main()
