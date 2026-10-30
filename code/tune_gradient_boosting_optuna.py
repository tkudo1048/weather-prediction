"""
勾配ブースティング（GradientBoostingClassifier）を Optuna（TPESampler によるベイズ最適化）でチューニングし、
GridSearchCV版（tune_gradient_boosting.py）・ロジスティック回帰と比較する。

- 探索は学習データ（1,488行）内の TimeSeriesSplit(n_splits=5) 交差検証のみ（シャッフルなし）。
- 評価データ（364行）は最良パラメータ決定後の最終評価にのみ使う（リーク防止）。
- 目的指標: CV Macro F1（GridSearchCV版と同一）。
- 実行方法: プロジェクト直下で `PYTHONPATH=code python code/tune_gradient_boosting_optuna.py`
"""
import time
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from compare_models import load_data, time_series_split, build_models, evaluate, FEATURE_COLS, TARGET_COL

N_TRIALS = 150
GRID_BEST = dict(n_estimators=200, max_depth=3, learning_rate=0.01, min_samples_leaf=20, subsample=0.6)


def make_pipe(**p):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", GradientBoostingClassifier(random_state=42, **p))])


def main():
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]
    cv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 600, step=10),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.4, 1.0),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        )
        s = cross_val_score(make_pipe(**p), X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
        trial.set_user_attr("std", float(s.std()))
        trial.set_user_attr("folds", [round(float(v), 3) for v in s])
        return s.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    elapsed = time.time() - t0
    b = study.best_trial
    print(f"試行数: {N_TRIALS}, 所要時間: {elapsed:.1f}秒")
    print("最良パラメータ:", b.params)
    print(f"CV Macro F1: {b.value:.3f} (±{b.user_attrs['std']:.3f}) folds={b.user_attrs['folds']} (trial #{b.number})")

    tdf = study.trials_dataframe()
    tdf["best_so_far"] = tdf["value"].cummax()
    for k in [10, 25, 50, 100, 150]:
        if k <= len(tdf):
            print(f"  {k}試行時点の最良CV: {tdf['best_so_far'].iloc[k-1]:.4f}")
    print(tdf.sort_values("value", ascending=False)[[c for c in tdf.columns if c.startswith("params_")] + ["value"]].head(10).to_string(index=False))
    tdf.to_csv("research/gb_optuna_trials.csv", index=False)

    # GridSearchCV最良の同条件CVスコア（参照）
    g = cross_val_score(make_pipe(**GRID_BEST), X_tr, y_tr, cv=cv, scoring="f1_macro")
    print(f"Grid最良 CV: {g.mean():.3f} (±{g.std():.3f}) folds={[round(v,3) for v in g]}")
    lr = build_models()["ロジスティック回帰"]
    lr_cv = cross_val_score(lr, X_tr, y_tr, cv=cv, scoring="f1_macro")
    print(f"LR CV: {lr_cv.mean():.3f} (±{lr_cv.std():.3f}) folds={[round(v,3) for v in lr_cv]}")

    models = {"ロジスティック回帰": lr, "GB(GridSearchCV)": make_pipe(**GRID_BEST), "GB(Optuna)": make_pipe(**b.params)}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        tra, trf, _ = evaluate(y_tr, m.predict(X_tr))
        eva, evf, cf = evaluate(y_ev, m.predict(X_ev))
        print(f"{name}: 学習Acc={tra:.3f} F1={trf:.3f} / 評価Acc={eva:.3f} F1={evf:.3f} / ギャップAcc={tra-eva:.3f} F1={trf-evf:.3f} / "
              + str({c: round(v, 3) for c, v in cf.items()}))


if __name__ == "__main__":
    main()
