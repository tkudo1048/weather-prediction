"""これまで試していない3点で、採用中のロジスティック回帰(LR)との優劣が変わるかを確かめる。

(A) LightGBM / XGBoost(勾配ブースティングの別実装)。Optuna 150試行、クラス重みは balanced。
(B) LRの調整(正則化の強さC、L1/L2の混ぜ具合 l1_ratio、class_weight の有無)。Optuna 150試行。
(C) クラス重みの強さ: 行の重み = balanced重み ** alpha (alpha=0,0.25,0.5,0.75,1.0)。
    LR / GB(sklearn) / LightGBM / XGBoost で比較。alpha は学習データ内CVで選ぶ。

- 分割は既存の load_data / time_series_split(評価は直近365日、シャッフルなし)。
- 探索・選択はすべて学習1,488行の中の TimeSeriesSplit(5) だけ。評価364行は最終評価にのみ使う。
- 基準(前日と同じ / 月別最頻 / 常に多数派)、過学習ギャップ、seed違い(5通り)、
  ブートストラップ(日単位・7日ブロック)95%信頼区間を出す。
- 実行: プロジェクト直下で `PYTHONPATH=code .venv/bin/python code/untested_items.py`
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from compare_models import load_data, time_series_split, build_models, evaluate, FEATURE_COLS, TARGET_COL, CLASSES
from train_model import persistence_baseline
from seed_robustness import load_best_params_42, boot_diff

warnings.filterwarnings("ignore")
N_TRIALS = 150
MODEL_SEEDS = [0, 1, 2, 3, 42]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CV = TimeSeriesSplit(n_splits=5)
OUT = "research/untested_"


class XGBLabel(BaseEstimator, ClassifierMixin):
    """XGBoost は整数ラベルが必要なので、文字ラベルとの変換だけを挟む。"""

    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **p):
        self.params.update(p)
        return self

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array(CLASSES)
        yi = pd.Series(y).map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        self.m_ = xgb.XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=4,
                                    verbosity=0, **self.params)
        self.m_.fit(X, yi, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.classes_[self.m_.predict(X).astype(int)]


def make(kind, seed, **p):
    if kind == "LR":
        clf = LogisticRegression(max_iter=5000, random_state=seed, **p)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])
    if kind == "GB":
        clf = GradientBoostingClassifier(random_state=seed, **p)
    elif kind == "LGBM":
        clf = lgb.LGBMClassifier(random_state=seed, n_jobs=4, verbosity=-1, subsample_freq=1, **p)
    else:
        clf = XGBLabel(random_state=seed, **p)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def weights(y, alpha):
    if alpha is None:
        return None
    return compute_sample_weight("balanced", y) ** alpha


def cv_score(kind, seed, p, X, y, alpha):
    """学習データ内の TimeSeriesSplit(5)。重みは各foldの学習部分だけから計算する。"""
    s = []
    for tr, te in CV.split(X):
        m = make(kind, seed, **p)
        w = weights(y.iloc[tr], alpha)
        m.fit(X.iloc[tr], y.iloc[tr], **({} if w is None else {"clf__sample_weight": w}))
        s.append(f1_score(y.iloc[te], m.predict(X.iloc[te]), average="macro"))
    return float(np.mean(s)), float(np.std(s))


def space(kind, trial):
    if kind == "LGBM":
        return dict(n_estimators=trial.suggest_int("n_estimators", 50, 600, step=10),
                    max_depth=trial.suggest_int("max_depth", 1, 5),
                    num_leaves=trial.suggest_int("num_leaves", 2, 31),
                    learning_rate=trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 1, 100, log=True),
                    subsample=trial.suggest_float("subsample", 0.4, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if kind == "XGB":
        return dict(n_estimators=trial.suggest_int("n_estimators", 50, 600, step=10),
                    max_depth=trial.suggest_int("max_depth", 1, 5),
                    learning_rate=trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
                    min_child_weight=trial.suggest_float("min_child_weight", 0.1, 20, log=True),
                    subsample=trial.suggest_float("subsample", 0.4, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    # LR: l1_ratio=0 が L2、0<l1_ratio<1 が elasticnet(L1とL2の混合)
    l1 = trial.suggest_float("l1_ratio", 0.0, 1.0)
    p = dict(C=trial.suggest_float("C", 1e-3, 100, log=True), l1_ratio=l1,
             solver="saga" if l1 > 0 else "lbfgs")
    p["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", "none"])
    return p


def lr_params(p):
    p = dict(p)
    if p.get("class_weight") == "none":
        p["class_weight"] = None
    p["solver"] = "saga" if p["l1_ratio"] > 0 else "lbfgs"
    return p


def search(kind, X, y, seed=42):
    def obj(trial):
        p = space(kind, trial)
        alpha = 1.0
        if kind == "LR":
            p = lr_params(p)
        m, s = cv_score(kind, seed, p, X, y, None if kind == "LR" else alpha)
        trial.set_user_attr("std", s)
        return m
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=N_TRIALS)
    df = st.trials_dataframe()
    df.to_csv(f"{OUT}{kind.lower()}_optuna_trials.csv", index=False)
    bp = st.best_params
    return lr_params(bp) if kind == "LR" else bp, st.best_value


def fit_eval(model, X_tr, y_tr, X_ev, y_ev, w=None):
    model.fit(X_tr, y_tr, **({} if w is None else {"clf__sample_weight": w}))
    tra, trf, _ = evaluate(y_tr, model.predict(X_tr))
    pred = np.asarray(model.predict(X_ev))
    eva, evf, cf = evaluate(y_ev, pred)
    row = dict(eval_acc=eva, eval_f1=evf, train_acc=tra, train_f1=trf, gap_acc=tra - eva, gap_f1=trf - evf)
    row.update({f"f1_{c}": v for c, v in cf.items()})
    return row, pred


def base_row(name, y_tr, p_tr, y_ev, p_ev):
    tra, trf, _ = evaluate(y_tr, p_tr)
    eva, evf, cf = evaluate(y_ev, p_ev)
    return dict(part="baseline", model=name, eval_acc=eva, eval_f1=evf, train_acc=tra, train_f1=trf,
                gap_acc=tra - eva, gap_f1=trf - evf, **{f"f1_{c}": v for c, v in cf.items()})


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    T0 = time.time()
    df = load_data()
    train_df, eval_df = time_series_split(df, eval_days=365)
    X_tr, y_tr = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_ev, y_ev = eval_df[FEATURE_COLS], eval_df[TARGET_COL]
    print(f"学習 {len(X_tr)}行 / 評価 {len(X_ev)}行 / 特徴量 {len(FEATURE_COLS)}", flush=True)
    rows, preds = [], {}

    # 基準
    maj = y_tr.mode()[0]
    rows.append(base_row("always_" + maj, y_tr, np.full(len(y_tr), maj), y_ev, np.full(len(y_ev), maj)))
    mm = train_df.groupby(train_df.date.dt.month)[TARGET_COL].agg(lambda s: s.mode()[0])
    rows.append(base_row("monthly_mode", y_tr, train_df.date.dt.month.map(mm).to_numpy(),
                         y_ev, eval_df.date.dt.month.map(mm).to_numpy()))
    rows.append(base_row("persistence", y_tr, persistence_baseline(train_df).to_numpy(),
                         y_ev, persistence_baseline(eval_df).to_numpy()))

    # 採用中のLR(既存設定そのもの)
    lr_cv = cv_score("LR", 42, dict(class_weight="balanced"), X_tr, y_tr, None)
    r, preds["LR_current"] = fit_eval(build_models()["ロジスティック回帰"], X_tr, y_tr, X_ev, y_ev)
    rows.append(dict(part="current", model="LR_current", seed=42, cv_f1=lr_cv[0], cv_std=lr_cv[1], **r))

    # (A)(B) Optuna 探索(学習データ内のみ)
    best = {}
    for kind in ["LGBM", "XGB", "LR"]:
        t0 = time.time()
        bp, bv = search(kind, X_tr, y_tr)
        best[kind] = bp
        mins = (time.time() - t0) / 60
        print(kind, "探索", round(mins, 1), "分 CV", round(bv, 4), bp, flush=True)
        part = "B_LR_tuned" if kind == "LR" else "A_" + kind
        for s in MODEL_SEEDS:
            w = None if kind == "LR" else weights(y_tr, 1.0)
            r, pred = fit_eval(make(kind, s, **bp), X_tr, y_tr, X_ev, y_ev, w)
            rows.append(dict(part=part, model=kind + "_tuned", seed=s, cv_f1=bv, params=str(bp),
                             search_minutes=mins, **r))
            if s == 42:
                preds[kind + "_tuned"] = pred

    # (C) 重みの強さ。パラメータは固定(LRは既存設定、GBは既存Optuna最良、LGBM/XGBは(A)の最良)。
    fixed = {"LR": dict(), "GB": load_best_params_42(), "LGBM": best["LGBM"], "XGB": best["XGB"]}
    for kind, p in fixed.items():
        t0 = time.time()
        for a in ALPHAS:
            cvm, cvs = cv_score(kind, 42, p, X_tr, y_tr, a)
            for s in MODEL_SEEDS:
                r, pred = fit_eval(make(kind, s, **p), X_tr, y_tr, X_ev, y_ev, weights(y_tr, a))
                rows.append(dict(part="C_alpha", model=kind, alpha=a, seed=s, cv_f1=cvm, cv_std=cvs, **r))
                if s == 42:
                    preds[f"C_{kind}_{a}"] = pred
        print(kind, "alpha完了", round((time.time() - t0) / 60, 1), "分", flush=True)
    res = pd.DataFrame(rows)
    # 学習データ内CVで選んだ alpha を記録
    c = res[(res.part == "C_alpha") & (res.seed == 42)]
    chosen = c.loc[c.groupby("model").cv_f1.idxmax()][["model", "alpha"]]
    res["alpha_chosen_by_cv"] = False
    for _, z in chosen.iterrows():
        res.loc[(res.part == "C_alpha") & (res.model == z.model) & (res.alpha == z.alpha), "alpha_chosen_by_cv"] = True
    res.to_csv(OUT + "results.csv", index=False)

    # ブートストラップ: 各候補(seed=42) − 現行LR
    y = y_ev.to_numpy()
    targets = ["LGBM_tuned", "XGB_tuned", "LR_tuned"] + [f"C_{z.model}_{z.alpha}" for _, z in chosen.iterrows()]
    bs = []
    for t in dict.fromkeys(targets):
        for b in [1, 7]:
            d = boot_diff(y, preds["LR_current"], preds[t], np.random.default_rng(42), block=b)
            obs = f1_score(y, preds[t], average="macro") - f1_score(y, preds["LR_current"], average="macro")
            bs.append(dict(candidate=t, block_days=b, diff_point=obs, ci_lo=np.percentile(d, 2.5),
                           ci_hi=np.percentile(d, 97.5), p_le0=(d <= 0).mean()))
    pd.DataFrame(bs).to_csv(OUT + "bootstrap.csv", index=False)
    print(pd.DataFrame(bs).round(3).to_string())
    print("合計", round((time.time() - T0) / 60, 1), "分")


if __name__ == "__main__":
    main()
