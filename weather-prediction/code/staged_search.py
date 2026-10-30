"""
勾配ブースティングのGrid探索を staged_predict で高速化する共通関数。

n_estimators 以外の組み合わせ × fold ごとに、n_estimators の最大値で1回だけ学習し、
staged_predict（木を1本ずつ足した途中段階の予測）から各水準のスコアを取り出す。
GridSearchCV と同じ並び順・同じ列名の cv_results 相当の DataFrame を返す。
"""
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

N_EST_KEY = "clf__n_estimators"


def _fit_one(pipe, params, X, y, tr, te, n_levels, return_train):
    """1つの組み合わせ×foldを最大本数で学習し、各水準のスコアを返す。"""
    m = clone(pipe).set_params(**params, **{N_EST_KEY: max(n_levels)})
    t0 = time.time()
    m.fit(X.iloc[tr], y.iloc[tr])
    fit_t = time.time() - t0
    imp, clf = m.named_steps["imputer"], m.named_steps["clf"]

    def scores(idx):
        Xt = imp.transform(X.iloc[idx])
        yt = y.iloc[idx]
        out = {}
        for k, pred in enumerate(clf.staged_predict(Xt), start=1):
            if k in n_levels:
                out[k] = f1_score(yt, pred, average="macro")
        return out

    t0 = time.time()
    te_s = scores(te)
    score_t = time.time() - t0
    tr_s = scores(tr) if return_train else None
    return te_s, tr_s, fit_t, score_t


def staged_grid_search(pipe, space, X, y, cv, n_jobs=-1, return_train_score=False):
    """GridSearchCV(scoring='f1_macro') 相当の結果を staged 方式で計算する。

    戻り値: (cv_results の DataFrame, best_index)
    best_index は GridSearchCV と同じく rank_test_score 最小の先頭（同点なら先に出てくる方）。
    """
    n_levels = sorted(space[N_EST_KEY])
    base = {k: v for k, v in space.items() if k != N_EST_KEY}
    base_list = list(ParameterGrid(base))
    splits = list(cv.split(X, y))
    jobs = [(b, f) for b in range(len(base_list)) for f in range(len(splits))]
    outs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(pipe, base_list[b], X, y, *splits[f], set(n_levels), return_train_score)
        for b, f in jobs)
    got = {(b, f): o for (b, f), o in zip(jobs, outs)}

    # GridSearchCV と同じ並び（ParameterGrid の順）で行を作る
    rows = []
    for p in ParameterGrid(space):
        b = base_list.index({k: v for k, v in p.items() if k != N_EST_KEY})
        n = p[N_EST_KEY]
        row = {}
        fits = [got[(b, f)][2] for f in range(len(splits))]
        sc_t = [got[(b, f)][3] for f in range(len(splits))]
        # 時間は最大本数での1回分（水準ごとの値ではない）
        row.update(mean_fit_time=np.mean(fits), std_fit_time=np.std(fits),
                   mean_score_time=np.mean(sc_t), std_score_time=np.std(sc_t))
        for k, v in p.items():
            row[f"param_{k}"] = v
        row["params"] = p
        te = [got[(b, f)][0][n] for f in range(len(splits))]
        for f, s in enumerate(te):
            row[f"split{f}_test_score"] = s
        row["mean_test_score"], row["std_test_score"] = np.mean(te), np.std(te)
        if return_train_score:
            trs = [got[(b, f)][1][n] for f in range(len(splits))]
            for f, s in enumerate(trs):
                row[f"split{f}_train_score"] = s
            row["mean_train_score"], row["std_train_score"] = np.mean(trs), np.std(trs)
        rows.append(row)
    res = pd.DataFrame(rows)
    # 列順を GridSearchCV に合わせる（train系は末尾）
    res["rank_test_score"] = rankdata(-res["mean_test_score"].to_numpy(), method="min").astype(np.int32)
    if return_train_score:
        tail = [c for c in res.columns if "train_score" in c]
        res = res[[c for c in res.columns if c not in tail] + tail]
    best = int(res["rank_test_score"].to_numpy().argmin())
    return res, best
