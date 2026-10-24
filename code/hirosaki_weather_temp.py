"""
弘前の「翌日の天気(晴/曇/雨/雪)」と「翌日の最高・最低気温」を予測する検証。

問い:
  (1) 弘前には天気概況の観測がない。観測値(降雪・降水量・日照)のルールで作った「ルール天気」は、
      青森の天気概況とどのくらい一致するか。
  (2) 翌日の弘前のルール天気を、夜の18特徴量 + GSM(previous_day2)で予測できるか。
      GSMの格子点は青森格子(41.0, 141.0)と弘前格子(応答の格子点)の2通り。
  (3) 翌日の弘前の最高・最低気温を、Ridge回帰で予測すると、GSMの予報気温そのものより良いか。

ルール天気(しきい値は青森で学習データだけで選んだものを、弘前では再調整せずに使う):
  降雪量 >= 2cm → 雪 / 降水量 >= 5mm → 雨 / 日照時間 ÷ その月の日照95%点 >= 0.6 → 晴 / それ以外 → 曇
  - 月別の日照95%点は、弘前の学習期間内(観測日 <= 学習の最終日)の日だけで計算する(評価期間を使わない)。
  - 降雪・降水量の欠測は0とみなす(青森でしきい値を選んだときと同じ)。日照が欠測で雪・雨に当たらない日は、ラベルなし。

データ:
  - 観測: data/raw/hirosaki_daily.csv(弘前アメダス)、data/raw/aomori_daily.csv(青森気象台の天気概況(昼))
  - 予報: data/raw/nwp/aomori_jma_gsm_2021-08-01_2026-08-31.json(青森格子、code/fetch_nwp.py)、
          data/raw/nwp/hirosaki_jma_gsm_day2_2021-08-01_2026-08-31.json(弘前格子、code/fetch_nwp_hirosaki.py)。
          Open-Meteo Previous Runs API(CC BY 4.0、出典 Open-Meteo.com)。previous_day2 だけを使う。
  - 日別集計は本番と共通の nwp_features.daily_features(対象日の06〜18時 JST)。

結合と分割:
  - 行 = 観測日D(既存の load_data の行。青森の翌日の天気概況がある日)。目的変数は弘前のD+1の値を、
    暦の日付(D + 1日)で結合する。特徴量はDの観測と、D+1を対象にした約48時間前の予報だけ。
  - 分割は既存の time_series_split(学習1,488行・評価364行)。目的変数が欠測の行は、分割後に除く。
  - 探索(Ridgeのalpha)は学習データ内の TimeSeriesSplit(5) だけ。評価データは最終評価にだけ使う。
  - 欠損補完(中央値)と標準化は Pipeline の中(学習データでだけ fit)。
  - 日付合わせの確認: 予報の対象日を D / D+2 にずらした版も評価する(ずらすと悪化するはず)。

実行: プロジェクト直下で `PYTHONPATH=code .venv/bin/python code/hirosaki_weather_temp.py`
出力:
  research/hirosaki_rule_label_check.csv … ルール天気の分布・青森の天気概況との一致率・混同行列
  research/hirosaki_weather_eval.csv     … 翌日のルール天気(4分類)の評価とブートストラップ
  research/hirosaki_temp_eval.csv        … 翌日の最高・最低気温の評価とブートストラップ
  research/hirosaki_gsm_grid_compare.csv … 青森格子と弘前格子のGSM日別値の比較
  research/hirosaki_join_check.csv       … 結合確認用の数行
"""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from compare_models import load_data, time_series_split, build_models, evaluate, CLASSES
from fetch_realtime import FEATURES
from fetch_nwp import out_path as aomori_nwp_path
from fetch_nwp_hirosaki import out_path as hirosaki_nwp_path
from nwp_features import daily_features, hourly_frame
from preprocess import load, HIROSAKI_COLS, AOMORI_COLS, to_label
from seed_robustness import boot_diff

SNOW_CM, PRECIP_MM, SUN_RATIO = 2.0, 5.0, 0.6  # 青森で学習データだけで選んだしきい値(弘前では再調整しない)
LEAD = "previous_day2"
PREFIX = {"aomori": "gsm2", "hirosaki": "hgsm2"}
N_BOOT = 1000
ALPHAS = np.logspace(-3, 3, 13)
SEASON = {"summer": [6, 7, 8], "winter": [12, 1, 2]}


# ---------------------------------------------------------------- ルール天気
def sun_q95_by_month(obs: pd.DataFrame, last_train_day: pd.Timestamp) -> pd.Series:
    """月別の日照95%点。学習期間内(観測日 <= last_train_day)の日だけで計算する。"""
    tr = obs.loc[obs.index <= last_train_day, "sunshine"]
    return tr.groupby(tr.index.month).quantile(0.95)


def rule_label(obs: pd.DataFrame, q95: pd.Series) -> pd.Series:
    snow = obs["snowfall"].fillna(0)
    prec = obs["precip_total"].fillna(0)
    ratio = obs["sunshine"] / obs.index.month.map(q95).to_numpy()
    lab = pd.Series(np.where(ratio >= SUN_RATIO, "晴", "曇"), index=obs.index, dtype=object)
    lab[ratio.isna()] = np.nan
    lab[prec >= PRECIP_MM] = "雨"
    lab[snow >= SNOW_CM] = "雪"
    return lab


def agreement_rows(name, a, b):
    """a=比べる側(行)、b=青森の天気概況(列)。一致率と混同行列を縦長の行にする。"""
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    rows = [dict(section="agreement", period=name, n=len(a), value=(a == b).mean())]
    for c in CLASSES:
        rows.append(dict(section="dist_compare", period=name, cls=c, n=int((a == c).sum()),
                         value=(a == c).mean(), aomori_n=int((b == c).sum()), aomori_share=(b == c).mean()))
    cm = pd.crosstab(a, b).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    for r in CLASSES:
        for c in CLASSES:
            rows.append(dict(section="confusion", period=name, cls=f"{r}->{c}", n=int(cm.loc[r, c])))
    for c in CLASSES:
        rows.append(dict(section="class_f1_vs_aomori", period=name, cls=c,
                         value=f1_score(b, a, labels=[c], average="macro", zero_division=0)))
    rows.append(dict(section="macro_f1_vs_aomori", period=name, n=len(a),
                     value=f1_score(b, a, labels=CLASSES, average="macro", zero_division=0)))
    return rows


# ---------------------------------------------------------------- データ
def gsm_daily(grid):
    path = aomori_nwp_path() if grid == "aomori" else hirosaki_nwp_path()
    d = json.loads(path.read_text(encoding="utf-8"))
    h = hourly_frame(d)
    dly = daily_features(h, PREFIX[grid], LEAD)
    # 参考の基準用: 0〜23時の毎時値の最高・最低(06〜18時だけだと朝の最低気温を取りこぼすため)
    t = h[["time", f"temperature_2m_{LEAD}"]].copy()
    t["day"] = t["time"].dt.normalize()
    g = t.groupby("day")[f"temperature_2m_{LEAD}"]
    full = pd.DataFrame({f"{PREFIX[grid]}_fullday_temp_max": g.max().where(g.count() >= 22),
                         f"{PREFIX[grid]}_fullday_temp_min": g.min().where(g.count() >= 22)})
    return dly, full, (d.get("latitude"), d.get("longitude"), d.get("elevation"))


def attach(df, dly, offset):
    """予報の対象日 D+offset の値を、観測日 D の行に置く(offset=1 が正しい結合)。"""
    x = dly.copy()
    x.index = x.index - pd.Timedelta(days=offset)
    x.index.name = "date"
    return df.merge(x.reset_index(), on="date", how="left")


def build(offset=1, cache=None):
    df = load_data()
    for grid in ("aomori", "hirosaki"):
        dly, full, _ = cache[grid]
        df = attach(df, dly.join(full, how="outer"), offset)
    return df


# ---------------------------------------------------------------- ブートストラップ
def boot_idx(n, rng, block):
    if block == 1:
        return rng.integers(0, n, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def boot_mae_diff(y, p_a, p_b, rng, block):
    """MAE(b) − MAE(a) のブートストラップ分布。負なら b のほうが誤差が小さい。"""
    ea, eb = np.abs(p_a - y), np.abs(p_b - y)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = boot_idx(len(y), rng, block)
        out[i] = eb[idx].mean() - ea[idx].mean()
    return out


# ---------------------------------------------------------------- (2) 天気
def weather_eval(df, gcols, cache):
    tr, ev = time_series_split(df, eval_days=365)
    n_tr0, n_ev0 = len(tr), len(ev)
    tr, ev = tr.dropna(subset=["target_h"]), ev.dropna(subset=["target_h"])
    print(f"[天気] 学習 {len(tr)}/{n_tr0}行  評価 {len(ev)}/{n_ev0}行(目的変数の欠測を除く)")
    y_tr, y_ev = tr["target_h"], ev["target_h"]

    def row(model, fs, p_tr, p_ev, **kw):
        tra, trf, _ = evaluate(y_tr, p_tr)
        eva, evf, cf = evaluate(y_ev, p_ev)
        r = dict(model=model, feature_set=fs, n_train=len(y_tr), n_eval=len(y_ev), eval_acc=eva, eval_f1=evf,
                 train_acc=tra, train_f1=trf, gap_acc=tra - eva, gap_f1=trf - evf)
        r.update({f"f1_{c}": cf[c] for c in CLASSES})
        r.update(kw)
        return r

    rows, preds = [], {}
    maj = y_tr.mode()[0]
    rows.append(row("majority", f"const={maj}", np.full(len(y_tr), maj), np.full(len(y_ev), maj)))
    mm = tr.groupby(tr.date.dt.month)["target_h"].agg(lambda s: s.mode()[0])
    rows.append(row("monthly_mode", "month", tr.date.dt.month.map(mm).to_numpy(), ev.date.dt.month.map(mm).to_numpy()))
    # 前日と同じ: 今日(D)の弘前のルール天気。今日が欠測なら学習データの多数派で埋める
    pers_tr, pers_ev = tr["today_h"].fillna(maj).to_numpy(), ev["today_h"].fillna(maj).to_numpy()
    preds["persistence"] = pers_ev
    rows.append(row("persistence", "today_rule_label", pers_tr, pers_ev,
                    n_today_missing_eval=int(ev["today_h"].isna().sum())))

    sets = {"base18": FEATURES, "base18+gsm2_aomori": FEATURES + gcols["aomori"],
            "base18+gsm2_hirosaki": FEATURES + gcols["hirosaki"]}
    cv = TimeSeriesSplit(n_splits=5)
    for name, cols in sets.items():
        lr = build_models()["ロジスティック回帰"]
        cvs = cross_val_score(lr, tr[cols], y_tr, cv=cv, scoring="f1_macro")
        lr.fit(tr[cols], y_tr)
        preds[name] = lr.predict(ev[cols])
        rows.append(row("LR", name, lr.predict(tr[cols]), preds[name], n_features=len(cols),
                        cv_f1_mean=cvs.mean(), cv_f1_std=cvs.std(),
                        n_eval_gsm_na=int(ev[cols].isna().any(axis=1).sum())))

    y = y_ev.to_numpy()
    rng = np.random.default_rng(42)
    pairs = [("base18+gsm2_aomori", "base18"), ("base18+gsm2_hirosaki", "base18"),
             ("base18+gsm2_hirosaki", "base18+gsm2_aomori"), ("base18", "persistence")]
    for b, a in pairs:
        point = (f1_score(y, preds[b], average="macro", labels=CLASSES, zero_division=0)
                 - f1_score(y, preds[a], average="macro", labels=CLASSES, zero_division=0))
        for meth, blk in [("iid_day", 1), ("block_7d", 7)]:
            d = boot_diff(y, np.asarray(preds[a]), np.asarray(preds[b]), rng, blk)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append(dict(model="LR_bootstrap", feature_set=f"{b}_minus_{a}", method=meth,
                             diff_point=point, ci_low=lo, ci_high=hi, p_ge0=(d >= 0).mean()))

    # 日付合わせの確認
    for off in (0, 2):
        dfo = add_targets(build(off, cache))
        tro, evo = time_series_split(dfo, eval_days=365)
        tro, evo = tro.dropna(subset=["target_h"]), evo.dropna(subset=["target_h"])
        for name in ("base18+gsm2_aomori", "base18+gsm2_hirosaki"):
            cols = sets[name]
            lr = build_models()["ロジスティック回帰"]
            lr.fit(tro[cols], tro["target_h"])
            eva, evf, cf = evaluate(evo["target_h"], lr.predict(evo[cols]))
            rows.append(dict(model="LR_shift_check", feature_set=f"{name}_offset{off}", n_eval=len(evo),
                             eval_acc=eva, eval_f1=evf, **{f"f1_{c}": cf[c] for c in CLASSES}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (3) 気温
def ridge():
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("reg", Ridge())])
    return GridSearchCV(pipe, {"reg__alpha": ALPHAS}, cv=TimeSeriesSplit(n_splits=5),
                        scoring="neg_mean_absolute_error")


def reg_metrics(y, p, months):
    e = p - y
    r = dict(n=len(y), mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()), bias=e.mean())
    for s, ms in SEASON.items():
        m = np.isin(months, ms)
        r[f"mae_{s}"] = np.abs(e[m]).mean()
        r[f"bias_{s}"] = e[m].mean()
        r[f"n_{s}"] = int(m.sum())
    return r


def temp_eval(df, gcols, cache):
    rows = []
    rng = np.random.default_rng(42)
    sets = {"base18": FEATURES, "base18+gsm2_aomori": FEATURES + gcols["aomori"],
            "base18+gsm2_hirosaki": FEATURES + gcols["hirosaki"]}
    for kind in ("max", "min"):
        tcol = f"target_t{kind}"
        tr, ev = time_series_split(df, eval_days=365)
        tr, ev = tr.dropna(subset=[tcol]), ev.dropna(subset=[tcol])
        y_tr, y_ev = tr[tcol].to_numpy(), ev[tcol].to_numpy()
        mon_tr = (tr.date + pd.Timedelta(days=1)).dt.month.to_numpy()  # 季節は目的変数の日(D+1)で分ける
        mon_ev = (ev.date + pd.Timedelta(days=1)).dt.month.to_numpy()
        # 比較はすべて同じ行で行う(GSMの素の予報が欠けている行を除いた共通の行)
        raw_cols = {"gsm_raw_aomori": f"gsm2_temp_{kind}", "gsm_raw_hirosaki": f"hgsm2_temp_{kind}",
                    "gsm_fullday_aomori(ref)": f"gsm2_fullday_temp_{kind}",
                    "gsm_fullday_hirosaki(ref)": f"hgsm2_fullday_temp_{kind}"}
        common = ev[list(raw_cols.values())].notna().all(axis=1).to_numpy()
        print(f"[最{'高' if kind == 'max' else '低'}気温] 学習 {len(tr)}行  評価 {len(ev)}行  共通の比較行 {common.sum()}")

        preds = {}
        preds["persistence"] = (ev[f"hiro_temp_{kind}"].to_numpy(), tr[f"hiro_temp_{kind}"].to_numpy())
        mmean = pd.Series(y_tr).groupby(mon_tr).mean()
        preds["monthly_mean"] = (pd.Series(mon_ev).map(mmean).to_numpy(), pd.Series(mon_tr).map(mmean).to_numpy())
        for name, c in raw_cols.items():
            preds[name] = (ev[c].to_numpy(), tr[c].to_numpy())
        # 参考の基準: GSMの素の予報 + 学習期間の月別平均誤差(実測 − 予報)で補正しただけのもの
        for grid in PREFIX:
            g_tr, g_ev = preds[f"gsm_raw_{grid}"][1], preds[f"gsm_raw_{grid}"][0]
            corr = pd.Series(y_tr - g_tr).groupby(mon_tr).mean()
            preds[f"gsm_biascorr_{grid}(ref)"] = (g_ev + pd.Series(mon_ev).map(corr).to_numpy(),
                                                  g_tr + pd.Series(mon_tr).map(corr).to_numpy())
        extra = {}
        for name, cols in sets.items():
            gs = ridge().fit(tr[cols], y_tr)
            preds[f"ridge_{name}"] = (gs.predict(ev[cols]), gs.predict(tr[cols]))
            extra[f"ridge_{name}"] = dict(alpha=gs.best_params_["reg__alpha"], cv_mae=-gs.best_score_,
                                          n_features=len(cols))

        for name, (p_ev, p_tr) in preds.items():
            r = dict(target=f"temp_{kind}", model=name, subset="common")
            r.update(reg_metrics(y_ev[common], p_ev[common], mon_ev[common]))
            ok_tr = ~np.isnan(p_tr)
            r["train_mae"] = np.abs(p_tr[ok_tr] - y_tr[ok_tr]).mean()
            r["gap_mae"] = r["mae"] - r["train_mae"]  # 評価MAE − 学習MAE(正なら評価で悪化)
            r.update(extra.get(name, {}))
            rows.append(r)
            if name.startswith("ridge_"):  # 評価の全行(GSM欠損の行は中央値補完)
                r2 = dict(target=f"temp_{kind}", model=name, subset="all_eval")
                r2.update(reg_metrics(y_ev, p_ev, mon_ev))
                rows.append(r2)

        yc = y_ev[common]
        pairs = [(f"ridge_{s}", g) for s in sets for g in ("gsm_raw_aomori", "gsm_raw_hirosaki")]
        pairs += [("ridge_base18+gsm2_hirosaki", "ridge_base18+gsm2_aomori"),
                  ("gsm_raw_hirosaki", "gsm_raw_aomori"), ("ridge_base18", "persistence")]
        pairs += [(f"ridge_base18+gsm2_{g}", f"gsm_biascorr_{g}(ref)") for g in PREFIX]
        pairs += [(f"ridge_base18+gsm2_{g}", f"gsm_fullday_{g}(ref)") for g in PREFIX]
        for b, a in pairs:
            pa, pb = preds[a][0][common], preds[b][0][common]
            point = np.abs(pb - yc).mean() - np.abs(pa - yc).mean()
            for meth, blk in [("iid_day", 1), ("block_7d", 7)]:
                d = boot_mae_diff(yc, pa, pb, rng, blk)
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(dict(target=f"temp_{kind}", model="bootstrap_mae_diff", subset=f"{b}_minus_{a}",
                                 method=meth, diff_point=point, ci_low=lo, ci_high=hi, p_le0=(d <= 0).mean()))

    # 日付合わせの確認(ずらすと悪化するはず)。GSMの素の予報と、18+GSMのRidge
    for off in (0, 2):
        dfo = add_targets(build(off, cache))
        for kind in ("max", "min"):
            tcol = f"target_t{kind}"
            tro, evo = time_series_split(dfo, eval_days=365)
            tro, evo = tro.dropna(subset=[tcol]), evo.dropna(subset=[tcol])
            mon = (evo.date + pd.Timedelta(days=1)).dt.month.to_numpy()
            for grid, pre in PREFIX.items():
                c = f"{pre}_temp_{kind}"
                ok = evo[c].notna().to_numpy()
                r = dict(target=f"temp_{kind}", model=f"shift_gsm_raw_{grid}", subset=f"offset{off}")
                r.update(reg_metrics(evo[tcol].to_numpy()[ok], evo[c].to_numpy()[ok], mon[ok]))
                rows.append(r)
                cols = FEATURES + gcols[grid]
                gs = ridge().fit(tro[cols], tro[tcol])
                r = dict(target=f"temp_{kind}", model=f"shift_ridge_base18+gsm2_{grid}", subset=f"offset{off}",
                         alpha=gs.best_params_["reg__alpha"])
                r.update(reg_metrics(evo[tcol].to_numpy(), gs.predict(evo[cols]), mon))
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
HIRO = None


def add_targets(df):
    """弘前のD+1の値(ルール天気・最高・最低気温)と、Dのルール天気を、暦の日付で結合する。"""
    nxt = HIRO.copy()
    nxt.index = nxt.index - pd.Timedelta(days=1)  # D+1 の値を D の行へ
    nxt = nxt.rename(columns={"label": "target_h", "temp_max": "target_tmax", "temp_min": "target_tmin"})
    nxt.index.name = "date"
    out = df.merge(nxt.reset_index(), on="date", how="left")
    today = HIRO[["label"]].rename(columns={"label": "today_h"})
    today.index.name = "date"
    return out.merge(today.reset_index(), on="date", how="left")


def main():
    global HIRO
    pd.set_option("display.width", 250)
    hiro = load("hirosaki", HIROSAKI_COLS)
    aomo = load("aomori", AOMORI_COLS)
    base = load_data()
    tr0, ev0 = time_series_split(base, eval_days=365)
    last_train_day = tr0.date.max()
    print(f"学習の最終日(観測日) {last_train_day.date()}  評価 {ev0.date.min().date()}〜{ev0.date.max().date()}")

    # ---- (1) ルール天気
    q95_h = sun_q95_by_month(hiro, last_train_day)
    hiro_lab = rule_label(hiro, q95_h)
    q95_a = sun_q95_by_month(aomo, last_train_day)
    aomo_rule = rule_label(aomo, q95_a)  # 参考: 同じ手順(学習期間内の95%点)で青森に当てはめた版
    aomo_obs = aomo["weather_day"].map(to_label)
    print("弘前の月別日照95%点(学習期間内):", q95_h.round(2).to_dict())
    print(f"ルール天気が作れない日(弘前): {int(hiro_lab.isna().sum())}  {hiro_lab.index[hiro_lab.isna()].date.tolist()}")

    lab_rows = [dict(section="sun_q95_hirosaki", period="train", cls=str(m), value=v) for m, v in q95_h.items()]
    lab_rows += [dict(section="sun_q95_aomori", period="train", cls=str(m), value=v) for m, v in q95_a.items()]
    idx = hiro_lab.index.intersection(aomo_obs.index)
    periods = {"all": idx, "train_period": idx[idx <= last_train_day + pd.Timedelta(days=1)],
               "eval_period": idx[idx > last_train_day + pd.Timedelta(days=1)]}
    for pname, ix in periods.items():
        lab_rows += [dict(r, compare="hirosaki_rule_vs_aomori_obs") for r in
                     agreement_rows(pname, hiro_lab.reindex(ix), aomo_obs.reindex(ix))]
        lab_rows += [dict(r, compare="aomori_rule_vs_aomori_obs(ref)") for r in
                     agreement_rows(pname, aomo_rule.reindex(ix), aomo_obs.reindex(ix))]
        lab_rows += [dict(r, compare="hirosaki_rule_vs_aomori_rule(ref)") for r in
                     agreement_rows(pname, hiro_lab.reindex(ix), aomo_rule.reindex(ix))]
    lab = pd.DataFrame(lab_rows)
    lab.to_csv("research/hirosaki_rule_label_check.csv", index=False)
    print(lab[lab.section.isin(["agreement", "macro_f1_vs_aomori"])].round(3).to_string())

    HIRO = pd.DataFrame({"label": hiro_lab, "temp_max": hiro["temp_max"], "temp_min": hiro["temp_min"]})

    # ---- GSM(青森格子・弘前格子)
    cache = {g: gsm_daily(g) for g in ("aomori", "hirosaki")}
    for g, (_, _, meta) in cache.items():
        print(f"GSM格子({g}): lat, lon, elevation = {meta}")
    gcols = {g: list(cache[g][0].columns) for g in cache}
    a_d, h_d = cache["aomori"][0], cache["hirosaki"][0]
    cmp_rows = []
    for ca, ch in zip(gcols["aomori"], gcols["hirosaki"]):
        j = pd.concat([a_d[ca], h_d[ch]], axis=1).dropna()
        cmp_rows.append(dict(feature=ca.replace("gsm2_", ""), n=len(j), mean_aomori_grid=j[ca].mean(),
                             mean_hirosaki_grid=j[ch].mean(), mean_diff_h_minus_a=(j[ch] - j[ca]).mean(),
                             corr=j[ca].corr(j[ch])))
    for g, (_, _, meta) in cache.items():
        cmp_rows.append(dict(feature=f"grid_{g}", lat=meta[0], lon=meta[1], elevation=meta[2]))
    pd.DataFrame(cmp_rows).to_csv("research/hirosaki_gsm_grid_compare.csv", index=False)
    print(pd.DataFrame(cmp_rows).round(3).to_string())

    df = add_targets(build(1, cache))
    chk = df[df.date.isin(pd.to_datetime(["2021-08-01", "2022-01-15", "2024-12-31", "2025-08-30",
                                          "2025-08-31", "2026-02-01", "2026-08-30"]))]
    chk = chk[["date", "target", "target_h", "today_h", "hiro_temp_max", "target_tmax", "target_tmin",
               "gsm2_temp_max", "hgsm2_temp_max", "gsm2_temp_min", "hgsm2_temp_min", "gsm2_precip_sum", "hgsm2_precip_sum"]]
    chk.to_csv("research/hirosaki_join_check.csv", index=False)
    print(chk.round(2).to_string(index=False))

    # ---- (2) 天気
    w = weather_eval(df, gcols, cache)
    w.to_csv("research/hirosaki_weather_eval.csv", index=False)
    print(w.round(3).to_string())

    # ---- (3) 気温
    t = temp_eval(df, gcols, cache)
    t.to_csv("research/hirosaki_temp_eval.csv", index=False)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
