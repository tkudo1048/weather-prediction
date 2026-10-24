"""
夜の予測(predict_tonight.py)の追加分: 弘前の翌日の最高・最低気温とルール天気(晴/曇/雨/雪)を予測し、
logs/predictions_hirosaki.csv に1行追記する。

- predict_tonight.py が、青森の予測・記録を終えたあとに run() を呼ぶ。青森が記録済みで早期終了する日は、
  このファイルの LOG_CSV に対象日の行がないときだけ呼ばれる(弘前だけの再試行。青森のGSMは取得しない)。
  ここでの失敗は、青森の記録に影響させない(predict_tonight.py 側でも例外を捕まえてログに出す)。
- モデル: models/hirosaki_temp_max.joblib / hirosaki_temp_min.joblib(Ridge)、hirosaki_weather.joblib
  (ロジスティック回帰)。学習は code/train_hirosaki.py。特徴量は夜の18個 + GSM day2(弘前格子)の11個。
- GSM: 弘前格子を1回のリクエストで取得(nwp_features.fetch_tonight_hirosaki。待機1.5秒・User-Agent・タイムアウト30秒)。
  生値は data/raw/nwp/realtime_hirosaki/<対象日>.json に保存(--cutoff・過去日の再現では保存しない)。
- 記録の規則は青森と同じ:
    青森が「暫定・記録しない」になる条件(最終観測23:00前 / 欠測あり / 実行日≠観測日)では、弘前も記録しない
    (GSMも取得しない)。--force では provisional=1 と理由を付けて記録し、同じ対象日の既存行は置き換える。
    --cutoff 付きは note に「cutoff再現 HH:MM」、cutoff 列も埋める。
    --force なしで同じ対象日の行がすでにあれば追記しない。
- 取得失敗・欠損・格子点の違い・例外のときは、弘前の予測は記録しない(18特徴量へのフォールバックはしない)。
  理由は画面とログ(logs/realtime.log)に出す。
"""
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fetch_realtime import FEATURES, log
from nwp_features import HIRO_GRID, HIRO_PREFIX, HIRO_REALTIME_DIR, fetch_tonight_hirosaki, save_realtime

ROOT = Path(__file__).resolve().parent.parent
LOG_CSV = ROOT / "logs" / "predictions_hirosaki.csv"
MODELS = {"max": ROOT / "models" / "hirosaki_temp_max.joblib",
          "min": ROOT / "models" / "hirosaki_temp_min.joblib",
          "weather": ROOT / "models" / "hirosaki_weather.joblib"}
CLASSES = ["晴", "曇", "雨", "雪"]
COLUMNS = ["run_at", "target_date", "temp_max_pred", "temp_min_pred", *[f"p_{c}" for c in CLASSES],
           "pred_weather_rule", "last_obs_time", "n_missing_features", "provisional", "cutoff", "note"]


def _skip(msg, level="error"):
    print(f"弘前: 記録しない({msg})")
    getattr(log, level)("弘前: 記録せず: %s", msg)


def run(obs_day: date, target: date, obs_feats: pd.Series, last_obs, n_missing: int, reasons: list,
        run_at: datetime, cutoff_str: str | None, force: bool, replay: bool, fetch=fetch_tonight_hirosaki):
    """弘前の予測と記録。戻り値は記録した行(dict)。記録しなかったときは None。"""
    provisional = bool(reasons)
    if provisional and not force:
        _skip("青森と同じく暫定のため予測しません", level="info")
        return None
    old = pd.read_csv(LOG_CSV, dtype=str) if LOG_CSV.exists() else pd.DataFrame(columns=COLUMNS)
    if len(old) and (old["target_date"] == target.isoformat()).any() and not force:
        print(f"弘前: 対象日 {target} の予測は記録済みのため追記しません")
        return None

    try:
        bundles = {k: joblib.load(p) for k, p in MODELS.items()}
    except Exception as e:
        _skip(f"モデルを読めない: {type(e).__name__}: {e}")
        return None
    for k, b in bundles.items():
        if tuple(b.get("grid") or ()) != HIRO_GRID or b.get("nwp_prefix") != HIRO_PREFIX:
            _skip(f"モデル {k} の格子点・接頭辞が想定と違う: {b.get('grid')} {b.get('nwp_prefix')}")
            return None

    try:
        gsm, raw = fetch(target)
    except Exception as e:  # NWPError のほか想定外の例外も。弘前は記録しない
        _skip(f"GSM(弘前格子)の取得に失敗: {type(e).__name__}: {e}")
        return None
    log.info("弘前: GSM(弘前格子)を取得: %s", gsm.round(2).to_dict())
    if not replay:
        try:
            path = save_realtime(target, raw, run_at, out_dir=HIRO_REALTIME_DIR,
                                 source=f"open-meteo previous-runs-api jma_gsm hirosaki grid {HIRO_GRID}")
            log.info("弘前: GSMの生値を保存: %s", path)
        except Exception as e:  # 保存だけの失敗では予測を止めない
            log.error("弘前: GSMの生値の保存に失敗: %s", e)

    feats = pd.concat([obs_feats[FEATURES], gsm])
    out = {}
    for k, b in bundles.items():
        X = pd.DataFrame([feats[b["features"]].astype(float).values], columns=b["features"])
        if X.iloc[0, len(FEATURES):].isna().any():
            _skip(f"GSMの特徴量に欠損: {X.columns[X.isna().iloc[0]].tolist()}")
            return None
        if k == "weather":
            proba = b["pipeline"].predict_proba(X)[0]
            out["proba"] = dict(zip(b["classes"], proba))
        else:
            out[k] = float(b["pipeline"].predict(X)[0])
    proba = out["proba"]
    pred = max(proba, key=proba.get)
    if not all(np.isfinite([out["max"], out["min"], *proba.values()])):
        _skip("予測値が数値でない")
        return None

    probs = " ".join(f"{c} {proba.get(c, 0):.0%}" for c in CLASSES)
    print(f"弘前:最高 {out['max']:.1f}℃ / 最低 {out['min']:.1f}℃ / 天気(ルール){pred} (確率 {probs})")
    notes = ([f"cutoff再現 {cutoff_str}"] if cutoff_str else []) + (["GSMは実行時に取得した値(再現)"] if replay else [])
    row = {"run_at": run_at.isoformat(timespec="seconds"), "target_date": target.isoformat(),
           "temp_max_pred": round(out["max"], 2), "temp_min_pred": round(out["min"], 2),
           **{f"p_{c}": round(float(proba.get(c, 0.0)), 4) for c in CLASSES},
           "pred_weather_rule": pred, "last_obs_time": str(last_obs) if last_obs is not None else "",
           "n_missing_features": n_missing, "provisional": int(provisional), "cutoff": cutoff_str or "",
           "note": " / ".join(notes + list(reasons))}
    if len(old) and force:
        old = old[old["target_date"] != target.isoformat()]
    new = pd.DataFrame([row]).astype(str)
    cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    res = pd.concat([old, new], ignore_index=True).reindex(columns=cols)
    LOG_CSV.parent.mkdir(exist_ok=True)
    res.to_csv(LOG_CSV, index=False, encoding="utf-8")
    log.info("弘前: 予測を記録: %s", row)
    print("弘前: 記録しました" + ("（provisional=1）" if provisional else ""))
    return row
