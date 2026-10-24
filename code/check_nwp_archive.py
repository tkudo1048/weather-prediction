"""
夜に取得したGSMの予報値(data/raw/nwp/realtime/<対象日>.json、predict_tonight.py が保存)と、
後日 Open-Meteo Previous Runs API から取り直したアーカイブの値を比べる。

目的: 学習に使ったアーカイブの previous_day2 が、「夜の時点で実際に取れた値」と同じかを確かめる。
     (同じなら、学習と本番で特徴量の作り方がずれていないと言える。違えば、評価値は楽観的な可能性がある)

- 後日取得は --fetch を付けて実行したときだけ。保存済みの全対象日を、1回のリクエストでまとめて取得する
  (対象日が今日より前のものだけ)。取得結果は data/raw/nwp/archive_check/ に保存し、
  --fetch なしの実行では、保存済みのうち最新のものを使う(アクセスしない)。
- 比べるもの(対象日ごと・変数ごと、06〜18時 JST):
    毎時値の差(夜の値 − アーカイブの値)の平均絶対値・最大絶対値・一致した時刻数。
    previous_day0 / day1 / day2 のうち、夜に取れたものすべて。
    あわせて、モデルに入れる previous_day2 の11特徴量(集計後)の差。
- 注意: previous_day0(最新の予報)は、夜の時点ではまだ更新途中の回のため、後日の値と違うのが普通。
  一致を確かめたいのは主に previous_day2(と day1)。
- 出力: research/nwp_archive_check.csv(毎時値の比較)、research/nwp_archive_check_features.csv(11特徴量の比較)
- 実行: プロジェクト直下で `.venv/bin/python code/check_nwp_archive.py`(取得するときは `--fetch`)
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nwp_features import (BASE_VARS, INST_HOURS, PROD_LEAD, PROD_PREFIX, REALTIME_DIR, daily_features,
                          hourly_frame, var_name)

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "data" / "raw" / "nwp" / "archive_check"
LEADS = ["previous_day0", "previous_day1", "previous_day2"]
OUT = ROOT / "research" / "nwp_archive_check.csv"
OUT_FEAT = ROOT / "research" / "nwp_archive_check_features.csv"


def load_realtime():
    recs = []
    for p in sorted(REALTIME_DIR.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        h = hourly_frame(r["response"])
        h = h[h["time"].dt.normalize() == pd.Timestamp(r["target_date"])]
        recs.append((date.fromisoformat(r["target_date"]), r["fetched_at"], h))
    return recs


def fetch_archive(start: date, end: date) -> Path:
    """start〜end の previous_day0/1/2 を1回のリクエストで取得して保存する。"""
    import requests
    from nwp_features import HEADERS, LAT, LON, SLEEP_SEC, URL
    import time
    hourly = ",".join(var_name(v, l) for v in BASE_VARS for l in LEADS)
    params = dict(latitude=LAT, longitude=LON, hourly=hourly, models="jma_gsm",
                  start_date=start.isoformat(), end_date=end.isoformat(), timezone="Asia/Tokyo")
    time.sleep(SLEEP_SEC)
    r = requests.get(URL, params=params, headers=HEADERS, timeout=60)
    print(f"GET {r.url} -> {r.status_code}")
    r.raise_for_status()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = ARCHIVE_DIR / f"aomori_jma_gsm_{start}_{end}_fetched{datetime.now():%Y%m%dT%H%M}.json"
    path.write_text(json.dumps(r.json(), ensure_ascii=False), encoding="utf-8")
    print(f"保存: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="アーカイブを1回のリクエストで取得する(ユーザーが実行するときのみ)")
    a = ap.parse_args()
    recs = [r for r in load_realtime() if r[0] < date.today()]
    if not recs:
        print(f"比べられる夜の記録がありません({REALTIME_DIR}、対象日が今日より前のもの)")
        return
    start, end = min(r[0] for r in recs), max(r[0] for r in recs)
    if a.fetch:
        path = fetch_archive(start, end)
    else:
        files = sorted(ARCHIVE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not files:
            print("保存済みのアーカイブがありません。`--fetch` を付けて実行してください(1回だけアクセスします)")
            return
        path = files[-1]
        print(f"保存済みのアーカイブを使います: {path}")
    arc = hourly_frame(json.loads(path.read_text(encoding="utf-8")))

    rows, frows = [], []
    for tgt, fetched_at, night in recs:
        day_arc = arc[arc["time"].dt.normalize() == pd.Timestamp(tgt)]
        if day_arc.empty:
            print(f"{tgt}: アーカイブに無い(取得範囲外)")
            continue
        m = night.merge(day_arc, on="time", suffixes=("_night", "_arc"))
        m = m[m["time"].dt.hour.isin(INST_HOURS)]
        for lead in LEADS:
            for v in BASE_VARS:
                c = var_name(v, lead)
                if f"{c}_night" not in m or f"{c}_arc" not in m:
                    continue
                d = (m[f"{c}_night"] - m[f"{c}_arc"]).dropna()
                rows.append(dict(target_date=tgt, fetched_at=fetched_at, lead=lead, variable=v,
                                 n_hours=len(d), n_equal=int((d.abs() < 1e-9).sum()),
                                 mean_abs_diff=float(d.abs().mean()) if len(d) else np.nan,
                                 max_abs_diff=float(d.abs().max()) if len(d) else np.nan))
        try:
            fn = daily_features(night, PROD_PREFIX, PROD_LEAD).loc[pd.Timestamp(tgt)]
            fa = daily_features(day_arc, PROD_PREFIX, PROD_LEAD).loc[pd.Timestamp(tgt)]
            for k in fn.index:
                frows.append(dict(target_date=tgt, feature=k, night=fn[k], archive=fa[k], diff=fn[k] - fa[k]))
        except KeyError:
            pass

    res = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)
    pd.DataFrame(frows).to_csv(OUT_FEAT, index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    if len(res):
        s = res.groupby("lead").agg(days=("target_date", "nunique"), hours=("n_hours", "sum"),
                                    equal=("n_equal", "sum"), max_abs_diff=("max_abs_diff", "max"))
        print("\nまとめ(lead別): 一致した時刻 / 比べた時刻")
        print(s.to_string())
    print(f"保存: {OUT} / {OUT_FEAT}")


if __name__ == "__main__":
    main()
