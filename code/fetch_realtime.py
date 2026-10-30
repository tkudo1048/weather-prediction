"""
アメダスの10分ごとの観測JSONを取得し、preprocess.py と同じ名前の「日別特徴量」に集計する。

データ元（公式APIではない内部JSON。形式が変わる可能性があるので例外処理とログを入れている）:
  最新時刻: https://www.jma.go.jp/bosai/amedas/data/latest_time.txt
  地点データ: https://www.jma.go.jp/bosai/amedas/data/point/{地点}/{YYYYMMDD}_{HH}.json
    HH は 00,03,...,21 の3時間ブロック。キーは YYYYMMDDHHMMSS、値は [値, 品質]。
  青森=31312（海面気圧）、弘前=31461（その他すべて）。保存期間はおよそ9〜10日。

気象庁の日別値との対応（すべて近似。ずれの実測は research/realtime_feature_check.csv）:
  - 1日の範囲は気象庁と同じく 00:10〜24:00（翌日 00:00 の記録まで）。
  - 平均（気温・湿度・風速・海面気圧）: 毎正時（01時〜24時）の値の平均。気象庁の日平均も毎時値の平均。
    夜23時台の実行では24時がまだ無いので、01時〜23時の23個で計算する。
  - 最高・最低気温: JSON の maxTemp/minTemp（その日の途中までの極値）と10分値の最大/最小の大きい方/小さい方。
  - 最大瞬間風速: JSON の gust（その日の途中までの最大）の最大値。
  - 最大風速: 10分平均風速の最大。最小湿度: 10分値の最小（気象庁は瞬間値ベースなので少し高めに出うる）。
  - 降水量合計: precipitation10m の合計。最大1時間: precipitation1h（10分ずらしの前1時間値）の最大。
  - 日照時間: sun10m（分）の合計 / 60。
  - 欠測・品質不良（品質≠0）は使わず、計算できない特徴量は NaN（モデル側の Imputer が中央値で埋める）。

実行例: PYTHONPATH=code python code/fetch_realtime.py --date 2026-10-02
"""
import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "amedas_10min"
LOG_DIR = ROOT / "logs"
BASE = "https://www.jma.go.jp/bosai/amedas/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal research; weather forecast pipeline)"}
SLEEP_SEC = 1.5
HIROSAKI, AOMORI = "31461", "31312"

# モデルが使う18特徴量（特徴量セット(c)）。順序は train_final.py で保存したものを正とする
FEATURES = [
    "hiro_precip_total", "hiro_precip_max_1h", "hiro_precip_max_10m",
    "hiro_temp_mean", "hiro_temp_max", "hiro_temp_min",
    "hiro_humidity_mean", "hiro_humidity_min",
    "hiro_wind_mean", "hiro_wind_max", "hiro_gust_max",
    "hiro_sunshine", "aomori_pressure_sea", "aomori_pressure_diff",
    "hiro_temp_diff", "hiro_humidity_diff",
    "season_sin", "season_cos",
]
MIN_HOURLY = 20  # 毎時値がこれ未満なら日平均は NaN（夜の実行では23個そろう想定）

log = logging.getLogger("realtime")
_last_request = 0.0


def setup_logging():
    if log.handlers:
        return
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_DIR / "realtime.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def _get(url: str) -> requests.Response:
    """気象庁への配慮: 前回のリクエストから1.5秒以上あける。"""
    global _last_request
    wait = SLEEP_SEC - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(url, headers=HEADERS, timeout=30)
    finally:
        _last_request = time.time()


def latest_time() -> datetime:
    r = _get(f"{BASE}/latest_time.txt")
    r.raise_for_status()
    return datetime.fromisoformat(r.text.strip()).replace(tzinfo=None)  # 日本時間として扱う


def fetch_block(code: str, day: date, hh: int, latest: datetime) -> dict:
    """3時間ブロックのJSONを返す。確定済み（最新時刻より前に終わった）ブロックだけキャッシュする。"""
    start = datetime(day.year, day.month, day.day, hh)
    if start > latest:
        return {}
    path = CACHE / code / f"{day:%Y%m%d}_{hh:02d}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"{BASE}/point/{code}/{day:%Y%m%d}_{hh:02d}.json"
    try:
        r = _get(url)
        if r.status_code == 404:
            log.warning("見つからない（保存期間切れの可能性）: %s", url)
            return {}
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("想定外の形式（dictではない）")
    except (requests.RequestException, ValueError) as e:
        log.error("取得失敗 %s: %s", url, e)
        return {}
    if start + timedelta(hours=3) <= latest:  # ブロックが確定していれば保存
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False))
    return data


def _val(rec: dict, key: str):
    """[値, 品質] から値を取り出す。欠測・品質不良・形式違いは NaN。"""
    v = rec.get(key)
    if isinstance(v, list) and len(v) == 2 and v[0] is not None and v[1] == 0:
        return float(v[0])
    return np.nan


def load_day(code: str, day: date, latest: datetime, cutoff: datetime | None = None) -> pd.DataFrame:
    """day の 00:10〜24:00 の10分記録を DataFrame（index=時刻）で返す。cutoff 以降は捨てる。"""
    blocks = [(day, hh) for hh in range(0, 24, 3)] + [(day + timedelta(days=1), 0)]
    recs = {}
    for d, hh in blocks:
        recs.update(fetch_block(code, d, hh, latest))
    keys = ["temp", "humidity", "wind", "gust", "maxTemp", "minTemp", "precipitation10m",
            "precipitation1h", "sun10m", "normalPressure"]
    rows = []
    for k, rec in recs.items():
        try:
            t = datetime.strptime(k, "%Y%m%d%H%M%S")
        except ValueError:
            log.warning("想定外のキー: %s", k)
            continue
        if isinstance(rec, dict):
            rows.append({"time": t, **{c: _val(rec, c) for c in keys}})
    if not rows:
        return pd.DataFrame(columns=keys)
    df = pd.DataFrame(rows).set_index("time").sort_index()
    lo = datetime(day.year, day.month, day.day, 0, 10)
    hi = lo + timedelta(hours=23, minutes=50)  # 24:00
    if cutoff is not None:
        hi = min(hi, cutoff)
    return df[(df.index >= lo) & (df.index <= hi)]


def _hourly_mean(s: pd.Series) -> float:
    h = s[s.index.minute == 0].dropna()
    return float(h.mean()) if len(h) >= MIN_HOURLY else np.nan


def _agg(s: pd.Series, how: str, min_n: int = 1) -> float:
    s = s.dropna()
    return float(getattr(s, how)()) if len(s) >= min_n else np.nan


def aggregate(hiro: pd.DataFrame, aomo: pd.DataFrame) -> dict:
    """1日分の10分記録を日別特徴量（前日差・季節以外）に集計する。"""
    f = {}
    n10 = 120  # 10分値が1日144個中これ未満なら合計系は NaN（欠けが多いと過小になるため）
    f["hiro_precip_total"] = _agg(hiro["precipitation10m"], "sum", n10)
    f["hiro_precip_max_1h"] = _agg(hiro["precipitation1h"], "max")
    f["hiro_precip_max_10m"] = _agg(hiro["precipitation10m"], "max")
    f["hiro_temp_mean"] = _hourly_mean(hiro["temp"])
    f["hiro_temp_max"] = np.nanmax([_agg(hiro["temp"], "max"), _agg(hiro["maxTemp"], "max")]) \
        if hiro[["temp", "maxTemp"]].notna().any().any() else np.nan
    f["hiro_temp_min"] = np.nanmin([_agg(hiro["temp"], "min"), _agg(hiro["minTemp"], "min")]) \
        if hiro[["temp", "minTemp"]].notna().any().any() else np.nan
    f["hiro_humidity_mean"] = _hourly_mean(hiro["humidity"])
    f["hiro_humidity_min"] = _agg(hiro["humidity"], "min")
    f["hiro_wind_mean"] = _hourly_mean(hiro["wind"])
    f["hiro_wind_max"] = _agg(hiro["wind"], "max")
    f["hiro_gust_max"] = _agg(hiro["gust"], "max")
    f["hiro_sunshine"] = _agg(hiro["sun10m"], "sum", n10) / 60
    f["aomori_pressure_sea"] = _hourly_mean(aomo["normalPressure"]) if len(aomo) else np.nan
    return f


def build_features(day: date, cutoff: datetime | None = None, latest: datetime | None = None):
    """day の特徴量（18個）を作る。前日差のため前日分も集計する。

    戻り値: (特徴量の pd.Series, 使えた観測の最終時刻, 当日分の集計値 dict)
    使えた観測の最終時刻は、弘前・青森の最終観測時刻の早いほう。どちらかが1件も無ければ None。
    """
    setup_logging()
    latest = latest or latest_time()
    log.info("対象日 %s / 最新観測 %s / 打ち切り %s", day, latest, cutoff)
    today_h, today_a = load_day(HIROSAKI, day, latest, cutoff), load_day(AOMORI, day, latest, cutoff)
    prev = day - timedelta(days=1)
    prev_f = aggregate(load_day(HIROSAKI, prev, latest), load_day(AOMORI, prev, latest))
    f = aggregate(today_h, today_a)
    f["aomori_pressure_diff"] = f["aomori_pressure_sea"] - prev_f["aomori_pressure_sea"]
    f["hiro_temp_diff"] = f["hiro_temp_mean"] - prev_f["hiro_temp_mean"]
    f["hiro_humidity_diff"] = f["hiro_humidity_mean"] - prev_f["hiro_humidity_mean"]
    doy = pd.Timestamp(day).dayofyear
    f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # last_obs: 弘前と青森それぞれの「当日の最終観測時刻」のうち早いほう（遅れている地点に合わせる）。
    # 片方の地点のデータが全く取れない場合: その地点の特徴量は欠測（NaN）になり、last_obs は None とする。
    # 取れた側の時刻を採用すると記録条件を満たしたように見えてしまうため、安全側（記録しない側）に倒す。
    obs_times = []
    for name, df in (("弘前", today_h), ("青森", today_a)):
        t = df.dropna(how="all").index.max() if len(df) else None
        if t is None or pd.isna(t):
            log.warning("%s の当日観測が1件も取れません（last_obs=None として記録しない側に倒す）", name)
            obs_times = None
            break
        obs_times.append(t)
    last_obs = min(obs_times) if obs_times else None
    feats = pd.Series({k: f[k] for k in FEATURES}, dtype=float)
    miss = feats.index[feats.isna()].tolist()
    if miss:
        log.warning("欠測の特徴量 %d個: %s", len(miss), miss)
    return feats, last_obs, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(), help="対象日 YYYY-MM-DD（既定: 今日）")
    a = ap.parse_args()
    feats, last_obs, _ = build_features(date.fromisoformat(a.date))
    print(f"使えた観測の最終時刻: {last_obs}")
    print(feats.round(3).to_string())


if __name__ == "__main__":
    main()
