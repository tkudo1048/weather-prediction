"""
数値予報(気象庁GSM)の「過去に出た予報値」を Open-Meteo Previous Runs API から取得して保存する。

- データ: Open-Meteo Previous Runs API(無料版、非商用利用・CC BY 4.0。出典: Open-Meteo.com)
- 地点: 青森(40.82, 140.77)。
- 変数: 各変数の *_previous_day1(対象時刻の約24時間前に出た予報)と
        *_previous_day2(約48時間前に出た予報)。毎時値、日本時間。
- 保存先: data/raw/nwp/aomori_jma_gsm_<start>_<end>.json(生のJSONをそのまま保存)。
  保存済みのファイルがあれば再取得しない(アクセスは必要最小限)。
- 実行: プロジェクト直下で `.venv/bin/python code/fetch_nwp.py`
"""
import json
import time
from pathlib import Path

import requests

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nwp"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.82, 140.77
START, END = "2021-08-01", "2026-08-31"
BASE_VARS = ["temperature_2m", "precipitation", "cloud_cover", "relative_humidity_2m",
             "pressure_msl", "wind_speed_10m", "snowfall"]
LEADS = ["previous_day1", "previous_day2"]
HEADERS = {"User-Agent": "weather-forecast-study/1.0 (personal non-commercial research)"}
SLEEP_SEC = 1.5


def out_path(start=START, end=END):
    return OUT_DIR / f"aomori_jma_gsm_{start}_{end}.json"


def fetch(start=START, end=END):
    path = out_path(start, end)
    if path.exists():
        print(f"保存済みのため再取得しません: {path}")
        return path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = ",".join(f"{v}_{l}" for v in BASE_VARS for l in LEADS)
    params = dict(latitude=LAT, longitude=LON, hourly=hourly, models="jma_gsm",
                  start_date=start, end_date=end, timezone="Asia/Tokyo")
    time.sleep(SLEEP_SEC)
    r = requests.get(URL, params=params, headers=HEADERS, timeout=120)
    print(f"GET {r.url} -> {r.status_code}")
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"保存: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


if __name__ == "__main__":
    fetch()
