"""
弘前の近くの格子点の数値予報(気象庁GSM、previous_day2)を、Open-Meteo Previous Runs API から取得して保存する。

- データ: Open-Meteo Previous Runs API(無料版、非商用利用・CC BY 4.0。出典: Open-Meteo.com)、models=jma_gsm
- 地点: 弘前(40.60, 140.46)を指定。実際に使われた格子点は、応答の latitude / longitude に入る。
- 変数: 青森格子(code/fetch_nwp.py)と同じ7変数の *_previous_day2(約48時間前に出た予報)だけ。毎時値、日本時間。
- 期間: 2021-08-01〜2026-08-31(青森格子の保存済みデータと同じ)。1回のリクエストで取得する。
- 保存先: data/raw/nwp/hirosaki_jma_gsm_day2_<start>_<end>.json(生のJSONをそのまま保存)。
  保存済みのファイルがあれば再取得しない(アクセスは必要最小限)。
- 実行: プロジェクト直下で `.venv/bin/python code/fetch_nwp_hirosaki.py`
"""
import json
import time
from pathlib import Path

import requests

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "nwp"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.60, 140.46
START, END = "2021-08-01", "2026-08-31"
BASE_VARS = ["temperature_2m", "precipitation", "cloud_cover", "relative_humidity_2m",
             "pressure_msl", "wind_speed_10m", "snowfall"]
LEAD = "previous_day2"
HEADERS = {"User-Agent": "weather-forecast-study/1.0 (personal non-commercial research)"}
SLEEP_SEC = 1.5


def out_path(start=START, end=END):
    return OUT_DIR / f"hirosaki_jma_gsm_day2_{start}_{end}.json"


def fetch(start=START, end=END):
    path = out_path(start, end)
    if path.exists():
        print(f"保存済みのため再取得しません: {path}")
        return path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = ",".join(f"{v}_{LEAD}" for v in BASE_VARS)
    params = dict(latitude=LAT, longitude=LON, hourly=hourly, models="jma_gsm",
                  start_date=start, end_date=end, timezone="Asia/Tokyo")
    time.sleep(SLEEP_SEC)
    r = requests.get(URL, params=params, headers=HEADERS, timeout=120)
    print(f"GET {r.url} -> {r.status_code}")
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"保存: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"格子点: latitude={data.get('latitude')} longitude={data.get('longitude')} elevation={data.get('elevation')}")
    return path


if __name__ == "__main__":
    fetch()
