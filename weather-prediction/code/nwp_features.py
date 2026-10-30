"""
数値予報(気象庁GSM)の毎時予報値を、日別の特徴量に集計する共通処理と、夜の実行用の取得処理。

学習(train_final.py、検証の nwp_features_eval.py)と夜の予測(predict_tonight.py)は、
どちらもこのファイルの daily_features() で集計する(集計方法を必ず一致させるため)。

集計(予報の対象日ごと、06〜18時 JST):
- 瞬間値(気温・雲量・湿度・気圧・風速)は 06:00〜18:00 の13時刻。気温は平均・最高・最低、他は平均。
- 積算値(降水量・降雪量。Open-Meteo の毎時値は「直前1時間の合計」)は 07:00〜18:00 の12時刻。
  降水量は合計・1時間最大・0.1mm以上の時間数、降雪量は合計。
- 有効な時刻が足りない日(瞬間値10未満 / 積算値9未満)は NaN(学習時はモデルの Imputer が中央値で埋める)。
- 風速は km/h → m/s に換算(観測と単位をそろえる)。
- 降水量の負の値(GSMに最小 -0.1mm 程度が含まれる)は、学習・予測ともにクリップしない。
  レビュー済みの評価(research/model-comparison.md の GSM の節)と同じ集計にそろえるため。

夜の取得(fetch_tonight):
- Open-Meteo Previous Runs API(無料版、非商用・CC BY 4.0。出典: Open-Meteo.com)、models=jma_gsm、
  青森(40.82, 140.77)、timezone=Asia/Tokyo。予報の対象日1日分を1回のリクエストで取得する。
- previous_day0(最新の予報。変数名に接尾辞なし)/ previous_day1 / previous_day2 を取得し、
  モデルに入れるのは previous_day2 だけ。day0/day1 は、後日アーカイブと突き合わせる検証の記録用。
- URL は環境変数 WEATHER_NWP_URL で差し替えられる(取得失敗時の動作確認用)。

弘前格子の夜の取得(fetch_tonight_hirosaki。弘前の最高・最低気温とルール天気のモデル用):
- 弘前(40.60, 140.46)を指定する(学習に使った code/fetch_nwp_hirosaki.py と同じ)。応答の格子点が
  学習時の (40.5, 140.5) と違えば NWPError(格子が変わったら学習と別の値になるため)。
- 集計は同じ daily_features、接頭辞は hgsm2(学習 code/train_hirosaki.py と同じ)。
- URL は環境変数 WEATHER_NWP_URL_HIROSAKI(なければ WEATHER_NWP_URL)で差し替えられる。
- 生値は data/raw/nwp/realtime_hirosaki/<対象日>.json に保存する。
"""
import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
REALTIME_DIR = ROOT / "data" / "raw" / "nwp" / "realtime"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.82, 140.77
BASE_VARS = ["temperature_2m", "precipitation", "cloud_cover", "relative_humidity_2m",
             "pressure_msl", "wind_speed_10m", "snowfall"]
HEADERS = {"User-Agent": "weather-forecast-study/1.0 (personal non-commercial research)"}
SLEEP_SEC = 1.5
TIMEOUT_SEC = 30

INST = {"temperature_2m": "temp", "cloud_cover": "cloud", "relative_humidity_2m": "rh",
        "pressure_msl": "pressure", "wind_speed_10m": "wind"}
ACCUM = {"precipitation": "precip", "snowfall": "snow"}
INST_HOURS, ACCUM_HOURS = range(6, 19), range(7, 19)
MIN_INST, MIN_ACCUM = 10, 9

# 本番で使う版: previous_day2(約48時間前に出た予報。夜の時点で理屈の上では公開済み(未検証。check_nwp_archive.pyで確かめる))
PROD_PREFIX, PROD_LEAD = "gsm2", "previous_day2"
NWP_VERSION = "gsm_previous_day2"

# 弘前格子(弘前の気温・ルール天気のモデル用)
HIRO_LAT, HIRO_LON = 40.60, 140.46        # 指定する地点(fetch_nwp_hirosaki.py と同じ)
HIRO_GRID = (40.5, 140.5)                 # 学習データの応答に入っていた格子点
HIRO_PREFIX = "hgsm2"
HIRO_REALTIME_DIR = ROOT / "data" / "raw" / "nwp" / "realtime_hirosaki"
HIRO_URL_ENV = "WEATHER_NWP_URL_HIROSAKI"


def var_name(v, lead):
    """previous_day0 は接尾辞なしの変数名(最新の予報)。"""
    return v if lead == "previous_day0" else f"{v}_{lead}"


def hourly_frame(data: dict) -> pd.DataFrame:
    h = pd.DataFrame(data["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    return h


def _agg(h, cols, hours, minimum, funcs):
    sub = h[h["time"].dt.hour.isin(hours)].copy()
    sub["day"] = sub["time"].dt.normalize()
    g = sub.groupby("day")
    out = {}
    for src, name in cols:
        cnt = g[src].count()
        for fname, f in funcs(name).items():
            s = f(g[src])
            out[fname] = s.where(cnt >= minimum)
    return pd.DataFrame(out)


def daily_features(h, prefix, lead):
    """予報の対象日ごと(index=対象日)に 06〜18時 JST の値を集計する。"""
    inst_cols = [(var_name(v, lead), n) for v, n in INST.items()]
    acc_cols = [(var_name(v, lead), n) for v, n in ACCUM.items()]

    def inst_funcs(n):
        fs = {f"{prefix}_{n}_mean": lambda g: g.mean()}
        if n == "temp":
            fs.update({f"{prefix}_temp_max": lambda g: g.max(), f"{prefix}_temp_min": lambda g: g.min()})
        return fs

    def acc_funcs(n):
        fs = {f"{prefix}_{n}_sum": lambda g: g.sum(min_count=1)}
        if n == "precip":
            fs.update({f"{prefix}_precip_max_1h": lambda g: g.max(),
                       f"{prefix}_precip_hours": lambda g: g.apply(lambda s: (s >= 0.1).sum())})
        return fs

    a = _agg(h, inst_cols, INST_HOURS, MIN_INST, inst_funcs)
    b = _agg(h, acc_cols, ACCUM_HOURS, MIN_ACCUM, acc_funcs)
    out = a.join(b, how="outer")
    out[f"{prefix}_wind_mean"] = out[f"{prefix}_wind_mean"] / 3.6  # km/h → m/s(観測と単位をそろえる)
    return out


def prod_feature_names(prefix=PROD_PREFIX):
    """本番モデルが使うGSMの11特徴量の名前(daily_features の列順)。弘前格子は prefix=HIRO_PREFIX。"""
    t = pd.Timestamp("2000-01-01")
    times = pd.date_range(t, periods=24, freq="h")
    dummy = pd.DataFrame({"time": times, **{var_name(v, PROD_LEAD): 0.0 for v in BASE_VARS}})
    return list(daily_features(dummy, prefix, PROD_LEAD).columns)


class NWPError(Exception):
    """夜の取得の失敗、または翌日分の欠損。"""


def fetch_day(target: date, leads=("previous_day0", "previous_day1", "previous_day2"),
              url: str | None = None, timeout: float = TIMEOUT_SEC,
              lat: float = LAT, lon: float = LON) -> dict:
    """予報の対象日 target の1日分(毎時)を1回のリクエストで取得し、生のJSON(dict)を返す。"""
    url = url or os.environ.get("WEATHER_NWP_URL") or URL
    hourly = ",".join(var_name(v, l) for v in BASE_VARS for l in leads)
    params = dict(latitude=lat, longitude=lon, hourly=hourly, models="jma_gsm",
                  start_date=target.isoformat(), end_date=target.isoformat(), timezone="Asia/Tokyo")
    time.sleep(SLEEP_SEC)
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        raise NWPError(f"GSMの取得に失敗: {type(e).__name__}: {e}") from e
    if not isinstance(data, dict) or "hourly" not in data:
        raise NWPError("GSMの応答が想定外の形式(hourly がない)")
    return data


def prod_features_from(data: dict, target: date, prefix: str = PROD_PREFIX) -> pd.Series:
    """取得したJSONから、本番モデル用のGSM 11特徴量(target の日)を作る。
    06〜18時の previous_day2 に1つでも欠損があれば NWPError(本番では18特徴量モデルに切り替える)。"""
    try:
        h = hourly_frame(data)
    except (KeyError, ValueError, TypeError) as e:
        raise NWPError(f"GSMの応答を読めない: {e}") from e
    cols = [var_name(v, PROD_LEAD) for v in BASE_VARS]
    missing_cols = [c for c in cols if c not in h.columns]
    if missing_cols:
        raise NWPError(f"GSMの応答に変数がない: {missing_cols}")
    day = h[h["time"].dt.normalize() == pd.Timestamp(target)]
    need = day[day["time"].dt.hour.isin(INST_HOURS)]
    if len(need) < len(INST_HOURS):
        raise NWPError(f"{target} 06〜18時の時刻が {len(need)}/{len(INST_HOURS)} しかない")
    n_na = int(need[cols].isna().sum().sum())
    if n_na:
        raise NWPError(f"{target} 06〜18時の previous_day2 に欠損 {n_na}個")
    dly = daily_features(day, prefix, PROD_LEAD)
    if pd.Timestamp(target) not in dly.index:
        raise NWPError(f"{target} の集計値が作れない")
    s = dly.loc[pd.Timestamp(target)].astype(float)
    if s.isna().any():
        raise NWPError(f"集計値に欠損: {s.index[s.isna()].tolist()}")
    return s


def fetch_tonight(target: date, fetch=fetch_day):
    """夜の実行用。(GSM 11特徴量の Series, 生のJSON) を返す。失敗時は NWPError。
    fetch は差し替え可能(ネットワークを使わない失敗の確認用)。"""
    data = fetch(target)
    return prod_features_from(data, target), data


def fetch_day_hirosaki(target: date) -> dict:
    """弘前格子の対象日1日分を1回のリクエストで取得する(URLは WEATHER_NWP_URL_HIROSAKI で差し替え可)。"""
    return fetch_day(target, url=os.environ.get(HIRO_URL_ENV) or None, lat=HIRO_LAT, lon=HIRO_LON)


def check_grid(data: dict, grid=HIRO_GRID, tol: float = 0.05):
    """応答の格子点が学習時と同じかを確かめる。違えば NWPError。"""
    try:
        lat, lon = float(data["latitude"]), float(data["longitude"])
    except (KeyError, TypeError, ValueError) as e:
        raise NWPError(f"GSMの応答に格子点がない: {e}") from e
    if abs(lat - grid[0]) > tol or abs(lon - grid[1]) > tol:
        raise NWPError(f"GSMの格子点が学習時と違う: 応答 ({lat}, {lon}) / 学習 {grid}")


def fetch_tonight_hirosaki(target: date, fetch=fetch_day_hirosaki):
    """夜の実行用(弘前格子)。(GSM 11特徴量(hgsm2_*)の Series, 生のJSON) を返す。失敗時は NWPError。"""
    data = fetch(target)
    check_grid(data)
    return prod_features_from(data, target, prefix=HIRO_PREFIX), data


def save_realtime(target: date, data: dict, run_at: datetime, out_dir: Path = REALTIME_DIR,
                  source: str = "open-meteo previous-runs-api jma_gsm") -> Path:
    """夜に取得した生値を data/raw/nwp/realtime/<対象日>.json に保存する(同じ対象日は上書き)。
    後日アーカイブと突き合わせるため(code/check_nwp_archive.py)。弘前格子は out_dir=HIRO_REALTIME_DIR。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target.isoformat()}.json"
    rec = {"target_date": target.isoformat(), "fetched_at": run_at.isoformat(timespec="seconds"),
           "source": source, "response": data}
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return path
