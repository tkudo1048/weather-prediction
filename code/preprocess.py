"""
data/raw の日別データを整形し、「今日の観測値 → 明日の天気（晴/曇/雨/雪）」の学習用データを作る。

出力: data/processed/dataset.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# 気象庁の表の並び（先頭の「日」を除く）
AOMORI_COLS = [
    "pressure_local", "pressure_sea",
    "precip_total", "precip_max_1h", "precip_max_10m",
    "temp_mean", "temp_max", "temp_min",
    "humidity_mean", "humidity_min",
    "wind_mean", "wind_max", "wind_max_dir", "gust_max", "gust_max_dir",
    "sunshine", "snowfall", "snow_depth",
    "weather_day", "weather_night",
]
HIROSAKI_COLS = [
    "precip_total", "precip_max_1h", "precip_max_10m",
    "temp_mean", "temp_max", "temp_min",
    "humidity_mean", "humidity_min",
    "wind_mean", "wind_max", "wind_max_dir", "gust_max", "gust_max_dir", "wind_mode_dir",
    "sunshine", "snowfall", "snow_depth",
]
TEXT_COLS = {"wind_max_dir", "gust_max_dir", "wind_mode_dir", "weather_day", "weather_night"}
ZERO_IF_NONE = {"precip_total", "precip_max_1h", "precip_max_10m", "snowfall", "snow_depth"}

LABELS = ["晴", "曇", "雨", "雪"]
_LABEL_RULES = [
    (r"^(快晴|晴)", "晴"),
    (r"^(薄曇|曇)", "曇"),
    (r"^(大雪|雪|みぞれ|ふぶき|暴風雪)", "雪"),
    (r"^(大雨|雨|霧雨|雷雨|暴風雨|雷)", "雨"),
]


def load(name: str, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{name}_daily.csv", encoding="utf-8-sig", dtype=str)
    return parse_daily(df, cols)


def parse_daily(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """取得した日別表（文字列のまま）を数値化する。月別CSVにも使えるよう load から分離。"""
    df = df.copy()
    # 列名は取得スクリプトの旧バグでずれているため、位置で付け直す
    df.columns = ["year", "month", "day"] + cols
    df["date"] = pd.to_datetime(df[["year", "month", "day"]].astype(int))
    df = df.drop(columns=["year", "month", "day"]).set_index("date").sort_index()

    for c in cols:
        if c in TEXT_COLS:
            df[c] = df[c].replace({"--": np.nan, "×": np.nan, "///": np.nan})
            continue
        s = df[c].str.strip()
        none = s == "--"
        # ")" 準正常値は採用、"]" 資料不足・"×" 欠測・"///" などは欠損
        s = s.str.replace(r"\s*\)$", "", regex=True)
        s = pd.to_numeric(s, errors="coerce")
        s[none] = 0.0 if c in ZERO_IF_NONE else np.nan
        df[c] = s
    return df


def to_label(text) -> str | float:
    if not isinstance(text, str):
        return np.nan
    for pattern, label in _LABEL_RULES:
        if re.match(pattern, text):
            return label
    return np.nan


def main():
    hiro = load("hirosaki", HIROSAKI_COLS)
    aomo = load("aomori", AOMORI_COLS)

    feats = hiro.drop(columns=["wind_max_dir", "gust_max_dir", "wind_mode_dir"]).add_prefix("hiro_")
    feats["aomori_pressure_sea"] = aomo["pressure_sea"]
    feats["aomori_pressure_diff"] = aomo["pressure_sea"].diff()
    feats["hiro_temp_diff"] = hiro["temp_mean"].diff()
    feats["hiro_humidity_diff"] = hiro["humidity_mean"].diff()

    today = aomo["weather_day"].map(to_label)
    for lab in LABELS:
        feats[f"today_{lab}"] = (today == lab).astype(float).where(today.notna())

    doy = feats.index.dayofyear
    feats["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["season_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 明日の天気を今日の行にぶら下げる（暦上の翌日と一致する場合のみ）
    full_idx = pd.date_range(feats.index.min(), feats.index.max(), freq="D")
    tomorrow = today.reindex(full_idx).shift(-1)
    feats = feats.reindex(full_idx)
    feats["target"] = tomorrow
    feats["target_text"] = aomo["weather_day"].reindex(full_idx).shift(-1)
    feats.index.name = "date"

    unmapped = aomo["weather_day"][aomo["weather_day"].notna() & today.isna()]
    missing_days = len(full_idx) - len(hiro)

    dataset = feats.dropna(subset=["target"])
    OUT.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT / "dataset.csv", encoding="utf-8-sig")

    print(f"期間: {full_idx.min().date()} 〜 {full_idx.max().date()}  抜けている日: {missing_days}")
    print(f"出力行数: {len(dataset)}  特徴量数: {dataset.shape[1] - 2}")
    print("ラベル分布:\n" + dataset["target"].value_counts().to_string())
    print("欠損の多い特徴量:\n" + dataset.isna().sum().sort_values(ascending=False).head(6).to_string())
    if len(unmapped):
        print(f"分類できなかった天気概況 {len(unmapped)}件:\n" + unmapped.value_counts().to_string())


if __name__ == "__main__":
    main()
