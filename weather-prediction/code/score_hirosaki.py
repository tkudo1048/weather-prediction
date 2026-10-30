"""
弘前の予測の答え合わせ: logs/predictions_hirosaki.csv の対象日が過ぎた行を、気象庁の日別データ(弘前アメダス)の
最高・最低気温と、同じルールで作った「ルール天気」と照合し、logs/scores_hirosaki.csv に保存する。

- 日別データは code/jma_daily.py の load_recent で取得(data/raw/recent/ にキャッシュ。必要な月だけ、
  1回の実行で月ごとに1回まで。待機1.5秒)。
- ルール天気: 降雪量>=2cm → 雪、降水量>=5mm → 雨、日照時間 ÷ その月の日照95%点 >= 0.6 → 晴、それ以外 → 曇。
  しきい値と月別95%点は、学習時に models/hirosaki_weather.joblib に保存した値を使う(学習と同じルール)。
  日照が欠測で雪・雨にも当たらない日は、天気の照合から外す。
- 気温: MAE(誤差の大きさの平均。1.0なら平均1℃外れる)と、誤差の平均(予測 − 実測。マイナスなら低めに予測しがち)。
  天気: 一致した日数。
- 対象日の0時以降に作った予測は採点しない。cutoff 列ありは cutoff_replay、provisional=1 か欠測ありは
  provisional_or_missing として、本番(main)と別に集計する(score_predictions.py と同じ分け方)。
- 30日未満は参考値(日数が少なく、数値が大きくぶれる)。
- 注意: ルール天気は観測値から作った天気で、実際に観測された天気そのものではない。
- 実行: プロジェクト直下で `.venv/bin/python code/score_hirosaki.py`
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from jma_daily import load_recent

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "logs" / "predictions_hirosaki.csv"
OUT = ROOT / "logs" / "scores_hirosaki.csv"
WEATHER_MODEL = ROOT / "models" / "hirosaki_weather.joblib"


def rule_label(obs: pd.DataFrame, q95: dict, rule: dict) -> pd.Series:
    """学習(code/hirosaki_weather_temp.py の rule_label)と同じ規則。q95 は bundle の月別日照95%点。"""
    snow = obs["snowfall"].fillna(0)
    prec = obs["precip_total"].fillna(0)
    ratio = obs["sunshine"] / obs.index.month.map(q95).to_numpy(dtype=float)
    lab = pd.Series(np.where(ratio >= rule["sun_ratio"], "晴", "曇"), index=obs.index, dtype=object)
    lab[ratio.isna()] = np.nan
    lab[prec >= rule["precip_mm"]] = "雨"
    lab[snow >= rule["snow_cm"]] = "雪"
    return lab


def main():
    if not PRED.exists():
        print("logs/predictions_hirosaki.csv がありません")
        return
    pred = pd.read_csv(PRED)
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred["run_at"] = pd.to_datetime(pred["run_at"])
    late = pred["run_at"] >= pred["target_date"]
    if late.any():
        print(f"対象日の0時以降に作った予測 {int(late.sum())}行は採点しません")
    pred = pred[~late]
    past = pred[pred["target_date"].dt.date < date.today()].copy()
    if past.empty:
        print("答え合わせできる（対象日が過ぎた）予測はまだありません")
        return
    hiro = load_recent("hirosaki", past.target_date.min().date(), past.target_date.max().date())
    if hiro.empty:
        print("気象庁の日別データ(弘前)が取れませんでした")
        return
    b = joblib.load(WEATHER_MODEL)
    lab = rule_label(hiro, b["sun_q95_by_month"], b["rule"])
    idx = past["target_date"]
    past["actual_temp_max"] = hiro["temp_max"].reindex(idx).values
    past["actual_temp_min"] = hiro["temp_min"].reindex(idx).values
    past["actual_weather_rule"] = lab.reindex(idx).values
    past["err_max"] = past["temp_max_pred"] - past["actual_temp_max"]
    past["err_min"] = past["temp_min_pred"] - past["actual_temp_min"]
    prov = past["provisional"].fillna(0).astype(int)
    nmiss = past["n_missing_features"].fillna(0).astype(int)
    cut = past["cutoff"].fillna("").astype(str).str.strip()
    past["group"] = np.select([cut != "", (prov == 1) | (nmiss > 0)],
                              ["cutoff_replay", "provisional_or_missing"], "main")
    past.to_csv(OUT, index=False, encoding="utf-8")
    cols = ["target_date", "group", "temp_max_pred", "actual_temp_max", "temp_min_pred", "actual_temp_min",
            "pred_weather_rule", "actual_weather_rule"]
    show = past[cols].copy()
    show["target_date"] = show["target_date"].dt.date
    print(show.round(1).to_string(index=False))
    print("※ 天気はルール天気(降雪・降水量・日照から作った天気)。観測された天気そのものではない")
    for g, label in [("main", "本番"), ("provisional_or_missing", "暫定・欠測あり（別集計）"),
                     ("cutoff_replay", "cutoff再現（別集計）")]:
        s = past[past.group == g]
        if s.empty:
            continue
        print(f"{label} {len(s)}日:")
        for k, name in (("max", "最高気温"), ("min", "最低気温")):
            e = s[f"err_{k}"].dropna()
            if len(e):
                print(f"  {name}: MAE {e.abs().mean():.2f}℃ / 誤差の平均(予測−実測) {e.mean():+.2f}℃({len(e)}日)")
            else:
                print(f"  {name}: 実測が取れた日がありません")
        w = s.dropna(subset=["actual_weather_rule"])
        print(f"  天気(ルール): 一致 {int((w.pred_weather_rule == w.actual_weather_rule).sum())}/{len(w)}日"
              + (f"(ルール天気が作れない日 {len(s) - len(w)})" if len(s) > len(w) else ""))
        if len(s) < 30:
            print("  ※ 日数が少ないため数値は大きくぶれます（参考値）")


if __name__ == "__main__":
    main()
