"""
予測の答え合わせ: logs/predictions.csv の各対象日について、気象庁の日別データ（青森の天気概況・昼）が
取れた日だけ preprocess.py の分類ルールで実際の天気（晴/曇/雨/雪）を付け、logs/scores.csv に保存する。
Accuracy（正解率）と Macro F1（4クラスのF1の平均。少ないクラスも同じ重みで見る指標）を表示する。

- 天気概況が取れない日・分類できない概況の日はスキップする。
- 対象日が過ぎた後に作った予測（run_at が対象日の0時以降。--cutoff での再現実行など）は採点しない。
- cutoff 列が空でない行（--cutoff での再現実行）は group=cutoff_replay として本番に混ぜない。
- provisional=1 や欠測ありの行は「本番」とは別に集計する（scores.csv には group 列で区別して残す）。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/score_predictions.py`
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from jma_daily import load_recent
from preprocess import to_label

ROOT = Path(__file__).resolve().parent.parent


def main():
    path = ROOT / "logs" / "predictions.csv"
    if not path.exists():
        print("logs/predictions.csv がありません")
        return
    pred = pd.read_csv(path)
    pred["target_date"] = pd.to_datetime(pred["target_date"])
    pred["run_at"] = pd.to_datetime(pred["run_at"])
    late = pred["run_at"] >= pred["target_date"]
    if late.any():
        print(f"対象日の0時以降に作った予測 {int(late.sum())}行は採点しません")
    pred = pred[~late]
    past = pred[pred["target_date"].dt.date < date.today()]
    if past.empty:
        print("答え合わせできる（対象日が過ぎた）予測はまだありません")
        return
    aomo = load_recent("aomori", past.target_date.min().date(), past.target_date.max().date())
    if aomo.empty:
        print("気象庁の日別データが取れませんでした")
        return
    actual_text = aomo["weather_day"].reindex(past["target_date"]).values
    past = past.assign(actual_text=actual_text, actual=[to_label(t) for t in actual_text])
    prov = past["provisional"].fillna(0).astype(int) if "provisional" in past else 0
    nmiss = past["n_missing_features"].fillna(0).astype(int)
    cut = past["cutoff"].fillna("").astype(str).str.strip() if "cutoff" in past else pd.Series("", index=past.index)
    past["group"] = np.select([cut != "", (prov == 1) | (nmiss > 0)],
                              ["cutoff_replay", "provisional_or_missing"], "main")
    scored = past.dropna(subset=["actual"])
    skipped = len(past) - len(scored)
    scored.to_csv(ROOT / "logs" / "scores.csv", index=False, encoding="utf-8")
    if scored.empty:
        print(f"実際の天気が付けられた日がありません（スキップ {skipped}日）")
        return
    print(scored[["target_date", "group", "pred", "actual", "actual_text"]].to_string(index=False))
    print(f"スキップ {skipped}日（実際の天気が付けられない）")
    for g, label in [("main", "本番"), ("provisional_or_missing", "暫定・欠測あり（別集計）"),
                     ("cutoff_replay", "cutoff再現（別集計）")]:
        s = scored[scored.group == g]
        if s.empty:
            continue
        acc = accuracy_score(s.actual, s.pred)
        f1 = f1_score(s.actual, s.pred, average="macro", zero_division=0)  # 登場したクラスのみで平均（学習時の評価と同じ）
        print(f"{label} {len(s)}日: Accuracy {acc:.3f} / Macro F1 {f1:.3f}")
        if len(s) < 30:
            print("  ※ 日数が少ないため数値は大きくぶれます（参考値）")

if __name__ == "__main__":
    main()
