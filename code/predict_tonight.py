"""
今夜の予測: 今日（または --date の日）の10分データから特徴量を作り、models/model.joblib で
翌日の天気（晴/曇/雨/雪）の確率を出して logs/predictions.csv に1行追記する。

記録する条件（本番扱い）。どれか1つでも満たさなければ「暫定」として表示だけし、記録しない。
- 観測日の最終観測時刻 last_obs が 23:00 以降（観測が1日の終わり近くまで届いている）。
  last_obs は弘前と青森の最終観測時刻の早いほう。どちらかの地点の当日データが1件も取れない場合は
  None となり、記録しない（その地点の特徴量も欠測になる）。
  最高・最低気温や最小湿度、最大風速などは途中までの観測でも欠測にならず静かにずれるため、
  欠測数ではなく「23:00以降まで観測が届いているか」で判定する。
- 欠測した特徴量が0個。
- 実行日（date.today()）が観測日と同じ（0時過ぎの実行や過去日の --date 指定は暫定扱い）。
  ただし --cutoff で打ち切り時刻を指定した再現実行は、この条件を問わない。
  --cutoff 付きで記録した行は note に「cutoff再現 HH:MM」を必ず入れ、cutoff 列も埋める。
  score_predictions.py はこの行を本番と別の group（cutoff_replay）で集計する。
- 同じ対象日（翌日）の本番行がすでにあれば追記しない。
- --force: 上記の条件を満たさなくても記録する（既存行は置き換え）。条件を満たさない行には provisional=1 を付ける。
- GSM(数値予報): models/model.joblib が GSMあり(nwp_version="gsm_previous_day2")のときは、
  実行時に Open-Meteo Previous Runs API から翌日 06〜18時の予報を1回のリクエストで取得し、
  previous_day2 を集計(code/nwp_features.py。学習と共通)してモデルに入れる。
  取得失敗、または翌日分に欠損があるときは models/model_18feat.joblib(18特徴量)で予測し、
  note に「GSM取得失敗のため18特徴量モデル」と記録する。provisional の扱いは上の3条件のまま
  (GSMの失敗だけでは暫定にしない)。使ったモデルは model 列に入る。
  取得は「記録済みなら追記しない」の判定の後なので、23:20 で記録済みなら 23:40 は青森のGSMを取得しない。
- 夜に取得したGSMの生値(previous_day0/1/2)は data/raw/nwp/realtime/<対象日>.json に保存する
  (後日アーカイブと突き合わせる検証用。code/check_nwp_archive.py)。
  --cutoff 指定・過去日の再現では保存しない(その時点の夜の値ではないため)。
- 弘前(2026-10-23〜): 青森の予測・記録のあとに、弘前の翌日の最高・最低気温とルール天気を予測し、
  別ファイル logs/predictions_hirosaki.csv に記録する(code/hirosaki_tonight.py)。記録の規則は青森と同じ
  (青森が暫定・記録しない条件では弘前も記録しない。--force / --cutoff も同じ扱い)。
  青森が記録済み(「記録済みのため追記しません」)の日は、logs/predictions_hirosaki.csv に対象日の行が
  ないときだけ弘前の予測を試みる(23:20 で弘前だけ失敗した日の 23:40 の再試行)。このとき青森は一切触らず、
  青森の GSM も取得しない。弘前の記録条件(3条件・provisional)は同じ。弘前の行もあれば何もしない(GSM取得なし)。
  弘前の失敗(GSM取得失敗・欠損・例外)は弘前だけ記録しない。青森の記録には影響させない。
- 実行: プロジェクト直下で `PYTHONPATH=code python code/predict_tonight.py`
  夜の実行の再現: `... predict_tonight.py --date 2026-10-02 --cutoff 23:30`
"""
import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fetch_realtime import build_features, log
from nwp_features import fetch_tonight, save_realtime

ROOT = Path(__file__).resolve().parent.parent
LOG_CSV = ROOT / "logs" / "predictions.csv"
MODEL = ROOT / "models" / "model.joblib"
MODEL_18 = ROOT / "models" / "model_18feat.joblib"
NOTE_NWP_FAIL = "GSM取得失敗のため18特徴量モデル"
MIN_LAST_OBS = time(23, 0)


def run_hirosaki(**kw):
    """弘前の予測(code/hirosaki_tonight.py)。どんな失敗も青森の記録に影響させないよう、ここで完全に隔離する。
    import もこの中で行う(弘前側のファイルやモデルの問題で、青森の実行が止まらないように)。"""
    try:
        import hirosaki_tonight
        hirosaki_tonight.run(**kw)
    except Exception as e:
        log.exception("弘前の予測で例外(弘前は記録しない。青森の記録には影響なし): %s", e)
        print(f"弘前: 記録しない(例外: {type(e).__name__}: {e})")


def record_reasons(obs_day: date, last_obs, n_missing: int, cutoff, run_at: datetime) -> list:
    """記録条件(3条件)を満たさない理由の一覧。空なら本番扱い。青森と弘前で共通。"""
    reasons = []
    if last_obs is None or pd.Timestamp(last_obs) < pd.Timestamp(datetime.combine(obs_day, MIN_LAST_OBS)):
        reasons.append(f"最終観測 {last_obs} が {obs_day} 23:00 より前")
    if n_missing > 0:
        reasons.append(f"欠測特徴量 {n_missing}個")
    if cutoff is None and run_at.date() != obs_day:
        reasons.append(f"実行日 {run_at.date()} が観測日 {obs_day} と違う（日付またぎ・過去日）")
    return reasons


def hirosaki_has_row(target: date) -> bool:
    """logs/predictions_hirosaki.csv に対象日の行があるか。読めないときは「ある」とみなし、何もしない側に倒す。"""
    try:
        import hirosaki_tonight
        path = hirosaki_tonight.LOG_CSV
        if not path.exists():
            return False
        h = pd.read_csv(path, dtype=str)
        return bool(len(h)) and bool((h["target_date"] == target.isoformat()).any())
    except Exception as e:
        log.error("弘前の記録を読めないため、弘前の再試行はしません: %s", e)
        return True


def retry_hirosaki_only(obs_day: date, target: date, cutoff, cutoff_str, run_at: datetime):
    """青森が記録済みで、弘前の行がない日の再試行。青森のログ・GSMには触らない。"""
    print(f"弘前: 対象日 {target} の行がないため、弘前だけ予測を試みます")
    log.info("青森は記録済み、弘前の行なし → 弘前だけ再試行: target=%s", target)
    try:
        feats, last_obs, _ = build_features(obs_day, cutoff=cutoff)
    except Exception as e:
        log.exception("弘前の再試行: 観測の取得で例外(弘前は記録しない): %s", e)
        print(f"弘前: 記録しない(観測の取得で例外: {type(e).__name__}: {e})")
        return
    n_missing = int(feats.isna().sum())
    reasons = record_reasons(obs_day, last_obs, n_missing, cutoff, run_at)
    if reasons:
        print("弘前: 暫定・記録しない（理由: " + " / ".join(reasons) + "）")
    replay = cutoff is not None or run_at.date() != obs_day
    run_hirosaki(obs_day=obs_day, target=target, obs_feats=feats.copy(), last_obs=last_obs, n_missing=n_missing,
                 reasons=reasons, run_at=run_at, cutoff_str=cutoff_str, force=False, replay=replay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(), help="観測日 YYYY-MM-DD（既定: 今日）")
    ap.add_argument("--cutoff", default=None, help="観測の打ち切り時刻 HH:MM（夜の実行の再現用）")
    ap.add_argument("--force", action="store_true", help="条件を満たさなくても記録する（provisional=1）。同じ対象日の既存行は置き換える")
    a = ap.parse_args()
    run_at = datetime.now()
    obs_day = date.fromisoformat(a.date)
    target = obs_day + timedelta(days=1)
    cutoff = None
    if a.cutoff:
        hh, mm = map(int, a.cutoff.split(":"))
        cutoff = datetime.combine(obs_day, time(0)) + timedelta(hours=hh, minutes=mm)

    old = pd.read_csv(LOG_CSV, dtype=str) if LOG_CSV.exists() else pd.DataFrame()
    if len(old) and (old["target_date"] == target.isoformat()).any() and not a.force:
        print(f"対象日 {target} の予測は記録済みのため追記しません（--force で上書き）")
        if hirosaki_has_row(target):
            print(f"弘前: 対象日 {target} の予測も記録済みのため何もしません")
            return
        retry_hirosaki_only(obs_day, target, cutoff, a.cutoff, run_at)
        return

    bundle = joblib.load(MODEL)
    feats, last_obs, _ = build_features(obs_day, cutoff=cutoff)
    n_missing = int(feats.isna().sum())  # 18特徴量(観測)の欠測数。記録条件はこれで判定する
    obs_feats = feats.copy()  # 弘前の予測用(18特徴量。青森のGSMを足す前)
    replay = cutoff is not None or run_at.date() != obs_day
    nwp_notes = []
    if bundle.get("nwp_version"):
        try:
            gsm, raw = fetch_tonight(target)
            if not replay:
                try:  # 生値の保存だけが失敗しても、取得できたGSMは予測に使う
                    path = save_realtime(target, raw, run_at)
                    log.info("GSMの生値を保存: %s", path)
                except Exception as e:
                    log.error("GSMの生値の保存に失敗: %s", e)
            feats = pd.concat([feats, gsm])
            log.info("GSM(%s)を取得: %s", bundle["nwp_version"], gsm.round(2).to_dict())
            if replay:
                nwp_notes.append("GSMは実行時に取得した値(再現)")
        except Exception as e:  # NWPError のほか想定外の例外でも、本番を止めず18特徴量モデルに切り替える
            log.error("%s → 18特徴量モデルで予測します", e)
            print(f"{e} → 18特徴量モデルで予測します")
            bundle = joblib.load(MODEL_18)
            nwp_notes.append(NOTE_NWP_FAIL)
    model_name = bundle.get("model_name") or ("lr_18feat" if not bundle.get("nwp_version") else bundle["nwp_version"])
    try:
        X = pd.DataFrame([feats[bundle["features"]].values], columns=bundle["features"])
        proba = bundle["pipeline"].predict_proba(X)[0]
    except Exception as e:
        if not bundle.get("nwp_version"):
            raise
        log.error("29特徴量モデルでの予測に失敗: %s → 18特徴量モデルで予測します", e)
        print(f"29特徴量モデルでの予測に失敗: {e} → 18特徴量モデルで予測します")
        bundle = joblib.load(MODEL_18)
        nwp_notes.append(NOTE_NWP_FAIL)
        model_name = bundle.get("model_name") or "lr_18feat"
        X = pd.DataFrame([feats[bundle["features"]].values], columns=bundle["features"])
        proba = bundle["pipeline"].predict_proba(X)[0]
    classes = bundle["classes"]
    pred = classes[int(np.argmax(proba))]

    reasons = record_reasons(obs_day, last_obs, n_missing, cutoff, run_at)
    provisional = bool(reasons)
    hiro_kw = dict(obs_day=obs_day, target=target, obs_feats=obs_feats, last_obs=last_obs, n_missing=n_missing,
                   reasons=list(reasons), run_at=run_at, cutoff_str=a.cutoff, force=a.force, replay=replay)

    print(f"観測日 {obs_day}（最終観測 {last_obs}）→ 対象日 {target} の予測")
    for c, p in zip(classes, proba):
        print(f"  {c}: {p:.1%}")
    print(f"予測クラス: {pred} / 欠測特徴量: {n_missing}個 / モデル: {model_name}({len(bundle['features'])}特徴量)")

    if provisional and not a.force:
        print("暫定・記録しない（理由: " + " / ".join(reasons) + "）")
        log.info("暫定のため記録せず: target=%s pred=%s 理由=%s", target, pred, reasons)
        run_hirosaki(**hiro_kw)  # 暫定のときは弘前も記録しない(GSMも取得しない)
        return

    row = {"run_at": run_at.isoformat(timespec="seconds"), "target_date": target.isoformat(),
           **{f"p_{c}": round(float(p), 4) for c, p in zip(classes, proba)},
           "pred": pred, "last_obs_time": str(last_obs) if last_obs is not None else "",
           "n_missing_features": n_missing, "provisional": int(provisional),
           "cutoff": a.cutoff or "",
           "note": " / ".join(([f"cutoff再現 {a.cutoff}"] if a.cutoff else []) + nwp_notes + reasons),
           "model": model_name}
    if len(old) and a.force:
        old = old[old["target_date"] != target.isoformat()]
    # 既存の列順を保ち、新しい列(model など)は末尾に足す。既存行の新しい列は空欄
    new = pd.DataFrame([row]).astype(str)
    cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
    out = pd.concat([old, new], ignore_index=True).reindex(columns=cols)
    LOG_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(LOG_CSV, index=False, encoding="utf-8")
    log.info("予測を記録: %s", row)
    print("記録しました" + ("（provisional=1: " + " / ".join(reasons) + "）" if provisional else ""))
    run_hirosaki(**hiro_kw)  # 青森の記録が終わってから


if __name__ == "__main__":
    main()
