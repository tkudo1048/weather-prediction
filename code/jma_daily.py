"""
最近の日別データ（気象庁「過去の気象データ」）を月単位で取得・キャッシュし、preprocess.py と同じ規則で数値化する。

- 検証（check_realtime_features.py）と答え合わせ（score_predictions.py / score_hirosaki.py）で使う。
- 取得結果は data/raw/recent/{hirosaki|aomori}_YYYYMM.csv に保存し、必要な日がそろっていれば再取得しない。
- 新しさの判定(cache_status):
  気象庁の月の表は月末まで行があり、まだ観測していない日は空欄・`///` になる。そのため「表の最後の行」ではなく、
  「観測値(空欄・`///`・`--`・`×` 以外)が入っている最後の行の日」(have_last)で判定する。
  - 当月など、必要な最終日(昨日まで。月末を超えない)に have_last が届いていなければ取り直す。
  - 過去の月で、月末日に値がないときも取り直す。ただし、月が終わってから2日以上たって取得したファイル
    (その時点で載っていた分が全部)なら、取り直さない(月末の欠測などで毎回取りに行かないため)。
  - 同じ日にすでに取得したファイル(更新日時=今日)は、その日のうちは取り直さない(気象庁への負荷を抑える)。
  - 取得は1回の実行で月ごとに1回まで。待機1.5秒、User-Agent は download_jma_data.fetch_month の設定。
"""
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from download_jma_data import STATIONS, SLEEP_SEC, fetch_month
from preprocess import AOMORI_COLS, HIROSAKI_COLS, parse_daily

CACHE = Path(__file__).resolve().parent.parent / "data" / "raw" / "recent"
COLS = {"hirosaki": HIROSAKI_COLS, "aomori": AOMORI_COLS}
NO_VALUE = {"", "///", "--", "×", "nan"}
PAST_MONTH_GRACE_DAYS = 2  # 月末からこの日数以上あとに取得したファイルは、過去の月として確定扱い


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _month_end(y: int, m: int) -> date:
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return nxt - timedelta(days=1)


def last_observed_day(raw: pd.DataFrame | None, y: int, m: int) -> date | None:
    """観測値が1つでも入っている最後の行の日。値のある行がなければ None。
    値の判定: 空欄・`///`(観測なし)・`--`(該当現象なし)・`×`(欠測)以外。`)` や `]` 付きの値は値として数える。"""
    if raw is None or not len(raw):
        return None
    vals = raw.iloc[:, 3:].fillna("").astype(str).apply(lambda s: s.str.strip())
    has = ~vals.isin(NO_VALUE)
    days = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
    ok = has.any(axis=1) & days.notna()
    if not ok.any():
        return None
    return date(y, m, int(days[ok].max()))


def cache_status(raw: pd.DataFrame | None, path: Path, y: int, m: int, need_last: date,
                 today: date | None = None) -> tuple[bool, str]:
    """(取り直すか, 理由)を返す。ネットワークにはアクセスしない(確認用に単独でも呼べる)。
    need_last: 全体で必要な最終日(昨日まで)。"""
    today = today or date.today()
    month_start, month_end = date(y, m, 1), _month_end(y, m)
    if need_last < month_start:
        return False, "この月はまだ必要ない"
    fetched = datetime.fromtimestamp(path.stat().st_mtime).date() if path.exists() else None
    if raw is None:
        return True, "キャッシュなし"
    if fetched == today:
        return False, "今日すでに取得済み(同じ日は取り直さない)"
    have_last = last_observed_day(raw, y, m)
    need_in_month = min(need_last, month_end)
    if have_last is not None and have_last >= need_in_month:
        return False, f"必要な日({need_in_month})まで値あり(最後の値 {have_last})"
    if need_in_month >= month_end and fetched is not None \
            and fetched >= month_end + timedelta(days=PAST_MONTH_GRACE_DAYS):
        return False, f"過去の月で、月末後({fetched})に取得済み(最後の値 {have_last})"
    return True, f"値のある最後の日 {have_last} < 必要な日 {need_in_month}(取得日 {fetched})"


def load_recent(name: str, start: date, end: date, allow_fetch: bool = True, dry_run: bool = False) -> pd.DataFrame:
    """start〜end を含む月の日別データを返す（index=日付）。取れない月は飛ばす。
    dry_run=True: 取得はせず、月ごとの判定(取り直すか・理由)を表示するだけ(キャッシュの内容で返す)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    info = STATIONS[name]
    # 必要な最終日（昨日まで。今日の日別値はまだ確定していない）
    need_last = min(end, date.today() - timedelta(days=1))
    frames = []
    for y, m in _months(start, end):
        path = CACHE / f"{name}_{y}{m:02d}.csv"
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str) if path.exists() else None
        stale, why = cache_status(raw, path, y, m, need_last)
        if dry_run:
            print(f"[{name}] {y}-{m:02d}: {'取り直す' if stale else '取り直さない'}({why})")
        elif stale and allow_fetch:
            try:
                df = fetch_month(info["page"], info["prec_no"], info["block_no"], y, m)
                time.sleep(SLEEP_SEC)
                if df is not None:
                    df.to_csv(path, index=False, encoding="utf-8-sig")
                    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
                    print(f"[{name}] {y}-{m:02d} を取り直しました(理由: {why}。最後の値 {last_observed_day(raw, y, m)})")
            except requests.RequestException as e:
                print(f"[{name}] {y}-{m:02d} の日別データ取得に失敗: {e}")
        if raw is not None and len(raw):
            frames.append(parse_daily(raw, COLS[name]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
