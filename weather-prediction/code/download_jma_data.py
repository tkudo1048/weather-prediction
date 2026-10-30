"""
気象庁「過去の気象データ・ダウンロード」ページから、日別データを取得するスクリプト。

対象地点:
  - 弘前 (prec_no=31, block_no=0166) : アメダス地点。気温・降水量・湿度・風・日照時間・雪を取得。
  - 青森 (prec_no=31, block_no=47575) : 気象台。天気概況(昼/夜)・気温などを取得。
    弘前は天気カテゴリを観測していないため、天気概況は青森のデータで代用する。

取得期間: 過去5年分（デフォルトでは今日から5年前の月初め〜先月末）

出力:
  data/raw/hirosaki_YYYYMM.csv
  data/raw/aomori_YYYYMM.csv
  各地点を1本のCSVに結合したもの (hirosaki_daily.csv / aomori_daily.csv) も出力する。

注意:
  気象庁のサーバーに配慮し、リクエストの間に待機時間を入れている。
  すでに保存済みの月別CSVは再取得しない(増分取得)。ただし、確定値への更新があり得るため、
  取得期間の最後の数か月(既定は2か月)は毎回取り直す。全期間を取り直すときは --refresh を付ける。
  初回(月別CSVがまだ無いとき)の実行には数分かかる。

使い方:
  python code/download_jma_data.py                       # 増分取得(既定)
  python code/download_jma_data.py --refresh             # 全期間を取り直す
  python code/download_jma_data.py --start 2021-08 --end 2026-08   # 期間を指定する
  python code/download_jma_data.py --refetch-months 0    # 保存済みの月を取り直さない(通信しない)
"""

import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.data.jma.go.jp/obd/stats/etrn/view/{page}.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (research use; JMA daily data download script)"}
SLEEP_SEC = 1.5  # 気象庁サーバーへの配慮のための待機時間

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = {
    "hirosaki": {"page": "daily_a1", "prec_no": 31, "block_no": "0166"},
    "aomori": {"page": "daily_s1", "prec_no": 31, "block_no": "47575"},
}


def month_range(start: date, end: date):
    """start〜end (両端含む) の各月の1日を列挙する。"""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_month(page: str, prec_no: int, block_no: str, year: int, month: int) -> pd.DataFrame | None:
    """指定地点・年月の日別データを1つのDataFrameとして取得する。"""
    params = {
        "prec_no": prec_no,
        "block_no": block_no,
        "year": year,
        "month": month,
        "day": 1,
        "view": "",
    }
    url = BASE_URL.format(page=page)
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    data_rows = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        # 先頭列が日付(1〜31の数字)である行だけをデータ行とみなす
        if cells and cells[0].isdigit():
            data_rows.append(cells)

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows)
    df.insert(0, "year", year)
    df.insert(1, "month", month)
    df = df.rename(columns={0: "day"})
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """月別データの列名を、位置で year, month, day, 1, 2, ... にそろえる。

    取得スクリプトの旧版で保存した月別CSVは、列名がずれている(0,1,day,3,...)。
    新しく取得した月と結合しても列がずれないよう、位置で付け直す。前処理(preprocess.py)も位置で列名を付けている。
    """
    df = df.copy()
    df.columns = ["year", "month", "day"] + [str(i) for i in range(1, df.shape[1] - 2)]
    return df


def download_station(name: str, page: str, prec_no: int, block_no: str, start: date, end: date,
                     out_dir: Path = OUT_DIR, refetch_months: int = 2, refresh: bool = False):
    months = list(month_range(start, end))
    # 取得期間の最後の refetch_months か月は、確定値への更新があり得るので毎回取り直す
    always_fetch = set(months[len(months) - refetch_months:]) if refetch_months > 0 else set()
    monthly_frames = []
    n_fetch = n_cache = 0
    for d in months:
        path = out_dir / f"{name}_{d.year}{d.month:02d}.csv"
        use_cache = path.exists() and not refresh and d not in always_fetch
        if use_cache:
            monthly_frames.append(_normalize(pd.read_csv(path, dtype=str, encoding="utf-8-sig")))
            n_cache += 1
            continue
        print(f"[{name}] {d.year}-{d.month:02d} を取得中...")
        try:
            df = fetch_month(page, prec_no, block_no, d.year, d.month)
        except requests.RequestException as e:
            print(f"  -> 取得失敗: {e}")
            df = None
        n_fetch += 1
        if df is not None:
            df = _normalize(df)
            monthly_frames.append(df)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif path.exists():
            # 取得に失敗しても、保存済みのデータがあればそれを使う
            monthly_frames.append(_normalize(pd.read_csv(path, dtype=str, encoding="utf-8-sig")))
        time.sleep(SLEEP_SEC)
    print(f"[{name}] 取得 {n_fetch}か月 / 保存済みを利用 {n_cache}か月")

    if monthly_frames:
        combined = pd.concat(monthly_frames, ignore_index=True)
        combined.to_csv(out_dir / f"{name}_daily.csv", index=False, encoding="utf-8-sig")
        print(f"[{name}] 結合ファイルを保存しました: {out_dir / f'{name}_daily.csv'} ({len(combined)}行)")
    else:
        print(f"[{name}] データを取得できませんでした。")


def _parse_month(s: str) -> date:
    y, m = s.split("-")
    return date(int(y), int(m), 1)


def main():
    ap = argparse.ArgumentParser(description="気象庁の日別データを取得する(既定は増分取得)")
    ap.add_argument("--start", help="取得開始月 YYYY-MM(既定: 終了月の5年前)")
    ap.add_argument("--end", help="取得終了月 YYYY-MM(既定: 先月)")
    ap.add_argument("--refetch-months", type=int, default=2,
                    help="取得期間の最後のNか月は、保存済みでも取り直す(既定: 2。0なら取り直さない)")
    ap.add_argument("--refresh", action="store_true", help="保存済みの月も含め、全期間を取り直す")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="保存先(既定: data/raw)")
    a = ap.parse_args()

    today = date.today()
    if a.end:
        end = _parse_month(a.end)
    else:
        # 今月はまだ月末まで揃っていない可能性があるため先月末までとする。
        end_year, end_month = today.year, today.month - 1
        if end_month == 0:
            end_year -= 1
            end_month = 12
        end = date(end_year, end_month, 1)
    start = _parse_month(a.start) if a.start else date(end.year - 5, end.month, 1)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"取得期間: {start.year}-{start.month:02d} 〜 {end.year}-{end.month:02d}"
          f"(取り直す月数: {'全部' if a.refresh else a.refetch_months})")

    for name, info in STATIONS.items():
        download_station(name, info["page"], info["prec_no"], info["block_no"], start, end,
                         out_dir=out_dir, refetch_months=a.refetch_months, refresh=a.refresh)


if __name__ == "__main__":
    main()
