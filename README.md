# weather-forecast

気象庁の観測データ(青森・弘前)から、翌日の天気と降雪量を予測する機械学習の実験です。2つのプロジェクトに分けています。

- `weather-prediction/`: 翌日の天気(晴/曇/雨/雪)と、弘前の最高・最低気温の予測。夜の自動実行もここ
- `snowfall-prediction/`: 翌日の降雪量の予測と、降雪についての調査。日別データと前処理は `weather-prediction/` のものを使う

共通のもの(ひとつ上の階層)は、仮想環境 `.venv`、`requirements.txt`、`research/`(検証の記録)です。
使い方は、それぞれのフォルダの README を見てください。
