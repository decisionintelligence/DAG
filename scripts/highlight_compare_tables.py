import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
ALL_CSV = RESULT_DIR / "compare_DAG_vs_SWAN_V3_TUNE_all.csv"
WIN_CSV = RESULT_DIR / "compare_DAG_vs_SWAN_V3_TUNE_wins.csv"
OUT_HTML = RESULT_DIR / "compare_DAG_vs_SWAN_V3_TUNE_highlight.html"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def td(text: str, cls: str = "") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return f"<td{class_attr}>{text}</td>"


def build_table(rows, title: str) -> str:
    headers = [
        "dataset",
        "horizon",
        "DAG_mse",
        "SWAN_V3_TUNE_mse",
        "DAG_mae",
        "SWAN_V3_TUNE_mae",
        "DAG_rmse",
        "SWAN_V3_TUNE_rmse",
        "winner_by_mse",
    ]

    head_html = "".join([f"<th>{h}</th>" for h in headers])
    body = []

    for r in rows:
        winner = r.get("winner_by_mse", "")
        dag_cls = "win" if winner == "DAG" else ""
        v3_cls = "win" if winner == "SWAN_V3_TUNE" else ""

        tds = [
            td(r.get("dataset", "")),
            td(r.get("horizon", "")),
            td(r.get("DAG_mse", ""), dag_cls),
            td(r.get("SWAN_V3_TUNE_mse", ""), v3_cls),
            td(r.get("DAG_mae", ""), dag_cls),
            td(r.get("SWAN_V3_TUNE_mae", ""), v3_cls),
            td(r.get("DAG_rmse", ""), dag_cls),
            td(r.get("SWAN_V3_TUNE_rmse", ""), v3_cls),
            td(winner, "winner"),
        ]
        body.append("<tr>" + "".join(tds) + "</tr>")

    return f"""
    <section>
      <h2>{title}</h2>
      <table>
        <thead><tr>{head_html}</tr></thead>
        <tbody>
          {''.join(body)}
        </tbody>
      </table>
    </section>
    """


def main():
    all_rows = read_csv(ALL_CSV)
    win_rows = read_csv(WIN_CSV)

    html = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <title>DAG vs SWAN_V3_TUNE 对比高亮</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ color: #666; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; table-layout: fixed; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; font-size: 12px; }}
    th {{ background: #f7f7f7; position: sticky; top: 0; }}
    .win {{ color: #d60000; font-weight: 700; }}
    .winner {{ font-weight: 700; }}
    section {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>DAG vs SWAN_V3_TUNE 对比表（胜出结果标红）</h1>
  <div class=\"note\">规则：按 winner_by_mse 判定胜出模型；胜出模型对应行的 mse/mae/rmse 三个值标红。</div>
  {build_table(all_rows, '表1：所有数据集对比表')}
  {build_table(win_rows, '表2：SWAN_V3_TUNE 胜出数据集对比表')}
</body>
</html>
"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
