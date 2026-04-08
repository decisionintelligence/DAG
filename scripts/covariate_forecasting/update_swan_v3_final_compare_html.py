import csv
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "result"
FINAL_SH = ROOT / "scripts" / "covariate_forecasting" / "SWAN_V3_FINAL.sh"
OUT_DIR = RESULT_DIR / "Z_showResult" / "SWAN_V3_FINAL"
OUT_CSV = OUT_DIR / "compare_DAG_vs_SWAN_V3_FINAL_current_best.csv"
OUT_HTML = OUT_DIR / "compare_DAG_vs_SWAN_V3_FINAL_current_best.html"

CANDIDATE_MODELS = ["SWAN_V3", "SWAN_V3_TUNE", "SWAN_V3_TUNE_R2", "SWAN_V3_TUNE_R3", "SWAN_V3_TUNE_R4"]
METRIC_NAME = "mse_norm"
HORIZON_PAT = re.compile(r'"horizon"\s*:\s*(\d+)')


def parse_horizon(strategy_args: str):
    m = HORIZON_PAT.search(strategy_args or "")
    return int(m.group(1)) if m else None


def parse_final_script_params():
    if not FINAL_SH.exists():
        return {}

    mapping = {}
    lines = FINAL_SH.read_text(encoding="utf-8").splitlines()

    for line in lines:
        if "run_benchmark.py" not in line:
            continue

        data_match = re.search(r"--data-name-list\s+([^\s]+)", line)
        strategy_match = re.search(r"--strategy-args\s+'([^']+)'", line)
        params_match = re.search(r"--model-hyper-params\s+'([^']+)'", line)

        if not data_match or not strategy_match or not params_match:
            continue

        dataset_token = data_match.group(1).strip().strip('"').strip("'")
        dataset = Path(dataset_token).stem

        horizon = parse_horizon(strategy_match.group(1))
        if horizon is None:
            continue

        params_json = params_match.group(1)
        mapping[(dataset, horizon)] = params_json

    return mapping


def scan_best_candidate_records():
    best = {}

    for dataset_dir in sorted(RESULT_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        if dataset == "Z_showResult":
            continue

        for model in CANDIDATE_MODELS:
            model_dir = dataset_dir / model
            if not model_dir.is_dir():
                continue

            if model in {"SWAN_V3_TUNE_R2", "SWAN_V3_TUNE_R3", "SWAN_V3_TUNE_R4"}:
                files = sorted(model_dir.glob("**/test_report*.csv"), key=lambda p: p.stat().st_mtime)
            else:
                files = sorted(model_dir.glob("test_report*.csv"), key=lambda p: p.stat().st_mtime)

            for fp in files:
                try:
                    with fp.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        headers = reader.fieldnames or []
                        if "strategy_args" not in headers or "metric_name" not in headers:
                            continue

                        value_col = headers[-1]
                        if value_col in {"strategy_args", "metric_name"} or ";" not in value_col:
                            continue

                        prefix, _ = value_col.split(";", 1)
                        if not prefix.startswith("SWANV3"):
                            continue

                        mtime = fp.stat().st_mtime
                        for row in reader:
                            if row.get("metric_name") != METRIC_NAME:
                                continue

                            horizon = parse_horizon(row.get("strategy_args", ""))
                            if horizon is None:
                                continue

                            try:
                                mse = float(row.get(value_col, ""))
                            except Exception:
                                continue

                            key = (dataset, horizon)
                            prev = best.get(key)
                            rec = {
                                "dataset": dataset,
                                "horizon": horizon,
                                "best_model": model,
                                "best_mse": mse,
                                "source_file": str(fp).replace("\\", "/"),
                                "mtime": mtime,
                            }
                            if prev is None or mse < prev["best_mse"] or (mse == prev["best_mse"] and mtime > prev["mtime"]):
                                best[key] = rec
                except Exception:
                    continue

    return best


def scan_dag_records():
    dag = {}

    for dataset_dir in sorted(RESULT_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        if dataset == "Z_showResult":
            continue

        model_dir = dataset_dir / "DAG"
        if not model_dir.is_dir():
            continue

        files = sorted(model_dir.glob("test_report*.csv"), key=lambda p: p.stat().st_mtime)
        for fp in files:
            try:
                with fp.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
                    if "strategy_args" not in headers or "metric_name" not in headers:
                        continue

                    value_col = headers[-1]
                    if value_col in {"strategy_args", "metric_name"}:
                        continue

                    mtime = fp.stat().st_mtime
                    for row in reader:
                        if row.get("metric_name") != METRIC_NAME:
                            continue

                        horizon = parse_horizon(row.get("strategy_args", ""))
                        if horizon is None:
                            continue

                        try:
                            mse = float(row.get(value_col, ""))
                        except Exception:
                            continue

                        key = (dataset, horizon)
                        prev = dag.get(key)
                        rec = {
                            "dag_mse": mse,
                            "source_file": str(fp).replace("\\", "/"),
                            "mtime": mtime,
                        }
                        if prev is None or mtime >= prev["mtime"]:
                            dag[key] = rec
            except Exception:
                continue

    return dag


def build_rows(best_map, dag_map, final_params_map):
    keys = sorted(set(best_map.keys()) & set(dag_map.keys()), key=lambda x: (x[0], x[1]))

    rows = []
    for dataset, horizon in keys:
        best_rec = best_map[(dataset, horizon)]
        dag_rec = dag_map[(dataset, horizon)]

        dag_mse = dag_rec["dag_mse"]
        best_mse = best_rec["best_mse"]
        delta = best_mse - dag_mse

        if best_mse < dag_mse:
            winner = "SWAN_V3_FINAL_BEST"
        elif best_mse > dag_mse:
            winner = "DAG"
        else:
            winner = "TIE"

        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "DAG_mse": dag_mse,
                "SWAN_V3_FINAL_current_best_mse": best_mse,
                "delta_best_minus_DAG": delta,
                "winner_by_mse": winner,
                "best_from_model": best_rec["best_model"],
                "best_source_file": best_rec["source_file"],
                "DAG_source_file": dag_rec["source_file"],
                "final_params_json": final_params_map.get((dataset, horizon), ""),
            }
        )

    return rows


def write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "horizon",
        "DAG_mse",
        "SWAN_V3_FINAL_current_best_mse",
        "delta_best_minus_DAG",
        "winner_by_mse",
        "best_from_model",
        "best_source_file",
        "DAG_source_file",
        "final_params_json",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **r,
                    "DAG_mse": f"{r['DAG_mse']:.10f}",
                    "SWAN_V3_FINAL_current_best_mse": f"{r['SWAN_V3_FINAL_current_best_mse']:.10f}",
                    "delta_best_minus_DAG": f"{r['delta_best_minus_DAG']:.10f}",
                }
            )


def build_html(rows):
    header = [
        "dataset",
        "horizon",
        "DAG_mse",
        "SWAN_V3_FINAL_current_best_mse",
        "delta_best_minus_DAG",
        "winner_by_mse",
        "best_from_model",
        "best_source_file",
        "final_params_json",
    ]

    th = "".join(f"<th>{h}</th>" for h in header)
    body_rows = []

    for r in rows:
        dag_cls = "win" if r["winner_by_mse"] == "DAG" else ""
        best_cls = "win" if r["winner_by_mse"] == "SWAN_V3_FINAL_BEST" else ""
        tie_cls = "tie" if r["winner_by_mse"] == "TIE" else ""

        tds = [
            f"<td>{html.escape(str(r['dataset']))}</td>",
            f"<td>{r['horizon']}</td>",
            f"<td class='{dag_cls}'>{r['DAG_mse']:.10f}</td>",
            f"<td class='{best_cls}'>{r['SWAN_V3_FINAL_current_best_mse']:.10f}</td>",
            f"<td class='{best_cls if r['delta_best_minus_DAG'] < 0 else dag_cls}'>{r['delta_best_minus_DAG']:.10f}</td>",
            f"<td class='{tie_cls}'>{html.escape(r['winner_by_mse'])}</td>",
            f"<td>{html.escape(r['best_from_model'])}</td>",
            f"<td class='path'>{html.escape(r['best_source_file'])}</td>",
            f"<td class='params'>{html.escape(r['final_params_json'])}</td>",
        ]
        body_rows.append("<tr>" + "".join(tds) + "</tr>")

    html_doc = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <title>DAG vs SWAN_V3_FINAL 当前最优对比</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    h1 {{ margin: 0 0 6px 0; }}
    .note {{ color: #555; margin: 0 0 16px 0; }}
    .summary {{ margin: 0 0 18px 0; font-size: 13px; color: #333; }}
    section {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; text-align: center; vertical-align: top; }}
    th {{ background: #f7f7f7; position: sticky; top: 0; z-index: 1; }}
    .win {{ color: #d60000; font-weight: 700; }}
    .tie {{ color: #555; font-weight: 700; }}
    .path {{ text-align: left; word-break: break-all; max-width: 260px; }}
    .params {{ text-align: left; word-break: break-all; max-width: 900px; font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>DAG vs SWAN_V3_FINAL 当前最优结果对比</h1>
  <div class="note">当前最优结果来源：SWAN_V3 / SWAN_V3_TUNE / SWAN_V3_TUNE_R2 / SWAN_V3_TUNE_R3 / SWAN_V3_TUNE_R4 中每个 dataset-horizon 的最小 mse_norm；最后一列为 SWAN_V3_FINAL.sh 对应参数配置。</div>
  <div class=\"summary\">总行数：{len(rows)}（仅展示 DAG 与当前最优都有结果的公共点）。</div>
  <section>
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
  </section>
</body>
</html>
"""

    OUT_HTML.write_text(html_doc, encoding="utf-8")


def main():
    final_params_map = parse_final_script_params()
    best_map = scan_best_candidate_records()
    dag_map = scan_dag_records()
    rows = build_rows(best_map, dag_map, final_params_map)

    if not rows:
        raise RuntimeError("No common dataset-horizon points found between DAG and current best SWAN_V3 records.")

    write_csv(rows)
    build_html(rows)

    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved HTML: {OUT_HTML}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
