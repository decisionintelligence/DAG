import csv
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
MODELS = ["DAG", "SWAN", "SWAN_V3"]
METRICS = ["mse_norm", "mae_norm", "rmse_norm"]


def parse_horizon(strategy_args: str):
    m = re.search(r'"horizon"\s*:\s*(\d+)', strategy_args)
    return int(m.group(1)) if m else None


def load_latest_records():
    latest = {}
    for dataset_dir in RESULT_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for model in MODELS:
            model_dir = dataset_dir / model
            if not model_dir.exists():
                continue
            files = sorted(model_dir.glob("test_report*.csv"), key=lambda p: p.stat().st_mtime)
            for f in files:
                mtime = f.stat().st_mtime
                try:
                    with f.open("r", encoding="utf-8") as fp:
                        reader = csv.DictReader(fp)
                        cols = reader.fieldnames or []
                        if "metric_name" not in cols or "strategy_args" not in cols:
                            continue
                        value_col = cols[-1]
                        for row in reader:
                            metric = row.get("metric_name")
                            if metric not in METRICS:
                                continue
                            h = parse_horizon(row.get("strategy_args", ""))
                            if h is None:
                                continue
                            try:
                                value = float(row.get(value_col, ""))
                            except Exception:
                                continue
                            key = (dataset, model, h, metric)
                            prev = latest.get(key)
                            if prev is None or mtime >= prev["mtime"]:
                                latest[key] = {
                                    "dataset": dataset,
                                    "model": model,
                                    "horizon": h,
                                    "metric": metric,
                                    "value": value,
                                    "mtime": mtime,
                                }
                except Exception:
                    continue
    return list(latest.values())


def build_compare_rows(records):
    by_point = {}
    for r in records:
        key = (r["dataset"], r["horizon"], r["metric"])
        item = by_point.setdefault(key, {"dataset": r["dataset"], "horizon": r["horizon"], "metric": r["metric"]})
        item[r["model"]] = r["value"]
    rows = []
    for _, item in by_point.items():
        if all(m in item for m in MODELS):
            rows.append(item)
    rows.sort(key=lambda x: (x["dataset"], x["horizon"], x["metric"]))
    return rows


def write_mse_csv(compare_rows, out_csv):
    mse_rows = [r for r in compare_rows if r["metric"] == "mse_norm"]
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset", "horizon", "DAG", "SWAN", "SWAN_V3"])
        for r in mse_rows:
            writer.writerow([r["dataset"], r["horizon"], r["DAG"], r["SWAN"], r["SWAN_V3"]])
    return mse_rows


def make_svg(compare_rows, out_svg):
    wins = {m: {k: 0 for k in MODELS} for m in METRICS}
    for metric in METRICS:
        sub = [r for r in compare_rows if r["metric"] == metric]
        for r in sub:
            best = min(MODELS, key=lambda x: r[x])
            wins[metric][best] += 1

    mse_rows = [r for r in compare_rows if r["metric"] == "mse_norm"]
    max_mse = max((max(r[m] for m in MODELS) for r in mse_rows), default=1.0)

    width = 2200
    height = 1300
    top_y0 = 80
    top_h = 280
    bottom_y0 = 460
    bottom_h = 740
    left = 90
    right = width - 40
    top_colors = {"DAG": "#1f77b4", "SWAN": "#2ca02c", "SWAN_V3": "#d62728"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="40" font-size="26" font-family="Arial" fill="#222">DAG vs SWAN vs SWAN_V3 结果对比图（公共结果点）</text>',
        '<text x="30" y="68" font-size="16" font-family="Arial" fill="#555">上图：各指标最优次数；下图：按 dataset-horizon 的 mse_norm 柱状对比（越低越好）</text>',
    ]

    # top chart
    parts.append(f'<line x1="{left}" y1="{top_y0 + top_h}" x2="{right}" y2="{top_y0 + top_h}" stroke="#888"/>')
    parts.append(f'<line x1="{left}" y1="{top_y0}" x2="{left}" y2="{top_y0 + top_h}" stroke="#888"/>')
    max_win = max((wins[m][k] for m in METRICS for k in MODELS), default=1)
    max_win = max(max_win, 1)
    group_w = 520
    bar_w = 70
    for i, metric in enumerate(METRICS):
        gx = left + 120 + i * group_w
        parts.append(f'<text x="{gx+20}" y="{top_y0 + top_h + 28}" font-size="16" font-family="Arial">{html.escape(metric)}</text>')
        for j, model in enumerate(MODELS):
            val = wins[metric][model]
            h = int((val / max_win) * (top_h - 40))
            x = gx + j * (bar_w + 26)
            y = top_y0 + top_h - h
            color = top_colors[model]
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}"/>')
            parts.append(f'<text x="{x+24}" y="{y-6}" font-size="14" font-family="Arial">{val}</text>')

    legend_x = right - 320
    legend_y = top_y0 + 20
    for i, m in enumerate(MODELS):
        y = legend_y + i * 24
        parts.append(f'<rect x="{legend_x}" y="{y-12}" width="14" height="14" fill="{top_colors[m]}"/>')
        parts.append(f'<text x="{legend_x+22}" y="{y}" font-size="14" font-family="Arial">{m}</text>')

    # bottom chart
    parts.append(f'<line x1="{left}" y1="{bottom_y0 + bottom_h}" x2="{right}" y2="{bottom_y0 + bottom_h}" stroke="#888"/>')
    parts.append(f'<line x1="{left}" y1="{bottom_y0}" x2="{left}" y2="{bottom_y0 + bottom_h}" stroke="#888"/>')

    n = len(mse_rows)
    if n > 0:
        slot = max((right - left - 40) / n, 6)
        bw = max(slot / 4, 1.5)
        for i, r in enumerate(mse_rows):
            base_x = left + 20 + i * slot
            label = f'{r["dataset"]}-h{r["horizon"]}'
            for j, model in enumerate(MODELS):
                val = r[model]
                h = int((val / max_mse) * (bottom_h - 50))
                x = base_x + j * bw
                y = bottom_y0 + bottom_h - h
                parts.append(f'<rect x="{x:.2f}" y="{y}" width="{bw:.2f}" height="{h}" fill="{top_colors[model]}" opacity="0.9"/>')
            if i % max(int(n / 24), 1) == 0:
                tx = base_x + bw
                ty = bottom_y0 + bottom_h + 18
                parts.append(f'<text x="{tx:.2f}" y="{ty}" font-size="10" font-family="Arial" transform="rotate(60 {tx:.2f},{ty})">{html.escape(label)}</text>')

    parts.append(f'<text x="{left}" y="{bottom_y0 - 14}" font-size="16" font-family="Arial">mse_norm 对比（DAG/SWAN/SWAN_V3）</text>')
    parts.append('</svg>')

    out_svg.write_text("\n".join(parts), encoding="utf-8")


def main():
    latest = load_latest_records()
    if not latest:
        raise RuntimeError("No comparable records found under result directory.")

    compare_rows = build_compare_rows(latest)
    if not compare_rows:
        raise RuntimeError("No common points found for DAG/SWAN/SWAN_V3.")
    table_path = RESULT_DIR / "model_compare_dag_swan_swanv3_mse.csv"
    write_mse_csv(compare_rows, table_path)

    out_svg = RESULT_DIR / "model_compare_dag_swan_swanv3.svg"
    make_svg(compare_rows, out_svg)
    print(f"Saved plot: {out_svg}")
    print(f"Saved table: {table_path}")


if __name__ == "__main__":
    main()
