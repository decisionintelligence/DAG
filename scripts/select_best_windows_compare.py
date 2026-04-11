import argparse
import base64
import json
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.recording import read_record_file


def parse_horizon(strategy_args: str) -> int:
    return int(json.loads(strategy_args)["horizon"])


def decode_blob(blob: str) -> Any:
    return pickle.loads(base64.b64decode(blob.encode("utf-8")))


def to_window_list(decoded_obj: Any) -> List[Any]:
    if isinstance(decoded_obj, list):
        return decoded_obj

    if isinstance(decoded_obj, np.ndarray):
        arr = decoded_obj
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        return [arr]

    return [decoded_obj]


def window_to_series(window_obj: Any) -> Tuple[np.ndarray, str]:
    if isinstance(window_obj, pd.DataFrame):
        y = window_obj.iloc[:, 0].to_numpy(dtype=float)
        key = str(window_obj.index[-1]) if len(window_obj.index) else "no_index"
        return y, key

    if isinstance(window_obj, np.ndarray):
        arr = window_obj
        if arr.ndim == 1:
            y = arr.astype(float)
        elif arr.ndim == 2:
            y = arr[:, 0].astype(float)
        elif arr.ndim == 3:
            y = arr[-1, :, 0].astype(float)
        else:
            raise ValueError(f"Unsupported ndarray ndim: {arr.ndim}")
        return y, "ndarray"

    raise ValueError(f"Unsupported window object type: {type(window_obj)}")


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("inf")
    yt = y_true[-n:]
    yp = y_pred[-n:]
    if metric == "mse":
        return float(np.mean((yt - yp) ** 2))
    return float(np.mean(np.abs(yt - yp)))


def parse_final_commands(final_sh: Path) -> Dict[int, str]:
    if not final_sh.exists():
        raise FileNotFoundError(f"FINAL script not found: {final_sh}")

    mapping: Dict[int, str] = {}
    for raw in final_sh.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "run_benchmark.py" not in line:
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except Exception:
            continue

        strategy_json: Optional[str] = None
        for i, tok in enumerate(tokens):
            if tok == "--strategy-args" and i + 1 < len(tokens):
                strategy_json = tokens[i + 1]
                break
        if not strategy_json:
            continue

        try:
            h = parse_horizon(strategy_json)
        except Exception:
            continue

        mapping[h] = line

    if not mapping:
        raise ValueError(f"No run_benchmark commands found in: {final_sh}")
    return mapping


def rebuild_command(
    command_line: str,
    dynamic_save_path: str,
    enforce_save_true_pred: bool,
) -> List[str]:
    tokens = shlex.split(command_line, posix=True)

    if not tokens:
        raise ValueError("Empty command line")

    if tokens[0].lower().startswith("python"):
        tokens[0] = sys.executable

    def set_arg(flag: str, value: str) -> None:
        for i, tok in enumerate(tokens):
            if tok == flag:
                if i + 1 < len(tokens):
                    tokens[i + 1] = value
                    return
        tokens.extend([flag, value])

    set_arg("--save-path", dynamic_save_path)
    if enforce_save_true_pred:
        set_arg("--save-true-pred", "true")

    return tokens


def has_prediction_logs(result_dir: Path) -> bool:
    if not result_dir.exists():
        return False
    for fp in result_dir.rglob("*"):
        if not fp.is_file():
            continue
        if fp.name.startswith("test_report"):
            continue
        if fp.name.endswith(".csv") or fp.name.endswith(".tar.gz"):
            return True
    return False


def run_dynamic_predictions_from_final(
    final_sh: Path,
    model_name: str,
    dynamic_rel_root: str,
    force_rerun: bool,
) -> Path:
    by_h = parse_final_commands(final_sh)
    output_root = PROJECT_ROOT / "result" / dynamic_rel_root / model_name

    for h, line in sorted(by_h.items()):
        rel_save = f"{dynamic_rel_root}/{model_name}/h{h}"
        abs_save = PROJECT_ROOT / "result" / rel_save
        abs_save.mkdir(parents=True, exist_ok=True)

        if (not force_rerun) and has_prediction_logs(abs_save):
            print(f"skip existing dynamic run: {abs_save}")
            continue

        cmd = rebuild_command(line, rel_save, enforce_save_true_pred=True)
        print(f"running dynamic predict: model={model_name}, horizon={h}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    return output_root


def collect_latest_rows_by_horizon(log_dir: Path) -> Dict[int, pd.Series]:
    candidates: List[Tuple[float, int, pd.Series]] = []

    for fp in log_dir.rglob("*"):
        if not fp.is_file():
            continue
        if not (fp.name.endswith(".csv") or fp.name.endswith(".tar.gz")):
            continue
        if fp.name.startswith("test_report"):
            continue

        try:
            df = read_record_file(str(fp))
        except Exception:
            continue

        required = {"strategy_args", "actual_data", "inference_data"}
        if not required.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            if pd.isna(row["actual_data"]) or pd.isna(row["inference_data"]):
                continue
            try:
                h = parse_horizon(row["strategy_args"])
            except Exception:
                continue
            candidates.append((fp.stat().st_mtime, h, row))

    if not candidates:
        raise ValueError(f"No prediction logs found in: {log_dir}")

    latest: Dict[int, Tuple[float, pd.Series]] = {}
    for mtime, h, row in candidates:
        if h not in latest or mtime > latest[h][0]:
            latest[h] = (mtime, row)

    return {h: pair[1] for h, pair in latest.items()}


def build_window_records(row: pd.Series, metric: str) -> List[Dict[str, Any]]:
    actual_windows = to_window_list(decode_blob(row["actual_data"]))
    pred_windows = to_window_list(decode_blob(row["inference_data"]))

    m = min(len(actual_windows), len(pred_windows))
    records: List[Dict[str, Any]] = []

    for i in range(m):
        y_true, key_true = window_to_series(actual_windows[i])
        y_pred, key_pred = window_to_series(pred_windows[i])
        n = min(len(y_true), len(y_pred))
        if n == 0:
            continue

        y_true = y_true[-n:]
        y_pred = y_pred[-n:]
        err = metric_value(y_true, y_pred, metric)
        key = key_true if key_true != "ndarray" else f"win_{i:05d}_{key_pred}"

        records.append(
            {
                "window_idx": i,
                "window_key": key,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": err,
                "horizon": n,
            }
        )

    records.sort(key=lambda x: x["error"])
    return records


def plot_true_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    x = np.arange(len(y_true))
    plt.figure(figsize=(10, 4))
    plt.plot(x, y_true, label="True", linewidth=2)
    plt.plot(x, y_pred, label="Pred", linewidth=2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_topk(
    rows_by_h: Dict[int, pd.Series],
    model_name: str,
    out_root: Path,
    top_k: int,
    metric: str,
    summary_rows: List[Dict[str, Any]],
) -> None:
    for h, row in sorted(rows_by_h.items()):
        records = build_window_records(row, metric=metric)
        best = records[:top_k]

        save_dir = out_root / f"h{h}" / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for rank, rec in enumerate(best, start=1):
            fname = f"top{rank:02d}_idx{rec['window_idx']:05d}_{metric}{rec['error']:.6f}.png"
            out_path = save_dir / fname
            title = (
                f"{model_name} | h={h} | rank={rank} | idx={rec['window_idx']} | "
                f"{metric}={rec['error']:.6f}"
            )
            plot_true_pred(rec["y_true"], rec["y_pred"], title, out_path)

            summary_rows.append(
                {
                    "horizon": h,
                    "model": model_name,
                    "rank": rank,
                    "window_idx": rec["window_idx"],
                    "window_key": rec["window_key"],
                    metric: rec["error"],
                    "image_path": str(out_path).replace("\\", "/"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run dynamic rolling predictions from FINAL.sh best params, then select and plot top-k "
            "best windows for targets_only and meteo_plus_targets, separated by horizon."
        )
    )
    parser.add_argument("--targets-final-sh", required=True, help="Path to targets_only FINAL.sh")
    parser.add_argument("--meteo-final-sh", required=True, help="Path to meteo_plus_targets FINAL.sh")
    parser.add_argument(
        "--dynamic-save-root",
        default="maize_NL_NL11_daily_window_dynamic",
        help=(
            "Relative save root under result/, used to store fresh dynamic predictions, "
            "e.g. maize_NL_NL11_daily_window_dynamic"
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun dynamic predictions even if dynamic logs already exist",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for selected window plots")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K best windows per horizon per model")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae", help="Ranking metric")
    args = parser.parse_args()

    targets_final_sh = Path(args.targets_final_sh)
    meteo_final_sh = Path(args.meteo_final_sh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_dynamic_dir = run_dynamic_predictions_from_final(
        final_sh=targets_final_sh,
        model_name="targets_only",
        dynamic_rel_root=args.dynamic_save_root,
        force_rerun=args.force_rerun,
    )
    meteo_dynamic_dir = run_dynamic_predictions_from_final(
        final_sh=meteo_final_sh,
        model_name="meteo_plus_targets",
        dynamic_rel_root=args.dynamic_save_root,
        force_rerun=args.force_rerun,
    )

    targets_rows = collect_latest_rows_by_horizon(targets_dynamic_dir)
    meteo_rows = collect_latest_rows_by_horizon(meteo_dynamic_dir)

    summary_rows: List[Dict[str, Any]] = []

    save_topk(
        rows_by_h=targets_rows,
        model_name="targets_only",
        out_root=out_dir,
        top_k=args.top_k,
        metric=args.metric,
        summary_rows=summary_rows,
    )
    save_topk(
        rows_by_h=meteo_rows,
        model_name="meteo_plus_targets",
        out_root=out_dir,
        top_k=args.top_k,
        metric=args.metric,
        summary_rows=summary_rows,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_fp = out_dir / "best_windows_summary.csv"
    summary_df.to_csv(summary_fp, index=False)
    print(f"saved: {summary_fp}")


if __name__ == "__main__":
    main()
