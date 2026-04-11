import argparse
import base64
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ts_benchmark.recording import read_record_file


def parse_horizon(strategy_args: str) -> int:
    if not isinstance(strategy_args, str):
        raise ValueError("strategy_args is not a string")
    data = json.loads(strategy_args)
    return int(data["horizon"])


def decode_blob(blob: str):
    return pickle.loads(base64.b64decode(blob.encode("utf-8")))


def extract_last_window(decoded_obj) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(decoded_obj, list):
        last = decoded_obj[-1]
        if isinstance(last, pd.DataFrame):
            y = last.iloc[:, 0].to_numpy(dtype=float)
            x = np.arange(len(y))
            return x, y
        if isinstance(last, np.ndarray):
            arr = last
            if arr.ndim == 1:
                y = arr.astype(float)
            else:
                y = arr[:, 0].astype(float)
            x = np.arange(len(y))
            return x, y

    if isinstance(decoded_obj, pd.DataFrame):
        y = decoded_obj.iloc[:, 0].to_numpy(dtype=float)
        x = np.arange(len(y))
        return x, y

    if isinstance(decoded_obj, np.ndarray):
        arr = decoded_obj
        if arr.ndim == 1:
            y = arr.astype(float)
        elif arr.ndim == 2:
            y = arr[:, 0].astype(float)
        elif arr.ndim == 3:
            y = arr[-1, :, 0].astype(float)
        else:
            raise ValueError(f"Unsupported ndarray ndim: {arr.ndim}")
        x = np.arange(len(y))
        return x, y

    raise ValueError(f"Unsupported decoded object type: {type(decoded_obj)}")


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
        raise ValueError(
            f"No prediction records found in {log_dir}. Run benchmark with --save-true-pred true first."
        )

    latest: Dict[int, Tuple[float, pd.Series]] = {}
    for mtime, h, row in candidates:
        if h not in latest or mtime > latest[h][0]:
            latest[h] = (mtime, row)

    return {h: pair[1] for h, pair in latest.items()}


def plot_one(row: pd.Series, out_path: Path) -> None:
    actual = decode_blob(row["actual_data"])
    pred = decode_blob(row["inference_data"])

    xa, ya = extract_last_window(actual)
    xp, yp = extract_last_window(pred)

    n = min(len(ya), len(yp))
    ya = ya[-n:]
    yp = yp[-n:]
    x = np.arange(n)

    plt.figure(figsize=(10, 4))
    plt.plot(x, ya, label="True", linewidth=2)
    plt.plot(x, yp, label="Pred", linewidth=2)
    plt.title(f"Forecast Last Window (horizon={n})")
    plt.xlabel("Step")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot forecast curves from benchmark log files saved with --save-true-pred true"
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Result directory under project, e.g. ./result/maize_NL_NL11_daily_otb_targets_only/SWAN_V3_FINAL",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output figure directory. Default: <result-dir>/plots",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_rows = collect_latest_rows_by_horizon(result_dir)

    for h, row in sorted(latest_rows.items()):
        out_path = out_dir / f"forecast_h{h}_last_window.png"
        plot_one(row, out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
