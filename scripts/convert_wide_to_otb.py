import argparse
from pathlib import Path

import pandas as pd


def wide_to_otb(df: pd.DataFrame, date_col: str, feature_cols: list[str]) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col])
    long_blocks = []

    for col in feature_cols:
        block = pd.DataFrame(
            {
                "date": dates,
                "data": pd.to_numeric(df[col], errors="coerce"),
                "cols": col,
            }
        )
        long_blocks.append(block)

    otb_df = pd.concat(long_blocks, axis=0, ignore_index=True)
    return otb_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert wide-format time series CSV to OTB long format (date,data,cols)."
    )
    parser.add_argument("--input", required=True, help="Input wide CSV path")
    parser.add_argument("--output", required=True, help="Output OTB CSV path")
    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column name in input CSV",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Optional target column name to move to the end of feature order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found in {input_path}")

    feature_cols = [col for col in df.columns if col != args.date_col]
    if not feature_cols:
        raise ValueError("No feature columns found in input CSV")

    if args.target_col is not None:
        if args.target_col not in feature_cols:
            raise ValueError(f"Target column '{args.target_col}' not found in features")
        feature_cols = [col for col in feature_cols if col != args.target_col] + [args.target_col]

    otb_df = wide_to_otb(df, args.date_col, feature_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otb_df.to_csv(output_path, index=False)

    print(f"Converted: {input_path}")
    print(f"Saved OTB: {output_path}")
    print(f"Rows: {len(otb_df)}, Features: {len(feature_cols)}")
    print(f"Feature order: {feature_cols}")


if __name__ == "__main__":
    main()
