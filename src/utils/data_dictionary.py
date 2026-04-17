
from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_data_dictionary(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    rows = []
    for col in df.columns:
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean() * 100),
                "example_value": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "",
                "description": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{dataset_name}_data_dictionary.csv", index=False)


def generate_missing_report(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": df.isna().mean().values * 100,
        }
    ).to_csv(output_dir / f"{dataset_name}_missing_report.csv", index=False)


def generate_basic_stats(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return
    stats = numeric_df.describe().T.reset_index().rename(columns={"index": "column"})
    stats.to_csv(output_dir / f"{dataset_name}_summary_stats.csv", index=False)


def process_dataset(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    generate_data_dictionary(df, dataset_name, output_dir)
    generate_missing_report(df, dataset_name, output_dir)
    generate_basic_stats(df, dataset_name, output_dir)
