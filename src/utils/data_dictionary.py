from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_data_dictionary(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    rows = []
    for column in df.columns:
        sample = df[column].dropna()
        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean() * 100.0),
                "example_value": str(sample.iloc[0]) if not sample.empty else "",
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
            "missing_pct": (df.isna().mean() * 100.0).values,
        }
    ).to_csv(output_dir / f"{dataset_name}_missing_report.csv", index=False)


def generate_basic_stats(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    ensure_dir(output_dir)
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return
    stats = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    stats.to_csv(output_dir / f"{dataset_name}_summary_stats.csv", index=False)


def process_dataset(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    generate_data_dictionary(df, dataset_name, output_dir)
    generate_missing_report(df, dataset_name, output_dir)
    generate_basic_stats(df, dataset_name, output_dir)
