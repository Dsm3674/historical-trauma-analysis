from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


def normalize_indicator(
    series: pd.Series,
    method: Literal["zscore", "minmax"] = "zscore",
) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")

    if method == "zscore":
        std = series.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    if method == "minmax":
        denominator = series.max() - series.min()
        if denominator == 0 or pd.isna(denominator):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.min()) / denominator

    raise ValueError(f"Unsupported normalization method: {method}")


def load_indicator_table(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"State", "Indicator", "Value", "Definition", "Source_Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["State"] = df["State"].astype(str).str.strip()
    df["Indicator"] = df["Indicator"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    duplicate_rows = df[df.duplicated(subset=["State", "Indicator"], keep=False)]
    if not duplicate_rows.empty:
        sample = duplicate_rows[["State", "Indicator"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(
            "Duplicate (State, Indicator) pairs found in indicator table. "
            f"Examples: {sample}"
        )

    if df["Value"].isna().all():
        raise ValueError("All indicator values are missing after numeric conversion.")

    return df


def load_weights(json_path: str | Path) -> dict[str, float]:
    weights = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Weights file must contain a non-empty JSON object.")

    cleaned = {}
    for key, value in weights.items():
        if value is None:
            raise ValueError(f"Weight for '{key}' is null.")
        try:
            numeric_value = float(value)
        except Exception as exc:
            raise ValueError(f"Weight for '{key}' is not numeric: {value}") from exc
        cleaned[str(key).strip()] = numeric_value

    return cleaned


def compute_index(
    indicator_df: pd.DataFrame,
    weights: dict[str, float],
    normalization: Literal["zscore", "minmax"] = "zscore",
    index_name: str = "Historical_Trauma_Index",
) -> pd.DataFrame:
    required = {"State", "Indicator", "Value"}
    missing = required - set(indicator_df.columns)
    if missing:
        raise ValueError(f"Indicator dataframe missing required columns: {sorted(missing)}")

    working = indicator_df.copy()
    working["State"] = working["State"].astype(str).str.strip()
    working["Indicator"] = working["Indicator"].astype(str).str.strip()
    working["Value"] = pd.to_numeric(working["Value"], errors="coerce")

    duplicate_rows = working[working.duplicated(subset=["State", "Indicator"], keep=False)]
    if not duplicate_rows.empty:
        sample = duplicate_rows[["State", "Indicator"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(
            "Duplicate (State, Indicator) pairs found before pivot. "
            f"Examples: {sample}"
        )

    pivot = working.pivot(index="State", columns="Indicator", values="Value").copy()

    if pivot.empty:
        raise ValueError("Pivoted indicator table is empty.")

    normalized_columns = {
        column: normalize_indicator(pivot[column], method=normalization)
        for column in pivot.columns
    }
    normalized = pd.DataFrame(normalized_columns, index=pivot.index)
    normalized[index_name] = 0.0

    usable_indicators = [c for c in normalized.columns if c != index_name]
    missing_weights = [indicator for indicator in usable_indicators if indicator not in weights]
    if missing_weights:
        raise ValueError(f"Missing weights for indicators: {missing_weights}")

    extra_weights = [indicator for indicator in weights if indicator not in usable_indicators]
    if extra_weights:
        # harmless but usually signals mismatch between config and data
        print(f"Warning: weights provided for unused indicators: {extra_weights}")

    total_weight = sum(weights[indicator] for indicator in usable_indicators)
    if total_weight == 0:
        raise ValueError("Total applied weight is zero.")

    for indicator in usable_indicators:
        normalized[index_name] += normalized[indicator] * weights[indicator]

    normalized[index_name] = normalized[index_name] / total_weight
    return normalized.reset_index()


def sensitivity_analysis(
    indicator_df: pd.DataFrame,
    primary_weights: dict[str, float],
    alternate_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    primary = compute_index(indicator_df, primary_weights, index_name="Index_Primary")

    if alternate_weights is None:
        unique_indicators = sorted(indicator_df["Indicator"].astype(str).str.strip().unique())
        equal_weight = 1.0 / len(unique_indicators)
        alternate_weights = {indicator: equal_weight for indicator in unique_indicators}

    alternate = compute_index(indicator_df, alternate_weights, index_name="Index_Alternate")

    merged = primary.merge(alternate[["State", "Index_Alternate"]], on="State")
    merged["Rank_Primary"] = merged["Index_Primary"].rank(ascending=False, method="dense")
    merged["Rank_Alternate"] = merged["Index_Alternate"].rank(ascending=False, method="dense")
    merged["Rank_Shift"] = merged["Rank_Alternate"] - merged["Rank_Primary"]
    return merged.sort_values("Rank_Primary")
