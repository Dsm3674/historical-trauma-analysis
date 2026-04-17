
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


def normalize_indicator(series: pd.Series, method: Literal["zscore", "minmax"] = "zscore") -> pd.Series:
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
    return df


def load_weights(json_path: str | Path) -> dict[str, float]:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def compute_index(
    indicator_df: pd.DataFrame,
    weights: dict[str, float],
    normalization: Literal["zscore", "minmax"] = "zscore",
    index_name: str = "Historical_Trauma_Index",
) -> pd.DataFrame:
    pivot = indicator_df.pivot_table(index="State", columns="Indicator", values="Value", aggfunc="first").copy()
    normalized_columns = {column: normalize_indicator(pivot[column], method=normalization) for column in pivot.columns}
    normalized = pd.DataFrame(normalized_columns, index=pivot.index)
    normalized[index_name] = 0.0
    usable_indicators = [c for c in normalized.columns if c != index_name]
    total_weight = sum(weights.get(indicator, 0.0) for indicator in usable_indicators)
    if total_weight == 0:
        raise ValueError("Total applied weight is zero.")
    for indicator in usable_indicators:
        normalized[index_name] += normalized[indicator] * weights.get(indicator, 0.0)
    normalized[index_name] = normalized[index_name] / total_weight
    return normalized.reset_index()


def sensitivity_analysis(
    indicator_df: pd.DataFrame,
    primary_weights: dict[str, float],
    alternate_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    primary = compute_index(indicator_df, primary_weights, index_name="Index_Primary")
    if alternate_weights is None:
        unique_indicators = sorted(indicator_df["Indicator"].unique())
        equal_weight = 1.0 / len(unique_indicators)
        alternate_weights = {indicator: equal_weight for indicator in unique_indicators}
    alternate = compute_index(indicator_df, alternate_weights, index_name="Index_Alternate")
    merged = primary.merge(alternate[["State", "Index_Alternate"]], on="State")
    merged["Rank_Primary"] = merged["Index_Primary"].rank(ascending=False, method="dense")
    merged["Rank_Alternate"] = merged["Index_Alternate"].rank(ascending=False, method="dense")
    merged["Rank_Shift"] = merged["Rank_Alternate"] - merged["Rank_Primary"]
    return merged.sort_values("Rank_Primary")
