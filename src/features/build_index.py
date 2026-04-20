from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

NormalizationMethod = Literal["zscore", "minmax"]


@dataclass
class IndexArtifacts:
    scores: pd.DataFrame
    normalized_matrix: pd.DataFrame
    indicator_diagnostics: pd.DataFrame


def normalize_indicator(series: pd.Series, method: NormalizationMethod = "zscore") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if method == "zscore":
        std = numeric.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series(np.zeros(len(numeric)), index=numeric.index)
        return (numeric - numeric.mean()) / std

    if method == "minmax":
        denominator = numeric.max() - numeric.min()
        if denominator == 0 or pd.isna(denominator):
            return pd.Series(np.zeros(len(numeric)), index=numeric.index)
        return (numeric - numeric.min()) / denominator

    raise ValueError(f"Unsupported normalization method: {method}")


def load_weights(json_path: str | Path) -> dict[str, float]:
    weights = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Weights file must contain a non-empty JSON object.")

    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        try:
            cleaned[str(key).strip()] = float(value)
        except Exception as exc:
            raise ValueError(f"Weight for '{key}' is not numeric: {value}") from exc
    return cleaned


def _pivot_indicator_table(indicator_df: pd.DataFrame) -> pd.DataFrame:
    required = {"State", "Indicator", "Value"}
    missing = required - set(indicator_df.columns)
    if missing:
        raise ValueError(f"Indicator dataframe missing required columns: {sorted(missing)}")

    working = indicator_df.copy()
    working["State"] = working["State"].astype(str).str.strip()
    working["Indicator"] = working["Indicator"].astype(str).str.strip()
    working["Value"] = pd.to_numeric(working["Value"], errors="coerce")

    duplicates = working[working.duplicated(subset=["State", "Indicator"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["State", "Indicator"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Duplicate (State, Indicator) pairs found before pivot. Examples: {sample}")

    pivot = working.pivot(index="State", columns="Indicator", values="Value").sort_index()
    if pivot.empty:
        raise ValueError("Pivoted indicator table is empty.")
    return pivot


def _build_indicator_diagnostics(
    pivot: pd.DataFrame,
    weights: dict[str, float],
    normalization: NormalizationMethod,
    drop_constant_indicators: bool,
) -> pd.DataFrame:
    rows = []
    for indicator in pivot.columns:
        series = pd.to_numeric(pivot[indicator], errors="coerce")
        non_null = int(series.notna().sum())
        unique_non_null = int(series.dropna().nunique())
        is_constant = unique_non_null <= 1
        rows.append(
            {
                "Indicator": indicator,
                "Configured_Weight": weights.get(indicator, np.nan),
                "Normalization_Method": normalization,
                "Observed_States": non_null,
                "Missing_States": int(series.isna().sum()),
                "Unique_NonNull_Values": unique_non_null,
                "Raw_Min": float(series.min()) if series.notna().any() else np.nan,
                "Raw_Max": float(series.max()) if series.notna().any() else np.nan,
                "Raw_Mean": float(series.mean()) if series.notna().any() else np.nan,
                "Raw_STD": float(series.std(ddof=0)) if series.notna().any() else np.nan,
                "Is_Constant_Within_Sample": is_constant,
                "Included_In_Primary_Index": bool(not is_constant or not drop_constant_indicators),
                "Exclusion_Reason": "constant_within_sample" if is_constant and drop_constant_indicators else "",
            }
        )
    return pd.DataFrame(rows).sort_values("Indicator").reset_index(drop=True)


def _weighted_row_mean(matrix: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weight_series = pd.Series(weights, dtype=float)
    aligned = matrix[weight_series.index]
    weighted = aligned.mul(weight_series, axis=1)
    denominator = aligned.notna().mul(weight_series, axis=1).sum(axis=1)
    return weighted.sum(axis=1, skipna=True) / denominator.replace({0.0: np.nan})


def compute_index(
    indicator_df: pd.DataFrame,
    weights: dict[str, float],
    normalization: NormalizationMethod = "zscore",
    index_name: str = "Historical_Trauma_Index",
    drop_constant_indicators: bool = True,
) -> IndexArtifacts:
    pivot = _pivot_indicator_table(indicator_df)
    diagnostics = _build_indicator_diagnostics(pivot, weights, normalization, drop_constant_indicators)

    indicators = list(pivot.columns)
    missing_weights = [indicator for indicator in indicators if indicator not in weights]
    if missing_weights:
        raise ValueError(f"Missing weights for indicators: {missing_weights}")

    included = diagnostics.loc[diagnostics["Included_In_Primary_Index"], "Indicator"].tolist()
    if not included:
        raise ValueError("All indicators were excluded; cannot compute the primary index.")

    normalized_columns = {
        indicator: normalize_indicator(pivot[indicator], method=normalization)
        for indicator in pivot.columns
    }
    normalized = pd.DataFrame(normalized_columns, index=pivot.index)

    applied_weights = {indicator: weights[indicator] for indicator in included}
    if sum(applied_weights.values()) == 0:
        raise ValueError("Total applied weight is zero.")

    index_scores = _weighted_row_mean(normalized[included], applied_weights)
    output = pd.DataFrame(
        {
            "State": normalized.index,
            index_name: index_scores.values,
            "Indicators_Used": normalized[included].notna().sum(axis=1).values,
            "Normalization_Method": normalization,
            "Constant_Indicators_Dropped": int((~diagnostics["Included_In_Primary_Index"]).sum()),
        }
    )

    normalized = normalized.reset_index().rename(columns={"index": "State"})
    return IndexArtifacts(scores=output, normalized_matrix=normalized, indicator_diagnostics=diagnostics)


def _equal_weights(indicators: list[str]) -> dict[str, float]:
    if not indicators:
        raise ValueError("Cannot create equal weights for an empty indicator set.")
    return {indicator: 1.0 for indicator in indicators}


def _pca_score_from_normalized(normalized: pd.DataFrame, primary_scores: pd.Series | None = None) -> pd.Series:
    matrix = normalized.to_numpy(dtype=float)
    centered = matrix - np.nanmean(matrix, axis=0)
    centered = np.nan_to_num(centered, nan=0.0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    component = vh[0]
    scores = centered @ component
    score_series = pd.Series(scores, index=normalized.index, dtype=float)

    if primary_scores is not None:
        aligned = pd.concat([score_series, primary_scores], axis=1).dropna()
        if not aligned.empty and aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) < 0:
            score_series = -score_series
    return score_series


def sensitivity_analysis(
    indicator_df: pd.DataFrame,
    primary_weights: dict[str, float],
    primary_normalization: NormalizationMethod = "zscore",
    alternate_normalizations: list[NormalizationMethod] | None = None,
    include_equal_weights: bool = True,
    include_pca: bool = True,
    leave_one_indicator_out: bool = True,
    index_name: str = "Historical_Trauma_Index",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alternate_normalizations = alternate_normalizations or []

    primary = compute_index(
        indicator_df,
        primary_weights,
        normalization=primary_normalization,
        index_name=index_name,
        drop_constant_indicators=True,
    )
    diagnostics = primary.indicator_diagnostics.copy()
    active_indicators = diagnostics.loc[diagnostics["Included_In_Primary_Index"], "Indicator"].tolist()
    primary_scores = primary.scores.set_index("State")[index_name]

    scheme_frames = []

    def add_scheme(frame: pd.DataFrame, scheme_name: str, scheme_type: str, normalization: str, note: str) -> None:
        working = frame.copy()
        working["Scheme"] = scheme_name
        working["Scheme_Type"] = scheme_type
        working["Normalization_Method"] = normalization
        working["Scheme_Note"] = note
        scheme_frames.append(working)

    add_scheme(primary.scores[["State", index_name]], "primary", "configured_weights", primary_normalization, "Configured weights after dropping constant indicators.")

    if include_equal_weights:
        equal = compute_index(
            indicator_df,
            _equal_weights(list(_pivot_indicator_table(indicator_df).columns)),
            normalization=primary_normalization,
            index_name=index_name,
            drop_constant_indicators=True,
        )
        add_scheme(equal.scores[["State", index_name]], "equal_weights", "equal_weights", primary_normalization, "Equal weights across included indicators.")

    for alt_normalization in alternate_normalizations:
        alternate = compute_index(
            indicator_df,
            primary_weights,
            normalization=alt_normalization,
            index_name=index_name,
            drop_constant_indicators=True,
        )
        add_scheme(
            alternate.scores[["State", index_name]],
            f"configured_weights_{alt_normalization}",
            "alternate_normalization",
            alt_normalization,
            "Configured weights under alternate normalization.",
        )

    if include_pca:
        normalized_primary = primary.normalized_matrix.set_index("State")[active_indicators]
        pca_scores = _pca_score_from_normalized(normalized_primary, primary_scores)
        pca_frame = pd.DataFrame({"State": pca_scores.index, index_name: pca_scores.values})
        add_scheme(
            pca_frame,
            "pca_pc1",
            "pca",
            primary_normalization,
            "First principal component score oriented to align with the primary index.",
        )

    if leave_one_indicator_out and len(active_indicators) > 1:
        for dropped_indicator in active_indicators:
            adjusted_weights = {k: v for k, v in primary_weights.items() if k != dropped_indicator}
            filtered_indicator_df = indicator_df.loc[
                indicator_df["Indicator"].astype(str).str.strip() != dropped_indicator
            ].copy()
            rerun = compute_index(
                filtered_indicator_df,
                adjusted_weights,
                normalization=primary_normalization,
                index_name=index_name,
                drop_constant_indicators=True,
            )
            add_scheme(
                rerun.scores[["State", index_name]],
                f"leave_out_{dropped_indicator}",
                "leave_one_indicator_out",
                primary_normalization,
                f"Configured weights after removing {dropped_indicator}.",
            )

    long_df = pd.concat(scheme_frames, ignore_index=True)
    long_df["Rank"] = long_df.groupby("Scheme")[index_name].rank(ascending=False, method="dense")

    primary_lookup = (
        long_df.loc[long_df["Scheme"] == "primary", ["State", index_name, "Rank"]]
        .rename(columns={index_name: "Primary_Score", "Rank": "Primary_Rank"})
    )
    long_df = long_df.merge(primary_lookup, on="State", how="left")
    long_df["Score_Delta_vs_Primary"] = long_df[index_name] - long_df["Primary_Score"]
    long_df["Rank_Shift_vs_Primary"] = long_df["Rank"] - long_df["Primary_Rank"]
    long_df = long_df.sort_values(["Scheme", "Rank", "State"]).reset_index(drop=True)

    summary = (
        long_df.groupby(["Scheme", "Scheme_Type", "Normalization_Method", "Scheme_Note"], dropna=False)
        .agg(
            Mean_Absolute_Rank_Shift=("Rank_Shift_vs_Primary", lambda series: float(np.nanmean(np.abs(series)))),
            Max_Absolute_Rank_Shift=("Rank_Shift_vs_Primary", lambda series: float(np.nanmax(np.abs(series)))),
            Mean_Absolute_Score_Delta=("Score_Delta_vs_Primary", lambda series: float(np.nanmean(np.abs(series)))),
        )
        .reset_index()
        .sort_values(["Scheme_Type", "Scheme"])
        .reset_index(drop=True)
    )

    return long_df, summary
