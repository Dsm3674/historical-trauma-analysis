from __future__ import annotations

import re

import numpy as np
import pandas as pd


AGGREGATED_COUNT_PATTERN = re.compile(r"\(n\s*=\s*(\d+)\)", re.IGNORECASE)


def _extract_aggregated_school_count(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(AGGREGATED_COUNT_PATTERN, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def load_boarding_school_listing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.strip().replace(" ", "_") for column in df.columns]
    return df


def build_state_boarding_school_features(
    df: pd.DataFrame,
    state_col: str = "State",
    school_col: str = "School_Name",
    open_year_col: str = "Open_Year",
    close_year_col: str = "Close_Year",
    burial_site_col: str = "Burial_Site_Indicator",
) -> pd.DataFrame:
    for required in [state_col, school_col]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    work = df.copy()
    work[state_col] = work[state_col].astype(str).str.strip()
    work[school_col] = work[school_col].astype(str).str.strip()
    work["_Aggregated_School_Count"] = _extract_aggregated_school_count(work[school_col])
    work["_Effective_School_Count"] = work["_Aggregated_School_Count"].fillna(1.0)

    if open_year_col in work.columns:
        work[open_year_col] = pd.to_numeric(work[open_year_col], errors="coerce")
    if close_year_col in work.columns:
        work[close_year_col] = pd.to_numeric(work[close_year_col], errors="coerce")
    if burial_site_col in work.columns:
        work[burial_site_col] = pd.to_numeric(work[burial_site_col], errors="coerce").fillna(0)

    grouped = (
        work.groupby(state_col)["_Effective_School_Count"]
        .sum()
        .reset_index(name="BoardingSchool_Count")
        .rename(columns={state_col: "State"})
    )
    grouped["BoardingSchool_Count"] = grouped["BoardingSchool_Count"].astype(float)

    if open_year_col in work.columns and close_year_col in work.columns:
        durations = work.copy()
        durations["School_Duration_Years"] = durations[close_year_col] - durations[open_year_col]
        duration_summary = (
            durations.groupby(state_col)["School_Duration_Years"]
            .agg(["mean", "max"])
            .reset_index()
            .rename(
                columns={
                    state_col: "State",
                    "mean": "Mean_BoardingSchool_Duration_Years",
                    "max": "Max_BoardingSchool_Duration_Years",
                }
            )
        )
        grouped = grouped.merge(duration_summary, on="State", how="left")

    if burial_site_col in work.columns:
        burial = (
            work.groupby(state_col)[burial_site_col]
            .apply(lambda series: int((series > 0).sum()))
            .reset_index(name="Schools_With_Burial_Site_Flag")
            .rename(columns={state_col: "State"})
        )
        grouped = grouped.merge(burial, on="State", how="left")

    return grouped.sort_values("State").reset_index(drop=True)


def add_population_normalized_features(
    features: pd.DataFrame,
    population: pd.DataFrame,
    count_col: str = "BoardingSchool_Count",
    population_col: str = "AI_AN_Population",
    per_n: int = 10_000,
) -> pd.DataFrame:
    """Add a population-rate feature next to the raw count.

    Reviewer concern: BoardingSchool_Count is extensive (scales with state
    size) while other indicators in the composite are intensive (durations,
    binary flags). z-score normalization preserves the scaling property and
    leaves the composite correlated with population. Convert to a rate so
    a rate-normalized co-primary composite can be built.
    """
    if count_col not in features.columns or population_col not in population.columns:
        return features
    pop = population[["State", population_col]].copy()
    pop["State"] = pop["State"].astype(str).str.strip()
    out = features.merge(pop, on="State", how="left")
    rate_col = f"BoardingSchool_Rate_per_{per_n // 1000}k_AI_AN"
    out[rate_col] = np.where(
        out[population_col] > 0,
        float(per_n) * out[count_col] / out[population_col],
        np.nan,
    )
    return out.drop(columns=[population_col])


def to_indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    definition_map = {
        "BoardingSchool_Count": "Number of boarding schools recorded for the state, using aggregated DOI counts when encoded in the source label and unique rows otherwise.",
        "BoardingSchool_Rate_per_10k_AI_AN": "BoardingSchool_Count divided by the state AI/AN population (x 10,000). Population-rate alternative used to address the reviewer concern that the raw count is an extensive variable mixed with intensive ones.",
        "Mean_BoardingSchool_Duration_Years": "Mean operating duration in years across listed boarding schools in the state.",
        "Max_BoardingSchool_Duration_Years": "Maximum observed operating duration in years across listed boarding schools in the state.",
        "Schools_With_Burial_Site_Flag": "Count of listed boarding schools in the state with a burial-site flag in the source file.",
    }
    value_columns = [column for column in df.columns if column != "State"]
    long_df = df.melt(
        id_vars=["State"],
        value_vars=value_columns,
        var_name="Indicator",
        value_name="Value",
    )
    long_df["Definition"] = long_df["Indicator"].map(definition_map).fillna(long_df["Indicator"])
    long_df["Source_Label"] = "Federal Indian Boarding School listing"
    return long_df
