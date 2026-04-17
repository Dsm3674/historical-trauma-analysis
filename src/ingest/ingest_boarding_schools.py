from __future__ import annotations

import pandas as pd


def load_boarding_school_listing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def build_state_boarding_school_features(
    df: pd.DataFrame,
    state_col: str = "State",
    school_col: str = "School_Name",
    open_year_col: str = "Open_Year",
    close_year_col: str = "Close_Year",
    burial_site_col: str = "Burial_Site_Indicator",
) -> pd.DataFrame:
    required = [state_col, school_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    work = df.copy()

    if open_year_col in work.columns:
        work[open_year_col] = pd.to_numeric(work[open_year_col], errors="coerce")
    if close_year_col in work.columns:
        work[close_year_col] = pd.to_numeric(work[close_year_col], errors="coerce")
    if burial_site_col in work.columns:
        work[burial_site_col] = pd.to_numeric(work[burial_site_col], errors="coerce").fillna(0)

    grouped = (
        work.groupby(state_col)[school_col]
        .nunique()
        .reset_index(name="BoardingSchool_Count")
        .rename(columns={state_col: "State"})
    )

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
            .apply(lambda s: int((s > 0).sum()))
            .reset_index(name="Schools_With_Burial_Site_Flag")
            .rename(columns={state_col: "State"})
        )
        grouped = grouped.merge(burial, on="State", how="left")

    return grouped


def to_indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    definition_map = {
        "BoardingSchool_Count": "Number of unique boarding schools recorded in the source listing for the state.",
        "Mean_BoardingSchool_Duration_Years": "Mean operating duration in years across listed boarding schools in the state.",
        "Max_BoardingSchool_Duration_Years": "Maximum observed operating duration in years across listed boarding schools in the state.",
        "Schools_With_Burial_Site_Flag": "Count of listed boarding schools in the state with a burial-site flag in the source file.",
    }

    value_cols = [c for c in df.columns if c != "State"]
    long_df = df.melt(id_vars=["State"], value_vars=value_cols, var_name="Indicator", value_name="Value")
    long_df["Definition"] = long_df["Indicator"].map(definition_map).fillna(long_df["Indicator"])
    long_df["Source_Label"] = "Federal Indian Boarding School listing"
    return long_df
