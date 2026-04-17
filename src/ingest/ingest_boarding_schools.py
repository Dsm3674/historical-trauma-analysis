
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SourceMeta:
    dataset_name: str
    source_org: str
    source_url: str
    accessed_on: str
    citation: str
    notes: str


def save_dataset_with_manifest(df: pd.DataFrame, output_csv: str | Path, meta: SourceMeta) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    manifest = {"rows": int(len(df)), "columns": list(df.columns), "source_meta": asdict(meta)}
    output_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_boarding_school_listing(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def build_state_boarding_school_features(
    df: pd.DataFrame,
    state_col: str = "State",
    school_col: str = "School_Name",
    open_year_col: str = "Open_Year",
    close_year_col: str = "Close_Year",
    burial_site_col: str | None = "Burial_Site_Indicator",
) -> pd.DataFrame:
    required = [state_col, school_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    work = df.copy()
    grouped = work.groupby(state_col)[school_col].nunique().reset_index(name="BoardingSchool_Count").rename(columns={state_col: "State"})
    if open_year_col in work.columns:
        first_open = work.groupby(state_col)[open_year_col].min().reset_index(name="First_Open_Year").rename(columns={state_col: "State"})
        grouped = grouped.merge(first_open, on="State", how="left")
    if close_year_col in work.columns:
        last_close = work.groupby(state_col)[close_year_col].max().reset_index(name="Last_Close_Year").rename(columns={state_col: "State"})
        grouped = grouped.merge(last_close, on="State", how="left")
    if burial_site_col and burial_site_col in work.columns:
        burial = work.groupby(state_col)[burial_site_col].apply(
            lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).sum())
        ).reset_index(name="Schools_With_Burial_Site_Flag").rename(columns={state_col: "State"})
        grouped = grouped.merge(burial, on="State", how="left")
    return grouped


def to_indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c != "State"]
    long_df = df.melt(id_vars=["State"], value_vars=value_cols, var_name="Indicator", value_name="Value")
    long_df["Definition"] = long_df["Indicator"]
    long_df["Source_Label"] = "Federal Indian Boarding School listing"
    return long_df


if __name__ == "__main__":
    raw_path = Path("data/raw/boarding_schools/boarding_school_listing.csv")
    raw = load_boarding_school_listing(raw_path)
    features = build_state_boarding_school_features(raw)
    features.to_csv("data/processed/boarding_school_features.csv", index=False)
    indicators = to_indicator_table(features)
    meta = SourceMeta(
        dataset_name="boarding_school_indicators",
        source_org="U.S. Department of the Interior / Bureau of Indian Affairs",
        source_url="https://www.bia.gov/",
        accessed_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
        citation="Federal Indian Boarding School Initiative source listing",
        notes="Derived from a cleaned boarding-school source file.",
    )
    save_dataset_with_manifest(indicators, "data/processed/boarding_school_indicators.csv", meta)
    print("Wrote data/processed/boarding_school_indicators.csv")
