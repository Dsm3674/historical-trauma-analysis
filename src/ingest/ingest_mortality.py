
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


def load_cdc_export(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def standardize_mortality_export(
    df: pd.DataFrame,
    year_col: str,
    state_col: str,
    cause_col: str,
    rate_col: str,
    group_col: str,
    group_value: str,
) -> pd.DataFrame:
    work = df.copy()
    for required in [year_col, state_col, cause_col, rate_col, group_col]:
        if required not in work.columns:
            raise ValueError(f"Missing required column: {required}")
    work = work.loc[work[group_col] == group_value].copy()
    work[rate_col] = pd.to_numeric(work[rate_col], errors="coerce")
    work = work.rename(
        columns={
            year_col: "Year",
            state_col: "State",
            cause_col: "Condition",
            rate_col: "Age_Adjusted_Rate_per_100k",
            group_col: "Population_Group",
        }
    )
    return work[["Year", "State", "Condition", "Population_Group", "Age_Adjusted_Rate_per_100k"]]


def build_disparity_ratios(indigenous_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    merged = indigenous_df.merge(reference_df, on=["Year", "State", "Condition"], suffixes=("_AIAN", "_Reference"))
    merged["Disparity_Ratio"] = merged["Age_Adjusted_Rate_per_100k_AIAN"] / merged["Age_Adjusted_Rate_per_100k_Reference"]
    return merged[
        [
            "Year",
            "State",
            "Condition",
            "Population_Group_AIAN",
            "Population_Group_Reference",
            "Age_Adjusted_Rate_per_100k_AIAN",
            "Age_Adjusted_Rate_per_100k_Reference",
            "Disparity_Ratio",
        ]
    ].rename(
        columns={
            "Age_Adjusted_Rate_per_100k_AIAN": "AI_AN_Rate_per_100k",
            "Age_Adjusted_Rate_per_100k_Reference": "Comparator_Rate_per_100k",
        }
    )


if __name__ == "__main__":
    indigenous_path = Path("data/raw/mortality/aian_mortality_export.csv")
    reference_path = Path("data/raw/mortality/reference_mortality_export.csv")
    indigenous_raw = load_cdc_export(indigenous_path)
    reference_raw = load_cdc_export(reference_path)
    indigenous = standardize_mortality_export(
        indigenous_raw, "Year", "State", "Cause_of_death", "Age_adjusted_rate", "Race", "American Indian or Alaska Native"
    )
    reference = standardize_mortality_export(
        reference_raw, "Year", "State", "Cause_of_death", "Age_adjusted_rate", "Race", "All races"
    )
    disparities = build_disparity_ratios(indigenous, reference)
    meta = SourceMeta(
        dataset_name="CDC WONDER mortality disparity extract",
        source_org="CDC / NCHS",
        source_url="https://wonder.cdc.gov/",
        accessed_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
        citation="CDC WONDER Underlying Cause of Death",
        notes="Built from user-downloaded CDC exports.",
    )
    save_dataset_with_manifest(disparities, "data/processed/mortality.csv", meta)
    print("Wrote data/processed/mortality.csv")
