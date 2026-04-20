from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.common import (
    DatasetBundle,
    SourceMeta,
    assert_columns,
    normalize_state_names,
    to_numeric_columns,
    utc_now,
)

VALID_STATE_NAMES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
}


class RealDataLoader:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def _read_csv(self, relative_path: str, required_columns: Iterable[str], dataset_name: str) -> pd.DataFrame:
        path = self.raw_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Required raw file not found: {path}. Add the real source file before running the pipeline."
            )

        df = pd.read_csv(path)
        df.columns = [column.strip() for column in df.columns]
        df = normalize_state_names(df)
        assert_columns(df, required_columns, dataset_name)

        if "State" in df.columns:
            bad_states = sorted(set(df["State"].dropna()) - VALID_STATE_NAMES)
            if bad_states:
                raise ValueError(f"{dataset_name} contains unrecognized state names: {bad_states[:10]}")

        return df

    def _validate_nonnegative(self, df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
        for column in columns:
            if column in df.columns and (pd.to_numeric(df[column], errors="coerce").dropna() < 0).any():
                raise ValueError(f"{dataset_name}.{column} contains negative values.")

    def _validate_percentage(self, df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
        for column in columns:
            if column in df.columns:
                series = pd.to_numeric(df[column], errors="coerce")
                invalid = ~series.dropna().between(0, 100, inclusive="both")
                if invalid.any():
                    raise ValueError(f"{dataset_name}.{column} contains values outside 0-100.")

    def load_population(self) -> DatasetBundle:
        df = self._read_csv(
            "population/population.csv",
            ["Year", "State", "AI_AN_Population", "Total_Population"],
            "population",
        )
        df = to_numeric_columns(df, ["Year", "AI_AN_Population", "Total_Population"])
        self._validate_nonnegative(df, ["Year", "AI_AN_Population", "Total_Population"], "population")

        if (df["AI_AN_Population"] > df["Total_Population"]).any():
            raise ValueError("population.AI_AN_Population exceeds Total_Population for at least one row.")

        df["AI_AN_Population_Percent"] = np.where(
            df["Total_Population"] > 0,
            100.0 * df["AI_AN_Population"] / df["Total_Population"],
            np.nan,
        )

        meta = SourceMeta(
            dataset_name="population",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "population/population.csv"),
            notes="AI/AN population share is required for compositional-confounding diagnostics.",
            geographic_unit="state",
        )
        return DatasetBundle("population", df, meta)

    def load_mortality(self) -> DatasetBundle:
        df = self._read_csv(
            "mortality/mortality.csv",
            ["Year", "State", "Condition", "AI_AN_Rate_per_100k", "Comparator_Rate_per_100k"],
            "mortality",
        )
        df = to_numeric_columns(df, ["Year", "AI_AN_Rate_per_100k", "Comparator_Rate_per_100k"])
        self._validate_nonnegative(df, ["Year", "AI_AN_Rate_per_100k", "Comparator_Rate_per_100k"], "mortality")

        if "Disparity_Ratio" not in df.columns:
            df["Disparity_Ratio"] = np.where(
                df["Comparator_Rate_per_100k"] > 0,
                df["AI_AN_Rate_per_100k"] / df["Comparator_Rate_per_100k"],
                np.nan,
            )

        meta = SourceMeta(
            dataset_name="mortality",
            source_org="CDC WONDER National Vital Statistics System",
            source_url="https://wonder.cdc.gov/ucd-icd10.html",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "mortality/mortality.csv"),
            notes="Age-adjusted diabetes mortality rates (ICD-10 E10-E14), 2020. AI/AN vs White comparator. States with suppressed AI/AN counts excluded.",
            geographic_unit="state",
        )
        return DatasetBundle("mortality", df, meta)

    def load_missing_persons(self) -> DatasetBundle:
        # Total_Missing is not required — NamUs May 2020 report provides
        # AI/AN counts only at the state level without total missing denominators.
        df = self._read_csv(
            "missing_persons/missing_persons.csv",
            ["Year", "State", "AI_AN_Missing"],
            "missing_persons",
        )
        df = to_numeric_columns(
            df,
            ["Year", "AI_AN_Missing", "AI_AN_Population_Percent"],
        )
        self._validate_nonnegative(
            df,
            ["Year", "AI_AN_Missing"],
            "missing_persons",
        )
        if "AI_AN_Population_Percent" in df.columns:
            self._validate_percentage(df, ["AI_AN_Population_Percent"], "missing_persons")

        # Compute derived columns only if Total_Missing is present
        if "Total_Missing" in df.columns:
            df = to_numeric_columns(df, ["Total_Missing"])
            self._validate_nonnegative(df, ["Total_Missing"], "missing_persons")
            if (df["AI_AN_Missing"] > df["Total_Missing"]).any():
                raise ValueError("missing_persons.AI_AN_Missing exceeds Total_Missing.")
            df["AI_AN_Percent_Missing"] = np.where(
                df["Total_Missing"] > 0,
                100.0 * df["AI_AN_Missing"] / df["Total_Missing"],
                np.nan,
            )
            if "AI_AN_Population_Percent" in df.columns:
                df["Overrepresentation_Ratio"] = np.where(
                    df["AI_AN_Population_Percent"] > 0,
                    df["AI_AN_Percent_Missing"] / df["AI_AN_Population_Percent"],
                    np.nan,
                )
            self._validate_percentage(
                df, ["AI_AN_Percent_Missing", "AI_AN_Population_Percent"], "missing_persons"
            )
            self._validate_nonnegative(df, ["Overrepresentation_Ratio"], "missing_persons")

        meta = SourceMeta(
            dataset_name="missing_persons",
            source_org="National Institute of Justice / NamUs",
            source_url="https://www.namus.gov",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "missing_persons/missing_persons.csv"),
            notes=(
                "AI/AN unresolved missing persons counts from NamUs May 1, 2020 report. "
                "Total missing counts not published at state level; AI_AN_Percent_Missing "
                "and Overrepresentation_Ratio therefore not computed."
            ),
            geographic_unit="state",
        )
        return DatasetBundle("missing_persons", df, meta)

    def load_historical_policy(self) -> DatasetBundle:
        df = self._read_csv(
            "historical_policy/historical_policy.csv",
            ["State", "Indicator", "Value", "Definition", "Source_Label"],
            "historical_policy",
        )
        df = to_numeric_columns(df, ["Value"])
        meta = SourceMeta(
            dataset_name="historical_policy",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "historical_policy/historical_policy.csv"),
            notes="Definitions and source labels are preserved for methods reporting.",
            geographic_unit="state",
        )
        return DatasetBundle("historical_policy", df, meta)

    def load_environmental_hazards(self) -> DatasetBundle:
        df = self._read_csv(
            "environmental/environmental_hazards.csv",
            ["State", "Indicator", "Value", "Definition", "Source_Label"],
            "environmental_hazards",
        )
        df = to_numeric_columns(df, ["Value"])
        meta = SourceMeta(
            dataset_name="environmental_hazards",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "environmental/environmental_hazards.csv"),
            notes="Environmental indicator construction is surfaced in methods outputs for manuscript specificity.",
            geographic_unit="state",
        )
        return DatasetBundle("environmental_hazards", df, meta)
