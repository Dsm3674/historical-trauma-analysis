
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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


class RealDataLoader:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def _read_csv(self, relative_path: str, required_columns: Iterable[str], dataset_name: str) -> pd.DataFrame:
        path = self.raw_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Required raw file not found: {path}. Add a real source file before running the pipeline."
            )
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df = normalize_state_names(df)
        assert_columns(df, required_columns, dataset_name)
        return df

    def load_population(self) -> DatasetBundle:
        df = self._read_csv(
            "population/population.csv",
            ["Year", "State", "AI_AN_Population", "Total_Population"],
            "population",
        )
        df = to_numeric_columns(df, ["Year", "AI_AN_Population", "Total_Population"])
        for opt in ["AI_AN_With_Disability", "AI_AN_In_Poverty"]:
            if opt in df.columns:
                df = to_numeric_columns(df, [opt])
        if "AI_AN_With_Disability" in df.columns:
            df["Disability_Rate"] = df["AI_AN_With_Disability"] / df["AI_AN_Population"]
        if "AI_AN_In_Poverty" in df.columns:
            df["Poverty_Rate"] = df["AI_AN_In_Poverty"] / df["AI_AN_Population"]
        meta = SourceMeta(
            dataset_name="population",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "population/population.csv"),
            notes="Add exact ACS variable documentation in the manifest before publication.",
            geographic_unit="state_or_county",
        )
        return DatasetBundle("population", df, meta)

    def load_mortality(self) -> DatasetBundle:
        df = self._read_csv(
            "mortality/mortality.csv",
            ["Year", "State", "Condition", "AI_AN_Rate_per_100k", "Comparator_Rate_per_100k"],
            "mortality",
        )
        df = to_numeric_columns(df, ["Year", "AI_AN_Rate_per_100k", "Comparator_Rate_per_100k"])
        if "Disparity_Ratio" not in df.columns:
            df["Disparity_Ratio"] = np.where(
                df["Comparator_Rate_per_100k"] > 0,
                df["AI_AN_Rate_per_100k"] / df["Comparator_Rate_per_100k"],
                np.nan,
            )
        meta = SourceMeta(
            dataset_name="mortality",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "mortality/mortality.csv"),
            notes="Prefer age-adjusted rates and explicit suppression handling.",
            geographic_unit="state_or_county",
        )
        return DatasetBundle("mortality", df, meta)

    def load_missing_persons(self) -> DatasetBundle:
        df = self._read_csv(
            "missing_persons/missing_persons.csv",
            ["Year", "State", "Total_Missing", "AI_AN_Missing"],
            "missing_persons",
        )
        df = to_numeric_columns(df, ["Year", "Total_Missing", "AI_AN_Missing", "AI_AN_Population_Percent"])
        if "AI_AN_Percent_Missing" not in df.columns:
            df["AI_AN_Percent_Missing"] = np.where(
                df["Total_Missing"] > 0,
                100.0 * df["AI_AN_Missing"] / df["Total_Missing"],
                np.nan,
            )
        if "Overrepresentation_Ratio" not in df.columns and "AI_AN_Population_Percent" in df.columns:
            df["Overrepresentation_Ratio"] = np.where(
                df["AI_AN_Population_Percent"] > 0,
                df["AI_AN_Percent_Missing"] / df["AI_AN_Population_Percent"],
                np.nan,
            )
        meta = SourceMeta(
            dataset_name="missing_persons",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(self.raw_dir / "missing_persons/missing_persons.csv"),
            notes="Coverage and inclusion rules must be documented because registries may be incomplete.",
            geographic_unit="state_or_county",
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
            notes="Each indicator must have a source and definition. Do not hard-code these in Python.",
            geographic_unit="state_or_county",
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
            notes="Counts and exposure proxies should come from a real extract, not a hard-coded dictionary.",
            geographic_unit="state_or_county",
        )
        return DatasetBundle("environmental_hazards", df, meta)

    def load_case_narratives(self) -> Optional[DatasetBundle]:
        path = self.raw_dir / "narratives" / "case_narratives.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        assert_columns(df, ["Case_ID", "Case_Type", "Narrative_Text"], "case_narratives")
        meta = SourceMeta(
            dataset_name="case_narratives",
            source_org="User-supplied source",
            source_url="",
            accessed_on=utc_now(),
            file_path=str(path),
            notes="Use only if a real, auditable, documented corpus exists.",
            restrictions="Add privacy and sharing restrictions here before release.",
        )
        return DatasetBundle("case_narratives", df, meta)
