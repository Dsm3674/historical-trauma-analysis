
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


def load_namus_export(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def build_state_missing_persons_features(
    df: pd.DataFrame,
    state_col: str,
    race_col: str,
    total_case_id_col: str,
    aian_labels: list[str] | None = None,
) -> pd.DataFrame:
    if aian_labels is None:
        aian_labels = [
            "American Indian / Alaska Native",
            "American Indian/Alaska Native",
            "American Indian or Alaska Native",
        ]
    required = [state_col, race_col, total_case_id_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    work = df.copy()
    work["is_aian_case"] = work[race_col].isin(aian_labels)
    total_by_state = work.groupby(state_col)[total_case_id_col].nunique().reset_index(name="Total_Missing")
    aian_by_state = work.loc[work["is_aian_case"]].groupby(state_col)[total_case_id_col].nunique().reset_index(name="AI_AN_Missing")
    merged = total_by_state.merge(aian_by_state, on=state_col, how="left")
    merged["AI_AN_Missing"] = merged["AI_AN_Missing"].fillna(0)
    merged["AI_AN_Percent_Missing"] = (merged["AI_AN_Missing"] / merged["Total_Missing"]) * 100.0
    return merged.rename(columns={state_col: "State"})


def merge_population_share(
    missing_df: pd.DataFrame,
    population_df: pd.DataFrame,
    population_col: str = "AI_AN_Population",
    total_population_col: str = "Total_Population",
) -> pd.DataFrame:
    required_pop = {"State", population_col, total_population_col}
    missing = required_pop - set(population_df.columns)
    if missing:
        raise ValueError(f"Population file is missing: {sorted(missing)}")
    pop = population_df[["State", population_col, total_population_col]].copy()
    pop["AI_AN_Population_Percent"] = (pop[population_col] / pop[total_population_col]) * 100.0
    merged = missing_df.merge(pop[["State", "AI_AN_Population_Percent"]], on="State", how="left")
    merged["Overrepresentation_Ratio"] = merged["AI_AN_Percent_Missing"] / merged["AI_AN_Population_Percent"]
    return merged


if __name__ == "__main__":
    raw_path = Path("data/raw/missing_persons/namus_missing_persons_export.csv")
    population_path = Path("data/processed/population.csv")
    raw = load_namus_export(raw_path)
    features = build_state_missing_persons_features(raw, "Current_State", "Race", "Case_Number")
    population = pd.read_csv(population_path)
    output = merge_population_share(features, population)
    meta = SourceMeta(
        dataset_name="NamUs missing persons state extract",
        source_org="NamUs / NIJ",
        source_url="https://namus.nij.ojp.gov/",
        accessed_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
        citation="NamUs export",
        notes="Coverage depends on cases present in NamUs and is not a full census.",
    )
    save_dataset_with_manifest(output, "data/processed/missing_persons.csv", meta)
    print("Wrote data/processed/missing_persons.csv")
