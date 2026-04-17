
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_UNITS_FOR_INFERENCE = 20


def require_minimum_units(df: pd.DataFrame, unit_col: str, min_units: int = MIN_UNITS_FOR_INFERENCE) -> None:
    n_units = df[unit_col].nunique()
    if n_units < min_units:
        raise ValueError(f"Inferential analysis blocked: only {n_units} unique {unit_col} values are available. Need at least {min_units}.")


def load_processed_dataset(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset: {csv_path}")
    return pd.read_csv(csv_path)


def build_master_analysis_table(
    trauma_index_path: str | Path,
    population_path: str | Path,
    mortality_path: str | Path,
    missing_persons_path: str | Path | None = None,
) -> pd.DataFrame:
    trauma = load_processed_dataset(trauma_index_path)
    population = load_processed_dataset(population_path)
    mortality = load_processed_dataset(mortality_path)

    if "State" not in trauma.columns or "State" not in population.columns:
        raise ValueError("Both trauma index and population files must include 'State'.")

    merged = trauma.merge(population, on="State", how="left")

    mortality_summary = (
        mortality.groupby("State", as_index=False)["Disparity_Ratio"]
        .mean()
        .rename(columns={"Disparity_Ratio": "Mean_Mortality_Disparity_Ratio"})
    )
    merged = merged.merge(mortality_summary, on="State", how="left")

    if missing_persons_path:
        missing_df = load_processed_dataset(missing_persons_path)
        if "State" in missing_df.columns:
            keep_cols = [c for c in ["State", "Overrepresentation_Ratio", "AI_AN_Percent_Missing"] if c in missing_df.columns]
            merged = merged.merge(missing_df[keep_cols], on="State", how="left")

    return merged


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "Variable"})


def spearman_correlation(x: pd.Series, y: pd.Series) -> dict:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return {"n": int(len(valid)), "spearman_rho": np.nan}
    x_rank = valid.iloc[:, 0].rank(method="average")
    y_rank = valid.iloc[:, 1].rank(method="average")
    rho = x_rank.corr(y_rank)
    return {"n": int(len(valid)), "spearman_rho": float(rho) if pd.notna(rho) else np.nan}


def exploratory_association_table(df: pd.DataFrame, target: str = "Historical_Trauma_Index") -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Expected '{target}' in master analysis table.")
    candidate_columns = ["Mean_Mortality_Disparity_Ratio", "Overrepresentation_Ratio", "AI_AN_Percent_Missing", "AI_AN_Population"]
    rows = []
    for col in candidate_columns:
        if col in df.columns:
            result = spearman_correlation(df[target], df[col])
            rows.append({"Outcome": col, "N": result["n"], "Association_Type": "Spearman", "Effect_Size": result["spearman_rho"], "Interpretation_Label": "exploratory"})
    return pd.DataFrame(rows)


def write_limitations_report(output_path: str | Path) -> None:
    text = """Limitations and Interpretation Constraints

1. This analysis is ecological and uses aggregate units. It must not be interpreted as individual-level evidence.
2. Associations are exploratory unless the number of unique geographic units is sufficiently large and additional validation is provided.
3. The historical trauma index is a proxy measure constructed from documented indicators; it is not a direct measure of lived experience.
4. Any policy recommendations should be framed as tentative and contingent on stronger validation and, where appropriate, community consultation.
5. If the study includes only a small number of states or counties, significance testing should be avoided or clearly labeled as unstable.
"""
    Path(output_path).write_text(text, encoding="utf-8")
