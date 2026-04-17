from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

MIN_UNITS_FOR_INFERENCE = 20


def build_master_analysis_table(
    trauma_index_path,
    population_path,
    mortality_path,
    missing_persons_path=None,
):
    trauma = pd.read_csv(trauma_index_path)
    population = pd.read_csv(population_path)
    mortality = pd.read_csv(mortality_path)

    if "State" not in trauma.columns or "State" not in population.columns:
        raise ValueError("Both trauma and population files must include 'State'.")

    merged = trauma.merge(population, on="State", how="left")

    mortality_summary = (
        mortality.groupby("State")["Disparity_Ratio"]
        .mean()
        .reset_index(name="Mean_Mortality_Disparity_Ratio")
    )
    merged = merged.merge(mortality_summary, on="State", how="left")

    if missing_persons_path:
        missing_df = pd.read_csv(missing_persons_path)
        keep = [c for c in ["State", "Overrepresentation_Ratio", "AI_AN_Percent_Missing"] if c in missing_df.columns]
        if keep:
            merged = merged.merge(missing_df[keep], on="State", how="left")

    return merged


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "Variable"})


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    return x.rank(method="average").corr(y.rank(method="average"))


def bootstrap_spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 3000, seed: int = 42) -> tuple[float, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 4:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    arr = valid.to_numpy()
    stats = []

    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        sx = pd.Series(sample[:, 0])
        sy = pd.Series(sample[:, 1])
        stats.append(_rank_corr(sx, sy))

    return (float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5)))


def evidence_label(n: int) -> str:
    return "exploratory" if n < MIN_UNITS_FOR_INFERENCE else "ecological_inferential"


def exploratory_association_table(df: pd.DataFrame, target: str = "Historical_Trauma_Index") -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Expected '{target}' in dataframe.")

    outcomes = [
        "Mean_Mortality_Disparity_Ratio",
        "Overrepresentation_Ratio",
        "AI_AN_Percent_Missing",
        "AI_AN_Population",
    ]

    rows = []
    for col in outcomes:
        if col not in df.columns:
            continue

        valid = df[[target, col]].dropna()
        n = len(valid)
        if n < 3:
            continue

        rho = _rank_corr(valid[target], valid[col])
        ci_low, ci_high = bootstrap_spearman_ci(valid[target], valid[col])

        rows.append(
            {
                "Outcome": col,
                "N": n,
                "Association_Type": "Spearman rank correlation",
                "Effect_Size": rho,
                "CI_2.5": ci_low,
                "CI_97.5": ci_high,
                "Evidence_Label": evidence_label(n),
                "Interpretation_Label": "exploratory" if n < MIN_UNITS_FOR_INFERENCE else "ecological_only_not_individual_level",
            }
        )

    return pd.DataFrame(rows)


def write_limitations_report(path):
    text = """Limitations:

- This is an ecological analysis and must not be interpreted as individual-level evidence.
- If the number of geographic units is small, associations should be interpreted as exploratory rather than confirmatory.
- The historical trauma index is a proxy measure constructed from documented indicators; it is not a direct measure of lived experience.
- No spatial clustering analysis is implemented in this codebase. The manuscript should not claim spatial clustering results unless a documented spatial module is added.
- No policy recommendations should be framed as definitive unless stronger validation, richer data, and appropriate community-informed interpretation are added.
- Any manuscript using this pipeline should clearly document source definitions, inclusion criteria, and indicator construction rules.
"""
    Path(path).write_text(text)
