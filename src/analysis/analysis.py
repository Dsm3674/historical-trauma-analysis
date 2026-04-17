from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def build_master_analysis_table(
    trauma_index_path,
    population_path,
    mortality_path,
    missing_persons_path=None,
):
    trauma = pd.read_csv(trauma_index_path)
    population = pd.read_csv(population_path)
    mortality = pd.read_csv(mortality_path)

    merged = trauma.merge(population, on="State", how="left")

    # ✅ FIX: merge by state (not broken global average)
    mortality_summary = (
        mortality.groupby("State")["Disparity_Ratio"]
        .mean()
        .reset_index(name="Mean_Mortality_Disparity_Ratio")
    )

    merged = merged.merge(mortality_summary, on="State", how="left")

    if missing_persons_path:
        missing_df = pd.read_csv(missing_persons_path)
        merged = merged.merge(
            missing_df[["State", "Overrepresentation_Ratio"]],
            on="State",
            how="left",
        )

    return merged


def exploratory_association_table(df):
    target = "Historical_Trauma_Index"
    cols = [
        "Mean_Mortality_Disparity_Ratio",
        "Overrepresentation_Ratio",
    ]

    results = []

    for col in cols:
        if col in df.columns:
            valid = df[[target, col]].dropna()

            if len(valid) < 3:
                continue

            rho = valid[target].rank().corr(valid[col].rank())

            results.append({
                "Outcome": col,
                "Correlation": rho,
                "N": len(valid),
                "Type": "Exploratory"
            })

    return pd.DataFrame(results)


def write_limitations_report(path):
    text = """Limitations:

- Ecological analysis (not individual-level)
- Small sample size limits inference
- No spatial clustering analysis implemented
- Trauma index is a proxy, not direct measure
"""
    Path(path).write_text(text)
