
from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED = ["State", "Indicator", "Value", "Definition", "Source_Label"]


def _load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[REQUIRED].copy()


def merge_indicator_tables(
    historical_policy_path: str | Path,
    environmental_path: str | Path,
    boarding_school_path: str | Path | None = None,
) -> pd.DataFrame:
    frames = [_load(historical_policy_path), _load(environmental_path)]
    if boarding_school_path:
        frames.append(_load(boarding_school_path))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["State", "Indicator", "Value"])
    return merged


if __name__ == "__main__":
    output = merge_indicator_tables(
        "data/raw/historical_policy/historical_policy.csv",
        "data/raw/environmental/environmental_hazards.csv",
        "data/processed/boarding_school_indicators.csv",
    )
    output.to_csv("data/processed/combined_indicator_table.csv", index=False)
    print("Wrote data/processed/combined_indicator_table.csv")
