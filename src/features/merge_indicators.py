from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["State", "Indicator", "Value", "Definition", "Source_Label"]


def load_file(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing {column} in {csv_path}")

    df = df[REQUIRED_COLUMNS].copy()
    df["State"] = df["State"].astype(str).str.strip()
    df["Indicator"] = df["Indicator"].astype(str).str.strip()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df


def merge_indicator_tables(policy: str | Path, environmental: str | Path, boarding: str | Path | None = None) -> pd.DataFrame:
    frames = [load_file(policy), load_file(environmental)]
    if boarding is not None and Path(boarding).exists():
        frames.append(load_file(boarding))

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["State", "Indicator", "Value"])

    duplicates = merged[merged.duplicated(subset=["State", "Indicator"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["State", "Indicator"]].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Duplicate (State, Indicator) pairs found when merging indicators. Examples: {sample}")

    return merged.sort_values(["State", "Indicator"]).reset_index(drop=True)
