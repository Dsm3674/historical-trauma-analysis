from pathlib import Path
import pandas as pd

REQUIRED = ["State", "Indicator", "Value", "Definition", "Source_Label"]


def load_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {path}")

    return df[REQUIRED].copy()


def merge_indicator_tables(policy, env, boarding=None):
    dfs = [
        load_file(policy),
        load_file(env),
    ]

    if boarding is not None and Path(boarding).exists():
        dfs.append(load_file(boarding))

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["State", "Indicator", "Value"])
    return merged
