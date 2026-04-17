import pandas as pd
from pathlib import Path

REQUIRED = ["State", "Indicator", "Value", "Definition", "Source_Label"]


def load_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in {path}")

    return df


def merge_indicator_tables(policy, env, boarding=None):
    dfs = [
        load_file(policy),
        load_file(env),
    ]

    if boarding and Path(boarding).exists():
        dfs.append(load_file(boarding))

    return pd.concat(dfs, ignore_index=True)
