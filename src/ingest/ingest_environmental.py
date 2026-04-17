
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


def load_environmental_file(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"State", "Indicator", "Value", "Definition", "Source_Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


if __name__ == "__main__":
    raw_path = Path("data/raw/environmental/environmental_hazards.csv")
    df = load_environmental_file(raw_path)
    meta = SourceMeta(
        dataset_name="environmental_hazards",
        source_org="User-supplied source",
        source_url="",
        accessed_on=pd.Timestamp.today().strftime("%Y-%m-%d"),
        citation="Environmental hazards extract",
        notes="Keep original source definitions and field notes in the raw folder.",
    )
    save_dataset_with_manifest(df, "data/processed/environmental_hazards.csv", meta)
    print("Wrote data/processed/environmental_hazards.csv")
