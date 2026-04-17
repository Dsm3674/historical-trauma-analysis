
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def summarize_csv(csv_path: Path, manifest_dir: Path) -> dict:
    df = pd.read_csv(csv_path)
    manifest_path = manifest_dir / f"{csv_path.stem}.manifest.json"
    source_meta = {}
    if manifest_path.exists():
        source_meta = json.loads(manifest_path.read_text(encoding="utf-8")).get("source_meta", {})
    return {"file": str(csv_path), "rows": int(len(df)), "columns": list(df.columns), "source_meta": source_meta}


def build_provenance_report(processed_dir: str | Path = "data/processed", manifest_dir: str | Path = "data/manifests") -> dict:
    processed_dir = Path(processed_dir)
    manifest_dir = Path(manifest_dir)
    report = {"datasets": []}
    for csv_path in sorted(processed_dir.glob("*.csv")):
        report["datasets"].append(summarize_csv(csv_path, manifest_dir))
    return report


if __name__ == "__main__":
    report = build_provenance_report()
    out_path = Path("data/processed/provenance_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
