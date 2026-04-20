from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_csv(csv_path: Path, manifest_dir: Path) -> dict:
    df = pd.read_csv(csv_path)
    manifest_path = manifest_dir / f"{csv_path.stem}.manifest.json"
    source_meta = {}
    if manifest_path.exists():
        source_meta = json.loads(manifest_path.read_text(encoding="utf-8")).get("source_meta", {})
    return {
        "file": str(csv_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "sha256": _sha256_file(csv_path),
        "source_meta": source_meta,
    }


def summarize_config(config_path: Path) -> dict:
    return {
        "file": str(config_path),
        "sha256": _sha256_file(config_path),
        "contents": json.loads(config_path.read_text(encoding="utf-8")),
    }


def build_provenance_report(
    processed_dir: str | Path = "data/processed",
    manifest_dir: str | Path = "data/manifests",
    config_dir: str | Path | None = None,
) -> dict:
    processed_dir = Path(processed_dir)
    manifest_dir = Path(manifest_dir)
    report = {"datasets": [], "configs": []}

    for csv_path in sorted(processed_dir.glob("*.csv")):
        report["datasets"].append(summarize_csv(csv_path, manifest_dir))

    if config_dir is not None:
        for config_path in sorted(Path(config_dir).glob("*.json")):
            report["configs"].append(summarize_config(config_path))

    return report
