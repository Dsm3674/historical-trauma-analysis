
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.data_dictionary import process_dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SourceMeta:
    dataset_name: str
    source_org: str
    source_url: str
    accessed_on: str
    file_path: str
    citation: str = ""
    notes: str = ""
    geographic_unit: str = ""
    temporal_coverage: str = ""
    restrictions: str = ""


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    source_meta: SourceMeta
    schema_version: str = "1.0"

    def save(self, out_dir: Path, manifest_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{self.name}.csv"
        manifest_path = manifest_dir / f"{self.name}.manifest.json"
        self.frame.to_csv(csv_path, index=False)
        payload = {
            "name": self.name,
            "rows": int(len(self.frame)),
            "columns": list(self.frame.columns),
            "schema_version": self.schema_version,
            "source_meta": asdict(self.source_meta),
            "saved_at_utc": utc_now(),
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        process_dataset(self.frame, self.name, out_dir)


def assert_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def to_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_state_names(df: pd.DataFrame, state_col: str = "State") -> pd.DataFrame:
    out = df.copy()
    if state_col in out.columns:
        out[state_col] = out[state_col].astype(str).str.strip()
    return out


def get_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plos_research_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "research_pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
