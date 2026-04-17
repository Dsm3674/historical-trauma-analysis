from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import pandas as pd

from src.analysis.analysis import (
    build_master_analysis_table,
    descriptive_summary,
    exploratory_association_table,
    write_limitations_report,
)
from src.features.build_index import compute_index, sensitivity_analysis
from src.features.merge_indicators import merge_indicator_tables
from src.ingest.ingest_boarding_schools import (
    load_boarding_school_listing,
    build_state_boarding_school_features,
    to_indicator_table,
)
from src.ingest.loader import RealDataLoader
from src.reporting.provenance import build_provenance_report
from src.utils.common import get_logger
from src.utils.data_dictionary import process_dataset


@dataclass
class AnalysisGuardrails:
    minimum_units_for_inference: int = 20
    allow_policy_recommendations: bool = False
    allow_nlp_module: bool = False


class ResearchPipeline:
    def __init__(self, project_root: Path, guardrails: Optional[AnalysisGuardrails] = None):
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.manifest_dir = self.data_dir / "manifests"
        self.report_dir = project_root / "reports"
        self.vis_dir = project_root / "visualizations"
        self.log_dir = project_root / "logs"
        self.guardrails = guardrails or AnalysisGuardrails()
        self.logger = get_logger(self.log_dir)
        self.loader = RealDataLoader(self.raw_dir)

        for folder in [self.processed_dir, self.manifest_dir, self.report_dir, self.vis_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def ingest(self) -> None:
        self.logger.info("Starting real-data ingestion.")

        bundles = [
            self.loader.load_population(),
            self.loader.load_mortality(),
            self.loader.load_missing_persons(),
            self.loader.load_historical_policy(),
            self.loader.load_environmental_hazards(),
        ]

        for bundle in bundles:
            bundle.save(self.processed_dir, self.manifest_dir)
            self.logger.info("Saved %s", bundle.name)

        boarding_raw = self.raw_dir / "boarding_schools" / "boarding_school_listing.csv"
        if boarding_raw.exists():
            self.logger.info("Building boarding-school indicators from raw listing.")
            raw = load_boarding_school_listing(boarding_raw)
            features = build_state_boarding_school_features(raw)
            indicators = to_indicator_table(features)

            features.to_csv(self.processed_dir / "boarding_school_features.csv", index=False)
            indicators.to_csv(self.processed_dir / "boarding_school_indicators.csv", index=False)

            process_dataset(features, "boarding_school_features", self.processed_dir)
            process_dataset(indicators, "boarding_school_indicators", self.processed_dir)
        else:
            self.logger.warning("No boarding-school raw file found. Boarding-school indicators will be excluded.")

    def build_indices(self) -> None:
        boarding_path = self.processed_dir / "boarding_school_indicators.csv"

        combined = merge_indicator_tables(
            self.raw_dir / "historical_policy" / "historical_policy.csv",
            self.raw_dir / "environmental" / "environmental_hazards.csv",
            boarding_path if boarding_path.exists() else None,
        )

        combined.to_csv(self.processed_dir / "combined_indicator_table.csv", index=False)
        process_dataset(combined, "combined_indicator_table", self.processed_dir)

        weights_path = self.project_root / "config" / "indicator_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        weights = json.loads(weights_path.read_text())

        index_df = compute_index(combined, weights, index_name="Historical_Trauma_Index")
        index_df.to_csv(self.processed_dir / "historical_trauma_index.csv", index=False)
        process_dataset(index_df, "historical_trauma_index", self.processed_dir)

        sensitivity_df = sensitivity_analysis(combined, weights)
        sensitivity_df.to_csv(self.processed_dir / "historical_trauma_index_sensitivity.csv", index=False)
        process_dataset(sensitivity_df, "historical_trauma_index_sensitivity", self.processed_dir)

    def run_analysis(self) -> None:
        master = build_master_analysis_table(
            self.processed_dir / "historical_trauma_index.csv",
            self.processed_dir / "population.csv",
            self.processed_dir / "mortality.csv",
            self.processed_dir / "missing_persons.csv",
        )
        master.to_csv(self.processed_dir / "master_analysis_table.csv", index=False)
        process_dataset(master, "master_analysis_table", self.processed_dir)

        summary = descriptive_summary(master)
        summary.to_csv(self.processed_dir / "descriptive_summary.csv", index=False)

        associations = exploratory_association_table(master, target="Historical_Trauma_Index")
        associations.to_csv(self.processed_dir / "exploratory_associations.csv", index=False)

        write_limitations_report(self.report_dir / "limitations_report.txt")

    def run(self) -> None:
        self.ingest()
        self.build_indices()
        self.run_analysis()

        provenance = build_provenance_report(self.processed_dir, self.manifest_dir)
        (self.processed_dir / "provenance_report.json").write_text(json.dumps(provenance, indent=2))
        self.logger.info("Pipeline complete.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ResearchPipeline(root).run()
