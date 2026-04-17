
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.analysis import build_master_analysis_table, descriptive_summary, exploratory_association_table, write_limitations_report
from src.features.build_index import compute_index, sensitivity_analysis
from src.features.merge_indicators import merge_indicator_tables
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

    def build_indices(self) -> None:
        combined = merge_indicator_tables(
            self.raw_dir / "historical_policy" / "historical_policy.csv",
            self.raw_dir / "environmental" / "environmental_hazards.csv",
            self.processed_dir / "boarding_school_indicators.csv" if (self.processed_dir / "boarding_school_indicators.csv").exists() else None,
        )
        combined.to_csv(self.processed_dir / "combined_indicator_table.csv", index=False)

        weights_path = self.project_root / "config" / "indicator_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        import json
        weights = json.loads(weights_path.read_text(encoding="utf-8"))
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
        self._plot(master)

    def _plot(self, master: pd.DataFrame) -> None:
        needed = {"Historical_Trauma_Index", "Mean_Mortality_Disparity_Ratio"}
        if not needed.issubset(master.columns):
            return
        plot_df = master[list(needed)].dropna()
        if plot_df.empty:
            return
        plt.figure(figsize=(7, 5))
        plt.scatter(plot_df["Historical_Trauma_Index"], plot_df["Mean_Mortality_Disparity_Ratio"])
        plt.xlabel("Historical_Trauma_Index")
        plt.ylabel("Mean_Mortality_Disparity_Ratio")
        plt.title("Exploratory Association")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "historical_index_vs_disparity.png", dpi=200)
        plt.close()

    def run(self) -> None:
        self.ingest()
        self.build_indices()
        self.run_analysis()
        provenance = build_provenance_report(self.processed_dir, self.manifest_dir)
        import json
        (self.processed_dir / "provenance_report.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        self.logger.info("Pipeline complete.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ResearchPipeline(root).run()
