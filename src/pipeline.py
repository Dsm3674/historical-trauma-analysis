from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import pandas as pd

from src.analysis.analysis import (
    build_master_analysis_table,
    build_sample_characterization_table,
    descriptive_summary,
    exploratory_association_table,
    generate_figures,
    summarize_included_vs_excluded,
    validate_main_analysis_scope,
    write_limitations_report,
    write_methods_summary,
)
from src.features.build_index import compute_index, load_weights, sensitivity_analysis
from src.features.merge_indicators import merge_indicator_tables
from src.ingest.ingest_boarding_schools import (
    build_state_boarding_school_features,
    load_boarding_school_listing,
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


class ResearchPipeline:
    def __init__(self, project_root: Path, guardrails: AnalysisGuardrails | None = None):
        self.project_root = project_root
        self.config_dir = project_root / "config"
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
        self.analysis_config = self._load_analysis_config()

        for folder in [self.processed_dir, self.manifest_dir, self.report_dir, self.vis_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def _load_analysis_config(self) -> dict[str, Any]:
        path = self.config_dir / "analysis_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing analysis config: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def ingest(self) -> None:
        self.logger.info("Starting ingestion.")
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
        if not boarding_raw.exists():
            raise FileNotFoundError(f"Missing boarding-school raw file: {boarding_raw}")

        raw_listing = load_boarding_school_listing(str(boarding_raw))
        features = build_state_boarding_school_features(raw_listing)
        indicators = to_indicator_table(features)

        features.to_csv(self.processed_dir / "boarding_school_features.csv", index=False)
        indicators.to_csv(self.processed_dir / "boarding_school_indicators.csv", index=False)
        process_dataset(features, "boarding_school_features", self.processed_dir)
        process_dataset(indicators, "boarding_school_indicators", self.processed_dir)

    def build_indices(self) -> None:
        combined = merge_indicator_tables(
            self.raw_dir / "historical_policy" / "historical_policy.csv",
            self.raw_dir / "environmental" / "environmental_hazards.csv",
            self.processed_dir / "boarding_school_indicators.csv",
        )
        combined.to_csv(self.processed_dir / "combined_indicator_table.csv", index=False)
        process_dataset(combined, "combined_indicator_table", self.processed_dir)

        weights = load_weights(self.config_dir / "indicator_weights.json")
        index_artifacts = compute_index(
            combined,
            weights,
            normalization=self.analysis_config["primary_normalization"],
            index_name=self.analysis_config["index_name"],
            drop_constant_indicators=bool(self.analysis_config.get("drop_constant_indicators", True)),
        )

        index_artifacts.scores.to_csv(self.processed_dir / "historical_trauma_index.csv", index=False)
        index_artifacts.normalized_matrix.to_csv(self.processed_dir / "historical_trauma_index_normalized_matrix.csv", index=False)
        index_artifacts.indicator_diagnostics.to_csv(
            self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv",
            index=False,
        )
        process_dataset(index_artifacts.scores, "historical_trauma_index", self.processed_dir)
        process_dataset(index_artifacts.normalized_matrix, "historical_trauma_index_normalized_matrix", self.processed_dir)
        process_dataset(index_artifacts.indicator_diagnostics, "historical_trauma_index_indicator_diagnostics", self.processed_dir)

        sensitivity_df, sensitivity_summary = sensitivity_analysis(
            combined,
            weights,
            primary_normalization=self.analysis_config["primary_normalization"],
            alternate_normalizations=self.analysis_config.get("sensitivity", {}).get("alternate_normalizations", []),
            include_equal_weights=bool(self.analysis_config.get("sensitivity", {}).get("include_equal_weights", True)),
            include_pca=bool(self.analysis_config.get("sensitivity", {}).get("include_pca", True)),
            leave_one_indicator_out=bool(self.analysis_config.get("sensitivity", {}).get("leave_one_indicator_out", True)),
            index_name=self.analysis_config["index_name"],
        )
        sensitivity_df.to_csv(self.processed_dir / "historical_trauma_index_sensitivity.csv", index=False)
        sensitivity_summary.to_csv(self.processed_dir / "historical_trauma_index_sensitivity_summary.csv", index=False)
        process_dataset(sensitivity_df, "historical_trauma_index_sensitivity", self.processed_dir)
        process_dataset(sensitivity_summary, "historical_trauma_index_sensitivity_summary", self.processed_dir)

    def run_analysis(self) -> None:
        master = build_master_analysis_table(
            self.processed_dir / "historical_trauma_index.csv",
            self.processed_dir / "population.csv",
            self.processed_dir / "mortality.csv",
            self.processed_dir / "missing_persons.csv",
        )
        validate_main_analysis_scope(master)
        master.to_csv(self.processed_dir / "master_analysis_table.csv", index=False)
        process_dataset(master, "master_analysis_table", self.processed_dir)

        sample_audit = build_sample_characterization_table(
            self.processed_dir / "historical_trauma_index.csv",
            self.processed_dir / "population.csv",
            self.processed_dir / "mortality.csv",
            self.processed_dir / "missing_persons.csv",
        )
        sample_audit.to_csv(self.processed_dir / "sample_characterization.csv", index=False)
        process_dataset(sample_audit, "sample_characterization", self.processed_dir)

        inclusion_summary = summarize_included_vs_excluded(sample_audit)
        inclusion_summary.to_csv(self.processed_dir / "sample_characterization_summary.csv", index=False)
        process_dataset(inclusion_summary, "sample_characterization_summary", self.processed_dir)

        summary = descriptive_summary(master)
        summary.to_csv(self.processed_dir / "descriptive_summary.csv", index=False)

        associations = exploratory_association_table(
            master,
            target=self.analysis_config["index_name"],
            outcomes=self.analysis_config.get("outcomes"),
            confounders=self.analysis_config.get("confounders", []),
            bootstrap_iterations=int(self.analysis_config.get("bootstrap_iterations", 4000)),
            permutation_iterations=int(self.analysis_config.get("permutation_iterations", 10000)),
            seed=int(self.analysis_config.get("seed", 42)),
        )
        associations.to_csv(self.processed_dir / "exploratory_associations.csv", index=False)
        process_dataset(associations, "exploratory_associations", self.processed_dir)

        sensitivity = pd.read_csv(self.processed_dir / "historical_trauma_index_sensitivity.csv")
        generate_figures(master, sensitivity, self.vis_dir, target=self.analysis_config["index_name"])

        write_limitations_report(self.report_dir / "limitations_report.txt")
        self._write_methods_summary()

    def _write_methods_summary(self) -> None:
        combined = pd.read_csv(self.processed_dir / "combined_indicator_table.csv")
        indicator_diagnostics = pd.read_csv(self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv")
        mortality = pd.read_csv(self.processed_dir / "mortality.csv")

        indicator_summary = (
            combined.groupby("Indicator")
            .agg(
                Definition=("Definition", "first"),
                Source_Label=("Source_Label", lambda series: " | ".join(sorted({str(v) for v in series.dropna()}))),
            )
            .reset_index()
            .sort_values("Indicator")
        )

        payload = {
            "index_name": self.analysis_config["index_name"],
            "primary_normalization": self.analysis_config["primary_normalization"],
            "drop_constant_indicators": bool(self.analysis_config.get("drop_constant_indicators", True)),
            "bootstrap_iterations": int(self.analysis_config.get("bootstrap_iterations", 4000)),
            "permutation_iterations": int(self.analysis_config.get("permutation_iterations", 10000)),
            "confounders": self.analysis_config.get("confounders", []),
            "outcomes": self.analysis_config.get("outcomes", []),
            "indicator_construction": indicator_summary.to_dict("records"),
            "constant_indicators_within_sample": indicator_diagnostics.loc[
                indicator_diagnostics["Is_Constant_Within_Sample"] == True,
                ["Indicator", "Exclusion_Reason"],
            ].to_dict("records"),
            "mortality_conditions": sorted({str(value) for value in mortality.get("Condition", pd.Series(dtype=str)).dropna()}),
            "mortality_aggregation": "Mean_Mortality_Disparity_Ratio is the state-level mean of available Disparity_Ratio values across listed conditions.",
            "missing_persons_metrics": {
                "AI_AN_Percent_Missing": "100 * AI_AN_Missing / Total_Missing",
                "Overrepresentation_Ratio": "AI_AN_Percent_Missing / AI_AN_Population_Percent",
            },
            "ejscreen_note": "Environmental variables are taken from the supplied state-level indicator table and preserved with definitions/source labels in combined_indicator_table.csv.",
        }
        write_methods_summary(self.report_dir / "analysis_methods_summary.json", payload)

    def run(self) -> None:
        self.ingest()
        self.build_indices()
        self.run_analysis()

        provenance = build_provenance_report(self.processed_dir, self.manifest_dir, self.config_dir)
        (self.processed_dir / "provenance_report.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        self.logger.info("Pipeline complete.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ResearchPipeline(root).run()
