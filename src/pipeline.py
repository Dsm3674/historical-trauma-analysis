from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
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
from src.analysis.bivariate import write_bivariate_associations
from src.analysis.construct_validity import build_construct_validity_report
from src.analysis.selection_audit import selection_bias_audit
from src.analysis.weight_perturbation import dirichlet_weight_perturbation
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

    @property
    def composite_column(self) -> str:
        return self.analysis_config.get("composite_column_name", "Structural_Exposure_Composite")

    @property
    def composite_display(self) -> str:
        return self.analysis_config.get("composite_display_name", "Structural Exposure Composite")

    @property
    def legacy_alias(self) -> str:
        return self.analysis_config.get("legacy_alias_column", "Historical_Trauma_Index")

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
        weighting_scheme = self.analysis_config.get("primary_weighting_scheme", "configured")

        # Build the configured-weight composite (the historical primary)
        configured = compute_index(
            combined, weights,
            normalization=self.analysis_config["primary_normalization"],
            index_name=self.composite_column,
            drop_constant_indicators=bool(self.analysis_config.get("drop_constant_indicators", True)),
            legacy_alias_column=self.legacy_alias,
        )

        # Always compute equal weights too (cheap; used as co-primary or fallback)
        all_indicators = list(combined["Indicator"].astype(str).str.strip().unique())
        equal_weights = {ind: 1.0 for ind in all_indicators}
        equal = compute_index(
            combined, equal_weights,
            normalization=self.analysis_config["primary_normalization"],
            index_name=self.composite_column,
            drop_constant_indicators=bool(self.analysis_config.get("drop_constant_indicators", True)),
            legacy_alias_column=self.legacy_alias,
        )

        # Write the chosen primary as historical_trauma_index.csv (preserves
        # downstream filenames). When 'both', configured is the canonical
        # primary file and equal-weight is written alongside as a co-primary
        # for readers/scripts that prefer the more stable specification.
        if weighting_scheme == "equal":
            primary = equal
        else:
            primary = configured

        primary.scores.to_csv(self.processed_dir / "historical_trauma_index.csv", index=False)
        primary.normalized_matrix.to_csv(
            self.processed_dir / "historical_trauma_index_normalized_matrix.csv", index=False,
        )
        primary.indicator_diagnostics.to_csv(
            self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv", index=False,
        )
        process_dataset(primary.scores, "historical_trauma_index", self.processed_dir)
        process_dataset(primary.normalized_matrix, "historical_trauma_index_normalized_matrix", self.processed_dir)
        process_dataset(primary.indicator_diagnostics, "historical_trauma_index_indicator_diagnostics", self.processed_dir)

        if weighting_scheme == "both":
            # Write the alternate as well, clearly named
            equal.scores.to_csv(self.processed_dir / "structural_composite_equal_weights.csv", index=False)
            process_dataset(equal.scores, "structural_composite_equal_weights", self.processed_dir)
            configured.scores.to_csv(self.processed_dir / "structural_composite_configured_weights.csv", index=False)
            process_dataset(configured.scores, "structural_composite_configured_weights", self.processed_dir)

        # Sensitivity analysis (uses configured weights as the named primary)
        sensitivity_df, sensitivity_summary = sensitivity_analysis(
            combined, weights,
            primary_normalization=self.analysis_config["primary_normalization"],
            alternate_normalizations=self.analysis_config.get("sensitivity", {}).get("alternate_normalizations", []),
            include_equal_weights=bool(self.analysis_config.get("sensitivity", {}).get("include_equal_weights", True)),
            include_pca=bool(self.analysis_config.get("sensitivity", {}).get("include_pca", True)),
            leave_one_indicator_out=bool(self.analysis_config.get("sensitivity", {}).get("leave_one_indicator_out", True)),
            index_name=self.composite_column,
            include_temporal_only_2020=bool(self.analysis_config.get("sensitivity", {}).get("include_temporal_only_2020", False)),
            temporal_only_indicators=[
                "Mean_BoardingSchool_Duration_Years",
                "Max_BoardingSchool_Duration_Years",
                "Schools_With_Burial_Site_Flag",
            ],
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
            composite_column_name=self.composite_column,
        )
        validate_main_analysis_scope(
            master,
            min_units=int(self.analysis_config.get("min_units_for_main_manuscript", 10)),
        )
        master.to_csv(self.processed_dir / "master_analysis_table.csv", index=False)
        process_dataset(master, "master_analysis_table", self.processed_dir)

        sample_audit = build_sample_characterization_table(
            self.processed_dir / "historical_trauma_index.csv",
            self.processed_dir / "population.csv",
            self.processed_dir / "mortality.csv",
            self.processed_dir / "missing_persons.csv",
            composite_column_name=self.composite_column,
        )
        sample_audit.to_csv(self.processed_dir / "sample_characterization.csv", index=False)
        process_dataset(sample_audit, "sample_characterization", self.processed_dir)

        inclusion_summary = summarize_included_vs_excluded(sample_audit)
        inclusion_summary.to_csv(self.processed_dir / "sample_characterization_summary.csv", index=False)
        process_dataset(inclusion_summary, "sample_characterization_summary", self.processed_dir)

        # Phase 6: formal selection-bias audit
        selection_audit_df = selection_bias_audit(
            sample_audit,
            self.processed_dir / "population.csv",
            n_perm=int(self.analysis_config.get("permutation_iterations", 10000)),
            seed=int(self.analysis_config.get("seed", 42)),
        )
        selection_audit_df.to_csv(self.processed_dir / "selection_bias_audit.csv", index=False)
        process_dataset(selection_audit_df, "selection_bias_audit", self.processed_dir)

        summary = descriptive_summary(master)
        summary.to_csv(self.processed_dir / "descriptive_summary.csv", index=False)

        # Primary association table (configured-weight composite by default)
        associations = exploratory_association_table(
            master,
            target=self.composite_column,
            outcomes=self.analysis_config.get("outcomes"),
            confounders=self.analysis_config.get("confounders", []),
            bootstrap_iterations=int(self.analysis_config.get("bootstrap_iterations", 4000)),
            permutation_iterations=int(self.analysis_config.get("permutation_iterations", 10000)),
            seed=int(self.analysis_config.get("seed", 42)),
            primary_outcomes_for_multiple_testing=self.analysis_config.get("primary_outcomes_for_multiple_testing"),
            min_units_for_inference=int(self.analysis_config.get("min_units_for_inference", 20)),
        )
        associations.to_csv(self.processed_dir / "exploratory_associations.csv", index=False)
        process_dataset(associations, "exploratory_associations", self.processed_dir)

        # Co-primary equal-weight associations when configured
        if self.analysis_config.get("primary_weighting_scheme") == "both":
            eq_path = self.processed_dir / "structural_composite_equal_weights.csv"
            if eq_path.exists():
                # Build a master table where the composite column is replaced
                # with equal-weight scores, leaving outcomes intact
                eq_scores = pd.read_csv(eq_path)[["State", self.composite_column]].rename(
                    columns={self.composite_column: f"{self.composite_column}_EqualWeights"}
                )
                eq_master = master.drop(columns=[self.composite_column], errors="ignore").merge(
                    eq_scores.rename(columns={f"{self.composite_column}_EqualWeights": self.composite_column}),
                    on="State", how="left",
                )
                eq_associations = exploratory_association_table(
                    eq_master,
                    target=self.composite_column,
                    outcomes=self.analysis_config.get("outcomes"),
                    confounders=self.analysis_config.get("confounders", []),
                    bootstrap_iterations=int(self.analysis_config.get("bootstrap_iterations", 4000)),
                    permutation_iterations=int(self.analysis_config.get("permutation_iterations", 10000)),
                    seed=int(self.analysis_config.get("seed", 42)),
                    primary_outcomes_for_multiple_testing=self.analysis_config.get("primary_outcomes_for_multiple_testing"),
                    min_units_for_inference=int(self.analysis_config.get("min_units_for_inference", 20)),
                )
                eq_associations["Weighting_Scheme"] = "equal"
                eq_associations.to_csv(self.processed_dir / "exploratory_associations_equal_weights.csv", index=False)
                process_dataset(eq_associations, "exploratory_associations_equal_weights", self.processed_dir)

        # Phase 4: bivariate per-indicator associations
        try:
            write_bivariate_associations(
                indicator_table_path=self.processed_dir / "combined_indicator_table.csv",
                master_path=self.processed_dir / "master_analysis_table.csv",
                output_path=self.processed_dir / "bivariate_indicator_associations.csv",
                outcomes=self.analysis_config.get("outcomes", []),
                confounders=self.analysis_config.get("confounders", []),
                bootstrap_iterations=int(self.analysis_config.get("bootstrap_iterations", 4000)),
                permutation_iterations=int(self.analysis_config.get("permutation_iterations", 10000)),
                seed=int(self.analysis_config.get("seed", 42)),
            )
            biv = pd.read_csv(self.processed_dir / "bivariate_indicator_associations.csv")
            process_dataset(biv, "bivariate_indicator_associations", self.processed_dir)
        except Exception as exc:
            self.logger.warning("Bivariate analysis failed: %s", exc)

        # Phase 3: construct validity diagnostics
        if self.analysis_config.get("construct_validity", {}).get("compute", True):
            try:
                build_construct_validity_report(
                    normalized_matrix_path=self.processed_dir / "historical_trauma_index_normalized_matrix.csv",
                    indicator_diagnostics_path=self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv",
                    output_csv=self.processed_dir / "construct_validity_report.csv",
                    output_json=self.report_dir / "construct_validity_report.json",
                    output_summary=self.report_dir / "construct_validity_summary.txt",
                )
                cv = pd.read_csv(self.processed_dir / "construct_validity_report.csv")
                process_dataset(cv, "construct_validity_report", self.processed_dir)
            except Exception as exc:
                self.logger.warning("Construct validity computation failed: %s", exc)

        # Phase 7: Dirichlet weight perturbation
        if self.analysis_config.get("sensitivity", {}).get("include_dirichlet_weight_perturbation", True):
            try:
                self._run_weight_perturbation(master)
            except Exception as exc:
                self.logger.warning("Weight perturbation failed: %s", exc)

        # Sensitivity figures
        sensitivity = pd.read_csv(self.processed_dir / "historical_trauma_index_sensitivity.csv")
        generate_figures(
            master, sensitivity, self.vis_dir,
            target=self.composite_column,
            display_label=self.composite_display,
        )

        main_analysis_state_count = int(master.loc[master["Included_In_Complete_Case"], "State"].nunique())
        index_state_count = int(master["State"].nunique())
        write_limitations_report(
            self.report_dir / "limitations_report.txt",
            main_analysis_state_count=main_analysis_state_count,
            index_state_count=index_state_count,
        )
        self._write_methods_summary()
        self._write_headline_findings(master, associations)

    def _run_weight_perturbation(self, master: pd.DataFrame) -> None:
        diagnostics = pd.read_csv(
            self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv"
        )
        included = diagnostics.loc[
            diagnostics["Included_In_Primary_Index"].astype(bool), "Indicator"
        ].tolist()
        if len(included) < 2:
            return

        combined = pd.read_csv(self.processed_dir / "combined_indicator_table.csv")
        pivot = combined.pivot(index="State", columns="Indicator", values="Value").reset_index()
        pivot = pivot[["State"] + [c for c in included if c in pivot.columns]]

        per_iter, summary = dirichlet_weight_perturbation(
            indicator_pivot=pivot,
            outcomes_df=master,
            outcome_columns=[c for c in self.analysis_config.get("outcomes", []) if c in master.columns],
            n_iterations=int(self.analysis_config.get("weight_perturbation_iterations", 2000)),
            seed=int(self.analysis_config.get("seed", 42)),
        )
        per_iter.to_csv(self.processed_dir / "weight_perturbation_iterations.csv", index=False)
        summary.to_csv(self.processed_dir / "weight_perturbation_summary.csv", index=False)
        if not summary.empty:
            process_dataset(summary, "weight_perturbation_summary", self.processed_dir)

    def _write_methods_summary(self) -> None:
        combined = pd.read_csv(self.processed_dir / "combined_indicator_table.csv")
        indicator_diagnostics = pd.read_csv(self.processed_dir / "historical_trauma_index_indicator_diagnostics.csv")
        mortality = pd.read_csv(self.processed_dir / "mortality.csv")
        master = pd.read_csv(self.processed_dir / "master_analysis_table.csv")

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
            "pipeline_version": self.analysis_config.get("pipeline_version", "2.0.0"),
            "composite_display_name": self.composite_display,
            "composite_column_name": self.composite_column,
            "legacy_alias_column": self.legacy_alias,
            "primary_weighting_scheme": self.analysis_config.get("primary_weighting_scheme", "configured"),
            "primary_normalization": self.analysis_config["primary_normalization"],
            "drop_constant_indicators": bool(self.analysis_config.get("drop_constant_indicators", True)),
            "bootstrap_iterations": int(self.analysis_config.get("bootstrap_iterations", 4000)),
            "permutation_iterations": int(self.analysis_config.get("permutation_iterations", 10000)),
            "weight_perturbation_iterations": int(self.analysis_config.get("weight_perturbation_iterations", 2000)),
            "min_units_for_inference": int(self.analysis_config.get("min_units_for_inference", 20)),
            "min_units_for_main_manuscript": int(self.analysis_config.get("min_units_for_main_manuscript", 10)),
            "seed": int(self.analysis_config.get("seed", 42)),
            "confounders": self.analysis_config.get("confounders", []),
            "outcomes": self.analysis_config.get("outcomes", []),
            "primary_outcomes_for_multiple_testing": self.analysis_config.get("primary_outcomes_for_multiple_testing", []),
            "index_state_count": int(master["State"].nunique()),
            "complete_case_state_count": int(master.loc[master["Included_In_Complete_Case"], "State"].nunique()),
            "complete_case_states": master.loc[master["Included_In_Complete_Case"], "State"].sort_values().tolist(),
            "outcome_sample_sizes": {
                outcome: int(master[[self.composite_column, outcome]].dropna().shape[0])
                for outcome in self.analysis_config.get("outcomes", [])
                if outcome in master.columns
            },
            "indicator_construction": indicator_summary.to_dict("records"),
            "constant_indicators_within_sample": indicator_diagnostics.loc[
                indicator_diagnostics["Is_Constant_Within_Sample"] == True,
                ["Indicator", "Exclusion_Reason"],
            ].to_dict("records"),
            "weighting_note": (
                "Configured weights are heuristic and not community-informed. "
                "When primary_weighting_scheme is 'both', the equal-weight composite "
                "is reported as a co-primary specification because at small n it "
                "produces a more stable rank ordering."
            ),
            "construct_validity_note": (
                "Construct-validity diagnostics in construct_validity_report.csv "
                "are descriptive aids only; at small ecological n the underlying "
                "statistics have wide sampling variability."
            ),
            "mortality_conditions": sorted({str(value) for value in mortality.get("Condition", pd.Series(dtype=str)).dropna()}),
            "mortality_aggregation": (
                "Per (State, Condition), Disparity_Ratio is averaged across years. "
                "Mean_Mortality_Disparity_Ratio is then the equal-weighted mean of "
                "per-condition values, giving each condition equal weight regardless "
                "of year-availability. Per-condition columns are also written "
                "(Disparity_Ratio_<Condition>)."
            ),
            "missing_persons_metrics": {
                "AI_AN_Missing": "Primary outcome (absolute count); see MMIR undercounting literature.",
                "AI_AN_Missing_Rate_per_100k_AI_AN": "Secondary, population-adjusted: 100000 * AI_AN_Missing / AI_AN_Population",
            },
            "boarding_school_note": "BoardingSchool_Count uses aggregated counts extracted from DOI-style source labels when available.",
            "ejscreen_note": "Environmental variables are state-level and not AI/AN-specific; temporally misaligned with 2020 outcomes.",
            "naming_change_note": (
                f"The composite was previously named 'Historical Trauma Index'. "
                f"It has been renamed to '{self.composite_display}' to better "
                f"reflect what it measures: a summary of observable structural "
                f"indicators, not lived historical trauma. The legacy column "
                f"'{self.legacy_alias}' is retained as a backward-compatible alias."
            ),
        }
        write_methods_summary(self.report_dir / "analysis_methods_summary.json", payload)

    def _write_headline_findings(self, master: pd.DataFrame, associations: pd.DataFrame) -> None:
        """Phase 10: a single JSON summarizing the headline empirical results."""
        complete_case_n = int(master.loc[master["Included_In_Complete_Case"], "State"].nunique())
        primary_outcomes = self.analysis_config.get("primary_outcomes_for_multiple_testing", [])

        findings = {
            "pipeline_version": self.analysis_config.get("pipeline_version", "2.0.0"),
            "composite_display_name": self.composite_display,
            "complete_case_state_count": complete_case_n,
            "by_outcome": [],
        }
        for _, row in associations.iterrows():
            outcome = row["Outcome"]
            entry = {
                "outcome": outcome,
                "is_primary_family_member": outcome in primary_outcomes,
                "n": int(row["N"]),
                "spearman_rho": float(row["Effect_Size"]) if pd.notna(row["Effect_Size"]) else None,
                "kendall_tau_b": float(row["Kendall_Tau_b"]) if pd.notna(row["Kendall_Tau_b"]) else None,
                "permutation_p_value": float(row["Permutation_P_Value"]) if pd.notna(row["Permutation_P_Value"]) else None,
                "partial_rho_after_population_adjustment": float(row["Partial_Spearman_Adjusted_Rho"]) if pd.notna(row["Partial_Spearman_Adjusted_Rho"]) else None,
                "partial_p_after_population_adjustment": float(row["Partial_Adjusted_P_Value"]) if pd.notna(row["Partial_Adjusted_P_Value"]) else None,
                "holm_adjusted_p_primary_family": float(row["Holm_Adjusted_P_Primary_Family"]) if "Holm_Adjusted_P_Primary_Family" in row and pd.notna(row["Holm_Adjusted_P_Primary_Family"]) else None,
                "bh_fdr_adjusted_p_primary_family": float(row["BH_FDR_Adjusted_P_Primary_Family"]) if "BH_FDR_Adjusted_P_Primary_Family" in row and pd.notna(row["BH_FDR_Adjusted_P_Primary_Family"]) else None,
                "evidence_label": row.get("Evidence_Label", ""),
            }
            # Significance after multiple-testing adjustment
            holm_p = entry["holm_adjusted_p_primary_family"]
            entry["significant_at_0p05_after_holm"] = bool(holm_p is not None and holm_p < 0.05)
            findings["by_outcome"].append(entry)

        findings["narrative_summary"] = self._build_narrative_summary(findings)

        out_path = self.report_dir / "headline_findings.json"
        out_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    @staticmethod
    def _build_narrative_summary(findings: dict) -> str:
        lines = []
        n = findings["complete_case_state_count"]
        lines.append(
            f"In a complete-case sample of {n} states, the {findings['composite_display_name']} "
            "was tested against each pre-specified outcome using permutation-based inference. "
            "The Holm-adjusted p-value within the primary outcome family is reported alongside "
            "each association."
        )
        any_significant = False
        for entry in findings["by_outcome"]:
            if not entry.get("is_primary_family_member"):
                continue
            rho = entry.get("spearman_rho")
            p = entry.get("permutation_p_value")
            holm = entry.get("holm_adjusted_p_primary_family")
            if rho is None or p is None:
                continue
            sig = entry.get("significant_at_0p05_after_holm")
            any_significant = any_significant or sig
            holm_str = f"{holm:.3f}" if holm is not None else "NA"
            lines.append(
                f" - {entry['outcome']}: rho = {rho:.3f}, perm p = {p:.3f}, "
                f"Holm-adjusted p = {holm_str}, n = {entry['n']}."
            )
        if any_significant:
            lines.append(
                "At least one primary association remained significant after multiple-testing "
                "adjustment within the primary family."
            )
        else:
            lines.append(
                "No primary association reached statistical significance after multiple-testing "
                "adjustment within the primary family. This is interpreted as a power-limited "
                "null result rather than evidence against the underlying theoretical framework."
            )
        return " ".join(lines)

    def run(self) -> None:
        self.ingest()
        self.build_indices()
        self.run_analysis()

        provenance = build_provenance_report(self.processed_dir, self.manifest_dir, self.config_dir)
        (self.processed_dir / "provenance_report.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8",
        )
        self.logger.info("Pipeline complete.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ResearchPipeline(root).run()
