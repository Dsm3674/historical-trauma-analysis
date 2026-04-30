# Reviewer Response Support

This note maps the current code outputs to the main reviewer concerns for manuscript `PONE-D-26-19133`.

## Reviewer concerns addressed in code

- Normalization is now explicit and fixed in `config/analysis_config.json`.
  The primary run uses `zscore`, and the alternate `minmax` specification is reported in `data/processed/historical_trauma_index_sensitivity_summary.csv`.

- Non-varying indicators are now detected and documented automatically.
  See `data/processed/historical_trauma_index_indicator_diagnostics.csv`.
  Within the current local snapshot, `Federal_BoardingSchool_Presence` is constant and excluded from scored differentiation. The same diagnostics also show whether any other indicator is constant in the retained sample.

- Sensitivity analysis is broader than the manuscript's original equal-weight check.
  The pipeline now writes:
  - `data/processed/historical_trauma_index_sensitivity.csv`
  - `data/processed/historical_trauma_index_sensitivity_summary.csv`
  These include equal weights, alternate normalization, PCA-based scoring, and leave-one-indicator-out reruns.

- Bootstrap intervals are no longer the only uncertainty measure.
  `data/processed/exploratory_associations.csv` now includes:
  - Spearman rho
  - Kendall's tau-b
  - permutation p-values
  - partial rank correlations controlling for `AI_AN_Population_Percent`
  - leave-one-state-out influence diagnostics

- Compositional confounding is addressed more directly in code.
  The adjusted columns `Partial_Spearman_Adjusted_Rho` and `Partial_Adjusted_P_Value` quantify how associations change after controlling for `AI_AN_Population_Percent`.
  The main missing-persons outcome is now `AI_AN_Missing_Rate_per_100k_AI_AN` rather than the raw `AI_AN_Missing` count.

- Sample-selection auditing is now explicit.
  The pipeline writes:
  - `data/processed/sample_characterization.csv`
  - `data/processed/sample_characterization_summary.csv`
  These files identify which states are present across domains and whether they enter the complete-case analysis.

- Data-construction details are surfaced for methods reporting.
  `reports/analysis_methods_summary.json` records:
  - primary normalization
  - complete-case state count and state list
  - outcome-specific sample sizes
  - indicator definitions and source labels
  - constant indicators within sample
  - the weighting caveat
  - mortality condition list
  - mortality aggregation rule
  - missing-person outcome formulas
  - the environmental-indicator note tied to the combined indicator table

- Provenance and reproducibility reporting remain explicit.
  The pipeline writes dataset manifests, data dictionaries, missingness reports, summary statistics, and `data/processed/provenance_report.json`.

## Reviewer concerns only partly addressed by code

- The current code adjusts only for `AI_AN_Population_Percent`.
  Region, urbanization, income, or other state covariates are not available in the current local inputs.

- Excluded-state comparisons are structurally supported but only informative if excluded states exist in the local raw inputs.
  In the current local snapshot, the audit distinguishes 20 index states from the 13 complete-case states retained for association analyses.

- Figure generation is implemented in code. The v2 figure set has been rendered in both `visualizations/` and `figures/`.

## Manuscript text that should be updated manually

- If you use the new outputs, revise Methods and Results to mention permutation p-values, Kendall's tau-b, partial correlations, and leave-one-state-out diagnostics.
- If your actual repository raw inputs differ from this local snapshot, regenerate outputs before copying exact values into the manuscript.
- Check the constant-indicator section carefully after each rerun. The current code recovers DOI-style aggregate boarding-school counts from labels such as `(n=21)`, so `BoardingSchool_Count` should no longer be constant when those counts are present in the raw file.
