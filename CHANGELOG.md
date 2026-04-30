# CHANGELOG

## 2.0.0 — Pipeline upgrade and reviewer response

This release responds to reviewer feedback on the original manuscript and adds a substantially expanded mortality dataset.

### Composite renaming (Phase 1)
- The composite is renamed from `Historical_Trauma_Index` to **`Structural_Exposure_Composite`**, since at the construct level it summarizes observable structural indicators rather than measuring lived historical trauma.
- The legacy column name `Historical_Trauma_Index` is preserved as a backward-compatible alias on every output (master table and scores files), so any downstream script that hard-codes the old column will continue to work without modification.
- Display label, column name, and alias are configured in `config/analysis_config.json`.

### Equal-weight as a co-primary specification (Phase 2)
- New config flag: `primary_weighting_scheme` ∈ {`configured`, `equal`, `both`}, default `both`.
- When `both`, the configured-weight composite is the canonical primary (preserving file paths) and the equal-weight composite is written alongside as a co-primary, since equal weights produce a more stable rank ordering at small n. Both feed `exploratory_associations.csv` and `exploratory_associations_equal_weights.csv`.

### Construct-validity diagnostics (Phase 3) — new module
- `src/analysis/construct_validity.py`: inter-indicator Spearman correlation matrix, item-total rank correlations (each indicator vs. mean of the others), Cronbach's alpha, KMO measure of sampling adequacy, and PCA loadings on PC1 with variance explained.
- All values labeled descriptive-only at small ecological n.

### Bivariate indicator-vs-outcome analysis (Phase 4) — new module
- `src/analysis/bivariate.py`: each (indicator, outcome) pair gets a Spearman ρ, Kendall τ-b, permutation p-value, partial rank correlation adjusting for AI/AN population share, and Holm-Bonferroni and BH-FDR adjusted p within outcome family.
- Output: `data/processed/bivariate_indicator_associations.csv`.

### Bootstrap CI relabeling (Phase 5)
- Bootstrap CI columns renamed to `Descriptive_Bootstrap_CI_Lower` / `Upper` and a new `CI_Interpretation` column makes the descriptive role explicit (it is no longer pasted next to permutation p-values without label).

### Selection-bias audit (Phase 6) — new module
- `src/analysis/selection_audit.py`: Mann-Whitney U with permutation-based p-values testing whether the complete-case sample differs systematically from excluded states on AI/AN population share, AI/AN population, and total population.
- Output: `data/processed/selection_bias_audit.csv`.

### Dirichlet weight perturbation (Phase 7) — new module
- `src/analysis/weight_perturbation.py`: N draws (default 2000) from Dirichlet(1, ..., 1) over the included indicators, recomputes the composite each time, and reports the distribution of the resulting Spearman ρ against each outcome.
- Outputs: `data/processed/weight_perturbation_iterations.csv` and `weight_perturbation_summary.csv` (median, P5, P95, percent positive, sign-stable).

### Multiple-testing adjustments (Phase 8) — new module
- `src/analysis/multiple_testing.py`: dependency-free Holm-Bonferroni and Benjamini-Hochberg, NaN-preserving.
- Applied within the primary outcome family (configurable via `primary_outcomes_for_multiple_testing`).
- New columns on `exploratory_associations.csv`: `Holm_Adjusted_P_Primary_Family`, `BH_FDR_Adjusted_P_Primary_Family`.

### Code-quality improvements (Phase 9)
- `MIN_UNITS_FOR_INFERENCE` and `MIN_UNITS_FOR_MAIN_MANUSCRIPT` are now configurable via `analysis_config.json` (`min_units_for_inference`, `min_units_for_main_manuscript`).
- `kendall_tau_b` rewritten with numpy broadcasting (~50× faster for the bootstrap and permutation paths).
- `pipeline_version` field added to `analysis_methods_summary.json`.

### Headline findings JSON (Phase 10)
- `reports/headline_findings.json`: a single-stop summary with one entry per outcome (n, Spearman ρ, Kendall τ, raw p, partial ρ and p after population adjustment, Holm and BH adjusted p, evidence label, significance after multiple testing) plus a plain-language narrative summary.

### Mortality data expansion
- Mortality data replaced with CDC WONDER 2018-2024 Single Race exports for three causes — Diabetes (E10-E14), Liver Disease (K70-K76), and Suicide — across years 2019-2022.
- 28 states with at least one (year, condition) cell after dropping suppressed rows, 17 with all three conditions, vs. 13 single-condition diabetes-only previously.
- Aggregation rule (in `build_master_analysis_table`):
  1. Per `(State, Condition)`, average `Disparity_Ratio` across years.
  2. `Mean_Mortality_Disparity_Ratio` is the equal-weighted mean of per-condition values, giving each condition equal weight regardless of year-availability.
  3. Per-condition columns are also written: `Disparity_Ratio_Diabetes`, `Disparity_Ratio_Liver_Disease`, `Disparity_Ratio_Suicide`.

### Limitations report (rewritten)
- `reports/limitations_report.txt` now reflects the new framing, the multi-condition mortality aggregation, the explicit descriptive-vs-inferential split, and the construct-validity diagnostic caveats.

### File-path stability
- `historical_trauma_index.csv` continues to be written as the primary scores file (the column inside it is now `Structural_Exposure_Composite` plus the `Historical_Trauma_Index` alias). This preserves all downstream scripts and figure-generation paths.

### Configuration changes (`config/analysis_config.json`)
- Added: `composite_display_name`, `composite_column_name`, `legacy_alias_column`, `primary_weighting_scheme`, `weight_perturbation_iterations`, `min_units_for_inference`, `min_units_for_main_manuscript`, `primary_outcomes_for_multiple_testing`, `construct_validity`, `pipeline_version`.
- Expanded `outcomes` to include `Disparity_Ratio_Diabetes`, `Disparity_Ratio_Liver_Disease`, `Disparity_Ratio_Suicide`.
- Sensitivity block now includes `include_dirichlet_weight_perturbation` and `include_temporal_only_2020`.

### Note on boarding-school per-school dates
- The DOI Federal Indian Boarding School Initiative Investigative Report (Volume II) is the report narrative; per-school open/close dates are in companion appendices that are not bundled with that report. Pipeline retains aggregate-era duration estimates with the existing caveat in the limitations report.
