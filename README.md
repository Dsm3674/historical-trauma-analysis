# Structural Exposure Composite — Reproducible Pipeline (v2.0.0)

A reproducible computational framework for ecological analysis of state-level
structural exposure indicators and contemporary disparities in Indigenous
communities.

> **Naming.** The composite previously called *Historical Trauma Index* has
> been renamed to **Structural Exposure Composite** to better reflect what it
> measures: a summary of observable structural indicators, not lived
> historical trauma. The legacy column `Historical_Trauma_Index` is preserved
> as a backward-compatible alias on every output.

## What's new in 2.0.0

This release responds to reviewer feedback. See [`CHANGELOG.md`](CHANGELOG.md)
for the full list of changes.

| Phase | Change | New file or column |
| --- | --- | --- |
| 1 | Composite renamed → `Structural_Exposure_Composite` | composite & legacy alias on every output |
| 2 | Equal-weight composite as **co-primary** | `structural_composite_equal_weights.csv`, `exploratory_associations_equal_weights.csv` |
| 3 | Construct-validity diagnostics | `construct_validity_report.csv`, `reports/construct_validity_summary.txt` |
| 4 | Bivariate indicator-vs-outcome view | `bivariate_indicator_associations.csv` |
| 5 | Bootstrap CI clearly relabeled | `Descriptive_Bootstrap_CI_*` + `CI_Interpretation` |
| 6 | Selection-bias audit (Mann-Whitney + permutation) | `selection_bias_audit.csv` |
| 7 | Dirichlet weight-perturbation sensitivity | `weight_perturbation_summary.csv` |
| 8 | Multiple-testing adjustments (Holm + BH-FDR) | `Holm_Adjusted_P_Primary_Family`, `BH_FDR_Adjusted_P_Primary_Family` |
| 9 | Configurable thresholds + vectorized Kendall τ | `min_units_for_inference` config knob |
| 10 | Headline findings JSON | `reports/headline_findings.json` |

In addition, the **mortality dataset has been replaced** with CDC WONDER
2018-2024 Single Race exports for three causes (Diabetes E10-E14, Liver Disease
K70-K76, Suicide) across years 2019-2022. Coverage rises from 13 single-condition
states to 28 states with at least one (year, condition) cell and 17 states with
all three conditions. See [CHANGELOG](CHANGELOG.md) for the aggregation rule.

## Run

```bash
python3 -m pip install numpy pandas matplotlib
python3 -m src.pipeline
```

Outputs are written to:

- `data/processed/` — analysis tables, indicator-level diagnostics, and per-table provenance
- `data/manifests/` — raw-data manifests with hashes
- `reports/` — `headline_findings.json`, `analysis_methods_summary.json`,
  `limitations_report.txt`, `construct_validity_summary.txt`,
  `reviewer_response_support.md`
- `visualizations/` — figures

## Current defaults

- Composite display name: **Structural Exposure Composite**
- Primary weighting scheme: **`both`** — configured weights remain the
  manuscript primary (preserving file paths and figures); the equal-weight
  composite is reported alongside as a co-primary specification, since at
  small n equal weights produce a more stable rank ordering.
- Primary normalization: `zscore`
- Constant indicators: dropped from scored index, documented in diagnostics
- Primary confounder for adjusted analyses: `AI_AN_Population_Percent`
- Primary missing-person outcomes: `AI_AN_Missing` (count) and
  `AI_AN_Missing_Rate_per_100k_AI_AN` (rate)
- Multiple-testing family: `Mean_Mortality_Disparity_Ratio` and
  `AI_AN_Missing_Rate_per_100k_AI_AN` (configurable in
  `analysis_config.json`)
- Sensitivity variants: equal weights, alternate normalization, PCA-based
  score, leave-one-indicator-out, Dirichlet weight perturbation, and
  temporal-only (2020-era indicators).

These defaults are configured in
[`config/analysis_config.json`](config/analysis_config.json) and
[`config/indicator_weights.json`](config/indicator_weights.json) and can be
changed without editing Python.

## Interpretation guidance

- All correlation results are **ecological** and must not be interpreted as
  individual-level evidence.
- At the current sample size (10s of states), bootstrap intervals are reported
  as **descriptive only** (CI columns are explicitly labeled). **Permutation
  p-values** are the primary inferential statistic, with **Holm and BH-FDR
  adjustments** within the primary outcome family.
- Construct-validity diagnostics (Cronbach α, KMO, PCA loadings, item-total
  correlations) are reported as **descriptive aids only**. At small n these
  statistics have wide sampling variability.
- The composite is **not** a measure of lived historical trauma. It is a
  summary of observable structural indicators, retained mainly to enable
  rank-based comparison and sensitivity analysis.
- Adjusted analyses control only for AI/AN population share. Urbanization,
  income, region, and IHS-facility proximity remain plausible omitted
  variables.

## Repository layout

```
src/
  ingest/         # raw-data loaders + boarding school feature builder
  features/       # composite construction + sensitivity analysis
  analysis/       # association tables + 4 new modules:
                  #   construct_validity.py    (Phase 3)
                  #   bivariate.py             (Phase 4)
                  #   selection_audit.py       (Phase 6)
                  #   weight_perturbation.py   (Phase 7)
                  #   multiple_testing.py      (Phase 8)
  reporting/      # provenance, manifests
  utils/          # logging, data-dictionary helpers
config/
  analysis_config.json     # all behavioral knobs in one place
  indicator_weights.json   # configured composite weights
data/
  raw/            # source CSVs
  processed/      # everything the pipeline writes (also data dictionaries
                  #   and missing reports per table)
  manifests/      # raw-data hashes
reports/          # human-readable outputs
visualizations/   # PNG figures
```
