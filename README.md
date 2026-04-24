# Historical Trauma Analysis Pipeline

This working copy rebuilds the exported research pipeline and adds reviewer-driven analysis safeguards:

- explicit normalization configuration through `config/analysis_config.json`
- automatic detection and exclusion of non-varying indicators from within-sample scoring
- richer sensitivity analysis across weighting schemes, normalization choices, PCA, and leave-one-indicator-out variants
- permutation-based inference, Kendall's tau, partial rank correlations, and leave-one-state-out diagnostics
- sample inclusion audits, outcome construction summaries, and reviewer-friendly figures

## Run

```bash
python3 -m pip install numpy pandas matplotlib
python3 -m src.pipeline
```

Outputs are written to:

- `data/processed/`
- `data/manifests/`
- `reports/`
- `visualizations/`

## Current default choices

- Primary normalization: `zscore`
- Constant indicators: dropped from scored index, but still documented in diagnostics
- Primary confounder for adjusted analyses: `AI_AN_Population_Percent`
- Primary missing-person outcome: `AI_AN_Missing_Rate_per_100k_AI_AN`
- Boarding-school counts: recovered from DOI-style aggregated labels such as `(n=21)` when present
- Sensitivity variants: equal weights, alternate normalization, PCA-based score, and leave-one-indicator-out reruns

These defaults are meant to address the current reviewer comments in code. They can be changed without editing Python by updating `config/analysis_config.json` or `config/indicator_weights.json`.
