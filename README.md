# Organized research pipeline

This version fixes the integration issues and adds a cleaner folder structure.

## Main fixes
- `analysis.py` now merges mortality by `State` instead of cross-joining one latest-year national average.
- `historical_trauma_index.csv` is the actual output used by the analysis layer.
- added `merge_indicators.py` so historical policy + environmental + boarding-school indicators can be combined before index construction.
- added `ingest_environmental.py`
- standardized processed filenames:
  - `population.csv`
  - `mortality.csv`
  - `missing_persons.csv`
  - `historical_trauma_index.csv`

## Folder structure
```text
plos_research_organized/
  config/
    indicator_weights.json
  data/
    raw/
      population/
      mortality/
      missing_persons/
      historical_policy/
      environmental/
      boarding_schools/
    processed/
    manifests/
  logs/
  reports/
  visualizations/
  src/
    ingest/
    features/
    analysis/
    reporting/
    utils/
    pipeline.py
```

## Recommended run order
1. Put your real raw files into `data/raw/...`
2. Run the dedicated ingestion scripts if you want standalone outputs
3. Run `src/pipeline.py`

## Raw files expected
- `data/raw/population/population.csv`
- `data/raw/mortality/mortality.csv`
- `data/raw/missing_persons/missing_persons.csv`
- `data/raw/historical_policy/historical_policy.csv`
- `data/raw/environmental/environmental_hazards.csv`

Optional standalone ingestion inputs:
- `data/raw/mortality/aian_mortality_export.csv`
- `data/raw/mortality/reference_mortality_export.csv`
- `data/raw/missing_persons/namus_missing_persons_export.csv`
- `data/raw/boarding_schools/boarding_school_listing.csv`

## Important note
Update `config/indicator_weights.json` so the keys exactly match your real indicator names.
