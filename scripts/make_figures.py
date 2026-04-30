import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from src.analysis.analysis import generate_figures

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
CONFIG = ROOT / "config" / "analysis_config.json"

def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    composite_column = config.get("composite_column_name", "Structural_Exposure_Composite")
    display_label = config.get("composite_display_name", "Structural Exposure Composite")

    master_path = PROCESSED / "master_analysis_table.csv"
    sensitivity_path = PROCESSED / "historical_trauma_index_sensitivity.csv"
    if not master_path.exists() or not sensitivity_path.exists():
        raise FileNotFoundError("Run python3 -m src.pipeline before generating figures.")

    master = pd.read_csv(master_path)
    sensitivity = pd.read_csv(sensitivity_path)
    generate_figures(master, sensitivity, FIGURES, target=composite_column, display_label=display_label)
    print("Saved figures to ./figures")


if __name__ == "__main__":
    main()
