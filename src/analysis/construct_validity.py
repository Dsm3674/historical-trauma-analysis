"""Construct-validity diagnostics for the structural-exposure composite.

Adds the following items, all computed on the within-sample non-constant
indicators after z-score normalization:

  - inter-indicator Spearman correlation matrix
  - item-total rank correlation for each indicator (correlation between
    the indicator and the composite computed *without* it)
  - Cronbach's alpha
  - PCA: variance explained by each PC, loadings on PC1, KMO measure
    of sampling adequacy

All outputs are clearly labeled "diagnostic only" — at n=13 these
statistics have very wide sampling variability and should be read as
descriptive aids, not validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3:
        return float("nan")
    a, b = valid.iloc[:, 0], valid.iloc[:, 1]
    if method == "spearman":
        a = a.rank(method="average")
        b = b.rank(method="average")
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(a_c * b_c) / denom)


def inter_indicator_correlations(normalized: pd.DataFrame) -> pd.DataFrame:
    cols = list(normalized.columns)
    rows = []
    for a in cols:
        for b in cols:
            rho = _safe_corr(normalized[a], normalized[b], method="spearman")
            rows.append({"Indicator_A": a, "Indicator_B": b, "Spearman_Rho": rho})
    return pd.DataFrame(rows)


def item_total_correlations(normalized: pd.DataFrame) -> pd.DataFrame:
    cols = list(normalized.columns)
    rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            rows.append({"Indicator": col, "Item_Total_Rho": float("nan"),
                         "N_Other_Indicators": 0})
            continue
        composite_without = normalized[others].mean(axis=1, skipna=True)
        rho = _safe_corr(normalized[col], composite_without, method="spearman")
        rows.append({"Indicator": col, "Item_Total_Rho": rho,
                     "N_Other_Indicators": len(others)})
    return pd.DataFrame(rows)


def cronbach_alpha(normalized: pd.DataFrame) -> dict:
    """Return Cronbach's alpha and the standard caveats."""
    matrix = normalized.dropna(axis=0, how="any")
    n_items = matrix.shape[1]
    n_units = matrix.shape[0]
    if n_items < 2 or n_units < 3:
        return {"Cronbach_Alpha": float("nan"), "N_Items": n_items,
                "N_Units": n_units, "Note": "insufficient_data"}
    item_var = matrix.var(axis=0, ddof=1).sum()
    total_var = matrix.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return {"Cronbach_Alpha": float("nan"), "N_Items": n_items,
                "N_Units": n_units, "Note": "total_variance_zero"}
    alpha = (n_items / (n_items - 1.0)) * (1.0 - item_var / total_var)
    return {"Cronbach_Alpha": float(alpha), "N_Items": int(n_items),
            "N_Units": int(n_units), "Note": "diagnostic_only_at_small_n"}


def pca_diagnostics(normalized: pd.DataFrame) -> dict:
    matrix = normalized.dropna(axis=0, how="any").to_numpy(dtype=float)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return {"Variance_Explained": [], "PC1_Loadings": {},
                "Note": "insufficient_data"}
    centered = matrix - matrix.mean(axis=0)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    total_var = float((s ** 2).sum())
    var_explained = [float(x) for x in (s ** 2) / total_var] if total_var > 0 else []
    loadings = {col: float(vh[0, i])
                for i, col in enumerate(normalized.dropna(axis=0, how="any").columns)}
    return {"Variance_Explained": var_explained, "PC1_Loadings": loadings,
            "Note": "diagnostic_only_at_small_n"}


def kmo_measure(normalized: pd.DataFrame) -> dict:
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    KMO < 0.5 is conventionally read as "indicators do not share enough
    common variance to support factor extraction"; >= 0.7 is acceptable.
    """
    matrix = normalized.dropna(axis=0, how="any").to_numpy(dtype=float)
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return {"KMO": float("nan"), "Note": "insufficient_data"}
    # Pearson correlation matrix
    corr = np.corrcoef(matrix, rowvar=False)
    if np.any(np.isnan(corr)):
        return {"KMO": float("nan"), "Note": "correlation_matrix_has_nan"}
    try:
        inv = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return {"KMO": float("nan"), "Note": "correlation_matrix_singular"}
    # Partial correlation magnitudes
    d = np.sqrt(np.diag(inv))
    if np.any(d == 0):
        return {"KMO": float("nan"), "Note": "zero_diagonal_in_inverse"}
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    np.fill_diagonal(corr, 0.0)
    num = float(np.sum(corr ** 2))
    den = float(np.sum(corr ** 2) + np.sum(partial ** 2))
    if den == 0:
        return {"KMO": float("nan"), "Note": "denominator_zero"}
    return {"KMO": float(num / den), "Note": "diagnostic_only_at_small_n"}


def build_construct_validity_report(
    normalized_matrix_path: str | Path,
    indicator_diagnostics_path: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    output_summary: str | Path,
) -> dict:
    """Build all construct-validity outputs.

    Parameters
    ----------
    normalized_matrix_path
        Path to historical_trauma_index_normalized_matrix.csv (long with State).
    indicator_diagnostics_path
        Path to historical_trauma_index_indicator_diagnostics.csv to learn
        which indicators were retained.
    """
    normalized = pd.read_csv(normalized_matrix_path)
    diag = pd.read_csv(indicator_diagnostics_path)

    if "State" in normalized.columns:
        normalized = normalized.set_index("State")

    # Restrict to indicators included in the primary index
    if "Included_In_Primary_Index" in diag.columns:
        included = diag.loc[diag["Included_In_Primary_Index"].astype(bool), "Indicator"].tolist()
    else:
        included = list(normalized.columns)
    included = [c for c in included if c in normalized.columns]
    if len(included) < 2:
        Path(output_summary).write_text(
            "Construct validity not computable: fewer than 2 included indicators.\n",
            encoding="utf-8",
        )
        return {"computed": False, "reason": "insufficient_indicators"}

    sub = normalized[included]

    inter = inter_indicator_correlations(sub)
    item_total = item_total_correlations(sub)
    alpha = cronbach_alpha(sub)
    pca = pca_diagnostics(sub)
    kmo = kmo_measure(sub)

    # Long-format CSV combining the most useful items
    rows = []
    for _, r in item_total.iterrows():
        rows.append({"Metric": "Item_Total_Rho", "Indicator": r["Indicator"],
                     "Value": r["Item_Total_Rho"]})
    rows.append({"Metric": "Cronbach_Alpha", "Indicator": "OVERALL",
                 "Value": alpha["Cronbach_Alpha"]})
    rows.append({"Metric": "KMO", "Indicator": "OVERALL", "Value": kmo["KMO"]})
    for i, v in enumerate(pca["Variance_Explained"]):
        rows.append({"Metric": "PCA_Variance_Explained",
                     "Indicator": f"PC{i+1}", "Value": v})
    for ind, ld in pca["PC1_Loadings"].items():
        rows.append({"Metric": "PCA_PC1_Loading", "Indicator": ind, "Value": ld})
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    inter.to_csv(str(output_csv).replace(".csv", "_inter_indicator.csv"), index=False)

    payload = {
        "n_units": int(sub.dropna(axis=0, how="any").shape[0]),
        "n_indicators": int(len(included)),
        "indicators": included,
        "cronbach_alpha": alpha,
        "kmo": kmo,
        "pca": pca,
        "interpretation_caveat": (
            "All construct-validity statistics here are reported as diagnostic "
            "aids. At small ecological sample sizes (n<30) Cronbach's alpha, "
            "KMO, and PCA loadings have wide sampling variability and should "
            "not be used to either validate or invalidate the composite on "
            "their own."
        ),
    }
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plain-language summary
    lines = []
    lines.append("Construct validity report")
    lines.append("=========================")
    lines.append(f"Indicators in composite (n={payload['n_indicators']}): "
                 f"{', '.join(included)}")
    lines.append(f"Units (states) with complete indicator data: "
                 f"{payload['n_units']}")
    lines.append("")
    if not np.isnan(alpha["Cronbach_Alpha"]):
        lines.append(f"Cronbach's alpha = {alpha['Cronbach_Alpha']:.3f} "
                     f"(diagnostic only at small n).")
    if not np.isnan(kmo["KMO"]):
        lines.append(f"KMO sampling adequacy = {kmo['KMO']:.3f}.")
    if pca["Variance_Explained"]:
        pct = pca["Variance_Explained"][0] * 100
        lines.append(f"PC1 explains {pct:.1f}% of total variance across "
                     f"the included indicators.")
    lines.append("")
    lines.append("Item-total correlations (each indicator vs. mean of the "
                 "others):")
    for _, r in item_total.iterrows():
        v = r["Item_Total_Rho"]
        lines.append(f"  {r['Indicator']}: {v:.3f}"
                     if not np.isnan(v) else f"  {r['Indicator']}: NA")
    lines.append("")
    lines.append("Caveat: at the present sample size these statistics have "
                 "high variance and are reported as descriptive aids only.")
    Path(output_summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"computed": True, **payload}
