"""Dirichlet-sampled weight perturbation for the structural composite.

For each of N draws from Dirichlet(1, 1, ..., 1) over the included
indicators, recompute the composite, compute the Spearman rank
correlation against each outcome, and summarize the resulting
distribution.

Reported as descriptive only at small ecological n: the distribution
characterizes "how robust the composite-vs-outcome signal is to weight
choice", not classical inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analysis import _rank_corr


def _normalize_zscore(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        m = col[finite].mean()
        s = col[finite].std(ddof=0)
        if s == 0 or not np.isfinite(s):
            out[:, j] = 0.0
            continue
        out[:, j] = np.where(finite, (col - m) / s, np.nan)
    return out


def dirichlet_weight_perturbation(
    indicator_pivot: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    outcome_columns: list[str],
    n_iterations: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run weight perturbation analysis.

    Parameters
    ----------
    indicator_pivot
        Wide indicator table, index = State, columns = indicators.
        Constant indicators should already be removed.
    outcomes_df
        Table with State + outcome columns. Joined to the pivot by State.
    outcome_columns
        Outcomes to evaluate.

    Returns
    -------
    (per_iteration_df, summary_df)
    """
    rng = np.random.default_rng(seed)

    if "State" in indicator_pivot.columns:
        indicator_pivot = indicator_pivot.set_index("State")

    indicator_names = list(indicator_pivot.columns)
    if len(indicator_names) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Align outcomes to indicator pivot index
    outcomes = outcomes_df.set_index("State")
    common = indicator_pivot.index.intersection(outcomes.index)
    pivot = indicator_pivot.loc[common]
    outcomes = outcomes.loc[common]

    raw = pivot.to_numpy(dtype=float)
    z = _normalize_zscore(raw)

    iterations: list[dict] = []
    for it in range(n_iterations):
        weights = rng.dirichlet(np.ones(len(indicator_names)))
        weighted = np.nansum(z * weights[None, :], axis=1)
        not_nan_mask = np.any(np.isfinite(z), axis=1)
        weighted = np.where(not_nan_mask, weighted, np.nan)

        composite = pd.Series(weighted, index=pivot.index, name="composite")
        row: dict = {"iteration": it}
        for j, w in enumerate(weights):
            row[f"weight__{indicator_names[j]}"] = float(w)
        for outcome in outcome_columns:
            if outcome not in outcomes.columns:
                continue
            rho = _rank_corr(composite, outcomes[outcome])
            row[f"rho__{outcome}"] = rho
        iterations.append(row)

    per_iter = pd.DataFrame(iterations)

    # Summary: for each outcome, distribution of rho across weight space
    summary_rows = []
    for outcome in outcome_columns:
        col = f"rho__{outcome}"
        if col not in per_iter.columns:
            continue
        s = per_iter[col].dropna()
        if s.empty:
            continue
        summary_rows.append({
            "Outcome": outcome,
            "N_Iterations": int(len(s)),
            "Median_Rho": float(s.median()),
            "Mean_Rho": float(s.mean()),
            "P5_Rho": float(s.quantile(0.05)),
            "P95_Rho": float(s.quantile(0.95)),
            "Pct_Rho_Positive": float((s > 0).mean()),
            "Pct_Rho_GE_0p3": float((s >= 0.3).mean()),
            "Pct_Rho_GE_0p5": float((s >= 0.5).mean()),
            "Sign_Stable": bool((s > 0).all() or (s < 0).all()),
            "Note": "descriptive_only_not_for_inference",
        })
    summary = pd.DataFrame(summary_rows)
    return per_iter, summary
