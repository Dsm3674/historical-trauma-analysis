"""Formal selection-bias audit between included and excluded states.

Tests whether the complete-case sample differs systematically from the
excluded states on observable structural characteristics. Uses Mann-Whitney
U with permutation-based p-values rather than asymptotic approximations,
appropriate for small samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def mann_whitney_u(included: np.ndarray, excluded: np.ndarray) -> float:
    """Compute the Mann-Whitney U statistic for `included`."""
    if len(included) == 0 or len(excluded) == 0:
        return float("nan")
    combined = np.concatenate([included, excluded])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    r1 = ranks[: len(included)].sum()
    n1 = len(included)
    u1 = r1 - n1 * (n1 + 1) / 2
    return float(u1)


def mann_whitney_permutation_p(
    included: np.ndarray,
    excluded: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sided permutation p-value for the Mann-Whitney U statistic.

    Returns (U_observed, p_value).
    """
    if len(included) == 0 or len(excluded) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    u_obs = mann_whitney_u(included, excluded)
    n1 = len(included)
    combined = np.concatenate([included, excluded])
    n_total = len(combined)
    u_null = n1 * len(excluded) / 2.0
    obs_dev = abs(u_obs - u_null)

    count = 0
    for _ in range(n_perm):
        shuffled = combined[rng.permutation(n_total)]
        u_perm = mann_whitney_u(shuffled[:n1], shuffled[n1:])
        if abs(u_perm - u_null) >= obs_dev:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return (float(u_obs), float(p))


def selection_bias_audit(
    sample_audit: pd.DataFrame,
    population_path: str | Path,
    n_perm: int = 10000,
    seed: int = 42,
    significance_threshold: float = 0.10,
) -> pd.DataFrame:
    """For each numeric covariate available, test whether the
    distribution differs between included and excluded states.

    Returns one row per covariate with: median values for both groups,
    Mann-Whitney U, permutation p, "selection_pressure_flag" if p <
    threshold, plus a plain-language interpretation.
    """
    population = pd.read_csv(population_path)
    if "AI_AN_Population" in population.columns and "Total_Population" in population.columns:
        population["AI_AN_Population_Percent"] = np.where(
            population["Total_Population"] > 0,
            100.0 * population["AI_AN_Population"] / population["Total_Population"],
            np.nan,
        )

    audit = sample_audit.merge(
        population[[c for c in ["State", "AI_AN_Population", "Total_Population",
                                "AI_AN_Population_Percent"] if c in population.columns]],
        on="State", how="left", suffixes=("", "_pop"),
    )

    candidates = [
        "AI_AN_Population_Percent",
        "AI_AN_Population",
        "Total_Population",
    ]
    candidates = [c for c in candidates if c in audit.columns]

    rows = []
    for covariate in candidates:
        included = audit.loc[audit["Included_In_Main_Analysis"] == True, covariate].dropna().to_numpy()
        excluded = audit.loc[audit["Included_In_Main_Analysis"] == False, covariate].dropna().to_numpy()
        u, p = mann_whitney_permutation_p(included, excluded, n_perm=n_perm, seed=seed)
        med_in = float(np.median(included)) if len(included) else float("nan")
        med_out = float(np.median(excluded)) if len(excluded) else float("nan")
        mean_in = float(np.mean(included)) if len(included) else float("nan")
        mean_out = float(np.mean(excluded)) if len(excluded) else float("nan")
        flag = bool(p < significance_threshold) if not np.isnan(p) else False
        rows.append({
            "Covariate": covariate,
            "N_Included": int(len(included)),
            "N_Excluded": int(len(excluded)),
            "Mean_Included": mean_in,
            "Mean_Excluded": mean_out,
            "Median_Included": med_in,
            "Median_Excluded": med_out,
            "MannWhitney_U": u,
            "Permutation_P_Value": p,
            f"Selection_Pressure_At_p_lt_{significance_threshold}": flag,
        })

    return pd.DataFrame(rows)
