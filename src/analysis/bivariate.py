"""Bivariate indicator-vs-outcome analysis.

A direct, composite-free view: for each (indicator, outcome) pair, report
the Spearman rank correlation, permutation p-value, partial correlation
adjusting for AI/AN population share, and Holm-adjusted p-value across
the indicator family for that outcome. This complements the composite
analysis and addresses the reviewer's recommendation to evaluate
indicators individually given the small sample.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analysis import (
    _rank_corr,
    bootstrap_spearman_ci,
    kendall_tau_b,
    partial_spearman,
    permutation_p_value,
)
from src.analysis.multiple_testing import holm_bonferroni, benjamini_hochberg


def bivariate_indicator_associations(
    master_df: pd.DataFrame,
    indicator_pivot: pd.DataFrame,
    outcomes: list[str],
    confounders: list[str] | None = None,
    bootstrap_iterations: int = 4000,
    permutation_iterations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute Spearman rho, perm p, partial rho, and Holm-adjusted p for
    every (indicator, outcome) pair.

    Parameters
    ----------
    master_df
        Master analysis table with State + outcomes + confounders.
    indicator_pivot
        Wide-format indicator table with one row per State and one
        column per indicator.
    outcomes
        Outcomes to test against each indicator.
    """
    confounders = confounders or []

    # Merge indicators into master by state
    if "State" not in indicator_pivot.columns:
        indicator_pivot = indicator_pivot.reset_index().rename(
            columns={"index": "State"}
        )

    merged = master_df.merge(indicator_pivot, on="State", how="left",
                             suffixes=("", "_ind"))

    indicator_cols = [c for c in indicator_pivot.columns if c != "State"]

    rows = []
    for outcome in outcomes:
        if outcome not in merged.columns:
            continue
        # Holm adjustment is applied within outcome across indicators
        per_outcome: list[dict] = []
        for indicator in indicator_cols:
            if indicator not in merged.columns:
                continue
            # Dedupe valid_cols so that when outcome == confounder we don't
            # end up with duplicate column names (which would make
            # valid[outcome] return a DataFrame instead of a Series).
            valid_cols = []
            for c in ["State", indicator, outcome] + [
                cf for cf in confounders if cf in merged.columns
            ]:
                if c not in valid_cols:
                    valid_cols.append(c)
            valid = merged[valid_cols].dropna(subset=[indicator, outcome])
            n = len(valid)
            if n < 4:
                continue
            # Skip if either the indicator or the outcome is constant in
            # this subsample — correlation/permutation tests are
            # undefined and would otherwise yield meaningless p-values.
            if valid[indicator].nunique() <= 1 or valid[outcome].nunique() <= 1:
                continue
            rho = _rank_corr(valid[indicator], valid[outcome])
            tau = kendall_tau_b(valid[indicator], valid[outcome])
            ci_lo, ci_hi = bootstrap_spearman_ci(
                valid[indicator], valid[outcome],
                n_boot=bootstrap_iterations, seed=seed,
            )
            perm_p, perm_mode = permutation_p_value(
                valid[indicator], valid[outcome], statistic="spearman",
                n_perm=permutation_iterations, seed=seed,
            )
            partial_rho = float("nan")
            partial_p = float("nan")
            # Drop outcome from the confounder set if it appears there
            # (degenerate case when AI_AN_Population_Percent is both)
            available_conf = [c for c in confounders
                              if c in valid.columns and c != outcome and c != indicator]
            if available_conf:
                p_rho, p_p, _ = partial_spearman(
                    valid[indicator], valid[outcome], valid[available_conf],
                    n_perm=permutation_iterations, seed=seed,
                )
                partial_rho, partial_p = p_rho, p_p
            per_outcome.append({
                "Outcome": outcome,
                "Indicator": indicator,
                "N": n,
                "Spearman_Rho": rho,
                "Kendall_Tau_b": tau,
                "Descriptive_Bootstrap_CI_Lower": ci_lo,
                "Descriptive_Bootstrap_CI_Upper": ci_hi,
                "CI_Interpretation": "descriptive_only_not_for_inference",
                "Permutation_P_Value": perm_p,
                "Permutation_Mode": perm_mode,
                "Partial_Spearman_Rho": partial_rho,
                "Partial_Permutation_P_Value": partial_p,
                "Confounders_Used": ", ".join(available_conf),
            })

        # Within-outcome multiple-testing correction
        if per_outcome:
            ps = [r["Permutation_P_Value"] for r in per_outcome]
            holm = holm_bonferroni(ps)
            bh = benjamini_hochberg(ps)
            for r, h, b in zip(per_outcome, holm, bh):
                r["Holm_Adjusted_P_Within_Outcome"] = h
                r["BH_FDR_Adjusted_P_Within_Outcome"] = b
            rows.extend(per_outcome)

    return pd.DataFrame(rows).sort_values(["Outcome", "Indicator"]).reset_index(drop=True)


def write_bivariate_associations(
    indicator_table_path: str | Path,
    master_path: str | Path,
    output_path: str | Path,
    outcomes: list[str],
    confounders: list[str] | None = None,
    bootstrap_iterations: int = 4000,
    permutation_iterations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    indicators_long = pd.read_csv(indicator_table_path)
    master = pd.read_csv(master_path)

    pivot = indicators_long.pivot(
        index="State", columns="Indicator", values="Value"
    ).reset_index()

    df = bivariate_indicator_associations(
        master, pivot, outcomes, confounders,
        bootstrap_iterations, permutation_iterations, seed,
    )
    df.to_csv(output_path, index=False)
    return df
