from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

# Default thresholds, overridable from config or callers
MIN_UNITS_FOR_INFERENCE = 20
MIN_UNITS_FOR_MAIN_MANUSCRIPT = 10


def build_master_analysis_table(
    trauma_index_path: str | Path,
    population_path: str | Path,
    mortality_path: str | Path,
    missing_persons_path: str | Path | None = None,
    composite_column_name: str = "Structural_Exposure_Composite",
) -> pd.DataFrame:
    trauma = pd.read_csv(trauma_index_path)
    population = pd.read_csv(population_path)
    mortality = pd.read_csv(mortality_path)

    if "State" not in trauma.columns or "State" not in population.columns:
        raise ValueError("Both trauma and population files must include 'State'.")

    population = population.copy()
    if "AI_AN_Population_Percent" not in population.columns:
        population["AI_AN_Population_Percent"] = np.where(
            population["Total_Population"] > 0,
            100.0 * population["AI_AN_Population"] / population["Total_Population"],
            np.nan,
        )

    merged = trauma.merge(
        population[["State", "AI_AN_Population", "Total_Population", "AI_AN_Population_Percent"]],
        on="State",
        how="left",
    )

    if "Disparity_Ratio" not in mortality.columns:
        mortality = mortality.copy()
        mortality["Disparity_Ratio"] = np.where(
            mortality["Comparator_Rate_per_100k"] > 0,
            mortality["AI_AN_Rate_per_100k"] / mortality["Comparator_Rate_per_100k"],
            np.nan,
        )

    # Multi-condition multi-year aggregation:
    #   1) per-(State, Condition) average across years
    #   2) per-condition wide columns
    #   3) Mean_Mortality_Disparity_Ratio is the equal-weighted mean of
    #      per-condition values
    if "Condition" in mortality.columns:
        per_state_condition = (
            mortality.groupby(["State", "Condition"])
            .agg(
                Disparity_Ratio=("Disparity_Ratio", "mean"),
                Years_Available=("Year", "nunique") if "Year" in mortality.columns else ("Disparity_Ratio", "size"),
            )
            .reset_index()
        )
        per_condition_wide = per_state_condition.pivot(
            index="State", columns="Condition", values="Disparity_Ratio"
        )
        per_condition_wide.columns = [f"Disparity_Ratio_{c}" for c in per_condition_wide.columns]
        per_condition_wide = per_condition_wide.reset_index()

        mortality_summary = (
            per_state_condition.groupby("State")
            .agg(
                Mean_Mortality_Disparity_Ratio=("Disparity_Ratio", "mean"),
                Mortality_Cause_Count=("Condition", "nunique"),
                Mortality_Cause_List=("Condition", lambda s: " | ".join(sorted({str(v) for v in s.dropna()}))),
            )
            .reset_index()
        )
        merged = merged.merge(mortality_summary, on="State", how="left")
        merged = merged.merge(per_condition_wide, on="State", how="left")
    else:
        mortality_summary = (
            mortality.groupby("State")
            .agg(Mean_Mortality_Disparity_Ratio=("Disparity_Ratio", "mean"))
            .reset_index()
        )
        merged = merged.merge(mortality_summary, on="State", how="left")

    # Missing persons
    if missing_persons_path:
        missing_df = pd.read_csv(missing_persons_path)
        missing_df = missing_df.copy()

        if "Total_Missing" in missing_df.columns and "AI_AN_Percent_Missing" not in missing_df.columns:
            missing_df["AI_AN_Percent_Missing"] = np.where(
                missing_df["Total_Missing"] > 0,
                100.0 * missing_df["AI_AN_Missing"] / missing_df["Total_Missing"],
                np.nan,
            )

        if "Overrepresentation_Ratio" not in missing_df.columns and "AI_AN_Percent_Missing" in missing_df.columns and "AI_AN_Population_Percent" in missing_df.columns:
            missing_df["Overrepresentation_Ratio"] = np.where(
                missing_df["AI_AN_Population_Percent"] > 0,
                missing_df["AI_AN_Percent_Missing"] / missing_df["AI_AN_Population_Percent"],
                np.nan,
            )

        keep = [c for c in [
            "State", "Total_Missing", "AI_AN_Missing", "AI_AN_Percent_Missing",
            "AI_AN_Population_Percent", "Overrepresentation_Ratio",
        ] if c in missing_df.columns]
        merged = merged.merge(missing_df[keep], on="State", how="left", suffixes=("", "_missing"))

        if "AI_AN_Population_Percent_missing" in merged.columns:
            merged["AI_AN_Population_Percent"] = merged["AI_AN_Population_Percent"].fillna(
                merged["AI_AN_Population_Percent_missing"]
            )
            merged = merged.drop(columns=["AI_AN_Population_Percent_missing"])

    if {"AI_AN_Missing", "AI_AN_Population"}.issubset(merged.columns):
        merged["AI_AN_Missing_Rate_per_100k_AI_AN"] = np.where(
            merged["AI_AN_Population"] > 0,
            100000.0 * merged["AI_AN_Missing"] / merged["AI_AN_Population"],
            np.nan,
        )

    # Backward-compat: ensure both names exist
    legacy = "Historical_Trauma_Index"
    if composite_column_name not in merged.columns and legacy in merged.columns:
        merged[composite_column_name] = merged[legacy]
    if legacy not in merged.columns and composite_column_name in merged.columns:
        merged[legacy] = merged[composite_column_name]

    core_columns = [
        composite_column_name,
        "Mean_Mortality_Disparity_Ratio",
        "AI_AN_Missing_Rate_per_100k_AI_AN",
        "AI_AN_Population_Percent",
    ]
    present_core = [c for c in core_columns if c in merged.columns]
    merged["Included_In_Complete_Case"] = merged[present_core].notna().all(axis=1)
    merged.attrs["n_states"] = int(merged["State"].dropna().astype(str).nunique())
    return merged.sort_values("State").reset_index(drop=True)


def build_sample_characterization_table(
    trauma_index_path: str | Path,
    population_path: str | Path,
    mortality_path: str | Path,
    missing_persons_path: str | Path | None = None,
    composite_column_name: str = "Structural_Exposure_Composite",
) -> pd.DataFrame:
    trauma = pd.read_csv(trauma_index_path)
    population = pd.read_csv(population_path)
    mortality = pd.read_csv(mortality_path)
    missing = pd.read_csv(missing_persons_path) if missing_persons_path else pd.DataFrame(columns=["State"])

    all_states = sorted(
        set(trauma.get("State", pd.Series(dtype=str)).dropna())
        | set(population.get("State", pd.Series(dtype=str)).dropna())
        | set(mortality.get("State", pd.Series(dtype=str)).dropna())
        | set(missing.get("State", pd.Series(dtype=str)).dropna())
    )
    audit = pd.DataFrame({"State": all_states})
    audit["In_Trauma_Index"] = audit["State"].isin(set(trauma["State"]))
    audit["In_Population"] = audit["State"].isin(set(population["State"]))
    audit["In_Mortality"] = audit["State"].isin(set(mortality["State"]))
    audit["In_Missing_Persons"] = audit["State"].isin(set(missing.get("State", pd.Series(dtype=str))))

    if {"State", "AI_AN_Population", "Total_Population"}.issubset(population.columns):
        pop = population[["State", "AI_AN_Population", "Total_Population"]].copy()
        pop["AI_AN_Population_Percent"] = np.where(
            pop["Total_Population"] > 0,
            100.0 * pop["AI_AN_Population"] / pop["Total_Population"],
            np.nan,
        )
        audit = audit.merge(pop[["State", "AI_AN_Population_Percent"]], on="State", how="left")

    master = build_master_analysis_table(
        trauma_index_path, population_path, mortality_path, missing_persons_path,
        composite_column_name=composite_column_name,
    )
    included = set(master.loc[master["Included_In_Complete_Case"], "State"])
    audit["Included_In_Main_Analysis"] = audit["State"].isin(included)
    return audit.sort_values("State").reset_index(drop=True)


def summarize_included_vs_excluded(sample_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for included_flag, label in [(True, "included"), (False, "excluded")]:
        subset = sample_audit.loc[sample_audit["Included_In_Main_Analysis"] == included_flag]
        rows.append(
            {
                "Group": label,
                "State_Count": int(len(subset)),
                "Mean_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].mean())
                    if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
                "Min_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].min())
                    if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
                "Max_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].max())
                    if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index().rename(columns={"index": "Variable"})


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 2:
        return np.nan
    return float(valid.iloc[:, 0].rank(method="average").corr(valid.iloc[:, 1].rank(method="average")))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denominator == 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denominator)


def kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    """Vectorized Kendall's tau-b. Same semantics as the previous loop
    implementation, ~50x faster via numpy broadcasting."""
    valid = pd.concat([x, y], axis=1).dropna()
    n = len(valid)
    if n < 2:
        return np.nan
    a = valid.iloc[:, 0].to_numpy(dtype=float)
    b = valid.iloc[:, 1].to_numpy(dtype=float)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    iu = np.triu_indices(n, k=1)
    sa = da[iu]
    sb = db[iu]
    concordant = int(np.sum((sa * sb) > 0))
    discordant = int(np.sum((sa * sb) < 0))
    tie_x = int(np.sum((sa == 0) & (sb != 0)))
    tie_y = int(np.sum((sa != 0) & (sb == 0)))
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return np.nan
    return float((concordant - discordant) / denom)


def bootstrap_spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 4000, seed: int = 42) -> tuple[float, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 4:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    arr = valid.to_numpy()
    stats = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        stats.append(_rank_corr(pd.Series(sample[:, 0]), pd.Series(sample[:, 1])))

    return (
        float(np.nanpercentile(stats, 2.5)),
        float(np.nanpercentile(stats, 97.5)),
    )


def permutation_p_value(
    x: pd.Series,
    y: pd.Series,
    statistic: str = "spearman",
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, str]:
    valid = pd.concat([x, y], axis=1).dropna()
    n = len(valid)
    if n < 4:
        return (np.nan, "insufficient_n")

    observed = (_rank_corr(valid.iloc[:, 0], valid.iloc[:, 1])
                if statistic == "spearman"
                else kendall_tau_b(valid.iloc[:, 0], valid.iloc[:, 1]))

    if n <= 8:
        reference = valid.iloc[:, 1].to_numpy()
        permuted = []
        for ordering in permutations(range(n)):
            shuffled = reference[list(ordering)]
            if statistic == "spearman":
                permuted.append(_rank_corr(valid.iloc[:, 0], pd.Series(shuffled)))
            else:
                permuted.append(kendall_tau_b(valid.iloc[:, 0], pd.Series(shuffled)))
        permuted = np.asarray(permuted, dtype=float)
        p_value = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))
        return (p_value, "exact_enumeration")

    rng = np.random.default_rng(seed)
    reference = valid.iloc[:, 1].to_numpy()
    permuted_stats = []
    for _ in range(n_perm):
        shuffled = rng.permutation(reference)
        if statistic == "spearman":
            permuted_stats.append(_rank_corr(valid.iloc[:, 0], pd.Series(shuffled)))
        else:
            permuted_stats.append(kendall_tau_b(valid.iloc[:, 0], pd.Series(shuffled)))
    permuted_stats = np.asarray(permuted_stats, dtype=float)
    p_value = float((np.sum(np.abs(permuted_stats) >= abs(observed)) + 1) / (len(permuted_stats) + 1))
    return (p_value, "permutation")


def _residualize(outcome: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariates)), covariates])
    beta, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return outcome - design @ beta


def partial_spearman(
    x: pd.Series,
    y: pd.Series,
    covariate_df: pd.DataFrame,
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, float, str]:
    combined = pd.concat([x.rename("x"), y.rename("y"), covariate_df], axis=1).dropna()
    if len(combined) < 4 or covariate_df.empty:
        return (np.nan, np.nan, "no_covariates")

    ranked = combined.rank(method="average")
    residual_x = _residualize(ranked["x"].to_numpy(dtype=float), ranked[covariate_df.columns].to_numpy(dtype=float))
    residual_y = _residualize(ranked["y"].to_numpy(dtype=float), ranked[covariate_df.columns].to_numpy(dtype=float))
    observed = _pearson_corr(residual_x, residual_y)

    rng = np.random.default_rng(seed)
    permuted = []
    for _ in range(n_perm):
        permuted.append(_pearson_corr(residual_x, rng.permutation(residual_y)))

    permuted = np.asarray(permuted, dtype=float)
    p_value = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (len(permuted) + 1))
    return (observed, p_value, "permutation")


def leave_one_state_out_summary(df: pd.DataFrame, target: str, outcome: str) -> dict[str, object]:
    valid = df[["State", target, outcome]].dropna()
    if len(valid) < 5:
        return {
            "LeaveOneOut_Min_Rho": np.nan,
            "LeaveOneOut_Max_Rho": np.nan,
            "Most_Influential_State": "",
            "Max_Absolute_Rho_Shift": np.nan,
        }

    full_rho = _rank_corr(valid[target], valid[outcome])
    rows = []
    for state in valid["State"]:
        subset = valid.loc[valid["State"] != state]
        rho = _rank_corr(subset[target], subset[outcome])
        rows.append({"State": state, "Rho": rho, "Abs_Shift": abs(rho - full_rho)})

    loo = pd.DataFrame(rows)
    influential = loo.sort_values("Abs_Shift", ascending=False).iloc[0]
    return {
        "LeaveOneOut_Min_Rho": float(loo["Rho"].min()),
        "LeaveOneOut_Max_Rho": float(loo["Rho"].max()),
        "Most_Influential_State": str(influential["State"]),
        "Max_Absolute_Rho_Shift": float(influential["Abs_Shift"]),
    }


def evidence_label(n: int, threshold: int = MIN_UNITS_FOR_INFERENCE) -> str:
    return "exploratory" if n < threshold else "ecological_inferential"


def interpretation_label(n: int, threshold: int = MIN_UNITS_FOR_INFERENCE) -> str:
    if n < threshold:
        return "exploratory_ecological_only"
    return "ecological_inferential_not_individual_level"


def exploratory_association_table(
    df: pd.DataFrame,
    target: str = "Structural_Exposure_Composite",
    outcomes: list[str] | None = None,
    confounders: list[str] | None = None,
    bootstrap_iterations: int = 4000,
    permutation_iterations: int = 10000,
    seed: int = 42,
    primary_outcomes_for_multiple_testing: list[str] | None = None,
    min_units_for_inference: int = MIN_UNITS_FOR_INFERENCE,
) -> pd.DataFrame:
    from src.analysis.multiple_testing import holm_bonferroni, benjamini_hochberg

    if target not in df.columns:
        raise ValueError(f"Expected '{target}' in dataframe.")

    outcomes = outcomes or [
        "Mean_Mortality_Disparity_Ratio",
        "AI_AN_Missing_Rate_per_100k_AI_AN",
        "AI_AN_Population_Percent",
    ]
    confounders = confounders or []

    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        # Dedupe column list to avoid DataFrame-not-Series indexing when
        # outcome equals a confounder
        col_list = []
        for c in ["State", target, outcome, *[cf for cf in confounders if cf in df.columns]]:
            if c not in col_list:
                col_list.append(c)
        valid = df[col_list].dropna(subset=[target, outcome])
        n = len(valid)
        if n < 3:
            continue

        rho = _rank_corr(valid[target], valid[outcome])
        tau = kendall_tau_b(valid[target], valid[outcome])
        ci_low, ci_high = bootstrap_spearman_ci(valid[target], valid[outcome], n_boot=bootstrap_iterations, seed=seed)
        perm_p, perm_mode = permutation_p_value(valid[target], valid[outcome], statistic="spearman", n_perm=permutation_iterations, seed=seed)

        available_confounders = [c for c in confounders
                                 if c in valid.columns and c != outcome and c != target]
        partial_rho = np.nan
        partial_p = np.nan
        partial_mode = ""
        confounder_label = ""
        if available_confounders:
            partial_rho, partial_p, partial_mode = partial_spearman(
                valid[target], valid[outcome], valid[available_confounders],
                n_perm=permutation_iterations, seed=seed,
            )
            confounder_label = ", ".join(available_confounders)

        leave_one_out = leave_one_state_out_summary(valid, target, outcome)
        rows.append(
            {
                "Outcome": outcome,
                "N": n,
                "Association_Type": "Spearman rank correlation",
                "Effect_Size": rho,
                "Kendall_Tau_b": tau,
                "Descriptive_Bootstrap_CI_Lower": ci_low,
                "Descriptive_Bootstrap_CI_Upper": ci_high,
                "CI_Interpretation": "descriptive_only_not_for_inference",
                "Permutation_P_Value": perm_p,
                "Permutation_Mode": perm_mode,
                "Partial_Spearman_Adjusted_Rho": partial_rho,
                "Partial_Adjusted_P_Value": partial_p,
                "Partial_Permutation_Mode": partial_mode,
                "Confounders_Used": confounder_label,
                "Evidence_Label": evidence_label(n, min_units_for_inference),
                "Interpretation_Label": interpretation_label(n, min_units_for_inference),
                **leave_one_out,
            }
        )

    out = pd.DataFrame(rows).sort_values("Outcome").reset_index(drop=True)

    if primary_outcomes_for_multiple_testing and not out.empty:
        primary_mask = out["Outcome"].isin(primary_outcomes_for_multiple_testing)
        primary_p = out.loc[primary_mask, "Permutation_P_Value"].tolist()
        if primary_p:
            holm = holm_bonferroni(primary_p)
            bh = benjamini_hochberg(primary_p)
            out["Holm_Adjusted_P_Primary_Family"] = np.nan
            out["BH_FDR_Adjusted_P_Primary_Family"] = np.nan
            out.loc[primary_mask, "Holm_Adjusted_P_Primary_Family"] = holm
            out.loc[primary_mask, "BH_FDR_Adjusted_P_Primary_Family"] = bh

    return out


def validate_main_analysis_scope(df: pd.DataFrame, min_units: int = MIN_UNITS_FOR_MAIN_MANUSCRIPT) -> None:
    n_states = int(df["State"].dropna().astype(str).nunique())
    if n_states < min_units:
        raise ValueError(
            f"Main manuscript analysis contains only {n_states} states. "
            f"At least {min_units} states are required for the current manuscript framing."
        )


def generate_figures(
    master_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "Structural_Exposure_Composite",
    display_label: str = "Structural Exposure Composite",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if target in master_df.columns:
        plt.figure(figsize=(7.5, 4.8))
        plt.hist(master_df[target].dropna(), bins=10, edgecolor="black")
        plt.xlabel(display_label)
        plt.ylabel("Number of States")
        plt.title(f"Distribution of {display_label}")
        plt.tight_layout()
        plt.savefig(output_dir / "index_distribution.png", dpi=300)
        plt.close()

    display_labels = {
        target: display_label,
        "Mean_Mortality_Disparity_Ratio": "Mean mortality disparity ratio",
        "AI_AN_Missing_Rate_per_100k_AI_AN": "AI/AN missing persons rate per 100,000 AI/AN population",
        "AI_AN_Missing": "AI/AN missing persons (count)",
    }
    for outcome, filenames in [
        ("Mean_Mortality_Disparity_Ratio", ["index_vs_mortality.png"]),
        ("AI_AN_Missing_Rate_per_100k_AI_AN",
         ["index_vs_ai_an_missing_rate.png", "index_vs_ai_an_missing.png"]),
    ]:
        if target not in master_df.columns or outcome not in master_df.columns:
            continue
        subset = master_df.dropna(subset=[target, outcome, "State"])
        if subset.empty:
            continue
        plt.figure(figsize=(8.0, 5.5))
        plt.scatter(subset[target], subset[outcome], alpha=0.9)
        for _, row in subset.iterrows():
            plt.annotate(row["State"], (row[target], row[outcome]),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
        plt.xlabel(display_labels.get(target, target))
        plt.ylabel(display_labels.get(outcome, outcome))
        plt.title(f"{display_labels.get(target, target)} vs {display_labels.get(outcome, outcome)}")
        plt.tight_layout()
        for filename in filenames:
            plt.savefig(output_dir / filename, dpi=300)
        plt.close()

    primary = sensitivity_df.loc[sensitivity_df["Scheme"] == "primary"].copy()
    equal = sensitivity_df.loc[sensitivity_df["Scheme"] == "equal_weights"].copy()
    if not primary.empty and not equal.empty:
        compare = primary[["State", "Rank"]].rename(columns={"Rank": "Primary_Rank"}).merge(
            equal[["State", "Rank"]].rename(columns={"Rank": "Equal_Weight_Rank"}),
            on="State", how="inner",
        )
        compare = compare.sort_values("Primary_Rank").reset_index(drop=True)
        y_positions = list(range(len(compare)))

        plt.figure(figsize=(8.5, 6.0))
        plt.plot(compare["Primary_Rank"], y_positions, marker="o", label="Primary (configured)")
        plt.plot(compare["Equal_Weight_Rank"], y_positions, marker="o", label="Equal weights")
        for idx, row in compare.iterrows():
            x_position = max(row["Primary_Rank"], row["Equal_Weight_Rank"]) + 0.15
            plt.text(x_position, idx, row["State"], va="center", fontsize=7)
        plt.xlabel("Rank")
        plt.ylabel("State")
        plt.title("Sensitivity Analysis of State Rankings")
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "sensitivity_ranks.png", dpi=300)
        plt.close()


def write_limitations_report(path: str | Path, main_analysis_state_count: int, index_state_count: int) -> None:
    text = f"""Limitations:

- This is an ecological analysis and must not be interpreted as individual-level evidence.
- The structural exposure composite is constructed from observable structural indicators; it is NOT a direct measure of historical trauma or lived experience. The previous label 'Historical Trauma Index' is retained as a backward-compatible column name only.
- Configured indicator weights are heuristic and not community-informed; the pipeline reports both configured and equal-weight composite scores as co-primary specifications, since equal weights produce a more stable rank ordering at small n.
- Sensitivity analyses (alternate normalization, PCA, leave-one-out, Dirichlet weight perturbation, temporal-only) are reported as robustness checks rather than validation.
- BoardingSchool_Count recovers DOI-style aggregated counts when encoded in the source label; duration estimates remain derived from aggregate era endpoints.
- Constant indicators are documented and excluded from within-sample scoring.
- With only {main_analysis_state_count} complete-case states ({index_state_count} index states overall), bootstrap intervals are reported as descriptive only (CI columns explicitly labeled). Permutation p-values are the primary inferential statistic. Multiple-testing adjustments (Holm-Bonferroni, BH-FDR) are reported across the primary outcome family.
- Construct-validity diagnostics (Cronbach alpha, KMO, PCA loadings, item-total correlations) are reported as descriptive aids only; at small n these statistics have wide sampling variability.
- Mortality data are pooled across years and conditions to reduce suppression-driven exclusion; equal weight is given to each condition rather than each (year, condition) cell.
- The missing-persons outcome is reported as both an absolute count (primary, per the MMIR undercounting literature) and a population-adjusted rate per 100,000 AI/AN residents (secondary).
- Adjusted analyses control only for AI/AN population share; urbanization, income, region, and IHS-facility proximity remain plausible omitted variables.
- The environmental indicator is state-level and not AI/AN-specific.
- State-level analyses can mask within-state heterogeneity and are not substitutes for tribal or community-governed analyses.
- No causal interpretation, policy-effect attribution, or community-consensus weighting should be claimed.
- Selection-bias audit (Mann-Whitney U with permutation p) tests whether the complete-case sample differs systematically from excluded states; results are in selection_bias_audit.csv.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_methods_summary(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
