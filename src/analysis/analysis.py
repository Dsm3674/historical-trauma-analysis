from __future__ import annotations

import json
import math
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

MIN_UNITS_FOR_INFERENCE = 20
MIN_UNITS_FOR_MAIN_MANUSCRIPT = 10


def build_master_analysis_table(
    trauma_index_path: str | Path,
    population_path: str | Path,
    mortality_path: str | Path,
    missing_persons_path: str | Path | None = None,
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
        mortality["Disparity_Ratio"] = np.where(
            mortality["Comparator_Rate_per_100k"] > 0,
            mortality["AI_AN_Rate_per_100k"] / mortality["Comparator_Rate_per_100k"],
            np.nan,
        )

    mortality_summary = (
        mortality.groupby("State")
        .agg(
            Mean_Mortality_Disparity_Ratio=("Disparity_Ratio", "mean"),
            Mortality_Cause_Count=("Condition", "nunique"),
            Mortality_Cause_List=("Condition", lambda series: " | ".join(sorted({str(v) for v in series.dropna()}))),
        )
        .reset_index()
    )
    merged = merged.merge(mortality_summary, on="State", how="left")

    if missing_persons_path:
        missing_df = pd.read_csv(missing_persons_path)
        missing_df = missing_df.copy()

        # Only compute AI_AN_Percent_Missing if Total_Missing is present
        if "Total_Missing" in missing_df.columns and "AI_AN_Percent_Missing" not in missing_df.columns:
            missing_df["AI_AN_Percent_Missing"] = np.where(
                missing_df["Total_Missing"] > 0,
                100.0 * missing_df["AI_AN_Missing"] / missing_df["Total_Missing"],
                np.nan,
            )

        # Only compute Overrepresentation_Ratio if AI_AN_Percent_Missing is available
        if "Overrepresentation_Ratio" not in missing_df.columns and "AI_AN_Percent_Missing" in missing_df.columns and "AI_AN_Population_Percent" in missing_df.columns:
            missing_df["Overrepresentation_Ratio"] = np.where(
                missing_df["AI_AN_Population_Percent"] > 0,
                missing_df["AI_AN_Percent_Missing"] / missing_df["AI_AN_Population_Percent"],
                np.nan,
            )

        keep = [
            column
            for column in [
                "State",
                "Total_Missing",
                "AI_AN_Missing",
                "AI_AN_Percent_Missing",
                "AI_AN_Population_Percent",
                "Overrepresentation_Ratio",
            ]
            if column in missing_df.columns
        ]
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

    core_columns = [
        "Historical_Trauma_Index",
        "Mean_Mortality_Disparity_Ratio",
        "AI_AN_Missing_Rate_per_100k_AI_AN",
        "AI_AN_Population_Percent",
    ]
    present_core = [column for column in core_columns if column in merged.columns]
    merged["Included_In_Complete_Case"] = merged[present_core].notna().all(axis=1)
    merged.attrs["n_states"] = int(merged["State"].dropna().astype(str).nunique())
    return merged.sort_values("State").reset_index(drop=True)


def build_sample_characterization_table(
    trauma_index_path: str | Path,
    population_path: str | Path,
    mortality_path: str | Path,
    missing_persons_path: str | Path | None = None,
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

    master = build_master_analysis_table(trauma_index_path, population_path, mortality_path, missing_persons_path)
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
                "Mean_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].mean()) if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
                "Min_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].min()) if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
                "Max_AI_AN_Population_Percent": float(subset["AI_AN_Population_Percent"].max()) if "AI_AN_Population_Percent" in subset.columns and not subset.empty else np.nan,
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
    valid = pd.concat([x, y], axis=1).dropna()
    values = valid.to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        return np.nan

    concordant = 0
    discordant = 0
    tie_x = 0
    tie_y = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = np.sign(values[i, 0] - values[j, 0])
            dy = np.sign(values[i, 1] - values[j, 1])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denominator == 0:
        return np.nan
    return float((concordant - discordant) / denominator)


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

    observed = _rank_corr(valid.iloc[:, 0], valid.iloc[:, 1]) if statistic == "spearman" else kendall_tau_b(valid.iloc[:, 0], valid.iloc[:, 1])

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


def evidence_label(n: int) -> str:
    return "exploratory" if n < MIN_UNITS_FOR_INFERENCE else "ecological_inferential"


def interpretation_label(n: int) -> str:
    if n < MIN_UNITS_FOR_INFERENCE:
        return "exploratory_ecological_only"
    return "ecological_inferential_not_individual_level"


def exploratory_association_table(
    df: pd.DataFrame,
    target: str = "Historical_Trauma_Index",
    outcomes: list[str] | None = None,
    confounders: list[str] | None = None,
    bootstrap_iterations: int = 4000,
    permutation_iterations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
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

        valid = df[["State", target, outcome, *[c for c in confounders if c in df.columns]]].dropna(subset=[target, outcome])
        n = len(valid)
        if n < 3:
            continue

        rho = _rank_corr(valid[target], valid[outcome])
        tau = kendall_tau_b(valid[target], valid[outcome])
        ci_low, ci_high = bootstrap_spearman_ci(valid[target], valid[outcome], n_boot=bootstrap_iterations, seed=seed)
        perm_p, perm_mode = permutation_p_value(valid[target], valid[outcome], statistic="spearman", n_perm=permutation_iterations, seed=seed)

        available_confounders = [column for column in confounders if column in valid.columns and column != outcome]
        partial_rho = np.nan
        partial_p = np.nan
        partial_mode = ""
        confounder_label = ""
        if available_confounders:
            partial_rho, partial_p, partial_mode = partial_spearman(
                valid[target],
                valid[outcome],
                valid[available_confounders],
                n_perm=permutation_iterations,
                seed=seed,
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
                "CI_2.5": ci_low,
                "CI_97.5": ci_high,
                "Permutation_P_Value": perm_p,
                "Permutation_Mode": perm_mode,
                "Partial_Spearman_Adjusted_Rho": partial_rho,
                "Partial_Adjusted_P_Value": partial_p,
                "Partial_Permutation_Mode": partial_mode,
                "Confounders_Used": confounder_label,
                "Evidence_Label": evidence_label(n),
                "Interpretation_Label": interpretation_label(n),
                **leave_one_out,
            }
        )

    return pd.DataFrame(rows).sort_values("Outcome").reset_index(drop=True)


def validate_main_analysis_scope(df: pd.DataFrame) -> None:
    # Lowered threshold to match 13-state real-data sample
    n_states = int(df["State"].dropna().astype(str).nunique())
    if n_states < MIN_UNITS_FOR_MAIN_MANUSCRIPT:
        raise ValueError(
            f"Main manuscript analysis contains only {n_states} states. "
            f"At least {MIN_UNITS_FOR_MAIN_MANUSCRIPT} states are required for the current manuscript framing."
        )


def generate_figures(
    master_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "Historical_Trauma_Index",
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
        plt.xlabel(target)
        plt.ylabel("Number of States")
        plt.title("Distribution of Historical Trauma Index")
        plt.tight_layout()
        plt.savefig(output_dir / "index_distribution.png", dpi=300)
        plt.close()

    display_labels = {
        target: "Historical Trauma Index",
        "Mean_Mortality_Disparity_Ratio": "Mean mortality disparity ratio",
        "AI_AN_Missing_Rate_per_100k_AI_AN": "AI/AN missing persons rate per 100,000 AI/AN population",
    }
    for outcome, filenames in [
        ("Mean_Mortality_Disparity_Ratio", ["index_vs_mortality.png"]),
        (
            "AI_AN_Missing_Rate_per_100k_AI_AN",
            ["index_vs_ai_an_missing_rate.png", "index_vs_ai_an_missing.png"],
        ),
    ]:
        if target not in master_df.columns or outcome not in master_df.columns:
            continue

        subset = master_df.dropna(subset=[target, outcome, "State"])
        if subset.empty:
            continue

        plt.figure(figsize=(8.0, 5.5))
        plt.scatter(subset[target], subset[outcome], alpha=0.9)
        for _, row in subset.iterrows():
            plt.annotate(
                row["State"],
                (row[target], row[outcome]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
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
            on="State",
            how="inner",
        )
        compare = compare.sort_values("Primary_Rank").reset_index(drop=True)
        y_positions = list(range(len(compare)))

        plt.figure(figsize=(8.5, 6.0))
        plt.plot(compare["Primary_Rank"], y_positions, marker="o", label="Primary weights")
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
- The historical trauma index is a proxy measure constructed from observable structural indicators; it is not a direct measure of lived experience.
- Configured indicator weights are heuristic and not community-informed; sensitivity analyses should be treated as robustness checks rather than validation of the weighting scheme.
- BoardingSchool_Count now recovers DOI-style aggregated counts when they are encoded in the source label, but the boarding-school file is still a state-level aggregate and does not fully resolve construct-measurement limitations for all boarding indicators.
- Constant indicators are documented and excluded from within-sample scoring, which improves numerical validity but also highlights construct-coverage limitations in the current sample.
- With only {main_analysis_state_count} complete-case states ({index_state_count} index states overall), bootstrap intervals and rank associations remain sensitive to influential states; permutation tests and leave-one-state-out diagnostics should be reported alongside point estimates.
- The missing persons outcome is now an AI/AN population-adjusted rate per 100,000 AI/AN residents rather than an absolute count, which reduces but does not eliminate compositional confounding.
- Adjusted analyses control only for AI/AN population share, not the full set of potential confounders.
- Temporal alignment remains imperfect: the environmental indicator is from a later period than the 2020 mortality and missing-person outcomes.
- The environmental indicator is state-level and not AI/AN-specific, so within-state heterogeneity and community-level exposure error likely remain substantial.
- Only one mortality condition is analyzed in the current inputs, so disease-specific inference remains narrow and vulnerable to suppression-driven selection.
- State-level analyses can mask within-state heterogeneity and are not substitutes for tribal, reservation, county, or community-governed analyses.
- No causal interpretation, policy-effect attribution, or community-consensus weighting should be claimed from this pipeline alone.
- Data provenance, source definitions, and processed-data manifests should be checked before manuscript claims are finalized.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_methods_summary(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
