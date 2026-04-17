from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
FIGURES = Path("figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Figure 1: index distribution
# -----------------------------
index_path = PROCESSED / "historical_trauma_index.csv"
if index_path.exists():
    df_index = pd.read_csv(index_path)

    plt.figure(figsize=(7.5, 4.8))
    plt.hist(df_index["Historical_Trauma_Index"].dropna(), bins=10, edgecolor="black")
    plt.xlabel("Historical Trauma Index")
    plt.ylabel("Number of States")
    plt.title("Distribution of Historical Trauma Index")
    plt.tight_layout()
    plt.savefig(FIGURES / "index_distribution.png", dpi=300)
    plt.close()

# --------------------------------------------
# Figure 2: index vs mortality disparity
# --------------------------------------------
master_path = PROCESSED / "master_analysis_table.csv"
if master_path.exists():
    df_master = pd.read_csv(master_path)
    needed = {"Historical_Trauma_Index", "Mean_Mortality_Disparity_Ratio", "State"}

    if needed.issubset(df_master.columns):
        df_plot = df_master.dropna(
            subset=["Historical_Trauma_Index", "Mean_Mortality_Disparity_Ratio"]
        )

        plt.figure(figsize=(7.5, 4.8))
        plt.scatter(
            df_plot["Historical_Trauma_Index"],
            df_plot["Mean_Mortality_Disparity_Ratio"],
        )

        for _, row in df_plot.iterrows():
            plt.annotate(
                row["State"],
                (row["Historical_Trauma_Index"], row["Mean_Mortality_Disparity_Ratio"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )

        plt.xlabel("Historical Trauma Index")
        plt.ylabel("Mean Mortality Disparity Ratio")
        plt.title("Historical Trauma Index vs Mortality Disparity Ratio")
        plt.tight_layout()
        plt.savefig(FIGURES / "index_vs_mortality.png", dpi=300)
        plt.close()

# --------------------------------------------
# Figure 3: sensitivity rank comparison
# --------------------------------------------
sens_path = PROCESSED / "historical_trauma_index_sensitivity.csv"
if sens_path.exists():
    df_sens = pd.read_csv(sens_path)
    needed = {"State", "Rank_Primary", "Rank_Alternate"}

    if needed.issubset(df_sens.columns):
        df_sens = df_sens.sort_values("Rank_Primary").reset_index(drop=True)

        plt.figure(figsize=(8, 6))
        y_vals = list(range(len(df_sens)))

        plt.plot(df_sens["Rank_Primary"], y_vals, marker="o", label="Primary weights")
        plt.plot(df_sens["Rank_Alternate"], y_vals, marker="o", label="Equal weights")

        for i, (_, row) in enumerate(df_sens.iterrows()):
            x_pos = max(row["Rank_Primary"], row["Rank_Alternate"]) + 0.2
            plt.text(x_pos, i, row["State"], va="center", fontsize=7)

        plt.xlabel("Rank")
        plt.ylabel("State")
        plt.title("Sensitivity Analysis of State Rankings")
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / "sensitivity_ranks.png", dpi=300)
        plt.close()

print("Saved figures to ./figures")
