import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import os

# ============================================================
# ASSIGNMENT 7: VISUAL ANALYSIS WITH MATPLOTLIB
# ============================================================
# This script includes:
# 1) Iris dataset: 3 visualizations focused on species trait differences
# 2) Loan dataset: 3 visualizations for exploratory pattern discovery
# 3) A short evidence-based paragraph summarizing loan insights
# ============================================================

# =========================
# Part 1: Iris visualizations
# =========================

def load_iris_data() -> pd.DataFrame:
    """
    Load Fisher's Iris dataset from sklearn and return a cleaned DataFrame.
    - Keeps data loading logic in one place.
    - Renames verbose sklearn column names to cleaner snake_case.
    - Adds human-readable species labels used by all visualizations.
    """
    # Load the built-in sklearn Iris object (features + numeric labels + label names).
    iris = datasets.load_iris()

    # Build the DataFrame from the numeric feature matrix.
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Rename long feature names so plotting code is easier to read.
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    # Convert numeric class IDs (0,1,2) to real species names.
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def iris_scatter_petal(df: pd.DataFrame):
    """
    Visualization 1:
    Scatter plot of petal length vs petal width, colored by species.

    Goal: show how species separate in petal space.
    """
    plt.figure(figsize=(7, 5))
    # Fixed palette keeps species-color mapping consistent across figures.
    palette = {"setosa": "#0ee9de", "versicolor": "#1d4ed8", "virginica": "#ba91dd"}

    # Plot one scatter layer per species so separation can be seen clearly.
    for species, color in palette.items():
        subset = df[df["species"] == species]
        plt.scatter(
            subset["petal_length"],
            subset["petal_width"],
            label=species.title(),
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            s=45,
        )

    plt.title("Iris Species Separation in Petal Space", fontsize=13, pad=10)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    # Light grid helps estimate relative positions without overpowering points.
    plt.grid(alpha=0.3)
    plt.legend(loc="best", title="Species")
    plt.tight_layout()
    plt.savefig("iris_petal_scatter.png", dpi=300)
    plt.show()


def iris_boxplot_sepal(df: pd.DataFrame):
    """
    Visualization 2:
    Boxplot of sepal length by species.

    Goal: compare central tendency and spread of sepal length across species.
    """
    plt.figure(figsize=(7, 5))

    # Fixed order keeps category order consistent across runs and charts.
    species_order = ["setosa", "versicolor", "virginica"]

    # Build one list/series per species for boxplot input.
    data = [df[df["species"] == s]["sepal_length"] for s in species_order]

    box = plt.boxplot(
        data,
        tick_labels=[s.title() for s in species_order],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )

    # Color each box to match species palette used in other Iris figures.
    box_colors = ["#0ee9de", "#1d4ed8", "#ba91dd"]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    plt.title("Sepal Length Distribution by Iris Species", fontsize=13, pad=10)
    plt.xlabel("Species")
    plt.ylabel("Sepal Length (cm)")
    plt.grid(visible=True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("iris_sepal_boxplot.png", dpi=300)
    plt.show()


def iris_subplots_pairwise(df: pd.DataFrame):
    """
    Visualization 3:
    2x2 subplot grid comparing different trait pairs.

    Top-left: sepal length vs sepal width
    Top-right: petal length vs sepal length
    Bottom-left: petal width vs sepal width
    Bottom-right: petal length vs petal width (again, but with different style)

    Goal: show multiple relationships in one figure.
    """
    # Create a 2x2 panel so multiple pairwise relationships can be compared at once.
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), layout="tight")

    species_list = ["setosa", "versicolor", "virginica"]
    palette = {"setosa": "#0ee9de", "versicolor": "#1d4ed8", "virginica": "#ba91dd"}

    # Plot each panel explicitly for readability (lecture-note style).
    for species in species_list:
        subset = df[df["species"] == species]
        color = palette[species]

        # Each subplot uses the same species color mapping for easy cross-panel reading.
        ax[0, 0].scatter(subset["sepal_length"], subset["sepal_width"], label=species.title(), color=color, alpha=0.8, edgecolor="black", linewidth=0.4, s=28)
        ax[0, 1].scatter(subset["petal_length"], subset["sepal_length"], label=species.title(), color=color, alpha=0.8, edgecolor="black", linewidth=0.4, s=28)
        ax[1, 0].scatter(subset["petal_width"], subset["sepal_width"], label=species.title(), color=color, alpha=0.8, edgecolor="black", linewidth=0.4, s=28)
        ax[1, 1].scatter(subset["petal_length"], subset["petal_width"], label=species.title(), color=color, alpha=0.8, edgecolor="black", linewidth=0.4, s=28)

    ax[0, 0].set_title("Sepal Length vs Width", fontsize=10)
    ax[0, 0].set_xlabel("Sepal Length")
    ax[0, 0].set_ylabel("Sepal Width")
    ax[0, 0].grid(visible=True, which="both", axis="both", alpha=0.25)

    ax[0, 1].set_title("Petal Length vs Sepal Length", fontsize=10)
    ax[0, 1].set_xlabel("Petal Length")
    ax[0, 1].set_ylabel("Sepal Length")
    ax[0, 1].grid(visible=True, which="both", axis="both", alpha=0.25)

    ax[1, 0].set_title("Petal Width vs Sepal Width", fontsize=10)
    ax[1, 0].set_xlabel("Petal Width")
    ax[1, 0].set_ylabel("Sepal Width")
    ax[1, 0].grid(visible=True, which="both", axis="both", alpha=0.25)

    ax[1, 1].set_title("Petal Length vs Width", fontsize=10)
    ax[1, 1].set_xlabel("Petal Length")
    ax[1, 1].set_ylabel("Petal Width")
    ax[1, 1].grid(visible=True, which="both", axis="both", alpha=0.25)

    # Use one shared legend to avoid repeating legends in all four panels.
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.54, 0.5),
        ncols=3,
        frameon=True,
        title="Species",
        fontsize=8,
        title_fontsize=9,
        markerscale=0.8,
        framealpha=0.85,
    )

    plt.savefig("iris_pairwise_subplots.png", dpi=300)
    plt.show()


# =========================
# Part 2: Loan visualizations
# =========================

def load_loan_data(path: str = r"C:\Users\jinas\anaconda\Library\include\absl\base\InfoB211\Assignment 7\LoanDataset - LoansDatasest.csv") -> pd.DataFrame:
    """
    Load the Kaggle loan dataset from a CSV file.

    """
    # Resolve the folder where this script lives.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = ""

    # 1) If the provided path exists, use it directly.
    if os.path.exists(path):
        candidate = path
    # 2) If the path is relative, try resolving it from this script folder.
    elif os.path.exists(os.path.join(script_dir, path)):
        candidate = os.path.join(script_dir, path)
    # 3) Otherwise, auto-detect a CSV in this folder that contains "loan" in its name.
    else:
        for filename in os.listdir(script_dir):
            lower_name = filename.lower()
            if lower_name.endswith(".csv") and "loan" in lower_name:
                candidate = os.path.join(script_dir, filename)
                break

    if candidate == "":
        raise FileNotFoundError(
            "Loan dataset not found. Put the loan CSV in this folder or pass a full file path."
        )

    # Read the selected loan file.
    df = pd.read_csv(candidate)

    # Standardize headers to lowercase_with_underscores so schema checks are easier.
    clean_cols = []
    for col in df.columns:
        clean_cols.append(col.strip().lower().replace(" ", "_"))
    df.columns = clean_cols

    # Build Loan_Status in Y/N format expected by all loan plotting functions.
    if "loan_status" in df.columns:
        df["Loan_Status"] = df["loan_status"].astype(str).str.upper()
    elif "current_loan_status" in df.columns:
        # Convert textual status labels to Y (approved/no-default) and N (not approved/default).
        status_values = []
        for value in df["current_loan_status"]:
            text = str(value).strip().upper()
            if text == "NO DEFAULT" or text == "Y":
                status_values.append("Y")
            elif text == "DEFAULT" or text == "N":
                status_values.append("N")
            else:
                status_values.append(None)
        df["Loan_Status"] = status_values

    # Build ApplicantIncome used in the income boxplot and paragraph stats.
    if "applicantincome" in df.columns:
        df["ApplicantIncome"] = pd.to_numeric(df["applicantincome"], errors="coerce")
    elif "customer_income" in df.columns:
        df["ApplicantIncome"] = pd.to_numeric(df["customer_income"], errors="coerce")

    # Build Credit_History used in the third loan chart.
    if "credit_history" in df.columns:
        df["Credit_History"] = pd.to_numeric(df["credit_history"], errors="coerce")
    elif "historical_default" in df.columns:
        # Convert historical default flags to a numeric proxy used for grouping.
        credit_values = []
        for value in df["historical_default"]:
            text = str(value).strip().upper()
            if text == "N":
                credit_values.append(1)
            elif text == "Y":
                credit_values.append(0)
            else:
                credit_values.append(None)
        df["Credit_History"] = credit_values

    # Stop early with a helpful message if core columns are still missing.
    if "Loan_Status" not in df.columns or "ApplicantIncome" not in df.columns:
        raise KeyError(
            "Loan file is missing required columns after cleanup. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def loan_bar_loan_status(df: pd.DataFrame):
    """
    Visualization 1:
    Bar chart of loan status counts.

    Goal: show overall approval vs non-approval distribution.
    """
    plt.figure(figsize=(7, 5))

    # Reindex ensures both Y and N appear even if one group is absent in data.
    status_counts = df["Loan_Status"].value_counts().reindex(["Y", "N"]).fillna(0)

    bars = plt.bar(
        ["Approved (Y)", "Not Approved (N)"],
        status_counts.values,
        color=["#0ee9de", "#ba91dd"],
        edgecolor="black",
    )

    total = status_counts.sum()
    # Annotate each bar with both absolute count and percent of total.
    for bar, value in zip(bars, status_counts.values):
        pct = (100 * value / total) if total else 0
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{int(value)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title("Loan Approval Distribution", fontsize=13, pad=10)
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Applications")
    plt.grid(visible=True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("loan_status_bar.png", dpi=300)
    plt.show()


def loan_boxplot_income_by_status(df: pd.DataFrame):
    """
    Visualization 2:
    Boxplot of applicant income by loan status.

    Goal: explore whether income differs between approved and not approved loans.
    """
    plt.figure(figsize=(7, 5))

    # Drop missing values so the boxplot and medians are computed on valid rows.
    subset = df[["ApplicantIncome", "Loan_Status"]].dropna()

    approved_income = subset[subset["Loan_Status"] == "Y"]["ApplicantIncome"]
    not_approved_income = subset[subset["Loan_Status"] == "N"]["ApplicantIncome"]

    box = plt.boxplot(
        [approved_income, not_approved_income],
        tick_labels=["Approved (Y)", "Not Approved (N)"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    box["boxes"][0].set_facecolor("#0ee9de")
    box["boxes"][1].set_facecolor("#ba91dd")
    box["boxes"][0].set_alpha(0.75)
    box["boxes"][1].set_alpha(0.75)

    plt.title("Applicant Income by Loan Status (Log Scale)", fontsize=13, pad=10)
    plt.xlabel("Loan Status")
    plt.ylabel("Applicant Income (log scale)")
    # Income is often right-skewed; log scale improves readability of spread.
    plt.yscale("log")
    plt.grid(visible=True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("loan_income_boxplot.png", dpi=300)
    plt.show()


def loan_credit_history_approval_rate(df: pd.DataFrame):
    """
    Visualization 3:
    Approval rate by credit history group.

    Goal: discover whether credit history has a strong relationship to approval.
    """
    # This chart is optional if the file does not contain a credit-history variable.
    if "Credit_History" not in df.columns:
        print("Skipping credit-history chart: 'Credit_History' column not found.")
        return

    plt.figure(figsize=(7, 5))
    # Keep only the two required columns and remove rows with missing values.
    subset = df[["Credit_History", "Loan_Status"]].dropna().copy()
    subset["approved_flag"] = (subset["Loan_Status"] == "Y").astype(int)

    # Group by credit history and compute approval rate as a percentage.
    rates = subset.groupby("Credit_History")["approved_flag"].mean() * 100
    rates = rates.sort_index()

    labels = [f"Credit_History={value}" for value in rates.index]
    bars = plt.bar(labels, rates.values, color=["#0ee9de", "#ba91dd"], edgecolor="black")

    for bar, rate in zip(bars, rates.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0, 100)
    plt.title("Loan Approval Rate by Credit History", fontsize=13, pad=10)
    plt.xlabel("Credit History")
    plt.ylabel("Approval Rate (%)")
    plt.grid(visible=True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("loan_credit_history_approval_rate.png", dpi=300)
    plt.show()


def loan_analysis_paragraph(df: pd.DataFrame) -> str:
    """
    Generate a brief analysis paragraph based on the loan visualizations.

    """
    # Recompute the same summary values used in the visualizations.
    status_counts = df["Loan_Status"].value_counts()
    total = status_counts.sum()
    approved_pct = 100 * status_counts.get("Y", 0) / total if total else 0
    not_approved_pct = 100 * status_counts.get("N", 0) / total if total else 0

    # Median income is more robust than mean for skewed income distributions.
    income_subset = df[["ApplicantIncome", "Loan_Status"]].dropna()
    approved_median = income_subset[income_subset["Loan_Status"] == "Y"]["ApplicantIncome"].median()
    not_approved_median = income_subset[income_subset["Loan_Status"] == "N"]["ApplicantIncome"].median()

    # Add credit-history interpretation only when that column exists.
    credit_sentence = ""
    if "Credit_History" in df.columns:
        credit_subset = df[["Credit_History", "Loan_Status"]].dropna().copy()
        credit_subset["approved_flag"] = (credit_subset["Loan_Status"] == "Y").astype(int)
        rates = credit_subset.groupby("Credit_History")["approved_flag"].mean() * 100
        if len(rates) >= 2:
            credit_sentence = (
                f" The credit-history chart shows the clearest pattern, with approval rates "
                f"ranging from about {rates.min():.1f}% to {rates.max():.1f}% across groups."
            )

    # Final paragraph ties the computed values back to the chart story.
    paragraph = (
        f"The visualizations indicate that approximately {approved_pct:.1f}% of loans are approved "
        f"and {not_approved_pct:.1f}% are not approved. The income boxplot (using a log scale) shows "
        f"substantial overlap between groups; the median income for approved applications is "
        f"{approved_median:,.0f}, compared with {not_approved_median:,.0f} for non-approved "
        f"applications, suggesting income alone is not a complete explanation of approval outcomes."
        f"{credit_sentence} These conclusions are supported by the approval distribution chart, the "
        f"income-by-status boxplot, and the approval-rate comparison by credit history."
    )
    return paragraph


# =========================
# Main
# =========================

def main():
    # Run all Iris visualizations first.
    iris_df = load_iris_data()
    iris_scatter_petal(iris_df)
    iris_boxplot_sepal(iris_df)
    iris_subplots_pairwise(iris_df)

    # Load loan dataset; if missing, print a clear message and stop loan section.
    try:
        loan_df = load_loan_data()
    except FileNotFoundError as error:
        print("\nLoan section skipped:")
        print(error)
        return

    # Run all three loan visualizations.
    loan_bar_loan_status(loan_df)
    loan_boxplot_income_by_status(loan_df)
    loan_credit_history_approval_rate(loan_df)

    # Print analysis paragraph for your write-up
    print("\n=== Loan Data Analysis Paragraph ===\n")
    print(loan_analysis_paragraph(loan_df))


if __name__ == "__main__":
    main()
