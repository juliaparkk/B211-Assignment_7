# Assignment 7 - Matplotlib Visual Analysis

## Project Purpose
This project performs visual analysis on two datasets using Matplotlib:

1. **Fisher's Iris dataset** (from sklearn) to compare physical flower traits across species.
2. **Loan dataset** (CSV file) to explore patterns related to loan outcomes.

The goal is to demonstrate how visualizations can communicate patterns, differences, and possible relationships in real data.

---

## Design and Implementation Overview
This project uses a **functional design** (separate functions) instead of an object-oriented class design.

### Why this design was used
- The assignment focuses on plotting and data analysis workflow.
- A function-by-function structure is easier to read for beginner/intermediate Python users.
- Each function has one clear responsibility (load data, plot one chart, or generate analysis text).

### Class Design Note
- **No custom classes were implemented** in this project.
- In this context, each function acts like a simple "method" for one task in the pipeline.
- Data is passed between functions using pandas DataFrames.

---

## File in this Project
- `Assignment 7 Matplotlib.py`: main script that loads data, creates six visualizations, and prints a summary paragraph.

---

## Data Model (Attributes / Fields Used)

### Iris DataFrame fields
- `sepal_length`: sepal length in centimeters.
- `sepal_width`: sepal width in centimeters.
- `petal_length`: petal length in centimeters.
- `petal_width`: petal width in centimeters.
- `species`: categorical species label (`setosa`, `versicolor`, `virginica`).

### Loan DataFrame fields (normalized for plotting)
The script standardizes possible CSV schemas into the following fields:
- `Loan_Status`: Y/N category used for status-based visualizations.
- `ApplicantIncome`: numeric income field used in boxplot and analysis.
- `Credit_History`: numeric/encoded credit history group used in approval-rate chart (when available).

---

## Function-by-Function Documentation

### 1) `load_iris_data()`
**Purpose:** Load and clean Iris data.

**Implementation details:**
- Reads Iris data from `sklearn.datasets`.
- Renames verbose feature names to snake_case for cleaner code.
- Converts numeric target labels to species names.

**Returns:**
- A cleaned pandas DataFrame with five columns.

---

### 2) `iris_scatter_petal(df)`
**Purpose:** Show species separation in petal space.

**Implementation details:**
- Creates a scatter plot of `petal_length` vs `petal_width`.
- Draws one point layer per species with a consistent color palette.
- Adds labels, legend, grid, and saves the figure.

**Output file:**
- `iris_petal_scatter.png`

---

### 3) `iris_boxplot_sepal(df)`
**Purpose:** Compare the distribution of sepal length by species.

**Implementation details:**
- Creates a boxplot using sepal length values grouped by species.
- Uses `tick_labels` for category labels.
- Applies custom colors and median styling.

**Output file:**
- `iris_sepal_boxplot.png`

---

### 4) `iris_subplots_pairwise(df)`
**Purpose:** Compare multiple Iris feature relationships in a single 2x2 panel.

**Implementation details:**
- Creates 4 scatter subplots with different x/y feature pairings.
- Reuses the same species colors across all subplots for consistency.
- Uses a shared figure legend and saves the panel.

**Output file:**
- `iris_pairwise_subplots.png`

---

### 5) `load_loan_data(path=...)`
**Purpose:** Load the loan CSV and normalize schema differences.

**Implementation details:**
- Attempts to load from provided path, script folder relative path, or auto-detected loan CSV name.
- Standardizes headers to lowercase with underscores.
- Maps source columns into expected plotting fields (`Loan_Status`, `ApplicantIncome`, `Credit_History`).
- Raises clear errors if required fields are missing.

**Returns:**
- A cleaned pandas DataFrame ready for loan visualizations.

---

### 6) `loan_bar_loan_status(df)`
**Purpose:** Show status distribution (Y vs N).

**Implementation details:**
- Counts values in `Loan_Status`.
- Draws bar chart with annotations for count and percentage.
- Adds chart formatting and saves image.

**Output file:**
- `loan_status_bar.png`

---

### 7) `loan_boxplot_income_by_status(df)`
**Purpose:** Compare income distributions by status.

**Implementation details:**
- Uses `ApplicantIncome` grouped by status.
- Applies log scale on y-axis to handle skewed income values.
- Uses colored boxplots and saves image.

**Output file:**
- `loan_income_boxplot.png`

---

### 8) `loan_credit_history_approval_rate(df)`
**Purpose:** Show approval-rate trend by credit history group.

**Implementation details:**
- Builds a binary flag from status (Y=1, N=0).
- Groups by `Credit_History` and computes mean percentage.
- Plots bar chart and annotates percentage values.
- Skips gracefully if `Credit_History` is unavailable.

**Output file:**
- `loan_credit_history_approval_rate.png`

---

### 9) `loan_analysis_paragraph(df)`
**Purpose:** Generate a short written interpretation of the loan charts.

**Implementation details:**
- Computes approval percentages.
- Computes median income by status.
- Optionally computes range of approval rates by credit history.
- Returns one formatted paragraph string.

**Returns:**
- A narrative paragraph summarizing key findings.

---

### 10) `main()`
**Purpose:** Orchestrate the full analysis pipeline.

**Implementation details:**
- Loads Iris data, creates three Iris charts.
- Loads Loan data, creates three Loan charts.
- Prints analysis paragraph.
- Handles missing loan file with a user-friendly message.

---

## How to Run
1. Open terminal in this project folder.
2. Run:

```bash
python "Assignment 7 Matplotlib.py"
```

3. Generated PNG figures will be saved in the same folder.

---

## Dependencies
- Python 3.x
- pandas
- matplotlib
- scikit-learn

Install if needed:

```bash
pip install pandas matplotlib scikit-learn
```

---

## Current Limitations
1. **No custom classes:** The project is function-based, not class-based.
2. **Schema assumptions:** Loan data normalization supports common column variants but may fail for unexpected schemas.
3. **Status mapping assumptions:** Y/N meaning depends on source dataset labels and may represent different business definitions.
4. **Static chart settings:** Figure sizes, colors, and styles are hard-coded.
5. **No statistical testing:** Insights are descriptive visual findings, not inferential statistical proof.
6. **No automated tests:** The script has no unit tests for data mapping or plotting logic.

---

## Future Improvements
1. Convert to class-based architecture (for example, `IrisAnalyzer` and `LoanAnalyzer`).
2. Add configuration file for paths, colors, and output names.
3. Add schema validation and warnings for ambiguous label mappings.
4. Add optional statistical summaries (confidence intervals, significance tests).
5. Add automated tests for loader and transformation logic.

---
## Loan Data Analysis Paragraph 
The visualizations indicate that approximately 79.0% of loans are approved and 21.0% are not approved. The income boxplot (using a log scale) shows substantial overlap between groups; the median income for approved applications is 60,000, compared with 42,000 for non-approved applications, suggesting income alone is not a complete explanation of approval outcomes. The credit-history chart shows the clearest pattern, with approval rates ranging from about 20.1% to 62.9% across groups. These conclusions are supported by the approval distribution chart, the income-by-status boxplot, and the approval-rate comparison by credit history.

## Author Notes
This README is designed to clearly communicate purpose, structure, implementation choices, and known constraints for grading and project review.
