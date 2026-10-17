
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
PLOT_DIR = REPORT_DIR / "eda_plots"


NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    # Preserve the original num target and derive a binary target separately.
    df["target_binary"] = (df["num"] > 0).astype(int)

    print("DATA QUALITY AUDIT")
    print("=" * 40)
    print(f"Rows: {len(df)}")
    print(f"Columns before binary target: {len(df.columns) - 1}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nOriginal target distribution:")
    print(df["num"].value_counts().sort_index().to_string())

    print("\nBinary target distribution:")
    print(df["target_binary"].value_counts().sort_index().to_string())

    print("\nNumeric feature summary:")
    print(df[NUMERIC_COLUMNS].describe().round(2).to_string())

    # 1. Class balance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target_binary")
    plt.title("Heart Disease Dataset: Binary Class Distribution")
    plt.xlabel("Target (0 = absence, 1 = presence)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "class_balance.png", dpi=150)
    plt.close()

    # 2. Numeric feature distributions
    df[NUMERIC_COLUMNS].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distributions of Numeric Features")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "numeric_distributions.png", dpi=150)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=False,
    )
    plt.title("Feature and Target Correlation Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. Missing-value counts
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        plt.figure(figsize=(7, 4))
        missing.plot(kind="bar")
        plt.title("Missing Values by Column")
        plt.xlabel("Column")
        plt.ylabel("Missing count")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "missing_values.png", dpi=150)
        plt.close()

    # Save an audit summary for reporting.
    summary = [
        f"Rows: {len(df)}",
        f"Original columns: {len(df.columns) - 1}",
        f"Duplicate rows: {df.duplicated().sum()}",
        f"Missing values:\n{df.isna().sum().to_string()}",
        f"Original target distribution:\n{df['num'].value_counts().sort_index().to_string()}",
        f"Binary target distribution:\n{df['target_binary'].value_counts().sort_index().to_string()}",
    ]

    (REPORT_DIR / "eda_summary.txt").write_text(
        "\n\n".join(summary), encoding="utf-8"
    )

    print(f"\nPlots saved to: {PLOT_DIR}")
    print(f"Summary saved to: {REPORT_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()
