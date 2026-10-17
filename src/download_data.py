
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "heart_disease_uci.csv"


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Heart Disease dataset from UCI...")
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features
    target = dataset.data.targets
    df = pd.concat([features, target], axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDataset shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    target_column = target.columns[0]
    print(f"\nTarget distribution ({target_column}):")
    print(df[target_column].value_counts(dropna=False).sort_index())

    print(f"\nSaved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

