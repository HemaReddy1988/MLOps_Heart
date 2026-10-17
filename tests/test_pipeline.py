import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import (
    DATA_FILE,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    build_pipeline,
)

from sklearn.linear_model import LogisticRegression


def test_dataset_exists_and_has_required_columns():
    assert DATA_FILE.exists()

    df = pd.read_csv(DATA_FILE)
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["num"]

    assert all(column in df.columns for column in required)
    assert len(df) > 0


def test_target_is_binary():
    df = pd.read_csv(DATA_FILE)
    target = (df["num"] > 0).astype(int)

    assert set(target.unique()) == {0, 1}


def test_pipeline_predicts_binary_classes():
    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = (df["num"] > 0).astype(int)

    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    )
    pipeline.fit(X, y)

    predictions = pipeline.predict(X.iloc[:10])

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})


def test_pipeline_handles_missing_feature_values():
    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = (df["num"] > 0).astype(int)

    pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    )
    pipeline.fit(X, y)

    # The source dataset contains missing values in ca and thal.
    sample = X.iloc[:10].copy()
    predictions = pipeline.predict(sample)

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})
