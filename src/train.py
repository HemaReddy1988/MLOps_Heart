
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "heart_disease_uci.csv"
ARTIFACT_DIR = ROOT / "artifacts"

NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang",
    "slope", "ca", "thal"
]


def build_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", model),
    ])


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = (df["num"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            },
            return_train_score=False,
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
            "cv_accuracy_std": cv_scores["test_accuracy"].std(),
            "cv_precision_mean": cv_scores["test_precision"].mean(),
            "cv_recall_mean": cv_scores["test_recall"].mean(),
            "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
        }

        results.append(metrics)

        print(f"\n{'=' * 55}")
        print(name)
        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

        # Save each complete preprocessing/model pipeline.
        joblib.dump(
            pipeline,
            ARTIFACT_DIR / f"{name}.joblib",
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        ARTIFACT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nSaved model comparison to artifacts/model_comparison.csv")
    print("Saved both model pipelines to artifacts/")


if __name__ == "__main__":
    main()
