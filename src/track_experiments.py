
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

from train import (
    ARTIFACT_DIR,
    CATEGORICAL_COLUMNS,
    DATA_FILE,
    NUMERIC_COLUMNS,
    build_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
TRACKING_DB = ROOT / "mlflow.db"
PLOT_DIR = ROOT / "reports" / "eda_plots"

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment("HeartDisease-Model-Comparison")


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = (df["num"] > 0).astype(int)

    fingerprint = __import__("hashlib").sha256(
        DATA_FILE.read_bytes()
    ).hexdigest()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"max_iter": 1000, "random_state": 42},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=42,
            ),
            {
                "n_estimators": 200,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for name, (estimator, params) in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = build_pipeline(estimator)

            cv_results = cross_validate(
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
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
                "cv_accuracy_std": cv_results["test_accuracy"].std(),
                "cv_precision_mean": cv_results["test_precision"].mean(),
                "cv_recall_mean": cv_results["test_recall"].mean(),
                "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            mlflow.log_params(params)
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("target_definition", "num > 0")
            mlflow.log_param("dataset_sha256", fingerprint)
            mlflow.log_metrics({
                key: float(value)
                for key, value in metrics.items()
            })

            # Log the actual model and its preprocessing pipeline.
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            # Attach EDA plots to each run for convenient review.
            for plot in PLOT_DIR.glob("*.png"):
                mlflow.log_artifact(str(plot), artifact_path="eda")

            print(f"\n{name}")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

    print(f"\nMLflow tracking database: {TRACKING_DB}")
    print("Experiment: HeartDisease-Model-Comparison")


if __name__ == "__main__":
    main()
