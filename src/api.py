
from functools import lru_cache
from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "logistic_regression.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heartguard")

app = FastAPI(
    title="HeartGuard Prediction API",
    description=(
        "Educational heart disease classification demo. "
        "Not intended for clinical diagnosis."
    ),
    version="1.0.0",
)

NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


class HeartFeatures(BaseModel):
    age: float = Field(gt=0, le=120)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    thalach: float = Field(gt=0, le=300)
    oldpeak: float = Field(ge=0, le=15)

    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    exang: int = Field(ge=0, le=1)
    slope: int | None = Field(default=None, ge=1, le=3)
    ca: float | None = Field(default=None, ge=0, le=3)
    thal: float | None = Field(default=None, ge=3, le=7)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    model_available = MODEL_PATH.is_file()
    return {
        "status": "ok" if model_available else "unavailable",
        "model_loaded": model_available,
    }


@app.post("/predict")
def predict(features: HeartFeatures):
    try:
        model = load_model()
        row = pd.DataFrame(
            [features.model_dump()],
            columns=FEATURE_COLUMNS,
        )

        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])

        logger.info("Prediction request completed successfully")

        return {
            "prediction": prediction,
            "probability_of_class_1": round(probability, 4),
            "model": "logistic_regression",
            "disclaimer": (
                "Educational output; not a medical diagnosis."
            ),
        }

    except FileNotFoundError:
        logger.exception("Trained model is unavailable")
        raise HTTPException(
            status_code=503,
            detail="Trained model is unavailable.",
        )
    except Exception:
        logger.exception("Prediction request failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check application logs.",
        )
