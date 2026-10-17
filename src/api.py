
from functools import lru_cache
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "logistic_regression.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heartguard")

REQUEST_COUNT = Counter(
    "heartguard_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "heartguard_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

app = FastAPI(
    title="HeartGuard Prediction API",
    description=(
        "Educational heart disease classification demo. "
        "Not intended for clinical diagnosis."
    ),
    version="1.1.0",
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


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception(
            "request_id=%s method=%s path=%s unhandled_error",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        path = request.url.path

        # Avoid high-cardinality metrics from arbitrary URL paths.
        endpoint = path if path in {
            "/health", "/predict", "/metrics"
        } else "other"

        REQUEST_COUNT.labels(
            request.method, endpoint, str(status_code)
        ).inc()
        REQUEST_LATENCY.labels(
            request.method, endpoint
        ).observe(duration)

        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_seconds=%.4f",
            request_id,
            request.method,
            path,
            status_code,
            duration,
        )


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


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


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
