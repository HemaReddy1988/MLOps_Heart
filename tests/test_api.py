
from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "age": 55,
        "trestbps": 130,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
        "sex": 1,
        "cp": 2,
        "fbs": 0,
        "restecg": 1,
        "exang": 0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability_of_class_1"] <= 1.0
    assert result["model"] == "logistic_regression"


def test_prediction_rejects_invalid_input():
    payload = {
        "age": 500,
        "trestbps": 130,
        "chol": 240,
        "thalach": 150,
        "oldpeak": 1.0,
        "sex": 1,
        "cp": 2,
        "fbs": 0,
        "restecg": 1,
        "exang": 0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
