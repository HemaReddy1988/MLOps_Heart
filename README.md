# HeartGuard: Reproducible Heart Disease Prediction

## 1. Overview

HeartGuard is an educational MLOps project using the UCI Heart Disease dataset. It demonstrates data acquisition, exploratory data analysis (EDA), preprocessing, model comparison, experiment tracking, automated testing, API serving, containerization, and continuous integration.

**Disclaimer:** This project is intended for educational purposes only. It is not clinically validated and must not be used for diagnosis or medical decision-making.

## 2. Dataset

The project uses the UCI Heart Disease dataset, acquired using `ucimlrepo`.

- Dataset size: 303 rows and 14 original columns.
- Original target: `num`.
- Binary target: `0` when `num == 0`, and `1` when `num > 0`.
- Missing values in the original dataset: 4 in `ca` and 2 in `thal`.
- Binary target distribution: 164 negative and 139 positive cases.

The dataset is downloaded at runtime and is not committed to the repository.

## 3. Technology Stack

- Python
- pandas, NumPy, Matplotlib and Seaborn
- scikit-learn
- MLflow
- pytest
- FastAPI and Uvicorn
- Docker
- GitHub Actions

## 4. Project Structure

- `src/download_data.py` — downloads the dataset.
- `src/eda.py` — performs data quality checks and generates EDA outputs.
- `src/train.py` — trains and evaluates Logistic Regression and Random Forest pipelines.
- `src/track_experiments.py` — logs experiment results using MLflow.
- `src/api.py` — serves predictions through FastAPI.
- `tests/` — contains pipeline and API tests.
- `reports/` — contains EDA outputs and report evidence.
- `artifacts/` — contains locally generated trained models and model comparison results.
- `Dockerfile` — defines the container image.
- `.github/workflows/ci.yml` — defines the automated CI workflow.

## 5. Environment Setup

### Linux / University VM

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 6. Data Acquisition and EDA

Download the dataset:

```bash
python src/download_data.py
```

Generate EDA summaries and plots:

```bash
python src/eda.py
```

Outputs are saved under `reports/`, including the EDA plots and summary.

## 7. Model Training

Run:

```bash
python src/train.py
```

The pipeline compares Logistic Regression and Random Forest using a stratified train/test split and stratified cross-validation. Preprocessing includes median imputation and standardization for numeric features, and most-frequent imputation and one-hot encoding for categorical features.

The script saves trained pipelines and model comparison results under `artifacts/`.

## 8. Experiment Tracking

Run:

```bash
python src/track_experiments.py
```

The experiment is named `HeartDisease-Model-Comparison`. MLflow records parameters, metrics, model artifacts and related experiment information.

To open the MLflow UI locally:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in a browser on the same machine.

## 9. Automated Testing

Run:

```bash
python -m pytest -v
```

The current test suite contains seven tests covering dataset structure, binary target handling, pipeline predictions, missing-value handling, API health, valid predictions and invalid input validation.

## 10. Prediction API

**Generate the model artifacts before starting the API.**

Start the service:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Prediction endpoint: `POST /predict`

Example request:

```json
{
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
  "thal": 3
}
```

The API returns a binary prediction, the estimated probability of class 1, the model name and an educational-use disclaimer.

## 11. Docker

**Build the model artifacts before building the image.**

```bash
python src/download_data.py
python src/train.py
docker build -t heartguard-api:1.0 .
docker run -d --name heartguard -p 8000:8000 heartguard-api:1.0
```

Verify the running container:

```bash
docker ps
curl http://127.0.0.1:8000/health
```

Open `http://127.0.0.1:8000/docs` in a browser on the machine hosting the container.

To stop the container:

```bash
docker stop heartguard
```

The host must permit Docker access and the selected port. External access depends on the host's network and firewall configuration.

## 12. Continuous Integration

The GitHub Actions workflow in `.github/workflows/ci.yml` installs dependencies, downloads the dataset, trains the models and runs the automated tests on pushes and pull requests targeting `main` or `master`.

The latest recorded workflow run passed.

## 13. Evaluation Results

Observed evaluation results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Cross-validation accuracy | 0.8471 | 0.8139 |
| Cross-validation ROC-AUC | 0.9025 | 0.8947 |
| Test accuracy | 0.8852 | 0.8689 |
| Test ROC-AUC | 0.9665 | 0.9481 |
| Test recall | 0.9286 | 0.9286 |

Logistic Regression was selected based on its observed cross-validation and held-out ROC-AUC results. The test set is small, so these estimates are preliminary and should not be interpreted as evidence of clinical suitability.

## 14. Limitations

- The dataset is small and is not representative of all patient populations.
- Evaluation metrics depend on the dataset split and preprocessing choices.
- The API is an educational demonstration, not a clinically validated service.
- Production use would require additional security, access control, monitoring, model validation and deployment controls.

## 15. Project Status

The dataset pipeline, EDA, model comparison, MLflow tracking, seven automated tests, GitHub Actions workflow, FastAPI service and Docker execution have been verified in the development environment. The project has also been reproduced and tested in the university virtual lab.

For submission, refer to the accompanying implementation evidence, report and demonstration video.


## 16. Kubernetes Deployment

The HeartGuard prediction API was deployed to a local Kubernetes cluster using Minikube in the university virtual lab.

### Kubernetes Resources

The deployment uses the following manifests:

- `k8s/deployment.yaml` — defines the API Deployment, container image, replica count, resource limits, and health probes.
- `k8s/service.yaml` — exposes the API using a Kubernetes NodePort Service.

### Deployment Steps

Start the Minikube cluster:

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```

Apply the Kubernetes manifests:

```bash
minikube kubectl -- apply -f k8s/
```

Check the deployment, pod, and service:

```bash
minikube kubectl -- get deployments,pods,services
```

Access the service:

```bash
minikube service heartguard-api-service --url
```

The API was verified through the Kubernetes Service. The `/health` endpoint returned HTTP 200, and the `/predict` endpoint returned a prediction response.

### Deployment Evidence

Screenshots document the running Kubernetes pod, deployment and service resources, and successful API endpoint tests.

### Deployment Scope

This implementation demonstrates deployment on a local Minikube Kubernetes cluster within the university virtual lab. It is not a public-cloud deployment, and external public accessibility has not been established.

## 17. Monitoring and Logging

HeartGuard includes application logging and Prometheus-compatible HTTP metrics.

### Request Logging

The FastAPI middleware records:
- HTTP method and request path
- HTTP status code
- Request duration in seconds
- Request ID in the response header (`X-Request-ID`)

Application logs are written to the container's standard output and can be inspected using Kubernetes logs.

### Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-formatted metrics, including:
- `heartguard_http_requests_total` — total HTTP requests, labelled by method, endpoint, and status.
- `heartguard_http_request_duration_seconds` — request latency histogram.

### Verification

The monitoring endpoint was tested through the Kubernetes NodePort Service:

```bash
curl http://192.168.49.2:30080/metrics
```

Request counters and latency metrics were verified after API traffic was generated.

The metrics endpoint is available for Prometheus scraping. A Prometheus/Grafana dashboard has not been configured in this implementation.
