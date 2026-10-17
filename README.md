# HeartGuard: Reproducible Heart Disease Prediction

## 1. Overview

HeartGuard is an educational MLOps project using the UCI Heart Disease dataset. It demonstrates data acquisition, exploratory data analysis (EDA), preprocessing, model comparison, experiment tracking, automated testing, API serving, containerization, continuous integration, Kubernetes deployment, and API monitoring.

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
- Prometheus Client
- Docker
- Kubernetes and Minikube
- GitHub Actions

## 4. Project Structure

- `src/download_data.py` — downloads the dataset.
- `src/eda.py` — performs data quality checks and generates EDA outputs.
- `src/train.py` — trains and evaluates Logistic Regression and Random Forest pipelines.
- `src/track_experiments.py` — logs experiment results using MLflow.
- `src/api.py` — serves predictions through FastAPI and exposes Prometheus-compatible metrics.
- `tests/` — contains pipeline and API tests.
- `reports/` — contains EDA outputs and report evidence.
- `artifacts/` — contains locally generated trained models and model comparison results.
- `k8s/deployment.yaml` — defines the Kubernetes Deployment, container image, health probes and resource settings.
- `k8s/service.yaml` — exposes the API through a Kubernetes LoadBalancer Service.
- `Dockerfile` — defines the container image.
- `.github/workflows/ci.yml` — defines the automated CI workflow.

Generated data and model artifacts are excluded from Git where configured in `.gitignore`. They must be generated before building the Docker image from a fresh checkout.

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

Outputs are saved under `reports/`, including EDA plots and the summary report.

## 7. Model Training

Run:

```bash
python src/train.py
```

The pipeline compares Logistic Regression and Random Forest using a stratified train/test split and stratified cross-validation.

Preprocessing includes median imputation and standardization for numeric features, and most-frequent imputation and one-hot encoding for categorical features.

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

Open `http://127.0.0.1:5000` in a browser on the same machine.

## 9. Automated Testing

Run:

```bash
python -m pytest -v
```

The current test suite contains seven tests covering dataset structure, binary target handling, pipeline predictions, missing-value handling, API health, valid predictions and invalid input validation.

The seven tests passed in the university VM after the monitoring changes were added.

## 10. Prediction API

Generate the model artifacts before starting the API.

Start the service:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Health endpoint

```bash
curl http://127.0.0.1:8000/health
```

### Prediction endpoint

`POST /predict`

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

The API returns a binary prediction, estimated probability of class 1, model identifier and educational-use disclaimer.

## 11. Docker

Generate the model artifacts before building the image from a fresh checkout:

```bash
python src/download_data.py
python src/train.py
docker build -t heartguard-api:1.1 .
```

Run the container locally:

```bash
docker run -d --name heartguard -p 8000:8000 heartguard-api:1.1
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

The GitHub Actions workflow in `.github/workflows/ci.yml` installs dependencies, downloads the dataset, trains the models and runs automated tests on pushes and pull requests targeting `main` or `master`.

The recorded CI workflow completed successfully. Check the repository's Actions page for the latest run status.

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
- The implementation does not include a configured Prometheus/Grafana dashboard.
- The Minikube LoadBalancer is local to the university virtual lab; public internet accessibility has not been established.
- Production use would require additional security, access control, monitoring, model validation and deployment controls.

## 15. Project Status

The dataset pipeline, EDA, model comparison, MLflow tracking, seven automated tests, GitHub Actions workflow, FastAPI service and Docker execution have been verified.

The API was also deployed to a local Minikube Kubernetes cluster in the university virtual lab. The Kubernetes LoadBalancer Service was tested, and the health, prediction and monitoring endpoints responded successfully.

For submission, refer to the accompanying implementation evidence document, assignment report and demonstration video.

## 16. Kubernetes Deployment

The HeartGuard prediction API was deployed to a local Kubernetes cluster using Minikube in the university virtual lab.

### Kubernetes Resources

- `k8s/deployment.yaml` — defines the API Deployment, container image, replica count, resource limits and health probes.
- `k8s/service.yaml` — exposes the API through a Kubernetes Service of type `LoadBalancer`.

The deployed container image is `heartguard-api:1.1`, which includes request logging and Prometheus metrics.

### Deployment Steps

Start the Minikube cluster:

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```

Verify the Kubernetes node:

```bash
minikube kubectl -- get nodes
```

Apply the Kubernetes manifests:

```bash
minikube kubectl -- apply -f k8s/
```

Check the Deployment, Pods and Services:

```bash
minikube kubectl -- get deployments,pods,services
```

Wait for the LoadBalancer tunnel to be available. In a separate terminal, start the tunnel and keep it running:

```bash
minikube tunnel
```

In the original terminal, retrieve the assigned LoadBalancer IP:

```bash
minikube kubectl -- get service heartguard-api-service
```

For the verified university lab session, the assigned private IP was `10.108.235.141`. This address may change when the cluster or tunnel is recreated.

### Verify the Endpoints

Health check:

```bash
curl -i http://10.108.235.141:8000/health
```

Prediction endpoint:

```bash
curl -i -X POST http://10.108.235.141:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"trestbps":130,"chol":240,"thalach":150,"oldpeak":1.0,"sex":1,"cp":2,"fbs":0,"restecg":1,"exang":0,"slope":2,"ca":0,"thal":3}'
```

The health and prediction endpoints returned HTTP 200 during verification.

### Deployment Scope

This implementation demonstrates a local Kubernetes deployment with a LoadBalancer Service and Minikube tunnel. The assigned address is private to the lab environment. This is not a public-cloud deployment, and external public accessibility has not been established.

Screenshots of the running Pod, Deployment, Service and endpoint verification are included in the implementation evidence document.

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

- `heartguard_http_requests_total` — total HTTP requests, labelled by method, endpoint and status.
- `heartguard_http_request_duration_seconds` — request latency histogram.

### Verification

The monitoring endpoint was tested through the Kubernetes LoadBalancer Service:

```bash
curl http://10.108.235.141:8000/metrics
```

Request counters and latency histograms were verified after generating API traffic. The metrics endpoint is available for Prometheus scraping.

A Prometheus/Grafana dashboard has not been configured in this implementation.
