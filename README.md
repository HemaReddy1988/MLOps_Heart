# HeartGuard: Reproducible Heart Disease Prediction

## Overview

HeartGuard is an educational MLOps project using the UCI Heart Disease dataset. It demonstrates data acquisition, exploratory data analysis, preprocessing, model comparison, experiment tracking, and automated testing.

This project is for learning and experimentation. It is not intended for clinical diagnosis or medical decision-making.

## Dataset

The project uses the UCI Heart Disease dataset obtained through `ucimlrepo`.

The original target column is `num`. For binary classification, the target is defined as:
- `0`: `num == 0`
- `1`: `num > 0`

## Technology Stack

- Python
- pandas, NumPy, Matplotlib and Seaborn
- scikit-learn
- MLflow
- pytest
- FastAPI and Uvicorn (API implementation planned)

## Setup

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Acquisition

Download the dataset:

```powershell
python src\download_data.py
```

## Exploratory Data Analysis

Run:

```powershell
python src\eda.py
```

EDA plots and summary outputs are saved under `reports`.

## Model Training

Run:

```powershell
python src\train.py
```

The training script compares Logistic Regression and Random Forest models.

## Experiment Tracking

Run:

```powershell
python src\track_experiments.py
```

The experiment is named `HeartDisease-Model-Comparison`. MLflow records model parameters, evaluation metrics, and model artifacts.

## Automated Tests

Run:

```powershell
python -m pytest -v
```

## Current Evaluation

The current experiment results favor Logistic Regression based on cross-validation and held-out ROC-AUC. These results are preliminary and should not be interpreted as evidence of clinical suitability.

## Project Status

Data acquisition, EDA, model comparison, MLflow tracking, and baseline automated tests have been implemented. Packaging validation, CI/CD, API serving, containerization, deployment, monitoring, and final reporting are being developed.
