# MLOps Assignment 01 — Heart Disease Classification
## Documentation & Reporting

**Course:** Machine Learning Operations (MLOps) — AIMLCZG523  
**Assignment:** End-to-End ML Model Development, CI/CD, and Production Deployment  
**Student:** Shaba Joseph / 2025AE05285  
**Date:** 20 October 2026  
**GitHub Repository:** [MLOPS](https://github.com/shabajoseph-git/MLOPS)

---

## 1. Executive Summary

This project demonstrates an end-to-end machine-learning workflow for heart-disease classification. The workflow includes exploratory data analysis (EDA), model training and evaluation, experiment tracking with MLflow, automated testing, a FastAPI inference service, containerization, and Kubernetes deployment.

The implementation uses Python modules and does not need to be converted into a Jupyter notebook unless the assignment instructions explicitly require one.

### Development Evidence

- **MLflow experiment:** `heart-disease-classification`
- **Experiment ID:** `952052864617791423`
- **Tracking store:** Local `mlruns/` directory
- **Recorded MLflow history:** 12 finished runs and 4 failed runs at the time checked
- **Automated tests:** `pytest -q` reported 3 passed
- **API health check:** Returned HTTP 200 with `{"status":"ok","model_loaded":"true"}`
- **Kubernetes deployment:** `heart-disease-api`; pod previously observed as `1/1 Running`

These results reflect the latest recorded checks and should be reverified before final submission.

---

## 2. Setup and Installation

### 2.1 Prerequisites

Install the following tools:

- Python 3.11
- Git
- Docker Desktop
- Minikube
- `kubectl`

Docker Desktop must be running before building or starting containers.

### 2.2 Package Dependencies

| Package | Purpose |
|---|---|
| `ucimlrepo` | Downloads datasets from the UCI Machine Learning Repository |
| `pandas`, `numpy` | Data processing and numerical calculations |
| `scikit-learn` | Model training, preprocessing, and evaluation |
| `matplotlib`, `seaborn` | EDA charts and visualizations |
| `mlflow` | Experiment tracking, metrics, and model artifacts |
| `joblib` | Saving and loading model objects |
| `fastapi`, `uvicorn` | Prediction API and web server |
| `pydantic` | API request and response validation |
| `pytest`, `httpx` | Automated testing and API test requests |
| `ruff` | Code linting and quality checks |
| `prometheus-client` | Exposes application metrics for monitoring |

### 2.3 Clone and Install

Run the following commands in Windows PowerShell if setting up a fresh copy:

```powershell
git clone https://github.com/shabajoseph-git/MLOPS.git
cd mlops_heart_disease_starter
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If using the existing local project, navigate to:

```powershell
cd E:\SHABA\MLOPS\mlops_heart_disease_starter
.\.venv\Scripts\Activate.ps1
```

**Note:** Verify that the GitHub repository contains the project folder and that its directory structure matches the commands above. If the starter project is inside a subfolder, adjust the `cd` command accordingly.

### 2.4 Train the Model

```powershell
python -m src.train
```

Training should execute the configured preprocessing and model evaluation workflow and record experiment information in MLflow.

### 2.5 Start the API

Open a second PowerShell terminal:

```powershell
cd E:\SHABA\MLOPS\mlops_heart_disease_starter
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Check the health endpoint:

```powershell
curl http://127.0.0.1:8000/health
```

Previously observed response:

```json
{"status":"ok","model_loaded":"true"}
```

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Use the request schema shown in Swagger UI to submit a valid `/predict` request.

### 2.6 Tests, Linting, and MLflow UI

Run the tests:

```powershell
pytest -q
```

Run the linter:

```powershell
ruff check .
```

Start MLflow:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select the `heart-disease-classification` experiment.

Preserve the `mlruns/` directory if experiment history is required for submission.

### 2.7 Docker and Kubernetes

Ensure Docker Desktop is running, then build the image:

```powershell
docker build -t heart-disease-mlflow:latest .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 heart-disease-mlflow:latest
```

For Kubernetes, use the manifest paths that actually exist in the repository:

```powershell
minikube start
kubectl apply -f k8s/
kubectl get deployments
kubectl get pods
kubectl get services
minikube service heart-disease-api --url
```

Test the `/health` endpoint using the service URL printed by Minikube. If the manifest directory or deployment name differs, adjust the commands accordingly.

---

## 3. Exploratory Data Analysis (EDA)

The dataset used during development was `data/heart_disease.csv`.

The EDA should describe:

- Dataset dimensions and data types
- Missing values and duplicate records
- Target-label distribution
- Numeric-feature distributions
- Categorical-feature frequencies
- Relationships and correlations between features

EDA supports data-quality checks and helps inform preprocessing and model selection. Correlation does not establish causation.

### 3.1 EDA Evidence

Insert the actual outputs generated by the project. First verify the real filenames in the `artifacts` directory.

| Evidence | Suggested file location |
|---|---|
| Dataset summary | `artifacts/eda/summary.csv` |
| Missing-value summary | `artifacts/eda/missing_values.csv` |
| Target-label distribution | `artifacts/eda/class_balance.png` |
| Correlation heatmap | `artifacts/eda/correlation_heatmap.png` |
| Cholesterol distribution | `artifacts/eda/hist_chol.png` |
| Oldpeak distribution | `artifacts/eda/hist_oldpeak.png` |
| Resting blood pressure distribution | `artifacts/eda/hist_trestbps.png` |

**Important:** These are suggested paths. Use the exact paths and extensions of files that exist in your project.

Example Markdown for inserting a plot:

```markdown
### Figure 3.1: Target-Label Distribution

![Target-Label Distribution](artifacts/eda/class_balance.png)
```

---

## 4. Modelling Choices and Evaluation

### 4.1 Logistic Regression

Logistic Regression was used as a baseline model for binary classification. It is relatively fast to train and comparatively easy to interpret.

Preprocessing and feature scaling should be fitted only on the training data, preferably within a scikit-learn pipeline, to reduce data leakage.

### 4.2 Random Forest

Random Forest is a tree-ensemble model capable of capturing nonlinear relationships and feature interactions. It was evaluated using the same data split and evaluation metrics as Logistic Regression.

### 4.3 Evaluation Strategy

The experiment workflow records metrics including cross-validation accuracy, test accuracy, precision, recall, and ROC-AUC, according to the training configuration.

The evaluation process compares candidate models and uses the configured selection criterion to choose the final model.

### 4.4 Table 1.1: Model Performance Comparison

| Model | CV Accuracy | Test Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.855187 | 0.868852 | 0.812500 | 0.928571 | 0.966450 |
| Random Forest | 0.801616 | 0.885246 | 0.818182 | 0.964286 | 0.946429 |

*Note: CV accuracy and test-set ROC-AUC are different metrics. The selected model is based on the cross-validation ROC-AUC criterion described below.*

### 4.5 Selected Model

**Selected model: Logistic Regression**

Logistic Regression was selected based on its higher mean five-fold cross-validation ROC-AUC score of **0.902478**, compared with **0.898511** for Random Forest.

Random Forest achieved higher test accuracy and recall. However, Logistic Regression achieved a higher test ROC-AUC of **0.966450**, compared with **0.946429** for Random Forest.

ROC-AUC measures a classifier's ability to distinguish between the two classes across different classification thresholds.

Therefore, Logistic Regression was selected according to the cross-validation ROC-AUC selection criterion used in the experiment.

### 4.6 Evaluation Summary

The selected Logistic Regression model achieved the following recorded results:

| Metric | Score |
|---|---:|
| Cross-validation accuracy | 85.52% |
| Test accuracy | 86.89% |
| Precision | 81.25% |
| Recall | 92.86% |
| ROC-AUC | 96.65% |

The recall score indicates that the model identified a high proportion of positive cases in the test set. However, this is an educational project and the model must not be used as a substitute for professional medical diagnosis.

---

## 5. MLflow Experiment Tracking Summary

MLflow records model-training runs, parameters, metrics, and artifacts to support experiment comparison and reproducibility.

| Item | Recorded detail |
|---|---|
| Experiment name | `heart-disease-classification` |
| Experiment ID | `952052864617791423` |
| Tracking store | Local `mlruns/` directory |
| Run history at last report | 12 finished; 4 failed |
| Parameters | Model and training configuration, as logged |
| Metrics | Accuracy, precision, recall, ROC-AUC and/or cross-validation metrics, as configured |
| Artifacts | Saved model and evaluation artifacts, depending on the run |

Failed runs are part of the experiment history and should not be hidden. Identify the best completed run that corresponds to the model used by the API.

### 5.1 MLflow Screenshots

Save genuine screenshots from your own MLflow interface.

| Screenshot | Evidence |
|---|---|
| `screenshots/mlflow/01-experiment-overview.png` | Experiment name and run list |
| `screenshots/mlflow/02-best-run-metrics.png` | Metrics for the selected model |
| `screenshots/mlflow/03-run-parameters-artifacts.png` | Run parameters and saved artifacts |

Example Markdown:

```markdown
### Figure 5.1: MLflow Experiment Overview

![MLflow Experiment Overview](screenshots/mlflow/01-experiment-overview.png)
```

---

## 6. System Architecture

The following diagram describes the intended end-to-end workflow.

```mermaid
flowchart TD
    A[Heart disease CSV] --> B[Data validation and EDA]
    B --> C[Preprocessing and model training]
    C --> D[Logistic Regression / Random Forest evaluation]
    D --> E[MLflow: parameters, metrics and artifacts]
    D --> F[Saved model artifact]
    F --> G[FastAPI inference service]
    G --> H[/health endpoint]
    G --> I[/predict endpoint]
    J[Git push / pull request] --> K[GitHub Actions CI]
    K --> L[Lint and automated tests]
    G --> M[Docker image]
    M --> N[Kubernetes deployment]
    N --> O[Service health verification]
```

### 6.1 Component Responsibilities

- **EDA:** Checks data quality and explores feature distributions.
- **Model training:** Trains and compares candidate models.
- **MLflow:** Records experiments, parameters, metrics, and artifacts.
- **FastAPI:** Provides health and prediction endpoints.
- **GitHub Actions:** Automates configured code-quality and testing checks.
- **Docker:** Packages the application and its dependencies.
- **Kubernetes:** Runs the containerized application and supports deployment validation.

---

## 7. CI/CD and Deployment Workflow Evidence

The workflow used during development was named **MLOps CI**. Confirm the actual workflow file, such as `.github/workflows/ci.yml`, before documenting its exact triggers and steps.

A successful CI run demonstrates that the configured checks passed. It does not, by itself, prove automated production deployment unless deployment is explicitly configured in the workflow.

### 7.1 CI/CD Screenshots

| Suggested screenshot | Evidence to show |
|---|---|
| `screenshots/cicd/01-github-actions-workflow.png` | GitHub Actions page with a successful run |
| `screenshots/cicd/02-ci-job-steps.png` | Job steps and their results |
| `screenshots/cicd/03-workflow-file.png` | Actual workflow YAML showing checkout, Python setup, installation, linting, and tests |

### 7.2 Deployment Screenshots

| Suggested screenshot | Evidence to show |
|---|---|
| `screenshots/deployment/01-docker-build.png` | Successful Docker image build |
| `screenshots/deployment/02-docker-container.png` | Running container or container status |
| `screenshots/deployment/03-kubernetes-pods.png` | `kubectl get pods` showing the API pod running |
| `screenshots/deployment/04-kubernetes-service.png` | Kubernetes service details |
| `screenshots/deployment/05-health-check.png` | `/health` response showing status OK and model loaded |
| `screenshots/deployment/06-api-prediction.png` | Swagger UI or valid prediction request and response |

Use screenshots from the actual tool corresponding to each filename. Do not use a test screenshot as evidence of a Docker build, or a Docker screenshot as evidence of a Kubernetes deployment.

### 7.3 Monitoring Screenshots

Include these only if the corresponding monitoring functionality has been implemented and verified.

| Suggested screenshot | Evidence to show |
|---|---|
| `screenshots/monitoring/01-api-healthchecks.png` | API health-check response |
| `screenshots/monitoring/02-api-metrics.png` | Application metrics endpoint |
| `screenshots/monitoring/03-api-logs.png` | Application logs |
| `screenshots/monitoring/04-prometheus-metrics.png` | Prometheus metrics or dashboard |
| `screenshots/monitoring/05-kubernetes-pod.png` | Kubernetes pod status and details |

Do not claim Prometheus monitoring or a dashboard is working unless it has actually been configured and tested.

---

## 8. Testing and Validation

The latest known automated test result during report preparation was **3 passed tests**.

### 8.1 Final Validation Checklist

- [ ] `ruff check .` completes successfully.
- [ ] `pytest -q` passes.
- [ ] Training completes and creates the expected outputs.
- [ ] MLflow shows the experiment and completed runs.
- [ ] The selected model and metrics match the final report.
- [ ] `/health` returns status OK and confirms that the model is loaded.
- [ ] `/predict` accepts a valid request and returns a response.
- [ ] The Docker image builds successfully.
- [ ] The Docker container starts successfully.
- [ ] The Kubernetes deployment and pod are healthy, if required.
- [ ] The final GitHub Actions run succeeds.
- [ ] Screenshots match the actual final results.

### 8.2 Test Evidence

Example Markdown:

```markdown
### Figure 8.1: Automated Test Results

![Automated Test Results](screenshots/testing/01-pytest-results.png)
```

Capture a genuine terminal screenshot showing the complete `pytest -q` output.

---

## 9. Repository and Submission Checklist

**GitHub repository:** [https://github.com/shabajoseph-git/MLOPS](https://github.com/shabajoseph-git/MLOPS)

Verify the remote repository and working-tree status:

```powershell
git remote -v
git status
```

### 9.1 Recommended Submission Contents

Include the following items if required by the assignment rubric:

- Source code and training scripts
- API implementation
- Unit and API tests
- `requirements.txt`
- Dockerfile and `.dockerignore`
- Kubernetes manifests
- GitHub Actions workflow
- MLflow evidence and required artifacts
- EDA outputs and evaluation results
- Genuine screenshots documenting CI/CD, testing, and deployment
- This report exported to PDF
- Working GitHub repository link

Avoid submitting `.venv`, caches, secrets, or unnecessary temporary files. Confirm whether the assignment specifically requires the `mlruns/` directory or model artifacts before deciding whether to include them.

### 9.2 Final Screenshot Directory

A possible organization is:

```text
screenshots/
├── mlflow/
│   ├── 01-experiment-overview.png
│   ├── 02-best-run-metrics.png
│   └── 03-run-parameters-artifacts.png
├── cicd/
│   ├── 01-github-actions-workflow.png
│   ├── 02-ci-job-steps.png
│   └── 03-workflow-file.png
├── deployment/
│   ├── 01-docker-build.png
│   ├── 02-docker-container.png
│   ├── 03-kubernetes-pods.png
│   ├── 04-kubernetes-service.png
│   ├── 05-health-check.png
│   └── 06-api-prediction.png
├── monitoring/
│   ├── 01-api-healthchecks.png
│   ├── 02-api-metrics.png
│   ├── 03-api-logs.png
│   ├── 04-prometheus-metrics.png
│   └── 05-kubernetes-pod.png
└── testing/
    └── 01-pytest-results.png
```

Create only the folders and screenshots that are relevant to your actual implementation. Update the image paths in this report to match the files you save.

---

## 10. Conclusion

This project demonstrates an end-to-end MLOps workflow covering exploratory data analysis, model comparison, experiment tracking, an inference API, automated testing, containerization, and Kubernetes deployment.

MLflow provides experiment history and model evaluation evidence, while automated checks and health validation help assess software quality and deployment readiness.

Before submission, verify the final metrics, ensure that all referenced screenshots exist, confirm that the repository link works, and export the completed report to PDF. The model is intended for academic experimentation and is not a clinical diagnostic device.
