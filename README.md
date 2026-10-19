# MLOps Assignment 01 — Heart Disease Prediction (Starter Project)

This repository is a learning starter for the MLOps assignment. You must run it, inspect the outputs, personalize the analysis, take your own screenshots, and be able to explain every part before submitting.

## What it contains
- `src/data.py`: downloads the UCI Heart Disease dataset using `ucimlrepo`
- `src/eda.py`: saves summary tables and EDA plots
- `src/train.py`: compares Logistic Regression and Random Forest; logs runs to MLflow; saves the best pipeline
- `src/api.py`: FastAPI `/predict` endpoint with prediction and probability
- `tests/`: basic data/model/API tests
- `.github/workflows/ci.yml`: lint, tests, and training workflow
- `Dockerfile`: container image for the API
- `k8s/`: example Kubernetes Deployment and Service
- `report/REPORT_TEMPLATE.md`: report structure to complete in your own words

## 1. Requirements
Install Python 3.11 or 3.12, Git, and VS Code. Docker Desktop is needed for the container step. Kubernetes can be done later with Minikube or Docker Desktop Kubernetes.

## 2. Create and activate a virtual environment (Windows PowerShell)
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```

## 3. Download data and make EDA outputs
```powershell
python -m src.data
python -m src.eda
```
The downloaded raw CSV is saved to `data/heart_disease.csv`; plots and summaries go to `artifacts/eda/`.

Dataset access needs an internet connection. The downloader uses the UCI Machine Learning Repository's `ucimlrepo` package. Inspect the downloaded columns and target before continuing.

## 4. Train models and track experiments
```powershell
python -m src.train
```
The script prints metrics, writes `artifacts/metrics.json`, saves `artifacts/model.joblib`, and records experiments in the local `mlruns/` folder. To inspect MLflow runs:
```powershell
mlflow ui --backend-store-uri ./mlruns
```
Then open the local address printed by MLflow in your browser (usually `http://127.0.0.1:5000`).

Model scores will vary with dataset version, split, and library versions. Do not copy expected scores from someone else; report your own run.

## 5. Run API locally
After training has created `artifacts/model.joblib`:
```powershell
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs`, choose `POST /predict`, and try the sample JSON below. The API is a demonstration only, not a medical device.

```json
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
  "fbs": 0, "restecg": 1, "thalch": 150, "exang": 0,
  "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
}
```
Feature codes must match the dataset's documented coding. The example is only a test input, not clinical advice.

## 6. Run tests
```powershell
pytest -q
```

## 7. Build and test Docker
Make sure Docker Desktop is running, and train the model first so `artifacts/model.joblib` exists:
```powershell
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```
In another terminal, test `http://127.0.0.1:8000/docs` or send a JSON request to `/predict`.

## 8. GitHub Actions
Create a GitHub repository and push your files. The workflow runs linting, tests, and training. Note: GitHub-hosted runners need internet access to download the dataset. Check the Actions tab and fix any failures.

## 9. Kubernetes (after Docker works)
Build the image in an environment accessible to your cluster. For a local Minikube cluster, build/load the image into Minikube and update `image:` in `k8s/deployment.yaml` if needed. Apply:
```powershell
kubectl apply -f k8s/
kubectl get pods
kubectl get services
```
The included Service is NodePort for local demonstration; change it if your instructor requires a LoadBalancer or Ingress. Capture real screenshots of successful deployment and endpoint testing.

## 10. Finish the assignment
- Personalize the EDA and discuss what you actually observe.
- Compare both models using your own cross-validation and test metrics.
- Add experiment screenshots, CI run screenshots, Docker proof, deployment proof, and API logs.
- Complete the 10-page report template and record a short video showing the pipeline.
- Add your GitHub repository URL and a public API URL if you actually deploy publicly.
- Do not claim a deployment or metric that you did not verify.

## Important limitations
This is a starter, not a guaranteed final submission. UCI's dataset can include missing values and coded categorical fields. Review `data/heart_disease.csv`, the generated plots, the metrics, and the model behavior. The prediction is educational and must not be used for diagnosis or treatment.
