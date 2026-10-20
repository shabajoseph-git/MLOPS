"""FastAPI inference service for the saved scikit-learn pipeline."""
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, ConfigDict, Field

from fastapi import Request
from prometheus_client import Counter, Histogram, make_asgi_app

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


# Configure API request logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# Prometheus request counter
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

# Prometheus response-time histogram
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
)

# Expose monitoring metrics at /metrics
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    method = request.method
    path = request.url.path
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception:
        logger.exception(
            "Request failed: method=%s path=%s",
            method,
            path,
        )
        raise
    finally:
        duration = time.perf_counter() - start_time

        if path != "/metrics" and not path.startswith("/metrics/"):
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=str(status),
            ).inc()

            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path,
            ).observe(duration)

            logger.info(
                "method=%s path=%s status=%s duration_seconds=%.4f",
                method,
                path,
                status,
                duration,
            )


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(..., ge=1, le=120)
    sex: float
    cp: float
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: float
    restecg: float
    thalach: float = Field(..., ge=0)
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(MODEL_PATH.exists()).lower()}


@app.post("/predict")
def predict(patient: PatientFeatures) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Train it first.")
    model = load(MODEL_PATH)
    row = pd.DataFrame([patient.model_dump()])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    confidence = float(max(probabilities))
    logger.info("prediction_request received; prediction=%s confidence=%.4f", prediction, confidence)
    return {
        "prediction": prediction,
        "label": "disease_present_in_dataset_label" if prediction == 1 else "disease_absent_in_dataset_label",
        "confidence": round(confidence, 4),
        "disclaimer": "Educational demo only; not for medical diagnosis or treatment."
    }
