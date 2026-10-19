"""FastAPI inference service for the saved scikit-learn pipeline."""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart-disease-api")
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


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
