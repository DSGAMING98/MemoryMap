from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "retention_model.pkl"


class PredictionRequest(BaseModel):
    delta_days: float = Field(ge=0, le=3650)
    history_seen: int = Field(ge=0, le=100000)
    history_accuracy: float = Field(ge=0, le=1)
    half_life_days: float = Field(default=1.0, gt=0, le=3650)
    threshold: float = Field(default=0.70, gt=0, lt=1)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained retention model is unavailable.")
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="MemoryMap Prediction API",
    version="1.1.1",
    docs_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/predict")
def model_status():
    model = load_model()
    return {
        "status": "ready",
        "version": "1.1.1",
        "model": model.__class__.__name__,
        "features": [
            "delta_days",
            "history_seen",
            "history_accuracy",
        ],
    }


@app.post("/api/predict")
def predict_retention(request: PredictionRequest):
    try:
        model = load_model()
        features = np.asarray(
            [[
                request.delta_days,
                request.history_seen,
                request.history_accuracy,
            ]],
            dtype=float,
        )
        model_prediction = float(model.predict(features)[0])
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    model_prediction = max(0.0, min(model_prediction, 1.0))
    fixed_prediction = 2 ** (
        -request.delta_days / request.half_life_days
    )
    revision_interval = -request.half_life_days * np.log2(
        request.threshold
    )
    revision_interval = max(1 / 24, min(float(revision_interval), 30.0))

    return {
        "model_prediction": model_prediction,
        "fixed_prediction": fixed_prediction,
        "revision_interval_days": revision_interval,
        "threshold": request.threshold,
        "model": model.__class__.__name__,
        "model_role": "public Duolingo bootstrap model",
    }
