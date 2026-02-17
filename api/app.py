
"""
app.py — FastAPI endpoint for jailbreak detection.

Endpoints:
  POST /detect   — Classify a prompt as jailbreak or benign
  GET  /health   — Service health check
  GET  /stats    — Prediction statistics

All predictions are logged to logs/predictions.jsonl for analytics.
"""

import os
from typing import Optional, List
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import (
    extract_handcrafted_features,
    extract_single_embedding,
    get_handcrafted_feature_names,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"

# Risk thresholds
RISK_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.0,
}

# Features that indicate jailbreak behavior (for flagged_features response)
FLAGGED_FEATURE_DESCRIPTIONS = {
    "jailbreak_keyword_freq": "Contains jailbreak-associated keywords",
    "instruction_pattern_count": "Contains instruction injection patterns",
    "imperative_verb_score": "High use of imperative/commanding verbs",
    "uppercase_ratio": "Abnormal capitalization pattern",
    "special_char_ratio": "Unusual special character usage",
    "quote_nesting_depth": "Deep quote nesting detected",
    "newline_count": "Complex multi-line structure",
    "exclamation_count": "Excessive exclamation marks",
}

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Jailbreak Detection API",
    description="Detect adversarial prompts attempting to manipulate LLMs into bypassing safety guidelines.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None
model_metadata = None


@app.on_event("startup")
def load_model():
    global model, model_metadata
    model_path = MODEL_DIR / "jailbreak_detector.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"

    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: Model not found at {model_path}. Run model_training.py first.")

    if meta_path.exists():
        with open(meta_path) as f:
            model_metadata = json.load(f)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="The user prompt to classify")


class DetectResponse(BaseModel):
    is_jailbreak: bool
    confidence: float
    risk_level: str
    flagged_features: List[str]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    test_metrics: Optional[dict] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_risk_level(confidence: float) -> str:
    if confidence >= RISK_THRESHOLDS["high"]:
        return "high"
    elif confidence >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def get_flagged_features(features: dict) -> list[str]:
    """Identify which features contributed to a positive detection."""
    flagged = []

    if features.get("jailbreak_keyword_freq", 0) >= 2:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["jailbreak_keyword_freq"])
    if features.get("instruction_pattern_count", 0) >= 1:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["instruction_pattern_count"])
    if features.get("imperative_verb_score", 0) >= 2:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["imperative_verb_score"])
    if features.get("uppercase_ratio", 0) >= 0.3:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["uppercase_ratio"])
    if features.get("special_char_ratio", 0) >= 0.15:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["special_char_ratio"])
    if features.get("quote_nesting_depth", 0) >= 3:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["quote_nesting_depth"])
    if features.get("newline_count", 0) >= 5:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["newline_count"])
    if features.get("exclamation_count", 0) >= 3:
        flagged.append(FLAGGED_FEATURE_DESCRIPTIONS["exclamation_count"])

    return flagged


def log_prediction(text: str, result: dict):
    """Append prediction to JSONL log file."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_preview": text[:200],
        "text_length": len(text),
        "is_jailbreak": result["is_jailbreak"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "flagged_features": result["flagged_features"],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/detect", response_model=DetectResponse)
def detect_jailbreak(request: DetectRequest):
    """Classify a prompt as jailbreak or benign."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run model_training.py first.",
        )

    start_time = time.time()

    # Extract features
    hc_features = extract_handcrafted_features(request.text)
    embedding = extract_single_embedding(request.text)

    hc_names = get_handcrafted_feature_names()
    hc_values = [hc_features[name] for name in hc_names]
    feature_vector = hc_values + embedding.tolist()

    X = pd.DataFrame(
        [feature_vector],
        columns=hc_names + [f"emb_{i}" for i in range(384)],
    )

    # Predict
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    is_jailbreak = bool(prediction == 1)
    confidence = float(round(proba, 4))

    risk_level = get_risk_level(confidence)
    flagged = get_flagged_features(hc_features) if is_jailbreak else []

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    result = {
        "is_jailbreak": is_jailbreak,
        "confidence": confidence,
        "risk_level": risk_level,
        "flagged_features": flagged,
        "processing_time_ms": elapsed_ms,
    }

    # Log prediction
    log_prediction(request.text, result)

    return DetectResponse(**result)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health check."""
    metrics = None
    model_name = None
    if model_metadata:
        model_name = model_metadata.get("model_name")
        metrics = {
            "precision": model_metadata.get("test_precision"),
            "recall": model_metadata.get("test_recall"),
            "f1_score": model_metadata.get("test_f1"),
        }

    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_name=model_name,
        test_metrics=metrics,
    )


@app.get("/stats")
def prediction_stats():
    """Return prediction statistics from the log file."""
    if not LOG_FILE.exists():
        return {"total_predictions": 0, "jailbreak_count": 0, "benign_count": 0}

    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    total = len(entries)
    jb_count = sum(1 for e in entries if e["is_jailbreak"])
    bn_count = total - jb_count

    avg_confidence = sum(e["confidence"] for e in entries) / total if total > 0 else 0

    risk_dist = {"low": 0, "medium": 0, "high": 0}
    for e in entries:
        risk_dist[e["risk_level"]] += 1

    return {
        "total_predictions": total,
        "jailbreak_count": jb_count,
        "benign_count": bn_count,
        "average_confidence": round(avg_confidence, 4),
        "risk_distribution": risk_dist,
        "recent_predictions": entries[-10:][::-1],
    }
