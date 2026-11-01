"""
Production REST API for Automated Content Moderation (Milestone: Week 10).
Built using FastAPI and Pydantic with asynchronous lifespan management.
Serves real-time single and batch prediction endpoints for content classification
(Normal, Toxic, Spam) with calibrated confidence scores and sub-millisecond latencies.
"""
import os
import sys
import time
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TextPreprocessor

# Global state container for production artifacts
class ModelArtifacts:
    model: Any = None
    vectorizer: Any = None
    preprocessor: Optional[TextPreprocessor] = None
    metadata: Dict[str, Any] = {}
    classes: List[str] = ["Normal", "Spam", "Toxic"]
    start_time: float = 0.0

artifacts = ModelArtifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifespan context manager: Loads model, vectorizer, and preprocessor
    into memory on startup, ensuring zero cold-start delay on user requests.
    """
    artifacts.start_time = time.time()
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    # Preferred model: production ensemble (Week 9), fallback: baseline (Week 7)
    prod_model_path = os.path.join(models_dir, "production_model.joblib")
    base_model_path = os.path.join(models_dir, "baseline_logreg.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "production_metadata.json")

    chosen_model_path = prod_model_path if os.path.exists(prod_model_path) else base_model_path
    
    if not os.path.exists(chosen_model_path):
        raise RuntimeError(f"Model artifact not found at {chosen_model_path}. Train model first.")
    if not os.path.exists(vec_path):
        raise RuntimeError(f"Vectorizer artifact not found at {vec_path}.")

    print(f"[API Lifecycle] Loading model from: {chosen_model_path}")
    artifacts.model = joblib.load(chosen_model_path)

    print(f"[API Lifecycle] Loading vectorizer from: {vec_path}")
    artifacts.vectorizer = joblib.load(vec_path)

    # Load metadata if present
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            artifacts.metadata = json.load(f)
            artifacts.classes = artifacts.metadata.get("target_classes", ["Normal", "Spam", "Toxic"])
    else:
        artifacts.classes = list(artifacts.model.classes_)

    # Initialize production text preprocessor with litotes & negation preservation
    artifacts.preprocessor = TextPreprocessor(
        lower=True,
        expand_contractions=True,
        remove_html=True,
        replace_urls=True,
        normalize_elongations=True,
        demask_obfuscation=True,
        resolve_veiled_toxicity=True,
        remove_punctuation=True,
        remove_stopwords=True,
        preserve_negations=True,
        lemmatize=True
    )
    # Pre-warm preprocessor and vectorizer to prevent first-request cold-start latency
    _ = artifacts.preprocessor.transform("warmup health check")
    _vec_warmup = artifacts.vectorizer.transform(["warmup"])
    _ = artifacts.model.predict(_vec_warmup)

    print(f"[API Lifecycle] Serving {len(artifacts.classes)} classes: {artifacts.classes}")
    print("[API Lifecycle] Initialization complete. Service ready to process requests.")
    yield
    print("[API Lifecycle] Shutting down service.")

# Initialize FastAPI application with formal OpenAPI metadata
app = FastAPI(
    title="Automated Content Moderation API",
    description=(
        "High-throughput NLP service for real-time classification of online text messages "
        "into Normal, Toxic, and Spam categories with compositional litotes detection."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration (Streamlit demo, browser dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Data Contracts & Schemas ---

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The raw textual message to be classified.",
        examples=["Honestly, you are not smart enough to understand this."]
    )

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text must not be empty or whitespace only.")
        return v

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="A batch list of text messages (1 to 100 items)."
    )

class PredictionResponse(BaseModel):
    text: str
    label: str = Field(..., description="Predicted class: 'Normal', 'Toxic', or 'Spam'.")
    confidence: float = Field(..., description="Model certainty probability [0.0 to 1.0].")
    probabilities: Dict[str, float] = Field(..., description="Posterior probability per target class.")
    is_flagged: bool = Field(..., description="True if content requires moderation (Toxic or Spam).")
    latency_ms: float = Field(..., description="Processing time in milliseconds.")

class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    target_classes: List[str]
    features_dimension: int
    uptime_seconds: float

# --- Endpoint Handlers ---

@app.get("/", tags=["System"])
async def root() -> Dict[str, Any]:
    """Root endpoint returning service identity and documentation pointers."""
    return {
        "service": "Automated Content Moderation API",
        "status": "online",
        "version": "1.0.0",
        "documentation_url": "/docs",
        "health_check_url": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Readiness and health probe for container orchestrators and monitoring agents."""
    is_ready = artifacts.model is not None and artifacts.vectorizer is not None
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded into memory."
        )
    
    uptime = time.time() - artifacts.start_time
    model_name = artifacts.metadata.get("model_name", type(artifacts.model).__name__)
    n_features = artifacts.vectorizer.max_features or len(artifacts.vectorizer.get_feature_names_out())

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_name,
        target_classes=artifacts.classes,
        features_dimension=n_features,
        uptime_seconds=round(uptime, 2)
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Moderation"])
async def predict_single(payload: PredictionRequest) -> PredictionResponse:
    """
    Classify a single incoming message.
    Returns the predicted label, confidence, class probabilities, and moderation flag.
    """
    if artifacts.model is None or artifacts.vectorizer is None or artifacts.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )

    t0 = time.perf_counter()
    raw_text = payload.text

    # Preprocessing with litotes preservation
    cleaned = artifacts.preprocessor.transform(raw_text)
    if not cleaned.strip():
        cleaned = "<EMPTY>"

    # Vectorisation & Inference
    vec_input = artifacts.vectorizer.transform([cleaned])
    pred_label = str(artifacts.model.predict(vec_input)[0])

    # Probability calibration check
    if hasattr(artifacts.model, "predict_proba"):
        probs_array = artifacts.model.predict_proba(vec_input)[0]
        model_classes = list(artifacts.model.classes_)
        prob_dict = {cls: round(float(probs_array[model_classes.index(cls)]), 4) for cls in artifacts.classes if cls in model_classes}
        confidence = float(max(probs_array))
    else:
        # Fallback for models without predict_proba
        prob_dict = {cls: (1.0 if cls == pred_label else 0.0) for cls in artifacts.classes}
        confidence = 1.0

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        text=raw_text,
        label=pred_label,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        is_flagged=(pred_label in ["Toxic", "Spam"]),
        latency_ms=round(latency_ms, 3)
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Moderation"])
async def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Vectorized batch inference for up to 100 messages simultaneously.
    Optimized for high-throughput chat stream ingestion.
    """
    if artifacts.model is None or artifacts.vectorizer is None or artifacts.preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable."
        )

    t_batch_start = time.perf_counter()
    texts = payload.texts
    results = []

    # Batch preprocessing
    cleaned_texts = []
    for t in texts:
        c = artifacts.preprocessor.transform(t)
        cleaned_texts.append(c if c.strip() else "<EMPTY>")

    # Vectorized transform and prediction
    vec_inputs = artifacts.vectorizer.transform(cleaned_texts)
    preds = artifacts.model.predict(vec_inputs)

    has_proba = hasattr(artifacts.model, "predict_proba")
    probs_matrix = artifacts.model.predict_proba(vec_inputs) if has_proba else None
    model_classes = list(artifacts.model.classes_)

    total_duration_ms = (time.perf_counter() - t_batch_start) * 1000.0
    avg_per_doc_ms = total_duration_ms / max(len(texts), 1)

    for i, raw_text in enumerate(texts):
        pred_label = str(preds[i])
        if has_proba and probs_matrix is not None:
            p_arr = probs_matrix[i]
            p_dict = {cls: round(float(p_arr[model_classes.index(cls)]), 4) for cls in artifacts.classes if cls in model_classes}
            conf = float(max(p_arr))
        else:
            p_dict = {cls: (1.0 if cls == pred_label else 0.0) for cls in artifacts.classes}
            conf = 1.0

        results.append(PredictionResponse(
            text=raw_text,
            label=pred_label,
            confidence=round(conf, 4),
            probabilities=p_dict,
            is_flagged=(pred_label in ["Toxic", "Spam"]),
            latency_ms=round(avg_per_doc_ms, 3)
        ))

    return BatchPredictionResponse(
        count=len(results),
        predictions=results,
        total_latency_ms=round(total_duration_ms, 3)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
