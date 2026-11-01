"""
Integration Test Suite for Automated Content Moderation API (Milestone: Week 10).
Tests system health probes, schema validation, explicit toxicity,
compositional litotes (veiled insults), benign negations, commercial spam,
and vectorized batch processing.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api import app

@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture that triggers the FastAPI lifespan context manager,
    loading production model artifacts into memory once for all tests.
    """
    with TestClient(app) as test_client:
        yield test_client

def test_root_endpoint(client):
    """Verify root endpoint returns HTTP 200 with service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation_url" in data

def test_health_check(client):
    """Verify health endpoint confirms model and vectorizer readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert set(data["target_classes"]) == {"Normal", "Spam", "Toxic"}
    assert data["features_dimension"] > 0
    assert data["uptime_seconds"] >= 0

def test_predict_normal_discourse(client):
    """Verify polite encyclopedic commentary is classified as Normal."""
    payload = {"text": "Thank you for updating the citations and improving the article references."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Normal"
    assert data["is_flagged"] is False
    assert data["confidence"] > 0.70
    assert "Normal" in data["probabilities"]
    assert data["latency_ms"] < 50.0

def test_predict_explicit_toxic(client):
    """Verify profane personal attacks are classified as Toxic and flagged."""
    payload = {"text": "You are a complete idiot and a moron, get lost!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Toxic"
    assert data["is_flagged"] is True
    assert data["confidence"] > 0.80

def test_predict_veiled_litotes_cognitive(client):
    """
    Verify supervisor's specific case: 'not smart enough' is flagged as Toxic
    via our compositional litotes detection and N-gram features.
    """
    payload = {"text": "Honestly, you are not smart enough to understand this topic."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Toxic"
    assert data["is_flagged"] is True
    assert data["confidence"] > 0.70

def test_predict_veiled_idiom_insult(client):
    """Verify idiomatic veiled insult ('room temperature IQ') is flagged as Toxic."""
    payload = {"text": "You clearly have a room temperature IQ, stop posting here."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Toxic"
    assert data["is_flagged"] is True

def test_predict_benign_negation(client):
    """
    Verify harmless impersonal negation with 'not ... enough' is NOT falsely flagged
    (preventing false positives).
    """
    payload = {"text": "This article is not long enough and needs more background citations."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Normal"
    assert data["is_flagged"] is False

def test_predict_commercial_spam(client):
    """Verify promotional phishing / spam broadcasts are flagged as Spam."""
    payload = {"text": "URGENT! You have won a £1000 cash prize! Call 0800-FREE-NOW or claim at http://win-free.biz"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Spam"
    assert data["is_flagged"] is True
    assert data["confidence"] > 0.85

def test_empty_input_validation(client):
    """Verify empty or blank whitespace requests are rejected with HTTP 422."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422

def test_batch_prediction(client):
    """Verify batch endpoint classifies multiple heterogeneous texts in vectorized mode."""
    batch_payload = {
        "texts": [
            "This is a wonderful contribution to our open source project.",
            "You are horribly incompetent and stupid.",
            "FREE gift card giveaway! Click http://claim-promo.site now!"
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["label"] == "Normal"
    assert data["predictions"][1]["label"] == "Toxic"
    assert data["predictions"][2]["label"] == "Spam"
    assert data["total_latency_ms"] < 100.0

def test_latency_performance(client):
    """Verify single message processing latency remains under 10ms."""
    payload = {"text": "A standard message benchmark for assessing real-time moderation throughput."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["latency_ms"] < 10.0
