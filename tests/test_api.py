"""API endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment_analysis.api.main import app

client = TestClient(app)


def test_root_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "version" in data


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "ReviewSense" in response.text
    assert "Review Sentiment Dashboard" in response.text
    assert "Positive Reviews" in response.text
    assert "Negative Reviews" in response.text


def test_predict_page():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "Single Review Prediction" in response.text


def test_batch_page():
    response = client.get("/batch")
    assert response.status_code == 200
    assert "Batch CSV" in response.text


def test_metrics_page():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "Model Metrics" in response.text


def test_analytics_page():
    response = client.get("/analytics")
    assert response.status_code == 200
    assert "Sentiment Analytics" in response.text


def test_metrics_api():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data


def test_analytics_api():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_reviews"] > 0
    assert "sentiment_distribution" in data


def test_predict_without_model():
    response = client.post("/api/predict", json={"text": "Great product!"})
    if response.status_code == 503:
        assert "not loaded" in response.json()["detail"].lower()
    else:
        assert response.status_code == 200
        assert response.json()["sentiment"] in ("positive", "neutral", "negative")


def test_batch_invalid_file():
    response = client.post(
        "/api/predict/batch",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 400
