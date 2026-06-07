"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class BatchPredictResponse(BaseModel):
    total: int
    results: list[dict[str, Any]]
    download_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str | None = None
    version: str


class ModelMetricsResponse(BaseModel):
    best_model: str | None = None
    best_f1_macro: float | None = None
    models: list[dict[str, Any]]
    metadata: dict[str, Any] | None = None
    mlflow_experiment: str | None = None


class AnalyticsResponse(BaseModel):
    total_reviews: int
    labeled_reviews: int
    sentiment_distribution: dict[str, int]
    rating_distribution: dict[str, int]
    avg_text_length: float
    median_text_length: float
    top_brands: list[dict[str, Any]]
    class_imbalance_ratio: float
