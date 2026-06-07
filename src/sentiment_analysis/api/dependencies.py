"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache

from sentiment_analysis.services.analytics_service import AnalyticsService
from sentiment_analysis.services.metrics_service import MetricsService
from sentiment_analysis.services.model_service import ModelService


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()


@lru_cache
def get_metrics_service() -> MetricsService:
    return MetricsService()


@lru_cache
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService()
