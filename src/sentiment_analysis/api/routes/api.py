"""JSON API routes."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from sentiment_analysis import __version__
from sentiment_analysis.api.dependencies import (
    get_analytics_service,
    get_metrics_service,
    get_model_service,
)
from sentiment_analysis.api.schemas import (
    AnalyticsResponse,
    BatchPredictResponse,
    HealthResponse,
    ModelMetricsResponse,
    PredictRequest,
    PredictResponse,
)
from sentiment_analysis.services.analytics_service import AnalyticsService
from sentiment_analysis.services.metrics_service import MetricsService
from sentiment_analysis.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["api"])

BATCH_RESULTS: dict[str, Path] = {}
UPLOAD_DIR = Path("artifacts") / "batch_results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        model_source=model_service.model_source if model_service.is_loaded else None,
        version=__version__,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    model_service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    try:
        result = model_service.predict_one(payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**result)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    text_column = _detect_text_column(df.columns.tolist())
    if text_column is None:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a text column (e.g. reviews.text, text, reviewText).",
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="CSV contains no review text.")

    try:
        results_df = model_service.predict_batch(texts)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    output_df = df.copy()
    for column in results_df.columns:
        output_df[column if column != "text" else "predicted_sentiment"] = results_df[column].values

    batch_id = str(uuid.uuid4())
    output_path = UPLOAD_DIR / f"batch_{batch_id}.csv"
    output_df.to_csv(output_path, index=False)
    BATCH_RESULTS[batch_id] = output_path

    records = []
    for _, row in results_df.iterrows():
        record = row.to_dict()
        prob_cols = [c for c in record if c not in ("text", "sentiment")]
        record["confidence"] = max((record[c] for c in prob_cols), default=1.0)
        records.append(record)

    return BatchPredictResponse(
        total=len(records),
        results=records[:100],
        download_url=f"/api/predict/batch/{batch_id}/download",
    )


@router.get("/predict/batch/{batch_id}/download")
async def download_batch(batch_id: str) -> FileResponse:
    path = BATCH_RESULTS.get(batch_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Batch result not found.")
    return FileResponse(path, filename=path.name, media_type="text/csv")


@router.get("/metrics", response_model=ModelMetricsResponse)
async def metrics(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> ModelMetricsResponse:
    data = metrics_service.get_metrics()
    return ModelMetricsResponse(**data)


@router.get("/metrics/confusion-matrix/{model_name}")
async def confusion_matrix(
    model_name: str,
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> FileResponse:
    path = metrics_service.get_confusion_matrix_path(model_name)
    if path is None:
        raise HTTPException(status_code=404, detail="Confusion matrix not found.")
    return FileResponse(path, media_type="image/png")


@router.get("/analytics", response_model=AnalyticsResponse)
async def analytics(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsResponse:
    try:
        data = analytics_service.get_analytics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return AnalyticsResponse(**data)


def _detect_text_column(columns: list[str]) -> str | None:
    candidates = (
        "reviews.text",
        "reviewText",
        "review_text",
        "text",
        "review",
        "content",
    )
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
