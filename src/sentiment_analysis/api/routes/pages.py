"""HTML page routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"page": "home"})


@router.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "predict.html", {"page": "predict"})


@router.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "batch.html", {"page": "batch"})


@router.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "metrics.html", {"page": "metrics"})


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "analytics.html", {"page": "analytics"})
