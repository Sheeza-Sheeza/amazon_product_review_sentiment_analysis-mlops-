"""FastAPI application entry point."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sentiment_analysis import __version__
from sentiment_analysis.api.routes import api, pages

logging.basicConfig(level=logging.INFO)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="Amazon Review Sentiment Analysis",
    description="Production-ready NLP sentiment API for e-commerce reviews",
    version=__version__,
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
app.include_router(pages.router)
app.include_router(api.router)


@app.get("/health")
async def root_health() -> dict[str, str]:
    return {"status": "ok"}
