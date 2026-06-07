"""Model loading from MLflow registry or local artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sentiment_analysis.data.preprocessing import clean_review_text

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ModelService:
    """Load and serve the production sentiment classifier."""

    def __init__(
        self,
        model_path: Path | None = None,
        mlflow_model_uri: str | None = None,
    ) -> None:
        self.model_path = model_path or Path(
            os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "best_model" / "model.joblib")
        )
        self.mlflow_model_uri = mlflow_model_uri or os.getenv("MLFLOW_MODEL_URI")
        self.model_source = "unloaded"
        self._model: Any = None
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> None:
        if self.mlflow_model_uri:
            try:
                import mlflow

                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                self._model = mlflow.sklearn.load_model(self.mlflow_model_uri)
                self.model_source = f"mlflow:{self.mlflow_model_uri}"
                logger.info("Loaded model from MLflow: %s", self.mlflow_model_uri)
                return
            except Exception as exc:
                logger.warning("MLflow model load failed: %s", exc)

        if self.model_path.exists():
            self._model = joblib.load(self.model_path)
            self.model_source = f"local:{self.model_path}"
            logger.info("Loaded model from disk: %s", self.model_path)
            return

        logger.warning("No model artifact found. Predictions will be unavailable.")

    def predict_one(self, text: str) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError(
                "Model is not loaded. Train a model or set MLFLOW_MODEL_URI / MODEL_PATH."
            )

        cleaned = clean_review_text(text)
        sentiment = self._model.predict([cleaned])[0]
        result: dict[str, Any] = {
            "text": text,
            "sentiment": str(sentiment),
            "confidence": 1.0,
            "probabilities": {},
        }

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba([cleaned])[0]
            classes = [str(label) for label in self._model.classes_]
            probabilities = {label: float(score) for label, score in zip(classes, proba)}
            result["probabilities"] = probabilities
            result["confidence"] = float(max(proba))

        return result

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        cleaned = [clean_review_text(text) for text in texts]
        predictions = self._model.predict(cleaned)

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(cleaned)
            classes = [str(label) for label in self._model.classes_]
            frame = pd.DataFrame(proba, columns=classes)
            frame.insert(0, "sentiment", predictions)
            frame.insert(0, "text", texts)
            return frame

        return pd.DataFrame({"text": texts, "sentiment": predictions})
