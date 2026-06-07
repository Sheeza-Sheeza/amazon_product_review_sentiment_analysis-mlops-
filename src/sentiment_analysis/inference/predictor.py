"""Model loading and prediction for deployment."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sentiment_analysis.data.preprocessing import clean_review_text


class SentimentPredictor:
    """Load a trained sklearn pipeline and score review text."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, texts: list[str]) -> list[str]:
        cleaned = [clean_review_text(text) for text in texts]
        return self.model.predict(cleaned).tolist()

    def predict_proba(self, texts: list[str]) -> pd.DataFrame:
        cleaned = [clean_review_text(text) for text in texts]
        probabilities = self.model.predict_proba(cleaned)
        classes = self.model.classes_
        return pd.DataFrame(probabilities, columns=classes)

    def predict_batch(self, texts: list[str]) -> pd.DataFrame:
        predictions = self.predict(texts)
        if hasattr(self.model, "predict_proba"):
            proba_df = self.predict_proba(texts)
            proba_df.insert(0, "sentiment", predictions)
            return proba_df
        return pd.DataFrame({"sentiment": predictions})
