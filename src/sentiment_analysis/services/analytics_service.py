"""Dataset analytics for the analytics dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sentiment_analysis.constants import RATING_TO_SENTIMENT
from sentiment_analysis.data.loader import load_review_dataset
from sentiment_analysis.data.preprocessing import clean_review_text, map_ratings_to_sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AnalyticsService:
    """Compute dataset-level sentiment analytics."""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or Path(
            os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "raw" / "7817_1.csv")
        )
        if not self.data_path.exists():
            fallback = PROJECT_ROOT / "7817_1.csv"
            if fallback.exists():
                self.data_path = fallback

    def get_analytics(self) -> dict:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        df, text_col, rating_col = load_review_dataset(self.data_path)
        texts = df[text_col].fillna("").astype(str)
        ratings = df[rating_col]

        sentiments = ratings.map(map_ratings_to_sentiment)
        labeled = sentiments.dropna()
        sentiment_counts = labeled.value_counts().to_dict()
        sentiment_counts = {str(key): int(value) for key, value in sentiment_counts.items()}

        rating_counts = ratings.dropna().astype(int).value_counts().sort_index()
        rating_distribution = {str(key): int(value) for key, value in rating_counts.items()}

        lengths = texts.map(lambda value: len(clean_review_text(value)))
        avg_length = float(lengths.mean())
        median_length = float(lengths.median())

        top_brands: list[dict] = []
        if "brand" in df.columns:
            brand_counts = df["brand"].value_counts().head(10)
            top_brands = [
                {"brand": str(brand), "count": int(count)}
                for brand, count in brand_counts.items()
            ]

        max_count = max(sentiment_counts.values()) if sentiment_counts else 1
        min_count = min(sentiment_counts.values()) if sentiment_counts else 1
        imbalance_ratio = round(max_count / min_count, 2) if min_count else 0.0

        return {
            "total_reviews": int(len(df)),
            "labeled_reviews": int(len(labeled)),
            "sentiment_distribution": sentiment_counts,
            "rating_distribution": rating_distribution,
            "avg_text_length": round(avg_length, 2),
            "median_text_length": round(median_length, 2),
            "top_brands": top_brands,
            "class_imbalance_ratio": imbalance_ratio,
        }
