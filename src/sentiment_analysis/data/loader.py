"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_analysis.constants import RATING_COLUMN_CANDIDATES, TEXT_COLUMN_CANDIDATES


def _detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV with fallback encodings for common encoding issues."""
    encodings = ("utf-8", "utf-8-sig", "latin-1", "cp1252")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Failed to load {path} with encodings {encodings}") from last_error


def load_review_dataset(
    path: Path,
    text_column: str | None = None,
    rating_column: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load review dataset and resolve text/rating column names."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = load_dataset(path)
    resolved_text = text_column or _detect_column(df.columns.tolist(), TEXT_COLUMN_CANDIDATES)
    resolved_rating = rating_column or _detect_column(df.columns.tolist(), RATING_COLUMN_CANDIDATES)

    if resolved_text is None:
        raise ValueError("Could not detect text column. Pass --text-column explicitly.")
    if resolved_rating is None:
        raise ValueError("Could not detect rating column. Pass --rating-column explicitly.")

    return df, resolved_text, resolved_rating
