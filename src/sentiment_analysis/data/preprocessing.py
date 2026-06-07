"""Text cleaning and label preparation."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from sentiment_analysis.constants import RATING_TO_SENTIMENT, SENTIMENT_LABELS

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s'.!,?-]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_review_text(text: str) -> str:
    """Normalize review text for NLP modeling."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHANUM_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def map_ratings_to_sentiment(rating: float | int) -> str | None:
    """Map star ratings to positive/neutral/negative sentiment labels."""
    if pd.isna(rating):
        return None
    return RATING_TO_SENTIMENT.get(int(rating))


def prepare_training_data(
    df: pd.DataFrame,
    text_column: str,
    rating_column: str,
    *,
    drop_duplicates: bool = True,
    min_text_length: int = 10,
) -> pd.DataFrame:
    """Build a clean modeling dataframe with text and sentiment labels."""
    working = df[[text_column, rating_column]].copy()
    working = working.rename(columns={text_column: "text_raw", rating_column: "rating"})

    working["text"] = working["text_raw"].fillna("").astype(str).map(clean_review_text)
    working["sentiment"] = working["rating"].map(map_ratings_to_sentiment)

    working = working.dropna(subset=["sentiment"])
    working = working[working["text"].str.len() >= min_text_length]

    if drop_duplicates:
        working = working.drop_duplicates(subset=["text", "sentiment"])

    working["sentiment"] = pd.Categorical(
        working["sentiment"],
        categories=list(SENTIMENT_LABELS),
        ordered=False,
    )
    return working.reset_index(drop=True)
