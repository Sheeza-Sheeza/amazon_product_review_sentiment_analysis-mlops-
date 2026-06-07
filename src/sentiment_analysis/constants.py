"""Shared constants for the sentiment analysis pipeline."""

SENTIMENT_LABELS = ("negative", "neutral", "positive")

RATING_TO_SENTIMENT: dict[int, str] = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

TEXT_COLUMN_CANDIDATES = (
    "reviews.text",
    "reviewText",
    "review_text",
    "text",
    "review",
    "content",
    "body",
)

RATING_COLUMN_CANDIDATES = (
    "reviews.rating",
    "rating",
    "label",
    "sentiment",
    "class",
    "target",
)
