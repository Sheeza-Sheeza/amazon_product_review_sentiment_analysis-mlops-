"""Preprocessing unit tests."""

from sentiment_analysis.data.preprocessing import clean_review_text, map_ratings_to_sentiment


def test_clean_review_text_removes_urls():
    text = "Great product! Visit https://example.com for more"
    cleaned = clean_review_text(text)
    assert "https" not in cleaned
    assert "great product" in cleaned


def test_map_ratings_to_sentiment():
    assert map_ratings_to_sentiment(1) == "negative"
    assert map_ratings_to_sentiment(2) == "negative"
    assert map_ratings_to_sentiment(3) == "neutral"
    assert map_ratings_to_sentiment(4) == "positive"
    assert map_ratings_to_sentiment(5) == "positive"
    assert map_ratings_to_sentiment(None) is None
