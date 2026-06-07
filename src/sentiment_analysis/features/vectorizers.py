"""Text vectorization for TF-IDF and embedding-based models."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_analysis.config import EmbeddingConfig, VectorizerConfig


def build_tfidf_vectorizer(config: VectorizerConfig) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with production-friendly defaults."""
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        strip_accents="unicode",
        lowercase=False,
    )


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer wrapping sentence-transformer embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def fit(self, X, y=None):  # noqa: N803
        self._load_model()
        return self

    def transform(self, X):  # noqa: N803
        self._load_model()
        embeddings = self._model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings)

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> EmbeddingTransformer:
        return cls(
            model_name=config.model_name,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
        )
