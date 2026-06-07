"""Model pipeline factory for sentiment classification."""

from __future__ import annotations

from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment_analysis.config import EmbeddingConfig, VectorizerConfig
from sentiment_analysis.features.vectorizers import EmbeddingTransformer, build_tfidf_vectorizer


def _class_weight(use_class_weights: bool) -> str | None:
    return "balanced" if use_class_weights else None


def build_tfidf_logistic_regression(
    vectorizer_config: VectorizerConfig,
    use_class_weights: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", build_tfidf_vectorizer(vectorizer_config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight=_class_weight(use_class_weights),
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_linear_svc(
    vectorizer_config: VectorizerConfig,
    use_class_weights: bool,
) -> Pipeline:
    base_svc = LinearSVC(
        class_weight=_class_weight(use_class_weights),
        random_state=42,
        max_iter=5_000,
    )
    return Pipeline(
        steps=[
            ("vectorizer", build_tfidf_vectorizer(vectorizer_config)),
            (
                "classifier",
                CalibratedClassifierCV(base_svc, cv=3, method="sigmoid"),
            ),
        ]
    )


def build_tfidf_multinomial_nb(vectorizer_config: VectorizerConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", build_tfidf_vectorizer(vectorizer_config)),
            ("classifier", MultinomialNB(alpha=0.1)),
        ]
    )


def build_embedding_logistic_regression(
    embedding_config: EmbeddingConfig,
    use_class_weights: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("embedder", EmbeddingTransformer.from_config(embedding_config)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight=_class_weight(use_class_weights),
                    random_state=42,
                ),
            ),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[..., Pipeline]] = {
    "tfidf_logistic_regression": build_tfidf_logistic_regression,
    "tfidf_linear_svc": build_tfidf_linear_svc,
    "tfidf_multinomial_nb": build_tfidf_multinomial_nb,
    "embedding_logistic_regression": build_embedding_logistic_regression,
}


def get_available_models() -> list[str]:
    return list(MODEL_BUILDERS.keys())


def build_model_pipeline(
    model_name: str,
    vectorizer_config: VectorizerConfig,
    embedding_config: EmbeddingConfig,
    use_class_weights: bool,
) -> Pipeline:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(get_available_models())}"
        )

    builder = MODEL_BUILDERS[model_name]
    if model_name.startswith("embedding_"):
        return builder(embedding_config=embedding_config, use_class_weights=use_class_weights)
    if model_name == "tfidf_multinomial_nb":
        return builder(vectorizer_config=vectorizer_config)
    return builder(vectorizer_config=vectorizer_config, use_class_weights=use_class_weights)
