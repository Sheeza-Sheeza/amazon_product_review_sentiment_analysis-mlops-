"""Configuration management for the sentiment analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_path: Path = Path("7817_1.csv")
    text_column: str | None = None
    rating_column: str | None = None
    drop_duplicates: bool = True
    min_text_length: int = 10


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class VectorizerConfig:
    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass
class TrainingConfig:
    cv_folds: int = 5
    use_class_weights: bool = True
    models: list[str] = field(
        default_factory=lambda: [
            "tfidf_logistic_regression",
            "tfidf_linear_svc",
            "tfidf_multinomial_nb",
            "embedding_logistic_regression",
        ]
    )


@dataclass
class MLflowConfig:
    enabled: bool = True
    experiment_name: str = "amazon_review_sentiment"
    tracking_uri: str | None = None
    run_name: str | None = None
    register_best_model: bool = False
    registered_model_name: str = "amazon_review_sentiment_classifier"


@dataclass
class OutputConfig:
    output_dir: Path = Path("artifacts")
    save_best_model: bool = True
    save_comparison_report: bool = True


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PipelineConfig:
        data = raw.get("data", {})
        split = raw.get("split", {})
        vectorizer = raw.get("vectorizer", {})
        embedding = raw.get("embedding", {})
        training = raw.get("training", {})
        mlflow_cfg = raw.get("mlflow", {})
        output = raw.get("output", {})

        if "data_path" in data:
            data["data_path"] = Path(data["data_path"])
        if "ngram_range" in vectorizer:
            vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
        if "output_dir" in output:
            output["output_dir"] = Path(output["output_dir"])

        return cls(
            data=DataConfig(**data),
            split=SplitConfig(**split),
            vectorizer=VectorizerConfig(**vectorizer),
            embedding=EmbeddingConfig(**embedding),
            training=TrainingConfig(**training),
            mlflow=MLflowConfig(**mlflow_cfg),
            output=OutputConfig(**output),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"]["data_path"] = str(self.data.data_path)
        payload["output"]["output_dir"] = str(self.output.output_dir)
        payload["vectorizer"]["ngram_range"] = list(self.vectorizer.ngram_range)
        return payload
