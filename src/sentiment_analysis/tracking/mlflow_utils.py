"""MLflow experiment tracking helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow.models import infer_signature

from sentiment_analysis.config import MLflowConfig, PipelineConfig


class MLflowTracker:
    """Thin wrapper around MLflow for reproducible experiment logging."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._active_run = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def setup(self) -> None:
        if not self.enabled:
            return
        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        *,
        nested: bool = False,
    ) -> Iterator[str | None]:
        if not self.enabled:
            yield None
            return

        with mlflow.start_run(
            run_name=run_name or self.config.run_name,
            nested=nested,
        ) as run:
            self._active_run = run
            yield run.info.run_id
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        flattened = _flatten_params(params)
        mlflow.log_params(flattened)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self.enabled or not path.exists():
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_model(
        self,
        model,
        artifact_path: str,
        X_sample,
        y_sample,
    ) -> None:
        if not self.enabled:
            return
        signature = infer_signature(X_sample, y_sample)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X_sample[:3],
        )

    def register_best_model(self, model_uri: str) -> None:
        if not self.enabled or not self.config.register_best_model:
            return
        mlflow.register_model(model_uri, self.config.registered_model_name)


def log_pipeline_config(tracker: MLflowTracker, config: PipelineConfig) -> None:
    tracker.log_params(config.to_dict())


def save_model_comparison(results: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for result in results:
        serializable.append(
            {
                "model_name": result["model_name"],
                "metrics": {
                    key: value
                    for key, value in result["metrics"].items()
                    if key not in {"classification_report", "confusion_matrix"}
                },
            }
        )
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return output_path


def _flatten_params(data: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, full_key))
        elif isinstance(value, (list, tuple)):
            flattened[full_key] = json.dumps(value)
        else:
            flattened[full_key] = value
    return flattened
