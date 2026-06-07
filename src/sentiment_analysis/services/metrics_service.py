"""Load model training metrics from artifacts and MLflow."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MetricsService:
    """Expose training metrics for the metrics dashboard."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or Path(
            os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
        )
        self.mlflow_experiment = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "amazon_review_sentiment"
        )

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def get_metrics(self) -> dict[str, Any]:
        summary = self._read_json(self.artifacts_dir / "training_summary.json") or {}
        comparison = self._read_json(self.artifacts_dir / "model_comparison.json") or []
        metadata = self._read_json(self.artifacts_dir / "best_model" / "metadata.json")

        models: list[dict[str, Any]] = []
        if summary.get("model_metrics"):
            models = summary["model_metrics"]
        elif comparison:
            models = [
                {
                    "model_name": item["model_name"],
                    **{
                        key: value
                        for key, value in item.get("metrics", {}).items()
                        if key not in {"classification_report", "confusion_matrix"}
                    },
                }
                for item in comparison
            ]

        per_model_details = []
        for model in models:
            name = model["model_name"]
            model_dir = self.artifacts_dir / name
            details = {"model_name": name, **model}
            report_path = model_dir / "classification_report.txt"
            cm_path = model_dir / "confusion_matrix.png"
            if report_path.exists():
                details["classification_report"] = report_path.read_text(encoding="utf-8")
            if cm_path.exists():
                details["confusion_matrix_url"] = f"/api/metrics/confusion-matrix/{name}"
            per_model_details.append(details)

        return {
            "best_model": summary.get("best_model"),
            "best_f1_macro": summary.get("best_f1_macro"),
            "models": per_model_details,
            "metadata": metadata,
            "mlflow_experiment": self.mlflow_experiment,
        }

    def get_confusion_matrix_path(self, model_name: str) -> Path | None:
        path = self.artifacts_dir / model_name / "confusion_matrix.png"
        return path if path.exists() else None
