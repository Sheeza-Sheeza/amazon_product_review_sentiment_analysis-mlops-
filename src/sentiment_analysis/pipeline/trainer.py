"""End-to-end training orchestration for sentiment models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment_analysis.config import PipelineConfig
from sentiment_analysis.data.loader import load_review_dataset
from sentiment_analysis.data.preprocessing import prepare_training_data
from sentiment_analysis.evaluation.metrics import (
    evaluate_model,
    plot_confusion_matrix,
    save_classification_report,
)
from sentiment_analysis.models.factory import build_model_pipeline
from sentiment_analysis.tracking.mlflow_utils import (
    MLflowTracker,
    log_pipeline_config,
    save_model_comparison,
)


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict[str, Any]
    model: Pipeline
    artifact_dir: Path
    mlflow_run_id: str | None = None


class SentimentTrainer:
    """Train and compare sentiment classifiers with MLflow-ready outputs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.tracker = MLflowTracker(config.mlflow)
        self.tracker.setup()

    def run(self) -> dict[str, Any]:
        prepared = self._prepare_data()
        X = prepared["text"]
        y = prepared["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.split.test_size,
            random_state=self.config.split.random_state,
            stratify=y if self.config.split.stratify else None,
        )

        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[TrainingResult] = []
        with self.tracker.start_run(run_name=self.config.mlflow.run_name):
            log_pipeline_config(self.tracker, self.config)
            self.tracker.log_metrics(
                {
                    "dataset_rows": float(len(prepared)),
                    "train_rows": float(len(X_train)),
                    "test_rows": float(len(X_test)),
                }
            )

            for model_name in self.config.training.models:
                result = self._train_single_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    output_dir=output_dir / model_name,
                    nested_mlflow=True,
                )
                results.append(result)

        best_result = max(results, key=lambda item: item.metrics["f1_macro"])
        summary = self._finalize(results, best_result, output_dir, X_test, y_test)
        return summary

    def _prepare_data(self) -> pd.DataFrame:
        df, text_col, rating_col = load_review_dataset(
            path=self.config.data.data_path,
            text_column=self.config.data.text_column,
            rating_column=self.config.data.rating_column,
        )
        prepared = prepare_training_data(
            df=df,
            text_column=text_col,
            rating_column=rating_col,
            drop_duplicates=self.config.data.drop_duplicates,
            min_text_length=self.config.data.min_text_length,
        )
        if prepared.empty:
            raise ValueError("No valid training rows after preprocessing.")
        return prepared

    def _train_single_model(
        self,
        model_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir: Path,
        nested_mlflow: bool = False,
    ) -> TrainingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = build_model_pipeline(
            model_name=model_name,
            vectorizer_config=self.config.vectorizer,
            embedding_config=self.config.embedding,
            use_class_weights=self.config.training.use_class_weights,
        )

        mlflow_run_id: str | None = None
        with self.tracker.start_run(run_name=model_name, nested=nested_mlflow) as run_id:
            mlflow_run_id = run_id
            self.tracker.log_params({"model_name": model_name})
            metrics = evaluate_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv_folds=self.config.training.cv_folds,
            )
            self.tracker.log_metrics(
                {
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                }
            )

            report_path = save_classification_report(
                metrics["classification_report"],
                output_dir / "classification_report.txt",
            )
            cm_path = plot_confusion_matrix(
                metrics["confusion_matrix"],
                output_dir / "confusion_matrix.png",
                title=f"Confusion Matrix - {model_name}",
            )
            self.tracker.log_artifact(report_path)
            self.tracker.log_artifact(cm_path)
            self.tracker.log_model(
                model=model,
                artifact_path="model",
                X_sample=X_train,
                y_sample=y_train,
            )

        joblib.dump(model, output_dir / "model.joblib")
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"classification_report"}
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return TrainingResult(
            model_name=model_name,
            metrics=metrics,
            model=model,
            artifact_dir=output_dir,
            mlflow_run_id=mlflow_run_id,
        )

    def _finalize(
        self,
        results: list[TrainingResult],
        best_result: TrainingResult,
        output_dir: Path,
        X_test,
        y_test,
    ) -> dict[str, Any]:
        comparison = [
            {
                "model_name": result.model_name,
                "metrics": result.metrics,
            }
            for result in results
        ]

        if self.config.output.save_comparison_report:
            save_model_comparison(comparison, output_dir / "model_comparison.json")

        if self.config.output.save_best_model:
            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_result.model, best_dir / "model.joblib")
            metadata = {
                "model_name": best_result.model_name,
                "selected_by": "f1_macro",
                "metrics": {
                    key: value
                    for key, value in best_result.metrics.items()
                    if key not in {"classification_report", "confusion_matrix"}
                },
                "trained_at": datetime.now().isoformat(),
            }
            (best_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

        summary = {
            "best_model": best_result.model_name,
            "best_f1_macro": best_result.metrics["f1_macro"],
            "results": comparison,
            "output_dir": str(output_dir.resolve()),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "best_model": summary["best_model"],
                    "best_f1_macro": summary["best_f1_macro"],
                    "output_dir": summary["output_dir"],
                    "model_metrics": [
                        {
                            "model_name": item["model_name"],
                            "accuracy": item["metrics"]["accuracy"],
                            "f1_macro": item["metrics"]["f1_macro"],
                            "f1_weighted": item["metrics"]["f1_weighted"],
                            "cv_f1_macro_mean": item["metrics"]["cv_f1_macro_mean"],
                        }
                        for item in comparison
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if self.config.mlflow.register_best_model and best_result.mlflow_run_id:
            model_uri = f"runs:/{best_result.mlflow_run_id}/model"
            self.tracker.register_best_model(model_uri)

        return summary
