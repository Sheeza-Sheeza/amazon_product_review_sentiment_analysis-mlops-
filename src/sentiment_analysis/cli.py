"""Package CLI entry point for training."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentiment_analysis.config import PipelineConfig
from sentiment_analysis.pipeline.trainer import SentimentTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare NLP sentiment classifiers."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override dataset path from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to train.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional MLflow run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)

    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.models is not None:
        config.training.models = args.models
    if args.no_mlflow:
        config.mlflow.enabled = False
    if args.mlflow_run_name is not None:
        config.mlflow.run_name = args.mlflow_run_name

    trainer = SentimentTrainer(config)
    summary = trainer.run()

    print("\n=== Training Complete ===")
    print(f"Best model : {summary['best_model']}")
    print(f"Best F1    : {summary['best_f1_macro']:.4f}")
    print(f"Artifacts  : {summary['output_dir']}")


if __name__ == "__main__":
    main()
