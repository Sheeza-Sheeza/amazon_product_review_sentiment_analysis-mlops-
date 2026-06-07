"""Model evaluation utilities."""

from sentiment_analysis.evaluation.metrics import (
    compute_metrics,
    evaluate_model,
    plot_confusion_matrix,
    save_classification_report,
)

__all__ = [
    "compute_metrics",
    "evaluate_model",
    "plot_confusion_matrix",
    "save_classification_report",
]
