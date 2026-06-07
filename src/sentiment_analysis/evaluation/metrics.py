"""Classification metrics and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from sentiment_analysis.constants import SENTIMENT_LABELS


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute standard multi-class classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def cross_validate_model(
    model: Pipeline,
    X,
    y,
    cv_folds: int,
) -> dict[str, float]:
    """Run stratified cross-validation and return summary metrics."""
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    cv_results: dict[str, float] = {}
    for metric_name, scorer in scoring.items():
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scorer, n_jobs=-1)
        cv_results[f"cv_{metric_name}_mean"] = float(np.mean(scores))
        cv_results[f"cv_{metric_name}_std"] = float(np.std(scores))
    return cv_results


def evaluate_model(
    model: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds: int,
) -> dict[str, Any]:
    """Fit model, evaluate on holdout set, and compute CV metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics.update(cross_validate_model(model, X_train, y_train, cv_folds))
    metrics["confusion_matrix"] = confusion_matrix(
        y_test, y_pred, labels=list(SENTIMENT_LABELS)
    ).tolist()
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        labels=list(SENTIMENT_LABELS),
        zero_division=0,
    )
    return metrics


def plot_confusion_matrix(
    confusion: list[list[int]],
    output_path: Path,
    title: str,
) -> Path:
    matrix = np.array(confusion)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(SENTIMENT_LABELS)))
    ax.set_yticks(range(len(SENTIMENT_LABELS)))
    ax.set_xticklabels(SENTIMENT_LABELS, rotation=30)
    ax.set_yticklabels(SENTIMENT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_classification_report(report: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path
