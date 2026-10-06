import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from typing import TypedDict

class ClassificationMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    TP: int
    FP: int
    FN: int
    TN: int

def evaluate_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> ClassificationMetrics:
    """Evaluate a binary classifier at a given probability threshold."""

    predictions: np.ndarray = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate a classifier across multiple probability thresholds."""

    results: list[dict[str, float | int]] = []

    for threshold in thresholds:
        metrics = evaluate_at_threshold(
            y_true=y_true,
            probabilities=probabilities,
            threshold=threshold,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TN": metrics["TN"],
            }
        )

    return pd.DataFrame(results)