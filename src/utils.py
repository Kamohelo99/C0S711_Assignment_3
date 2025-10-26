"""
utils.py
========

This module contains utility functions for metric computation and other
helper routines used across training scripts.  The functions provided
here are simple and intended as placeholders; you may need to extend
them or replace them entirely depending on your evaluation strategy.
"""

from typing import List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute macro precision, recall and F1 for a multi‑label task.

    This function binarises the predictions at 0.5 by default before
    computing metrics.  If you wish to adjust thresholds per class you
    should implement that externally and pass the binary predictions
    here.

    Args:
        y_true: Array of shape `(N, C)` with ground truth binary labels.
        y_pred: Array of shape `(N, C)` with predicted probabilities or
            binary predictions.

    Returns:
        A tuple `(precision, recall, f1)` representing the macro‑averaged
        precision, recall and F1 score across classes.
    """
    # Binarise predictions at 0.5
    y_bin = (y_pred >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average='macro', zero_division=0
    )
    return precision, recall, f1


def compute_map(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute mean average precision (mAP) across classes.

    Args:
        y_true: Binary matrix `(N, C)` of ground truth labels.
        y_scores: Matrix `(N, C)` of predicted scores (probabilities).

    Returns:
        The mean of the average precision score for each class.
    """
    n_classes = y_true.shape[1]
    ap_values = []
    for c in range(n_classes):
        try:
            ap = average_precision_score(y_true[:, c], y_scores[:, c])
        except ValueError:
            ap = 0.0
        ap_values.append(ap)
    return float(np.mean(ap_values))
