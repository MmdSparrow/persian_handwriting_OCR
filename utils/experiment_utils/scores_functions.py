import math

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "accuracy_calculator",
    "auc_score_calculator",
    "balanced_accuracy_calculator",
    "f1_score_calculator",
    "macro_f1_score_calculator",
    "macro_precision_calculator",
    "macro_recall_calculator",
    "precision_calculator",
    "sensitivity_calculator",
    "specificity_calculator",
    "weighted_f1_calculator",
]


def __not_sample_class_handler(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else math.nan


def accuracy_calculator(tp, tn, fp, fn, **_) -> float:
    return __not_sample_class_handler(tp + tn, tp + tn + fp + fn)


def sensitivity_calculator(tp, fn, **_) -> float:
    return __not_sample_class_handler(tp, tp + fn)


def specificity_calculator(tn, fp, **_) -> float:
    return __not_sample_class_handler(tn, tn + fp)


def precision_calculator(tp, fp, **_) -> float:
    return __not_sample_class_handler(tp, tp + fp)


def f1_score_calculator(tp, fp, fn, **_) -> float:
    return __not_sample_class_handler(2 * tp, 2 * tp + fp + fn)


def auc_score_calculator(labels, probabilities, **_) -> float:
    labels = labels.numpy()
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return math.nan
    return roc_auc_score(labels, probabilities)


def balanced_accuracy_calculator(labels, predictions):
    balanced_accuracy_score(labels, predictions)


def macro_f1_score_calculator(labels, predictions):
    return f1_score(labels, predictions, average="macro", zero_division=0)


def weighted_f1_score_calculator(labels, predictions):
    return f1_score(labels, predictions, average="weighted", zero_division=0)


def macro_recall_calculator(labels, predictions):
    return recall_score(labels, predictions, average="macro", zero_division=0)


def macro_precision_calculator(labels, predictions):
    return precision_score(labels, predictions, average="macro", zero_division=0)
