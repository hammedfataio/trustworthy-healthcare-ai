"""Evaluation utilities for uncertainty-based error detection.

This module supports EXP-003 of the Trustworthy Healthcare AI research
programme.

The central question is whether higher predictive uncertainty is associated
with a greater likelihood that the baseline classifier is incorrect.

For error-detection evaluation:

    correct prediction   -> error target = 0
    incorrect prediction -> error target = 1

The uncertainty score is then treated as a continuous score for detecting
prediction errors.

Primary evaluation metrics:
- Error prevalence
- Error-detection AUROC
- Error-detection AUPRC

Important
---------
These metrics evaluate whether uncertainty ranks incorrect predictions above
correct predictions. They do not establish clinical safety or model
trustworthiness by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ErrorDetectionMetrics:
    """Container for uncertainty-based error-detection metrics."""

    n_samples: int
    n_errors: int
    error_prevalence: float
    error_detection_auroc: float
    error_detection_auprc: float


def _as_1d_array(
    values: np.ndarray | list | tuple,
    *,
    name: str,
) -> np.ndarray:
    """Convert an input sequence to a one-dimensional NumPy array."""

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    return array


def _validate_same_length(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
) -> None:
    """Require two arrays to contain the same number of samples."""

    if len(first) != len(second):
        raise ValueError(
            f"{first_name} and {second_name} must have the same length."
        )


def _validate_binary_labels(
    labels: np.ndarray,
    *,
    name: str,
) -> None:
    """Validate that labels contain only binary values 0 and 1."""

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")


def _validate_uncertainty_scores(
    uncertainty_scores: np.ndarray,
) -> None:
    """Validate uncertainty scores used for error detection."""

    try:
        finite = np.isfinite(uncertainty_scores)
    except TypeError as exc:
        raise ValueError(
            "uncertainty_scores must contain numeric values."
        ) from exc

    if not np.all(finite):
        raise ValueError(
            "uncertainty_scores must contain only finite values."
        )


def prediction_correctness(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> np.ndarray:
    """Return prediction correctness as binary indicators.

    Returns
    -------
    np.ndarray
        1 indicates a correct prediction.
        0 indicates an incorrect prediction.
    """

    y_true_array = _as_1d_array(
        y_true,
        name="y_true",
    )

    y_pred_array = _as_1d_array(
        y_pred,
        name="y_pred",
    )

    _validate_same_length(
        y_true_array,
        y_pred_array,
        first_name="y_true",
        second_name="y_pred",
    )

    _validate_binary_labels(
        y_true_array,
        name="y_true",
    )

    _validate_binary_labels(
        y_pred_array,
        name="y_pred",
    )

    return (y_true_array == y_pred_array).astype(np.int64)


def prediction_errors(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> np.ndarray:
    """Return binary error indicators.

    Returns
    -------
    np.ndarray
        0 indicates a correct prediction.
        1 indicates an incorrect prediction.

    Notes
    -----
    EXP-003 treats prediction error as the positive class for uncertainty
    evaluation.
    """

    correctness = prediction_correctness(
        y_true=y_true,
        y_pred=y_pred,
    )

    return 1 - correctness


def error_prevalence(
    error_targets: np.ndarray | list[int],
) -> float:
    """Calculate the proportion of predictions that are incorrect."""

    errors = _as_1d_array(
        error_targets,
        name="error_targets",
    )

    _validate_binary_labels(
        errors,
        name="error_targets",
    )

    return float(np.mean(errors))


def evaluate_error_detection(
    error_targets: np.ndarray | list[int],
    uncertainty_scores: np.ndarray | list[float],
) -> ErrorDetectionMetrics:
    """Evaluate uncertainty as a score for detecting prediction errors.

    Parameters
    ----------
    error_targets:
        Binary indicators where:

            0 = correct prediction
            1 = incorrect prediction

    uncertainty_scores:
        Continuous uncertainty scores. Larger values are assumed to indicate
        greater uncertainty.

    Returns
    -------
    ErrorDetectionMetrics
        Number of samples, number of errors, error prevalence,
        error-detection AUROC, and error-detection AUPRC.

    Raises
    ------
    ValueError
        If inputs are empty, multidimensional, mismatched, non-binary,
        non-finite, or if the error targets contain only one class.

    Notes
    -----
    AUROC evaluates ranking discrimination between incorrect and correct
    predictions.

    AUPRC is especially important when prediction errors are relatively rare
    because it reflects performance relative to the prevalence of the
    positive error class.
    """

    errors = _as_1d_array(
        error_targets,
        name="error_targets",
    )

    uncertainty = _as_1d_array(
        uncertainty_scores,
        name="uncertainty_scores",
    )

    _validate_same_length(
        errors,
        uncertainty,
        first_name="error_targets",
        second_name="uncertainty_scores",
    )

    _validate_binary_labels(
        errors,
        name="error_targets",
    )

    _validate_uncertainty_scores(
        uncertainty,
    )

    unique_classes = np.unique(errors)

    if unique_classes.size < 2:
        raise ValueError(
            "error_targets must contain both correct and incorrect "
            "predictions to compute error-detection AUROC and AUPRC."
        )

    n_samples = int(errors.size)
    n_errors = int(errors.sum())
    prevalence = float(errors.mean())

    auroc = float(
        roc_auc_score(
            errors,
            uncertainty,
        )
    )

    auprc = float(
        average_precision_score(
            errors,
            uncertainty,
        )
    )

    return ErrorDetectionMetrics(
        n_samples=n_samples,
        n_errors=n_errors,
        error_prevalence=prevalence,
        error_detection_auroc=auroc,
        error_detection_auprc=auprc,
    )


def summarize_uncertainty_by_correctness(
    correctness: np.ndarray | list[int],
    uncertainty_scores: np.ndarray | list[float],
) -> dict[str, dict[str, float | int]]:
    """Summarize uncertainty separately for correct and incorrect predictions.

    Parameters
    ----------
    correctness:
        Binary indicators where:

            1 = correct prediction
            0 = incorrect prediction

    uncertainty_scores:
        Continuous uncertainty scores.

    Returns
    -------
    dict
        Summary statistics for correct and incorrect predictions.

    Notes
    -----
    This function provides descriptive evidence only. Statistical differences
    should not be inferred solely from these summary values.
    """

    correctness_array = _as_1d_array(
        correctness,
        name="correctness",
    )

    uncertainty = _as_1d_array(
        uncertainty_scores,
        name="uncertainty_scores",
    )

    _validate_same_length(
        correctness_array,
        uncertainty,
        first_name="correctness",
        second_name="uncertainty_scores",
    )

    _validate_binary_labels(
        correctness_array,
        name="correctness",
    )

    _validate_uncertainty_scores(
        uncertainty,
    )

    summaries: dict[str, dict[str, float | int]] = {}

    groups = {
        "correct": correctness_array == 1,
        "incorrect": correctness_array == 0,
    }

    for group_name, mask in groups.items():
        values = uncertainty[mask]

        if values.size == 0:
            summaries[group_name] = {
                "count": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
                "max": float("nan"),
            }

            continue

        summaries[group_name] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        }

    return summaries
