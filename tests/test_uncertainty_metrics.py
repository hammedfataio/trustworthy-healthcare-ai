"""Tests for uncertainty-based error-detection evaluation.

These tests validate the evaluation utilities used in EXP-003.

The EXP-003 convention is:

    correct prediction   -> error target = 0
    incorrect prediction -> error target = 1

Higher uncertainty is interpreted as stronger evidence that a prediction
may be incorrect.
"""

import numpy as np
import pytest

from src.evaluation.uncertainty_metrics import (
    ErrorDetectionMetrics,
    error_prevalence,
    evaluate_error_detection,
    prediction_correctness,
    prediction_errors,
    summarize_uncertainty_by_correctness,
)


def test_prediction_correctness():
    """Correctness indicators should match true/predicted labels."""

    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 1, 1]

    result = prediction_correctness(y_true, y_pred)

    expected = np.array([1, 0, 1, 0])

    assert np.array_equal(result, expected)


def test_prediction_errors():
    """Incorrect predictions should be encoded as error target 1."""

    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 1, 1]

    result = prediction_errors(y_true, y_pred)

    expected = np.array([0, 1, 0, 1])

    assert np.array_equal(result, expected)


def test_correctness_and_errors_are_complements():
    """Correctness and error indicators should sum to one."""

    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 1]

    correctness = prediction_correctness(y_true, y_pred)
    errors = prediction_errors(y_true, y_pred)

    assert np.array_equal(
        correctness + errors,
        np.ones(len(y_true), dtype=np.int64),
    )


def test_error_prevalence():
    """Error prevalence should equal the proportion of error targets."""

    errors = [0, 1, 0, 1]

    prevalence = error_prevalence(errors)

    assert prevalence == pytest.approx(0.5)


def test_error_prevalence_with_single_error():
    """Prevalence calculation should work for imbalanced errors."""

    errors = [0, 0, 0, 1]

    prevalence = error_prevalence(errors)

    assert prevalence == pytest.approx(0.25)


def test_perfect_error_detection():
    """Perfect uncertainty ranking should produce AUROC and AUPRC of 1."""

    errors = np.array([0, 0, 1, 1])

    uncertainty = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    metrics = evaluate_error_detection(
        errors,
        uncertainty,
    )

    assert isinstance(metrics, ErrorDetectionMetrics)

    assert metrics.n_samples == 4
    assert metrics.n_errors == 2
    assert metrics.error_prevalence == pytest.approx(0.5)
    assert metrics.error_detection_auroc == pytest.approx(1.0)
    assert metrics.error_detection_auprc == pytest.approx(1.0)


def test_reversed_error_ranking_has_zero_auroc():
    """Completely reversed uncertainty ranking should produce AUROC 0."""

    errors = np.array([0, 0, 1, 1])

    uncertainty = np.array([
        0.90,
        0.80,
        0.20,
        0.10,
    ])

    metrics = evaluate_error_detection(
        errors,
        uncertainty,
    )

    assert metrics.error_detection_auroc == pytest.approx(0.0)


def test_constant_uncertainty_has_chance_level_auroc():
    """Constant scores should provide no ranking discrimination."""

    errors = np.array([0, 1, 0, 1])

    uncertainty = np.array([
        0.5,
        0.5,
        0.5,
        0.5,
    ])

    metrics = evaluate_error_detection(
        errors,
        uncertainty,
    )

    assert metrics.error_detection_auroc == pytest.approx(0.5)

    # With identical scores, average precision corresponds
    # to the prevalence of the positive error class.
    assert metrics.error_detection_auprc == pytest.approx(0.5)


def test_error_detection_counts_samples_and_errors():
    """Returned metadata should reflect the supplied targets."""

    errors = np.array([0, 0, 1, 0, 1])

    uncertainty = np.array([
        0.10,
        0.20,
        0.70,
        0.30,
        0.80,
    ])

    metrics = evaluate_error_detection(
        errors,
        uncertainty,
    )

    assert metrics.n_samples == 5
    assert metrics.n_errors == 2
    assert metrics.error_prevalence == pytest.approx(0.4)


def test_mismatched_prediction_lengths_raise_error():
    """True and predicted labels must contain the same number of samples."""

    y_true = [0, 1, 0]
    y_pred = [0, 1]

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        prediction_correctness(y_true, y_pred)


@pytest.mark.parametrize(
    "y_true,y_pred",
    [
        ([0, 2], [0, 1]),
        ([0, 1], [0, 2]),
        ([-1, 1], [0, 1]),
    ],
)
def test_nonbinary_prediction_labels_raise_error(
    y_true,
    y_pred,
):
    """Classification labels must be binary."""

    with pytest.raises(
        ValueError,
        match="must contain only binary values",
    ):
        prediction_correctness(
            y_true,
            y_pred,
        )


def test_empty_prediction_input_raises_error():
    """Empty classification arrays must be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        prediction_correctness([], [])


def test_multidimensional_prediction_input_raises_error():
    """Prediction labels must be one-dimensional."""

    y_true = np.array([[0, 1]])
    y_pred = np.array([[0, 1]])

    with pytest.raises(
        ValueError,
        match="must be one-dimensional",
    ):
        prediction_correctness(
            y_true,
            y_pred,
        )


def test_invalid_error_targets_raise_error():
    """Error targets must contain only 0 and 1."""

    errors = [0, 2, 1]

    uncertainty = [0.1, 0.5, 0.9]

    with pytest.raises(
        ValueError,
        match="must contain only binary values",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


def test_mismatched_error_detection_lengths_raise_error():
    """Error targets and uncertainty scores must align sample-by-sample."""

    errors = [0, 1, 0]

    uncertainty = [0.1, 0.9]

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


@pytest.mark.parametrize(
    "uncertainty",
    [
        [0.1, np.nan, 0.8],
        [0.1, np.inf, 0.8],
        [0.1, -np.inf, 0.8],
    ],
)
def test_nonfinite_uncertainty_scores_raise_error(
    uncertainty,
):
    """NaN and infinite uncertainty scores must be rejected."""

    errors = [0, 1, 0]

    with pytest.raises(
        ValueError,
        match="must contain only finite values",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


def test_nonnumeric_uncertainty_scores_raise_error():
    """Non-numeric uncertainty scores must be rejected."""

    errors = [0, 1]

    uncertainty = ["low", "high"]

    with pytest.raises(
        ValueError,
        match="must contain numeric values",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


@pytest.mark.parametrize(
    "errors",
    [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ],
)
def test_single_error_class_raises_error(errors):
    """AUROC/AUPRC require both correct and incorrect predictions."""

    uncertainty = [0.1, 0.2, 0.3, 0.4]

    with pytest.raises(
        ValueError,
        match="must contain both correct and incorrect",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


def test_empty_error_detection_input_raises_error():
    """Empty error-detection inputs must be rejected."""

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        evaluate_error_detection([], [])


def test_multidimensional_uncertainty_scores_raise_error():
    """Uncertainty scores must be one-dimensional."""

    errors = np.array([0, 1])

    uncertainty = np.array([
        [0.1, 0.9],
    ])

    with pytest.raises(
        ValueError,
        match="must be one-dimensional",
    ):
        evaluate_error_detection(
            errors,
            uncertainty,
        )


def test_uncertainty_summary_separates_correct_and_incorrect():
    """Summary statistics should separate correct and incorrect cases."""

    correctness = np.array([
        1,
        1,
        0,
        0,
    ])

    uncertainty = np.array([
        0.10,
        0.20,
        0.70,
        0.90,
    ])

    summary = summarize_uncertainty_by_correctness(
        correctness,
        uncertainty,
    )

    assert summary["correct"]["count"] == 2
    assert summary["incorrect"]["count"] == 2

    assert summary["correct"]["mean"] == pytest.approx(0.15)
    assert summary["incorrect"]["mean"] == pytest.approx(0.80)

    assert summary["incorrect"]["mean"] > summary["correct"]["mean"]


def test_uncertainty_summary_contains_expected_statistics():
    """Summary output should expose the planned descriptive statistics."""

    correctness = [1, 1, 0, 0]

    uncertainty = [
        0.10,
        0.20,
        0.70,
        0.90,
    ]

    summary = summarize_uncertainty_by_correctness(
        correctness,
        uncertainty,
    )

    expected_keys = {
        "count",
        "mean",
        "median",
        "std",
        "min",
        "q25",
        "q75",
        "max",
    }

    assert set(summary["correct"].keys()) == expected_keys
    assert set(summary["incorrect"].keys()) == expected_keys


def test_uncertainty_summary_handles_missing_group():
    """Summary should explicitly represent a group with no samples."""

    correctness = [1, 1, 1]

    uncertainty = [0.1, 0.2, 0.3]

    summary = summarize_uncertainty_by_correctness(
        correctness,
        uncertainty,
    )

    assert summary["correct"]["count"] == 3

    assert summary["incorrect"]["count"] == 0
    assert np.isnan(summary["incorrect"]["mean"])
    assert np.isnan(summary["incorrect"]["median"])


def test_summary_mismatched_lengths_raise_error():
    """Correctness and uncertainty arrays must align sample-by-sample."""

    correctness = [1, 0, 1]

    uncertainty = [0.1, 0.9]

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        summarize_uncertainty_by_correctness(
            correctness,
            uncertainty,
        )


def test_summary_rejects_nonbinary_correctness():
    """Correctness indicators must contain only 0 and 1."""

    correctness = [1, 2, 0]

    uncertainty = [0.1, 0.5, 0.9]

    with pytest.raises(
        ValueError,
        match="must contain only binary values",
    ):
        summarize_uncertainty_by_correctness(
            correctness,
            uncertainty,
        )
