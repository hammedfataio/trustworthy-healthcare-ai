"""EXP-003: Predictive Uncertainty for Error Detection.

Research question
-----------------
Does predictive uncertainty provide useful information for distinguishing
incorrect from correct predictions produced by the frozen baseline
medical-image classifier?

Experimental sequence
---------------------
1. Load the PneumoniaMNIST held-out test split.
2. Reconstruct the frozen EXP-001 BaselineCNN architecture.
3. Load the frozen EXP-001 checkpoint.
4. Generate held-out test predictions.
5. Verify that the predictions reproduce the EXP-001 baseline.
6. Compute deterministic binary predictive entropy.
7. Treat prediction error as the positive error-detection class.
8. Evaluate uncertainty using AUROC and AUPRC.
9. Produce prediction-level and summary artifacts.
10. Save artifacts only after integrity checks succeed.

Research integrity
------------------
EXP-003 does not retrain the baseline classifier.

The held-out test set is not used to select the classification threshold,
uncertainty method, model parameters, or uncertainty parameters.

Predictive entropy is evaluated as a deterministic uncertainty baseline.
It must not be interpreted as a complete estimate of epistemic uncertainty.

EXP-003 artifacts are not saved unless the loaded checkpoint reproduces the
frozen EXP-001 baseline prediction behaviour.
"""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medmnist import INFO
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.evaluation.uncertainty_metrics import (
    evaluate_error_detection,
    prediction_correctness,
    prediction_errors,
    summarize_uncertainty_by_correctness,
)
from src.uncertainty.entropy import (
    binary_predictive_entropy,
    normalized_binary_predictive_entropy,
)


# =============================================================================
# Experiment configuration
# =============================================================================

EXPERIMENT_ID = "EXP-003"
EXPERIMENT_TITLE = "Predictive Uncertainty for Error Detection"

SEED = 42

DATA_FLAG = "pneumoniamnist"

BATCH_SIZE = 64

CLASSIFICATION_THRESHOLD = 0.5

HIGH_CONFIDENCE_THRESHOLD = 0.90


# =============================================================================
# Frozen EXP-001 reference evidence
# =============================================================================

EXPECTED_TEST_SAMPLES = 624

EXPECTED_ACCURACY = 0.884615

EXPECTED_TN = 168
EXPECTED_FP = 66
EXPECTED_FN = 6
EXPECTED_TP = 384

ACCURACY_TOLERANCE = 1e-6


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "experiment_001_baseline_cnn.pt"
)

RESULTS_TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

PREDICTION_RESULTS_PATH = (
    RESULTS_TABLE_DIR
    / "experiment_003_prediction_level_results.csv"
)

UNCERTAINTY_SUMMARY_PATH = (
    RESULTS_TABLE_DIR
    / "experiment_003_uncertainty_summary.csv"
)

ERROR_DETECTION_METRICS_PATH = (
    RESULTS_TABLE_DIR
    / "experiment_003_error_detection_metrics.csv"
)

BASELINE_INTEGRITY_PATH = (
    RESULTS_TABLE_DIR
    / "experiment_003_baseline_integrity.csv"
)


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = SEED) -> None:
    """Set random seeds used by the experiment."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Frozen EXP-001 model
# =============================================================================


class BaselineCNN(nn.Module):
    """CNN architecture used by the frozen EXP-001 baseline.

    EXP-003 must evaluate the existing EXP-001 model rather than train a
    replacement classifier.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 7 * 7,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw binary-classification logits."""

        features = self.features(x)

        logits = self.classifier(features)

        return logits


# =============================================================================
# Dataset
# =============================================================================


def create_test_loader() -> DataLoader:
    """Create the frozen PneumoniaMNIST held-out test loader."""

    info = INFO[DATA_FLAG]

    data_class = getattr(
        medmnist,
        info["python_class"],
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_dataset = data_class(
        split="test",
        transform=transform,
        download=True,
    )

    if len(test_dataset) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "PneumoniaMNIST test-set integrity check failed. "
            f"Expected {EXPECTED_TEST_SAMPLES} samples but "
            f"received {len(test_dataset)}."
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader


# =============================================================================
# Frozen checkpoint loading
# =============================================================================


def load_frozen_model(
    device: torch.device,
) -> BaselineCNN:
    """Load the frozen EXP-001 model checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "The frozen EXP-001 checkpoint was not found.\n\n"
            f"Expected checkpoint:\n{CHECKPOINT_PATH}\n\n"
            "EXP-003 must evaluate the frozen EXP-001 classifier. "
            "A replacement model must not be trained inside EXP-003."
        )

    model = BaselineCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model


# =============================================================================
# Inference
# =============================================================================


def collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labels and probabilities for the held-out test set."""

    all_labels: list[np.ndarray] = []

    all_probabilities: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            logits = model(
                images
            ).view(-1)

            probabilities = torch.sigmoid(
                logits
            )

            labels_array = (
                labels
                .view(-1)
                .cpu()
                .numpy()
                .astype(np.int64)
            )

            probabilities_array = (
                probabilities
                .cpu()
                .numpy()
            )

            all_labels.append(
                labels_array
            )

            all_probabilities.append(
                probabilities_array
            )

    y_true = np.concatenate(
        all_labels
    )

    probabilities = np.concatenate(
        all_probabilities
    )

    if len(y_true) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Inference produced an unexpected number of labels. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"received {len(y_true)}."
        )

    if len(probabilities) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Inference produced an unexpected number of probabilities. "
            f"Expected {EXPECTED_TEST_SAMPLES}, "
            f"received {len(probabilities)}."
        )

    return (
        y_true,
        probabilities,
    )


# =============================================================================
# Prediction construction
# =============================================================================


def probabilities_to_predictions(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Convert probabilities to frozen-threshold binary predictions."""

    return (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(np.int64)


# =============================================================================
# EXP-001 baseline integrity verification
# =============================================================================


def calculate_baseline_integrity(
    y_true: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, int | float | bool]:
    """Calculate baseline statistics required for integrity verification."""

    accuracy = float(
        accuracy_score(
            y_true,
            predicted_labels,
        )
    )

    matrix = confusion_matrix(
        y_true,
        predicted_labels,
        labels=[0, 1],
    )

    if matrix.shape != (2, 2):
        raise RuntimeError(
            "Unexpected confusion-matrix shape during baseline "
            f"verification: {matrix.shape}"
        )

    tn, fp, fn, tp = matrix.ravel()

    accuracy_match = bool(
        np.isclose(
            accuracy,
            EXPECTED_ACCURACY,
            atol=ACCURACY_TOLERANCE,
            rtol=0.0,
        )
    )

    confusion_matrix_match = bool(
        int(tn) == EXPECTED_TN
        and int(fp) == EXPECTED_FP
        and int(fn) == EXPECTED_FN
        and int(tp) == EXPECTED_TP
    )

    baseline_verified = bool(
        accuracy_match
        and confusion_matrix_match
    )

    return {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "expected_accuracy": EXPECTED_ACCURACY,
        "accuracy_match": accuracy_match,
        "tn": int(tn),
        "expected_tn": EXPECTED_TN,
        "fp": int(fp),
        "expected_fp": EXPECTED_FP,
        "fn": int(fn),
        "expected_fn": EXPECTED_FN,
        "tp": int(tp),
        "expected_tp": EXPECTED_TP,
        "confusion_matrix_match": confusion_matrix_match,
        "baseline_verified": baseline_verified,
    }


def enforce_baseline_integrity(
    baseline_integrity: dict[str, int | float | bool],
) -> None:
    """Stop EXP-003 if the frozen EXP-001 baseline is not reproduced."""

    if not bool(
        baseline_integrity["baseline_verified"]
    ):
        message = f"""
EXP-001 BASELINE INTEGRITY CHECK FAILED

EXP-003 has been stopped before uncertainty evidence was saved.

Observed:
    Accuracy: {baseline_integrity['accuracy']}
    TN:       {baseline_integrity['tn']}
    FP:       {baseline_integrity['fp']}
    FN:       {baseline_integrity['fn']}
    TP:       {baseline_integrity['tp']}

Expected:
    Accuracy: {EXPECTED_ACCURACY}
    TN:       {EXPECTED_TN}
    FP:       {EXPECTED_FP}
    FN:       {EXPECTED_FN}
    TP:       {EXPECTED_TP}

Possible causes include:
- incorrect checkpoint;
- preprocessing mismatch;
- dataset mismatch;
- architecture mismatch;
- label handling difference;
- threshold difference.

The discrepancy must be investigated before EXP-003 results are accepted.
"""

        raise RuntimeError(
            message.strip()
        )


# =============================================================================
# Prediction-level evidence
# =============================================================================


def build_prediction_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build the EXP-003 prediction-level evidence table."""

    predicted_labels = probabilities_to_predictions(
        probabilities
    )

    correctness = prediction_correctness(
        y_true=y_true,
        y_pred=predicted_labels,
    )

    errors = prediction_errors(
        y_true=y_true,
        y_pred=predicted_labels,
    )

    predictive_entropy = binary_predictive_entropy(
        probabilities
    )

    normalized_entropy = (
        normalized_binary_predictive_entropy(
            probabilities
        )
    )

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities,
    )

    table = pd.DataFrame(
        {
            "sample_id": np.arange(
                len(y_true),
                dtype=np.int64,
            ),
            "true_label": y_true.astype(
                np.int64
            ),
            "predicted_probability": probabilities,
            "predicted_label": predicted_labels,
            "correct": correctness,
            "error": errors,
            "confidence": confidence,
            "predictive_entropy": predictive_entropy,
            "normalized_predictive_entropy": normalized_entropy,
        }
    )

    return table


# =============================================================================
# Prediction-level integrity checks
# =============================================================================


def run_prediction_integrity_checks(
    prediction_table: pd.DataFrame,
) -> None:
    """Validate prediction-level evidence before artifact creation."""

    if prediction_table.empty:
        raise RuntimeError(
            "Prediction table is empty."
        )

    if len(prediction_table) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Prediction table contains an unexpected number "
            f"of samples. Expected {EXPECTED_TEST_SAMPLES}, "
            f"received {len(prediction_table)}."
        )

    required_columns = {
        "sample_id",
        "true_label",
        "predicted_probability",
        "predicted_label",
        "correct",
        "error",
        "confidence",
        "predictive_entropy",
        "normalized_predictive_entropy",
    }

    missing_columns = (
        required_columns
        - set(prediction_table.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Prediction table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    probabilities = prediction_table[
        "predicted_probability"
    ].to_numpy()

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            "Predicted probabilities contain non-finite values."
        )

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            "Predicted probabilities fall outside [0, 1]."
        )

    confidence = prediction_table[
        "confidence"
    ].to_numpy()

    if not np.all(
        np.isfinite(confidence)
    ):
        raise RuntimeError(
            "Confidence values contain non-finite values."
        )

    if np.any(
        (confidence < 0.5)
        | (confidence > 1.0)
    ):
        raise RuntimeError(
            "Confidence values fall outside the expected [0.5, 1.0] range."
        )

    entropy = prediction_table[
        "predictive_entropy"
    ].to_numpy()

    if not np.all(
        np.isfinite(entropy)
    ):
        raise RuntimeError(
            "Predictive entropy contains non-finite values."
        )

    if np.any(
        entropy < 0.0
    ):
        raise RuntimeError(
            "Predictive entropy contains negative values."
        )

    normalized_entropy = prediction_table[
        "normalized_predictive_entropy"
    ].to_numpy()

    if not np.all(
        np.isfinite(normalized_entropy)
    ):
        raise RuntimeError(
            "Normalized predictive entropy contains "
            "non-finite values."
        )

    tolerance = 1e-10

    if np.any(
        normalized_entropy
        < -tolerance
    ):
        raise RuntimeError(
            "Normalized predictive entropy contains "
            "unexpected negative values."
        )

    if np.any(
        normalized_entropy
        > 1.0 + tolerance
    ):
        raise RuntimeError(
            "Normalized predictive entropy exceeds "
            "its expected upper bound."
        )

    correctness = prediction_table[
        "correct"
    ].to_numpy()

    errors = prediction_table[
        "error"
    ].to_numpy()

    if not np.array_equal(
        correctness + errors,
        np.ones(
            len(prediction_table),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Correctness and error indicators are inconsistent."
        )

    expected_sample_ids = np.arange(
        EXPECTED_TEST_SAMPLES,
        dtype=np.int64,
    )

    observed_sample_ids = prediction_table[
        "sample_id"
    ].to_numpy()

    if not np.array_equal(
        observed_sample_ids,
        expected_sample_ids,
    ):
        raise RuntimeError(
            "Prediction sample ordering is inconsistent."
        )


# =============================================================================
# Uncertainty summaries
# =============================================================================


def build_uncertainty_summary(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive uncertainty statistics by correctness."""

    summary = summarize_uncertainty_by_correctness(
        correctness=prediction_table[
            "correct"
        ].to_numpy(),
        uncertainty_scores=prediction_table[
            "predictive_entropy"
        ].to_numpy(),
    )

    rows: list[dict[str, object]] = []

    for group_name, statistics in summary.items():

        row = {
            "prediction_group": group_name,
            **statistics,
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Error-detection metrics
# =============================================================================


def build_error_detection_metrics(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the primary EXP-003 error-detection metrics."""

    metrics = evaluate_error_detection(
        error_targets=prediction_table[
            "error"
        ].to_numpy(),
        uncertainty_scores=prediction_table[
            "predictive_entropy"
        ].to_numpy(),
    )

    metrics_dict = asdict(
        metrics
    )

    accuracy = float(
        accuracy_score(
            prediction_table[
                "true_label"
            ],
            prediction_table[
                "predicted_label"
            ],
        )
    )

    metrics_row = {
        "experiment_id": EXPERIMENT_ID,
        "uncertainty_method": "binary_predictive_entropy",
        "n_samples": metrics_dict[
            "n_samples"
        ],
        "n_errors": metrics_dict[
            "n_errors"
        ],
        "n_correct": (
            metrics_dict["n_samples"]
            - metrics_dict["n_errors"]
        ),
        "accuracy": accuracy,
        "error_prevalence": metrics_dict[
            "error_prevalence"
        ],
        "error_detection_auroc": metrics_dict[
            "error_detection_auroc"
        ],
        "error_detection_auprc": metrics_dict[
            "error_detection_auprc"
        ],
    }

    return pd.DataFrame(
        [metrics_row]
    )


# =============================================================================
# High-confidence error analysis
# =============================================================================


def identify_high_confidence_errors(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return incorrect predictions with confidence >= configured threshold."""

    high_confidence_errors = prediction_table[
        (prediction_table["error"] == 1)
        & (
            prediction_table["confidence"]
            >= HIGH_CONFIDENCE_THRESHOLD
        )
    ].copy()

    return high_confidence_errors.sort_values(
        by="predictive_entropy",
        ascending=True,
    )


# =============================================================================
# Artifact persistence
# =============================================================================


def save_artifacts(
    prediction_table: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    metrics_table: pd.DataFrame,
    baseline_integrity: dict[str, int | float | bool],
) -> None:
    """Save EXP-003 evidence after all required checks succeed."""

    if not bool(
        baseline_integrity["baseline_verified"]
    ):
        raise RuntimeError(
            "Artifact saving was blocked because the "
            "EXP-001 baseline was not verified."
        )

    RESULTS_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_integrity_table = pd.DataFrame(
        [baseline_integrity]
    )

    prediction_table.to_csv(
        PREDICTION_RESULTS_PATH,
        index=False,
    )

    uncertainty_summary.to_csv(
        UNCERTAINTY_SUMMARY_PATH,
        index=False,
    )

    metrics_table.to_csv(
        ERROR_DETECTION_METRICS_PATH,
        index=False,
    )

    baseline_integrity_table.to_csv(
        BASELINE_INTEGRITY_PATH,
        index=False,
    )


# =============================================================================
# Console reporting
# =============================================================================


def print_baseline_integrity(
    baseline_integrity: dict[str, int | float | bool],
) -> None:
    """Print EXP-001 reproduction information."""

    print()
    print("=" * 72)
    print("EXP-001 BASELINE INTEGRITY")
    print("=" * 72)

    print(
        f"Samples: "
        f"{baseline_integrity['n_samples']}"
    )

    print(
        f"Observed accuracy: "
        f"{float(baseline_integrity['accuracy']):.6f}"
    )

    print(
        f"Expected accuracy: "
        f"{EXPECTED_ACCURACY:.6f}"
    )

    print()
    print("Observed confusion matrix")

    print(
        f"TN={baseline_integrity['tn']}  "
        f"FP={baseline_integrity['fp']}  "
        f"FN={baseline_integrity['fn']}  "
        f"TP={baseline_integrity['tp']}"
    )

    print()
    print("Expected confusion matrix")

    print(
        f"TN={EXPECTED_TN}  "
        f"FP={EXPECTED_FP}  "
        f"FN={EXPECTED_FN}  "
        f"TP={EXPECTED_TP}"
    )

    print()

    print(
        "Baseline verified: "
        f"{baseline_integrity['baseline_verified']}"
    )

    print("=" * 72)


def print_experiment_summary(
    metrics_table: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    prediction_table: pd.DataFrame,
) -> None:
    """Print a concise EXP-003 experimental summary."""

    metrics = metrics_table.iloc[0]

    print()
    print("=" * 72)
    print(
        f"{EXPERIMENT_ID} — {EXPERIMENT_TITLE}"
    )
    print("=" * 72)

    print(
        f"Samples: "
        f"{int(metrics['n_samples'])}"
    )

    print(
        f"Correct predictions: "
        f"{int(metrics['n_correct'])}"
    )

    print(
        f"Errors: "
        f"{int(metrics['n_errors'])}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Error prevalence: "
        f"{metrics['error_prevalence']:.6f}"
    )

    print(
        f"Error-detection AUROC: "
        f"{metrics['error_detection_auroc']:.6f}"
    )

    print(
        f"Error-detection AUPRC: "
        f"{metrics['error_detection_auprc']:.6f}"
    )

    print()
    print("UNCERTAINTY BY CORRECTNESS")
    print("-" * 72)

    print(
        uncertainty_summary.to_string(
            index=False
        )
    )

    high_confidence_errors = (
        identify_high_confidence_errors(
            prediction_table
        )
    )

    print()
    print(
        "High-confidence errors "
        f"(confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(high_confidence_errors)}"
    )

    if not high_confidence_errors.empty:

        lowest_entropy_error = (
            high_confidence_errors
            .iloc[0]
        )

        print()
        print(
            "Lowest-entropy high-confidence error"
        )

        print(
            f"sample_id: "
            f"{int(lowest_entropy_error['sample_id'])}"
        )

        print(
            f"true_label: "
            f"{int(lowest_entropy_error['true_label'])}"
        )

        print(
            f"predicted_label: "
            f"{int(lowest_entropy_error['predicted_label'])}"
        )

        print(
            f"probability: "
            f"{lowest_entropy_error['predicted_probability']:.6f}"
        )

        print(
            f"confidence: "
            f"{lowest_entropy_error['confidence']:.6f}"
        )

        print(
            f"predictive_entropy: "
            f"{lowest_entropy_error['predictive_entropy']:.6f}"
        )

    print()
    print("ARTIFACTS")
    print("-" * 72)

    print(
        BASELINE_INTEGRITY_PATH
    )

    print(
        PREDICTION_RESULTS_PATH
    )

    print(
        UNCERTAINTY_SUMMARY_PATH
    )

    print(
        ERROR_DETECTION_METRICS_PATH
    )

    print("=" * 72)


# =============================================================================
# Main experiment
# =============================================================================


def main() -> None:
    """Execute EXP-003 with baseline-integrity protection."""

    print()
    print("=" * 72)
    print(
        f"{EXPERIMENT_ID}: {EXPERIMENT_TITLE}"
    )
    print("=" * 72)

    print(
        "Setting reproducibility controls..."
    )

    set_seed(
        SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()
    print(
        "Loading frozen PneumoniaMNIST test split..."
    )

    test_loader = create_test_loader()

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print()
    print(
        "Loading frozen EXP-001 checkpoint..."
    )

    model = load_frozen_model(
        device=device
    )

    print(
        "Checkpoint loaded successfully."
    )

    print()
    print(
        "Generating frozen-model predictions..."
    )

    y_true, probabilities = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    predicted_labels = probabilities_to_predictions(
        probabilities
    )

    print()
    print(
        "Verifying EXP-001 baseline integrity..."
    )

    baseline_integrity = calculate_baseline_integrity(
        y_true=y_true,
        predicted_labels=predicted_labels,
    )

    print_baseline_integrity(
        baseline_integrity
    )

    enforce_baseline_integrity(
        baseline_integrity
    )

    print()
    print(
        "EXP-001 baseline reproduced successfully."
    )

    print()
    print(
        "Computing deterministic predictive entropy..."
    )

    prediction_table = build_prediction_table(
        y_true=y_true,
        probabilities=probabilities,
    )

    print(
        "Running prediction-level integrity checks..."
    )

    run_prediction_integrity_checks(
        prediction_table
    )

    print(
        "Prediction-level integrity checks passed."
    )

    print()
    print(
        "Evaluating uncertainty as an error-detection signal..."
    )

    uncertainty_summary = (
        build_uncertainty_summary(
            prediction_table
        )
    )

    metrics_table = (
        build_error_detection_metrics(
            prediction_table
        )
    )

    print()
    print(
        "Saving verified EXP-003 artifacts..."
    )

    save_artifacts(
        prediction_table=prediction_table,
        uncertainty_summary=uncertainty_summary,
        metrics_table=metrics_table,
        baseline_integrity=baseline_integrity,
    )

    print(
        "Artifacts saved successfully."
    )

    print_experiment_summary(
        metrics_table=metrics_table,
        uncertainty_summary=uncertainty_summary,
        prediction_table=prediction_table,
    )

    print()
    print(
        "EXP-003 execution completed."
    )


if __name__ == "__main__":
    main()
