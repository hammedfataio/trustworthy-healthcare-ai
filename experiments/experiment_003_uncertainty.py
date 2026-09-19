"""EXP-003: Predictive Uncertainty for Error Detection.

Research question
-----------------
Does predictive uncertainty provide useful information for distinguishing
incorrect from correct predictions produced by the frozen baseline
medical-image classifier?

This experiment:

1. Loads the PneumoniaMNIST test split.
2. Reconstructs the frozen EXP-001 BaselineCNN architecture.
3. Loads the frozen EXP-001 checkpoint.
4. Generates test-set probabilities.
5. Computes deterministic binary predictive entropy.
6. Treats prediction error as the positive class.
7. Evaluates uncertainty using error-detection AUROC and AUPRC.
8. Saves prediction-level and summary artifacts.

Important
---------
The test set is used only for final evaluation.

No uncertainty threshold, model parameter, or method is selected using
test-set results.

Predictive entropy is a deterministic uncertainty baseline and should not
be interpreted as complete epistemic uncertainty.
"""

from __future__ import annotations

import random
from pathlib import Path

import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medmnist import INFO
from sklearn.metrics import accuracy_score
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


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED = 42
DATA_FLAG = "pneumoniamnist"
BATCH_SIZE = 64
CLASSIFICATION_THRESHOLD = 0.5

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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    """Set random seeds used by the experiment."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Frozen EXP-001 model architecture
# ---------------------------------------------------------------------------


class BaselineCNN(nn.Module):
    """CNN architecture used in EXP-001.

    This architecture must remain consistent with the model whose parameters
    are stored in the frozen EXP-001 checkpoint.

    EXP-003 does not retrain this model.
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
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw binary-classification logits."""

        features = self.features(x)

        return self.classifier(features)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def create_test_loader() -> DataLoader:
    """Create the frozen PneumoniaMNIST test-set loader."""

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

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_frozen_model(
    device: torch.device,
) -> BaselineCNN:
    """Load the frozen EXP-001 CNN checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "EXP-001 checkpoint was not found at:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "EXP-003 must use the frozen EXP-001 model. "
            "Do not train a replacement model inside this experiment."
        )

    model = BaselineCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # Support either a raw state_dict or a checkpoint dictionary
    # containing a model_state_dict entry.
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


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

            labels = labels.view(-1).cpu().numpy()

            logits = model(images).view(-1)

            probabilities = torch.sigmoid(logits)

            all_labels.append(
                labels.astype(np.int64)
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.concatenate(
        all_labels
    )

    probabilities = np.concatenate(
        all_probabilities
    )

    return (
        y_true,
        probabilities,
    )


# ---------------------------------------------------------------------------
# Artifact creation
# ---------------------------------------------------------------------------


def build_prediction_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build the prediction-level EXP-003 evidence table."""

    predicted_labels = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(np.int64)

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
            "true_label": y_true.astype(np.int64),
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


def build_uncertainty_summary(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive uncertainty statistics."""

    summary = summarize_uncertainty_by_correctness(
        correctness=prediction_table["correct"].to_numpy(),
        uncertainty_scores=prediction_table[
            "predictive_entropy"
        ].to_numpy(),
    )

    rows = []

    for group_name, statistics in summary.items():

        row = {
            "prediction_group": group_name,
            **statistics,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_error_detection_metrics(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate primary EXP-003 error-detection metrics."""

    metrics = evaluate_error_detection(
        error_targets=prediction_table[
            "error"
        ].to_numpy(),
        uncertainty_scores=prediction_table[
            "predictive_entropy"
        ].to_numpy(),
    )

    accuracy = accuracy_score(
        prediction_table["true_label"],
        prediction_table["predicted_label"],
    )

    metrics_row = {
        "n_samples": metrics.n_samples,
        "n_errors": metrics.n_errors,
        "n_correct": (
            metrics.n_samples
            - metrics.n_errors
        ),
        "accuracy": float(accuracy),
        "error_prevalence": metrics.error_prevalence,
        "error_detection_auroc": (
            metrics.error_detection_auroc
        ),
        "error_detection_auprc": (
            metrics.error_detection_auprc
        ),
    }

    return pd.DataFrame(
        [metrics_row]
    )


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------


def run_integrity_checks(
    prediction_table: pd.DataFrame,
) -> None:
    """Check assumptions before EXP-003 artifacts are accepted."""

    if prediction_table.empty:
        raise RuntimeError(
            "Prediction table is empty."
        )

    if len(prediction_table) != 624:
        raise RuntimeError(
            "Unexpected PneumoniaMNIST test-set size. "
            f"Expected 624 samples but received "
            f"{len(prediction_table)}."
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

    numeric_columns = [
        "predicted_probability",
        "confidence",
        "predictive_entropy",
        "normalized_predictive_entropy",
    ]

    for column in numeric_columns:

        values = prediction_table[
            column
        ].to_numpy()

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                f"{column} contains non-finite values."
            )

    probabilities = prediction_table[
        "predicted_probability"
    ].to_numpy()

    if np.any(
        (probabilities < 0.0)
        | (probabilities > 1.0)
    ):
        raise RuntimeError(
            "Predicted probabilities fall outside [0, 1]."
        )

    entropy = prediction_table[
        "predictive_entropy"
    ].to_numpy()

    if np.any(entropy < 0.0):
        raise RuntimeError(
            "Predictive entropy contains negative values."
        )

    normalized_entropy = prediction_table[
        "normalized_predictive_entropy"
    ].to_numpy()

    tolerance = 1e-10

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
            "Correctness and error indicators "
            "are inconsistent."
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_experiment_summary(
    metrics_table: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    prediction_table: pd.DataFrame,
) -> None:
    """Print a concise EXP-003 result summary."""

    metrics = metrics_table.iloc[0]

    print()
    print("=" * 72)
    print("EXP-003 — Predictive Uncertainty for Error Detection")
    print("=" * 72)

    print(
        f"Samples: {int(metrics['n_samples'])}"
    )

    print(
        f"Correct predictions: "
        f"{int(metrics['n_correct'])}"
    )

    print(
        f"Errors: {int(metrics['n_errors'])}"
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
    print("Uncertainty by correctness")
    print("-" * 72)

    print(
        uncertainty_summary.to_string(
            index=False
        )
    )

    high_confidence_errors = prediction_table[
        (prediction_table["error"] == 1)
        & (prediction_table["confidence"] >= 0.90)
    ]

    print()
    print(
        "High-confidence errors "
        "(confidence >= 0.90): "
        f"{len(high_confidence_errors)}"
    )

    if not high_confidence_errors.empty:

        lowest_entropy_error = (
            high_confidence_errors
            .sort_values(
                "predictive_entropy",
                ascending=True,
            )
            .iloc[0]
        )

        print(
            "Lowest-entropy high-confidence error:"
        )

        print(
            f"  sample_id: "
            f"{int(lowest_entropy_error['sample_id'])}"
        )

        print(
            f"  true_label: "
            f"{int(lowest_entropy_error['true_label'])}"
        )

        print(
            f"  predicted_label: "
            f"{int(lowest_entropy_error['predicted_label'])}"
        )

        print(
            f"  probability: "
            f"{lowest_entropy_error['predicted_probability']:.6f}"
        )

        print(
            f"  confidence: "
            f"{lowest_entropy_error['confidence']:.6f}"
        )

        print(
            f"  predictive_entropy: "
            f"{lowest_entropy_error['predictive_entropy']:.6f}"
        )

    print()
    print("Artifacts")
    print("-" * 72)

    print(PREDICTION_RESULTS_PATH)
    print(UNCERTAINTY_SUMMARY_PATH)
    print(ERROR_DETECTION_METRICS_PATH)

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute EXP-003."""

    print(
        "Starting EXP-003: "
        "Predictive Uncertainty for Error Detection"
    )

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        "Loading PneumoniaMNIST test split..."
    )

    test_loader = create_test_loader()

    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    print(
        "Loading frozen EXP-001 checkpoint..."
    )

    model = load_frozen_model(
        device=device
    )

    print(
        "Generating frozen-model predictions..."
    )

    y_true, probabilities = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    print(
        "Computing deterministic predictive entropy..."
    )

    prediction_table = build_prediction_table(
        y_true=y_true,
        probabilities=probabilities,
    )

    print(
        "Running evidence-integrity checks..."
    )

    run_integrity_checks(
        prediction_table
    )

    print(
        "Evaluating uncertainty as an error detector..."
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

    RESULTS_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    print_experiment_summary(
        metrics_table=metrics_table,
        uncertainty_summary=uncertainty_summary,
        prediction_table=prediction_table,
    )


if __name__ == "__main__":
    main()
