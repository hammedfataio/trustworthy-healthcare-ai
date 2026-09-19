"""Predictive-entropy utilities for uncertainty analysis.

This module implements deterministic predictive entropy for binary
classification.

The functions are used in EXP-003 as the first uncertainty baseline for
investigating whether uncertainty scores can help distinguish incorrect
predictions from correct predictions.

Important
---------
Predictive entropy derived from a single deterministic model's output
probability is not a complete measure of epistemic uncertainty. A model may
still be confidently wrong and therefore produce low predictive entropy for
an incorrect prediction.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_eps(eps: float) -> None:
    """Validate the numerical-stability constant."""

    if not 0.0 < eps < 0.5:
        raise ValueError("eps must satisfy 0 < eps < 0.5.")


def _validate_numpy_probabilities(probabilities: np.ndarray) -> None:
    """Validate a NumPy array containing probabilities."""

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain only finite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie within [0, 1].")


def _validate_torch_probabilities(probabilities: torch.Tensor) -> None:
    """Validate a PyTorch tensor containing probabilities."""

    if not torch.isfinite(probabilities).all().item():
        raise ValueError("Probabilities must contain only finite values.")

    if torch.any((probabilities < 0.0) | (probabilities > 1.0)).item():
        raise ValueError("Probabilities must lie within [0, 1].")


def binary_predictive_entropy(
    probabilities: np.ndarray | list[float],
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute binary predictive entropy using NumPy.

    For a predicted positive-class probability ``p``:

        H(p) = -p log(p) - (1 - p) log(1 - p)

    Natural logarithms are used, so maximum binary entropy is ``log(2)``
    when ``p = 0.5``.

    Parameters
    ----------
    probabilities:
        Positive-class probabilities in the closed interval [0, 1].

    eps:
        Small numerical-stability constant used before evaluating logarithms.

    Returns
    -------
    np.ndarray
        Predictive entropy for each supplied probability.

    Raises
    ------
    ValueError
        If ``eps`` is invalid, probabilities are non-finite, or values fall
        outside [0, 1].
    """

    _validate_eps(eps)

    probabilities_array = np.asarray(probabilities, dtype=np.float64)

    _validate_numpy_probabilities(probabilities_array)

    clipped = np.clip(probabilities_array, eps, 1.0 - eps)

    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )

    return entropy


def normalized_binary_predictive_entropy(
    probabilities: np.ndarray | list[float],
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute normalized binary predictive entropy.

    Binary entropy is divided by ``log(2)`` so that the theoretical range is
    approximately [0, 1].

    A value near 0 represents low predictive entropy, while a value near 1
    represents maximum binary predictive entropy.
    """

    entropy = binary_predictive_entropy(
        probabilities=probabilities,
        eps=eps,
    )

    return entropy / np.log(2.0)


def torch_binary_predictive_entropy(
    probabilities: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute binary predictive entropy for a PyTorch tensor.

    The returned tensor remains on the same device as the input tensor.

    Parameters
    ----------
    probabilities:
        Tensor containing positive-class probabilities in [0, 1].

    eps:
        Small numerical-stability constant.

    Returns
    -------
    torch.Tensor
        Predictive entropy with the same shape and device as the input.

    Notes
    -----
    This deterministic entropy baseline measures uncertainty implied by the
    model's predicted probability. It should not be interpreted as complete
    epistemic uncertainty.
    """

    _validate_eps(eps)

    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor.")

    _validate_torch_probabilities(probabilities)

    if not probabilities.is_floating_point():
        probabilities = probabilities.to(dtype=torch.float32)

    # torch.clamp with scalar bounds is safer across floating-point dtypes.
    finfo = torch.finfo(probabilities.dtype)
    lower = max(eps, finfo.eps)
    upper = min(1.0 - eps, 1.0 - finfo.eps)

    clipped = torch.clamp(
        probabilities,
        min=lower,
        max=upper,
    )

    entropy = -(
        clipped * torch.log(clipped)
        + (1.0 - clipped) * torch.log(1.0 - clipped)
    )

    return entropy
