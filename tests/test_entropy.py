"""Tests for predictive-entropy utilities used in EXP-003."""

import numpy as np
import pytest
import torch

from src.uncertainty.entropy import (
    binary_predictive_entropy,
    normalized_binary_predictive_entropy,
    torch_binary_predictive_entropy,
)


def test_binary_entropy_is_maximum_at_half():
    """Binary predictive entropy should equal log(2) at p = 0.5."""

    probabilities = np.array([0.5])

    entropy = binary_predictive_entropy(probabilities)

    assert np.isclose(
        entropy[0],
        np.log(2.0),
        atol=1e-10,
    )


def test_binary_entropy_is_low_near_probability_extremes():
    """Probabilities near 0 and 1 should have very low entropy."""

    probabilities = np.array([1e-9, 1.0 - 1e-9])

    entropy = binary_predictive_entropy(probabilities)

    assert np.all(entropy < 1e-6)


def test_binary_entropy_is_symmetric():
    """H(p) should equal H(1-p)."""

    probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    entropy_left = binary_predictive_entropy(probabilities)
    entropy_right = binary_predictive_entropy(1.0 - probabilities)

    assert np.allclose(
        entropy_left,
        entropy_right,
        atol=1e-12,
    )


def test_normalized_entropy_is_one_at_half():
    """Normalized entropy should equal 1 at p = 0.5."""

    probabilities = np.array([0.5])

    entropy = normalized_binary_predictive_entropy(probabilities)

    assert np.isclose(
        entropy[0],
        1.0,
        atol=1e-10,
    )


def test_normalized_entropy_stays_within_expected_range():
    """Normalized entropy should remain approximately within [0, 1]."""

    probabilities = np.linspace(0.0, 1.0, 101)

    entropy = normalized_binary_predictive_entropy(probabilities)

    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= 1.0 + 1e-12)


def test_exact_zero_and_one_are_handled_safely():
    """Exact boundary probabilities should not produce NaN or infinity."""

    probabilities = np.array([0.0, 1.0])

    entropy = binary_predictive_entropy(probabilities)

    assert np.all(np.isfinite(entropy))
    assert np.all(entropy >= 0.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([-0.1]),
        np.array([1.1]),
        np.array([-0.1, 0.5]),
        np.array([0.5, 1.1]),
    ],
)
def test_invalid_probability_range_raises_value_error(probabilities):
    """Probabilities outside [0, 1] must be rejected."""

    with pytest.raises(
        ValueError,
        match=r"Probabilities must lie within \[0, 1\]",
    ):
        binary_predictive_entropy(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
    ],
)
def test_nonfinite_probabilities_raise_value_error(probabilities):
    """NaN and infinite probabilities must be rejected."""

    with pytest.raises(
        ValueError,
        match="Probabilities must contain only finite values",
    ):
        binary_predictive_entropy(probabilities)


@pytest.mark.parametrize(
    "eps",
    [
        0.0,
        -1e-12,
        0.5,
        0.6,
    ],
)
def test_invalid_eps_raises_value_error(eps):
    """The numerical-stability constant must satisfy 0 < eps < 0.5."""

    probabilities = np.array([0.5])

    with pytest.raises(
        ValueError,
        match="eps must satisfy",
    ):
        binary_predictive_entropy(
            probabilities,
            eps=eps,
        )


def test_list_input_is_supported():
    """The NumPy implementation should also accept a Python list."""

    probabilities = [0.25, 0.5, 0.75]

    entropy = binary_predictive_entropy(probabilities)

    assert isinstance(entropy, np.ndarray)
    assert entropy.shape == (3,)


def test_numpy_output_shape_matches_input_shape():
    """Entropy should preserve the shape of NumPy probability arrays."""

    probabilities = np.array(
        [
            [0.1, 0.2],
            [0.8, 0.9],
        ]
    )

    entropy = binary_predictive_entropy(probabilities)

    assert entropy.shape == probabilities.shape


def test_torch_entropy_is_maximum_at_half():
    """PyTorch entropy should equal log(2) at p = 0.5."""

    probabilities = torch.tensor(
        [0.5],
        dtype=torch.float64,
    )

    entropy = torch_binary_predictive_entropy(probabilities)

    expected = torch.tensor(
        np.log(2.0),
        dtype=torch.float64,
    )

    assert torch.isclose(
        entropy[0],
        expected,
        atol=1e-10,
    )


def test_torch_entropy_preserves_shape():
    """PyTorch entropy should preserve the input tensor shape."""

    probabilities = torch.tensor(
        [
            [0.1, 0.2],
            [0.8, 0.9],
        ],
        dtype=torch.float32,
    )

    entropy = torch_binary_predictive_entropy(probabilities)

    assert entropy.shape == probabilities.shape


def test_torch_entropy_preserves_device():
    """Returned entropy should remain on the input tensor device."""

    probabilities = torch.tensor(
        [0.2, 0.5, 0.8],
        dtype=torch.float32,
    )

    entropy = torch_binary_predictive_entropy(probabilities)

    assert entropy.device == probabilities.device


def test_torch_integer_input_is_converted_to_float():
    """Integer probability tensors should be safely converted."""

    probabilities = torch.tensor(
        [0, 1],
        dtype=torch.int64,
    )

    entropy = torch_binary_predictive_entropy(probabilities)

    assert entropy.is_floating_point()
    assert torch.all(torch.isfinite(entropy))


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([-0.1]),
        torch.tensor([1.1]),
    ],
)
def test_torch_invalid_probability_range_raises_value_error(probabilities):
    """PyTorch probabilities outside [0, 1] must be rejected."""

    with pytest.raises(
        ValueError,
        match=r"Probabilities must lie within \[0, 1\]",
    ):
        torch_binary_predictive_entropy(probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([float("-inf")]),
    ],
)
def test_torch_nonfinite_probabilities_raise_value_error(probabilities):
    """PyTorch NaN and infinite values must be rejected."""

    with pytest.raises(
        ValueError,
        match="Probabilities must contain only finite values",
    ):
        torch_binary_predictive_entropy(probabilities)


def test_torch_requires_tensor_input():
    """The PyTorch implementation should reject non-tensor input."""

    with pytest.raises(
        TypeError,
        match="probabilities must be a torch.Tensor",
    ):
        torch_binary_predictive_entropy([0.2, 0.5, 0.8])


def test_numpy_and_torch_implementations_agree():
    """NumPy and PyTorch implementations should produce equivalent results."""

    values = [0.05, 0.2, 0.5, 0.8, 0.95]

    numpy_probabilities = np.array(
        values,
        dtype=np.float64,
    )

    torch_probabilities = torch.tensor(
        values,
        dtype=torch.float64,
    )

    numpy_entropy = binary_predictive_entropy(
        numpy_probabilities
    )

    torch_entropy = torch_binary_predictive_entropy(
        torch_probabilities
    )

    assert np.allclose(
        numpy_entropy,
        torch_entropy.detach().cpu().numpy(),
        atol=1e-10,
    )
