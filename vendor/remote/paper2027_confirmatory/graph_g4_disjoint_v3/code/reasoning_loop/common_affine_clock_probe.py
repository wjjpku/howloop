"""Linear algebra for common affine clock probes across several maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactCommonDirectionTest:
    stacked_matrix: np.ndarray
    singular_values: np.ndarray
    unit_probe: np.ndarray


def stack_row_state_probe_constraints(weights: list[np.ndarray]) -> np.ndarray:
    """Stack ``W_i-I`` for row-state maps ``h -> h W_i + b_i``.

    A scalar probe is represented by a column vector ``omega`` and reads
    ``h @ omega``.  Its shift is independent of ``h`` exactly when
    ``(W_i-I) @ omega == 0`` for every map.
    """

    if not weights:
        raise ValueError("at least one weight matrix is required")
    dimension = weights[0].shape[0]
    identity = np.eye(dimension)
    if any(weight.shape != (dimension, dimension) for weight in weights):
        raise ValueError("all weights must be square with the same dimension")
    return np.concatenate([(weight - identity) for weight in weights], axis=0)


def stack_left_invariance_constraints(weights: list[np.ndarray]) -> np.ndarray:
    """Backward-compatible alias for the row-state probe constraints."""

    return stack_row_state_probe_constraints(weights)


def exact_common_direction_test(weights: list[np.ndarray]) -> ExactCommonDirectionTest:
    """Return the full singular test and best unit approximate direction."""

    stacked = stack_row_state_probe_constraints(weights)
    _, singular_values, right = np.linalg.svd(stacked, full_matrices=False)
    return ExactCommonDirectionTest(
        stacked_matrix=stacked,
        singular_values=singular_values,
        unit_probe=right[-1].copy(),
    )


def orient_probe(probe: np.ndarray, biases: np.ndarray, target_sign: float = -1.0) -> np.ndarray:
    """Choose the otherwise arbitrary sign from the mean bias projection."""

    mean_delta = float((np.asarray(biases).T @ probe).mean())
    if mean_delta == 0.0:
        return probe.copy()
    return probe * np.sign(target_sign * mean_delta)


def scale_probe_to_mean_delta(
    probe: np.ndarray, biases: np.ndarray, target: float
) -> np.ndarray:
    """Scale a direction so the mean ``probe @ b_i`` equals ``target``."""

    mean_bias = np.asarray(biases).mean(axis=1)
    denominator = float(mean_bias @ probe)
    if abs(denominator) < 1e-15:
        raise ValueError("probe has zero mean bias projection")
    return probe * (target / denominator)


def minimum_residual_mean_delta_probe(
    weights: list[np.ndarray], biases: np.ndarray, target: float
) -> np.ndarray:
    """Minimize global invariance residual with a fixed mean affine shift."""

    stacked = stack_row_state_probe_constraints(weights)
    gram = stacked.T @ stacked
    mean_bias = np.asarray(biases).mean(axis=1)
    inverse_bias = np.linalg.solve(gram, mean_bias)
    denominator = float(mean_bias @ inverse_bias)
    return inverse_bias * (target / denominator)


def minimum_residual_fixed_delta_probe(
    weights: list[np.ndarray], biases: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Minimize global invariance residual subject to every affine shift target."""

    stacked = stack_row_state_probe_constraints(weights)
    gram = stacked.T @ stacked
    bias_matrix = np.asarray(biases)
    target = np.asarray(targets, dtype=np.float64)
    if bias_matrix.ndim != 2 or target.shape != (bias_matrix.shape[1],):
        raise ValueError("biases must be [dimension, maps] with one target per map")
    inverse_biases = np.linalg.solve(gram, bias_matrix)
    constraint_metric = bias_matrix.T @ inverse_biases
    coefficients = np.linalg.lstsq(constraint_metric, target, rcond=None)[0]
    probe = inverse_biases @ coefficients
    achieved = bias_matrix.T @ probe
    if not np.allclose(achieved, target, rtol=1e-8, atol=1e-10):
        raise ValueError("fixed affine-shift constraints are mutually inconsistent")
    return probe
