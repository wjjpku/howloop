"""Exact adversarial linear probes under a natural-state R-squared constraint.

For a linear probe ``T(x) = x @ w + b``, the intercept can be profiled out by
centering ``x`` and ``y``.  The natural-state sum of squared errors is then a
quadratic function of ``w``, while the mean intervention shift is ``q @ w``.
This module solves the resulting quadratically constrained linear problem in
closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeGeometry:
    """Centered least-squares geometry shared by all intervention objectives."""

    x_mean: np.ndarray
    y_mean: float
    centered_x: np.ndarray
    centered_y: np.ndarray
    sst: float
    gram_eigenvalues: np.ndarray
    gram_eigenvectors: np.ndarray
    numerical_rank: int
    rank_tolerance: float
    ols_weight: np.ndarray
    ols_bias: float
    ols_sse: float
    ols_r2: float


@dataclass(frozen=True)
class AdversarialEndpoints:
    """Minimum and maximum intervention-shift probes for one R2 threshold."""

    status: str
    threshold_r2: float
    null_objective_norm: float
    min_weight: np.ndarray | None
    min_bias: float | None
    max_weight: np.ndarray | None
    max_bias: float | None
    min_objective: float
    max_objective: float
    predicted_half_width: float


@dataclass(frozen=True)
class LinearTargetProbe:
    """Best natural-state probe satisfying several linear shift targets."""

    status: str
    weight: np.ndarray
    bias: float
    constraint_residual_norm: float
    excess_sse_over_ols: float


def r2_score(x: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float) -> float:
    """Return ordinary sample R-squared for a linear probe."""

    residual = x @ weight + bias - y
    centered = y - y.mean()
    sst = float(centered @ centered)
    return float(1.0 - (residual @ residual) / sst)


def fit_scaled_ridge(
    x: np.ndarray, y: np.ndarray, ridge: float
) -> tuple[np.ndarray, float]:
    """Match the scaled ridge convention used by the original age probe."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    covariance = xc.T @ xc / x.shape[0]
    cross = xc.T @ yc / x.shape[0]
    scale = max(float(np.diag(covariance).mean()), 1e-8)
    weight = np.linalg.solve(
        covariance + ridge * scale * np.eye(x.shape[1]), cross
    )
    bias = y_mean - float(x_mean @ weight)
    return weight, bias


def build_probe_geometry(
    x: np.ndarray, y: np.ndarray, *, rcond: float = 1e-12
) -> ProbeGeometry:
    """Build the exact empirical least-squares ellipsoid in float64."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("expected x=[examples, features] and matching y=[examples]")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    gram = xc.T @ xc
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    tolerance = float(eigenvalues[0] * rcond) if eigenvalues.size else 0.0
    keep = eigenvalues > tolerance
    numerical_rank = int(keep.sum())
    vectors = eigenvectors[:, keep]
    values = eigenvalues[keep]
    cross = xc.T @ yc
    ols_weight = vectors @ ((vectors.T @ cross) / values)
    ols_bias = y_mean - float(x_mean @ ols_weight)
    residual = x @ ols_weight + ols_bias - y
    ols_sse = float(residual @ residual)
    sst = float(yc @ yc)
    return ProbeGeometry(
        x_mean=x_mean,
        y_mean=y_mean,
        centered_x=xc,
        centered_y=yc,
        sst=sst,
        gram_eigenvalues=eigenvalues,
        gram_eigenvectors=eigenvectors,
        numerical_rank=numerical_rank,
        rank_tolerance=tolerance,
        ols_weight=ols_weight,
        ols_bias=ols_bias,
        ols_sse=ols_sse,
        ols_r2=float(1.0 - ols_sse / sst),
    )


def adversarial_probe_endpoints(
    geometry: ProbeGeometry,
    objective: np.ndarray,
    *,
    threshold_r2: float,
    null_tolerance: float = 1e-10,
) -> AdversarialEndpoints:
    """Solve min/max ``objective @ w`` subject to natural R2 >= threshold.

    If the objective has a component in the empirical null space of centered
    natural states, the interval is unbounded: that component can be scaled
    without changing any natural-state prediction.
    """

    q = np.asarray(objective, dtype=np.float64)
    dimension = geometry.ols_weight.shape[0]
    if q.shape != (dimension,):
        raise ValueError(f"expected objective shape {(dimension,)}, got {q.shape}")
    max_sse = float((1.0 - threshold_r2) * geometry.sst)
    slack_sse = max_sse - geometry.ols_sse
    if slack_sse < -1e-9 * max(geometry.sst, 1.0):
        raise ValueError(
            f"R2 threshold {threshold_r2:.9g} exceeds OLS R2 {geometry.ols_r2:.9g}"
        )
    slack_sse = max(slack_sse, 0.0)

    rank = geometry.numerical_rank
    vectors = geometry.gram_eigenvectors[:, :rank]
    values = geometry.gram_eigenvalues[:rank]
    projected = vectors.T @ q
    q_range = vectors @ projected
    q_null = q - q_range
    null_norm = float(np.linalg.norm(q_null))
    if null_norm > null_tolerance * max(float(np.linalg.norm(q)), 1.0):
        return AdversarialEndpoints(
            status="unbounded_empirical_null_space",
            threshold_r2=threshold_r2,
            null_objective_norm=null_norm,
            min_weight=None,
            min_bias=None,
            max_weight=None,
            max_bias=None,
            min_objective=float("-inf"),
            max_objective=float("inf"),
            predicted_half_width=float("inf"),
        )

    inverse_q = vectors @ (projected / values)
    q_inverse_q = float(q @ inverse_q)
    center = float(q @ geometry.ols_weight)
    if q_inverse_q <= 0.0 or slack_sse == 0.0:
        min_weight = geometry.ols_weight.copy()
        max_weight = geometry.ols_weight.copy()
        half_width = 0.0
    else:
        scale = float(np.sqrt(slack_sse / q_inverse_q))
        delta = scale * inverse_q
        min_weight = geometry.ols_weight - delta
        max_weight = geometry.ols_weight + delta
        half_width = float(np.sqrt(slack_sse * q_inverse_q))
    min_bias = geometry.y_mean - float(geometry.x_mean @ min_weight)
    max_bias = geometry.y_mean - float(geometry.x_mean @ max_weight)
    return AdversarialEndpoints(
        status="bounded",
        threshold_r2=threshold_r2,
        null_objective_norm=null_norm,
        min_weight=min_weight,
        min_bias=min_bias,
        max_weight=max_weight,
        max_bias=max_bias,
        min_objective=center - half_width,
        max_objective=center + half_width,
        predicted_half_width=half_width,
    )


def minimum_loss_probe_for_linear_targets(
    geometry: ProbeGeometry,
    objectives: np.ndarray,
    targets: np.ndarray,
    *,
    residual_tolerance: float = 1e-9,
) -> LinearTargetProbe:
    """Minimize natural-state SSE subject to ``objectives @ w = targets``.

    This is the equality-constrained companion to the adversarial interval.
    It answers whether one shared probe can realize several desired J shifts
    simultaneously, and how much natural-state fit that requirement costs.
    """

    q = np.asarray(objectives, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    dimension = geometry.ols_weight.shape[0]
    if q.ndim != 2 or q.shape[1] != dimension:
        raise ValueError(
            f"expected objectives=[constraints, {dimension}], got {q.shape}"
        )
    if target.shape != (q.shape[0],):
        raise ValueError(f"expected targets shape {(q.shape[0],)}, got {target.shape}")

    rank = geometry.numerical_rank
    vectors = geometry.gram_eigenvectors[:, :rank]
    values = geometry.gram_eigenvalues[:rank]
    inverse_q_transpose = vectors @ ((vectors.T @ q.T) / values[:, None])
    constraint_metric = q @ inverse_q_transpose
    desired_change = target - q @ geometry.ols_weight
    coefficients = np.linalg.pinv(constraint_metric, rcond=1e-12) @ desired_change
    correction = inverse_q_transpose @ coefficients
    weight = geometry.ols_weight + correction
    bias = geometry.y_mean - float(geometry.x_mean @ weight)
    residual_norm = float(np.linalg.norm(q @ weight - target))
    excess_sse = float(correction @ (geometry.centered_x.T @ geometry.centered_x) @ correction)
    status = "satisfied" if residual_norm <= residual_tolerance else "infeasible_retained_space"
    return LinearTargetProbe(
        status=status,
        weight=weight,
        bias=bias,
        constraint_residual_norm=residual_norm,
        excess_sse_over_ols=excess_sse,
    )
