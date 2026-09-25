"""Pure utilities for Parity content-by-clock causal interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FactorialCell:
    """One content-relation by recurrent-phase intervention cell."""

    content_opposite: bool
    phase_offset: int

    @property
    def name(self) -> str:
        content = "opposite" if self.content_opposite else "same"
        return f"content_{content}__phase_{self.phase_offset:+d}"


def balanced_donor_indices(
    receiver_labels: torch.Tensor,
    donor_labels: torch.Tensor,
    *,
    opposite: bool,
    seed: int,
) -> torch.Tensor:
    """Choose deterministic donor indices with the requested parity relation."""

    receiver_cpu = receiver_labels.detach().to(device="cpu", dtype=torch.long)
    donor_cpu = donor_labels.detach().to(device="cpu", dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    wanted_labels = 1 - receiver_cpu if opposite else receiver_cpu
    selected = torch.empty_like(receiver_cpu)
    for wanted in (0, 1):
        receiver_positions = torch.nonzero(
            wanted_labels == wanted, as_tuple=False
        ).flatten()
        candidates = torch.nonzero(donor_cpu == wanted, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"donor class {wanted} is absent")
        assignments: list[torch.Tensor] = []
        remaining = int(receiver_positions.numel())
        while remaining > 0:
            permutation = candidates[
                torch.randperm(candidates.numel(), generator=generator)
            ]
            take = min(remaining, int(permutation.numel()))
            assignments.append(permutation[:take])
            remaining -= take
        if assignments:
            selected[receiver_positions] = torch.cat(assignments)
    return selected.to(device=donor_labels.device)


def patch_answer_plane(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: torch.Tensor,
    *,
    answer_position: int,
) -> torch.Tensor:
    """Replace the answer's two plane coordinates and preserve its complement."""

    if basis.ndim != 2 or basis.shape[1] != 2:
        raise ValueError("content-clock basis must have two columns")
    if receiver.shape != donor.shape:
        raise ValueError("receiver and donor states must have identical shapes")
    live_basis = basis.to(receiver.device, receiver.dtype)
    output = receiver.clone()
    delta = (donor[:, answer_position] - receiver[:, answer_position]) @ live_basis
    output[:, answer_position] += delta @ live_basis.T
    return output


def patch_answer_complement(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: torch.Tensor,
    *,
    answer_position: int,
) -> torch.Tensor:
    """Apply an equal-norm complement patch while preserving plane coordinates."""

    live_basis = basis.to(receiver.device, receiver.dtype)
    projector = torch.eye(
        live_basis.shape[0], device=receiver.device, dtype=receiver.dtype
    ) - live_basis @ live_basis.T
    output = receiver.clone()
    full_delta = donor[:, answer_position] - receiver[:, answer_position]
    plane_delta = (full_delta @ live_basis) @ live_basis.T
    complement_delta = full_delta @ projector
    plane_norm = torch.linalg.vector_norm(plane_delta, dim=-1, keepdim=True)
    complement_norm = torch.linalg.vector_norm(
        complement_delta, dim=-1, keepdim=True
    )
    complement_delta = complement_delta * plane_norm / complement_norm.clamp_min(1e-12)
    output[:, answer_position] += complement_delta
    return output


def project_answer(
    state: torch.Tensor,
    basis: torch.Tensor,
    answer_position: int,
) -> torch.Tensor:
    """Return float32 two-dimensional answer-token coordinates."""

    live_basis = basis.to(device=state.device, dtype=torch.float32)
    return state[:, answer_position].float() @ live_basis


def coordinate_angles(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert `[batch, 2]` coordinates to wrapped angles."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [batch, 2]")
    return torch.atan2(coordinates[:, 1], coordinates[:, 0])


def class_midpoint(
    coordinates: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return the midpoint between binary class centroids in a coordinate plane."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [batch, 2]")
    if labels.ndim != 1 or labels.shape[0] != coordinates.shape[0]:
        raise ValueError("labels must have shape [batch]")
    centroids = []
    for label in (0, 1):
        selected = coordinates[labels == label]
        if selected.shape[0] == 0:
            raise ValueError(f"class {label} is absent")
        centroids.append(selected.mean(dim=0))
    return 0.5 * (centroids[0] + centroids[1])


def wrap_angle(value: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""

    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def circular_difference(left: float, right: float) -> float:
    """Return the signed shortest angular displacement `left - right`."""

    return wrap_angle(left - right)


def circular_mean(values: Sequence[float]) -> float:
    """Return the circular mean of a nonempty angle sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot average an empty angle sequence")
    return float(np.arctan2(np.sin(array).mean(), np.cos(array).mean()))


def circular_mae(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute shortest angular error."""

    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted angles must be equally nonempty")
    errors = [abs(circular_difference(left, right)) for left, right in zip(observed, predicted)]
    return float(np.mean(errors))


def factorial_interaction_error(
    angles: dict[tuple[bool, int], float],
    *,
    phase_offset: int,
) -> float:
    """Circular difference-in-differences for the content by phase design."""

    content_at_shift = circular_difference(
        angles[(True, phase_offset)], angles[(False, phase_offset)]
    )
    content_at_zero = circular_difference(angles[(True, 0)], angles[(False, 0)])
    return circular_difference(content_at_shift, content_at_zero)


def _group_rows(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_factorial_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize factorial cells and compute circular interaction residuals."""

    if not rows:
        raise ValueError("cannot summarize empty factorial rows")
    group_keys = (
        "backbone_seed",
        "length",
        "data_seed",
        "tested_phase_offset",
        "phase_offset",
        "continuation_calls",
        "control_family",
        "condition",
        "content_relation",
        "receiver_label",
    )
    summaries: list[dict[str, Any]] = []
    for key, selected in sorted(_group_rows(rows, group_keys).items()):
        output = dict(zip(group_keys, key))
        observed = [float(row["observed_angle"]) for row in selected]
        predicted = [float(row["predicted_angle"]) for row in selected]
        if "observed_phase_x" in selected[0]:
            mean_observed_x = float(
                np.mean([float(row["observed_phase_x"]) for row in selected])
            )
            mean_observed_y = float(
                np.mean([float(row["observed_phase_y"]) for row in selected])
            )
            mean_observed_angle = float(
                np.arctan2(mean_observed_y, mean_observed_x)
            )
        else:
            mean_observed_x = float("nan")
            mean_observed_y = float("nan")
            mean_observed_angle = circular_mean(observed)
        output.update(
            {
                "examples": len(selected),
                "mean_observed_x": mean_observed_x,
                "mean_observed_y": mean_observed_y,
                "mean_observed_angle": mean_observed_angle,
                "mean_predicted_angle": circular_mean(predicted),
                "circular_mae": circular_mae(observed, predicted),
                "mean_content_correct": float(
                    np.mean([float(row["content_correct"]) for row in selected])
                ),
                "mean_receiver_content_correct": float(
                    np.mean(
                        [float(row["receiver_content_correct"]) for row in selected]
                    )
                ),
                "mean_oriented_margin": float(
                    np.mean([float(row["oriented_margin"]) for row in selected])
                ),
                "mean_patch_norm": float(
                    np.mean([float(row["patch_norm"]) for row in selected])
                ),
                "mean_complement_displacement": float(
                    np.mean(
                        [float(row["complement_displacement"]) for row in selected]
                    )
                ),
            }
        )
        if "donor_target_angle" in selected[0]:
            if "donor_target_phase_x" in selected[0]:
                mean_target_x = float(
                    np.mean(
                        [float(row["donor_target_phase_x"]) for row in selected]
                    )
                )
                mean_target_y = float(
                    np.mean(
                        [float(row["donor_target_phase_y"]) for row in selected]
                    )
                )
                output["mean_donor_target_x"] = mean_target_x
                output["mean_donor_target_y"] = mean_target_y
                output["mean_donor_target_angle"] = float(
                    np.arctan2(mean_target_y, mean_target_x)
                )
            else:
                output["mean_donor_target_angle"] = circular_mean(
                    [float(row["donor_target_angle"]) for row in selected]
                )
            output["mean_donor_target_circular_error"] = circular_mae(
                observed,
                [float(row["donor_target_angle"]) for row in selected],
            )
        if "target_readout_match" in selected[0]:
            output["mean_target_readout_match"] = float(
                np.mean([float(row["target_readout_match"]) for row in selected])
            )
        if "target_oriented_margin" in selected[0]:
            output["mean_target_oriented_margin"] = float(
                np.mean([float(row["target_oriented_margin"]) for row in selected])
            )
        summaries.append(output)

    interaction_keys = (
        "backbone_seed",
        "length",
        "data_seed",
        "tested_phase_offset",
        "continuation_calls",
        "control_family",
        "condition",
        "receiver_label",
    )
    interactions: list[dict[str, Any]] = []
    for key, selected in sorted(_group_rows(summaries, interaction_keys).items()):
        output = dict(zip(interaction_keys, key))
        tested_offset = int(output["tested_phase_offset"])
        angle_map: dict[tuple[bool, int], float] = {}
        for row in selected:
            relation = str(row["content_relation"])
            angle_map[(relation == "opposite", int(row["phase_offset"]))] = float(
                row["mean_observed_angle"]
            )
        required = {
            (False, 0),
            (True, 0),
            (False, tested_offset),
            (True, tested_offset),
        }
        if required.issubset(angle_map):
            output["factorial_interaction_error"] = factorial_interaction_error(
                angle_map, phase_offset=tested_offset
            )
            output["absolute_factorial_interaction_error"] = abs(
                float(output["factorial_interaction_error"])
            )
            interactions.append(output)
    return summaries, interactions
