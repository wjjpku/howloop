#!/usr/bin/env python3
"""Test whether Parity's local four-call behavior is a hidden-state phase orbit.

The primary object is the answer-token parity contrast

    q_(n,d) = E[h_(n,n+d) | y=1] - E[h_(n,n+d) | y=0].

Discovery and evaluation lengths/seeds are disjoint.  A candidate two-dimensional
phase plane is discovered only from the period-four Fourier coefficients of the
discovery contrasts.  The script then tests held-out rank, dynamics, four-step
closure, MLP-skip lag, and causal phase-component interchange.  It can analyze
both the raw frozen backbone and the released rank-48 controller trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# The phase plane is discovered only on the first set.  Evaluation and causal
# interventions deliberately use held-out lengths; the latter may share the
# evaluation lengths but must never share the discovery lengths.
DEFAULT_DISCOVERY_LENGTHS = (12, 16, 20, 24, 32, 40, 64, 80)
DEFAULT_EVALUATION_LENGTHS = (10, 14, 18, 22, 28, 36, 48, 72, 100)
DEFAULT_CAUSAL_LENGTHS = (10, 22, 36, 72)

try:
    from paper_length_telomere import (
        generate_paper_batch,
        load_backbone,
        load_controller,
        pick_device,
    )
except ModuleNotFoundError:  # package import in tests
    from reasoning_loop.paper_length_telomere import (
        generate_paper_batch,
        load_backbone,
        load_controller,
        pick_device,
    )


@dataclass
class ContrastRecord:
    variant: str
    split: str
    length: int
    relative_depth: int
    class0_mean: torch.Tensor
    class1_mean: torch.Tensor
    state_mean: torch.Tensor
    accuracy: float
    margin: float
    examples: int
    class0_examples: int
    class1_examples: int

    @property
    def contrast(self) -> torch.Tensor:
        return self.class1_mean - self.class0_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--discovery-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_DISCOVERY_LENGTHS,
    )
    parser.add_argument(
        "--evaluation-lengths",
        nargs="+",
        type=int,
        default=DEFAULT_EVALUATION_LENGTHS,
    )
    parser.add_argument(
        "--causal-lengths", nargs="+", type=int, default=DEFAULT_CAUSAL_LENGTHS
    )
    parser.add_argument("--relative-start", type=int, default=0)
    parser.add_argument("--relative-end", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--causal-batch-size", type=int, default=256)
    parser.add_argument("--discovery-seed", type=int, default=2026081101)
    parser.add_argument("--evaluation-seed", type=int, default=2026081201)
    parser.add_argument("--causal-seed", type=int, default=2026081301)
    parser.add_argument(
        "--random-controls",
        type=int,
        default=100,
        help="number of energy-matched random two-dimensional-plane controls",
    )
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="reject checkpoints outside the corrected input-once Parity protocol",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def orthonormalize(columns: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(columns.float(), mode="reduced")
    return q


def random_plane(
    dimension: int,
    *,
    seed: int,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(dimension, 2, generator=generator)
    if exclude is not None:
        candidate = candidate - exclude @ (exclude.T @ candidate)
    return orthonormalize(candidate)


def phase_patch(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: torch.Tensor,
    *,
    positions: slice | int,
) -> torch.Tensor:
    """Replace only donor-minus-receiver components in ``span(basis)``."""
    output = receiver.clone()
    delta = donor[:, positions].float() - receiver[:, positions].float()
    live_basis = basis.to(delta.device, delta.dtype)
    projected = (delta @ live_basis) @ live_basis.T
    output[:, positions] = receiver[:, positions].float() + projected
    return output


def complement_patch(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: torch.Tensor,
    *,
    positions: slice | int,
) -> torch.Tensor:
    output = receiver.clone()
    delta = donor[:, positions].float() - receiver[:, positions].float()
    live_basis = basis.to(delta.device, delta.dtype)
    projected = (delta @ live_basis) @ live_basis.T
    output[:, positions] = receiver[:, positions].float() + delta - projected
    return output


def magnitude_matched_random_patch(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    phase_basis: torch.Tensor,
    control_basis: torch.Tensor,
    *,
    position: int,
) -> torch.Tensor:
    output = receiver.clone()
    delta = donor[:, position].float() - receiver[:, position].float()
    phase_live = phase_basis.to(delta.device, delta.dtype)
    control_live = control_basis.to(delta.device, delta.dtype)
    phase_delta = (delta @ phase_live) @ phase_live.T
    random_delta = (delta @ control_live) @ control_live.T
    phase_norm = torch.linalg.vector_norm(phase_delta, dim=-1, keepdim=True)
    random_norm = torch.linalg.vector_norm(random_delta, dim=-1, keepdim=True)
    random_delta = random_delta * phase_norm / random_norm.clamp_min(1e-12)
    output[:, position] = receiver[:, position].float() + random_delta
    return output


def rotate_plane_component(
    receiver: torch.Tensor,
    basis: torch.Tensor,
    *,
    angle_radians: float,
    position: int,
) -> torch.Tensor:
    """Rotate just the answer-token component in a two-dimensional plane."""

    output = receiver.clone()
    live_basis = basis.to(receiver.device, torch.float32)
    original = receiver[:, position].float()
    coordinates = original @ live_basis
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]],
        device=original.device,
        dtype=original.dtype,
    )
    rotated = coordinates @ rotation.T
    delta = (rotated - coordinates) @ live_basis.T
    output[:, position] = original + delta
    return output


def remove_plane_component(
    receiver: torch.Tensor, basis: torch.Tensor, *, position: int
) -> torch.Tensor:
    """Delete the answer-token component in the discovered plane."""

    output = receiver.clone()
    live_basis = basis.to(receiver.device, torch.float32)
    original = receiver[:, position].float()
    output[:, position] = original - (original @ live_basis) @ live_basis.T
    return output


def keep_plane_component(
    receiver: torch.Tensor, basis: torch.Tensor, *, position: int
) -> torch.Tensor:
    """Keep only the discovered-plane component at the answer position."""

    output = receiver.clone()
    live_basis = basis.to(receiver.device, torch.float32)
    original = receiver[:, position].float()
    output[:, position] = (original @ live_basis) @ live_basis.T
    return output


def magnitude_matched_random_rotation(
    receiver: torch.Tensor,
    phase_basis: torch.Tensor,
    control_basis: torch.Tensor,
    *,
    angle_radians: float,
    position: int,
) -> torch.Tensor:
    """Match the phase-rotation update norm in an orthogonal random plane."""

    phase_rotated = rotate_plane_component(
        receiver,
        phase_basis,
        angle_radians=angle_radians,
        position=position,
    )
    random_rotated = rotate_plane_component(
        receiver,
        control_basis,
        angle_radians=angle_radians,
        position=position,
    )
    output = receiver.clone()
    phase_delta = phase_rotated[:, position].float() - receiver[:, position].float()
    random_delta = random_rotated[:, position].float() - receiver[:, position].float()
    scale = torch.linalg.vector_norm(phase_delta, dim=-1, keepdim=True) / torch.linalg.vector_norm(
        random_delta, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    output[:, position] = receiver[:, position].float() + random_delta * scale
    return output


def step_state(
    model: Any,
    state: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
    output_step: int,
    skip_mlp: bool = False,
    skip_attention: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    controlled = state
    if (
        controller is not None
        and controller_anchor is not None
        and output_step > controller_anchor
    ):
        controlled = controller(controlled)
    if len(model.layers) != 1:
        raise ValueError("four-phase analysis expects one shared physical layer")
    layer = model.layers[0]
    pre_attention = controlled + embeddings
    attention_output = (
        torch.zeros_like(pre_attention)
        if skip_attention
        else layer.attention(layer.attention_norm(pre_attention))
    )
    post_attention = pre_attention + attention_output
    mlp_output = layer.mlp(layer.mlp_norm(post_attention))
    pre_final_norm = post_attention if skip_mlp else post_attention + mlp_output
    final_state = model.final_norm(pre_final_norm)
    return final_state, {
        "source": state,
        "controlled_source": controlled,
        "pre_attention": pre_attention,
        "post_attention": post_attention,
        "mlp_output": mlp_output,
        "pre_final_norm": pre_final_norm,
        "final_state": final_state,
    }


def oriented_margin(
    logits: torch.Tensor, labels: torch.Tensor, answer_position: int
) -> torch.Tensor:
    answer = logits[:, answer_position]
    rows = torch.arange(answer.shape[0], device=answer.device)
    return answer[rows, labels] - answer[rows, 1 - labels]


def score_state(
    model: Any,
    state: torch.Tensor,
    labels: torch.Tensor,
    answer_position: int,
) -> tuple[float, float]:
    logits = model.decode(state).float()
    predictions = logits[:, answer_position].argmax(dim=-1)
    return (
        float(predictions.eq(labels).float().mean().cpu()),
        float(oriented_margin(logits, labels, answer_position).mean().cpu()),
    )


@torch.inference_mode()
def collect_contrasts(
    model: Any,
    spec: Any,
    *,
    variant: str,
    split: str,
    lengths: Sequence[int],
    relative_start: int,
    relative_end: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
) -> list[ContrastRecord]:
    accumulators: dict[tuple[int, int], dict[str, Any]] = {}
    for length_index, length in enumerate(lengths):
        for batch_index in range(batches):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 10007 * length_index + batch_index)
            batch = generate_paper_batch(
                spec,
                batch_size=batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            labels = batch.targets[:, length]
            state = torch.zeros_like(model.read_in(batch.inputs))
            final_step = length + relative_end
            for output_step in range(1, final_step + 1):
                embeddings = model.input_embeddings(
                    batch.inputs, step_index=output_step
                )
                state, _ = step_state(
                    model,
                    state,
                    embeddings,
                    controller=controller,
                    controller_anchor=controller_anchor,
                    output_step=output_step,
                )
                relative_depth = output_step - length
                if not relative_start <= relative_depth <= relative_end:
                    continue
                answer = state[:, length].float()
                logits = model.decode(state).float()
                predicted = logits[:, length].argmax(dim=-1)
                margins = oriented_margin(logits, labels, length)
                key = (length, relative_depth)
                if key not in accumulators:
                    accumulators[key] = {
                        "class_sum": [
                            torch.zeros(answer.shape[-1], dtype=torch.float64),
                            torch.zeros(answer.shape[-1], dtype=torch.float64),
                        ],
                        "class_count": [0, 0],
                        "state_sum": torch.zeros(
                            answer.shape[-1], dtype=torch.float64
                        ),
                        "correct": 0.0,
                        "margin": 0.0,
                        "count": 0,
                    }
                acc = accumulators[key]
                answer_cpu = answer.cpu().double()
                labels_cpu = labels.cpu()
                for label in (0, 1):
                    selected = labels_cpu.eq(label)
                    acc["class_sum"][label] += answer_cpu[selected].sum(dim=0)
                    acc["class_count"][label] += int(selected.sum())
                acc["state_sum"] += answer_cpu.sum(dim=0)
                acc["correct"] += float(predicted.eq(labels).sum().cpu())
                acc["margin"] += float(margins.sum().cpu())
                acc["count"] += batch_size

    records: list[ContrastRecord] = []
    for (length, relative_depth), acc in sorted(accumulators.items()):
        if min(acc["class_count"]) == 0:
            raise RuntimeError("a parity class was absent from an evaluation cell")
        records.append(
            ContrastRecord(
                variant=variant,
                split=split,
                length=length,
                relative_depth=relative_depth,
                class0_mean=(
                    acc["class_sum"][0] / acc["class_count"][0]
                ).float(),
                class1_mean=(
                    acc["class_sum"][1] / acc["class_count"][1]
                ).float(),
                state_mean=(acc["state_sum"] / acc["count"]).float(),
                accuracy=acc["correct"] / acc["count"],
                margin=acc["margin"] / acc["count"],
                examples=acc["count"],
                class0_examples=acc["class_count"][0],
                class1_examples=acc["class_count"][1],
            )
        )
    return records


def grouped_contrasts(
    records: Sequence[ContrastRecord],
) -> dict[int, tuple[list[int], torch.Tensor]]:
    grouped: dict[int, list[ContrastRecord]] = {}
    for record in records:
        grouped.setdefault(record.length, []).append(record)
    output: dict[int, tuple[list[int], torch.Tensor]] = {}
    for length, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item.relative_depth)
        output[length] = (
            [row.relative_depth for row in rows],
            torch.stack([row.contrast for row in rows]),
        )
    return output


def harmonic_coefficients(
    depths: Sequence[int], values: torch.Tensor, *, period: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(depths, dtype=values.dtype)
    design = torch.stack(
        (
            torch.cos(2.0 * math.pi * x / period),
            torch.sin(2.0 * math.pi * x / period),
        ),
        dim=1,
    )
    centered = values - values.mean(dim=0, keepdim=True)
    coefficients = torch.linalg.lstsq(design, centered).solution
    fitted = design @ coefficients
    return coefficients[0], coefficients[1], fitted


def discover_phase_plane(records: Sequence[ContrastRecord]) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for depths, values in grouped_contrasts(records).values():
        cosine, sine, _ = harmonic_coefficients(depths, values, period=4.0)
        vectors.extend((cosine, sine))
    stacked = torch.stack(vectors)
    _, _, right_t = torch.linalg.svd(stacked, full_matrices=False)
    return right_t[:2].T.contiguous()


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (
        torch.linalg.vector_norm(left, dim=1)
        * torch.linalg.vector_norm(right, dim=1)
    ).clamp_min(1e-12)
    return float(((left * right).sum(dim=1) / denominator).mean())


def fit_phase_dynamics(
    records: Sequence[ContrastRecord], basis: torch.Tensor
) -> dict[str, Any]:
    x_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    transition_group_rows: list[torch.Tensor] = []
    full_centered: list[torch.Tensor] = []
    two_left: list[torch.Tensor] = []
    two_right: list[torch.Tensor] = []
    four_left: list[torch.Tensor] = []
    four_right: list[torch.Tensor] = []
    harmonic_summaries: dict[int, list[float]] = {p: [] for p in range(2, 9)}
    grouped_values = grouped_contrasts(records)
    for group_index, (depths, values) in enumerate(grouped_values.values()):
        centered = values - values.mean(dim=0, keepdim=True)
        full_centered.append(centered)
        # Fit the shared transition on uncentered coordinates with one affine
        # intercept per length.  This does not confuse a finite-window sample
        # mean with the center of a damped spiral.
        z = values @ basis
        x_rows.append(z[:-1])
        y_rows.append(z[1:])
        group_indicator = torch.zeros(len(z) - 1, len(grouped_values))
        group_indicator[:, group_index] = 1.0
        transition_group_rows.append(group_indicator)
        centered_z = centered @ basis
        if len(z) >= 3:
            two_left.append(centered_z[:-2])
            two_right.append(centered_z[2:])
        if len(z) >= 5:
            four_left.append(centered_z[:-4])
            four_right.append(centered_z[4:])
        denominator = float(centered.square().sum().clamp_min(1e-20))
        for period in harmonic_summaries:
            _, _, fitted = harmonic_coefficients(
                depths, values, period=float(period)
            )
            harmonic_summaries[period].append(
                float(fitted.square().sum()) / denominator
            )

    full = torch.cat(full_centered)
    z_full = full @ basis
    singular = torch.linalg.svdvals(full)
    plane_energy = float(z_full.square().sum() / full.square().sum())
    line_energy = float(z_full[:, 0].square().sum() / full.square().sum())
    intrinsic_rank1 = float(singular[0].square() / singular.square().sum())
    intrinsic_rank2 = float(singular[:2].square().sum() / singular.square().sum())

    x = torch.cat(x_rows)
    y = torch.cat(y_rows)
    group_design = torch.cat(transition_group_rows)
    affine_design = torch.cat((x, group_design), dim=1)
    affine_solution = torch.linalg.lstsq(affine_design, y).solution
    transition = affine_solution[:2]
    prediction = affine_design @ affine_solution
    transition_r2 = 1.0 - float(
        (y - prediction).square().sum()
        / (y - y.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-20)
    )
    left, transition_singular, right_t = torch.linalg.svd(transition)
    polar_rotation = left @ right_t
    signed_angle = math.atan2(
        float(polar_rotation[0, 1]), float(polar_rotation[0, 0])
    )
    angle = abs(signed_angle)
    eigenvalues = torch.linalg.eigvals(transition)
    positive_imag = eigenvalues[eigenvalues.imag.argmax()]
    eigen_angle = float(torch.angle(positive_imag))
    if eigen_angle < 0:
        eigen_angle += 2.0 * math.pi
    eigen_modulus = float(torch.abs(positive_imag))

    two_l = torch.cat(two_left)
    two_r = torch.cat(two_right)
    four_l = torch.cat(four_left)
    four_r = torch.cat(four_right)
    alpha_two = -float((two_l * two_r).sum() / two_l.square().sum())
    alpha_four = float((four_l * four_r).sum() / four_l.square().sum())
    two_error = float(
        torch.linalg.vector_norm(two_r + alpha_two * two_l)
        / torch.linalg.vector_norm(two_r).clamp_min(1e-20)
    )
    four_error = float(
        torch.linalg.vector_norm(four_r - alpha_four * four_l)
        / torch.linalg.vector_norm(four_r).clamp_min(1e-20)
    )
    return {
        "shared_phase_line_energy_fraction": line_energy,
        "shared_phase_plane_energy_fraction": plane_energy,
        "unconstrained_rank1_energy_fraction": intrinsic_rank1,
        "unconstrained_rank2_energy_fraction": intrinsic_rank2,
        "transition_matrix": transition.tolist(),
        "transition_singular_values": transition_singular.tolist(),
        "transition_r2": transition_r2,
        "polar_rotation_signed_angle_radians": signed_angle,
        "polar_rotation_angle_radians": angle,
        "polar_rotation_period_calls": 2.0 * math.pi / angle,
        "complex_eigen_angle_radians": eigen_angle,
        "complex_eigen_period_calls": 2.0 * math.pi / eigen_angle,
        "complex_eigen_modulus": eigen_modulus,
        "two_step_antialignment_cosine": cosine_mean(two_l, -two_r),
        "four_step_alignment_cosine": cosine_mean(four_l, four_r),
        "best_two_step_negative_scale": alpha_two,
        "best_four_step_positive_scale": alpha_four,
        "scaled_two_step_relative_error": two_error,
        "scaled_four_step_relative_error": four_error,
        "harmonic_energy_fraction_by_period": {
            str(period): float(np.mean(values))
            for period, values in harmonic_summaries.items()
        },
    }


def record_rows(
    records: Sequence[ContrastRecord], bases: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = grouped_contrasts(records)
    centers = {
        length: values.mean(dim=0) for length, (_, values) in grouped.items()
    }
    for record in records:
        basis = bases[record.variant]
        centered = record.contrast - centers[record.length]
        coordinate = centered @ basis
        rows.append(
            {
                "variant": record.variant,
                "split": record.split,
                "length": record.length,
                "relative_depth": record.relative_depth,
                "accuracy": record.accuracy,
                "oriented_margin": record.margin,
                "contrast_norm": float(torch.linalg.vector_norm(record.contrast)),
                "centered_contrast_norm": float(torch.linalg.vector_norm(centered)),
                "phase_x": float(coordinate[0]),
                "phase_y": float(coordinate[1]),
                "phase_radius": float(torch.linalg.vector_norm(coordinate)),
                "phase_angle": float(math.atan2(float(coordinate[1]), float(coordinate[0]))),
                "examples": record.examples,
                "class0_examples": record.class0_examples,
                "class1_examples": record.class1_examples,
            }
        )
    return rows


def lengthwise_dynamics_rows(
    records: Sequence[ContrastRecord], bases: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[ContrastRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.variant, record.split, record.length), []
        ).append(record)
    rows: list[dict[str, Any]] = []
    for (variant, split, length), selected in sorted(grouped.items()):
        metrics = fit_phase_dynamics(selected, bases[variant])
        rows.append(
            {
                "variant": variant,
                "split": split,
                "length": length,
                "shared_phase_plane_energy_fraction": metrics[
                    "shared_phase_plane_energy_fraction"
                ],
                "unconstrained_rank1_energy_fraction": metrics[
                    "unconstrained_rank1_energy_fraction"
                ],
                "unconstrained_rank2_energy_fraction": metrics[
                    "unconstrained_rank2_energy_fraction"
                ],
                "transition_r2": metrics["transition_r2"],
                "rotation_angle_radians": metrics[
                    "polar_rotation_angle_radians"
                ],
                "rotation_period_calls": metrics["polar_rotation_period_calls"],
                "complex_eigen_period_calls": metrics[
                    "complex_eigen_period_calls"
                ],
                "complex_eigen_modulus": metrics["complex_eigen_modulus"],
                "two_step_antialignment_cosine": metrics[
                    "two_step_antialignment_cosine"
                ],
                "four_step_alignment_cosine": metrics[
                    "four_step_alignment_cosine"
                ],
                "best_four_step_positive_scale": metrics[
                    "best_four_step_positive_scale"
                ],
                "scaled_four_step_relative_error": metrics[
                    "scaled_four_step_relative_error"
                ],
                "period4_harmonic_energy_fraction": metrics[
                    "harmonic_energy_fraction_by_period"
                ]["4"],
            }
        )
    return rows


def phase_origin_locking_summary(
    centroid_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    variants = sorted({str(row["variant"]) for row in centroid_rows})
    for variant in variants:
        selected = sorted(
            [
                row
                for row in centroid_rows
                if row["variant"] == variant
                and row["split"] == "evaluation"
                and int(row["relative_depth"]) == 0
            ],
            key=lambda item: int(item["length"]),
        )
        angles = np.array([float(row["phase_angle"]) for row in selected])
        lengths = np.array([float(row["length"]) for row in selected])
        resultant = np.mean(np.exp(1j * angles))
        unwrapped = np.unwrap(angles)
        design = np.stack((lengths, np.ones_like(lengths)), axis=1)
        slope, intercept = np.linalg.lstsq(design, unwrapped, rcond=None)[0]
        prediction = slope * lengths + intercept
        denominator = np.square(unwrapped - unwrapped.mean()).sum()
        r2 = (
            1.0 - float(np.square(unwrapped - prediction).sum() / denominator)
            if denominator > 1e-20
            else 1.0
        )
        concentration = abs(resultant)
        output.append(
            {
                "variant": variant,
                "evaluation_lengths": " ".join(str(int(value)) for value in lengths),
                "phase_origin_angles_radians": " ".join(
                    f"{value:.9f}" for value in angles
                ),
                "circular_resultant_length": float(concentration),
                "circular_standard_deviation": float(
                    math.sqrt(max(0.0, -2.0 * math.log(max(concentration, 1e-20))))
                ),
                "circular_mean_angle_radians": float(np.angle(resultant)),
                "unwrapped_angle_slope_per_length": float(slope),
                "unwrapped_linear_fit_r2": r2,
            }
        )
    return output


def readout_alignment(model: Any, basis: torch.Tensor) -> dict[str, float]:
    direction = (
        model.read_out.weight[1].detach().float().cpu()
        - model.read_out.weight[0].detach().float().cpu()
    )
    projection = basis @ (basis.T @ direction)
    return {
        "readout_direction_in_phase_plane_energy_fraction": float(
            projection.square().sum() / direction.square().sum()
        ),
        "readout_direction_norm": float(torch.linalg.vector_norm(direction)),
    }


@torch.inference_mode()
def natural_states(
    model: Any,
    inputs: torch.Tensor,
    *,
    steps: int,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
) -> list[torch.Tensor]:
    state = torch.zeros_like(model.read_in(inputs))
    states: list[torch.Tensor] = []
    for output_step in range(1, steps + 1):
        embeddings = model.input_embeddings(inputs, step_index=output_step)
        state, _ = step_state(
            model,
            state,
            embeddings,
            controller=controller,
            controller_anchor=controller_anchor,
            output_step=output_step,
        )
        states.append(state)
    return states


@torch.inference_mode()
def continue_state(
    model: Any,
    state: torch.Tensor,
    inputs: torch.Tensor,
    *,
    start_step: int,
    calls: int,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
) -> torch.Tensor:
    for output_step in range(start_step + 1, start_step + calls + 1):
        embeddings = model.input_embeddings(inputs, step_index=output_step)
        state, _ = step_state(
            model,
            state,
            embeddings,
            controller=controller,
            controller_anchor=controller_anchor,
            output_step=output_step,
        )
    return state


def state_phase_coordinate(
    state: torch.Tensor,
    *,
    position: int,
    labels: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    answer = state[:, position].float()
    contrast = answer[labels == 1].mean(0) - answer[labels == 0].mean(0)
    return contrast.cpu() @ basis


@torch.inference_mode()
def run_causal_interchange(
    model: Any,
    spec: Any,
    *,
    variant: str,
    lengths: Sequence[int],
    basis: torch.Tensor,
    batch_size: int,
    seed: int,
    random_controls: int,
    device: torch.device,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    readout_direction = (
        model.read_out.weight[1].detach().float().cpu()
        - model.read_out.weight[0].detach().float().cpu()
    )
    readout_basis = (readout_direction / readout_direction.norm()).unsqueeze(1)
    control_planes = [
        random_plane(
            basis.shape[0],
            seed=seed + 1000 + index,
            exclude=basis,
        )
        for index in range(random_controls)
    ]
    for length_index, length in enumerate(lengths):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 10007 * length_index)
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=length,
            max_length=length,
            fixed_length=length,
            generator=generator,
        ).to(device)
        labels = batch.targets[:, length]
        states = natural_states(
            model,
            batch.inputs,
            steps=length + 8,
            controller=controller,
            controller_anchor=controller_anchor,
        )
        receiver = states[length - 1]
        for donor_offset in (1, 2, 3, 4):
            donor = states[length + donor_offset - 1]
            conditions: list[tuple[str, torch.Tensor]] = [
                ("receiver", receiver),
                ("full_answer", phase_patch(receiver, donor, torch.eye(basis.shape[0]), positions=length)),
                ("full_all", donor),
                ("phase_answer", phase_patch(receiver, donor, basis, positions=length)),
                ("phase_all", phase_patch(receiver, donor, basis, positions=slice(None))),
                ("complement_answer", complement_patch(receiver, donor, basis, positions=length)),
                ("readout_answer", phase_patch(receiver, donor, readout_basis, positions=length)),
                ("phase_remove_answer", remove_plane_component(receiver, basis, position=length)),
                ("phase_keep_answer", keep_plane_component(receiver, basis, position=length)),
                (
                    "phase_rotate_pi_over_2_answer",
                    rotate_plane_component(
                        receiver,
                        basis,
                        angle_radians=math.pi / 2.0,
                        position=length,
                    ),
                ),
                (
                    "phase_rotate_pi_answer",
                    rotate_plane_component(
                        receiver,
                        basis,
                        angle_radians=math.pi,
                        position=length,
                    ),
                ),
                (
                    "phase_rotate_3pi_over_2_answer",
                    rotate_plane_component(
                        receiver,
                        basis,
                        angle_radians=3.0 * math.pi / 2.0,
                        position=length,
                    ),
                ),
            ]
            for control_index, control_basis in enumerate(control_planes):
                conditions.append(
                    (
                        f"random2d_equalnorm_answer_{control_index}",
                        magnitude_matched_random_patch(
                            receiver,
                            donor,
                            basis,
                            control_basis,
                            position=length,
                        ),
                    )
                )
                conditions.append(
                    (
                        f"random2d_rotation_pi_answer_{control_index}",
                        magnitude_matched_random_rotation(
                            receiver,
                            basis,
                            control_basis,
                            angle_radians=math.pi,
                            position=length,
                        ),
                    )
                )
            for condition, initial in conditions:
                live = initial
                for continuation_calls in range(0, 5):
                    if continuation_calls:
                        live = continue_state(
                            model,
                            live,
                            batch.inputs,
                            start_step=length + continuation_calls - 1,
                            calls=1,
                            controller=controller,
                            controller_anchor=controller_anchor,
                        )
                    accuracy, margin = score_state(model, live, labels, length)
                    target = states[
                        length + donor_offset + continuation_calls - 1
                    ]
                    base = states[length + continuation_calls - 1]
                    target_accuracy, target_margin = score_state(
                        model, target, labels, length
                    )
                    base_accuracy, base_margin = score_state(
                        model, base, labels, length
                    )
                    coordinate = state_phase_coordinate(
                        live,
                        position=length,
                        labels=labels,
                        basis=basis,
                    )
                    target_coordinate = state_phase_coordinate(
                        target,
                        position=length,
                        labels=labels,
                        basis=basis,
                    )
                    base_coordinate = state_phase_coordinate(
                        base,
                        position=length,
                        labels=labels,
                        basis=basis,
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "length": length,
                            "donor_offset": donor_offset,
                            "continuation_calls": continuation_calls,
                            "condition": condition,
                            "accuracy": accuracy,
                            "oriented_margin": margin,
                            "target_accuracy": target_accuracy,
                            "target_margin": target_margin,
                            "base_accuracy": base_accuracy,
                            "base_margin": base_margin,
                            "absolute_margin_error_to_target": abs(margin - target_margin),
                            "base_absolute_margin_error_to_target": abs(base_margin - target_margin),
                            "phase_x": float(coordinate[0]),
                            "phase_y": float(coordinate[1]),
                            "target_phase_x": float(target_coordinate[0]),
                            "target_phase_y": float(target_coordinate[1]),
                            "base_phase_x": float(base_coordinate[0]),
                            "base_phase_y": float(base_coordinate[1]),
                            "phase_error_to_target": float(
                                torch.linalg.vector_norm(
                                    coordinate - target_coordinate
                                )
                            ),
                            "base_phase_error_to_target": float(
                                torch.linalg.vector_norm(
                                    base_coordinate - target_coordinate
                                )
                            ),
                            "examples": batch_size,
                            "evaluation_seed": seed + 10007 * length_index,
                        }
                    )
    return rows


def summarize_causal(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition.startswith("random2d_equalnorm_answer_"):
            condition = "random2d_equalnorm_answer"
        elif condition.startswith("random2d_rotation_pi_answer_"):
            condition = "random2d_rotation_pi_answer"
        continuation_group = (
            "continued" if int(row["continuation_calls"]) >= 1 else "immediate"
        )
        for donor_group in ("all", str(int(row["donor_offset"]))):
            grouped.setdefault(
                (str(row["variant"]), condition, continuation_group + "/" + donor_group),
                [],
            ).append(row)
    output: list[dict[str, Any]] = []
    for (variant, condition, continuation_donor), selected in sorted(grouped.items()):
        continuation_group, donor_group = continuation_donor.split("/", 1)
        margin_sse = sum(
            (float(row["oriented_margin"]) - float(row["target_margin"])) ** 2
            for row in selected
        )
        base_margin_sse = sum(
            (float(row["base_margin"]) - float(row["target_margin"])) ** 2
            for row in selected
        )
        phase_sse = sum(float(row["phase_error_to_target"]) ** 2 for row in selected)
        base_phase_sse = sum(
            float(row["base_phase_error_to_target"]) ** 2 for row in selected
        )
        output.append(
            {
                "variant": variant,
                "condition": condition,
                "continuation_group": continuation_group,
                "donor_offset": donor_group,
                "cells": len(selected),
                "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in selected])),
                "mean_target_accuracy": float(np.mean([float(row["target_accuracy"]) for row in selected])),
                "mean_absolute_margin_error_to_target": float(
                    np.mean(
                        [float(row["absolute_margin_error_to_target"]) for row in selected]
                    )
                ),
                "margin_translation_recovery": 1.0 - margin_sse / max(base_margin_sse, 1e-20),
                "phase_translation_recovery": 1.0 - phase_sse / max(base_phase_sse, 1e-20),
            }
        )
    return output


@torch.inference_mode()
def run_branch_skip_hidden_lag(
    model: Any,
    spec: Any,
    *,
    branch: str,
    lengths: Sequence[int],
    basis: torch.Tensor,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if branch not in {"attention", "mlp"}:
        raise ValueError("branch must be attention or mlp")
    rows: list[dict[str, Any]] = []
    for length_index, length in enumerate(lengths):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 20011 * length_index)
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=length,
            max_length=length,
            fixed_length=length,
            generator=generator,
        ).to(device)
        labels = batch.targets[:, length]
        clean = natural_states(
            model,
            batch.inputs,
            steps=length + 2,
            controller=None,
            controller_anchor=None,
        )
        skip_steps = sorted({1, max(1, length // 4), max(1, length // 2), max(1, 3 * length // 4), length})
        for skipped_step in skip_steps:
            state = torch.zeros_like(model.read_in(batch.inputs))
            skipped_states: list[torch.Tensor] = []
            for output_step in range(1, length + 3):
                embeddings = model.input_embeddings(
                    batch.inputs, step_index=output_step
                )
                state, _ = step_state(
                    model,
                    state,
                    embeddings,
                    controller=None,
                    controller_anchor=None,
                    output_step=output_step,
                    skip_mlp=(branch == "mlp" and output_step == skipped_step),
                    skip_attention=(
                        branch == "attention" and output_step == skipped_step
                    ),
                )
                skipped_states.append(state)
            for eval_step in (length, length + 1):
                skipped = skipped_states[eval_step - 1]
                clean_same = clean[eval_step - 1]
                clean_lagged = clean[eval_step - 2]
                skipped_coordinate = state_phase_coordinate(
                    skipped, position=length, labels=labels, basis=basis
                )
                same_coordinate = state_phase_coordinate(
                    clean_same, position=length, labels=labels, basis=basis
                )
                lagged_coordinate = state_phase_coordinate(
                    clean_lagged, position=length, labels=labels, basis=basis
                )
                accuracy, margin = score_state(model, skipped, labels, length)
                rows.append(
                    {
                        "length": length,
                        "skipped_branch": branch,
                        f"skipped_{branch}_step": skipped_step,
                        "evaluation_step": eval_step,
                        "accuracy": accuracy,
                        "oriented_margin": margin,
                        "phase_distance_to_clean_same_step": float(
                            torch.linalg.vector_norm(skipped_coordinate - same_coordinate)
                        ),
                        "phase_distance_to_clean_previous_step": float(
                            torch.linalg.vector_norm(skipped_coordinate - lagged_coordinate)
                        ),
                        "closer_to_previous_phase": float(
                            torch.linalg.vector_norm(skipped_coordinate - lagged_coordinate)
                            < torch.linalg.vector_norm(skipped_coordinate - same_coordinate)
                        ),
                        "examples": batch_size,
                        "evaluation_seed": seed + 20011 * length_index,
                    }
                )
    return rows


def run_mlp_skip_hidden_lag(
    model: Any,
    spec: Any,
    *,
    lengths: Sequence[int],
    basis: torch.Tensor,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    return run_branch_skip_hidden_lag(
        model, spec, branch="mlp", lengths=lengths, basis=basis,
        batch_size=batch_size, seed=seed, device=device,
    )


def run_attention_skip_hidden_lag(
    model: Any,
    spec: Any,
    *,
    lengths: Sequence[int],
    basis: torch.Tensor,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    return run_branch_skip_hidden_lag(
        model, spec, branch="attention", lengths=lengths, basis=basis,
        batch_size=batch_size, seed=seed, device=device,
    )


def summarize_skip_hidden_lag(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty branch-skip analysis")
    return {
        "rows": len(rows),
        "fraction_closer_to_clean_previous_phase": float(
            np.mean([float(row["closer_to_previous_phase"]) for row in rows])
        ),
        "mean_distance_to_same": float(
            np.mean([float(row["phase_distance_to_clean_same_step"]) for row in rows])
        ),
        "mean_distance_to_previous": float(
            np.mean([float(row["phase_distance_to_clean_previous_step"]) for row in rows])
        ),
    }


@torch.inference_mode()
def run_component_phase(
    model: Any,
    spec: Any,
    *,
    variant: str,
    lengths: Sequence[int],
    basis: torch.Tensor,
    batch_size: int,
    seed: int,
    device: torch.device,
    controller: torch.nn.Module | None,
    controller_anchor: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_index, length in enumerate(lengths):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 30011 * length_index)
        batch = generate_paper_batch(
            spec,
            batch_size=batch_size,
            min_length=length,
            max_length=length,
            fixed_length=length,
            generator=generator,
        ).to(device)
        labels = batch.targets[:, length]
        state = torch.zeros_like(model.read_in(batch.inputs))
        for output_step in range(1, length + 5):
            embeddings = model.input_embeddings(
                batch.inputs, step_index=output_step
            )
            state, components = step_state(
                model,
                state,
                embeddings,
                controller=controller,
                controller_anchor=controller_anchor,
                output_step=output_step,
            )
            relative_depth = output_step - length
            if relative_depth not in (0, 1, 2, 3, 4):
                continue
            for component_name in (
                "source",
                "controlled_source",
                "pre_attention",
                "post_attention",
                "pre_final_norm",
                "final_state",
            ):
                coordinate = state_phase_coordinate(
                    components[component_name],
                    position=length,
                    labels=labels,
                    basis=basis,
                )
                rows.append(
                    {
                        "variant": variant,
                        "length": length,
                        "output_step": output_step,
                        "relative_depth": relative_depth,
                        "component": component_name,
                        "phase_x": float(coordinate[0]),
                        "phase_y": float(coordinate[1]),
                        "phase_radius": float(torch.linalg.vector_norm(coordinate)),
                        "phase_angle": math.atan2(float(coordinate[1]), float(coordinate[0])),
                        "examples": batch_size,
                    }
                )
    return rows


def summarize_component_progress(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    for row in rows:
        key = (
            str(row["variant"]),
            int(row["length"]),
            int(row["relative_depth"]),
        )
        grouped.setdefault(key, {})[str(row["component"])] = np.array(
            [float(row["phase_x"]), float(row["phase_y"])], dtype=np.float64
        )
    stage_pairs = (
        ("controller_J", "source", "controlled_source"),
        ("input_reinjection", "controlled_source", "pre_attention"),
        ("attention", "pre_attention", "post_attention"),
        ("MLP", "post_attention", "pre_final_norm"),
        ("final_LayerNorm", "pre_final_norm", "final_state"),
    )
    accumulators: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (variant, _length, _depth), components in grouped.items():
        if not all(
            name in components
            for name in (
                "source",
                "controlled_source",
                "pre_attention",
                "post_attention",
                "pre_final_norm",
                "final_state",
            )
        ):
            continue
        total = components["final_state"] - components["source"]
        total_squared = float(total @ total)
        if total_squared < 1e-20:
            continue
        total_norm = math.sqrt(total_squared)
        for stage, before, after in stage_pairs:
            delta = components[after] - components[before]
            signed_progress = float(delta @ total) / total_squared
            relative_norm = float(np.linalg.norm(delta)) / total_norm
            accumulators.setdefault((variant, stage), []).append(
                (signed_progress, relative_norm)
            )
    output: list[dict[str, Any]] = []
    for (variant, stage), values in sorted(accumulators.items()):
        progress = np.array([value[0] for value in values])
        norms = np.array([value[1] for value in values])
        output.append(
            {
                "variant": variant,
                "stage": stage,
                "cells": len(values),
                "mean_signed_progress_along_full_phase_step": float(
                    progress.mean()
                ),
                "median_signed_progress_along_full_phase_step": float(
                    np.median(progress)
                ),
                "mean_stage_delta_to_full_phase_step_norm": float(norms.mean()),
            }
        )
    return output


def plot_orbits(
    rows: Sequence[dict[str, Any]], out_dir: Path
) -> None:
    variants = sorted({str(row["variant"]) for row in rows})
    figure, axes = plt.subplots(1, len(variants), figsize=(7.2 * len(variants), 6.2), squeeze=False)
    for axis, variant in zip(axes[0], variants, strict=True):
        selected_lengths = [
            length
            for length in (14, 22, 48, 100)
            if any(
                int(row["length"]) == length
                and row["variant"] == variant
                and row["split"] == "evaluation"
                for row in rows
            )
        ]
        if not selected_lengths:
            selected_lengths = sorted(
                {
                    int(row["length"])
                    for row in rows
                    if row["variant"] == variant and row["split"] == "evaluation"
                }
            )[-4:]
        for length in selected_lengths:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and int(row["length"]) == length
                    and row["split"] == "evaluation"
                ],
                key=lambda item: int(item["relative_depth"]),
            )
            if not selected:
                continue
            x = np.array([float(row["phase_x"]) for row in selected])
            y = np.array([float(row["phase_y"]) for row in selected])
            axis.plot(x, y, marker="o", label=f"n={length}")
            for index in range(min(5, len(selected))):
                axis.annotate(
                    f"d={selected[index]['relative_depth']}",
                    (x[index], y[index]),
                    fontsize=8,
                )
        axis.axhline(0, color="#bbbbbb", linewidth=0.8)
        axis.axvline(0, color="#bbbbbb", linewidth=0.8)
        axis.set_title(f"{variant}: held-out parity-contrast orbit")
        axis.set_xlabel("phase coordinate 1")
        axis.set_ylabel("phase coordinate 2")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out_dir / "heldout_phase_orbits.png", dpi=190)
    plt.close(figure)


def plot_behavior_and_causal(
    centroid_rows: Sequence[dict[str, Any]],
    causal_rows: Sequence[dict[str, Any]],
    out_dir: Path,
) -> None:
    variants = sorted({str(row["variant"]) for row in centroid_rows})
    figure, axes = plt.subplots(2, len(variants), figsize=(7.0 * len(variants), 9.5), squeeze=False)
    for column, variant in enumerate(variants):
        axis = axes[0, column]
        for length in (20, 32, 64, 100):
            selected = sorted(
                [
                    row
                    for row in centroid_rows
                    if row["variant"] == variant
                    and row["split"] == "evaluation"
                    and int(row["length"]) == length
                ],
                key=lambda item: int(item["relative_depth"]),
            )
            if selected:
                axis.plot(
                    [int(row["relative_depth"]) for row in selected],
                    [float(row["oriented_margin"]) for row in selected],
                    marker="o",
                    label=f"n={length}",
                )
        if not axis.lines:
            available = sorted(
                {
                    int(row["length"])
                    for row in centroid_rows
                    if row["variant"] == variant and row["split"] == "evaluation"
                }
            )[-4:]
            for length in available:
                selected = sorted(
                    [
                        row
                        for row in centroid_rows
                        if row["variant"] == variant
                        and row["split"] == "evaluation"
                        and int(row["length"]) == length
                    ],
                    key=lambda item: int(item["relative_depth"]),
                )
                axis.plot(
                    [int(row["relative_depth"]) for row in selected],
                    [float(row["oriented_margin"]) for row in selected],
                    marker="o",
                    label=f"n={length}",
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{variant}: local readout margin")
        axis.set_xlabel("relative depth d=t-n")
        axis.set_ylabel("correct-minus-opposite margin")
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)

        axis = axes[1, column]
        selected = [
            row
            for row in causal_rows
            if row["variant"] == variant
            and int(row["donor_offset"]) == 2
            and int(row["length"]) == (20 if any(int(item["length"]) == 20 and item["variant"] == variant for item in causal_rows) else min(int(item["length"]) for item in causal_rows if item["variant"] == variant))
        ]
        plotted = {
            "receiver": "unpatched d=0",
            "phase_answer": "2D phase patch",
            "readout_answer": "readout-1D patch",
            "full_all": "full-state donor",
        }
        for condition, label in plotted.items():
            condition_rows = sorted(
                [row for row in selected if row["condition"] == condition],
                key=lambda item: int(item["continuation_calls"]),
            )
            if condition_rows:
                axis.plot(
                    [int(row["continuation_calls"]) for row in condition_rows],
                    [float(row["oriented_margin"]) for row in condition_rows],
                    marker="o",
                    label=label,
                )
        target_rows = sorted(
            [row for row in selected if row["condition"] == "receiver"],
            key=lambda item: int(item["continuation_calls"]),
        )
        if target_rows:
            axis.plot(
                [int(row["continuation_calls"]) for row in target_rows],
                [float(row["target_margin"]) for row in target_rows],
                color="black",
                linestyle="--",
                marker="x",
                label="natural d=2+j target",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{variant}: d=0 receives d=2 phase")
        axis.set_xlabel("calls after intervention")
        axis.set_ylabel("correct-minus-opposite margin")
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / "behavior_and_phase_interchange.png", dpi=190)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        {
            "status": "running",
            "analysis": "parity_hidden_four_phase_test",
            "pid": os.getpid(),
            "started_at_unix": started,
            "checkpoint": str(args.checkpoint),
            "controller": None if args.controller is None else str(args.controller),
            "paper_mode": bool(args.paper_mode),
            "requested_device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "output_dir": str(args.out_dir),
        },
    )
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(
        args.checkpoint, device=device, paper_mode=args.paper_mode
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if spec.name != "parity":
        raise ValueError("this analysis is fixed to Parity")
    if len(model.layers) != 1 or model.config.d_model != 256 or model.config.n_heads != 64:
        raise ValueError("checkpoint does not match the authoritative Parity architecture")

    controller = None
    controller_payload = None
    controller_anchor = None
    if args.controller is not None:
        controller, controller_payload = load_controller(args.controller, device=device)
        controller.eval()
        for parameter in controller.parameters():
            parameter.requires_grad_(False)
        controller_anchor = int(controller_payload["anchor_step"])
        if controller_anchor != 1:
            raise ValueError("controlled comparison expects anchor_step=1")

    if args.smoke:
        discovery_lengths = (12, 20)
        evaluation_lengths = (14, 22)
        causal_lengths = (14,)
        relative_end = 6
        batch_size = min(args.batch_size, 64)
        batches = 1
        causal_batch_size = min(args.causal_batch_size, 64)
        random_controls = min(args.random_controls, 2)
    else:
        discovery_lengths = tuple(args.discovery_lengths)
        evaluation_lengths = tuple(args.evaluation_lengths)
        causal_lengths = tuple(args.causal_lengths)
        relative_end = args.relative_end
        batch_size = args.batch_size
        batches = args.batches
        causal_batch_size = args.causal_batch_size
        random_controls = args.random_controls
    if set(discovery_lengths) & set(evaluation_lengths):
        raise ValueError("discovery and evaluation lengths must be disjoint")
    if set(discovery_lengths) & set(causal_lengths):
        raise ValueError("discovery and causal-intervention lengths must be disjoint")
    if args.relative_start > relative_end or relative_end < 4:
        raise ValueError("the relative-depth window must contain a four-step comparison")

    variants: list[tuple[str, torch.nn.Module | None, int | None]] = [
        ("raw", None, None)
    ]
    if controller is not None:
        variants.append(("controlled", controller, controller_anchor))

    all_records: list[ContrastRecord] = []
    bases: dict[str, torch.Tensor] = {}
    dynamics: dict[str, dict[str, Any]] = {}
    for variant, live_controller, live_anchor in variants:
        discovery = collect_contrasts(
            model,
            spec,
            variant=variant,
            split="discovery",
            lengths=discovery_lengths,
            relative_start=args.relative_start,
            relative_end=relative_end,
            batch_size=batch_size,
            batches=batches,
            seed=args.discovery_seed,
            device=device,
            controller=live_controller,
            controller_anchor=live_anchor,
        )
        evaluation = collect_contrasts(
            model,
            spec,
            variant=variant,
            split="evaluation",
            lengths=evaluation_lengths,
            relative_start=args.relative_start,
            relative_end=relative_end,
            batch_size=batch_size,
            batches=batches,
            seed=args.evaluation_seed,
            device=device,
            controller=live_controller,
            controller_anchor=live_anchor,
        )
        basis = discover_phase_plane(discovery)
        bases[variant] = basis
        dynamics[variant] = {
            "discovery": fit_phase_dynamics(discovery, basis),
            "heldout_evaluation": {
                **fit_phase_dynamics(evaluation, basis),
                **readout_alignment(model, basis),
            },
        }
        all_records.extend(discovery)
        all_records.extend(evaluation)

    centroid_rows = record_rows(all_records, bases)
    write_csv(args.out_dir / "centroid_phase_trajectory.csv", centroid_rows)
    lengthwise_rows = lengthwise_dynamics_rows(all_records, bases)
    write_csv(args.out_dir / "lengthwise_phase_dynamics.csv", lengthwise_rows)
    phase_origin_rows = phase_origin_locking_summary(centroid_rows)
    write_csv(args.out_dir / "phase_origin_locking.csv", phase_origin_rows)

    causal_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for variant, live_controller, live_anchor in variants:
        causal_rows.extend(
            run_causal_interchange(
                model,
                spec,
                variant=variant,
                lengths=causal_lengths,
                basis=bases[variant],
                batch_size=causal_batch_size,
                seed=args.causal_seed,
                random_controls=random_controls,
                device=device,
                controller=live_controller,
                controller_anchor=live_anchor,
            )
        )
        component_rows.extend(
            run_component_phase(
                model,
                spec,
                variant=variant,
                lengths=causal_lengths,
                basis=bases[variant],
                batch_size=causal_batch_size,
                seed=args.causal_seed + 401,
                device=device,
                controller=live_controller,
                controller_anchor=live_anchor,
            )
        )
    causal_summary = summarize_causal(causal_rows)
    write_csv(args.out_dir / "causal_phase_interchange.csv", causal_rows)
    write_csv(args.out_dir / "causal_phase_interchange_summary.csv", causal_summary)
    write_csv(args.out_dir / "component_phase_trajectory.csv", component_rows)
    component_progress_rows = summarize_component_progress(component_rows)
    write_csv(args.out_dir / "component_phase_progress.csv", component_progress_rows)

    skip_lengths = tuple(length for length in causal_lengths if length <= 20)
    if not skip_lengths:
        skip_lengths = (min(causal_lengths),)
    mlp_skip_rows = run_mlp_skip_hidden_lag(
        model,
        spec,
        lengths=skip_lengths,
        basis=bases["raw"],
        batch_size=causal_batch_size,
        seed=args.causal_seed + 809,
        device=device,
    )
    write_csv(args.out_dir / "mlp_skip_hidden_phase_lag.csv", mlp_skip_rows)
    attention_skip_rows = run_attention_skip_hidden_lag(
        model,
        spec,
        lengths=skip_lengths,
        basis=bases["raw"],
        batch_size=causal_batch_size,
        seed=args.causal_seed + 911,
        device=device,
    )
    write_csv(
        args.out_dir / "attention_skip_hidden_phase_lag.csv", attention_skip_rows
    )

    basis_rows = []
    for variant, basis in bases.items():
        for coordinate in range(2):
            for dimension, value in enumerate(basis[:, coordinate]):
                basis_rows.append(
                    {
                        "variant": variant,
                        "phase_coordinate": coordinate + 1,
                        "hidden_dimension": dimension,
                        "weight": float(value),
                    }
                )
    write_csv(args.out_dir / "phase_basis.csv", basis_rows)

    cross_variant: dict[str, Any] | None = None
    if "controlled" in bases:
        singular = torch.linalg.svdvals(bases["raw"].T @ bases["controlled"])
        cross_variant = {
            "subspace_cosines": singular.tolist(),
            "principal_angles_degrees": [
                math.degrees(math.acos(min(1.0, max(-1.0, float(value)))))
                for value in singular
            ],
        }

    plot_orbits(centroid_rows, args.out_dir)
    plot_behavior_and_causal(centroid_rows, causal_rows, args.out_dir)

    skip_summary = summarize_skip_hidden_lag(mlp_skip_rows)
    attention_skip_summary = summarize_skip_hidden_lag(attention_skip_rows)

    summary = {
        "status": "complete",
        "analysis": "parity_hidden_four_phase_test",
        "evidence_mode": "claim-oriented bounded causal test on one frozen backbone seed",
        "paper_mode": bool(args.paper_mode),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": int(backbone_payload["step"]),
        "backbone_seed": int(backbone_payload["seed"]),
        "controller": None if args.controller is None else str(args.controller),
        "controller_anchor": controller_anchor,
        "controller_metadata": (
            None
            if controller_payload is None
            else {
                "backbone_checkpoint_named_by_controller": controller_payload.get(
                    "checkpoint"
                ),
                "controller_seed": controller_payload.get("seed"),
                "loss": controller_payload.get("loss"),
                "state_loss_weight": controller_payload.get("state_loss_weight"),
                "trained_logical_lengths": controller_payload.get(
                    "controller_sampled_logical_lengths"
                ),
            }
        ),
        "architecture": {
            "task": spec.name,
            "d_model": model.config.d_model,
            "attention_heads": model.config.n_heads,
            "mlp_width": model.config.d_mlp,
            "shared_physical_layers": len(model.layers),
            "token_embedding_injection": model.config.token_embedding_injection,
            "position_embedding": model.config.position_embedding,
            "position_injection": model.config.position_injection,
            "final_layer_norm": "inside every recurrent call",
            "loss_placement": "answer CE only at registered T(n)=n",
            "trained_lengths": [1, spec.train_max_length],
        },
        "protocol": {
            "primary_state": "post-call final-LayerNorm answer-token residual",
            "primary_signal": "class1-minus-class0 hidden-state centroid",
            "phase_plane_discovery": "top two hidden directions of period-4 Fourier coefficients on discovery lengths",
            "discovery_lengths": list(discovery_lengths),
            "evaluation_lengths": list(evaluation_lengths),
            "causal_lengths": list(causal_lengths),
            "relative_depth_window": [args.relative_start, relative_end],
            "examples_per_centroid_length": batch_size * batches,
            "causal_examples_per_length": causal_batch_size,
            "random_equal_norm_controls": random_controls,
            "discovery_seed": args.discovery_seed,
            "evaluation_seed": args.evaluation_seed,
            "causal_seed": args.causal_seed,
        },
        "dynamics": dynamics,
        "cross_variant_phase_plane": cross_variant,
        "mlp_skip_hidden_lag": skip_summary,
        "attention_skip_hidden_lag": attention_skip_summary,
        "causal_summary": causal_summary,
        "component_phase_progress": component_progress_rows,
        "lengthwise_phase_dynamics": lengthwise_rows,
        "phase_origin_locking": phase_origin_rows,
        "evidence_boundary": [
            "one authoritative backbone seed; data seeds are not backbone replications",
            "a two-dimensional class-contrast orbit is not by itself a complete circuit",
            "immediate readout changes are weaker evidence than persistence after continued recurrent calls",
            "the local post-endpoint orbit is tested separately from length-dependent distant recurrence echoes",
        ],
        "elapsed_seconds": time.time() - started,
        "actual_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved()) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "code_revision": git_revision(Path(__file__).resolve().parents[1]),
        "files": {
            "centroid_phase_trajectory": "centroid_phase_trajectory.csv",
            "lengthwise_phase_dynamics": "lengthwise_phase_dynamics.csv",
            "phase_origin_locking": "phase_origin_locking.csv",
            "phase_basis": "phase_basis.csv",
            "causal_phase_interchange": "causal_phase_interchange.csv",
            "causal_phase_interchange_summary": "causal_phase_interchange_summary.csv",
            "mlp_skip_hidden_phase_lag": "mlp_skip_hidden_phase_lag.csv",
            "attention_skip_hidden_phase_lag": "attention_skip_hidden_phase_lag.csv",
            "component_phase_trajectory": "component_phase_trajectory.csv",
            "component_phase_progress": "component_phase_progress.csv",
            "heldout_phase_orbits": "heldout_phase_orbits.png",
            "behavior_and_phase_interchange": "behavior_and_phase_interchange.png",
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_run_manifest(
        manifest_path,
        {
            "status": "complete",
            "analysis": "parity_hidden_four_phase_test",
            "pid": os.getpid(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
            "elapsed_seconds": summary["elapsed_seconds"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "controller": None if args.controller is None else str(args.controller),
            "paper_mode": bool(args.paper_mode),
            "actual_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "output_dir": str(args.out_dir),
            "summary": "summary.json",
        },
    )
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = run(args)
    except Exception:
        write_run_manifest(
            args.out_dir / "run_manifest.json",
            {
                "status": "failed",
                "analysis": "parity_hidden_four_phase_test",
                "pid": os.getpid(),
                "checkpoint": str(args.checkpoint),
                "controller": (
                    None if args.controller is None else str(args.controller)
                ),
                "paper_mode": bool(args.paper_mode),
                "requested_device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "output_dir": str(args.out_dir),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
