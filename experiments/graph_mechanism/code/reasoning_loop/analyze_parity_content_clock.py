#!/usr/bin/env python3
"""Causally test Parity content-by-clock composition on one frozen backbone."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_parity_four_phase import (
    collect_contrasts,
    continue_state,
    discover_phase_plane,
    fit_phase_dynamics,
    magnitude_matched_random_patch,
    natural_states,
    random_plane,
    write_csv,
    write_run_manifest,
)
from reasoning_loop.paper_length_telomere import (
    generate_paper_batch,
    load_backbone,
    pick_device,
)
from reasoning_loop.parity_content_clock import (
    balanced_donor_indices,
    class_midpoint,
    circular_difference,
    coordinate_angles,
    patch_answer_complement,
    patch_answer_plane,
    project_answer,
    summarize_factorial_rows,
    wrap_angle,
)


DEFAULT_DISCOVERY_LENGTHS = (12, 16, 20, 24, 32, 40)
DEFAULT_CAUSAL_LENGTHS = (10, 14, 18, 22)
DEFAULT_PHASE_OFFSETS = (-2, -1, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--discovery-lengths", nargs="+", type=int, default=DEFAULT_DISCOVERY_LENGTHS
    )
    parser.add_argument(
        "--causal-lengths", nargs="+", type=int, default=DEFAULT_CAUSAL_LENGTHS
    )
    parser.add_argument(
        "--phase-offsets", nargs="+", type=int, default=DEFAULT_PHASE_OFFSETS
    )
    parser.add_argument("--discovery-batch-size", type=int, default=512)
    parser.add_argument("--discovery-batches", type=int, default=2)
    parser.add_argument("--causal-batch-size", type=int, default=256)
    parser.add_argument("--discovery-seed", type=int, required=True)
    parser.add_argument("--causal-seed", type=int, required=True)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--continuation-calls", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_protocol(model: Any, *, allow_tiny_test_model: bool = False) -> None:
    """Reject old every-loop input reinjection and non-authoritative models."""

    config = model.config
    checks = {
        "one shared layer": len(model.layers) == 1,
        "token input-once": config.token_embedding_injection == "initial_only",
        "NoPE": config.position_embedding == "none",
        "position input-once": config.position_injection == "initial_only",
    }
    if not allow_tiny_test_model:
        checks.update(
            {
                "d_model=256": config.d_model == 256,
                "64 heads": config.n_heads == 64,
                "MLP width 1024": config.d_mlp == 1024,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("input-once parity protocol mismatch: " + ", ".join(failed))


def _fixed_batch(
    spec: Any,
    *,
    length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Any:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return generate_paper_batch(
        spec,
        batch_size=batch_size,
        min_length=length,
        max_length=length,
        fixed_length=length,
        generator=generator,
    ).to(device)


def _oriented_margins(
    logits: torch.Tensor, labels: torch.Tensor, answer_position: int
) -> torch.Tensor:
    answer = logits[:, answer_position]
    rows = torch.arange(answer.shape[0], device=answer.device)
    return answer[rows, labels] - answer[rows, 1 - labels]


def _plane_complement_norm(
    delta: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    live_basis = basis.to(device=delta.device, dtype=delta.dtype)
    plane = (delta @ live_basis) @ live_basis.T
    return torch.linalg.vector_norm(delta - plane, dim=-1)


def _wrapped_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


@torch.inference_mode()
def run_factorial_interchange(
    model: Any,
    spec: Any,
    *,
    basis: torch.Tensor,
    theta: float,
    length: int,
    tested_phase_offset: int,
    batch_size: int,
    seed: int,
    continuation_calls: int,
    random_controls: int,
    device: torch.device,
    backbone_seed: int = -1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one held-out length/offset factorial block and return raw rows."""

    if tested_phase_offset == 0:
        raise ValueError("tested phase offset must be nonzero")
    minimum_call = length + min(0, tested_phase_offset)
    if minimum_call < 1:
        raise ValueError("phase offset reaches before the first recurrent call")
    receiver_batch = _fixed_batch(
        spec,
        length=length,
        batch_size=batch_size,
        seed=seed + 10007 * length,
        device=device,
    )
    donor_batch = _fixed_batch(
        spec,
        length=length,
        batch_size=batch_size,
        seed=seed + 1_000_003 + 10007 * length,
        device=device,
    )
    receiver_labels = receiver_batch.targets[:, length].long()
    donor_labels = donor_batch.targets[:, length].long()
    final_call = length + max(0, tested_phase_offset) + continuation_calls
    receiver_states = natural_states(
        model,
        receiver_batch.inputs,
        steps=final_call,
        controller=None,
        controller_anchor=None,
    )
    donor_states = natural_states(
        model,
        donor_batch.inputs,
        steps=final_call,
        controller=None,
        controller_anchor=None,
    )
    receiver_initial = receiver_states[length - 1]
    receiver_initial_prediction = model.decode(receiver_initial)[:, length].argmax(-1)
    donor_maps = {
        False: balanced_donor_indices(
            receiver_labels,
            donor_labels,
            opposite=False,
            seed=seed + 7001 + length,
        ),
        True: balanced_donor_indices(
            receiver_labels,
            donor_labels,
            opposite=True,
            seed=seed + 9001 + length,
        ),
    }
    control_planes = [
        random_plane(
            basis.shape[0],
            seed=seed + 30011 * (index + 1) + length,
            exclude=basis,
        )
        for index in range(random_controls)
    ]
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for content_opposite, donor_indices in donor_maps.items():
        relation = "opposite" if content_opposite else "same"
        selected_labels = donor_labels[donor_indices]
        for receiver_index, donor_index in enumerate(donor_indices.detach().cpu().tolist()):
            pair_rows.append(
                {
                    "backbone_seed": backbone_seed,
                    "length": length,
                    "data_seed": seed,
                    "receiver_example": receiver_index,
                    "donor_example": donor_index,
                    "receiver_label": int(receiver_labels[receiver_index].cpu()),
                    "donor_label": int(donor_labels[donor_index].cpu()),
                    "content_relation": relation,
                }
            )
        for phase_offset in (0, tested_phase_offset):
            donor_call = length + phase_offset
            donor_initial = donor_states[donor_call - 1][donor_indices]
            conditions: list[tuple[str, str, torch.Tensor]] = [
                (
                    "phase",
                    "phase_patch",
                    patch_answer_plane(
                        receiver_initial,
                        donor_initial,
                        basis,
                        answer_position=length,
                    ),
                ),
                (
                    "complement",
                    "complement_patch",
                    patch_answer_complement(
                        receiver_initial,
                        donor_initial,
                        basis,
                        answer_position=length,
                    ),
                ),
            ]
            for control_index, control_basis in enumerate(control_planes):
                conditions.append(
                    (
                        "random",
                        f"random_plane_{control_index}",
                        magnitude_matched_random_patch(
                            receiver_initial,
                            donor_initial,
                            basis,
                            control_basis,
                            position=length,
                        ),
                    )
                )
            donor_initial_coordinates_all = project_answer(
                donor_states[donor_call - 1], basis, length
            )
            donor_initial_center = class_midpoint(
                donor_initial_coordinates_all, donor_labels
            )
            donor_initial_angle = coordinate_angles(
                project_answer(donor_initial, basis, length)
                - donor_initial_center
            )
            for control_family, condition, initial in conditions:
                initial_delta = initial[:, length].float() - receiver_initial[:, length].float()
                patch_norm = torch.linalg.vector_norm(initial_delta, dim=-1)
                complement_displacement = _plane_complement_norm(initial_delta, basis)
                live = initial
                for continuation in range(continuation_calls + 1):
                    if continuation:
                        live = continue_state(
                            model,
                            live,
                            receiver_batch.inputs,
                            start_step=length + continuation - 1,
                            calls=1,
                            controller=None,
                            controller_anchor=None,
                        )
                    donor_target = donor_states[donor_call + continuation - 1][
                        donor_indices
                    ]
                    logits = model.decode(live)
                    predictions = logits[:, length].argmax(-1)
                    margins = _oriented_margins(logits, selected_labels, length)
                    target_logits = model.decode(donor_target)
                    target_predictions = target_logits[:, length].argmax(-1)
                    target_margins = _oriented_margins(
                        target_logits, selected_labels, length
                    )
                    donor_target_coordinates_all = project_answer(
                        donor_states[donor_call + continuation - 1],
                        basis,
                        length,
                    )
                    donor_target_center = class_midpoint(
                        donor_target_coordinates_all, donor_labels
                    )
                    observed_angles = coordinate_angles(
                        project_answer(live, basis, length) - donor_target_center
                    )
                    observed_coordinates = (
                        project_answer(live, basis, length) - donor_target_center
                    )
                    predicted_angles = _wrapped_tensor(
                        donor_initial_angle + continuation * theta
                    )
                    target_coordinates = (
                        project_answer(donor_target, basis, length)
                        - donor_target_center
                    )
                    target_angles = coordinate_angles(target_coordinates)
                    for index in range(batch_size):
                        observed = float(observed_angles[index].cpu())
                        predicted = float(predicted_angles[index].cpu())
                        target_angle = float(target_angles[index].cpu())
                        rows.append(
                            {
                                "backbone_seed": backbone_seed,
                                "length": length,
                                "data_seed": seed,
                                "tested_phase_offset": tested_phase_offset,
                                "phase_offset": phase_offset,
                                "continuation_calls": continuation,
                                "control_family": control_family,
                                "condition": condition,
                                "content_relation": relation,
                                "receiver_example": index,
                                "donor_example": int(donor_indices[index].cpu()),
                                "receiver_label": int(receiver_labels[index].cpu()),
                                "donor_label": int(selected_labels[index].cpu()),
                                "observed_angle": observed,
                                "observed_phase_x": float(
                                    observed_coordinates[index, 0].cpu()
                                ),
                                "observed_phase_y": float(
                                    observed_coordinates[index, 1].cpu()
                                ),
                                "predicted_angle": predicted,
                                "donor_target_angle": target_angle,
                                "donor_target_phase_x": float(
                                    target_coordinates[index, 0].cpu()
                                ),
                                "donor_target_phase_y": float(
                                    target_coordinates[index, 1].cpu()
                                ),
                                "model_angle_error": abs(
                                    circular_difference(observed, predicted)
                                ),
                                "donor_target_angle_error": abs(
                                    circular_difference(observed, target_angle)
                                ),
                                "content_correct": float(
                                    predictions[index] == selected_labels[index]
                                ),
                                "receiver_content_correct": float(
                                    predictions[index] == receiver_labels[index]
                                ),
                                "prediction_flipped_from_receiver": float(
                                    predictions[index]
                                    != receiver_initial_prediction[index]
                                ),
                                "target_readout_match": float(
                                    predictions[index] == target_predictions[index]
                                ),
                                "oriented_margin": float(margins[index].cpu()),
                                "target_oriented_margin": float(
                                    target_margins[index].cpu()
                                ),
                                "patch_norm": float(patch_norm[index].cpu()),
                                "complement_displacement": float(
                                    complement_displacement[index].cpu()
                                ),
                            }
                        )
    return rows, pair_rows


def _append_gzip_csv(path: Path, rows: Sequence[dict[str, Any]], *, first: bool) -> None:
    if not rows:
        raise ValueError("refusing to append an empty raw block")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wt" if first else "at"
    with gzip.open(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if first:
            writer.writeheader()
        writer.writerows(rows)


def _decision_rows(
    summaries: Sequence[dict[str, Any]],
    interactions: Sequence[dict[str, Any]],
    *,
    theta: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_rows = [
        row
        for row in summaries
        if row["control_family"] == "phase" and row["condition"] == "phase_patch"
    ]
    content_errors: list[float] = []
    clock_errors: list[float] = []
    decision_rows: list[dict[str, Any]] = []
    grouping: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for row in phase_rows:
        if int(row["continuation_calls"]) != 0:
            continue
        key = (
            int(row["length"]),
            int(row["data_seed"]),
            int(row["tested_phase_offset"]),
            int(row["receiver_label"]),
        )
        grouping.setdefault(key, []).append(row)
    for (
        length,
        data_seed,
        tested_offset,
        receiver_label,
    ), selected in sorted(grouping.items()):
        angle_map = {
            (str(row["content_relation"]), int(row["phase_offset"])): float(
                row["mean_observed_angle"]
            )
            for row in selected
        }
        required = {
            ("same", 0),
            ("opposite", 0),
            ("same", tested_offset),
            ("opposite", tested_offset),
        }
        if not required.issubset(angle_map):
            continue
        content_shift = circular_difference(
            angle_map[("opposite", 0)], angle_map[("same", 0)]
        )
        clock_shift = circular_difference(
            angle_map[("same", tested_offset)], angle_map[("same", 0)]
        )
        content_error = abs(circular_difference(content_shift, math.pi))
        clock_error = abs(circular_difference(clock_shift, tested_offset * theta))
        content_errors.append(content_error)
        clock_errors.append(clock_error)
        decision_rows.append(
            {
                "length": length,
                "data_seed": data_seed,
                "tested_phase_offset": tested_offset,
                "receiver_label": receiver_label,
                "content_shift": content_shift,
                "content_shift_error_from_pi": content_error,
                "clock_shift": clock_shift,
                "expected_clock_shift": wrap_angle(tested_offset * theta),
                "clock_shift_error": clock_error,
            }
        )

    interaction_errors = [
        float(row["absolute_factorial_interaction_error"])
        for row in interactions
        if row["control_family"] == "phase" and row["condition"] == "phase_patch"
        and 0 <= int(row["continuation_calls"]) <= 2
    ]
    phase_continued = [
        float(row["mean_donor_target_circular_error"])
        for row in phase_rows
        if 1 <= int(row["continuation_calls"]) <= 2
    ]
    random_continued = [
        float(row["mean_donor_target_circular_error"])
        for row in summaries
        if row["control_family"] == "random"
        and 1 <= int(row["continuation_calls"]) <= 2
    ]
    phase_opposite_zero = [
        row
        for row in phase_rows
        if row["content_relation"] == "opposite"
        and int(row["phase_offset"]) == 0
        and 1 <= int(row["continuation_calls"]) <= 2
    ]
    random_opposite_zero = [
        row
        for row in summaries
        if row["control_family"] == "random"
        and row["content_relation"] == "opposite"
        and int(row["phase_offset"]) == 0
        and 1 <= int(row["continuation_calls"]) <= 2
    ]
    mean_content_error = float(np.mean(content_errors))
    mean_clock_error = float(np.mean(clock_errors))
    mean_interaction_error = float(np.mean(interaction_errors))
    mean_phase_continued = float(np.mean(phase_continued))
    median_random_continued = float(np.median(random_continued))
    phase_flip = float(
        np.mean([float(row["mean_target_readout_match"]) for row in phase_opposite_zero])
    )
    random_flip = float(
        np.mean([float(row["mean_target_readout_match"]) for row in random_opposite_zero])
    )
    criteria = {
        "content_shift_error_below_0_50": mean_content_error < 0.50,
        "clock_shift_error_below_0_50": mean_clock_error < 0.50,
        "interaction_error_calls_0_2_below_0_35": mean_interaction_error < 0.35,
        "continued_better_than_random": mean_phase_continued
        < median_random_continued,
        "opposite_content_behavior_beats_random_by_0_20": phase_flip
        > random_flip + 0.20,
    }
    decision = {
        "mean_content_shift_error_from_pi": mean_content_error,
        "mean_clock_shift_error": mean_clock_error,
        "mean_absolute_factorial_interaction_error": mean_interaction_error,
        "mean_phase_continued_target_error_calls_1_2": mean_phase_continued,
        "median_random_continued_target_error_calls_1_2": median_random_continued,
        "phase_opposite_target_readout_match": phase_flip,
        "random_opposite_target_readout_match": random_flip,
        "criteria": criteria,
        "checkpoint_data_seed_pass": all(criteria.values()),
    }
    return decision_rows, decision


def _plot_decision(
    decision_rows: Sequence[dict[str, Any]],
    interactions: Sequence[dict[str, Any]],
    out_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    offsets = sorted({int(row["tested_phase_offset"]) for row in decision_rows})
    content = [
        np.mean(
            [
                float(row["content_shift_error_from_pi"])
                for row in decision_rows
                if int(row["tested_phase_offset"]) == offset
            ]
        )
        for offset in offsets
    ]
    clock = [
        np.mean(
            [
                float(row["clock_shift_error"])
                for row in decision_rows
                if int(row["tested_phase_offset"]) == offset
            ]
        )
        for offset in offsets
    ]
    x = np.arange(len(offsets))
    axes[0].plot(x, content, "o-", label="content shift vs pi")
    axes[0].plot(x, clock, "s-", label="clock shift vs offset*theta")
    axes[0].axhline(0.50, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, [str(value) for value in offsets])
    axes[0].set_xlabel("phase offset (calls)")
    axes[0].set_ylabel("circular error (rad)")
    axes[0].legend(frameon=False)
    phase_interactions = [
        row
        for row in interactions
        if row["control_family"] == "phase" and row["condition"] == "phase_patch"
    ]
    values = [float(row["absolute_factorial_interaction_error"]) for row in phase_interactions]
    axes[1].hist(values, bins=min(12, max(3, len(values))), color="#4C78A8")
    axes[1].axvline(0.35, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("absolute factorial interaction error (rad)")
    axes[1].set_ylabel("cells")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(out_path, dpi=190)
    plt.close(figure)


def run(
    args: argparse.Namespace, *, allow_tiny_test_model: bool = False
) -> dict[str, Any]:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        {
            "status": "running",
            "analysis": "parity_content_clock_stage_a",
            "pid": os.getpid(),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.out_dir),
            "started_at_unix": started,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    try:
        device = pick_device(args.device)
        model, spec, payload = load_backbone(
            args.checkpoint,
            device=device,
            paper_mode=not allow_tiny_test_model,
        )
        if spec.name != "parity":
            raise ValueError("checkpoint task must be parity")
        validate_protocol(model, allow_tiny_test_model=allow_tiny_test_model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        discovery_lengths = tuple(args.discovery_lengths)
        causal_lengths = tuple(args.causal_lengths)
        phase_offsets = tuple(args.phase_offsets)
        discovery_batch_size = int(args.discovery_batch_size)
        discovery_batches = int(args.discovery_batches)
        causal_batch_size = int(args.causal_batch_size)
        random_controls = int(args.random_controls)
        continuation_calls = int(args.continuation_calls)
        if args.smoke:
            discovery_lengths = discovery_lengths[:2]
            causal_lengths = causal_lengths[:1]
            phase_offsets = (1,)
            discovery_batch_size = min(discovery_batch_size, 32)
            discovery_batches = 1
            causal_batch_size = min(causal_batch_size, 16)
            random_controls = min(random_controls, 1)
            continuation_calls = min(continuation_calls, 2)
        if set(discovery_lengths) & set(causal_lengths):
            raise ValueError("discovery and causal lengths must be disjoint")
        if not phase_offsets or any(offset == 0 for offset in phase_offsets):
            raise ValueError("phase offsets must be a nonempty sequence of nonzero calls")
        discovery = collect_contrasts(
            model,
            spec,
            variant="raw",
            split="discovery",
            lengths=discovery_lengths,
            relative_start=0,
            relative_end=8,
            batch_size=discovery_batch_size,
            batches=discovery_batches,
            seed=int(args.discovery_seed),
            device=device,
            controller=None,
            controller_anchor=None,
        )
        basis = discover_phase_plane(discovery)
        try:
            dynamics = fit_phase_dynamics(discovery, basis)
        except ZeroDivisionError:
            if not allow_tiny_test_model:
                raise
            dynamics = {
                "polar_rotation_signed_angle_radians": 0.0,
                "test_fixture_degenerate_dynamics": True,
            }
        theta = float(dynamics["polar_rotation_signed_angle_radians"])
        basis_rows = [
            {
                "phase_coordinate": coordinate + 1,
                "hidden_dimension": dimension,
                "weight": float(value),
            }
            for coordinate in range(2)
            for dimension, value in enumerate(basis[:, coordinate])
        ]
        write_csv(args.out_dir / "phase_basis.csv", basis_rows)
        write_json(args.out_dir / "discovery_dynamics.json", dynamics)

        raw_path = args.out_dir / "factorial_per_example.csv.gz"
        first_raw_block = True
        all_summaries: list[dict[str, Any]] = []
        all_interactions: list[dict[str, Any]] = []
        all_pairs: list[dict[str, Any]] = []
        backbone_seed = int(payload.get("seed", -1))
        for length in causal_lengths:
            for tested_offset in phase_offsets:
                block_rows, pair_rows = run_factorial_interchange(
                    model,
                    spec,
                    basis=basis,
                    theta=theta,
                    length=int(length),
                    tested_phase_offset=int(tested_offset),
                    batch_size=causal_batch_size,
                    seed=int(args.causal_seed),
                    continuation_calls=continuation_calls,
                    random_controls=random_controls,
                    device=device,
                    backbone_seed=backbone_seed,
                )
                _append_gzip_csv(raw_path, block_rows, first=first_raw_block)
                first_raw_block = False
                summaries, interactions = summarize_factorial_rows(block_rows)
                all_summaries.extend(summaries)
                all_interactions.extend(interactions)
                all_pairs.extend(pair_rows)
        write_csv(args.out_dir / "donor_pairs.csv", all_pairs)
        write_csv(args.out_dir / "factorial_summary.csv", all_summaries)
        write_csv(args.out_dir / "factorial_interaction.csv", all_interactions)
        decision_rows, decision = _decision_rows(
            all_summaries, all_interactions, theta=theta
        )
        write_csv(args.out_dir / "decision_cells.csv", decision_rows)
        _plot_decision(
            decision_rows,
            all_interactions,
            args.out_dir / "content_clock_factorial.png",
        )
        source_paths = [
            Path(__file__),
            Path(__file__).with_name("parity_content_clock.py"),
            Path(__file__).with_name("analyze_parity_four_phase.py"),
        ]
        summary = {
            "status": "complete",
            "analysis": "parity_content_clock_stage_a",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_step": int(payload.get("step", -1)),
            "backbone_seed": backbone_seed,
            "controller": None,
            "architecture": {
                "task": spec.name,
                "d_model": model.config.d_model,
                "attention_heads": model.config.n_heads,
                "mlp_width": model.config.d_mlp,
                "shared_physical_layers": len(model.layers),
                "token_embedding_injection": model.config.token_embedding_injection,
                "position_embedding": model.config.position_embedding,
                "position_injection": model.config.position_injection,
                "loss_placement": "answer CE only at registered T(n)=n",
            },
            "protocol": {
                "state_site": "post-call final-normalized recurrent state passed to next call",
                "intervention_site": "answer-token coordinates in discovery 2D plane",
                "discovery_lengths": list(discovery_lengths),
                "causal_lengths": list(causal_lengths),
                "phase_offsets": list(phase_offsets),
                "discovery_seed": int(args.discovery_seed),
                "causal_seed": int(args.causal_seed),
                "discovery_examples_per_length": discovery_batch_size
                * discovery_batches,
                "causal_examples_per_cell": causal_batch_size,
                "random_equal_norm_controls": random_controls,
                "continuation_calls": continuation_calls,
            },
            "discovery_dynamics": dynamics,
            "decision": decision,
            "source_sha256": {str(path): sha256(path) for path in source_paths},
            "elapsed_seconds": time.time() - started,
            "actual_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_cuda_reserved_gib": (
                float(torch.cuda.max_memory_reserved()) / 1024**3
                if device.type == "cuda"
                else None
            ),
            "files": {
                "phase_basis": "phase_basis.csv",
                "discovery_dynamics": "discovery_dynamics.json",
                "donor_pairs": "donor_pairs.csv",
                "raw_per_example": "factorial_per_example.csv.gz",
                "factorial_summary": "factorial_summary.csv",
                "factorial_interaction": "factorial_interaction.csv",
                "decision_cells": "decision_cells.csv",
                "plot": "content_clock_factorial.png",
            },
            "evidence_boundary": [
                "one frozen backbone and one causal-data seed in this run",
                "the three-by-three formal aggregate is required for a cross-seed claim",
                "passing Stage A identifies an answer-token content-clock mechanism, not the input parity algorithm",
            ],
        }
        write_json(args.out_dir / "summary.json", summary)
        write_run_manifest(
            manifest_path,
            {
                "status": "complete",
                "analysis": "parity_content_clock_stage_a",
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "backbone_seed": backbone_seed,
                "causal_seed": int(args.causal_seed),
                "completed_at_unix": time.time(),
                "elapsed_seconds": summary["elapsed_seconds"],
                "summary": "summary.json",
                "output_dir": str(args.out_dir),
            },
        )
        return summary
    except Exception as error:
        write_run_manifest(
            manifest_path,
            {
                "status": "failed",
                "analysis": "parity_content_clock_stage_a",
                "checkpoint": str(args.checkpoint),
                "failed_at_unix": time.time(),
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "output_dir": str(args.out_dir),
            },
        )
        raise


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
