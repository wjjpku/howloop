from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.typed_relation_circuit import load_checkpoint
from reasoning_loop.typed_relation_composition import make_relation_batch
from reasoning_loop.typed_relation_overloop_geometry import (
    cell_step,
    decode,
    energy_fraction,
    mean_margin,
    project_readout_sensitive,
    readout_classification_jacobian,
    unroll,
)
from reasoning_loop.typed_relation_train import pick_device


CONDITION_LABELS = {
    "receiver_natural": "接收问题自身 residual",
    "random_valid_question": "随机真实问题 residual",
    "same_answer_question": "同答案随机问题 residual",
    "different_answer_question": "异答案随机问题 residual",
    "random_question_from_D0": "随机问题，从 D0 生成",
    "random_question_from_D1": "随机问题，从 D1 生成",
    "covariance_matched_hidden": "协方差匹配随机 hidden 经模型",
    "shuffled_state_edges": "真实 state + 错配关系边经模型",
    "isotropic_reference": "各向同性向量（仅几何对照）",
}


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "PingFang SC",
                "Arial Unicode MS",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def target_matched_indices(
    receiver_targets: torch.Tensor,
    donor_targets: torch.Tensor,
    *,
    same: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    """Choose a donor for every receiver with the requested answer relation."""
    if receiver_targets.ndim != 1 or donor_targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if receiver_targets.device != donor_targets.device:
        raise ValueError("receiver and donor targets must share a device")
    indices = torch.empty_like(receiver_targets)
    for answer in receiver_targets.unique():
        receiver_mask = receiver_targets.eq(answer)
        donor_mask = donor_targets.eq(answer)
        eligible = donor_mask if same else ~donor_mask
        choices = eligible.nonzero(as_tuple=False).squeeze(1)
        if choices.numel() == 0:
            relation = "same" if same else "different"
            raise ValueError(f"no {relation}-answer donor is available")
        count = int(receiver_mask.sum().item())
        sampled = torch.randint(
            choices.numel(),
            (count,),
            device=choices.device,
            generator=generator,
        )
        indices[receiver_mask] = choices[sampled]
    return indices


def match_row_norm(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if vector.shape != reference.shape:
        raise ValueError("vector and reference must have the same shape")
    scale = reference.norm(dim=1, keepdim=True) / vector.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    return vector * scale


def covariance_matched_sample(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Gaussian sample matching the empirical mean and full covariance."""
    if reference.ndim != 2 or reference.shape[0] < reference.shape[1] + 1:
        raise ValueError("reference must have more rows than columns")
    centered = reference - reference.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (reference.shape[0] - 1)
    jitter = covariance.diagonal().mean().clamp_min(1e-8) * 1e-5
    factor = torch.linalg.cholesky(
        covariance
        + jitter
        * torch.eye(
            reference.shape[1], device=reference.device, dtype=reference.dtype
        )
    )
    noise = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    return reference.mean(dim=0, keepdim=True) + noise @ factor.T


def _prediction_retention(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    target: torch.Tensor,
) -> float:
    base_correct = base_logits.argmax(dim=1).eq(target)
    if not base_correct.any():
        return float("nan")
    return float(
        patched_logits.argmax(dim=1)[base_correct]
        .eq(target[base_correct])
        .float()
        .mean()
        .item()
    )


def _row(
    *,
    common: dict[str, Any],
    condition: str,
    normalization: str,
    patch_steps: int,
    receiver_state: torch.Tensor,
    receiver_natural_delta: torch.Tensor,
    donor_delta: torch.Tensor,
    target: torch.Tensor,
    model: torch.nn.Module,
    jacobian: torch.Tensor,
) -> dict[str, Any]:
    applied = (
        match_row_norm(donor_delta, receiver_natural_delta)
        if normalization == "receiver_norm_matched"
        else donor_delta
    )
    base_logits = decode(model, receiver_state)
    patched_logits = decode(model, receiver_state + applied)
    sensitive, _ = project_readout_sensitive(jacobian, applied)
    cosine = torch.nn.functional.cosine_similarity(
        applied, receiver_natural_delta, dim=1, eps=1e-12
    )
    return {
        **common,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "normalization": normalization,
        "patch_steps": patch_steps,
        "accuracy": float(patched_logits.argmax(dim=1).eq(target).float().mean()),
        "retention": _prediction_retention(base_logits, patched_logits, target),
        "margin": mean_margin(patched_logits, target),
        "margin_change_from_D2": mean_margin(patched_logits, target)
        - mean_margin(base_logits, target),
        "mean_applied_norm": float(applied.norm(dim=1).mean()),
        "mean_natural_norm": float(receiver_natural_delta.norm(dim=1).mean()),
        "sensitive_energy_fraction": energy_fraction(sensitive, applied),
        "cosine_with_receiver_natural": float(cosine.mean()),
    }


@torch.no_grad()
def analyze_checkpoint(
    checkpoint: Path,
    *,
    examples: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model, payload = load_checkpoint(checkpoint, device)
    if model.shared_cell is None or int(payload["train_loops"]) != 2:
        raise ValueError("checkpoint must be a two-loop shared-cell model")
    generator = torch.Generator(device=device).manual_seed(seed)
    order = payload.get("composition_order", "f_then_g")
    receiver = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=order,
    )
    donor = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=order,
    )
    edge_donor = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=order,
    )
    receiver_states = unroll(model, receiver, max_loops=4)
    donor_states = unroll(model, donor, max_loops=4)
    receiver_h2 = receiver_states[2]
    target = receiver.targets[:, 1]
    donor_target = donor.targets[:, 1]
    same_indices = target_matched_indices(
        target, donor_target, same=True, generator=generator
    )
    different_indices = target_matched_indices(
        target, donor_target, same=False, generator=generator
    )

    donor_edges = model.encode_edges(donor)
    mismatched_edges = model.encode_edges(edge_donor)
    visible = torch.ones(
        (examples, 2 * model.cfg.node_count), dtype=torch.bool, device=device
    )
    gaussian_h2 = covariance_matched_sample(donor_states[2], generator=generator)
    gaussian_states = [gaussian_h2]
    mismatched_states = [donor_states[2]]
    for _ in range(2):
        gaussian_states.append(
            cell_step(model, gaussian_states[-1], donor_edges, visible)
        )
        mismatched_states.append(
            cell_step(model, mismatched_states[-1], mismatched_edges, visible)
        )

    jacobian = readout_classification_jacobian(model, receiver_h2)
    common = {
        "model_seed": int(payload["seed"]),
        "checkpoint_step": int(payload["step"]),
        "node_count": model.cfg.node_count,
        "d_model": model.cfg.d_model,
        "checkpoint": str(checkpoint.resolve()),
        "examples": examples,
        "D2_accuracy": float(decode(model, receiver_h2).argmax(1).eq(target).float().mean()),
        "D2_margin": mean_margin(decode(model, receiver_h2), target),
    }

    rows: list[dict[str, Any]] = []
    for patch_steps in (1, 2):
        natural_delta = receiver_states[2 + patch_steps] - receiver_h2
        valid_delta = donor_states[2 + patch_steps] - donor_states[2]
        pre_solution_delta = donor_states[patch_steps] - donor_states[0]
        bridge_delta = donor_states[1 + patch_steps] - donor_states[1]
        gaussian_delta = gaussian_states[patch_steps] - gaussian_states[0]
        mismatched_delta = mismatched_states[patch_steps] - mismatched_states[0]
        isotropic = torch.randn(
            natural_delta.shape,
            device=device,
            dtype=natural_delta.dtype,
            generator=generator,
        )

        deltas = {
            "receiver_natural": natural_delta,
            "random_valid_question": valid_delta,
            "same_answer_question": valid_delta[same_indices],
            "different_answer_question": valid_delta[different_indices],
            "random_question_from_D0": pre_solution_delta,
            "random_question_from_D1": bridge_delta,
            "covariance_matched_hidden": gaussian_delta,
            "shuffled_state_edges": mismatched_delta,
            "isotropic_reference": isotropic,
        }
        for condition, delta in deltas.items():
            if condition == "receiver_natural":
                normalizations = ("raw",)
            elif condition == "isotropic_reference":
                normalizations = ("receiver_norm_matched",)
            else:
                normalizations = ("raw", "receiver_norm_matched")
            for normalization in normalizations:
                rows.append(
                    _row(
                        common=common,
                        condition=condition,
                        normalization=normalization,
                        patch_steps=patch_steps,
                        receiver_state=receiver_h2,
                        receiver_natural_delta=natural_delta,
                        donor_delta=delta,
                        target=target,
                        model=model,
                        jacobian=jacobian,
                    )
                )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["normalization"], row["patch_steps"])].append(
            row
        )
    metrics = (
        "accuracy",
        "retention",
        "margin",
        "margin_change_from_D2",
        "mean_applied_norm",
        "mean_natural_norm",
        "sensitive_energy_fraction",
        "cosine_with_receiver_natural",
    )
    aggregate: list[dict[str, Any]] = []
    for (condition, normalization, patch_steps), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "normalization": normalization,
            "patch_steps": patch_steps,
            "model_count": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(item)
    return aggregate


def _condition_order() -> list[str]:
    return list(CONDITION_LABELS)


def _value(
    aggregate: list[dict[str, Any]],
    condition: str,
    normalization: str,
    steps: int,
    metric: str,
) -> float:
    matches = [
        row
        for row in aggregate
        if row["condition"] == condition
        and row["normalization"] == normalization
        and row["patch_steps"] == steps
    ]
    if (
        not matches
        and condition == "receiver_natural"
        and normalization == "receiver_norm_matched"
    ):
        matches = [
            row
            for row in aggregate
            if row["condition"] == condition
            and row["normalization"] == "raw"
            and row["patch_steps"] == steps
        ]
    if len(matches) != 1:
        return float("nan")
    return float(matches[0][f"{metric}_mean"])


def plot_accuracy_heatmap(aggregate: list[dict[str, Any]], path: Path) -> None:
    conditions = _condition_order()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.8), constrained_layout=True)
    for axis, normalization, title in (
        (axes[0], "raw", "模型实际生成的 residual（原始大小）"),
        (axes[1], "receiver_norm_matched", "逐样本匹配接收问题 residual 范数"),
    ):
        data = np.array(
            [
                [
                    _value(aggregate, condition, normalization, steps, "accuracy")
                    for steps in (1, 2)
                ]
                for condition in conditions
            ]
        )
        image = axis.imshow(data, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks((0, 1), ("加 1-step residual", "加 2-step residual"))
        axis.set_yticks(range(len(conditions)))
        axis.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
        axis.set_title(title)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = data[row, column]
                label = "—" if np.isnan(value) else f"{value:.3f}"
                color = "white" if not np.isnan(value) and (value < 0.25 or value > 0.75) else "black"
                axis.text(column, row, label, ha="center", va="center", color=color)
    fig.colorbar(image, ax=axes, shrink=0.82, label="接收问题答案 accuracy（越红越稳定）")
    fig.suptitle("随机输入经模型生成的 residual：能否移植到另一个已解 hidden state？", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_geometry(aggregate: list[dict[str, Any]], path: Path) -> None:
    conditions = _condition_order()
    normalization = "receiver_norm_matched"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), constrained_layout=True)
    y = np.arange(len(conditions))
    for steps, color, offset, label in (
        (1, "#1565C0", -0.18, "1-step residual"),
        (2, "#C62828", 0.18, "2-step residual"),
    ):
        sensitive = [
            _value(
                aggregate,
                condition,
                normalization,
                steps,
                "sensitive_energy_fraction",
            )
            for condition in conditions
        ]
        margin_change = [
            _value(
                aggregate,
                condition,
                normalization,
                steps,
                "margin_change_from_D2",
            )
            for condition in conditions
        ]
        axes[0].barh(y + offset, sensitive, height=0.32, color=color, label=label)
        axes[1].barh(y + offset, margin_change, height=0.32, color=color, label=label)
    for axis in axes:
        axis.set_yticks(y)
        axis.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
        axis.axvline(0, color="black", linewidth=0.8)
        axis.grid(axis="x", alpha=0.25)
        axis.legend()
    axes[0].set_xlabel("相对接收 h₂ readout 敏感子空间的能量比例")
    axes[0].set_title("residual 在接收问题输出空间中的方向")
    axes[1].set_xlabel("移植后 margin − D2 margin")
    axes[1].set_title("residual 对接收问题答案余量的作用")
    fig.suptitle("范数匹配后：差异来自方向，而不是 residual 大小", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    selected = {}
    for condition in (
        "receiver_natural",
        "random_valid_question",
        "same_answer_question",
        "different_answer_question",
        "covariance_matched_hidden",
        "isotropic_reference",
    ):
        normalization = "raw" if condition == "receiver_natural" else "receiver_norm_matched"
        selected[condition] = {
            f"D2_plus_{steps}_residual_accuracy": _value(
                aggregate, condition, normalization, steps, "accuracy"
            )
            for steps in (1, 2)
        }
        selected[condition].update(
            {
                f"D2_plus_{steps}_residual_margin_change": _value(
                    aggregate,
                    condition,
                    normalization,
                    steps,
                    "margin_change_from_D2",
                )
                for steps in (1, 2)
            }
        )
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch model-generated residuals from random inputs into solved states."
    )
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=92_271)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.examples < 128:
        parser.error("examples must be at least 128")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_plots()
    device = pick_device(args.device)
    rows: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(args.checkpoints):
        rows.extend(
            analyze_checkpoint(
                checkpoint,
                examples=args.examples,
                seed=args.seed + 10_000 * index,
                device=device,
            )
        )
    aggregate = aggregate_rows(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "per_checkpoint.csv", rows)
    _write_csv(args.out_dir / "aggregate.csv", aggregate)
    plot_accuracy_heatmap(aggregate, args.out_dir / "model_generated_residual_accuracy.png")
    plot_geometry(aggregate, args.out_dir / "model_generated_residual_geometry.png")
    summary = {
        "definition": (
            "A random residual is generated by running the trained recurrent block "
            "on an independently sampled input trajectory; isotropic vectors are "
            "reported only as a geometric reference."
        ),
        "device": str(device),
        "examples_per_checkpoint": args.examples,
        "checkpoints": [str(path.resolve()) for path in args.checkpoints],
        "selected_results": summarize(aggregate),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
