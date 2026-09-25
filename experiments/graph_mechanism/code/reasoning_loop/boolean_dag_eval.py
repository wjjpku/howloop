from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.boolean_dag_data import (
    BooleanDAGBatch,
    BooleanDAGConfig,
    make_boolean_dag_batch,
    make_topology_matched_boolean_dag_batch,
    permute_batch_slots,
    wavefront_targets,
)
from reasoning_loop.boolean_dag_model import (
    BooleanDAGModelConfig,
    PeriodicBooleanDAGTransformer,
    build_boolean_dag_model,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed


def _root_indices(batch: BooleanDAGBatch) -> torch.Tensor:
    return batch.root_mask.long().argmax(dim=1)


def _restore_slot_order(values: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    return values.gather(
        2,
        inverse[:, None, :, None].expand(-1, values.shape[1], -1, values.shape[-1]),
    )


@torch.no_grad()
def permutation_audit(
    model: nn.Module,
    batch: BooleanDAGBatch,
    *,
    max_readouts: int,
    permutations: int,
) -> dict[str, float | int]:
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    model.eval()
    baseline = model.forward_all(batch, max_steps=max_readouts)["logits_by_step"]
    max_difference = 0.0
    disagreement = 0.0
    compared = 0
    for _ in range(permutations):
        permuted, inverse = permute_batch_slots(batch)
        actual = model.forward_all(permuted, max_steps=max_readouts)["logits_by_step"]
        restored = _restore_slot_order(actual, inverse)
        max_difference = max(max_difference, float((restored - baseline).abs().max().cpu()))
        disagreement += float(restored.argmax(dim=-1).ne(baseline.argmax(dim=-1)).sum().cpu())
        compared += restored.shape[0] * restored.shape[1] * restored.shape[2]
    return {
        "permutations": permutations,
        "max_abs_logit_difference": max_difference,
        "prediction_disagreement": disagreement / compared,
    }


def balanced_class_weights(labels: torch.Tensor, *, classes: int) -> torch.Tensor:
    counts = labels.bincount(minlength=classes).to(dtype=torch.float32)
    if counts.shape[0] != classes or (counts == 0).any():
        raise ValueError("probe calibration data must contain every state class")
    weights = counts.reciprocal()
    return weights / weights.mean()


def stratified_probe_weights(
    kinds: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes: int,
) -> torch.Tensor:
    if kinds.shape != labels.shape:
        raise ValueError("kinds and labels must have identical shapes")
    cell_indices = kinds.long() * classes + labels.long()
    cell_counts = cell_indices.bincount(minlength=4 * classes).to(dtype=torch.float32)
    sample_weights = cell_counts[cell_indices].reciprocal()
    return sample_weights / sample_weights.mean()


def fit_frozen_state_probe(
    model: nn.Module,
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    device: torch.device,
    batches: int,
    batch_size: int,
    optimization_steps: int,
    max_readouts: int | None = None,
) -> nn.Linear:
    readouts = model_cfg.steps if max_readouts is None else max_readouts
    readouts = min(readouts, model_cfg.steps)
    model.eval()
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    node_kinds: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(batches):
            batch = make_boolean_dag_batch(data_cfg, batch_size, device)
            output = model.forward_all(
                batch,
                max_steps=readouts,
                return_states=True,
            )
            states.append(output["states_by_step"].detach().reshape(-1, model_cfg.d_model))
            targets.append(wavefront_targets(batch, readouts=readouts).reshape(-1))
            node_kinds.append(
                batch.kinds[:, None, :].expand(-1, readouts, -1).reshape(-1)
            )
    features = torch.cat(states)
    labels = torch.cat(targets)
    kinds = torch.cat(node_kinds)
    sample_weights = stratified_probe_weights(kinds, labels, classes=3).to(device)
    probe = nn.Linear(model_cfg.d_model, 3).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3, weight_decay=1e-4)
    minibatch_size = min(8192, len(features))
    for _ in range(optimization_steps):
        indices = torch.randint(0, len(features), (minibatch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        per_sample_loss = F.cross_entropy(
            probe(features[indices]),
            labels[indices],
            reduction="none",
        )
        loss = (per_sample_loss * sample_weights[indices]).sum() / sample_weights[indices].sum()
        loss.backward()
        optimizer.step()
    return probe.eval()


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, float("nan")),
    )


def _save_heatmap(
    values: np.ndarray,
    *,
    depths: list[int],
    readouts: list[int],
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.7 * len(readouts) + 2.5), max(4.5, 0.55 * len(depths) + 2.0))
    )
    fig.patch.set_facecolor("white")
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(readouts)), readouts)
    ax.set_yticks(range(len(depths)), depths)
    ax.set_xlabel("readout step")
    ax.set_ylabel("root DAG depth")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "-" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value < 0.45 else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def _evaluate_model(
    model: nn.Module,
    *,
    state_probe: nn.Linear | None,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    depths: list[int],
    readouts: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    topology_control: bool = False,
) -> dict[str, torch.Tensor]:
    shape = (len(depths), readouts)
    sums = {
        "root_correct": torch.zeros(shape, device=device),
        "root_probability": torch.zeros(shape, device=device),
        "root_count": torch.zeros(shape, device=device),
        "state_correct": torch.zeros(shape, device=device),
        "state_count": torch.zeros(shape, device=device),
        "exact_state_correct": torch.zeros(shape, device=device),
        "premature": torch.zeros(shape, device=device),
        "unresolved_count": torch.zeros(shape, device=device),
        "delayed": torch.zeros(shape, device=device),
        "resolved_count": torch.zeros(shape, device=device),
        "resolved_value_correct": torch.zeros(shape, device=device),
    }
    max_level = data_cfg.eval_max_depth
    level_correct = torch.zeros(len(depths), readouts, max_level + 1, device=device)
    level_count = torch.zeros_like(level_correct)
    model.eval()
    for depth_index, depth in enumerate(depths):
        fixed_depths = torch.full((batch_size,), depth, device=device, dtype=torch.long)
        for _ in range(batches):
            if topology_control:
                batch = make_topology_matched_boolean_dag_batch(
                    data_cfg,
                    batch_size,
                    device,
                    root_depth=depth,
                )
            else:
                batch = make_boolean_dag_batch(
                    data_cfg,
                    batch_size,
                    device,
                    depths=fixed_depths,
                )
            output = model.forward_all(
                batch,
                max_steps=readouts,
                return_states=state_probe is not None,
            )
            task_logits = output["logits_by_step"]
            state_logits = (
                state_probe(output["states_by_step"])
                if state_probe is not None
                else task_logits
            )
            root_indices = _root_indices(batch)
            root_logits = task_logits.gather(
                2,
                root_indices[:, None, None, None].expand(-1, readouts, 1, 3),
            ).squeeze(2)
            root_targets = (batch.root_values + 1)[:, None].expand(-1, readouts)
            root_predictions = root_logits.argmax(dim=-1)
            root_probabilities = root_logits.softmax(dim=-1).gather(
                2, root_targets[:, :, None]
            ).squeeze(2)
            sums["root_correct"][depth_index] += root_predictions.eq(root_targets).sum(dim=0)
            sums["root_probability"][depth_index] += root_probabilities.sum(dim=0)
            sums["root_count"][depth_index] += batch_size

            targets = wavefront_targets(batch, readouts=readouts)
            predictions = state_logits.argmax(dim=-1)
            correct = predictions.eq(targets)
            resolved = targets.ne(0)
            unresolved = ~resolved
            sums["state_correct"][depth_index] += correct.sum(dim=(0, 2))
            sums["state_count"][depth_index] += batch_size * data_cfg.node_count
            sums["exact_state_correct"][depth_index] += correct.all(dim=2).sum(dim=0)
            sums["premature"][depth_index] += (predictions.ne(0) & unresolved).sum(dim=(0, 2))
            sums["unresolved_count"][depth_index] += unresolved.sum(dim=(0, 2))
            sums["delayed"][depth_index] += (predictions.eq(0) & resolved).sum(dim=(0, 2))
            sums["resolved_count"][depth_index] += resolved.sum(dim=(0, 2))
            sums["resolved_value_correct"][depth_index] += (correct & resolved).sum(dim=(0, 2))
            for level in range(max_level + 1):
                level_mask = batch.levels.eq(level)[:, None, :].expand(-1, readouts, -1)
                level_correct[depth_index, :, level] += (correct & level_mask).sum(dim=(0, 2))
                level_count[depth_index, :, level] += level_mask.sum(dim=(0, 2))

    return {
        "root_accuracy": _safe_ratio(sums["root_correct"], sums["root_count"]).cpu(),
        "root_probability": _safe_ratio(sums["root_probability"], sums["root_count"]).cpu(),
        "state_accuracy": _safe_ratio(sums["state_correct"], sums["state_count"]).cpu(),
        "exact_state_accuracy": _safe_ratio(
            sums["exact_state_correct"], sums["root_count"]
        ).cpu(),
        "premature_rate": _safe_ratio(sums["premature"], sums["unresolved_count"]).cpu(),
        "delayed_rate": _safe_ratio(sums["delayed"], sums["resolved_count"]).cpu(),
        "resolved_value_accuracy": _safe_ratio(
            sums["resolved_value_correct"], sums["resolved_count"]
        ).cpu(),
        "state_accuracy_by_level": _safe_ratio(level_correct, level_count).cpu(),
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    out_dir: Path,
    depths: list[int],
    max_readouts: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    permutation_trials: int = 16,
    probe_batches: int = 8,
    probe_batch_size: int = 256,
    probe_steps: int = 200,
    topology_control: bool = False,
    standard_continuation: str = "truncate",
) -> dict[str, Any]:
    if not depths or min(depths) < 1 or max(depths) > 8:
        raise ValueError("depths must be within 1 through 8")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    data_cfg = BooleanDAGConfig(**checkpoint_data["data_config"])
    model_cfg = BooleanDAGModelConfig(**checkpoint_data["model_config"])
    architecture = checkpoint_data["architecture"]
    loss_mode = checkpoint_data["loss_mode"]
    if standard_continuation not in {"truncate", "cycle"}:
        raise ValueError("standard_continuation must be truncate or cycle")
    if architecture != "standard" and standard_continuation != "truncate":
        raise ValueError("standard_continuation only applies to standard checkpoints")
    readouts = (
        min(max_readouts, model_cfg.steps)
        if architecture == "standard" and standard_continuation == "truncate"
        else max_readouts
    )
    if architecture == "standard" and standard_continuation == "cycle":
        model = PeriodicBooleanDAGTransformer(
            data_cfg,
            model_cfg,
            period=model_cfg.steps,
        ).to(device)
    else:
        model = build_boolean_dag_model(
            architecture=architecture,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
        ).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    set_seed(20260710)

    state_probe = None
    state_metric_source = "native_intermediate_head"
    if loss_mode == "final":
        state_probe = fit_frozen_state_probe(
            model,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
            batches=probe_batches,
            batch_size=probe_batch_size,
            optimization_steps=probe_steps,
        )
        state_metric_source = "frozen_shared_linear_probe_correlational"

    metrics = _evaluate_model(
        model,
        state_probe=state_probe,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        depths=depths,
        readouts=readouts,
        batches=batches,
        batch_size=batch_size,
        device=device,
        topology_control=topology_control,
    )
    audit_depths = torch.full(
        (batch_size,),
        min(max(depths), data_cfg.train_max_depth),
        device=device,
        dtype=torch.long,
    )
    audit_batch = make_boolean_dag_batch(
        data_cfg,
        batch_size,
        device,
        depths=audit_depths,
    )
    audit = permutation_audit(
        model,
        audit_batch,
        max_readouts=readouts,
        permutations=permutation_trials,
    )
    if audit["prediction_disagreement"] > 1e-5:
        raise RuntimeError(f"slot permutation audit failed: {audit}")

    out_dir.mkdir(parents=True, exist_ok=True)
    readout_labels = list(range(1, readouts + 1))
    for key, title, label, filename in (
        ("root_accuracy", "Root accuracy by graph depth and readout", "accuracy", "root_accuracy.png"),
        (
            "root_probability",
            "Correct-root probability by graph depth and readout",
            "probability",
            "root_probability.png",
        ),
        (
            "state_accuracy",
            f"Node wavefront state accuracy ({state_metric_source})",
            "accuracy",
            "state_accuracy.png",
        ),
        (
            "premature_rate",
            f"Premature resolution rate ({state_metric_source})",
            "rate",
            "premature_rate.png",
        ),
        (
            "delayed_rate",
            f"Delayed resolution rate ({state_metric_source})",
            "rate",
            "delayed_rate.png",
        ),
    ):
        _save_heatmap(
            metrics[key].numpy(),
            depths=depths,
            readouts=readout_labels,
            title=title,
            colorbar_label=label,
            path=out_dir / filename,
        )

    rows: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(depths):
        for readout_index, readout in enumerate(readout_labels):
            rows.append(
                {
                    "depth": depth,
                    "readout": readout,
                    "root_accuracy": float(metrics["root_accuracy"][depth_index, readout_index]),
                    "root_probability": float(
                        metrics["root_probability"][depth_index, readout_index]
                    ),
                    "state_accuracy": float(metrics["state_accuracy"][depth_index, readout_index]),
                    "exact_state_accuracy": float(
                        metrics["exact_state_accuracy"][depth_index, readout_index]
                    ),
                    "premature_rate": float(
                        metrics["premature_rate"][depth_index, readout_index]
                    ),
                    "delayed_rate": float(metrics["delayed_rate"][depth_index, readout_index]),
                    "resolved_value_accuracy": float(
                        metrics["resolved_value_accuracy"][depth_index, readout_index]
                    ),
                }
            )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "permutation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if state_probe is not None:
        torch.save(state_probe.state_dict(), out_dir / "state_probe.pt")
    summary = {
        "task_version": checkpoint_data.get("task_version", "legacy_unversioned"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "architecture": architecture,
        "standard_continuation": (
            standard_continuation if architecture == "standard" else None
        ),
        "loss_mode": loss_mode,
        "depths": depths,
        "readouts": readout_labels,
        "examples_per_depth": batches * batch_size,
        "evaluation_distribution": (
            "topology_matched_depth8_master" if topology_control else "native_depth_specific"
        ),
        "state_metric_source": state_metric_source,
        "permutation_audit": audit,
        **{key: value.tolist() for key, value in metrics.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Boolean-DAG checkpoints across depth and readout.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--max-readouts", type=int, default=10)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--permutation-trials", type=int, default=16)
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--topology-control", action="store_true")
    parser.add_argument(
        "--standard-continuation",
        choices=["truncate", "cycle"],
        default="truncate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        depths=args.depths,
        max_readouts=args.max_readouts,
        batches=args.batches,
        batch_size=args.batch_size,
        device=pick_device(args.device),
        permutation_trials=args.permutation_trials,
        probe_batches=args.probe_batches,
        probe_batch_size=args.probe_batch_size,
        probe_steps=args.probe_steps,
        topology_control=args.topology_control,
        standard_continuation=args.standard_continuation,
    )


if __name__ == "__main__":
    main()
