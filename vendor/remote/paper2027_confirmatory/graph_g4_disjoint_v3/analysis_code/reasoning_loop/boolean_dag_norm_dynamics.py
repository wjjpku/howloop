from __future__ import annotations

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

from reasoning_loop.boolean_dag_data import BooleanDAGConfig, make_boolean_dag_batch
from reasoning_loop.boolean_dag_model import BooleanDAGModelConfig, build_boolean_dag_model


def decompose_group_update(
    incoming: torch.Tensor,
    update: torch.Tensor,
    *,
    groups: int,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if incoming.shape != update.shape:
        raise ValueError("incoming and update must have identical shapes")
    if incoming.ndim < 1:
        raise ValueError("incoming and update must have at least one dimension")
    d_model = incoming.shape[-1]
    if groups < 1 or groups > d_model or d_model % groups:
        raise ValueError("groups must be positive and divide the hidden dimension")
    group_size = d_model // groups
    incoming_groups = incoming.float().reshape(
        *incoming.shape[:-1], groups, group_size
    )
    update_groups = update.float().reshape(*update.shape[:-1], groups, group_size)
    denominator = incoming_groups.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    coefficient = (incoming_groups * update_groups).sum(
        dim=-1, keepdim=True
    ) / denominator
    radial = coefficient * incoming_groups
    tangential = update_groups - radial
    radial_energy = radial.square().sum(dim=(-1, -2))
    tangential_energy = tangential.square().sum(dim=(-1, -2))
    total_energy = update_groups.square().sum(dim=(-1, -2))
    safe_total = total_energy.clamp_min(eps)
    return {
        "total_energy": total_energy,
        "radial_energy": radial_energy,
        "tangential_energy": tangential_energy,
        "radial_fraction": radial_energy / safe_total,
        "tangential_fraction": tangential_energy / safe_total,
    }


def summarize_dynamics_tensors(
    output: dict[str, torch.Tensor],
    *,
    groups: int,
) -> list[dict[str, float | int]]:
    required = {
        "incoming_by_step",
        "pre_outer_by_step",
        "recurrent_states_by_step",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"missing dynamics tensors: {sorted(missing)}")
    incoming = output["incoming_by_step"]
    pre_outer = output["pre_outer_by_step"]
    recurrent = output["recurrent_states_by_step"]
    if incoming.shape != pre_outer.shape or incoming.shape != recurrent.shape:
        raise ValueError("all dynamics tensors must have identical shapes")
    if incoming.ndim != 4:
        raise ValueError("dynamics tensors must have shape [batch, loop, node, hidden]")

    rows: list[dict[str, float | int]] = []
    for loop_index in range(incoming.shape[1]):
        loop_incoming = incoming[:, loop_index]
        pre_update = pre_outer[:, loop_index] - loop_incoming
        effective_update = recurrent[:, loop_index] - loop_incoming
        pieces = decompose_group_update(
            loop_incoming,
            pre_update,
            groups=groups,
        )
        total_energy = pieces["total_energy"].sum().clamp_min(1e-12)
        radial_fraction = pieces["radial_energy"].sum() / total_energy
        tangential_fraction = pieces["tangential_energy"].sum() / total_energy
        rows.append(
            {
                "loop": loop_index + 1,
                "state_cosine": float(
                    F.cosine_similarity(
                        recurrent[:, loop_index].float(),
                        loop_incoming.float(),
                        dim=-1,
                    ).mean()
                ),
                "state_norm": float(
                    recurrent[:, loop_index].float().norm(dim=-1).mean()
                ),
                "pre_update_norm": float(pre_update.float().norm(dim=-1).mean()),
                "effective_update_norm": float(
                    effective_update.float().norm(dim=-1).mean()
                ),
                "radial_fraction": float(radial_fraction),
                "tangential_fraction": float(tangential_fraction),
            }
        )
    return rows


def write_dynamics_rows(
    rows: list[dict[str, Any]],
    *,
    csv_path: Path,
    json_path: Path,
) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _save_heatmap(
    values: np.ndarray,
    *,
    depths: list[int],
    readouts: int,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(7.0, 0.65 * readouts + 2.0), max(4.0, 0.55 * len(depths) + 1.5))
    )
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(readouts), range(1, readouts + 1))
    ax.set_yticks(range(len(depths)), depths)
    ax.set(xlabel="loop readout", ylabel="DAG depth", title=title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.45 else "black",
            )
    fig.colorbar(image, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _save_dynamics_plot(
    rows: list[dict[str, Any]],
    *,
    trained_horizon: int,
    path: Path,
) -> None:
    metrics = (
        ("state_cosine", "Consecutive-state cosine"),
        ("pre_update_norm", "Pre-normalization update norm"),
        ("effective_update_norm", "Effective boundary update norm"),
        ("tangential_fraction", "Group-tangential energy fraction"),
    )
    loops = sorted({int(row["loop"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [
            np.mean([float(row[key]) for row in rows if int(row["loop"]) == loop])
            for loop in loops
        ]
        ax.plot(loops, values, marker="o", linewidth=1.8)
        ax.axvline(trained_horizon, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="loop")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


@torch.inference_mode()
def analyze_norm_checkpoint(
    *,
    checkpoint: Path,
    out_dir: Path,
    depths: list[int],
    readouts: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    seed: int = 20_260_714,
) -> dict[str, Any]:
    if not depths or min(depths) < 1:
        raise ValueError("depths must be nonempty and positive")
    if readouts < 1 or batches < 1 or batch_size < 2 or batch_size % 2:
        raise ValueError("readouts and batches must be positive; batch_size must be even")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    data_cfg = BooleanDAGConfig(**checkpoint_data["data_config"])
    model_cfg = BooleanDAGModelConfig(**checkpoint_data["model_config"])
    if max(depths) > data_cfg.eval_max_depth:
        raise ValueError("requested depth exceeds checkpoint evaluation configuration")
    model = build_boolean_dag_model(
        architecture=checkpoint_data["architecture"],
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()

    root_correct = torch.zeros(len(depths), readouts, device=device)
    root_probability = torch.zeros_like(root_correct)
    root_count = torch.zeros_like(root_correct)
    dynamics_keys = (
        "state_cosine",
        "state_norm",
        "pre_update_norm",
        "effective_update_norm",
        "radial_fraction",
        "tangential_fraction",
    )
    dynamics_sums = {
        key: torch.zeros(len(depths), readouts, dtype=torch.float64) for key in dynamics_keys
    }
    dynamics_count = torch.zeros(len(depths), readouts, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(seed)

    for depth_index, depth in enumerate(depths):
        fixed_depths = torch.full(
            (batch_size,), depth, device=device, dtype=torch.long
        )
        for _ in range(batches):
            batch = make_boolean_dag_batch(
                data_cfg,
                batch_size,
                device,
                depths=fixed_depths,
                generator=generator,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                output = model.forward_all(
                    batch,
                    max_steps=readouts,
                    return_dynamics=True,
                )
            logits = output["logits_by_step"]
            root_indices = batch.root_mask.long().argmax(dim=1)
            root_logits = logits.gather(
                2,
                root_indices[:, None, None, None].expand(-1, readouts, 1, 3),
            ).squeeze(2)
            targets = (batch.root_values + 1)[:, None].expand(-1, readouts)
            root_correct[depth_index] += root_logits.argmax(dim=-1).eq(targets).sum(dim=0)
            root_probability[depth_index] += root_logits.float().softmax(dim=-1).gather(
                2, targets[:, :, None]
            ).squeeze(2).sum(dim=0)
            root_count[depth_index] += batch_size

            dynamics = summarize_dynamics_tensors(
                output,
                groups=max(1, model_cfg.outer_norm_groups),
            )
            for loop_index, row in enumerate(dynamics):
                for key in dynamics_keys:
                    dynamics_sums[key][depth_index, loop_index] += float(row[key])
                dynamics_count[depth_index, loop_index] += 1

    accuracy = (root_correct / root_count.clamp_min(1)).cpu()
    probability = (root_probability / root_count.clamp_min(1)).cpu()
    dynamics_means = {
        key: values / dynamics_count.clamp_min(1) for key, values in dynamics_sums.items()
    }
    behavior_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(depths):
        for loop_index in range(readouts):
            behavior_rows.append(
                {
                    "depth": depth,
                    "loop": loop_index + 1,
                    "root_accuracy": float(accuracy[depth_index, loop_index]),
                    "correct_probability": float(probability[depth_index, loop_index]),
                }
            )
            dynamics_rows.append(
                {
                    "depth": depth,
                    "loop": loop_index + 1,
                    **{
                        key: float(dynamics_means[key][depth_index, loop_index])
                        for key in dynamics_keys
                    },
                }
            )

    train_indices = [
        index for index, depth in enumerate(depths) if depth <= data_cfg.train_max_depth
    ]
    if not train_indices:
        raise ValueError("evaluation must include at least one training-distribution depth")
    horizon_index = min(model_cfg.steps, readouts) - 1
    loop4_accuracy = float(accuracy[train_indices, horizon_index].mean())
    loop4_probability = float(probability[train_indices, horizon_index].mean())
    late_start = min(model_cfg.steps, readouts)
    late_accuracy = accuracy[train_indices, late_start:]
    overloop_degradation = (
        max(0.0, loop4_accuracy - float(late_accuracy.min()))
        if late_accuracy.numel()
        else 0.0
    )
    overloop_retention = (
        float(late_accuracy.mean()) if late_accuracy.numel() else loop4_accuracy
    )
    late_dynamics_rows = [
        row
        for row in dynamics_rows
        if row["depth"] <= data_cfg.train_max_depth and row["loop"] > model_cfg.steps
    ]
    if not late_dynamics_rows:
        late_dynamics_rows = [
            row
            for row in dynamics_rows
            if row["depth"] <= data_cfg.train_max_depth
            and row["loop"] == min(model_cfg.steps, readouts)
        ]
    late_effective_update_norm = float(
        np.mean([row["effective_update_norm"] for row in late_dynamics_rows])
    )
    late_tangential_fraction = float(
        np.mean([row["tangential_fraction"] for row in late_dynamics_rows])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "behavior.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(behavior_rows[0]))
        writer.writeheader()
        writer.writerows(behavior_rows)
    write_dynamics_rows(
        dynamics_rows,
        csv_path=out_dir / "dynamics.csv",
        json_path=out_dir / "dynamics.json",
    )
    _save_heatmap(
        accuracy.numpy(),
        depths=depths,
        readouts=readouts,
        title="Root accuracy by DAG depth and recurrent loop",
        path=out_dir / "root_accuracy.png",
    )
    _save_dynamics_plot(
        dynamics_rows,
        trained_horizon=model_cfg.steps,
        path=out_dir / "dynamics.png",
    )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "parameter_count": checkpoint_data.get("parameter_count"),
        "data_config": checkpoint_data["data_config"],
        "model_config": checkpoint_data["model_config"],
        "depths": depths,
        "readouts": readouts,
        "examples_per_depth": batches * batch_size,
        "root_accuracy": accuracy.tolist(),
        "correct_probability": probability.tolist(),
        "behavior_rows": behavior_rows,
        "dynamics_rows": dynamics_rows,
        "pilot_metrics": {
            "loop4_accuracy": loop4_accuracy,
            "loop4_correct_probability": loop4_probability,
            "overloop_retention": overloop_retention,
            "overloop_degradation": overloop_degradation,
            "late_effective_update_norm": late_effective_update_norm,
            "late_tangential_fraction": late_tangential_fraction,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
