from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.boolean_dag_causal import make_leaf_counterfactual
from reasoning_loop.boolean_dag_data import UNKNOWN, BooleanDAGConfig, make_boolean_dag_batch
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.ouro_boolean_dag import OuroBooleanDAG, OuroBooleanDAGConfig


def load_ouro_boolean_dag_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[OuroBooleanDAG, BooleanDAGConfig, OuroBooleanDAGConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_cfg = BooleanDAGConfig(**checkpoint["data_config"])
    model_cfg = OuroBooleanDAGConfig(**checkpoint["model_config"])
    model = OuroBooleanDAG(data_cfg, model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, data_cfg, model_cfg, checkpoint


def _root_values(values: torch.Tensor, root_mask: torch.Tensor) -> torch.Tensor:
    root_slots = root_mask.long().argmax(dim=1)
    return values.gather(
        2,
        root_slots[:, None, None].expand(-1, values.shape[1], 1),
    ).squeeze(2)


def _root_states(states: torch.Tensor, root_mask: torch.Tensor) -> torch.Tensor:
    root_slots = root_mask.long().argmax(dim=1)
    return states.gather(
        2,
        root_slots[:, None, None, None].expand(
            -1, states.shape[1], 1, states.shape[-1]
        ),
    ).squeeze(2)


@torch.no_grad()
def analyze_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    max_loops: int = 8,
    batch_size: int = 256,
    batches: int = 4,
    seed: int = 0,
    device_name: str = "auto",
    amp: bool = True,
) -> dict[str, Any]:
    if max_loops < 1 or batch_size < 2 or batch_size % 2 or batches < 1:
        raise ValueError("max_loops and batches must be positive; batch_size must be positive and even")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(device_name)
    set_seed(seed)
    model, data_cfg, model_cfg, checkpoint = load_ouro_boolean_dag_checkpoint(
        checkpoint_path, device
    )
    generator = torch.Generator(device=device).manual_seed(seed + 20_260_715)
    depth_count = data_cfg.eval_max_depth
    shape = (depth_count, max_loops)
    native_correct = torch.zeros(shape, device=device)
    donor_correct = torch.zeros(shape, device=device)
    masked_leaf_correct = torch.zeros(shape, device=device)
    flip_pair_correct = torch.zeros(shape, device=device)
    flip_prediction_changed = torch.zeros(shape, device=device)
    entropy_sum = torch.zeros(shape, device=device)
    update_rms_sum = torch.zeros(shape, device=device)
    pre_outer_rms_sum = torch.zeros(shape, device=device)
    adjacent_cosine_sum = torch.zeros((depth_count, max_loops - 1), device=device)
    level_correct = torch.zeros((depth_count + 1, max_loops), device=device)
    level_count = torch.zeros(depth_count + 1, device=device)
    native_count = torch.zeros(depth_count, device=device)
    flip_count = torch.zeros(depth_count, device=device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for depth in range(1, depth_count + 1):
        requested_depths = torch.full(
            (batch_size,), depth, device=device, dtype=torch.long
        )
        for _ in range(batches):
            base = make_boolean_dag_batch(
                data_cfg,
                batch_size,
                device,
                depths=requested_depths,
                generator=generator,
            )
            donor = make_leaf_counterfactual(base, generator=generator)
            masked = replace(
                base,
                initial_states=torch.full_like(base.initial_states, UNKNOWN),
            )
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                base_output = model.forward_all(
                    base,
                    max_steps=max_loops,
                    return_dynamics=True,
                )
                donor_logits = model.forward_all(donor, max_steps=max_loops)[
                    "logits_by_step"
                ]
                masked_logits = model.forward_all(masked, max_steps=max_loops)[
                    "logits_by_step"
                ]

            base_root_logits = _root_states(
                base_output["logits_by_step"], base.root_mask
            ).float()
            donor_root_logits = _root_states(donor_logits, donor.root_mask).float()
            masked_root_logits = _root_states(masked_logits, masked.root_mask).float()
            base_predictions = base_root_logits.argmax(dim=-1)
            donor_predictions = donor_root_logits.argmax(dim=-1)
            masked_predictions = masked_root_logits.argmax(dim=-1)
            base_targets = (base.root_values + 1)[:, None]
            donor_targets = (donor.root_values + 1)[:, None]
            native_correct[depth - 1] += base_predictions.eq(base_targets).sum(dim=0)
            donor_correct[depth - 1] += donor_predictions.eq(donor_targets).sum(dim=0)
            masked_leaf_correct[depth - 1] += masked_predictions.eq(base_targets).sum(dim=0)
            native_count[depth - 1] += batch_size
            node_predictions = base_output["logits_by_step"].argmax(dim=-1)
            node_targets = (base.values + 1)[:, None, :]
            for level in range(depth_count + 1):
                at_level = base.levels.eq(level)
                if at_level.any():
                    level_correct[level] += (
                        node_predictions.eq(node_targets) & at_level[:, None, :]
                    ).sum(dim=(0, 2))
                    level_count[level] += at_level.sum()

            root_flipped = base.root_values.ne(donor.root_values)
            if root_flipped.any():
                both_correct = base_predictions.eq(base_targets) & donor_predictions.eq(
                    donor_targets
                )
                flip_pair_correct[depth - 1] += both_correct[root_flipped].sum(dim=0)
                flip_prediction_changed[depth - 1] += base_predictions[root_flipped].ne(
                    donor_predictions[root_flipped]
                ).sum(dim=0)
                flip_count[depth - 1] += root_flipped.sum()

            probabilities = base_root_logits.softmax(dim=-1)
            entropy_sum[depth - 1] += (
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            ).sum(dim=0)
            incoming = _root_states(
                base_output["incoming_by_step"], base.root_mask
            ).float()
            recurrent = _root_states(
                base_output["recurrent_states_by_step"], base.root_mask
            ).float()
            pre_outer = _root_states(
                base_output["pre_outer_by_step"], base.root_mask
            ).float()
            update_rms_sum[depth - 1] += (recurrent - incoming).square().mean(dim=-1).sqrt().sum(dim=0)
            pre_outer_rms_sum[depth - 1] += pre_outer.square().mean(dim=-1).sqrt().sum(dim=0)
            if max_loops > 1:
                adjacent_cosine_sum[depth - 1] += torch.nn.functional.cosine_similarity(
                    recurrent[:, :-1], recurrent[:, 1:], dim=-1
                ).sum(dim=0)

    native_denom = native_count[:, None].clamp_min(1)
    flip_denom = flip_count[:, None].clamp_min(1)
    metrics = {
        "native_accuracy": (native_correct / native_denom).cpu().tolist(),
        "counterfactual_accuracy": (donor_correct / native_denom).cpu().tolist(),
        "leaf_masked_accuracy": (masked_leaf_correct / native_denom).cpu().tolist(),
        "flip_pair_both_correct": (flip_pair_correct / flip_denom).cpu().tolist(),
        "flip_prediction_changed": (flip_prediction_changed / flip_denom).cpu().tolist(),
        "root_logit_entropy": (entropy_sum / native_denom).cpu().tolist(),
        "root_update_rms": (update_rms_sum / native_denom).cpu().tolist(),
        "root_pre_outer_rms": (pre_outer_rms_sum / native_denom).cpu().tolist(),
        "root_adjacent_state_cosine": (
            adjacent_cosine_sum / native_denom if max_loops > 1 else adjacent_cosine_sum
        ).cpu().tolist(),
        "node_level_direct_readout_accuracy": (
            level_correct / level_count[:, None].clamp_min(1)
        ).cpu().tolist(),
        "node_level_examples": level_count.long().cpu().tolist(),
        "examples_per_depth": int(batch_size * batches),
        "flipped_examples_per_depth": flip_count.long().cpu().tolist(),
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "parameter_count": int(
            checkpoint.get("parameter_count", sum(p.numel() for p in model.parameters()))
        ),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "analysis_config": {
            "max_loops": max_loops,
            "batch_size": batch_size,
            "batches": batches,
            "seed": seed,
        },
        "metrics": metrics,
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_heatmap(
        np.asarray(metrics["native_accuracy"]),
        output_dir / "native_depth_loop_accuracy.png",
        title="Native root accuracy",
        value_format=".2f",
    )
    _plot_heatmap(
        np.asarray(metrics["flip_pair_both_correct"]),
        output_dir / "counterfactual_flip_pair_accuracy.png",
        title="Both answers correct after leaf counterfactual",
        value_format=".2f",
    )
    _plot_heatmap(
        np.asarray(metrics["leaf_masked_accuracy"]),
        output_dir / "leaf_masked_accuracy.png",
        title="Accuracy after masking all leaf values",
        value_format=".2f",
    )
    _plot_heatmap(
        np.asarray(metrics["root_update_rms"]),
        output_dir / "root_update_rms.png",
        title="Root-state update RMS",
        value_format=".2f",
        fixed_scale=False,
    )
    _plot_heatmap(
        np.asarray(metrics["node_level_direct_readout_accuracy"]),
        output_dir / "node_level_loop_accuracy.png",
        title="Direct node readout by true DAG level",
        value_format=".2f",
    )
    return summary


def _plot_heatmap(
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    value_format: str,
    fixed_scale: bool = True,
) -> None:
    fig, axis = plt.subplots(
        figsize=(max(6.0, values.shape[1] * 0.75), max(4.5, values.shape[0] * 0.48)),
        constrained_layout=True,
    )
    kwargs = {"vmin": 0.0, "vmax": 1.0} if fixed_scale else {}
    image = axis.imshow(values, cmap="viridis", aspect="auto", **kwargs)
    axis.set(
        xlabel="executed loop count",
        ylabel="DAG depth",
        title=title,
        xticks=np.arange(values.shape[1]),
        yticks=np.arange(values.shape[0]),
        xticklabels=np.arange(1, values.shape[1] + 1),
        yticklabels=np.arange(1, values.shape[0] + 1),
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.45 else "black"
            axis.text(
                column,
                row,
                format(values[row, column], value_format),
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an Ouro Boolean-DAG checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-loops", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_checkpoint(
        args.checkpoint,
        args.output_dir,
        max_loops=args.max_loops,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
        amp=args.amp,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
