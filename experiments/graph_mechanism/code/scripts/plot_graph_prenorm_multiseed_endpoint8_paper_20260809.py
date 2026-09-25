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

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import set_seed
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def checkpoint(root: Path, seed: int) -> Path:
    return (
        root
        / f"D8_L8_seed{seed}"
        / f"graphpath_N8_D8_d256_B2_L8_seed{seed}"
        / "best.pt"
    )


@torch.no_grad()
def evaluate_seed(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    loops: int,
    data_seed: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint_path, device)
    if (cfg.node_count, cfg.max_depth, cfg.max_loops, cfg.n_layers) != (8, 8, 8, 2):
        raise ValueError(f"unexpected checkpoint config: {vars(cfg)}")

    correct = np.zeros(loops + 1, dtype=np.int64)
    probability_sum = np.zeros(loops + 1, dtype=np.float64)
    examples = 0
    set_seed(data_seed)

    for _ in range(batches):
        tokens, targets, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        endpoint = targets[:, cfg.max_depth - 1]

        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        states = [state]
        for loop_index in range(loops):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            states.append(state)

        for loop_index, selected_state in enumerate(states):
            logits = logits_from_raw_state(model, selected_state)
            probability = logits.softmax(dim=-1)
            correct[loop_index] += int(logits.argmax(dim=-1).eq(endpoint).sum().item())
            probability_sum[loop_index] += float(
                probability.gather(1, endpoint[:, None]).sum().item()
            )
        examples += batch_size

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "examples": examples,
        "endpoint_top1_accuracy": correct / examples,
        "endpoint_mean_probability": probability_sum / examples,
    }


def plot(results: dict[int, dict[str, Any]], output_path: Path) -> None:
    seeds = sorted(results)
    accuracy = np.stack(
        [np.asarray(results[seed]["endpoint_top1_accuracy"]) for seed in seeds]
    )
    figure, axes = plt.subplots(
        len(seeds),
        1,
        figsize=(7.05, 4.8),
        sharex=True,
    )
    image = None
    for row, (axis, seed) in enumerate(zip(axes, seeds, strict=True)):
        image = axis.imshow(
            accuracy[row : row + 1],
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            interpolation="nearest",
            aspect="auto",
        )
        axis.axvline(7.5, color="#ef4444", linestyle="--", linewidth=0.9)
        axis.axvline(8.5, color="#ef4444", linestyle="--", linewidth=0.9)
        axis.set_yticks([])
        axis.set_ylabel(
            f"seed {seed}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8.5,
            labelpad=10,
        )
        axis.tick_params(axis="x", labelsize=7)

    axes[-1].set_xticks(range(accuracy.shape[1]))
    axes[-1].set_xlabel("recurrent loop", fontsize=8.5)
    figure.suptitle(
        "Top-1 accuracy against the trained target $f_G^8(s)$\n"
        "Random permutation graphs; red lines enclose the supervised loop-8 readout",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    assert image is not None
    figure.subplots_adjust(left=0.12, right=0.89, bottom=0.10, top=0.87, hspace=0.43)
    colorbar_axis = figure.add_axes((0.915, 0.14, 0.016, 0.68))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(r"accuracy against $f_G^8(s)$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.savefig(output_path)
    plt.close(figure)


def write_outputs(results: dict[int, dict[str, Any]], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for seed, result in results.items():
        accuracy = np.asarray(result["endpoint_top1_accuracy"])
        probability = np.asarray(result["endpoint_mean_probability"])
        for loop_index in range(accuracy.size):
            rows.append(
                {
                    "seed": seed,
                    "loop": loop_index,
                    "endpoint_top1_accuracy": float(accuracy[loop_index]),
                    "endpoint_mean_probability": float(probability[loop_index]),
                    "examples": int(result["examples"]),
                }
            )
    with (out_dir / "multiseed_endpoint8_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "backbone": "Pre-Norm D8L8, two shared physical blocks, CE only after loop 8",
            "graph_distribution": "unconditioned random permutation graphs from fixed_depth_batch",
            "target": "f_G^8(start)",
            "metric": "top-1 exact match to the eight-step target at every recurrent loop",
            "evaluated_loops": 16,
            "same_evaluation_stream_across_seeds": True,
        },
        "seeds": {
            str(seed): {
                "checkpoint": result["checkpoint"],
                "checkpoint_step": result["checkpoint_step"],
                "examples": result["examples"],
                "endpoint_top1_accuracy": result["endpoint_top1_accuracy"].tolist(),
                "endpoint_mean_probability": result["endpoint_mean_probability"].tolist(),
            }
            for seed, result in results.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_existing_summary(out_dir: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return {
        int(seed): {
            "endpoint_top1_accuracy": np.asarray(
                result["endpoint_top1_accuracy"], dtype=np.float64
            ),
            "endpoint_mean_probability": np.asarray(
                result["endpoint_mean_probability"], dtype=np.float64
            ),
        }
        for seed, result in payload["seeds"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--loops", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=8_092_206)
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="replot summary.json in out-dir without rerunning model evaluation",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_existing:
        plot(
            read_existing_summary(args.out_dir),
            args.out_dir / "multiseed_endpoint8_accuracy.pdf",
        )
        return
    if args.checkpoint_root is None:
        parser.error("--checkpoint-root is required unless --plot-existing is used")
    device = torch.device(args.device)
    results: dict[int, dict[str, Any]] = {}
    for seed in range(6):
        results[seed] = evaluate_seed(
            checkpoint(args.checkpoint_root, seed),
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            loops=args.loops,
            data_seed=args.data_seed,
        )
        print(f"completed seed {seed}", flush=True)
    write_outputs(results, args.out_dir)
    plot(results, args.out_dir / "multiseed_endpoint8_accuracy.pdf")


if __name__ == "__main__":
    main()
