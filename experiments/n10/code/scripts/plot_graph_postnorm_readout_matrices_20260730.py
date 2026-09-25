from __future__ import annotations

import argparse
import csv
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


def checkpoint(root: Path, seed: int, *, postnorm: bool) -> Path:
    family = f"D8_L8_postnorm_seed{seed}" if postnorm else f"D8_L8_seed{seed}"
    return (
        root
        / family
        / f"graphpath_N8_D8_d256_B2_L8_seed{seed}"
        / "best.pt"
    )


@torch.no_grad()
def readout_matrix(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    data_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, cfg, _ = load_checkpoint(checkpoint_path, device)
    if (cfg.node_count, cfg.max_depth, cfg.max_loops) != (8, 8, 8):
        raise ValueError(f"unexpected config in {checkpoint_path}")
    correct = np.zeros((9, 9), dtype=np.float64)
    counts = np.zeros((9, 9), dtype=np.float64)
    position_probability = np.zeros((9, 10), dtype=np.float64)
    example_count = 0
    set_seed(data_seed)
    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        all_targets = torch.cat([start[:, None], targets], dim=1)
        endpoint = all_targets[:, cfg.max_depth]
        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        states = [state]
        for loop_index in range(cfg.max_loops):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            states.append(state)
        for age, selected_state in enumerate(states):
            logits = logits_from_raw_state(model, selected_state)
            probability = logits.softmax(dim=-1)
            prediction = logits.argmax(dim=-1)
            multiplicity = torch.zeros_like(probability)
            multiplicity.scatter_add_(
                1,
                all_targets,
                torch.ones_like(all_targets, dtype=probability.dtype),
            )
            path_probability = probability.gather(1, all_targets)
            path_probability = path_probability / multiplicity.gather(
                1, all_targets
            )
            other_probability = (
                probability * multiplicity.eq(0)
            ).sum(dim=1, keepdim=True)
            distributed = torch.cat(
                [path_probability, other_probability], dim=1
            )
            if not torch.allclose(
                distributed.sum(dim=1),
                torch.ones(
                    distributed.shape[0],
                    device=distributed.device,
                    dtype=distributed.dtype,
                ),
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError("position probability rows do not sum to one")
            position_probability[age] += (
                distributed.sum(dim=0).float().cpu().numpy()
            )
            for position in range(cfg.max_depth + 1):
                target = all_targets[:, position]
                valid = (
                    torch.ones_like(target, dtype=torch.bool)
                    if position == cfg.max_depth
                    else target.ne(endpoint)
                )
                correct[age, position] += float(
                    prediction[valid].eq(target[valid]).sum()
                )
                counts[age, position] += int(valid.sum())
        example_count += batch_size
    return correct / counts, counts, position_probability / example_count


def plot_condition(
    matrices: dict[int, np.ndarray],
    *,
    condition_label: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(15.8, 9.0), dpi=220)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.065,
        right=0.94,
        bottom=0.07,
        top=0.89,
        wspace=0.18,
        hspace=0.28,
    )
    axes = np.asarray(
        [[figure.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
    )
    colorbar_axis = figure.add_subplot(grid[:, 3])
    image = None
    for axis, seed in zip(axes.ravel(), sorted(matrices), strict=True):
        values = matrices[seed]
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            aspect="equal",
        )
        for age in range(values.shape[0]):
            best_position = int(np.argmax(values[age]))
            axis.add_patch(
                plt.Rectangle(
                    (best_position - 0.47, age - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor="#22d3ee",
                    linewidth=1.25,
                )
            )
            for position in range(values.shape[1]):
                value = values[age, position]
                color = "white" if value < 0.58 else "black"
                axis.text(
                    position,
                    age,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=color,
                )
        axis.set_title(f"{condition_label}, seed {seed}")
        axis.set_xticks(range(9))
        axis.set_yticks(range(9))
        axis.set_xlabel(r"decoded graph position $f^p(start)$")
        axis.grid(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"hidden age $h_k$")
    assert image is not None
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("strict readout accuracy")
    figure.suptitle(
        "Final readout applied at every recurrent boundary\n"
        "cyan box = row maximum; non-endpoint columns exclude endpoint coincidences",
        fontsize=12,
    )
    figure.savefig(output_path)
    plt.close(figure)


def plot_position_distribution(
    matrices: dict[int, np.ndarray],
    *,
    condition_label: str,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(17.0, 9.0), dpi=220)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        left=0.06,
        right=0.94,
        bottom=0.07,
        top=0.88,
        wspace=0.18,
        hspace=0.30,
    )
    axes = np.asarray(
        [[figure.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]
    )
    colorbar_axis = figure.add_subplot(grid[:, 3])
    image = None
    xlabels = [rf"$f^{position}$" for position in range(9)] + ["OTHER"]
    for axis, seed in zip(axes.ravel(), sorted(matrices), strict=True):
        values = matrices[seed]
        if not np.allclose(values.sum(axis=1), 1.0, atol=2e-6):
            raise ValueError(f"seed {seed} distribution rows do not sum to one")
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
        )
        for age in range(values.shape[0]):
            best_position = int(np.argmax(values[age]))
            axis.add_patch(
                plt.Rectangle(
                    (best_position - 0.47, age - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor="#f97316",
                    linewidth=1.25,
                )
            )
            for position in range(values.shape[1]):
                value = values[age, position]
                color = "white" if value < 0.52 else "black"
                axis.text(
                    position,
                    age,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=color,
                )
        axis.set_title(f"{condition_label}, seed {seed}")
        axis.set_xticks(range(10), xlabels, fontsize=8)
        axis.set_yticks(range(9))
        axis.set_xlabel("position class")
        axis.grid(False)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"hidden age $h_k$")
    assert image is not None
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("mean readout probability mass")
    figure.suptitle(
        "Row-normalized position-class probability distribution\n"
        "repeated-node mass is split across matching positions; rows sum to 1",
        fontsize=12,
    )
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-root", type=Path, required=True)
    parser.add_argument("--pre-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=7_745_000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for condition, root, postnorm in (
        ("pre_layernorm_warmup500", args.pre_root, False),
        ("post_layernorm_warmup2000", args.post_root, True),
    ):
        matrices: dict[int, np.ndarray] = {}
        distributions: dict[int, np.ndarray] = {}
        for seed in range(6):
            values, counts, position_distribution = readout_matrix(
                checkpoint(root, seed, postnorm=postnorm),
                device=device,
                batch_size=args.batch_size,
                batches=args.batches,
                data_seed=args.data_seed,
            )
            matrices[seed] = values
            distributions[seed] = position_distribution
            for age in range(9):
                for position in range(9):
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "age": age,
                            "path_position": position,
                            "strict_readout_accuracy": float(
                                values[age, position]
                            ),
                            "valid_count": int(counts[age, position]),
                        }
                    )
                for position in range(10):
                    distribution_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "age": age,
                            "position_class": (
                                f"f^{position}" if position < 9 else "OTHER"
                            ),
                            "mean_probability": float(
                                position_distribution[age, position]
                            ),
                            "row_sum": float(
                                position_distribution[age].sum()
                            ),
                            "example_count": args.batch_size * args.batches,
                        }
                    )
            print(f"complete: {condition} seed {seed}", flush=True)
        plot_condition(
            matrices,
            condition_label=(
                "Post-LN, warmup 2000"
                if postnorm
                else "Pre-LN, warmup 500"
            ),
            output_path=args.out_dir / f"{condition}_strict_readout_matrix.png",
        )
        plot_position_distribution(
            distributions,
            condition_label=(
                "Post-LN, warmup 2000"
                if postnorm
                else "Pre-LN, warmup 500"
            ),
            output_path=(
                args.out_dir
                / f"{condition}_position_probability_distribution.png"
            ),
        )
    with (args.out_dir / "strict_readout_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "position_probability_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(distribution_rows[0])
        )
        writer.writeheader()
        writer.writerows(distribution_rows)


if __name__ == "__main__":
    main()
