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

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
    paired_fixed_depth_batch,
)
from reasoning_loop.graph_path_loop import LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
    replace_answer_state,
)


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("run must be NAME=/path/to/checkpoint.pt")
    name, raw_path = text.split("=", 1)
    if not name:
        raise ValueError("run name must not be empty")
    return name, Path(raw_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_state(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
) -> torch.Tensor:
    state = model.token_embed(tokens)
    if model.block_style == "legacy":
        state = state + model.pos_embed.unsqueeze(0)
    return state


def cache_macro_states(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    loops: int,
) -> list[torch.Tensor]:
    state = initial_state(model, tokens)
    states = [state]
    for loop_index in range(loops):
        state = apply_shared_stack(model, state, loop_index=loop_index)
        states.append(state)
    return states


def cache_physical_block_states(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    loops: int,
) -> list[torch.Tensor]:
    state = initial_state(model, tokens)
    states = [state]
    for loop_index in range(loops):
        for block_index in model.active_block_indices(loop_index):
            state = model.blocks[block_index](state)
            states.append(state)
        if model.outer_norm is not None:
            state = model.outer_norm(state)
    return states


def path_accuracy(
    logits: torch.Tensor,
    *,
    start: torch.Tensor,
    targets: torch.Tensor,
    positions: int,
    endpoint_position: int,
) -> list[float]:
    prediction = logits.argmax(dim=-1)
    all_targets = torch.cat((start[:, None], targets[:, :positions]), dim=1)
    endpoint = all_targets[:, endpoint_position]
    values: list[float] = []
    for position in range(positions + 1):
        target = all_targets[:, position]
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if position == endpoint_position
            else target.ne(endpoint)
        )
        values.append(float(prediction[valid].eq(target[valid]).float().mean()))
    return values


def plot_heatmap(
    values: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    path: Path,
) -> None:
    width = max(7.0, 0.7 * len(column_labels) + 2.8)
    height = max(4.5, 0.43 * len(row_labels) + 2.3)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        values,
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
        aspect="auto",
    )
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        maximum = int(np.argmax(values[row]))
        axis.add_patch(
            plt.Rectangle(
                (maximum - 0.47, row - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor="#22d3ee",
                linewidth=1.2,
            )
        )
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 0.58 else "black",
            )
    figure.colorbar(image, ax=axis, label="strict readout accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def intervention_schedules(base_loops: int) -> dict[str, list[int]]:
    schedules = {"baseline": [1] * base_loops}
    for slot in range(base_loops):
        skip = [1] * base_loops
        skip[slot] = 0
        schedules[f"skip_{slot + 1}"] = skip
        repeat = [1] * base_loops
        repeat[slot] = 2
        schedules[f"repeat_{slot + 1}"] = repeat
    return schedules


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    overloops: int,
    seed: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.n_layers != 2 or cfg.max_depth != 8 or cfg.max_loops != 8:
        raise ValueError("expected the D8L8 model with two physical blocks")
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    physical_sum = np.zeros((2 * cfg.max_loops + 1, cfg.max_depth + 1))
    macro_sum = np.zeros((overloops + 1, cfg.max_depth + 1))
    schedule_names = list(intervention_schedules(4))
    schedule_sum = np.zeros((len(schedule_names), cfg.max_depth + 1))
    transplant_immediate_sum = np.zeros((3, 3))
    transplant_next_sum = np.zeros((3, 3))
    donor_full_next_sum = np.zeros(3)

    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=max(cfg.max_depth, 2 * overloops),
        )
        physical_states = cache_physical_block_states(
            model,
            tokens,
            loops=cfg.max_loops,
        )
        for index, state in enumerate(physical_states):
            physical_sum[index] += np.asarray(
                path_accuracy(
                    logits_from_raw_state(model, state),
                    start=start,
                    targets=targets,
                    positions=cfg.max_depth,
                    endpoint_position=cfg.max_depth,
                )
            )

        macro_states = cache_macro_states(model, tokens, loops=overloops)
        for index, state in enumerate(macro_states):
            macro_sum[index] += np.asarray(
                path_accuracy(
                    logits_from_raw_state(model, state),
                    start=start,
                    targets=targets,
                    positions=cfg.max_depth,
                    endpoint_position=cfg.max_depth,
                )
            )

        initial = initial_state(model, tokens)
        for condition_index, schedule in enumerate(
            intervention_schedules(4).values()
        ):
            state = initial.clone()
            actual_loop = 0
            for update_count in schedule:
                for _ in range(update_count):
                    state = apply_shared_stack(
                        model,
                        state,
                        loop_index=actual_loop,
                    )
                    actual_loop += 1
            schedule_sum[condition_index] += np.asarray(
                path_accuracy(
                    logits_from_raw_state(model, state),
                    start=start,
                    targets=targets,
                    positions=cfg.max_depth,
                    endpoint_position=cfg.max_depth,
                )
            )

        (
            donor_tokens,
            donor_targets,
            receiver_tokens,
            _,
            _,
            _,
        ) = paired_fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        donor_states = cache_macro_states(model, donor_tokens, loops=4)
        receiver_states = cache_macro_states(model, receiver_tokens, loops=3)
        for donor_index, donor_loop in enumerate((1, 2, 3)):
            immediate_target = donor_targets[:, 2 * donor_loop - 1]
            next_target = donor_targets[:, 2 * donor_loop + 1]
            donor_next = apply_shared_stack(
                model,
                donor_states[donor_loop],
                loop_index=donor_loop,
            )
            donor_full_next_sum[donor_index] += float(
                logits_from_raw_state(model, donor_next)
                .argmax(dim=-1)
                .eq(next_target)
                .float()
                .mean()
            )
            for receiver_index, receiver_loop in enumerate((1, 2, 3)):
                patched = replace_answer_state(
                    receiver_states[receiver_loop],
                    donor_states[donor_loop],
                )
                transplant_immediate_sum[donor_index, receiver_index] += float(
                    logits_from_raw_state(model, patched)
                    .argmax(dim=-1)
                    .eq(immediate_target)
                    .float()
                    .mean()
                )
                continued = apply_shared_stack(
                    model,
                    patched,
                    loop_index=receiver_loop,
                )
                transplant_next_sum[donor_index, receiver_index] += float(
                    logits_from_raw_state(model, continued)
                    .argmax(dim=-1)
                    .eq(next_target)
                    .float()
                    .mean()
                )

    physical = physical_sum / batches
    macro = macro_sum / batches
    schedule = schedule_sum / batches
    transplant_immediate = transplant_immediate_sum / batches
    transplant_next = transplant_next_sum / batches
    donor_full_next = donor_full_next_sum / batches

    path_labels = [f"f^{position}" for position in range(cfg.max_depth + 1)]
    physical_labels = ["h0"] + [
        f"L{(index - 1) // 2 + 1}.B{(index - 1) % 2 + 1}"
        for index in range(1, 2 * cfg.max_loops + 1)
    ]
    macro_labels = [f"h{index}" for index in range(overloops + 1)]
    plot_heatmap(
        physical,
        row_labels=physical_labels,
        column_labels=path_labels,
        title=f"{name}: readout after every physical block",
        path=run_dir / "physical_block_trajectory.png",
    )
    plot_heatmap(
        macro,
        row_labels=macro_labels,
        column_labels=path_labels,
        title=f"{name}: natural recurrent trajectory and overloop",
        path=run_dir / "macro_overloop_trajectory.png",
    )
    plot_heatmap(
        schedule,
        row_labels=schedule_names,
        column_labels=path_labels,
        title=f"{name}: four-slot skip/repeat intervention",
        path=run_dir / "skip_repeat_final_readout.png",
    )
    plot_heatmap(
        transplant_next,
        row_labels=["donor h1", "donor h2", "donor h3"],
        column_labels=["receiver h1", "receiver h2", "receiver h3"],
        title=f"{name}: donor answer-state causal continuation by one loop",
        path=run_dir / "answer_state_transplant_next.png",
    )

    trajectory_rows: list[dict[str, Any]] = []
    for state_index, row in enumerate(physical):
        for position, accuracy in enumerate(row):
            trajectory_rows.append(
                {
                    "model": name,
                    "state_index": state_index,
                    "state_label": physical_labels[state_index],
                    "path_position": position,
                    "accuracy": float(accuracy),
                }
            )
    write_csv(run_dir / "physical_block_trajectory.csv", trajectory_rows)

    macro_rows: list[dict[str, Any]] = []
    for loop_index, row in enumerate(macro):
        for position, accuracy in enumerate(row):
            macro_rows.append(
                {
                    "model": name,
                    "loop": loop_index,
                    "path_position": position,
                    "accuracy": float(accuracy),
                }
            )
    write_csv(run_dir / "macro_overloop_trajectory.csv", macro_rows)

    schedule_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(schedule_names):
        updates = sum(intervention_schedules(4)[condition])
        for position, accuracy in enumerate(schedule[condition_index]):
            schedule_rows.append(
                {
                    "model": name,
                    "condition": condition,
                    "actual_updates": updates,
                    "path_position": position,
                    "accuracy": float(accuracy),
                }
            )
    write_csv(run_dir / "skip_repeat_final_readout.csv", schedule_rows)

    transplant_rows: list[dict[str, Any]] = []
    for donor_index, donor_loop in enumerate((1, 2, 3)):
        for receiver_index, receiver_loop in enumerate((1, 2, 3)):
            transplant_rows.append(
                {
                    "model": name,
                    "donor_loop": donor_loop,
                    "receiver_loop": receiver_loop,
                    "immediate_donor_position_accuracy": float(
                        transplant_immediate[donor_index, receiver_index]
                    ),
                    "next_donor_position_accuracy": float(
                        transplant_next[donor_index, receiver_index]
                    ),
                    "donor_full_state_next_accuracy": float(
                        donor_full_next[donor_index]
                    ),
                }
            )
    write_csv(run_dir / "answer_state_transplant.csv", transplant_rows)

    expected_physical = [
        float(physical[index, min(index, cfg.max_depth)])
        for index in range(1, 2 * cfg.max_loops + 1)
    ]
    expected_macro = [
        float(macro[index, min(2 * index, cfg.max_depth)])
        for index in range(1, cfg.max_loops + 1)
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": vars(cfg),
        "examples": batch_size * batches,
        "expected_physical_block_accuracy": expected_physical,
        "mean_expected_physical_block_accuracy_first_8": float(
            np.mean(expected_physical[:8])
        ),
        "expected_macro_accuracy": expected_macro,
        "mean_expected_macro_accuracy_first_4": float(np.mean(expected_macro[:4])),
        "endpoint_accuracy_by_macro_loop": macro[:, cfg.max_depth].tolist(),
        "skip_f6_accuracy": {
            name: float(schedule[index, 6])
            for index, name in enumerate(schedule_names)
            if name.startswith("skip_")
        },
        "repeat_endpoint_accuracy": {
            name: float(schedule[index, 8])
            for index, name in enumerate(schedule_names)
            if name.startswith("repeat_")
        },
        "answer_state_transplant_immediate_mean": float(
            transplant_immediate.mean()
        ),
        "answer_state_transplant_next_mean": float(transplant_next.mean()),
        "donor_full_state_next_accuracy": donor_full_next.tolist(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME=/path/to/checkpoint.pt",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--overloops", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries = {}
    for name, checkpoint in map(parse_run_spec, args.run):
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            output_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            overloops=args.overloops,
            seed=args.seed,
        )
        print(f"complete: {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
