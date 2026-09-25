from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_loop import LoopedGraphPathTransformer, pick_device, set_seed
from reasoning_loop.graph_path_stepwise import StepwiseGraphPathConfig, make_stepwise_batch


def build_update_schedule(
    base_loops: int,
    *,
    mode: str,
    intervention_loop: int | None,
) -> list[int]:
    if base_loops < 1:
        raise ValueError("base_loops must be >= 1")
    if mode not in {"baseline", "skip", "repeat"}:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "baseline":
        if intervention_loop is not None:
            raise ValueError("intervention_loop must be omitted for baseline")
        return [1] * base_loops
    if intervention_loop is None or not 1 <= intervention_loop <= base_loops:
        raise ValueError(f"intervention_loop must be in [1, {base_loops}]")
    schedule = [1] * base_loops
    schedule[intervention_loop - 1] = 0 if mode == "skip" else 2
    return schedule


def rollout_targets(
    successors: torch.Tensor,
    start: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    if successors.ndim != 2:
        raise ValueError("successors must have shape [batch, node_count]")
    if start.ndim != 1 or start.shape[0] != successors.shape[0]:
        raise ValueError("start must have shape [batch]")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    targets = [start]
    current = start
    for _ in range(steps):
        current = successors.gather(1, current[:, None]).squeeze(1)
        targets.append(current)
    return torch.stack(targets, dim=1)


def replace_answer_state(receiver_x: torch.Tensor, donor_x: torch.Tensor) -> torch.Tensor:
    if receiver_x.shape != donor_x.shape or receiver_x.ndim != 3:
        raise ValueError("receiver_x and donor_x must have identical [batch, seq, d_model] shapes")
    patched = receiver_x.clone()
    patched[:, -1, :] = donor_x[:, -1, :]
    return patched


def apply_shared_stack(
    model: LoopedGraphPathTransformer,
    x: torch.Tensor,
    *,
    loop_index: int = 0,
) -> torch.Tensor:
    return model.apply_loop(x, loop_index=loop_index)


def logits_from_raw_state(model: LoopedGraphPathTransformer, x: torch.Tensor) -> torch.Tensor:
    answer_state = model.ln_final(x[:, -1, :])
    return model.unembed(answer_state)[:, : model.cfg.node_count]


def run_update_schedule(
    initial_x: torch.Tensor,
    schedule: list[int],
    *,
    update_fn: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    if not schedule or any(update_count < 0 for update_count in schedule):
        raise ValueError("schedule must contain non-negative update counts")
    x = initial_x.clone()
    states: list[torch.Tensor] = []
    for update_count in schedule:
        for _ in range(update_count):
            x = update_fn(x)
        states.append(x)
    return states


def cache_raw_states(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    max_loop: int,
) -> list[torch.Tensor]:
    if max_loop < 1:
        raise ValueError("max_loop must be >= 1")
    x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    states: list[torch.Tensor] = []
    for loop_index in range(max_loop):
        x = apply_shared_stack(model, x, loop_index=loop_index)
        states.append(x)
    return states


def _mean_accuracy_and_probability(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = logits.argmax(dim=-1)
    probs = logits.softmax(dim=-1)
    accuracy = pred.eq(target).float().mean()
    probability = probs.gather(1, target[:, None]).squeeze(1).mean()
    return accuracy, probability


@torch.no_grad()
def cross_time_batch_metrics(
    *,
    model: LoopedGraphPathTransformer,
    donor_tokens: torch.Tensor,
    donor_targets: torch.Tensor,
    receiver_tokens: torch.Tensor,
    receiver_targets: torch.Tensor,
    receiver_successors: torch.Tensor,
    donor_loops: list[int],
    receiver_loops: list[int],
    max_delta: int,
    final_target_position: int,
) -> dict[str, torch.Tensor]:
    if not donor_loops or min(donor_loops) < 1:
        raise ValueError("donor_loops must contain 1-indexed positive values")
    if not receiver_loops or min(receiver_loops) < 1:
        raise ValueError("receiver_loops must contain 1-indexed positive values")
    if max_delta < 0:
        raise ValueError("max_delta must be >= 0")
    if donor_targets.shape[1] < max(max(donor_loops), final_target_position):
        raise ValueError("donor_targets do not cover requested positions")
    if receiver_targets.shape[1] < max(receiver_loops) + max_delta:
        raise ValueError("receiver_targets do not cover receiver_loop + max_delta")

    donor_states = cache_raw_states(model, donor_tokens, max_loop=max(donor_loops))
    receiver_states = cache_raw_states(model, receiver_tokens, max_loop=max(receiver_loops))
    shape = (len(donor_loops), len(receiver_loops), max_delta + 1)
    metrics = {
        "transplant_acc": torch.zeros(shape, device=donor_tokens.device),
        "transplant_prob": torch.zeros(shape, device=donor_tokens.device),
        "receiver_path_acc": torch.zeros(shape, device=donor_tokens.device),
        "receiver_path_prob": torch.zeros(shape, device=donor_tokens.device),
        "donor_final_acc": torch.zeros(shape, device=donor_tokens.device),
        "donor_final_prob": torch.zeros(shape, device=donor_tokens.device),
    }

    donor_final = donor_targets[:, final_target_position - 1]
    for donor_idx, donor_loop in enumerate(donor_loops):
        donor_current = donor_targets[:, donor_loop - 1]
        transplant_targets = rollout_targets(
            receiver_successors,
            donor_current,
            steps=max_delta,
        )
        donor_x = donor_states[donor_loop - 1]
        for receiver_idx, receiver_loop in enumerate(receiver_loops):
            patched_x = replace_answer_state(
                receiver_states[receiver_loop - 1],
                donor_x,
            )
            for delta in range(max_delta + 1):
                logits = logits_from_raw_state(model, patched_x)
                target = transplant_targets[:, delta]
                acc, prob = _mean_accuracy_and_probability(logits, target)
                metrics["transplant_acc"][donor_idx, receiver_idx, delta] = acc
                metrics["transplant_prob"][donor_idx, receiver_idx, delta] = prob

                receiver_target = receiver_targets[:, receiver_loop + delta - 1]
                acc, prob = _mean_accuracy_and_probability(logits, receiver_target)
                metrics["receiver_path_acc"][donor_idx, receiver_idx, delta] = acc
                metrics["receiver_path_prob"][donor_idx, receiver_idx, delta] = prob

                acc, prob = _mean_accuracy_and_probability(logits, donor_final)
                metrics["donor_final_acc"][donor_idx, receiver_idx, delta] = acc
                metrics["donor_final_prob"][donor_idx, receiver_idx, delta] = prob

                if delta < max_delta:
                    patched_x = apply_shared_stack(
                        model,
                        patched_x,
                        loop_index=receiver_loop + delta,
                    )
    return metrics


@torch.no_grad()
def schedule_batch_metrics(
    *,
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    targets_by_position: torch.Tensor,
    start: torch.Tensor,
    base_loops: int,
    max_path_position: int,
) -> dict[str, torch.Tensor | list[str]]:
    if max_path_position < 0 or targets_by_position.shape[1] < max_path_position:
        raise ValueError("targets_by_position do not cover max_path_position")
    condition_names = ["baseline"]
    schedules = [
        build_update_schedule(base_loops, mode="baseline", intervention_loop=None)
    ]
    for mode in ("skip", "repeat"):
        for intervention_loop in range(1, base_loops + 1):
            condition_names.append(f"{mode}_{intervention_loop}")
            schedules.append(
                build_update_schedule(
                    base_loops,
                    mode=mode,
                    intervention_loop=intervention_loop,
                )
            )

    all_targets = torch.cat(
        [start[:, None], targets_by_position[:, :max_path_position]],
        dim=1,
    )
    shape = (len(condition_names), base_loops, max_path_position + 1)
    accuracy = torch.zeros(shape, device=tokens.device)
    probability = torch.zeros(shape, device=tokens.device)
    cumulative_updates = torch.zeros(
        (len(condition_names), base_loops),
        dtype=torch.long,
        device=tokens.device,
    )
    initial_x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    for condition_idx, schedule in enumerate(schedules):
        cumulative_updates[condition_idx] = torch.tensor(
            schedule,
            dtype=torch.long,
            device=tokens.device,
        ).cumsum(dim=0)
        state = initial_x.clone()
        states: list[torch.Tensor] = []
        actual_update_index = 0
        for update_count in schedule:
            for _ in range(update_count):
                state = apply_shared_stack(
                    model,
                    state,
                    loop_index=actual_update_index,
                )
                actual_update_index += 1
            states.append(state)
        for slot_idx, state in enumerate(states):
            logits = logits_from_raw_state(model, state)
            for position in range(max_path_position + 1):
                acc, prob = _mean_accuracy_and_probability(
                    logits,
                    all_targets[:, position],
                )
                accuracy[condition_idx, slot_idx, position] = acc
                probability[condition_idx, slot_idx, position] = prob
    return {
        "condition_names": condition_names,
        "accuracy": accuracy,
        "probability": probability,
        "cumulative_updates": cumulative_updates,
    }


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME=/path/to/checkpoint.pt, got {text!r}")
    name, path = text.split("=", 1)
    if not name:
        raise ValueError("run name must not be empty")
    return name, Path(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(
    values: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    width = max(6.5, 0.75 * len(col_labels) + 2.5)
    height = max(4.5, 0.48 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value < 0.45 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    donor_loops: list[int],
    receiver_loops: list[int],
    max_delta: int,
    batch_size: int,
    batches: int,
    base_loops: int,
    max_path_position: int,
) -> dict[str, Any]:
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be >= 1")
    checkpoint_data = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    cfg = StepwiseGraphPathConfig(**checkpoint_data["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()

    path_positions = max(
        max(donor_loops),
        max(receiver_loops) + max_delta,
        max_path_position,
        cfg.max_depth,
    )
    cross_sums: dict[str, torch.Tensor] | None = None
    schedule_sums: dict[str, torch.Tensor] | None = None
    condition_names: list[str] | None = None
    cumulative_updates: torch.Tensor | None = None
    for _ in range(batches):
        donor_tokens, donor_targets, _, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        receiver_tokens, receiver_targets, receiver_successors, receiver_start = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        cross_batch = cross_time_batch_metrics(
            model=model,
            donor_tokens=donor_tokens,
            donor_targets=donor_targets,
            receiver_tokens=receiver_tokens,
            receiver_targets=receiver_targets,
            receiver_successors=receiver_successors,
            donor_loops=donor_loops,
            receiver_loops=receiver_loops,
            max_delta=max_delta,
            final_target_position=cfg.max_depth,
        )
        schedule_batch = schedule_batch_metrics(
            model=model,
            tokens=receiver_tokens,
            targets_by_position=receiver_targets,
            start=receiver_start,
            base_loops=base_loops,
            max_path_position=max_path_position,
        )
        if cross_sums is None:
            cross_sums = {key: torch.zeros_like(value) for key, value in cross_batch.items()}
            schedule_sums = {
                "accuracy": torch.zeros_like(schedule_batch["accuracy"]),
                "probability": torch.zeros_like(schedule_batch["probability"]),
            }
            condition_names = list(schedule_batch["condition_names"])
            cumulative_updates = schedule_batch["cumulative_updates"].clone()
        for key, value in cross_batch.items():
            cross_sums[key] += value
        assert schedule_sums is not None
        schedule_sums["accuracy"] += schedule_batch["accuracy"]
        schedule_sums["probability"] += schedule_batch["probability"]

    assert cross_sums is not None
    assert schedule_sums is not None
    assert condition_names is not None
    assert cumulative_updates is not None
    cross = {key: value.div(batches).cpu() for key, value in cross_sums.items()}
    schedule = {key: value.div(batches).cpu() for key, value in schedule_sums.items()}
    cumulative_updates = cumulative_updates.cpu()

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    donor_labels = [str(loop) for loop in donor_loops]
    receiver_labels = [str(loop) for loop in receiver_loops]
    for delta in range(max_delta + 1):
        save_heatmap(
            cross["transplant_acc"][:, :, delta].numpy(),
            row_labels=donor_labels,
            col_labels=receiver_labels,
            title=f"{name}: donor variable after cross-time patch, delta={delta}",
            colorbar_label="accuracy",
            path=run_dir / f"cross_time_transplant_acc_delta{delta}.png",
        )
    save_heatmap(
        cross["donor_final_acc"][:, :, 0].numpy(),
        row_labels=donor_labels,
        col_labels=receiver_labels,
        title=f"{name}: donor final answer immediately after patch",
        colorbar_label="accuracy to donor f^D(start)",
        path=run_dir / "cross_time_donor_final_acc_delta0.png",
    )
    save_heatmap(
        schedule["accuracy"][:, -1, :].numpy(),
        row_labels=condition_names,
        col_labels=[f"f^{position}" for position in range(max_path_position + 1)],
        title=f"{name}: final-slot output after skip/repeat",
        colorbar_label="accuracy",
        path=run_dir / "schedule_final_slot_accuracy.png",
    )

    cross_rows: list[dict[str, Any]] = []
    for donor_idx, donor_loop in enumerate(donor_loops):
        for receiver_idx, receiver_loop in enumerate(receiver_loops):
            for delta in range(max_delta + 1):
                cross_rows.append(
                    {
                        "model": name,
                        "donor_loop": donor_loop,
                        "receiver_loop": receiver_loop,
                        "delta": delta,
                        "transplant_acc": float(cross["transplant_acc"][donor_idx, receiver_idx, delta]),
                        "transplant_prob": float(cross["transplant_prob"][donor_idx, receiver_idx, delta]),
                        "receiver_path_acc": float(cross["receiver_path_acc"][donor_idx, receiver_idx, delta]),
                        "donor_final_acc": float(cross["donor_final_acc"][donor_idx, receiver_idx, delta]),
                    }
                )
    write_csv(run_dir / "cross_time_rows.csv", cross_rows)

    schedule_rows: list[dict[str, Any]] = []
    for condition_idx, condition in enumerate(condition_names):
        for slot in range(base_loops):
            for position in range(max_path_position + 1):
                schedule_rows.append(
                    {
                        "model": name,
                        "condition": condition,
                        "slot": slot + 1,
                        "cumulative_updates": int(cumulative_updates[condition_idx, slot]),
                        "path_position": position,
                        "accuracy": float(schedule["accuracy"][condition_idx, slot, position]),
                        "probability": float(schedule["probability"][condition_idx, slot, position]),
                    }
                )
    write_csv(run_dir / "schedule_rows.csv", schedule_rows)

    receiver_sensitivity = cross["transplant_acc"].std(dim=1, unbiased=False)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "loss_mode": checkpoint_data.get("loss_mode", ""),
        "config": asdict(cfg),
        "examples": batch_size * batches,
        "cross_time": {
            "donor_loops": donor_loops,
            "receiver_loops": receiver_loops,
            "max_delta": max_delta,
            "shape": list(cross["transplant_acc"].shape),
            "transplant_acc": cross["transplant_acc"].tolist(),
            "receiver_path_acc": cross["receiver_path_acc"].tolist(),
            "donor_final_acc": cross["donor_final_acc"].tolist(),
            "mean_transplant_acc_by_delta": cross["transplant_acc"].mean(dim=(0, 1)).tolist(),
            "mean_donor_final_acc_by_delta": cross["donor_final_acc"].mean(dim=(0, 1)).tolist(),
            "mean_receiver_sensitivity_by_delta": receiver_sensitivity.mean(dim=0).tolist(),
        },
        "schedule": {
            "condition_names": condition_names,
            "accuracy_shape": list(schedule["accuracy"].shape),
            "cumulative_updates": cumulative_updates.tolist(),
            "final_slot_accuracy": schedule["accuracy"][:, -1, :].tolist(),
            "final_slot_probability": schedule["probability"][:, -1, :].tolist(),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-time transplant and skip/repeat diagnostics for no-depth graph-path models."
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--donor-loops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--receiver-loops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-delta", type=int, default=3)
    parser.add_argument("--base-loops", type=int, default=6)
    parser.add_argument("--max-path-position", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    for name, checkpoint in [parse_run_spec(text) for text in args.run]:
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            donor_loops=args.donor_loops,
            receiver_loops=args.receiver_loops,
            max_delta=args.max_delta,
            batch_size=args.batch_size,
            batches=args.batches,
            base_loops=args.base_loops,
            max_path_position=args.max_path_position,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"models": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
