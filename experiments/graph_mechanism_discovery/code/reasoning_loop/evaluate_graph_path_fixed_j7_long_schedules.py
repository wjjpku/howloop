"""Evaluate a fixed J_7 rollback map on literal long F/J schedules.

The controller was trained only on legal in-horizon bridges, where
``J_7`` denotes the H8 -> H7 map.  This evaluator deliberately reuses that
same map after states H9, H10, ... .  It therefore measures a genuine
long-rollout/OOD maintenance behaviour, not a new in-distribution age path.

F advances the frozen shared Transformer by one loop and advances the graph
target by the corresponding phase jump.  J changes only the residual stream:
it never advances the graph target.  All curves are read immediately after F,
so H9 and H10 mean the first and second overloop states reached from H8.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _repeat_pattern(pattern: tuple[str, ...]) -> Iterable[str]:
    while True:
        yield from pattern


def schedule_actions(
    *,
    family: str,
    max_forwards: int,
    prefix_forwards: int = 0,
) -> tuple[str, ...]:
    """Return literal F/J actions, stopping just after ``max_forwards`` Fs."""

    if max_forwards < 1:
        raise ValueError("max_forwards must be positive")
    patterns = {
        "F_only": ("F",),
        "FFJ": ("F", "F", "J"),
        "prefix_FJ": ("F", "J"),
        "FFJJ": ("F", "F", "J", "J"),
        "FFFFJJJJ": ("F", "F", "F", "F", "J", "J", "J", "J"),
        "F7J7": ("F",) * 7 + ("J",) * 7,
    }
    if family not in patterns:
        raise ValueError(f"unknown schedule family: {family}")
    if family != "prefix_FJ" and prefix_forwards:
        raise ValueError("only prefix_FJ accepts a nonzero prefix")

    actions: list[str] = []
    forwards = 0
    if family == "prefix_FJ":
        if prefix_forwards < 1:
            raise ValueError("prefix_FJ needs a positive F prefix")
        for _ in range(min(prefix_forwards, max_forwards)):
            actions.append("F")
            forwards += 1
    for action in _repeat_pattern(patterns[family]):
        if action == "F" and forwards >= max_forwards:
            break
        actions.append(action)
        if action == "F":
            forwards += 1
            if forwards == max_forwards:
                break
    return tuple(actions)


def schedule_specs(
    *, max_forwards: int, prefix_lengths: tuple[int, ...]
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    specs: list[tuple[str, str, tuple[str, ...]]] = [
        ("F_only", "F F F ... (no-J control)", schedule_actions(
            family="F_only", max_forwards=max_forwards
        )),
        ("FFJ", "(F F J)^infinity", schedule_actions(
            family="FFJ", max_forwards=max_forwards
        )),
    ]
    for prefix in prefix_lengths:
        specs.append(
            (
                f"F{prefix}_then_FJ",
                f"F^{prefix} then (F J)^infinity",
                schedule_actions(
                    family="prefix_FJ",
                    prefix_forwards=prefix,
                    max_forwards=max_forwards,
                ),
            )
        )
    specs.extend(
        (
            ("FFJJ", "(F F J J)^infinity", schedule_actions(
                family="FFJJ", max_forwards=max_forwards
            )),
            ("FFFFJJJJ", "(F F F F J J J J)^infinity", schedule_actions(
                family="FFFFJJJJ", max_forwards=max_forwards
            )),
            ("F7J7", "(F^7 J^7)^infinity", schedule_actions(
                family="F7J7", max_forwards=max_forwards
            )),
        )
    )
    return tuple(specs)


def _metrics(model, state: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    logits = logits_from_raw_state(model, state)
    probabilities = logits.softmax(dim=-1)
    target_probability = probabilities.gather(1, target[:, None]).squeeze(1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    answer = state[:, -1, :].float()
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(target).float().mean()),
        "target_probability": float(target_probability.mean()),
        "readout_entropy": float(entropy.mean()),
        "state_rms": float(
            state.float().flatten(1).norm(dim=-1).mean()
            / math.sqrt(state.shape[1] * state.shape[2])
        ),
        "answer_norm": float(answer.norm(dim=-1).mean()),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["schedule"], row["schedule_text"], row["forward_count"], row["state"])
        ].append(row)
    summary: list[dict[str, Any]] = []
    metric_names = ("accuracy", "target_probability", "readout_entropy", "state_rms", "answer_norm")
    for (schedule, schedule_text, forward_count, state), parts in grouped.items():
        output: dict[str, Any] = {
            "schedule": schedule,
            "schedule_text": schedule_text,
            "forward_count": forward_count,
            "state": state,
            "observations": len(parts),
        }
        for metric in metric_names:
            values = np.asarray([float(part[metric]) for part in parts])
            output[f"{metric}_mean"] = float(values.mean())
            output[f"{metric}_sem"] = float(
                values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        summary.append(output)
    return sorted(summary, key=lambda row: (row["schedule"], row["state"]))


def _plot(summary: list[dict[str, Any]], path: Path) -> None:
    figure, (acc_axis, norm_axis) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=180)
    schedule_order = []
    for row in summary:
        if row["schedule"] not in schedule_order:
            schedule_order.append(row["schedule"])
    for schedule in schedule_order:
        rows = [row for row in summary if row["schedule"] == schedule]
        states = np.asarray([row["state"] for row in rows])
        accuracy = np.asarray([row["accuracy_mean"] for row in rows])
        sem = np.asarray([row["accuracy_sem"] for row in rows])
        label = rows[0]["schedule_text"]
        (line,) = acc_axis.plot(states, accuracy, marker="o", markersize=3, label=label)
        acc_axis.fill_between(states, accuracy - sem, accuracy + sem, color=line.get_color(), alpha=0.13)
        norm_axis.plot(states, [row["state_rms_mean"] for row in rows], marker="o", markersize=3, label=label)
    for axis in (acc_axis, norm_axis):
        axis.axvline(9, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        axis.axvline(10, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        axis.set_xlabel("external state number (H8 is the common initial state)")
        axis.grid(alpha=0.22)
    acc_axis.set_ylabel("graph-path accuracy after F")
    acc_axis.set_ylim(-0.03, 1.03)
    acc_axis.set_title("Long rollout accuracy; dashed lines: H9 and H10")
    norm_axis.set_ylabel("full residual RMS")
    norm_axis.set_title("Residual scale during the same rollouts")
    handles, labels = acc_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=826901)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--examples-per-seed", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-forwards", type=int, default=32)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=(1, 2, 4, 7))
    parser.add_argument(
        "--rollback-source-age", type=int, default=8,
        help="8 denotes J_7, the learned H8 -> H7 map.",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    if args.rollback_source_age != 8:
        raise ValueError("this experiment is specifically the fixed J_7 (H8 -> H7) test")
    if args.max_forwards < 2:
        raise ValueError("at least two forwards are needed to inspect H9 and H10")
    if any(prefix < 1 for prefix in args.prefix_lengths):
        raise ValueError("all prefix lengths must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase_payload["trajectory_positions_including_initial"]]
    if len(phase_positions) <= 8:
        raise ValueError("phase summary must include H1 through H8")
    jump = phase_positions[2] - phase_positions[1]
    if jump <= 0:
        raise ValueError("phase positions must have a positive one-loop graph jump")

    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    if payload.get("checkpoint") != str(args.checkpoint):
        raise ValueError("J artifact and frozen backbone checkpoint differ")
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=payload.get("map_architecture", "diagonal_lora"),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    bank = bank.frozen()
    positions = tuple(range(cfg.seq_len))
    specs = schedule_specs(
        max_forwards=args.max_forwards,
        prefix_lengths=tuple(args.prefix_lengths),
    )

    detail_rows: list[dict[str, Any]] = []
    for seed_offset in args.evaluation_seeds:
        data_seed = args.seed + int(seed_offset)
        set_seed(data_seed)
        for batch_index in range(args.examples_per_seed // args.batch_size):
            _, path_targets, successors, _ = fixed_depth_batch(
                cfg, args.batch_size, device, path_positions=cfg.max_depth
            )
            endpoint = path_targets[:, cfg.max_depth - 1]
            h8 = _aligned_state_at_age(
                model=model,
                cfg=cfg,
                successors=successors,
                current=endpoint,
                age=8,
                phase_position=phase_positions[8],
            )
            for schedule, schedule_text, actions in specs:
                state = h8.clone()
                current = endpoint.clone()
                initial = _metrics(model, state, current)
                detail_rows.append(
                    {
                        "schedule": schedule,
                        "schedule_text": schedule_text,
                        "seed": data_seed,
                        "batch": batch_index,
                        "event": "initial_H8",
                        "action_index": 0,
                        "forward_count": 0,
                        "j7_count": 0,
                        "state": 8,
                        "raw_loop_index": 7,
                        **initial,
                    }
                )
                forward_count = 0
                j7_count = 0
                for action_index, action in enumerate(actions, start=1):
                    if action == "F":
                        # H8 is produced by loop_index=7.  Its next state H9
                        # therefore uses loop_index=8, then H10 uses 9, etc.
                        raw_loop_index = 8 + forward_count
                        state = model.apply_loop(state, loop_index=raw_loop_index)
                        current = advance_nodes(successors, current, steps=jump)
                        forward_count += 1
                        event = "after_F"
                        visible_state = 8 + forward_count
                    else:
                        state = bank.rollback(
                            state,
                            source_age=args.rollback_source_age,
                            positions=positions,
                        )
                        j7_count += 1
                        event = "after_J7"
                        # J is intentionally not interpreted as a true age
                        # decrement once the schedule is beyond H8.
                        visible_state = 8 + forward_count
                        raw_loop_index = 7 + forward_count
                    metrics = _metrics(model, state, current)
                    detail_rows.append(
                        {
                            "schedule": schedule,
                            "schedule_text": schedule_text,
                            "seed": data_seed,
                            "batch": batch_index,
                            "event": event,
                            "action_index": action_index,
                            "forward_count": forward_count,
                            "j7_count": j7_count,
                            "state": visible_state,
                            "raw_loop_index": raw_loop_index,
                            **metrics,
                        }
                    )

    forward_rows = [row for row in detail_rows if row["event"] in {"initial_H8", "after_F"}]
    summary_rows = _summarize(forward_rows)
    key_states = {8, 9, 10, 12, 16, 20, 24, 32, 40}
    key_rows = [row for row in summary_rows if row["state"] in key_states]
    _write_csv(args.out_dir / "long_schedule_per_batch.csv", detail_rows)
    _write_csv(args.out_dir / "long_schedule_forward_summary.csv", summary_rows)
    _write_csv(args.out_dir / "long_schedule_key_states.csv", key_rows)
    _plot(summary_rows, args.out_dir / "long_schedule_accuracy_and_norm.png")
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "backbone_loss_placement": "final-only CE at H8",
        "initial_state": "exact aligned H8 on the same random graph/current target",
        "F_semantics": "one frozen shared stack and one graph-target phase jump",
        "J_semantics": "fixed J_7 = learned H8->H7 map; residual-only; target unchanged",
        "OOD_boundary": "every J_7 after H9 is an out-of-training-range reuse",
        "phase_jump_per_F": jump,
        "max_forward_steps_after_H8": args.max_forwards,
        "evaluation_seeds": list(args.evaluation_seeds),
        "examples_per_seed": args.examples_per_seed,
        "batch_size": args.batch_size,
        "schedules": [
            {"name": name, "text": text, "actions": "".join(actions)}
            for name, text, actions in specs
        ],
        "key_state_rows": key_rows,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"key_state_rows": key_rows, "status": "complete"}, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
