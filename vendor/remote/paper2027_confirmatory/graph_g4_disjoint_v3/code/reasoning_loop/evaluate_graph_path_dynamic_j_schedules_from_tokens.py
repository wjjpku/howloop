"""Evaluate literal long F/J schedules from a natural H1 residual.

H1 is produced by running the raw input-token residual through the first
frozen Transformer loop.  The requested schedule begins after that point.
The logical phase is updated by +1 under F and -1 under J.  Inside H2..H8,
the controller is selected by the current phase; above H8 it is clamped to J7
(the learned H8->H7 map).  Starting at H1 makes all equal-length F/J motifs
close exactly inside the trained H1..H8 phase range.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.evaluate_graph_path_fixed_j7_long_schedules import _metrics
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank


def repeated_actions(
    *,
    motif: tuple[str, ...],
    max_forwards: int,
    prefix_forwards: int = 0,
) -> tuple[str, ...]:
    if max_forwards < 1 or prefix_forwards < 0:
        raise ValueError("invalid forward count")
    actions = ["F"] * min(prefix_forwards, max_forwards)
    forwards = len(actions)
    while forwards < max_forwards:
        for action in motif:
            actions.append(action)
            if action == "F":
                forwards += 1
                if forwards == max_forwards:
                    return tuple(actions)
    return tuple(actions)


def schedule_specs(
    *, scheduled_forwards: int, prefix_lengths: tuple[int, ...]
) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    specs: list[tuple[str, str, str, tuple[str, ...]]] = [
        (
            "F_only",
            "F F F ... (no-J control)",
            "raw_control",
            repeated_actions(motif=("F",), max_forwards=scheduled_forwards),
        ),
        (
            "FFJ",
            "(F F J)^infinity; dynamic J, cap at J7",
            "literal",
            repeated_actions(motif=("F", "F", "J"), max_forwards=scheduled_forwards),
        ),
    ]
    for prefix in prefix_lengths:
        specs.append(
            (
                f"F{prefix}_then_FJ",
                f"F^{prefix} then (F J)^infinity; dynamic J",
                "literal",
                repeated_actions(
                    motif=("F", "J"),
                    prefix_forwards=prefix,
                    max_forwards=scheduled_forwards,
                ),
            )
        )
    motifs = (
        ("FFJJ", ("F", "F", "J", "J")),
        ("FFFFJJJJ", ("F", "F", "F", "F", "J", "J", "J", "J")),
        ("F7J7", ("F",) * 7 + ("J",) * 7),
    )
    for name, motif in motifs:
        specs.append(
            (
                name,
                f"({name})^infinity; dynamic J from natural H1",
                "trained_phase_range",
                repeated_actions(motif=motif, max_forwards=scheduled_forwards),
            )
        )
    return tuple(specs)


def controller_for_phase(logical_phase: int) -> tuple[int, int, str]:
    """Return source age, user-facing J index, and domain label."""

    if logical_phase < 2:
        raise ValueError("no learned rollback exists below H1")
    if logical_phase > 8:
        return 8, 7, "high_boundary_J7_cap"
    return logical_phase, logical_phase - 1, "trained_adjacent_phase"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["schedule"], row["schedule_text"], row["boundary_policy"], row["forward_count"])
        ].append(row)
    output: list[dict[str, Any]] = []
    metrics = ("accuracy", "target_probability", "readout_entropy", "state_rms", "answer_norm")
    for (schedule, text, policy, forward_count), values in groups.items():
        item: dict[str, Any] = {
            "schedule": schedule,
            "schedule_text": text,
            "boundary_policy": policy,
            "forward_count": forward_count,
            "logical_phase": int(values[0]["logical_phase"]),
            "j_count": int(values[0]["j_count"]),
            "observations": len(values),
        }
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values])
            item[f"{metric}_mean"] = float(array.mean())
            item[f"{metric}_sem"] = float(
                array.std(ddof=1) / math.sqrt(len(array)) if len(array) > 1 else 0.0
            )
        output.append(item)
    return sorted(output, key=lambda row: (row["schedule"], row["forward_count"]))


def _j_usage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedules = sorted({row["schedule"] for row in rows})
    output: list[dict[str, Any]] = []
    for schedule in schedules:
        selected = [row for row in rows if row["schedule"] == schedule and row["event"] == "after_J"]
        counts = Counter((int(row["user_j_index"]), row["j_domain"]) for row in selected)
        for (j_index, domain), count in sorted(counts.items()):
            output.append(
                {
                    "schedule": schedule,
                    "user_j_index": j_index,
                    "domain": domain,
                    "batch_level_calls": count,
                }
            )
    return output


def _plot(summary: list[dict[str, Any]], path: Path) -> None:
    figure, (acc_axis, phase_axis) = plt.subplots(1, 2, figsize=(16, 5.8), dpi=180)
    order: list[str] = []
    for row in summary:
        if row["schedule"] not in order:
            order.append(row["schedule"])
    for schedule in order:
        rows = [row for row in summary if row["schedule"] == schedule]
        x = np.asarray([row["forward_count"] for row in rows])
        y = np.asarray([row["accuracy_mean"] for row in rows])
        sem = np.asarray([row["accuracy_sem"] for row in rows])
        label = rows[0]["schedule_text"]
        (line,) = acc_axis.plot(x, y, marker="o", markersize=2.6, linewidth=1.2, label=label)
        acc_axis.fill_between(x, y - sem, y + sem, color=line.get_color(), alpha=0.10)
        phase_axis.plot(x, [row["logical_phase"] for row in rows], marker="o", markersize=2.6, linewidth=1.2, label=label)
    for axis in (acc_axis, phase_axis):
        axis.axvline(9, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        axis.axvline(10, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        axis.grid(alpha=0.22)
        axis.set_xlabel("physical hidden-state index (natural H1 is the start)")
    acc_axis.set_ylim(-0.03, 1.03)
    acc_axis.set_ylabel("diagnostic graph-path accuracy after F")
    acc_axis.set_title("Dynamic-J long schedules from natural H1")
    phase_axis.set_ylabel("logical phase after the F output")
    phase_axis.set_title("Schedule-induced phase trajectory")
    handles, labels = acc_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    figure.tight_layout(rect=(0, 0.22, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=827401)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--examples-per-seed", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-state", type=int, default=40)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=(1, 2, 4, 7))
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    if args.max_state < 10:
        raise ValueError("max-state must cover H9 and H10")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload = torch.load(args.bank_artifact, map_location="cpu", weights_only=False)
    if payload.get("checkpoint") != str(args.checkpoint):
        raise ValueError("J artifact and backbone checkpoint differ")
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
        scheduled_forwards=args.max_state - 1,
        prefix_lengths=tuple(args.prefix_lengths),
    )
    rows: list[dict[str, Any]] = []
    for seed_offset in args.evaluation_seeds:
        data_seed = args.seed + int(seed_offset)
        set_seed(data_seed)
        for batch_index in range(args.examples_per_seed // args.batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, args.batch_size, device, path_positions=cfg.max_depth
            )
            h0 = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
            h1 = model.apply_loop(h0, loop_index=0)
            h1_target = advance_nodes(successors, start, steps=1)
            for schedule, text, boundary_policy, actions in specs:
                state = h1.clone()
                current = h1_target.clone()
                forward_count = 1
                logical_phase = 1
                j_count = 0
                rows.append(
                    {
                        "schedule": schedule,
                        "schedule_text": text,
                        "boundary_policy": boundary_policy,
                        "seed": data_seed,
                        "batch": batch_index,
                        "event": "initial_H1",
                        "action_index": 0,
                        "forward_count": forward_count,
                        "logical_phase": logical_phase,
                        "j_count": j_count,
                        "user_j_index": "",
                        "source_age": "",
                        "j_domain": "",
                        **_metrics(model, state, current),
                    }
                )
                for action_index, action in enumerate(actions, start=1):
                    if action == "F":
                        state = model.apply_loop(state, loop_index=forward_count)
                        current = advance_nodes(successors, current, steps=1)
                        forward_count += 1
                        logical_phase += 1
                        row = {
                            "schedule": schedule,
                            "schedule_text": text,
                            "boundary_policy": boundary_policy,
                            "seed": data_seed,
                            "batch": batch_index,
                            "event": "after_F",
                            "action_index": action_index,
                            "forward_count": forward_count,
                            "logical_phase": logical_phase,
                            "j_count": j_count,
                            "user_j_index": "",
                            "source_age": "",
                            "j_domain": "",
                            **_metrics(model, state, current),
                        }
                    else:
                        source_age, user_j_index, domain = controller_for_phase(logical_phase)
                        state = bank.rollback(state, source_age=source_age, positions=positions)
                        logical_phase -= 1
                        j_count += 1
                        row = {
                            "schedule": schedule,
                            "schedule_text": text,
                            "boundary_policy": boundary_policy,
                            "seed": data_seed,
                            "batch": batch_index,
                            "event": "after_J",
                            "action_index": action_index,
                            "forward_count": forward_count,
                            "logical_phase": logical_phase,
                            "j_count": j_count,
                            "user_j_index": user_j_index,
                            "source_age": source_age,
                            "j_domain": domain,
                            **_metrics(model, state, current),
                        }
                    rows.append(row)
    forward_rows = [row for row in rows if row["event"] in {"initial_H1", "after_F"}]
    summary = _summarize(forward_rows)
    key_counts = {1, 2, 4, 6, 8, 9, 10, 12, 16, 24, 32, 40}
    key_rows = [row for row in summary if row["forward_count"] in key_counts]
    usage_rows = _j_usage(rows)
    _write_csv(args.out_dir / "dynamic_j_per_batch.csv", rows)
    _write_csv(args.out_dir / "dynamic_j_forward_summary.csv", summary)
    _write_csv(args.out_dir / "dynamic_j_key_states.csv", key_rows)
    _write_csv(args.out_dir / "dynamic_j_usage.csv", usage_rows)
    _plot(summary, args.out_dir / "dynamic_j_accuracy_and_phase.png")
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "initial_state": "natural H1 obtained by one frozen F from raw input tokens; schedule begins at H1",
        "external_state": "physical hidden-state index including the initialization F that produced H1",
        "logical_phase": "+1 per F and -1 per J",
        "J_selection": "trained phase map in H2..H8; fixed J7 cap above H8",
        "readout_boundary": "backbone supervised only at F8; all other F-count logits are diagnostic",
        "examples": args.examples_per_seed * len(args.evaluation_seeds),
        "schedules": [
            {"name": name, "text": text, "boundary_policy": policy, "actions": "".join(actions)}
            for name, text, policy, actions in specs
        ],
        "key_state_rows": key_rows,
        "j_usage_rows": usage_rows,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "key_state_rows": key_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
