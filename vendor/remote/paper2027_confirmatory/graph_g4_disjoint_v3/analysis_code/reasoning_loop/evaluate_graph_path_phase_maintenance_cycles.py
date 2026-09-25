"""Test phase-valid rollback-then-forward maintenance cycles from H8.

Unlike an ``F...J`` rescue schedule, each learned rollback is used while the
residual is still at the adjacent source phase on which it was trained.  A
cycle first maps H8 down to H(8-m), then applies m frozen F updates back to
H8.  The graph target advances once per F while the residual phase is restored
to the only final-only readout phase, H8.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.evaluate_graph_path_fixed_j7_long_schedules import _metrics
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank


CYCLES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("J7_F", (8,)),
    ("J7_J6_FF", (8, 7)),
    ("J7_J6_J5_J4_FFFF", (8, 7, 6, 5)),
    ("J7_to_J1_F7", (8, 7, 6, 5, 4, 3, 2)),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["schedule"], row["schedule_text"], row["external_state"])].append(row)
    summary: list[dict[str, Any]] = []
    for (schedule, schedule_text, external_state), parts in grouped.items():
        item: dict[str, Any] = {
            "schedule": schedule,
            "schedule_text": schedule_text,
            "external_state": external_state,
            "observations": len(parts),
        }
        for metric in ("accuracy", "target_probability", "readout_entropy", "state_rms", "answer_norm"):
            values = np.asarray([float(part[metric]) for part in parts])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sem"] = float(
                values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            )
        summary.append(item)
    return sorted(summary, key=lambda row: (row["schedule"], row["external_state"]))


def _plot(summary: list[dict[str, Any]], path: Path) -> None:
    figure, (acc_axis, norm_axis) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    order: list[str] = []
    for row in summary:
        if row["schedule"] not in order:
            order.append(row["schedule"])
    for schedule in order:
        rows = [row for row in summary if row["schedule"] == schedule]
        x = np.asarray([row["external_state"] for row in rows])
        y = np.asarray([row["accuracy_mean"] for row in rows])
        sem = np.asarray([row["accuracy_sem"] for row in rows])
        label = rows[0]["schedule_text"]
        (line,) = acc_axis.plot(x, y, marker="o", markersize=3.5, label=label)
        acc_axis.fill_between(x, y - sem, y + sem, color=line.get_color(), alpha=0.14)
        norm_axis.plot(x, [row["state_rms_mean"] for row in rows], marker="o", markersize=3.5, label=label)
    for axis in (acc_axis, norm_axis):
        axis.axvline(9, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.axvline(10, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        axis.grid(alpha=0.23)
        axis.set_xlabel("external graph state (H8 is the common start)")
    acc_axis.set_ylim(-0.03, 1.03)
    acc_axis.set_ylabel("graph-path accuracy at restored H8 phase")
    acc_axis.set_title("Rollback-then-forward phase maintenance")
    norm_axis.set_ylabel("full residual RMS")
    norm_axis.set_title("Residual scale at restored H8 phase")
    handles, labels = acc_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
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
    parser.add_argument("--seed", type=int, default=827101)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--examples-per-seed", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-forwards", type=int, default=35)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.examples_per_seed % args.batch_size:
        raise ValueError("examples-per-seed must be divisible by batch-size")
    if args.max_forwards < 2:
        raise ValueError("max-forwards must cover H9 and H10")
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
    jump = phase_positions[2] - phase_positions[1]
    if len(phase_positions) <= 8 or jump <= 0:
        raise ValueError("phase summary must cover H1..H8 with positive phase jump")
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

    rows: list[dict[str, Any]] = []
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
            all_specs = (("F_only", (), "raw F continuation"),) + tuple(
                (name, sources, name.replace("_", " ")) for name, sources in CYCLES
            )
            for name, rollback_sources, text in all_specs:
                state = h8.clone()
                current = endpoint.clone()
                rows.append({
                    "schedule": name, "schedule_text": text, "seed": data_seed,
                    "batch": batch_index, "external_state": 8,
                    "forward_count": 0, "rollback_depth": len(rollback_sources),
                    "event": "initial_H8", **_metrics(model, state, current),
                })
                forwards = 0
                if name == "F_only":
                    while forwards < args.max_forwards:
                        state = model.apply_loop(state, loop_index=8 + forwards)
                        current = advance_nodes(successors, current, steps=jump)
                        forwards += 1
                        rows.append({
                            "schedule": name, "schedule_text": text, "seed": data_seed,
                            "batch": batch_index, "external_state": 8 + forwards,
                            "forward_count": forwards, "rollback_depth": 0,
                            "event": "raw_F", **_metrics(model, state, current),
                        })
                    continue
                cycle_size = len(rollback_sources)
                while forwards + cycle_size <= args.max_forwards:
                    for source_age in rollback_sources:
                        state = bank.rollback(
                            state, source_age=source_age, positions=positions
                        )
                    # After J8,...,J(9-m), this sequentially restores H(8-m)
                    # to H8.  The F loop index equals its input logical age.
                    for input_age in range(8 - cycle_size, 8):
                        state = model.apply_loop(state, loop_index=input_age)
                        current = advance_nodes(successors, current, steps=jump)
                        forwards += 1
                    rows.append({
                        "schedule": name, "schedule_text": text, "seed": data_seed,
                        "batch": batch_index, "external_state": 8 + forwards,
                        "forward_count": forwards, "rollback_depth": cycle_size,
                        "event": "restored_H8", **_metrics(model, state, current),
                    })

    summary = _summarize(rows)
    _write_csv(args.out_dir / "phase_maintenance_per_batch.csv", rows)
    _write_csv(args.out_dir / "phase_maintenance_summary.csv", summary)
    _plot(summary, args.out_dir / "phase_maintenance_accuracy_and_norm.png")
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "initial_state": "exact aligned H8 with identical graph/current target",
        "semantics": "J maps run before F, at their legal adjacent source phases; target advances only under F",
        "cycles": [
            {"name": name, "rollback_source_ages": list(sources), "forward_count_per_cycle": len(sources)}
            for name, sources in CYCLES
        ],
        "examples": args.examples_per_seed * len(args.evaluation_seeds),
        "summary_rows": summary,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "summary_rows": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
