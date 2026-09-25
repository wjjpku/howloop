from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import (
    fixed_depth_batch,
    load_checkpoint,
)
from reasoning_loop.graph_path_functional_circuit import (
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_hidden_norm import (
    _aggregate_norms,
    _load_initializer,
    _record_norms,
)
from reasoning_loop.graph_path_telomere_initializer_semantic_readout import (
    permutation_orbit,
    semantic_offsets,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map


CONDITIONS = (
    "no_control",
    "periodic_initializer",
    "R_only",
    "initializer_then_R",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def periodic_initializer_cycles(
    continuation_loops: int,
    *,
    period: int,
) -> tuple[int, ...]:
    if continuation_loops < 1 or period < 1:
        raise ValueError("continuation_loops and period must be positive")
    return tuple(range(1, continuation_loops + 1, period))


def _aggregate_behavior(
    sums: dict[tuple[str, int], dict[str, float]],
    mode_counts: dict[tuple[str, int, int], int],
    *,
    node_count: int,
    jump: int,
) -> list[dict[str, Any]]:
    rows = []
    for (condition, cycle), values in sorted(sums.items()):
        full_count = values["orbit8_count"]
        mode_parts = [
            (offset, mode_counts[(condition, cycle, offset)])
            for offset in range(node_count)
        ]
        mode_offset, mode_count = max(mode_parts, key=lambda part: part[1])
        nonendpoint_count = values["nonendpoint_count"]
        rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "current_target_definition": f"f^({jump}*cycle)(H8_endpoint)",
                "expected_offset_mod_8": (jump * cycle) % node_count,
                "accuracy_all": values["correct_all"] / values["count"],
                "accuracy_orbit8": (
                    values["correct_orbit8"] / full_count
                    if full_count
                    else float("nan")
                ),
                "orbit8_count": int(full_count),
                "accuracy_nonendpoint": (
                    values["correct_nonendpoint"] / nonendpoint_count
                    if nonendpoint_count
                    else float("nan")
                ),
                "nonendpoint_count": int(nonendpoint_count),
                "orbit8_top1_mode_offset": mode_offset,
                "orbit8_top1_mode_fraction": (
                    mode_count / full_count if full_count else float("nan")
                ),
                "off_orbit_fraction": (
                    values["off_orbit"] / values["count"]
                ),
            }
        )
    return rows


def _plot_norm_with_accuracy(
    norm_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    continuation_loops: int,
    trigger_cycles: tuple[int, ...],
    accuracy_field: str,
    filename: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm_index = {
        (
            str(row["condition"]),
            int(row["cycle"]),
            str(row["position_group"]),
        ): row
        for row in norm_rows
        if row["site"] == "loop_output"
    }
    behavior_index = {
        (str(row["condition"]), int(row["cycle"])): row
        for row in behavior_rows
    }
    groups = ("all", "answer", "graph", "metadata")
    condition_labels = {
        "no_control": "No control",
        "periodic_initializer": "Periodic initializer",
        "R_only": "R every loop",
        "initializer_then_R": "Initializer, then R",
    }
    line_colors = {
        "no_control": "#4c78a8",
        "periodic_initializer": "#f58518",
        "R_only": "#54a24b",
        "initializer_then_R": "#e45756",
    }
    natural_h3 = {
        group: norm_index[("natural", 3, group)]["q50"] for group in groups
    }
    natural_h8 = {
        group: norm_index[("natural", 8, group)]["q50"] for group in groups
    }
    figure, axes = plt.subplots(
        len(groups),
        len(CONDITIONS),
        figsize=(19, 14),
        constrained_layout=True,
        sharex=True,
    )
    cycles = np.arange(0, continuation_loops + 1)
    scatter = None
    for row_index, group in enumerate(groups):
        for column, condition in enumerate(CONDITIONS):
            axis = axes[row_index, column]
            parts = [
                norm_index[(condition, cycle, group)]
                for cycle in cycles
            ]
            median = np.array([part["q50"] for part in parts])
            q25 = np.array([part["q25"] for part in parts])
            q75 = np.array([part["q75"] for part in parts])
            axis.fill_between(
                cycles,
                q25,
                q75,
                color=line_colors[condition],
                alpha=0.13,
            )
            axis.plot(
                cycles,
                median,
                color=line_colors[condition],
                linewidth=1.8,
                alpha=0.8,
            )
            point_cycles = np.arange(1, continuation_loops + 1)
            accuracies = np.array(
                [
                    behavior_index[(condition, cycle)][accuracy_field]
                    for cycle in point_cycles
                ]
            )
            scatter = axis.scatter(
                point_cycles,
                median[1:],
                c=accuracies,
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
                s=33,
                edgecolors="black",
                linewidths=0.35,
                zorder=3,
            )
            axis.axhline(
                natural_h3[group],
                color="#777777",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )
            axis.axhline(
                natural_h8[group],
                color="#222222",
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
            )
            if condition == "periodic_initializer":
                for trigger in trigger_cycles:
                    axis.axvline(
                        trigger,
                        color="#9c2f2f",
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.45,
                    )
            if row_index == 0:
                axis.set_title(condition_labels[condition])
            if column == 0:
                axis.set_ylabel(f"{group} token L2 norm")
            if row_index == len(groups) - 1:
                axis.set_xlabel("Continuation loop")
            axis.grid(alpha=0.18)
    if scatter is None:
        raise RuntimeError("accuracy scatter was not created")
    colorbar = figure.colorbar(
        scatter,
        ax=axes,
        location="right",
        shrink=0.72,
        pad=0.01,
    )
    colorbar.set_label(
        "Current-step accuracy"
        + (" (orbit length 8)" if accuracy_field == "accuracy_orbit8" else "")
    )
    figure.suptitle(
        "Hidden-state norm; point color is accuracy for that same loop\n"
        "dashed = natural H3 median, dotted = natural H8 median",
        fontsize=15,
    )
    figure.savefig(out_dir / filename, dpi=180)
    plt.close(figure)


def _plot_accuracy(
    behavior_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    trigger_cycles: tuple[int, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "no_control": "No control",
        "periodic_initializer": "Periodic initializer",
        "R_only": "R every loop",
        "initializer_then_R": "Initializer, then R",
    }
    colors = {
        "no_control": "#4c78a8",
        "periodic_initializer": "#f58518",
        "R_only": "#54a24b",
        "initializer_then_R": "#e45756",
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    metrics = (
        ("accuracy_all", "All graphs: exact current-step node accuracy"),
        ("accuracy_orbit8", "Orbit length 8"),
        ("accuracy_nonendpoint", "Target differs from starting endpoint"),
    )
    for axis, (field, title) in zip(axes, metrics, strict=True):
        for condition in CONDITIONS:
            parts = [
                row for row in behavior_rows if row["condition"] == condition
            ]
            cycles = [int(row["cycle"]) for row in parts]
            values = [float(row[field]) for row in parts]
            axis.plot(
                cycles,
                values,
                color=colors[condition],
                label=labels[condition],
                marker="o",
                markersize=3.5,
                linewidth=1.7,
            )
        for trigger in trigger_cycles:
            axis.axvline(
                trigger,
                color="#9c2f2f",
                linestyle=":",
                linewidth=0.8,
                alpha=0.4,
            )
        axis.set_title(title)
        axis.set_xlabel("Continuation loop")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Current-step accuracy")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Current-step target is f^(2c)(H8 endpoint); "
        "vertical dotted lines mark periodic-I triggers",
        fontsize=13,
    )
    figure.savefig(out_dir / "periodic_current_step_accuracy.png", dpi=180)
    plt.close(figure)


def _plot_semantic_offsets(
    behavior_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    trigger_cycles: tuple[int, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "no_control": "#4c78a8",
        "periodic_initializer": "#f58518",
        "R_only": "#54a24b",
        "initializer_then_R": "#e45756",
    }
    labels = {
        "no_control": "No control",
        "periodic_initializer": "Periodic initializer",
        "R_only": "R every loop",
        "initializer_then_R": "Initializer, then R",
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, condition in zip(axes.flat, CONDITIONS, strict=True):
        parts = [
            row for row in behavior_rows if row["condition"] == condition
        ]
        cycles = [int(row["cycle"]) for row in parts]
        expected = [int(row["expected_offset_mod_8"]) for row in parts]
        actual = [int(row["orbit8_top1_mode_offset"]) for row in parts]
        axis.plot(
            cycles,
            expected,
            color="#222222",
            linestyle="--",
            marker=".",
            label="Expected offset mod 8",
        )
        axis.plot(
            cycles,
            actual,
            color=colors[condition],
            marker="o",
            markersize=4,
            label="Top-1 mode offset",
        )
        for trigger in trigger_cycles:
            axis.axvline(
                trigger,
                color="#9c2f2f",
                linestyle=":",
                linewidth=0.8,
                alpha=0.4,
            )
        axis.set_title(labels[condition])
        axis.set_xlabel("Continuation loop")
        axis.set_ylabel("Forward offset on length-8 orbit")
        axis.set_yticks(range(8))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Length-8 orbit readout; vertical dotted lines mark "
        "periodic-I trigger cycles",
        fontsize=14,
    )
    figure.savefig(out_dir / "periodic_semantic_offset.png", dpi=180)
    plt.close(figure)


@torch.no_grad()
def run_experiment(
    *,
    checkpoint: Path,
    phase_summary_path: Path,
    initializer_artifact: Path,
    feedback_artifact: Path,
    feedback_label: str,
    out_dir: Path,
    device_name: str,
    batch_size: int,
    batches: int,
    continuation_loops: int,
    initializer_period: int,
    seed: int,
    physical_gpu: int | None,
    prelaunch_used_mib: int | None,
    prelaunch_free_mib: int | None,
    declared_peak_gib: float,
    reserve_gib: float,
    shared_gpu: bool,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        fraction = float(
            os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.04")
        )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    if cfg.max_depth != 8 or cfg.max_loops != 8 or cfg.n_layers != 2:
        raise ValueError("experiment is fixed to the D8L8 two-block model")
    phase_summary = json.loads(phase_summary_path.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    jump = phase_positions[3] - phase_positions[2]
    initializer, initializer_positions = _load_initializer(
        initializer_artifact,
        device=device,
    )
    feedback, feedback_positions = _load_map(
        feedback_artifact,
        label=feedback_label,
        device=device,
    )
    expected_positions = intervention_groups(cfg.node_count)[
        "answer_graph_metadata"
    ]
    if (
        initializer_positions != expected_positions
        or feedback_positions != expected_positions
    ):
        raise ValueError("map positions do not match the Block2 interface")
    trigger_cycles = periodic_initializer_cycles(
        continuation_loops,
        period=initializer_period,
    )

    position_groups = explicit_depth_position_groups(cfg.node_count)
    groups = {
        "all": tuple(range(cfg.seq_len)),
        "answer": position_groups["answer"],
        "graph": position_groups["graph"],
        "metadata": position_groups["query_metadata"],
    }
    norm_storage: dict[
        tuple[str, int, str, str],
        list[torch.Tensor],
    ] = defaultdict(list)
    behavior_sums: dict[
        tuple[str, int],
        dict[str, float],
    ] = defaultdict(lambda: defaultdict(float))
    mode_counts: dict[tuple[str, int, int], int] = defaultdict(int)

    set_seed(seed)
    for _ in range(batches):
        tokens, path_targets, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        natural_state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        _record_norms(
            norm_storage,
            condition="natural",
            cycle=0,
            site="loop_output",
            state=natural_state,
            groups=groups,
        )
        for loop_index in range(cfg.max_loops):
            natural_step = run_one_loop(
                model,
                natural_state,
                loop_index=loop_index,
            )
            natural_state = natural_step.state
            _record_norms(
                norm_storage,
                condition="natural",
                cycle=loop_index + 1,
                site="loop_output",
                state=natural_state,
                groups=groups,
            )

        endpoint = path_targets[:, cfg.max_depth - 1]
        orbit, orbit_lengths = permutation_orbit(successors, endpoint)
        orbit8 = orbit_lengths.eq(cfg.node_count)
        states = {
            condition: natural_state.clone() for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            _record_norms(
                norm_storage,
                condition=condition,
                cycle=0,
                site="loop_output",
                state=states[condition],
                groups=groups,
            )

        for cycle in range(1, continuation_loops + 1):
            target = advance_nodes(
                successors,
                endpoint,
                steps=jump * cycle,
            )
            nonendpoint = target.ne(endpoint)
            steps = {}
            for condition in CONDITIONS:
                transform = None
                if (
                    condition == "periodic_initializer"
                    and cycle in trigger_cycles
                ):
                    transform = (expected_positions, initializer)
                elif condition == "R_only":
                    transform = (expected_positions, feedback)
                elif condition == "initializer_then_R":
                    transform = (
                        (expected_positions, initializer)
                        if cycle == 1
                        else (expected_positions, feedback)
                    )
                step = run_one_loop(
                    model,
                    states[condition],
                    loop_index=cfg.max_loops + cycle - 1,
                    block2_position_transform=transform,
                )
                steps[condition] = step
                _record_norms(
                    norm_storage,
                    condition=condition,
                    cycle=cycle,
                    site="loop_output",
                    state=step.state,
                    groups=groups,
                )
                prediction = step.logits.argmax(dim=-1)
                offsets = semantic_offsets(
                    prediction,
                    orbit,
                    orbit_lengths,
                )
                values = behavior_sums[(condition, cycle)]
                values["count"] += batch_size
                values["correct_all"] += int(prediction.eq(target).sum())
                values["orbit8_count"] += int(orbit8.sum())
                values["correct_orbit8"] += int(
                    (prediction.eq(target) & orbit8).sum()
                )
                values["nonendpoint_count"] += int(nonendpoint.sum())
                values["correct_nonendpoint"] += int(
                    (prediction.eq(target) & nonendpoint).sum()
                )
                values["off_orbit"] += int(offsets.eq(-1).sum())
                orbit8_offsets = offsets[orbit8]
                for offset in range(cfg.node_count):
                    mode_counts[(condition, cycle, offset)] += int(
                        orbit8_offsets.eq(offset).sum()
                    )
            states = {
                condition: steps[condition].state for condition in CONDITIONS
            }

    norm_rows = _aggregate_norms(norm_storage)
    behavior_rows = _aggregate_behavior(
        behavior_sums,
        mode_counts,
        node_count=cfg.node_count,
        jump=jump,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "periodic_norm_quantiles.csv", norm_rows)
    _write_csv(out_dir / "periodic_current_step_accuracy.csv", behavior_rows)
    _plot_norm_with_accuracy(
        norm_rows,
        behavior_rows,
        out_dir=out_dir,
        continuation_loops=continuation_loops,
        trigger_cycles=trigger_cycles,
        accuracy_field="accuracy_all",
        filename="periodic_norm_accuracy_all.png",
    )
    _plot_norm_with_accuracy(
        norm_rows,
        behavior_rows,
        out_dir=out_dir,
        continuation_loops=continuation_loops,
        trigger_cycles=trigger_cycles,
        accuracy_field="accuracy_orbit8",
        filename="periodic_norm_accuracy_orbit8.png",
    )
    _plot_accuracy(
        behavior_rows,
        out_dir=out_dir,
        trigger_cycles=trigger_cycles,
    )
    _plot_semantic_offsets(
        behavior_rows,
        out_dir=out_dir,
        trigger_cycles=trigger_cycles,
    )

    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "loss_placement": "no training; current-step readout evaluation",
        "trained_loop_count": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "graphs": batch_size * batches,
        "seed": seed,
        "jump_per_loop": jump,
        "initializer_period": initializer_period,
        "initializer_trigger_cycles": trigger_cycles,
        "current_step_target": "f^(2*continuation_cycle)(H8_endpoint)",
        "behavior_rows": behavior_rows,
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "shared_gpu": shared_gpu,
            "observed_peak_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "files": {
            "norm_quantiles": "periodic_norm_quantiles.csv",
            "current_step_accuracy": "periodic_current_step_accuracy.csv",
            "figures": [
                "periodic_norm_accuracy_all.png",
                "periodic_norm_accuracy_orbit8.png",
                "periodic_current_step_accuracy.png",
                "periodic_semantic_offset.png",
            ],
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure periodic-initializer hidden norms and current-step accuracy."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--initializer-artifact", type=Path, required=True)
    parser.add_argument("--feedback-artifact", type=Path, required=True)
    parser.add_argument("--feedback-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--continuation-loops", type=int, default=24)
    parser.add_argument("--initializer-period", type=int, default=6)
    parser.add_argument("--seed", type=int, default=135001)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prelaunch-used-mib", type=int)
    parser.add_argument("--prelaunch-free-mib", type=int)
    parser.add_argument("--declared-peak-gib", type=float, default=1.5)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    parser.add_argument("--shared-gpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(
        checkpoint=args.checkpoint,
        phase_summary_path=args.phase_summary,
        initializer_artifact=args.initializer_artifact,
        feedback_artifact=args.feedback_artifact,
        feedback_label=args.feedback_label,
        out_dir=args.out_dir,
        device_name=args.device,
        batch_size=args.batch_size,
        batches=args.batches,
        continuation_loops=args.continuation_loops,
        initializer_period=args.initializer_period,
        seed=args.seed,
        physical_gpu=args.physical_gpu,
        prelaunch_used_mib=args.prelaunch_used_mib,
        prelaunch_free_mib=args.prelaunch_free_mib,
        declared_peak_gib=args.declared_peak_gib,
        reserve_gib=args.reserve_gib,
        shared_gpu=args.shared_gpu,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
