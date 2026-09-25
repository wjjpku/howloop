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
from reasoning_loop.graph_path_telomere_initializer_semantic_readout import (
    _load_initializer,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_shared_power_eval import _load_map


QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
CONDITIONS = (
    "no_control",
    "initializer_only",
    "R_only",
    "initializer_then_R",
)
SITES = (
    "loop_input",
    "block2_pre_transform",
    "block2_post_transform",
    "post_attention_residual",
    "loop_output",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def norm_quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().flatten()
    if flat.numel() == 0:
        raise ValueError("cannot summarize an empty norm sample")
    quantiles = torch.quantile(
        flat,
        torch.tensor(QUANTILES, device=flat.device),
    )
    result = {
        f"q{int(100 * probability):02d}": float(value)
        for probability, value in zip(QUANTILES, quantiles, strict=True)
    }
    result.update(
        {
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "count": int(flat.numel()),
        }
    )
    return result


def _record_norms(
    storage: dict[tuple[str, int, str, str], list[torch.Tensor]],
    *,
    condition: str,
    cycle: int,
    site: str,
    state: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
) -> None:
    norms = state.float().norm(dim=-1)
    for group_name, positions in groups.items():
        storage[(condition, cycle, site, group_name)].append(
            norms[:, list(positions)].flatten().cpu()
        )


def _aggregate_norms(
    storage: dict[tuple[str, int, str, str], list[torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    for (condition, cycle, site, group), parts in sorted(storage.items()):
        rows.append(
            {
                "condition": condition,
                "cycle": cycle,
                "site": site,
                "position_group": group,
                **norm_quantiles(torch.cat(parts)),
            }
        )
    return rows


def _row_index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {
        (
            str(row["condition"]),
            int(row["cycle"]),
            str(row["site"]),
            str(row["position_group"]),
        ): row
        for row in rows
    }


def _plot_norms(rows: list[dict[str, Any]], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = _row_index(rows)
    group_order = ("all", "answer", "graph", "metadata")
    colors = {
        "no_control": "#4c78a8",
        "initializer_only": "#f58518",
        "R_only": "#54a24b",
        "initializer_then_R": "#e45756",
    }
    labels = {
        "no_control": "No control",
        "initializer_only": "Initializer only",
        "R_only": "R only",
        "initializer_then_R": "Initializer + R",
    }

    figure, axes = plt.subplots(
        2,
        len(group_order),
        figsize=(17, 7.5),
        constrained_layout=True,
    )
    for column, group in enumerate(group_order):
        axis = axes[0, column]
        cycles = list(range(0, 9))
        natural = [
            index[("natural", cycle, "loop_output", group)]
            for cycle in cycles
        ]
        median = np.array([row["q50"] for row in natural])
        q25 = np.array([row["q25"] for row in natural])
        q75 = np.array([row["q75"] for row in natural])
        q05 = np.array([row["q05"] for row in natural])
        q95 = np.array([row["q95"] for row in natural])
        axis.fill_between(cycles, q05, q95, color="#9ecae9", alpha=0.22)
        axis.fill_between(cycles, q25, q75, color="#4c78a8", alpha=0.28)
        axis.plot(cycles, median, color="#1f4e79", marker="o", linewidth=2)
        axis.set_title(f"Natural trajectory: {group}")
        axis.set_xlabel("Natural loop")
        if column == 0:
            axis.set_ylabel("Per-token hidden-state L2 norm")
        axis.grid(alpha=0.2)

        axis = axes[1, column]
        continuation_cycles = list(range(0, 9))
        for condition in CONDITIONS:
            condition_rows = [
                index[(condition, cycle, "loop_output", group)]
                for cycle in continuation_cycles
            ]
            median = np.array([row["q50"] for row in condition_rows])
            q25 = np.array([row["q25"] for row in condition_rows])
            q75 = np.array([row["q75"] for row in condition_rows])
            q05 = np.array([row["q05"] for row in condition_rows])
            q95 = np.array([row["q95"] for row in condition_rows])
            axis.fill_between(
                continuation_cycles,
                q05,
                q95,
                color=colors[condition],
                alpha=0.035,
            )
            axis.fill_between(
                continuation_cycles,
                q25,
                q75,
                color=colors[condition],
                alpha=0.10,
            )
            axis.plot(
                continuation_cycles,
                median,
                color=colors[condition],
                label=labels[condition],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
            )
        axis.set_title(f"Continuation from H8: {group}")
        axis.set_xlabel("Continuation loop (0 = H8)")
        if column == 0:
            axis.set_ylabel("Per-token hidden-state L2 norm")
        axis.grid(alpha=0.2)
    axes[1, -1].legend(loc="best", fontsize=8)
    path = out_dir / "hidden_state_norm_trajectories.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)

    site_order = (
        "block2_pre_transform",
        "block2_post_transform",
        "post_attention_residual",
        "loop_output",
    )
    figure, axes = plt.subplots(
        len(site_order),
        2,
        figsize=(12, 13),
        constrained_layout=True,
    )
    for row_index, site in enumerate(site_order):
        for column, group in enumerate(("all", "answer")):
            axis = axes[row_index, column]
            cycles = list(range(1, 9))
            for condition in CONDITIONS:
                parts = [
                    index[(condition, cycle, site, group)]
                    for cycle in cycles
                ]
                median = np.array([row["q50"] for row in parts])
                q25 = np.array([row["q25"] for row in parts])
                q75 = np.array([row["q75"] for row in parts])
                axis.fill_between(
                    cycles,
                    q25,
                    q75,
                    color=colors[condition],
                    alpha=0.09,
                )
                axis.plot(
                    cycles,
                    median,
                    color=colors[condition],
                    label=labels[condition],
                    marker="o",
                    linewidth=1.7,
                    markersize=3,
                )
            axis.set_title(f"{site.replace('_', ' ')}: {group}")
            axis.set_xlabel("Continuation loop")
            axis.set_ylabel("L2 norm")
            axis.grid(alpha=0.2)
    axes[0, -1].legend(loc="best", fontsize=8)
    path_by_site = out_dir / "hidden_state_norm_by_site.png"
    figure.savefig(path_by_site, dpi=180)
    plt.close(figure)
    return [path.name, path_by_site.name]


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

    explicit_groups = explicit_depth_position_groups(cfg.node_count)
    sequence_positions = tuple(range(cfg.seq_len))
    groups = {
        "all": sequence_positions,
        "answer": explicit_groups["answer"],
        "graph": explicit_groups["graph"],
        "metadata": explicit_groups["query_metadata"],
    }
    storage: dict[
        tuple[str, int, str, str],
        list[torch.Tensor],
    ] = defaultdict(list)

    set_seed(seed)
    for _ in range(batches):
        tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        natural_state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        _record_norms(
            storage,
            condition="natural",
            cycle=0,
            site="loop_output",
            state=natural_state,
            groups=groups,
        )
        for loop_index in range(cfg.max_loops):
            _record_norms(
                storage,
                condition="natural",
                cycle=loop_index,
                site="loop_input",
                state=natural_state,
                groups=groups,
            )
            natural_step = run_one_loop(
                model,
                natural_state,
                loop_index=loop_index,
            )
            for site, state in (
                (
                    "block2_pre_transform",
                    natural_step.block2_hidden_pre_intervention,
                ),
                ("block2_post_transform", natural_step.block2_hidden_in),
                (
                    "post_attention_residual",
                    natural_step.block2_residual_mid,
                ),
                ("loop_output", natural_step.state),
            ):
                _record_norms(
                    storage,
                    condition="natural",
                    cycle=loop_index + 1,
                    site=site,
                    state=state,
                    groups=groups,
                )
            natural_state = natural_step.state

        states = {
            condition: natural_state.clone() for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            _record_norms(
                storage,
                condition=condition,
                cycle=0,
                site="loop_output",
                state=states[condition],
                groups=groups,
            )
        for cycle in range(1, continuation_loops + 1):
            steps = {}
            for condition in CONDITIONS:
                _record_norms(
                    storage,
                    condition=condition,
                    cycle=cycle,
                    site="loop_input",
                    state=states[condition],
                    groups=groups,
                )
                transform = None
                if condition == "initializer_only" and cycle == 1:
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
                for site, state in (
                    (
                        "block2_pre_transform",
                        step.block2_hidden_pre_intervention,
                    ),
                    ("block2_post_transform", step.block2_hidden_in),
                    (
                        "post_attention_residual",
                        step.block2_residual_mid,
                    ),
                    ("loop_output", step.state),
                ):
                    _record_norms(
                        storage,
                        condition=condition,
                        cycle=cycle,
                        site=site,
                        state=state,
                        groups=groups,
                    )
            states = {
                condition: steps[condition].state for condition in CONDITIONS
            }

    rows = _aggregate_norms(storage)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "hidden_norm_quantiles.csv", rows)
    figure_files = _plot_norms(rows, out_dir)
    index = _row_index(rows)
    immediate_ratios = {}
    for condition in CONDITIONS:
        immediate_ratios[condition] = []
        for cycle in range(1, continuation_loops + 1):
            before = index[
                (condition, cycle, "block2_pre_transform", "all")
            ]["q50"]
            after = index[
                (condition, cycle, "block2_post_transform", "all")
            ]["q50"]
            immediate_ratios[condition].append(after / before)

    summary = {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "initializer_artifact": str(initializer_artifact),
        "feedback_artifact": str(feedback_artifact),
        "feedback_label": feedback_label,
        "loss_placement": "no training; hidden-state norm observation only",
        "trained_loop_count": cfg.max_loops,
        "evaluated_natural_loops": cfg.max_loops,
        "evaluated_continuation_loops": continuation_loops,
        "shared_block_count": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "d_model": cfg.d_model,
        "inner_norm_style": getattr(cfg, "inner_norm_style", "pre_layernorm"),
        "graphs": batch_size * batches,
        "seed": seed,
        "gpu_runtime": {
            "physical_gpu": physical_gpu,
            "prelaunch_used_mib": prelaunch_used_mib,
            "prelaunch_free_mib": prelaunch_free_mib,
            "declared_peak_gib": declared_peak_gib,
            "reserve_gib": reserve_gib,
            "shared_gpu": shared_gpu,
        },
        "map_timing": {
            "initializer_only": "initializer at continuation cycle 1 only",
            "R_only": "R at every continuation cycle",
            "initializer_then_R": (
                "initializer at cycle 1, then R at cycles 2 onward"
            ),
        },
        "median_post_over_pre_transform_ratio_all_positions": immediate_ratios,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda"
            else 0.0
        ),
        "files": {
            "quantiles": "hidden_norm_quantiles.csv",
            "figures": figure_files,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hidden-state norm distributions under I and R."
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
    parser.add_argument("--continuation-loops", type=int, default=8)
    parser.add_argument("--seed", type=int, default=133001)
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
