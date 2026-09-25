from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.long_cycle_graph_path import make_single_cycle_successors
from reasoning_loop.graph_path_loop import set_seed
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def checkpoint(root: Path, seed: int, node_count: int = 8) -> Path:
    return (
        root
        / f"D8_L8_seed{seed}"
        / f"graphpath_N{node_count}_D8_d256_B2_L8_seed{seed}"
        / "best.pt"
    )


def path_target_positions(*, node_count: int, max_depth: int) -> tuple[int, ...]:
    del node_count
    return tuple(range(max_depth + 1))


def validate_checkpoint_config(cfg: Any, *, expected_node_count: int) -> None:
    actual = (cfg.node_count, cfg.max_depth, cfg.max_loops, cfg.n_layers)
    expected = (expected_node_count, 8, 8, 2)
    if actual != expected:
        raise ValueError(f"unexpected checkpoint config: {vars(cfg)}")


def validate_seed_panel(seeds: list[int]) -> tuple[int, ...]:
    seed_ids = tuple(seeds)
    if len(seed_ids) != 6:
        raise ValueError("the paper panel requires exactly six seeds")
    if len(set(seed_ids)) != len(seed_ids) or any(seed < 0 for seed in seed_ids):
        raise ValueError("seed ids must be unique non-negative integers")
    return seed_ids


def validate_candidate_seeds(seeds: list[int]) -> tuple[int, ...]:
    seed_ids = tuple(seeds)
    if len(seed_ids) < 6:
        raise ValueError("the diversity candidate pool requires at least six seeds")
    if len(set(seed_ids)) != len(seed_ids) or any(seed < 0 for seed in seed_ids):
        raise ValueError("seed ids must be unique non-negative integers")
    return seed_ids


def single_cycle_fixed_depth_batch(
    cfg: Any,
    *,
    batch_size: int,
    device: torch.device,
    path_positions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if path_positions >= cfg.node_count:
        raise ValueError("path positions must be shorter than the single cycle")
    successors = make_single_cycle_successors(
        batch_size=batch_size,
        node_count=cfg.node_count,
        device=device,
    )
    start = torch.randint(0, cfg.node_count, (batch_size,), device=device)
    return fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=start,
    )


def select_maximin_panel(
    results: dict[int, dict[str, Any]],
    *,
    panel_size: int = 6,
    selection_boundaries: int = 9,
    performance_gate: float = 0.99,
) -> tuple[int, ...]:
    eligible = [
        seed
        for seed in sorted(results)
        if np.asarray(results[seed]["top1_match_fraction"])[8, 8]
        >= performance_gate
    ]
    if len(eligible) < panel_size:
        raise ValueError(
            f"only {len(eligible)} seeds pass the performance gate; need {panel_size}"
        )
    vectors = {
        seed: np.asarray(results[seed]["mean_readout_probability"])[
            :selection_boundaries
        ].reshape(-1)
        for seed in eligible
    }
    best_combo: tuple[int, ...] | None = None
    best_score: tuple[float, float] | None = None
    for combo in itertools.combinations(eligible, panel_size):
        distances = [
            float(np.sqrt(np.mean((vectors[first] - vectors[second]) ** 2)))
            for first, second in itertools.combinations(combo, 2)
        ]
        score = (min(distances), float(np.mean(distances)))
        if best_score is None or score > best_score or (
            score == best_score and combo < best_combo
        ):
            best_combo = combo
            best_score = score
    assert best_combo is not None
    return best_combo


def panel_distance_statistics(
    results: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    selection_boundaries: int,
) -> dict[str, Any]:
    vectors = {
        seed: np.asarray(results[seed]["mean_readout_probability"])[
            :selection_boundaries
        ].reshape(-1)
        for seed in seeds
    }
    pairwise = [
        {
            "seed_a": first,
            "seed_b": second,
            "rms_distance": float(
                np.sqrt(np.mean((vectors[first] - vectors[second]) ** 2))
            ),
        }
        for first, second in itertools.combinations(seeds, 2)
    ]
    distances = [entry["rms_distance"] for entry in pairwise]
    return {
        "minimum_pairwise_rms_distance": min(distances),
        "mean_pairwise_rms_distance": float(np.mean(distances)),
        "pairwise_distances": pairwise,
    }


@torch.no_grad()
def evaluate_seed(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    loops: int,
    data_seed: int,
    expected_node_count: int = 8,
    graph_distribution: str = "random_permutation",
    max_path_position: int | None = None,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint_path, device)
    validate_checkpoint_config(cfg, expected_node_count=expected_node_count)
    if max_path_position is None:
        max_path_position = cfg.max_depth
    if max_path_position < 1:
        raise ValueError("max path position must be positive")

    probability_sum = np.zeros((loops + 1, max_path_position + 1), dtype=np.float64)
    top1_match_count = np.zeros_like(probability_sum, dtype=np.int64)
    examples = 0
    set_seed(data_seed)

    for _ in range(batches):
        if graph_distribution == "single_cycle":
            tokens, targets, _, start = single_cycle_fixed_depth_batch(
                cfg,
                batch_size=batch_size,
                device=device,
                path_positions=max_path_position,
            )
        elif graph_distribution == "random_permutation":
            tokens, targets, _, start = fixed_depth_batch(
                cfg,
                batch_size,
                device,
                path_positions=max_path_position,
            )
        else:
            raise ValueError(f"unknown graph distribution: {graph_distribution}")
        path_nodes = torch.cat((start[:, None], targets), dim=1)
        if graph_distribution == "single_cycle" and not all(
            torch.unique(row).numel() == path_nodes.shape[1] for row in path_nodes
        ):
            raise RuntimeError("single-cycle path targets unexpectedly overlap")

        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        states = [state]
        for loop_index in range(loops):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            states.append(state)

        for loop_index, selected_state in enumerate(states):
            logits = logits_from_raw_state(model, selected_state)
            probability = logits.softmax(dim=-1)
            prediction = logits.argmax(dim=-1)
            probability_sum[loop_index] += (
                probability.gather(1, path_nodes).sum(dim=0).float().cpu().numpy()
            )
            top1_match_count[loop_index] += (
                prediction[:, None].eq(path_nodes).sum(dim=0).cpu().numpy()
            )
        examples += batch_size

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "examples": examples,
        "mean_readout_probability": probability_sum / examples,
        "top1_match_fraction": top1_match_count / examples,
    }


def plot(
    results: dict[int, dict[str, Any]],
    output_path: Path,
    *,
    node_count: int,
    graph_distribution: str = "random_permutation",
) -> None:
    figure = plt.figure(figsize=(7.05, 5.4))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.905,
        bottom=0.09,
        top=0.91,
        wspace=0.18,
        hspace=0.30,
    )
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    image = None
    for axis, seed in zip(axes, sorted(results), strict=True):
        values = np.asarray(results[seed]["mean_readout_probability"])
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        maxima = values.argmax(axis=1)
        axis.plot(
            maxima,
            np.arange(values.shape[0]),
            "o",
            ms=2.1,
            color="white",
            mec="black",
            mew=0.25,
        )
        axis.axhline(8.5, color="#ef4444", linestyle="--", linewidth=0.8)
        axis.set_title(f"seed {seed}", fontsize=9, pad=2)
        positions = path_target_positions(
            node_count=node_count, max_depth=values.shape[1] - 1
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [rf"$f^{position}$" for position in positions], fontsize=6.5
        )
        axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("path target", fontsize=7.5, labelpad=1)
    for axis in axes[::3]:
        axis.set_ylabel("loop boundary", fontsize=8)

    assert image is not None
    colorbar_axis = figure.add_axes((0.92, 0.12, 0.014, 0.76))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("mean readout probability", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    title = (
        f"Intermediate readouts on newly sampled random {node_count}-cycle graphs"
        if graph_distribution == "single_cycle"
        else "Intermediate readouts on newly sampled random permutation graphs"
    )
    figure.text(
        0.075,
        0.975,
        title,
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    figure.savefig(output_path)
    plt.close(figure)


def plot_overview(
    results: dict[int, dict[str, Any]],
    output_path: Path,
    *,
    node_count: int,
    graph_distribution: str,
    metric_key: str = "mean_readout_probability",
) -> None:
    seed_ids = sorted(results)
    columns = 4
    rows = int(np.ceil(len(seed_ids) / columns))
    figure = plt.figure(figsize=(9.5, 2.45 * rows + 0.65))
    grid = figure.add_gridspec(
        rows,
        columns,
        left=0.065,
        right=0.92,
        bottom=0.075,
        top=0.92,
        wspace=0.18,
        hspace=0.34,
    )
    axes = [
        figure.add_subplot(grid[row, column])
        for row in range(rows)
        for column in range(columns)
    ]
    image = None
    for axis, seed in zip(axes, seed_ids):
        values = np.asarray(results[seed][metric_key])
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        maxima = values.argmax(axis=1)
        axis.plot(
            maxima,
            np.arange(values.shape[0]),
            "o",
            ms=1.9,
            color="white",
            mec="black",
            mew=0.22,
        )
        axis.axhline(8.5, color="#ef4444", linestyle="--", linewidth=0.75)
        axis.set_title(f"seed {seed}", fontsize=8.5, pad=2)
        positions = path_target_positions(
            node_count=node_count, max_depth=values.shape[1] - 1
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [rf"$f^{position}$" for position in positions], fontsize=6
        )
        axis.set_yticks([0, 4, 8, 12, 16])
        axis.tick_params(axis="y", labelsize=6.5)
        axis.set_xlabel("path target", fontsize=7, labelpad=1)
    for axis in axes[len(seed_ids) :]:
        axis.set_visible(False)
    for axis in axes[::columns]:
        axis.set_ylabel("loop boundary", fontsize=7.5)

    assert image is not None
    colorbar_axis = figure.add_axes((0.935, 0.12, 0.012, 0.73))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    metric_label = (
        "top-1 output fraction"
        if metric_key == "top1_match_fraction"
        else "mean readout probability"
    )
    colorbar.set_label(metric_label, fontsize=7.5)
    colorbar.ax.tick_params(labelsize=6.5)
    distribution = (
        f"random {node_count}-cycle graphs"
        if graph_distribution == "single_cycle"
        else "random permutation graphs"
    )
    figure.text(
        0.065,
        0.978,
        f"All candidate seeds: {metric_label} on newly sampled {distribution}",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    figure.savefig(output_path)
    plt.close(figure)


def write_outputs(
    results: dict[int, dict[str, Any]],
    out_dir: Path,
    data_seed: int,
    *,
    node_count: int,
    graph_distribution: str = "random_permutation",
) -> None:
    rows: list[dict[str, Any]] = []
    for seed, result in results.items():
        probability = np.asarray(result["mean_readout_probability"])
        top1_match = np.asarray(result["top1_match_fraction"])
        for loop_index in range(probability.shape[0]):
            for path_position in range(probability.shape[1]):
                rows.append(
                    {
                        "seed": seed,
                        "loop": loop_index,
                        "path_position": path_position,
                        "mean_readout_probability": float(
                            probability[loop_index, path_position]
                        ),
                        "top1_match_fraction": float(
                            top1_match[loop_index, path_position]
                        ),
                        "examples": int(result["examples"]),
                    }
                )
    with (out_dir / "multiseed_random_path_readout_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_path_position = (
        np.asarray(next(iter(results.values()))["mean_readout_probability"]).shape[1]
        - 1
    )
    summary = {
        "protocol": {
            "backbone": "Pre-Norm D8L8, two shared physical blocks, CE only after loop 8",
            "node_count": node_count,
            "graph_distribution": (
                f"uniformly sampled single {node_count}-cycle permutations"
                if graph_distribution == "single_cycle"
                else "unconditioned random permutation graphs from fixed_depth_batch"
            ),
            "path_targets": f"f_G^0(start),...,f_G^{max_path_position}(start)",
            "stored_metrics": [
                "mean readout probability assigned to each path target",
                "top-1 output fraction matching each path target",
            ],
            "evaluated_loops": 16,
            "examples_per_seed": next(iter(results.values()))["examples"],
            "data_seed": data_seed,
            "same_evaluation_stream_across_seeds": True,
            "evaluated_seed_ids": sorted(results),
            "target_distinctness": (
                f"guaranteed: f_G^0 through f_G^{max_path_position} are distinct "
                f"because {max_path_position} < {node_count}"
                if graph_distribution == "single_cycle"
                else "not guaranteed because random permutations can contain short cycles"
            ),
            "collision_note": (
                "none among plotted path targets"
                if graph_distribution == "single_cycle"
                else "path targets can denote the same node on short permutation cycles; columns are not mutually exclusive"
            ),
        },
        "seeds": {
            str(seed): {
                "checkpoint": result["checkpoint"],
                "checkpoint_step": result["checkpoint_step"],
                "examples": result["examples"],
                "mean_readout_probability": result["mean_readout_probability"].tolist(),
                "top1_match_fraction": result["top1_match_fraction"].tolist(),
            }
            for seed, result in results.items()
        },
    }
    if len(results) == 6:
        summary["protocol"]["panel_seed_ids"] = sorted(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_selection(
    results: dict[int, dict[str, Any]],
    selected: tuple[int, ...],
    out_dir: Path,
    *,
    performance_gate: float,
    selection_boundaries: int,
) -> None:
    eligible = [
        seed
        for seed in sorted(results)
        if np.asarray(results[seed]["top1_match_fraction"])[8, 8]
        >= performance_gate
    ]
    payload = {
        "selection_method": (
            "enumerate all six-seed subsets after the performance gate; maximize "
            "minimum pairwise RMS distance between flattened mean-readout matrices, "
            "then mean pairwise distance, then lexicographic seed ids"
        ),
        "candidate_seed_ids": sorted(results),
        "eligible_seed_ids": eligible,
        "selected_seed_ids": list(selected),
        "performance_gate": {
            "metric": "top1_match_fraction at loop 8 for f^8(start)",
            "threshold": performance_gate,
            "values": {
                str(seed): float(
                    np.asarray(results[seed]["top1_match_fraction"])[8, 8]
                )
                for seed in sorted(results)
            },
        },
        "selection_boundaries": list(range(selection_boundaries)),
        **panel_distance_statistics(
            results,
            selected,
            selection_boundaries=selection_boundaries,
        ),
    }
    (out_dir / "selected_panel.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument(
        "--graph-distribution",
        choices=("random_permutation", "single_cycle"),
        default="random_permutation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--loops", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=8_091_709)
    parser.add_argument("--select-diverse-panel", action="store_true")
    parser.add_argument("--performance-gate", type=float, default=0.99)
    parser.add_argument("--selection-boundaries", type=int, default=9)
    parser.add_argument("--max-path-position", type=int)
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument("--plot-both-metrics", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results: dict[int, dict[str, Any]] = {}
    seed_ids = (
        validate_candidate_seeds(args.seeds)
        if args.select_diverse_panel or args.overview_only
        else validate_seed_panel(args.seeds)
    )
    for seed in seed_ids:
        results[seed] = evaluate_seed(
            checkpoint(args.checkpoint_root, seed, node_count=args.node_count),
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            loops=args.loops,
            data_seed=args.data_seed,
            expected_node_count=args.node_count,
            graph_distribution=args.graph_distribution,
            max_path_position=args.max_path_position,
        )
        print(f"completed seed {seed}", flush=True)
    write_outputs(
        results,
        args.out_dir,
        args.data_seed,
        node_count=args.node_count,
        graph_distribution=args.graph_distribution,
    )
    if args.overview_only:
        metrics = (
            ("top1_match_fraction", "top1_accuracy"),
            ("mean_readout_probability", "mean_probability"),
        )
        if not args.plot_both_metrics:
            metrics = metrics[:1]
        max_path_position = (
            np.asarray(next(iter(results.values()))["mean_readout_probability"]).shape[1]
            - 1
        )
        for metric_key, suffix in metrics:
            for extension in ("pdf", "png"):
                plot_overview(
                    results,
                    args.out_dir
                    / f"all_candidate_seeds_f0_f{max_path_position}_{suffix}.{extension}",
                    node_count=args.node_count,
                    graph_distribution=args.graph_distribution,
                    metric_key=metric_key,
                )
        return
    if args.select_diverse_panel:
        selected_seed_ids = select_maximin_panel(
            results,
            panel_size=6,
            selection_boundaries=args.selection_boundaries,
            performance_gate=args.performance_gate,
        )
        write_selection(
            results,
            selected_seed_ids,
            args.out_dir,
            performance_gate=args.performance_gate,
            selection_boundaries=args.selection_boundaries,
        )
        plot_overview(
            results,
            args.out_dir / "all_candidate_seeds_cycle10_readout.pdf",
            node_count=args.node_count,
            graph_distribution=args.graph_distribution,
        )
        plot_overview(
            results,
            args.out_dir / "all_candidate_seeds_cycle10_readout.png",
            node_count=args.node_count,
            graph_distribution=args.graph_distribution,
        )
        results = {seed: results[seed] for seed in selected_seed_ids}
    plot(
        results,
        args.out_dir / "multiseed_random_path_readout.pdf",
        node_count=args.node_count,
        graph_distribution=args.graph_distribution,
    )
    plot(
        results,
        args.out_dir / "multiseed_random_path_readout.png",
        node_count=args.node_count,
        graph_distribution=args.graph_distribution,
    )


if __name__ == "__main__":
    main()
